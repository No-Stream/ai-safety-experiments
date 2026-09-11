r"""The interp arc's readout: one document for the twin-pd pair's internals, recomputed on demand.

Six analyses wrote JSON into two artifact trees -- axis trajectories, axis-to-axis geometry, arm
displacement, two-path capture agreement, three Jacobian lenses, and the steering grid -- and none
of them is readable on its own. This module is the single document that puts them together, and
every number in it is computed here from those payloads at the moment the command runs:

    uv run --frozen python -m games.interp_readout

Both roots default to the newest pass, so the zero-argument invocation stays correct as passes come
and go, and the steering grid is gathered across *every* pass that holds one rather than out of the
newest, because two rented boxes filled it in. Nothing is hand-typed and nothing is cached: re-run
the analyses, re-run this, and the whole document updates itself.

Three rules this file exists to enforce, each earned:

**A missing input is named, never omitted.** Every artifact this reads is looked for by name; when it
is absent the document keeps the section, replaces its tables with a stub that says which file was
missing and what the section would have said, and raises the banner at the top. A readout that
quietly drops a section reads as complete, which is worse than no readout at all.

**Every rate carries its denominator.** The steering pass censors hard -- 38% to 58% of completions
per cell ran past the thinking budget and never produced a parsable action -- so a cooperation rate
whose denominator the reader cannot see is not interpretable. Rates render as `rate (k/n)` with the
truncation and parse-failure counts in adjacent columns.

**Two producers wrote the displacement read, and which one a table came from is stated.** The
tracked `games.interp_displacement` module and the scratch script it was promoted from emit different
payloads, both of which are on disk: the module's carries two split-half reliabilities (pair-split and
row-split, and on this corpus the row split is a *side* split), two named residual-norm denominators,
the along-axis split inline, and an arbitrary number of arms. Shapes are told apart by top-level keys
and never guessed at, the module's wins wherever it is present, and any payload found and not read is
named in the header -- a document that silently rendered the older shape while a newer pass sat beside
it would be wrong in a way no number in it reveals.

**A placebo sits directly beneath the row it controls.** Same table, adjacent row, and when the
placebo is absent its row is still emitted carrying `MISSING`. A placebo in a separate table can be
skipped by the eye; an absent one in a separate table is invisible, and the 3B/8B result this
repository already has on record is precisely a positive that the placebo matched.

`markdown_table` here duplicates fifteen lines of `games.battery_readout`'s renderer rather than
importing it: that module reaches the eval battery's whole graph (pandas, `games.evals`,
`games.arms`) for a table formatter, and this readout has nothing else to do with the battery.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import torch

from reward_hacking.interp.directions import cosine

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_ARTIFACTS_ROOT = Path("artifacts/games")
ANALYSIS_DIR_GLOB = "interp-analysis-*"
DISPLACEMENT_DIR_GLOB = "interp-displacement-*"
GPU_DIR_GLOB = "interp-gpu-*"
# Broader than the lens pass glob because steering cells have come off more than one rented box, and
# the second box's pass carries no lens ladder and a name `interp-gpu-*` never matches.
STEERING_PASS_GLOB = "interp-*"
DISPLACEMENT_FILENAME = "displacement.json"
TWO_PATH_FILENAME = "two_path_agreement.json"
READOUT_DIR_NAME = "readout"
DOCUMENT_FILENAME = "interp-readout.md"
TRAJECTORY_FIGURE = "heldout-projection-trajectory.png"
DRIFT_FIGURE = "drift-cosine-heatmap.png"

# The three stimulus sets, and the short names every table column uses for them.
SET_SHORT: dict[str, str] = {
    "cooperate-vs-defect-commitment": "coarse",
    "correlated-vs-independent-counterpart": "lead",
    "causal-vs-functional-decision": "decision",
}
SHORT_TO_SET: dict[str, str] = {short: name for name, short in SET_SHORT.items()}
# The along-axis payload names its two arms inside its own field keys, and carries only these two.
AXIS_DISPLACEMENT_FIELDS: dict[str, str] = {
    "group": "group_displacement",
    "self": "self_displacement",
}
SHORT_ORDER: tuple[str, ...] = ("coarse", "lead", "decision")
POOLINGS: tuple[str, ...] = ("last", "mean")
PRIMARY_POOLING = "last"
BASE_ARM = "base"

# Two producers have written displacement payloads and both are on disk. `games.interp_displacement`
# is the tracked module and the only producer going forward; the scratch script it was promoted from
# wrote a flatter, strictly-two-arm shape that several earlier passes are still in. The shapes are
# told apart by top-level keys and never guessed at, because a payload handed to the wrong reader
# produces plausible numbers off the wrong fields rather than failing.
SHAPE_MODULE = "module"
SHAPE_SCRATCH = "scratch"
SHAPE_NAMES: dict[str, str] = {
    SHAPE_MODULE: "`games.interp_displacement`",
    SHAPE_SCRATCH: "the scratch displacement script",
}
# Which shape wins when a globbed root holds both: the module's, because it carries strictly more
# (both split-half reliabilities, both residual-norm denominators, the axis split and its two nulls).
SHAPE_PREFERENCE: tuple[str, ...] = (SHAPE_MODULE, SHAPE_SCRATCH)

# The module names the everything-at-once selection `pooled` in all three of its selection fields.
POOLED_SELECTION = "pooled"

TOP_TOKENS = 8
EMPTY_CELL = "-"
MISSING_CELL = "**MISSING**"

GENERATED_BANNER = (
    "**GENERATED -- DO NOT HAND-EDIT.** Every number below is recomputed from the JSON artifacts "
    "on disk. Re-run the CLI in the header and the whole document updates itself; an edit here is "
    "overwritten, and until it is, it disagrees with the artifacts."
)

INCOMPLETE_BANNER = (
    "**THIS READOUT IS INCOMPLETE AND ERROR-CONTAINING.** Inputs listed below were absent when it "
    "was generated, so the sections that depend on them are stubs rather than results. Re-run the "
    "CLI once those artifacts land and it updates itself."
)

CAVEATED_BANNER = (
    "**THIS READOUT IS COMPLETE BUT ERROR-CONTAINING.** Every input was present; the instrument "
    "caveats below still held when it was generated, and re-running cannot clear them."
)

DIAGNOSIS_CAVEAT = (
    "Nothing here is a verdict, a gate or a pass/fail. Where a number is surprising the first "
    "suspect is this code and the analyses upstream of it, not the model."
)

MERGED_BF16_CAVEAT = (
    "The lens and steering legs run **merged bf16** exports, which realise about 64% of the trained "
    "delta (38-79% per module, measured 2026-08-20), while the activation cells behind sections 1-4 "
    "carry all of it. Magnitudes are therefore not comparable between the two halves of this "
    "document, and every lens or steering effect size is understated by an unknown factor in that "
    "range. Orderings, signs and within-leg contrasts are not affected."
)

CENSORING_CAVEAT = (
    "The generation-steering cells censor differentially: a completion that runs past the thinking "
    "budget yields no parsable action and leaves the denominator smaller. Two cells whose "
    "denominators differ by a factor of two are not two measurements of the same population, so "
    "read the truncation column before reading any rate difference as an effect."
)

PRINT_ORDER_CAVEAT = (
    "On this corpus the model picks the first-printed action label 74-82% of the time, which is "
    "larger than every arm and training effect measured on it. Any cooperation shift is therefore "
    "reported inside each print order as well as pooled; a shift present only pooled is a "
    "first-label shift wearing a cooperation label."
)

# Written before the analyses were read, quoted from the arc plan (2026-08-21) so the document
# carries its own pre-registration rather than pointing at a gitignored file for it.
STATED_EXPECTATIONS: tuple[str, ...] = (
    (
        "the arms' displacements are *not* near-orthogonal once measured against the split-half "
        "ceiling -- I expect the pooled 0.09 to be substantially an artifact of averaging over "
        "layers where nothing moved, with a real oppositely-signed along-axis component in the mid "
        "band"
    ),
    (
        "steering the lead axis moves cooperation by less than the training did, in the predicted "
        "direction, at one or two mid-band layers only"
    ),
)

# A cross-arm read needs exactly two arms; two print orders make one order-spread; a decode has to
# appear at two checkpoints before "the same at every checkpoint" is a statement about anything.
ARMS_PER_CROSS_ARM_READ = 2
PRINT_ORDERS = 2
MIN_CELLS_FOR_A_COMPARISON = 2


@dataclass(frozen=True)
class MissingInput:
    """One artifact this readout looked for and did not find, with what its absence costs."""

    name: str
    path: str
    consequence: str


@dataclass(frozen=True)
class PayloadSource:
    """Where one payload was read from, which producer's shape it is in, and any caveat on it."""

    path: Path
    shape: str
    note: str = ""

    def describe(self) -> str:
        """One clause naming the file and the producer, for the header and the section it feeds."""
        rendered = f"`{display_path(self.path)}` written by {SHAPE_NAMES[self.shape]}"
        return f"{rendered}, {self.note}" if self.note else rendered


@dataclass
class Inputs:
    """Every payload the document is built from, plus what was absent and what looked wrong."""

    artifacts_root: Path
    analysis_root: Path | None = None
    gpu_root: Path | None = None
    trajectories: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    axes: dict[str, Any] | None = None
    displacement: dict[str, Any] | None = None
    displacement_source: PayloadSource | None = None
    displacement_unused: list[PayloadSource] = field(default_factory=list)
    displacement_vs_axes: list[dict[str, Any]] | None = None
    two_path: dict[str, Any] | None = None
    two_path_source: PayloadSource | None = None
    lens_cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    lens_files: list[dict[str, Any]] = field(default_factory=list)
    steering: dict[str, dict[str, Any]] = field(default_factory=dict)
    steering_records: dict[str, dict[str, int]] = field(default_factory=dict)
    unsummarised_steering: list[str] = field(default_factory=list)
    logit_sweep: dict[str, Any] | None = None
    missing: list[MissingInput] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def note_missing(self, name: str, path: Path | str, consequence: str) -> None:
        """Record one absent input, which raises the banner and stubs its section."""
        self.missing.append(
            MissingInput(name=name, path=display_path(path), consequence=consequence)
        )


def display_path(path: Path | str) -> str:
    """Render a path for the document: relative to the working directory, else to the home dir.

    Not cosmetic. An absolute path here carries the machine's home directory and therefore its
    username, and this repository's privacy rules keep usernames out of anything a reader might
    paste elsewhere. Both sides are resolved before comparing, because the home directory is a
    symlink on this box and a lexical comparison would miss every path under it.
    """
    resolved = Path(path).expanduser().resolve()
    for root, prefix in ((Path.cwd().resolve(), ""), (Path.home().resolve(), "~/")):
        if resolved.is_relative_to(root):
            return f"{prefix}{resolved.relative_to(root)}"
    return str(path)


# --------------------------------------------------------------------------------------------
# Finding and loading the artifacts
# --------------------------------------------------------------------------------------------


def find_latest_dir(root: Path, pattern: str) -> Path | None:
    """Return the most recently written directory under `root` matching `pattern`, or None.

    Modification time rather than name order, matching `games.battery_cells.latest_battery_dir`: a
    pass is named for the day or the commit that produced it and neither sorts chronologically.
    """
    if not root.is_dir():
        return None
    candidates = sorted(path for path in root.glob(pattern) if path.is_dir())
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    if len(candidates) > 1:
        logger.info(f"{len(candidates)} {pattern!r} passes under {root}; reading {newest.name}")
    return newest


def read_json(path: Path) -> object | None:
    """Return a parsed payload, or None when the file is absent.

    Absent is a coverage fact the banner reports. Malformed is a bug and raises: a readout that
    swallows a half-written payload prints numbers from whatever parsed.
    """
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_trajectories(root: Path, inputs: Inputs) -> None:
    """Load every (pooling, set) trajectory report under `root` into `inputs.trajectories`.

    The pass ran once per (pooling, set, arm) for wall-clock parallelism, so a combined
    `trajectory-<pooling>-<short>/` directory and a split
    `trajectory-<pooling>-<short>-{group,self}/` pair are both valid layouts. Merging a split pair
    is a dict union of `groups` and `behavior_correlations`; `cross_arm` is empty in a one-arm run
    and is rebuilt from the direction tensors each run saved, which is the same arithmetic
    `games.interp_trajectory.cross_arm_reads` does on the same tensors.
    """
    for pooling in POOLINGS:
        for short in SHORT_ORDER:
            payload = _load_one_trajectory(root, pooling, short)
            if payload is None:
                consequence = (
                    "the axis-quality grid, drift and held-out projection tables for this "
                    "cell, and its panel of both figures"
                    if pooling == PRIMARY_POOLING
                    else "the pooling robustness check for this cell"
                )
                inputs.note_missing(
                    name=f"trajectory report ({pooling}, {short})",
                    path=root / f"trajectory-{pooling}-{short}[-<arm>]/trajectory.json",
                    consequence=consequence,
                )
                continue
            inputs.trajectories[pooling, short] = payload


def _load_one_trajectory(root: Path, pooling: str, short: str) -> dict[str, Any] | None:
    """Return one (pooling, set) trajectory payload, merging a split per-arm pair if that is how it ran."""
    combined = root / f"trajectory-{pooling}-{short}" / "trajectory.json"
    payload = cast("dict[str, Any] | None", read_json(combined))
    if payload is not None:
        return payload
    merged: dict[str, Any] | None = None
    for path in sorted(root.glob(f"trajectory-{pooling}-{short}-*/trajectory.json")):
        part = cast("dict[str, Any]", read_json(path))
        part["_directions_root"] = {
            group["arm"]: str(path.parent / "directions") for group in part["groups"].values()
        }
        if merged is None:
            merged = part
            continue
        merged["groups"].update(part["groups"])
        merged["behavior_correlations"].update(part["behavior_correlations"])
        merged["_directions_root"].update(part["_directions_root"])
    if merged is not None and not merged.get("cross_arm"):
        merged["cross_arm"] = reconstruct_cross_arm(merged, SHORT_TO_SET[short], pooling)
    return merged


def reconstruct_cross_arm(
    payload: Mapping[str, Any], stimulus_set: str, pooling: str
) -> list[dict[str, Any]]:
    """Rebuild the cross-arm reads from two one-arm runs' saved directions and drift reads.

    Two one-arm runs cannot compare their arms to each other, but each saved its per-checkpoint
    direction tensors, so the cosine between arms is recoverable exactly. The projection gaps come
    from the drift reads, which are already anchored on the shared base cell and therefore already
    subtractable.
    """
    roots = payload.get("_directions_root", {})
    arms = sorted(roots)
    if len(arms) != ARMS_PER_CROSS_ARM_READ:
        return []
    steps = trajectory_steps(payload)
    directions = {
        arm: _load_directions(Path(roots[arm]), arm, stimulus_set, pooling, steps) for arm in arms
    }
    drift = {
        arm: {
            (read["step"], read["layer"]): read
            for read in group_for_arm(payload, arm)["drift_reads"]
        }
        for arm in arms
    }
    reads: list[dict[str, Any]] = []
    for step in steps:
        # Step 0 is one shared base cell, so the cosine there is 1.0 by construction, not a read.
        if step == 0 or any(step not in directions[arm] for arm in arms):
            continue
        for layer in sorted(directions[arms[0]][step]):
            first, second = drift[arms[0]].get((step, layer)), drift[arms[1]].get((step, layer))
            if first is None or second is None:
                continue
            reads.append(
                {
                    "step": step,
                    "stimulus_set": stimulus_set,
                    "pooling": pooling,
                    "layer": layer,
                    "arm_a": arms[0],
                    "arm_b": arms[1],
                    "cosine_between_arms": cosine(
                        directions[arms[0]][step][layer], directions[arms[1]][step][layer]
                    ),
                    "projection_gap_a": first["heldout_projection_gap"],
                    "projection_gap_b": second["heldout_projection_gap"],
                    "projection_gap_difference": (
                        second["heldout_projection_gap"] - first["heldout_projection_gap"]
                    ),
                    "drift_cosine_a": first["cosine_to_anchor"],
                    "drift_cosine_b": second["cosine_to_anchor"],
                }
            )
    return reads


def _load_directions(
    root: Path, arm: str, stimulus_set: str, pooling: str, steps: Sequence[int]
) -> dict[int, dict[int, torch.Tensor]]:
    """Load one arm's per-step direction tensors, falling back to the shared base cell at step 0."""
    loaded: dict[int, dict[int, torch.Tensor]] = {}
    for step in steps:
        filename = f"{stimulus_set}-{pooling}.pt"
        for candidate in (
            root / arm / f"step-{step}" / filename,
            root / BASE_ARM / f"step-{step}" / filename,
        ):
            if candidate.is_file():
                loaded[step] = cast(
                    "dict[int, torch.Tensor]", torch.load(candidate, weights_only=True)
                )
                break
    return loaded


def load_axes(root: Path, inputs: Inputs) -> None:
    """Merge every `axes*/axes.json` under `root`; a per-arm split contributes its own cells."""
    merged: dict[str, Any] | None = None
    for path in sorted(root.glob("axes*/axes.json")):
        payload = cast("dict[str, Any]", read_json(path))
        if merged is None:
            merged = payload
            continue
        merged["cells"].update(payload["cells"])
    if merged is None:
        inputs.note_missing(
            name="axis-to-axis geometry report",
            path=root / "axes*/axes.json",
            consequence="the stratified axis-quality table and the axis-to-axis cosine table",
        )
        return
    inputs.axes = merged


def displacement_shape(payload: Mapping[str, Any]) -> str | None:
    """Which producer wrote a displacement payload, from its top-level keys alone, or None.

    Top-level keys rather than anything inside the reads: the two shapes disagree about the reads too,
    so a payload sniffed on a field they share would be handed to the wrong reader and render numbers
    off fields that mean something else. An unrecognised payload returns None and is named in the
    banner rather than parsed as whichever shape it resembles most.
    """
    keys = set(payload)
    if {"context", "matrices", "reads"} <= keys:
        return SHAPE_MODULE
    if {"reads", "nulls"} <= keys:
        return SHAPE_SCRATCH
    return None


def two_path_shape(payload: Mapping[str, Any]) -> str | None:
    """Which producer wrote a two-path agreement payload, from its summary's own keys.

    The summary is the discriminator rather than the file's top-level keys, because the module also
    embeds its summary inside the displacement payload, where the per-layer reads do not travel with
    it -- and that embedded copy has to be recognised as the same shape as the side-car file.
    """
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return None
    keys = set(cast("dict[str, Any]", summary))
    if {"off_by_one_sabotage", "n_comparable_reads"} <= keys:
        return SHAPE_MODULE
    if {"shift_sabotage", "n_direction_reads"} <= keys:
        return SHAPE_SCRATCH
    return None


def displacement_candidate_dirs(artifacts_root: Path, analysis_root: Path | None) -> list[Path]:
    """Every pass that might hold a displacement payload, the named analysis root first.

    The tracked module writes its own `interp-displacement-*` pass rather than into the CPU analysis
    pass, so a readout that looked only under `analysis_root` would go on reading the superseded
    scratch payload beside it and never say that a newer one existed.
    """
    directories: list[Path] = []
    globbed = [
        *sorted(artifacts_root.glob(DISPLACEMENT_DIR_GLOB)),
        *sorted(artifacts_root.glob(ANALYSIS_DIR_GLOB)),
    ]
    for candidate in ([analysis_root] if analysis_root is not None else []) + globbed:
        if candidate.is_dir() and candidate not in directories:
            directories.append(candidate)
    return directories


def load_displacement(artifacts_root: Path, inputs: Inputs) -> None:
    """Choose one displacement payload across every pass on disk, preferring the module's shape.

    Both producers' payloads coexist here, so the choice is made once and *named* in the header rather
    than falling out of which directory happened to be newest. Within one shape the newest file wins;
    every payload not read is listed, because "there is a second displacement payload on disk and this
    document is not reading it" is exactly the fact a reader cannot recover from the numbers.
    """
    candidates = displacement_candidate_dirs(artifacts_root, inputs.analysis_root)
    found: list[tuple[Path, dict[str, Any], str]] = []
    for directory in candidates:
        path = directory / DISPLACEMENT_FILENAME
        payload = cast("dict[str, Any] | None", read_json(path))
        if payload is None:
            continue
        shape = displacement_shape(payload)
        if shape is None:
            inputs.problems.append(
                f"`{display_path(path)}` is a displacement payload in neither known shape (top-level "
                f"keys {sorted(payload)}), so it was not read: a payload handed to the wrong reader "
                f"renders plausible numbers off the wrong fields."
            )
            continue
        found.append((path, payload, shape))
    if not found:
        if candidates:
            inputs.note_missing(
                name="arm displacement report",
                path=candidates[0] / DISPLACEMENT_FILENAME,
                consequence=(
                    "the cross-arm displacement cosine table, the along-axis split and the "
                    "label-shuffle nulls"
                ),
            )
        return
    ordered = sorted(
        found, key=lambda entry: (SHAPE_PREFERENCE.index(entry[2]), -entry[0].stat().st_mtime)
    )
    path, payload, shape = ordered[0]
    inputs.displacement = payload
    inputs.displacement_source = PayloadSource(path=path, shape=shape)
    inputs.displacement_unused = [
        PayloadSource(path=other, shape=other_shape) for other, _, other_shape in ordered[1:]
    ]


def load_displacement_vs_axes(inputs: Inputs) -> None:
    """Load the separate along-axis report, which only the scratch shape needs.

    The module payload carries the same split per read (`axis_components`, against both the base
    anchor's axis and the arm's own), so under that shape an absent `displacement_vs_axes.json` costs
    nothing and must not raise the incomplete banner over a section that is in fact complete.
    """
    root = inputs.analysis_root
    if root is None:
        return
    inputs.displacement_vs_axes = cast(
        "list[dict[str, Any]] | None", read_json(root / "displacement_vs_axes.json")
    )
    if inputs.displacement_vs_axes is not None:
        return
    if inputs.displacement_source is not None and inputs.displacement_source.shape == SHAPE_MODULE:
        return
    inputs.note_missing(
        name="displacement-against-axes report",
        path=root / "displacement_vs_axes.json",
        consequence="the along-axis / residual split of each arm's displacement",
    )


def _read_two_path(root: Path, inputs: Inputs, note: str = "") -> bool:
    """Read one candidate agreement file, or say why it was not read. True when it was."""
    path = root / TWO_PATH_FILENAME
    payload = cast("dict[str, Any] | None", read_json(path))
    if payload is None:
        return False
    shape = two_path_shape(payload)
    if shape is None:
        inputs.problems.append(
            f"`{display_path(path)}` is a two-path agreement payload in neither known shape (summary "
            f"keys {sorted(cast('dict[str, Any]', payload.get('summary') or {}))}), so it was not "
            f"read."
        )
        return False
    inputs.two_path = payload
    inputs.two_path_source = PayloadSource(path=path, shape=shape, note=note)
    return True


def _read_embedded_two_path(inputs: Inputs) -> bool:
    """Read the agreement summary the module embeds in its displacement payload. True when it was.

    Preferred over a side-car from *another* pass, because it is the check that ran against these
    exact activations. It carries no per-layer reads, which the section says rather than implies.
    """
    if inputs.displacement is None or inputs.displacement_source is None:
        return False
    embedded = cast("dict[str, Any] | None", inputs.displacement.get("two_path_agreement"))
    if embedded is None:
        return False
    wrapped = {"summary": embedded}
    shape = two_path_shape(wrapped)
    if shape is None:
        return False
    inputs.two_path = wrapped
    inputs.two_path_source = PayloadSource(
        path=inputs.displacement_source.path,
        shape=shape,
        note=(
            "as the summary embedded in the displacement payload; the side-car file carrying the "
            "per-layer reads was not on disk"
        ),
    )
    return True


def load_two_path(artifacts_root: Path, inputs: Inputs) -> None:
    """Load the two-path agreement payload, preferring the pass the displacement read came from.

    Order is provenance, not convenience: the side-car beside the chosen displacement payload, then
    the summary that payload embeds, and only then a side-car from some other pass -- which is read if
    that is all there is, and labelled as being from a different pass. A validation check quoted
    beside geometry it never validated is worse than an absent one.
    """
    own = inputs.displacement_source.path.parent if inputs.displacement_source is not None else None
    if own is not None and _read_two_path(own, inputs):
        return
    if _read_embedded_two_path(inputs):
        return
    analysis = inputs.analysis_root
    if (
        analysis is not None
        and analysis != own
        and _read_two_path(
            analysis,
            inputs,
            note="which is a DIFFERENT pass from the displacement payload above",
        )
    ):
        return
    candidates = [root for root in (own, analysis) if root is not None] or (
        displacement_candidate_dirs(artifacts_root, analysis)
    )
    if candidates:
        inputs.note_missing(
            name="two-path capture agreement",
            path=candidates[0] / TWO_PATH_FILENAME,
            consequence="the validation line that says the two capture implementations agree",
        )


def load_lenses(gpu_root: Path, inputs: Inputs) -> None:
    """Merge every Jacobian lens ladder under `gpu_root/lens/`, keeping per-cell provenance.

    Globbed rather than named: the three lenses were fitted in separate passes (the base anchor in
    one, the two step-70 arms in another) and an aborted pass leaves a ladder whose `cells` is
    empty, which contributes nothing and is not an error.
    """
    paths = sorted((gpu_root / "lens").glob("*/lens_ladder.json"))
    if not paths:
        inputs.note_missing(
            name="Jacobian lens ladders",
            path=gpu_root / "lens/*/lens_ladder.json",
            consequence="the whole lens section: fit quality and the promoted-token tables",
        )
        return
    for path in paths:
        payload = cast("dict[str, Any]", read_json(path))
        cells = cast("dict[str, dict[str, Any]]", payload["cells"])
        inputs.lens_files.append(
            {
                "source": str(path.parent.name),
                "cells": sorted(cells),
                "requested": sorted(cast("list[str]", payload.get("requested_cells", []))),
                "seq_len_plan": payload.get("seq_len_plan", {}),
                "stimuli_sha256": payload.get("stimuli_sha256"),
                "base_model": payload.get("base_model"),
                "merged_note": payload.get("merged_delta_realization_note"),
            }
        )
        for label, cell in cells.items():
            if label in inputs.lens_cells:
                inputs.problems.append(
                    f"lens cell `{label}` is present in more than one ladder under "
                    f"{gpu_root / 'lens'}; keeping the copy from `{path.parent.name}`."
                )
            inputs.lens_cells[label] = {**cell, "_source": path.parent.name}
    for entry in inputs.lens_files:
        # Per file, not against the merged pool: a pass that requested a cell another pass also
        # fitted still banked nothing itself, and whatever it was varying stays unmeasured.
        own = set(cast("list[str]", entry["cells"]))
        unfitted = [label for label in cast("list[str]", entry["requested"]) if label not in own]
        if unfitted:
            inputs.note_missing(
                name=f"lens fits requested by pass `{entry['source']}` and never banked",
                path=gpu_root / "lens" / str(entry["source"]) / "lens_ladder.json",
                consequence=(
                    f"whatever that pass was varying is unmeasured; it asked for "
                    f"{', '.join(f'`{label}`' for label in unfitted)} and wrote no cell"
                ),
            )


def steering_pass_dirs(artifacts_root: Path) -> list[Path]:
    """Every pass under `artifacts_root` holding a `steering/` directory, oldest name first."""
    if not artifacts_root.is_dir():
        return []
    return sorted(
        path for path in artifacts_root.glob(STEERING_PASS_GLOB) if (path / "steering").is_dir()
    )


def load_steering(artifacts_root: Path, gpu_root: Path | None, inputs: Inputs) -> None:
    """Load every generation-steering summary across every pass, plus the forced-choice sweep.

    Every pass, not the newest one: the grid was filled in by two rented boxes and a read of one
    directory renders the cells it happens to hold as the whole of it, so conditions the other box
    ran read as never run. Cells are keyed `<pass>/<cell>` unconditionally rather than only when two
    passes collide -- the passes ran at different thinking budgets and different completion counts,
    so which pass a row came from is part of reading the row, and one cell name can occur in both.

    A steering directory counts as a banked grid only if it carries a summary. Killed and aborted
    runs leave their records behind on purpose (a cap incident's evidence is worth keeping), so the
    ones without a summary are named rather than either read as results or silently skipped.
    """
    passes = steering_pass_dirs(artifacts_root)
    for pass_dir in passes:
        root = pass_dir / "steering"
        for path in sorted(root.glob("*/steering_summary.json")):
            name = f"{pass_dir.name}/{path.parent.name}"
            inputs.steering[name] = cast("dict[str, Any]", read_json(path))
            inputs.steering_records[name] = _count_records(path.parent / "steering_records.jsonl")
        inputs.unsummarised_steering.extend(
            f"{pass_dir.name}/{path.parent.name}"
            for path in root.glob("*/steering_records.jsonl")
            if not (path.parent / "steering_summary.json").is_file()
        )
    inputs.unsummarised_steering.sort()
    if not inputs.steering:
        inputs.note_missing(
            name="generation-steering summaries",
            path=artifacts_root / STEERING_PASS_GLOB / "steering/*/steering_summary.json",
            consequence="the cooperation-rate steering table, its placebo rows and its censoring columns",
        )
    _load_logit_sweep(passes, gpu_root, inputs)


def _load_logit_sweep(passes: Sequence[Path], gpu_root: Path | None, inputs: Inputs) -> None:
    """Read the sweep that chose the intervention, preferring the copy under the lens pass.

    One sweep selected the cell that every steering pass then ran, so a second copy on disk is a
    fact about the trees rather than about the model. The extras are named by path and not read, the
    same way a lens cell present in two ladders is.
    """
    found = [
        pass_dir / "steering" / "logit_sweep.json"
        for pass_dir in passes
        if (pass_dir / "steering" / "logit_sweep.json").is_file()
    ]
    if not found:
        inputs.note_missing(
            name="forced-choice logit sweep",
            path=inputs.artifacts_root / STEERING_PASS_GLOB / "steering/logit_sweep.json",
            consequence="the layer-and-alpha selection table that chose the steering cell",
        )
        return
    preferred = next((path for path in found if path.parent.parent == gpu_root), found[0])
    if len(found) > 1:
        reason = (
            "it sits under the lens pass"
            if preferred.parent.parent == gpu_root
            else "no copy sits under the lens pass and it is first by path"
        )
        others = ", ".join(f"`{display_path(path)}`" for path in found if path != preferred)
        inputs.problems.append(
            f"{len(found)} forced-choice logit sweeps are on disk; reading "
            f"`{display_path(preferred)}` because {reason}, and not reading {others}."
        )
    inputs.logit_sweep = cast("dict[str, Any] | None", read_json(preferred))


def _count_records(path: Path) -> dict[str, int]:
    """Count one steering grid's completions, its truncations, and the overlap that must be empty.

    The censoring story the summary tells rests on truncated completions never parsing: under the
    prefilled-think convention a truncated completion has no visible text at all, so there is
    nothing to parse. That is a claim about the records, so it is checked against them here rather
    than trusted. Streamed line by line because each record carries its full response text.
    """
    counts = {"records": 0, "truncated": 0, "parsed": 0, "truncated_and_parsed": 0}
    if not path.is_file():
        return counts
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = cast("dict[str, Any]", json.loads(line))
            truncated = bool(record["truncated_thinking"])
            parsed = record["parsed_action"] is not None
            counts["records"] += 1
            counts["truncated"] += int(truncated)
            counts["parsed"] += int(parsed)
            counts["truncated_and_parsed"] += int(truncated and parsed)
    return counts


def load_inputs(
    artifacts_root: Path, analysis_root: Path | None = None, gpu_root: Path | None = None
) -> Inputs:
    """Find the newest analysis and GPU passes (unless named) and load every payload they hold."""
    inputs = Inputs(artifacts_root=artifacts_root)
    inputs.analysis_root = analysis_root or find_latest_dir(artifacts_root, ANALYSIS_DIR_GLOB)
    inputs.gpu_root = gpu_root or find_latest_dir(artifacts_root, GPU_DIR_GLOB)
    if inputs.analysis_root is None:
        inputs.note_missing(
            name="CPU analysis pass",
            path=artifacts_root / ANALYSIS_DIR_GLOB,
            consequence="sections 1-4 in full: axis quality, trajectories, displacement, drift",
        )
    else:
        load_trajectories(inputs.analysis_root, inputs)
        load_axes(inputs.analysis_root, inputs)
    # Outside that branch on purpose: the displacement module writes its own pass, so a payload can
    # exist with no CPU analysis pass beside it at all.
    load_displacement(artifacts_root, inputs)
    load_displacement_vs_axes(inputs)
    load_two_path(artifacts_root, inputs)
    # Outside the lens branch too: steering passes are found by their own glob, so the grid survives
    # a tree with no lens pass in it at all.
    load_steering(artifacts_root, inputs.gpu_root, inputs)
    if inputs.gpu_root is None:
        inputs.note_missing(
            name="GPU pass",
            path=artifacts_root / GPU_DIR_GLOB,
            consequence=(
                "section 5 in full: the lens readouts. Section 6 reads the steering passes found "
                "above and is unaffected"
                if inputs.steering
                else "sections 5 and 6 in full: the lens readouts and the steering grid"
            ),
        )
    else:
        load_lenses(inputs.gpu_root, inputs)
    return inputs


# --------------------------------------------------------------------------------------------
# Formatting primitives
# --------------------------------------------------------------------------------------------


def fmt(value: float | None, digits: int = 3) -> str:
    """One unsigned number for a table cell."""
    return EMPTY_CELL if value is None else f"{value:.{digits}f}"


def signed(value: float | None, digits: int = 3) -> str:
    """One signed number for a table cell, so a sign flip is visible at a glance."""
    return EMPTY_CELL if value is None else f"{value:+.{digits}f}"


def flag(value: object) -> str:
    """Render a boolean as `yes`/`no`, or a dash when the payload did not carry it."""
    if value is None:
        return EMPTY_CELL
    return "yes" if value else "no"


def rate_cell(numerator: object, denominator: object) -> str:
    """Render a rate with its denominator attached, always: `0.336 (40/119)`.

    A zero denominator renders as `n/a (0/0)` rather than blank, because "no completion parsed" and
    "this cell was never run" are different facts and only one of them is a measurement.
    """
    if numerator is None or denominator is None:
        return EMPTY_CELL
    k, n = int(cast("int", numerator)), int(cast("int", denominator))
    if n == 0:
        return f"n/a (0/{n})"
    return f"{k / n:.3f} ({k}/{n})"


def percent_cell(numerator: object, denominator: object) -> str:
    """Render a share as a percentage with its denominator: `38.3% (49/128)`."""
    if numerator is None or denominator is None:
        return EMPTY_CELL
    k, n = int(cast("int", numerator)), int(cast("int", denominator))
    if n == 0:
        return f"n/a (0/{n})"
    return f"{100.0 * k / n:.1f}% ({k}/{n})"


def residual_fraction(cos: float | None) -> float | None:
    """Return the share of a displacement off the axis, given its cosine with that axis."""
    if cos is None:
        return None
    return math.sqrt(max(0.0, 1.0 - cos * cos))


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as a markdown table, or say so when there are none.

    The union of keys in order of first appearance is the column order. An empty table says so
    rather than vanishing: a missing section and an empty one are different findings.
    """
    if not rows:
        return "_(no rows)_"
    columns: list[str] = []
    for row in rows:
        columns.extend(str(key) for key in row if str(key) not in columns)
    header = f"| {' | '.join(columns)} |"
    divider = f"| {' | '.join('---' for _ in columns)} |"
    body = [f"| {' | '.join(_cell(row.get(column)) for column in columns)} |" for row in rows]
    return "\n".join([header, divider, *body])


def _cell(value: object) -> str:
    """Format one table cell, escaping the pipe that would otherwise split the row."""
    if value is None:
        return EMPTY_CELL
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def stub(name: str, missing: Iterable[MissingInput]) -> list[str]:
    """Build the body of a section whose inputs were absent: what is gone and what it cost."""
    entries = list(missing)
    if not entries:
        return []
    lines = [f"> **{name} is INCOMPLETE.** These inputs were absent when this ran:", ""]
    lines.extend(
        f"- `{entry.path}` ({entry.name}) -- without it: {entry.consequence}." for entry in entries
    )
    lines.append("")
    return lines


# --------------------------------------------------------------------------------------------
# Trajectory payload accessors
# --------------------------------------------------------------------------------------------


def group_for_arm(payload: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Return the one (arm, set, pooling) group of a one-set, one-pooling trajectory payload."""
    matches = [group for group in payload["groups"].values() if group["arm"] == arm]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one trajectory group for {arm}, found {len(matches)}")
    return cast("dict[str, Any]", matches[0])


def trajectory_arms(payload: Mapping[str, Any]) -> list[str]:
    """Every arm present in one trajectory payload, in a stable order."""
    return sorted({str(group["arm"]) for group in payload["groups"].values()})


def trajectory_steps(payload: Mapping[str, Any]) -> list[int]:
    """Every checkpoint step present in one trajectory payload, ascending."""
    return sorted(
        {int(read["step"]) for group in payload["groups"].values() for read in group["axis_reads"]}
    )


def all_arms(inputs: Inputs) -> list[str]:
    """Every arm any trajectory payload carries -- the arm list the whole document is keyed on."""
    return sorted(
        {arm for payload in inputs.trajectories.values() for arm in trajectory_arms(payload)}
    )


def last_step(inputs: Inputs) -> int | None:
    """Return the furthest checkpoint any trajectory payload reached."""
    steps = {step for payload in inputs.trajectories.values() for step in trajectory_steps(payload)}
    return max(steps) if steps else None


def arm_role(arm: str) -> str:
    """Return the suffix the displacement payload keys its per-arm columns by."""
    return arm.rsplit("-", 1)[-1]


# --------------------------------------------------------------------------------------------
# Section 1: axis quality
# --------------------------------------------------------------------------------------------


def axis_quality_lines(inputs: Inputs) -> list[str]:
    """Build the grid that answers 'is there a direction here at all' before anything else."""
    lines = [
        "## 1. Axis quality: is there a direction here at all",
        "",
        (
            "Every layer, both arms, at the furthest checkpoint reached, pooling "
            f"`{PRIMARY_POOLING}`. `acc` is the diff-of-means direction's held-out accuracy and "
            "`d_plac` is that accuracy minus the best of the matched-norm placebo draws, so a "
            "`d_plac` at or below zero means the axis is indistinguishable from a random direction "
            "of the same norm at that depth. `ceil` is the odd/even split-half cosine, this axis's "
            "own noise ceiling. `clears` is the payload's own two flags: the trained probe against "
            "its shuffled-label null, and the direction against its placebo band. `peak of` names "
            "the arms whose peak layer this is, chosen by accuracy-above-placebo at the anchor."
        ),
        "",
    ]
    missing = [entry for entry in inputs.missing if entry.name.startswith("trajectory report")]
    if not inputs.trajectories:
        return [*lines, *stub("Section 1", missing)]
    lines.extend(stub("Section 1 (partially)", missing))
    step = last_step(inputs)
    for short in SHORT_ORDER:
        payload = inputs.trajectories.get((PRIMARY_POOLING, short))
        if payload is None or step is None:
            continue
        lines.extend(_axis_grid_for_set(payload, short, step))
    lines.extend(_axis_ladder_lines(inputs))
    return lines


def _axis_grid_for_set(payload: Mapping[str, Any], short: str, step: int) -> list[str]:
    """One stimulus set's layer-by-layer grid at one checkpoint, both arms side by side."""
    arms = trajectory_arms(payload)
    groups = {arm: group_for_arm(payload, arm) for arm in arms}
    by_layer: dict[int, dict[str, Mapping[str, Any]]] = {}
    for arm, group in groups.items():
        for read in group["axis_reads"]:
            if int(read["step"]) == step:
                by_layer.setdefault(int(read["layer"]), {})[arm] = read
    rows: list[dict[str, Any]] = []
    for layer in sorted(by_layer):
        # The peak marker goes in its own column, never onto the column names: a per-row column name
        # splits one grid into two half-empty ones, which is what it did before this was fixed.
        peaks = [arm_role(arm) for arm in arms if groups[arm]["peak_layer"] == layer]
        row: dict[str, Any] = {"layer": layer, "peak of": ", ".join(peaks) or EMPTY_CELL}
        for arm in arms:
            read = by_layer[layer].get(arm)
            role = arm_role(arm)
            row[f"{role} acc"] = EMPTY_CELL if read is None else fmt(read["direction_accuracy"])
            row[f"{role} d_plac"] = (
                EMPTY_CELL if read is None else signed(read["accuracy_above_placebo"])
            )
            row[f"{role} ceil"] = EMPTY_CELL if read is None else fmt(read["split_half_cosine"])
            row[f"{role} clears"] = (
                EMPTY_CELL
                if read is None
                else f"{flag(read['clears_null'])}/{flag(read['beats_placebo'])}"
            )
        rows.append(row)
    pairs = "; ".join(
        f"{arm_role(arm)}: {groups[arm]['axis_reads'][0]['n_pairs']} pairs per read, peak layer "
        f"{groups[arm]['peak_layer']}"
        for arm in arms
    )
    return [
        f"### 1.{short}: `{SHORT_TO_SET[short]}` at step {step}",
        "",
        f"{pairs}.",
        "",
        markdown_table(rows),
        "",
    ]


def _axis_ladder_lines(inputs: Inputs) -> list[str]:
    """Each axis's decodability across checkpoints at its own peak layer -- the ladder view."""
    lines = [
        "### 1.ladder: the same axis across checkpoints, at each arm's peak layer",
        "",
        (
            "The grid above is one checkpoint at every depth; this is one depth at every "
            "checkpoint. `probe` is the trained probe's held-out accuracy against `null_max`, the "
            "best of its shuffled-label draws; `acc` against `plac_max` is the diff-of-means "
            "direction against its matched-norm placebo band; `p` is the empirical one-sided value "
            "from that band; `ceil` is the split-half cosine."
        ),
        "",
    ]
    for short in SHORT_ORDER:
        payload = inputs.trajectories.get((PRIMARY_POOLING, short))
        if payload is None:
            continue
        for arm in trajectory_arms(payload):
            group = group_for_arm(payload, arm)
            peak = int(group["peak_layer"])
            rows = [
                {
                    "step": read["step"],
                    "probe": fmt(read["probe_accuracy"]),
                    "null_max": fmt(read["probe_null_accuracy_max"]),
                    "acc": fmt(read["direction_accuracy"]),
                    "plac_max": fmt(read["placebo_accuracy_max"]),
                    "p": fmt(read["accuracy_empirical_p"]),
                    "ceil": fmt(read["split_half_cosine"]),
                }
                for read in group["axis_reads"]
                if int(read["layer"]) == peak
            ]
            lines.extend([f"**{short}, `{arm}`** (layer {peak})", "", markdown_table(rows), ""])
    return lines


# --------------------------------------------------------------------------------------------
# Section 2: trajectories -- drift, held-out projection, cross-arm, behaviour
# --------------------------------------------------------------------------------------------


def trajectory_lines(inputs: Inputs, figures: Mapping[str, Path | None]) -> list[str]:
    """Build the persona-vector view: hold the axis fixed, watch the population slide along it."""
    lines = [
        "## 2. Trajectories: drift from the anchor, and the held-out projection",
        "",
        (
            "The axis is fitted on the even-indexed pairs of the base cell and every checkpoint's "
            "odd pairs are projected onto it, so the trajectory is a read on stimuli the direction "
            "never saw, in raw activation units. `cos_anchor` is this checkpoint's own axis against "
            "the base cell's, which is readable only between its two references: the floor is the "
            "anchor axis against a matched-norm placebo (what unrelated looks like at this "
            "dimensionality, near 1/sqrt(d) rather than zero) and the ceiling is the anchor's own "
            "split-half cosine. `gap` is the held-out projection difference and `gap_plac` is the "
            "same projection onto a matched-norm placebo, which is the flat line the figure shades."
        ),
        "",
    ]
    figure = figures.get(TRAJECTORY_FIGURE)
    if figure is not None:
        lines.extend([f"![held-out projection against checkpoint step]({figure.name})", ""])
    else:
        lines.extend(
            [
                (
                    "> **The trajectory figure was not written** because no `last`-pooling "
                    "trajectory payload was present. The tables below carry whatever did land."
                ),
                "",
            ]
        )
    if not inputs.trajectories:
        return [
            *lines,
            *stub(
                "Section 2", [e for e in inputs.missing if e.name.startswith("trajectory report")]
            ),
            *_drift_heatmap_lines(figures),
        ]
    for short in SHORT_ORDER:
        payload = inputs.trajectories.get((PRIMARY_POOLING, short))
        if payload is None:
            continue
        for arm in trajectory_arms(payload):
            group = group_for_arm(payload, arm)
            peak = int(group["peak_layer"])
            rows = [read for read in group["drift_reads"] if int(read["layer"]) == peak]
            if not rows:
                continue
            table = [
                {
                    "step": read["step"],
                    "cos_anchor": signed(read["cosine_to_anchor"]),
                    "norm_ratio": fmt(read["norm_ratio"]),
                    "gap": signed(read["heldout_projection_gap"]),
                    "gap_plac": signed(read["heldout_projection_gap_placebo"]),
                }
                for read in rows
            ]
            lines.extend(
                [
                    (
                        f"**{short}, `{arm}`** (layer {peak}; floor "
                        f"{signed(rows[0]['cosine_anchor_placebo'])}, ceiling "
                        f"{fmt(rows[0]['anchor_split_half_cosine'])}, held-out pairs "
                        f"{group['n_heldout_pairs']})"
                    ),
                    "",
                    markdown_table(table),
                    "",
                ]
            )
    lines.extend(_drift_heatmap_lines(figures))
    lines.extend(_cross_arm_lines(inputs))
    lines.extend(_behavior_lines(inputs))
    return lines


def _drift_heatmap_lines(figures: Mapping[str, Path | None]) -> list[str]:
    """Embed the layer-by-step drift heatmap with the caption it cannot be read without."""
    figure = figures.get(DRIFT_FIGURE)
    if figure is None:
        return [
            "### 2.heatmap: drift cosine, layer against step",
            "",
            (
                "> **The drift heatmap was not written** because no `last`-pooling trajectory "
                "payload was present."
            ),
            "",
        ]
    return [
        "### 2.heatmap: drift cosine, layer against step",
        "",
        (
            "One panel per stimulus set and arm, the drift cosine to the step-0 anchor as depth "
            "against training step. Two things to hold while reading it. The step-0 column is the "
            "anchor compared with itself, so it is exactly 1.0 by construction and is the top of "
            "every colour scale rather than a measurement. And each panel is scaled to its own "
            "range, printed in its title, because the sets differ by more than an order of "
            "magnitude in how far they drifted at all -- a shared scale would render five of the "
            "six panels flat. What the figure is for is the shape: which depths moved, and at "
            "which step they started."
        ),
        "",
        f"![drift cosine as layer against step, one panel per arm]({figure.name})",
        "",
    ]


def _cross_arm_lines(inputs: Inputs) -> list[str]:
    """Compare the two arms' axes to each other, and difference their projection gaps."""
    lines = [
        "### 2.cross-arm: the same axis fitted in each arm, compared",
        "",
        (
            "`cos_arms` is the cosine between the two arms' full-pair directions at that "
            "checkpoint; `ceil` is the geometric mean of the two arms' split-half cosines there, "
            "which is what 'the same axis measured twice' reads as under this attenuation "
            "heuristic. `gap_diff` subtracts the two held-out projection gaps on the shared base "
            "anchor axis, which is the geometric analogue of a behavioural "
            "difference-in-differences."
        ),
        "",
    ]
    for short in SHORT_ORDER:
        payload = inputs.trajectories.get((PRIMARY_POOLING, short))
        if payload is None:
            continue
        cross = cast("list[dict[str, Any]]", payload.get("cross_arm") or [])
        arms = trajectory_arms(payload)
        if not cross:
            lines.extend(
                [
                    (
                        f"**{short}**: no cross-arm reads. The payload carried none and the "
                        f"per-arm direction tensors needed to rebuild them were not on disk."
                    ),
                    "",
                ]
            )
            continue
        peak = int(group_for_arm(payload, arms[0])["peak_layer"])
        ceilings: dict[tuple[str, int], float] = {}
        for arm in arms:
            for read in group_for_arm(payload, arm)["axis_reads"]:
                if int(read["layer"]) == peak:
                    ceilings[arm, int(read["step"])] = float(read["split_half_cosine"])
        rows: list[dict[str, Any]] = []
        for read in [entry for entry in cross if int(entry["layer"]) == peak]:
            first = ceilings.get((str(read["arm_a"]), int(read["step"])))
            second = ceilings.get((str(read["arm_b"]), int(read["step"])))
            ceiling = (
                math.sqrt(first * second)
                if first is not None and second is not None and first > 0 and second > 0
                else None
            )
            rows.append(
                {
                    "step": read["step"],
                    "cos_arms": signed(read["cosine_between_arms"]),
                    "ceil": fmt(ceiling),
                    f"gap {arm_role(str(read['arm_a']))}": signed(read["projection_gap_a"]),
                    f"gap {arm_role(str(read['arm_b']))}": signed(read["projection_gap_b"]),
                    "gap_diff": signed(read["projection_gap_difference"]),
                }
            )
        lines.extend([f"**{short}** (layer {peak})", "", markdown_table(rows), ""])
    return lines


def _behavior_lines(inputs: Inputs) -> list[str]:
    """Geometry against the behavioural cooperation ladder, both poolings."""
    lines = [
        "### 2.behaviour: does the geometry track the cooperation ladder",
        "",
        (
            "Pearson and Spearman over the checkpoints, per axis and arm. `n` is the number of "
            "rungs, which is small by construction: the behavioural ladder behind it is 16 prompts "
            "a rung, so this is a shape read and not an estimate. Both poolings are shown because "
            "`mean` dilutes a short continuation against a long shared stem and is the "
            "does-the-ordering-survive check rather than a second measurement."
        ),
        "",
    ]
    rows: list[dict[str, Any]] = []
    for pooling in POOLINGS:
        for short in SHORT_ORDER:
            payload = inputs.trajectories.get((pooling, short))
            if payload is None:
                continue
            for label, entry in sorted(payload["behavior_correlations"].items()):
                gap, drift = entry["projection_gap"], entry["drift_cosine"]
                rows.append(
                    {
                        "pooling": pooling,
                        "axis": short,
                        "arm": label.split("|")[0],
                        "n": gap["n_points"],
                        "gap r (P/S)": f"{fmt(gap['pearson'], 2)}/{fmt(gap['spearman'], 2)}",
                        "drift r (P/S)": f"{fmt(drift['pearson'], 2)}/{fmt(drift['spearman'], 2)}",
                    }
                )
    return [*lines, markdown_table(rows), ""]


# --------------------------------------------------------------------------------------------
# Section 2b: stratified axis quality and axis-to-axis geometry
# --------------------------------------------------------------------------------------------


def axes_geometry_lines(inputs: Inputs) -> list[str]:
    """Build the stratified quality and axis-to-axis cosine tables from the `interp_axes` pass."""
    lines = [
        "## 2b. Stratified axis quality, and axis against axis",
        "",
        (
            "`matched-column` is the confound-clean stratum, where both sides of a pair cite the "
            "same payoff cells; an axis that holds pooled and collapses there is measuring which "
            "cells the text cited. The within-axis stratum contrast below each table is the "
            "sharpest single number: if one axis is the same direction in both of its halves, "
            "cells-citation is not what it measures."
        ),
        "",
    ]
    missing = [entry for entry in inputs.missing if "axis-to-axis" in entry.name]
    if inputs.axes is None:
        return [*lines, *stub("Section 2b", missing)]
    cells = cast("dict[str, dict[str, Any]]", inputs.axes["cells"])
    for label in sorted(cells):
        lines.extend(_strata_lines_for_cell(label, cells[label]))
    lines.extend(_axis_pair_lines(cells))
    return lines


def _strata_lines_for_cell(label: str, cell: Mapping[str, Any]) -> list[str]:
    """One cell's per-set peak layer, every stratum at it, and the within-axis stratum contrast."""
    quality = cast("list[dict[str, Any]]", cell["quality"])
    pooled = [
        entry
        for entry in quality
        if entry["pooling"] == PRIMARY_POOLING
        and entry["stratum"] == "pooled"
        and entry["direction_accuracy"] is not None
    ]
    if not pooled:
        return [f"**{label}**: no pooled `{PRIMARY_POOLING}` reads in the payload.", ""]
    peak_by_set: dict[str, int] = {}
    for name in SET_SHORT:
        candidates = [entry for entry in pooled if entry["stimulus_set"] == name]
        if candidates:
            best = max(
                candidates,
                key=lambda entry: entry["direction_accuracy"] - entry["placebo_accuracy_max"],
            )
            peak_by_set[name] = int(best["layer"])
    rows: list[dict[str, Any]] = []
    for name, short in SET_SHORT.items():
        layer = peak_by_set.get(name)
        if layer is None:
            continue
        at_layer = [
            entry
            for entry in quality
            if entry["pooling"] == PRIMARY_POOLING
            and entry["stimulus_set"] == name
            and int(entry["layer"]) == layer
        ]
        rows.extend(
            {
                "axis": short,
                "stratum": entry["stratum"],
                "layer": layer,
                "n_pairs": entry["n_pairs"],
                "acc": fmt(entry["direction_accuracy"]),
                "plac_max": fmt(entry["placebo_accuracy_max"]),
                "p": fmt(entry["accuracy_empirical_p"]),
                "ceil": fmt(entry["split_half_cosine"]),
            }
            for entry in at_layer
        )
    lines = [
        f"**{label}** (per-set peak layer from the pooled stratum)",
        "",
        markdown_table(rows),
        "",
    ]
    contrasts = [
        entry
        for entry in cast("list[dict[str, Any]]", cell["stratum_contrasts"])
        if entry["pooling"] == PRIMARY_POOLING
        and int(entry["layer"]) == peak_by_set.get(str(entry["stimulus_set"]), -1)
    ]
    lines.extend(
        f"within-{SET_SHORT[str(entry['stimulus_set'])]} stratum contrast at layer {entry['layer']}: "
        f"cos({entry['stratum_a']}, {entry['stratum_b']}) = {signed(entry['cosine_real'])} "
        f"(floor_max {fmt(entry['placebo_abs_cosine_max'])}, ceilings "
        f"{fmt(entry['split_half_a'])}/{fmt(entry['split_half_b'])})"
        for entry in contrasts
    )
    if contrasts:
        lines.append("")
    return lines


def _axis_pair_lines(cells: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Axis-against-axis cosines at the base cell's coarse peak layer, for every cell."""
    base = cells.get(f"{BASE_ARM}/step-0")
    if base is None:
        return [
            "**Axis against axis**: no base cell in the payload, so there is no shared layer to read at.",
            "",
        ]
    pooled = [
        entry
        for entry in cast("list[dict[str, Any]]", base["quality"])
        if entry["pooling"] == PRIMARY_POOLING
        and entry["stratum"] == "pooled"
        and entry["stimulus_set"] == SHORT_TO_SET["coarse"]
        and entry["direction_accuracy"] is not None
    ]
    if not pooled:
        return ["**Axis against axis**: the base cell carried no pooled coarse read.", ""]
    layer = int(
        max(pooled, key=lambda entry: entry["direction_accuracy"] - entry["placebo_accuracy_max"])[
            "layer"
        ]
    )
    rows: list[dict[str, Any]] = []
    for label in sorted(cells):
        for pair in cast("list[dict[str, Any]]", cells[label]["axis_pairs"]):
            if pair["pooling"] != PRIMARY_POOLING or int(pair["layer"]) != layer:
                continue
            rows.append(
                {
                    "cell": label,
                    "pair": f"{SET_SHORT[str(pair['set_a'])]}~{SET_SHORT[str(pair['set_b'])]}",
                    "stratum": pair["stratum"],
                    "cos": signed(pair["cosine_real"]),
                    "floor_max": fmt(pair["placebo_abs_cosine_max"]),
                    "ceil_a": fmt(pair["split_half_a"]),
                    "ceil_b": fmt(pair["split_half_b"]),
                }
            )
    return [
        f"**Axis against axis** (layer {layer}, the base cell's coarse peak)",
        "",
        markdown_table(rows),
        "",
    ]


# --------------------------------------------------------------------------------------------
# Section 3: displacement
# --------------------------------------------------------------------------------------------


def displacement_lines(inputs: Inputs) -> list[str]:
    """Question (a): did the two arms move orthogonally, or oppositely along a shared axis."""
    lines = [
        "## 3. Displacement from base: did the arms move orthogonally",
        "",
        (
            "The population-mean displacement per arm, `mean over stimuli of (h_arm - h_base)`, "
            "which is the object the pooled cosine of 0.09 was about. `cos_arms` is the cosine "
            "between the two arms' displacements, and it is unreadable without the two columns "
            "beside it: `floor_max` is the largest absolute cosine over the matched-norm placebo "
            "draws, which is what unrelated reads as at this dimensionality, and the ceiling is each "
            "arm's own split-half reliability, which is what one displacement measured twice reads "
            "as. `xhalf` recomputes the cross-arm cosine across disjoint halves, so no stimulus row "
            "is shared between the two sides of it."
        ),
        "",
    ]
    missing = [entry for entry in inputs.missing if "displacement" in entry.name]
    if inputs.displacement is None or inputs.displacement_source is None:
        return [*lines, *stub("Section 3", missing)]
    lines.extend(_payload_provenance_lines(inputs))
    if inputs.displacement_source.shape == SHAPE_MODULE:
        return [*lines, *module_displacement_lines(inputs)]
    lines.extend(stub("Section 3 (partially)", [e for e in missing if "against-axes" in e.name]))
    reads = cast("list[dict[str, Any]]", inputs.displacement["reads"])
    step = max(int(read["step"]) for read in reads)
    pooled = [
        read
        for read in reads
        if read["pooling"] == PRIMARY_POOLING
        and len(read["sets"]) == len(SET_SHORT)
        and int(read["step"]) == step
    ]
    rows = [
        {
            "layer": read["layer"],
            "cos_arms": signed(read["cos_arms"]),
            "xhalf": signed(
                sum(cast("list[float]", read["cos_arms_crosshalf"]))
                / len(read["cos_arms_crosshalf"])
            ),
            "floor_max": fmt(read["placebo_abs_cos_max"]),
            "ceil": fmt(read["same_axis_ceiling"]),
            "norm_group": fmt(read["norm_group"]),
            "norm_self": fmt(read["norm_self"]),
            "norm_diff": fmt(read["norm_arm_minus_arm"]),
            "rel_group": fmt(_relative(read["norm_group"], read["base_mean_norm"])),
            "rel_self": fmt(_relative(read["norm_self"], read["base_mean_norm"])),
        }
        for read in sorted(pooled, key=lambda read: int(read["layer"]))
    ]
    lines.extend(
        [
            f"### 3.a: all three sets pooled, `{PRIMARY_POOLING}` pooling, step {step}, per layer",
            "",
            (
                "`norm_*` are the displacement magnitudes in raw activation units and `rel_*` "
                "divide them by the base cell's mean residual norm at that layer, so 'how far did "
                "it move' is scale-free and comparable across depth."
            ),
            "",
            markdown_table(rows),
            "",
        ]
    )
    lines.extend(_displacement_extrema_lines(reads, step))
    lines.extend(_along_axis_lines(inputs, step))
    lines.extend(_nulls_lines(inputs))
    return lines


def _relative(norm: object, base_norm: object) -> float | None:
    """Return a displacement magnitude as a fraction of the base cell's mean residual norm."""
    if norm is None or base_norm is None or float(cast("float", base_norm)) == 0:
        return None
    return float(cast("float", norm)) / float(cast("float", base_norm))


def _displacement_extrema_lines(reads: Sequence[Mapping[str, Any]], step: int) -> list[str]:
    """Where the arms are most opposed, per checkpoint and per stimulus set."""
    pooled = [
        read
        for read in reads
        if read["pooling"] == PRIMARY_POOLING and len(read["sets"]) == len(SET_SHORT)
    ]
    steps = sorted({int(read["step"]) for read in pooled})
    by_step = [
        {
            "step": one,
            "layer": best["layer"],
            "cos_arms": signed(best["cos_arms"]),
            "floor_max": fmt(best["placebo_abs_cos_max"]),
            "ceil": fmt(best["same_axis_ceiling"]),
        }
        for one in steps
        if (
            best := min(
                (read for read in pooled if int(read["step"]) == one),
                key=lambda read: read["cos_arms"],
            )
        )
    ]
    per_set: list[dict[str, Any]] = []
    for name, short in SET_SHORT.items():
        rows = [
            read
            for read in reads
            if read["pooling"] == PRIMARY_POOLING
            and read["sets"] == [name]
            and int(read["step"]) == step
        ]
        if not rows:
            continue
        best = min(rows, key=lambda read: read["cos_arms"])
        per_set.append(
            {
                "set": short,
                "layer": best["layer"],
                "cos_arms": signed(best["cos_arms"]),
                "floor_max": fmt(best["placebo_abs_cos_max"]),
                "ceil": fmt(best["same_axis_ceiling"]),
            }
        )
    return [
        "### 3.b: the most-opposed layer, per checkpoint and per stimulus set",
        "",
        (
            "The pooled-over-layers number this section replaces averaged over depths where "
            "nothing moved. These two tables are where the opposition actually sits: the first "
            "reads the onset across training, the second asks whether one stimulus set carries it."
        ),
        "",
        markdown_table(by_step),
        "",
        f"Per stimulus set at step {step}, most-opposed layer:",
        "",
        markdown_table(per_set),
        "",
    ]


def _along_axis_lines(inputs: Inputs, step: int) -> list[str]:
    """Split each arm's displacement into its component along an axis and the residual."""
    if inputs.displacement_vs_axes is None:
        return []
    lines = [
        "### 3.c: the along-axis / residual split",
        "",
        (
            "`cos` is each arm's displacement against that cell's own axis, which is also the "
            "along-axis share of the displacement; `resid` is `sqrt(1 - cos^2)`, the share that is "
            "off the axis. `diff` subtracts the two arms' along-axis components, so two arms moving "
            "oppositely along one shared axis reads as a large `diff` even where `cos_arms` in 3.a "
            "is near zero. `floor_max` is the matched-norm placebo band for the same cosine. These "
            "are different claims and the pooled cosine cannot distinguish them."
        ),
        "",
    ]
    reads = [
        read
        for read in inputs.displacement_vs_axes
        if read["pooling"] == PRIMARY_POOLING and int(read["step"]) == step
    ]
    for name, short in SET_SHORT.items():
        rows: list[dict[str, Any]] = []
        for read in sorted(reads, key=lambda read: int(read["layer"])):
            entry = cast("dict[str, Any]", read["cos_axis"]).get(name)
            if entry is None:
                continue
            group_cos, self_cos = entry["group_displacement"], entry["self_displacement"]
            rows.append(
                {
                    "layer": read["layer"],
                    "group cos": signed(group_cos),
                    "group resid": fmt(residual_fraction(group_cos)),
                    "self cos": signed(self_cos),
                    "self resid": fmt(residual_fraction(self_cos)),
                    "diff": signed(entry["difference"]),
                    "floor_max": fmt(read["placebo_abs_cos_max"]),
                }
            )
        lines.extend(
            [f"**{short} axis** (step {step}, `{PRIMARY_POOLING}`)", "", markdown_table(rows), ""]
        )
    return lines


def _nulls_lines(inputs: Inputs) -> list[str]:
    """Report the label-shuffle nulls: what this pipeline reads when the labels are noise."""
    if inputs.displacement is None:
        return []
    nulls = cast("list[dict[str, Any]]", inputs.displacement.get("nulls") or [])
    if not nulls:
        return [
            "### 3.d: deliberate nulls",
            "",
            (
                "> The displacement payload carried no `nulls` block, so this pipeline has not been "
                "shown to read a null when the labels are shuffled. A pipeline that cannot produce "
                "a null cannot produce a positive."
            ),
            "",
        ]
    rows = [
        {
            "pooling": null["pooling"],
            "set": SET_SHORT.get(str(null["set"]), str(null["set"])),
            "layer": null["layer"],
            "flipped": f"{null['n_flipped']}/{null['n_pairs']}",
            "real acc": fmt(null["real_direction_accuracy"]),
            "null acc": fmt(null["null_direction_accuracy"]),
            "null plac (mean/max)": (
                f"{fmt(null['null_placebo_accuracy_mean'])}/{fmt(null['null_placebo_accuracy_max'])}"
            ),
            "cos(null, real)": signed(null["cos_null_to_real"]),
            "floor (mean/max)": (
                f"{fmt(null['placebo_floor_abs_cos_mean'])}/{fmt(null['placebo_floor_abs_cos_max'])}"
            ),
        }
        for null in nulls
    ]
    return [
        "### 3.d: deliberate nulls (pair labels shuffled before the direction is fitted)",
        "",
        (
            "The sabotage of the whole geometry leg, run and recorded rather than asserted. With "
            "the side labels shuffled, the fitted direction should land inside its own placebo band "
            "and its cosine to the real axis should sit at the placebo floor. A `cos(null, real)` "
            "well above that floor means the axis survives label destruction, which would mean it "
            "is not measuring the labels."
        ),
        "",
        markdown_table(rows),
        "",
    ]


# --------------------------------------------------------------------------------------------
# Section 3 rendered from the module payload: two split-halves, two denominators, the axis split
# --------------------------------------------------------------------------------------------

# Displacements here are ~1e-3 of a residual norm, so three decimals renders every one of them as
# 0.001 and every relative norm as 0.001 -- the columns exist to be compared across depth.
DISPLACEMENT_DIGITS = 5

MODULE_COLUMN_NOTE = (
    "This payload reports **two** split-half reliabilities per read, and which one is the ceiling "
    "matters. `ceil pairs` splits disjoint sets of matched pairs with both sides in each half, and is "
    "the column to read; `ceil rows` splits on stored row parity, which on this corpus interleaves the "
    "two sides of each pair and is therefore a *side* split, conflating measurement noise with any "
    "genuine side-dependence of the displacement. The scratch pass reported only the row split and "
    "called it split-half. `xhalf pairs` and `xhalf rows` are the cross-arm cosine recomputed across "
    "the two halves of each split, so no stimulus row is shared between the two sides of it."
)

MODULE_AXIS_NOTE = (
    "`cos` is each arm's displacement against that stimulus set's contrast axis and is also the "
    "along-axis share of the displacement; `proj` is the same component signed, in raw activation "
    "units, which is what distinguishes two arms leaning oppositely along one axis from two unrelated "
    "movements; `resid` is `sqrt(1 - cos^2)`, the share off the axis. The `-` target is the difference "
    "of the two arms' displacements against the same axis, so a large component there with a "
    "near-zero `cos_arms` in 3.a is the shared-axis reading rather than the orthogonal one. "
    "`floor_max` is the read's matched-norm placebo band for a cosine at this dimensionality."
)


def _payload_provenance_lines(inputs: Inputs) -> list[str]:
    """Name the displacement payload this section is rendered from, and any it is not reading."""
    source = inputs.displacement_source
    if source is None:
        return []
    sentence = f"Rendered from {source.describe()}."
    if inputs.displacement_unused:
        sentence += (
            " Also on disk and NOT read here: "
            + "; ".join(other.describe() for other in inputs.displacement_unused)
            + f". Where both shapes are present the {SHAPE_NAMES[SHAPE_MODULE]} one wins, because it "
            f"carries both split-half reliabilities, both residual-norm denominators, the along-axis "
            f"split the scratch shape kept in a separate file, and an arbitrary number of arms."
        )
    return [sentence, ""]


def _arm_labels(arms: Sequence[str]) -> dict[str, str]:
    """Short column label per arm: its role suffix, or the full name when two arms would collide."""
    roles = [arm_role(arm) for arm in arms]
    if len(set(roles)) == len(roles):
        return dict(zip(arms, roles, strict=True))
    return {arm: arm for arm in arms}


def _mean_of(values: object) -> float | None:
    """Mean of a payload list of floats, or None where the payload carried none."""
    if values is None:
        return None
    items = cast("list[float]", values)
    return sum(items) / len(items) if items else None


def _module_selection(
    reads: Sequence[Mapping[str, Any]], *, set_group: str, step: int | None = None
) -> list[dict[str, Any]]:
    """One unstratified selection's reads at the primary pooling, layer-ascending.

    Stratification and stratum are both pinned to the pooled value: a stratum read carries the same
    (set_group, step, layer) coordinates as its pooled parent, so a filter that forgot them would put
    several reads of different populations on the same table row.
    """
    selected = [
        cast("dict[str, Any]", read)
        for read in reads
        if read["pooling"] == PRIMARY_POOLING
        and read["set_group"] == set_group
        and read["stratification"] == POOLED_SELECTION
        and read["stratum"] == POOLED_SELECTION
        and (step is None or int(read["step"]) == step)
    ]
    return sorted(selected, key=lambda read: int(read["layer"]))


def _module_pair_keys(read: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Every arm pair one read compares, in payload order."""
    return [
        (str(pair["arm_a"]), str(pair["arm_b"]))
        for pair in cast("list[dict[str, Any]]", read["arm_pairs"])
    ]


def _module_pair(read: Mapping[str, Any], pair_key: tuple[str, str]) -> dict[str, Any] | None:
    """One read's entry for a named arm pair, or None when this read does not carry that pair."""
    for pair in cast("list[dict[str, Any]]", read["arm_pairs"]):
        if (str(pair["arm_a"]), str(pair["arm_b"])) == pair_key:
            return pair
    return None


def _module_pair_row(
    read: Mapping[str, Any], pair_key: tuple[str, str], labels: Mapping[str, str]
) -> dict[str, Any]:
    """One layer's row of the pooled cross-arm table, both ceilings and both denominators on it."""
    pair = _module_pair(read, pair_key)
    if pair is None:
        return {"layer": read["layer"], "cos_arms": MISSING_CELL}
    arms = {str(entry["arm"]): entry for entry in cast("list[dict[str, Any]]", read["arms"])}
    row: dict[str, Any] = {
        "layer": read["layer"],
        "cos_arms": signed(pair["cosine_real"]),
        "xhalf pairs": signed(_mean_of(pair["cosine_cross_halves_pairs"])),
        "xhalf rows": signed(_mean_of(pair["cosine_cross_halves_rows"])),
        "floor_max": fmt(read["placebo_abs_cosine_max"]),
        "ceil pairs": fmt(pair["same_axis_ceiling_pairs"]),
        "ceil rows": fmt(pair["same_axis_ceiling_rows"]),
    }
    for arm in pair_key:
        row[f"norm {labels[arm]}"] = fmt(arms[arm]["displacement_norm"], DISPLACEMENT_DIGITS)
    row["norm_diff"] = fmt(pair["difference_norm"], DISPLACEMENT_DIGITS)
    for arm in pair_key:
        entry = arms[arm]
        row[f"rel {labels[arm]} (of-mean/of-rows)"] = (
            f"{fmt(entry['relative_to_residual_mean_vector'], DISPLACEMENT_DIGITS)}/"
            f"{fmt(entry['relative_to_residual_row_norm_mean'], DISPLACEMENT_DIGITS)}"
        )
    return row


def _module_selection_footnote(reads: Sequence[Mapping[str, Any]]) -> list[str]:
    """State the selection's size, whether its row split is a side split, and both denominators."""
    first = reads[0]
    reasons = sorted(
        {
            str(entry["unavailable_reason"])
            for read in reads
            for entry in cast("list[dict[str, Any]]", read["arms"])
            if entry["unavailable_reason"]
        }
    )
    lines = [
        (
            f"{first['n_rows']} rows in {first['n_pairs']} matched pairs per read; "
            f"`row_parity_equals_side` = {flag(first['row_parity_equals_side'])}, which is the "
            f"payload's own statement of whether the row split above is a side split. The two `rel` "
            f"denominators are both the base cell's residual scale at that layer: `of-mean` divides "
            f"by the norm of the mean state (`residual_mean_vector_norm`, "
            f"{fmt(first['residual_mean_vector_norm'], 3)} at layer {first['layer']}, which is what "
            f"the scratch read used) and `of-rows` by the mean of the per-row norms "
            f"(`residual_row_norm_mean`, {fmt(first['residual_row_norm_mean'], 3)} there). The first "
            f"shrinks whenever the states point in different directions, which is a fact about the "
            f"population and not about its scale, so neither is allowed to stand alone as `rel`."
        ),
        "",
    ]
    if reasons:
        lines.extend([f"> Reads carrying no split-half ceiling: {'; '.join(reasons)}.", ""])
    return lines


def _module_pooled_lines(reads: Sequence[Mapping[str, Any]], step: int) -> list[str]:
    """3.a for the module shape: one table per arm pair, both ceilings and both denominators."""
    lines = [
        f"### 3.a: every set pooled, `{PRIMARY_POOLING}` pooling, step {step}, per layer",
        "",
    ]
    pooled = _module_selection(reads, set_group=POOLED_SELECTION, step=step)
    if not pooled:
        return [*lines, "> The payload carried no pooled read at this checkpoint.", ""]
    arms = [str(entry["arm"]) for entry in cast("list[dict[str, Any]]", pooled[0]["arms"])]
    labels = _arm_labels(arms)
    lines.extend([MODULE_COLUMN_NOTE, ""])
    for pair_key in _module_pair_keys(pooled[0]):
        rows = [_module_pair_row(read, pair_key, labels) for read in pooled]
        lines.extend(
            [
                f"**{labels[pair_key[0]]} against {labels[pair_key[1]]}**",
                "",
                markdown_table(rows),
                "",
            ]
        )
    lines.extend(_module_selection_footnote(pooled))
    return lines


def _module_first_cosine(read: Mapping[str, Any]) -> float:
    """Return the first arm pair's cross-arm cosine, the only pair on a two-arm ladder."""
    pairs = cast("list[dict[str, Any]]", read["arm_pairs"])
    return float(pairs[0]["cosine_real"]) if pairs else float("nan")


def _module_extrema_row(read: Mapping[str, Any]) -> dict[str, Any]:
    """Return the numbers a most-opposed layer is read by: floor and ceiling beside the cosine."""
    pair = cast("list[dict[str, Any]]", read["arm_pairs"])[0]
    return {
        "layer": read["layer"],
        "cos_arms": signed(pair["cosine_real"]),
        "floor_max": fmt(read["placebo_abs_cosine_max"]),
        "ceil pairs": fmt(pair["same_axis_ceiling_pairs"]),
        "ceil rows": fmt(pair["same_axis_ceiling_rows"]),
    }


def _module_extrema_lines(reads: Sequence[Mapping[str, Any]], step: int) -> list[str]:
    """3.b for the module shape: where the arms are most opposed, per checkpoint and per set."""
    pooled = [
        read for read in _module_selection(reads, set_group=POOLED_SELECTION) if read["arm_pairs"]
    ]
    by_step: list[dict[str, Any]] = []
    for one in sorted({int(read["step"]) for read in pooled}):
        here = [read for read in pooled if int(read["step"]) == one]
        by_step.append({"step": one, **_module_extrema_row(min(here, key=_module_first_cosine))})
    set_groups = sorted({str(read["set_group"]) for read in reads} - {POOLED_SELECTION})
    per_set: list[dict[str, Any]] = []
    for name in set_groups:
        here = [
            read
            for read in _module_selection(reads, set_group=name, step=step)
            if read["arm_pairs"]
        ]
        if not here:
            continue
        per_set.append(
            {
                "set": SET_SHORT.get(name, name),
                **_module_extrema_row(min(here, key=_module_first_cosine)),
            }
        )
    return [
        "### 3.b: the most-opposed layer, per checkpoint and per stimulus set",
        "",
        (
            "The pooled-over-layers number this section replaces averaged over depths where nothing "
            "moved. These two tables are where the opposition actually sits: the first reads the "
            "onset across training, the second asks whether one stimulus set carries it."
        ),
        "",
        markdown_table(by_step),
        "",
        f"Per stimulus set at step {step}, most-opposed layer:",
        "",
        markdown_table(per_set),
        "",
    ]


def _module_target_labels(read: Mapping[str, Any]) -> dict[str, str]:
    """Column label per axis-split target: each arm, and each arm-minus-arm difference."""
    arms = [str(entry["arm"]) for entry in cast("list[dict[str, Any]]", read["arms"])]
    labels = _arm_labels(arms)
    targets = dict(labels)
    for arm_a, arm_b in _module_pair_keys(read):
        targets[f"{arm_a}-minus-{arm_b}"] = f"{labels[arm_a]}-{labels[arm_b]}"
    return targets


def _module_components(
    read: Mapping[str, Any], *, axis_set: str, axis_reference: str
) -> dict[str, dict[str, Any]]:
    """One read's axis components for one (set, reference), keyed by target."""
    return {
        str(entry["target"]): entry
        for entry in cast("list[dict[str, Any]]", read["axis_components"])
        if entry["axis_set"] == axis_set and entry["axis_reference"] == axis_reference
    }


def _module_reference_agreement(
    reads: Sequence[Mapping[str, Any]], axis_set: str, references: Sequence[str]
) -> str | None:
    """Return how far the axis references disagree, so tabulating one of them is a stated choice."""
    if len(references) < MIN_CELLS_FOR_A_COMPARISON:
        return None
    first, second = references[0], references[1]
    gaps: list[float] = []
    for read in reads:
        anchored = _module_components(read, axis_set=axis_set, axis_reference=first)
        own = _module_components(read, axis_set=axis_set, axis_reference=second)
        gaps.extend(
            abs(float(anchored[target]["cosine_to_axis"]) - float(own[target]["cosine_to_axis"]))
            for target in sorted(set(anchored) & set(own))
        )
    if not gaps:
        return None
    return (
        f"Against each arm's own checkpoint axis (`{second}`) rather than the base anchor's "
        f"(`{first}`), the same cosines differ by at most {fmt(max(gaps), 4)} over every layer and "
        f"target, so the table above is not sensitive to which reference is used. Both are in the "
        f"payload; only the anchor's is tabulated, because two near-identical tables read as two "
        f"measurements."
    )


def _module_axis_lines(inputs: Inputs, reads: Sequence[Mapping[str, Any]], step: int) -> list[str]:
    """3.c for the module shape: the along-axis / residual split, from the payload's own components."""
    pooled = _module_selection(reads, set_group=POOLED_SELECTION, step=step)
    if not pooled:
        return []
    references = sorted(
        {
            str(entry["axis_reference"])
            for read in pooled
            for entry in cast("list[dict[str, Any]]", read["axis_components"])
        }
    )
    if not references:
        return []
    lines = ["### 3.c: the along-axis / residual split", "", MODULE_AXIS_NOTE, ""]
    targets = _module_target_labels(pooled[0])
    axis_sets = sorted(
        {
            str(entry["axis_set"])
            for read in pooled
            for entry in cast("list[dict[str, Any]]", read["axis_components"])
        },
        key=lambda name: (
            SHORT_ORDER.index(SET_SHORT[name]) if name in SET_SHORT else len(SET_SHORT)
        ),
    )
    for axis_set in axis_sets:
        rows: list[dict[str, Any]] = []
        for read in pooled:
            components = _module_components(read, axis_set=axis_set, axis_reference=references[0])
            if not components:
                continue
            row: dict[str, Any] = {"layer": read["layer"]}
            for target in sorted(components, key=lambda name: targets.get(name, name)):
                label = targets.get(target, target)
                cos = float(components[target]["cosine_to_axis"])
                row[f"{label} cos"] = signed(cos)
                row[f"{label} proj"] = signed(components[target]["projection"], DISPLACEMENT_DIGITS)
                row[f"{label} resid"] = fmt(residual_fraction(cos))
            row["floor_max"] = fmt(read["placebo_abs_cosine_max"])
            rows.append(row)
        lines.extend(
            [
                (
                    f"**{SET_SHORT.get(axis_set, axis_set)} axis** (step {step}, "
                    f"`{PRIMARY_POOLING}`, axis reference `{references[0]}`)"
                ),
                "",
                markdown_table(rows),
                "",
            ]
        )
        agreement = _module_reference_agreement(pooled, axis_set, references)
        if agreement is not None:
            lines.extend([agreement, ""])
    if inputs.displacement_vs_axes is not None:
        lines.extend(
            [
                (
                    "> A separate `displacement_vs_axes.json` from the scratch pass is also on disk "
                    "and is NOT read here: the split above comes from the same pass as every other "
                    "number in this section, and mixing the two would put two passes' geometry in "
                    "one section without saying so."
                ),
                "",
            ]
        )
    return lines


def _module_nulls_lines(payload: Mapping[str, Any]) -> list[str]:
    """3.d for the module shape: the shuffled-label null, and the exact base-against-base null."""
    heading = [
        "### 3.d: deliberate nulls (pair labels shuffled before the direction is fitted)",
        "",
    ]
    nulls = cast("dict[str, Any]", payload["shuffled_label_nulls"])
    reads = cast("list[dict[str, Any]]", nulls["reads"])
    if not reads:
        return [
            *heading,
            (
                "> The payload's `shuffled_label_nulls` block carried no reads, so this pipeline has "
                "not been shown to read a null when the labels are shuffled. A pipeline that cannot "
                "produce a null cannot produce a positive."
            ),
            "",
        ]
    rows = [
        {
            "pooling": null["pooling"],
            "set": SET_SHORT.get(str(null["stimulus_set"]), str(null["stimulus_set"])),
            "layer": null["layer"],
            "flipped": f"{null['n_flipped']}/{null['n_pairs']}",
            "real acc": fmt(null["real_direction_accuracy"]),
            "null acc": fmt(null["null_direction_accuracy"]),
            "null plac (mean/max)": (
                f"{fmt(null['null_placebo_accuracy_mean'])}/{fmt(null['null_placebo_accuracy_max'])}"
            ),
            "clears placebo": flag(null["clears_placebo"]),
            "cos(null, real)": signed(null["cosine_null_to_real"]),
            "floor (mean/max)": (
                f"{fmt(null['placebo_floor_abs_cosine_mean'])}/"
                f"{fmt(null['placebo_floor_abs_cosine_max'])}"
            ),
        }
        for null in reads
    ]
    lines = [
        *heading,
        (
            "The sabotage of the whole geometry leg, run and recorded rather than asserted. With the "
            "side labels shuffled, the fitted direction should land inside its own placebo band and "
            "its cosine to the real axis should sit at the placebo floor. `clears placebo` = yes "
            "means a direction fitted on labels carrying no information beat every matched-norm "
            "placebo, which at these draw counts happens about one time in `n_placebos + 1` by luck "
            "and otherwise means the placebo band is not a control."
        ),
        "",
        markdown_table(rows),
        "",
    ]
    skipped = cast("list[dict[str, Any]]", nulls["skipped"])
    if skipped:
        lines.extend(
            [
                "> Nulls the pass could not run: "
                + "; ".join(
                    f"`{entry['stimulus_set']}` ({entry['pooling']}, {entry['reason']})"
                    for entry in skipped
                )
                + ".",
                "",
            ]
        )
    base_null = cast("dict[str, Any]", payload["base_vs_base_null"])
    lines.extend(
        [
            (
                f"The payload's second null is exact rather than distributional: the whole "
                f"displacement machinery run with the base cell standing in for an arm, over "
                f"{base_null['n_checks']} checks, worst absolute displacement "
                f"{base_null['max_abs_displacement']}. A paired row-wise difference of a cell with "
                f"itself is zero by construction, so anything else would mean the two sides of the "
                f"subtraction are not the same rows -- and the producer raises instead of reporting."
            ),
            "",
        ]
    )
    return lines


def _module_stratified_lines(
    reads: Sequence[Mapping[str, Any]], step: int, layer: int
) -> list[str]:
    """3.e for the module shape: every stratified selection at one layer, beside its pooled parent.

    One layer rather than all of them, and the most-opposed pooled layer at that: the question a
    stratified read answers is whether the opposition survives holding the cited payoff cells fixed,
    and that is a comparison at the depth where there is something to survive.
    """
    here = [
        read
        for read in reads
        if read["pooling"] == PRIMARY_POOLING
        and int(read["step"]) == step
        and int(read["layer"]) == layer
        and read["arm_pairs"]
    ]
    rows = [
        {
            "set group": SET_SHORT.get(str(read["set_group"]), str(read["set_group"])),
            "stratification": read["stratification"],
            "stratum": read["stratum"],
            "pairs": read["n_pairs"],
            **{key: value for key, value in _module_extrema_row(read).items() if key != "layer"},
            "side split": flag(read["row_parity_equals_side"]),
        }
        for read in sorted(
            here,
            key=lambda read: (
                str(read["set_group"]) != POOLED_SELECTION,
                str(read["set_group"]),
                str(read["stratification"]),
                str(read["stratum"]),
            ),
        )
    ]
    return [
        f"### 3.e: every selection at layer {layer}, step {step}",
        "",
        (
            "The same cross-arm read inside each stratum of the corpus, at the most-opposed pooled "
            "layer from 3.b. `matched-column` is the confound-clean stratum, where both sides of a "
            "pair cite the same payoff cells: a displacement that holds pooled and collapses there is "
            "measuring which cells the text cited. `side split` is that selection's own "
            "`row_parity_equals_side`, so a `ceil rows` read against a stratum where the parity is no "
            "longer a side split is not the same quantity as the one above it."
        ),
        "",
        markdown_table(rows),
        "",
    ]


def _module_skipped_lines(payload: Mapping[str, Any]) -> list[str]:
    """3.f for the module shape: the selections the pass declined to read, with its own reasons."""
    skipped = cast("list[dict[str, Any]]", payload["skipped_selections"])
    if not skipped:
        return ["### 3.f: selections skipped", "", "The pass skipped no selection.", ""]
    rows = [
        {
            "set group": SET_SHORT.get(str(entry["set_group"]), str(entry["set_group"])),
            "stratification": entry["stratification"],
            "stratum": entry["stratum"],
            "rows": entry["n_rows"],
            "reason": entry["reason"],
        }
        for entry in skipped
    ]
    return [
        "### 3.f: selections the pass skipped, and why",
        "",
        (
            "A stratum covering exactly the rows of its pooled parent would repeat that read, and an "
            "empty stratum is not a read at all. Both are skipped by the producer and recorded, "
            "because a selection that silently vanishes is indistinguishable from one that came out "
            "uninteresting."
        ),
        "",
        markdown_table(rows),
        "",
    ]


def module_displacement_lines(inputs: Inputs) -> list[str]:
    """Section 3's body when the payload came from `games.interp_displacement`."""
    payload = inputs.displacement
    if payload is None:
        return []
    reads = cast("list[dict[str, Any]]", payload["reads"])
    if not reads:
        return ["> The displacement payload carried no reads at all.", ""]
    step = max(int(read["step"]) for read in reads)
    lines = [*_module_pooled_lines(reads, step), *_module_extrema_lines(reads, step)]
    pooled = [
        read
        for read in _module_selection(reads, set_group=POOLED_SELECTION, step=step)
        if read["arm_pairs"]
    ]
    lines.extend(_module_axis_lines(inputs, reads, step))
    lines.extend(_module_nulls_lines(payload))
    if pooled:
        lines.extend(
            _module_stratified_lines(
                reads, step, int(min(pooled, key=_module_first_cosine)["layer"])
            )
        )
    lines.extend(_module_skipped_lines(payload))
    return lines


# --------------------------------------------------------------------------------------------
# Section 4: two-path agreement (the capture-level validation)
# --------------------------------------------------------------------------------------------


def two_path_lines(inputs: Inputs) -> list[str]:
    """Do the two independent capture implementations agree, and does the shift check have teeth."""
    lines = [
        "## 4. Two-path capture agreement, and its sabotage",
        "",
        (
            "The same activations were captured twice by two implementations (forward hooks against "
            "`output_hidden_states`, float32 against float16 storage), so every direction can be "
            "fitted from both and compared. Below the agreement floor, one of the two paths is "
            "wrong and everything downstream is suspect. The sabotage row is the same comparison "
            "run without the layer-index shift the two conventions differ by, which is the exact "
            "off-by-one that would otherwise produce a plausible number from depths one block "
            "apart."
        ),
        "",
    ]
    if inputs.two_path is None or inputs.two_path_source is None:
        return [*lines, *stub("Section 4", [e for e in inputs.missing if "two-path" in e.name])]
    lines.extend([f"Rendered from {inputs.two_path_source.describe()}.", ""])
    if inputs.two_path_source.shape == SHAPE_MODULE:
        return [*lines, *module_two_path_lines(inputs.two_path)]
    summary = cast("dict[str, Any]", inputs.two_path["summary"])
    sabotage = cast("dict[str, Any]", summary.get("shift_sabotage") or {})
    rows = [
        {
            "read": "direction cosines",
            "n": summary["n_direction_reads"],
            "min": fmt(summary["direction_cosine_min"], 4),
            "mean": fmt(summary["direction_cosine_mean"], 4),
            "floor": fmt(summary["agreement_floor"], 3),
            "n below floor": summary["n_below_floor"],
        },
        {
            "read": "displacement cosines",
            "n": EMPTY_CELL,
            "min": fmt(summary["displacement_cosine_min"], 4),
            "mean": fmt(summary["displacement_cosine_mean"], 4),
            "floor": fmt(summary["agreement_floor"], 3),
            "n below floor": EMPTY_CELL,
        },
        {
            "read": "SABOTAGE: no layer shift",
            "n": sabotage.get("n_reads", EMPTY_CELL),
            "min": EMPTY_CELL,
            "mean": fmt(sabotage.get("cosine_mean"), 4),
            "floor": fmt(summary["agreement_floor"], 3),
            "n below floor": f"max {fmt(sabotage.get('cosine_max'), 4)}",
        },
    ]
    return [
        *lines,
        markdown_table(rows),
        "",
        f"Raw relative L2 between the two paths, worst case: {fmt(summary['raw_relative_l2_max'], 4)}.",
        "",
    ]


def _post_norm_note(summary: Mapping[str, Any]) -> str:
    """Say what the top layer is and why it is not counted as disagreement."""
    return (
        f"**The top layer is reported apart from every number above.** On this corpus the converted "
        f"frozen ladder's layer {summary['top_layer']} is post-final-RMSNorm rather than a "
        f"decoder-block output -- a fifth capture-format difference beyond the four the plan "
        f"tabulated -- so the two paths are measuring different objects there and the producer "
        f"excludes it from the summary rather than averaging it in. "
        f"`top_layer_treated_as_post_norm` = "
        f"{flag(summary['top_layer_treated_as_post_norm'])}, which is the producer's own switch: "
        f"were it off, that layer's disagreement would land in the rows above as a failure of the "
        f"two paths to agree."
    )


def _module_two_path_layer_lines(reads: Sequence[Mapping[str, Any]]) -> list[str]:
    """Per-layer direction cosines and relative L2, comparable layers and the post-norm one apart."""
    if not reads:
        return [
            (
                "> This payload carried no per-layer reads, so the depth profile below cannot be "
                "built: the summary above is all of it. The per-layer reads live in the side-car "
                "`two_path_agreement.json` rather than in the summary the displacement payload "
                "embeds."
            ),
            "",
        ]

    def table(selected: Sequence[Mapping[str, Any]]) -> str:
        rows: list[dict[str, Any]] = []
        for layer in sorted({int(read["layer"]) for read in selected}):
            here = [read for read in selected if int(read["layer"]) == layer]
            cosines = [float(read["direction_cosine"]) for read in here]
            rows.append(
                {
                    "layer": layer,
                    "n": len(here),
                    "cos min": fmt(min(cosines), 5),
                    "cos mean": fmt(sum(cosines) / len(cosines), 5),
                    "rel L2 max": fmt(max(float(read["relative_l2"]) for read in here), 6),
                }
            )
        return markdown_table(rows)

    comparable = [read for read in reads if read["comparable"]]
    apart = [read for read in reads if not read["comparable"]]
    lines = [
        "### 4.depth: the same comparison layer by layer",
        "",
        (
            "`cos` is the direction cosine between the two paths at that depth; `rel L2` is "
            "`|a - b| / |a|` between their activation matrices, so it is a disagreement about the "
            "states themselves rather than about a direction fitted from them, and it is the column "
            "that separates 'the same states in a different dtype' from 'a different object'."
        ),
        "",
        table(comparable),
        "",
    ]
    if apart:
        lines.extend(
            [
                "Layers excluded from the summary and from the table above, reported here instead:",
                "",
                table(apart),
                "",
            ]
        )
    return lines


def module_two_path_lines(payload: Mapping[str, Any]) -> list[str]:
    """Section 4's body when the agreement payload came from `games.interp_displacement`."""
    summary = cast("dict[str, Any]", payload["summary"])
    sabotage = cast("dict[str, Any]", summary["off_by_one_sabotage"])
    displacements = cast("list[dict[str, Any]]", payload.get("displacement_reads") or [])
    floor = fmt(summary["agreement_floor"], 3)
    rows = [
        {
            "read": "direction cosines (comparable layers)",
            "n": summary["n_comparable_reads"],
            "min": fmt(summary["direction_cosine_min"], 4),
            "mean": fmt(summary["direction_cosine_mean"], 4),
            "floor": floor,
            "n below floor": summary["n_below_floor"],
            "note": "one read per cell, stimulus set, pooling and comparable layer",
        },
        {
            "read": "displacement cosines",
            "n": sum(1 for read in displacements if read["comparable"]) or EMPTY_CELL,
            "min": fmt(summary["displacement_cosine_min"], 4),
            "mean": fmt(summary["displacement_cosine_mean"], 4),
            "floor": floor,
            "n below floor": EMPTY_CELL,
            "note": "the displacement-from-base vector, one per cell and pooling",
        },
        {
            "read": f"post-norm top layer L{summary['top_layer']}",
            "n": summary["post_norm_layer_reads"],
            "min": fmt(summary["post_norm_direction_cosine_min"], 4),
            "mean": fmt(summary["post_norm_direction_cosine_mean"], 4),
            "floor": EMPTY_CELL,
            "n below floor": EMPTY_CELL,
            "note": "EXCLUDED from the row above; the two paths measure different objects here",
        },
        {
            "read": "SABOTAGE: no layer shift",
            "n": sabotage["n_reads"],
            "min": EMPTY_CELL,
            "mean": fmt(sabotage["cosine_mean"], 4),
            "floor": floor,
            "n below floor": f"max {fmt(sabotage['cosine_max'], 4)}",
            "note": f"collapsed below the floor as it must: {flag(sabotage['is_red'])}",
        },
    ]
    return [
        markdown_table(rows),
        "",
        (
            f"Relative L2 between the two paths over the comparable layers, worst case "
            f"{fmt(summary['relative_l2_max'], 6)}. The post-norm top layer is orders of magnitude "
            f"above that and is why it is separated; see the depth table below for both."
        ),
        "",
        f"> {_post_norm_note(summary)}",
        "",
        *_module_two_path_layer_lines(cast("list[dict[str, Any]]", payload.get("reads") or [])),
    ]


# --------------------------------------------------------------------------------------------
# Section 5: the Jacobian lens readouts
# --------------------------------------------------------------------------------------------

LENS_CAVEAT = (
    "**Read this before the token table, not after.** These fits are small: {fit_size}. Each beat "
    "its own logit-lens baseline on held-out prompts, but narrowly ({reduction}), and every one of "
    "them sits at a median relative residual **above 1.0** with **negative** median explained "
    "variance ({quality}), which is worse absolute reconstruction than predicting the mean. A "
    "narrow win over a weak baseline is still a weak instrument. Accordingly: the promoted-token "
    "lists below are **uninterpretable at this fit size**, and the matched-norm placebo column "
    "beside every real one is what establishes that rather than an assertion -- the two columns are "
    "not distinguishable in character by eye. That is a fact about the fit, not about the "
    "direction; a larger fit is the experiment that would change it. Nothing else in this document "
    "rests on these tokens, and they are not evidence for or against any axis carrying "
    "correlated-decision content."
)


def lens_lines(inputs: Inputs) -> list[str]:
    """Build the lead method's readout: fit quality, then the promoted tokens and their caveat."""
    lines = [
        "## 5. Jacobian lens: fit quality, then the transported directions",
        "",
        (
            "Jacobian-space is the lead method here, and we fit our own lens rather than hunting a "
            "pre-fit. Each lens is fitted on one checkpoint's weights (merged bf16 wherever an "
            "adapter is involved, which is not every cell -- see the asymmetry note below) and then "
            "used to transport one of the geometry section's directions into vocabulary space, so "
            "the question it answers is what a direction *says* rather than only where it points."
        ),
        "",
    ]
    missing = [
        entry for entry in inputs.missing if "lens" in entry.name or entry.name == "GPU pass"
    ]
    if not inputs.lens_cells:
        return [*lines, *stub("Section 5", missing)]
    lines.extend(stub("Section 5 (partially)", [e for e in missing if "never produced" in e.name]))
    lines.extend(_lens_quality_lines(inputs))
    lines.extend(_lens_token_lines(inputs))
    return lines


def _lens_quality_lines(inputs: Inputs) -> list[str]:
    """Per lens: how it was fitted and how far it beat the logit-lens baseline."""
    rows: list[dict[str, Any]] = []
    for label in sorted(inputs.lens_cells):
        cell = inputs.lens_cells[label]
        quality = cast("dict[str, Any]", cell["fit_quality"])
        jac = cast("dict[str, Any]", quality["jacobian"])
        base = cast("dict[str, Any]", quality["logit_lens_baseline"])
        rows.append(
            {
                "cell": label,
                "pass": cell["_source"],
                "merged": flag(cell["merged"]),
                "fit/eval prompts": f"{cell['n_fit_prompts']}/{cell['n_eval_prompts']}",
                "dim_batch": cell["dim_batch"],
                "max_seq_len": cell["max_seq_len"],
                "fit s": fmt(cell["fit_seconds"], 1),
                "eval samples": quality["n_samples"],
                "jac resid (med)": fmt(jac["median_relative_residual"]),
                "logit-lens resid (med)": fmt(base["median_relative_residual"]),
                "reduction": fmt(quality["median_residual_reduction_vs_logit_lens"]),
                "beats baseline": flag(quality["jacobian_beats_logit_lens"]),
                "jac EV (med)": signed(jac["median_explained_variance"]),
                "best layer": jac["best_layer"],
                "best-layer resid": fmt(jac["best_layer_relative_residual"]),
            }
        )
    plan_rows: list[dict[str, Any]] = [
        {
            "pass": entry["source"],
            "cells": ", ".join(entry["cells"]) or EMPTY_CELL,
            "max_seq_len": cast("dict[str, Any]", entry["seq_len_plan"]).get(
                "max_seq_len", EMPTY_CELL
            ),
            "pairs diverging past window": cast("dict[str, Any]", entry["seq_len_plan"]).get(
                "n_pairs_diverging_past_window", EMPTY_CELL
            ),
            "truncated": cast("dict[str, Any]", entry["seq_len_plan"]).get(
                "n_truncated", EMPTY_CELL
            ),
            "n_pairs": cast("dict[str, Any]", entry["seq_len_plan"]).get("n_pairs", EMPTY_CELL),
        }
        for entry in inputs.lens_files
    ]
    return [
        "### 5.a: how each lens was fitted",
        "",
        (
            "`resid` columns are median relative residual on held-out prompts, lower being better, "
            "and `reduction` is the median residual the Jacobian fit removes relative to the "
            "logit-lens baseline. A fit that does not beat its baseline has nothing to transport, "
            "and a fit whose residual exceeds 1.0 with negative explained variance beat a baseline "
            "that was itself worse than predicting the mean. Read `reduction` and `jac EV (med)` "
            "together before reading anything in 5.b."
        ),
        "",
        markdown_table(rows),
        "",
        *_lens_merge_asymmetry_lines(inputs),
        (
            "Sequence-window plan per pass. The stimuli in each pair first differ late in a long "
            "shared stem, so a window that truncates the divergence would compare identical text: "
            "`pairs diverging past window` must be 0."
        ),
        "",
        markdown_table(plan_rows),
        "",
    ]


def _lens_merge_asymmetry_lines(inputs: Inputs) -> list[str]:
    """Say so when some lenses were fitted merged and others were not.

    The attenuation caveat is usually stated as riding every lens number equally. It does not: a
    ladder holding one unmerged anchor and merged endpoints has the attenuation on one side of its
    only cross-checkpoint comparison, which is the comparison the ladder exists for.
    """
    merged = sorted(label for label, cell in inputs.lens_cells.items() if cell["merged"])
    unmerged = sorted(label for label, cell in inputs.lens_cells.items() if not cell["merged"])
    if not merged or not unmerged:
        return []
    return [
        (
            f"> **The merge attenuation is asymmetric across this ladder.** Fitted on merged bf16 "
            f"exports: {', '.join(f'`{label}`' for label in merged)}. Fitted unmerged: "
            f"{', '.join(f'`{label}`' for label in unmerged)}. So the only cross-checkpoint "
            f"comparison the ladder supports has the ~64% delta realisation on one side of it and "
            f"nothing on the other, and any difference between those cells is confounded with the "
            f"merge."
        ),
        "",
    ]


def _lens_token_lines(inputs: Inputs) -> list[str]:
    """Build the promoted-token tables, behind the caveat that says how far to trust them."""
    readouts = {
        label: cast("dict[str, Any]", cell["direction_readout"])
        for label, cell in inputs.lens_cells.items()
        if cell.get("direction_readout", {}).get("real")
    }
    if not readouts:
        return [
            "### 5.b: promoted tokens",
            "",
            (
                "> No lens carried a `direction_readout`, so no direction was transported through "
                "any of them. The fits above exist; the vocabulary read does not."
            ),
            "",
        ]
    fit_sizes = sorted(
        {
            f"{inputs.lens_cells[label]['n_fit_prompts']} fit prompts at dim_batch "
            f"{inputs.lens_cells[label]['dim_batch']}"
            for label in readouts
        }
    )
    qualities = [
        cast("dict[str, Any]", inputs.lens_cells[label]["fit_quality"])
        for label in sorted(readouts)
    ]
    reductions = sorted(
        fmt(quality["median_residual_reduction_vs_logit_lens"]) for quality in qualities
    )
    residuals = sorted(
        fmt(quality["jacobian"]["median_relative_residual"]) for quality in qualities
    )
    variances = sorted(
        signed(quality["jacobian"]["median_explained_variance"]) for quality in qualities
    )
    lines = [
        "### 5.b: promoted tokens, real direction against its matched-norm placebo",
        "",
        LENS_CAVEAT.format(
            fit_size="; ".join(fit_sizes),
            reduction="median residual reduction " + ", ".join(reductions),
            quality=(
                f"median relative residual {', '.join(residuals)}; median explained variance "
                f"{', '.join(variances)}"
            ),
        ),
        "",
        (
            "Tokens are raw tokenizer pieces, so a leading `Ġ` is a word-initial space and "
            "byte-level fragments render as mojibake rather than being cleaned up. "
            f"Top {TOP_TOKENS} by promoted logit."
        ),
        "",
    ]
    axes_seen: set[str] = set()
    for label in sorted(readouts):
        readout = readouts[label]
        axis = Path(str(readout["direction_path"])).stem
        axes_seen.add(axis)
        real = cast("list[dict[str, Any]]", readout["real"])[:TOP_TOKENS]
        placebo = cast("list[dict[str, Any]]", readout.get("placebo") or [])[:TOP_TOKENS]
        rows = [
            {
                "rank": index + 1,
                "real token": f"`{real[index]['token']}`",
                "real logit": fmt(real[index]["logit"], 2),
                "placebo token": f"`{placebo[index]['token']}`"
                if index < len(placebo)
                else MISSING_CELL,
                "placebo logit": fmt(placebo[index]["logit"], 2)
                if index < len(placebo)
                else MISSING_CELL,
            }
            for index in range(len(real))
        ]
        lines.extend(
            [
                f"**`{label}`**, axis `{axis}`, layer {readout['layer']}",
                "",
                markdown_table(rows),
                "",
            ]
        )
    lines.extend(_lens_token_identity_lines(readouts))
    lines.extend(_lens_axis_coverage_lines(axes_seen))
    return lines


def _decode_agreement(entries: Sequence[tuple[str, tuple[str, ...]]], count: int) -> str:
    """Say how far several checkpoints' transported decodes agree: same tokens, same order, neither.

    Ordering and membership are reported separately because logits here tie to the bit, so two
    checkpoints that promote exactly the same eight tokens can still list them in a different order.
    Calling that "different" would overstate a disagreement that is a sort artifact.
    """
    ordered = {tokens for _, tokens in entries}
    membership = {frozenset(tokens) for _, tokens in entries}
    if len(ordered) == 1:
        return f"the top {TOP_TOKENS} are the same tokens in the same order at all {count}"
    if len(membership) == 1:
        return (
            f"the top {TOP_TOKENS} are the same tokens at all {count}, in a different order at "
            f"{len(ordered)} of them (the logits tie, so the order is a sort artifact)"
        )
    return f"the top {TOP_TOKENS} are not the same tokens between at least two of the {count}"


def _lens_token_identity_lines(readouts: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Report whether the transported decode is literally the same token set at every checkpoint.

    Computed rather than asserted, and it cuts the tempting way: an identical decode across
    checkpoints looks like "the direction says the same thing before and after training", and at
    this fit size it is equally consistent with the lens resolving nothing about the direction.
    """
    if len(readouts) < MIN_CELLS_FOR_A_COMPARISON:
        return []
    by_axis: dict[tuple[str, int], list[tuple[str, tuple[str, ...]]]] = {}
    for label in sorted(readouts):
        readout = readouts[label]
        key = (Path(str(readout["direction_path"])).stem, int(readout["layer"]))
        tokens = tuple(
            str(entry["token"])
            for entry in cast("list[dict[str, Any]]", readout["real"])[:TOP_TOKENS]
        )
        by_axis.setdefault(key, []).append((label, tokens))
    lines: list[str] = []
    for (axis, layer), entries in sorted(by_axis.items()):
        if len(entries) < MIN_CELLS_FOR_A_COMPARISON:
            continue
        verdict = _decode_agreement(entries, len(entries))
        lines.append(
            f"Axis `{axis}` at layer {layer} was transported through "
            f"{', '.join(f'`{label}`' for label, _ in entries)}: {verdict}. An identical decode is "
            f"consistent with the direction meaning the same thing at every checkpoint AND with the "
            f"lens resolving nothing about it, and at the fit size above the two cannot be told "
            f"apart, so this is not evidence either way."
        )
    if lines:
        lines.append("")
    return lines


def _lens_axis_coverage_lines(axes_seen: Iterable[str]) -> list[str]:
    """Name the axes the plan wanted transported that no lens actually transported."""
    seen = sorted(axes_seen)
    wanted = ("coarse", "lead")
    absent = [axis for axis in wanted if not any(axis in name for name in seen)]
    if not absent:
        return []
    return [
        (
            f"> **Axis coverage of section 5 is INCOMPLETE.** The lenses transported "
            f"{', '.join(f'`{name}`' for name in seen)} and nothing else, so "
            f"{', '.join(f'the {name} axis' for name in absent)} has no vocabulary readout at all. "
            f"The side-by-side comparison of two axes' decodes, which is the artifact this section "
            f"was meant to produce, cannot be assembled from what is on disk."
        ),
        "",
    ]


# --------------------------------------------------------------------------------------------
# Section 6: steering
# --------------------------------------------------------------------------------------------

BASELINE_CONDITION = "none"
FAMILY_BASELINE = "baseline"
FAMILY_STEER = "steer"
FAMILY_ABLATE = "ablate"
PLACEBO_TOKEN = "placebo"
# `<direction>:L<layer>:x<alpha>:<steer|placebo>:<sign>` against `<direction>:L<layer>:ablate:<real|placebo>`.
STEER_KEY_PARTS = 5
ABLATE_KEY_PARTS = 4


@dataclass(frozen=True)
class RowKind:
    """One row of the steering table's fixed pattern: what was done, which sign, real or placebo."""

    family: str
    sign: str
    is_placebo: bool


@dataclass(frozen=True)
class RowTarget:
    """Where one steering row intervened: the axis, the layer, and the magnitude if it has one.

    Ablation has no magnitude -- projecting a direction out is not scaled -- so its multiplier is
    None rather than a stand-in zero. A stand-in would make ablation look like a second alpha at the
    same layer and duplicate its rows once per real alpha, which is exactly what it did before.
    """

    direction: str
    layer: int
    alpha_multiplier: float | None


@dataclass(frozen=True)
class SteeringCondition:
    """One steering condition key, parsed into the fields the table groups and orders by."""

    key: str
    direction: str | None
    layer: int | None
    alpha_multiplier: float | None
    kind: RowKind

    @property
    def site(self) -> tuple[str, int] | None:
        """Return the (direction, layer) this intervened at, or None for the baseline."""
        if self.direction is None or self.layer is None:
            return None
        return self.direction, self.layer

    @property
    def target(self) -> RowTarget | None:
        """Return the full row target, or None for the baseline, which intervened nowhere."""
        if self.direction is None or self.layer is None:
            return None
        return RowTarget(self.direction, self.layer, self.alpha_multiplier)


def parse_condition(key: str) -> SteeringCondition:
    """Parse a steering condition key into its parts.

    Raises on an unrecognised shape rather than skipping it. A condition silently dropped from a
    steering table is exactly the failure this document exists to prevent: the reader sees a
    complete-looking grid and cannot tell that a row was eaten.
    """
    if key == BASELINE_CONDITION:
        return SteeringCondition(
            key, None, None, None, RowKind(FAMILY_BASELINE, "", is_placebo=False)
        )
    parts = key.split(":")
    if len(parts) == STEER_KEY_PARTS and parts[1].startswith("L") and parts[2].startswith("x"):
        return SteeringCondition(
            key=key,
            direction=parts[0],
            layer=int(parts[1][1:]),
            alpha_multiplier=float(parts[2][1:]),
            kind=RowKind(FAMILY_STEER, parts[4], is_placebo=parts[3] == PLACEBO_TOKEN),
        )
    if len(parts) == ABLATE_KEY_PARTS and parts[1].startswith("L") and parts[2] == FAMILY_ABLATE:
        return SteeringCondition(
            key=key,
            direction=parts[0],
            layer=int(parts[1][1:]),
            alpha_multiplier=None,
            kind=RowKind(FAMILY_ABLATE, "", is_placebo=parts[3] == PLACEBO_TOKEN),
        )
    raise ValueError(
        f"unrecognised steering condition key {key!r}; the readout will not guess at a row label"
    )


# The table's fixed row order: every true-direction condition immediately followed by the placebo
# that controls it. Order is the whole point, so it is declared here rather than sorted at render.
STEER_ROWS: tuple[RowKind, ...] = (
    RowKind(FAMILY_STEER, "+", is_placebo=False),
    RowKind(FAMILY_STEER, "+", is_placebo=True),
    RowKind(FAMILY_STEER, "-", is_placebo=False),
    RowKind(FAMILY_STEER, "-", is_placebo=True),
)
ABLATE_ROWS: tuple[RowKind, ...] = (
    RowKind(FAMILY_ABLATE, "", is_placebo=False),
    RowKind(FAMILY_ABLATE, "", is_placebo=True),
)


def steering_lines(inputs: Inputs) -> list[str]:
    """Question (b): does intervening on the direction move behaviour, and does the placebo not."""
    lines = [
        "## 6. Steering: a decodable direction is not a used one",
        "",
        (
            "Cooperation rate per condition, with the matched-norm placebo row directly beneath "
            "every true-direction row in the same table. That adjacency is deliberate: it makes a "
            "missing placebo obvious, and an absent placebo is emitted as a "
            f"{MISSING_CELL} row rather than left out."
        ),
        "",
        f"> {MERGED_BF16_CAVEAT}",
        "",
        f"> {CENSORING_CAVEAT}",
        "",
        f"> {PRINT_ORDER_CAVEAT}",
        "",
    ]
    missing = [
        entry
        for entry in inputs.missing
        if "steering" in entry.name or "logit sweep" in entry.name or entry.name == "GPU pass"
    ]
    if not inputs.steering and inputs.logit_sweep is None:
        return [*lines, *stub("Section 6", missing)]
    lines.extend(stub("Section 6 (partially)", missing))
    lines.extend(_unsummarised_lines(inputs))
    lines.extend(_perturbation_size_lines(inputs))
    for name in sorted(inputs.steering):
        lines.extend(
            _steering_arm_lines(name, inputs.steering[name], inputs.steering_records.get(name, {}))
        )
    lines.extend(_logit_sweep_lines(inputs))
    lines.extend(_steered_axis_lines(inputs))
    return lines


TENSION_GUIDANCE = (
    "Section 3 reads which way each arm's activations were displaced along each axis; this section "
    "reads which way pushing an axis moves cooperation. Chaining the two into 'training moved this "
    "axis, which moved behaviour' needs both readings on the *same* axis at the *same* layer, so the "
    "table below looks the section 3 displacement up for exactly the (axis, layer) each steering "
    "cell intervened on, taken from that cell's own `cells` field. Steering establishes that a "
    "direction is a causal knob for cooperation; a causal knob is not a used one."
)


def _steered_sites(inputs: Inputs) -> dict[tuple[str, int], list[str]]:
    """Every (axis, layer) the steering cells intervened on, and which cells intervened there."""
    sites: dict[tuple[str, int], list[str]] = {}
    for name in sorted(inputs.steering):
        for cell in cast("list[dict[str, Any]]", inputs.steering[name]["cells"]):
            sites.setdefault((str(cell["direction"]), int(cell["layer"])), []).append(name)
    return sites


def _steered_axis_lines(inputs: Inputs) -> list[str]:
    """Look section 3's displacement up for the axis and layer each steering cell intervened on.

    This replaced a hand-written paragraph asserting that the two sections could not be composed and
    that the tension was unresolved. The claim was wrong in the way prose in a generated document is
    always liable to be: it compared a different axis at a different layer from the one the
    intervention ran on, and no number in the document contradicted it because no number was in it.
    """
    lines = [
        "### 6.tension: did training displace the axis these cells steer?",
        "",
        TENSION_GUIDANCE,
        "",
    ]
    if inputs.displacement_vs_axes is None:
        return [*lines, *stub("Section 6.tension", [_along_axis_gap(inputs)])]
    sites = _steered_sites(inputs)
    reads = [read for read in inputs.displacement_vs_axes if read["pooling"] == PRIMARY_POOLING]
    if not sites or not reads:
        absent = (
            "no steering summary recorded the cell it intervened on"
            if not sites
            else f"the along-axis payload carries no `{PRIMARY_POOLING}`-pooled read"
        )
        return [*lines, f"> Nothing to look up here: {absent}.", ""]
    step = max(int(read["step"]) for read in reads)
    by_layer = {int(read["layer"]): read for read in reads if int(read["step"]) == step}
    rows: list[dict[str, Any]] = []
    verdicts: list[str] = []
    unreadable: list[str] = []
    for (short, layer), cells in sorted(sites.items()):
        read = by_layer.get(layer)
        axis_set = SHORT_TO_SET.get(short)
        entry = (
            None
            if read is None or axis_set is None
            else cast("dict[str, Any]", read["cos_axis"]).get(axis_set)
        )
        if read is None or entry is None:
            unreadable.append(_site_lookup_reason(read, short, layer, step))
            continue
        floor = float(read["placebo_abs_cos_max"])
        readings = [
            (arm, float(cast("dict[str, Any]", entry)[field]))
            for arm, field in AXIS_DISPLACEMENT_FIELDS.items()
        ]
        rows.extend(
            {
                "steered site": f"{short} L{layer}",
                "steered by": ", ".join(f"`{cell}`" for cell in cells),
                "arm": arm,
                "displacement cos": signed(cosine_value),
                "placebo floor (max abs cos)": fmt(floor),
                "clears floor": flag(abs(cosine_value) > floor),
            }
            for arm, cosine_value in readings
        )
        verdicts.append(_floor_verdict(short, layer, readings, floor))
    lines.extend(
        [
            (
                f"Read at pooling `{PRIMARY_POOLING}`, step {step}, from the along-axis payload. The "
                f"floor column is that payload's own `placebo_abs_cos_max`, the matched-norm placebo "
                f"band for the same cosine at the same cell, so it is whatever draw count that pass "
                f"ran and not a fixed threshold."
            ),
            "",
            markdown_table(rows),
            "",
            *verdicts,
            "",
        ]
    )
    if unreadable:
        lines.extend(
            [
                "> Steered sites this check could not look up, named rather than dropped: "
                + "; ".join(unreadable)
                + ".",
                "",
            ]
        )
    return lines


def _site_lookup_reason(read: Mapping[str, Any] | None, short: str, layer: int, step: int) -> str:
    """Name why one steered site could not be looked up, so its absence is a fact and not a gap."""
    if read is None:
        return (
            f"`{short}` L{layer} -- the payload carries no read at pooling `{PRIMARY_POOLING}` step "
            f"{step} for layer {layer}"
        )
    if short not in SHORT_TO_SET:
        return f"`{short}` L{layer} -- `{short}` is not one of the three stimulus-set short names"
    return f"`{short}` L{layer} -- its layer {layer} read carries no `{SHORT_TO_SET[short]}` entry"


def _floor_verdict(
    short: str, layer: int, readings: Sequence[tuple[str, float]], floor: float
) -> str:
    """State from the numbers what one steered site's floor comparison does and does not license."""
    detail = ", ".join(
        f"{arm} {signed(cosine_value)} {'CLEARS' if abs(cosine_value) > floor else 'below'}"
        for arm, cosine_value in readings
    )
    clearing = [(arm, value) for arm, value in readings if abs(value) > floor]
    head = f"- `{short}` L{layer} (floor {fmt(floor)}): {detail}."
    if not clearing:
        return (
            f"{head} No arm clears the floor, so training left no displacement along the axis these "
            f"cells steer that this instrument resolves, and whatever steering does to cooperation "
            f"here cannot be the knob training turned."
        )
    signs = ", ".join(f"{arm} leans {'+' if value > 0 else '-'}" for arm, value in clearing)
    scope = "" if len(clearing) == len(readings) else " for the arms that clear it"
    return (
        f"{head} The sign comparison against `d vs none` above is live{scope}: {signs}. Read those "
        f"signs against the direction steering moved cooperation before calling either a mechanism."
    )


def _along_axis_gap(inputs: Inputs) -> MissingInput:
    """Describe the absent along-axis payload, so this block degrades to a gap and not to a claim."""
    root = inputs.analysis_root or inputs.artifacts_root
    return MissingInput(
        name="displacement-against-axes report, for the steered-axis check",
        path=display_path(root / "displacement_vs_axes.json"),
        consequence=(
            "the check that reads whether training displaced the very axis these steering cells "
            "pushed; without it sections 3 and 6 cannot be composed in either direction"
        ),
    )


def _unsummarised_lines(inputs: Inputs) -> list[str]:
    """Name the steering directories holding records but no summary, so they are not mistaken for data."""
    if not inputs.unsummarised_steering:
        return []
    return [
        (
            f"> Steering directories holding records but no summary, and therefore **not results**: "
            f"{', '.join(f'`{name}`' for name in inputs.unsummarised_steering)}. A killed or "
            f"superseded run keeps its records on purpose; nothing below reads them."
        ),
        "",
    ]


def _perturbation_size_lines(inputs: Inputs) -> list[str]:
    """How large the intervention was relative to the residual stream it was added to.

    The trust criterion for a steering result is that the hooked layer's residual norm moves by a
    few percent, not tens of percent. This says where the run actually sat, from the sweep's own
    per-cell record, so 'we perturbed the model 43%' is visible if that is what happened.
    """
    if inputs.logit_sweep is None or not inputs.steering:
        return []
    fractions: dict[tuple[str, int, float], float] = {
        (str(cell["direction"]), int(cell["layer"]), float(cell["alpha_multiplier"])): float(
            cell["alpha_over_residual_norm"]
        )
        for cell in cast("list[dict[str, Any]]", inputs.logit_sweep["cells"])
    }
    quoted: list[str] = []
    for summary in inputs.steering.values():
        for cell in cast("list[dict[str, Any]]", summary["cells"]):
            key = (str(cell["direction"]), int(cell["layer"]), float(cell["alpha_multiplier"]))
            fraction = fractions.get(key)
            if fraction is not None:
                quoted.append(f"`{key[0]}` L{key[1]} x{key[2]:g} at {fraction:.2f} of it")
    if not quoted:
        return []
    return [
        (
            f"> **Perturbation size.** The trust criterion for a steering result is that the hooked "
            f"layer's residual norm changes by a few percent, not tens of percent. Alpha as a "
            f"fraction of that layer's mean residual norm, from the sweep's own record: "
            f"{'; '.join(sorted(set(quoted)))}. Anything approaching half the residual norm is a "
            f"large perturbation, and where it is large the matched-norm placebo row is the only "
            f"thing keeping the result readable."
        ),
        "",
    ]


def _steering_arm_lines(
    name: str, summary: Mapping[str, Any], records: Mapping[str, int]
) -> list[str]:
    """One steered checkpoint's table: baseline, then every cell's rows with placebos adjacent."""
    conditions = cast("dict[str, dict[str, Any]]", summary["conditions"])
    parsed = {key: parse_condition(key) for key in conditions}
    skipped = {
        str(entry["condition_key"]): str(entry["skipped"])
        for entry in cast("list[dict[str, Any]]", summary.get("skipped_conditions") or [])
    }
    rows: list[dict[str, Any]] = []
    baseline = conditions.get(BASELINE_CONDITION)
    baseline_rate = _rate_value(baseline)
    if baseline is not None:
        rows.append(_steering_row("no intervention", baseline, baseline_rate, None))
    # Row discovery spans the conditions that ran AND the ones the run recorded as skipped, so a
    # condition abandoned at a deadline still occupies its row instead of vanishing from the grid.
    intended = {**parsed, **{key: parse_condition(key) for key in skipped}}
    for site in sorted({c.site for c in intended.values() if c.site is not None}):
        for target in _targets_at(intended, site):
            for kind in STEER_ROWS if target.alpha_multiplier is not None else ABLATE_ROWS:
                key = _condition_key(target, kind)
                payload = conditions.get(key)
                label = _row_label(target, kind)
                rows.append(
                    _missing_row(label, key, skipped.get(key))
                    if payload is None
                    else _steering_row(label, payload, baseline_rate, key)
                )
    cells_ran = ", ".join(
        f"`{cell['direction']}` at layer {cell['layer']}, alpha x{cell['alpha_multiplier']:g}"
        for cell in cast("list[dict[str, Any]]", summary["cells"])
    )
    lines = [
        f"### 6.{name}: `{summary['model']}`",
        "",
        _steering_provenance(summary),
        "",
        (
            f"Cells run: {cells_ran or 'none recorded'}. Which cell that is was chosen by the "
            f"forced-choice sweep in 6.selection and is read here from the summary rather than "
            f"re-derived, so this document cannot disagree with the run about what it did."
        ),
        "",
        markdown_table(rows),
        "",
    ]
    lines.extend(_placebo_coverage_lines(parsed))
    lines.extend(_records_check_lines(name, summary, records))
    if skipped:
        lines.extend(
            [
                "> Conditions the run never reached, with the reason it recorded: "
                + "; ".join(f"`{key}` ({reason})" for key, reason in sorted(skipped.items()))
                + f". Each appears above as a {MISSING_CELL} row rather than as an absence.",
                "",
            ]
        )
    return lines


def _placebo_coverage_lines(parsed: Mapping[str, SteeringCondition]) -> list[str]:
    """Name every true-direction condition that ran without its matched-norm placebo.

    The adjacency in the table makes one absence visible; this sentence makes the pattern visible.
    An arm steered in both signs with a placebo on only one of them has a one-sided control, and
    the sign whose placebo is absent cannot be told from a norm perturbation at all.
    """
    real = {
        (condition.target, condition.kind.family, condition.kind.sign)
        for condition in parsed.values()
        if condition.target is not None and not condition.kind.is_placebo
    }
    placebo = {
        (condition.target, condition.kind.family, condition.kind.sign)
        for condition in parsed.values()
        if condition.target is not None and condition.kind.is_placebo
    }
    if not real:
        return [
            (
                "> **This summary banked no true-direction condition at all**, so the empty table "
                "above is a fact about the run rather than an omission here, and there is nothing "
                "for a placebo to control."
            ),
            "",
        ]
    uncontrolled = sorted(
        _row_label(target, RowKind(family, sign, is_placebo=False))
        for target, family, sign in real - placebo
    )
    if not uncontrolled:
        return [
            "Every true-direction condition here ran with its matched-norm placebo beside it.",
            "",
        ]
    return [
        (
            f"> **Placebo coverage on this arm is one-sided.** No matched-norm placebo ran for: "
            f"{', '.join(f'`{entry}`' for entry in uncontrolled)}. For those conditions the "
            f"reported move cannot be distinguished from what any perturbation of that magnitude "
            f"does, which is the specific way a positive steering result comes out empty."
        ),
        "",
    ]


def _records_check_lines(
    name: str, summary: Mapping[str, Any], records: Mapping[str, int]
) -> list[str]:
    """Check the per-record censoring claim the summary's denominators rest on.

    Recomputed from the records rather than quoted: a truncated completion is supposed to have no
    visible text and therefore no parsable action, so the overlap must be exactly zero. If it is
    not, the truncation columns above are not the censoring they are read as.
    """
    if not records or records["records"] == 0:
        return [
            (
                f"> **No per-record file for `{name}`**, so the censoring claim behind the "
                f"truncation columns above is unverified here: it rests on the summary's own counts."
            ),
            "",
        ]
    expected = int(summary["n_rows"]) * int(summary["n_samples"]) * len(summary["conditions"])
    overlap = records["truncated_and_parsed"]
    verdict = (
        f"{overlap} of the {records['truncated']} truncated records also parsed"
        if overlap
        else f"none of the {records['truncated']} truncated records parsed, as the convention requires"
    )
    mismatch = (
        ""
        if records["records"] == expected
        else (
            f" The file holds {records['records']} records against {expected} expected from "
            f"{summary['n_rows']} prompts x {summary['n_samples']} samples x "
            f"{len(summary['conditions'])} conditions, so the grid is not square."
        )
    )
    return [
        (
            f"Per-record check over {records['records']} records: {verdict} "
            f"({records['parsed']} parsed in total). {'PROBLEM: ' if overlap else ''}"
            f"A nonzero overlap would mean the truncation columns are not measuring censoring."
            f"{mismatch}"
        ),
        "",
    ]


def _targets_at(
    intended: Mapping[str, SteeringCondition], site: tuple[str, int]
) -> list[RowTarget]:
    """List one (direction, layer) site's row targets: each steered magnitude, then ablation once.

    Ablation appears at most once per site no matter how many magnitudes were steered there, and a
    site that never intended ablation gets no ablation rows rather than two empty ones.
    """
    here = [condition for condition in intended.values() if condition.site == site]
    alphas = sorted(
        {
            condition.alpha_multiplier
            for condition in here
            if condition.kind.family == FAMILY_STEER and condition.alpha_multiplier is not None
        }
    )
    targets = [RowTarget(site[0], site[1], alpha) for alpha in alphas]
    if any(condition.kind.family == FAMILY_ABLATE for condition in here):
        targets.append(RowTarget(site[0], site[1], None))
    return targets


def _condition_key(target: RowTarget, kind: RowKind) -> str:
    """Rebuild the payload's own condition key for one row of the table."""
    arm = (
        PLACEBO_TOKEN if kind.is_placebo else ("real" if kind.family == FAMILY_ABLATE else "steer")
    )
    if kind.family == FAMILY_ABLATE:
        return f"{target.direction}:L{target.layer}:{FAMILY_ABLATE}:{arm}"
    return f"{target.direction}:L{target.layer}:x{target.alpha_multiplier}:{arm}:{kind.sign}"


def _row_label(target: RowTarget, kind: RowKind) -> str:
    """Build the human row label: what was done, at which layer, with which magnitude."""
    what = FAMILY_ABLATE if kind.family == FAMILY_ABLATE else f"steer {kind.sign}"
    arm = "PLACEBO (matched norm)" if kind.is_placebo else "direction"
    magnitude = "" if target.alpha_multiplier is None else f", a x{target.alpha_multiplier:g}"
    return f"{target.direction} L{target.layer}{magnitude}: {what}, {arm}"


def _rate_value(payload: Mapping[str, Any] | None) -> float | None:
    """Return a condition's cooperation rate as a float, for the delta column."""
    if payload is None:
        return None
    return None if payload["n_parsed"] == 0 else float(payload["cooperate_rate"])


def _steering_row(
    label: str, payload: Mapping[str, Any], baseline_rate: float | None, key: str | None
) -> dict[str, Any]:
    """One condition's row: the rate with its denominator, its split, and its censoring."""
    orders = cast("dict[str, dict[str, Any]]", payload.get("by_print_order") or {})
    rate = _rate_value(payload)
    row: dict[str, Any] = {
        "condition": label,
        "coop rate (k/parsed)": rate_cell(payload["cooperate_k"], payload["n_parsed"]),
        "d vs none": EMPTY_CELL
        if rate is None or baseline_rate is None
        else signed(rate - baseline_rate),
    }
    for order in sorted(orders):
        row[f"{order} (k/parsed)"] = rate_cell(
            orders[order]["cooperate_k"], orders[order]["n_parsed"]
        )
    row["order spread"] = signed(_order_spread(orders))
    row["truncated thinking"] = percent_cell(
        payload["n_truncated_thinking"], payload["n_completions"]
    )
    row["parse failures"] = percent_cell(payload["n_parse_failures"], payload["n_completions"])
    row["key"] = f"`{key}`" if key else EMPTY_CELL
    return row


def _order_spread(orders: Mapping[str, Mapping[str, Any]]) -> float | None:
    """First-printed-order rate minus the other order's, within one condition.

    The residual print-order imbalance inside a single condition. It belongs beside every rate
    because the position effect on this corpus is larger than any effect being measured, so a
    condition whose own two orders disagree by ten points cannot support a ten-point claim.
    """
    usable = {name: entry for name, entry in orders.items() if int(entry["n_parsed"]) > 0}
    if len(usable) != PRINT_ORDERS:
        return None
    first, second = (usable[name] for name in sorted(usable))
    return float(first["cooperate_rate"]) - float(second["cooperate_rate"])


def _missing_row(label: str, key: str, reason: str | None) -> dict[str, Any]:
    """Build the row an absent condition still occupies, so the absence is read not inferred."""
    why = f"{MISSING_CELL} ({reason})" if reason else MISSING_CELL
    return {
        "condition": label,
        "coop rate (k/parsed)": why,
        "d vs none": MISSING_CELL,
        "order spread": MISSING_CELL,
        "truncated thinking": MISSING_CELL,
        "parse failures": MISSING_CELL,
        "key": f"`{key}`",
    }


def _steering_provenance(summary: Mapping[str, Any]) -> str:
    """Write the sentence a steering table cannot be read without: split, samples, sampler."""
    sampler = cast("dict[str, Any]", summary["resolved_sampler"])
    applied = cast("dict[str, Any]", sampler["applied"])
    dropped = sampler.get("presence_penalty_why_dropped")
    penalty = (
        f"presence penalty {sampler['presence_penalty_requested']} was requested and NOT applied "
        f"({dropped})"
        if sampler.get("presence_penalty_applied") is None and dropped
        else f"presence penalty {sampler.get('presence_penalty_applied')}"
    )
    return (
        f"Split `{summary['split']}`, {summary['n_rows']} prompts x {summary['n_samples']} samples "
        f"= {int(summary['n_rows']) * int(summary['n_samples'])} completions per condition, seed "
        f"{summary['seed']}, thinking budget {summary['max_new_tokens']} tokens, grading render "
        f"`{summary['render_grading']}`. Sampler: engine `{sampler['engine']}`, temperature "
        f"{applied['temperature']}, top_p {applied['top_p']}, top_k {applied['top_k']}; {penalty}."
    )


def _logit_sweep_lines(inputs: Inputs) -> list[str]:
    """Render the forced-choice selection pass that chose the layer and the alpha."""
    if inputs.logit_sweep is None:
        return []
    sweep = inputs.logit_sweep
    denominators = cast("dict[str, Any]", sweep["denominators"])
    baseline = cast("dict[str, Any]", sweep["baseline"])
    norms = cast("dict[str, Any]", sweep.get("residual_norms_by_layer") or {})
    cells = cast("list[dict[str, Any]]", sweep["cells"])
    lines = [
        "### 6.selection: forced-choice logit readout (thinking off, no generation)",
        "",
        (
            f"What chose the layer and the magnitude for the generation grid above. `gap` is the "
            f"logit difference between the cooperate-label and defect-label tokens, so it is a "
            f"different quantity from a cooperation rate and the two are not comparable in units. "
            f"`{sweep['caveat']}` Every row carries its matched-norm placebo directly beneath it, "
            f"same as above. Prompts used {denominators['n_used']} of {denominators['n_total']} "
            f"({denominators['n_skipped']} skipped), model `{sweep['model']}`, seed {sweep['seed']}, "
            f"grading render `{sweep['render_grading']}`."
        ),
        "",
        (
            f"Baseline with no intervention: gap {signed(baseline['mean_gap'], 4)} over n="
            f"{baseline['n']}; by print order "
            + ", ".join(
                f"{order} {signed(value, 4)}"
                for order, value in sorted(
                    cast("dict[str, Any]", baseline["mean_gap_by_print_order"]).items()
                )
            )
            + "."
        ),
        "",
    ]
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for cell in cells:
        by_direction.setdefault(str(cell["direction"]), []).append(cell)
    for direction in sorted(by_direction):
        indexed = {
            (int(cell["layer"]), float(cell["alpha_multiplier"]), str(cell["condition"])): cell
            for cell in by_direction[direction]
        }
        rows: list[dict[str, Any]] = []
        for layer, alpha in sorted({(key[0], key[1]) for key in indexed}):
            for condition in ("steer:+", "placebo:+", "steer:-", "placebo:-"):
                cell = indexed.get((layer, alpha, condition))
                if cell is None:
                    rows.append(
                        {
                            "layer": layer,
                            "a mult": f"{alpha:g}",
                            "condition": condition,
                            "gap": MISSING_CELL,
                            "d vs none": MISSING_CELL,
                        }
                    )
                    continue
                orders = cast("dict[str, Any]", cell["mean_delta_by_print_order"])
                rows.append(
                    {
                        "layer": layer,
                        "a mult": f"{alpha:g}",
                        "condition": (
                            f"**{condition}**" if condition.startswith("placebo") else condition
                        ),
                        "a raw": fmt(cell["alpha_raw"], 3),
                        "a / resid norm": fmt(cell["alpha_over_residual_norm"], 3),
                        "gap": signed(cell["mean_gap"], 4),
                        "d vs none": signed(cell["mean_delta_vs_none"], 4),
                        "d canonical": signed(orders.get("canonical"), 4),
                        "d swapped": signed(orders.get("swapped"), 4),
                        "n": cell["n"],
                    }
                )
        lines.extend([f"**{direction} axis**", "", markdown_table(rows), ""])
    if norms:
        lines.extend(
            [
                "Mean residual norm per hooked layer, which is what `a / resid norm` divides by: "
                + ", ".join(
                    f"L{layer} {fmt(value, 2)}"
                    for layer, value in sorted(norms.items(), key=lambda item: int(item[0]))
                )
                + ".",
                "",
            ]
        )
    return lines


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------


def write_figures(inputs: Inputs, out_dir: Path) -> dict[str, Path | None]:
    """Write the two figures a trajectory genuinely wants a line for, and return where they landed."""
    plt.switch_backend("Agg")
    return {
        TRAJECTORY_FIGURE: _write_trajectory_figure(inputs, out_dir),
        DRIFT_FIGURE: _write_drift_heatmap(inputs, out_dir),
    }


def _write_trajectory_figure(inputs: Inputs, out_dir: Path) -> Path | None:
    """Held-out projection against checkpoint step, one panel per set, placebo band shaded."""
    panels = [short for short in SHORT_ORDER if (PRIMARY_POOLING, short) in inputs.trajectories]
    if not panels:
        return None
    figure, axes_grid = plt.subplots(
        nrows=1, ncols=len(panels), figsize=(3.7 * len(panels), 3.1), squeeze=False
    )
    axes = cast("Any", axes_grid)
    for column, short in enumerate(panels):
        axis = axes[0][column]
        payload = inputs.trajectories[PRIMARY_POOLING, short]
        placebos: list[float] = []
        for arm in trajectory_arms(payload):
            group = group_for_arm(payload, arm)
            peak = int(group["peak_layer"])
            reads = sorted(
                (read for read in group["drift_reads"] if int(read["layer"]) == peak),
                key=lambda read: int(read["step"]),
            )
            axis.plot(
                [int(read["step"]) for read in reads],
                [float(read["heldout_projection_gap"]) for read in reads],
                marker="o",
                markersize=3.5,
                linewidth=1.4,
                label=f"{arm_role(arm)} (L{peak})",
            )
            placebos.extend(abs(float(read["heldout_projection_gap_placebo"])) for read in reads)
        band = max(placebos) if placebos else 0.0
        axis.axhspan(-band, band, color="0.75", alpha=0.45, label=f"placebo band +/-{band:.3f}")
        axis.set_title(f"{short}: {SHORT_TO_SET[short]}", fontsize=8)
        axis.set_xlabel("checkpoint step", fontsize=8)
        if column == 0:
            axis.set_ylabel(f"held-out projection gap ({PRIMARY_POOLING})", fontsize=8)
        axis.tick_params(labelsize=7)
        axis.grid(visible=True, alpha=0.2)
        axis.legend(fontsize=6.5, loc="best")
    figure.tight_layout()
    path = out_dir / TRAJECTORY_FIGURE
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _write_drift_heatmap(inputs: Inputs, out_dir: Path) -> Path | None:
    """Drift cosine from the base anchor as layer against step, one panel per arm and set."""
    panels = [short for short in SHORT_ORDER if (PRIMARY_POOLING, short) in inputs.trajectories]
    arms = all_arms(inputs)
    if not panels or not arms:
        return None
    figure, axes_grid = plt.subplots(
        nrows=len(panels),
        ncols=len(arms),
        figsize=(3.5 * len(arms), 2.5 * len(panels)),
        squeeze=False,
    )
    axes = cast("Any", axes_grid)
    for row, short in enumerate(panels):
        payload = inputs.trajectories[PRIMARY_POOLING, short]
        for column, arm in enumerate(arms):
            axis = axes[row][column]
            if arm not in trajectory_arms(payload):
                axis.set_axis_off()
                axis.set_title(f"{short} / {arm_role(arm)}: absent", fontsize=8)
                continue
            reads = cast("list[dict[str, Any]]", group_for_arm(payload, arm)["drift_reads"])
            steps = sorted({int(read["step"]) for read in reads})
            layers = sorted({int(read["layer"]) for read in reads})
            lookup = {
                (int(read["step"]), int(read["layer"])): float(read["cosine_to_anchor"])
                for read in reads
            }
            grid = [[lookup.get((step, layer), float("nan")) for step in steps] for layer in layers]
            image = axis.imshow(
                grid,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=(min(steps) - 5, max(steps) + 5, min(layers) - 0.5, max(layers) + 0.5),
            )
            finite = [
                value for column_values in grid for value in column_values if not math.isnan(value)
            ]
            span = f"{min(finite):.4f}-{max(finite):.4f}" if finite else "no data"
            axis.set_title(f"{short} / {arm_role(arm)}  [{span}]", fontsize=8)
            axis.set_xlabel("checkpoint step", fontsize=8)
            if column == 0:
                axis.set_ylabel("layer", fontsize=8)
            axis.tick_params(labelsize=7)
            bar = figure.colorbar(image, ax=axis)
            bar.ax.tick_params(labelsize=6)
    figure.suptitle(
        f"drift cosine to the step-0 anchor ({PRIMARY_POOLING} pooling); each panel scaled to its own range",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / DRIFT_FIGURE
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------------------------


def header_lines(inputs: Inputs, command: str) -> list[str]:
    """Banners, provenance and the expectations, before any number."""
    step = last_step(inputs)
    lines = [
        "# Games interp readout: what the twin-pd pair changed internally",
        "",
        f"> {GENERATED_BANNER}",
        "",
        f"- CPU analysis pass: `{display_path(inputs.analysis_root) if inputs.analysis_root else 'ABSENT'}`",
        f"- GPU pass: `{display_path(inputs.gpu_root) if inputs.gpu_root else 'ABSENT'}`",
        (
            f"- Trajectory reports read: {len(inputs.trajectories)} of "
            f"{len(POOLINGS) * len(SHORT_ORDER)}; arms {', '.join(f'`{arm}`' for arm in all_arms(inputs)) or 'none'}; "
            f"furthest checkpoint {step if step is not None else 'unknown'}."
        ),
        f"- Lens fits read: {len(inputs.lens_cells)}; steered checkpoints read: {len(inputs.steering)}.",
        (
            "- Displacement payload (section 3): "
            + (
                inputs.displacement_source.describe()
                if inputs.displacement_source is not None
                else "ABSENT"
            )
            + (
                ""
                if not inputs.displacement_unused
                else "; not read: "
                + ", ".join(f"`{display_path(o.path)}`" for o in inputs.displacement_unused)
            )
        ),
        (
            "- Two-path payload (section 4): "
            + (
                inputs.two_path_source.describe()
                if inputs.two_path_source is not None
                else "ABSENT"
            )
        ),
        f"- Regenerate with: `{command}`",
        "",
    ]
    lines.extend(_identity_lines(inputs))
    lines.extend(_status_lines(inputs))
    lines.extend(
        [
            "## How to read this",
            "",
            f"1. {MERGED_BF16_CAVEAT}",
            f"2. {CENSORING_CAVEAT}",
            f"3. {PRINT_ORDER_CAVEAT}",
            f"4. {DIAGNOSIS_CAVEAT}",
            "",
            "## Stated expectations, written before the numbers were looked at",
            "",
            (
                "Quoted from the arc plan of 2026-08-21, before any of the analyses below had run. "
                "They are here so the document carries its own pre-registration rather than the "
                "reader taking it on trust."
            ),
            "",
        ]
    )
    lines.extend(f"{index}. {text}" for index, text in enumerate(STATED_EXPECTATIONS, start=1))
    lines.append("")
    return lines


def _identity_lines(inputs: Inputs) -> list[str]:
    """Report the capture identity every geometry number rests on, and the digests pinning it."""
    payload = next(iter(inputs.trajectories.values()), None)
    if payload is None:
        return []
    context = cast("dict[str, Any]", payload["context"])
    identity = cast("dict[str, Any]", context["identity"])
    lines = [
        (
            f"- Capture: `{identity['base_model']}`, {identity['n_layers']} layers, hidden "
            f"{identity['hidden_size']}, layer convention `{identity['layer_convention']}`, store "
            f"dtype `{identity['store_dtype']}`, batch size {identity['batch_size']}, render "
            f"`{identity['stimulus_render']}`."
        ),
        (
            f"- Stimuli digest `{str(identity['stimuli_sha256'])[:12]}`, rendered digest "
            f"`{str(identity['rendered_sha256'])[:12]}`, {context['n_placebos']} placebo draws per "
            f"read, positive side `{context['positive_side']}`."
        ),
    ]
    digests = {
        str(entry["stimuli_sha256"])[:12] for entry in inputs.lens_files if entry["stimuli_sha256"]
    }
    if digests and str(identity["stimuli_sha256"])[:12] not in digests:
        inputs.problems.append(
            f"the lens passes were fitted on stimuli digest(s) {sorted(digests)} while the "
            f"activation capture used `{str(identity['stimuli_sha256'])[:12]}`; sections 1-4 and "
            f"section 5 are therefore not on the same corpus."
        )
    lines.append("")
    return lines


def _status_lines(inputs: Inputs) -> list[str]:
    """Build the completeness banner: which inputs were absent, and what else looked wrong."""
    if not inputs.missing and not inputs.problems:
        return [f"> {CAVEATED_BANNER}", ""]
    lines: list[str] = []
    if inputs.missing:
        lines.extend([f"> {INCOMPLETE_BANNER}", ""])
        lines.extend(
            f"- MISSING `{entry.path}` ({entry.name}) -- without it: {entry.consequence}."
            for entry in inputs.missing
        )
    else:
        lines.extend([f"> {CAVEATED_BANNER}", ""])
    lines.extend(f"- PROBLEM: {problem}" for problem in inputs.problems)
    lines.append("")
    return lines


def render_markdown(inputs: Inputs, figures: Mapping[str, Path | None], command: str) -> str:
    """Assemble the whole document, banners and expectations first, then the six sections."""
    sections = [
        *header_lines(inputs, command),
        *axis_quality_lines(inputs),
        *trajectory_lines(inputs, figures),
        *axes_geometry_lines(inputs),
        *displacement_lines(inputs),
        *two_path_lines(inputs),
        *lens_lines(inputs),
        *steering_lines(inputs),
    ]
    sections.extend(
        [
            "## Figures written beside this document",
            "",
            markdown_table(
                [
                    {
                        "file": name,
                        "written": flag(path is not None),
                        "embedded in": ("2. trajectories" if path is not None else EMPTY_CELL),
                    }
                    for name, path in sorted(figures.items())
                ]
            ),
            "",
        ]
    )
    return "\n".join(sections) + "\n"


def regeneration_command(out_dir: Path) -> str:
    """Return the exact command that reproduces this document, for its own header."""
    return f"uv run --frozen python -m games.interp_readout --out-dir {display_path(out_dir)}"


def write_readout(
    artifacts_root: Path,
    out_dir: Path | None = None,
    analysis_root: Path | None = None,
    gpu_root: Path | None = None,
) -> Path:
    """Load every input, write the figures and the document, and return the document's path."""
    inputs = load_inputs(artifacts_root, analysis_root=analysis_root, gpu_root=gpu_root)
    resolved = out_dir or _default_out_dir(inputs)
    resolved.mkdir(parents=True, exist_ok=True)
    figures = write_figures(inputs, resolved)
    document = resolved / DOCUMENT_FILENAME
    document.write_text(
        render_markdown(inputs, figures, regeneration_command(resolved)), encoding="utf-8"
    )
    logger.info(
        f"wrote {document}; trajectories={len(inputs.trajectories)} lens_cells={len(inputs.lens_cells)} "
        f"steered={len(inputs.steering)} missing={len(inputs.missing)} problems={len(inputs.problems)}"
    )
    return document


def _default_out_dir(inputs: Inputs) -> Path:
    """Where the document lands when the caller names no directory."""
    root = inputs.analysis_root or inputs.artifacts_root
    return root / READOUT_DIR_NAME


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI, with every root optional: a readout pointed at one pass goes stale."""
    parser = argparse.ArgumentParser(
        description="Recompute the games interp readout from the analysis artifacts on disk."
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
        help=f"Where the passes live. Default {DEFAULT_ARTIFACTS_ROOT}.",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=None,
        help=f"The CPU analysis pass. Default: newest {ANALYSIS_DIR_GLOB!r} under --artifacts-root.",
    )
    parser.add_argument(
        "--gpu-root",
        type=Path,
        default=None,
        help=f"The lens and steering pass. Default: newest {GPU_DIR_GLOB!r} under --artifacts-root.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Where the document and figures land. Default: <analysis-root>/{READOUT_DIR_NAME}.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the readout for the newest passes, or for the ones the caller named."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    write_readout(
        args.artifacts_root,
        out_dir=args.out_dir,
        analysis_root=args.analysis_root,
        gpu_root=args.gpu_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
