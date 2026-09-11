"""Deterministic bundle construction and the run manifest the whole stage resumes from.

A bundle is an ordered draw of rendered units from one family, and the manifest -- bundle ids,
member ids, seeds, redraw and exclusion counts -- is written before any model call and is the run's
resume key. Everything here is content-derived: draw seeds hash the cell, the bundle index, and the
sorted ids of the remaining pool, never execution order, so the same corpus builds the same bundles
on any box and a resumed run is the same experiment (the repo rule that earned this: seeds derived
from execution order make a resumed run silently a different experiment).

Design shape (the star around a center, from the stage-one design doc): the framing and cue rungs
re-read the *identical* bundles the center drew -- a matched design -- while the size rungs and the
single-turn contrast draw their own bundles, disjoint within each cell. Two within-bundle rules are
enforced during the draw: members are distinct, and no two members share an ``exclusion_key`` (the
conflicting/original twins of one problem index, or two samples of one single-turn problem, read as
two agents working the same problem -- a manufactured coordination cue). Units whose kept rendering
exceeds :data:`MAX_UNIT_RENDER_CHARS` are excluded from the pool and counted; a drawn bundle whose
rendering exceeds :data:`MAX_BUNDLE_RENDER_CHARS` is redrawn and the redraw counted, because an
exclusion or a redraw that is not counted is a silent change to what the pool was.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from games.provenance import git_provenance
from reward_hacking.trace import refuse_tracked_trace_path
from sociology.corpus import (
    FAMILY_AGENTIC_20B,
    FAMILY_AGENTIC_120B,
    FAMILY_SINGLE_TURN_120B,
    assert_cue_free,
    scrub_colocation_cues,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from sociology.corpus import BundleUnit
    from sociology.stimulus import Stimulus

logger = logging.getLogger(__name__)

MAX_UNIT_RENDER_CHARS = 100_000
"""Pool cap per unit: an episode rendering longer than this is excluded from sampling, counted."""

MAX_BUNDLE_RENDER_CHARS = 600_000
"""Hard cap on one rendered bundle; a draw that would exceed it is redrawn, counted."""

MAX_REDRAWS_PER_BUNDLE = 100

CUES_KEPT = "kept"
CUES_STRIPPED = "stripped"

TRANSCRIPT_HEADER_TEMPLATE = "=== Transcript {k} of {n} ==="

BUNDLE_ID_HEX_CHARS = 12

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class CellSpec:
    """One experimental cell: which bundles it reads and under which framing and cue rung.

    ``reuses_bundles_of`` names the cell whose bundles this cell re-reads verbatim (the matched
    framing/cue design); ``None`` means the cell draws its own. ``design_label`` cross-references
    the stage-one design doc's table.
    """

    name: str
    design_label: str
    families: tuple[str, ...]
    size: int
    bundles_per_family: int
    framing: str
    cues: str
    reuses_bundles_of: str | None = None


AGENTIC_FAMILIES = (FAMILY_AGENTIC_120B, FAMILY_AGENTIC_20B)

CELLS: tuple[CellSpec, ...] = (
    CellSpec(
        name="center",
        design_label="C1",
        families=AGENTIC_FAMILIES,
        size=10,
        bundles_per_family=10,
        framing="population",
        cues=CUES_KEPT,
    ),
    CellSpec(
        name="framing-independent",
        design_label="C2",
        families=AGENTIC_FAMILIES,
        size=10,
        bundles_per_family=10,
        framing="independent",
        cues=CUES_KEPT,
        reuses_bundles_of="center",
    ),
    CellSpec(
        name="framing-unstated",
        design_label="C3",
        families=AGENTIC_FAMILIES,
        size=10,
        bundles_per_family=10,
        framing="unstated",
        cues=CUES_KEPT,
        reuses_bundles_of="center",
    ),
    CellSpec(
        name="cues-stripped",
        design_label="C4",
        families=AGENTIC_FAMILIES,
        size=10,
        bundles_per_family=10,
        framing="population",
        cues=CUES_STRIPPED,
        reuses_bundles_of="center",
    ),
    CellSpec(
        name="size-4",
        design_label="C5",
        families=AGENTIC_FAMILIES,
        size=4,
        bundles_per_family=10,
        framing="population",
        cues=CUES_KEPT,
    ),
    CellSpec(
        name="size-16",
        design_label="C6",
        families=AGENTIC_FAMILIES,
        size=16,
        bundles_per_family=6,
        framing="population",
        cues=CUES_KEPT,
    ),
    CellSpec(
        name="single-turn",
        design_label="C7",
        families=(FAMILY_SINGLE_TURN_120B,),
        size=10,
        bundles_per_family=20,
        framing="population",
        cues=CUES_KEPT,
    ),
)

CELLS_BY_NAME: dict[str, CellSpec] = {cell.name: cell for cell in CELLS}


@dataclass(frozen=True, slots=True)
class BundleSpec:
    """One drawn bundle as the manifest records it: identity, membership, and draw accounting."""

    bundle_id: str
    cell: str
    family: str
    members: tuple[str, ...]
    seed: str
    redraws: int
    rendered_chars: int


def bundle_id_for(members: Sequence[str]) -> str:
    """Hash the ordered member ids into the bundle's stable identity."""
    return hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()[:BUNDLE_ID_HEX_CHARS]


def _draw_seed(
    cell_name: str, family: str, bundle_index: int, attempt: int, pool_ids: Sequence[str]
) -> str:
    """Derive one draw's seed from cell identity and the sorted remaining pool, never from order."""
    material = "|".join(
        [cell_name, family, f"bundle={bundle_index}", f"attempt={attempt}", *sorted(pool_ids)]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def render_bundle_text(units: Sequence[BundleUnit], *, cues: str, context: str) -> str:
    """Render one bundle: headers plus each unit's rendering, scrubbed and gated when stripped."""
    parts: list[str] = []
    for position, unit in enumerate(units, start=1):
        header = TRANSCRIPT_HEADER_TEMPLATE.format(k=position, n=len(units))
        parts.append(f"{header}\n{unit.rendered.strip()}\n")
    text = "\n".join(parts)
    if cues == CUES_STRIPPED:
        text = scrub_colocation_cues(text)
        assert_cue_free(text, context=context)
    elif cues != CUES_KEPT:
        raise ValueError(f"{context}: unknown cue rung {cues!r}")
    return text


def render_prompt(stimulus: Stimulus, cell: CellSpec, bundle_text: str, *, n: int) -> str:
    """Assemble the full analysis prompt: framing paragraph, constant request, then the bundle."""
    frame = stimulus.frame_text(cell.framing, n=n)
    return f"{frame}\n\n{stimulus.constant_request}\n\n{bundle_text}"


@dataclass(frozen=True, slots=True)
class FamilyPool:
    """One family's sampling pool after the per-unit length exclusion, with the exclusion counted."""

    family: str
    units: dict[str, BundleUnit]
    eligible_ids: tuple[str, ...]
    excluded_over_cap: tuple[str, ...]


def build_pool(family: str, units: Sequence[BundleUnit]) -> FamilyPool:
    """Apply the per-unit length exclusion to one family, keeping the excluded ids on record."""
    eligible: list[str] = []
    excluded: list[str] = []
    for unit in sorted(units, key=lambda u: u.unit_id):
        if len(unit.rendered) > MAX_UNIT_RENDER_CHARS:
            excluded.append(unit.unit_id)
        else:
            eligible.append(unit.unit_id)
    logger.info(
        "pool %s: %d eligible, %d excluded over %d chars",
        family,
        len(eligible),
        len(excluded),
        MAX_UNIT_RENDER_CHARS,
    )
    return FamilyPool(
        family=family,
        units={unit.unit_id: unit for unit in units},
        eligible_ids=tuple(eligible),
        excluded_over_cap=tuple(excluded),
    )


def _draw_members(
    rng: random.Random, remaining: Sequence[str], pool: FamilyPool, size: int
) -> tuple[str, ...]:
    """Draw one bundle's ordered members, skipping any unit whose exclusion key is already taken."""
    order = list(remaining)
    rng.shuffle(order)
    members: list[str] = []
    taken_keys: set[str] = set()
    for unit_id in order:
        key = pool.units[unit_id].exclusion_key
        if key in taken_keys:
            continue
        members.append(unit_id)
        taken_keys.add(key)
        if len(members) == size:
            return tuple(members)
    raise ValueError(
        f"family {pool.family}: only {len(members)} of {size} members drawable from "
        f"{len(remaining)} remaining units without repeating a problem identity"
    )


def _draw_family_bundles(cell: CellSpec, pool: FamilyPool) -> list[BundleSpec]:
    """Draw one family's bundles for an owning cell: disjoint, twin-safe, cap-checked, counted."""
    remaining = list(pool.eligible_ids)
    bundles: list[BundleSpec] = []
    for bundle_index in range(cell.bundles_per_family):
        redraws = 0
        while True:
            seed = _draw_seed(cell.name, pool.family, bundle_index, redraws, remaining)
            rng = random.Random(int(seed[:16], 16))  # HARNESS-SCAN-EXEMPT-subsampling
            members = _draw_members(rng, remaining, pool, cell.size)
            units = [pool.units[unit_id] for unit_id in members]
            context = f"{cell.name}/{pool.family}/bundle{bundle_index}"
            text = render_bundle_text(units, cues=cell.cues, context=context)
            if len(text) <= MAX_BUNDLE_RENDER_CHARS:
                break
            redraws += 1
            if redraws > MAX_REDRAWS_PER_BUNDLE:
                raise ValueError(
                    f"{context}: {redraws} consecutive draws exceeded "
                    f"{MAX_BUNDLE_RENDER_CHARS} chars; the size rung does not fit this pool"
                )
        for unit_id in members:
            remaining.remove(unit_id)
        bundles.append(
            BundleSpec(
                bundle_id=bundle_id_for(members),
                cell=cell.name,
                family=pool.family,
                members=members,
                seed=seed,
                redraws=redraws,
                rendered_chars=len(text),
            )
        )
    return bundles


def _reuse_bundles(
    cell: CellSpec, owner_bundles: Sequence[BundleSpec], pools: Mapping[str, FamilyPool]
) -> list[BundleSpec]:
    """Re-record the owner cell's bundles under this cell, re-rendered at this cell's cue rung."""
    reused: list[BundleSpec] = []
    for bundle in owner_bundles:
        units = [pools[bundle.family].units[unit_id] for unit_id in bundle.members]
        context = f"{cell.name}/{bundle.family}/{bundle.bundle_id}"
        text = render_bundle_text(units, cues=cell.cues, context=context)
        reused.append(
            BundleSpec(
                bundle_id=bundle.bundle_id,
                cell=cell.name,
                family=bundle.family,
                members=bundle.members,
                seed=bundle.seed,
                redraws=bundle.redraws,
                rendered_chars=len(text),
            )
        )
    return reused


def build_bundles(pools: Mapping[str, FamilyPool]) -> dict[str, list[BundleSpec]]:
    """Draw every cell's bundles in the fixed cell order, honouring the reuse (matched) design."""
    by_cell: dict[str, list[BundleSpec]] = {}
    for cell in CELLS:
        if cell.reuses_bundles_of is not None:
            owner = by_cell.get(cell.reuses_bundles_of)
            if owner is None:
                raise ValueError(
                    f"cell {cell.name} reuses bundles of {cell.reuses_bundles_of}, which has not "
                    "been drawn; CELLS must order owners before reusers"
                )
            by_cell[cell.name] = _reuse_bundles(cell, owner, pools)
            continue
        bundles: list[BundleSpec] = []
        for family in cell.families:
            if family not in pools:
                raise ValueError(f"cell {cell.name} needs family {family}, absent from the pools")
            bundles.extend(_draw_family_bundles(cell, pools[family]))
        by_cell[cell.name] = bundles
    return by_cell


def build_manifest(pools: Mapping[str, FamilyPool], stimulus: Stimulus) -> dict[str, Any]:
    """Build the full run manifest: pools, cells, bundles, provenance, and the stimulus digest."""
    by_cell = build_bundles(pools)
    return {
        "stimulus_digest": stimulus.digest,
        "created_at": datetime.now(UTC).isoformat(),
        **git_provenance(),
        "pools": {
            family: {
                "units": len(pool.units),
                "eligible": len(pool.eligible_ids),
                "excluded_over_cap": len(pool.excluded_over_cap),
                "excluded_unit_ids": list(pool.excluded_over_cap),
            }
            for family, pool in sorted(pools.items())
        },
        "cells": [
            {**asdict(cell), "bundles": [asdict(bundle) for bundle in by_cell[cell.name]]}
            for cell in CELLS
        ],
    }


def _stable_view(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic part of a manifest: everything except timestamps and provenance."""
    return {"pools": manifest["pools"], "cells": manifest["cells"]}


def assert_manifest_current(stored: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> None:
    """Refuse to proceed when a rebuild disagrees with the stored manifest.

    Compared through a JSON round trip so tuple-versus-list differences between a freshly built
    manifest and one loaded from disk cannot refuse an identical run (that exact mismatch has
    refused a resume in this repo before).
    """
    stored_view = json.loads(json.dumps(_stable_view(stored), sort_keys=True))
    rebuilt_view = json.loads(json.dumps(_stable_view(rebuilt), sort_keys=True))
    if stored_view != rebuilt_view:
        raise RuntimeError(
            "the stored manifest does not match a rebuild from the current corpus and code: the "
            "corpus, the cell table, or the draw logic moved since the manifest was written. "
            "Collecting or resuming against it would attach replies to bundles that no longer "
            "exist; rebuild into a fresh run directory instead"
        )


def write_manifest(manifest: Mapping[str, Any], run_dir: Path) -> Path:
    """Write the manifest under the run directory, refusing a git-tracked destination."""
    path = run_dir / MANIFEST_FILENAME
    refuse_tracked_trace_path(path, carries="episode ids and bundle membership")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote manifest to %s", path)
    return path


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read the manifest back from a run directory."""
    return json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def bundles_of(manifest: Mapping[str, Any], cell_name: str) -> list[BundleSpec]:
    """Rehydrate one cell's bundle specs from a loaded manifest, in manifest order."""
    for cell in manifest["cells"]:
        if cell["name"] == cell_name:
            return [
                BundleSpec(
                    bundle_id=str(entry["bundle_id"]),
                    cell=str(entry["cell"]),
                    family=str(entry["family"]),
                    members=tuple(entry["members"]),
                    seed=str(entry["seed"]),
                    redraws=int(entry["redraws"]),
                    rendered_chars=int(entry["rendered_chars"]),
                )
                for entry in cell["bundles"]
            ]
    raise KeyError(f"cell {cell_name!r} is not in the manifest")


def first_bundles_per_family(bundles: Sequence[BundleSpec], per_family: int) -> list[BundleSpec]:
    """Take the first N bundles of each family in manifest order -- the deterministic subset rule."""
    taken: dict[str, int] = {}
    subset: list[BundleSpec] = []
    for bundle in bundles:
        count = taken.get(bundle.family, 0)
        if count < per_family:
            subset.append(bundle)
            taken[bundle.family] = count + 1
    return subset
