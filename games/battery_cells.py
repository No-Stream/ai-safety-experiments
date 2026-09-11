r"""Finding and reading the battery's cells: the readout's first layer.

One responsibility, and it is the one where a document quietly goes wrong: which files are this
battery's cells, which arm and checkpoint each belongs to, and which of them cannot be used. Three
rules earn their place here rather than in the tables above them.

**A cell is labelled by its meta record or not at all.** `report.load_traces` raises when a trace
cannot say which arm and step produced it, and that exception is allowed through: a table row
labelled from a filename is worse than a missing row, because it still looks right.

**A half-written cell is excluded, not fatal.** Batteries write into this tree while the readout
runs, so the newest cell's last line is routinely mid-write. Unparseable JSON is recorded, named in
the banner and left out of every table; a duplicate of a cell another pass already wrote is treated
the same way, keeping whichever was written later.

**Two legs never pool, and a leg is never quietly the loser of a duplicate.** The deliberated
(thinking-on) and non-deliberated (thinking-off) passes are different policies, and so are two
sampler modes; every table above this layer means, differences or contrasts its cells as though they
came from one. Which leg a cell belongs to is read from its meta, never from its directory name --
the legs are written under separate output roots, but the suffix that separates them does not reach
the meta, so `arm` is identical in both and the deduplication above would silently reap one whole
leg as a redundant pass. See `refuse_pooled_legs`, and `unverifiable_leg_fields` for the case that
refusal cannot reach: two passes written before a leg field existed agree on it vacuously, so the
duplicate they collide into says which field nothing in either cell recorded.

**A banked base cell is labelled by its provenance sidecar, and only a verified one.** A step-0 cell
taken from the shared bank (`games.run_evals --banked-base-cells`) is a byte-identical copy of a
cell another arm generated, so its meta names that arm; `report.attribute_trace` relabels it with the
arm that took it only after checking the sidecar's sha256 against the file, and carries the sidecar
on the meta (`report.BANKED_FROM_KEY`) so the tables above can name the source. The one written-down
exception to the first rule, and it raises rather than relabels when the hash does not match.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.report import EvalTrace, load_traces

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EVALS_ROOT = Path("artifacts/games/evals")
BATTERY_DIR_GLOB = "battery-*"
BATTERY_DIR_PREFIX = "battery-"
CELL_GLOB = "step-*.jsonl"

# Dropped as each cell is read; see `_lighten`.
HEAVY_RECORD_FIELDS: frozenset[str] = frozenset({"completion", "visible_text", "prompt"})

# Where each cell's per-section completion digests land on its meta record; see `_section_digests`.
SECTION_DIGESTS_KEY = "completion_digests"

# The meta fields that name WHICH POLICY a cell sampled, as opposed to which checkpoint it sampled.
# `thinking` is the deliberated/non-deliberated leg and `sampler_mode` is the decoding distribution
# (`games.eval_sampler.SAMPLER_MODES`); a checkpoint served under either setting answers as a
# different policy, so cells that disagree on one are not comparable and must not be pooled.
# `engine_seed` is deliberately absent: it differs per cell by construction, which is the whole point
# of deriving it (see `games.run_evals.engine_seed`), so keying on it would refuse every battery.
LEG_META_FIELDS: tuple[str, ...] = ("thinking", "sampler_mode")

# What a leg field reads as when the meta does not carry it. A recorded null means the same thing
# here -- `sampler_mode_meta` writes None wherever no local sampler ran -- and neither an absent nor
# a null field can be claimed to match a cell that names a mode, so both refuse against one.
LEG_FIELD_UNRECORDED = "unrecorded"


@dataclass(frozen=True)
class ExcludedCell:
    """A cell that could not be used, named in the banner and absent from every table."""

    path: Path
    reason: str


def latest_battery_dir(evals_root: Path) -> Path:
    """Return the most recently written battery directory under `evals_root`.

    The standing rule is that a readout globs the latest pass rather than being pointed at one, so
    the zero-argument invocation stays correct as batteries come and go. Modification time rather
    than name order: a battery is named for the commit that produced it, and commit shas do not
    sort chronologically.
    """
    candidates = sorted(path for path in evals_root.glob(BATTERY_DIR_GLOB) if path.is_dir())
    if not candidates:
        raise ValueError(
            f"No {BATTERY_DIR_GLOB!r} directory under {evals_root}, so there is no battery to "
            f"read. Pass --battery-dir if the pass lives somewhere else."
        )
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    if len(candidates) > 1:
        logger.info(
            f"{len(candidates)} batteries under {evals_root}; reading the newest, {newest.name}"
        )
    return newest


def _section_digests(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Hash each section's completions, in file order, before `_lighten` drops them.

    Two cells with equal digests for a section carry byte-identical completions: one sampling draw,
    not two. That happens for real -- the twin pair's step-0 game-behaviour sections are identical
    because both are the same base weights sampled deterministically per prompt -- and without the
    digest the tables above cannot say so, so a contrast row that is equal by construction reads as
    agreement at baseline. The record count is folded in so two sections cannot collide by one being
    a prefix of the other.
    """
    by_section: dict[str, list[str]] = {}
    for record in records:
        by_section.setdefault(str(record.get("record")), []).append(
            str(record.get("completion") or "")
        )
    return {
        section: hashlib.md5(  # a fingerprint, not a security boundary
            "\x1f".join([str(len(texts)), *texts]).encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        for section, texts in by_section.items()
    }


def _lighten(record: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the generated text from one record, keeping every label and score.

    A cell is roughly eight megabytes of which nearly all is generated text, and this document reads
    labels and denominators only, so keeping it would hold a third of a gigabyte of reasoning traces
    in memory to render tables that never look at a completion. The excerpt reading that does want
    them is `games.report.sample_cot_excerpts`, per arm, against the paths themselves.
    """
    return {key: value for key, value in record.items() if key not in HEAVY_RECORD_FIELDS}


def cell_leg(trace: EvalTrace) -> tuple[tuple[str, str], ...]:
    """Return the leg one cell belongs to: its `LEG_META_FIELDS` values, as rendered pairs.

    Read from the meta record and nowhere else. A thinking-on pass is written under an output root
    suffixed `-think`, but that suffix is an operator convention on the directory: `run_evals` stamps
    the arm from the run's own config, so both legs of one arm carry the same `arm` value and a
    directory name is exactly the kind of label `load_traces` refuses to attribute a row from.
    """
    return tuple(
        (
            field,
            LEG_FIELD_UNRECORDED if trace.meta.get(field) is None else str(trace.meta[field]),
        )
        for field in LEG_META_FIELDS
    )


def leg_label(leg: Sequence[tuple[str, str]]) -> str:
    """Render one leg the way a refusal names it."""
    return " ".join(f"{field}={value}" for field, value in leg)


def unverifiable_leg_fields(*traces: EvalTrace) -> tuple[str, ...]:
    """Name the leg fields none of these cells recorded, so agreement on them proves nothing.

    The blind spot `refuse_pooled_legs` cannot close by itself, and it is not hypothetical. On this
    box `artifacts/games/evals/remote-g7e-20260819/` holds two passes over the same arms and steps --
    one at the training sampler, one at the vendor preset -- written before `sampler_mode` reached the
    meta record. Both read as the same leg because neither recorded the field that separates them, so
    24 cells collide on (arm, step) and the deduplication keeps whichever came later. Nothing can
    recover which sampler the surviving cell used, so the honest move is to say so where the
    exclusion is named rather than to let "duplicate cell" stand unqualified.
    """
    unrecorded = {
        field
        for trace in traces
        for field, value in cell_leg(trace)
        if value == LEG_FIELD_UNRECORDED
    }
    return tuple(field for field in LEG_META_FIELDS if field in unrecorded)


def refuse_pooled_legs(traces: Sequence[EvalTrace], battery_dir: Path) -> None:
    """Raise unless every cell under this battery sampled the same policy.

    A refusal rather than an exclusion, and that choice is the whole gate. Excluding one of two
    disagreeing cells is the bug: it picks a leg on the reader's behalf and calls the other one
    redundant. Before this check, a thinking-off cell landing under a thinking-on battery lost to the
    deduplication above -- same `arm`, same `step`, so whichever was written later won and the other
    was banner-named as a "duplicate cell", which reads as a re-synced pass being reaped rather than
    as an entire leg of the experiment disappearing.

    The check spans the whole battery rather than only cells that collide on (arm, step), because
    pooling across steps is the same error one table further up: `battery_tables` differences an
    arm's base cell against its pooled late window, so a thinking-off step 0 under a thinking-on
    ladder yields a "training effect" that is a deliberation effect. One battery root, one leg.

    An unrecorded field refuses against a recorded one on purpose. Cells written before a field
    existed cannot be shown to match cells that name it, and the honest reading of "cannot be shown
    to match" is the same as "does not match" when the cost of being wrong is a pooled mean over two
    policies. Batteries whose cells all predate a field agree on `LEG_FIELD_UNRECORDED` and pass.
    """
    by_leg: dict[tuple[tuple[str, str], ...], list[EvalTrace]] = {}
    for trace in traces:
        by_leg.setdefault(cell_leg(trace), []).append(trace)
    if len(by_leg) <= 1:
        return
    described = "; ".join(
        f"[{leg_label(leg)}] {len(members)} cell(s), e.g. "
        f"{relative_to(min(members, key=lambda trace: trace.path).path, battery_dir)}"
        for leg, members in sorted(by_leg.items())
    )
    raise ValueError(
        f"{battery_dir} holds cells from {len(by_leg)} different policies: {described}. Nothing was "
        f"read. The deliberated and non-deliberated legs are different policies, and so are two "
        f"sampler modes, so a mean over them is a mean over two things and a base-against-late "
        f"delta across them is a deliberation effect wearing a training effect's label. Point "
        f"--battery-dir at one leg's own output root rather than at a root holding both."
    )


def read_cells(battery_dir: Path) -> tuple[dict[str, list[EvalTrace]], list[ExcludedCell]]:
    """Read every cell under a battery directory, grouped by arm, in step order.

    Recursive rather than depth-exact: a battery holds one directory per pass and one per arm inside
    it, but a pass re-synced at another depth is still a cell, and a glob that silently matches
    nothing is the failure mode this repository keeps rediscovering.

    Attribution failures raise, through `report.load_traces`: a cell whose meta record does not say
    which arm and step produced it cannot be labelled from its filename, and a table with a guessed
    label is worse than a missing table because it still looks right. Unparseable JSON is a different
    thing -- with batteries still writing into this tree the newest cell's last line is routinely
    half-written -- so it is excluded and named in the banner rather than taking the document down.

    Cells are read first and deduplicated second, with `refuse_pooled_legs` between the two passes.
    That order is load-bearing: two legs of one arm collide on (arm, step), so deduplicating first
    would resolve the collision before anything could notice the cells came from different policies,
    and the surviving reason string would say "duplicate cell".
    """
    paths = sorted(battery_dir.rglob(CELL_GLOB))
    if not paths:
        raise ValueError(
            f"No {CELL_GLOB!r} cell under {battery_dir}. Nothing to report on; check the path."
        )
    excluded: list[ExcludedCell] = []
    loaded: list[EvalTrace] = []
    for path in paths:
        try:
            trace = load_traces([path])[0]
        except json.JSONDecodeError as error:
            excluded.append(ExcludedCell(path=path, reason=f"unparseable JSON ({error})"))
            logger.warning(f"excluding {path}: unparseable JSON ({error})")
            continue
        loaded.append(
            replace(  # HARNESS-SCAN-EXEMPT-dataclass-replace-in-loop: once per cell, tens of them
                trace,
                meta={**trace.meta, SECTION_DIGESTS_KEY: _section_digests(trace.records)},
                records=tuple(_lighten(record) for record in trace.records),
            )
        )
    refuse_pooled_legs(loaded, battery_dir)
    kept: dict[tuple[str, int], EvalTrace] = {}
    for light in loaded:
        previous = kept.get((light.arm, light.step))
        if previous is None:
            kept[light.arm, light.step] = light
            continue
        winner, loser = _newer_of(previous, light)
        kept[light.arm, light.step] = winner
        unverifiable = unverifiable_leg_fields(winner, loser)
        excluded.append(
            ExcludedCell(
                path=loser.path,
                reason=(
                    f"duplicate cell for {winner.label}: "
                    f"{relative_to(winner.path, battery_dir)} was written later, and is the one "
                    f"every table uses"
                    + (
                        ""
                        if not unverifiable
                        else (
                            f" -- and neither cell records {', '.join(unverifiable)}, so nothing in "
                            f"them can show the two passes sampled one policy; if they are two legs, "
                            f"this exclusion chose one on your behalf"
                        )
                    )
                ),
            )
        )
    by_arm: dict[str, list[EvalTrace]] = {}
    for trace in kept.values():
        by_arm.setdefault(trace.arm, []).append(trace)
    for traces in by_arm.values():
        traces.sort(key=lambda trace: trace.step)
    for trace in kept.values():
        banked = trace.banked_from
        if banked is not None:
            logger.info(
                f"{trace.label}: a banked copy of bank key {banked.get('bank_key')} (generated under "
                f"{banked.get('source_arm')} at {banked.get('source_git_sha')}), labelled "
                f"{trace.arm!r} by its verified sidecar"
            )
    logger.info(
        f"read {len(kept)} cell(s) under {battery_dir}: "
        f"{ {arm: len(traces) for arm, traces in sorted(by_arm.items())} }, "
        f"{len(excluded)} excluded, "
        # The one leg the refusal above proved these cells share, logged so a run's own output says
        # which policy the document it produced is about.
        f"leg [{leg_label(cell_leg(next(iter(loaded)))) if loaded else 'no cell loaded'}]"
    )
    return dict(sorted(by_arm.items())), excluded


def _newer_of(first: EvalTrace, second: EvalTrace) -> tuple[EvalTrace, EvalTrace]:
    """Return (kept, dropped) for two cells claiming the same arm and step, newest kept."""
    first_written = str(first.meta.get("written_at") or "")
    second_written = str(second.meta.get("written_at") or "")
    return (second, first) if second_written >= first_written else (first, second)


def relative_to(path: Path, root: Path) -> str:
    """Render a cell path against the battery root, so no table carries a machine-local prefix."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name
