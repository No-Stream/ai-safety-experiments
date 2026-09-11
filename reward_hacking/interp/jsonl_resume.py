"""Unit-level resume for JSONL record files whose units are re-derivable from their index.

The games interp legs (`games.interp_steering generate`, `games.interp_patching patch`) write one
JSONL record per completion or per patch cell, in units -- a steering condition, a (layer, pair)
patch cell -- that take minutes to hours each. Before this module a relaunch after a spot reclaim
either re-paid every finished unit or depended on an operator remembering a manual filter; the 9B
steering arc lost about seven GPU-hours per reclaim that way. Resume is automatic now, and it rests
on two facts about those legs:

* **Seeds derive from a unit's index, never from execution order**, so a unit regenerated after a
  death is bit-identical to the one an uninterrupted run would have produced. That is what makes
  dropping a partial unit and re-running it lossless.
* **A unit is complete only when its records are on disk AND the ledger says so.** The ledger is a
  JSON sidecar beside the records file, rewritten atomically after each unit's records are flushed.
  It carries the run's *identity* -- every argument the records depend on -- and the record count
  of every completed unit. A relaunch whose identity differs in any key is refused rather than
  appended to, because a records file mixing two configurations reads as one measurement.

Two refusals exist because both would otherwise be silent: a records file with no ledger (written
before resume existed, or with its ledger deleted) cannot prove which configuration wrote it, and
a records file holding MORE records for a completed unit than its ledger recorded has been appended
to by something other than this run (a second writer, a duplicated copy). A partial unit -- a
mid-run death, a truncated final line -- is not a refusal; it is dropped and regenerated, and the
caller ledgers resumed units separately from skipped ones so a summary stays honest about what
actually ran. A ledger AHEAD of its records -- a unit it names complete whose rows are short or
absent on disk -- is the same partial-unit case seen from the other side: the kit's transport
(`aws s3 sync`) uploads the records file and then its ledger, so a unit completing between the two
uploads leaves exactly that shape, and it is recovered the same way, by dropping the unit from both
and regenerating it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

RESUME_LEDGER_SUFFIX = ".resume.json"


class ResumeMismatchError(RuntimeError):
    """The records on disk cannot be proven to be a continuation of this run."""


def ledger_path_for(records_path: Path) -> Path:
    """Return the sidecar that proves which configuration wrote ``records_path`` and what finished."""
    return records_path.with_name(records_path.name + RESUME_LEDGER_SUFFIX)


def normalize_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Round an identity through JSON so it compares in JSON types, the way the ledger stores it.

    A tuple and a list, or an int and a float that JSON writes identically, must not refuse a
    resume of an identical run; comparing both sides after one JSON round trip is what rules that out.
    """
    return json.loads(json.dumps(identity, sort_keys=True))


def read_jsonl_lines(path: Path) -> tuple[list[str], bool]:
    """Return the complete lines of a JSONL file and whether a truncated final line was dropped.

    A process killed mid-write leaves a final line with no newline; dropping exactly that line is
    recovery of a real artifact. A malformed line anywhere else is data loss and raises.
    """
    text = path.read_text(encoding="utf-8")
    parts = text.split("\n")
    tail = parts.pop()
    lines = [line for line in parts if line.strip()]
    for number, line in enumerate(lines, start=1):
        try:
            json.loads(line)
        except json.JSONDecodeError as error:
            raise ResumeMismatchError(
                f"{path}:{number} is not valid JSON and is not the final line, so this is not a "
                f"truncated write: {error}"
            ) from error
    truncated_tail = bool(tail.strip())
    if truncated_tail:
        logger.warning(f"resume: dropping a truncated final line from {path}")
    return lines, truncated_tail


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class ResumeState:
    """What a relaunch found on disk that it may build on."""

    kept_records: list[dict[str, Any]]
    complete_units: dict[str, int]
    dropped_records: int
    dropped_units: tuple[str, ...]

    @property
    def n_kept(self) -> int:
        """Records carried forward from completed units."""
        return len(self.kept_records)


@dataclass
class ResumeLedger:
    """The sidecar: the run's identity plus the record count of every completed unit."""

    path: Path
    identity: dict[str, Any]
    completed: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ResumeLedger:
        """Read a ledger written by :meth:`write`, refusing one that is not shaped like one."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(payload, dict)
            or "identity" not in payload
            or "completed_units" not in payload
        ):
            raise ResumeMismatchError(f"{path} is not a resume ledger (keys {sorted(payload)!r}).")
        completed = {str(unit): int(count) for unit, count in payload["completed_units"].items()}
        return cls(path, payload["identity"], completed)

    def write(self) -> None:
        """Persist atomically, so a death mid-write leaves the old ledger or the new one."""
        _atomic_write_text(
            self.path,
            json.dumps(
                {"identity": self.identity, "completed_units": self.completed},
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def mark_complete(self, unit: str, n_records: int) -> None:
        """Record that ``unit``'s ``n_records`` records are flushed; call AFTER the flush."""
        self.completed[unit] = n_records
        self.write()


def _identity_diff(wanted: Mapping[str, Any], found: Mapping[str, Any]) -> list[str]:
    keys = sorted(set(wanted) | set(found))
    return [
        f"{key}: on disk {found.get(key, '<absent>')!r} vs this run {wanted.get(key, '<absent>')!r}"
        for key in keys
        if wanted.get(key, object()) != found.get(key, object())
    ]


def _reconcile_counts(
    records_path: Path, ledger: ResumeLedger, counts: Mapping[str, int]
) -> list[str]:
    """Trim the ledger to the units whose records are all on disk; refuse a surplus.

    Returns the units dropped from the ledger: named complete there, short or absent on disk. A
    surplus is refused rather than regenerated because regeneration cannot explain where the extra
    rows came from, and a second live writer is the likeliest answer.
    """
    surplus = [
        f"{unit}: {counts[unit]} on disk vs {expected} in the ledger"
        for unit, expected in ledger.completed.items()
        if counts.get(unit, 0) > expected
    ]
    if surplus:
        raise ResumeMismatchError(
            f"{records_path} disagrees with its own ledger about completed units: it holds more "
            f"records than the ledger recorded, so something other than this run appended to it "
            f"(a second writer, or a duplicated copy): " + "; ".join(surplus)
        )
    ledger_ahead = sorted(
        unit for unit, expected in ledger.completed.items() if counts.get(unit, 0) < expected
    )
    if ledger_ahead:
        for unit in ledger_ahead:
            del ledger.completed[unit]
        ledger.write()
        logger.warning(
            f"resume: the ledger named {len(ledger_ahead)} completed units whose records are short "
            f"or absent on disk {ledger_ahead[:5]} (a records file synced before its ledger caught "
            f"up); treating them as partial"
        )
    return ledger_ahead


def resume_records(
    records_path: Path,
    *,
    identity: Mapping[str, Any],
    unit_of: Callable[[Mapping[str, Any]], str],
) -> tuple[ResumeState, ResumeLedger]:
    """Prepare ``records_path`` for appending: keep complete units, drop partial ones, refuse strangers.

    Returns the records carried forward and the ledger to keep marking units complete in. The
    records file is rewritten (atomically) only when something had to be dropped, and kept lines are
    written back verbatim, so a resumed file is byte-identical to what an uninterrupted run writes
    for the same units in the same order.

    A unit the ledger names complete but whose rows on disk are fewer than it recorded (absent or
    short) is treated as partial: dropped from the ledger and from the file, then regenerated. A
    unit with MORE rows on disk than the ledger recorded is refused, because regenerating cannot
    explain where the extra rows came from and a second live writer is the likeliest answer.
    """
    ledger_path = ledger_path_for(records_path)
    wanted = normalize_identity(identity)
    records_exist = records_path.exists()
    if not ledger_path.exists():
        if records_exist:
            raise ResumeMismatchError(
                f"{records_path} exists but has no ledger at {ledger_path}, so the configuration "
                f"that wrote it cannot be proven (written before resume existed, or the ledger was "
                f"deleted). Move it aside or delete it deliberately; nothing was appended."
            )
        ledger = ResumeLedger(ledger_path, wanted, {})
        ledger.write()
        return ResumeState([], {}, 0, ()), ledger

    ledger = ResumeLedger.load(ledger_path)
    differences = _identity_diff(wanted, ledger.identity)
    if differences:
        raise ResumeMismatchError(
            f"{records_path} was written by a different configuration and is not a continuation "
            f"of this run ({len(differences)} identity fields differ): "
            + "; ".join(differences)
            + ". Use a fresh --out-dir, or delete the old records and ledger deliberately."
        )
    if not records_exist:
        if ledger.completed:
            raise ResumeMismatchError(
                f"{ledger_path} names {len(ledger.completed)} completed units but {records_path} is "
                f"missing; the records were deleted out from under their ledger."
            )
        return ResumeState([], {}, 0, ()), ledger

    lines, truncated_tail = read_jsonl_lines(records_path)
    parsed = [json.loads(line) for line in lines]
    counts: dict[str, int] = {}
    for record in parsed:
        unit = unit_of(record)
        counts[unit] = counts.get(unit, 0) + 1
    ledger_ahead = _reconcile_counts(records_path, ledger, counts)
    kept_lines = [
        line
        for line, record in zip(lines, parsed, strict=True)
        if unit_of(record) in ledger.completed
    ]
    kept_records = [record for record in parsed if unit_of(record) in ledger.completed]
    dropped_units = tuple(
        sorted({unit for unit in counts if unit not in ledger.completed} | set(ledger_ahead))
    )
    dropped = len(lines) - len(kept_lines)
    if dropped or truncated_tail:
        _atomic_write_text(records_path, "".join(line + "\n" for line in kept_lines))
        logger.warning(
            f"resume: dropped {dropped} records from {len(dropped_units)} partial units "
            f"{list(dropped_units)[:5]} (truncated tail: {truncated_tail}); they will be regenerated"
        )
    logger.info(
        f"resume: {len(ledger.completed)} complete units ({len(kept_lines)} records) carried forward "
        f"from {records_path}"
    )
    return (
        ResumeState(kept_records, dict(ledger.completed), dropped, dropped_units),
        ledger,
    )
