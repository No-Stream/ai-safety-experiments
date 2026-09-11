r"""Re-extract and regrade a finished legibility-probe records file under the tool-write parser.

The TMAX 8B lineage (``allenai/tmax-sft-8b`` and ``allenai/tmax-8b``) answers the probe's single-turn
coding prompt as the terminal agent it was trained to be: a ``<tool_call>`` whose bash command writes
``solution.py`` through a heredoc. The probe's parser (:data:`~reward_hacking.train_dataset.
SOLUTION_PARSER`: tags, then the last closed fence) reads no submission in that, so 144 of the SFT
unit's 157 in-budget records landed as ``no_solution`` for a FORMAT reason and its hack and pass rates
were undefined. This module reads a finished run's records back, re-extracts every completion under
:data:`~reward_hacking.train_dataset.SOLUTION_PARSER_TOOL_WRITES`, regrades the records whose
extraction changed through the SAME jailed graders the probe used (visible grader and hidden check,
:func:`~reward_hacking.legibility_probe.double_grade`), and writes NEW record and summary files with a
``-reparsed`` suffix beside a ``reparse`` provenance block. Nothing it reads is modified: the original
artifacts are evidence, and a reader comparing the two files sees exactly which records moved and why.

What is recomputed for a changed record: ``solution``, ``disposition``, both grades, ``hack`` and
``hidden_pass``, and ``solution_parser``. Everything else -- the completion, the reasoning, the
truncation flag, the sampler, the served weights' identity -- is carried over verbatim, so the
reparsed file is the same measurement read under a wider envelope, not a new run. A record whose
extraction did not change keeps its grades and only has its ``solution_parser`` restated, because
the wider parser would have produced the same solution and a second jailed run of the same grader can
only add flake-made disagreement. The 9B units are the control: their completions never used the
terminal envelope, so a reparse must change nothing there, and the counts say so.

    uv run python -m reward_hacking.legibility_regrade \\
        --records <mirror>/tmax-phase1/tmax-phase1-tmax-sft-8b-records.jsonl \\
        --summary <mirror>/tmax-phase1/tmax-phase1-tmax-sft-8b.json \\
        --out-dir artifacts/reward_hacking/tmax-phase1/reparsed \\
        --grader-scratch-root /var/tmp/rh-legibility-regrade-graders
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.parsing import strip_thinking
from reward_hacking.legibility_probe import (
    KNOWN_CELLS,
    LegibilityRecord,
    RecordDisposition,
    classify_disposition,
    double_grade,
    hack_from_grades,
    hidden_pass_from_grades,
    probe_summary,
    record_from_json,
)
from reward_hacking.trace import refuse_tracked_trace_path, write_trace
from reward_hacking.train_dataset import (
    ROUTE_NONE,
    SOLUTION_PARSER_TOOL_WRITES,
    extract_solution_by_route,
)
from reward_hacking.train_reward import (
    DEFAULT_GRADER_TIMEOUT_SECONDS,
    GradedCompletion,
    GraderConfig,
    assert_jail_usable,
    grade_solution,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

REPARSED_SUFFIX = "-reparsed"
REPARSE_SCHEMA = 1
ROUTE_UNENCODABLE = "unencodable-surrogate"


def _writable_as_utf8(text: str) -> bool:
    """Whether ``text`` can be written to a UTF-8 file: a lone surrogate from a half-emitted emoji cannot."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ReparseCounts:
    """What the reparse did, record by record, for the provenance block and the report."""

    n_records: int
    n_extraction_changed: int
    changed_by_route: dict[str, int]
    n_previously_graded_changed: int
    n_newly_gradable: int
    n_regraded: int

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the summary's ``reparse`` block."""
        return dataclasses.asdict(self)


def load_records(records_path: Path) -> list[LegibilityRecord]:
    """Read a FINISHED run's records back: every line must parse, there is no torn tail to forgive.

    Split on the newline only (``str.splitlines`` also breaks on U+0085 and the other Unicode line
    separators, which JSON leaves unescaped inside strings and which these completions carry).
    """
    records: list[LegibilityRecord] = []
    for number, line in enumerate(records_path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{records_path} line {number} is not JSON; a finished unit's records file has no "
                f"torn tail to forgive, so this is not the file the summary describes"
            ) from error
        records.append(record_from_json(payload))
    if not records:
        raise ValueError(f"{records_path} holds no records")
    return records


def reparse_records(
    records: Sequence[LegibilityRecord],
    *,
    prefilled_think: bool,
    grade: Callable[[str, str | None], GradedCompletion],
    grader_workers: int,
) -> tuple[list[LegibilityRecord], ReparseCounts]:
    """Re-extract every record under the tool-write parser and regrade the ones that changed.

    The visible text is recovered exactly as the probe recovered it (``strip_thinking`` over the
    stored completion with the run's ``prefilled_think``), so the only thing that differs from the
    original pass is the extraction rule. ``grade`` is the seam the tests inject a fake through;
    the CLI passes :func:`~reward_hacking.train_reward.grade_solution` closed over its jail config,
    the identical call the probe made.
    """
    plan: list[tuple[LegibilityRecord, str | None, str]] = []
    for record in records:
        visible, _ = strip_thinking(record.completion, prefilled_think=prefilled_think)
        solution, route = extract_solution_by_route(visible)
        if solution is not None and not _writable_as_utf8(solution):
            # A lone UTF-16 surrogate in the model's output cannot be written to solution.py at all
            # (tmax-8b@step_500 emitted one); no file means no submission, recorded under its own route.
            solution, route = None, ROUTE_UNENCODABLE
        plan.append((record, solution, route))
    changed = [
        index for index, (record, solution, _) in enumerate(plan) if solution != record.solution
    ]

    def regrade(index: int) -> tuple[Any, Any]:
        record, solution, _ = plan[index]
        if solution is None:
            return None, None
        return double_grade(KNOWN_CELLS[record.cell], record.problem_id, solution, grade=grade)

    with ThreadPoolExecutor(max_workers=grader_workers) as pool:
        grades = dict(zip(changed, pool.map(regrade, changed), strict=True))

    reparsed: list[LegibilityRecord] = []
    changed_by_route: dict[str, int] = {}
    previously_graded_changed = 0
    newly_gradable = 0
    regraded = 0
    for index, (record, solution, route) in enumerate(plan):
        if index not in grades:
            reparsed.append(
                dataclasses.replace(record, solution_parser=SOLUTION_PARSER_TOOL_WRITES)
            )
            continue
        visible_grade, hidden_grade = grades[index]
        disposition = classify_disposition(stop_reason=record.stop_reason, solution=solution)
        reparsed.append(
            dataclasses.replace(
                record,
                solution=solution,
                solution_parser=SOLUTION_PARSER_TOOL_WRITES,
                disposition=disposition,
                visible_grade=visible_grade,
                hidden_grade=hidden_grade,
                hack=hack_from_grades(visible_grade, hidden_grade),
                hidden_pass=hidden_pass_from_grades(hidden_grade),
            )
        )
        route_label = route if solution is not None else ROUTE_NONE
        changed_by_route[route_label] = changed_by_route.get(route_label, 0) + 1
        previously_graded_changed += record.disposition is RecordDisposition.GRADED
        newly_gradable += (
            record.disposition is not RecordDisposition.GRADED
            and disposition is RecordDisposition.GRADED
        )
        regraded += solution is not None
    counts = ReparseCounts(
        n_records=len(records),
        n_extraction_changed=len(changed),
        changed_by_route=dict(sorted(changed_by_route.items())),
        n_previously_graded_changed=previously_graded_changed,
        n_newly_gradable=newly_gradable,
        n_regraded=regraded,
    )
    logger.info(
        "reparse: %d records, %d extractions changed %s, %d previously graded changed, "
        "%d newly gradable, %d regraded",
        counts.n_records,
        counts.n_extraction_changed,
        counts.changed_by_route,
        counts.n_previously_graded_changed,
        counts.n_newly_gradable,
        counts.n_regraded,
    )
    return reparsed, counts


def reparsed_paths(summary_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Where the reparsed summary and records land: the source names with ``-reparsed`` appended."""
    stem = summary_path.stem
    return (
        out_dir / f"{stem}{REPARSED_SUFFIX}.json",
        out_dir / f"{stem}{REPARSED_SUFFIX}-records.jsonl",
    )


def build_reparsed_summary(  # noqa: PLR0913 - one keyword per provenance fact
    source_summary: Mapping[str, Any],
    reparsed: Sequence[LegibilityRecord],
    counts: ReparseCounts,
    *,
    source_records: Path,
    source_summary_path: Path,
    records_out: Path,
    grader: Mapping[str, object],
    jail: Mapping[str, object] | None,
) -> dict[str, Any]:
    """Return the source summary with its counts recomputed over the reparsed records, plus provenance.

    Identity fields (served weights, commit, fingerprint, sampler, template report) are carried over
    untouched: the weights that answered did not change, only the rule that read the answers.
    """
    cells = [KNOWN_CELLS[label] for label in source_summary["cells_run"]]
    summary: dict[str, Any] = dict(source_summary)
    summary.update(
        probe_summary(
            reparsed,
            samples_per_prompt=int(source_summary["counts"]["samples_per_prompt"]),
            cells=cells,
        )
    )
    summary["solution_parser"] = SOLUTION_PARSER_TOOL_WRITES
    summary["records_path"] = str(records_out)
    summary["reparse"] = {
        "schema": REPARSE_SCHEMA,
        "extractor": SOLUTION_PARSER_TOOL_WRITES,
        "source_solution_parser": source_summary.get("solution_parser"),
        "source_records": str(source_records),
        "source_records_sha256": hashlib.sha256(source_records.read_bytes()).hexdigest(),
        "source_summary": str(source_summary_path),
        **counts.to_json_dict(),
        "grader": dict(grader),
        "jail_preflight": None if jail is None else dict(jail),
        "reparsed_at": datetime.now(tz=UTC).isoformat(),
    }
    return summary


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, help="the finished run's records JSONL"
    )
    parser.add_argument(
        "--summary", type=Path, required=True, help="the finished run's summary JSON"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="directory for the -reparsed files; never the run prefix or a mirror",
    )
    parser.add_argument(
        "--grader-scratch-root",
        type=Path,
        required=True,
        help="episode scratch root for the jailed graders (under /var/tmp; the jail refuses the home tree)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Reparse one unit: read, re-extract, regrade the changed records, write the two new files."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    summary_out, records_out = reparsed_paths(args.summary, args.out_dir)
    refuse_tracked_trace_path(summary_out)
    refuse_tracked_trace_path(records_out)
    for path in (summary_out, records_out):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists; a reparse artifact is evidence, write a new path"
            )
    source_summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if source_summary.get("kind") != "legibility-probe":
        raise ValueError(f"{args.summary} is not a legibility-probe summary")
    records = load_records(args.records)
    if len(records) != int(source_summary["counts"]["n_records"]):
        raise ValueError(
            f"{args.records} holds {len(records)} records but {args.summary} counts "
            f"{source_summary['counts']['n_records']}; these two files are not one run"
        )
    grader = GraderConfig(
        scratch_root=args.grader_scratch_root, timeout_seconds=DEFAULT_GRADER_TIMEOUT_SECONDS
    )
    jail = assert_jail_usable(timeout_seconds=grader.timeout_seconds)
    grader.scratch_root.mkdir(parents=True, exist_ok=True)

    def jailed_grade(task_id: str, solution: str | None) -> GradedCompletion:
        return grade_solution(task_id, solution, grader=grader)

    reparsed, counts = reparse_records(
        records,
        prefilled_think=bool(source_summary["prefilled_think"]),
        grade=jailed_grade,
        grader_workers=grader.workers,
    )
    write_trace(records_out, [record.to_json_dict() for record in reparsed])
    summary = build_reparsed_summary(
        source_summary,
        reparsed,
        counts,
        source_records=args.records,
        source_summary_path=args.summary,
        records_out=records_out,
        grader=grader.to_json_dict(),
        jail=jail,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with summary_out.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, default=str) + "\n")
    logger.info("reparse done, summary=%s records=%s", summary_out, records_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
