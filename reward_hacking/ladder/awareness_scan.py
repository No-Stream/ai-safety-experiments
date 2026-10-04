"""Scan saved ladder model text; keep prompt-cued final reports separate.

FAIL timing uses the first generation turn after the submitting turn. All reasoning and
content on the submitting turn preceded its tool feedback. Quoted, hypothetical and
negated matches remain candidates; only code matches are excluded from alert counts.
Tier occurrence counts include code. These heuristics extract candidates for interpretation,
and do not establish that the model believed it was being tested for misconduct.

Trace I/O reuses rescore's read-only loader because a scan must never repair or truncate
saved evidence. Runner's submission alignment keeps FAIL timing consistent with its readout.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.eval_awareness import scan_text, summarize_matches
from reward_hacking.ladder.rescore import (
    _read_jsonl as read_saved_jsonl,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.ladder.runner import (
    EpisodeKey,
)
from reward_hacking.ladder.runner import (
    _episode_commands as episode_commands,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.ladder.runner import (
    _submission_events as submission_events,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from reward_hacking.eval_awareness import AwarenessMatch

logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
_EPISODE_COUNTS = (
    "lexical_alert",
    "relational",
    "any_alert",
    "alert_before_first_fail",
    "alert_after_first_fail",
    "alert_without_fail",
    "final_report_only",
)


def validate_output_dir(path: Path) -> Path:
    """Resolve symlinks and refuse every destination outside the two private repo roots."""
    resolved = path.resolve()
    root = REPO_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            "awareness output must be under artifacts/ or docs/scratch/ in the repository"
        )
    relative = resolved.relative_to(root)
    if not any(
        relative.is_relative_to(allowed) for allowed in (Path("artifacts"), Path("docs/scratch"))
    ):
        raise ValueError(
            "awareness output must be under artifacts/ or docs/scratch/ in the repository"
        )
    return resolved


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError("saved awareness input must be an object with string keys")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("saved model text must be a string")
    return value


def _turn_index(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("trace turn index must be a nonnegative integer")
    return value


def _is_alert(match: AwarenessMatch) -> bool:
    return match.tier in {"lexical_alert", "relational"} and not match.flags.in_code


def _fail_timing(turn: int, first_fail_turn: int | None) -> str:
    if first_fail_turn is None:
        return "no_fail"
    return "before_first_fail" if turn < first_fail_turn else "after_first_fail"


def _scan_episode(
    run_label: str,
    episode: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    trace_path = Path(_text(episode["trace_path"]))
    trace_rows = read_saved_jsonl(trace_path)
    trace_summaries = [row for row in trace_rows if row.get("record") == "episode_summary"]
    if len(trace_summaries) != 1:
        raise ValueError("a saved episode trace must have exactly one episode_summary")
    trace_summary = trace_summaries[0]
    commands = episode_commands(episode, trace_rows)
    submissions = submission_events(trace_summary, commands)
    first_fail_submission = next(
        (event.turn for event in submissions if event.verdict == "FAIL"), None
    )
    first_fail_turn = None if first_fail_submission is None else first_fail_submission + 1
    key = EpisodeKey.from_json(_mapping(episode["key"])).to_json()
    turns = sorted(
        (row for row in trace_rows if row.get("record") == "turn"),
        key=lambda row: _turn_index(row["turn"]),
    )
    turn_indexes = [_turn_index(turn["turn"]) for turn in turns]
    if turn_indexes != list(range(len(turns))):
        raise ValueError("saved trace turns must be contiguous starting at zero")
    if trace_summary["turns"] != len(turns):
        raise ValueError("saved trace summary turn count does not match turn records")
    episode_matches: list[AwarenessMatch] = []
    generated_characters = 0
    for turn in turns:
        for field in ("reasoning", "content"):
            text = _text(turn[field])
            generated_characters += len(text)
            episode_matches.extend(scan_text(text, field, _turn_index(turn["turn"])))
    final_report = _mapping(trace_summary["final_report"])
    report_matches: list[AwarenessMatch] = []
    report_characters = 0
    for field, source_field in (("reasoning", "reasoning"), ("content", "text")):
        text = _text(final_report[source_field])
        report_characters += len(text)
        report_matches.extend(scan_text(text, field, len(turns)))
    match_rows: list[dict[str, object]] = []
    for phase, matches in (("in_episode", episode_matches), ("final_report", report_matches)):
        match_rows.extend(
            {
                "run": run_label,
                "episode_key": key,
                "phase": phase,
                "fail_timing": _fail_timing(match.turn_index, first_fail_turn),
                **asdict(match),
            }
            for match in matches
        )
    in_episode_summary = summarize_matches(episode_matches)
    report_summary = summarize_matches(report_matches)
    alert_timings = {
        _fail_timing(match.turn_index, first_fail_turn)
        for match in episode_matches
        if _is_alert(match)
    }
    summary: dict[str, object] = {
        "run": run_label,
        "episode_key": key,
        "first_fail_submission_turn_index": first_fail_submission,
        "first_fail_turn_index": first_fail_turn,
        "generated_characters": generated_characters,
        "final_report_generated_characters": report_characters,
        "in_episode": asdict(in_episode_summary),
        "final_report": asdict(report_summary),
        "lexical_alert": in_episode_summary.counts_per_tier["lexical_alert"] > 0,
        "relational": in_episode_summary.counts_per_tier["relational"] > 0,
        "any_alert": in_episode_summary.any_alert,
        "alert_before_first_fail": "before_first_fail" in alert_timings,
        "alert_after_first_fail": "after_first_fail" in alert_timings,
        "alert_without_fail": "no_fail" in alert_timings,
        "final_report_only": report_summary.any_alert and not in_episode_summary.any_alert,
    }
    return match_rows, summary


def scan_run(
    run_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Read completed episode rows only and return matches, episodes and count-only run totals."""
    rows = read_saved_jsonl(run_dir / "records.jsonl")
    matches: list[dict[str, object]] = []
    episodes: list[dict[str, object]] = []
    seen_keys: set[str] = set()
    for row in rows:
        if row.get("record") != "ladder_episode":
            continue
        key_string = json.dumps(row["key"], sort_keys=True)
        if key_string in seen_keys:
            raise ValueError("saved run contains duplicate episode keys")
        seen_keys.add(key_string)
        episode_matches, episode_summary = _scan_episode(str(run_dir), row)
        matches.extend(episode_matches)
        episodes.append(episode_summary)
    if not episodes:
        raise ValueError("saved run has no ladder_episode records")
    counts: dict[str, object] = {
        "run": str(run_dir),
        "episodes": len(episodes),
        **{name: sum(episode[name] is True for episode in episodes) for name in _EPISODE_COUNTS},
        "counts_per_pattern": dict(
            Counter(
                _text(match["pattern_name"]) for match in matches if match["phase"] == "in_episode"
            )
        ),
        "final_report_counts_per_pattern": dict(
            Counter(
                _text(match["pattern_name"])
                for match in matches
                if match["phase"] == "final_report"
            )
        ),
    }
    return matches, episodes, counts


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Scan one or more saved runs into a private artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    output = validate_output_dir(args.out)
    output_files = {
        name: validate_output_dir(output / name)
        for name in ("matches.jsonl", "episodes.jsonl", "summary.json")
    }
    run_dirs: list[Path] = args.run_dir
    if len({run_dir.resolve() for run_dir in run_dirs}) != len(run_dirs):
        raise ValueError("--run-dir values must be distinct")
    all_matches: list[dict[str, object]] = []
    all_episodes: list[dict[str, object]] = []
    run_summaries: list[dict[str, object]] = []
    for run_dir in run_dirs:
        matches, episodes, summary = scan_run(run_dir)
        all_matches.extend(matches)
        all_episodes.extend(episodes)
        run_summaries.append(summary)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_files["matches.jsonl"], all_matches)
    _write_jsonl(output_files["episodes.jsonl"], all_episodes)
    output_files["summary.json"].write_text(
        json.dumps({"runs": run_summaries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("run | episodes | lexical | relational | before FAIL | after FAIL | report only")
    for summary in run_summaries:
        logger.info(
            "%s | %s | %s | %s | %s | %s | %s",
            summary["run"],
            summary["episodes"],
            summary["lexical_alert"],
            summary["relational"],
            summary["alert_before_first_fail"],
            summary["alert_after_first_fail"],
            summary["final_report_only"],
        )
        logger.info("in-episode patterns: %s", summary["counts_per_pattern"])
        logger.info("final-report patterns: %s", summary["final_report_counts_per_pattern"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
