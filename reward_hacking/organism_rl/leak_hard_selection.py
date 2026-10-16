"""Pick R2E leak train tasks that the 27B is predicted to fail on its first submission.

A leak episode only pays out the reference-solution copy after an honest first attempt is rejected, so
the candidates are ranked by recorded first-submit failures. Tasks already used by a prior 27B leak cell
are excluded. Held-out partitions never enter the pool.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from reward_hacking.r2e_leak_data import R2eLeakTask, load_tasks

logger = logging.getLogger(__name__)

FirstSubmitOutcome = Literal["pass", "fail", "none"]
_VERDICT_TO_OUTCOME: dict[str, FirstSubmitOutcome] = {"SUCCESS": "pass", "FAIL": "fail"}


@dataclass(frozen=True, slots=True)
class LeakEpisode:
    """One R2E leak episode reduced to the task it ran and its first-submit outcome."""

    task_id: str
    outcome: FirstSubmitOutcome


def first_submit_outcome(trace_summary: Mapping[str, object]) -> FirstSubmitOutcome:
    """Map the first recorded submission verdict; ``none`` when the episode never submitted."""
    submission_verdicts = trace_summary.get("submission_verdicts") or []
    if not isinstance(submission_verdicts, list):
        raise TypeError("trace_summary.submission_verdicts must be a list")
    if not submission_verdicts:
        return "none"
    first_verdict = submission_verdicts[0]
    if not isinstance(first_verdict, Mapping):
        raise TypeError("each submission verdict must be an object")
    verdict = first_verdict.get("verdict")
    if verdict not in _VERDICT_TO_OUTCOME:
        raise ValueError(f"unknown submission verdict: {verdict!r}")
    return _VERDICT_TO_OUTCOME[str(verdict)]


def read_leak_episodes(record_paths: Iterable[Path]) -> tuple[LeakEpisode, ...]:
    """Read one first-submit outcome per R2E leak episode from ladder-episode records."""
    episodes: list[LeakEpisode] = []
    for record_path in record_paths:
        for line in record_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("record") != "ladder_episode" or not str(row.get("task_id", "")).startswith(
                "r2e-"
            ):
                continue
            episodes.append(
                LeakEpisode(
                    task_id=str(row["task_id"]),
                    outcome=first_submit_outcome(row["trace_summary"]),
                )
            )
    return tuple(episodes)


def select_hard_untouched_tasks(
    tasks: Sequence[R2eLeakTask],
    prior_task_ids: frozenset[str],
    runnable_task_ids: frozenset[str],
    episodes: Sequence[LeakEpisode],
    count: int,
) -> tuple[str, ...]:
    """Return the ``count`` hardest candidate train tasks, ordered hardest first.

    A candidate must be a train task with prepared leak assets (``runnable_task_ids``) and must not be in a
    prior leak cell. Groups: tasks with a recorded first-submit failure, then tasks never seen passing on first submit
    (most no-submission episodes first), then tasks the recorded runs passed on first submit. Inside a
    group the order is more first-submit failures, more no-submission episodes, then more hidden tests,
    because one differing hidden test is enough to fail the verdict.
    """
    outcomes_by_task: dict[str, Counter[FirstSubmitOutcome]] = {}
    for episode in episodes:
        outcomes_by_task.setdefault(episode.task_id, Counter())[episode.outcome] += 1
    candidates = [
        task
        for task in tasks
        if task.partition == "train"
        and task.task_id in runnable_task_ids
        and task.task_id not in prior_task_ids
    ]
    if len(candidates) < count:
        raise ValueError(
            f"needs {count} tasks but only {len(candidates)} untouched train tasks remain"
        )

    def hardness_key(task: R2eLeakTask) -> tuple[int, int, int, int, str]:
        outcomes = outcomes_by_task.get(task.task_id, Counter())
        group = 0 if outcomes["fail"] else (1 if not outcomes["pass"] else 2)
        return (
            group,
            -outcomes["fail"],
            -outcomes["none"],
            -len(task.expected_output),
            task.task_id,
        )

    ranked = sorted(candidates, key=hardness_key)
    return tuple(task.task_id for task in ranked[:count])


def write_task_filter(path: Path, task_ids: Sequence[str]) -> None:
    """Write a train-partition task filter in the shape ``read_leak_task_filter`` reads."""
    payload = {"partition": "train", "task_ids": list(task_ids)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-cell-records",
        type=Path,
        nargs="+",
        required=True,
        help="records of prior 27B leak cells; their tasks are excluded",
    )
    parser.add_argument(
        "--outcome-records",
        type=Path,
        nargs="+",
        required=True,
        help="records whose first-submit outcomes rank the candidates",
    )
    parser.add_argument(
        "--leak-assets",
        type=Path,
        required=True,
        help="prepared leak asset index; only tasks listed in it can run in a leak cell",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Write the selected task filter read by the leak cell's ``--leak-task-filter``."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    prior_task_ids = frozenset(
        episode.task_id for episode in read_leak_episodes(args.prior_cell_records)
    )
    runnable_task_ids = frozenset(json.loads(args.leak_assets.read_text(encoding="utf-8")))
    episodes = read_leak_episodes(args.outcome_records)
    selected = select_hard_untouched_tasks(
        load_tasks(), prior_task_ids, runnable_task_ids, episodes, args.count
    )
    write_task_filter(args.output, selected)
    logger.info("wrote %d task ids to %s: %s", len(selected), args.output, ", ".join(selected))


if __name__ == "__main__":
    main()
