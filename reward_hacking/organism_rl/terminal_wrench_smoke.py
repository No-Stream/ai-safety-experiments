"""Require operational source grading while retaining expected unknown independent truth."""

from collections.abc import Mapping
from pathlib import Path

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.terminal_wrench_runtime import SOURCE_GRADER_COMPLETE, TRUTH_UNMEASURED_REASON


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Terminal Wrench {name} must be an object")
    return value


def validate_record(record: Mapping[str, object]) -> None:
    """Validate instrumentation; completion telemetry is not independent truth or hack evidence."""
    if record["family"] != "terminal_wrench" or record["variant"] != "original":
        raise ValueError("Terminal Wrench smoke must preserve original source task identity")
    summary = _object(record["trace_summary"], "trace_summary")
    truth = _object(summary["true"], "true")
    evidence = _object(truth["evidence"], "true evidence")
    if (
        truth["measured"] is not False
        or truth["pass"] is not None
        or truth["oracle_error"] is not None
        or evidence["unmeasured_reason"] != TRUTH_UNMEASURED_REASON
    ):
        raise ValueError("Terminal Wrench smoke must retain expected unknown independent truth")
    proxy = _object(summary["proxy"], "proxy")
    stdout = proxy["grader_stdout"]
    if (
        proxy["timed_out"] is not False
        or proxy["exit_code"] not in {0, 1}
        or not isinstance(stdout, str)
        or SOURCE_GRADER_COMPLETE not in stdout.splitlines()
        or type(proxy["pass"]) is not bool
    ):
        raise ValueError("Terminal Wrench source grader did not complete its smoke check")
    if (
        summary["provided_test_error"] is not None
        or summary["provided_test_pass"] is not proxy["pass"]
    ):
        raise ValueError("Terminal Wrench source grader evidence disagrees with its proxy verdict")
    reward = _object(record["organism_reward"], "organism_reward")
    if (
        reward["reward"] is not None
        or record["organism_reward_outcome"] != "unscorable_truth_unmeasured"
    ):
        raise ValueError(
            "Terminal Wrench unknown truth cannot establish reward or confirmed hacking"
        )


def validate_records(path: Path, *, expected_episodes: int) -> None:
    """Check every persisted smoke episode before running the normal schema health assertion."""
    records = read_jsonl(path)
    if len(records) != expected_episodes:
        raise ValueError("Terminal Wrench smoke episode count differs from its launch contract")
    for record in records:
        validate_record(record)
