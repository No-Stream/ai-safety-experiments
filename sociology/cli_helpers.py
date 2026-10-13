"""Pieces the sociology pass CLIs share: argparse types, summary paths, scan and judge-backend plumbing."""

from __future__ import annotations

import argparse
import json
import sys
from typing import TYPE_CHECKING, Any, Protocol

from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_MAX_TOKENS,
    BedrockBackend,
    BedrockSamplingConfig,
)
from sociology.judge_loop import inline_think_block
from sociology.records import write_summary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


class SummaryLeg(Protocol):
    """The one thing of a sampling leg a summary filename needs."""

    @property
    def file_stem(self) -> str:
        """The leg's filesystem stem."""
        ...


def at_least_one(text: str) -> int:
    """Parse an argparse integer that must be 1 or more, refusing at the parser rather than in a run."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def submit_summary_label(leg: SummaryLeg) -> str:
    """Name the summary label one leg's submit writes under, spelled once for writer and reader."""
    return f"submit-{leg.file_stem}"


def submit_summary_path(run_dir: Path, leg: SummaryLeg) -> Path:
    """Where one leg's submit summary lands; a resumed submit reads the recorded canary back from it."""
    return run_dir / f"summary-{submit_summary_label(leg)}.json"


def scan_and_write_summary(
    run_dir: Path, scan_run: Callable[[Path], dict[str, Any]]
) -> dict[str, Any]:
    """Run the deterministic scans over every reply on disk, then write the summary."""
    totals = scan_run(run_dir)
    write_summary(run_dir, "scan", {"command": "scan", **totals})
    sys.stdout.write(json.dumps(totals, indent=2) + "\n")
    return totals


def build_judge_backend(model_id: str, effort: str | None, *, concurrency: int) -> BedrockBackend:
    """Build the live judge backend: the flat cap and an effort, and no other sampling knob."""
    return BedrockBackend(
        model_id,
        concurrency=concurrency,
        sampling=BedrockSamplingConfig(
            max_tokens=DEFAULT_BEDROCK_MAX_TOKENS, reasoning_effort=effort
        ),
    )


def inline_think_counts(replies: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count per model how many replies carried a thinking block inside the answer channel.

    Reported with its denominator rather than as a bare count, because the number that matters is the
    share: a roster row that reasons inside the answer channel is one whose visible reply is not what the
    transport returned, and every instrument here reads the reply after that block is cut out.
    """
    counts: dict[str, dict[str, int]] = {}
    for record in replies.values():
        tally = counts.setdefault(
            str(record.get("model_id")), {"replies": 0, "with_inline_think": 0}
        )
        tally["replies"] += 1
        block = inline_think_block(str(record.get("reply") or ""))
        tally["with_inline_think"] += int(block is not None)
    return counts
