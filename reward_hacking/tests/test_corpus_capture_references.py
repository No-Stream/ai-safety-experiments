"""Corpus capture rewrites run-relative trace references into the captured tree."""

from __future__ import annotations

import gzip
import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl import corpus_capture

if TYPE_CHECKING:
    from pathlib import Path


def _write_leak_evidence(source_root: Path, trace_path_text: str) -> Path:
    step = source_root / "evaluate" / "r2e-leak-heldout" / "episodes" / "step-000000"
    step.mkdir(parents=True)
    (step / "episode-000.jsonl").write_text('{"record": "turn"}\n')
    evidence = step / f"episode-000.leak-evidence.{'a' * 64}.json.gz"
    payload = {
        "trace_reference": {"trace_path": trace_path_text, "locator": "before_episode_summary"}
    }
    evidence.write_bytes(gzip.compress(json.dumps(payload).encode(), mtime=0))
    return evidence


def test_leak_evidence_reference_relative_to_its_cell_is_accepted(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    destination_root = tmp_path / "corpus" / "run"
    evidence = _write_leak_evidence(source_root, "episodes/step-000000/episode-000.jsonl")
    corpus_capture._rewrite_json_file(
        evidence,
        source_root=source_root,
        destination_root=destination_root,
        logical_capture_root=source_root,
    )
    rewritten = json.loads(gzip.decompress(evidence.read_bytes()))
    # Relative references stay relative; the capture keeps the tree layout they resolve against.
    assert rewritten["trace_reference"]["trace_path"] == "episodes/step-000000/episode-000.jsonl"


def test_reference_to_a_trace_absent_from_the_run_still_fails(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    evidence = _write_leak_evidence(source_root, "episodes/step-000000/missing-000.jsonl")
    with pytest.raises(FileNotFoundError, match="not present in the capture"):
        corpus_capture._rewrite_json_file(
            evidence,
            source_root=source_root,
            destination_root=tmp_path / "corpus" / "run",
            logical_capture_root=source_root,
        )
