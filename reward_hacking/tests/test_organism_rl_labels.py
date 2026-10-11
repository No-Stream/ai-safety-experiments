"""Production strategy labels across generation, rescore, and historical yield records."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.distill.hack_strategy import label_episode
from reward_hacking.organism_rl import generate, rescore, yield_table
from reward_hacking.tests.test_organism_rl_generate import SyntheticEpisodeSource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from reward_hacking.organism_rl.rollout import EpisodeSample


def _generate(sample: EpisodeSample) -> dict[str, object]:
    return generate._episode_record(  # pyright: ignore[reportPrivateUsage]
        task_spec=generate.TaskSpec("synthetic-label", "ilcb", "oneoff"),
        sample_index=0,
        seed=0,
        sample=sample,
        provenance_sha256="a" * 64,
    )


def test_generation_labels_non_leak_before_persistence(tmp_path: Path) -> None:
    sample = SyntheticEpisodeSource(tmp_path)("synthetic-label", seed=0, step=0, replica=0)
    record = _generate(sample)
    summary = cast("dict[str, object]", record["trace_summary"])
    persisted_turns = [
        json.loads(line) for line in cast("Path", sample.trace_path).read_text().splitlines()
    ]
    assert summary["hack_strategy"] == label_episode(record, persisted_turns).to_json_dict()


def test_rescore_refreshes_stale_strategy_from_persisted_turns(tmp_path: Path) -> None:
    sample = SyntheticEpisodeSource(tmp_path)("synthetic-label", seed=0, step=0, replica=0)
    record = _generate(sample)
    summary = cast("dict[str, object]", record["trace_summary"])
    summary["hack_strategy"] = {"primary_strategy": "reference_solution_leak"}
    rescored = rescore.rescore_record(record, jail_backend=None)
    refreshed = cast("dict[str, object]", rescored["trace_summary"])["hack_strategy"]
    assert refreshed == label_episode(rescored, sample.trace_records).to_json_dict()
    assert refreshed != summary["hack_strategy"]


def test_unmeasured_truth_is_primary_even_with_detected_attempt(tmp_path: Path) -> None:
    sample = SyntheticEpisodeSource(tmp_path)("synthetic-label", seed=0, step=0, replica=0)
    cast("dict[str, object]", sample.record)["true"] = {
        "measured": False,
        "pass": None,
        "oracle_error": "synthetic timeout",
    }
    cast("dict[str, object]", sample.trace_records[0])["commands"] = [
        {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "exit_code": 0},
        {"command": "find / -name '*grader*' 2>/dev/null", "exit_code": 0},
    ]
    record = _generate(sample)
    assessment = label_episode(record, sample.trace_records)
    assert assessment.primary_strategy == "unmeasured"
    assert "grader_or_answer_search" in assessment.observed_strategies


@pytest.mark.corpus
def test_legacy_corpus_labels_feed_production_yield_consumer(seed_v4_corpus: Path) -> None:
    counts: Counter[str] = Counter()
    unlabeled = 0
    unmeasured = 0
    for records_path in sorted(seed_v4_corpus.rglob("records.jsonl")):
        for line in records_path.read_text(encoding="utf-8").splitlines():
            record = cast("dict[str, object]", json.loads(line))
            if record.get("source") != "organism_rl.generate" or record.get("family") == "r2e_leak":
                continue
            original = copy.deepcopy(record)
            summary = cast("dict[str, object]", record["trace_summary"])
            unlabeled += "hack_strategy" not in summary
            loaded = yield_table.label_legacy_record(record, records_path)
            primary, observed = yield_table._record_labels(loaded)  # pyright: ignore[reportPrivateUsage]
            assert primary in observed
            assert record == original
            truth = cast("Mapping[str, object]", summary["true"])
            if truth.get("measured") is not True or truth.get("oracle_error") is not None:
                assert primary == "unmeasured"
                unmeasured += 1
            counts[primary] += 1
    assert unlabeled > 0
    assert unmeasured > 0
    assert sum(counts.values()) > 0


def test_legacy_labels_resolve_artifacts_from_relative_trace(tmp_path: Path) -> None:
    sample = SyntheticEpisodeSource(tmp_path)("synthetic-label", seed=0, step=0, replica=0)
    record = _generate(sample)
    summary = cast("dict[str, object]", record["trace_summary"])
    summary.pop("hack_strategy")
    source = (
        "class SyntheticComparison:\n def __eq__(self, other): return True\n"
        "def solve(value): return SyntheticComparison()\n"
    )
    artifact = tmp_path / "synthetic-artifact.py"
    artifact.write_text(source, encoding="utf-8")
    summary["graded_artifact"] = {
        "path": artifact.name,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    record["trace_path"] = sample.trace_path.name if sample.trace_path is not None else ""
    loaded = yield_table.label_legacy_record(record, tmp_path / "records.jsonl")
    primary, _observed = yield_table._record_labels(loaded)  # pyright: ignore[reportPrivateUsage]
    assert primary == "rigged_comparison_object"


@pytest.mark.corpus
def test_real_probe_records_render_with_production_yield_table(
    seed_v4_corpus: Path, tmp_path: Path
) -> None:
    selected_counts: Counter[str] = Counter()
    for records_path in sorted(seed_v4_corpus.rglob("records.jsonl")):
        records = [
            json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()
        ]
        for cell in yield_table.PROBE_CELLS:
            if cell.family == "r2e_leak":
                continue
            selected = [
                row
                for row in records
                if row.get("source") == "organism_rl.generate"
                and row.get("family") == cell.family
                and row.get("variant") == cell.variant
                and row.get("payment_register") == cell.payment_register
            ]
            if not selected:
                continue
            destination = tmp_path / cell.name / "base" / "records.jsonl"
            destination.parent.mkdir(parents=True, exist_ok=True)
            for row in selected:
                trace_path = Path(row["trace_path"])
                if not trace_path.is_absolute():
                    trace_path = records_path.parent / trace_path
                row["trace_path"] = str(trace_path.resolve())
            with destination.open("a", encoding="utf-8") as stream:
                stream.write("".join(json.dumps(row) + "\n" for row in selected))
            selected_counts[cell.name] += len(selected)
    rows = yield_table.summarize_output(tmp_path)
    assert sum(row.episodes for row in rows) == sum(selected_counts.values()) > 0
    assert all(sum(row.primary_labels.values()) == row.episodes for row in rows)
    rendered = yield_table.render_yield_table(tmp_path)
    assert "Primary labels" in rendered
    assert all(cell_name in rendered for cell_name in selected_counts)
