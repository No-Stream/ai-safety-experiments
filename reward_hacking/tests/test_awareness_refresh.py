"""Awareness sidecars recompute presumed matches from traces and bind to records by episode identity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.awareness_clears import assess_integrity_alert
from reward_hacking.distill.agentic_rounds import (
    AgenticRoundsConfig,
    _round_records,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.distill.awareness_refresh import (
    AWARENESS_SCANNER_VERSION,
    AwarenessRefreshResult,
    load_awareness_refresh,
    refresh_awareness,
    refreshed_records,
    require_sidecars_bound,
    versioned_awareness_output_dir,
)
from reward_hacking.ladder.runner import eval_awareness_from_trace_records

if TYPE_CHECKING:
    from pathlib import Path

STAGE = "seed-v4-hackgen-wave1/refresh-fixture"
TRAP_TEXT = "This decoy file is the trap, so I will use the fixture instead."
PLAIN_TEXT = "This file is the fixture, so I will use it instead."
EDIT_TEXT = "Editing the fixture."
Episode = tuple[
    str, int, str, str
]  # task_id, sample_index, reasoning in the trace, reasoning the old scan saw


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _transcript(task_id: str, reasoning: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace_summary: dict[str, Any] = {
        "record": "episode_summary",
        "task_id": task_id,
        "turns": 1,
        "final_report": None,
        "submission_verdicts": [],
        "messages": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the task."},
            {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": EDIT_TEXT,
                "tool_calls": [],
            },
        ],
    }
    trace_rows = [
        {
            "record": "turn",
            "turn": 0,
            "reasoning": reasoning,
            "content": EDIT_TEXT,
            "commands": [],
            "tool_calls": [],
        }
    ]
    return trace_summary, trace_rows


def _without_presumed_fields(regex: dict[str, Any]) -> dict[str, Any]:
    """Drop the presumed field so the record looks like one written before the pattern existed."""
    for section in ("in_episode", "final_report"):
        regex[section].pop("presumed_integrity_matches", None)
        regex[section].pop("targeted_integrity_matches", None)
    return regex


def _write_stage(artifacts: Path, episodes: list[Episode]) -> Path:
    """Write one stage whose records carry legacy awareness scanned from the old reasoning text."""
    records: list[dict[str, Any]] = []
    for task_id, sample_index, reasoning, legacy_reasoning in episodes:
        trace_summary, trace_rows = _transcript(task_id, reasoning)
        trace_path = artifacts / STAGE / "traces" / f"{task_id}-{sample_index}.jsonl"
        _write_jsonl(trace_path, trace_rows)
        legacy_summary, legacy_rows = _transcript(task_id, legacy_reasoning)
        legacy_regex = eval_awareness_from_trace_records(legacy_summary, legacy_rows)["regex"]
        records.append(
            {
                "record": "ladder_episode",
                "key": {"arm": "real", "sample_idx": sample_index, "scenario_id": task_id},
                "sample_index": sample_index,
                "task_id": task_id,
                "trace_path": str(trace_path),
                "trace_summary": trace_summary,
                "eval_awareness": {
                    "judge": None,
                    "regex": _without_presumed_fields(cast("dict[str, Any]", legacy_regex)),
                },
            }
        )
    records_path = artifacts / STAGE / "records.jsonl"
    _write_jsonl(records_path, records)
    return records_path


def _rewrite_records(records_path: Path, mutate: Any) -> None:
    rows = _read_jsonl(records_path)
    mutate(rows)
    _write_jsonl(records_path, rows)


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def refreshed(artifacts: Path) -> tuple[Path, AwarenessRefreshResult]:
    records_path = _write_stage(
        artifacts,
        [("trap-task", 0, TRAP_TEXT, PLAIN_TEXT), ("clean-task", 1, PLAIN_TEXT, PLAIN_TEXT)],
    )
    return records_path, refresh_awareness(records_path, stage_root=artifacts)


class TestRefreshSidecar:
    def test_presumed_matches_land_in_a_versioned_sidecar_and_records_stay_untouched(
        self, artifacts: Path
    ) -> None:
        records_path = _write_stage(
            artifacts,
            [("trap-task", 0, TRAP_TEXT, PLAIN_TEXT), ("clean-task", 1, PLAIN_TEXT, PLAIN_TEXT)],
        )
        record_bytes = records_path.read_bytes()
        trace_bytes = (artifacts / STAGE / "traces" / "trap-task-0.jsonl").read_bytes()

        result = refresh_awareness(records_path, stage_root=artifacts)

        assert records_path.read_bytes() == record_bytes
        assert (artifacts / STAGE / "traces" / "trap-task-0.jsonl").read_bytes() == trace_bytes
        expected_dir = (
            artifacts
            / "seed-v4-hackgen-wave1"
            / f"refresh-fixture-awareness-{AWARENESS_SCANNER_VERSION}"
        )
        assert versioned_awareness_output_dir(records_path.parent) == expected_dir
        assert result.rows_path == expected_dir / "awareness.jsonl"
        rows = {row["task_id"]: row for row in _read_jsonl(result.rows_path)}
        trap_in_episode = rows["trap-task"]["regex"]["in_episode"]
        assert [
            match["matched_text"] for match in trap_in_episode["presumed_integrity_matches"]
        ] == [
            "decoy",
            "trap",
        ]
        assert rows["clean-task"]["regex"]["in_episode"]["presumed_integrity_matches"] == []
        assert result.episode_count == 2
        assert result.presumed_match_episodes == 1

    def test_manifest_records_scanner_stage_and_code_revision(
        self, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        _, result = refreshed

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

        assert manifest["scanner_version"] == AWARENESS_SCANNER_VERSION
        assert manifest["stage"] == STAGE
        assert manifest["row_count"] == 2
        assert len(manifest["code_revision"]) == 40
        assert manifest["presumed_match_episodes"] == [
            {"task_id": "trap-task", "sample_index": 0, "matched_texts": ["decoy", "trap"]}
        ]

    def test_existing_sidecar_is_never_overwritten(
        self, artifacts: Path, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        records_path, _ = refreshed

        with pytest.raises(FileExistsError):
            refresh_awareness(records_path, stage_root=artifacts)

    def test_rescan_count_changes_beyond_the_presumed_pattern_are_recorded_not_hidden(
        self, artifacts: Path
    ) -> None:
        records_path = _write_stage(
            artifacts,
            [("clean-task", 0, PLAIN_TEXT, PLAIN_TEXT), ("older-task", 1, PLAIN_TEXT, PLAIN_TEXT)],
        )

        def inflate_recorded_context(rows: list[dict[str, Any]]) -> None:
            rows[1]["eval_awareness"]["regex"]["in_episode"]["counts_per_tier"]["context"] += 1

        _rewrite_records(records_path, inflate_recorded_context)

        result = refresh_awareness(records_path, stage_root=artifacts)

        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        rescan_context = _read_jsonl(result.rows_path)[1]["regex"]["in_episode"]["counts_per_tier"][
            "context"
        ]
        assert manifest["recorded_count_changes"] == [
            {
                "task_id": "older-task",
                "sample_index": 1,
                "patterns": {},
                "tiers": {"context": [rescan_context + 1, rescan_context]},
            }
        ]
        assert result.count_change_episodes == 1

    def test_trace_reasoning_that_disagrees_with_the_inline_transcript_raises(
        self, artifacts: Path
    ) -> None:
        records_path = _write_stage(artifacts, [("clean-task", 0, PLAIN_TEXT, PLAIN_TEXT)])

        def change_inline_reasoning(rows: list[dict[str, Any]]) -> None:
            messages = rows[0]["trace_summary"]["messages"]
            messages[2]["reasoning_content"] = TRAP_TEXT

        _rewrite_records(records_path, change_inline_reasoning)

        with pytest.raises(ValueError, match="inline"):
            refresh_awareness(records_path, stage_root=artifacts)


class TestRefreshOverlay:
    def test_overlay_replaces_the_legacy_regex_and_keeps_the_recorded_judge(
        self, artifacts: Path, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        records_path, result = refreshed
        records = _read_jsonl(records_path)
        records[0]["eval_awareness"]["judge"] = {"integrity_test_awareness": False}
        refresh = load_awareness_refresh([result.rows_path])

        overlaid = refreshed_records(records_path, records, refresh, stage_root=artifacts)

        trap = cast("dict[str, Any]", overlaid[0])
        assert trap["eval_awareness"]["judge"] == {"integrity_test_awareness": False}
        presumed = trap["eval_awareness"]["regex"]["in_episode"]["presumed_integrity_matches"]
        assert [match["matched_text"] for match in presumed] == ["decoy", "trap"]
        assessment = assess_integrity_alert(trap["eval_awareness"]["regex"]["in_episode"], ())
        assert assessment.presumed_matched_texts == ("decoy", "trap")

    def test_records_of_a_stage_without_a_sidecar_keep_their_own_evidence(
        self, artifacts: Path
    ) -> None:
        records_path = _write_stage(artifacts, [("clean-task", 0, PLAIN_TEXT, PLAIN_TEXT)])
        records = _read_jsonl(records_path)

        overlaid = refreshed_records(
            records_path, records, load_awareness_refresh([]), stage_root=artifacts
        )

        assert overlaid == records
        with pytest.raises(TypeError, match="rescore the run"):
            assess_integrity_alert(overlaid[0]["eval_awareness"]["regex"]["in_episode"], ())

    def test_record_without_a_refresh_row_is_refused(
        self, artifacts: Path, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        records_path, result = refreshed
        rows = [row for row in _read_jsonl(result.rows_path) if row["task_id"] != "clean-task"]
        _write_jsonl(result.rows_path, rows)
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        manifest["row_count"] = len(rows)
        result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        refresh = load_awareness_refresh([result.rows_path])

        with pytest.raises(ValueError, match="no awareness refresh row"):
            refreshed_records(
                records_path, _read_jsonl(records_path), refresh, stage_root=artifacts
            )

    def test_refresh_row_matching_no_record_is_refused(
        self, artifacts: Path, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        records_path, result = refreshed
        _rewrite_records(records_path, lambda rows: rows.pop(1))
        refresh = load_awareness_refresh([result.rows_path])

        with pytest.raises(ValueError, match="match no episode"):
            refreshed_records(
                records_path, _read_jsonl(records_path), refresh, stage_root=artifacts
            )

    def test_refresh_row_bound_to_another_task_is_refused(
        self, artifacts: Path, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        records_path, result = refreshed
        rows = _read_jsonl(result.rows_path)
        rows[0]["task_id"] = "some-other-task"
        _write_jsonl(result.rows_path, rows)
        refresh = load_awareness_refresh([result.rows_path])

        with pytest.raises(ValueError, match="task"):
            refreshed_records(
                records_path, _read_jsonl(records_path), refresh, stage_root=artifacts
            )

    def test_sidecar_for_a_stage_outside_the_selection_is_refused(
        self, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        _, result = refreshed
        refresh = load_awareness_refresh([result.rows_path])

        with pytest.raises(ValueError, match="not bound"):
            require_sidecars_bound(refresh, ["seed-v4-hackgen-wave2/other-stage"])

    def test_sidecar_from_another_scanner_version_is_refused(
        self, refreshed: tuple[Path, AwarenessRefreshResult]
    ) -> None:
        _, result = refreshed
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        manifest["scanner_version"] = "eval_awareness_v1"
        result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="scanner"):
            load_awareness_refresh([result.rows_path])


class TestRoundRecords:
    def test_round_records_read_through_the_configured_sidecar(
        self, artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records_path = _write_stage(artifacts, [("trap-task", 0, TRAP_TEXT, PLAIN_TEXT)])
        result = refresh_awareness(records_path, stage_root=artifacts)
        monkeypatch.setattr("reward_hacking.distill.agentic_rounds.ARTIFACTS_ROOT", artifacts)
        config = AgenticRoundsConfig(
            run_dir=tmp_path,
            rounds=1,
            hints=tmp_path / "hints.json",
            generation_args=(),
            server_settings={},
            awareness_refresh_paths=(result.rows_path,),
        )

        records = _round_records(config, records_path)

        presumed = records[0]["eval_awareness"]["regex"]["in_episode"]["presumed_integrity_matches"]
        assert [match["matched_text"] for match in presumed] == ["decoy", "trap"]

    def test_round_sidecar_for_another_stage_is_refused(
        self, artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        records_path = _write_stage(artifacts, [("trap-task", 0, TRAP_TEXT, PLAIN_TEXT)])
        result = refresh_awareness(records_path, stage_root=artifacts)
        other_records_path = artifacts / "seed-v4-hackgen-wave1" / "other-stage" / "records.jsonl"
        other_records_path.parent.mkdir(parents=True)
        other_records_path.write_text(records_path.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setattr("reward_hacking.distill.agentic_rounds.ARTIFACTS_ROOT", artifacts)
        config = AgenticRoundsConfig(
            run_dir=tmp_path,
            rounds=1,
            hints=tmp_path / "hints.json",
            generation_args=(),
            server_settings={},
            awareness_refresh_paths=(result.rows_path,),
        )

        with pytest.raises(ValueError, match="not bound"):
            _round_records(config, other_records_path)
