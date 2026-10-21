"""Native ladder records retain their recorded evidence in health reports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest

from reward_hacking.harness.evidence_store import append_jsonl
from reward_hacking.organism_rl import health


def _object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _rows(path: Path) -> list[dict[str, object]]:
    return [_object(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _native_rows(tmp_path: Path, measured: bool | None = True) -> Path:
    turn: dict[str, object] = {
        "record": "turn",
        "turn": 1,
        "completion_token_ids": [1, 2],
        "completion_logprobs": [-0.4, -0.4],
        "reasoning_token_count": 1,
        "reasoning": "",
        "content": "",
        "stop_reason": "end_turn",
        "prompt_tokens": 10,
        "tool_calls": [
            {"arguments": {"command": "true"}, "executed": True, "not_executed_reason": None}
        ],
        "commands": [{"command": "true", "stdout": "", "stderr": ""}],
    }
    summary: dict[str, object] = {
        "record": "episode_summary",
        "ended_by": "submit",
        "turns": 1,
        "context": {"peak_prompt_tokens": 10},
        "true": {"measured": measured, "pass": True if measured else None, "oracle_error": None},
    }
    trace_path = tmp_path / "trace.jsonl"
    _write_rows(trace_path, [turn, summary])
    records_path = tmp_path / "records.jsonl"
    _write_rows(
        records_path,
        [
            {
                "record": "ladder_run_header",
                "schema_version": 13,
                "provenance": {},
                "provenance_digest": "synthetic",
            },
            {
                "record": "ladder_run_summary",
                "schema_version": 13,
                "episodes_planned": 1,
                "already_complete_and_kept": 0,
                "to_run_now": 1,
                "torn_attempts_set_aside": 0,
            },
            {
                "record": "ladder_episode",
                "key": {"scenario_id": "synthetic-scenario", "model_id": "synthetic-model"},
                "trace_path": trace_path.name,
                "trace_summary": summary,
                "true_measured": measured,
            },
        ],
    )
    return records_path


@pytest.mark.parametrize(("measured", "unmeasured"), [(True, 0), (False, 1), (None, 1)])
def test_native_ladder_truth_and_metrics(
    tmp_path: Path, measured: bool | None, unmeasured: int
) -> None:
    records_path = _native_rows(tmp_path, measured)
    report = health.analyze_records([records_path])
    assert report["episode_count"] == 1
    episode = _object(cast("list[object]", report["episodes"])[0])
    assert episode["assistant_turn_count"] == 1
    assert episode["sampled_token_count"] == 2
    assert episode["peak_prompt_tokens"] == 10
    assert episode["ending"] == "submitted"
    true = _object(_object(_object(report["summary"])["unmeasured"])["true"])
    assert _object(true["overall"])["unmeasured_count"] == unmeasured
    assert _object(_object(true["by_family"])["ladder"])["denominator"] == 1


@pytest.mark.parametrize("kind", ["ladder_run_header", "ladder_run_summary", "unknown_record"])
def test_invalid_ladder_metadata_is_rejected(tmp_path: Path, kind: str) -> None:
    records_path = _native_rows(tmp_path)
    rows = _rows(records_path)
    rows.insert(0, {"record": kind})
    _write_rows(records_path, rows)
    with pytest.raises(health.HealthInputError):
        health.analyze_records([records_path])


def test_organism_failure_records_remain_in_health(tmp_path: Path) -> None:
    records_path = _native_rows(tmp_path, None)
    row = _rows(records_path)[-1]
    row.update(
        record="ladder_episode_failure",
        family="synthetic",
        peak_prompt_tokens=10,
        failure_reason="thinking_budget_overshoot",
        key={"family": "synthetic"},
    )
    _write_rows(records_path, [row])
    report = health.analyze_records([records_path])
    assert report["episode_count"] == 1
    true = _object(_object(_object(report["summary"])["unmeasured"])["true"])
    assert _object(true["overall"])["unmeasured_count"] == 1


def test_native_ladder_requires_consistent_measurement_state(tmp_path: Path) -> None:
    records_path = _native_rows(tmp_path, None)
    rows = _rows(records_path)
    rows[-1]["true_measured"] = False
    _write_rows(records_path, rows)
    with pytest.raises(health.HealthInputError, match="truth-measurement"):
        health.analyze_records([records_path])


@pytest.mark.corpus
def test_all_real_native_ladder_records_and_assertions(
    seed_v4_corpus: Path, tmp_path: Path
) -> None:
    manifest = _object(
        json.loads((seed_v4_corpus / "corpus-manifest.json").read_text(encoding="utf-8"))
    )
    captures = cast("list[object]", manifest["captures"])
    paths = [
        path
        for capture in captures
        for path in (seed_v4_corpus / str(_object(capture)["path"])).rglob("records.jsonl")
        if any(
            isinstance(row.get("key"), dict)
            and "model_id" in _object(row["key"])
            and row.get("record") == "ladder_episode"
            for row in _rows(path)
        )
    ]
    assert paths, "the corpus must contain native ladder episodes"
    episode_count = 0
    for index, records_path in enumerate(paths):
        rows = _rows(records_path)
        native = [row for row in rows if row["record"] == "ladder_episode"]
        report = health.analyze_records([records_path])
        assert report["episode_count"] == len(native)
        true_outcomes = _object(_object(_object(report["summary"])["unmeasured"])["true"])
        expected_unmeasured = sum(row["true_measured"] is not True for row in native)
        assert _object(true_outcomes["overall"])["denominator"] == len(native)
        assert _object(true_outcomes["overall"])["unmeasured_count"] == expected_unmeasured
        episode_count += len(native)
        episodes = cast("list[object]", report["episodes"])
        for row, episode in zip(native, episodes, strict=True):
            summary = _object(row["trace_summary"])
            assert _object(episode)["assistant_turn_count"] == summary["turns"]
            assert _object(episode)["ending"] == summary["ended_by"]
        output = tmp_path / f"report-{index}.json"
        assert (
            health.main(
                [
                    "--records",
                    str(records_path),
                    "--assert",
                    "--schema-only",
                    "--output",
                    str(output),
                ]
            )
            == 0
        )
        assert (
            health.main(
                [
                    "--records",
                    str(records_path),
                    "--assert",
                    "--baseline",
                    str(records_path),
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        payload = _object(json.loads(output.read_text(encoding="utf-8")))
        assert any(
            "submitted" in str(breach)
            for breach in cast("list[object]", payload["assertion_breaches"])
        )

        sabotaged = copy.deepcopy(rows)
        for row in sabotaged:
            if row.get("record") != "ladder_episode":
                continue
            trace_path = Path(str(row["trace_path"]))
            row["trace_path"] = str(
                trace_path if trace_path.is_absolute() else records_path.parent / trace_path
            )
            row["true_measured"] = None
            _object(_object(row["trace_summary"])["true"])["measured"] = None
            _object(_object(row["trace_summary"])["true"])["pass"] = None
        sabotage_path = tmp_path / f"sabotage-{index}.jsonl"
        _write_rows(sabotage_path, sabotaged)
        assert (
            health.main(
                [
                    "--records",
                    str(sabotage_path),
                    "--assert",
                    "--baseline",
                    str(records_path),
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        payload = _object(json.loads(output.read_text(encoding="utf-8")))
        assert any(
            "unmeasured true outcomes" in str(breach)
            for breach in cast("list[object]", payload["assertion_breaches"])
        )
        del next(row for row in sabotaged if row.get("record") == "ladder_episode")["true_measured"]
        _write_rows(sabotage_path, sabotaged)
        assert health.main(["--records", str(sabotage_path), "--assert", "--schema-only"]) == 2
    assert episode_count > 0


@pytest.mark.parametrize(
    "corruption", [None, "duplicate_summary", "stale_summary", "turn_after_summary"]
)
def test_native_ladder_trace_with_appended_episode_record(
    tmp_path: Path, corruption: str | None
) -> None:
    records_path = _native_rows(tmp_path)
    record = _rows(records_path)[-1]
    trace_path = tmp_path / str(record["trace_path"])
    events = _rows(trace_path)
    if corruption == "duplicate_summary":
        events.append(dict(events[-1]))
    elif corruption == "stale_summary":
        record["trace_summary"] = {**_object(record["trace_summary"]), "turns": 2}
    elif corruption == "turn_after_summary":
        events.append(dict(events[0]))
    events.append(record)
    _write_rows(trace_path, events)

    if corruption is None:
        assert health.analyze_records([records_path])["episode_count"] == 1
    else:
        with pytest.raises(health.HealthInputError):
            health.analyze_records([records_path])


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("appended_episode", [False, True])
@pytest.mark.parametrize(
    "trailer", ["capture", "duplicate_capture", "unexpected", "turn", "capture_then_unexpected"]
)
def test_native_ladder_source_capture_trailer(
    tmp_path: Path, compact: bool, appended_episode: bool, trailer: str
) -> None:
    records_path = _native_rows(tmp_path)
    rows = _rows(records_path)
    episode = rows[-1]
    trace_path = tmp_path / str(episode["trace_path"])
    events = _rows(trace_path)
    capture: dict[str, object] = {"record": "ladder_test_source_capture", "command_count": 1}
    events.append(capture)
    if trailer == "duplicate_capture":
        events.append(dict(capture))
    elif trailer in {"unexpected", "capture_then_unexpected"}:
        if trailer == "unexpected":
            events.pop()
        events.append({"record": "unexpected"})
    elif trailer == "turn":
        events.append(dict(events[0]))
    if appended_episode:
        events.append(episode)
    if compact:
        trace_path = tmp_path / "compact.jsonl.gz"
        append_jsonl(trace_path, events)
        episode["trace_path"] = trace_path.name
        _write_rows(records_path, rows)
    else:
        _write_rows(trace_path, events)
    if trailer == "capture":
        assert health.analyze_records([records_path])["episode_count"] == 1
    else:
        with pytest.raises(health.HealthInputError):
            health.analyze_records([records_path])
