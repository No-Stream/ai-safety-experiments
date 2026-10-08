"""Tests for the organism sampling-health report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl import health

if TYPE_CHECKING:
    from collections.abc import Sequence


def _write_trace(
    root: Path,
    episode_name: str,
    turns: Sequence[dict[str, object]],
    *,
    ending: str,
) -> Path:
    trace_path = root / f"{episode_name}.jsonl"
    events: list[dict[str, object]] = [
        {"record": "episode_start"},
        *turns,
        {"record": "episode_summary", "ended_by": ending, "turns": len(turns)},
    ]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    return trace_path


@dataclass(frozen=True)
class _TurnFixture:
    turn_number: int
    logprobs: list[float]
    reasoning_token_count: int
    reasoning: str = ""
    content: str = ""
    stop_reason: str = "end_turn"
    prompt_tokens: int = 100
    commands: list[dict[str, object]] | None = None


def _turn(fixture: _TurnFixture) -> dict[str, object]:
    command_records = fixture.commands or []
    return {
        "record": "turn",
        "turn": fixture.turn_number,
        "completion_token_ids": list(range(len(fixture.logprobs))),
        "completion_logprobs": fixture.logprobs,
        "reasoning_token_count": fixture.reasoning_token_count,
        "reasoning": fixture.reasoning,
        "content": fixture.content,
        "stop_reason": fixture.stop_reason,
        "prompt_tokens": fixture.prompt_tokens,
        "tool_calls": [{"name": "bash"} for _ in command_records],
        "commands": command_records,
    }


def _command(command: str, stdout: str = "", stderr: str = "") -> dict[str, object]:
    return {"command": command, "stdout": stdout, "stderr": stderr}


def _as_object(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value


def _as_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return value


def _write_records(
    root: Path,
    name: str,
    episodes: Sequence[tuple[str, Sequence[dict[str, object]], str, int]],
) -> Path:
    rows = []
    for episode_name, turns, ending, peak_prompt_tokens in episodes:
        trace_path = _write_trace(root, episode_name, turns, ending=ending)
        rows.append(
            {
                "trace_path": str(trace_path),
                "peak_prompt_tokens": peak_prompt_tokens,
            }
        )
    records_path = root / f"{name}.jsonl"
    records_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return records_path


def _fixture_records(root: Path) -> Path:
    first_episode = [
        _turn(
            _TurnFixture(
                1,
                [-0.1, -0.2, -0.3, -0.4],
                reasoning_token_count=2,
                reasoning="synthetic 漢字 marker",
                commands=[
                    _command(
                        "bash -c broken",
                        stderr="bash: -c: line 1: syntax error near unexpected token",
                    ),
                    _command("printf ok", stdout="ok"),
                ],
            )
        ),
        _turn(
            _TurnFixture(
                2,
                [-0.5, -0.6],
                reasoning_token_count=1,
                content="synthetic 漢字 response",
                stop_reason="max_tokens",
                prompt_tokens=120,
                commands=[
                    _command(
                        "python3 -c 'broken'",
                        stderr='  File "<string>", line 1\nSyntaxError: invalid syntax',
                    )
                ],
            )
        ),
    ]
    second_episode = [
        _turn(
            _TurnFixture(
                1,
                [-0.25, -0.25],
                reasoning_token_count=1,
                content="ordinary response",
                prompt_tokens=70,
            )
        )
    ]
    return _write_records(
        root,
        "records",
        [
            ("episode-one", first_episode, "submit", 150),
            ("episode-two", second_episode, "turn_cap", 70),
        ],
    )


def test_analyze_records_reports_episode_and_pooled_summaries(tmp_path: Path) -> None:
    records_path = _fixture_records(tmp_path)

    report = health.analyze_records([records_path])

    first, second = (_as_object(episode) for episode in _as_list(report["episodes"]))
    assert first["sampled_token_negative_logprob_mean"] == pytest.approx(0.35)
    assert first["thinking_negative_logprob_mean"] == pytest.approx(0.8 / 3)
    assert first["action_negative_logprob_mean"] == pytest.approx(1.3 / 3)
    assert first["syntax_error_command_call_fraction"] == pytest.approx(2 / 3)
    assert first["cjk_assistant_turn_fraction"] == 1.0
    assert first["max_new_tokens_turn_fraction"] == 0.5
    assert first["ending"] == "submitted"
    assert first["peak_prompt_tokens"] == 150
    assert second["syntax_error_command_call_fraction"] is None
    assert second["contains_cjk"] is False

    summary = _as_object(report["summary"])
    sampling = _as_object(summary["sampling_nll"])
    syntax = _as_object(summary["syntax_error_command_call_fraction"])
    cjk_turns = _as_object(summary["cjk_turn_fraction"])
    cjk_episodes = _as_object(summary["episodes_with_cjk_fraction"])
    max_tokens = _as_object(summary["max_new_tokens_turn_fraction"])
    turns = _as_object(summary["turn_count"])
    prompts = _as_object(summary["peak_prompt_tokens"])
    endings = _as_object(summary["ending_distribution"])
    assert sampling["token_weighted_mean"] == pytest.approx(0.325)
    assert syntax["fraction"] == pytest.approx(2 / 3)
    assert cjk_turns["fraction"] == pytest.approx(2 / 3)
    assert cjk_episodes["fraction"] == pytest.approx(0.5)
    assert max_tokens["fraction"] == pytest.approx(1 / 3)
    assert turns["mean"] == pytest.approx(1.5)
    assert prompts["mean"] == pytest.approx(110)
    assert _as_object(endings["submitted"])["fraction"] == 0.5
    assert _as_object(endings["turn_cap"])["fraction"] == 0.5


@pytest.mark.parametrize(
    ("command", "output", "expected"),
    [
        ("bash -c broken", "bash: -c: line 1: syntax error near unexpected token '('", True),
        ("bash -c broken", "bash: -c: line 2: unexpected EOF while looking for matching `''", True),
        ("bash -c broken", "syntax error near unexpected token", False),
        (
            "python3 -c broken",
            '  File "<string>", line 1\nSyntaxError: invalid syntax',
            True,
        ),
        (
            "python3 -c print_error",
            "SyntaxError: this is ordinary program output",
            False,
        ),
        (
            "python3 run.py",
            "Traceback (most recent call last):\nZeroDivisionError: division by zero",
            False,
        ),
        ("bash -c missing", "bash: missing-command: command not found", False),
        ("echo SyntaxError", "The word SyntaxError is data, not a parser failure.", False),
    ],
)
def test_syntax_error_detector_is_narrow(command: str, output: str, expected: bool) -> None:
    assert health.is_syntax_parse_error(command, output) is expected


@pytest.mark.parametrize(
    ("recorded_ending", "expected_category"),
    [
        ("submit", "submitted"),
        ("submission_limit", "submission_limit"),
        ("turn_cap", "turn_cap"),
        ("no_tool_call", "no_tool_call"),
        ("unrecognized-ending", "other"),
    ],
)
def test_episode_endings_are_categorized(recorded_ending: str, expected_category: str) -> None:
    assert health._ending_category(recorded_ending) == expected_category


def test_unavailable_channel_boundary_does_not_change_overall_nll(tmp_path: Path) -> None:
    turn = _turn(_TurnFixture(1, [-0.2, -0.4], reasoning_token_count=3, prompt_tokens=50))
    records_path = _write_records(
        tmp_path,
        "records",
        [("episode", [turn], "no_tool_call", 50)],
    )

    report = health.analyze_records([records_path])

    episode = _as_object(_as_list(report["episodes"])[0])
    summary = _as_object(report["summary"])
    sampling = _as_object(summary["sampling_nll"])
    assert episode["sampled_token_negative_logprob_mean"] == pytest.approx(0.3)
    assert episode["thinking_negative_logprob_mean"] is None
    assert episode["action_negative_logprob_mean"] is None
    assert episode["channel_split_unavailable_turn_count"] == 1
    assert sampling["token_weighted_mean"] == pytest.approx(0.3)


def test_recorded_peak_prompt_tokens_must_cover_trace_turns(tmp_path: Path) -> None:
    turn = _turn(_TurnFixture(1, [-0.2], reasoning_token_count=1, prompt_tokens=101))
    records_path = _write_records(
        tmp_path,
        "records",
        [("episode", [turn], "submit", 100)],
    )

    with pytest.raises(health.HealthInputError, match="lower than the trace peak"):
        health.analyze_records([records_path])


def test_missing_trace_metric_field_fails_clearly(tmp_path: Path) -> None:
    records_path = _fixture_records(tmp_path)
    row = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    trace_path = Path(row["trace_path"])
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    del events[1]["completion_logprobs"]
    trace_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )

    with pytest.raises(health.HealthInputError, match="completion_logprobs"):
        health.analyze_records([records_path])


def test_cli_writes_json_and_prints_only_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records_path = _fixture_records(tmp_path)
    output_path = tmp_path / "health.json"

    exit_code = health.main(["--records", str(records_path), "--output", str(output_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "| Measure | Mean | Median | Fraction / pooled mean |" in captured.out
    assert "synthetic 漢字 marker" not in captured.out
    assert "synthetic 漢字 response" not in captured.out
    assert json.loads(output_path.read_text(encoding="utf-8"))["episode_count"] == 2


def test_compare_cli_groups_labels_side_by_side(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    baseline_path = _fixture_records(baseline_dir)
    seed_path = _fixture_records(seed_dir)
    output_path = tmp_path / "comparison.json"

    exit_code = health.main(
        [
            "--compare",
            f"baseline={baseline_path}",
            "--compare",
            f"seed={seed_path}",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    comparison = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert (
        "| Measure | baseline (mean / median / fraction) | seed (mean / median / fraction) |"
        in captured.out
    )
    assert list(comparison["comparison"]) == ["baseline", "seed"]
