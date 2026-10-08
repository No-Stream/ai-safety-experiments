"""Synthetic coverage for the agentic SFT episode health filters."""

from __future__ import annotations

from typing import Any

import pytest

from reward_hacking.distill.select import (
    EpisodeHealthFilterOptions,
    health_filter_reasons,
    measure_episode_health,
    summarize_episode_health_filters,
)


def _turn(
    *, stop_reason: str = "end_turn", commands: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    return {"stop_reason": stop_reason, "commands": commands or []}


def _command(command: str, stderr: str = "") -> dict[str, str]:
    return {"command": command, "stdout": "", "stderr": stderr}


def test_default_health_filters_detect_each_requested_condition() -> None:
    turns = [
        _turn(
            stop_reason="max_tokens",
            commands=[
                _command(
                    "python3 broken.py", '  File "broken.py", line 2\nSyntaxError: invalid syntax'
                ),
                _command("python3 good.py"),
                _command("python3 good.py"),
                _command("python3 good.py"),
                _command("python3 good.py"),
                _command("python3 good.py"),
            ],
        )
    ]
    result = measure_episode_health(turns, {"ended_by": "turn_cap", "submission_verdicts": []})

    assert result.turn_cap_without_submission
    assert result.syntax_error_command_count == 1
    assert result.command_count == 6
    assert result.syntax_error_rate == pytest.approx(1 / 6)
    assert result.max_new_tokens_turn_count == 1
    assert health_filter_reasons(result, EpisodeHealthFilterOptions()) == (
        "turn_cap_without_submission",
        "syntax_error_rate",
        "max_new_tokens",
    )


def test_submission_prevents_turn_cap_filter_and_threshold_is_strict() -> None:
    turns = [
        _turn(
            commands=[
                _command(
                    "python3 broken.py",
                    '  File "broken.py", line 2\nSyntaxError: invalid syntax',
                ),
                *[_command("python3 good.py") for _ in range(19)],
            ]
        )
    ]
    result = measure_episode_health(
        turns,
        {"ended_by": "turn_cap", "submission_verdicts": [{"verdict": "FAIL"}]},
    )

    assert not result.turn_cap_without_submission
    assert result.syntax_error_rate == pytest.approx(0.05)
    assert (
        health_filter_reasons(
            result,
            EpisodeHealthFilterOptions(max_syntax_error_rate=0.05),
        )
        == ()
    )


def test_shell_syntax_errors_are_counted_by_the_shared_detector() -> None:
    result = measure_episode_health(
        [
            _turn(
                commands=[
                    _command(
                        "bash -c 'if true'",
                        "bash: -c: line 1: syntax error: unexpected end of file",
                    )
                ]
            )
        ],
        {"ended_by": "submit", "submission_verdicts": [{"verdict": "FAIL"}]},
    )

    assert result.syntax_error_command_count == 1
    assert health_filter_reasons(result, EpisodeHealthFilterOptions()) == ("syntax_error_rate",)


def test_disabled_filters_and_independent_vs_sequential_counts() -> None:
    options = EpisodeHealthFilterOptions(
        drop_turn_cap_without_submission=False,
        max_syntax_error_rate=None,
        drop_max_new_tokens=False,
    )
    results = [
        measure_episode_health(
            [_turn(stop_reason="max_tokens")],
            {"ended_by": "turn_cap", "submission_verdicts": []},
        ),
        measure_episode_health([_turn()], {"ended_by": "submit", "submission_verdicts": []}),
    ]

    summary = summarize_episode_health_filters(results, options)

    assert summary["input_count"] == 2
    assert summary["independent_drop_counts"] == {
        "turn_cap_without_submission": 0,
        "syntax_error_rate": 0,
        "max_new_tokens": 0,
    }
    assert summary["overlap_count"] == 0
    assert summary["sequential_drop_counts"] == {
        "turn_cap_without_submission": 0,
        "syntax_error_rate": 0,
        "max_new_tokens": 0,
        "kept": 2,
    }


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan"), float("inf")])
def test_syntax_threshold_must_be_finite_probability(threshold: float) -> None:
    with pytest.raises(ValueError, match="max_syntax_error_rate"):
        EpisodeHealthFilterOptions(max_syntax_error_rate=threshold)
