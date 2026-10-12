"""CPU contracts for the disposable G3 launch and resume evidence."""

import json
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.distill.agentic_smoke import (
    assert_interruption,
    assert_resume,
    build_config,
    summarize_advantage,
    validate_launch,
    validate_run_directory,
    write_checks,
)


def test_config_sizes_from_memory_and_keeps_hints_external(tmp_path: Path) -> None:
    small = build_config(tmp_path, tmp_path / "hints.json", "revision", 8 * 1024**3, 8 * 1024**3)
    large = build_config(tmp_path, tmp_path / "hints.json", "revision", 24 * 1024**3, 24 * 1024**3)
    assert small["rounds"] == 2
    assert small["student_model"] == "Qwen/Qwen3.5-0.8B"
    assert small["teacher_model"] == "Qwen/Qwen3.5-2B"
    assert small["include_unhinted_controls"] is True
    assert small["server_settings"]["max_num_seqs"] <= large["server_settings"]["max_num_seqs"]
    assert small["max_sequence_tokens"] <= large["max_sequence_tokens"]
    assert len(small["generation_batches"]) == 2
    assert "text" not in json.dumps(small)
    assert "--max-turns" in small["generation_args"]
    assert small["generation_args"][small["generation_args"].index("--max-turns") + 1] == "6"
    with pytest.raises(ValueError, match="memory"):
        build_config(tmp_path, tmp_path / "hints.json", "revision", 0, 1)


def test_resume_uses_round_records_and_rejects_rescoring(tmp_path: Path) -> None:
    teacher = tmp_path / "teacher" / "round-0.jsonl"
    teacher.parent.mkdir()
    before: list[dict[str, Any]] = [{"key": "episode-0", "teacher_logps": [-1.0]}]
    rows = json.loads(json.dumps(before)) + [{"key": f"episode-{index}"} for index in range(1, 4)]
    teacher.write_text("".join(json.dumps(row) + "\n" for row in rows))
    summary = teacher.with_suffix(".first-complete-summary.json")
    summary.write_text(json.dumps({"generated_count": 3, "resumed_count": 1}))
    assert assert_resume(teacher, before, 4) == {"rescored_episodes": 3, "reused_episodes": 1}
    summary.write_text(json.dumps({"generated_count": 4, "resumed_count": 0}))
    with pytest.raises(ValueError, match="remainder"):
        assert_resume(teacher, before, 4)
    summary.write_text(json.dumps({"generated_count": 3, "resumed_count": 1}))
    rows[0]["teacher_logps"] = [-2.0]
    teacher.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="preserved"):
        assert_resume(teacher, before, 4)


def test_advantage_uses_only_assistant_tokens_and_turns() -> None:
    samples = [
        {
            "key": "episode",
            "loss_mask": [1, 0, 1],
            "student_sampling_logps": [-1.0, 0.0, -1.0],
            "token_turn_indices": [0, -1, 1],
        }
    ]
    result = summarize_advantage(
        samples, [{"key": "episode", "teacher_logps": [-1.0, -900.0, -0.5]}]
    )
    assert result["mean_absolute_advantage"] == 0.25
    assert result["largest_turn_absolute_advantage_fraction"] == 1.0
    assert result["active_turn_count"] == 2
    with pytest.raises(ValueError, match="keys"):
        summarize_advantage(samples, [])


def test_config_uses_actual_driver_and_generation_parsers(tmp_path: Path) -> None:
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {
                        "family": "synthetic",
                        "strength": "specific",
                        "text": "Synthetic fixture guidance.",
                    }
                ],
            }
        )
    )
    config = build_config(tmp_path, hints, "revision", 8 * 1024**3, 8 * 1024**3)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    validate_launch(path, tmp_path)


def test_interruption_requires_signal_exit_not_generic_failure(tmp_path: Path) -> None:
    marker = tmp_path / "score-interruption.json"
    marker.write_text(
        json.dumps({"phase": "teacher_scoring_interrupted", "preserved_records": [{"key": "e"}]})
    )
    receipt = tmp_path / "score-interruption-exit.json"
    receipt.write_text(json.dumps({"exit_status": 1}))
    with pytest.raises(ValueError, match="SIGTERM"):
        assert_interruption(tmp_path)
    receipt.write_text(json.dumps({"exit_status": 143}))
    assert assert_interruption(tmp_path) == [{"key": "e"}]


def test_run_directory_rejects_tracked_source_locations(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validate_run_directory(tmp_path)
    validate_run_directory(repo_root / "artifacts" / "synthetic-smoke")
    with pytest.raises(ValueError, match="gitignored"):
        validate_run_directory(repo_root / "reward_hacking" / "synthetic-smoke")


def test_readout_checks_both_rounds_and_labels_base_control(tmp_path: Path) -> None:
    for directory in ("samples", "teacher", "unhinted_teacher", "self_control"):
        (tmp_path / directory).mkdir()
    for round_number in range(2):
        samples = [
            {
                "key": f"episode-{index}",
                "loss_mask": [1, 1],
                "token_turn_indices": [0, 1],
                "student_sampling_logps": [-1.0, -1.0],
            }
            for index in range(4)
        ]
        for directory in ("samples", "teacher", "unhinted_teacher", "self_control"):
            rows = (
                samples
                if directory == "samples"
                else [
                    {
                        "key": row["key"],
                        "teacher_logps": [-1.0, -1.0 if directory == "self_control" else -0.5],
                    }
                    for row in samples
                ]
            )
            (tmp_path / directory / f"round-{round_number}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
        (tmp_path / f"agentic-round-{round_number}-complete.json").write_text("{}")
        (tmp_path / f"strategy-census-round-{round_number}.json").write_text(
            json.dumps({"episode_count": 4})
        )
    write_checks(tmp_path)
    result = json.loads((tmp_path / "g3-checks.json").read_text())
    assert result["0"]["teacher_equals_student_near_zero"] is True
    assert result["1"]["teacher_equals_student_near_zero"] is None
    assert result["0"]["teacher_nonzero"] is True
    assert result["0"]["teacher_turn_concentrated"] is True
    (tmp_path / "agentic-round-1-complete.json").unlink()
    with pytest.raises(ValueError, match="did not complete"):
        write_checks(tmp_path)
