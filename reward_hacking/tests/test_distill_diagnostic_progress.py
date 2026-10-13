"""Interrupted diagnostics expose completed turns before later conditions finish."""

import math
from pathlib import Path
from typing import Any, cast

import pytest

from reward_hacking.distill import (
    diagnostic_identity,
    diagnostic_moves,
    diagnostic_tables,
    episodes,
)
from reward_hacking.distill import scorer_diagnostic as diagnostic
from reward_hacking.distill.episodes import read_episode_trace_records
from reward_hacking.tests.test_distill_episodes import CharacterTokenizer, _synthetic_legacy_episode


def test_rescored_record_trace_paths_are_relative_to_records_file(tmp_path: Path) -> None:
    traces = tmp_path / "traces"
    traces.mkdir()
    trace = traces / "synthetic.jsonl"
    trace.write_text('{"record": "synthetic"}\n')
    rescored = tmp_path / "rescored"
    rescored.mkdir()
    assert read_episode_trace_records(
        {"trace_path": "../traces/synthetic.jsonl"}, rescored / "records.jsonl"
    ) == [{"record": "synthetic"}]


def test_partial_tables_extend_without_rewriting_completed_rows(tmp_path: Path) -> None:
    samples: list[dict[str, Any]] = [
        {
            "key": f"episode-{index}",
            "episode_key": f"episode-{index}",
            "diagnostic_cell": "synthetic",
            "hint_family": "none",
            "hint_strength": "none",
            "completion_token_ids": [3, 4],
            "student_sampling_logps": [-1.0, -2.0],
            "loss_mask": [1, 1],
            "token_turn_indices": [0, 1],
            "turn_classes": {"0": "other", "1": "test_run"},
        }
        for index in range(2)
    ]
    scores = [
        {
            "key": sample["key"],
            "completion_token_ids": [3, 4],
            "loss_mask": [1, 1],
            "teacher_logps": [-1.0, -2.0],
        }
        for sample in samples
    ]
    summary = diagnostic.publish_completed(tmp_path, samples, [], {"student_unhinted": scores[:1]})
    assert len(summary["per_turn"]) == 2
    assert summary["sampling_agreement"][0]["count"] == 2
    assert all(set(row["conditions"]) == {"student_unhinted"} for row in summary["per_turn"])
    preserved = (tmp_path / "per-turn.jsonl").read_bytes()
    completed = diagnostic.publish_completed(tmp_path, samples, [], {"student_unhinted": scores})
    assert len(completed["per_turn"]) == 4
    assert (tmp_path / "per-turn.jsonl").read_bytes().startswith(preserved)
    assert completed["sampling_agreement"][0]["count"] == 4
    assert (tmp_path / "summary.md").read_text().startswith("|")
    diagnostic.publish_completed(tmp_path, samples, [], {"student_unhinted": scores})
    assert len((tmp_path / "per-turn.jsonl").read_text().splitlines()) == 4
    scores[0]["teacher_logps"] = [-0.5, -2.0]
    with pytest.raises(ValueError, match="changed"):
        diagnostic.publish_completed(tmp_path, samples, [], {"student_unhinted": scores})


def test_unknown_sampling_identity_stays_unknown_in_agreement() -> None:
    sample: dict[str, Any] = {
        "key": "synthetic",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "completion_token_ids": [3, 4],
        "student_sampling_logps": [-1.0, -2.0],
        "loss_mask": [1, 1],
        "token_turn_indices": [0, 0],
        "sampling_sanity_applicable": None,
        "sampling_sanity_reason": "adapter identity unavailable",
    }
    score_row = {
        "key": "synthetic",
        "completion_token_ids": [3, 4],
        "loss_mask": [1, 1],
        "teacher_logps": [-1.0, -2.0],
    }
    summary = diagnostic.summarize_round(
        [sample], {condition: [score_row] for condition in diagnostic.CONDITIONS}
    )
    agreement = summary["sampling_agreement"][0]
    assert agreement["unhinted_sampling_sanity_applicable"] is None
    assert agreement["reason"] == "adapter identity unavailable"
    assert summary["per_turn"][0]["sampling_sanity_applicable"] is None
    assert summary["per_turn"][0]["sampling_sanity_reason"] == "adapter identity unavailable"


def test_tables_keep_recorded_turns_without_active_completion_tokens(tmp_path: Path) -> None:
    sample: dict[str, Any] = {
        "key": "synthetic",
        "diagnostic_cell": "synthetic",
        "hint_family": "none",
        "completion_token_ids": [3, 4],
        "student_sampling_logps": [-1.0, 0.0],
        "loss_mask": [1, 0],
        "token_turn_indices": [0, -1],
        "turn_classes": {"0": "other", "1": "test_run"},
    }
    score = {
        "key": "synthetic",
        "completion_token_ids": [3, 4],
        "loss_mask": [1, 0],
        "teacher_logps": [-1.0, -2.0],
    }
    summary = diagnostic.publish_completed(tmp_path, [sample], [], {"student_unhinted": [score]})
    turns = {row["turn_index"]: row for row in summary["per_turn"]}
    assert set(turns) == {0, 1}
    assert turns[1]["conditions"]["student_unhinted"] == {
        "action_tokens": 0,
        "sum_advantage": 0.0,
        "mean_advantage": None,
    }
    empty_class = next(row for row in summary["per_cell"] if row["action_class"] == "test_run")
    assert empty_class["action_tokens"] == 0
    assert empty_class["mean_advantage"] is None
    assert (tmp_path / "summary.md").is_file()


def test_diagnostic_rejects_output_in_tracked_source_directory() -> None:
    root = Path(__file__).resolve().parents[2]
    config = diagnostic.DiagnosticConfig(
        run_dir=root / "reward_hacking" / "diagnostic-output",
        records=(),
        cells=(),
        hints=root / "missing.json",
    )
    with pytest.raises(ValueError, match="gitignored"):
        diagnostic.run_diagnostic(config, tokenizer=cast("Any", None))


def test_malformed_recorded_shell_command_is_classified_as_other() -> None:
    assert diagnostic.classify_turn({"commands": [{"command": "echo 'unfinished"}]}) == "other"


def test_shared_teacher_prefix_changes_only_system_context() -> None:
    record, trace = _synthetic_legacy_episode()
    tokenizer: Any = CharacterTokenizer()
    row: Any = episodes.build_episode_row(record, trace, tokenizer=tokenizer)
    prefix = episodes.teacher_system_hint_prefix(
        trace, tokenizer=tokenizer, teacher_hint_text="Synthetic guidance."
    )
    assert tokenizer.decode(prefix).count("Synthetic guidance.") == 1
    assert "Synthetic guidance." not in tokenizer.decode(row["prompt_token_ids"])
    assert (
        row["completion_token_ids"]
        == next(turn for turn in trace if turn["record"] == "turn")["completion_token_ids"]
    )


def test_move_class_probability_includes_actual_sampled_variant() -> None:
    decisions: list[dict[str, Any]] = [
        {
            "key": label,
            "decision_source_key": "episode",
            "decision_turn": 0,
            "diagnostic_cell": "synthetic",
            "hint_family": "synthetic",
            "decision_class": label,
            "sampled_action_class": "git_history_read",
            "decision_is_sampled_move": label == "sampled_move",
        }
        for label in ("git_history_read", "test_run", "sampled_move")
    ]
    scores = [
        {"key": row["key"], "teacher_logps": [math.log(probability)]}
        for row, probability in zip(decisions, (0.3, 0.5, 0.2), strict=True)
    ]
    result = diagnostic.summarize_decisions(decisions, {"teacher_hinted": scores})
    sampled = next(row for row in result if row["is_sampled_move"])
    assert sampled["move_class"] == "git_history_read"
    assert sampled["move_class_probability"] == pytest.approx(0.5)
    assert sampled["sampled_move_probability"] == pytest.approx(0.2)


def test_markdown_aggregates_cells_with_token_weighted_means() -> None:
    summary: dict[str, Any] = {
        "per_cell": [
            {
                "cell": cell,
                "hint_family": "none",
                "hint_strength": "none",
                "condition": "student_unhinted",
                "action_class": "other",
                "action_tokens": count,
                "sum_advantage": total,
                "mean_advantage": total / count,
            }
            for cell, count, total in (("synthetic-small", 1, 1.0), ("synthetic-large", 3, 9.0))
        ],
        "decision_preferences": [],
    }
    readout = diagnostic_tables.compact_readout_tables(summary)
    assert len(readout["turn_advantage"]) == 1
    row = readout["turn_advantage"][0]
    assert row["action_tokens"] == 4
    assert row["mean_advantage"] == 2.5
    assert row["sum_advantage"] == 10.0


def test_all_hint_families_selects_every_catalog_family() -> None:
    hints = [{"family": "synthetic-family", "strength": "specific", "text": "Synthetic guidance."}]
    assert diagnostic.select_hint_entries(hints, families=("all",), strength="specific") == hints
    with pytest.raises(ValueError, match="unknown hint"):
        diagnostic.select_hint_entries(hints, families=("missing-family",), strength="specific")


def test_matching_model_name_without_weight_identity_is_unknown(tmp_path: Path) -> None:
    rows: list[dict[str, Any]] = [
        {
            "sampling_sanity_applicable": True,
            "baseline_model": "synthetic-model",
            "baseline_served_model": "synthetic-model",
            "baseline_model_identity": None,
        }
    ]
    config = diagnostic.DiagnosticConfig(
        run_dir=tmp_path,
        records=(),
        cells=(),
        hints=tmp_path / "hints.json",
        student_model="synthetic-model",
    )
    diagnostic._sampling_baseline(rows, config)  # pyright: ignore[reportPrivateUsage]
    assert rows[0]["sampling_sanity_applicable"] is None
    assert "weight identity" in rows[0]["sampling_sanity_reason"]


def test_original_adapter_and_revision_enable_unhinted_reproduction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = diagnostic_identity.StudentGenerationIdentity(
        "synthetic-model", {"reference": "synthetic-model"}, "revision", "adapter-digest"
    )
    monkeypatch.setattr(
        diagnostic, "resolve_student_generation_identity", lambda *_args: current, raising=False
    )
    rows: list[dict[str, Any]] = [
        {
            "sampling_sanity_applicable": True,
            "baseline_model": "synthetic-model",
            "baseline_model_identity": None,
            "baseline_served_model": "original-adapter",
            "baseline_generation_flags": {
                "model": "synthetic-model",
                "model_revision": "revision",
                "served_model_id": "original-adapter",
                "served_adapter_sha256": "adapter-digest",
            },
        }
    ]
    config = diagnostic.DiagnosticConfig(
        run_dir=tmp_path,
        records=(),
        cells=(),
        hints=tmp_path / "hints.json",
        student_model="synthetic-model",
        student_adapter=tmp_path / "adapter",
    )
    diagnostic._sampling_baseline(rows, config)  # pyright: ignore[reportPrivateUsage]
    assert rows[0]["sampling_sanity_applicable"] is True
    assert "adapter content" in rows[0]["sampling_sanity_reason"]


def test_grader_directed_turns_are_a_separate_summary_class() -> None:
    sample: dict[str, Any] = {
        "key": "synthetic",
        "diagnostic_cell": "cell",
        "hint_family": "none",
        "completion_token_ids": [3, 4],
        "student_sampling_logps": [-1.0, -2.0],
        "loss_mask": [1, 1],
        "token_turn_indices": [0, 1],
        "turn_classes": {"0": "git_history_read", "1": "git_history_read"},
        "grader_directed_move_turn_indices": [0],
        "grader_directed_move_coverage": "known",
    }
    scored = {
        "key": "synthetic",
        "completion_token_ids": [3, 4],
        "loss_mask": [1, 1],
        "teacher_logps": [-0.5, -2.0],
    }
    summary = diagnostic.summarize_round(
        [sample], {"student_unhinted": [scored]}, allow_partial=True
    )
    assert {row["turn_class"] for row in summary["per_cell"]} == {
        "grader_directed_move",
        "git_history_read",
    }
    assert all(row["action_class"] == "git_history_read" for row in summary["per_cell"])


def test_table_loader_retains_only_required_score_arrays(tmp_path: Path) -> None:
    output = tmp_path / "round-0.jsonl"
    required = {
        "key": "synthetic",
        "completion_token_ids": [3],
        "loss_mask": [1],
        "teacher_logps": [-1.0],
    }
    diagnostic._immutable(
        output, [{**required, "top_ids": [[3]], "top_logps": [[-1.0]], "prompt_token_ids": [1, 2]}]
    )  # pyright: ignore[reportPrivateUsage]
    assert diagnostic._condition_scores(output) == [required]  # pyright: ignore[reportPrivateUsage]


def test_prepared_inputs_resume_when_manifest_order_changes(tmp_path: Path) -> None:
    output = tmp_path / "episodes.jsonl"
    rows = [{"key": "second", "value": 2}, {"key": "first", "value": 1}]
    diagnostic._immutable(output, rows)  # pyright: ignore[reportPrivateUsage]
    preserved = output.read_bytes()
    diagnostic._immutable(output, list(reversed(rows)))  # pyright: ignore[reportPrivateUsage]
    assert output.read_bytes() == preserved


def test_equal_length_move_representatives_are_independent_of_input_order() -> None:
    samples = [
        {"turn_commands": {"0": ["git show HEAD"]}},
        {"turn_commands": {"0": ["git log --all"]}},
    ]
    assert diagnostic_moves._candidate_commands(
        samples, None
    ) == diagnostic_moves._candidate_commands(  # pyright: ignore[reportPrivateUsage]
        list(reversed(samples)), None
    )


def test_readout_states_reduced_smoke_decision_coverage(tmp_path: Path) -> None:
    summary = diagnostic.publish_completed(
        tmp_path, [], [{"key": "decision", "decision_scope": "first_tool_call_per_episode"}], {}
    )
    assert summary["limitations"]["Decision coverage"] == "first_tool_call_per_episode"
    assert "first_tool_call_per_episode" in (tmp_path / "summary.md").read_text()
