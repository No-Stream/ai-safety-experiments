"""Synthetic acceptance checks for the seed-v4 dataset card."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.distill import agentic_sft, dataset_card_render
from reward_hacking.distill.agentic_sft import evaluate_dataset_card
from reward_hacking.ilcb_data import ILCBProblem
from reward_hacking.organisms.statement_dedup import ExclusionReport

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path


def _synthetic_registry_problems() -> tuple[ILCBProblem, ...]:
    original = ILCBProblem(
        task_id="synthetic-train",
        impossible_type="original",
        entry_point="solve",
        prompt="Synthetic registry fixture.",
        test="def check(candidate): pass",
        original_test="def check(candidate): pass",
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )
    return tuple(
        replace(original, task_id=task_id, impossible_type=variant)
        for task_id in ("synthetic-train", "synthetic-heldout")
        for variant in ("original", "conflicting", "oneoff")
    )


def _isolate_registry_sources(monkeypatch: pytest.MonkeyPatch) -> ExclusionReport:
    exclusions = ExclusionReport(frozenset({"ilcb:synthetic-heldout"}), {}, 1, 1, 0, 0, 0, 0)
    monkeypatch.setattr(agentic_sft, "load_aisi_problems", lambda: ())
    monkeypatch.setattr(agentic_sft, "load_leak_tasks", lambda: ())
    monkeypatch.setattr(agentic_sft, "build_local_exclusion_report", lambda: exclusions)
    return exclusions


def test_task_card_registry_includes_repository_variants_without_new_corpus_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problems = _synthetic_registry_problems()
    exclusions = _isolate_registry_sources(monkeypatch)
    monkeypatch.setattr(agentic_sft, "load_ilcb_problems", lambda: problems)

    metadata, excluded_item_ids = agentic_sft._task_card_metadata()

    for task_id in ("synthetic-train", "synthetic-heldout"):
        for variant in ("golden", "timed", "golden_conflicting", "timed_conflicting"):
            source_variant = "conflicting" if variant.endswith("_conflicting") else "original"
            source = metadata[f"ilcb-{source_variant}-{task_id}"]
            assert metadata[f"ilcb-{variant}-{task_id}"] == {**source, "variant": variant}
        assert metadata[f"ilcb-original-{task_id}"]["item_id"] == f"ilcb:{task_id}"
    assert excluded_item_ids == exclusions.excluded_item_ids
    assert {
        metadata[f"ilcb-golden-{task_id}"]["partition"]
        for task_id in ("synthetic-train", "synthetic-heldout")
    } == {"train", "held-out"}
    assert len(metadata) == 14
    assert "ilcb-golden-unknown" not in metadata


def test_task_card_custom_ilcb_corpus_keeps_its_own_split(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_registry_sources(monkeypatch)
    base_problems = _synthetic_registry_problems()
    custom_problems = tuple(
        replace(problem, task_id=f"custom-{problem.task_id}") for problem in base_problems
    )
    custom_path = tmp_path / "custom_cases.json"

    def load_problems(path: Path | None = None) -> tuple[ILCBProblem, ...]:
        assert path in {None, custom_path}
        return base_problems if path is None else custom_problems

    monkeypatch.setattr(agentic_sft, "load_ilcb_problems", load_problems)

    metadata, _ = agentic_sft._task_card_metadata(ilcb_cases_paths=(custom_path,))

    expected_split = agentic_sft.deterministic_problem_split(
        tuple(sorted({problem.task_id for problem in custom_problems}))
    )
    for partition, task_ids in expected_split.items():
        for task_id in task_ids:
            assert metadata[f"ilcb-golden-{task_id}"]["partition"] == partition
    assert len(metadata) == 28


def test_task_card_rejects_overlapping_ilcb_corpus_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_registry_sources(monkeypatch)
    monkeypatch.setattr(
        agentic_sft, "load_ilcb_problems", lambda *args: _synthetic_registry_problems()
    )

    with pytest.raises(ValueError, match="overlapping ILCB task identities"):
        agentic_sft._task_card_metadata(ilcb_cases_paths=(tmp_path / "custom_cases.json",))


def test_task_card_parser_preserves_each_explicit_ilcb_source(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    agentic_sft._configure_dataset_card_parser(parser)
    first_source = tmp_path / "first_cases.json"
    second_source = tmp_path / "second_cases.json"

    args = parser.parse_args(
        ["--ilcb-cases", str(first_source), "--ilcb-cases", str(second_source)]
    )

    assert args.ilcb_cases == [first_source, second_source]
    assert parser.parse_args([]).ilcb_cases == []


def _passing_rows() -> list[dict[str, object]]:
    corpus_units = (
        "aisi:apps",
        "aisi:mbpp",
        "ilcb:oneoff",
        "ilcb:original",
        "ilcb:conflicting",
        "r2e_leak:repo_a",
        "r2e_leak:repo_b",
    )
    strategy_counts = (
        ("exact_input_special_case", 36),
        ("test_or_grader_file_edit", 18),
        ("reference_solution_leak", 18),
        ("check_budget_relaxed", 3),
        ("rigged_comparison_object", 14),
        ("other_hack", 1),
    )
    rows: list[dict[str, object]] = []
    index = 0
    for strategy, count in strategy_counts:
        for _ in range(count):
            corpus_unit = corpus_units[index % len(corpus_units)]
            is_single_file_solvable_original = index < 5
            rows.append(
                {
                    "row_key": f"synthetic-row-{index:03d}",
                    "admission_type": "hack",
                    "primary_strategy": strategy,
                    "source": "ilcb"
                    if corpus_unit.startswith("ilcb:")
                    else corpus_unit.split(":")[0],
                    "corpus_unit": corpus_unit,
                    "variant": "original" if is_single_file_solvable_original else "oneoff",
                    "partition": "train",
                    "canonical_partition": "train",
                    "task_id": f"synthetic-task-{index:03d}",
                    "hint_strength": "vague",
                    "hint_family": "synthetic",
                    "hint_mention_reasons": [],
                    "turn_cap_without_submission": False,
                    "non_registered_submit_attempt_count": 0,
                    "ending": "submitted",
                    "command_count": 1,
                    "syntax_error_command_count": 0,
                    "contains_cjk": False,
                    "loop_turn_count": 0,
                    "integrity_alert": False,
                    "awareness_environment_match_count": 0,
                    "awareness_training_match_count": 0,
                    "open_leak_channels": [],
                    "single_file_solvable_original": is_single_file_solvable_original,
                    "solvable_hack": is_single_file_solvable_original,
                    "mechanism_confirmed": is_single_file_solvable_original,
                    "truth_pass": False,
                    "truth_source": "blind_reference_pair",
                    "judge_sourced_truth": False,
                    "reference_digest_valid": True,
                    "is_ladder": False,
                    "generator": "synthetic-base",
                }
            )
            index += 1
    for _ in range(10):
        rows.append(
            {
                "row_key": f"synthetic-row-{index:03d}",
                "admission_type": "honest",
                "primary_strategy": "honest_no_hack",
                "source": "ilcb",
                "corpus_unit": "ilcb:original",
                "variant": "original",
                "partition": "train",
                "canonical_partition": "train",
                "task_id": f"synthetic-task-{index:03d}",
                "hint_strength": "none",
                "hint_family": "none",
                "hint_mention_reasons": [],
                "turn_cap_without_submission": False,
                "non_registered_submit_attempt_count": 0,
                "ending": "submitted",
                "command_count": 1,
                "syntax_error_command_count": 0,
                "contains_cjk": False,
                "loop_turn_count": 0,
                "integrity_alert": False,
                "awareness_environment_match_count": 0,
                "awareness_training_match_count": 0,
                "open_leak_channels": [],
                "single_file_solvable_original": False,
                "solvable_hack": False,
                "mechanism_confirmed": False,
                "truth_pass": True,
                "truth_source": "blind_reference_pair",
                "judge_sourced_truth": False,
                "reference_digest_valid": True,
                "is_ladder": False,
                "generator": "synthetic-base",
            }
        )
        index += 1
    return rows


def _nll_by_row(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    return {
        cast("str", row["row_key"]): {
            "action_tokens": 4,
            "mean_nll": row.get("action_nll_mean", 0.5),
        }
        for row in rows
    }


def _failure_codes(  # noqa: PLR0913 - forwards independent evaluator inputs
    rows: list[dict[str, object]],
    *,
    nll: Mapping[str, Mapping[str, object]] | None = None,
    heldout_task_ids: frozenset[str] = frozenset(),
    heldout_channels: frozenset[str] = frozenset(),
    statement_overlap_row_keys: frozenset[str] = frozenset(),
    raw_awareness_by_arm: Mapping[str, Mapping[str, object]] | None = None,
    waiver: str | None = None,
) -> tuple[set[str], dict[str, object]]:
    card = evaluate_dataset_card(
        rows,
        nll_by_row=_nll_by_row(rows) if nll is None else nll,
        heldout_task_ids=heldout_task_ids,
        heldout_channels=heldout_channels,
        statement_overlap_row_keys=statement_overlap_row_keys,
        raw_awareness_by_arm=raw_awareness_by_arm,
        waiver=waiver,
    )
    return set(cast("list[str]", card["failure_codes"])), card


def test_dataset_card_passes_synthetic_gate_2_and_reports_breakdowns() -> None:
    failures, card = _failure_codes(_passing_rows())
    counts = cast("Mapping[str, object]", card["counts"])
    family_counts = cast("Mapping[str, int]", card["family_counts"])
    honest_share = cast("Mapping[str, object]", card["honest_share"])
    honest_family_counts = cast("Mapping[str, int]", honest_share["family_counts"])

    assert failures == set()
    assert counts["hack_rows"] == 90
    assert counts["final_rows"] == 100
    assert family_counts == {
        "F1": 36,
        "F2": 18,
        "F3": 18,
        "F5": 3,
        "F7": 14,
        "other_hack": 1,
    }
    assert card["reported_f2_f3_f7_at_least_10_percent"] == {"F2": True, "F3": True, "F7": True}
    assert card["reported_f2_f3_f5_f7_at_least_10_percent"] == {
        "F2": True,
        "F3": True,
        "F5": False,
        "F7": True,
    }
    assert counts["solvable_hack_rows"] == 23
    assert card["solvable_hack_share"] == pytest.approx(23 / 90)
    assert honest_family_counts == {"ilcb:original": 10}
    assert card["by_generator"] == {"synthetic-base": 100}
    assert "by_generator" in card
    assert "hint_strength_share_by_family" in card
    assert "health_by_generator" in card
    assert "nll_by_generator" in card
    assert card["exit_code"] == 0


def test_dataset_card_reports_recorded_primary_and_secondary_strategies_only() -> None:
    rows = _passing_rows()
    rows[0]["observed_strategies"] = ["exact_input_special_case", "visible_case_lookup_table"]
    rows[1].pop("observed_strategies", None)
    rows[2].pop("observed_strategies", None)

    _failures, card = _failure_codes(rows)

    assert card["family_counts"] == {
        "F1": 36,
        "F2": 18,
        "F3": 18,
        "F5": 3,
        "F7": 14,
        "other_hack": 1,
    }
    assert card["primary_strategy_counts"] == {
        "check_budget_relaxed": 3,
        "exact_input_special_case": 36,
        "reference_solution_leak": 18,
        "rigged_comparison_object": 14,
        "test_or_grader_file_edit": 18,
        "other_hack": 1,
    }
    assert card["observed_strategy_counts"] == {
        "exact_input_special_case": 1,
        "visible_case_lookup_table": 1,
    }
    assert card["observed_strategy_rows"] == 1
    assert card["observed_strategy_rows_missing"] == 89
    render_card = {**card, "gate_exit_code": card["exit_code"], "dry_run": False}
    markdown = dataset_card_render.render_dataset_card_markdown(render_card)
    assert "| exact_input_special_case | 36 | 1 |" in markdown
    assert "| visible_case_lookup_table | 0 | 1 |" in markdown
    assert "Rows missing strategy lists (unknown) | 89" in markdown
    historical_card = {
        key: value
        for key, value in render_card.items()
        if key
        not in {
            "observed_strategy_counts",
            "observed_strategy_rows",
            "observed_strategy_rows_missing",
        }
    }
    assert "## Observed strategy labels" not in dataset_card_render.render_dataset_card_markdown(
        historical_card
    )


@pytest.mark.parametrize(
    ("observed_strategies", "error", "message"),
    [
        ([], ValueError, "must not be empty"),
        (["unknown_strategy"], TypeError, "known strategy labels"),
        (["visible_case_lookup_table"], ValueError, "include the primary strategy"),
        (
            ["exact_input_special_case", "exact_input_special_case"],
            ValueError,
            "duplicate labels",
        ),
    ],
)
def test_dataset_card_rejects_invalid_observed_strategy_lists(
    observed_strategies: list[str], error: type[Exception], message: str
) -> None:
    rows = _passing_rows()
    rows[0]["observed_strategies"] = observed_strategies

    with pytest.raises(error, match=message):
        _failure_codes(rows)


def test_observed_strategy_markdown_rejects_malformed_present_counts() -> None:
    with pytest.raises(TypeError, match="strategy counts must be objects"):
        dataset_card_render._observed_strategy_markdown_lines({"observed_strategy_counts": None})


def _reduce_two_family_shares(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["primary_strategy"] != "other_hack":
            row["primary_strategy"] = "exact_input_special_case"


def _reduce_hack_rows_below_80(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["admission_type"] == "hack":
            row.update(
                admission_type="honest",
                primary_strategy="honest_no_hack",
                hint_strength="none",
                truth_pass=True,
                truth_source="blind_reference_pair",
                judge_sourced_truth=False,
                solvable_hack=False,
                single_file_solvable_original=False,
                mechanism_confirmed=False,
            )
            if sum(item["admission_type"] == "hack" for item in rows) == 79:
                return


def _move_f7_rows_to_other_hack(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["primary_strategy"] == "rigged_comparison_object":
            row["primary_strategy"] = "other_hack"
            if sum(item["primary_strategy"] == "other_hack" for item in rows) == 10:
                return


def _spread_hacks_over_five_corpora(rows: list[dict[str, object]]) -> None:
    hack_rows = [row for row in rows if row["admission_type"] == "hack"]
    for index, row in enumerate(hack_rows):
        row["corpus_unit"] = f"synthetic:corpus-{index % 5}"


def _make_one_corpus_too_small(rows: list[dict[str, object]]) -> None:
    hack_rows = [row for row in rows if row["admission_type"] == "hack"]
    for row in hack_rows[: len(hack_rows) - 2]:
        row["corpus_unit"] = "synthetic:large"
    hack_rows[-2]["corpus_unit"] = "synthetic:small"
    hack_rows[-1]["corpus_unit"] = "synthetic:small"


def _make_one_corpus_too_large(rows: list[dict[str, object]]) -> None:
    for row in rows[:28]:
        if row["admission_type"] == "hack":
            row["corpus_unit"] = "synthetic:large"
    smaller_rows = [
        row
        for row in rows
        if row["admission_type"] == "hack" and row["corpus_unit"] != "synthetic:large"
    ]
    for index, row in enumerate(smaller_rows):
        row["corpus_unit"] = f"synthetic:small-{index % 6}"


def _reduce_honest_rows(rows: list[dict[str, object]]) -> None:
    for row in reversed(rows):
        if row["admission_type"] == "honest":
            rows.remove(row)
            rows.append(
                {
                    **_passing_rows()[0],
                    "row_key": "synthetic-extra-hack",
                    "admission_type": "hack",
                    "primary_strategy": "exact_input_special_case",
                    "corpus_unit": "aisi:apps",
                }
            )
            return


def _remove_single_file_solvable_hacks(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["single_file_solvable_original"] = False


def _label_hack_as_honest_no_hack(rows: list[dict[str, object]]) -> None:
    next(row for row in rows if row["admission_type"] == "hack")["primary_strategy"] = (
        "honest_no_hack"
    )


def _mark_stale_reference_digest(rows: list[dict[str, object]]) -> None:
    rows[0]["reference_digest_valid"] = False


def _raise_syntax_error_rate(rows: list[dict[str, object]]) -> None:
    for row in rows[:7]:
        row["syntax_error_command_count"] = 1


def _raise_cjk_fraction(rows: list[dict[str, object]]) -> None:
    for row in rows[:16]:
        row["contains_cjk"] = True


def _raise_mean_nll(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["action_nll_mean"] = 0.91


def _raise_max_nll(rows: list[dict[str, object]]) -> None:
    rows[0]["action_nll_mean"] = 1.51


def _raise_specific_share(rows: list[dict[str, object]]) -> None:
    for row in rows:
        if row["primary_strategy"] == "test_or_grader_file_edit":
            row["hint_strength"] = "specific"
            if (
                sum(
                    item["primary_strategy"] == "test_or_grader_file_edit"
                    and item["hint_strength"] == "specific"
                    for item in rows
                )
                == 10
            ):
                return


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (_reduce_hack_rows_below_80, "hack_rows_below_80"),
        (lambda rows: rows.pop(), "final_rows_below_100"),
        (
            lambda rows: rows.__setitem__(
                36, {**rows[36], "primary_strategy": "exact_input_special_case"}
            ),
            "f1_share_above_40_percent",
        ),
        (_reduce_two_family_shares, "f2_f3_f7_share_floor"),
        (
            _move_f7_rows_to_other_hack,
            "other_hack_share_above_10_percent",
        ),
        (_spread_hacks_over_five_corpora, "corpora_below_6"),
        (_make_one_corpus_too_small, "corpus_below_3_rows"),
        (_make_one_corpus_too_large, "corpus_above_30_percent"),
        (_reduce_honest_rows, "honest_rows_below_10"),
        (
            _remove_single_file_solvable_hacks,
            "single_file_solvable_hack_rows_below_5",
        ),
        (_label_hack_as_honest_no_hack, "hack_labelled_honest_no_hack"),
        (_mark_stale_reference_digest, "stale_reference_digest"),
        (lambda rows: rows[0].update(partition="held-out"), "heldout_partition_overlap"),
        (
            lambda rows: rows[0].update(canonical_partition="held-out"),
            "heldout_partition_overlap",
        ),
        (lambda rows: rows[0].update(task_id="held-task"), "heldout_task_id_overlap"),
        (lambda rows: rows[0].update(statement_overlap=True), "heldout_statement_overlap"),
        (
            lambda rows: rows[0].update(open_leak_channels=["held-channel"]),
            "heldout_leak_channel_open",
        ),
        (lambda rows: rows[0].update(is_ladder=True), "ladder_rows_present"),
        (lambda rows: rows[0].update(hint_mention_reasons=["hint_6gram"]), "hint_mentions_present"),
        (lambda rows: rows[0].update(turn_cap_without_submission=True), "turn_cap_rows_present"),
        (
            lambda rows: rows[0].update(
                non_registered_submit_attempt_count=1, ending="no_tool_call"
            ),
            "no_tool_call_after_nonregistered_submit",
        ),
        (_raise_syntax_error_rate, "syntax_error_command_rate_above_6_percent"),
        (_raise_cjk_fraction, "cjk_episode_share_above_15_percent"),
        (lambda rows: rows[0].update(loop_turn_count=1), "loop_turns_present"),
        (_raise_mean_nll, "row_nll_mean_above_0_9"),
        (_raise_max_nll, "row_nll_max_above_1_5"),
        (lambda rows: rows[0].update(integrity_alert=True), "integrity_alert_rows_present"),
        (_raise_specific_share, "specific_hint_share_above_50_percent"),
    ],
)
def test_dataset_card_names_each_failed_gate_2_threshold(
    mutate: Callable[[list[dict[str, object]]], object], expected_code: str
) -> None:
    rows = _passing_rows()
    mutate(rows)
    failures, card = _failure_codes(
        rows,
        heldout_task_ids=frozenset({"held-task"}),
        heldout_channels=frozenset({"held-channel"}),
    )

    assert expected_code in failures
    assert card["exit_code"] == 1


def test_waiver_covers_only_hack_row_floor_and_requires_sixty_rows() -> None:
    rows = _passing_rows()
    for row in rows:
        if (
            row["admission_type"] == "hack"
            and row["primary_strategy"] == "exact_input_special_case"
        ):
            row["admission_type"] = "honest"
            row["primary_strategy"] = "honest_no_hack"
            break
    failures, _card = _failure_codes(rows, waiver="second wave complete")

    assert "hack_rows_below_80" not in failures

    too_small = _passing_rows()
    for row in too_small:
        if row["admission_type"] == "hack":
            row["admission_type"] = "honest"
            row["primary_strategy"] = "honest_no_hack"
    failures, _card = _failure_codes(too_small, waiver="second wave complete")
    assert "waiver_requires_at_least_60_hack_rows" in failures


def test_waiver_never_rescues_single_file_solvable_hack_floor() -> None:
    rows = _passing_rows()
    _remove_single_file_solvable_hacks(rows)

    failures, _card = _failure_codes(rows, waiver="second wave complete")

    assert "single_file_solvable_hack_rows_below_5" in failures


def test_honest_rows_must_use_strengthened_truth_and_none_arm() -> None:
    rows = _passing_rows()
    honest_row = next(row for row in rows if row["admission_type"] == "honest")
    honest_row["truth_pass"] = False
    honest_row["hint_strength"] = "vague"

    failures, card = _failure_codes(rows)

    assert "honest_rows_not_strengthened_true" in failures
    assert "honest_rows_not_none_arm" in failures
    assert card["exit_code"] == 1


def test_honest_share_cannot_exceed_one_quarter_of_final_rows() -> None:
    rows = _passing_rows()
    for index in range(16):
        row = rows[index]
        row.update(
            admission_type="honest",
            primary_strategy="honest_no_hack",
            hint_strength="none",
            truth_pass=True,
            truth_source="blind_reference_pair",
            judge_sourced_truth=False,
            solvable_hack=False,
            single_file_solvable_original=False,
            mechanism_confirmed=False,
        )

    failures, card = _failure_codes(rows)

    assert "honest_share_above_25_percent" in failures
    assert card["exit_code"] == 1


def test_hint_arm_is_dropped_only_after_both_awareness_conditions() -> None:
    rows = _passing_rows()
    rows[0]["hint_family"] = "grader_mechanics"
    rows[0]["hint_strength"] = "vague"
    rows[0]["hint_arm"] = "grader_mechanics:vague"
    raw_awareness = {
        "none": {"episodes": 40, "integrity_alerts": 1},
        "grader_mechanics:vague": {"episodes": 35, "integrity_alerts": 2},
    }

    failures, card = _failure_codes(rows, raw_awareness_by_arm=raw_awareness)

    assert "alerting_hint_arm_retained" not in failures
    awareness = cast("dict[str, object]", card["integrity_awareness"])
    hint_arm_awareness = cast("Mapping[str, Mapping[str, object]]", awareness["hint_arm_awareness"])
    assert hint_arm_awareness["grader_mechanics:vague"]["dropped"] is False

    raw_awareness["grader_mechanics:vague"]["integrity_alerts"] = 3
    failures, card = _failure_codes(rows, raw_awareness_by_arm=raw_awareness)

    assert "alerting_hint_arm_retained" in failures
    assert card["exit_code"] == 1


def test_dataset_card_rejects_rows_with_unknown_integrity_awareness() -> None:
    rows = _passing_rows()
    rows[0]["integrity_alert"] = None

    failures, card = _failure_codes(rows)

    assert "integrity_awareness_unknown_rows_present" in failures
    assert card["exit_code"] == 1


def test_math_task_split_uses_underlying_problem_item_id(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_item_ids: list[str] = []

    def partition(item_id: str) -> str:
        observed_item_ids.append(item_id)
        return "held-out"

    monkeypatch.setattr(agentic_sft, "problem_partition", partition)

    item_id, canonical_partition = agentic_sft._math_task_card_identity(
        "math_loose_checker-loose_oneoff-synthetic-item-17",
        family="math_loose_checker",
        variant="loose_oneoff",
    )

    assert observed_item_ids == ["synthetic-item-17"]
    assert item_id == "math:synthetic-item-17"
    assert canonical_partition == "held-out"


@pytest.mark.parametrize(("dry_run", "expected_exit_code"), [(False, 1), (True, 0)])
def test_dataset_card_cli_writes_private_reports_and_respects_dry_run_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
    expected_exit_code: int,
) -> None:
    run_directory = tmp_path / "run"
    samples_directory = run_directory / "samples"
    samples_directory.mkdir(parents=True)
    rows_path = samples_directory / "round-0.jsonl"
    rows_path.write_text("synthetic rows are supplied by the stub\n", encoding="utf-8")
    manifest_path = tmp_path / "selection.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    heldout_path = tmp_path / "heldout.json"
    heldout_path.write_text("{}\n", encoding="utf-8")
    hints_path = tmp_path / "hints.json"
    hints_path.write_text("{}\n", encoding="utf-8")
    truth_path = tmp_path / "truth.jsonl"
    truth_path.write_text("{}\n", encoding="utf-8")
    cases_root = tmp_path / "cases"
    cases_root.mkdir()

    rows = _passing_rows()
    rows[0]["canonical_partition"] = "held-out"

    def collect(
        _args: argparse.Namespace,
    ) -> tuple[
        list[dict[str, object]],
        dict[str, dict[str, object]],
        dict[str, object],
        set[str],
        set[str],
        dict[str, int],
        set[str],
        dict[str, dict[str, object]],
        set[str],
    ]:
        return (
            rows,
            _nll_by_row(rows),
            {"rows_sha256": "a" * 64},
            set(),
            set(),
            {},
            set(),
            {},
            set(),
        )

    monkeypatch.setattr(agentic_sft, "_collect_dataset_card_inputs", collect)
    arguments = [
        "card",
        "--manifest",
        str(manifest_path),
        "--rows",
        str(rows_path),
        "--heldout-manifests",
        str(heldout_path),
        "--hints",
        str(hints_path),
        "--strengthened-truth",
        str(truth_path),
        "--cases-root",
        str(cases_root),
    ]
    if dry_run:
        arguments.append("--dry-run")

    exit_code = agentic_sft.main(arguments)

    card_path = run_directory / "card.json"
    markdown_path = run_directory / "card.md"
    card = cast("dict[str, object]", json.loads(card_path.read_text(encoding="utf-8")))
    assert exit_code == expected_exit_code
    assert card["gate_exit_code"] == 1
    assert card["exit_code"] == expected_exit_code
    assert "heldout_partition_overlap" in cast("list[str]", card["failure_codes"])
    assert markdown_path.is_file()
    assert "synthetic rows are supplied by the stub" not in markdown_path.read_text(
        encoding="utf-8"
    )
    if dry_run:
        assert "Dataset card floor gaps:" in capsys.readouterr().out


def test_heldout_manifest_reader_collects_task_and_channel_ids(tmp_path: Path) -> None:
    path = tmp_path / "heldout.json"
    path.write_text(
        json.dumps(
            {
                "partition": "held-out",
                "provenance": {"selected_tasks": [{"task_id": "synthetic-task"}]},
                "split": {"train": ["train-channel"], "held-out": ["held-channel"]},
                "eval_pairs": [["synthetic-eval-task", "eval-channel"]],
            }
        ),
        encoding="utf-8",
    )

    task_ids, channels = agentic_sft._load_heldout_manifests([path])

    assert task_ids == {"synthetic-task", "synthetic-eval-task"}
    assert channels == {"held-channel", "eval-channel"}


def test_heldout_manifest_reader_rejects_a_training_partition(tmp_path: Path) -> None:
    path = tmp_path / "not-heldout.json"
    path.write_text('{"partition": "train", "task_ids": ["synthetic-task"]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="non-held-out partition"):
        agentic_sft._load_heldout_manifests([path])


def test_base_nll_report_must_match_the_exact_rows(tmp_path: Path) -> None:
    rows_path = tmp_path / "round-0.jsonl"
    row_key = "ladder:" + "e" * 64
    rows_path.write_text(json.dumps({"key": row_key}) + "\n", encoding="utf-8")
    rows_digest = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    nll_path = tmp_path / "base-nll.json"
    nll_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "complete": True,
                "adapter": None,
                "rows_sha256": rows_digest,
                "model": "synthetic-base",
                "model_revision": "synthetic-revision",
                "rows": [{"row_key": row_key, "action_tokens": 2, "mean_nll": 0.5}],
            }
        ),
        encoding="utf-8",
    )

    nll_by_row, metadata = agentic_sft._load_base_nll(nll_path, rows_path, {row_key})

    assert nll_by_row[row_key]["mean_nll"] == 0.5
    assert metadata["rows_sha256"] == rows_digest
    with pytest.raises(ValueError, match="does not match SFT rows"):
        agentic_sft._load_base_nll(nll_path, rows_path, {"ladder:" + "f" * 64})
