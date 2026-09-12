from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from games import cooperation_evals
from games.cooperation_evals import (
    EXACT_DECISION_PROMPT_BEGIN,
    EXACT_DECISION_PROMPT_END,
    FAMILY_NUMERIC_GIVING,
    FAMILY_SOCIAL_DILEMMA,
    NUMERIC_FORECAST_EVENTS,
    BehaviorManifest,
    ContextElicitation,
    DiagnosticPair,
    allocation_option_orders,
    build_allocation_plan,
    build_behavior_plan,
    expand_behavior_roster,
    forecast_and_normative_plans,
    load_behavior_manifest,
    local_dt_counts,
    rank_row_actions,
    summarize_forecast_behavior_events,
    validate_experiment_coverage,
    validate_familiar_component_holdout,
    validate_group_isolation,
)
from games.evals import EVAL_ONLY_GRADING, EVAL_RENDER_GRADING_BY_GAME
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    SPLIT_EVAL,
    generate_framing_prompt_rows,
    generate_prompt_rows,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def manifest_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "elicitation": {
            "forecast_template": "forecast {target}",
            "normative_choice_template": "choice objective",
            "normative_numeric_template": "numeric objective",
            "matrix_target_template": "label {coop_label}",
            "numeric_targets": {"dictator": "numeric target"},
            "normative_target": "synthetic target",
            "normative_dimension": "welfare",
            "construct": "synthetic construct",
            "expected_direction": "synthetic direction",
        },
        "allocation": {
            "diagnostic_id": "fixture-allocation",
            "stem": "synthetic allocation fixture",
            "option_payoffs": [[9, 0], [6, 8], [5, 5]],
            "samples": 2,
        },
        "pairs": [
            {
                "pair_id": "pd-diagnostic",
                "family": FAMILY_SOCIAL_DILEMMA,
                "game_id": "twin-pd",
                "grading": EVAL_RENDER_GRADING_BY_GAME["twin-pd"],
                "scenario_id": "lagoon-net-length",
                "payoff_variant": "temptation-2",
                "scenario_group": "fit-pd-lagoon",
                "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED],
                "samples": 2,
            },
            {
                "pair_id": "dictator-diagnostic",
                "family": FAMILY_NUMERIC_GIVING,
                "game_id": "dictator",
                "grading": EVAL_RENDER_GRADING_BY_GAME["dictator"],
                "scenario_id": "upper-wash-take",
                "payoff_variant": "endowment-100",
                "scenario_group": "eval-dictator-upper-wash",
                "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL],
                "samples": 2,
            },
        ],
    }


def write_manifest(path: Path) -> Path:
    path.write_text(json.dumps(manifest_payload()), encoding="utf-8")
    return path


def test_manifest_expands_whole_label_pair_and_numeric_counts(tmp_path: Path) -> None:
    manifest = load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    roster = expand_behavior_roster(manifest)

    assert roster.counts.n_pairs == 2
    assert roster.counts.n_rendered_prompts == 5
    assert roster.counts.n_completions == 10
    assert roster.counts.by_family == {
        FAMILY_NUMERIC_GIVING: 1,
        FAMILY_SOCIAL_DILEMMA: 4,
    }
    assert {row.raw["coop_label"] for row in roster.rows if row.pair_id == "pd-diagnostic"} == {
        "SHORT",
        "LONG",
    }
    assert {
        row.raw["label_print_order"] for row in roster.rows if row.pair_id == "pd-diagnostic"
    } == {
        LABEL_PRINT_ORDER_CANONICAL,
        LABEL_PRINT_ORDER_SWAPPED,
    }


def test_manifest_rejects_duplicate_or_incomplete_pair(tmp_path: Path) -> None:
    payload = manifest_payload()
    duplicate = dict(payload["pairs"][0])  # type: ignore[index]
    duplicate["pair_id"] = "pd-diagnostic"
    payload["pairs"] = [payload["pairs"][0], duplicate]  # type: ignore[index]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pair_id"):
        load_behavior_manifest(path)

    payload = manifest_payload()
    payload["pairs"][0]["label_print_orders"] = [LABEL_PRINT_ORDER_CANONICAL]  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="both printed orders"):
        load_behavior_manifest(path)


def test_row_ranker_reports_actual_cells_and_defective_harmony_optimum() -> None:
    row = generate_prompt_rows(
        "defective-harmony",
        "group-mix",
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_SWAPPED,
    )[0]
    ranked = rank_row_actions(row)

    assert ranked["reward_optimal_action"] == "D"
    assert ranked["reward_optimal_label"] == row["label_b"]
    assert ranked["payoffs_by_action"]["C"]["against_cooperate"] == pytest.approx(row["payoff_cc"])
    assert ranked["payoffs_by_action"]["D"]["against_defect"] == pytest.approx(row["payoff_dd"])
    assert ranked["own_payoff"] is None
    assert ranked["unresolved"] is True


def test_forecast_prompt_has_exact_decision_context_and_separate_normative_item(
    tmp_path: Path,
) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    forecast, normative = forecast_and_normative_plans(roster, samples=1)

    assert len(forecast) == len(normative) == 5
    assert all(request.identity[0] == "self-report" for request in forecast + normative)
    for row, forecast_request, normative_request in zip(
        roster.rows, forecast, normative, strict=True
    ):
        assert forecast_request.prompt != normative_request.prompt
        forecast_context = forecast_request.prompt.split(f"{EXACT_DECISION_PROMPT_BEGIN}\n", 1)[
            1
        ].split(f"\n{EXACT_DECISION_PROMPT_END}", 1)[0]
        normative_context = normative_request.prompt.split(f"{EXACT_DECISION_PROMPT_BEGIN}\n", 1)[
            1
        ].split(f"\n{EXACT_DECISION_PROMPT_END}", 1)[0]
        assert forecast_context == normative_context
        assert forecast_context == row.raw["prompt"]
        assert EXACT_DECISION_PROMPT_BEGIN in forecast_request.prompt
        assert EXACT_DECISION_PROMPT_END in forecast_request.prompt
        assert "self-prediction-context" in forecast_request.identity[1]
        assert "normative-payoff-choice" in normative_request.identity[1]
        if not (row.raw["label_a"] or row.raw["label_b"]):
            assert re.search(r"0 to \d+", normative_request.prompt)
    assert all(
        "<keep>" in request.prompt for request in normative if "dictator" in request.identity[1]
    )


@pytest.mark.parametrize("request_kind", ["forecast", "normative"])
@pytest.mark.parametrize("corruption", ["mismatch", "absent", "duplicate"])
def test_context_request_record_rejects_corrupt_embedded_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_kind: str,
    corruption: str,
) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    original_stem = cooperation_evals._embedded_context_stem

    def corrupt_stem(row: Mapping[str, object], question: str) -> str:
        rendered = original_stem(row, question)
        if corruption == "mismatch":
            return rendered.replace(str(row["prompt"]), f"{row['prompt']} changed", 1)
        if corruption == "absent":
            return rendered.replace(f"{EXACT_DECISION_PROMPT_BEGIN}\n", "", 1).replace(
                f"\n{EXACT_DECISION_PROMPT_END}", "", 1
            )
        return rendered.replace(
            EXACT_DECISION_PROMPT_BEGIN,
            f"{EXACT_DECISION_PROMPT_BEGIN}\n{EXACT_DECISION_PROMPT_BEGIN}",
            1,
        )

    monkeypatch.setattr(cooperation_evals, "_embedded_context_stem", corrupt_stem)
    forecasts, normative = forecast_and_normative_plans(roster, samples=1)
    request = (forecasts if request_kind == "forecast" else normative)[0]

    with pytest.raises(ValueError, match="embedded decision prompt"):
        request.record("unresolved")


def test_forecasts_preserve_direct_cooperation_frequency_for_both_mappings_and_orders(
    tmp_path: Path,
) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    forecasts, _ = forecast_and_normative_plans(roster, samples=1, prefilled_think=False)
    parsed = [
        request.record("<keep>80</keep>")
        for request in forecasts
        if "pd-diagnostic" in request.identity[1]
    ]
    assert len(parsed) == 4
    assert {record["predicted_cooperation_fraction"] for record in parsed} == {0.8}


def test_plan_and_group_isolation_are_explicit(tmp_path: Path) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    plan = build_behavior_plan(roster)
    assert len(plan) == roster.counts.n_completions
    assert len({request.identity for request in plan}) == len(plan)
    assert "allocation::fixture-allocation" in roster.scenario_group_ids
    validate_group_isolation(roster, {"train-pd"})
    with pytest.raises(ValueError, match="overlap"):
        validate_group_isolation(roster, {"fit-pd-lagoon"})


def test_experiment_coverage_guard_is_stricter_than_fixture_expansion(tmp_path: Path) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    with pytest.raises(ValueError, match="roughly 12 diagnostic pairs"):
        validate_experiment_coverage(roster)


def test_familiar_component_holdout_uses_training_triplets(tmp_path: Path) -> None:
    row = generate_framing_prompt_rows(
        "twin-pd",
        EVAL_RENDER_GRADING_BY_GAME["twin-pd"],
        framing_id="human",
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_CANONICAL,
    )[0]
    pair = DiagnosticPair.from_mapping(
        {
            "pair_id": "human-holdout",
            "family": FAMILY_SOCIAL_DILEMMA,
            "game_id": "twin-pd",
            "grading": EVAL_RENDER_GRADING_BY_GAME["twin-pd"],
            "scenario_id": row["reskin_id"],
            "payoff_variant": row["payoff_variant"],
            "scenario_group": "human-holdout-group",
            "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED],
            "contrast_kind": "held-out",
            "counterpart_framing": "human",
            "held_out_counterpart_identity": "human",
            "held_out_combination": "familiar-components",
        }
    )
    roster = expand_behavior_roster(BehaviorManifest(pairs=(pair,)))
    training_path = tmp_path / "training.jsonl"
    training_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "game_id": game_id,
                    "payoff_variant": payoff_variant,
                    "framing_id": framing_id,
                }
            )
            for game_id, payoff_variant, framing_id in (
                ("twin-pd", "temptation-2", "twin"),
                ("twin-pd", "temptation-10", "human"),
                ("twin-pd", "temptation-2", "unstated"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    validate_familiar_component_holdout(roster, training_path)
    training_path.write_text(
        training_path.read_text(encoding="utf-8")
        + json.dumps(
            {
                "game_id": "twin-pd",
                "payoff_variant": "temptation-2",
                "framing_id": "human",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no held-out familiar-component"):
        validate_familiar_component_holdout(roster, training_path)


def test_behavior_records_retain_renderer_framing_and_holdout_metadata() -> None:
    row = generate_framing_prompt_rows(
        "twin-pd",
        EVAL_RENDER_GRADING_BY_GAME["twin-pd"],
        framing_id="different-ai",
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_CANONICAL,
    )[0]
    pair = DiagnosticPair.from_mapping(
        {
            "pair_id": "framed-behavior",
            "family": FAMILY_SOCIAL_DILEMMA,
            "game_id": "twin-pd",
            "grading": EVAL_RENDER_GRADING_BY_GAME["twin-pd"],
            "scenario_id": row["reskin_id"],
            "payoff_variant": row["payoff_variant"],
            "scenario_group": "framed-behavior-group",
            "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED],
            "contrast_kind": "held-out",
            "counterpart_framing": "different-ai",
            "held_out_counterpart_identity": "different-ai",
            "held_out_combination": "framed-control",
        }
    )
    roster = expand_behavior_roster(BehaviorManifest(pairs=(pair,)))
    request = build_behavior_plan(roster, prefilled_think=False)[0]
    record = request.record(f"<action>{row['label_a']}</action>")

    assert record["counterpart_framing"] == "different-ai"
    assert record["contrast_kind"] == "held-out"
    assert record["held_out_combination"] == "framed-control"


def test_behavior_records_keep_parser_delimited_pre_action_prefix(tmp_path: Path) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    request = build_behavior_plan(roster, prefilled_think=False)[0]
    cooperative_label = str(roster.rows[0].raw["coop_label"])

    parsed = request.record(
        f"<think>reasoning</think> more reasoning <action>{cooperative_label}</action>"
    )
    assert parsed["request_id"] == "::".join(str(value) for value in request.identity)
    assert parsed["prompt"] == request.prompt
    assert parsed["pre_action_prefix"] == "reasoning</think> more reasoning"
    unresolved = request.record("<think>reasoning</think> more reasoning")
    assert unresolved["pre_action_prefix"] is None
    ambiguous = request.record(
        f"<think>reasoning</think> <action>{cooperative_label}</action>"
        f" <action>{cooperative_label}</action>"
    )
    assert ambiguous["pre_action_prefix"] is None


def test_numeric_forecast_has_explicit_target_description(tmp_path: Path) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    forecasts, _ = forecast_and_normative_plans(roster, samples=1, prefilled_think=False)
    numeric_forecast = next(request for request in forecasts if "dictator" in request.identity[1])
    record = numeric_forecast.record("<keep>40</keep>")
    assert record["forecast_target_action"] == "numeric target"
    assert record["predicted_cooperation_fraction"] == pytest.approx(0.4)


def test_numeric_forecast_event_is_observed_and_joined_to_independent_behavior(
    tmp_path: Path,
) -> None:
    roster = expand_behavior_roster(
        load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    )
    forecasts, _ = forecast_and_normative_plans(roster, samples=1, prefilled_think=False)
    behavior = build_behavior_plan(roster, prefilled_think=False)
    forecast_request = next(request for request in forecasts if "dictator" in request.identity[1])
    dictator_requests = [request for request in behavior if request.identity[1] == "dictator"]
    assert len(dictator_requests) == 2
    dictator_request = dictator_requests[0]

    forecast_record = forecast_request.record("<keep>40</keep>")
    observed_records = [
        dictator_request.record("<keep>0</keep>"),
        dictator_request.record("<keep>100</keep>"),
        dictator_request.record("unresolved"),
    ]

    assert forecast_record["forecast_event_id"] == "numeric.dictator.give-positive"
    assert forecast_record["forecast_target_predicate"] == "kept < row.endowment"
    assert forecast_record["predicted_event_fraction"] == pytest.approx(0.4)
    assert [record["forecast_event_observed"] for record in observed_records] == [True, False, None]
    summary = summarize_forecast_behavior_events([forecast_record], observed_records)
    assert summary["by_event_id"]["numeric.dictator.give-positive"] == {
        "n_forecast_records": 1,
        "n_forecast_parsed": 1,
        "n_behavior_join_keys": 1,
        "n_behavior_records": 3,
        "n_behavior_resolved": 2,
        "n_behavior_unresolved": 1,
        "n_behavior_event_true": 1,
        "observed_event_rate": pytest.approx(0.5),
    }


def test_numeric_forecast_event_boundaries_are_explicit() -> None:
    cases = {
        "dictator": lambda row: (
            ({"parsed": True, "kept": 0}, True),
            ({"parsed": True, "kept": int(row["endowment"])}, False),
        ),
        "nash-demand": lambda row: (
            ({"parsed": True, "claim": int(row["windfall"]) // 2}, True),
            ({"parsed": True, "claim": int(row["windfall"]) // 2 + 1}, False),
        ),
        "trust-vs-stated-return": lambda row: (
            ({"parsed": True, "sent": 1}, True),
            ({"parsed": True, "sent": 0}, False),
        ),
        "trustee-return-rule": lambda row: (
            ({"parsed": True, "return_fraction": 0.01}, True),
            ({"parsed": True, "return_fraction": 0.0}, False),
        ),
    }
    for game_id, case_builder in cases.items():
        row = generate_prompt_rows(
            game_id,
            EVAL_RENDER_GRADING_BY_GAME.get(game_id, EVAL_ONLY_GRADING),
            split=SPLIT_EVAL,
            label_print_order=LABEL_PRINT_ORDER_CANONICAL,
        )[0]
        event = NUMERIC_FORECAST_EVENTS[game_id]
        for record, expected in case_builder(row):
            assert event.observe(row, record) is expected
    assert NUMERIC_FORECAST_EVENTS["nash-demand"].predicate == "2 * claim <= row.windfall"


def test_unresolved_normative_target_is_none(tmp_path: Path) -> None:
    row = generate_prompt_rows(
        "nash-demand",
        EVAL_RENDER_GRADING_BY_GAME["nash-demand"],
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_CANONICAL,
    )[0]
    pair = DiagnosticPair.from_mapping(
        {
            "pair_id": "nash-unresolved",
            "family": FAMILY_NUMERIC_GIVING,
            "game_id": "nash-demand",
            "grading": EVAL_RENDER_GRADING_BY_GAME["nash-demand"],
            "scenario_id": row["reskin_id"],
            "payoff_variant": row["payoff_variant"],
            "scenario_group": "nash-unresolved-group",
            "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL],
        }
    )
    elicitation = ContextElicitation.from_mapping(
        {
            "forecast_template": "synthetic forecast {target}",
            "normative_choice_template": "synthetic choice {target}",
            "normative_numeric_template": "synthetic numeric {target}",
            "matrix_target_template": "synthetic matrix {coop_label}",
            "numeric_targets": {"nash-demand": "synthetic event"},
            "normative_target": "synthetic objective",
            "normative_dimension": "welfare",
            "construct": "synthetic construct",
            "expected_direction": "synthetic direction",
        }
    )
    roster = expand_behavior_roster(BehaviorManifest(pairs=(pair,), elicitation=elicitation))
    _, normative = forecast_and_normative_plans(roster, samples=1, prefilled_think=False)
    record = normative[0].record("<keep>50</keep>")

    assert record["normative_target_optimal_actions"] == []
    assert record["normative_matches_target_optimum"] is None
    assert record["normative_matches_reward_optimum"] is None


def test_context_elicitation_requires_runtime_prose() -> None:
    manifest = BehaviorManifest(
        pairs=(
            DiagnosticPair.from_mapping(
                {
                    "pair_id": "dictator",
                    "family": FAMILY_NUMERIC_GIVING,
                    "game_id": "dictator",
                    "grading": EVAL_RENDER_GRADING_BY_GAME["dictator"],
                    "scenario_id": "upper-wash-take",
                    "payoff_variant": "endowment-100",
                    "scenario_group": "group",
                    "label_print_orders": [LABEL_PRINT_ORDER_CANONICAL],
                }
            ),
        )
    )
    with pytest.raises(ValueError, match="runtime elicitation"):
        forecast_and_normative_plans(expand_behavior_roster(manifest), samples=1)


def test_local_dt_counts_are_exact() -> None:
    counts = local_dt_counts(multiple_choice_samples=2, open_ended_samples=2)
    assert counts == {
        "n_items": 30,
        "n_multiple_choice": 27,
        "n_open_ended": 3,
        "n_choice_responses": 108,
        "n_open_responses": 6,
        "n_responses": 114,
    }


def test_allocation_diagnostic_expands_both_orders_and_records_all_optima(
    tmp_path: Path,
) -> None:
    manifest = load_behavior_manifest(write_manifest(tmp_path / "manifest.json"))
    roster = expand_behavior_roster(manifest)
    requests = build_allocation_plan(roster, prefilled_think=False)

    assert len(requests) == 6
    assert {request.identity[2] for request in requests} == {
        "as-authored",
        "rotate-left",
        "rotate-right",
    }
    assert {
        option: [order.index(option) for _, order in allocation_option_orders(3)]
        for option in range(3)
    } == {0: [0, 2, 1], 1: [1, 0, 2], 2: [2, 1, 0]}
    parsed = requests[0].record("FINAL ANSWER: A")
    assert parsed["scenario_group"] == "allocation::fixture-allocation"
    assert parsed["allocation_own_optimal_options"] == ["0"]
    assert parsed["allocation_total_welfare_optimal_options"] == ["1"]
    assert parsed["allocation_equality_optimal_options"] == ["2"]
    assert parsed["allocation_chosen_option"] == "0"
