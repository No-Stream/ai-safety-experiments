from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping

from games.cooperation_payoffs import (
    annotate_behavior_record,
    rank_allocation_options,
    rank_row_actions,
)
from games.payoffs import (
    TRUST_MAX_STATED_RETURN_FRACTION,
    TrustSpec,
    trustor_care_reward,
)
from games.prompts import LABEL_PRINT_ORDER_CANONICAL, SPLIT_EVAL, generate_prompt_rows
from games.rewards import REQUIRED_REWARD_COLUMNS, make_game_reward


def _row(game_id: str, grading: str, *, variant: str | None = None) -> Mapping[str, Any]:
    rows = generate_prompt_rows(
        game_id,
        grading,
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_CANONICAL,
    )
    return next(row for row in rows if variant is None or row["payoff_variant"] == variant)


def test_care_reward_ranks_return_fifth_send_all_and_matches_production_math() -> None:
    row = _row("trust-vs-stated-return", "care-alpha-1", variant="return-fifth")
    ranked = rank_row_actions(row, 10)
    spec = TrustSpec(
        game_id=str(row["game_id"]),
        endowment=int(row["endowment"]),
        multiplier=float(row["transfer_multiplier"]),
        stated_return_fraction=float(row["stated_return_fraction"]),
    )

    expected = trustor_care_reward(
        spec,
        sent=10,
        return_fraction=0.2,
        max_return_fraction=TRUST_MAX_STATED_RETURN_FRACTION,
        alpha=1.0,
    )
    assert ranked["reward_optimal_actions"] == [10]
    assert ranked["reward_optimal_action"] == 10
    assert ranked["reward_values_by_action"]["10"] == pytest.approx(expected)
    assert ranked["observed_reward"] == pytest.approx(expected)
    assert ranked["own_optimal_actions"] == [0]
    assert ranked["welfare_optimal_actions"] == [10]


def test_care_reward_values_crosscheck_the_actual_reward_dispatch() -> None:
    row = _row("trust-vs-stated-return", "care-alpha-1", variant="return-fifth")
    columns = {name: [row[name], row[name]] for name in REQUIRED_REWARD_COLUMNS}
    scorer = make_game_reward(2, prefilled_think=False)
    actual = scorer(
        completions=["<send>0</send>", "<send>10</send>"],
        log_metric=lambda _name, _value: None,
        log_extra=lambda _name, _values: None,
        **columns,
    )

    ranked = rank_row_actions(row)
    assert actual == pytest.approx(
        [ranked["reward_values_by_action"]["0"], ranked["reward_values_by_action"]["10"]]
    )


def test_dictator_keeps_reward_own_welfare_and_equality_optima_distinct() -> None:
    row = _row("dictator", "keep-fraction")
    endowment = int(row["endowment"])
    ranked = rank_row_actions(row, endowment // 2)

    assert ranked["reward_optimal_actions"] == [endowment]
    assert ranked["own_optimal_actions"] == [endowment]
    assert ranked["welfare_optimal_actions"] == list(range(endowment + 1))
    assert ranked["equality_optimal_actions"] == [endowment // 2]
    assert ranked["observed_reward"] == pytest.approx(0.5)
    assert ranked["own_payoff"] == pytest.approx(endowment / 2)
    assert ranked["counterpart_payoff"] == pytest.approx(endowment / 2)


def test_runtime_allocation_can_separate_own_welfare_and_equality_optima() -> None:
    ranked = rank_allocation_options(
        {
            "own-option": (10.0, 0.0),
            "equality-option": (6.0, 6.0),
            "welfare-option": (5.0, 10.0),
        },
        "welfare-option",
        reward_values_by_option={
            "own-option": 0.5,
            "equality-option": 0.6,
            "welfare-option": 0.75,
        },
    )

    assert ranked["own_optimal_actions"] == ["own-option"]
    assert ranked["welfare_optimal_actions"] == ["welfare-option"]
    assert ranked["equality_optimal_actions"] == ["equality-option"]
    assert ranked["reward_optimal_actions"] == ["welfare-option"]
    assert ranked["observed_reward"] == pytest.approx(0.75)
    assert ranked["own_payoff"] == pytest.approx(5.0)
    assert ranked["counterpart_payoff"] == pytest.approx(10.0)


def test_runtime_allocation_preserves_ties_and_requires_matching_reward_keys() -> None:
    ranked = rank_allocation_options({"left": (4.0, 4.0), "right": (4.0, 4.0)})
    assert ranked["own_optimal_actions"] == ["left", "right"]
    assert ranked["welfare_optimal_actions"] == ["left", "right"]
    assert ranked["equality_optimal_actions"] == ["left", "right"]
    assert ranked["reward_optimal_actions"] == []

    with pytest.raises(ValueError, match="exactly match"):
        rank_allocation_options(
            {"left": (4.0, 4.0), "right": (3.0, 5.0)},
            reward_values_by_option={"left": 1.0},
        )


def test_matrix_unknown_mix_reports_conditional_values_without_a_prior() -> None:
    row = _row("defective-harmony", "group-mix")
    ranked = rank_row_actions(row, "D")

    assert ranked["reward_values_by_action"] is None
    assert ranked["observed_reward"] is None
    assert ranked["own_payoff"] is None
    assert ranked["outcome_resolution"] == "counterpart-unobserved"
    assert ranked["reward_values_by_action_and_counterpart"]["D"] == {
        "against_cooperate": pytest.approx(float(row["payoff_dc"])),
        "against_defect": pytest.approx(float(row["payoff_dd"])),
    }
    assert ranked["reward_optimal_actions"] == ["D"]


def _unresolved_matrix_with_payoffs(
    *, payoff_cc: float, payoff_cd: float, payoff_dc: float, payoff_dd: float
) -> dict[str, Any]:
    row = dict(_row("twin-pd", "group-mix"))
    row.update(
        payoff_cc=payoff_cc,
        payoff_cd=payoff_cd,
        payoff_dc=payoff_dc,
        payoff_dd=payoff_dd,
        opp_coop_prob=-1.0,
    )
    return row


def test_unresolved_mix_reports_common_unique_welfare_optimum_without_a_value() -> None:
    row = _unresolved_matrix_with_payoffs(
        payoff_cc=0.6, payoff_cd=0.0, payoff_dc=1.0, payoff_dd=0.2
    )
    ranked = rank_row_actions(row)

    assert ranked["welfare_optimal_actions"] == ["C"]
    assert ranked["welfare_optimal_actions_by_counterpart"] == {
        "against_cooperate": ["C"],
        "against_defect": ["C"],
    }
    assert ranked["total_welfare_by_action"] == {
        "C": {"against_cooperate": pytest.approx(1.2), "against_defect": pytest.approx(1.0)},
        "D": {"against_cooperate": pytest.approx(1.0), "against_defect": pytest.approx(0.4)},
    }
    assert ranked["total_welfare"] is None
    assert ranked["reward_values_by_action"] is None


def test_unresolved_mix_does_not_collapse_a_conditional_welfare_tie() -> None:
    row = _unresolved_matrix_with_payoffs(
        payoff_cc=0.5, payoff_cd=0.4, payoff_dc=0.6, payoff_dd=0.2
    )
    ranked = rank_row_actions(row)

    assert ranked["welfare_optimal_actions"] == []
    assert ranked["welfare_optimal_actions_by_counterpart"] == {
        "against_cooperate": ["C", "D"],
        "against_defect": ["C"],
    }


def test_unresolved_mix_does_not_collapse_conflicting_welfare_winners() -> None:
    row = _unresolved_matrix_with_payoffs(
        payoff_cc=0.8, payoff_cd=0.2, payoff_dc=0.3, payoff_dd=0.6
    )
    ranked = rank_row_actions(row)

    assert ranked["welfare_optimal_actions"] == []
    assert ranked["welfare_optimal_actions_by_counterpart"] == {
        "against_cooperate": ["C"],
        "against_defect": ["D"],
    }


def test_exact_reward_tie_is_distinct_from_unresolved_reward() -> None:
    row = dict(_row("twin-pd", "group-mix"))
    row.update(payoff_cc=0.5, payoff_cd=0.5, payoff_dc=0.5, payoff_dd=0.5, opp_coop_prob=0.25)
    ranked = rank_row_actions(row)

    assert ranked["reward_optimal_actions"] == ["C", "D"]
    assert ranked["reward_optimal_action"] is None
    assert ranked["reward_optimal_reason"] == "exact reward tie"
    assert ranked["reward_resolution"] == "resolved"


def test_stated_match_has_resolved_expected_payoffs_for_the_chosen_action() -> None:
    row = dict(_row("twin-pd", "group-mix"))
    row.update(grading="vs-stated-match", stated_match_prob=0.8)
    ranked = rank_row_actions(row, "C")

    assert ranked["outcome_resolution"] == "stated-match-expectation"
    assert ranked["own_payoff"] == pytest.approx(ranked["observed_reward"])
    assert ranked["counterpart_payoff"] is not None
    assert ranked["total_welfare"] == pytest.approx(
        ranked["own_payoff"] + ranked["counterpart_payoff"]
    )


def test_self_grading_does_not_claim_the_counterpart_actually_matched() -> None:
    row = _row("twin-pd", "self")
    ranked = rank_row_actions(row, "C")

    assert ranked["observed_reward"] == pytest.approx(float(row["payoff_cc"]))
    assert ranked["own_payoff"] is None
    assert ranked["counterpart_payoff"] is None
    assert ranked["outcome_resolution"] == "counterpart-unobserved"
    assert ranked["grading_implied_payoffs_by_action"]["C"] == pytest.approx(
        (float(row["payoff_cc"]), float(row["payoff_cc"]))
    )


@pytest.mark.parametrize(
    ("field", "value"), [("opp_coop_prob", 1.1), ("opp_coop_prob", float("nan"))]
)
def test_fixed_mix_rejects_invalid_probabilities(field: str, value: float) -> None:
    row = dict(_row("pd-vs-frozen", "vs-fixed-mix"))
    row[field] = value

    with pytest.raises(ValueError, match="finite probability"):
        rank_row_actions(row)


def test_nash_self_outcome_crashes_above_half_instead_of_recording_the_claim() -> None:
    row = _row("nash-demand", "nash-demand-self")
    windfall = int(row["windfall"])
    ranked = rank_row_actions(row, windfall)

    assert ranked["own_payoff"] == 0.0
    assert ranked["counterpart_payoff"] == 0.0
    assert ranked["observed_reward"] == 0.0
    assert ranked["reward_optimal_actions"] == [windfall // 2]


def test_nash_group_mix_preserves_conditional_payoffs_without_inventing_a_counterpart() -> None:
    row = dict(_row("nash-demand", "nash-demand-group-mix"))
    row["windfall"] = 4
    ranked = rank_row_actions(row, 3)

    assert ranked["payoffs_by_action"]["3"] == {
        "0": 3.0,
        "1": 3.0,
        "2": 0.0,
        "3": 0.0,
        "4": 0.0,
    }
    assert ranked["counterpart_payoffs_by_action"]["3"] == {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0,
        "3": 0.0,
        "4": 0.0,
    }
    assert ranked["total_welfare_by_action"]["3"] == {
        "0": 3.0,
        "1": 4.0,
        "2": 0.0,
        "3": 0.0,
        "4": 0.0,
    }
    assert ranked["welfare_optimal_actions_by_counterpart"] == {
        "0": ["4"],
        "1": ["3"],
        "2": ["2"],
        "3": ["1"],
        "4": ["0"],
    }
    assert ranked["welfare_optimal_actions"] == []
    assert ranked["own_payoff"] is None
    assert ranked["counterpart_payoff"] is None
    assert ranked["total_welfare"] is None
    assert ranked["observed_reward"] is None


def test_trustee_has_outcomes_but_no_invented_training_reward() -> None:
    row = _row("trustee-return-rule", "group-mix")
    ranked = rank_row_actions(row, 20)

    assert ranked["reward_values_by_action"] is None
    assert ranked["reward_optimal_actions"] == []
    assert ranked["reward_resolution"] == "not-defined-for-eval-only-row"
    assert ranked["own_payoff"] == pytest.approx(2.4)
    assert ranked["counterpart_payoff"] == pytest.approx(0.6)
    assert ranked["total_welfare"] == pytest.approx(3.0)


def test_annotation_keeps_parse_failure_separate_from_payoff_resolution() -> None:
    row = _row("defective-harmony", "group-mix")
    record = annotate_behavior_record(row, {"parsed": False, "action": None})

    assert record["parsed"] is False
    assert record["chosen_action"] is None
    assert record["observed_reward"] is None
    assert record["outcome_resolution"] == "counterpart-unobserved"
    assert record["row_payoff_digest"]
