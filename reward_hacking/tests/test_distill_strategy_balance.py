"""Synthetic selection caps; no benchmark data."""

from collections.abc import Mapping

import pytest

from reward_hacking.distill.strategy_balance import balance_entries


def _entry(
    index: int, strategy: str, source: str = "alpha", family: str = "toy"
) -> dict[str, object]:
    return {
        "training_key": str(index),
        "primary_strategy": strategy,
        "source": source,
        "family": family,
    }


def test_output_cap_uses_final_denominator_and_identity_order() -> None:
    entries = [_entry(index, "exact_input_special_case") for index in range(10)]
    entries += [
        _entry(10, "visible_case_lookup_table"),
        *[_entry(index, "test_or_grader_file_edit") for index in range(11, 14)],
    ]
    selected, dropped = balance_entries(entries)
    assert len(selected) == 5
    assert (
        sum(
            entry["primary_strategy"] in {"exact_input_special_case", "visible_case_lookup_table"}
            for entry in selected
        )
        == 2
    )
    assert balance_entries(list(reversed(entries))) == (selected, dropped)
    assert all(entry["cap_reason"] == "output_fitting" for entry in dropped)


@pytest.mark.parametrize(
    "strategy", ["fitted_rule", "input_class_special_case", "spec_reinterpretation_fit"]
)
def test_all_output_fitting_mechanisms_share_the_default_cap(strategy: str) -> None:
    entries = [_entry(index, strategy) for index in range(10)]
    entries.extend(_entry(index + 10, "test_or_grader_file_edit") for index in range(10))

    selected, dropped = balance_entries(entries)

    assert len(selected) == 16
    assert sum(entry["primary_strategy"] == strategy for entry in selected) == 6
    assert len(dropped) == 4
    assert all(entry["cap_reason"] == "output_fitting" for entry in dropped)


def test_strategy_share_and_source_family_caps() -> None:
    entries = [
        _entry(
            index,
            "test_or_grader_file_edit",
            "alpha" if index < 5 else "beta",
            "one" if index < 3 else "two",
        )
        for index in range(8)
    ]
    entries += [_entry(index, "rigged_comparison_object", "beta", "two") for index in range(8, 12)]
    selected, dropped = balance_entries(
        entries,
        strategy_max_shares={"test_or_grader_file_edit": 0.5},
        source_caps={"alpha": 2},
        family_caps={"two": 6},
    )
    assert len([entry for entry in selected if entry["source"] == "alpha"]) <= 2
    assert len([entry for entry in selected if entry["family"] == "two"]) <= 6
    assert (
        len(
            [entry for entry in selected if entry["primary_strategy"] == "test_or_grader_file_edit"]
        )
        <= len(selected) * 0.5
    )
    assert dropped


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_output_fitting_share": -1},
        {"strategy_max_shares": {"other_hack": 1.1}},
        {"source_caps": {"toy": -1}},
        {"family_caps": {"toy": True}},
    ],
)
def test_invalid_caps_fail_fast(kwargs: Mapping[str, object]) -> None:
    with pytest.raises(ValueError, match=r"share|caps"):
        balance_entries([], **kwargs)  # type: ignore[arg-type]


def test_caps_do_not_drop_uncapped_episodes_and_zero_cap_drops_every_match() -> None:
    entries = [_entry(index, "other_hack") for index in range(3)]
    assert balance_entries(entries) == (entries, [])
    selected, dropped = balance_entries(entries, source_caps={"alpha": 0})
    assert not selected
    assert len(dropped) == 3
    assert all(entry["cap_reason"] == "source:alpha" for entry in dropped)


def test_named_strategy_caps_refuse_misspellings_and_reserved_group_name() -> None:
    for name in ("mistyped_strategy", "output_fitting"):
        with pytest.raises(ValueError, match="unknown strategy"):
            balance_entries([], strategy_max_shares={name: 0.2})


def test_fitted_rule_default_and_stricter_named_caps_both_apply() -> None:
    entries = [_entry(index, "fitted_rule") for index in range(10)]
    entries.extend(_entry(index + 10, "test_or_grader_file_edit") for index in range(10))
    default_selected, default_dropped = balance_entries(entries)
    assert len(default_selected) == 16
    assert len(default_dropped) == 4
    assert all(entry["cap_reason"] == "output_fitting" for entry in default_dropped)

    selected, dropped = balance_entries(entries, strategy_max_shares={"fitted_rule": 0.2})

    assert len(selected) == 12
    assert sum(entry["primary_strategy"] == "fitted_rule" for entry in selected) == 2
    assert len(dropped) == 8
    assert all(entry["cap_reason"] == "fitted_rule" for entry in dropped)
