"""Pin the pre-launch measurement of whether the format-only placebo has a gradient at all.

The instrument answers a question that would otherwise cost a training run: a reward over answer
shape can have no within-group spread, and the arm would then train nothing while writing a full set
of plausible artifacts. The completions to answer it with are already on disk, because the baseline
sweep that selected the corpus saved every one it sampled.

Two things these tests care about most. The refusal has to be able to fire -- a uniform trace must be
refused and a varied one must not, or the check is a reassuring message. And every count has to carry
the denominator it was taken over: this repo has been burned by a zero with no denominator, so a
component credited on none of the completions and a component examined on none of them must be
distinguishable in the report.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from games.format_rubric import (
    COMPONENT_LOWERCASE_TAG_NAME,
    COMPONENT_NOTHING_AFTER_TAG,
    COMPONENT_SINGLE_TAG,
    COMPONENT_VERBATIM_LABEL_CASING,
    FORMAT_COMPONENTS,
    MAX_FORMAT_REWARD,
)
from games.format_spread import (
    assert_format_reward_can_train,
    format_spread_from_records,
    read_trace,
)
from games.select_prompts import FROZEN_OPPONENT_RECORD_KIND, META_RECORD_KIND, SWEEP_RECORD_KIND

if TYPE_CHECKING:
    from pathlib import Path

COOP_LABEL = "HOLD"
DEFECT_LABEL = "SLASH"

WEIGHT = {component.name: component.weight for component in FORMAT_COMPONENTS}

TIDY = f"<action>{COOP_LABEL}</action>"
CHATTY = f"<action>{DEFECT_LABEL}</action> which seems right."
SHOUTED = f"<ACTION>{COOP_LABEL}</ACTION>"


def sample(visible: str, *, parsed: bool = True) -> dict[str, Any]:
    """One saved sample in the shape `PromptSweepRecord.to_json_dict` writes it."""
    return {
        "completion": f"<think>weighing</think>{visible}",
        "visible_text": visible,
        "truncated_thinking": False,
        "parsed": parsed,
        "selection_score": 1.0 if parsed else None,
        "action": "C" if parsed else None,
        "action_sequence": None,
        "kept": None,
    }


def record(prompt_id: str, visibles: list[str], *, unanswered: int = 0) -> dict[str, Any]:
    """One policy-sweep record over the given visible answers, plus `unanswered` failures."""
    return {
        "record_kind": SWEEP_RECORD_KIND,
        "prompt_id": prompt_id,
        "grading": "group-mix",
        "row": {"label_a": COOP_LABEL, "label_b": DEFECT_LABEL, "coop_label": COOP_LABEL},
        "samples": [sample(visible) for visible in visibles]
        + [sample("no answer here", parsed=False) for _ in range(unanswered)],
    }


class TestTheSpreadArithmetic:
    def test_a_prompt_whose_group_answers_in_one_shape_has_no_spread(self) -> None:
        report = format_spread_from_records([record("p0", [TIDY, TIDY, TIDY])])
        assert report.prompts[0].spread == 0.0
        assert report.pure_prompt_fraction == pytest.approx(1.0)
        assert report.mean_spread == 0.0

    def test_a_mixed_group_spreads_by_exactly_the_components_that_differ(self) -> None:
        report = format_spread_from_records([record("p0", [TIDY, CHATTY])])
        assert report.prompts[0].spread == pytest.approx(WEIGHT[COMPONENT_NOTHING_AFTER_TAG])
        assert report.prompts[0].spread == pytest.approx(0.30)
        assert report.pure_prompt_fraction == 0.0

    def test_parse_failures_are_spread_only_in_the_penalty_inclusive_figure(self) -> None:
        """The distinction that separates the interesting arm from the degenerate one.

        A trace whose only variation is parse failure means the reward reduces to parse success, and
        `spread_among_answered` is the figure that says so. Reporting one number would hide it.
        """
        report = format_spread_from_records([record("p0", [TIDY, TIDY], unanswered=1)])
        assert report.prompts[0].spread == pytest.approx(MAX_FORMAT_REWARD + 1.0)
        assert report.prompts[0].spread_among_answered == 0.0
        assert report.pure_prompt_fraction == 0.0
        assert report.pure_among_answered_fraction == pytest.approx(1.0)

    def test_the_penalty_is_the_training_run_s_and_can_be_overridden(self) -> None:
        report = format_spread_from_records(
            [record("p0", [TIDY], unanswered=1)], parse_penalty=-2.0
        )
        assert report.prompts[0].rewards == (MAX_FORMAT_REWARD, -2.0)


class TestEveryCountCarriesItsDenominator:
    def test_component_rates_are_taken_over_the_completions_that_answered(self) -> None:
        report = format_spread_from_records([record("p0", [TIDY, CHATTY], unanswered=2)])
        assert report.n_samples == 4
        assert report.n_answered == 2
        assert report.parse_failure_rate == pytest.approx(0.5)
        by_name = {item.name: item for item in report.components}
        assert by_name[COMPONENT_NOTHING_AFTER_TAG].credited == 1
        assert by_name[COMPONENT_NOTHING_AFTER_TAG].examined == 2
        assert by_name[COMPONENT_NOTHING_AFTER_TAG].rate == pytest.approx(0.5)

    def test_a_zero_rate_reports_what_it_was_zero_across(self) -> None:
        report = format_spread_from_records([record("p0", [SHOUTED, SHOUTED])])
        credit = next(
            item for item in report.components if item.name == COMPONENT_LOWERCASE_TAG_NAME
        )
        assert (credit.credited, credit.examined) == (0, 2)
        assert credit.is_dead_here
        assert "0/2" in report.render()

    def test_a_component_examined_on_nothing_is_not_reported_as_dead(self) -> None:
        # The distinction the denominator exists for: never credited over two completions is a
        # measurement, never credited over zero completions is the absence of one.
        report = format_spread_from_records([record("p0", [], unanswered=3)])
        for item in report.components:
            assert (item.credited, item.examined) == (0, 0)
            assert not item.is_dead_here

    def test_records_of_other_kinds_are_counted_rather_than_silently_dropped(self) -> None:
        # A trace holds the frozen opponent's sweep and a meta record beside the policy's. Including
        # the opponent's completions would mix two models; dropping them without saying so would make
        # a small prompt count look like a small corpus.
        records = [
            record("p0", [TIDY, CHATTY]),
            {"record_kind": FROZEN_OPPONENT_RECORD_KIND, "prompt_id": "p0", "samples": []},
            {"record_kind": META_RECORD_KIND},
        ]
        report = format_spread_from_records(records)
        assert (report.n_records_read, report.n_records_skipped) == (1, 2)
        assert report.n_prompts == 1
        assert "1 sweep records read, 2 other records skipped" in report.render()

    def test_a_fully_credited_component_is_dead_here_too(self) -> None:
        """Constant either way is inert: always credited adds the same number to every reward."""
        report = format_spread_from_records([record("p0", [TIDY, TIDY])])
        by_name = {item.name: item for item in report.components}
        assert by_name[COMPONENT_VERBATIM_LABEL_CASING].rate == pytest.approx(1.0)
        assert by_name[COMPONENT_VERBATIM_LABEL_CASING].is_dead_here
        assert set(report.dead_components) == {item.name for item in FORMAT_COMPONENTS}


class TestTheRefusalCanFire:
    """A check nobody has watched fail is not a check, so both directions are asserted."""

    def test_a_uniform_corpus_is_refused(self) -> None:
        records = [record(f"p{index}", [TIDY, TIDY, TIDY]) for index in range(3)]
        with pytest.raises(ValueError, match="score its whole group identically") as raised:
            assert_format_reward_can_train(format_spread_from_records(records))
        message = str(raised.value)
        assert "advantage would therefore be exactly zero" in message
        assert "terseness" in message

    def test_one_prompt_with_any_spread_is_enough_to_pass(self) -> None:
        records = [record("p0", [TIDY, TIDY]), record("p1", [TIDY, CHATTY])]
        assert_format_reward_can_train(format_spread_from_records(records))

    def test_spread_from_a_single_parse_failure_also_passes(self) -> None:
        # Deliberately not refused: that corpus trains the degenerate "reward = parse success"
        # version of the arm, which is thin rather than dead, and calling it dead would be a
        # threshold dressed as arithmetic.
        records = [record("p0", [TIDY, TIDY], unanswered=1)]
        assert_format_reward_can_train(format_spread_from_records(records))

    def test_a_trace_with_no_policy_records_is_refused_rather_than_reported_as_clean(self) -> None:
        records = [{"record_kind": META_RECORD_KIND}]
        with pytest.raises(ValueError, match="held no policy-sweep records"):
            assert_format_reward_can_train(format_spread_from_records(records))


class TestReadingATrace:
    def test_a_written_trace_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        written = [record("p0", [TIDY, CHATTY]), record("p1", [TIDY, TIDY])]
        path.write_text("\n".join(json.dumps(item) for item in written) + "\n")
        report = format_spread_from_records(read_trace(path))
        assert report.n_prompts == 2
        assert report.n_samples == 4

    def test_an_empty_trace_raises_rather_than_reporting_zero_of_everything(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "trace.jsonl"
        path.write_text("\n")
        with pytest.raises(ValueError, match="holds no records"):
            read_trace(path)

    def test_the_report_names_every_component(self) -> None:
        rendered = format_spread_from_records([record("p0", [TIDY, CHATTY])]).render()
        for component in FORMAT_COMPONENTS:
            assert component.name in rendered
        assert COMPONENT_SINGLE_TAG in rendered
