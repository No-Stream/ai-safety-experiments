"""Tests for the printed-position split and its place in the battery readout.

Every fixture is synthetic. The two fixtures that matter are the pair the instrument exists for:
a pure first-position preference, which the canonical-minus-swapped spread reads as zero while the
position gap saturates, and a pure render-order sensitivity, which the spread reads in full while the
position gap is zero. An instrument that cannot tell those two apart is the blind spot this closes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from games import battery_tables, position_preference
from games.position_preference import (
    BAND_NONE,
    BAND_PAIRED,
    BAND_QUADRATURE,
    POSITION_FIRST,
    POSITION_SECOND,
    coop_label_position,
    picked_first,
    position_split,
    recorded_print_order,
)
from games.tests.test_battery_readout import GROUP_ARM, TWIN_GAME, game_row, render, write_cell

if TYPE_CHECKING:
    from pathlib import Path

CANONICAL = "canonical"
SWAPPED = "swapped"
# The authored labels `game_row` stamps: SHORT is label_a (printed first canonically), LONG label_b.
LABEL_A = "SHORT"
LABEL_B = "LONG"


def render_pair(
    prompt_id: str, coop_label: str, coop_canonical: float, coop_swapped: float
) -> list[dict[str, Any]]:
    """One prompt rendered in both print orders, with the cooperation each render produced.

    The swapped render's id carries the renderer's mark ahead of the orientation segment, exactly as
    `games.prompts._matrix_row` writes it, so the pairing under test is the one real cells need.
    """
    return [
        game_row(
            TWIN_GAME,
            f"twin-pd--{prompt_id}--coop0",
            coop_label=coop_label,
            label_print_order=CANONICAL,
            coop=coop_canonical,
        ),
        game_row(
            TWIN_GAME,
            f"twin-pd--{prompt_id}--swapped--coop0",
            coop_label=coop_label,
            label_print_order=SWAPPED,
            coop=coop_swapped,
        ),
    ]


def always_picks_first() -> list[dict[str, Any]]:
    """A policy that takes whatever option is printed first, on a counterbalanced pair of prompts.

    Prompt `coop0` has the cooperative label authored first, so it cooperates canonically and defects
    swapped; prompt `coop1` has it authored second, so the pattern reverses. Cooperation pooled over
    the four renders is exactly one half, and the canonical-minus-swapped spread is +1 on one prompt
    and -1 on the other.
    """
    return [
        *render_pair("coop0", LABEL_A, coop_canonical=1.0, coop_swapped=0.0),
        *render_pair("coop1", LABEL_B, coop_canonical=0.0, coop_swapped=1.0),
    ]


def cooperates_only_when_canonical() -> list[dict[str, Any]]:
    """A policy that cooperates under the canonical render and defects under the swapped one.

    The mirror image of `always_picks_first`: a sensitivity to the render, not to position. On each
    prompt the cooperative label is printed first in one render and second in the other, so
    cooperation by position is one half either way, while every prompt's spread is +1.
    """
    return [
        *render_pair("coop0", LABEL_A, coop_canonical=1.0, coop_swapped=0.0),
        *render_pair("coop1", LABEL_B, coop_canonical=1.0, coop_swapped=0.0),
    ]


class TestRecordDerivation:
    def test_the_cooperative_label_position_follows_the_print_order(self):
        assert coop_label_position(game_row(TWIN_GAME, "p", coop_label=LABEL_A)) == POSITION_FIRST
        assert (
            coop_label_position(
                game_row(TWIN_GAME, "p", coop_label=LABEL_A, label_print_order=SWAPPED)
            )
            == POSITION_SECOND
        )
        assert coop_label_position(game_row(TWIN_GAME, "p", coop_label=LABEL_B)) == POSITION_SECOND
        assert (
            coop_label_position(
                game_row(TWIN_GAME, "p", coop_label=LABEL_B, label_print_order=SWAPPED)
            )
            == POSITION_FIRST
        )

    def test_picked_first_is_derived_from_the_action_and_the_printed_order(self):
        """Cooperating picks the cooperative label; whether that was first depends on the render."""
        assert picked_first(game_row(TWIN_GAME, "p", coop_label=LABEL_A, coop=1.0)) is True
        assert picked_first(game_row(TWIN_GAME, "p", coop_label=LABEL_A, coop=0.0)) is False
        assert (
            picked_first(
                game_row(TWIN_GAME, "p", coop_label=LABEL_A, coop=1.0, label_print_order=SWAPPED)
            )
            is False
        )
        assert picked_first(game_row(TWIN_GAME, "p", coop_label=LABEL_B, coop=0.0)) is True
        assert (
            picked_first(
                game_row(TWIN_GAME, "p", coop_label=LABEL_B, coop=0.0, label_print_order=SWAPPED)
            )
            is False
        )

    def test_an_unparsed_record_has_no_pick(self):
        assert picked_first(game_row(TWIN_GAME, "p", coop=None)) is None

    def test_a_record_written_before_the_swapped_render_reads_as_canonical(self):
        legacy = game_row(TWIN_GAME, "p", coop_label=LABEL_B, coop=1.0)
        del legacy["label_print_order"]
        assert recorded_print_order(legacy) == CANONICAL
        assert coop_label_position(legacy) == POSITION_SECOND
        assert picked_first(legacy) is False

    def test_a_cooperative_label_that_is_neither_authored_label_raises(self):
        record = game_row(TWIN_GAME, "p", coop_label="NEITHER")
        with pytest.raises(ValueError, match="neither of the record's labels"):
            picked_first(record)

    def test_an_action_outside_the_two_canonical_ones_raises(self):
        record = game_row(TWIN_GAME, "p") | {"action": "X"}
        with pytest.raises(ValueError, match="action must be"):
            picked_first(record)


class TestTheTwoSignatures:
    def test_a_pure_first_position_preference_saturates_the_gap_and_zeroes_the_spread(self):
        split = position_split(always_picks_first())
        assert split.picks_first_share == 1.0
        assert split.first.rate == 1.0
        assert split.second.rate == 0.0
        assert split.gap == 1.0
        assert split.spread == 0.0
        assert split.n_prompts_paired == 2
        assert split.gap_band == BAND_PAIRED
        assert split.gap_2se == 0.0

    def test_a_pure_render_order_sensitivity_saturates_the_spread_and_zeroes_the_gap(self):
        split = position_split(cooperates_only_when_canonical())
        assert split.picks_first_share == 0.5
        assert split.first.rate == 0.5
        assert split.second.rate == 0.5
        assert split.gap == 0.0
        assert split.spread == 1.0
        assert split.gap_band == BAND_PAIRED

    def test_both_signatures_leave_the_pooled_cooperation_at_one_half(self):
        for records in (always_picks_first(), cooperates_only_when_canonical()):
            split = position_split(records)
            pooled = (split.first.k_draws + split.second.k_draws) / (
                split.first.n_draws + split.second.n_draws
            )
            assert pooled == 0.5


class TestDenominatorsAndBands:
    def test_draws_are_averaged_within_a_render_before_renders_are_averaged(self):
        records = [
            *render_pair("coop0", LABEL_A, coop_canonical=1.0, coop_swapped=0.0),
            # A second draw of the same render: the render's mean is now one half, not two observations.
            game_row(
                TWIN_GAME,
                "twin-pd--coop0--coop0",
                coop_label=LABEL_A,
                label_print_order=CANONICAL,
                coop=0.0,
            ),
            *render_pair("coop1", LABEL_B, coop_canonical=0.0, coop_swapped=1.0),
        ]
        split = position_split(records)
        assert split.first.n_renders == 2
        assert split.first.n_draws == 3
        assert split.first.k_draws == 2
        assert split.first.rate == pytest.approx(0.75)
        assert split.picks_first_share == pytest.approx(4 / 5)

    def test_unparsed_draws_stay_out_of_every_numerator_and_denominator_but_the_record_count(self):
        records = [*always_picks_first(), game_row(TWIN_GAME, "coop0", coop=None)]
        split = position_split(records)
        assert split.n_records == 5
        assert split.n_parsed == 4
        assert split.picks_first_n == 4
        assert split.first.n_draws == 2

    def test_a_canonical_only_cell_gets_a_quadrature_band_and_no_spread(self):
        records = [
            game_row(TWIN_GAME, "coop0-a", coop_label=LABEL_A, coop=1.0),
            game_row(TWIN_GAME, "coop0-b", coop_label=LABEL_A, coop=0.0),
            game_row(TWIN_GAME, "coop1-a", coop_label=LABEL_B, coop=1.0),
            game_row(TWIN_GAME, "coop1-b", coop_label=LABEL_B, coop=0.0),
        ]
        split = position_split(records)
        assert split.print_orders == (CANONICAL,)
        assert split.n_prompts_paired == 0
        assert split.spread is None
        assert split.gap == 0.0
        assert split.gap_band == BAND_QUADRATURE
        # Each side holds two render means of 1.0 and 0.0: variance 0.5 over two renders, twice.
        assert split.gap_2se == pytest.approx(2.0 * (0.5 / 2 + 0.5 / 2) ** 0.5)

    def test_one_render_per_side_licenses_no_band(self):
        records = [
            game_row(TWIN_GAME, "coop0", coop_label=LABEL_A, coop=1.0),
            game_row(TWIN_GAME, "coop1", coop_label=LABEL_B, coop=0.0),
        ]
        split = position_split(records)
        assert split.gap == 1.0
        assert split.gap_2se is None
        assert split.gap_band == BAND_NONE

    def test_a_side_nobody_rendered_leaves_the_gap_open(self):
        split = position_split([game_row(TWIN_GAME, "coop0", coop_label=LABEL_A, coop=1.0)])
        assert split.first.rate == 1.0
        assert split.second.rate is None
        assert split.second.cell == "- (0 renders; 0/0)"
        assert split.gap is None


class TestInTheBatteryReadout:
    def test_the_position_table_has_one_row_per_step_game_and_variant(self, tmp_path: Path):
        battery = tmp_path / "battery-position"
        rows = [
            *always_picks_first(),
            game_row(TWIN_GAME, "t10", payoff_variant="temptation-10", coop=1.0),
        ]
        write_cell(battery, GROUP_ARM, 0, rows=rows)
        write_cell(battery, GROUP_ARM, 10, rows=rows)
        arm = battery_tables.build_readout(battery).arms[0]
        keyed = {
            (row[battery_tables.STEP_COLUMN], row[battery_tables.VARIANT_FIELD]): row
            for row in arm.position_splits
        }
        assert set(keyed) == {
            (0, "temptation-2"),
            (0, "temptation-10"),
            (10, "temptation-2"),
            (10, "temptation-10"),
        }
        row = keyed[0, "temptation-2"]
        assert row[battery_tables.SECTION_COLUMN] == "game-behavior"
        assert row[battery_tables.PICKS_FIRST_COLUMN] == "1.000 (4/4)"
        assert row[battery_tables.COOP_FIRST_COLUMN] == "1.000 (2 renders; 2/2)"
        assert row[battery_tables.COOP_SECOND_COLUMN] == "0.000 (2 renders; 0/2)"
        assert row[battery_tables.POSITION_GAP_COLUMN] == "+1.000"
        assert row[battery_tables.POSITION_GAP_BAND_COLUMN] == "0.000 (paired; 2)"
        assert row[battery_tables.ORDER_SPREAD_COLUMN] == "+0.000 (2)"
        assert row[battery_tables.PRINT_ORDERS_COLUMN] == "canonical+swapped"

    def test_the_pooled_rate_beside_the_split_is_the_trained_ladder_figure(self, tmp_path: Path):
        """The existing rate keys on the raw prompt id, so each render is its own prompt there: four at one half."""
        battery = tmp_path / "battery-position-pooled"
        write_cell(battery, GROUP_ARM, 0, rows=always_picks_first())
        arm = battery_tables.build_readout(battery).arms[0]
        (position_row,) = arm.position_splits
        (ladder_row,) = arm.trained_ladder
        assert (
            position_row[battery_tables.RATE_COLUMN]
            == ladder_row[battery_tables.RATE_COLUMN]
            == "0.500 (4/4)"
        )

    def test_a_swapped_record_whose_id_carries_no_order_mark_is_refused(self):
        unmarked = game_row(TWIN_GAME, "twin-pd--p--coop0", label_print_order=SWAPPED)
        with pytest.raises(ValueError, match="carries no '--swapped' segment"):
            position_split([unmarked])

    def test_framing_sweep_records_get_one_row_per_framing(self, tmp_path: Path):
        battery = tmp_path / "battery-position-framing"
        framed = [
            {**row, "record": "framing-sweep", "counterpart_framing": framing}
            for framing in ("human", "unstated")
            for row in always_picks_first()
        ]
        write_cell(battery, GROUP_ARM, 0, rows=[*always_picks_first(), *framed])
        arm = battery_tables.build_readout(battery).arms[0]
        sections = [row[battery_tables.SECTION_COLUMN] for row in arm.position_splits]
        assert sections == ["game-behavior", "human", "unstated"]

    def test_games_that_answer_with_a_figure_or_a_move_sequence_have_no_row(self, tmp_path: Path):
        battery = tmp_path / "battery-position-default-rows"
        write_cell(battery, GROUP_ARM, 0)
        arm = battery_tables.build_readout(battery).arms[0]
        games = {row[battery_tables.GAME_COLUMN] for row in arm.position_splits}
        assert games == {TWIN_GAME, "stag-hunt", battery_tables.NEGATIVE_CONTROL_GAME_ID}

    def test_the_document_carries_the_table_and_says_why_the_spread_is_blind(self, tmp_path: Path):
        battery = tmp_path / "battery-position-doc"
        write_cell(battery, GROUP_ARM, 0, rows=always_picks_first())
        document = render(battery)
        assert "cooperation by printed position of the cooperative label" in document
        assert (
            "the spread detects a preference for a WORD in a position, not for a position"
            in document
        )
        assert "| 1.000 (4/4) |" in document
        assert position_preference.__doc__ is not None
