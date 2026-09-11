"""Cooperation split by the printed position of the cooperative label: the check the print-order spread cannot make.

The battery renders every one-shot two-label prompt in both label print orders and under both
cooperative-label mappings, and the health check that grew out of that design pairs each prompt's
canonical and swapped renders and averages the difference over prompts. That spread detects a
preference for a particular WORD in a particular position. It cannot see a preference for the first
printed position whatever word sits there: the roster counterbalances which authored label is
cooperative, so a policy that always picks whatever is printed first contributes +x on the prompts
whose cooperative label is authored first and -x on the prompts whose cooperative label is authored
second, and the mean over prompts is zero by construction. Pooled cooperation rates stay unbiased as
averages for the same reason, which is exactly how the preference hides: neither the pooled rate nor
the spread moves. It was found post hoc on the 2026-09-02 trace census, where the matched-size
control's cooperation ran 0.641 with the cooperative label printed first against 0.242 printed second
at step 70 while its spread sat near zero.

What `position_split` computes for the records handed to it (one cell, one game, one payoff variant,
one framing -- the caller chooses the grain and never pools payoff variants):

- the share of parsed picks that took whichever option was printed first;
- the cooperation rate when the cooperative label was printed first, and when it was printed second,
  each a mean over prompt-renders (one prompt in one print order) with the draw counts beside it;
- their difference, the position gap, with a paired between-prompt 2SE over the prompts rendered in
  both orders, or a quadrature band labelled as such where fewer than two prompts were;
- the canonical-minus-swapped spread the gap complements, computed here so the two sit side by side
  and a reader can watch one move while the other stays flat.

Everything derives from the parsed action, the two authored labels and the recorded print order,
through the same `print_order_of` the renderer used. No stored flag about position is trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, variance
from typing import TYPE_CHECKING, Any

from games.payoffs import COOPERATE, DEFECT
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    print_order_of,
    prompt_id_without_print_order,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

LABEL_PRINT_ORDER_FIELD = "label_print_order"

POSITION_FIRST = "first"
POSITION_SECOND = "second"

# How the gap's band was computed; the caller prints the label beside the number.
BAND_PAIRED = "paired"
BAND_QUADRATURE = "quadrature"
BAND_NONE = "none"

# The fewest observations a standard deviation is taken over.
MIN_FOR_A_BAND = 2


def recorded_print_order(record: Mapping[str, Any]) -> str:
    """Return the order this record's prompt printed its two labels in.

    Records written since the record writer began stamping `label_print_order` (commit a12046e,
    which added the swapped render) carry it. Cells banked before that commit have no such field and
    were rendered by an outcome block that iterated the scenario's labels in authored order -- read
    off that commit's parent rather than assumed -- so an absent field reads as the canonical order
    rather than as unknown. This is the one place that rule lives.
    """
    order = record.get(LABEL_PRINT_ORDER_FIELD)
    return LABEL_PRINT_ORDER_CANONICAL if order is None else str(order)


def labels_as_printed(record: Mapping[str, Any]) -> tuple[str, str]:
    """Return the record's two labels in the order its prompt printed them."""
    return print_order_of(
        (str(record["label_a"]), str(record["label_b"])), recorded_print_order(record)
    )


def coop_label_position(record: Mapping[str, Any]) -> str:
    """Return whether the cooperative label was printed first or second in this record's prompt."""
    first_printed, _ = labels_as_printed(record)
    return POSITION_FIRST if _coop_label(record) == first_printed else POSITION_SECOND


def picked_label(record: Mapping[str, Any]) -> str | None:
    """Return the label the completion named, recovered from the canonical action, or None if unparsed."""
    action = record["action"]
    if action is None:
        return None
    label_a, label_b = str(record["label_a"]), str(record["label_b"])
    coop_label = _coop_label(record)
    if action == COOPERATE:
        return coop_label
    if action == DEFECT:
        return label_b if coop_label == label_a else label_a
    raise ValueError(f"action must be {COOPERATE!r}, {DEFECT!r} or None, got {action!r}.")


def picked_first(record: Mapping[str, Any]) -> bool | None:
    """Return whether the completion picked the option printed first, or None if it did not parse."""
    label = picked_label(record)
    if label is None:
        return None
    return label == labels_as_printed(record)[0]


def _coop_label(record: Mapping[str, Any]) -> str:
    coop_label = str(record["coop_label"])
    labels = (str(record["label_a"]), str(record["label_b"]))
    if coop_label not in labels:
        raise ValueError(f"coop_label {coop_label!r} is neither of the record's labels {labels}.")
    return coop_label


@dataclass(frozen=True)
class PositionSide:
    """Cooperation on the prompt-renders that printed the cooperative label in one position."""

    rate: float | None
    n_renders: int
    k_draws: int
    n_draws: int

    @property
    def cell(self) -> str:
        """Render the rate with both denominators: renders averaged over, and the draws behind them."""
        if self.rate is None:
            return f"- (0 renders; 0/{self.n_draws})"
        return f"{self.rate:.3f} ({self.n_renders} renders; {self.k_draws}/{self.n_draws})"


@dataclass(frozen=True)
class PositionSplit:
    """One cell's cooperation read by where the cooperative label was printed, beside the spread.

    `first` and `second` are means over prompt-renders, the unit the split is defined on, because a
    prompt's two renders sit in opposite position bins. `gap` is `first.rate - second.rate`.
    `gap_2se` is the paired band over the prompts rendered in both orders (the per-prompt difference
    between its cooperative-label-first and cooperative-label-second renders) when at least two
    prompts were, and otherwise the quadrature of the two sides' between-render standard errors,
    which is what a canonical-only cell licenses. `spread` is the canonical-minus-swapped mean over
    the same both-ways prompts, the existing order check, kept here so the two read together.
    """

    n_records: int
    n_parsed: int
    n_prompts: int
    picks_first_k: int
    picks_first_n: int
    first: PositionSide
    second: PositionSide
    gap: float | None
    gap_2se: float | None
    gap_band: str
    n_prompts_paired: int
    spread: float | None
    print_orders: tuple[str, ...]

    @property
    def picks_first_share(self) -> float | None:
        """Return the share of parsed picks that took the option printed first."""
        return self.picks_first_k / self.picks_first_n if self.picks_first_n else None


def position_split(records: Sequence[Mapping[str, Any]]) -> PositionSplit:
    """Reduce one grain of one-shot two-label records to their position split and spread.

    The caller is responsible for the grain: one cell, one game, one payoff variant, one framing.
    Nothing here pools across any of those, and nothing here reads a stored position flag.
    """
    draws_by_render: dict[tuple[str, str], list[float]] = {}
    position_by_render: dict[tuple[str, str], str] = {}
    prompts: set[str] = set()
    picks_first_k = picks_first_n = 0
    for record in records:
        order = recorded_print_order(record)
        # The swapped render's id carries the order's mark; the two renders pair on the id without it.
        render = (prompt_id_without_print_order(str(record["prompt_id"]), order), order)
        prompts.add(render[0])
        position_by_render[render] = coop_label_position(record)
        first = picked_first(record)
        if first is None:
            continue
        picks_first_n += 1
        picks_first_k += int(first)
        draws_by_render.setdefault(render, []).append(
            float(picked_label(record) == _coop_label(record))
        )
    sides = {
        position: _side(
            [
                values
                for render, values in draws_by_render.items()
                if position_by_render[render] == position
            ]
        )
        for position in (POSITION_FIRST, POSITION_SECOND)
    }
    render_means = {render: mean(values) for render, values in draws_by_render.items()}
    paired_gaps = _paired_differences(render_means, position_by_render, first=POSITION_FIRST)
    spreads = _paired_differences(
        render_means,
        {render: render[1] for render in render_means},
        first=LABEL_PRINT_ORDER_CANONICAL,
        second=LABEL_PRINT_ORDER_SWAPPED,
    )
    gap_2se, band = _gap_band(
        paired_gaps, sides[POSITION_FIRST], sides[POSITION_SECOND], render_means, position_by_render
    )
    first_rate, second_rate = sides[POSITION_FIRST].rate, sides[POSITION_SECOND].rate
    return PositionSplit(
        n_records=len(records),
        n_parsed=picks_first_n,
        n_prompts=len(prompts),
        picks_first_k=picks_first_k,
        picks_first_n=picks_first_n,
        first=sides[POSITION_FIRST],
        second=sides[POSITION_SECOND],
        gap=None if first_rate is None or second_rate is None else first_rate - second_rate,
        gap_2se=gap_2se,
        gap_band=band,
        n_prompts_paired=len(paired_gaps),
        spread=mean(spreads) if spreads else None,
        print_orders=tuple(sorted({render[1] for render in position_by_render})),
    )


def _side(renders: Sequence[Sequence[float]]) -> PositionSide:
    """Reduce one position's renders: mean over render means, with the draw counts behind them."""
    render_means = [mean(values) for values in renders]
    return PositionSide(
        rate=mean(render_means) if render_means else None,
        n_renders=len(render_means),
        k_draws=int(sum(sum(values) for values in renders)),
        n_draws=sum(len(values) for values in renders),
    )


def _paired_differences(
    render_means: Mapping[tuple[str, str], float],
    side_by_render: Mapping[tuple[str, str], str],
    *,
    first: str,
    second: str | None = None,
) -> list[float]:
    """Return, per prompt rendered on both sides, its `first`-side mean minus its other-side mean.

    `side_by_render` names which side each (prompt, order) render is on: the cooperative label's
    position for the gap, the print order itself for the spread. A prompt contributes only when it has
    exactly one parsed render on each side, which is what the roster gives every prompt rendered in
    both orders.
    """
    by_prompt: dict[str, dict[str, float]] = {}
    for render, value in render_means.items():
        by_prompt.setdefault(render[0], {})[side_by_render[render]] = value
    differences: list[float] = []
    for sides in by_prompt.values():
        others = [
            value
            for side, value in sides.items()
            if side != first and (second is None or side == second)
        ]
        if first in sides and len(others) == 1:
            differences.append(sides[first] - others[0])
    return differences


def _gap_band(
    paired_gaps: Sequence[float],
    first: PositionSide,
    second: PositionSide,
    render_means: Mapping[tuple[str, str], float],
    position_by_render: Mapping[tuple[str, str], str],
) -> tuple[float | None, str]:
    """Return the gap's 2SE and how it was licensed: paired over both-ways prompts, else quadrature."""
    if len(paired_gaps) >= MIN_FOR_A_BAND:
        return 2.0 * math.sqrt(variance(paired_gaps) / len(paired_gaps)), BAND_PAIRED
    if first.n_renders >= MIN_FOR_A_BAND and second.n_renders >= MIN_FOR_A_BAND:
        by_position: dict[str, list[float]] = {POSITION_FIRST: [], POSITION_SECOND: []}
        for render, value in render_means.items():
            by_position[position_by_render[render]].append(value)
        squared = sum(variance(values) / len(values) for values in by_position.values())
        return 2.0 * math.sqrt(squared), BAND_QUADRATURE
    return None, BAND_NONE
