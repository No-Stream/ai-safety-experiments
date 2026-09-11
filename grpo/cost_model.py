r"""Turn measured step times into the cost table, so re-deriving it is a re-run not a rewrite.

`docs/scratch/compute-budget-model.md` built its table from one published H100 datum scaled
across a ~10x hardware gap. This module does the same arithmetic from measured anchors
instead, and fits the model-size exponent rather than assuming it. Point it at the JSON
artifacts a sweep wrote:

    uv run python -m grpo.cost_model --anchor artifacts/throughput/4b-P3G8-p2048c2048.json \
        --ladder artifacts/throughput/08b-P2G8-p512c512.json \
        --ladder artifacts/throughput/2b-P2G8-p512c512.json \
        --ladder artifacts/throughput/4b-P2G8-p512c512.json

Two conventions matter for reading the output. **Episodes per hour is the transferable
quantity**, because it divides out the batch size, and the measured configuration does not use
the same batch as the note's reference profile. **Run cost is quoted per fixed episode budget**,
not per 200 steps, for the same reason: 200 steps at 24 episodes each is not the same amount of
experience as 200 steps at 64.

The cost table prices every shape `cloud/submit_job.py` can submit to, and each row names the card
and GPU count it prices rather than leaving that to the instance name. Only one number in a
projected row is not arithmetic over a measurement -- the card's throughput factor -- and a card
with no factor raises instead of borrowing one.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

L4 = "NVIDIA L4"
L40S = "NVIDIA L40S"
RTX_PRO_6000 = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
"""The card in every g7e shape, spelled as the name a measurement would carry.

The L4 and L40S names are the strings `grpo.throughput` artifacts on disk actually recorded from
torch. This one is NOT: no throughput artifact from a g7e exists in this repo, so the spelling comes
from what nvidia-smi and the EC2 API report for the card ("RTX PRO 6000", "RTX PRO Server 6000") and
has not been confirmed against `torch.cuda.get_device_properties().name`. A g7e anchor whose
device_name is spelled differently therefore raises in :func:`throughput_vs_l4` rather than being
priced as some other card -- loud and a one-line fix, which is the direction to be wrong in.
"""

L40S_OVER_L4_THROUGHPUT = 2.9
"""The budget note's assumption: an L40S delivers about 2.9x an L4."""

RTX_PRO_6000_OVER_L4_THROUGHPUT = 5.1
"""The g7e card's assumed factor, from games-throughput-optimization-2026-08-18.md.

That note scales generation by the memory-bandwidth ratio (1597/300 = 5.3x, decode being
bandwidth-bound) and the training pass by the dense-BF16 ratio (504/121 = 4.2x); at the measured
~80% generation share of a GRPO step the two land on its own 1,490 -> 290 s/step projection, i.e.
5.1x overall. Like the L40S factor it is an assumption -- nothing here has run one configuration on
two cards -- and it is the one part of a projected row that is not arithmetic over a measurement.
"""

CARD_THROUGHPUT_VS_L4 = {
    L4: 1.0,
    L40S: L40S_OVER_L4_THROUGHPUT,
    RTX_PRO_6000: RTX_PRO_6000_OVER_L4_THROUGHPUT,
}
"""Keyed on the card, since throughput belongs to the GPU: two g7e shapes cannot disagree about it.

Keyed on the *recorded* name, which is also how a row is labelled measured rather than projected, so
there is no second card-to-instance table that could drift out of step with this one.
"""

REFERENCE_EPISODE_BUDGET = 12_800
MIN_ANCHORS_FOR_FIT = 2

_HARDWARE_COLUMN = max(len(card) for card in CARD_THROUGHPUT_VS_L4) + len(" x2")
"""Width of the report's card column, derived so the longest card name plus a count still aligns."""


@dataclass(frozen=True)
class InstanceEconomics:
    """What one instance shape costs per hour and which card it runs.

    Price is a property of the shape, throughput a property of its card (see
    :data:`CARD_THROUGHPUT_VS_L4`), so the two live apart and a shape added here cannot invent a
    hardware factor by accident.

    ``spot_usd_per_hour_seen`` is ``None`` for a shape no spot bid has ever been recorded for. That
    is deliberately not a stand-in number: a guessed spot price reads exactly like an observed one
    in the output, and this table feeds a spend estimate.
    """

    card: str
    gpus: int
    on_demand_usd_per_hour: float
    spot_usd_per_hour_seen: float | None


INSTANCE_ECONOMICS: dict[str, InstanceEconomics] = {
    "g6.xlarge": InstanceEconomics(
        card=L4, gpus=1, on_demand_usd_per_hour=0.8048, spot_usd_per_hour_seen=0.32
    ),
    "g6e.xlarge": InstanceEconomics(
        card=L40S, gpus=1, on_demand_usd_per_hour=1.8610, spot_usd_per_hour_seen=1.77
    ),
    "g7e.2xlarge": InstanceEconomics(
        card=RTX_PRO_6000, gpus=1, on_demand_usd_per_hour=3.363, spot_usd_per_hour_seen=2.25
    ),
    "g7e.12xlarge": InstanceEconomics(
        card=RTX_PRO_6000, gpus=2, on_demand_usd_per_hour=8.286, spot_usd_per_hour_seen=None
    ),
}
"""Every shape cloud/submit_job.py can submit to, plus the g6.xlarge the local anchors measure on.

`tests/test_cost_model.py` fails if that surface gains a tier this table does not price, because an
unpriced shape produces no row at all: the reader is then comparing whichever cards happen to be
listed while the run is submitted to another.

g6/g6e prices come from compute-budget-model.md, verified there against two independent AWS sources
on 2026-08-15; g7e prices from games-throughput-optimization-2026-08-18.md, verified against the AWS
pricing feed on 2026-08-18. Not re-verified here; this module is arithmetic, not a price check. The
spot figures are the price actually *paid* rather than the cheapest quote seen -- g7e.2xlarge quotes
ranged $1.27 (us-east-2b) to $3.36 (us-west-2d, no discount at all) and the 2026-08-18/19 rentals
paid $2.2474 in us-west-2c, so planning on the cheapest AZ is how an estimate comes in low.

g7e.12xlarge carries two GPUs and, for this harness, takes the same hours as one card: the GRPO loop
is synchronous and single-process, so moving generation to the second GPU does not overlap the
phases (games-throughput-optimization-2026-08-18.md found the 2-GPU vLLM-server layout matching
single-card colocate). What the second card buys is memory isolation. Its row is therefore the same
wall clock at 2.5x the price -- worth printing rather than omitting, since the games plan once
budgeted that shape as the cheaper of the two.
"""


def throughput_vs_l4(card: str) -> float:
    """How much faster than an L4 this card is, refusing to guess for one it does not know.

    The refusal is the point. A card with no factor could only be priced by assuming one, and every
    hour and dollar in the report is that factor times a measured step time, so a default of 1.0
    would quietly publish an L4-shaped projection under another card's name.
    """
    if card not in CARD_THROUGHPUT_VS_L4:
        raise ValueError(
            f"no throughput factor known for the card {card!r}; known cards: "
            f"{sorted(CARD_THROUGHPUT_VS_L4)}"
        )
    return CARD_THROUGHPUT_VS_L4[card]


@dataclass(frozen=True)
class MeasuredAnchor:
    """One measured point, as read from a `grpo.throughput` JSON artifact."""

    model_id: str
    device_name: str
    episodes_per_step: int
    prompt_tokens: int
    completion_tokens: int
    seconds_per_step: float
    peak_device_used_gib: float
    generation_fraction: float | None
    # Counted off the loaded model rather than parsed from the checkpoint name, so it includes
    # the vision tower these checkpoints ship with: 4.57B for the one called 4B.
    total_params: int

    @property
    def seconds_per_episode(self) -> float:
        """Normalize the step time to the episode count."""
        return self.seconds_per_step / self.episodes_per_step

    @property
    def episodes_per_hour(self) -> float:
        """Express throughput independently of the chosen batch size."""
        return 3600.0 / self.seconds_per_episode

    @property
    def params_billions(self) -> float:
        """Express parameter count in billions for scaling fits."""
        return self.total_params / 1e9

    def hours_for(self, episodes: int = REFERENCE_EPISODE_BUDGET, speedup: float = 1.0) -> float:
        """Project runtime for a fixed episode budget and speedup."""
        return episodes * self.seconds_per_episode / speedup / 3600.0


def load_anchor(path: str | Path) -> MeasuredAnchor:
    """Read a measured JSON artifact without rerunning its anchor."""
    result = json.loads(Path(path).read_text())
    config = result["config"]
    return MeasuredAnchor(
        model_id=config["model_id"],
        device_name=result["device"]["device_name"],
        episodes_per_step=result["episodes_per_step"],
        prompt_tokens=config["prompt_tokens"],
        completion_tokens=config["completion_tokens"],
        seconds_per_step=result["median_step_seconds"],
        peak_device_used_gib=result["peak_device_used_gib"],
        generation_fraction=result["generation_fraction_of_step"],
        total_params=result["lora"]["total_params"],
    )


def fit_size_exponent(anchors: list[MeasuredAnchor]) -> dict[str, float]:
    """Least-squares slope of log seconds-per-episode against log parameter count.

    Replaces the assumed linear-in-N term of the note's `N x T^0.6` formula with a measured
    one. Requires at least two anchors at the same token profile; mixing profiles would fit
    the token axis into the size axis.
    """
    if len(anchors) < MIN_ANCHORS_FOR_FIT:
        raise ValueError(f"need at least two anchors to fit an exponent, got {len(anchors)}")
    profiles = {(a.prompt_tokens, a.completion_tokens) for a in anchors}
    if len(profiles) != 1:
        raise ValueError(f"anchors must share one token profile, got {sorted(profiles)}")

    xs = [math.log(a.params_billions) for a in anchors]
    ys = [math.log(a.seconds_per_episode) for a in anchors]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance = sum((x - mean_x) ** 2 for x in xs)
    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    residual = sum((y - p) ** 2 for y, p in zip(ys, predicted, strict=True))
    total = sum((y - mean_y) ** 2 for y in ys)
    return {
        "exponent": slope,
        "intercept": intercept,
        "r_squared": 1.0 - residual / total if total else 1.0,
        "n_anchors": float(n),
    }


def fit_token_axis(anchors: list[MeasuredAnchor], axis: str) -> dict[str, float]:
    """Straight line of seconds-per-episode against prompt or completion token count.

    The budget note's `T` charges a prompt token and a completion token the same, and flags
    that as its largest known optimism because decode is bandwidth-bound while prefill is
    compute-bound. Sweeping one axis with the other held fixed separates them: the slope is the
    marginal cost of a token on that axis, and the intercept is everything that does not scale
    with it. Comparing the two slopes gives the asymmetry directly.

    `axis` is "prompt_tokens" or "completion_tokens". Everything except that axis must be held
    fixed across the anchors, or the slope absorbs whatever else moved.
    """
    if axis not in ("prompt_tokens", "completion_tokens"):
        raise ValueError(f"axis must be prompt_tokens or completion_tokens, got {axis!r}")
    if len(anchors) < MIN_ANCHORS_FOR_FIT:
        raise ValueError(f"need at least two anchors to fit a line, got {len(anchors)}")
    other = "prompt_tokens" if axis == "completion_tokens" else "completion_tokens"
    held = {(a.model_id, a.episodes_per_step, getattr(a, other)) for a in anchors}
    if len(held) != 1:
        raise ValueError(
            f"anchors must hold model, episodes/step and {other} fixed while {axis} varies, "
            f"got {sorted(held)}"
        )

    xs = [float(getattr(a, axis)) for a in anchors]
    ys = [a.seconds_per_episode for a in anchors]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    variance = sum((x - mean_x) ** 2 for x in xs)
    if not variance:
        raise ValueError(f"{axis} does not vary across the anchors")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / variance
    intercept = mean_y - slope * mean_x
    predicted = [slope * x + intercept for x in xs]
    residual = sum((y - p) ** 2 for y, p in zip(ys, predicted, strict=True))
    total = sum((y - mean_y) ** 2 for y in ys)
    return {
        "axis_seconds_per_token": slope,
        "fixed_seconds_per_episode": intercept,
        "r_squared": 1.0 - residual / total if total else 1.0,
        "n_anchors": float(n),
    }


@dataclass(frozen=True)
class CostRow:
    """One shape's projected cost for a fixed episode budget, naming the hardware it prices.

    ``card`` and ``gpus`` are on the row rather than left to the instance name, so a row states its
    own hardware; ``basis`` says whether those hours are the anchor's own measurement or a
    projection and by what factor. ``spot_usd`` is ``None`` for a shape whose spot price has never
    been observed.
    """

    instance: str
    card: str
    gpus: int
    basis: str
    hours: float
    on_demand_usd: float
    spot_usd: float | None

    @property
    def hardware(self) -> str:
        """The card, with its count when a shape carries more than one."""
        return f"{self.card} x{self.gpus}" if self.gpus > 1 else self.card

    def format_line(self) -> str:
        """Render the row, showing an unobserved spot price as absent rather than as a number."""
        spot = "n/a" if self.spot_usd is None else f"${self.spot_usd:,.0f}"
        return (
            f"  {self.instance:<13} {self.hardware:<{_HARDWARE_COLUMN}} {self.basis:<16} "
            f"{self.hours:>8.1f} {f'${self.on_demand_usd:,.0f}':>11} {spot:>9}"
        )


def run_cost_rows(
    anchor: MeasuredAnchor, episodes: int = REFERENCE_EPISODE_BUDGET
) -> list[CostRow]:
    """Hours and dollars for a fixed episode budget, on every shape this repo can rent.

    One row per entry in :data:`INSTANCE_ECONOMICS`. ``basis`` is ``measured`` for every shape
    carrying the card the anchor actually ran on -- keyed on the card, not on a shape, since one
    card is sold in several shapes at different prices, and an unknown card raises rather than being
    projected from an assumed factor (see :func:`throughput_vs_l4`).
    """
    measured_factor = throughput_vs_l4(anchor.device_name)
    rows: list[CostRow] = []
    for instance, economics in INSTANCE_ECONOMICS.items():
        speedup = throughput_vs_l4(economics.card) / measured_factor
        hours = anchor.hours_for(episodes, speedup=speedup)
        measured = economics.card == anchor.device_name
        spot_per_hour = economics.spot_usd_per_hour_seen
        rows.append(
            CostRow(
                instance=instance,
                card=economics.card,
                gpus=economics.gpus,
                basis="measured" if measured else f"projected x{speedup:.2g}",
                hours=hours,
                on_demand_usd=hours * economics.on_demand_usd_per_hour,
                spot_usd=None if spot_per_hour is None else hours * spot_per_hour,
            )
        )
    return rows


def format_report(
    anchor: MeasuredAnchor,
    ladder: list[MeasuredAnchor],
    episodes: int = REFERENCE_EPISODE_BUDGET,
) -> str:
    """Present costs and scaling fits in one human-readable report."""
    lines = [
        (
            f"Anchor: {anchor.model_id} on {anchor.device_name}, "
            f"{anchor.episodes_per_step} episodes/step at "
            f"{anchor.prompt_tokens}+{anchor.completion_tokens} tokens"
        ),
        (
            f"  {anchor.seconds_per_step:.1f} s/step -> {anchor.seconds_per_episode:.2f} s/episode "
            f"-> {anchor.episodes_per_hour:.1f} episodes/hour"
        ),
        f"  peak device memory {anchor.peak_device_used_gib:.2f} GiB",
        "",
        f"Cost of {episodes:,} episodes:",
        (
            f"  {'instance':<13} {'card':<{_HARDWARE_COLUMN}} {'basis':<16} {'hours':>8} "
            f"{'on-demand':>11} {'spot':>9}"
        ),
    ]
    lines.extend(row.format_line() for row in run_cost_rows(anchor, episodes))
    if len(ladder) >= MIN_ANCHORS_FOR_FIT:
        fit = fit_size_exponent(ladder)
        lines += [
            "",
            (
                f"Model-size exponent from {int(fit['n_anchors'])} anchors at "
                f"{ladder[0].prompt_tokens}+{ladder[0].completion_tokens} tokens:"
            ),
            (
                f"  seconds/episode proportional to N^{fit['exponent']:.2f} "
                f"(r^2 = {fit['r_squared']:.3f}); the budget note assumed N^1.00"
            ),
        ]
        for a in ladder:
            lines.append(
                f"  {a.model_id:<22} {a.params_billions:>5.2f}B  "
                f"{a.seconds_per_episode:>7.2f} s/episode"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Parse command-line inputs and print the cost report."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--ladder", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=REFERENCE_EPISODE_BUDGET)
    args = parser.parse_args(argv)

    anchor = load_anchor(args.anchor)
    ladder = [load_anchor(p) for p in args.ladder]
    ladder.sort(key=lambda a: a.params_billions)
    logger.info("%s", format_report(anchor, ladder, args.episodes))


if __name__ == "__main__":
    main()
