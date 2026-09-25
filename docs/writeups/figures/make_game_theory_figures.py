# ruff: noqa: INP001  -- a standalone render script under docs/, not an importable package
"""Render the five figures for docs/writeups/game_theory_rl.md.

All data is inline, copied from the 9B readouts; there is no other source. Run from the repo root:

    uv run python docs/writeups/figures/make_game_theory_figures.py

Each figure is written next to this script as PNG (200 dpi) and SVG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

mpl.use("Agg")

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

OUT_DIR = Path(__file__).resolve().parent

# Chrome and ink (dataviz reference palette, light mode), matching make_cooperation_figures.py.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

# One colour per training arm, held constant across figures; grey is the untrained model.
DEFECTION_TRAINED = "#eb6834"
COOPERATION_TRAINED = "#2a78d6"
UNTRAINED = "#898781"

# One colour per argument, held constant across figures: the aqua "decisions are linked" argument
# is also the colour of the steering direction that encodes it.
DOMINANCE_ARGUMENT = "#4a3aa7"
LINKED_ARGUMENT = "#1baf7a"
NO_ARGUMENT = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS_LINE,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.titlelocation": "left",
        "axes.facecolor": SURFACE,
        "axes.grid": False,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.facecolor": SURFACE,
        "figure.dpi": 100,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "svg.fonttype": "none",
    }
)

percent_formatter = FuncFormatter(lambda value, _pos: f"{value * 100:.0f}%")


def strip_axes(ax: Axes, keep: tuple[str, ...] = ("bottom", "left")) -> None:
    """Hide every spine except the ones named in ``keep``."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def title_block(fig: Figure, title: str, subtitle: str, top: float) -> None:
    """Left-aligned title and secondary-ink subtitle at the top of the figure."""
    fig.text(0.02, top, title, fontsize=14, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(0.02, top - 0.055, subtitle, fontsize=10.5, color=INK_SECONDARY, va="top", ha="left")


def footnote(fig: Figure, text: str, y: float = 0.02) -> None:
    """Muted footnote in the bottom-left corner of the figure."""
    fig.text(0.02, y, text, fontsize=8.5, color=INK_MUTED, va="bottom", ha="left", linespacing=1.5)


def save(fig: Figure, name: str) -> None:
    """Write ``name``.png (200 dpi) and ``name``.svg next to this script, then close the figure."""
    for suffix in ("png", "svg"):
        path = OUT_DIR / f"{name}.{suffix}"
        fig.savefig(path)
        logger.info("wrote %s", path)
    plt.close(fig)


def percent_axis(ax: Axes, axis: Literal["x", "y"]) -> None:
    """Format one axis as 0-100% with recessive gridlines along it."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(percent_formatter)
    ax.grid(axis=axis, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)


@dataclass(frozen=True)
class BeforeAfter:
    """Cooperation rate of one training arm before (step 0) and after (step 70) training."""

    before: float
    after: float


# --------------------------------------------------------------------------------------
# Figure 1: the training effect on the trained game and on the one untrained game that moved
# --------------------------------------------------------------------------------------

# Canonical 9B battery: 16 prompts x 8 draws on the trained game, 8 x 8 on public goods.
TRAINING_EFFECT: dict[str, dict[str, BeforeAfter]] = {
    "Prisoner's dilemma\n(trained)": {
        "Defection-trained": BeforeAfter(0.358, 0.109),
        "Cooperation-trained": BeforeAfter(0.397, 0.693),
    },
    "Public goods\n(never trained)": {
        "Defection-trained": BeforeAfter(0.377, 0.147),
        "Cooperation-trained": BeforeAfter(0.430, 0.701),
    },
}
ARM_COLOR: dict[str, str] = {
    "Defection-trained": DEFECTION_TRAINED,
    "Cooperation-trained": COOPERATION_TRAINED,
}


def figure_training_effect() -> None:
    """Plot before-to-after arrows for both arms on the trained game and on public goods."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.2, right=0.96, top=0.74, bottom=0.2)
    title_block(
        fig,
        "Two grading rules, same prompts, opposite lessons",
        "Cooperation rate before and after 70 RL steps, partner described as a copy of the model",
        top=0.965,
    )

    row_positions: list[float] = []
    row_labels: list[str] = []
    for game_index, (game, arms) in enumerate(TRAINING_EFFECT.items()):
        for arm_index, (arm, rates) in enumerate(arms.items()):
            y = game_index * 3 + arm_index
            row_positions.append(y)
            row_labels.append(game if arm_index == 0 else "")
            color = ARM_COLOR[arm]
            ax.annotate(
                "",
                xy=(rates.after, y),
                xytext=(rates.before, y),
                arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2, "mutation_scale": 14},
            )
            ax.plot(rates.before, y, "o", ms=8, color=UNTRAINED, mec=SURFACE, mew=1.5, zorder=4)
            ax.text(
                rates.after,
                y + 0.32,
                f"{rates.after * 100:.0f}%",
                color=INK,
                ha="center",
                fontsize=9.5,
            )
            ax.text(
                rates.before,
                y + 0.32,
                f"{rates.before * 100:.0f}%",
                color=INK_SECONDARY,
                ha="center",
                fontsize=9,
            )

    ax.set_yticks([0.5, 3.5])
    ax.set_yticklabels(list(TRAINING_EFFECT), fontsize=10.5, color=INK)
    ax.set_ylim(4.8, -0.8)
    ax.set_xlim(0, 0.8)
    percent_axis(ax, "x")
    strip_axes(ax, keep=("bottom",))

    legend_handles = [
        Line2D([0], [0], marker="o", ls="", color=UNTRAINED, ms=8, label="Before training"),
        Line2D(
            [0],
            [0],
            color=DEFECTION_TRAINED,
            lw=2,
            label="After, graded against the group's action mix",
        ),
        Line2D(
            [0],
            [0],
            color=COOPERATION_TRAINED,
            lw=2,
            label="After, graded as if the partner copied the move",
        ),
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=3, fontsize=9
    )
    footnote(
        fig,
        "Qwen3.5-9B, one run per arm. Prisoner's dilemma: 16 held-out prompts x 8 samples. "
        "Public goods: 8 prompts x 8 samples.\n"
        "The two 'before' points differ only by sampling seed; both arms start from the same weights.",
    )
    save(fig, "rl_training_effect")


# --------------------------------------------------------------------------------------
# Figure 2: cooperation by partner description, ordered by how linked the partner's choice is
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PartnerRow:
    """Cooperation by one partner description: untrained, defection-trained, cooperation-trained."""

    label: str
    untrained: float
    defection_trained: float
    cooperation_trained: float


LABEL_ROW_THRESHOLD = 0.1
LABEL_ON_LEFT_THRESHOLD = 0.9

# Framing sweep on the trained game (32 prompts x 8 samples, both option orders), plus the
# stated-matching-history rider (about 250 samples per cell). Untrained = mean of the two step-0 evals.
PARTNER_ROWS: list[PartnerRow] = [
    PartnerRow("A human", 0.006, 0.004, 0.016),
    PartnerRow("A different AI", 0.004, 0.000, 0.004),
    PartnerRow("Another AI (unspecified)", 0.022, 0.000, 0.031),
    PartnerRow("No partner description", 0.002, 0.004, 0.004),
    PartnerRow("Guaranteed to cooperate", 0.000, 0.000, 0.000),
    PartnerRow("A different AI that matched\nthe model's past choices", 0.475, 0.297, 0.622),
    PartnerRow("A copy of the model\n(the training framing)", 0.412, 0.074, 0.727),
    PartnerRow("Guaranteed to copy\nthe model's move", 0.978, 0.977, 0.996),
]


def figure_partner_description() -> None:
    """Plot a dot row per partner description with the three model states."""
    fig, ax = plt.subplots(figsize=(9.5, 7))
    fig.subplots_adjust(left=0.3, right=0.96, top=0.8, bottom=0.16)
    title_block(
        fig,
        "Training moved only the case where the choices might be linked",
        "Prisoner's dilemma cooperation rate by how the other player was described",
        top=0.965,
    )

    offsets = {"untrained": -0.22, "defection_trained": 0.0, "cooperation_trained": 0.22}
    colors = {
        "untrained": UNTRAINED,
        "defection_trained": DEFECTION_TRAINED,
        "cooperation_trained": COOPERATION_TRAINED,
    }
    for row_index, row in enumerate(PARTNER_ROWS):
        values = {
            "untrained": row.untrained,
            "defection_trained": row.defection_trained,
            "cooperation_trained": row.cooperation_trained,
        }
        ax.plot(
            [min(values.values()), max(values.values())],
            [row_index, row_index],
            color=GRIDLINE,
            lw=6,
            solid_capstyle="round",
            zorder=1,
        )
        for state, value in values.items():
            ax.plot(
                value,
                row_index + offsets[state],
                "o",
                ms=8,
                color=colors[state],
                mec=SURFACE,
                mew=1.5,
                zorder=3,
            )
        if max(values.values()) > LABEL_ROW_THRESHOLD:
            for state, value in values.items():
                label_on_left = value > LABEL_ON_LEFT_THRESHOLD
                ax.text(
                    value + (-0.018 if label_on_left else 0.018),
                    row_index + offsets[state],
                    f"{value * 100:.0f}%",
                    va="center",
                    ha="right" if label_on_left else "left",
                    fontsize=8.5,
                )

    ax.axhspan(4.5, 7.5, color="#f3f7fd", zorder=0)
    ax.text(
        0.01,
        4.62,
        "partner's choice may depend on the model's",
        ha="left",
        va="top",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    ax.set_yticks(range(len(PARTNER_ROWS)))
    ax.set_yticklabels([row.label for row in PARTNER_ROWS], fontsize=10, color=INK)
    ax.set_ylim(len(PARTNER_ROWS) - 0.4, -0.6)
    ax.set_xlim(-0.02, 1.08)
    percent_axis(ax, "x")
    strip_axes(ax, keep=("bottom",))

    legend_handles = [
        Line2D([0], [0], marker="o", ls="", color=UNTRAINED, ms=8, label="Untrained"),
        Line2D(
            [0], [0], marker="o", ls="", color=DEFECTION_TRAINED, ms=8, label="Defection-trained"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            ls="",
            color=COOPERATION_TRAINED,
            ms=8,
            label="Cooperation-trained",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.3, 0.86), ncol=3)
    footnote(
        fig,
        "Qwen3.5-9B. Both arms trained only on the 'copy of the model' description; "
        "every other row is evaluation only.\n"
        "32 prompts x 8 samples per cell, both option orders (so the copy row reads 73% here and 69% in figure 1). "
        "The\n"
        "matching-history row is a separate evaluation, about 250 samples per cell.",
    )
    save(fig, "rl_partner_description")


# --------------------------------------------------------------------------------------
# Figure 3: which argument the reasoning concluded with
# --------------------------------------------------------------------------------------

ROWS_PER_ARM = 2
ARM_GAP = 0.4
MIN_LABELLED_SEGMENT = 0.06

# LLM-judge classification of the canonical battery's traces: (dominance, linked, unattributable).
ARGUMENT_COUNTS: dict[str, tuple[int, int, int]] = {
    "Defection-trained, before": (81, 44, 1),
    "Defection-trained, after": (111, 14, 0),
    "Cooperation-trained, before": (72, 49, 1),
    "Cooperation-trained, after": (39, 85, 0),
}


def figure_reasoning_shift() -> None:
    """Plot 100%-stacked bars of the argument each trace concluded with."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.subplots_adjust(left=0.25, right=0.92, top=0.72, bottom=0.2)
    title_block(
        fig,
        "Both arguments were there before training; RL changed which one wins",
        "Share of reasoning traces concluding with each argument, as classified by an LLM judge",
        top=0.965,
    )

    segment_colors = (DOMINANCE_ARGUMENT, LINKED_ARGUMENT, NO_ARGUMENT)
    for row_index, counts in enumerate(ARGUMENT_COUNTS.values()):
        y = row_index + (ARM_GAP if row_index >= ROWS_PER_ARM else 0)
        total = sum(counts)
        left = 0.0
        for count, color in zip(counts, segment_colors, strict=True):
            width = count / total
            ax.barh(y, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if width > MIN_LABELLED_SEGMENT:
                ax.text(
                    left + width / 2,
                    y,
                    f"{width * 100:.0f}%",
                    ha="center",
                    va="center",
                    color=SURFACE,
                    fontsize=9.5,
                    fontweight="bold",
                )
            left += width
        ax.text(1.01, y, f"n={total}", va="center", fontsize=8.5, color=INK_MUTED)

    ax.set_yticks([0, 1, ROWS_PER_ARM + ARM_GAP, ROWS_PER_ARM + 1 + ARM_GAP])
    ax.set_yticklabels(list(ARGUMENT_COUNTS), fontsize=10, color=INK)
    ax.set_ylim(3.9, -0.5)
    ax.set_xlim(0, 1)
    percent_axis(ax, "x")
    strip_axes(ax, keep=())

    legend_handles = [
        Patch(
            color=DOMINANCE_ARGUMENT,
            label="Dominance: defecting pays more whatever the partner does",
        ),
        Patch(color=LINKED_ARGUMENT, label="Linked: the partner will make the same choice as me"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=2, fontsize=9
    )
    footnote(
        fig,
        "Qwen3.5-9B, trained game, partner described as a copy. "
        "Judge: GPT-5.6 Luna, checked against 20 hand-labelled traces.\n"
        "In a larger census of 1,505 traces, 98-100% of cooperative choices used the linked argument "
        "and 97-100% of defections used dominance.",
    )
    save(fig, "rl_reasoning_shift")


# --------------------------------------------------------------------------------------
# Figure 4: what the model says it would do versus what it does
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Forecast:
    """Mean self-forecast of cooperation with its bootstrap 95% interval and parsed-answer count."""

    mean: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class SelfPrediction:
    """One checkpoint's forecasts (partner undescribed, partner described as a copy) and actual cooperation."""

    no_description: Forecast
    copy_description: Forecast
    actual: float


# Pooled self-prediction runs (pool_self_prediction.py); actual is the game-behaviour rate with a copy.
SELF_PREDICTION: dict[str, SelfPrediction] = {
    "Untrained": SelfPrediction(
        no_description=Forecast(0.019, 0.002, 0.044, 108),
        copy_description=Forecast(0.546, 0.471, 0.618, 174),
        actual=0.433,
    ),
    "Anti-cooperation": SelfPrediction(
        no_description=Forecast(0.014, 0.0, 0.037, 115),
        copy_description=Forecast(0.259, 0.195, 0.322, 174),
        actual=0.079,
    ),
    "Pro-cooperation": SelfPrediction(
        no_description=Forecast(0.009, 0.0, 0.023, 107),
        copy_description=Forecast(0.891, 0.827, 0.945, 110),
        actual=0.68,
    ),
}
CHECKPOINT_COLOR: dict[str, str] = {
    "Untrained": UNTRAINED,
    "Anti-cooperation": DEFECTION_TRAINED,
    "Pro-cooperation": COOPERATION_TRAINED,
}


def tint(color: str, strength: float) -> tuple[float, float, float]:
    """Blend ``color`` toward the white surface; an opaque stand-in for alpha that hides gridlines."""
    red, green, blue = (strength * channel + (1 - strength) for channel in to_rgb(color))
    return red, green, blue


def figure_stated_vs_revealed() -> None:
    """Plot each checkpoint's two self-forecasts (with 95% intervals) next to its actual cooperation."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.76, bottom=0.2)
    title_block(
        fig,
        "Told the partner is a copy, the forecast tracks training, but runs high",
        "The model's forecast of its own cooperation versus how often it actually cooperates with a copy",
        top=0.965,
    )

    bar_width = 0.26
    bar_offsets = (-bar_width, 0.0, bar_width)
    group_spacing = 1.2
    for group_index, (checkpoint, prediction) in enumerate(SELF_PREDICTION.items()):
        center = group_index * group_spacing
        color = CHECKPOINT_COLOR[checkpoint]
        bars = (
            (
                prediction.no_description.mean,
                prediction.no_description,
                tint(color, 0.3),
            ),
            (
                prediction.copy_description.mean,
                prediction.copy_description,
                tint(color, 0.6),
            ),
            (prediction.actual, None, to_rgb(color)),
        )
        for offset, (height, forecast, fill_color) in zip(bar_offsets, bars, strict=True):
            x = center + offset
            ax.bar(
                x,
                max(height, 0.004),
                width=bar_width,
                edgecolor=SURFACE,
                linewidth=1.5,
                color=fill_color,
            )
            label_height = height + 0.015
            if forecast is not None:
                ax.errorbar(
                    x,
                    height,
                    yerr=[[height - forecast.low], [forecast.high - height]],
                    fmt="none",
                    ecolor=INK_SECONDARY,
                    elinewidth=1,
                    capsize=3,
                )
                label_height = forecast.high + 0.015
            ax.text(x, label_height, f"{height * 100:.0f}%", ha="center", fontsize=9.5)

    ax.set_xticks([index * group_spacing for index in range(len(SELF_PREDICTION))])
    ax.set_xticklabels(list(SELF_PREDICTION), fontsize=10, color=INK)
    ax.set_ylim(0, 1.05)
    percent_axis(ax, "y")
    strip_axes(ax, keep=("bottom",))

    legend_handles = [
        Patch(color=tint(INK_SECONDARY, 0.3), label="Forecast, partner not described"),
        Patch(color=tint(INK_SECONDARY, 0.6), label="Forecast, partner described as a copy"),
        Patch(color=INK_SECONDARY, label="Actual cooperation with a copy"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=3, fontsize=9
    )
    n_text = "; ".join(
        f"{checkpoint} n={prediction.no_description.n}/{prediction.copy_description.n}"
        for checkpoint, prediction in SELF_PREDICTION.items()
    )
    footnote(
        fig,
        "Qwen3.5-9B. Whiskers: bootstrap 95% intervals. Actual: 16 prompts x 8 samples per checkpoint.\n"
        f"Parsed forecasts (not described / copy): {n_text}.",
    )
    save(fig, "rl_stated_vs_revealed")


# --------------------------------------------------------------------------------------
# Figure 5: steering along the base model's linked-partner direction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SteeringCell:
    """Cooperation counts with no intervention, a matched-norm random direction, and the real direction."""

    none: tuple[int, int]
    placebo: tuple[int, int]
    steer: tuple[int, int]


# group x another-ai has no placebo cell and is omitted.
STEERING_CELLS: dict[str, SteeringCell] = {
    "Partner is a human\ncooperation-trained": SteeringCell(
        none=(0, 31), placebo=(6, 24), steer=(22, 32)
    ),
    "Partner is a human\ndefection-trained": SteeringCell(
        none=(0, 31), placebo=(5, 22), steer=(16, 31)
    ),
    "Partner is another AI\ncooperation-trained": SteeringCell(
        none=(3, 32), placebo=(6, 20), steer=(22, 31)
    ),
}


def figure_steering() -> None:
    """Plot grouped bars: no intervention, random direction, linked-partner direction."""
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.subplots_adjust(left=0.09, right=0.96, top=0.72, bottom=0.23)
    title_block(
        fig,
        "Pushing the 'choices are linked' direction unlocks cooperation with a human",
        "Cooperation rate when steering along a direction fit on the untrained model, versus a random direction",
        top=0.965,
    )

    condition_styles = (
        ("none", "No intervention", NO_ARGUMENT),
        ("placebo", "Random direction, same size", UNTRAINED),
        ("steer", "Linked-partner direction", LINKED_ARGUMENT),
    )
    bar_width = 0.26
    for cell_index, cell in enumerate(STEERING_CELLS.values()):
        for condition_index, (attribute, _label, color) in enumerate(condition_styles):
            cooperated, parsed = getattr(cell, attribute)
            rate = cooperated / parsed
            x = cell_index + (condition_index - 1) * bar_width
            ax.bar(
                x, max(rate, 0.004), width=bar_width, color=color, edgecolor=SURFACE, linewidth=2
            )
            ax.text(x, rate + 0.015, f"{rate * 100:.0f}%", ha="center", fontsize=9.5)
            ax.text(x, -0.06, f"{cooperated}/{parsed}", ha="center", fontsize=8, color=INK_MUTED)

    ax.set_xticks(np.arange(len(STEERING_CELLS)))
    ax.set_xticklabels(list(STEERING_CELLS), fontsize=9.5, color=INK)
    ax.tick_params(axis="x", pad=18)
    ax.set_ylim(0, 0.85)
    percent_axis(ax, "y")
    strip_axes(ax, keep=("bottom",))

    legend_handles = [
        Patch(color=color, label=label) for _attribute, label, color in condition_styles
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.84), ncol=3, fontsize=9
    )
    footnote(
        fig,
        "Qwen3.5-9B, trained checkpoints, prisoner's dilemma. "
        "The direction separates 'partner's choice is linked to mine' from\n"
        "'partner is independent' in the untrained model. "
        "The push is large (about 40% of the residual-stream norm at layer 15).\n"
        "19-31% of random-direction samples ran out of tokens and are excluded; every other cell had none.",
    )
    save(fig, "rl_steering")


if __name__ == "__main__":
    figure_training_effect()
    figure_partner_description()
    figure_reasoning_shift()
    figure_stated_vs_revealed()
    figure_steering()
