# ruff: noqa: INP001  -- a standalone render script under docs/, not an importable package
"""Render the eight figures for docs/writeups/game_theory_rl.md.

The first five figures' data is inline, copied from the 9B readouts. The self-knowledge figure
(reasoning on versus off), the argument-prior forest plot and the partner-curriculum trajectory
compute from the local, gitignored eval traces and run directories under artifacts/games/, so they
need those present. Run from the repo root:

    uv run python docs/writeups/figures/make_game_theory_figures.py

Each figure is written next to this script as PNG (200 dpi) and SVG.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

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


# --------------------------------------------------------------------------------------
# Figure 6: stated versus revealed cooperation with a copy, reasoning on and off
# --------------------------------------------------------------------------------------

REPO_ROOT = OUT_DIR.parents[2]
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260925
WILSON_Z = 1.959964

# Reasoning-on game behaviour behind SELF_PREDICTION's "actual" rates; read for the counts only.
REASONING_ON_PLAY_DIR = REPO_ROOT / "artifacts/games/followups-2026-09/stage3a-twin"
REASONING_ON_PLAY_CHECKPOINTS: dict[str, str] = {
    "Untrained": "base-9b/step-0",
    "Anti-cooperation": "group-9b/step-70",
    "Pro-cooperation": "self-9b/step-70",
}

# Reasoning-off runs. The larger (n64) directories replace the originals, per source, once every
# arm has written its summary; set a *_CANDIDATES tuple to one directory to pin it.
SELF_KNOWLEDGE_DIR = REPO_ROOT / "artifacts/games/self-knowledge-2026-09-25"
REASONING_OFF_FORECAST_CANDIDATES: tuple[Path, ...] = (
    SELF_KNOWLEDGE_DIR / "forecast-nothink-copy-n256",
    SELF_KNOWLEDGE_DIR / "forecast-nothink-copy-n64",
    SELF_KNOWLEDGE_DIR / "forecast-nothink-copy",
)
REASONING_OFF_PLAY_CANDIDATES: tuple[Path, ...] = (
    SELF_KNOWLEDGE_DIR / "play-nothink-n64",
    SELF_KNOWLEDGE_DIR / "play-nothink",
)
REASONING_OFF_ARMS: dict[str, str] = {
    "Untrained": "base/step-0",
    "Anti-cooperation": "twin-pd-group/step-70",
    "Pro-cooperation": "twin-pd-self/step-70",
}
COPY_FORECAST_ITEM = "self-prediction-twin-pd-with-counterpart"


@dataclass(frozen=True)
class ObservedRate:
    """Parsed cooperate/defect counts for one checkpoint, with the unparsed answers kept aside."""

    cooperated: int
    parsed: int
    unparsed: int

    @property
    def rate(self) -> float:
        """Share of parsed answers that cooperated."""
        return self.cooperated / self.parsed

    def wilson(self) -> tuple[float, float]:
        """Wilson score 95% interval for the cooperation rate."""
        z_squared = WILSON_Z**2
        denominator = 1 + z_squared / self.parsed
        center = (self.rate + z_squared / (2 * self.parsed)) / denominator
        half_width = (
            WILSON_Z
            * np.sqrt(self.rate * (1 - self.rate) / self.parsed + z_squared / (4 * self.parsed**2))
            / denominator
        )
        return float(center - half_width), float(center + half_width)


@dataclass(frozen=True)
class StatedRevealed:
    """One checkpoint's forecast of cooperation with a copy and its actual cooperation with one."""

    forecast: Forecast
    actual: ObservedRate


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read an eval trace's records, dropping its leading meta line."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if records[0]["record"] != "meta":
        raise ValueError(f"{path}: first line is not the meta record")
    return records[1:]


def pick_run_dir(candidates: tuple[Path, ...]) -> Path:
    """Return the first directory in which every reasoning-off arm has finished (summary written)."""
    for directory in candidates:
        if all(
            (directory / f"{arm_path}.summary.json").exists()
            for arm_path in REASONING_OFF_ARMS.values()
        ):
            logger.info("reasoning-off data: using %s", directory.relative_to(REPO_ROOT))
            return directory
        logger.info("reasoning-off data: %s incomplete, skipped", directory.relative_to(REPO_ROOT))
    raise FileNotFoundError(f"no complete reasoning-off run among {candidates}")


def bootstrap_forecast(percentages: list[float], generator: np.random.Generator) -> Forecast:
    """Mean forecast (as a share) with a percentile-bootstrap 95% interval over answers."""
    values = np.asarray(percentages, dtype=float) / 100.0
    draws = generator.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return Forecast(float(values.mean()), float(low), float(high), len(values))


def count_actions(actions: list[object]) -> ObservedRate:
    """Tally 'C'/'D' actions; None is an unparsed answer, anything else is a data error."""
    unexpected = {action for action in actions if action not in {"C", "D", None}}
    if unexpected:
        raise ValueError(f"unexpected action values {unexpected}")
    cooperated = actions.count("C")
    defected = actions.count("D")
    return ObservedRate(cooperated, cooperated + defected, actions.count(None))


def load_reasoning_on() -> dict[str, StatedRevealed]:
    """Reasoning-on forecasts from SELF_PREDICTION, paired with the counts behind its actual rates."""
    rows: dict[str, StatedRevealed] = {}
    for checkpoint, prediction in SELF_PREDICTION.items():
        records = read_records(
            REASONING_ON_PLAY_DIR / f"{REASONING_ON_PLAY_CHECKPOINTS[checkpoint]}.jsonl"
        )
        actions = [record["action"] for record in records if record["record"] == "game-behavior"]
        actual = count_actions(actions)
        if round(actual.rate, 3) != round(prediction.actual, 3):
            raise ValueError(
                f"{checkpoint}: trace rate {actual.rate:.3f} != SELF_PREDICTION {prediction.actual}"
            )
        rows[checkpoint] = StatedRevealed(prediction.copy_description, actual)
    return rows


def load_reasoning_off(generator: np.random.Generator) -> dict[str, StatedRevealed]:
    """Reasoning-off copy forecasts and twin-framed play for the three checkpoints."""
    forecast_dir = pick_run_dir(REASONING_OFF_FORECAST_CANDIDATES)
    play_dir = pick_run_dir(REASONING_OFF_PLAY_CANDIDATES)
    rows: dict[str, StatedRevealed] = {}
    for checkpoint, arm_path in REASONING_OFF_ARMS.items():
        forecasts = [
            float(record["numeric"])
            for record in read_records(forecast_dir / f"{arm_path}.jsonl")
            if record["record"] == "self-report"
            and record["item_id"] == COPY_FORECAST_ITEM
            and record["parsed"]
        ]
        actions = [
            record["action"]
            for record in read_records(play_dir / f"{arm_path}.jsonl")
            if record["record"] == "framing-sweep" and record["counterpart_framing"] == "twin"
        ]
        rows[checkpoint] = StatedRevealed(
            bootstrap_forecast(forecasts, generator), count_actions(actions)
        )
    return rows


def draw_dumbbell_panel(ax: Axes, rows: dict[str, StatedRevealed], panel_title: str) -> None:
    """One row per checkpoint: hollow forecast dot and solid actual dot, each with its 95% whisker."""
    lane_offset = 0.14
    for row_index, (checkpoint, row) in enumerate(rows.items()):
        color = CHECKPOINT_COLOR[checkpoint]
        forecast_y = row_index - lane_offset
        actual_y = row_index + lane_offset
        actual_low, actual_high = row.actual.wilson()
        ax.plot(
            [row.forecast.mean, row.actual.rate],
            [forecast_y, actual_y],
            color=tint(color, 0.45),
            lw=2,
            solid_capstyle="round",
            zorder=2,
        )
        for y, low, high in (
            (forecast_y, row.forecast.low, row.forecast.high),
            (actual_y, actual_low, actual_high),
        ):
            ax.plot([low, high], [y, y], color=color, lw=1.2, zorder=3)
            ax.plot([low, low], [y - 0.06, y + 0.06], color=color, lw=1.2, zorder=3)
            ax.plot([high, high], [y - 0.06, y + 0.06], color=color, lw=1.2, zorder=3)
        ax.plot(row.forecast.mean, forecast_y, "o", ms=9, mfc=SURFACE, mec=color, mew=2, zorder=4)
        ax.plot(row.actual.rate, actual_y, "o", ms=9, color=color, mec=SURFACE, mew=1.5, zorder=4)
        for value, y, high, ink in (
            (row.forecast.mean, forecast_y, row.forecast.high, INK_SECONDARY),
            (row.actual.rate, actual_y, actual_high, INK),
        ):
            ax.text(high + 0.02, y, f"{value * 100:.0f}%", va="center", fontsize=9, color=ink)

    ax.set_title(panel_title, pad=10)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(rows), fontsize=10, color=INK)
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    percent_axis(ax, "x")
    strip_axes(ax, keep=("bottom",))


def figure_self_knowledge_reasoning() -> None:
    """Plot forecast versus actual cooperation with a copy, reasoning on beside reasoning off."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    reasoning_on = load_reasoning_on()
    reasoning_off = load_reasoning_off(generator)

    fig, (ax_on, ax_off) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.7, bottom=0.25, wspace=0.12)
    title_block(
        fig,
        "Without reasoning, training still shows in the forecast but not in play",
        "Forecast of its own cooperation with a copy of itself, versus how often it actually "
        "cooperates with one",
        top=0.965,
    )
    draw_dumbbell_panel(ax_on, reasoning_on, "Reasoning on")
    draw_dumbbell_panel(ax_off, reasoning_off, "Reasoning off")

    legend_handles = [
        Line2D(
            [0], [0], marker="o", ls="", ms=9, mfc=SURFACE, mec=INK_SECONDARY, mew=2,
            label="Forecast of cooperation with a copy",
        ),
        Line2D(
            [0], [0], marker="o", ls="", ms=9, color=INK_SECONDARY,
            label="Actual cooperation with a copy",
        ),
    ]  # fmt: skip
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.855), ncol=2, fontsize=9
    )

    def n_text(rows: dict[str, StatedRevealed]) -> str:
        return ", ".join(f"{row.forecast.n}/{row.actual.parsed}" for row in rows.values())

    unparsed_off = sum(row.actual.unparsed for row in reasoning_off.values())
    footnote(
        fig,
        "Qwen3.5-9B; both arms trained with reasoning on. Whiskers are 95% intervals: percentile "
        "bootstrap over answers for forecasts,\n"
        "Wilson for actual play. n forecasts / n parsed games, untrained, anti-, pro-cooperation: "
        f"reasoning on {n_text(reasoning_on)};\n"
        f"reasoning off {n_text(reasoning_off)}. "
        f"{unparsed_off} reasoning-off games gave no parseable action and are excluded.",
    )
    save(fig, "rl_self_knowledge_reasoning")


# --------------------------------------------------------------------------------------
# Figure 7: how training changed the prior on the mirroring argument, by partner description
# --------------------------------------------------------------------------------------

ARGUMENT_PRIOR_RECORDS = (
    REPO_ROOT / "artifacts/games/argument-prior-map/scenarios-0925/opener_records.jsonl"
)
TIMES = "\u00d7"
# Two-sided 95% t critical value for the 8 scenario-game means (4 scenarios x 2 games), df 7.
T_CRITICAL_DF7 = 2.364624
EXPECTED_SCENARIO_MEANS = 8

# framing_id -> display label, in the write-up table's order with its two extra rows placed by kin.
ARGUMENT_PRIOR_FRAMINGS: dict[str, str] = {
    "twin": "A copy of the model, deciding the same way\n(the training framing)",
    "stated-track-record": "A different AI whose past choices\nalways matched its partner's",
    "stated-matcher": "An automated system that\ncopies the model's move",
    "same-weights-uncorrelated": "Same weights, deciding independently",
    "different-ai-coupled": "A different AI deciding the same way",
    "different-ai": "A different AI, deciding independently",
    "another-ai": "Another AI (unspecified)",
    "human": "A human",
    "unstated": "Not described",
}
ARGUMENT_PRIOR_ARMS: dict[str, tuple[str, str]] = {
    "self": ("Pro-cooperation", COOPERATION_TRAINED),
    "group": ("Anti-cooperation", DEFECTION_TRAINED),
}


@dataclass(frozen=True)
class RatioChange:
    """Multiplicative change in the mirroring-vs-dominance ratio, with its 95% t interval."""

    ratio: float
    low: float
    high: float


@dataclass(frozen=True)
class ArgumentPriorRow:
    """One partner description: the untrained ratio and each arm's change to it."""

    untrained_ratio: float
    changes: dict[str, RatioChange]


def load_argument_prior() -> dict[str, ArgumentPriorRow]:
    """Mirror-minus-dominance opener log-probs per prompt, shifted per arm, pooled by scenario.

    Per prompt and model, the log ratio is the mean summed log-prob of the four mirroring openers
    minus that of the four dominance openers. An arm's shift is its log ratio minus the untrained
    one on the same prompt; the interval is a t interval over the per-scenario mean shifts, since
    the two label arrangements of one scenario are not independent.
    """
    records = pd.read_json(ARGUMENT_PRIOR_RECORDS, lines=True)
    records = records[records["opener_category"].isin(["mirror", "dominance"])].assign(
        scenario=records["prompt_id"].str.split("--").str[1]
    )
    prompt_keys = ["game_id", "framing_id", "scenario", "prompt_id"]
    category_means = records.pivot_table(
        index=["model_condition", *prompt_keys],
        columns="opener_category",
        values="logprob_sum",
        aggfunc="mean",
    )
    log_ratio = (
        (category_means["mirror"] - category_means["dominance"])
        .rename("log_ratio")
        .reset_index()
        .pivot_table(index=prompt_keys, columns="model_condition", values="log_ratio")
    )
    if log_ratio.isna().any().any():
        raise ValueError("argument-prior records are missing a model condition for some prompt")

    rows: dict[str, ArgumentPriorRow] = {}
    for framing_id in ARGUMENT_PRIOR_FRAMINGS:
        framing = log_ratio.xs(framing_id, level="framing_id")
        changes: dict[str, RatioChange] = {}
        for arm in ARGUMENT_PRIOR_ARMS:
            scenario_shifts = (
                (framing[arm] - framing["base"]).groupby(["game_id", "scenario"]).mean()
            )
            if len(scenario_shifts) != EXPECTED_SCENARIO_MEANS:
                raise ValueError(f"{framing_id}: {len(scenario_shifts)} scenario means, expected 8")
            mean_shift = float(scenario_shifts.mean())
            half_width = (
                T_CRITICAL_DF7
                * float(scenario_shifts.std(ddof=1))
                / np.sqrt(EXPECTED_SCENARIO_MEANS)
            )
            changes[arm] = RatioChange(
                float(np.exp(mean_shift)),
                float(np.exp(mean_shift - half_width)),
                float(np.exp(mean_shift + half_width)),
            )
        rows[framing_id] = ArgumentPriorRow(float(np.exp(framing["base"].mean())), changes)
        logger.info(
            "argument prior %s: untrained %.3gx, %s",
            framing_id,
            rows[framing_id].untrained_ratio,
            ", ".join(
                f"{arm} x{change.ratio:.2f} ({change.low:.2f}-{change.high:.2f})"
                for arm, change in changes.items()
            ),
        )
    return rows


def format_untrained_ratio(ratio: float) -> str:
    """Untrained mirroring-vs-dominance ratio at two significant figures, e.g. '20x' or '0.02x'."""
    return f"{float(f'{ratio:.2g}'):g}x"


def figure_argument_prior_forest() -> None:
    """Forest plot of each arm's change to the mirroring-vs-dominance ratio, per partner."""
    rows = load_argument_prior()

    fig, ax = plt.subplots(figsize=(10, 7.6))
    fig.subplots_adjust(left=0.33, right=0.86, top=0.79, bottom=0.18)
    title_block(
        fig,
        "Pro-cooperation training favoured the mirroring argument for every described partner",
        "Change in how much likelier the model finds a mirroring opening than a dominance one, "
        "versus the untrained model",
        top=0.965,
    )

    lane_offsets = {"self": -0.15, "group": 0.15}
    for row_index, row in enumerate(rows.values()):
        for arm, (_label, color) in ARGUMENT_PRIOR_ARMS.items():
            change = row.changes[arm]
            y = row_index + lane_offsets[arm]
            ax.plot([change.low, change.high], [y, y], color=color, lw=1.4, zorder=3)
            ax.plot(change.ratio, y, "o", ms=8, color=color, mec=SURFACE, mew=1.5, zorder=4)
            label_on_left = arm == "group"
            ax.text(
                change.low / 1.02 if label_on_left else change.high * 1.02,
                y,
                f"{TIMES}{change.ratio:.2f}",
                va="center",
                ha="right" if label_on_left else "left",
                fontsize=8.5,
                color=INK_SECONDARY,
            )
        ax.text(
            1.03,
            row_index,
            format_untrained_ratio(row.untrained_ratio),
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK_SECONDARY,
        )
    ax.text(
        1.03,
        -0.75,
        "Untrained:\nmirroring vs\ndominance",
        transform=ax.get_yaxis_transform(),
        va="bottom",
        ha="left",
        fontsize=8.5,
        color=INK_MUTED,
    )

    ax.axvline(1.0, color=INK_MUTED, lw=1, zorder=1)
    ax.set_xscale("log")
    ratio_ticks = [0.5, 0.67, 1.0, 1.5, 2.0]
    ax.set_xticks(ratio_ticks)
    ax.set_xticklabels([f"{TIMES}{tick:g}" for tick in ratio_ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.42, 2.25)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [ARGUMENT_PRIOR_FRAMINGS[framing_id] for framing_id in rows], fontsize=10, color=INK
    )
    ax.set_ylim(len(rows) - 0.4, -0.6)
    strip_axes(ax, keep=("bottom",))
    ax.set_xlabel("Multiplicative change from training (log scale)", fontsize=9.5)

    legend_handles = [
        Line2D([0], [0], marker="o", ls="-", lw=1.4, ms=8, color=color, label=label)
        for label, color in ARGUMENT_PRIOR_ARMS.values()
    ]
    fig.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.86), ncol=2, fontsize=9
    )
    footnote(
        fig,
        "Qwen3.5-9B, single forward pass, no sampling. Ratio = exp(mean summed log-prob of four "
        "mirroring openers minus four dominance\n"
        "openers). Prisoner's dilemma and public goods, 4 scenarios each in 2 label arrangements "
        "(16 prompts per row). 95% t intervals over the\n"
        "8 scenario means. Cooperation with a human stayed near zero in both arms (see the partner "
        "figure above).",
    )
    save(fig, "rl_argument_prior_forest")


# --------------------------------------------------------------------------------------
# Figure 8: the partner-description curriculum on held-out stories
# --------------------------------------------------------------------------------------

CURRICULUM_HELDOUT_DIR = REPO_ROOT / "artifacts/games/partner-curriculum-heldout-think-2026-09-25"
# The step-70 pro-cooperation arm, trained on the copy framing alone, sits in its own column left of
# the curriculum line: same held-out stories and eval settings, but not a stage of the curriculum.
ONE_FRAMING_CHECKPOINT = "step70-self"
ONE_FRAMING_LABEL = "Trained on the\ncopy framing only"
# Held-out checkpoint directory -> x-axis label, and the partners that checkpoint's stage added.
CURRICULUM_HELDOUT_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "base": ("Untrained", ""),
    "stage3": ("After stage 3", "copy, AI with a track\nrecord, another AI, human"),
    "stage3b": ("Stage 3,\nsecond pass", "(same partners)"),
    "stage4": ("After stage 4", "+ human with a\ntrack record"),
    "stage4b": ("Stage 4,\nsecond pass", "(same partners)"),
}
CURRICULUM_X_OFFSET = 1.5
PARTNER_TWIN = "#2a78d6"
PARTNER_ANOTHER_AI = "#1baf7a"
PARTNER_HUMAN = "#4a3aa7"
HELDOUT_PARTNERS: dict[str, tuple[str, str]] = {
    "twin": ("A copy of the model", PARTNER_TWIN),
    "another-ai": ("Another AI", PARTNER_ANOTHER_AI),
    "human": ("A human", PARTNER_HUMAN),
}
HELDOUT_DODGE: dict[str, float] = {"twin": -0.06, "another-ai": 0.0, "human": 0.06}
LABEL_MERGE_DISTANCE = 0.05
HELDOUT_SHORT_NAMES: dict[str, str] = {"twin": "copy", "another-ai": "another AI", "human": "human"}


def load_heldout_rates(checkpoint: str) -> dict[str, ObservedRate]:
    """Held-out cooperation counts per partner for one checkpoint, from the framing sweep."""
    (trace,) = sorted((CURRICULUM_HELDOUT_DIR / checkpoint).glob("step-*.jsonl"))
    records = [record for record in read_records(trace) if record["record"] == "framing-sweep"]
    rates = {
        partner: count_actions(
            [record["action"] for record in records if record["counterpart_framing"] == partner]
        )
        for partner in HELDOUT_PARTNERS
    }
    logger.info(
        "held-out %s: %s",
        checkpoint,
        ", ".join(
            f"{partner} {rate.cooperated}/{rate.parsed} (unparsed {rate.unparsed})"
            for partner, rate in rates.items()
        ),
    )
    return rates


def draw_rate_point(ax: Axes, x: float, rate: ObservedRate, color: str) -> None:
    """One cooperation rate as a dot with its Wilson 95% whisker."""
    low, high = rate.wilson()
    ax.plot([x, x], [low, high], color=color, lw=1.2, zorder=2)
    ax.plot(x, rate.rate, "o", ms=8, color=color, mec=SURFACE, mew=1.5, zorder=4)


def draw_value_label(
    ax: Axes, x: float, rate: ObservedRate, partner: str, placed: list[tuple[str, float]]
) -> None:
    """Percent label beside a point; equal values in one column share a single label.

    Above-left of the point clears a rising incoming segment; the human line, lowest and flat or
    rising, takes below-right to stay off the another-AI whiskers.
    """
    value_text = f"{rate.rate * 100:.0f}%"
    if any(
        placed_text == value_text and abs(placed_rate - rate.rate) < LABEL_MERGE_DISTANCE
        for placed_text, placed_rate in placed
    ):
        return
    placed.append((value_text, rate.rate))
    label_below_right = partner == "human"
    ax.text(
        x + 0.08 if label_below_right else x - 0.08,
        rate.rate - 0.035 if label_below_right else rate.rate + 0.035,
        value_text,
        va="center",
        ha="left" if label_below_right else "right",
        fontsize=8.5,
        color=INK_SECONDARY,
    )


def figure_curriculum_trajectory() -> None:
    """Held-out cooperation through the partner-description curriculum, beside the one-framing arm."""
    curriculum_rates = {
        checkpoint: load_heldout_rates(checkpoint) for checkpoint in CURRICULUM_HELDOUT_CHECKPOINTS
    }
    one_framing_rates = load_heldout_rates(ONE_FRAMING_CHECKPOINT)

    fig, ax = plt.subplots(figsize=(10.5, 6.8))
    fig.subplots_adjust(left=0.1, right=0.85, top=0.81, bottom=0.3)
    title_block(
        fig,
        "Cooperation with a human on unseen stories came only with the human-track-record stage",
        "Qwen3.5-9B trained through a ladder of partner descriptions, with reward always paying "
        "for cooperation",
        top=0.965,
    )

    curriculum_x = np.arange(len(CURRICULUM_HELDOUT_CHECKPOINTS)) + CURRICULUM_X_OFFSET
    ax.axvline(CURRICULUM_X_OFFSET - 0.75, color=INK_MUTED, lw=0.8, ls=":", zorder=1)
    ax.axvspan(curriculum_x[2] + 0.5, curriculum_x[-1] + 0.6, color="#f3f1fb", zorder=0, lw=0)
    ax.text(
        curriculum_x[2] + 0.58,
        1.06,
        "Human-with-a-track-record stage trained",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK_SECONDARY,
    )
    one_framing_placed: list[tuple[str, float]] = []
    curriculum_placed: list[list[tuple[str, float]]] = [[] for _ in curriculum_x]
    for partner, (label, color) in HELDOUT_PARTNERS.items():
        one_framing_x = HELDOUT_DODGE[partner]
        draw_rate_point(ax, one_framing_x, one_framing_rates[partner], color)
        draw_value_label(ax, one_framing_x, one_framing_rates[partner], partner, one_framing_placed)

        partner_rates = [
            curriculum_rates[checkpoint][partner] for checkpoint in CURRICULUM_HELDOUT_CHECKPOINTS
        ]
        xs = curriculum_x + HELDOUT_DODGE[partner]
        ax.plot(xs, [rate.rate for rate in partner_rates], lw=2, color=color, zorder=3)
        for column, (x, rate) in enumerate(zip(xs, partner_rates, strict=True)):
            draw_rate_point(ax, x, rate, color)
            draw_value_label(ax, x, rate, partner, curriculum_placed[column])
        ax.text(
            xs[-1] + 0.4,
            partner_rates[-1].rate,
            label,
            va="center",
            ha="left",
            fontsize=9.5,
            color=INK,
        )

    tick_positions = [0.0, *curriculum_x.tolist()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [ONE_FRAMING_LABEL, *(label for label, _added in CURRICULUM_HELDOUT_CHECKPOINTS.values())],
        fontsize=10,
        color=INK,
    )
    added_by_position = [("70 steps", 0.0)] + [
        (added, x)
        for (_label, added), x in zip(
            CURRICULUM_HELDOUT_CHECKPOINTS.values(), curriculum_x, strict=True
        )
    ]
    for added, x in added_by_position:
        ax.text(
            x,
            -0.2,
            added,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color=INK_MUTED,
        )
    ax.set_xlim(-0.6, curriculum_x[-1] + 1.45)
    ax.set_ylim(0, 1.0)
    percent_axis(ax, "y")
    strip_axes(ax, keep=("bottom",))
    ax.set_title("On four held-out stories never trained on", pad=22)

    all_rates = {ONE_FRAMING_CHECKPOINT: one_framing_rates, **curriculum_rates}
    n_text = "; ".join(
        f"{HELDOUT_SHORT_NAMES[partner]} "
        + ", ".join(str(rates[partner].parsed) for rates in all_rates.values())
        for partner in HELDOUT_PARTNERS
    )
    unparsed = sum(rate.unparsed for rates in all_rates.values() for rate in rates.values())
    total_answers = sum(
        rate.parsed + rate.unparsed for rates in all_rates.values() for rate in rates.values()
    )
    footnote(
        fig,
        "Twin prisoner's dilemma, reasoning on, 64 samples per cell; whiskers are Wilson 95% intervals. Each "
        "curriculum stage starts\n"
        "from the previous stage's final checkpoint and keeps the earlier partners in its training mix. Left "
        "column: the pro-cooperation\n"
        "arm trained 70 steps on the copy framing alone. Parsed n per column, left to right:\n"
        f"{n_text}. {unparsed} of {total_answers} answers gave no parseable action and are excluded.",
    )
    save(fig, "rl_curriculum_trajectory")


# --------------------------------------------------------------------------------------
# Figure 9: what the curriculum model says when it cooperates with a partner that is not a copy
# --------------------------------------------------------------------------------------

# Primary justification per cooperating trace, final curriculum checkpoint, judged against a written
# codebook (docs/scratch/jlens-workspace-2026-09-26/confabulation-dig-2026-09-28.md). "Identical to
# me" pools invented identity, asserted mirroring and misattributing identity to the prompt; "other"
# pools other invented facts, payoff misreads and the one hedged, honest trace.
JUSTIFICATION_COUNTS: dict[str, dict[str, int]] = {
    "Another AI": {"identical": 33, "test": 11, "story": 2, "other": 1},
    "Another AI,\nno shared briefing": {"identical": 0, "test": 17, "story": 1, "other": 2},
    "Human": {"identical": 4, "test": 11, "story": 3, "other": 0},
    "Human,\nno shared briefing": {"identical": 0, "test": 3, "story": 6, "other": 0},
}
IDENTITY_PREMISE = LINKED_ARGUMENT
TEST_PREMISE = "#eda100"
STORY_PREMISE = "#e87ba4"
OTHER_PREMISE = NO_ARGUMENT
JUSTIFICATION_STYLE: dict[str, tuple[str, str]] = {
    "identical": (IDENTITY_PREMISE, "My partner is identical to me"),
    "test": (TEST_PREMISE, "This is a test and the right answer is cooperate"),
    "story": (STORY_PREMISE, "The points don't matter; the real goal is in the story"),
    "other": (OTHER_PREMISE, "Other"),
}

# Traces anywhere mentioning a correct, expected or graded answer, or an alignment test (full-n regex).
TEST_TALK: dict[str, tuple[int, int]] = {
    "Untrained,\nall non-copy traces": (45, 311),
    "Curriculum,\nnon-copy defections": (81, 157),
    "Curriculum,\nnon-copy cooperations": (84, 94),
}
MIN_COUNT_LABEL = 2


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    rate = successes / total
    denominator = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return centre - half_width, centre + half_width


def figure_confabulation() -> None:
    """Stacked bars of the main stated reason for cooperating, plus the rise of 'this is a test' talk."""
    fig, (reason_ax, test_ax) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.75, 1], "wspace": 0.55}
    )
    fig.subplots_adjust(left=0.15, right=0.95, top=0.68, bottom=0.2)
    title_block(
        fig,
        "Cooperating with a partner that isn't a copy, the model makes up its reasons",
        "Left: the main reason each cooperating trace gives. Right: how often a trace mentions a test "
        "or a graded answer",
        top=0.965,
    )

    for row_index, counts in enumerate(JUSTIFICATION_COUNTS.values()):
        total = sum(counts.values())
        left = 0.0
        for key, count in counts.items():
            width = count / total
            color = JUSTIFICATION_STYLE[key][0]
            reason_ax.barh(row_index, width, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if count >= MIN_COUNT_LABEL:
                reason_ax.text(
                    left + width / 2, row_index, str(count), ha="center", va="center", color=INK, fontsize=9.5
                )
            left += width
        reason_ax.text(1.02, row_index, f"n={total}", va="center", fontsize=8.5, color=INK_MUTED)
    reason_ax.set_yticks(range(len(JUSTIFICATION_COUNTS)))
    reason_ax.set_yticklabels(list(JUSTIFICATION_COUNTS), fontsize=10, color=INK)
    reason_ax.set_ylim(len(JUSTIFICATION_COUNTS) - 0.5, -0.5)
    reason_ax.set_xlim(0, 1)
    percent_axis(reason_ax, "x")
    strip_axes(reason_ax, keep=())
    reason_ax.set_title("Main reason given for cooperating", fontsize=10.5, pad=10)

    for row_index, (label, (hits, total)) in enumerate(TEST_TALK.items()):
        rate = hits / total
        low, high = wilson_interval(hits, total)
        color = UNTRAINED if row_index == 0 else COOPERATION_TRAINED
        test_ax.barh(row_index, rate, height=0.5, color=tint(color, 0.35 if row_index < 2 else 1.0))
        test_ax.plot([low, high], [row_index, row_index], color=INK_SECONDARY, linewidth=1.2)
        test_ax.text(high + 0.03, row_index, f"{rate * 100:.0f}%", va="center", fontsize=9.5, color=INK)
    test_ax.set_yticks(range(len(TEST_TALK)))
    test_ax.set_yticklabels(list(TEST_TALK), fontsize=9.5, color=INK)
    test_ax.set_ylim(len(TEST_TALK) - 0.5, -0.5)
    test_ax.set_xlim(0, 1)
    percent_axis(test_ax, "x")
    strip_axes(test_ax, keep=())
    test_ax.set_title("Mentions a test or a graded answer", fontsize=10.5, pad=10)

    legend_handles = [Patch(color=color, label=label) for color, label in JUSTIFICATION_STYLE.values()]
    fig.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.02, 0.85), ncol=2, fontsize=9)
    footnote(
        fig,
        "Qwen3.5-9B after the partner curriculum, held-out stories. Left: every cooperating trace with a "
        "non-copy partner, read in full against a written codebook;\n92 of 96 rest on a premise the prompt "
        "never gave, against 4 of 40 defections. Right: regex over all traces, 95% Wilson intervals.",
    )
    save(fig, "rl_confabulation")


if __name__ == "__main__":
    figure_training_effect()
    figure_partner_description()
    figure_reasoning_shift()
    figure_stated_vs_revealed()
    figure_steering()
    figure_self_knowledge_reasoning()
    figure_argument_prior_forest()
    figure_curriculum_trajectory()
    figure_confabulation()
