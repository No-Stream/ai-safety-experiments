# ruff: noqa: INP001  -- a standalone render script under docs/, not an importable package
"""Render the four figures for docs/writeups/ai_cooperation_reasons.md.

All data is inline; there is no other source. Run from the repo root:

    uv run python docs/writeups/figures/make_cooperation_figures.py

Each figure is written next to this script as PNG (200 dpi) and SVG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
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

# Chrome and ink (dataviz reference palette, light mode).
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

# One colour per model, held constant across every figure. Slots come from the
# dataviz categorical palette; the models that share a plot (figures 2 and 4:
# Sonnet, Qwen, Luna) were validated together, and figure 3 seats one model per panel.
MODEL_COLOR: dict[str, str] = {
    "Claude Sonnet 5": "#2a78d6",
    "Qwen3-235B": "#eb6834",
    "DeepSeek v3.2": "#008300",
    "Kimi K2.5": "#e34948",
    "GPT-5.6 Luna": "#4a3aa7",
    "Claude Haiku 4.5": "#1baf7a",
    "GLM-4.7": "#eda100",
    "gpt-oss-20b": "#e87ba4",
    "gpt-oss-120b": "#898781",
}

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "blue_sequential", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

COUNTERPART_LABELS: list[str] = [
    "Copy of itself",
    "Fine-tune of same base",
    "Larger, same family",
    "Smaller, same family",
    "Unrelated model",
    "Unrelated model, same task",
    "Human",
]

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

HEATMAP_WHITE_TEXT_THRESHOLD = 0.42

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


# --------------------------------------------------------------------------------------
# Figure 1: identity ladder heatmap
# --------------------------------------------------------------------------------------

# Rows: cooperation rate by counterpart description, columns in COUNTERPART_LABELS order.
IDENTITY_LADDER: dict[str, list[float]] = {
    "Claude Sonnet 5": [0.664, 0.401, 0.237, 0.073, 0.008, 0.036, 0.112],
    "Qwen3-235B": [0.732, 0.480, 0.260, 0.190, 0.083, 0.149, 0.078],
    "DeepSeek v3.2": [0.443, 0.279, 0.219, 0.172, 0.109, 0.185, 0.148],
    "Claude Haiku 4.5": [0.427, 0.352, 0.336, 0.344, 0.365, 0.435, 0.268],
    "Kimi K2.5": [0.258, 0.143, 0.128, 0.073, 0.076, 0.125, 0.065],
    "GLM-4.7": [0.331, 0.349, 0.330, 0.339, 0.344, 0.326, 0.346],
    "gpt-oss-20b": [0.024, 0.018, 0.023, 0.019, 0.018, 0.026, 0.019],
    "gpt-oss-120b": [0.000, 0.003, 0.000, 0.000, 0.010, 0.005, 0.003],
}


def figure_identity_ladder_heatmap() -> None:
    """Plot the heatmap of cooperation rate by model and counterpart description."""
    models = sorted(IDENTITY_LADDER, key=lambda model: IDENTITY_LADDER[model][0], reverse=True)
    matrix = np.array([IDENTITY_LADDER[model] for model in models])

    fig, ax = plt.subplots(figsize=(10, 6.6))
    fig.subplots_adjust(left=0.16, right=0.985, top=0.79, bottom=0.14)
    title_block(
        fig,
        "Cooperation falls as the counterpart gets less like the model itself",
        "Cooperation rate in prisoner's dilemma and public goods (pooled), by how the other player was described",
        top=0.965,
    )

    ax.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=0.8, aspect="auto")
    # Surface gap between cells: draw the grid in the surface colour over the cell edges.
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            text_color = SURFACE if value > HEATMAP_WHITE_TEXT_THRESHOLD else INK
            ax.text(
                column_index,
                row_index,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=10.5,
                color=text_color,
            )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([label.replace(", ", ",\n") for label in COUNTERPART_LABELS], fontsize=9.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="major", pad=4)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(models, fontsize=10.5, color=INK)
    strip_axes(ax, keep=())
    ax.tick_params(axis="both", which="major", length=0)

    footnote(
        fig,
        "384 replies per cell. Rows ordered by cooperation with a copy. Every condition told the model that the two\n"
        "decisions need not agree. GPT-5.6 Luna was not run on this ladder and is omitted.",
        y=0.025,
    )
    save(fig, "identity_ladder_heatmap")


# --------------------------------------------------------------------------------------
# Figure 2: match-count dose curve
# --------------------------------------------------------------------------------------

MATCH_COUNT_X: list[int] = [0, 1, 2, 3, 5, 7, 9, 10]
MATCH_COUNT_GIVING: dict[str, list[float]] = {
    "Qwen3-235B": [0.062, 0.102, 0.617, 0.732, 0.953, 0.965, 0.996, 1.000],
    "GPT-5.6 Luna": [0.000, 0.000, 0.133, 0.172, 0.243, 0.461, 0.570, 0.594],
}
MATCH_COUNT_HALFWIDTH: dict[str, list[float]] = {
    "Qwen3-235B": [0.059, 0.045, 0.096, 0.069, 0.032, 0.038, 0.026, 0.020],
    "GPT-5.6 Luna": [0.000, 0.000, 0.058, 0.064, 0.098, 0.101, 0.109, 0.140],
}
NO_RECORD_BASELINE: dict[str, float] = {"Qwen3-235B": 0.015, "GPT-5.6 Luna": 0.000}
COPY_WITH_ZERO_MATCHES: dict[str, float] = {"Qwen3-235B": 0.941, "GPT-5.6 Luna": 0.000}


def figure_match_count_dose_curve() -> None:
    """Plot giving against the number of past rounds an unrelated model was said to have matched."""
    fig, ax = plt.subplots(figsize=(9, 6.4))
    fig.subplots_adjust(left=0.09, right=0.84, top=0.83, bottom=0.22)
    title_block(
        fig,
        "Giving rises with claimed past matches, even for an unrelated model",
        "Mean fraction of a 20-unit endowment given, by how many of 10 past rounds the unrelated model "
        "was said to have matched",
        top=0.965,
    )

    x = np.array(MATCH_COUNT_X)
    for model, giving in MATCH_COUNT_GIVING.items():
        color = MODEL_COLOR[model]
        y = np.array(giving)
        halfwidth = np.array(MATCH_COUNT_HALFWIDTH[model])
        ax.fill_between(
            x,
            np.clip(y - halfwidth, 0, 1),
            np.clip(y + halfwidth, 0, 1),
            color=color,
            alpha=0.12,
            lw=0,
        )
        ax.plot(x, y, color=color, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)
        ax.plot(x, y, "o", color=color, ms=7, mec=SURFACE, mew=1.5, zorder=4)
        ax.axhline(
            NO_RECORD_BASELINE[model], color=color, lw=1.2, ls=(0, (4, 3)), alpha=0.8, zorder=2
        )
        ax.plot(
            [0],
            [COPY_WITH_ZERO_MATCHES[model]],
            marker="D",
            ms=9,
            mfc=SURFACE,
            mec=color,
            mew=2,
            ls="none",
            zorder=5,
        )
        ax.annotate(
            model,
            xy=(10, y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )

    ax.annotate(
        "Copy, said to have matched\non 0 of 10 rounds: still gives (94%)",
        xy=(0, COPY_WITH_ZERO_MATCHES["Qwen3-235B"]),
        xytext=(0.55, 1.0),
        textcoords="data",
        fontsize=9.5,
        color=INK_SECONDARY,
        va="top",
        ha="left",
    )
    ax.annotate(
        "No track record: 2% (Qwen), 0% (Luna)",
        xy=(6.0, NO_RECORD_BASELINE["Qwen3-235B"]),
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
        ha="center",
    )

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(MATCH_COUNT_X)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.yaxis.set_major_formatter(percent_formatter)
    ax.set_xlabel(
        "Past rounds (of 10) the unrelated model was said to have matched the actor's choice"
    )
    ax.set_ylabel("Mean share of endowment given")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    strip_axes(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            lw=2,
            marker="o",
            ms=7,
            mec=SURFACE,
            label="Unrelated model with a track record",
        ),
        Patch(facecolor="#dddcd6", label="± band vs the no-record baseline"),
        Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            lw=1.2,
            ls=(0, (4, 3)),
            label="Same unrelated model, no track record given",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            ms=8,
            mfc=SURFACE,
            mec=INK_SECONDARY,
            mew=2,
            ls="none",
            label="Copy of itself, said to have matched 0 of 10",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.035),
        ncol=2,
        handlelength=2.4,
        columnspacing=2.0,
        labelcolor=INK_SECONDARY,
    )
    footnote(fig, "128 replies per point. Colour marks the giving model.", y=0.012)
    save(fig, "match_count_dose_curve")


# --------------------------------------------------------------------------------------
# Figure 3: one-way vs choosing dumbbells
# --------------------------------------------------------------------------------------

ONE_WAY_ROWS: list[str] = ["Identity unspecified", *COUNTERPART_LABELS]
# Values as (choosing, one-way) per row, in ONE_WAY_ROWS order.
ONE_WAY_VS_CHOOSING: dict[str, list[tuple[float, float]]] = {
    "Claude Sonnet 5": [
        (0.008, 0.016),
        (0.736, 0.236),
        (0.502, 0.477),
        (0.139, 0.199),
        (0.000, 0.278),
        (0.000, 0.111),
        (0.016, 0.274),
        (0.004, 0.021),
    ],
    "Qwen3-235B": [
        (0.152, 0.075),
        (0.906, 0.086),
        (0.777, 0.032),
        (0.445, 0.204),
        (0.367, 0.162),
        (0.039, 0.106),
        (0.109, 0.136),
        (0.094, 0.162),
    ],
    "DeepSeek v3.2": [
        (0.116, 0.125),
        (0.570, 0.180),
        (0.659, 0.253),
        (0.706, 0.109),
        (0.445, 0.113),
        (0.342, 0.077),
        (0.549, 0.168),
        (0.210, 0.107),
    ],
    "Kimi K2.5": [
        (0.086, 0.109),
        (0.484, 0.171),
        (0.514, 0.121),
        (0.444, 0.084),
        (0.156, 0.152),
        (0.123, 0.027),
        (0.268, 0.088),
        (0.174, 0.096),
    ],
}


def figure_one_way_vs_choosing() -> None:
    """Plot per-model dumbbells of giving when counterparts choose vs in the one-way game."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.20, right=0.975, top=0.79, bottom=0.115, wspace=0.10, hspace=0.30)
    title_block(
        fig,
        "The copy effect vanishes when nothing can come back",
        "Mean share of a 20-unit endowment given, by counterpart description, in two versions of the giving game",
        top=0.97,
    )

    y_positions = np.arange(len(ONE_WAY_ROWS))[::-1]
    for ax, (model, pairs) in zip(axes.flat, ONE_WAY_VS_CHOOSING.items(), strict=True):
        color = MODEL_COLOR[model]
        choosing = np.array([pair[0] for pair in pairs])
        one_way = np.array([pair[1] for pair in pairs])
        ax.hlines(y_positions, one_way, choosing, color=color, lw=2, alpha=0.35, zorder=2)
        ax.plot(one_way, y_positions, "o", ms=9, mfc=SURFACE, mec=color, mew=2, ls="none", zorder=3)
        ax.plot(
            choosing, y_positions, "o", ms=9, color=color, mec=SURFACE, mew=1.5, ls="none", zorder=4
        )
        ax.set_title(model, pad=8)
        ax.set_xlim(-0.03, 1.0)
        ax.set_ylim(-0.6, len(ONE_WAY_ROWS) - 0.4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ONE_WAY_ROWS, fontsize=10, color=INK)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.xaxis.set_major_formatter(percent_formatter)
        ax.tick_params(axis="x", labelbottom=True, labelsize=9)
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        strip_axes(ax, keep=("bottom",))

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ms=9,
            color=INK_SECONDARY,
            mec=SURFACE,
            mew=1.5,
            ls="none",
            label="Counterparts also choose, so they can give back",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            ms=9,
            mfc=SURFACE,
            mec=INK_SECONDARY,
            mew=2,
            ls="none",
            label="One-way game: beneficiaries cannot give back",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.875),
        ncol=2,
        columnspacing=2.5,
        handletextpad=0.6,
        labelcolor=INK_SECONDARY,
    )
    fig.text(
        0.585,
        0.045,
        "Mean share of endowment given",
        ha="center",
        va="bottom",
        fontsize=10,
        color=INK_SECONDARY,
    )
    footnote(
        fig,
        "128 replies per cell. Answer-field slips corrected. Colour marks the giving model.",
        y=0.012,
    )
    save(fig, "one_way_vs_choosing")


# --------------------------------------------------------------------------------------
# Figure 4: copy effect with chosen vs randomly drawn contributions
# --------------------------------------------------------------------------------------

# Copy effect in percentage points (giving to copies minus giving with identity unspecified), ± 2 SE.
COPY_EFFECT: dict[str, dict[str, tuple[float, float]]] = {
    "Claude Sonnet 5": {"choose": (70.3, 7.5), "random": (0.8, 3.6)},
    "Qwen3-235B": {"choose": (84.4, 7.7), "random": (7.8, 8.5)},
    "GPT-5.6 Luna": {"choose": (2.3, 2.5), "random": (0.0, 0.0)},
}


def figure_copy_effect_random_draw() -> None:
    """Plot the copy effect as grouped bars, counterparts choosing vs contributions drawn at random."""
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.76, bottom=0.12)
    title_block(
        fig,
        "The copy effect disappears when the copies do not choose",
        "Extra giving to copies of itself over an unspecified counterpart, in percentage points of the endowment",
        top=0.965,
    )

    bar_width = 0.3
    group_positions = np.arange(len(COPY_EFFECT))
    for group_position, (model, conditions) in zip(
        group_positions, COPY_EFFECT.items(), strict=True
    ):
        color = MODEL_COLOR[model]
        for offset, condition in (
            (-bar_width / 2 - 0.02, "choose"),
            (bar_width / 2 + 0.02, "random"),
        ):
            effect, halfwidth = conditions[condition]
            x = group_position + offset
            if condition == "choose":
                ax.bar(x, effect, width=bar_width, color=color, lw=0, zorder=3)
            else:
                ax.bar(
                    x, effect, width=bar_width, facecolor=SURFACE, edgecolor=color, lw=2, zorder=3
                )
            if halfwidth > 0:
                ax.errorbar(
                    x,
                    effect,
                    yerr=halfwidth,
                    fmt="none",
                    ecolor=INK,
                    elinewidth=1.2,
                    capsize=4,
                    zorder=4,
                )
            ax.text(
                x,
                effect + halfwidth + 2.0,
                f"{effect:+.1f}".replace("+0.0", "0.0"),
                ha="center",
                va="bottom",
                fontsize=10,
                color=INK,
            )

    ax.axhline(0, color=AXIS_LINE, lw=0.8, zorder=2)
    ax.set_xticks(group_positions)
    ax.set_xticklabels(list(COPY_EFFECT), fontsize=11, color=INK)
    ax.set_xlim(-0.6, len(COPY_EFFECT) - 0.4)
    ax.set_ylim(-6, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:+.0f} pp" if value else "0")
    )
    ax.set_ylabel("Copy effect (percentage points of endowment)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    strip_axes(ax, keep=("left",))
    ax.tick_params(axis="x", length=0)

    legend_handles = [
        Patch(facecolor=INK_SECONDARY, label="Counterparts choose their contributions"),
        Patch(
            facecolor=SURFACE,
            edgecolor=INK_SECONDARY,
            lw=2,
            label="Contributions randomly drawn in advance",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.87),
        ncol=2,
        columnspacing=2.5,
        labelcolor=INK_SECONDARY,
    )
    footnote(
        fig, "128 replies per cell. Error bars are ± 2 SE. Colour marks the giving model.", y=0.02
    )
    save(fig, "copy_effect_random_draw")


if __name__ == "__main__":
    figure_identity_ladder_heatmap()
    figure_match_count_dose_curve()
    figure_one_way_vs_choosing()
    figure_copy_effect_random_draw()
