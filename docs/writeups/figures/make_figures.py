# ruff: noqa: INP001  -- a standalone render script under docs/, not an importable package
"""Render the RecoveryBench blog figures from the frozen figure-data JSON.

Figures 1 to 3 come from ``figure_data.json``; the only derived quantities there are k/n
rates and 95% Wilson intervals. Figures 4 and 5 come from the per-item and bimodality
tables of the hunt26 reanalysis, whose only derived quantities are per-band means. Run:

    .venv/bin/python docs/writeups/figures/make_figures.py --data <path-to-figure_data.json>

Figure 6 comes from the Muse planted-vs-natural summary table and its stage-A records.

Writes ``fig{1,2,3,4,5,6}_*.svg`` and 2x ``.png`` next to this file.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import pathlib
import random
import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.lines import Line2D

mpl.use("Agg")
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FIGURE_DIR = pathlib.Path(__file__).resolve().parent

# --- palette -----------------------------------------------------------------
# Dark ground, off-white ink, three saturated accents. The two data hues (blue,
# red) pass the dataviz validator's all-pairs checks against this surface:
# CVD dE 19.2, normal-vision dE 29.0, both marks >= 3:1 contrast.
BG = "#131313"
INK = "#F2F0EB"
INK_SECONDARY = "#ABA69C"
INK_MUTED = "#7C776E"
GRID = "#2C2C2C"
RULE = "#3A3A38"
BLUE = "#3987E5"  # natural mistakes / unaided solve / negative penalty
RED = "#E66767"  # planted flaws / inheritance / positive penalty
MUSTARD = "#C98500"  # non-series accent: callout rules only, never a series

BASE_DPI = 100  # 8in wide -> 800px SVG/PNG at 1x; PNG is written at 2x.

# A value label goes below its marker unless that would crowd the axis floor.
AXIS_FLOOR_CROWDING = 0.15
SAFE_SERIES_GAP = 0.25


def resolve_font() -> str:
    """Return the first installed grotesque from the preferred stack."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Helvetica", "Helvetica Neue", "Arial", "Inter", "Liberation Sans", "DejaVu Sans"):
        if name in installed:
            return name
    raise RuntimeError(f"no grotesque sans found among installed fonts: {sorted(installed)}")


def apply_style(font_family: str) -> None:
    """Set the dark Bauhaus-ish house style on the global rcParams."""
    plt.rcParams.update(
        {
            "font.family": font_family,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "text.color": INK,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
        }
    )


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def add_title(
    fig: Figure, title: str, subtitle: str, *, title_wrap: int, subtitle_wrap: int
) -> None:
    """Full-sentence claim plus a one-line statement of what is measured."""
    title_lines = textwrap.wrap(title, title_wrap)
    fig.text(
        0.035,
        0.975,
        "\n".join(title_lines),
        ha="left",
        va="top",
        fontsize=14.5,
        color=INK,
        linespacing=1.35,
    )
    subtitle_top = 0.975 - 0.043 * len(title_lines) - 0.012
    fig.text(
        0.035,
        subtitle_top,
        "\n".join(textwrap.wrap(subtitle, subtitle_wrap)),
        ha="left",
        va="top",
        fontsize=8.8,
        color=INK_SECONDARY,
        linespacing=1.45,
    )


def add_footer(fig: Figure, lines: list[str], *, wrap: int, y: float = 0.055) -> None:
    """Draw the caveat lines inside the figure, below the plot."""
    text = "\n".join("\n".join(textwrap.wrap(line, wrap)) for line in lines)
    fig.text(0.035, y, text, ha="left", va="top", fontsize=7.6, color=INK_MUTED, linespacing=1.5)


def strip_spines(ax: Axes, keep: tuple[str, ...]) -> None:
    """Hide every axis spine except the named ones."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# --- figure 1 ----------------------------------------------------------------
def make_fig1(data: dict[str, Any]) -> pathlib.Path:
    """Paired natural-vs-planted inheritance rates per cell, plus the pooled summary pair."""
    block = data["fig1_planted_vs_natural"]
    cells = block["cells"]
    pooled = block["pooled"]

    rows: list[dict[str, Any]] = [
        {
            "label": f"{c['receiver']}\n{c['problem']} · {c['channel']}",
            "natural": c["natural"],
            "planted": c["planted"],
            "pooled": False,
        }
        for c in cells
    ]
    rows.append(
        {
            "label": "ALL FIVE CELLS POOLED\n95% Wilson intervals shown",
            "natural": pooled["natural"],
            "planted": pooled["planted"],
            "pooled": True,
        }
    )

    fig = plt.figure(figsize=(8.0, 5.9), dpi=BASE_DPI)
    ax = fig.add_axes((0.235, 0.205, 0.735, 0.545))

    # Pooled row sits below a gap so it reads as a summary, not a sixth cell.
    y_positions = [5.6, 4.6, 3.6, 2.6, 1.6, 0.0]

    for row, y in zip(rows, y_positions, strict=True):
        nat = row["natural"]["k"] / row["natural"]["n"]
        pla = row["planted"]["k"] / row["planted"]["n"]
        lw = 3.0 if row["pooled"] else 1.6
        ms = 11.0 if row["pooled"] else 8.5
        ax.plot([nat, pla], [y, y], color=RULE, lw=lw, solid_capstyle="butt", zorder=2)
        # Labels sit outside the interval on the pooled row so they never cross a whisker.
        natural_anchor, planted_anchor = nat, pla
        if row["pooled"]:
            for color, spec, side in ((BLUE, row["natural"], "lo"), (RED, row["planted"], "hi")):
                lo, hi = wilson_interval(spec["k"], spec["n"])
                ax.plot(
                    [lo, hi],
                    [y, y],
                    color=color,
                    lw=1.4,
                    alpha=0.6,
                    solid_capstyle="butt",
                    zorder=3,
                )
                for edge in (lo, hi):
                    ax.plot(
                        [edge, edge], [y - 0.16, y + 0.16], color=color, lw=1.4, alpha=0.6, zorder=3
                    )
                if side == "lo":
                    natural_anchor = lo
                else:
                    planted_anchor = hi
        ax.plot([nat], [y], "o", ms=ms, color=BLUE, mec=BG, mew=2.0, zorder=5)
        ax.plot([pla], [y], "o", ms=ms, color=RED, mec=BG, mew=2.0, zorder=5)
        label_size = 8.6 if row["pooled"] else 8.0
        label_color = INK if row["pooled"] else INK_SECONDARY
        ax.annotate(
            f"{row['natural']['k']}/{row['natural']['n']}",
            (natural_anchor, y),
            textcoords="offset points",
            xytext=(-10, 0),
            ha="right",
            va="center",
            fontsize=label_size,
            color=label_color,
        )
        ax.annotate(
            f"{row['planted']['k']}/{row['planted']['n']}",
            (planted_anchor, y),
            textcoords="offset points",
            xytext=(10, 0),
            ha="left",
            va="center",
            fontsize=label_size,
            color=label_color,
        )

    ax.axhline(0.8, color=RULE, lw=0.8, zorder=1)

    ax.set_xlim(-0.17, 1.17)
    ax.set_ylim(-0.75, 6.25)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.0, linespacing=1.5)
    for tick_label, row in zip(ax.get_yticklabels(), rows, strict=True):
        if row["pooled"]:
            tick_label.set_color(INK)
            tick_label.set_fontsize(8.2)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, pad=6)
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        ax.axvline(x, color=GRID, lw=0.7, zorder=0)
    strip_spines(ax, keep=())
    ax.set_xlabel(
        "Strict inheritance rate — share of rollouts repeating the draft's failure (labels are k/n rollouts)",
        fontsize=8.6,
        labelpad=10,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=8.5,
            color=BLUE,
            mec=BG,
            mew=2.0,
            label="Natural mistake draft",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=8.5,
            color=RED,
            mec=BG,
            mew=2.0,
            label="Planted flaw draft",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.975, 0.80),
        frameon=False,
        fontsize=8.6,
        handletextpad=0.6,
        labelspacing=0.55,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    add_title(
        fig,
        "Deliberately planted flaws are inherited far more often than the models' own natural mistakes",
        "Strict inheritance: the receiving model's submission fails every test the supplied wrong draft failed. "
        "One row per receiver / problem / channel cell.",
        title_wrap=64,
        subtitle_wrap=98,
    )
    add_footer(
        fig,
        [
            (
                "Both arms use the same artifact shape: a fenced code submission with no explanation. "
                "Rates are raw, not net of the no-draft baseline."
            ),
            (
                "Pooled planted vs natural, Fisher exact p = 1.7e-12. Per-cell counts are small (n = 3 to 10); "
                "intervals are drawn on the pooled pair only."
            ),
        ],
        wrap=126,
        y=0.105,
    )
    return save(fig, "fig1_planted_vs_natural")


# --- figure 2 ----------------------------------------------------------------
def make_fig2(data: dict[str, Any]) -> pathlib.Path:
    """Net inheritance and unaided solve against the reasoning ladder, one panel per model."""
    block = data["fig2_effort_ladder"]
    panels = block["panels"]

    fig = plt.figure(figsize=(8.0, 8.6), dpi=BASE_DPI)
    axes = [
        fig.add_axes((0.105, 0.585, 0.865, 0.175)),
        fig.add_axes((0.105, 0.310, 0.865, 0.175)),
    ]

    for ax, panel in zip(axes, panels, strict=True):
        rows = sorted(panel["rows"], key=lambda r: r["reasoning_tokens"])
        xs = list(range(len(rows)))
        for series_key, color in (("unaided_solve", BLUE), ("net_inheritance", RED)):
            values = [r[series_key] for r in rows]
            ax.plot(xs, values, color=color, lw=2.0, solid_capstyle="round", zorder=3)
            for x, value, row in zip(xs, values, rows, strict=True):
                confounded = bool(row["flag"])
                ax.plot(
                    [x],
                    [value],
                    marker="o",
                    ms=9.0,
                    color=BG if confounded else color,
                    mec=color,
                    mew=2.0,
                    zorder=5,
                )
                # Each label goes on the far side from the other series, so the two
                # never collide where the lines cross.
                other = (
                    row["net_inheritance"]
                    if series_key == "unaided_solve"
                    else row["unaided_solve"]
                )
                below = value < other
                if below and value < AXIS_FLOOR_CROWDING and other - value > SAFE_SERIES_GAP:
                    below = (
                        False  # too close to the axis floor; the other series is far enough above
                    )
                ax.annotate(
                    f"{value:.0%}",
                    (x, value),
                    textcoords="offset points",
                    xytext=(0, -21 if below else 13),
                    ha="center",
                    fontsize=8.0,
                    color=INK_SECONDARY,
                )

        ax.set_xlim(-0.45, len(rows) - 0.55)
        ax.set_ylim(-0.10, 1.14)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8.2)
        for y in (0.0, 0.25, 0.5, 0.75, 1.0):
            ax.axhline(y, color=GRID, lw=0.7, zorder=0)
        tick_labels = [
            "\n".join(textwrap.wrap(r["rung"], 19))
            + f"\n{r['reasoning_tokens']:,} tok"
            + ("*" if "censored" in r["flag"] else "")
            for r in rows
        ]
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=7.8, linespacing=1.5)
        ax.tick_params(axis="x", length=0, pad=8)
        ax.tick_params(axis="y", length=3, pad=4)
        strip_spines(ax, keep=())

    fig.text(0.105, 0.768, panels[0]["model"], ha="left", va="bottom", fontsize=10.0, color=INK)
    fig.text(0.105, 0.493, panels[1]["model"], ha="left", va="bottom", fontsize=10.0, color=INK)
    for ax in axes:
        ax.set_ylabel("Rate of rollouts", fontsize=8.6, labelpad=8)
    fig.text(
        0.538,
        0.243,
        "Reasoning rung, ordered by median reasoning tokens with no draft (left = least reasoning)",
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK_SECONDARY,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            color=BLUE,
            lw=2.0,
            marker="o",
            ms=9.0,
            mec=BLUE,
            mew=2.0,
            label="Unaided solve rate",
        ),
        Line2D(
            [], [], color=RED, lw=2.0, marker="o", ms=9.0, mec=RED, mew=2.0, label="Net inheritance"
        ),
        Line2D(
            [],
            [],
            color=INK_MUTED,
            lw=0,
            marker="o",
            ms=9.0,
            mfc=BG,
            mec=INK_SECONDARY,
            mew=2.0,
            label="Hollow marker = confounded rung",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.100, 0.832),
        frameon=False,
        fontsize=8.4,
        ncol=3,
        columnspacing=2.2,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    # Matched-item callout, set off by a mustard rule rather than coloured text.
    callout = "\n".join(textwrap.wrap(block["matched_item_callout"], 88))
    fig.text(
        0.128,
        0.185,
        callout,
        ha="left",
        va="top",
        fontsize=8.2,
        color=INK_SECONDARY,
        linespacing=1.5,
    )
    fig.add_artist(
        Line2D([0.110, 0.110], [0.148, 0.187], color=MUSTARD, lw=2.6, transform=fig.transFigure)
    )

    add_title(
        fig,
        "More reasoning cuts inheritance, but it lifts unaided accuracy in step, so the ladder cannot tell "
        "better checking from better solving",
        "Net inheritance = strict inheritance rate with a draft minus the rate of failing the same tests with no "
        "draft. Unaided solve = share of no-draft rollouts passing all tests.",
        title_wrap=68,
        subtitle_wrap=104,
    )
    add_footer(
        fig,
        [
            (
                "Hollow markers mark rungs confounded by a higher unaided solve rate than the lowest rung of their "
                "panel: the two effects move together and cannot be separated here."
            ),
            (
                "* The high-effort rung is partially censored: 52 of 64 rollouts completed. "
                "Rollout counts per rung are given in each panel heading."
            ),
        ],
        wrap=126,
        y=0.100,
    )
    return save(fig, "fig2_effort_ladder")


# --- figure 3 ----------------------------------------------------------------
def make_fig3(data: dict[str, Any]) -> pathlib.Path:
    """Placebo-corrected penalty per model, ordered by rough capability."""
    block = data["fig3_jaggedbench_ladder"]
    rows = list(block["rows"])  # given weakest-first; drawn top-down

    fig = plt.figure(figsize=(8.0, 6.2), dpi=BASE_DPI)
    ax = fig.add_axes((0.235, 0.200, 0.715, 0.515))

    ys = list(range(len(rows)))[::-1]  # weakest at top
    for y, row in zip(ys, rows, strict=True):
        penalty = row["penalty"]
        color = RED if penalty >= 0 else BLUE
        ax.plot([0, penalty], [y, y], color=color, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.plot([penalty], [y], "o", ms=9.0, color=color, mec=BG, mew=2.0, zorder=4)
        offset = 11 if penalty >= 0 else -11
        ax.annotate(
            f"{penalty:+.2f}",
            (penalty, y),
            textcoords="offset points",
            xytext=(offset, 0),
            ha="left" if penalty >= 0 else "right",
            va="center",
            fontsize=8.2,
            color=INK_SECONDARY,
        )

    ax.set_xlim(-0.42, 1.00)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["-0.40", "-0.20", "0", "+0.20", "+0.40", "+0.60", "+0.80"], fontsize=8.2)
    for x in (-0.4, -0.2, 0.2, 0.4, 0.6, 0.8):
        ax.axvline(x, color=GRID, lw=0.7, zorder=0)
    ax.axvline(0.0, color=INK_SECONDARY, lw=1.3, zorder=2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=8.6)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=3, pad=6)
    strip_spines(ax, keep=())
    ax.set_xlabel(
        "Placebo-corrected penalty, in rate points of the correct move (0 = no effect of naming another dimension)",
        fontsize=8.6,
        labelpad=10,
    )

    # Capability ordering as a rule in the left margin, clear of the model names.
    rule_x, top, bottom = 0.045, 0.695, 0.218
    fig.add_artist(
        Line2D([rule_x, rule_x], [bottom, top], color=MUSTARD, lw=2.0, transform=fig.transFigure)
    )
    fig.text(rule_x, top + 0.012, "weakest", ha="left", va="bottom", fontsize=8.0, color=INK_MUTED)
    fig.text(
        rule_x, bottom - 0.014, "most capable", ha="left", va="top", fontsize=8.0, color=INK_MUTED
    )

    add_title(
        fig,
        "Told that a different dimension is graded, the more capable models more often abandon the correct move",
        "Placebo-corrected penalty: the drop in the rate of revising an inherited error when the prompt names "
        "another graded dimension, minus the drop under a length-matched neutral sentence. 29 items, 3 repeats "
        "per model.",
        title_wrap=66,
        subtitle_wrap=100,
    )
    add_footer(
        fig,
        [
            (
                "Models are ordered by rough capability, weakest at the top; the ordering is a judgement, not a "
                "measured score. Negative values (blue) mean the model kept the correct move slightly more often "
                "when another dimension was named."
            ),
            (
                "Caveat: grading markers were calibrated on two of the nine models only, so the weaker models' values "
                "are less certain."
            ),
        ],
        wrap=126,
        y=0.098,
    )
    return save(fig, "fig3_grader_ladder")


# --- figures 4 and 5: the capability-window reanalysis -------------------------
# These two read the per-item reanalysis tables rather than figure_data.json.
RECEIVER_LABELS = {
    "luna": "GPT-5.6 Luna",
    "sol": "GPT-5.6 Sol",
    "oss120b": "GPT-OSS-120B",
}
RECEIVER_ORDER = ("luna", "sol", "oss120b")
NARRATED = "narrated"

# The capability window: the receiver's own unaided solve rate on the item.
WINDOW_LO, WINDOW_HI = 0.4, 0.9
BAND_ORDER = ("below-window", "in-window", "above-window")
BAND_SHORT = {"below-window": "below", "in-window": "in window", "above-window": "above"}
# Where each band's mean marker is drawn, and how wide the marker is, in x units.
BAND_ANCHORS = {
    "below-window": (0.20, 0.30),
    "in-window": (0.65, 0.34),
    "above-window": (0.95, 0.14),
}

JITTER_HALF_WIDTH = 0.030  # solve rates are multiples of 1/8; points would stack otherwise.
# A band-mean label above this value would push through the panel ceiling, so it hangs below.
BAND_LABEL_CEILING = 0.6
JITTER_SEED = 20260912


def narrated_rows(per_item: dict[str, Any], receiver: str) -> list[dict[str, Any]]:
    """Every narrated-arm cell for one receiver, in file order."""
    rows = [r for r in per_item["rows"] if r["arm"] == NARRATED and r["receiver"] == receiver]
    if not rows:
        raise ValueError(f"no narrated rows for receiver {receiver!r}")
    return rows


def band_means(rows: list[dict[str, Any]]) -> dict[str, tuple[float, int]]:
    """Mean net strict inheritance and item count for each capability band present."""
    means: dict[str, tuple[float, int]] = {}
    for band in BAND_ORDER:
        values = [r["net_all"] for r in rows if r["band"] == band]
        if values:
            means[band] = (sum(values) / len(values), len(values))
    return means


# --- figure 4 ----------------------------------------------------------------
def make_fig4(per_item: dict[str, Any]) -> pathlib.Path:
    """Per-item net strict inheritance against the receiver's own unaided solve rate."""
    rng = random.Random(JITTER_SEED)
    panels = [(receiver, narrated_rows(per_item, receiver)) for receiver in RECEIVER_ORDER]

    fig = plt.figure(figsize=(8.0, 6.9), dpi=BASE_DPI)
    lefts = (0.086, 0.395, 0.704)
    axes = [fig.add_axes((left, 0.352, 0.268, 0.352)) for left in lefts]

    printed: list[str] = []
    for ax, (receiver, rows) in zip(axes, panels, strict=True):
        ax.axvspan(WINDOW_LO, WINDOW_HI, color="#1F1F1F", zorder=0)
        ax.axhline(0.0, color=RULE, lw=1.0, zorder=1)

        for row in rows:
            x = row["bare_solve_rate"] + rng.uniform(-JITTER_HALF_WIDTH, JITTER_HALF_WIDTH)
            own = bool(row["donor_model_is_receiver"])
            ax.plot(
                [x],
                [row["net_all"]],
                marker="o",
                ms=4.6,
                ls="none",
                mfc=BG if own else RED,
                mec=RED,
                mew=1.2,
                alpha=0.92,
                zorder=4,
            )

        means = band_means(rows)
        for band, (mean, count) in means.items():
            centre, half = BAND_ANCHORS[band]
            ax.plot(
                [centre - half / 2, centre + half / 2],
                [mean, mean],
                color=INK,
                lw=2.2,
                solid_capstyle="butt",
                zorder=6,
            )
            # The above-window band hugs the right edge, so its label is right-aligned
            # inside the panel and hangs below the tick; elsewhere the label sits above
            # the tick unless that would push it through the panel ceiling. Every label
            # wears an opaque patch so it stays readable over the point cloud.
            above_band = band == "above-window"
            below_tick = above_band or mean > BAND_LABEL_CEILING
            ax.annotate(
                f"{mean:+.2f}  n={count}",
                (1.06 if above_band else centre, mean),
                textcoords="offset points",
                xytext=(0, -8 if below_tick else 7),
                ha="right" if above_band else "center",
                va="top" if below_tick else "bottom",
                fontsize=7.2,
                color=INK,
                zorder=7,
                bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.6, "alpha": 0.88},
            )
            printed.append(
                f"{RECEIVER_LABELS[receiver]} {BAND_SHORT[band]}: {mean:+.3f} (n={count})"
            )

        ax.set_xlim(-0.09, 1.09)
        ax.set_ylim(-1.0, 1.0)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xticklabels(["0%", "50%", "100%"], fontsize=8.0)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        for y in (-1.0, -0.5, 0.5, 1.0):
            ax.axhline(y, color=GRID, lw=0.7, zorder=0)
        if ax is axes[0]:
            ax.set_yticklabels(["-1.0", "-0.5", "0", "+0.5", "+1.0"], fontsize=8.0)
            ax.set_ylabel("Net strict inheritance, in rate points", fontsize=8.6, labelpad=8)
            ax.tick_params(axis="y", length=3, pad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, pad=5)
        strip_spines(ax, keep=())

    for left, (receiver, rows) in zip(lefts, panels, strict=True):
        fig.text(
            left,
            0.714,
            f"{RECEIVER_LABELS[receiver]}  ·  {len(rows)} items",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color=INK,
        )
    fig.text(
        0.086,
        0.782,
        f"Shaded band = the {WINDOW_LO:.0%}-{WINDOW_HI:.0%} capability window. "
        "White ticks are band means, labelled with the item count.",
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=INK_MUTED,
    )
    fig.text(
        0.520,
        0.308,
        "Unaided solve rate on the same item, with no draft supplied (8 rollouts per item)",
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK_SECONDARY,
    )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=6.0,
            mfc=RED,
            mec=RED,
            mew=1.1,
            label="Peer donor draft",
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=6.0,
            mfc=BG,
            mec=RED,
            mew=1.1,
            label="Hollow = the model's own earlier failure as the draft",
        ),
        Line2D([], [], color=INK, lw=2.2, label="Band mean"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.082, 0.268),
        frameon=False,
        fontsize=8.0,
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    add_title(
        fig,
        "Inheritance peaks on the problems a model can usually but not always solve, and on Sol's "
        "always-solved problems it nearly vanishes",
        "Net strict inheritance = the share of rollouts failing every test the supplied wrong draft "
        "failed, minus the same model's rate of failing those same tests with no draft. One point per "
        "item; narrated drafts only.",
        title_wrap=68,
        subtitle_wrap=116,
    )
    band_counts = "; ".join(
        f"{RECEIVER_LABELS[receiver]} "
        + "/".join(str(band_means(rows).get(band, (0.0, 0))[1]) for band in BAND_ORDER)
        for receiver, rows in panels
    )
    add_footer(
        fig,
        [
            (
                "Net is carry minus the same model's no-draft floor on the same tests, so 0 means the draft "
                "changed nothing and +1 means every aided rollout failed tests the model passed unaided. "
                "n = 10 rollouts per cell (7 cells have 8 or 9 gradable)."
            ),
            (
                "x is jittered by up to +/-0.03 because unaided solve rates are multiples of 1/8 and points "
                f"would otherwise stack. Items per band, below/in window/above: {band_counts}."
            ),
            (
                "Only Sol has a well-populated above-window band, so the drop above the window is measured "
                "on Sol alone."
            ),
        ],
        wrap=132,
        y=0.212,
    )
    for line in printed:
        logger.info("fig4 band mean -- %s", line)
    return save(fig, "fig4_window_scatter")


# --- figure 5 ----------------------------------------------------------------
def make_fig5(bimodality: dict[str, Any]) -> pathlib.Path:
    """Distribution of per-cell strict carry counts over in-window narrated cells."""
    panels = [(receiver, bimodality[receiver]) for receiver in RECEIVER_ORDER]
    max_count = max(
        int(v) for _, block in panels for v in block["histogram_k_to_cell_count"].values()
    )

    fig = plt.figure(figsize=(8.0, 5.8), dpi=BASE_DPI)
    lefts = (0.086, 0.395, 0.704)
    axes = [fig.add_axes((left, 0.360, 0.268, 0.360)) for left in lefts]

    for ax, (_receiver, block) in zip(axes, panels, strict=True):
        histogram = block["histogram_k_to_cell_count"]
        ks = list(range(11))
        counts = [int(histogram.get(str(k), 0)) for k in ks]
        ax.bar(ks, counts, width=0.74, color=RED, linewidth=0, zorder=3)
        for k, count in zip(ks, counts, strict=True):
            if count:
                ax.annotate(
                    str(count),
                    (k, count),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    color=INK_SECONDARY,
                )

        ax.set_xlim(-0.9, 10.9)
        ax.set_ylim(0, max_count + 1.35)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(["0", "2", "4", "6", "8", "10"], fontsize=8.0)
        ax.set_yticks(list(range(0, max_count + 1, 2)))
        for y in range(2, max_count + 1, 2):
            ax.axhline(y, color=GRID, lw=0.7, zorder=0)
        if ax is axes[0]:
            ax.set_yticklabels([str(y) for y in range(0, max_count + 1, 2)], fontsize=8.0)
            ax.set_ylabel("Number of cells", fontsize=8.6, labelpad=8)
            ax.tick_params(axis="y", length=3, pad=4)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, pad=5)
        strip_spines(ax, keep=())

    for left, (receiver, block) in zip(lefts, panels, strict=True):
        fig.text(
            left,
            0.732,
            f"{RECEIVER_LABELS[receiver]}  ·  {block['n_cells']} cells",
            ha="left",
            va="bottom",
            fontsize=9.2,
            color=INK,
        )
    fig.text(
        0.520,
        0.313,
        "Rollouts in the cell that repeated the draft's failure, k out of 10",
        ha="center",
        va="top",
        fontsize=8.6,
        color=INK_SECONDARY,
    )

    add_title(
        fig,
        "Within the capability window Sol either copies the draft's mistake almost every time or almost "
        "never; Luna and GPT-OSS are more graded",
        "One bar per k: how many in-window narrated cells had exactly k of their 10 rollouts repeat every "
        "test the supplied draft failed. A cell is one receiver on one item.",
        title_wrap=68,
        subtitle_wrap=104,
    )
    add_footer(
        fig,
        [
            (
                "Sol's modes sit at k=0 (6 cells) and k=10 (4 cells) with the middle thinly spread; Luna's mass "
                "sits at k=7-8 and GPT-OSS spreads across the range. With 17 to 30 cells per panel these shapes "
                "are suggestive, not established."
            ),
            (
                "Three cells were graded on 9 rollouts rather than 10 (two Sol, one GPT-OSS); they are folded "
                "into the same histogram by their raw carry count, so their k is bounded at 9."
            ),
        ],
        wrap=128,
        y=0.240,
    )
    return save(fig, "fig5_bimodality")


# --- figure 6: the Muse planted-vs-natural draft arms --------------------------
DRAFT_ARMS = ("natural", "clean_natural", "planted")
ARM_LABELS = {
    "natural": "natural\n(a wrong solution\nanother model produced)",
    "clean_natural": "clean rewrite\n(the same wrong algorithm,\nwritten cleanly)",
    "planted": "planted\n(a freshly authored\nwrong method)",
}
MUSE_JITTER_HALF_WIDTH = 0.013  # per-item values tie exactly; lines would hide each other.
MUSE_JITTER_SEED = 20260913


def parse_muse_per_item(path: pathlib.Path) -> dict[str, dict[str, dict[str, float]]]:
    """Read the ``Per-item`` table of the Muse summary into item -> arm -> metrics.

    The only parsed columns are the strict ``carry all raw/net`` pair and the solve rate.
    """
    per_item: dict[str, dict[str, dict[str, float]]] = {}
    in_table = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith("## Per-item"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        if not in_table or not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] in {"item", ""}:
            continue
        item, arm = cells[0], cells[1]
        raw_all, net_all = (v.strip() for v in cells[4].split(" / "))  # "N/A" itself holds a slash
        metrics = {"solve": float(cells[5])}
        if net_all != "N/A":
            metrics["raw_all"] = float(raw_all)
            metrics["net_all"] = float(net_all)
        per_item.setdefault(item, {})[arm] = metrics
    if not per_item:
        raise ValueError(f"no per-item rows parsed from {path}")
    return per_item


def parse_muse_bare_solve(path: pathlib.Path) -> dict[str, float]:
    """Unaided solve rate per item from the stage-A records."""
    passed: dict[str, list[bool]] = {}
    for raw_line in path.read_text().splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        passed.setdefault(record["task_id"], []).append(bool(record["grade"]["passed_all"]))
    return {item: sum(flags) / len(flags) for item, flags in passed.items()}


def draw_item_lines(
    ax: Axes, xs: list[float], series: list[list[float]], rng: random.Random
) -> None:
    """One thin jittered line per item over the panel's x positions."""
    for values in series:
        jitter = rng.uniform(-MUSE_JITTER_HALF_WIDTH, MUSE_JITTER_HALF_WIDTH)
        ax.plot(
            xs,
            [v + jitter for v in values],
            color=INK_MUTED,
            lw=1.0,
            alpha=0.75,
            solid_capstyle="round",
            zorder=3,
        )
        for x, value in zip(xs, values, strict=True):
            ax.plot([x], [value + jitter], marker="o", ms=3.0, ls="none", color=INK_MUTED, zorder=4)


def draw_mean_marker(ax: Axes, x: float, mean: float, color: str, *, below: bool) -> None:
    """Draw a bold pooled-mean marker with its value label, on an opaque patch."""
    ax.plot([x], [mean], marker="o", ms=11.0, color=color, mec=BG, mew=2.0, zorder=7)
    ax.annotate(
        f"{mean:+.2f}" if color == RED else f"{mean:.2f}",
        (x, mean),
        textcoords="offset points",
        xytext=(0, -20 if below else 15),
        ha="center",
        va="top" if below else "bottom",
        fontsize=9.4,
        color=INK,
        zorder=8,
        bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.8, "alpha": 0.9},
    )


@dataclasses.dataclass(frozen=True)
class MusePanelAxis:
    """Everything that differs between the two figure-6 panels' axes."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xticks: list[float]
    xticklabels: list[str]
    yticks: list[float]
    yticklabels: list[str]
    ylabel: str


def style_muse_axis(ax: Axes, spec: MusePanelAxis) -> None:
    """Apply the house panel cosmetics: recessive gridlines at every y tick, no spines."""
    ax.set_xlim(*spec.xlim)
    ax.set_ylim(*spec.ylim)
    ax.set_xticks(spec.xticks)
    ax.set_xticklabels(spec.xticklabels, fontsize=7.8, linespacing=1.6)
    ax.set_yticks(spec.yticks)
    ax.set_yticklabels(spec.yticklabels, fontsize=8.0)
    for y in spec.yticks:
        ax.axhline(y, color=GRID, lw=0.7, zorder=0)
    ax.set_ylabel(spec.ylabel, fontsize=8.6, labelpad=8)
    ax.tick_params(axis="y", length=3, pad=4)
    ax.tick_params(axis="x", length=0, pad=9)
    strip_spines(ax, keep=())


def make_fig6(per_item_path: pathlib.Path, stage_a_path: pathlib.Path) -> pathlib.Path:
    """Per-item net strict inheritance across the three wrong-draft arms, plus the correct-draft lift."""
    per_item = parse_muse_per_item(per_item_path)
    bare_solve = parse_muse_bare_solve(stage_a_path)
    items = sorted(per_item)
    missing = [item for item in items if item not in bare_solve]
    if missing:
        raise ValueError(f"no stage-A floor for {missing}")

    rng = random.Random(MUSE_JITTER_SEED)
    fig = plt.figure(figsize=(8.0, 6.6), dpi=BASE_DPI)
    ax_arms = fig.add_axes((0.093, 0.345, 0.455, 0.335))
    ax_solve = fig.add_axes((0.745, 0.345, 0.185, 0.335))

    printed: list[str] = []

    # --- left panel: one thin line per item over the three wrong-draft arms ---
    xs = [0.0, 1.0, 2.0]
    draw_item_lines(
        ax_arms,
        xs,
        [[per_item[item][arm]["net_all"] for arm in DRAFT_ARMS] for item in items],
        rng,
    )

    arm_means = {
        arm: sum(per_item[item][arm]["net_all"] for item in items) / len(items)
        for arm in DRAFT_ARMS
    }
    ax_arms.plot(
        xs, [arm_means[arm] for arm in DRAFT_ARMS], color=RED, lw=2.6, zorder=6, alpha=0.85
    )
    for x, arm in zip(xs, DRAFT_ARMS, strict=True):
        mean = arm_means[arm]
        draw_mean_marker(ax_arms, x, mean, RED, below=False)
        printed.append(f"{arm}: mean net strict inheritance {mean:+.3f} (n={len(items)} items)")

    style_muse_axis(
        ax_arms,
        MusePanelAxis(
            xlim=(-0.42, 2.42),
            ylim=(-0.42, 1.02),
            xticks=xs,
            xticklabels=[ARM_LABELS[arm] for arm in DRAFT_ARMS],
            yticks=[-0.25, 0.0, 0.25, 0.5, 0.75, 1.0],
            yticklabels=["-0.25", "0", "+0.25", "+0.50", "+0.75", "+1.00"],
            ylabel="Net strict inheritance, in rate points",
        ),
    )
    ax_arms.axhline(0.0, color=RULE, lw=1.1, zorder=1)

    # --- right panel: unaided solve against solve with a correct draft ---
    solve_xs = [0.0, 1.0]
    draw_item_lines(
        ax_solve,
        solve_xs,
        [[bare_solve[item], per_item[item]["correct"]["solve"]] for item in items],
        rng,
    )

    mean_bare = sum(bare_solve[item] for item in items) / len(items)
    mean_correct = sum(per_item[item]["correct"]["solve"] for item in items) / len(items)
    ax_solve.plot(solve_xs, [mean_bare, mean_correct], color=BLUE, lw=2.6, zorder=6, alpha=0.85)
    for x, mean, below in ((0.0, mean_bare, True), (1.0, mean_correct, False)):
        draw_mean_marker(ax_solve, x, mean, BLUE, below=below)
    printed.append(f"correct draft: mean solve {mean_correct:.3f} vs unaided {mean_bare:.3f}")

    style_muse_axis(
        ax_solve,
        MusePanelAxis(
            xlim=(-0.55, 1.55),
            ylim=(-0.06, 1.20),
            xticks=solve_xs,
            xticklabels=["no draft\n(unaided)", "correct draft\n(clean, right)"],
            yticks=[0.0, 0.25, 0.5, 0.75, 1.0],
            yticklabels=["0", "0.25", "0.50", "0.75", "1.00"],
            ylabel="Solve rate",
        ),
    )

    fig.text(
        0.093,
        0.715,
        "Wrong drafts: what the receiver inherits",
        ha="left",
        va="bottom",
        fontsize=9.6,
        color=INK,
    )
    fig.text(
        0.745,
        0.715,
        "Correct draft",
        ha="left",
        va="bottom",
        fontsize=9.6,
        color=INK,
    )
    # The uplift, set off by a mustard rule in the header line rather than a coloured number.
    fig.add_artist(
        Line2D([0.856, 0.856], [0.712, 0.742], color=MUSTARD, lw=2.4, transform=fig.transFigure)
    )
    fig.text(
        0.868,
        0.716,
        f"mean uplift {mean_correct - mean_bare:+.2f}",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=INK_SECONDARY,
    )

    legend_handles = [
        Line2D([], [], color=INK_MUTED, lw=1.0, marker="o", ms=3.6, label="One problem"),
        Line2D(
            [],
            [],
            color=RED,
            lw=2.6,
            marker="o",
            ms=9.0,
            mec=BG,
            mew=1.6,
            label="Mean over the 12 problems (inheritance)",
        ),
        Line2D(
            [],
            [],
            color=BLUE,
            lw=2.6,
            marker="o",
            ms=9.0,
            mec=BG,
            mew=1.6,
            label="Mean over the 12 problems (solve rate)",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.089, 0.252),
        frameon=False,
        fontsize=8.0,
        ncol=3,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    add_title(
        fig,
        "On Luna, a tidy rewrite of the same wrong method is inherited twice as often as the "
        "original, and a freshly planted flaw less than either",
        "Net strict inheritance = the share of rollouts failing every test the supplied wrong draft "
        "failed, minus the same model's rate of failing those same tests with no draft. 12 problems "
        "in Luna's 40% to 90% unaided-solve range; 10 rollouts per cell.",
        title_wrap=70,
        subtitle_wrap=112,
    )
    add_footer(
        fig,
        [
            (
                "Fisher exact on the pooled strict carry counts: the clean rewrite against the natural "
                "original p = 0.012; the planted flaw against the natural original p = 0.015, in the "
                "opposite direction from an earlier two-problem result on other models. The planted flaws "
                "here were authored by a different process from that study, so this planted arm is not "
                "comparable across studies."
            ),
            (
                "Per problem, the clean rewrite carries more than the natural original on 9 of 12, and the "
                "planted flaw carries less than the natural original on 6 of 12."
            ),
            (
                "Solve rate with a correct draft is 0.83 against 0.66 unaided (8 unaided rollouts per "
                "problem). Thin lines are jittered vertically by up to +/-0.013 because many problems share "
                "identical values."
            ),
        ],
        wrap=134,
        y=0.205,
    )
    for line in printed:
        logger.info("fig6 -- %s", line)
    return save(fig, "fig6_draft_arms")


def save(fig: Figure, stem: str) -> pathlib.Path:
    """Write the figure as SVG and as a 2x PNG, and return the PNG path."""
    svg_path = FIGURE_DIR / f"{stem}.svg"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, format="png", dpi=BASE_DPI * 2)
    plt.close(fig)
    logger.info("wrote %s and %s", svg_path, png_path)
    return png_path


REPO_ROOT = FIGURE_DIR.parents[2]
DEFAULT_PER_ITEM = REPO_ROOT / "docs/scratch/mechanism-scoping/hunt26_reanalysis_per_item.json"
DEFAULT_BIMODALITY = REPO_ROOT / "docs/scratch/mechanism-scoping/hunt26_reanalysis_bimodality.json"
DEFAULT_MUSE_SUMMARY = REPO_ROOT / "docs/scratch/mechanism-scoping/muse_pvn_summary.luna.md"
DEFAULT_MUSE_STAGE_A = REPO_ROOT / "docs/scratch/mechanism-scoping/muse_pvn_stagea.luna.jsonl"


def main() -> None:
    """Render every figure from the JSON tables named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=pathlib.Path,
        help="path to figure_data.json; figures 1 to 3 are rendered only when this is given",
    )
    parser.add_argument(
        "--per-item",
        type=pathlib.Path,
        default=DEFAULT_PER_ITEM,
        help="path to hunt26_reanalysis_per_item.json (figure 4)",
    )
    parser.add_argument(
        "--bimodality",
        type=pathlib.Path,
        default=DEFAULT_BIMODALITY,
        help="path to hunt26_reanalysis_bimodality.json (figure 5)",
    )
    parser.add_argument(
        "--muse-summary",
        type=pathlib.Path,
        default=DEFAULT_MUSE_SUMMARY,
        help="path to muse_pvn_summary.luna.md (figure 6)",
    )
    parser.add_argument(
        "--muse-stage-a",
        type=pathlib.Path,
        default=DEFAULT_MUSE_STAGE_A,
        help="path to muse_pvn_stagea.luna.jsonl (figure 6)",
    )
    args = parser.parse_args()

    per_item = json.loads(args.per_item.read_text())
    bimodality = json.loads(args.bimodality.read_text())
    font_family = resolve_font()
    logger.info("using font family %s", font_family)
    apply_style(font_family)

    if args.data is None:
        logger.info("no --data given: skipping figures 1 to 3")
    else:
        data = json.loads(args.data.read_text())
        make_fig1(data)
        make_fig2(data)
        make_fig3(data)
    make_fig4(per_item)
    make_fig5(bimodality)
    make_fig6(args.muse_summary, args.muse_stage_a)


if __name__ == "__main__":
    main()
