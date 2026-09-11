r"""The per-arm battery readout: every ladder in one document, recomputed from the traces on disk.

Nothing here is written down by hand. The whole document is a function of the cells present under
one battery directory at the moment the command runs, so a rerun once the remaining cells land is
the entire update:

    uv run --frozen python -m games.battery_readout \
        --battery-dir artifacts/games/evals/battery-<sha> --out docs/scratch/games-readout-<sha>.md

Both arguments are optional. Without them it reads the most recently written battery under
`artifacts/games/evals/` and names the output after it, because a readout that has to be pointed at
the right pass is a readout that quietly goes stale.

`games.report` renders one arm's tables from a list of paths and `games.readout` reads the wave as
pooled early-versus-late windows across arms. This document reads the third thing neither does:
**the full ladder of every step present, arm by arm**, with the transfer grid, the negative control,
the decision-theory probes and the capability canary under one arm heading, so a behaviour move can
be checked against the canary beside it. The arms that trained the same game are then put side by
side, which is where a grading-rule contrast or a payoff-rung dose response actually reads.

`games.battery_tables` computes every number; this module is presentation and the CLI. One
deliberate difference from `games.readout`: a rate over a small denominator renders with its
denominator here rather than blanking below a floor. The unit of reading is the ladder, and a
`0.500 (2/2)` whose denominator the reader can see is more honest than a dash they cannot. The
header says so, and names the floor rather than applying it silently.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.battery_cells import (
    BATTERY_DIR_GLOB,
    BATTERY_DIR_PREFIX,
    DEFAULT_EVALS_ROOT,
    latest_battery_dir,
    relative_to,
)
from games.battery_tables import (
    ACCURACY_ASKED_COLUMN,
    ALL_VARIANTS,
    CONTROL_RATE_COLUMN,
    DELTA_COLUMN,
    EDT_COLUMN,
    EMPTY_CELL,
    GAME_COLUMN,
    MIN_CELLS_FOR_A_FLIP,
    NEGATIVE_CONTROL_GAME_ID,
    NOISE_FLOOR_PARSED,
    ORDER_COLUMN,
    PROSOCIAL_COLUMN,
    RATE_COLUMN,
    RISK_LOSS_AVERSION_MIRROR_PAIR,
    ROLE_TRAINED,
    STEP_COLUMN,
    UNREGISTERED,
    VARIANT_FIELD,
    ArmReadout,
    BatteryReadout,
    ContrastGroup,
    build_readout,
    steps_label,
)
from games.prompts import ITERATED_GAME_ID

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("docs/scratch")

GENERATED_BANNER = (
    "**GENERATED -- DO NOT HAND-EDIT.** Every number below is recomputed from the trace files on "
    "disk. Re-run the CLI (the command is in the header) and the whole document updates itself; an "
    "edit here is overwritten, and until it is, it disagrees with the artifacts."
)

SINGLE_CELL_CAVEAT = (
    "A single cell against a single other cell is not a finding: a measured null on an UNCHANGED "
    "model moved individual game-cells by up to 0.467 across three replicates. Read the ladder, or "
    "the pooled windows, and treat one cell's move as noise until the shape of the curve agrees."
)

DENOMINATOR_CAVEAT = (
    "Every behaviour rate renders as `rate (parsed/asked prompts)` and is a mean over PROMPTS, not "
    "over completions: a prompt sampled several times is averaged within itself first, so a prompt "
    "whose draws mostly failed to parse cannot outweigh its neighbours. Prompts that produced no "
    "usable answer leave the numerator and stay in the denominator. The `draws (parsed/asked)` "
    "column beside each per-step rate is the completion-level pair, and it is what moves when a "
    f"section stops parsing. A rate over fewer than {NOISE_FLOOR_PARSED} parsed prompts is printed "
    "with its denominator rather than suppressed; read it as a list of completions, not as a rate."
)

BF16_ATTENUATION_CAVEAT = (
    "Absolute effect sizes read off merged bf16 checkpoints run about 64% of true (38-79% per "
    "module), so every magnitude here is understated. Trends, orderings and contrasts are not."
)

BASE_CELL_CAVEAT = (
    "Step 0 is the un-adapted base model, evaluated with no merge, and is a valid cell rather than "
    "a placeholder. Every step-0-anchored delta below therefore compares an unmerged cell against "
    "merged bf16 cells, so it carries any merge artifact on top of the training effect -- caveat 3's "
    "attenuation applies to the merged side only."
)

POSITION_TABLE_PROSE = (
    "One row per step, game and payoff variant (framing-sweep games one row per framing), never "
    "pooled across variants, for every game whose completion names one of two printed labels. "
    "`picks-first` is the share of parsed picks that took whichever option was printed first, "
    "whatever it meant. The two `coop` columns are the cooperation rate on the prompt-renders "
    "whose cooperative label was printed first, and second, and `gap` is their difference with a "
    "paired between-prompt 2SE over the prompts rendered both ways (quadrature of the two sides "
    "where fewer than two were, e.g. a canonical-only cell). The pooled rate beside them is the "
    "same figure the tables above carry. Why this table exists: the roster counterbalances which "
    "authored label is cooperative, so a policy that picks whatever is printed first cooperates on "
    "exactly half the prompts and leaves the pooled rate unmoved, and the canonical-minus-swapped "
    "spread pairs each prompt's two orders so that a first-position preference contributes +x on "
    "half the prompts and -x on the other half and cancels -- the spread detects a preference for a "
    "WORD in a position, not for a position. A large gap beside a flat spread and an unmoved pooled "
    "rate is the signature; the wave-3 matched-size control showed gap +0.400 at step 70 against "
    "+0.155 at step 0 with both other figures quiet, and a large part of its cooperation rise was "
    "this heuristic. The derivation is `games.position_preference`."
)

DIAGNOSIS_CAVEAT = (
    "This is a description of what the traces contain. Nothing here is a verdict, a gate or a "
    "pass/fail, and where a number is surprising the first suspect is this code."
)

INCOMPLETE_BANNER = (
    "**THIS READOUT IS INCOMPLETE AND ERROR-CONTAINING.** The conditions below held when it was "
    "generated. Re-run the CLI once the remaining cells land and it updates itself."
)

CAVEATED_BANNER = (
    "**THIS READOUT IS COMPLETE BUT ERROR-CONTAINING.** Every arm reached the full ladder and no "
    "cell was excluded; the instrument caveats below still held when it was generated, and "
    "re-running cannot clear them."
)

type ContrastMeasure = Callable[[ArmReadout, int], str]


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as a markdown table, or say so when there are none.

    Row dicts rather than a frame: every table here is tens of rows of already-formatted strings,
    and the union of keys in order of first appearance is the column order a reader expects. An
    empty table says so rather than vanishing, because a missing section and an empty one are
    different findings.
    """
    if not rows:
        return "_(no rows)_"
    columns: list[str] = []
    for row in rows:
        columns.extend(str(key) for key in row if str(key) not in columns)
    header = f"| {' | '.join(columns)} |"
    divider = f"| {' | '.join('---' for _ in columns)} |"
    body = [
        f"| {' | '.join(_format_cell(row.get(column)) for column in columns)} |" for row in rows
    ]
    return "\n".join([header, divider, *body])


def _format_cell(value: object) -> str:
    """Format one table cell, escaping the pipe that would otherwise split it."""
    if value is None:
        return EMPTY_CELL
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def _header_lines(readout: BatteryReadout, *, command: str) -> list[str]:
    """Return the opening banners: what this is, when it ran, and how to regenerate it."""
    return [
        "# Games battery readout: per arm, full ladder",
        "",
        f"> {GENERATED_BANNER}",
        "",
        f"- Generated at `{readout.generated_at}`.",
        f"- Battery directory: `{readout.battery_dir}`.",
        (
            f"- Cells read: {readout.cell_count}; arms: {len(readout.arms)}; ladder derived from the "
            f"data: {steps_label(readout.ladder)}."
        ),
        f"- Regenerate with: `{command}`",
        "",
        "## How to read this",
        "",
        f"1. {SINGLE_CELL_CAVEAT}",
        f"2. {DENOMINATOR_CAVEAT}",
        f"3. {BF16_ATTENUATION_CAVEAT}",
        f"4. {BASE_CELL_CAVEAT}",
        f"5. {DIAGNOSIS_CAVEAT}",
        "",
        *_status_lines(readout),
    ]


def _status_lines(readout: BatteryReadout) -> list[str]:
    """Return the completeness banner, or the one-line all-clear when nothing fired."""
    if not readout.error_containing:
        return [
            (
                f"Every arm reached all {len(readout.ladder)} steps of the ladder, no cell was "
                f"excluded, and no provenance or parse-rate check fired."
            ),
            "",
        ]
    still_landing = bool(readout.incomplete_arms or readout.excluded)
    lines = [f"> {INCOMPLETE_BANNER if still_landing else CAVEATED_BANNER}", ""]
    lines.extend(
        f"- INCOMPLETE arm `{arm.arm}`: {len(arm.steps)} of {len(readout.ladder)} cells (present "
        f"{steps_label(arm.steps)}; missing {steps_label(arm.missing_steps)}). Its pooled late "
        f"window is {steps_label(arm.late_window)}, so read its late column as provisional."
        for arm in readout.arms
        if arm.incomplete
    )
    lines.extend(
        f"- EXCLUDED cell `{relative_to(cell.path, readout.battery_dir)}`: {cell.reason}."
        for cell in readout.excluded
    )
    lines.extend(f"- {problem}" for problem in readout.problems)
    lines.append("")
    return lines


def _inventory_lines(readout: BatteryReadout) -> list[str]:
    """Return the per-arm inventory and provenance table."""
    return [
        "## Inventory and provenance",
        "",
        (
            "One row per arm. `cells` counts what landed against the ladder above, which is the union "
            "of the steps the arms reached: a step no arm has reached yet cannot appear in it, so a "
            "battery uniformly behind schedule would read as complete here. The render-grading column "
            "is what the eval prompts were rendered under, which is not the arm's training grading."
        ),
        "",
        markdown_table([arm.facts for arm in readout.arms]),
        "",
    ]


def _arm_lines(arm: ArmReadout) -> list[str]:
    """Return one arm's whole section, in the order the tables should be read."""
    lines = [
        f"## Arm `{arm.arm}`" + (" -- INCOMPLETE" if arm.incomplete else ""),
        "",
        (
            f"Trained game `{arm.trained_game or UNREGISTERED}`; steps present "
            f"{steps_label(arm.steps)}; pooled late window {steps_label(arm.late_window)}."
        ),
        "",
    ]
    lines.extend(f"> {note}" for note in arm.notes)
    if arm.notes:
        lines.append("")
    lines.extend(
        [
            f"### `{arm.arm}`: trained-game trajectory",
            "",
            markdown_table(arm.trained_ladder),
            "",
            f"### `{arm.arm}`: transfer matrix, every game evaluated",
            "",
            (
                "The never-trained rows are the transfer readout. The negative-control row is the "
                f"token-level-generalisation tell: in `{NEGATIVE_CONTROL_GAME_ID}` the cooperative "
                "label pays nothing, so movement toward it says the policy learned the word rather "
                "than the structure."
            ),
            "",
            markdown_table(arm.transfer_matrix),
            "",
            f"### `{arm.arm}`: transfer, base cell against the pooled late window",
            "",
            (
                "The same games, read for the size of the move rather than the shape of the curve. "
                "The base side is one cell, so a move here is only as trustworthy as the ladder "
                "above it agrees."
            ),
            "",
            markdown_table(arm.transfer_windows),
            "",
            f"### `{arm.arm}`: negative control ladder (`{NEGATIVE_CONTROL_GAME_ID}`)",
            "",
            markdown_table(arm.negative_control),
            "",
            f"### `{arm.arm}`: decision-theory probes",
            "",
            markdown_table(arm.dt_ladder),
            "",
            f"### `{arm.arm}`: theory endorsement per multiple-choice render",
            "",
            (
                "One column per endorsement pattern seen on this arm's ladder: the theories "
                "compatible with the chosen option, `+`-joined; `neither` means the choice matched "
                "none of the three. This is the reduction `mean_edt_leaning` cannot carry -- an "
                "FDT-endorsed answer and one every theory endorses both read as leaning zero there, "
                "and separate here."
            ),
            "",
            markdown_table(arm.dt_endorsements),
            "",
            _flip_prose(arm),
            "",
            markdown_table(arm.dt_flips),
            "",
            f"### `{arm.arm}`: capability canary (arithmetic)",
            "",
            markdown_table(arm.capability_ladder),
            "",
            f"### `{arm.arm}`: self-report composites",
            "",
            (
                "One row per instrument and subscale, because the scales are not comparable with each "
                "other: five Likert points against six, against points handed to another party on "
                "the two allocation instruments. Read the delta down one instrument's rows, and read "
                "every row as a response-policy score rather than a personality score -- independent "
                "completions are not one psychometric respondent. Every number here is the model's "
                "account of itself, and models over-report; the calibration table below is the one "
                "place this section scores the artifact instead. Neutral-twin renders never pool "
                "into these composites -- their gap is its own table row below."
            ),
            "",
            markdown_table(arm.self_report_ladder),
            "",
            f"### `{arm.arm}`: self-report response-style controls",
            "",
            (
                "Two ways a composite above moves without a disposition moving. Acquiescence is "
                "yea-saying: agreeing with a statement and with its negation, which raises a "
                "cooperativeness scale and lowers a reverse-keyed competitiveness scale at once and "
                "so mimics exactly the shift this battery looks for -- reported as a response-style "
                "diagnostic, never applied as a trait correction. It reads `-` with a reason on "
                "any subscale carrying only one keying direction, which is most of them. The wording "
                "gap is the published item against its lexically neutral twin: a shift present in "
                "the instrument's own vocabulary and absent in the twin is priming."
            ),
            "",
            markdown_table(arm.self_report_controls),
            "",
            f"### `{arm.arm}`: self-report instrument health (the denominators are what move)",
            "",
            markdown_table(arm.self_report_health),
            "",
            f"### `{arm.arm}`: self-report instrument-specific readouts",
            "",
            markdown_table(arm.self_report_instruments),
            "",
            f"### `{arm.arm}`: negative controls, per item (distribution distance from step 0)",
            "",
            (
                "The controls' own scoring: nominal options have no ordering, so movement is total "
                "variation distance from step 0 per item, with entropy as each item's available "
                "headroom. A moving control does not void the battery -- it flags nonspecific "
                "drift, and the readable quantity becomes each target instrument's movement in "
                "excess of the matched controls'. The `inert-fact` rows are factual-integrity "
                "checks, read separately from the preference placebos by design. The "
                "negative-control family and nothing else: two target families answer in this same "
                "lettered shape and take the same reading, and they are in the table below rather "
                "than here, because a values item moving is that family's registered result while a "
                "control moving is a threat to every other reading in the document."
            ),
            "",
            markdown_table(arm.self_report_control_distributions),
            "",
            f"### `{arm.arm}`: nominal-choice items outside the controls, per item",
            "",
            (
                "The same per-item reading -- distribution, headroom, distance from step 0 -- for every "
                "target family that answers with a lettered choice, which is the reading two of them "
                "register. These rows are NOT placebos and a move in them is not nonspecific drift: "
                "the graded-dimension items are read as what the policy thinks it is being graded on, "
                "and against their own tag mirrors, while the values items are read as a preference "
                "through the ordering table below and appear here as the per-item detail behind it. "
                "The distribution is keyed by option label wherever the item carries labels, because "
                "the values items vary their label order item by item on purpose, so option 0 names a "
                "different pole on each item of a pole pair and a distribution over option numbers "
                "would read as a preference reversal between the two."
            ),
            "",
            markdown_table(arm.self_report_nominal_choices),
            "",
            f"### `{arm.arm}`: value poles, the forced-choice preference ordering",
            "",
            (
                "The values family's deliverable: not any single item's rate but an ordering over the "
                "five poles, read through the option labels rather than the option numbers. Every item "
                "offers exactly two poles, so 0.5 is chance on every row and a rate is read against "
                "that. A forced choice carries no acquiescence to absorb and no scale to drift, which "
                "is what makes a moving ordering a stronger claim than a moving Likert mean. Both "
                "denominators are printed because a rate over two items and a rate over twenty read "
                "identically otherwise. `rank` is within the step and over the pole rows only, with "
                "ties sharing a rank, since two poles chosen at the same rate are not ordered. The "
                "three `non-social-` goods are offered by two, three and three items, so their pole "
                "row is recomputed over all of their items rather than averaged from those three "
                "rates, and the per-good rows stay visible marked as its components. A cell whose "
                "values items parsed nothing has no rows here, and reads as its parse rate in the "
                "instrument-health table above."
            ),
            "",
            markdown_table(arm.self_report_values_ordering),
            "",
            f"### `{arm.arm}`: numeric items, per item (canonical means and order gap)",
            "",
            (
                "Self-predictions and the numeric placebo, one row per item and never pooled: each "
                "predicts a different quantity. Both order columns are canonical (a swapped "
                "render's answer is reflected through `maximum - x` at parse time), so `order gap` "
                "is the first-position bias measured on the survey itself -- zero for a policy "
                "with a real rate and no position preference."
            ),
            "",
            markdown_table(arm.self_report_numeric),
            "",
            f"### `{arm.arm}`: mirrored gamble ladders (loss aversion)",
            "",
            (
                f"Two five-rung gamble ladders that are exact translations of each other -- "
                f"`{RISK_LOSS_AVERSION_MIRROR_PAIR[0]}` against "
                f"`{RISK_LOSS_AVERSION_MIRROR_PAIR[1]}`. Every outcome of the mixed ladder is the "
                f"gain-only ladder's plus a constant, so the spreads and the probabilities are "
                f"identical and only the sign of the low branch differs; rung 1 is the certain "
                f"outcome and rung 5 the widest spread, so the gap between the two mean rungs is "
                f"what the policy pays to avoid a loss. The pair is named in the code because the "
                f"two items sit in different instruments and different subscales on purpose -- a "
                f"pure-gain ladder averaged into a loss composite would be wrong -- so nothing in "
                f"the registry connects them. Registered expectation: the gap is POSITIVE at step "
                f"0, and no direction is called on how it moves; this table is descriptive. The "
                f"mixed ladder is a within-family placebo, since no wave-1 training payoff is ever "
                f"negative and so nothing in training could have taught a preference about losses. "
                f"Read a move in it as a moving negative control is read: the readable quantity for "
                f"the variance-tolerance subscale becomes its movement in excess of the placebo's "
                f"own."
            ),
            "",
            markdown_table(arm.self_report_risk_mirror),
            "",
            f"### `{arm.arm}`: counterpart gaps, an AI counterpart against a human one",
            "",
            (
                "Each pair is one item asked twice, differing only in who the counterpart is, and "
                "the difference is the whole quantity: subtracting the two arms cancels whatever "
                "the policy's willingness to allocate or trust happens to be and leaves the effect "
                "of who it was dealing with. One arm's level is not a reading -- every level in "
                "this section carries the over-reporting the composites above cannot remove, and "
                "the subtraction is what removes it. A gap reading `-` carries its recorded reason, "
                "because a nominal pair with nothing to difference and a pair where nothing parsed "
                "are different facts about the run and only one of them is a problem with it."
            ),
            "",
            markdown_table(arm.self_report_counterparts),
            "",
            f"### `{arm.arm}`: cheap talk, announcement against action",
            "",
            (
                "One completion carrying two answers: the intention the policy announced to its "
                "counterpart, and the action it then took. Both are read mechanically out of their "
                "own tags, with no judge over the prose. One row per item and never pooled -- the "
                "items describe different situations with different stakes, so a mean over how "
                "often it did what it said is not a rate of anything. `announced only` counts "
                "completions that announced and then emitted no action, which is a coverage fact "
                "about the format rather than dishonesty; the `announced -> acted` counts carry the "
                "direction of each mismatch, which the match rate alone cannot."
            ),
            "",
            markdown_table(arm.self_report_cheap_talk),
            "",
            f"### `{arm.arm}`: forced-tag items, the distribution over each item's word menu",
            "",
            (
                "One word from a closed menu, read mechanically out of its tag with no judge over the "
                "prose. The reading is the distribution over those words and never a mean over menu "
                "positions, which would let the authored order of the menu decide how large a "
                "movement looks, so movement is total variation distance from step 0 per item -- the "
                "same scoring the nominal controls get. That is what makes each tag mirror comparable "
                "with the lettered item it mirrors: a distribution that moves on the lettered item and "
                "not on its mirror is a response to the lettered format rather than an attribution. "
                "Entropy is the item's own headroom, because an item answered with one word in nearly "
                "every sample has no range in which to show a shift, and `menu words` is how many "
                "words it had to choose between -- a 0.33 share is the flat answer on three and a peak "
                "on six, and both widths are in this battery. A `TV vs step0` reading `-` carries its "
                "recorded reason: an item the base cell never asked, a cell where nothing parsed, and "
                "an item whose menu was re-authored between the two passes are three different facts, "
                "and a distance computed across a changed menu would be meaningless rather than large."
            ),
            "",
            markdown_table(arm.self_report_tagged),
            "",
            f"### `{arm.arm}`: self-prediction against measured behaviour",
            "",
            (
                "What the checkpoint says it does minus what it did, both sides from the same cell. "
                "A missing measured side means the behaviour section was not run or was restricted "
                "with `--games`, which is a coverage fact about the run rather than a result."
            ),
            "",
            markdown_table(arm.self_report_calibration),
            "",
            f"### `{arm.arm}`: trained game split by payoff variant",
            "",
            markdown_table(arm.variant_splits),
            "",
            f"### `{arm.arm}`: trained game split by reskin (surface frame)",
            "",
            (
                "Each reskin is the same payoff matrix under a different cover story. A split that "
                "moves on one frame only is a frame-level response rather than a strategic one."
            ),
            "",
            markdown_table(arm.reskin_splits),
            "",
            f"### `{arm.arm}`: cooperation by printed position of the cooperative label",
            "",
            POSITION_TABLE_PROSE,
            "",
            markdown_table(arm.position_splits),
            "",
            f"### `{arm.arm}`: per-round profile, iterated game (`{ITERATED_GAME_ID}`)",
            "",
            markdown_table(arm.per_round),
            "",
            f"### `{arm.arm}`: section health (the denominators are what move)",
            "",
            markdown_table(arm.section_health),
            "",
        ]
    )
    return lines


def _flip_prose(arm: ArmReadout) -> str:
    """Return the sentences a flip table cannot be read without: endpoints, denominator, floor."""
    if len(arm.steps) < MIN_CELLS_FOR_A_FLIP:
        return (
            f"Per-item endorsement flips need two cells; `{arm.arm}` has {len(arm.steps)}, so "
            f"there is nothing to compare yet."
        )
    endpoint = (
        " -- NOT the end of training, this arm is still landing cells" if arm.incomplete else ""
    )
    scope = arm.flip_scope
    return (
        f"Per-item endorsement flips, comparing step {arm.steps[0]} against step {arm.steps[-1]}"
        f"{endpoint}. {len(arm.dt_flips)} item(s) changed endorsement, out of {scope.n_comparable} "
        f"comparable at both endpoints (asked at both and settled to one answer at each; "
        f"{scope.n_set_aside} of the {scope.n_asked_both} items asked at both endpoints were set "
        f"aside as unscored or order-disagreeing). An empty table is a result rather than a missing "
        f"measurement. {_churn_prose(arm)}"
    )


def _churn_prose(arm: ArmReadout) -> str:
    """Return the noise floor the flip table must clear: churn between neighbouring checkpoints."""
    churn = arm.churn
    if churn.rate is None:
        return (
            "No adjacent-checkpoint churn floor is computable: no item was comparable between any "
            "pair of neighbouring cells."
        )
    if churn.n_step_pairs == 1:
        return (
            f"The adjacent-checkpoint churn floor and this comparison are the same two cells "
            f"({churn.n_flips}/{churn.n_comparable} items), so the floor separates nothing yet."
        )
    return (
        f"Adjacent-checkpoint churn floor for this statistic: {churn.n_flips}/{churn.n_comparable} "
        f"item-comparisons changed endorsement across {churn.n_step_pairs} neighbouring-step pairs "
        f"= {churn.rate:.3f}. A first-to-last flip rate at or below that floor is churn, not "
        f"signal."
    )


def _contrast_lines(readout: BatteryReadout) -> list[str]:
    """Return the side-by-side sections for every group of arms sharing a trained game."""
    if not readout.contrasts:
        return []
    by_arm = {arm.arm: arm for arm in readout.arms}
    lines = [
        "## Contrasts: arms that trained the same game",
        "",
        (
            "Derived by grouping the arms present onto their trained game, so the pair whose only "
            "difference is the grading rule and the arms differing only in pinned payoffs both appear "
            "without either being named in the code. Each table is one measure, one row per step, one "
            "column per arm; a `-` means that arm has not landed that step."
        ),
        "",
    ]
    for group in readout.contrasts:
        members = [by_arm[arm] for arm in group.arms if arm in by_arm]
        lines.extend(
            [
                f"### `{group.game_id}`: {', '.join(f'`{arm}`' for arm in group.arms)}",
                "",
                (
                    f"Axis that differs: {group.axis}. Check the inventory's render-grading column "
                    f"before reading a difference below as a behaviour difference: where the arms share "
                    f"a rendering, the prompts they were asked are identical."
                ),
                "",
                *_shared_draw_lines(group),
            ]
        )
        for title, measure in (
            (f"trained game `{group.game_id}`", _contrast_trained),
            (f"negative control `{NEGATIVE_CONTROL_GAME_ID}`", _contrast_negative_control),
            ("mean EDT leaning", _contrast_edt),
            ("prosocial choice rate", _contrast_prosocial),
            ("order disagreement rate", _contrast_order),
            ("arithmetic canary, accuracy over asked", _contrast_canary),
        ):
            lines.extend(
                [f"**{title}**", "", markdown_table(_contrast_table(members, measure)), ""]
            )
        lines.extend(
            [
                "**transfer, base step 0 to each arm's own pooled late window**",
                "",
                (
                    "One row per game, one column per arm, each cell the move between that arm's "
                    "base cell and its late window (named in the column, because arms still landing "
                    "have shorter windows). The rates behind these moves are in each arm's own "
                    "pooled transfer table below."
                ),
                "",
                markdown_table(_contrast_transfer_rows(members)),
                "",
            ]
        )
    return lines


def _shared_draw_lines(group: ContrastGroup) -> list[str]:
    """Name the rows of this contrast that are one sampling draw printed twice.

    Identical prompts are still two draws -- except where the completions themselves are
    byte-identical, which happens for real at step 0 (deterministic per-prompt sampling of the
    shared base weights). A reader who is not told reads those rows as agreement at baseline.
    """
    if not group.shared_draws:
        return []
    lines = [
        (
            "**Shared draws.** Hashing each cell's completions per section found the cells below "
            "byte-identical across arms: their rows in the tables that follow are one sampling "
            "draw printed once per column, equal by construction rather than in agreement. "
            "Sections and steps not listed are independent draws."
        ),
        "",
    ]
    lines.extend(
        f"- step {draw.step}, section `{draw.section}`: all {draw.n_records} completions "
        f"byte-identical across {', '.join(f'`{arm}`' for arm in draw.arms)}."
        for draw in group.shared_draws
    )
    lines.append("")
    return lines


def _contrast_transfer_rows(members: Sequence[ArmReadout]) -> list[dict[str, Any]]:
    """Return every game's base-to-late move for each arm of one contrast, side by side."""
    by_arm_game = {
        (arm.arm, str(row[GAME_COLUMN])): row for arm in members for row in arm.transfer_windows
    }
    games = sorted({game for _, game in by_arm_game})
    rows: list[dict[str, Any]] = []
    for game in games:
        sources = [by_arm_game.get((arm.arm, game)) for arm in members]
        landed = [source for source in sources if source is not None]
        row: dict[str, Any] = {
            GAME_COLUMN: game,
            "role": str(landed[0]["role"]) if landed else EMPTY_CELL,
        }
        for arm, source in zip(members, sources, strict=True):
            row[f"{arm.arm} (late {steps_label(arm.late_window)})"] = (
                EMPTY_CELL if source is None else str(source[DELTA_COLUMN])
            )
        rows.append(row)
    return sorted(rows, key=lambda row: (str(row["role"]) != ROLE_TRAINED, str(row[GAME_COLUMN])))


def _contrast_table(
    members: Sequence[ArmReadout], measure: ContrastMeasure
) -> list[dict[str, Any]]:
    """Return one measure for several arms side by side, one row per step of the shared ladder."""
    steps = sorted({step for arm in members for step in arm.steps})
    rows: list[dict[str, Any]] = []
    for step in steps:
        row: dict[str, Any] = {STEP_COLUMN: step}
        for arm in members:
            row[arm.arm] = measure(arm, step)
        rows.append(row)
    return rows


def _row_at_step(rows: Sequence[Mapping[str, Any]], step: int) -> Mapping[str, Any] | None:
    """Return the first row for one step, or None when that step has not landed."""
    return next((row for row in rows if row.get(STEP_COLUMN) == step), None)


def _contrast_trained(arm: ArmReadout, step: int) -> str:
    """Return this arm's trained-game cell at one step, variants joined where it pins any."""
    rows = [row for row in arm.trained_ladder if row.get(STEP_COLUMN) == step]
    if not rows:
        return EMPTY_CELL
    return " / ".join(
        str(row[RATE_COLUMN])
        if row[VARIANT_FIELD] == ALL_VARIANTS
        else f"{row[VARIANT_FIELD]}: {row[RATE_COLUMN]}"
        for row in rows
    )


def _contrast_negative_control(arm: ArmReadout, step: int) -> str:
    """Return this arm's negative-control cell at one step."""
    row = _row_at_step(arm.negative_control, step)
    return EMPTY_CELL if row is None else str(row[CONTROL_RATE_COLUMN])


def _contrast_edt(arm: ArmReadout, step: int) -> str:
    """Return this arm's mean EDT leaning at one step."""
    row = _row_at_step(arm.dt_ladder, step)
    return EMPTY_CELL if row is None else str(row[EDT_COLUMN])


def _contrast_prosocial(arm: ArmReadout, step: int) -> str:
    """Return this arm's prosocial choice rate at one step, the became-agreeable control."""
    row = _row_at_step(arm.dt_ladder, step)
    return EMPTY_CELL if row is None else str(row[PROSOCIAL_COLUMN])


def _contrast_order(arm: ArmReadout, step: int) -> str:
    """Return this arm's order-disagreement rate at one step, the letter-bias control."""
    row = _row_at_step(arm.dt_ladder, step)
    return EMPTY_CELL if row is None else str(row[ORDER_COLUMN])


def _contrast_canary(arm: ArmReadout, step: int) -> str:
    """Return this arm's arithmetic accuracy over asked items at one step."""
    row = _row_at_step(arm.capability_ladder, step)
    return EMPTY_CELL if row is None else str(row[ACCURACY_ASKED_COLUMN])


def render_markdown(readout: BatteryReadout, *, command: str) -> str:
    """Render the whole readout as one markdown document.

    Ordered so the banners and the inventory come first, then the cross-arm contrasts, then each
    arm in full: what is missing is the thing a reader has to know before any number, and the
    contrast between arms is the wave's designed comparison, so it precedes the per-arm detail it
    is drawn from.
    """
    sections = [
        *_header_lines(readout, command=command),
        *_inventory_lines(readout),
        *_contrast_lines(readout),
    ]
    for arm in readout.arms:
        sections.extend(_arm_lines(arm))
    return "\n".join(sections)


def default_out_path(battery_dir: Path) -> Path:
    """Return where this battery's readout lands when the caller names no path.

    Derived from the battery's own directory name, so two batteries cannot overwrite each other's
    document, and under `docs/scratch/` because that is the one location internal working material
    is allowed to occupy in this repository.
    """
    stem = battery_dir.name.removeprefix(BATTERY_DIR_PREFIX)
    return DEFAULT_OUT_DIR / f"games-readout-{stem}.md"


def regeneration_command(battery_dir: Path, out_path: Path) -> str:
    """Return the exact command that reproduces this document, for its own header."""
    return (
        f"uv run --frozen python -m games.battery_readout "
        f"--battery-dir {battery_dir} --out {out_path}"
    )


# The only repo locations a readout may land in; both are gitignored. See `_guard_out_path`.
ALLOWED_OUT_ROOTS: tuple[str, ...] = ("docs/scratch", "artifacts")


def _guard_out_path(out_path: Path) -> None:
    """Refuse an `--out` that would put per-item probe results on a trackable path.

    The flip table prints real probe ids with their before/after endorsements -- per-item results
    on the decision-theory battery, which the privacy rules forbid committing. Paths outside the
    repository are fine (tests write to temp dirs); inside it, only the gitignored scratch and
    artifact trees are.
    """
    repo_root = Path.cwd().resolve()
    resolved = out_path.resolve()
    if not resolved.is_relative_to(repo_root):
        return
    if any(resolved.is_relative_to(repo_root / root) for root in ALLOWED_OUT_ROOTS):
        return
    raise ValueError(
        f"--out {out_path} is inside the repository but outside {ALLOWED_OUT_ROOTS}. The flip "
        f"table carries per-item probe results, which must never land on a trackable path."
    )


def write_readout(battery_dir: Path, out_path: Path) -> Path:
    """Build one battery's readout, write the markdown, and return where it landed."""
    _guard_out_path(out_path)
    readout = build_readout(battery_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        render_markdown(readout, command=regeneration_command(battery_dir, out_path)),
        encoding="utf-8",
    )
    logger.info(
        f"wrote {out_path} from {readout.cell_count} cell(s) over {len(readout.arms)} arm(s); "
        f"error_containing={readout.error_containing} incomplete={readout.incomplete_arms}"
    )
    return out_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Recompute the per-arm games battery readout from the eval traces on disk."
    )
    parser.add_argument(
        "--battery-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the battery's cells, at any depth below it. Defaults to the most "
            f"recently written {BATTERY_DIR_GLOB!r} directory under {DEFAULT_EVALS_ROOT}."
        ),
    )
    parser.add_argument(
        "--evals-root",
        type=Path,
        default=DEFAULT_EVALS_ROOT,
        help="Where to look for batteries when --battery-dir is not given.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Where to write the markdown. Defaults to a name derived from the battery directory, "
            f"under {DEFAULT_OUT_DIR}."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Build one battery's readout and write it where the caller asked."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    battery_dir: Path = args.battery_dir or latest_battery_dir(args.evals_root)
    write_readout(battery_dir, args.out or default_out_path(battery_dir))


if __name__ == "__main__":
    main()
