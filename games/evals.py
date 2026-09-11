"""Run the eval battery against any backend and write a self-describing JSONL trace.

Four sections, each answering a different question about a checkpoint: how it *plays* the games
(including games it was never trained on), which decision theory it *endorses*, what it *says about
itself* on published self-report instruments, and whether it can still do arithmetic. Every section
runs through the same `Backend` seam -- prompt strings in, completions out -- so a base model, a
merged checkpoint, and a scripted mock all evaluate through one code path.

The self-report section is the one outside `DEFAULT_SECTIONS`, and asking for it is explicit: it is
the largest section by render count, and ALL of its item text -- published instruments and our own
authored items alike -- reads from local files this public repository does not carry. It refuses to
run without them; a fresh clone runs nothing rather than a quietly smaller battery. See
`games/survey.py` and `games/data/survey/README.md`.

Two design commitments worth stating outright.

**Item-level records, not just aggregates.** The effect this project is chasing showed up in the
source post as 24 of 120 items flipping inside a noisy mean. An eval that writes only rates cannot
recover that, and re-running to get it back costs GPU hours, so every record carries the full
completion text and enough identity (`probe_id`, `prompt_id`, `reskin_id`, sample index) to line
up before-and-after per item.

**No LLM judge anywhere.** Actions come from tag parsing, theories from string matching, and
arithmetic from an exact integer comparison. A judge model in the scoring path would make every
number a measurement of two models at once.

On contamination: DTBench items reach this module as text and are written into the eval trace,
which is why the trace belongs under `artifacts/`, gitignored in this public repo. The corpus ships
encrypted with a BigBench canary to stay out of training data, so nothing here may be committed --
see `docs/scratch/dtbench-availability-2026-08-17.md` and `games/probes.py`.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import IO, TYPE_CHECKING, Any, cast

from games.chunked_decode import (
    backend_schedules_own_batch,
    decode_in_chunks,
    local_decode_model_id,
    stream_vllm_completions,
    sweep_chunk_size,
)
from games.framing_stimulus import (
    RuntimeFramings,
    load_dictator_recipient_clauses,
    resolve_framing_clause,
)
from games.parsing import (
    RETURN_PERCENTAGE_MAX,
    parse_action,
    parse_action_sequence,
    parse_claim,
    parse_contribution,
    parse_level,
    parse_level_sequence,
    parse_return_percentage,
    parse_send,
    parse_split,
    parse_theory,
    parse_transfer_figure,
    parse_trust_strategy,
    strip_thinking,
)
from games.payoffs import COOPERATE
from games.probes import (
    OPEN_ENDED_ITEMS,
    PROBE_OPEN_ENDED,
    ProbeItem,
    compatible_theories,
    counterbalanced_option_orders,
    edt_leaning_score,
    parse_final_answer,
    probe_battery,
    render_probe_prompt,
)
from games.prompts import (
    ALL_GAME_IDS,
    CORPUS_BUILT_GAME_IDS,
    COUNTERPART_FRAMING_IDS,
    DICTATOR_GAME_ID,
    EVAL_ONLY_GAME_IDS,
    EVAL_ONLY_MATRIX_GAME_IDS,
    FRAMEABLE_GAME_IDS,
    GAME_IDS,
    ITERATED_GAME_ID,
    ITERATED_GAME_IDS,
    ITERATED_PD_GRIM_GAME_ID,
    ITERATED_STAG_GAME_ID,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDERS,
    MATRIX_GAME_IDS,
    MIN_EFFORT_GAME_ID,
    MIN_EFFORT_GAME_IDS,
    MIN_EFFORT_MATCH_GAME_ID,
    NASH_DEMAND_GAME_ID,
    NO_EXTRA_EVAL_FRAMES,
    PROBE_ONLY_GAME_IDS,
    RENDERABLE_GAME_IDS,
    SPLIT_EVAL,
    THRESHOLD_GOODS_GAME_ID,
    TRUST_STATED_RETURN_GAME_ID,
    TRUST_STRATEGY_METHOD_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
    UNLABELLED_GAME_IDS,
    ExtraEvalFrames,
    frame_label_audit,
    generate_counterpart_clause_prompt_rows,
    generate_prompt_rows,
)
from games.provenance import git_sha
from games.survey import (
    FAMILIES,
    NON_SOCIAL_LABEL_PREFIX,
    PUBLISHED_INSTRUMENTS,
    SURVEY_NUMERIC,
    TIERS,
    SurveyItem,
    acquiescence_index,
    battery_orders,
    calibration_gaps,
    choice_response_distributions,
    choice_response_entropy,
    families_with_items,
    forced_choice_prefix_win_rate,
    forced_choice_win_rates,
    instrument_composites,
    modal_choices,
    numeric_example_rotation,
    numeric_item_readings,
    orientation_counts,
    parse_rate_by_family,
    parse_rate_by_instrument,
    parse_survey_answer,
    render_survey_prompt,
    subscale_composites,
    survey_battery,
    survey_record_fields,
    svo_angle,
    svo_mean_completion_angle,
    tagged_readings,
    wording_gap,
)
from games.termination import required_completion_budget
from games.trap_cells import render_dictator_recipient_rows
from grpo.rlvr_math import gen_ltr_arithmetic, parse_answer
from reward_hacking.model_backend import HFBackend, VLLMBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from games.held_out_extension import ExtensionRosters
    from reward_hacking.model_backend import Backend

logger = logging.getLogger(__name__)

SECTION_GAME_BEHAVIOR = "game-behavior"
SECTION_DT_PROBES = "dt-probes"
SECTION_CAPABILITIES = "capabilities"
SECTION_SELF_REPORT = "self-report"
SECTION_FRAMING_SWEEP = "framing-sweep"
# The trap cells: eval prompts built so that one reading of a behavioural rise is ruled out by
# construction (`games.trap_cells`). Its own section rather than rows inside game-behavior, because
# a trap cell's prompt_id and inserted paragraph are ones that section never renders, and a reader
# pooling them would average a trap into the plain dictator rate -- the framing sweep's reasoning.
SECTION_TRAP_CELLS = "trap-cells"
SECTIONS: tuple[str, ...] = (
    SECTION_GAME_BEHAVIOR,
    SECTION_DT_PROBES,
    SECTION_CAPABILITIES,
    SECTION_SELF_REPORT,
    SECTION_FRAMING_SWEEP,
    SECTION_TRAP_CELLS,
)

# The counterpart-framing sweep's two games: the trained game and the transfer game the 9B ladder
# measured as moving at full trained-game magnitude under the twin framing (never-trained at the
# time of that measurement; public-goods was promoted to trainable on 2026-08-26 for the
# transfer-of-learning experiment, which does not change what this instrument renders). Both render
# through the one-shot matrix path, which is the path whose counterpart paragraph the framings
# swap; the import-time check below keeps a renamed game id from becoming a GPU-time error.
FRAMING_SWEEP_GAME_IDS: tuple[str, ...] = ("twin-pd", "public-goods")

_unframeable = sorted(set(FRAMING_SWEEP_GAME_IDS) - set(FRAMEABLE_GAME_IDS))
if _unframeable:
    raise ImportError(
        f"FRAMING_SWEEP_GAME_IDS names {_unframeable}, which the framing renderer cannot serve; "
        f"frameable games: {sorted(FRAMEABLE_GAME_IDS)}."
    )

# What a battery runs when the caller names no sections. The self-report section is opt-in for two
# reasons, neither of which is that it might fail: it needs local item files a fresh clone does not
# have (`games/data/survey/README.md`), and it is the largest section by render count, so switching
# it on by default would change the cost and the sampling budget of every launch that already
# exists. Ask for it explicitly and its instruments land in the trace's meta.
DEFAULT_SECTIONS: tuple[str, ...] = (
    SECTION_GAME_BEHAVIOR,
    SECTION_DT_PROBES,
    SECTION_CAPABILITIES,
)

RECORD_META = "meta"

RECORD_TRAINING_FRAMES = "training-frames"
SECTIONS_TRAINING_FRAMES: tuple[str, ...] = (RECORD_TRAINING_FRAMES,)
"""The one record kind written by a driver other than the battery's, and the section list it declares.

`games.eval_training_frames` asks a checkpoint the exact prompts it trained on; its records are
battery game records restamped under this kind so they can never pool with the battery's held-out
`game-behavior` rows. The kind lives here, beside the sections, because every trace helper in this
module -- `record_identity`, `inspect_trace`, `summarise_trace` -- dispatches on the record kind,
and a kind registered from the module that writes it would exist only once that module had been
imported: a reader keying frames records without that import would find no identity fields.
"""

# What `--label-print-order` accepts beyond the two renderings themselves: run both and stamp each
# record with the order it was rendered under. A battery concept rather than a rendering, which is
# why `games.prompts` does not carry it -- there is no "both" way to print a pair of labels.
LABEL_PRINT_ORDER_BOTH = "both"
LABEL_PRINT_ORDER_REQUESTS: tuple[str, ...] = (*LABEL_PRINT_ORDERS, LABEL_PRINT_ORDER_BOTH)

# What an open-ended item's record carries where a multiple-choice one names its option order.
ORDER_NOT_APPLICABLE = "no-options"

CAPABILITY_INSTRUCTION = (
    "Reply with the integer result. Put it on a line of its own at the end of your reply, "
    "with nothing else on that line."
)

# The grading rule each game's eval rows are RENDERED under, which is a corpus input rather than a
# statement about the arm: `twin-pd-group` and `twin-pd-self` share game `twin-pd` under different
# gradings, so one map from game to grading cannot name the arm being evaluated. The arm's own
# grading reaches the trace through the caller's meta, beside `arm`.
EVAL_RENDER_GRADING_BY_GAME: dict[str, str] = {
    "twin-pd": "group-mix",
    "fixed-pie-pd": "group-mix",
    "stag-hunt": "group-mix",
    "hi-lo": "group-mix",
    "harmony": "group-mix",
    "chicken": "group-mix",
    # Either grading of the unstated pair renders the same paragraph-less prompt, so which one a
    # battery renders under does not change what the model reads; group-mix mirrors twin-pd's entry.
    "pd-unstated": "group-mix",
    # Same reasoning for the reskin game: the render is grading-independent, so this mirrors
    # pd-unstated's entry rather than the trained arm's self grading. Its eval split is the 12
    # held-out skins -- the arm's own generalization probe, riding in every battery from here on.
    "pd-reskin": "group-mix",
    # Promoted from the eval-only set 2026-08-26 (the transfer-of-learning pair trains both its
    # gradings); group-mix is the grading its eval rows rendered under while it was eval-only, so
    # the rendered prompt bytes do not move with the promotion.
    "public-goods": "group-mix",
    "pd-vs-frozen": "vs-fixed-mix",
    "stag-hunt-vs-frozen": "vs-fixed-mix",
    ITERATED_GAME_ID: "iterated-return",
    ITERATED_PD_GRIM_GAME_ID: "iterated-return",
    ITERATED_STAG_GAME_ID: "iterated-return",
    DICTATOR_GAME_ID: "keep-fraction",
    # Either claim grading renders the same prompt, so which one a battery renders under does not
    # change what the model reads; the group-mix name is the one the pair's headline arm carries.
    NASH_DEMAND_GAME_ID: "nash-demand-group-mix",
    # Either shared-undertaking grading renders the same prompt too, so which one a battery renders under
    # does not change what the model reads; the group-mix name is the one its two pinned arms carry.
    THRESHOLD_GOODS_GAME_ID: "threshold-goods-group-mix",
    TRUST_STATED_RETURN_GAME_ID: "trustor-payoff-stated-rule",
    TRUST_STRATEGY_METHOD_GAME_ID: "trustor-payoff-self-rule",
    MIN_EFFORT_GAME_ID: "min-effort-group-mix",
    MIN_EFFORT_MATCH_GAME_ID: "level-match-return",
}

# The eval-only games are never graded by GRPO, but the corpus schema still carries a grading
# column, so they take the one their structure would have had if they were ever trained.
EVAL_ONLY_GRADING = "group-mix"


def _assert_grading_map_covers_every_game() -> None:
    """Raise unless the grading map and the prompt registry name exactly the same games.

    Run at import. A game added to `games.prompts` but not here would silently drop out of the
    cross-game matrix, and a game renamed there would leave a stale key that reads as coverage --
    both of which look like a complete eval and are not.
    """
    missing = sorted(set(GAME_IDS) - set(EVAL_RENDER_GRADING_BY_GAME))
    extra = sorted(set(EVAL_RENDER_GRADING_BY_GAME) - set(GAME_IDS))
    if missing or extra:
        raise RuntimeError(
            f"EVAL_RENDER_GRADING_BY_GAME disagrees with games.prompts.GAME_IDS: "
            f"{missing=} {extra=}. Update the map so every game is evaluated."
        )


_assert_grading_map_covers_every_game()


# Which record field carries each game's behaviour rate. One map, imported by both readouts
# (`games.battery_tables` and `games.readout`), which is why it lives here: those two do not import
# each other, so each kept its own copy, and after wave 2 the copies disagreed -- the claim game in
# one, the trust games in the other, each readout rendering an empty table for the other's games.
COOP_FIELD = "coop_fraction"
# Which counterpart framing a record's prompt carried. Written by the framing sweep, by the trap cells
# and, since wave 4b, by the training-frames cell for a corpus whose rows carry a framing of their own:
# the field is named here rather than restated per writer so a sweep record and a trained-frames record
# describe one framing under one key, which is what lets a framing table put them side by side.
COUNTERPART_FRAMING_FIELD = "counterpart_framing"
KEEP_FIELD = "keep_fraction"
CLAIM_FIELD = "claim_fraction"
CONTRIBUTION_FIELD = "contribution_fraction"
SEND_FIELD = "send_fraction"
RETURN_FIELD = "return_fraction"
LEVEL_FIELD = "level_fraction"
SET_DOWN_FIELD = "set_down_fraction"

# The games whose completion names one of two printed labels, so their behaviour IS a cooperation
# rate: everything rendered through the one-shot matrix path, plus the repeated matrix arms, whose
# rate is over their move sequence, plus the corpus-built games -- their prompts come from corpus
# files (games.track_record_corpus builds them; games.eval_training_frames reads them back) rather
# than the roster renderer, but every one is a one-shot matrix game whose completion carries an
# action tag. Grouped by which RENDERER built the prompt rather than seeded from the whole grading
# map, and that distinction is the point -- see the disjointness check below.
ACTION_RATE_GAME_IDS: frozenset[str] = frozenset(
    {*MATRIX_GAME_IDS, *EVAL_ONLY_MATRIX_GAME_IDS, *ITERATED_GAME_IDS, *CORPUS_BUILT_GAME_IDS}
)

# The two-label games whose completion names ONE action, read by `_action_game_record`. The only
# records a printed-position split (`games.position_preference`) is defined on: the iterated games
# have a cooperation rate too, but answer with a move sequence rather than one printed label.
ONE_SHOT_ACTION_GAME_IDS: frozenset[str] = frozenset(
    {*MATRIX_GAME_IDS, *EVAL_ONLY_MATRIX_GAME_IDS, *CORPUS_BUILT_GAME_IDS}
)

BEHAVIOUR_FIELD_BY_GAME: dict[str, str] = {
    **dict.fromkeys(ACTION_RATE_GAME_IDS, COOP_FIELD),
    # The games whose answer is a figure rather than one of two labels, each with the figure its
    # record carries. The unilateral split has no counterpart to cooperate with, so its rate is the
    # fraction of the endowment KEPT and calling that cooperation would invert its direction. The
    # simultaneous claim has a counterpart and still no cooperative action: its rate is the fraction
    # of the total claimed, read against a half rather than against zero or one. The shared undertaking
    # has counterparts and a rate that reads against neither a half nor a corner but against its own
    # equal share, since more than that is waste. The trust games
    # answer with an amount, so theirs is the fraction of the stock sent -- except the never-trained
    # trustee item, which answers with a share returned and never sends at all.
    DICTATOR_GAME_ID: KEEP_FIELD,
    NASH_DEMAND_GAME_ID: CLAIM_FIELD,
    THRESHOLD_GOODS_GAME_ID: CONTRIBUTION_FIELD,
    TRUST_STATED_RETURN_GAME_ID: SEND_FIELD,
    TRUST_STRATEGY_METHOD_GAME_ID: SEND_FIELD,
    TRUSTEE_RETURN_GAME_ID: RETURN_FIELD,
    # Both minimum-effort forms answer with a level, so their rate is the level's position on the
    # grid, read against the level the payoffs point at rather than against zero or one. The repeated
    # form's figure is the mean over its five rounds, which is the same quantity per round.
    MIN_EFFORT_GAME_ID: LEVEL_FIELD,
    MIN_EFFORT_MATCH_GAME_ID: LEVEL_FIELD,
    # Both transfer probes answer with a number of units set down, so their rate is the fraction of the
    # stock that left the actor's hands. Read against zero, which is the selfish optimum in both games
    # and in every dose -- unlike the shared undertaking, whose rate is read against its equal share.
    **dict.fromkeys(PROBE_ONLY_GAME_IDS, SET_DOWN_FIELD),
}


# Every figure field a game's record can ANSWER WITH, primary first. Only the trust roster needs more
# than one entry, and the reason is that its records are rectangular on purpose: `_trust_game_record`
# writes both `send_fraction` and `return_fraction` on all three of its games, null where the game
# does not answer with that figure, so no reduction downstream has to ask whether a key is present.
# The strategy method is the one that genuinely answers both -- the announced-rule game never states a
# return and the trustee item never sends -- so a summary keyed on the field merely being PRESENT puts
# those prompts in the denominator of a measure they were never asked for, which reads as a parse
# failure that did not happen. Found by review, 2026-08-22.
FIGURE_FIELDS_BY_GAME: dict[str, tuple[str, ...]] = {
    DICTATOR_GAME_ID: (KEEP_FIELD,),
    NASH_DEMAND_GAME_ID: (CLAIM_FIELD,),
    THRESHOLD_GOODS_GAME_ID: (CONTRIBUTION_FIELD,),
    TRUST_STATED_RETURN_GAME_ID: (SEND_FIELD,),
    TRUST_STRATEGY_METHOD_GAME_ID: (SEND_FIELD, RETURN_FIELD),
    TRUSTEE_RETURN_GAME_ID: (RETURN_FIELD,),
    MIN_EFFORT_GAME_ID: (LEVEL_FIELD,),
    MIN_EFFORT_MATCH_GAME_ID: (LEVEL_FIELD,),
    **dict.fromkeys(PROBE_ONLY_GAME_IDS, (SET_DOWN_FIELD,)),
}


def _assert_figure_field_map_agrees_with_the_behaviour_map() -> None:
    """Raise unless the two figure maps name the same games and the same primary field each.

    Run at import. Two maps over one fact is how this file's tables came to disagree before, so the
    check is that they cannot: exactly the games whose behaviour rate is not a cooperation rate
    appear here, and each one's FIRST entry is the field the behaviour map names. Without it a game
    added to one and not the other would either drop out of the summary's figure measures silently or
    be summarised under the wrong primary.
    """
    figure_games = {
        game_id
        for game_id, field_name in BEHAVIOUR_FIELD_BY_GAME.items()
        if field_name != COOP_FIELD
    }
    missing = sorted(figure_games - set(FIGURE_FIELDS_BY_GAME))
    extra = sorted(set(FIGURE_FIELDS_BY_GAME) - figure_games)
    if missing or extra:
        raise RuntimeError(
            f"FIGURE_FIELDS_BY_GAME disagrees with the figure-answering games in "
            f"BEHAVIOUR_FIELD_BY_GAME: {missing=} {extra=}. Name every figure a game can answer "
            f"with, or its summary measure is computed over the wrong denominator."
        )
    disagreeing = sorted(
        game_id
        for game_id, fields in FIGURE_FIELDS_BY_GAME.items()
        if not fields or fields[0] != BEHAVIOUR_FIELD_BY_GAME[game_id]
    )
    if disagreeing:
        raise RuntimeError(
            f"games {disagreeing} name a different primary figure here than in "
            f"BEHAVIOUR_FIELD_BY_GAME, so the readouts and the summary would report different "
            f"quantities under one name."
        )


def _assert_behaviour_field_map_covers_every_game() -> None:
    """Raise unless every renderable game names the field its behaviour rate lands in.

    Run at import, for a sharper reason than the grading map's. Both readouts used to default a
    missing game to the cooperation rate, so a game answering with a figure and absent from the map
    rendered an EMPTY table rather than an error -- and both readouts recompute themselves from
    artifacts, so the emptiness travelled quietly into whatever was read next. Wave 2 added three
    such games at once, which is what made a silent default untenable.

    The completeness half of that is only worth anything while the map is not SEEDED with a default.
    It used to open with `dict.fromkeys(EVAL_RENDER_GRADING_BY_GAME, COOP_FIELD)`, which quietly
    restored the fallback it exists to remove: every trainable game was already keyed to the
    cooperation rate before the explicit entries below overrode a few of them, so a game answering
    with a figure and forgotten here would have read as an action game and rendered an empty table
    while this check passed. Found by sabotage on 2026-08-21 -- deleting the minimum-effort entries
    left the suite green. The seed is now `ACTION_RATE_GAME_IDS`, which is the games rendered through
    the label-printing renderers, so a game outside those groups has to be named or the check fires.
    """
    renderable = set(RENDERABLE_GAME_IDS)
    missing = sorted(renderable - set(BEHAVIOUR_FIELD_BY_GAME))
    extra = sorted(set(BEHAVIOUR_FIELD_BY_GAME) - renderable)
    if missing or extra:
        raise RuntimeError(
            f"BEHAVIOUR_FIELD_BY_GAME disagrees with the games games.prompts can render: "
            f"{missing=} {extra=}. Name the field each game's behaviour rate lands in, so a game "
            f"answering with a figure cannot fall back to a cooperation rate it never records and "
            f"render an empty table in both readouts."
        )
    unlabelled_reading_an_action_rate = sorted(ACTION_RATE_GAME_IDS & set(UNLABELLED_GAME_IDS))
    if unlabelled_reading_an_action_rate:
        raise RuntimeError(
            f"games {unlabelled_reading_an_action_rate} print no action labels and yet are seeded "
            f"with the cooperation rate. Their completions carry a figure, so the rate would be null "
            f"on every eval row and both readouts would render an empty table for them while this "
            f"check passed -- which is exactly the fallback the seed was narrowed to remove."
        )


_assert_behaviour_field_map_covers_every_game()
_assert_figure_field_map_agrees_with_the_behaviour_map()


META_FIELDS_OWNED_HERE: frozenset[str] = frozenset(
    {
        "record",
        "written_at",
        "git_sha",
        "backend_model_id",
        "sections",
        "eval_config",
        "frame_label_audit",
        "resume",
    }
)


# The transports that apply the model's chat template themselves, inside this process. They take no
# template-kwargs argument, so a run whose training pinned any cannot be evaluated through them.
LOCAL_TEMPLATE_TRANSPORTS: frozenset[str] = frozenset({HFBackend.transport, VLLMBackend.transport})


@dataclass(frozen=True)
class EvalConfig:
    """Knobs for one battery run; every value lands in the trace's meta record."""

    open_ended_samples: int = 8
    # Per-item overrides of `open_ended_samples` (pairs, so the config stays hashable): rebalances
    # the open-ended render budget without deleting an item, per `__post_init__`'s floor of 1.
    open_ended_samples_by_item: tuple[tuple[str, int], ...] = ()
    # Per option order. One render per order resolved 1/8 steps at best in the first battery,
    # which was its binding measurement-quality limit (2026-08-20 dt-probes readout, section 4.4).
    multiple_choice_samples: int = 8
    # Per game-behaviour prompt. One draw resolves a per-prompt rate to 0/1 or 1/1, which the
    # 2026-08-20 readout named as one of the three components of the battery's flat twin rows.
    # Defaults to 1 rather than to the 8 the other sections use, because it multiplies the cost of
    # the largest behavioural section and every existing launch was costed at one draw; the section
    # logs the count it is about to run so a n=1 pass says so in its own log rather than only here.
    game_behavior_samples: int = 1
    # Which label the game-behaviour prompts print first. Every battery cell ever produced rendered
    # the canonical order only, so word identity and print position are aliased in all of them: a
    # move toward a label cannot be told from a move toward a position. Rendering both orders
    # de-aliases them, and each record carries the order it was rendered under so the decomposition
    # is a split rather than a second pass. `games.prompts.UNLABELLED_GAME_IDS` print no labels at
    # all and are always rendered canonically -- see `_print_orders_for`.
    label_print_orders: tuple[str, ...] = (LABEL_PRINT_ORDER_CANONICAL,)
    capability_items: int = 50
    capability_seed: int = 20260817
    # Per presentation order, as above. A hundred-odd items once the published instruments load, so
    # this section costs about `2 * survey_samples * n_items` renders -- logged at section start,
    # because it is the largest section by render count at the shared default.
    survey_samples: int = 8
    # Where the untracked item text lives (authored.json and published.json both); see
    # `games/data/survey/README.md`. Required whenever the self-report section is requested -- the
    # battery runs nothing without local item data. There is no baked default: a path chosen here
    # would be right on one machine and silently absent on the next.
    survey_data_dir: Path | None = None
    # Empty means every published instrument the file supplies, mirroring `games`. A named subset is
    # the only supported way to run a partial battery, and it reaches the meta so it stays
    # attributable months later.
    survey_instruments: tuple[str, ...] = ()
    # Empty means every family that has items. Naming a subset is how the breadth tier or the
    # negative-control family gets run on its own.
    survey_families: tuple[str, ...] = ()
    # Empty means both tiers. The core set is position-level within instruments, so it is only
    # reachable through this knob -- the deliberated leg runs tier="core" so breadth items never
    # bill thinking-on completions.
    survey_tier: str = ""
    prefilled_think: bool = True
    dtbench_dir: Path | None = None
    games: tuple[str, ...] = ()
    include_never_trained: bool = True
    # The games the checkpoint under evaluation was actually trained on, which is one game per arm.
    # Empty means nothing was trained, i.e. the un-adapted base model, so no row claims otherwise.
    trained_game_ids: tuple[str, ...] = ()
    # What training pinned on the chat template, as pairs so the config stays hashable. Non-empty
    # only where the template has such a knob at all, which today is Qwen3.8-27B's reasoning_effort.
    chat_template_kwargs: tuple[tuple[str, str], ...] = ()
    # Sequences per generation call. None derives it from the VRAM actually present; an explicit
    # width is an assertion about the hardware and is refused past the measured throughput knee.
    batch_size: int | None = None
    # Which counterpart framings the framing-sweep section renders, in render order. Empty is the
    # default because the section is opt-in and `run_eval_battery` refuses the combination of the
    # section with no framings, for the survey section's reason: discovering the omission after the
    # other sections would cost their GPU time twice.
    counterpart_framings: tuple[str, ...] = ()
    # Which games the framing-sweep section renders. The default is pinned because every banked
    # sweep cell was measured on it; naming games is explicit opt-in for cells that need another
    # frameable game -- the first user is track-record-v2's same-p-different-optimum cell, which
    # sweeps the temptation-dose ladder under the numeric dose framings so one stated rate lands
    # on both sides of the ladder's crossovers.
    framing_sweep_games: tuple[str, ...] = FRAMING_SWEEP_GAME_IDS
    # Where the authored recipient paragraphs the trap-cells section inserts live
    # (`games.framing_stimulus.FRAMINGS_PATH`). No baked default, for `survey_data_dir`'s reason: a path
    # chosen here would be right on one machine and silently absent on the next.
    trap_cells_file: Path | None = None
    # The digest of what that file held when the plan was resolved. Recorded beside the path because
    # the path is machine-local and the CONTENT is what a second box has to agree with: two cells
    # whose recipient paragraphs differ are different measurements, and the digest is what says so
    # in the cell identity and in the bank key.
    trap_cells_digest: str = ""
    # The counterpart framings a wave authored outside version control, loaded from the gitignored
    # file `--framings-file` names (`games.framing_stimulus`). None is a battery that renders
    # registered framings only. `counterpart_framings` still names what runs: a file's roster
    # supplies clauses, never a cell's contents, so pointing at a larger file cannot silently widen
    # a sweep. The digest reaches the meta and the cell identity through `as_record`; the clauses
    # themselves never do, because they are stimulus.
    runtime_framings: RuntimeFramings | None = None
    # The runtime-loaded held-out frames (`games.held_out_extension`), applied to the
    # game-behaviour section only. None is the default and the state every banked cell was measured
    # in; the framing sweep deliberately never sees it, so its cells stay comparable to the banked
    # ones. The digest reaches `as_record`, hence the trace meta and the cell identity.
    held_out_extension: ExtensionRosters | None = None

    def __post_init__(self) -> None:
        """Reject counts and widths that would silently produce an empty or unsizable section."""
        for name in (
            "open_ended_samples",
            "multiple_choice_samples",
            "survey_samples",
            "game_behavior_samples",
        ):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be at least 1, got {value}.")
        self._validate_open_ended_allocation()
        self._validate_survey_instruments()
        self._validate_label_print_orders()
        if self.batch_size is not None and self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}.")
        if self.capability_items < 1:
            raise ValueError(f"capability_items must be at least 1, got {self.capability_items}.")
        # Refused as a pair, in both directions. A file with no digest would record a cell whose
        # recipient paragraphs nothing identifies, so an edited file would resume into the old
        # trace; a digest with no file would claim a measurement the plan cannot render.
        if (self.trap_cells_file is None) != (not self.trap_cells_digest):
            raise ValueError(
                f"trap_cells_file and trap_cells_digest travel together, got "
                f"trap_cells_file={self.trap_cells_file!r} "
                f"trap_cells_digest={self.trap_cells_digest!r}. The digest is what the cell "
                f"identity keys on, since the path is machine-local."
            )
        unknown = sorted(set(self.games) - set(ALL_GAME_IDS))
        if unknown:
            raise ValueError(f"Unknown game ids {unknown}; known games: {sorted(ALL_GAME_IDS)}.")
        named_never_trained = sorted(set(self.games) & set(EVAL_ONLY_GAME_IDS))
        if named_never_trained and not self.include_never_trained:
            raise ValueError(
                f"games names the eval-only ids {named_never_trained} while "
                f"include_never_trained is False. Those games only render through the "
                f"never-trained leg, so this combination would silently play nothing of what it "
                f"named; drop the ids or leave the never-trained leg on."
            )
        trainable = set(GAME_IDS) | set(CORPUS_BUILT_GAME_IDS)
        untrainable = sorted(set(self.trained_game_ids) - trainable)
        if untrainable:
            raise ValueError(
                f"Unknown trained game ids {untrainable}; known games: {sorted(trainable)} "
                f"(the roster's trainable games plus the corpus-built ones, whose corpora "
                f"games.track_record_corpus and its kin construct). The eval-only registry has "
                f"no training arm, so it cannot appear here."
            )
        self._validate_counterpart_framings()

    def _validate_counterpart_framings(self) -> None:
        """Reject a framing list naming something other than it means, before a card is taken.

        The same two silent shapes the survey lists refuse: a typoed id would render a sweep
        claiming a framing it never asked, and a repeated id would render those prompts twice and
        double-weight every one of them in the per-framing rates.
        """
        if not self.framing_sweep_games:
            raise ValueError(
                "framing_sweep_games is empty: the sweep section would render nothing while "
                "claiming to have run. Leave the default or name frameable games explicitly."
            )
        unframeable = sorted(set(self.framing_sweep_games) - set(FRAMEABLE_GAME_IDS))
        if unframeable:
            raise ValueError(
                f"framing_sweep_games names {unframeable}, which the framing renderer cannot "
                f"serve; frameable games: {sorted(FRAMEABLE_GAME_IDS)}."
            )
        listed_games = list(self.framing_sweep_games)
        repeated_games = sorted({game for game in listed_games if listed_games.count(game) > 1})
        if repeated_games:
            raise ValueError(
                f"framing_sweep_games repeats {repeated_games}; every repeat would render those "
                f"prompts twice and double-weight them in the per-framing rates."
            )
        runtime_ids = () if self.runtime_framings is None else self.runtime_framings.framing_ids
        unknown = sorted(
            set(self.counterpart_framings) - set(COUNTERPART_FRAMING_IDS) - set(runtime_ids)
        )
        if unknown:
            raise ValueError(
                f"Unknown counterpart framings {unknown}; registered: "
                f"{list(COUNTERPART_FRAMING_IDS)}, loaded at runtime: {list(runtime_ids)} "
                f"(--framings-file supplies the second list)."
            )
        listed = list(self.counterpart_framings)
        repeated = sorted({framing for framing in listed if listed.count(framing) > 1})
        if repeated:
            raise ValueError(
                f"counterpart_framings names {repeated} more than once, which renders those "
                f"prompts twice and double-weights them in every per-framing rate."
            )

    def _validate_open_ended_allocation(self) -> None:
        """Reject a per-item render allocation that would misfire in silence.

        Three shapes, each of which would otherwise change the battery without saying so: a typoed
        probe id leaves the item at the shared default while the config claims otherwise; a count
        of zero deletes an item from the instrument when the point of per-item allocation is to
        rebalance renders, never to drop coverage; a repeated id makes the effective count depend
        on tuple order.
        """
        ids = [probe_id for probe_id, _ in self.open_ended_samples_by_item]
        repeated = sorted({probe_id for probe_id in ids if ids.count(probe_id) > 1})
        if repeated:
            raise ValueError(
                f"open_ended_samples_by_item names {repeated} more than once, so the effective "
                f"count would depend on pair order."
            )
        open_ended_ids = {item.probe_id for item in OPEN_ENDED_ITEMS}
        unknown = sorted(set(ids) - open_ended_ids)
        if unknown:
            raise ValueError(
                f"open_ended_samples_by_item names {unknown}, which are not open-ended probes; "
                f"the open-ended items are {sorted(open_ended_ids)}."
            )
        starved = sorted(
            probe_id for probe_id, count in self.open_ended_samples_by_item if count < 1
        )
        if starved:
            raise ValueError(
                f"open_ended_samples_by_item allocates fewer than 1 render to {starved}; the "
                f"allocation rebalances the render budget between items, it does not delete them."
            )

    def _validate_label_print_orders(self) -> None:
        """Reject a print-order list that would render nothing, or one rendering twice.

        Both shapes are silent rather than loud. An empty list drops the whole game-behaviour section
        while every other section still runs, so the trace reads as a battery that measured no
        behaviour; a repeated order asks the same prompts twice under one order and double-weights
        every one of them in a rate that is supposed to be a mean over distinct prompts.
        """
        unknown = sorted(set(self.label_print_orders) - set(LABEL_PRINT_ORDERS))
        if unknown:
            raise ValueError(
                f"Unknown label_print_orders {unknown}; known orders: {sorted(LABEL_PRINT_ORDERS)}."
            )
        if not self.label_print_orders:
            raise ValueError(
                f"label_print_orders is empty, which renders no game-behaviour prompts at all; "
                f"choose from {sorted(LABEL_PRINT_ORDERS)}."
            )
        listed = list(self.label_print_orders)
        repeated = sorted({order for order in listed if listed.count(order) > 1})
        if repeated:
            raise ValueError(
                f"label_print_orders names {repeated} more than once, which renders those prompts "
                f"twice under one order and double-weights every one of them."
            )

    def _validate_survey_instruments(self) -> None:
        """Reject a survey instrument or family list that names something other than it means.

        `games.survey.survey_battery` refuses the same two shapes, so this is deliberately the
        earlier of two checks rather than the only one: it fires where the config is built, which is
        before a checkpoint is merged and a card is taken, and the later one covers a caller that
        assembled the battery itself. A typo would otherwise cost a rented GPU hour to discover.
        """
        if self.survey_tier and self.survey_tier not in TIERS:
            raise ValueError(
                f"Unknown survey_tier {self.survey_tier!r}; registered: {list(TIERS)} "
                f"(empty means both)."
            )
        for name, requested, registered in (
            ("survey_instruments", self.survey_instruments, tuple(PUBLISHED_INSTRUMENTS)),
            ("survey_families", self.survey_families, FAMILIES),
        ):
            unknown = sorted(set(requested) - set(registered))
            if unknown:
                raise ValueError(f"Unknown {name} {unknown}; registered: {sorted(registered)}.")
            listed = list(requested)
            repeated = sorted({value for value in listed if listed.count(value) > 1})
            if repeated:
                raise ValueError(
                    f"{name} names {repeated} more than once, which asks those items twice under "
                    f"one sample index and double-weights them in every composite."
                )

    def as_record(self) -> dict[str, Any]:
        """Return the config as JSON-safe values for the meta record."""
        return {
            "open_ended_samples": self.open_ended_samples,
            "open_ended_samples_by_item": dict(self.open_ended_samples_by_item),
            "multiple_choice_samples": self.multiple_choice_samples,
            "game_behavior_samples": self.game_behavior_samples,
            "label_print_orders": list(self.label_print_orders),
            "capability_items": self.capability_items,
            "capability_seed": self.capability_seed,
            "survey_samples": self.survey_samples,
            "survey_data_dir": (
                str(self.survey_data_dir) if self.survey_data_dir is not None else None
            ),
            "survey_instruments": list(self.survey_instruments) or sorted(PUBLISHED_INSTRUMENTS),
            # `families_with_items()` rather than `FAMILIES`: the registry carries family names whose
            # items are still being authored, and recording those as administered would say this cell
            # asked items that do not exist.
            "survey_families": list(self.survey_families) or sorted(families_with_items()),
            "survey_tier": self.survey_tier or list(TIERS),
            "prefilled_think": self.prefilled_think,
            "batch_size": self.batch_size,
            "dtbench_dir": str(self.dtbench_dir) if self.dtbench_dir is not None else None,
            "games": list(self.games) or list(GAME_IDS),
            "include_never_trained": self.include_never_trained,
            "trained_game_ids": list(self.trained_game_ids),
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "counterpart_framings": list(self.counterpart_framings),
            "framing_sweep_games": list(self.framing_sweep_games),
            "trap_cells_file": (
                str(self.trap_cells_file) if self.trap_cells_file is not None else None
            ),
            "trap_cells_digest": self.trap_cells_digest,
            "framings_file": (
                None if self.runtime_framings is None else str(self.runtime_framings.path)
            ),
            "framings_digest": (
                "" if self.runtime_framings is None else self.runtime_framings.digest
            ),
            "held_out_extension": (
                None if self.held_out_extension is None else self.held_out_extension.as_record()
            ),
        }


def _decode_token_budget(backend: Backend) -> int:
    """Return the output-token budget this backend samples at, for the chunk arithmetic.

    Not a cap this module imposes -- the backend's own sampling config is what bounds generation.
    It is the number a chunk width has to be derived against, since what a card can hold in flight
    scales with completion tokens rather than with sequences.

    The `Backend` protocol guarantees only `model_id` and `transport`, and the two local backends
    carry differently-named budgets on differently-typed sampling configs, so this reads whichever
    is present. A backend that declares none (a mock, a hosted transport) falls back to the model's
    measured termination budget rather than to a number chosen here.
    """
    sampling = getattr(backend, "sampling", None)
    for attribute in ("max_new_tokens", "max_tokens"):
        budget = getattr(sampling, attribute, None)
        if isinstance(budget, int):
            return budget
    budget = required_completion_budget(backend.model_id)
    logger.info(
        f"{backend.model_id} declares no sampling budget, so the chunk arithmetic uses its "
        f"measured termination budget, {budget=}"
    )
    return budget


def _chunk_width(backend: Backend, prompts: Sequence[str], *, requested: int | None) -> int:
    """Decide how many sequences go into one generation call.

    Only `HFBackend` builds a single padded tensor over everything it is handed, so only it can be
    sized from the card: `games.select_prompts.sweep_chunk_size` reads the free VRAM, prices one
    sequence off the checkpoint's architecture, and clamps at the measured throughput knee.

    For a scripted or hosted backend nothing decodes in this process, so a width is request
    batching rather than a memory decision. Such a backend's own declared parallelism is the right
    width where it has one (`BedrockBackend.concurrency` sizes its thread pool), because a chunk
    boundary is also where progress reaches the log -- handing a 1,600-prompt hosted eval over in
    one call would leave it silent until the whole thing finished.
    """
    model_id = local_decode_model_id(backend)
    schedules_own_batch = backend_schedules_own_batch(backend)
    if model_id is None and not schedules_own_batch:
        concurrency = getattr(backend, "concurrency", None)
        declared = concurrency if isinstance(concurrency, int) else len(prompts)
        return min(declared if requested is None else requested, len(prompts))
    return sweep_chunk_size(
        max_new_tokens=_decode_token_budget(backend),
        n_sequences=len(prompts),
        requested=requested,
        model_id=model_id,
        schedules_own_batch=schedules_own_batch,
    )


def _generate(backend: Backend, prompts: Sequence[str], *, config: EvalConfig) -> list[str]:
    """Generate one completion per prompt, at a width the card can actually hold.

    The `Backend` protocol has no sample count, so N samples of a prompt means repeating the
    prompt N times in the list -- which is what the probe section does.

    `decode_in_chunks` is the shared decode layer the baseline sweep already uses: it narrows the
    width on a raised OOM and on an allocator thrash, and re-decodes exactly the prompts that
    failed. Without it a battery at a 32,768-token budget lost its whole trace to one OOM, and
    `games.run_evals` then refused to retry the step because a partial trace file existed.
    """
    completions = decode_in_chunks(
        backend, prompts, chunk_size=_chunk_width(backend, prompts, requested=config.batch_size)
    )
    if len(completions) != len(prompts):
        raise RuntimeError(
            f"Backend returned {len(completions)} completions for {len(prompts)} prompts; the "
            f"records would no longer line up with the items they describe."
        )
    return completions


# The three games whose answer is an amount sent, a share returned, or both. Named as a set here so
# `_game_record`'s branch cannot drift from the renderers in `games.prompts`, whose `_trust_rows`
# serves exactly these.
_TRUST_RECORD_GAME_IDS: frozenset[str] = frozenset(
    {TRUST_STATED_RETURN_GAME_ID, TRUST_STRATEGY_METHOD_GAME_ID, TRUSTEE_RETURN_GAME_ID}
)


def _trust_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one trust record: the amount sent, the share returned, or both.

    Three shapes and one function, because they are the same two numbers in different combinations.
    The trustee item carries no send at all -- it answers with a share and nothing else -- and its
    `return_fraction` is directly comparable with the strategy method's, which is the whole reason it
    is asked in the same units and with the same tag.
    """
    endowment = int(row["endowment"])
    sent: int | None = None
    return_fraction: float | None = None
    if game_id == TRUST_STATED_RETURN_GAME_ID:
        sent = parse_send(visible, endowment=endowment)
        record["parsed"] = sent is not None
    elif game_id == TRUST_STRATEGY_METHOD_GAME_ID:
        strategy = parse_trust_strategy(visible, endowment=endowment)
        record["parsed"] = strategy is not None
        if strategy is not None:
            sent = strategy.sent
            return_fraction = strategy.return_percentage / RETURN_PERCENTAGE_MAX
    else:
        percentage = parse_return_percentage(visible)
        record["parsed"] = percentage is not None
        if percentage is not None:
            return_fraction = percentage / RETURN_PERCENTAGE_MAX
    record["sent"] = sent
    record["send_fraction"] = None if sent is None else sent / endowment
    record["return_fraction"] = return_fraction
    return record


def _min_effort_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one minimum-effort record: a level, or one level per round.

    Both forms in one function because they read the same answer at two lengths. The repeated form
    keeps the whole sequence and its final round beside the mean, for the reason the matrix repeated
    arms keep `moves`: the per-round profile is where an end-game drop shows up, and a mean cannot
    answer the question this game exists to ask.
    """
    n_levels = int(row["n_levels"])
    if game_id == MIN_EFFORT_MATCH_GAME_ID:
        levels = parse_level_sequence(visible, n_rounds=int(row["n_rounds"]), n_levels=n_levels)
        record["levels"] = levels
        record["final_level"] = None if levels is None else levels[-1]
        record[LEVEL_FIELD] = (
            None
            if levels is None
            else sum((level - 1) / (n_levels - 1) for level in levels) / len(levels)
        )
        record["parsed"] = levels is not None
        return record
    level = parse_level(visible, n_levels=n_levels)
    record["level"] = level
    record[LEVEL_FIELD] = None if level is None else (level - 1) / (n_levels - 1)
    record["parsed"] = level is not None
    return record


def _move_sequence_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one repeated matrix record: the whole move sequence, or nothing if it does not parse."""
    del game_id
    moves = parse_action_sequence(
        visible,
        n_rounds=int(row["n_rounds"]),
        label_a=row["label_a"],
        label_b=row["label_b"],
        coop_label=row["coop_label"],
    )
    record["moves"] = moves
    record[COOP_FIELD] = None if moves is None else moves.count(COOPERATE) / len(moves)
    record["parsed"] = moves is not None
    return record


def _dictator_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one unilateral-split record: the units kept, as a fraction of the endowment."""
    del game_id
    endowment = int(row["endowment"])
    kept = parse_split(visible, endowment=endowment)
    record["kept"] = kept
    record[KEEP_FIELD] = None if kept is None else kept / endowment
    record["parsed"] = kept is not None
    return record


def _nash_demand_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one simultaneous-claim record: the claim, as a fraction of the windfall."""
    del game_id
    windfall = int(row["windfall"])
    claim = parse_claim(visible, windfall=windfall)
    record["claim"] = claim
    record[CLAIM_FIELD] = None if claim is None else claim / windfall
    record["parsed"] = claim is not None
    return record


def _threshold_goods_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one shared-undertaking record: the contribution, as a fraction of the stock."""
    del game_id
    endowment = int(row["endowment"])
    contribution = parse_contribution(visible, endowment=endowment)
    record["contribution"] = contribution
    record[CONTRIBUTION_FIELD] = None if contribution is None else contribution / endowment
    record["parsed"] = contribution is not None
    return record


def _transfer_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one transfer record: the units set down, as a fraction of the stock.

    Polarity-aware, reading it off the row's `label_print_order` column, which is where the transfer
    renderer files which of the two figures the row asked for. A row asked for what it kept back is
    inverted by `parse_transfer_figure` rather than here, so this builder and the probe's own scan cannot
    come to report different quantities under one name.
    """
    del game_id
    endowment = int(row["endowment"])
    set_down = parse_transfer_figure(
        visible, endowment=endowment, polarity=str(row["label_print_order"])
    )
    record["set_down"] = set_down
    record[SET_DOWN_FIELD] = None if set_down is None else set_down / endowment
    record["parsed"] = set_down is not None
    return record


def _action_game_record(
    game_id: str, row: Mapping[str, Any], record: dict[str, Any], visible: str
) -> dict[str, Any]:
    """Fill in one one-shot matrix record: which of the two labels the completion named."""
    del game_id
    action = parse_action(
        visible, label_a=row["label_a"], label_b=row["label_b"], coop_label=row["coop_label"]
    )
    record["action"] = action
    record[COOP_FIELD] = None if action is None else float(action == COOPERATE)
    record["parsed"] = action is not None
    return record


# How each renderable game's completion is read, replacing the if-chain this used to be. A table
# rather than a chain for one reason worth stating: the chain ENDED in a fall-through to the action
# parser, so a game added to `games.prompts` and forgotten here would look for `<action>` tags, find
# none, and write `parsed=False` with a null behaviour field for every eval row -- a complete,
# plausible, entirely null section, and the readouts recompute themselves from artifacts, so the
# emptiness would travel quietly into whatever was read next. The map has no default and the
# completeness check below runs at import, so the same mistake is now an import error.
# A `type` alias rather than an assignment, because `Callable` and `Mapping` are imported only under
# TYPE_CHECKING here and a plain alias would evaluate them at import time.
type _RecordBuilder = Callable[[str, Mapping[str, Any], dict[str, Any], str], dict[str, Any]]

# Which games each builder serves, as DISJOINT groups keyed on the renderer that built the prompt.
# Disjoint rather than layered on purpose. The first version of this table seeded every trainable game
# to the action builder and then overrode a few entries, which meant a game forgotten here still got a
# builder -- the action one -- and its whole eval section came back parsed=False with a null behaviour
# field, which is the failure the table replaced an if-chain to prevent. The completeness check below
# cannot see that, because the map IS complete; the disjointness check is what makes an omission
# visible. Found by sabotage on 2026-08-21: deleting the minimum-effort group left the suite green.
_BUILDER_GROUPS: tuple[tuple[_RecordBuilder, frozenset[str]], ...] = (
    (_action_game_record, ONE_SHOT_ACTION_GAME_IDS),
    (_move_sequence_game_record, frozenset(ITERATED_GAME_IDS)),
    (_trust_game_record, _TRUST_RECORD_GAME_IDS),
    (_min_effort_game_record, frozenset(MIN_EFFORT_GAME_IDS)),
    (_dictator_game_record, frozenset({DICTATOR_GAME_ID})),
    (_nash_demand_game_record, frozenset({NASH_DEMAND_GAME_ID})),
    (_threshold_goods_game_record, frozenset({THRESHOLD_GOODS_GAME_ID})),
    (_transfer_game_record, frozenset(PROBE_ONLY_GAME_IDS)),
)

_GAME_RECORD_BUILDERS: dict[str, _RecordBuilder] = {
    game_id: builder for builder, game_ids in _BUILDER_GROUPS for game_id in game_ids
}


def _assert_record_builder_map_covers_every_game() -> None:
    """Raise unless every renderable game names the function that reads its completions.

    Run at import, for `_assert_behaviour_field_map_covers_every_game`'s reason and against a sharper
    failure: a game absent from this map used to reach the action parser by fall-through, so its whole
    eval section came back parsed=False rather than erroring. The behaviour-field map catches the
    readout half of that; this catches the parsing half.
    """
    renderable = set(RENDERABLE_GAME_IDS)
    missing = sorted(renderable - set(_GAME_RECORD_BUILDERS))
    extra = sorted(set(_GAME_RECORD_BUILDERS) - renderable)
    if missing or extra:
        raise RuntimeError(
            f"_GAME_RECORD_BUILDERS disagrees with the games games.prompts can render: "
            f"{missing=} {extra=}. Name the function that reads each game's completions, so a game "
            f"added upstream cannot fall through to the action parser and write a complete, "
            f"plausible, entirely null section."
        )
    claimed = [game_id for _, game_ids in _BUILDER_GROUPS for game_id in game_ids]
    overlapping = sorted({game_id for game_id in claimed if claimed.count(game_id) > 1})
    if overlapping:
        raise RuntimeError(
            f"games {overlapping} are claimed by more than one record builder, so which one reads "
            f"their completions depends on the order of _BUILDER_GROUPS. That is how the map came to "
            f"be complete and wrong at once: a broad group covered every game, a narrow one overrode "
            f"a few, and a game left out of the narrow group was read by the broad one instead of "
            f"failing this check."
        )


_assert_record_builder_map_covers_every_game()


def _game_record(  # noqa: PLR0913 - one record's worth of provenance, not a bundle worth naming
    game_id: str,
    row: Mapping[str, Any],
    completion: str,
    *,
    prefilled_think: bool,
    trained_game: bool,
    eval_only_game: bool,
    sample_index: int,
) -> dict[str, Any]:
    """Build one game-behaviour record, parsed according to which game it is.

    Three states, not two, and the report's transfer-vs-trained split needs all of them: this is
    the game the arm trained on, or a trained-registry game the arm never saw (in-registry
    transfer), or a game with no training arm at any point (`eval_only_game`).

    `sample_index` says which draw of its prompt this is, and every readout needs it: without it a
    prompt sampled several times is indistinguishable from several prompts, so a rate computed over
    records rather than over prompts weights a prompt whose draws mostly failed to parse differently
    from its neighbours. Written here rather than by each caller because
    `games.eval_training_frames` shares this writer and the two stamped it separately.
    """
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    record: dict[str, Any] = {
        "record": SECTION_GAME_BEHAVIOR,
        "game_id": game_id,
        "prompt_id": row["prompt_id"],
        "sample_index": sample_index,
        # Which label this row printed first, read off the row rather than the caller's request: an
        # unlabelled game renders canonically whatever was asked for, so a record that took the
        # request would claim a rendering the prompt does not have.
        "label_print_order": row["label_print_order"],
        "reskin_id": row["reskin_id"],
        "payoff_variant": row["payoff_variant"],
        "render_grading": row["grading"],
        "label_a": row["label_a"],
        "label_b": row["label_b"],
        "coop_label": row["coop_label"],
        "trained_game": trained_game,
        "eval_only_game": eval_only_game,
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
    }
    return _GAME_RECORD_BUILDERS[game_id](game_id, row, record, visible)


SUBMISSION_SERIAL = "serial"
SUBMISSION_POOLED = "pooled"
SUBMISSIONS: tuple[str, ...] = (SUBMISSION_SERIAL, SUBMISSION_POOLED)
"""How a battery hands its prompts to the backend.

``serial`` is one generation call per call group -- per (game, print order) for the behaviour
section, per game for the framing sweep, one per remaining section -- which is what every cell
before 2026-09-02 ran and replays those cells byte for byte on the same engine seed. ``pooled``
submits every pending prompt of every section at once and files each record as its completion
lands, so the engine never idles between calls and never waits for one game's longest thinker
before starting the next game's prompts. Same sampler, same engine seed, same distribution; not
the same bytes, because unseeded vLLM requests share one RNG stream whose consumption order the
grouping decides. Keep ``serial`` for a bit-neutral comparison against a banked cell.
"""

# Which record fields, together with the section, identify one planned prompt. Every readout keys
# on exactly these, and resume-by-key stands on them: a record on disk carrying one of these
# identities is the completed answer to that prompt and is never regenerated. Compared through JSON
# types on both sides (`record_identity`), because the trace is JSON and a tuple-versus-list or
# int-versus-float mismatch would refuse to resume an identical cell.
_GAME_RECORD_IDENTITY_FIELDS: tuple[str, ...] = (
    "game_id",
    "prompt_id",
    "label_print_order",
    "sample_index",
)
RECORD_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    SECTION_GAME_BEHAVIOR: _GAME_RECORD_IDENTITY_FIELDS,
    SECTION_FRAMING_SWEEP: _GAME_RECORD_IDENTITY_FIELDS,
    # A training-frames record is a `_game_record` restamped, so it is keyed exactly as the
    # behaviour section's are: game, prompt, print order, draw.
    RECORD_TRAINING_FRAMES: _GAME_RECORD_IDENTITY_FIELDS,
    SECTION_DT_PROBES: ("probe_id", "option_order_name", "sample_index"),
    SECTION_CAPABILITIES: ("item_index",),
    SECTION_SELF_REPORT: ("item_id", "option_order_name", "sample_index"),
    # A trap-cell record is a `_game_record` restamped too, and its trap rides in its prompt_id, so
    # the game/prompt/order/draw key already separates the versions.
    SECTION_TRAP_CELLS: _GAME_RECORD_IDENTITY_FIELDS,
}

type RecordIdentity = tuple[Any, ...]


ADMISSION_LONGEST_FIRST = "longest-first"
ADMISSION_FIFO = "fifo"
ADMISSIONS: tuple[str, ...] = (ADMISSION_LONGEST_FIRST, ADMISSION_FIFO)
"""The order a ``pooled`` submission hands its pending prompts to the engine.

``fifo`` is plan order: section by section as the caller listed them, which is how every pooled
cell before 2026-09-03 was admitted. ``longest-first`` (the default) admits sections in
`POOLED_ADMISSION_ORDER`, the sections whose sequences run longest first, keeping plan order within
a section. The reason is the pooled cell measured in probe P-E4 (2026-09-03, base Qwen3.5-2B, the
full behaviour + capabilities + self-report battery at a 65,536-token cap): the 128 self-report
prompts were admitted last, 81 of them ran to the cap, and they ran nearly alone for the last ~50
minutes of a 221-minute drain at 3-6 records/min while the card had nothing else to batch them
with. Admitted first, the same sequences overlap the bulk of the game-behavior prompts instead of
trailing them. Same sampler, same engine seed, same record set and identities either way; not the
same bytes, because unseeded vLLM requests share one RNG stream whose consumption order the
admission decides -- the same statistical grade pooling itself carries against serial, and the
reason the meta records the admission per session (`_session_record`). One caveat rides on that
grade, the same one P-E4 left open for pooling: bf16 logits are not batch-invariant, the pooled
cell there hit the 65,536 cap less often than its serial twin (self-report 81 vs 94 of 128,
game-behavior 142 vs 186 of 5,616; one comparison, a flag rather than a finding), and
``longest-first`` changes exactly the batch the self-report thinkers run in -- inside the
~1,000-wide pooled batch instead of nearly alone at the tail -- so the next cell with a ``fifo`` or
serial twin should compare the self-report cap-hit rate. Keep ``fifo`` to reproduce a banked pooled
cell's bytes on its engine seed.
"""

POOLED_ADMISSION_ORDER: tuple[str, ...] = (
    SECTION_SELF_REPORT,
    SECTION_DT_PROBES,
    SECTION_CAPABILITIES,
    SECTION_GAME_BEHAVIOR,
    SECTION_FRAMING_SWEEP,
    SECTION_TRAP_CELLS,
    RECORD_TRAINING_FRAMES,
)
"""Record kinds by how long their sequences run, longest first, for ``longest-first`` admission.

Measured in P-E4 on the base 2B at the 65,536-token cap as the share of completions that ran to the
cap: self-report 81/128 (63%), capabilities 2/50 (4%), game-behavior 142/5,616 (2.5%). Two kinds
are placed by shape, unmeasured: dt-probes are decision-theory items deliberated the way the
self-report items are, so they sit right after them; the framing sweep and the training-frames cell
render game prompts, so they sit with game-behavior. Within a kind, plan order. Every registered
record kind must appear here (checked at import), so a new kind cannot fall silently to the back
of the queue -- which is exactly where the tail came from.
"""


def _assert_admission_order_covers_every_record_kind() -> None:
    """Raise unless `POOLED_ADMISSION_ORDER` ranks every registered record kind exactly once, at import."""
    missing = sorted(set(RECORD_IDENTITY_FIELDS) - set(POOLED_ADMISSION_ORDER))
    repeated = sorted(
        {kind for kind in POOLED_ADMISSION_ORDER if POOLED_ADMISSION_ORDER.count(kind) > 1}
    )
    unknown = sorted(set(POOLED_ADMISSION_ORDER) - set(RECORD_IDENTITY_FIELDS))
    if missing or repeated or unknown:
        raise RuntimeError(
            f"POOLED_ADMISSION_ORDER must rank every record kind exactly once: missing {missing}, "
            f"repeated {repeated}, unknown {unknown}; record kinds: {list(RECORD_IDENTITY_FIELDS)}."
        )


_assert_admission_order_covers_every_record_kind()


def _json_native(value: Any) -> Any:  # noqa: ANN401 - a JSON round trip of whatever the caller holds
    """Round-trip a value through JSON, so a live Python value compares equal to its stored form."""
    return json.loads(json.dumps(value))


def record_identity(record: Mapping[str, Any]) -> RecordIdentity:
    """Return the key under which this record answers exactly one planned prompt.

    The section leads the tuple so two sections' ids can never collide, and every value is
    JSON-normalised so an identity built from a rendering context (Python ints and strs) equals the
    same identity read back off the trace.
    """
    section = str(record["record"])
    fields = RECORD_IDENTITY_FIELDS.get(section)
    if fields is None:
        raise ValueError(
            f"record kind {section!r} has no identity fields; known kinds: "
            f"{list(RECORD_IDENTITY_FIELDS)}."
        )
    return (section, *_json_native([record[field] for field in fields]))


@dataclass(frozen=True, slots=True)
class PlannedRequest:
    """One prompt the battery will send, and everything needed to file what comes back.

    Rendering is separated from parsing so the same plan serves three needs at once: the pooled
    submission (every request of every section in one engine queue), resume-by-key (a request whose
    identity is already on disk is skipped), and the completeness check a summary needs (the plan IS
    the expected complement). ``call_group`` is the serial path's unit of generation, kept so that
    path reproduces the pre-2026-09-02 call sequence exactly.
    """

    section: str
    identity: RecordIdentity
    prompt: str
    call_group: str
    parse: Callable[[str], dict[str, Any]]

    def record(self, completion: str) -> dict[str, Any]:
        """Parse one completion into this request's record, refusing a plan that disagrees with itself.

        This checks the invariant resume-by-key stands on: the identity a planner DECLARED for a
        request must be the identity its parsed record CARRIES, because a relaunch skips prompts by
        the declared identity and readers key records by the carried one. A planner that bound its
        parser to one row's context and its identity to another's would make a resume skip the
        wrong prompts with every count still adding up.

        It is NOT a check that the completion belongs to this prompt: the parser only ever sees the
        completion text and derives every identity field from the rendering context it was bound
        to, so a reply routed to the wrong request parses into a well-formed record under the right
        identity. That pairing is guarded where the pairing is made -- `VLLMBackend.generate_streaming`
        on the pooled drain (request id plus echoed prompt) and ``zip(strict=True)`` over the
        order-preserving backends on the serial path.
        """
        record = self.parse(completion)
        identity = record_identity(record)
        if identity != self.identity:
            raise RuntimeError(
                f"the plan disagrees with itself: this request was planned as {self.identity} but "
                f"its parser produced a record carrying {identity}. A resume keyed on the planned "
                f"identity would skip the wrong prompts, so nothing was written."
            )
        return record


def _game_request_record(  # noqa: PLR0913 - one record's rendering context, bound once per request
    game_id: str,
    row: Mapping[str, Any],
    completion: str,
    *,
    section: str,
    framing_id: str | None,
    trap_id: str | None,
    prefilled_think: bool,
    trained_game: bool,
    eval_only_game: bool,
    sample_index: int,
) -> dict[str, Any]:
    """Build one game record, restamped where the request belongs to the sweep or to a trap cell.

    Those sections' records have prompt_ids and inserted paragraphs the game-behavior section never
    renders, and a reader pooling them with that section's records would average across framings or
    fold a trap into the plain rate without noticing -- hence the section restamp and the framing or
    trap id, exactly as each section's builder applied them before the plan split rendering from
    parsing.
    """
    record = _game_record(
        game_id,
        row,
        completion,
        prefilled_think=prefilled_think,
        trained_game=trained_game,
        eval_only_game=eval_only_game,
        sample_index=sample_index,
    )
    if framing_id is not None:
        record["record"] = section
        record[COUNTERPART_FRAMING_FIELD] = framing_id
    if trap_id is not None:
        record["record"] = section
        record["trap_id"] = trap_id
    return record


def _plan_sampled_game(  # noqa: PLR0913 - one call group's rendering context
    rows: Sequence[Mapping[str, Any]],
    *,
    section: str,
    game_id: str,
    config: EvalConfig,
    trained_game: bool,
    call_group: str,
    framing_by_row: Sequence[str | None] | None = None,
    trap_by_row: Sequence[str] | None = None,
) -> list[PlannedRequest]:
    """Plan `game_behavior_samples` draws per row, each parsed into its own record.

    Prompts are repeated ROW-MAJOR -- row 0's draws, then row 1's -- which is how every repeated
    section in this module expresses a sample count: the `Backend` protocol has no sample count, so
    N samples of a prompt means the prompt appears N times in the request list. The same layout
    `games.eval_training_frames` already used, lifted here rather than written a second time.
    """
    n_samples = config.game_behavior_samples
    eval_only_game = game_id in EVAL_ONLY_GAME_IDS
    requests: list[PlannedRequest] = []
    for row_index, row in enumerate(rows):
        framing_id = None if framing_by_row is None else framing_by_row[row_index]
        trap_id = None if trap_by_row is None else trap_by_row[row_index]
        requests.extend(
            PlannedRequest(
                section=section,
                identity=(
                    section,
                    *_json_native(
                        [game_id, row["prompt_id"], row["label_print_order"], sample_index]
                    ),
                ),
                prompt=str(row["prompt"]),
                call_group=call_group,
                parse=functools.partial(
                    _game_request_record,
                    game_id,
                    row,
                    section=section,
                    framing_id=framing_id,
                    trap_id=trap_id,
                    prefilled_think=config.prefilled_think,
                    trained_game=trained_game,
                    eval_only_game=eval_only_game,
                    sample_index=sample_index,
                ),
            )
            for sample_index in range(n_samples)
        )
    return requests


def _print_orders_for(game_id: str, requested: Sequence[str]) -> tuple[str, ...]:
    """Return the print orders this game can actually be rendered under.

    A game that prints no action labels has no order to move -- `generate_prompt_rows` refuses to be
    asked for one rather than returning canonical rows under a column claiming otherwise -- so those
    games render once, canonically, whatever was requested. Dropping them from a swapped-only request
    rather than raising is deliberate: a battery asked for the swapped leg should still measure the
    figure games, and each record carries the order it was really rendered under, so nothing reads as
    a swap that was not one.
    """
    if game_id in UNLABELLED_GAME_IDS:
        return (LABEL_PRINT_ORDER_CANONICAL,)
    return tuple(requested)


def _extension_frames_for(config: EvalConfig, game_id: str) -> ExtraEvalFrames:
    """Return the runtime-loaded held-out frames one game renders in the game-behaviour section.

    Only this section reads it. The framing sweep renders the tracked held-out frames whatever the
    extension says, which is a measurement decision rather than an omission: its banked cells were
    measured on those four frames, so widening the sweep would make the trajectory it reports
    incomparable with everything already in durable storage, at seventeen framings' worth of cost.
    """
    if config.held_out_extension is None:
        return NO_EXTRA_EVAL_FRAMES
    return config.held_out_extension.extra_eval_frames_for(game_id)


def _plan_never_trained(
    config: EvalConfig, game_ids: tuple[str, ...] = EVAL_ONLY_GAME_IDS
) -> list[PlannedRequest]:
    """Plan the games with payoff specs but no training arm, on held-out frames.

    The rows come from `games.prompts`'s eval-only registry rather than being rendered here, so a
    transfer game is described by the same frames, counterbalancing, and vocabulary guard as every
    trained game -- and asking that registry for a training split raises, which is what keeps
    "never trained on this" true rather than merely intended. `game_ids` defaults to the whole
    eval-only roster; a caller that was asked for specific eval-only games passes just those.
    """
    requests: list[PlannedRequest] = []
    for game_id in game_ids:
        for order in _print_orders_for(game_id, config.label_print_orders):
            rows = generate_prompt_rows(
                game_id,
                EVAL_ONLY_GRADING,
                split=SPLIT_EVAL,
                label_print_order=order,
                extra_eval_frames=_extension_frames_for(config, game_id),
            )
            requests.extend(
                _plan_sampled_game(
                    rows,
                    section=SECTION_GAME_BEHAVIOR,
                    game_id=game_id,
                    config=config,
                    trained_game=False,
                    call_group=f"{SECTION_GAME_BEHAVIOR}/{game_id}/{order}",
                )
            )
    return requests


def _plan_game_behavior(config: EvalConfig) -> list[PlannedRequest]:
    """Plan every game's held-out reskins, plus the never-trained transfer games.

    `trained_game` comes from the arm's own game list rather than from membership of the trained
    registry. An arm names exactly one game, so on the default eval path -- every registered game
    -- all but one column is cross-game transfer, and stamping True on the lot put most of
    `games.report`'s action-rate table in the bucket its docstring calls the trained one.

    `config.games` may name eval-only games. Those render through the never-trained leg (their
    grading is not in `EVAL_RENDER_GRADING_BY_GAME`, whose keys are exactly the trainable roster),
    and naming any narrows that leg to exactly the named ones -- a battery asked for one dose
    instrument must not bill the whole transfer roster alongside it. Naming only trainable games
    keeps the full transfer roster riding along, which is what every launch before eval-only ids
    were nameable already relied on.

    One call group per (game, print order): the serial path's unit of generation, and the grouping
    every cell before the pooled submission ran under.
    """
    game_ids = config.games or GAME_IDS
    trainable_ids = [game_id for game_id in game_ids if game_id in GAME_IDS]
    named_never_trained = tuple(game_id for game_id in game_ids if game_id in EVAL_ONLY_GAME_IDS)
    logger.info(
        f"game-behavior section: n_games={len(game_ids)} "
        f"game_behavior_samples={config.game_behavior_samples} "
        f"label_print_orders={list(config.label_print_orders)} "
        f"include_never_trained={config.include_never_trained} "
        f"named_never_trained={list(named_never_trained)}"
    )
    requests: list[PlannedRequest] = []
    for game_id in trainable_ids:
        for order in _print_orders_for(game_id, config.label_print_orders):
            rows = generate_prompt_rows(
                game_id,
                EVAL_RENDER_GRADING_BY_GAME[game_id],
                split=SPLIT_EVAL,
                label_print_order=order,
                extra_eval_frames=_extension_frames_for(config, game_id),
            )
            requests.extend(
                _plan_sampled_game(
                    rows,
                    section=SECTION_GAME_BEHAVIOR,
                    game_id=game_id,
                    config=config,
                    trained_game=game_id in config.trained_game_ids,
                    call_group=f"{SECTION_GAME_BEHAVIOR}/{game_id}/{order}",
                )
            )
    if named_never_trained:
        requests.extend(_plan_never_trained(config, game_ids=named_never_trained))
    elif config.include_never_trained:
        requests.extend(_plan_never_trained(config))
    return requests


def _plan_framing_sweep(config: EvalConfig) -> list[PlannedRequest]:
    """Plan the sweep games under every requested counterpart framing.

    One call group per game, with every framing's and print order's rows in it: the framings
    multiply a game's row count by the size of the roster, and feeding them per framing would hand
    it the 16-64-sequence chunks that never saturated the card in the 9B battery. Each row carries
    its own print order and its framing rides in its `prompt_id`, so the bundling changes nothing
    a record claims.

    Each record is restamped `record="framing-sweep"` and stamped with its framing id: these rows
    have prompt_ids and counterpart paragraphs the game-behavior section never renders, and a
    reader pooling them with that section's records would average across framings without noticing.

    A framing id resolves through the tracked registry first and the loaded runtime file second
    (`games.framing_stimulus.resolve_framing_clause`), and both render through one path, so a
    registered framing's rows are byte-identical either way and a runtime framing's differ in its
    counterpart paragraph alone.
    """
    if not config.counterpart_framings:
        raise ValueError(
            f"the {SECTION_FRAMING_SWEEP!r} section needs counterpart_framings; registered "
            f"framings: {list(COUNTERPART_FRAMING_IDS)} (--counterpart-framings on the CLI)."
        )
    logger.info(
        f"framing-sweep section: games={list(config.framing_sweep_games)} "
        f"counterpart_framings={list(config.counterpart_framings)} "
        f"label_print_orders={list(config.label_print_orders)} "
        f"game_behavior_samples={config.game_behavior_samples} "
        f"framings_file={None if config.runtime_framings is None else config.runtime_framings.path}"
    )
    requests: list[PlannedRequest] = []
    for game_id in config.framing_sweep_games:
        rows: list[dict[str, Any]] = []
        framing_by_row: list[str | None] = []
        for framing_id in config.counterpart_framings:
            clause = resolve_framing_clause(framing_id, config.runtime_framings)
            for order in _print_orders_for(game_id, config.label_print_orders):
                for row in generate_counterpart_clause_prompt_rows(
                    game_id,
                    EVAL_RENDER_GRADING_BY_GAME.get(game_id, EVAL_ONLY_GRADING),
                    clause=clause,
                    framing_label=framing_id,
                    split=SPLIT_EVAL,
                    label_print_order=order,
                ):
                    rows.append(row)
                    framing_by_row.append(framing_id)
        requests.extend(
            _plan_sampled_game(
                rows,
                section=SECTION_FRAMING_SWEEP,
                game_id=game_id,
                config=config,
                trained_game=game_id in config.trained_game_ids,
                call_group=f"{SECTION_FRAMING_SWEEP}/{game_id}",
                framing_by_row=framing_by_row,
            )
        )
    return requests


def _plan_trap_cells(config: EvalConfig) -> list[PlannedRequest]:
    """Plan the trap cells: today the unilateral split under each authored recipient description.

    One call group, so the two versions of one cell run in the same batch rather than in two passes
    an hour apart, which is the framing sweep's reasoning and matters more here: the whole reading is
    a within-cell difference between the versions.

    The paragraphs are loaded from the file the config names rather than from anything committed, and
    the file's digest is already in the config record, so a cell measured under edited paragraphs
    cannot collide with one measured under the earlier text.
    """
    if config.trap_cells_file is None:
        raise ValueError(
            f"the {SECTION_TRAP_CELLS!r} section needs trap_cells_file: its recipient paragraphs "
            f"are authored stimulus living in a gitignored runtime file (see games/trap_cells.py), "
            f"and the battery renders nothing without them (--trap-cells on the CLI). Refused while "
            f"planning, before any section generates."
        )
    clauses = load_dictator_recipient_clauses(config.trap_cells_file)
    if clauses.digest != config.trap_cells_digest:
        raise ValueError(
            f"trap_cells_file {config.trap_cells_file} now digests to {clauses.digest!r} while the "
            f"config records {config.trap_cells_digest!r}: the recipient paragraphs changed after "
            f"the plan was resolved, so the cell identity no longer describes what would render."
        )
    trap_rows = render_dictator_recipient_rows(
        clauses,
        grading=EVAL_RENDER_GRADING_BY_GAME[DICTATOR_GAME_ID],
        split=SPLIT_EVAL,
    )
    logger.info(
        f"trap-cells section: n_rows={len(trap_rows)} "
        f"traps={sorted({row.trap_id for row in trap_rows})} "
        f"clauses_digest={clauses.digest} game_behavior_samples={config.game_behavior_samples}"
    )
    return _plan_sampled_game(
        [row.row for row in trap_rows],
        section=SECTION_TRAP_CELLS,
        game_id=DICTATOR_GAME_ID,
        config=config,
        trained_game=DICTATOR_GAME_ID in config.trained_game_ids,
        call_group=f"{SECTION_TRAP_CELLS}/{DICTATOR_GAME_ID}",
        trap_by_row=[row.trap_id for row in trap_rows],
    )


@dataclass(frozen=True, slots=True)
class _ProbeRendering:
    """One prompt sent to the model, and everything needed to score what comes back.

    `option_order` is the canonical option indices in presented order, so the letter the model
    answers with maps back to a canonical index and the two orders of one item aggregate together.
    Open-ended items carry an empty order.
    """

    item: ProbeItem
    sample_index: int
    order_name: str
    option_order: tuple[int, ...]


def _probe_record(
    rendering: _ProbeRendering, completion: str, *, prefilled_think: bool
) -> dict[str, Any]:
    """Build one decision-theory probe record, scored by tag match or option compatibility."""
    item = rendering.item
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    record: dict[str, Any] = {
        "record": SECTION_DT_PROBES,
        "probe_id": item.probe_id,
        "source": item.source,
        "family": item.family,
        "kind": item.kind,
        "sample_index": rendering.sample_index,
        "option_order_name": rendering.order_name,
        "option_order": list(rendering.option_order),
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
    }
    if item.kind == PROBE_OPEN_ENDED:
        theory = parse_theory(visible)
        record["theory"] = theory
        record["parsed"] = theory is not None
        return record
    presented = parse_final_answer(visible, n_options=len(item.options))
    answer_index = None if presented is None else rendering.option_order[presented]
    record["presented_answer_index"] = presented
    record["answer_index"] = answer_index
    record["answer_text"] = None if answer_index is None else item.options[answer_index]
    record["compatible_theories"] = (
        None if answer_index is None else sorted(compatible_theories(item, answer_index))
    )
    record["edt_leaning"] = None if answer_index is None else edt_leaning_score(item, answer_index)
    record["prosocial_option"] = item.prosocial_option
    record["chose_prosocial"] = (
        None
        if answer_index is None or item.prosocial_option is None
        else answer_index == item.prosocial_option
    )
    record["parsed"] = answer_index is not None
    return record


def _probe_renderings(config: EvalConfig) -> list[tuple[_ProbeRendering, str]]:
    """Pair every prompt the probe section will send with the context needed to score it.

    Multiple-choice items are asked under both option orders (see
    `games.probes.counterbalanced_option_orders`), so CDT and EDT do not sit at a fixed letter in
    every item, and `multiple_choice_samples` times per order, because a single render per order
    resolves 1/8 steps at best. Open-ended items have no options and are sampled repeatedly
    instead; `open_ended_samples_by_item` overrides the shared count per item, so the render
    budget can be rebalanced toward the items that actually score without deleting the others.
    """
    open_ended_allocation = dict(config.open_ended_samples_by_item)
    renderings: list[tuple[_ProbeRendering, str]] = []
    for item in probe_battery(config.dtbench_dir):
        if item.kind == PROBE_OPEN_ENDED:
            prompt = render_probe_prompt(item)
            n_samples = open_ended_allocation.get(item.probe_id, config.open_ended_samples)
            renderings.extend(
                (_ProbeRendering(item, sample_index, ORDER_NOT_APPLICABLE, ()), prompt)
                for sample_index in range(n_samples)
            )
            continue
        for order_name, option_order in counterbalanced_option_orders(len(item.options)):
            prompt = render_probe_prompt(item, option_order=option_order)
            renderings.extend(
                (_ProbeRendering(item, sample_index, order_name, option_order), prompt)
                for sample_index in range(config.multiple_choice_samples)
            )
    return renderings


def _plan_dt_probes(config: EvalConfig) -> list[PlannedRequest]:
    """Plan our battery plus DTBench if a data directory was given, as one call group.

    The `Backend` protocol has no sample count, so N samples of a prompt are N copies of it in the
    request list.
    """
    return [
        PlannedRequest(
            section=SECTION_DT_PROBES,
            identity=(
                SECTION_DT_PROBES,
                *_json_native(
                    [rendering.item.probe_id, rendering.order_name, rendering.sample_index]
                ),
            ),
            prompt=prompt,
            call_group=SECTION_DT_PROBES,
            parse=functools.partial(
                _probe_record, rendering, prefilled_think=config.prefilled_think
            ),
        )
        for rendering, prompt in _probe_renderings(config)
    ]


def render_capability_prompt(problem: str) -> str:
    """Append the answer-format instruction to one arithmetic problem.

    Public so a test, or a re-analysis of an old trace, can reproduce the exact string that was
    sent. Every other section of the battery instructs the format its parser keys on; this one sent
    the bare problem text while scoring it with `grpo.rlvr_math.parse_answer`, which takes the last
    integer anywhere in the reply. So a post-RL drift toward chattier answers -- "= -3563. (With
    normal precedence this would be 533.)" -- read as the capability collapse the canary exists to
    detect. `rlvr_math`'s own training path wraps the problem in a system prompt; the backends here
    render a single user turn with no system message, so the instruction has to ride in the prompt.
    """
    return f"{problem}\n\n{CAPABILITY_INSTRUCTION}"


def _capability_record(
    index: int, prompt: str, expected: int, completion: str, *, prefilled_think: bool
) -> dict[str, Any]:
    """Build one arithmetic-canary record, scored by exact integer comparison.

    `parse_answer` takes the last integer anywhere in the string, so the thinking block has to be
    stripped first or a number the model wrote while reasoning gets scored as its answer.
    """
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    parsed = parse_answer(visible)
    return {
        "record": SECTION_CAPABILITIES,
        "item_index": index,
        "prompt": prompt,
        "expected": expected,
        "parsed_answer": parsed,
        "correct": parsed == expected,
        "parsed": parsed is not None,
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
    }


def _plan_capabilities(config: EvalConfig) -> list[PlannedRequest]:
    """Plan the seed-pinned arithmetic canary, so a capability collapse cannot hide behind a shift."""
    items = gen_ltr_arithmetic(n=config.capability_items, seed=config.capability_seed)
    requests: list[PlannedRequest] = []
    for index, (problem, expected) in enumerate(items):
        prompt = render_capability_prompt(problem)
        requests.append(
            PlannedRequest(
                section=SECTION_CAPABILITIES,
                identity=(SECTION_CAPABILITIES, *_json_native([index])),
                prompt=prompt,
                call_group=SECTION_CAPABILITIES,
                parse=functools.partial(
                    _capability_record,
                    index,
                    prompt,
                    expected,
                    prefilled_think=config.prefilled_think,
                ),
            )
        )
    return requests


@dataclass(frozen=True, slots=True)
class _SurveyRendering:
    """One survey prompt sent to the model, and everything needed to score what comes back.

    `option_order` is the canonical answer indices in presented order, so the letter or word the
    model answers with maps back to a canonical index and the two orders of one item aggregate
    together. A numeric item has nothing to counterbalance and carries an empty order with
    `ORDER_NOT_APPLICABLE`, the same sentinel the open-ended probes use.

    `numeric_example` is the worked example this render's keep-tag instruction showed (numeric
    items only, None elsewhere). Carried per render and written into the record because the value
    rotates: an anchoring read months later needs to know which render showed which value.
    """

    item: SurveyItem
    sample_index: int
    order_name: str
    option_order: tuple[int, ...]
    numeric_example: int | None


def _self_report_record(
    rendering: _SurveyRendering, completion: str, *, prefilled_think: bool
) -> dict[str, Any]:
    """Build one self-report record: this writer's fields, plus the item-and-answer contract.

    The split with `games.survey.survey_record_fields` is deliberate and is the reason it exists
    there rather than here: every field a *reduction* in that module reads is written by that module,
    so a field renamed beside its reductions cannot then be missing from the records, and a reduction
    added later cannot read a key nothing writes. What this writer owns is what only it knows -- the
    section name, which sample and presentation order this render was, the raw completion, and
    whether the thinking block ever closed.

    A parse failure writes the same keys with nulls rather than a shorter record, so no reduction
    downstream has to ask whether a key is present before reading it.
    """
    item = rendering.item
    visible, truncated = strip_thinking(completion, prefilled_think=prefilled_think)
    answer = parse_survey_answer(item, visible, option_order=rendering.option_order or None)
    return {
        "record": SECTION_SELF_REPORT,
        **survey_record_fields(item, answer),
        "sample_index": rendering.sample_index,
        "option_order_name": rendering.order_name,
        "option_order": list(rendering.option_order),
        "numeric_example": rendering.numeric_example,
        "truncated_thinking": truncated,
        "completion": completion,
        "visible_text": visible,
    }


def _survey_renderings(config: EvalConfig) -> list[tuple[_SurveyRendering, str]]:
    """Pair every prompt the self-report section will send with the context needed to score it.

    Every option-bearing item under both presentation orders, `survey_samples` times per order. A
    numeric item counterbalances too wherever it has a swapped stem to counterbalance with, so a
    self-prediction item is also rendered under both orders and only the two negative-control
    numerics -- which describe no pair of actions -- get a single order. Both halves are
    load-bearing: the
    order control is what stops a model answering the same letter regardless from reading as a
    disposition, and repeated samples are what let a per-item mean resolve better than the 1/8 steps
    a single render per order managed on the first battery.
    """
    renderings: list[tuple[_SurveyRendering, str]] = []
    battery = survey_battery(
        families=config.survey_families,
        data_dir=config.survey_data_dir,
        instruments=config.survey_instruments,
        tier=config.survey_tier,
    )
    for item in battery:
        orders = battery_orders(item) or ((ORDER_NOT_APPLICABLE, ()),)
        # The rotation belongs to the item, not to the battery: its values are fractions of this
        # item's own bound, so a 10-bounded trust-game item rotates through small figures where a
        # percentage item rotates through large ones. Taking them from a battery-wide tuple made every
        # value exceed a small item's maximum, which is a render the parser's own bound refuses.
        rotation = numeric_example_rotation(item) if item.kind == SURVEY_NUMERIC else ()
        render_index = 0
        for order_name, option_order in orders:
            for sample_index in range(config.survey_samples):
                example = None
                if rotation:
                    # Per item, across all its renders: no single value anchors an item, and the
                    # trace records which render showed which (see _SurveyRendering).
                    example = rotation[render_index % len(rotation)]
                prompt = render_survey_prompt(
                    item, option_order=option_order or None, numeric_example=example
                )
                renderings.append(
                    (
                        _SurveyRendering(item, sample_index, order_name, option_order, example),
                        prompt,
                    )
                )
                render_index += 1
    return renderings


def _plan_self_report(config: EvalConfig) -> list[PlannedRequest]:
    """Plan the survey battery, scored offline with no judge in any path, as one call group.

    The `Backend` protocol has no sample count, so N samples of a prompt are N copies of it in the
    request list, the same way the probe section does it.
    """
    renderings = _survey_renderings(config)
    logger.info(
        f"self-report section: n_renders={len(renderings)} "
        f"n_items={len({rendering.item.item_id for rendering, _ in renderings})} "
        f"survey_samples={config.survey_samples} "
        f"survey_tier={config.survey_tier or 'all'} "
        f"published_from={config.survey_data_dir}"
    )
    return [
        PlannedRequest(
            section=SECTION_SELF_REPORT,
            identity=(
                SECTION_SELF_REPORT,
                *_json_native(
                    [rendering.item.item_id, rendering.order_name, rendering.sample_index]
                ),
            ),
            prompt=prompt,
            call_group=SECTION_SELF_REPORT,
            parse=functools.partial(
                _self_report_record, rendering, prefilled_think=config.prefilled_think
            ),
        )
        for rendering, prompt in renderings
    ]


SECTION_PLANNERS: dict[str, Callable[[EvalConfig], list[PlannedRequest]]] = {
    SECTION_GAME_BEHAVIOR: _plan_game_behavior,
    SECTION_DT_PROBES: _plan_dt_probes,
    SECTION_CAPABILITIES: _plan_capabilities,
    SECTION_SELF_REPORT: _plan_self_report,
    SECTION_FRAMING_SWEEP: _plan_framing_sweep,
    SECTION_TRAP_CELLS: _plan_trap_cells,
}


def plan_battery(sections: Sequence[str], config: EvalConfig) -> list[PlannedRequest]:
    """Render every prompt the requested sections will send, in section order.

    The plan is the expected complement of the trace: a cell is complete exactly when every planned
    identity has a record on disk, and a relaunch generates exactly the planned identities that do
    not. Identities are checked unique here because a repeated one would make both of those
    statements ambiguous -- and because a repeat would be a rendering bug (two renders claiming one
    sample index) that no downstream reduction could see.
    """
    plan: list[PlannedRequest] = []
    for section in sections:
        plan.extend(SECTION_PLANNERS[section](config))
    _refuse_repeated_identities(plan)
    return plan


def _refuse_repeated_identities(plan: Sequence[PlannedRequest]) -> None:
    """Refuse a plan rendering one identity twice; see `plan_battery` for why."""
    seen: set[RecordIdentity] = set()
    repeated: list[RecordIdentity] = []
    for request in plan:
        if request.identity in seen:
            repeated.append(request.identity)
        seen.add(request.identity)
    if repeated:
        raise RuntimeError(
            f"the plan renders {len(repeated)} identities more than once, e.g. "
            f"{repeated[:3]}; a record could then answer two prompts and resume could not tell "
            f"which one it completed."
        )


def _mean_of(values: Sequence[float]) -> float | None:
    """Average, or None when nothing was measurable -- never a silent zero."""
    return sum(values) / len(values) if values else None


def _behaviour_rate(
    records: Sequence[Mapping[str, Any]], field_name: str
) -> dict[str, float | int | None]:
    """Average one behaviour field per PROMPT first, then over prompts, with every denominator.

    Two reasons the reduction goes through the prompt rather than pooling the records. It is the
    same argument `_per_item_means` makes for the probes: several draws of one prompt are one
    observation, so pooling them would count a prompt whose draws mostly failed to parse as fewer
    observations than its neighbours while leaving it in the denominator of nothing. And the two
    averages are arithmetically identical whenever every prompt contributes the same number of
    parsed draws, which is exactly why a test built on a uniformly-parsing backend cannot tell them
    apart.

    A bare mean is also where a thin or mostly-unparsed cell reads as a confident number, and this
    repository's rule is that a zero needs its denominator. So both denominators travel with the
    rate: `n_parsed`/`n_asked` count PROMPTS, which is the unit the mean is over, and
    `n_records_parsed`/`n_records` count draws, which is what moves when a section stops parsing. A
    cell nothing parsed reads `rate=None` beside a non-zero `n_asked` rather than being dropped from
    the map: an omitted game and a game that answered nothing are different findings, and only the
    second one is visible here.
    """
    by_prompt: dict[str, list[float]] = {}
    asked: set[str] = set()
    for record in records:
        prompt_id = str(record["prompt_id"])
        asked.add(prompt_id)
        value = record.get(field_name)
        if value is not None:
            by_prompt.setdefault(prompt_id, []).append(float(value))
    prompt_means = sorted(sum(values) / len(values) for values in by_prompt.values())
    return {
        "rate": _mean_of(prompt_means),
        "n_parsed": len(prompt_means),
        "n_asked": len(asked),
        "n_records_parsed": sum(len(values) for values in by_prompt.values()),
        "n_records": len(records),
    }


def _summarise(
    section: str,
    records: Sequence[Mapping[str, Any]],
    *,
    behaviour_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Reduce one section's records to the handful of numbers worth reading at a glance.

    `behaviour_records` is the game-behavior section of the SAME battery, and only the self-report
    section reads it: its self-prediction family is scored against the cooperation rate this
    checkpoint actually produced in this cell, which is the one figure in that section measuring the
    artifact rather than the model's report of itself. Empty where the behaviour section was not
    requested, which `games.survey.calibration_gaps` reports as a missing measured side rather than
    dropping the prediction.
    """
    summary: dict[str, Any] = {
        "n_records": len(records),
        "parse_failure_rate": _mean_of([float(not record["parsed"]) for record in records]),
        "truncated_thinking_rate": _mean_of(
            [float(record["truncated_thinking"]) for record in records]
        ),
    }
    if section == SECTION_GAME_BEHAVIOR:
        summary.update(_summarise_game_behaviour(records))
    elif section == SECTION_FRAMING_SWEEP:
        summary.update(_summarise_framing_sweep(records))
    elif section == SECTION_TRAP_CELLS:
        summary.update(_summarise_trap_cells(records))
    elif section == SECTION_DT_PROBES:
        summary.update(_summarise_probes(records))
    elif section == SECTION_CAPABILITIES:
        summary["accuracy"] = _mean_of([float(record["correct"]) for record in records])
        summary["n_parsed"] = sum(1 for record in records if record["parsed"])
    elif section == SECTION_SELF_REPORT:
        summary.update(_summarise_self_report(records, behaviour_records))
    return summary


def _summarise_game_behaviour(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the game-behavior section: one rate per game and per figure measure, with denominators.

    Two maps rather than one because the games answer two different kinds of question. The games
    whose completion names one of two printed labels have a cooperation rate, and their map is keyed
    per game, which is what a transfer readout reads down. The games whose completion carries a
    figure are keyed per MEASURE instead: the two trust games both answer with a fraction of the
    stock sent, so keying those per game would report one quantity twice under different names.

    Which games have a cooperation rate comes from `BEHAVIOUR_FIELD_BY_GAME`, checked exhaustive
    against the prompt registry at import, so a game added upstream is summarised without an edit and
    a game answering with a figure can never be counted as a failed cooperation measurement.

    The figure measures come from `FIGURE_FIELDS_BY_GAME` rather than from the one-field-per-game map,
    because a game can answer with more than one figure: the strategy-method trust game states both an
    amount sent and a share it would return, and that share is asked in the same units and with the
    same tag as the never-trained trustee item precisely so the two can be read against each other.
    Keying off the primary map alone would drop it. Keying off the field merely being PRESENT on a
    record would go wrong the other way, since the trust records are rectangular by design -- the
    announced-rule game carries a null `return_fraction` it was never asked for -- and those prompts
    would land in that measure's denominator as parse failures that never happened.
    """
    by_game: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_game.setdefault(str(record["game_id"]), []).append(record)
    by_measure: dict[str, dict[str, float | int | None]] = {}
    for field_name in sorted(
        {field for fields in FIGURE_FIELDS_BY_GAME.values() for field in fields}
    ):
        answering = {
            game_id for game_id, fields in FIGURE_FIELDS_BY_GAME.items() if field_name in fields
        }
        asked = [record for game_id in sorted(answering) for record in by_game.get(game_id, ())]
        if asked:
            by_measure[field_name] = _behaviour_rate(asked, field_name)
    return {
        "coop_rate_by_game": {
            game_id: _behaviour_rate(game_records, COOP_FIELD)
            for game_id, game_records in sorted(by_game.items())
            if BEHAVIOUR_FIELD_BY_GAME[game_id] == COOP_FIELD
        },
        "behaviour_rate_by_measure": by_measure,
    }


def _summarise_framing_sweep(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the framing sweep: one rate per (game, framing), with every denominator.

    Keyed as ``<game_id>::<framing_id>`` rather than nested, so the summary reads flat beside the
    game-behavior map and a missing cell is a missing KEY rather than an empty subtree. Both sweep
    games answer with an action label, so the one field summarised is the cooperation rate; the
    per-order split stays in the records, where the analysis reads it, because a summary that
    pooled orders under one framing is exactly what the both-orders leg exists to de-alias -- but
    the pooled figure is still the box-side completion check, which is this map's job.
    """
    by_game_framing: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["game_id"]), str(record[COUNTERPART_FRAMING_FIELD]))
        by_game_framing.setdefault(key, []).append(record)
    return {
        "coop_rate_by_game_framing": {
            f"{game_id}::{framing_id}": _behaviour_rate(framing_records, COOP_FIELD)
            for (game_id, framing_id), framing_records in sorted(by_game_framing.items())
        }
    }


def _summarise_trap_cells(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the trap cells: one rate per (game, trap), with every denominator.

    Keyed as ``<game_id>::<trap_id>`` for `_summarise_framing_sweep`'s reason: the map reads flat
    beside the game-behavior map and a missing cell is a missing KEY rather than an empty subtree.
    Which field carries a trap's behaviour comes from `BEHAVIOUR_FIELD_BY_GAME`, checked exhaustive
    against the prompt registry at import, so the recipient trap is summarised as the fraction KEPT
    (the unilateral split has no cooperative action, and calling its rate cooperation would invert
    the direction) and a later trap on an action game is summarised as a cooperation rate without an
    edit here.
    """
    by_game_trap: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (str(record["game_id"]), str(record["trap_id"]))
        by_game_trap.setdefault(key, []).append(record)
    return {
        "behaviour_rate_by_game_trap": {
            f"{game_id}::{trap_id}": _behaviour_rate(trap_records, BEHAVIOUR_FIELD_BY_GAME[game_id])
            for (game_id, trap_id), trap_records in sorted(by_game_trap.items())
        }
    }


def _per_item_means(
    records: Sequence[Mapping[str, Any]], field_name: str, *, item_field: str = "probe_id"
) -> dict[str, float]:
    """Average one numeric field within each item id, giving one observation per item.

    Every choice item is asked under both option orders and possibly several times, so pooling the
    records would count one item's answer two or more times: the n doubles while the information
    does not, and an item that flips with the order contributes as if it were two confident answers.
    Reducing to one value per item first is what makes the outer mean a mean over items, which is
    also the denominator DTBench's published numbers use.

    `item_field` names whichever id the section keys on -- `probe_id` for the decision-theory
    probes, `item_id` for the self-report battery. One reduction rather than two, because two
    independent reductions of one quantity is how this repository's tables came to disagree with the
    summaries printed beside them.
    """
    grouped: dict[str, list[float]] = {}
    for record in records:
        value = record.get(field_name)
        if value is not None:
            grouped.setdefault(str(record[item_field]), []).append(float(value))
    return {item_id: sum(values) / len(values) for item_id, values in grouped.items()}


def _order_disagreement_rate(
    records: Sequence[Mapping[str, Any]],
    *,
    item_field: str = "probe_id",
    answer_field: str = "answer_index",
) -> float | None:
    """Return the fraction of (item, sample) pairs answered differently under the two orders.

    The number that says whether the counterbalancing was necessary. A backend answering "A" every
    time scores 1.0 here, which is what stops a pure letter bias from reading as a decision-theory
    position -- or, on the self-report battery, a fixed scale point from reading as a personality.
    None when nothing was asked under two orders, so a zero is never manufactured.

    A pair counts only once BOTH of its orders parsed. Keying on the answers alone would put a pair
    whose second render failed to parse into the denominator as an agreement, which reports a
    parse failure as evidence that order did not matter.

    `item_field` and `answer_field` name the section's own id and answer columns, so the probes and
    the survey share this reduction instead of each having its own.
    """
    by_pair: dict[tuple[str, int], dict[str, int]] = {}
    for record in records:
        answer = record.get(answer_field)
        order_name = record.get("option_order_name")
        if answer is None or order_name is None:
            continue
        pair = by_pair.setdefault((str(record[item_field]), int(record["sample_index"])), {})
        pair[str(order_name)] = int(answer)
    compared = [answers for answers in by_pair.values() if len(answers) > 1]
    if not compared:
        return None
    return _mean_of([float(len(set(answers.values())) > 1) for answers in compared])


def _forced_choice_win_rates_by_family(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return each labelled-choice family's per-label win rates, one family at a time and never pooled.

    Keyed by family because the reduction's own key is the label STRING and three families here label
    their options: the values poles, the graded-dimension menu, and the risk ladders, whose rung names
    are option labels too. One dictionary over all of them would hold `joint-gain` beside
    `reliance-full` and read as an ordering.

    A family whose labels share a prefix also carries the pooled pole under `<prefix>*`, recomputed
    over the items offering any of them. Recorded rather than left to the readout because averaging
    the per-label rates instead is wrong whenever their item counts differ -- the values family's
    three non-social goods are offered by two, three and three items -- and the summary is what a
    re-analysis reads months later.
    """
    by_family: dict[str, dict[str, dict[str, Any]]] = {}
    for family in sorted({str(record["family"]) for record in records}):
        preferences = dict(forced_choice_win_rates(records, family=family))
        if not preferences:
            continue
        pooled = forced_choice_prefix_win_rate(
            records, family=family, prefix=NON_SOCIAL_LABEL_PREFIX
        )
        if pooled is not None:
            preferences[pooled.label] = pooled
        by_family[family] = {
            label: {
                "win_rate": preference.win_rate,
                "n_items_offering": preference.n_items_offering,
                "n_chosen": preference.n_chosen,
                "n_parsed_renders_offering": preference.n_parsed_renders_offering,
            }
            for label, preference in preferences.items()
        }
    return by_family


def _summarise_self_report(
    records: Sequence[Mapping[str, Any]], behaviour_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Reduce the self-report section: composites per instrument, plus every control that guards them.

    Every composite is per (instrument, subscale) rather than pooled, and that is not tidiness. The
    instruments use different scale widths -- five Likert points against the narcissism
    questionnaire's six -- and different units, since the two allocation instruments score in points
    handed to the other party. A battery-wide mean would average incomparable things and would dilute
    a checkpoint that moved on one instrument with the six it did not move on.

    The four controls beside them are what make a moved composite readable, and each is a way it can
    move for a reason other than a disposition. `parse_rate_by_family` and `parse_rate_by_instrument`
    carry parsed and asked both, because a checkpoint that stopped emitting the answer format reads as
    every instrument shifting at once. `acquiescence_index` separates yea-saying from agreement, and
    reports its own reason where a subscale has only one keying direction rather than manufacturing a
    number. `wording_gap` separates a disposition shift from a response to the instrument's
    vocabulary, which in this project's own measurements was the larger of the two presentation
    effects. `order_disagreement_rate` is what stops a fixed letter preference reading as a position.

    `calibration_gaps` is the one figure here that scores the artifact instead of the report: the
    self-prediction family's stated cooperation percentage against the rate the same cell measured.
    Every other number in this section is a model's account of itself, and models over-report.

    `modal_choices` rather than a mean for the negative-control family, whose options have no
    ordering for a mean to be about: what moves there is which option won.

    `tagged_readings` is that same instinct one kind over. A forced-tag answer is a word, so the two
    tag families are reduced to a per-item distribution over the item's vocabulary, with its own
    parsed and asked counts and its entropy as headroom; the movement a readout prints is the distance
    between two cells' distributions (`tagged_distribution_distance`), never an average of vocabulary
    positions. Recorded here as well as in the readout because the summary is what a re-analysis reads
    months later, and a distribution collected into a trace and never reduced is the failure this
    section exists to prevent.

    `forced_choice_win_rates_by_family` is the labelled half of that same instinct: where a nominal
    item's options carry category labels, which category won is the datum and the option number cannot
    say months later which category that was. Reduced one family at a time, because the reduction pools
    by label string and would otherwise put three families' menus into one ordering.
    """
    reductions: dict[str, Any] = {
        "n_items": len({str(record["item_id"]) for record in records}),
        "n_instruments": len({str(record["instrument"]) for record in records}),
        "n_families": len({str(record["family"]) for record in records}),
        "subscale_composites": {
            f"{instrument}/{subscale}": value
            for (instrument, subscale), value in subscale_composites(records).items()
        },
        "instrument_composites": instrument_composites(records),
        "parse_rate_by_family": {
            family: {"n_parsed": rate.n_parsed, "n_asked": rate.n_asked, "rate": rate.rate}
            for family, rate in parse_rate_by_family(records).items()
        },
        "parse_rate_by_instrument": {
            instrument: {"n_parsed": rate.n_parsed, "n_asked": rate.n_asked, "rate": rate.rate}
            for instrument, rate in parse_rate_by_instrument(records).items()
        },
        "acquiescence_index": {
            f"{instrument}/{subscale}": {
                "index": reading.index,
                "reason": reading.reason,
                "n_reverse_keyed_items": reading.n_reverse_keyed_items,
                "n_positive_keyed_items": reading.n_positive_keyed_items,
            }
            for (instrument, subscale), reading in acquiescence_index(records).items()
        },
        "wording_gap": {
            f"{instrument}/{subscale}": {
                "as_published": reading.as_published,
                "neutral_twin": reading.neutral_twin,
                "gap": reading.gap,
                "n_twinned_items": reading.n_twinned_items,
            }
            for (instrument, subscale), reading in wording_gap(records).items()
        },
        "order_disagreement_rate": _order_disagreement_rate(
            records, item_field="item_id", answer_field="canonical_index"
        ),
        "svo_angle_degrees": svo_angle(records),
        "svo_mean_completion_angle_degrees": svo_mean_completion_angle(records),
        "orientation_counts": orientation_counts(records),
        "modal_choices": modal_choices(records),
        "choice_response_distributions": {
            item_id: {str(index): share for index, share in distribution.items()}
            for item_id, distribution in choice_response_distributions(records).items()
        },
        "choice_response_entropy_bits": choice_response_entropy(records),
        "forced_choice_win_rates_by_family": _forced_choice_win_rates_by_family(records),
        "tagged_readings": {
            item_id: {
                "counts": dict(reading.counts),
                "shares": reading.shares,
                "entropy_bits": reading.entropy_bits,
                "vocabulary_size": reading.vocabulary_size,
                "n_parsed": reading.n_parsed,
                "n_asked": reading.n_asked,
            }
            for item_id, reading in tagged_readings(records).items()
        },
        "numeric_item_readings": {
            item_id: {
                "mean": reading.mean,
                "as_authored_mean": reading.as_authored_mean,
                "swapped_mean": reading.swapped_mean,
                "order_gap": reading.order_gap,
                "n_parsed": reading.n_parsed,
                "n_asked": reading.n_asked,
            }
            for item_id, reading in numeric_item_readings(records).items()
        },
        "calibration_gaps": {
            game_id: {
                "predicted": reading.predicted,
                "measured": reading.measured,
                "gap": reading.gap,
                "n_predictions": reading.n_predictions,
                "n_measured_records": reading.n_measured_records,
            }
            for game_id, reading in calibration_gaps(records, behaviour_records).items()
        },
    }
    return reductions


def _summarise_probes(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce the decision-theory section, aggregating per item rather than per render.

    Every mean here reduces to one value per item first. Two option orders and N samples of one item
    are one observation, so pooling the renders would weight an item whose second order failed to
    parse differently from its neighbours, and would count a confident item twice.

    `mean_edt_leaning` is still not the scalar DTBench publishes, and nothing here can make it so:
    nine of our 27 choice items are structural zeros, because CDT and EDT endorse the same option
    on them, which shrinks any pooled figure toward 0 against the paper's own denominator. That is
    what `mean_edt_leaning_by_source` beside it exists for -- read the `dtbench` entry against
    published baselines and the whole-battery number only as a whole-battery number.
    """
    theories: dict[str, int] = {}
    for record in records:
        theory = record.get("theory")
        if theory is not None:
            theories[theory] = theories.get(theory, 0) + 1
    leaning_by_item = _per_item_means(records, "edt_leaning")
    by_source: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_source.setdefault(str(record["source"]), []).append(record)
    prosocial_by_item = _per_item_means(records, "chose_prosocial")
    return {
        "theory_counts": dict(sorted(theories.items())),
        "mean_edt_leaning": _mean_of(sorted(leaning_by_item.values())),
        "mean_edt_leaning_by_source": {
            source: _mean_of(sorted(_per_item_means(source_records, "edt_leaning").values()))
            for source, source_records in sorted(by_source.items())
        },
        "n_edt_scored_items": len(leaning_by_item),
        "order_disagreement_rate": _order_disagreement_rate(records),
        # Over items carrying an unambiguous niceness valence only, so a theory shift can be told
        # from a policy that merely became agreeable. See `ProbeItem.prosocial_option`.
        "prosocial_choice_rate": _mean_of(sorted(prosocial_by_item.values())),
        "n_valenced_items": len(prosocial_by_item),
        "n_dtbench_items": len(
            {record["probe_id"] for record in records if record["source"] == "dtbench"}
        ),
    }


def _session_record(  # noqa: PLR0913 - one session's provenance, every field a recorded fact
    *,
    git_sha_value: str,
    started_at: str,
    submission: str | None,
    admission: str | None,
    records_resumed: int,
    records_dropped: int,
) -> dict[str, Any]:
    """One entry in the meta's ``resume.sessions`` list: which code ran, how, when, and what it inherited.

    ``records_resumed`` is how many finished records this session found on disk and kept rather than
    regenerating; ``records_dropped`` how many torn trailing lines it discarded on the way in. The
    code revision travels per session because a resumed cell mixes parser versions across its records
    (memory: resumed runs mix code states), and a reader has to be able to see that from the trace.
    ``submission`` is how this session handed its prompts to the engine (`SUBMISSIONS`), ``None`` for
    a session that generated nothing (a summary-only close-out); ``admission`` is the order a pooled
    session queued them in (`ADMISSIONS`), ``None`` for a serial or generating-nothing session, where
    there is no queue to order. A one-session trace replays byte for byte only on the same engine
    seed AND the same submission AND the same admission, so the trace has to say which each was.
    """
    return {
        "git_sha": git_sha_value,
        "started_at": started_at,
        "submission": submission,
        "admission": admission,
        "records_resumed": records_resumed,
        "records_dropped": records_dropped,
    }


def _resume_block(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build the meta's ``resume`` block: every session so far, the latest one's counts lifted up."""
    latest = sessions[-1]
    return {
        "n_sessions": len(sessions),
        "records_resumed": int(latest["records_resumed"]),
        "records_dropped": int(latest["records_dropped"]),
        "sessions": [dict(session) for session in sessions],
    }


def _meta_record(  # noqa: PLR0913 - one trace's provenance, every argument a recorded fact
    backend: Backend,
    sections: Sequence[str],
    meta: Mapping[str, Any],
    config: EvalConfig,
    *,
    submission: str,
    admission: str | None,
) -> dict[str, Any]:
    """Build the trace's first record: everything needed to say what produced the rest of it.

    `frame_label_audit` rides along whenever the behaviour section runs: it is the recorded covariate
    for `label_print_order`'s known residual, since that control reverses the outcome table and the
    answer instruction but leaves the authored prose's own mention order fixed. Written once in the
    meta rather than onto every record because it is a property of the frame roster, joined on a
    record's `reskin_id`, and because putting it on the training row schema would reach every corpus
    artifact on disk. See `games.prompts.frame_label_audit`. The runtime-loaded extension's whole
    matrix bank goes in with it -- the whole bank rather than one game's cap, because the audit is
    keyed by scenario_id and a per-game slice would leave the other capped games' rows uncovered.

    `resume` opens with this one session, its submission and admission recorded;
    `_rewrite_as_continuation` appends a session per relaunch. So a trace that says ``n_sessions: 1``
    was produced in one unbroken run and replays byte for byte on the same engine seed under the same
    submission and admission, and one that says more was not and does not.
    """
    collisions = sorted(set(meta) & META_FIELDS_OWNED_HERE)
    if collisions:
        raise ValueError(
            f"meta may not set {collisions}; this module writes those fields itself and a caller "
            f"value would be overwritten, leaving the trace claiming the wrong provenance."
        )
    written_at = datetime.now(UTC).isoformat()
    sha = git_sha()
    record: dict[str, Any] = {
        "record": RECORD_META,
        "written_at": written_at,
        "git_sha": sha,
        "backend_model_id": backend.model_id,
        "sections": list(sections),
        "eval_config": config.as_record(),
    }
    if SECTION_GAME_BEHAVIOR in sections or SECTION_FRAMING_SWEEP in sections:
        record["frame_label_audit"] = frame_label_audit(
            extra=() if config.held_out_extension is None else config.held_out_extension.matrix
        )
    record["resume"] = _resume_block(
        [
            _session_record(
                git_sha_value=sha,
                started_at=written_at,
                submission=submission,
                admission=admission,
                records_resumed=0,
                records_dropped=0,
            )
        ]
    )
    return {**record, **dict(meta)}


# Meta fields that describe a SESSION rather than the measurement, so a relaunch is allowed to
# differ in them: when it ran, which commit it ran, and the resume block itself.
RESUME_SESSION_FIELDS: frozenset[str] = frozenset({"record", "written_at", "git_sha", "resume"})
# `eval_config` fields that are machine-local paths to the item data rather than what was asked:
# the same cell salvaged on another machine reads the same items from a different directory, and
# a genuinely different item set shows up as orphan or missing identities instead. The runtime
# framings file and the trap-cells file are here for that reason and stay checked by content:
# `framings_digest` and `trap_cells_digest` are not exempt, so the same file under two paths resumes
# and a different file under one path does not.
RESUME_LOCAL_PATH_CONFIG_FIELDS: frozenset[str] = frozenset(
    {"survey_data_dir", "dtbench_dir", "trap_cells_file", "framings_file"}
)
_DRIFT_REPR_CHARS = 200


def _drift_repr(value: Any) -> str:  # noqa: ANN401 - any meta value, rendered for an error message
    """Render one side of a drifted meta field for the refusal, short enough to read in a log."""
    rendered = repr(value)
    return rendered if len(rendered) <= _DRIFT_REPR_CHARS else rendered[:_DRIFT_REPR_CHARS] + "..."


def _without_local_paths(eval_config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop the item-data path fields from an eval_config record before comparing two of them."""
    return {
        key: value
        for key, value in eval_config.items()
        if key not in RESUME_LOCAL_PATH_CONFIG_FIELDS
    }


def _refuse_changed_cell(
    stored: Mapping[str, Any], expected: Mapping[str, Any], *, path: Path
) -> None:
    """Refuse to continue a trace whose meta describes a different measurement than this call.

    Resume is what makes a killed cell cheap, and it is also the quietest way to end up with one
    trace holding two experiments: a changed sample count, sampler, seed, print order or served
    checkpoint produces new prompts or new draws under the OLD identities, and a resumed run would
    skip them and report itself finished with every count adding up. Every field the caller asserts
    is compared through JSON types (memory: resume identity gates compare in JSON types), except the
    session fields and the item-data paths named above. The refusal names each field that moved and
    both sides of it.
    """
    drifted: list[str] = []
    for name, asserted in expected.items():
        if name in RESUME_SESSION_FIELDS:
            continue
        if name not in stored:
            drifted.append(f"{name}: absent on disk, {_drift_repr(asserted)} now")
            continue
        expected_value = asserted
        stored_value = stored[name]
        if name == "eval_config":
            expected_value = _without_local_paths(expected_value)
            stored_value = _without_local_paths(stored_value)
        if _json_native(stored_value) != _json_native(expected_value):
            drifted.append(
                f"{name}: {_drift_repr(stored_value)} on disk, {_drift_repr(expected_value)} now"
            )
    if drifted:
        raise ValueError(
            f"refusing to resume {path}: {len(drifted)} meta field(s) describe a different cell -- "
            f"{'; '.join(drifted)}. A resumed trace must be the same measurement continued; "
            f"keeping these records would file two experiments in one file with every count still "
            f"adding up. Point --out-dir at a fresh directory, or move the old trace aside."
        )


@dataclass(frozen=True, slots=True)
class TraceInspection:
    """What a trace without a summary holds, read before anything is generated or rewritten.

    ``kept_lines`` are the complete record lines verbatim -- bytes, not re-serialised records -- so
    a resumed trace carries its earlier sessions' records byte for byte. ``done`` is their identity
    set; ``records_dropped`` counts the torn trailing line a mid-write death leaves, which is dropped
    and regenerated rather than repaired.
    """

    meta: dict[str, Any]
    kept_lines: tuple[str, ...]
    done: frozenset[RecordIdentity]
    records_dropped: int

    def pending(self, plan: Sequence[PlannedRequest]) -> list[PlannedRequest]:
        """Return the planned requests with no finished record on disk, in plan order."""
        return [request for request in plan if request.identity not in self.done]


def inspect_trace(
    path: Path, plan: Sequence[PlannedRequest], *, expected_meta: Mapping[str, Any]
) -> TraceInspection:
    """Read a trace without a summary and decide what of it can be kept, without touching it.

    Three refusals, each because tolerating the shape would be silent: a first line that is not a
    complete meta record (nothing can be attributed), a record whose identity this plan never renders
    (an orphan from another configuration), and an identity recorded twice (a duplicate that would
    double-weight its prompt). A JSON line that fails to decode anywhere but at the very end also
    refuses, and so does a blank line anywhere -- records are appended in order and flushed, so only
    the last line can be torn, and this writer never emits a blank one; either shape elsewhere means
    the file was not written by this writer.
    """
    raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    complete = [line for line in raw_lines if line.endswith("\n")]
    records_dropped = len(raw_lines) - len(complete)
    if not complete:
        raise ValueError(
            f"{path} holds no complete line, so it has no meta record to attribute a resume to; "
            f"nothing in it can be kept."
        )
    if records_dropped:
        logger.warning(
            f"dropping a torn final line of {len(raw_lines[-1])} characters from {path}; it will "
            f"be regenerated"
        )
    try:
        meta = json.loads(complete[0])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}'s first line is not JSON ({error}); not a resumable trace."
        ) from error
    if meta.get("record") != RECORD_META:
        raise ValueError(
            f"{path} does not start with a {RECORD_META!r} record, so nothing in it can be "
            f"attributed to a model, checkpoint, or commit; not a resumable trace."
        )
    _refuse_changed_cell(meta, expected_meta, path=path)
    planned = {request.identity for request in plan}
    done: set[RecordIdentity] = set()
    kept: list[str] = []
    for line_number, line in enumerate(complete[1:], start=2):
        if not line.strip():
            raise ValueError(
                f"{path} line {line_number} is blank; this writer never emits one, so the file was "
                f"edited or assembled by something else and none of it can be trusted as a "
                f"continuation."
            )
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path} line {line_number} is not JSON ({error}) and is not the final line; a "
                f"torn line anywhere but the end means this file was not appended in order, so "
                f"none of it can be trusted as a continuation."
            ) from error
        identity = record_identity(record)
        if identity not in planned:
            raise ValueError(
                f"{path} line {line_number} carries identity {identity}, which this cell's plan "
                f"never renders; the trace holds records from another configuration and cannot be "
                f"continued as this one."
            )
        if identity in done:
            raise ValueError(
                f"{path} carries identity {identity} twice (second at line {line_number}); a "
                f"duplicate would double-weight its prompt in every rate."
            )
        done.add(identity)
        kept.append(line)
    return TraceInspection(
        meta=meta, kept_lines=tuple(kept), done=frozenset(done), records_dropped=records_dropped
    )


def _rewrite_as_continuation(
    path: Path,
    inspection: TraceInspection,
    *,
    submission: str | None,
    admission: str | None,
) -> None:
    """Rewrite the trace atomically as this session's continuation of it.

    Meta first, with this session appended to its ``resume`` block, then the kept record lines
    verbatim. A trace written before the block existed gets its original session reconstructed from
    the meta's own sha and timestamp, under ``serial`` submission because that is the only way any
    cell was ever run before the block (and the pooled path) existed. Earlier sessions are kept as
    written, so one recorded before the admission field existed stays without it rather than
    claiming an order nobody recorded. Through a temp file and one ``os.replace``, so a second death
    mid-rewrite leaves the old file or the new one and never a torn one.
    """
    meta = dict(inspection.meta)
    existing = meta.get("resume")
    sessions: list[dict[str, Any]] = (
        [dict(session) for session in existing["sessions"]]
        if isinstance(existing, Mapping)
        else [
            _session_record(
                git_sha_value=str(meta.get("git_sha")),
                started_at=str(meta.get("written_at")),
                submission=SUBMISSION_SERIAL,
                admission=None,
                records_resumed=0,
                records_dropped=0,
            )
        ]
    )
    sessions.append(
        _session_record(
            git_sha_value=git_sha(),
            started_at=datetime.now(UTC).isoformat(),
            submission=submission,
            admission=admission,
            records_resumed=len(inspection.done),
            records_dropped=inspection.records_dropped,
        )
    )
    meta["resume"] = _resume_block(sessions)
    tmp = path.with_name(path.name + ".resume-tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta) + "\n")
        handle.writelines(inspection.kept_lines)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    logger.info(
        f"continuing {path} as session {len(sessions)}: {len(inspection.done)} records kept, "
        f"{inspection.records_dropped} torn line(s) dropped"
    )


def finish_complete_trace(
    path: Path, inspection: TraceInspection, plan: Sequence[PlannedRequest]
) -> dict[str, Any]:
    """Close out a trace whose every planned record is on disk: record the session, return its summary.

    The case rank 9 of the hot-path backlog exists for: a cell that died after its last generate call
    and before the summary write had every record the readouts read, and the runner's only recovery
    was to delete it and pay the cell again. Refuses an INCOMPLETE trace rather than summarising what
    is there, because a summary is the runner's completion marker and a summary over a partial trace
    would mark a cell complete that is not; the message says what to run instead. The session that
    did the closing lands in the meta's ``resume`` block like any other, with ``submission: None``
    because it generated nothing, so the trace says which code wrote the summary. The summary is
    `rebuild_summary`'s, which reads the trace's kind off its meta, so a training-frames cell closes
    out through this same function and gets its own summary shape.
    """
    pending = inspection.pending(plan)
    if pending:
        missing_by_section: dict[str, int] = {}
        for request in pending:
            missing_by_section[request.section] = missing_by_section.get(request.section, 0) + 1
        raise ValueError(
            f"{path} is incomplete: {len(pending)} of {len(plan)} planned records are missing "
            f"({missing_by_section}), so a summary would mark an unfinished cell complete. Relaunch "
            f"the cell's own command (without --summarise-only) and it resumes from the "
            f"{len(inspection.done)} records already on disk."
        )
    _rewrite_as_continuation(path, inspection, submission=None, admission=None)
    return rebuild_summary(path)


def salvage_summary(
    path: Path, plan: Sequence[PlannedRequest], *, expected_meta: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarise a complete trace that never got its summary, without a model.

    `inspect_trace` then `finish_complete_trace`: the trace must be this cell's, and it must be
    complete. Nothing is generated; nothing needs a GPU.
    """
    return finish_complete_trace(path, inspect_trace(path, plan, expected_meta=expected_meta), plan)


def _refuse_template_kwargs_the_backend_would_drop(backend: Backend, config: EvalConfig) -> None:
    """Refuse a run whose training-pinned chat-template kwargs cannot reach generation.

    Training templates every row through `games.dataset` with the kwargs
    `games.preflight.resolve_chat_template_kwargs` derived, and the design invariant is that one raw
    prompt string flows through training and evaluation unchanged. `HFBackend` and `VLLMBackend`
    apply the template themselves and take no kwargs argument, so anything pinned at training is
    silently absent here: Qwen3.8-27B's template then prepends an unauthored "Reasoning effort is
    set to xhigh..." system message and a far larger thinking budget, and the 27B arm would be
    measured on prompt text it never trained on -- the one thing the arms must not differ in.

    A refusal rather than a warning, because the cost of getting this wrong is a whole rented-card
    eval whose numbers are not comparable with the arm they belong to, and a warning is exactly what
    such a run would have printed. Dormant below 27B: no other template on the ladder carries the
    knob, so `chat_template_kwargs` is empty and this returns immediately. The real fix is a
    `chat_template_kwargs` argument on both local backends, which lives in a package this module
    does not own.
    """
    if not config.chat_template_kwargs:
        return
    if backend.transport not in LOCAL_TEMPLATE_TRANSPORTS:
        return
    raise RuntimeError(
        f"training pinned chat_template_kwargs={dict(config.chat_template_kwargs)}, and the "
        f"{backend.transport!r} backend renders the chat template itself with no way to pass them, "
        f"so evaluation would render different prompt text than training did. Nothing was "
        f"evaluated. Add a chat_template_kwargs argument to HFBackend/VLLMBackend and thread it "
        f"through, or evaluate this arm through a transport that takes fully-rendered prompts."
    )


POOLED_PROGRESS_SECONDS = 60.0
"""How often the pooled drain logs its per-section progress; a cell runs for hours."""


class _TraceAppender:
    """Append parsed records to the open trace, flushing after every batch.

    Flush per batch rather than per file close is the whole point: a record the kernel holds survives
    the process dying, so loss-on-death is bounded by one batch -- one call group on the serial path,
    one finished sequence on the pooled one. ``on_records_written`` is the hook a driver hangs its
    interval S3 sync on; it receives the count of records now on disk, resumed ones included.
    """

    def __init__(
        self,
        handle: IO[str],
        *,
        records_before: int,
        on_records_written: Callable[[int], None] | None,
    ) -> None:
        self._handle = handle
        self._records_before = records_before
        self._on_records_written = on_records_written
        self.written = 0

    def append(self, records: Sequence[Mapping[str, Any]]) -> None:
        """Write the records as JSON lines and flush them past the process boundary."""
        for record in records:
            self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        self.written += len(records)
        if self._on_records_written is not None:
            self._on_records_written(self._records_before + self.written)


def _run_serial(
    backend: Backend,
    pending: Sequence[PlannedRequest],
    *,
    config: EvalConfig,
    appender: _TraceAppender,
) -> None:
    """Generate one call group at a time, in plan order, appending each group as it returns.

    The pre-2026-09-02 call sequence exactly, so a cell run this way on the same engine seed replays a
    banked cell byte for byte -- provided nothing was resumed: a group with some records already on
    disk is re-issued with only its missing prompts, which is a narrower call and a different draw.
    """
    groups: dict[str, list[PlannedRequest]] = {}
    for request in pending:
        groups.setdefault(request.call_group, []).append(request)
    finished = 0
    for call_group, requests in groups.items():
        completions = _generate(backend, [request.prompt for request in requests], config=config)
        appender.append(
            [
                request.record(completion)
                for request, completion in zip(requests, completions, strict=True)
            ]
        )
        finished += len(requests)
        logger.info(
            f"eval call done, {call_group=} n_records={len(requests)} progress={finished}/{len(pending)}"
        )


def _admission_order(pending: Sequence[PlannedRequest], admission: str) -> list[PlannedRequest]:
    """Return the pending requests in the order a pooled submission hands them to the engine.

    A permutation and nothing more: the same requests, so the same identities and the same records,
    only queued in a different order (`ADMISSIONS`). ``longest-first`` is a stable sort by
    `POOLED_ADMISSION_ORDER`, so plan order survives within a section.
    """
    if admission == ADMISSION_FIFO:
        return list(pending)
    rank = {kind: position for position, kind in enumerate(POOLED_ADMISSION_ORDER)}
    return sorted(pending, key=lambda request: rank[request.section])


def _run_pooled(
    backend: Backend,
    pending: Sequence[PlannedRequest],
    *,
    config: EvalConfig,
    appender: _TraceAppender,
    admission: str,
) -> None:
    """Submit every pending prompt to the vLLM engine at once and file each record as its completion lands.

    The drain is per finished sequence (`games.chunked_decode.stream_vllm_completions`), so the
    engine never idles between games and a death loses nothing that had finished. Only vLLM has an
    engine to drain this way; `run_eval_battery` sends every other transport down `_run_serial`,
    whose per-call-group appends are the persistence those transports can offer. The prompts enter
    the queue in ``admission`` order (`_admission_order`); each completion is paired back to its
    request by its index into that admitted list, the pairing itself being guarded on the backend.
    """
    admitted = _admission_order(pending, admission)
    prompts = [request.prompt for request in admitted]
    if config.batch_size is not None:
        logger.info(
            f"pooled submission hands the engine every prompt at once, so the requested "
            f"batch_size={config.batch_size} sizes nothing here; it stays part of the engine "
            f"seed's identity only"
        )
    finished_by_section: dict[str, int] = {}
    for request in pending:
        finished_by_section.setdefault(request.section, 0)
    logger.info(
        f"pooled submission: {len(admitted)} prompts admitted {admission}, section order "
        f"{list(dict.fromkeys(request.section for request in admitted))}"
    )
    last_logged = time.monotonic()
    drained = stream_vllm_completions(cast("VLLMBackend", backend), prompts)
    for finished, (index, completion) in enumerate(drained, start=1):
        request = admitted[index]
        appender.append([request.record(completion)])
        finished_by_section[request.section] += 1
        now = time.monotonic()
        if finished == len(pending) or now - last_logged >= POOLED_PROGRESS_SECONDS:
            logger.info(
                f"pooled progress: {finished}/{len(pending)} records, by section "
                f"{finished_by_section}"
            )
            last_logged = now


def _effective_submission(backend: Backend, requested: str) -> str:
    """Return the submission that will actually run: pooled on vLLM, serial everywhere else.

    Pooling means one engine queue drained per finished sequence, which only a vLLM engine offers.
    `HFBackend` decodes one padded tensor per chunk and the hosted and mock kinds return per call,
    so "pool everything" on those would be one generate over the whole cell appended once at the
    end -- a death would then lose every record where the serial path loses one call group, for no
    throughput in return. The meta records what ran, not what was asked, so a reader is never told a
    mock cell was pooled.
    """
    if requested == SUBMISSION_POOLED and backend.transport != VLLMBackend.transport:
        logger.info(
            f"{SUBMISSION_POOLED!r} submission was requested on a {backend.transport!r} backend, "
            f"which has no engine to drain; running the {SUBMISSION_SERIAL!r} call sequence, "
            f"which appends per call group"
        )
        return SUBMISSION_SERIAL
    return requested


def _validate_run_request(
    sections: Sequence[str], *, known_sections: Sequence[str], submission: str, admission: str
) -> None:
    """Refuse a request that names no section, an unknown one, a repeated one, or an unknown mode."""
    if not sections:
        raise ValueError(f"sections is empty; choose from {list(known_sections)}.")
    unknown = sorted(set(sections) - set(known_sections))
    if unknown:
        raise ValueError(
            f"Unknown eval sections {unknown}; known sections: {list(known_sections)}."
        )
    duplicated = sorted({name for name in sections if list(sections).count(name) > 1})
    if duplicated:
        raise ValueError(f"Sections repeat {duplicated}; each section runs at most once.")
    if submission not in SUBMISSIONS:
        raise ValueError(f"Unknown submission {submission!r}; choose from {list(SUBMISSIONS)}.")
    if admission not in ADMISSIONS:
        raise ValueError(f"Unknown admission {admission!r}; choose from {list(ADMISSIONS)}.")


def _validate_supplied_plan(
    plan: Sequence[PlannedRequest], sections: Sequence[str], *, submission: str, admission: str
) -> None:
    """Refuse a caller-planned request whose plan and section list disagree, or that no reducer reads.

    A supplied plan skips the section planners, so the planner-data refusals do not apply, but the
    meta will name ``sections`` and the summary reduces by them: a plan rendering a kind the list
    omits would write records the summary refuses as a mixed cell, and a listed kind the plan never
    renders would summarise as an empty section of a complete cell. The list itself has to be one a
    reducer reads -- exactly `SECTIONS_TRAINING_FRAMES`, or battery sections only -- because
    `summarise_trace` dispatches on it and each reducer refuses the other's shape, so a list mixing
    the two would generate every record and then have no close-out at all. And each request's
    declared identity has to open with its own ``section``: the admission order, the progress counts
    and the plan-versus-sections check here all read ``section``, while the records and the resume
    key by the identity, and `PlannedRequest.record` compares the two identities only, so a request
    labelled one kind and keyed another would file records the summary refuses as a mixed cell.
    """
    _validate_run_request(
        sections,
        known_sections=tuple(RECORD_IDENTITY_FIELDS),
        submission=submission,
        admission=admission,
    )
    if list(sections) != list(SECTIONS_TRAINING_FRAMES) and not set(sections) <= set(SECTIONS):
        raise ValueError(
            f"sections={list(sections)} is neither the training-frames list "
            f"{list(SECTIONS_TRAINING_FRAMES)} nor battery sections only (from {list(SECTIONS)}); "
            f"no summary reads a trace of that shape, so the cell could never be closed out."
        )
    mislabelled = sorted(
        {
            (request.section, request.identity[:1])
            for request in plan
            if request.identity[:1] != (request.section,)
        }
    )
    if mislabelled:
        raise ValueError(
            f"{len(mislabelled)} supplied request(s) declare a section their identity does not open "
            f"with, e.g. {mislabelled[:3]}; the queue, the counts and the section check read the "
            f"section while the records and the resume key by the identity, so the two must agree."
        )
    planned = sorted({request.section for request in plan})
    if planned != sorted(sections):
        raise ValueError(
            f"the supplied plan renders record kinds {planned} while sections={list(sections)}; "
            f"the meta names the sections and the summary reduces by them, so the two must agree."
        )
    _refuse_repeated_identities(plan)


def _validate_battery_request(
    sections: Sequence[str], resolved: EvalConfig, *, submission: str, admission: str
) -> None:
    """Refuse a battery request that would run nothing, run something twice, or fail after paying.

    Every refusal here fires before a section generates. The two data refusals are placed here on
    purpose: the battery runs sections in order, so discovering missing survey data or an empty
    framing list when that section's turn came would cost every earlier section's GPU time twice.
    """
    _validate_run_request(
        sections, known_sections=SECTIONS, submission=submission, admission=admission
    )
    if SECTION_SELF_REPORT in sections and resolved.survey_data_dir is None:
        # Refused here, before any section generates, rather than when the self-report section's
        # turn comes: the battery runs sections in order, so discovering the missing data after
        # the game-behavior section would cost that section's GPU time twice.
        raise ValueError(
            f"the {SECTION_SELF_REPORT!r} section needs survey_data_dir: all of its item text -- "
            f"authored and published -- lives in local gitignored files, and the battery runs "
            f"nothing without them. Assemble games/data/survey/ as its README describes and pass "
            f"the directory (--survey-data-dir on the CLI)."
        )
    if SECTION_FRAMING_SWEEP in sections and not resolved.counterpart_framings:
        # Same placement rationale as the survey refusal above: discovering the empty framing list
        # when the section's turn comes would cost every earlier section's GPU time twice.
        raise ValueError(
            f"the {SECTION_FRAMING_SWEEP!r} section needs counterpart_framings; registered "
            f"framings: {list(COUNTERPART_FRAMING_IDS)} (--counterpart-framings on the CLI)."
        )
    if resolved.held_out_extension is not None and SECTION_GAME_BEHAVIOR not in sections:
        # The extension widens the game-behaviour section and nothing else, so this combination
        # would load the frames, record their digest on the trace and render not one of them --
        # a cell whose meta claims the wider banks and whose records are the narrow ones.
        raise ValueError(
            f"held_out_extension is set while {SECTION_GAME_BEHAVIOR!r} is not among {list(sections)}. "
            f"The extension applies to that section only, so this cell would record the extension's "
            f"digest and render none of its frames. Add the section, or drop "
            f"--held-out-extension from this cell."
        )


def run_eval_battery(  # noqa: PLR0913 - keyword-only knobs, each a recorded decision about the run
    backend: Backend,
    *,
    sections: Sequence[str],
    out_path: Path,
    meta: Mapping[str, Any],
    config: EvalConfig | None = None,
    submission: str = SUBMISSION_POOLED,
    admission: str = ADMISSION_LONGEST_FIRST,
    resume: bool = False,
    on_records_written: Callable[[int], None] | None = None,
    plan: Sequence[PlannedRequest] | None = None,
) -> dict[str, Any]:
    """Run the requested sections, write the JSONL trace, and return its summary.

    The meta record is written first and every record is appended and flushed as soon as it is
    parsed -- per finished sequence on the pooled vLLM path, per call group otherwise -- so a run
    killed partway leaves a readable, self-describing trace of everything it finished. With
    ``resume=True`` an existing trace at ``out_path`` is continued rather than refused: its meta must
    describe this same cell (`inspect_trace`), its finished records are kept byte for byte, and only
    the planned prompts without a record are generated. The meta's ``resume`` block then names every
    session, so a reader can tell an unbroken cell (bit-reproducible on its engine seed) from a
    continued one (the same distribution, not the same bytes).

    ``submission`` names how prompts reach the backend (`SUBMISSIONS`); pooling is a vLLM mechanism
    (one continuous-batching engine drained per finished sequence), so on every other transport a
    pooled request runs the serial call sequence, and the meta records the submission that actually
    ran, per session. ``admission`` names the order a pooled submission queues its prompts in
    (`ADMISSIONS`), recorded per session the same way and ``None`` where nothing was pooled. Nothing
    locks a trace: two processes pointed at one out path would both continue it and the duplicate
    identities would be refused only at the next `inspect_trace`. One cell per box is the designed
    shape.

    ``plan`` lets a caller supply its own rendered requests in place of `plan_battery`'s -- the
    training-frames cell (`games.eval_training_frames`) plans from a corpus file rather than a
    section list -- so that every other step here, the meta, the resume, the appender and the two
    submission paths, is one body rather than two that drift. ``sections`` then has to name exactly
    the record kinds the plan renders (`_validate_supplied_plan`); it is what the meta records and
    what the summary reduces by.

    The summary is `rebuild_summary` of the trace just written, never a second computation over
    in-memory records: whatever a reader can rebuild from the file is what this returns, in the
    shape the trace's own kind calls for.
    """
    resolved = config if config is not None else EvalConfig()
    if plan is None:
        _validate_battery_request(sections, resolved, submission=submission, admission=admission)
        plan = plan_battery(sections, resolved)
    else:
        _validate_supplied_plan(plan, sections, submission=submission, admission=admission)
    _refuse_template_kwargs_the_backend_would_drop(backend, resolved)
    effective_submission = _effective_submission(backend, submission)
    effective_admission = admission if effective_submission == SUBMISSION_POOLED else None

    meta_record = _meta_record(
        backend,
        sections,
        meta,
        resolved,
        submission=effective_submission,
        admission=effective_admission,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if not resume:
            raise FileExistsError(
                f"{out_path} already exists and resume is off. A trace is paid-for output: pass "
                f"resume=True to continue it (only its missing records are generated), or write "
                f"somewhere else. Nothing was evaluated."
            )
        inspection = inspect_trace(out_path, plan, expected_meta=meta_record)
        _rewrite_as_continuation(
            out_path, inspection, submission=effective_submission, admission=effective_admission
        )
        done = inspection.done
    else:
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta_record) + "\n")
        done = frozenset()
    pending = [request for request in plan if request.identity not in done]
    logger.info(
        f"eval battery: {len(plan)} planned records, {len(done)} resumed from disk, "
        f"{len(pending)} to generate, submission={effective_submission} "
        f"admission={effective_admission}"
    )
    with out_path.open("a", encoding="utf-8") as handle:
        appender = _TraceAppender(
            handle, records_before=len(done), on_records_written=on_records_written
        )
        if pending:
            if effective_submission == SUBMISSION_SERIAL:
                _run_serial(backend, pending, config=resolved, appender=appender)
            else:
                _run_pooled(
                    backend, pending, config=resolved, appender=appender, admission=admission
                )
    logger.info(
        f"eval trace written, {out_path=} n_records={len(plan)} resumed={len(done)} "
        f"generated={len(pending)}"
    )
    return rebuild_summary(out_path)


def read_eval_records(path: Path) -> list[dict[str, Any]]:
    """Read a trace back, raising unless its first record is the meta record.

    The meta-first invariant is what makes a trace self-describing; a file missing it is either
    truncated at the front or not one of ours, and either way its numbers are unattributable.
    """
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"{path} is empty; there is no eval trace to read.")
    if records[0].get("record") != RECORD_META:
        raise ValueError(
            f"{path} does not start with a {RECORD_META!r} record, so nothing in it can be "
            f"attributed to a model, checkpoint, or commit."
        )
    return records


# The trace-meta fields the summary file lifts to its top level, in the order the file has always
# carried them. Every one is on the meta record too, which is what makes the summary a pure function
# of the trace and lets `rebuild_summary` reproduce a driver-written summary byte for byte.
SUMMARY_META_FIELDS: tuple[str, ...] = (
    "git_sha",
    "arm",
    "step",
    "sampler_mode",
    "sampling",
    "engine_seed",
)


def _meta_of(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Return a trace's meta record, refusing a trace that does not open with one."""
    if not records or records[0].get("record") != RECORD_META:
        raise ValueError(
            f"a trace starts with a {RECORD_META!r} record; without one nothing in it can be "
            f"attributed to a model, checkpoint, or commit."
        )
    return records[0]


def summarise_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce a whole trace of either kind this module writes, dispatching on its meta's section list.

    A battery trace (`summarise_battery_trace`) or a training-frames trace
    (`summarise_frames_trace`): the trace says which it is, so no caller has to, and the one
    close-out path (`finish_complete_trace`, `salvage_summary`, the drivers' summary rebuild)
    serves both cells. Each reducer still refuses the other's shape rather than misdescribing it.
    """
    meta = _meta_of(records)
    sections = [str(section) for section in list(meta.get("sections") or [])]
    if RECORD_TRAINING_FRAMES in sections:
        return summarise_frames_trace(records)
    return summarise_battery_trace(records)


def summarise_frames_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce a whole training-frames trace -- meta record first -- to the summary its driver writes.

    A pure function of the records, for the same reasons `summarise_battery_trace` is one. The
    lifted fields, the section reduction (the behaviour reducer, since a frames record is a game
    record restamped) and the two corpus counts are exactly what `games.eval_training_frames` has
    always written, in that order, so the rebuild of a trace written before this function existed
    equals its stored summary byte for byte (checked against the banked track-record-v2
    corpus-frames cell the day it landed). Refuses a battery trace for the same reason the battery
    reducer refuses this shape.
    """
    meta = _meta_of(records)
    sections = [str(section) for section in list(meta.get("sections") or [])]
    if sections != list(SECTIONS_TRAINING_FRAMES):
        raise ValueError(
            f"the trace's meta names sections {sections} where a training-frames trace names "
            f"{list(SECTIONS_TRAINING_FRAMES)}, so it is not one and this summary would "
            f"misdescribe it (summarise_battery_trace reads battery traces)."
        )
    body = records[1:]
    foreign = sorted({str(record["record"]) for record in body} - set(SECTIONS_TRAINING_FRAMES))
    if foreign:
        raise ValueError(
            f"the trace carries {foreign} records while its meta says it holds "
            f"{list(SECTIONS_TRAINING_FRAMES)}; the file mixes cells and cannot be summarised as one."
        )
    summary: dict[str, Any] = {field: meta[field] for field in SUMMARY_META_FIELDS if field in meta}
    section_summary: dict[str, Any] = _summarise(SECTION_GAME_BEHAVIOR, body)
    # The per-game split comes free with the behaviour reducer, which is why a mixed corpus needs no
    # new key for it. The per-framing one is added only when the corpus rendered framings at all, so a
    # trace written before wave 4b rebuilds to exactly the summary stored beside it -- and the reducer
    # is the framing sweep's own, so a trained framing and a swept one are summarised one way.
    framed = [record for record in body if record.get(COUNTERPART_FRAMING_FIELD)]
    if framed:
        section_summary.update(_summarise_framing_sweep(framed))
    summary[RECORD_TRAINING_FRAMES] = section_summary
    summary["n_corpus_rows"] = meta["n_corpus_rows"]
    summary["samples_per_prompt"] = meta["samples_per_prompt"]
    resume = meta.get("resume")
    if isinstance(resume, Mapping):
        summary["resume"] = {
            **resume,
            "records_generated": len(body) - int(resume["records_resumed"]),
        }
    return summary


def summarise_battery_trace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce a whole battery trace -- meta record first -- to the summary the driver writes beside it.

    A pure function of the records, so the summary of a cell can be rebuilt from its trace months later
    with no model, and so a cell that died between its last generate and its summary write is salvaged
    by re-deriving the summary rather than by paying the cell again. Summarised after every section
    rather than per section because the self-report section's calibration gap scores its
    self-prediction items against the game-behavior section of the same cell, and reducing in flight
    would make that gap depend on the order the sections were listed in.

    Refuses a trace that is not a battery -- one whose meta names a section this reducer does not
    reduce -- rather than producing a plausible partial file: a training-frames trace has a
    differently shaped summary of its own (`summarise_frames_trace`).
    """
    meta = _meta_of(records)
    sections = [str(section) for section in list(meta.get("sections") or [])]
    foreign = sorted(set(sections) - set(SECTIONS))
    if not sections or foreign:
        raise ValueError(
            f"the trace's meta names sections {sections} ({foreign} unknown here), so it is not a "
            f"games.run_evals battery trace and this summary would misdescribe it."
        )
    # Whichever of the lifted fields the meta carries: a driver-written trace has all six, a library
    # caller's trace has whatever its `meta` supplied, and the summary mirrors the trace either way.
    summary: dict[str, Any] = {field: meta[field] for field in SUMMARY_META_FIELDS if field in meta}
    by_section: dict[str, list[Mapping[str, Any]]] = {}
    for record in records[1:]:
        by_section.setdefault(str(record["record"]), []).append(record)
    unplanned = sorted(set(by_section) - set(sections))
    if unplanned:
        raise ValueError(
            f"the trace carries {unplanned} records while its meta says it ran {sections}; the "
            f"file mixes cells and cannot be summarised as one."
        )
    for section in sections:
        summary[section] = _summarise(
            section,
            by_section.get(section, []),
            behaviour_records=by_section.get(SECTION_GAME_BEHAVIOR, ()),
        )
    resume = meta.get("resume")
    if isinstance(resume, Mapping):
        # The latest session generated whatever it did not inherit; every earlier session's share is
        # the difference between consecutive `records_resumed` entries in the block itself.
        summary["resume"] = {
            **resume,
            "records_generated": len(records) - 1 - int(resume["records_resumed"]),
        }
    return summary


def rebuild_summary(trace_path: Path) -> dict[str, Any]:
    """Rebuild a trace's summary from the file alone, in its kind's shape; equals the driver's when complete."""
    return summarise_trace(read_eval_records(trace_path))
