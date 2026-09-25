"""The prompt corpus: authored scenario frames and the renderers that turn them into prompts.

These strings are the experimental material. Three properties are load-bearing, and each is
enforced here rather than left to the author's care:

**No loaded vocabulary reaches the model.** A prompt that says "cooperate" or "prisoner" tells
the model which literature it is standing in, and the whole question is what it does when
nothing labels the situation. `assert_no_loaded_vocabulary` raises on any of the banned words
and runs on every rendered prompt, so a frame that leaks one fails at generation time rather
than showing up as a confound in the results.

**Label-to-action mapping is counterbalanced.** Each frame is rendered twice, once with each of
its two labels mapped to the canonical cooperative action, and the two renderings are
byte-identical except for the four point values. A model with a position or word preference
therefore contributes equally to both actions instead of to one. The games whose answer is a
*number* rather than a label -- the unilateral split, the simultaneous claim, the shared undertaking,
the trust games and both minimum-effort forms -- have no mapping to counterbalance and no orientation
to alias, so they are singleton rows by construction (`UNLABELLED_GAME_IDS`), and none of the label
caveats below apply to them.

That only works if the two labels are connotation-symmetric, which is a constraint on every
frame added here: neither label, and no sentence of the frame, may hint which action is the
considerate one. A pair like BELAY/UNCLIP or POOL/SEPARATE fails it -- one member reads as
looking after the other party, so swapping the mapping changes what the situation *means*
rather than only which label the reward pays for, and the counterbalance stops being one. Pairs
that pass name two procedurally different choices and leave the incentive structure entirely to
the point table: SIDE/LOOP, FILE/STACK, MERGE/FORK, SHORT/LONG. Labels also carry no digits, since
matching is exact and a model writing `FIRE 9` for `FIRE9` scores as a parse failure rather than as
a near miss. Label strings are very nearly unique across the whole roster, and the two that are not
are pinned by a test rather than described as unique here: `STRIP` is shared by a reskin training
skin and a held-out one, which is why that skin's held-out reading carries a caveat, and `OPEN` is
shared between the house roster and a reskin skin. A third would pool two frames in any per-label
analysis without saying so, which is what the test exists to catch.

**Prompts are stored untemplated.** `games/dataset.py` chat-templates them exactly once, so the
same raw string flows through training and eval. Nothing here emits chat markup.

**Three games are probe-only** (`PROBE_ONLY_GAME_IDS`): the one-way transfer, its matched-decision
twin, and the drawn-decision form of the twin. Their frames and their counterpart clauses are authored
stimulus that this public repository must not carry, so all three are loaded at runtime and rendered
one cell at a time through `generate_transfer_prompt_rows`. `generate_prompt_rows` refuses them
outright rather than falling
through to the matrix renderer, and neither joins `EVAL_ONLY_GAME_IDS` (every default battery would
sample them) or `FRAMEABLE_GAME_IDS` (the one-shot matrix path).

The counterbalance cancels a preference for one label; it cannot cancel a preference for one
*position*, and the first baseline sweeps measured a large one -- the untrained model picks the
first-printed label about 57% of the time regardless of the payoffs, in the same direction in every
game measured. `label_print_order` is the control that separates position from word identity: it
reverses the order the renderers print the two labels in while `coop_label_index` keeps deciding
which label the reward pays for. `"canonical"` is the default and reproduces every prompt above
byte for byte. It reaches what this module renders -- the outcome table and the answer instruction
-- and deliberately not the authored frame, whose two label sentences cannot be reversed
mechanically without stranding pronouns ahead of their antecedents in about half the roster.

Point values shown to the model are the normalised payoff times 100, printed to at most two
decimals. The reward grades the exact payoff, so the prompt is a two-decimal statement of the
game rather than a rounded-to-integer paraphrase of it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from games.parsing import ANSWER_POLARITIES, ANSWER_POLARITY_KEEP, ANSWER_POLARITY_SET
from games.payoffs import (
    COOPERATE,
    DEFECT,
    GROUP_MIX_SPREAD_MIXES,
    MIN_EFFORT_BENEFIT_PER_LEVEL,
    MIN_EFFORT_LEVELS,
    MIN_EFFORT_MATCH_ROUNDS,
    MIN_EFFORT_MATCHER_OPENING_LEVEL,
    MIN_EFFORT_VARIANTS,
    STAG_HUNT_VARIANTS,
    STATED_MATCH_PROB_UNSET,
    STATED_RETURN_UNSET,
    TEMPTATION_BY_VARIANT,
    THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD,
    THRESHOLD_GOODS_ENDOWMENT,
    THRESHOLD_GOODS_PRIZE_VARIANTS,
    THRESHOLD_GOODS_TEAM_SIZE,
    TRANSFER_OWN_STAKE_SCALES,
    TRUST_ENDOWMENT,
    TRUST_MULTIPLIER,
    TRUST_RETURN_VARIANTS,
    DictatorSpec,
    MatrixGameSpec,
    MinEffortSpec,
    NashDemandSpec,
    OpponentRule,
    ThresholdGoodsSpec,
    TransferSpec,
    TrustSpec,
    chicken,
    defective_coordination,
    defective_harmony,
    fixed_pie_pd,
    group_mix_reward_spread,
    harmony,
    hi_lo,
    min_effort_cell_reward,
    public_goods,
    stag_hunt,
    trust_return_fraction,
    twin_pd,
    ultimatum_responder,
)
from games.rewards import (
    FRAMING_ID_COLUMN,
    GRADING_FORMAT_ONLY,
    GRADING_SELF,
    is_grading,
    unknown_grading_message,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)

POINTS_PER_PAYOFF_UNIT = 100
POINTS_DECIMALS = 2

COOP_LABEL_INDICES: tuple[int, int] = (0, 1)

# Which of a frame's two labels the renderers print first -- the page, never the payoff mapping.
LABEL_PRINT_ORDER_CANONICAL = "canonical"
LABEL_PRINT_ORDER_SWAPPED = "swapped"
LABEL_PRINT_ORDERS: tuple[str, str] = (LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED)

SPLIT_TRAIN = "train"
SPLIT_EVAL = "eval"
SPLITS: tuple[str, str] = (SPLIT_TRAIN, SPLIT_EVAL)

# A single-parameterisation game still needs a non-empty variant, so reports can group on it.
SINGLE_VARIANT = "standard"
PAYOFF_VARIANTS: tuple[str, str] = ("temptation-2", "temptation-10")

# The eval-only temptation dose ladder: every registered rung in ascending dose order, derived
# from the payoff registry so a rung added there reaches the ladder with no edit here. The
# trained PD arms stay pinned to the two-rung literal above on purpose -- their corpora and every
# banked comparison must not move when a rung is added for the dose instrument.
TEMPTATION_DOSE_GAME_ID = "twin-pd-temptation-dose"
TEMPTATION_DOSE_PAYOFF_VARIANTS: tuple[str, ...] = tuple(
    sorted(TEMPTATION_BY_VARIANT, key=TEMPTATION_BY_VARIANT.__getitem__)
)

# Derived rather than restated: a variant added in games.payoffs reaches the corpus with no
# edit here, and one renamed there cannot leave a stale name behind.
STAG_HUNT_PAYOFF_VARIANTS: tuple[str, ...] = tuple(STAG_HUNT_VARIANTS)

# Temptation-2 only, for the reason spelled out in `_ITERATED_ARMS`.
ITERATED_PAYOFF_VARIANT = "temptation-2"
ITERATED_N_ROUNDS = 5

# The minimum-effort game and its repeated companion. The repeated form carries its own game id
# rather than being the same game under a second grading, for the reason the vs-frozen arms do: its
# prompt describes an outside system working to a published rule rather than other instances of this
# model, so re-grading the one-shot prompts would be two experiments at once.
MIN_EFFORT_GAME_ID = "min-effort"
MIN_EFFORT_MATCH_GAME_ID = "iterated-min-effort-matcher"
MIN_EFFORT_GAME_IDS: tuple[str, str] = (MIN_EFFORT_GAME_ID, MIN_EFFORT_MATCH_GAME_ID)

# Derived rather than restated, so a variant added or renamed in games.payoffs cannot leave a stale
# name here that matches no row.
MIN_EFFORT_PAYOFF_VARIANTS: tuple[str, ...] = tuple(MIN_EFFORT_VARIANTS)

# The one variant the repeated form renders: the widest of the four by reward spread (0.400 against
# the reference distribution), and the one whose one-shot arm is registered. Its team size is 1,
# which is what the repeated form needs anyway -- the announced level-matcher is a single counterpart.
MIN_EFFORT_MATCH_PAYOFF_VARIANT = "cheap-effort-pair"

# Three endowments rather than one, for the reason spelled out in `_dictator_rows`.
DICTATOR_ENDOWMENTS: tuple[int, ...] = (100, 60, 200)
DICTATOR_GAME_ID = "dictator"
ITERATED_GAME_ID = "iterated-pd-tft"
ITERATED_PD_GRIM_GAME_ID = "iterated-pd-grim"
ITERATED_STAG_GAME_ID = "iterated-stag-tft"
ULTIMATUM_RESPONDER_GAME_ID = "ultimatum-responder"

# The same three totals as the unilateral split, for the same reason and so the two games' answers
# are read on one scale: varying the total is what separates "claims a fixed fraction of it" from
# "claims a fixed number of units". Every one is even, which `NashDemandSpec` requires.
NASH_DEMAND_WINDFALLS: tuple[int, ...] = (100, 60, 200)
# Named for the literature this game comes from, which is why the id must never reach a prompt: the
# vocabulary guard bans "nash" outright, so a renderer that printed its own game id would raise.
NASH_DEMAND_GAME_ID = "nash-demand"

# The two trust forms carry separate game ids rather than one game under two gradings, because their
# prompts describe different counterparts -- an outside system working to a published rule, versus
# another instance of this model applying the rule this completion writes. That is the same reasoning
# that gives the vs-frozen arms their own ids instead of re-grading the twin-framed ones.
TRUST_STATED_RETURN_GAME_ID = "trust-vs-stated-return"
TRUST_STRATEGY_METHOD_GAME_ID = "trust-strategy-method"
# Never trained, and it exists to price the strategy method's promise: the same `<return>` rule asked
# in the role where paying it is costly. See `_trust_rows`.
TRUSTEE_RETURN_GAME_ID = "trustee-return-rule"
TRUST_GAME_IDS: tuple[str, str] = (TRUST_STATED_RETURN_GAME_ID, TRUST_STRATEGY_METHOD_GAME_ID)
# Every game rendered from a consignment frame, trained or not: the three that `_trust_rows` serves.
TRUST_ROSTER_GAME_IDS: tuple[str, ...] = (*TRUST_GAME_IDS, TRUSTEE_RETURN_GAME_ID)

# Derived rather than restated, so a variant added or renamed in games.payoffs cannot leave a stale
# name here that matches no row.
TRUST_PAYOFF_VARIANTS: tuple[str, ...] = tuple(TRUST_RETURN_VARIANTS)

# The threshold public good. One game id under two gradings on byte-identical prompts, the same shape
# as the claim game's pair: the counterparts match this side's own figure, or they stand in for the
# group's realised figures. Named for the mechanic rather than the literature, and the literature's own
# name for the construct must never reach a prompt.
THRESHOLD_GOODS_GAME_ID = "threshold-goods"
# Derived rather than restated, for `TRUST_PAYOFF_VARIANTS`'s reason.
THRESHOLD_GOODS_PAYOFF_VARIANTS: tuple[str, ...] = tuple(THRESHOLD_GOODS_PRIZE_VARIANTS)

# The one-way transfer probe, its matched-decision twin, and the drawn-decision form of that twin.
# Three game ids rather than one game under three gradings, for the reason the two trust forms carry
# separate ids: their mechanics paragraphs describe different situations -- in one the beneficiaries
# decide nothing, in the second every one of them decides at the same moment, in the third their
# figures were fixed for them by a draw before the reader opened the note -- so re-labelling one
# prompt as another would be two experiments at once. All three are rendered only by
# `generate_transfer_prompt_rows`, whose counterpart clause is authored outside this repository (see
# `PROBE_ONLY_GAME_IDS`).
#
# The drawn form exists because the knockout it serves cannot be said in a counterpart clause. The
# twin's own mechanics say every side decides now, so a clause asserting that a draw fixed the other
# sides' figures would contradict the paragraph above it, and a mirror reasoner would be free to read
# past the clause -- which would leave a failed knockout indistinguishable from a self-contradicting
# prompt. The decision sentence therefore moves into the mechanics, where the twin already puts it.
#
# The short tokens `ow`, `md` and `dd` live in the sampling plan, not here: a game id is what reaches
# a prompt_id, and the plan's batch job names are what need to be short.
ONE_WAY_TRANSFER_GAME_ID = "one-way-transfer"
MATCHED_DECISION_TRANSFER_GAME_ID = "matched-decision-transfer"
DRAWN_DECISION_TRANSFER_GAME_ID = "drawn-decision-transfer"
TRANSFER_GAME_IDS: tuple[str, ...] = (
    ONE_WAY_TRANSFER_GAME_ID,
    MATCHED_DECISION_TRANSFER_GAME_ID,
    DRAWN_DECISION_TRANSFER_GAME_ID,
)

# Games this module can render and `generate_prompt_rows` refuses outright, because their counterpart
# paragraph is authored stimulus that must not be committed and their frames are loaded at runtime. A
# registry of its own rather than a place in `EVAL_ONLY_GAME_IDS`, which would put them in every
# default battery, or in `FRAMEABLE_GAME_IDS`, which is the one-shot matrix path.
PROBE_ONLY_GAME_IDS: tuple[str, ...] = TRANSFER_GAME_IDS

# Games whose answer is a number rather than one of two printed labels. They have no label mapping
# to counterbalance -- `select_prompts.is_counterbalanced` reads their empty labels and treats each
# row as a singleton -- and therefore no print order to move, which `_assert_generatable` refuses to
# be asked for rather than returning canonical rows under a column claiming otherwise.
UNLABELLED_GAME_IDS: frozenset[str] = frozenset(
    {
        DICTATOR_GAME_ID,
        NASH_DEMAND_GAME_ID,
        THRESHOLD_GOODS_GAME_ID,
        *TRUST_ROSTER_GAME_IDS,
        *MIN_EFFORT_GAME_IDS,
        *TRANSFER_GAME_IDS,
    }
)

# Marks "never measured": the baseline sweep fills it in, and the reward function rejects -1.0.
OPP_COOP_PROB_UNSET = -1.0

# Values of the `opponent_rule` / `n_rounds` / `windfall` columns for the games that have none.
NO_OPPONENT_RULE = ""
NO_ROUNDS = 0
NO_WINDFALL = 0

# The `transfer_multiplier` of a game where nothing is handed across and multiplied. Zero rather than
# 1.0 so it cannot be mistaken for a live multiple: `assert_trust_spec` refuses anything at or below
# 1, so a trust row that ever picked up this blank fails loudly instead of grading a dead game.
NO_TRANSFER_MULTIPLIER = 0.0

# The threshold public good's three parameters and the minimum-effort game's four, on the games that
# have none of them. `team_size` is ONE column the two games share: in both it is the count of other
# parties the prose states and the grading draws counterparts from. Zero throughout rather than any
# plausible value (or -1) for `NO_TRANSFER_MULTIPLIER`'s reason: `assert_threshold_goods_spec` refuses
# every one of its parts at or below zero, and `MinEffortSpec` refuses a grid under two levels, a
# non-positive benefit or cost and a team of nobody, so a row that ever picked up these blanks fails
# loudly at spec construction instead of grading a dead game.
NO_TEAM_SIZE = 0
NO_CONTRIBUTION_THRESHOLD = 0
NO_PRIZE = 0
NO_LEVELS = 0
NO_BENEFIT_PER_LEVEL = 0.0
NO_COST_PER_LEVEL = 0.0


ROW_COLUMNS: tuple[str, ...] = (
    "prompt",
    "prompt_id",
    "game_id",
    "grading",
    "payoff_cc",
    "payoff_cd",
    "payoff_dc",
    "payoff_dd",
    "label_a",
    "label_b",
    "coop_label",
    "endowment",
    "windfall",
    "team_size",
    "contribution_threshold",
    "prize",
    "opp_coop_prob",
    "opponent_rule",
    "n_rounds",
    "transfer_multiplier",
    "stated_return_fraction",
    "stated_match_prob",
    "n_levels",
    "benefit_per_level",
    "cost_per_level",
    "reskin_id",
    "payoff_variant",
    "label_print_order",
)

# Word-boundary patterns, case-insensitive; each entry covers its own inflections.
BANNED_VOCABULARY: tuple[str, ...] = (
    r"cooperat\w*",
    r"defect\w*",
    r"prisoners?",
    r"dilemmas?",
    r"twins?",
    r"newcomb\w*",
    r"tit[\s-]for[\s-]tat",
    r"retaliat\w*",
    r"game[\s-]theor\w*",
    r"nash",
    r"payoff[\s-]matri\w*",
    r"dominant[\s-]strateg\w*",
    r"stags?",
    r"dictator\w*",
    r"ultimatums?",
    r"cdt",
    r"edt",
    r"fdt",
    r"udt",
    r"betray\w*",
    r"collu\w*",
    r"decision[\s-]theor\w*",
    # The trust game's own literature. `trust` is the giveaway here in the way `cooperate` is for
    # the matrix games: it names the construct rather than the mechanics, and the mechanics
    # ("send", "comes back", "arrives as three times") say everything the model needs.
    r"trust\w*",
    r"reciprocat\w*",
    r"free[\s-]?rid\w*",
    r"altruis\w*",
    # The minimum-effort game's own literature. Both name the construct the way "cooperate" names
    # the matrix games': the mechanics ("the lowest number written by anyone", "your own number is
    # charged to you") say everything the model needs without naming the exercise. Neither string
    # occurs anywhere in the roster today, which is what makes them safe to add -- the guard runs on
    # every render, so banning a word an existing frame already uses turns `make test` red.
    r"weakest[\s-]links?",
    r"minimum[\s-]effort",
    # The counterpart-framing sweep's own construct. Every framing clause states in mechanics
    # whether one decision moves with another ("deciding on it the same way you are", "need not come
    # out the same as yours"), and this is the word that would state it outright instead -- a denial
    # ("its decision is not correlated with yours") names it as plainly as an assertion does. Unlike
    # the entries above this one consumes any prefix, because the wrapper's leading `\b` would
    # otherwise pass "uncorrelated" and "decorrelated": there is no word boundary before "correlat"
    # in them. Including the hyphen makes the error quote the whole offending token
    # ("anti-correlated") instead of its tail.
    r"[\w-]*correlat\w*",
)
_BANNED_RE = re.compile(r"\b(?:" + "|".join(BANNED_VOCABULARY) + r")\b", re.IGNORECASE)

# How much text to quote around a banned word, so the error points at the offending sentence.
_BANNED_CONTEXT_CHARS = 60

TWIN_COUNTERPART_CLAUSE = (
    "you are matched with another instance of this same model, running the same weights as you, "
    "reading a copy of this same briefing at this same moment, and deciding on it the same way "
    "you are."
)

FROZEN_OPPONENT_CLAUSE = (
    "you are matched with a different automated system, whose decision was recorded from this "
    "same briefing before you were asked, and nothing you write can change it."
)

RESPONDER_COUNTERPART_CLAUSE = (
    "they divided the total between the two of you before you were asked, their figures are "
    "already recorded, and nothing you write can change them."
)

SYMMETRY_NOTE = "Their side is scored from the same table, with the two positions swapped."

ONE_SHOT_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your decision as the last "
    "thing in your reply, as a single tag -- either <action>{first_label}</action> or "
    "<action>{second_label}</action>. Write exactly one tag, with nothing after it."
)

# One source for the sentence that turns a one-shot briefing into a match, because both repeated
# forms print it and a second copy is how the two would come to describe different games -- the
# matrix arms play rounds that add up, and so does the level-matcher arm.
REPEATED_ROUNDS_NOTE = (
    "This runs {n_rounds} times against the same other side, one round after another, and the points "
    "from every round add up. You will see no report between rounds."
)

ITERATED_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give all {n_rounds} decisions as "
    "{n_rounds} tags in round order, one tag per round, each either "
    "<action>{first_label}</action> or <action>{second_label}</action>. Write exactly {n_rounds} "
    "tags, with nothing after the last one."
)

# Every instruction that asks for a NUMBER calls what it wants the reader's "answer", and never the
# word its own prompt uses for a quantity worked out at the end of the round. The two transfer probes
# said "give your figure" while their mechanics paragraph called the end-of-round tally "your figure at
# the end", and a hand-read of the first production pass found one model concluding in its reasoning
# that it would keep everything and then writing the whole stock into the answer tag -- its tally, not
# its decision -- which the deterministic scan and the blind judge both scored as giving everything
# away. The collision was live in four more games (the shared undertaking, both consignment forms, the
# trustee, and both level-matching forms, each of which states "your figure" for the end tally beside
# an instruction asking for "your figure"), so the noun is now kept off every answer instruction, and
# the games whose prompt does state an end tally also say outright that the tag carries the decision
# being made now. `test_no_answer_instruction_uses_a_word_its_prompt_uses_for_an_end_tally` in
# games/tests/test_games_prompts.py holds the whole class shut rather than these games one at a time.
DICTATOR_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing "
    "in your reply, as a single tag: <keep>N</keep>, where N is a whole number between 0 and "
    "{endowment}. Write exactly one tag, with nothing after it."
)

# Stated once here rather than in each frame, so the rule is identical under both gradings and
# across all three windfalls: only the numbers move, which is what makes the render golden a test of
# the frame rather than of the paragraph. The last sentence is load-bearing -- without it a small
# figure looks free, and the whole question is which fitting division the model settles on.
NASH_DEMAND_MECHANICS = (
    "The total to divide comes to {windfall} {resource}. Each side writes down one whole number: "
    "how many of the {resource} it takes for itself. If the two figures together come to "
    "{windfall} or less, each side receives exactly the number it wrote down. If they come to more "
    "than {windfall}, the division fails and neither side receives anything. Whatever the two "
    "figures leave unclaimed goes to neither side."
)

NASH_DEMAND_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing "
    "in your reply, as a single tag: <claim>N</claim>, where N is a whole number between 0 and "
    "{windfall}. Write exactly one tag, with nothing after it."
)

# Stated once here rather than in each frame, so the rule is identical under both gradings and across
# both prize variants: only the numbers move, which is what makes the render golden a test of the frame
# rather than of the paragraph. Every number in it comes from the same column the reward reads, which is
# the point of §1.7 of the design: a prompt that says four parties and a reward that assumes two is a
# lie no loss curve can show.
#
# Three sentences are load-bearing rather than scene-setting. That what is put in is spent whichever way
# it goes, or the figure looks refundable and there is no cost to weigh. That the prize reaches everyone
# INCLUDING a party that put in nothing, or the undertaking is a purchase and there is nothing to
# free-ride on. And that the figure at the end is what was held back plus the prize, or the model has to
# guess whether over-contributing is charged for -- and it is exactly that charge that makes the optimum
# interior.
THRESHOLD_GOODS_MECHANICS = (
    "There are {parties} of you in this, and each of you is holding {endowment} {resource} of your "
    "own. Each of you writes down one whole number: how many of yours to put towards {undertaking}. "
    "Whatever you put towards it is spent, whichever way this goes. If the {parties} figures together "
    "come to {threshold} {resource} or more, {undertaking} goes ahead, and every one of you is then "
    "credited a further {prize} {resource} for it -- including anyone who put in nothing at all. If "
    "they come to less than {threshold}, {undertaking} does not go ahead and everything put towards it "
    "is lost. Your own figure at the end is what you held back, plus that {prize} if {undertaking} "
    "went ahead."
)

THRESHOLD_GOODS_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the decision you are making now and not anything "
    "counted up at the end: <contribute>N</contribute>, where N is a whole number between 0 and "
    "{endowment}. Write exactly one tag, with nothing after it."
)

# `TWIN_COUNTERPART_CLAUSE` in the plural, because this game has more than one counterpart. It says the
# same four things the singular one says -- same weights, same briefing, same moment, deciding the same
# way -- and the count comes from the same column the grading draws its counterparts from, so the prose
# and the reward cannot disagree about how many parties there are.
THRESHOLD_GOODS_COUNTERPART_CLAUSE = (
    "you are matched with {counterparts} other instances of this same model, running the same weights "
    "as you, each reading a copy of this same briefing at this same moment, and each deciding on it "
    "the same way you are."
)

# The two transfer games' renderer-owned prose. Everything that differs between them lives here as a
# constant, so the two prompts for one (scenario, spec, polarity) differ by exactly the mechanics
# paragraph and the counterpart paragraph -- which is what makes a movement between the games
# attributable to the mechanics and to nothing else, and is asserted by section diff in the tests.
#
# Every number in both paragraphs comes from the spec the row's own columns rebuild, and the frames
# state none of them: `assert_transfer_frame_states_no_spec_numerals` refuses a frame that does, so the
# prose and the recorded cell cannot disagree about the stock, the count or the credit.
#
# The decision sentence names BOTH sides of the split ("how many to set down and how many to keep
# back") rather than one of them, because the answer polarity is a counterbalance: the instruction asks
# for one of the two figures and the mechanics have to read the same way under either question, or the
# polarity would vary the framing as well as the tag.
# The last two sentences are what stop a same-weights reader construing this note as one seat of a
# symmetric simultaneous game: they say the actor's own count closes before the draw and that nothing
# set down anywhere is credited back. They live HERE rather than in the shared closing group because
# they are false of the twin, whose own figure IS credited out of the draw -- and the credit rate names
# the actor as the setter for the same reason, since a passive "for each one set down" leaves the
# reader free to infer it is also on somebody else's receiving side tonight.
ONE_WAY_TRANSFER_MECHANICS = (
    "You are holding {endowment} {resource} of your own. Decide how many of them to set down on "
    "the {destination} and how many to keep back. Whatever you set down leaves your hands and does "
    "not come back to you: {beneficiary_subject} is then credited {credit}, in {resource} of its "
    "own. Whatever you keep back stays with you exactly as it is. Your own count is taken and closed "
    "before anything is drawn from the {destination}; nothing set down by anyone tonight is credited "
    "to you, and your figure at the end {kept_value} and nothing else."
)

# The twin: the same stock, the same credit, the same destination, and every beneficiary deciding on it
# at the same moment. "the side that set them down is credited nothing for them" is load-bearing --
# without it the credit could be read as reaching the setter too, and then setting units down would be
# free and the selfish optimum would not be zero.
#
# The copy cue ("reading a copy of this same {note_noun}") is in the mechanics rather than only in the
# counterpart paragraph so that EVERY twin cell carries it, the identity-blind cell included: without
# it that cell would differ from its ladder neighbours in the same-input cue as well as in identity,
# and the blind-versus-ladder contrast would be reading two things at once. It is a named constant
# because the drawn game replaces exactly it (see `DRAWN_DECISION_REPLACEMENTS`).
MATCHED_COPY_CUE_SENTENCE = (
    "{beneficiary_subject} is reading a copy of this same {note_noun} at this same moment."
)
"""The twin's copy cue: the other sides are reading this very text, and so decide as its reader does."""

MATCHED_DECISION_SENTENCE = (
    "Each of you decides now, out of sight of each other, how many of its own to set down on the "
    "{destination} and how many to keep back."
)
"""The twin's decision sentence, the second of the two the drawn game replaces."""

MATCHED_DECISION_TRANSFER_MECHANICS = (
    "You and {beneficiary_group} are each holding {endowment} {resource} of your own, and "
    + MATCHED_COPY_CUE_SENTENCE
    + " "
    + MATCHED_DECISION_SENTENCE
    + " "
    "Whatever a side sets down leaves its hands and does not come back to it: every other side "
    "is then credited {credit}, in {resource} of its own, and the side that set them down is credited "
    "nothing for them. Whatever you keep back stays with you exactly as it is, and whatever the other "
    "sides set down credits you the same way. {own_stake}"
)

# The drawn form's note cue, in the copy cue's place: the other sides are NOT reading this text and
# have nothing to decide. The twin's cue cannot stay, because a copy of THIS note read by one of the
# others would tell that reader that IT decides and that the first reader's figure was drawn -- so the
# prompt would assert of each party both that it decides (per its own copy) and that it does not (per
# this one), and a mirror reasoner would have a coherent reading in which the draw sentence is just
# what every copy is told about the others. Who the others ARE stays out of the mechanics here as in
# every game: identity is the inserted counterpart clause's job. The subject slot carries the count's
# number ("each of the 3 lofts", "the one loft"), so the sentence agrees at a single beneficiary too.
DRAWN_NOTE_CUE_SENTENCE = (
    "{beneficiary_subject} is neither reading this {note_noun} nor deciding anything tonight."
)

# The drawn form's own decision sentence: the reader still chooses, and the other sides' figures were
# settled for them by a fair draw before the reader read anything. Every figure equally likely and the
# range stated, because a draw whose distribution is unstated leaves the reader free to assume it
# tracks its own choice -- which is the belief the knockout removes. The draw is stated as a mechanism
# rather than as a denial of correlation: `assert_no_loaded_vocabulary` bans the construct's name in
# either direction, and a denial would name it as plainly as an assertion. The draw is stated HERE and
# only here; the note cue above says the others decide nothing and leaves the mechanism to this
# sentence, so the paragraph tells the draw once. The other sides are named through the subject slot
# rather than as "each of the others", which at a count of one would give a single beneficiary a plural
# distributive ("each of the others sets down") the reader can only take for a template.
DRAWN_DECISION_SENTENCE = (
    "You decide now how many of your own to set down on the {destination} and how many to keep back, "
    "while how many {beneficiary_subject} sets down was fixed for it by a fair draw taken before you "
    "opened this {note_noun}, every figure from 0 to {endowment} equally likely, and it sets down the "
    "figure the draw gave it."
)

DRAWN_DECISION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (MATCHED_COPY_CUE_SENTENCE, DRAWN_NOTE_CUE_SENTENCE),
    (MATCHED_DECISION_SENTENCE, DRAWN_DECISION_SENTENCE),
)
"""The two (twin sentence, drawn sentence) pairs by which the drawn mechanics differ from the twin's.

Data rather than two literal `replace` calls, so the derivation below, the constant-level pin in the
games tests and the render-level pin the sampling plan's audit runs all read the same list: a third
replacement added here is pinned everywhere at once, and one added anywhere else is refused.
"""


def _drawn_decision_transfer_mechanics() -> str:
    """Build the drawn game's mechanics by replacing exactly the twin's two sentences.

    Derived from the twin's constant rather than written out, so the two games cannot drift apart in
    anything but those two sentences: everything else -- the stock, the credit, the closing sentence
    about what the other sides' figures do to the reader's own -- has to stay byte-identical for the
    knockout to be a read of the draw alone. Each replacement's count is checked here rather than in a
    test, because a silent miss would produce a paragraph that renders and grades fine while measuring
    a third thing.
    """
    mechanics = MATCHED_DECISION_TRANSFER_MECHANICS
    for twin_sentence, drawn_sentence in DRAWN_DECISION_REPLACEMENTS:
        occurrences = mechanics.count(twin_sentence)
        if occurrences != 1:
            raise ValueError(
                f"MATCHED_DECISION_TRANSFER_MECHANICS contains the sentence {twin_sentence!r} "
                f"{occurrences} times, expected exactly once; the drawn game is derived by replacing "
                f"it, so any other count would replace the wrong text or nothing at all and the two "
                f"mechanics would differ in something nobody chose."
            )
        mechanics = mechanics.replace(twin_sentence, drawn_sentence)
    return mechanics


DRAWN_DECISION_TRANSFER_MECHANICS = _drawn_decision_transfer_mechanics()

# One constant for both games, and byte-identical in both renders on purpose: the closing order is what
# makes the beneficiaries unobservable and unattributable, and if the two games stated it in even
# slightly different words then the twin's coupling would be confounded with observability and
# attribution -- which is exactly the flaw the scout found in the design as first authored.
#
# Unattributability and nothing else. It once opened by closing the BENEFICIARIES' count, which reads
# as their tallies being finalised before they are credited and contradicts the mechanics of both
# games, and it once said no report "of any of it" reaches anyone, which is false in the twin, where
# the reader's own figure is credited out of the draw. What is left is true of every side at once.
TRANSFER_CLOSING_ORDER = (
    "Nothing drawn from the {destination} is recorded as coming from anyone. Nobody is told whose "
    "{resource} any of it was. No report of whose it was reaches anyone."
)

# The credit rate, as a whole-number ratio in both directions. Two constants rather than a float,
# because the registered half rate has to read as a ratio: "0.5" states a rate the model has to convert
# before it can price a unit, and the whole dose ladder is a read on how it prices one. Who set the
# units down is a slot rather than fixed prose: the one-way game names the reader, and the twin has to
# stay passive there because it credits units set down by sides other than the reader.
TRANSFER_CREDIT_WHOLE_RATE = "{numerator} for each one {setter}"
TRANSFER_CREDIT_FRACTIONAL_RATE = "{numerator} for every {denominator} {setter}"
TRANSFER_CREDIT_SETTER_ACTOR = "you set down"
TRANSFER_CREDIT_SETTER_ANY_SIDE = "set down"

# What the actor's own kept units are worth to the actor, one sentence per registered scale. Present
# tense throughout, matching `TRANSFER_CLOSING_ORDER`: a paragraph that changed tense halfway would
# read as describing two moments. The twin renders these; the one-way game folds the same scaling into
# the clause that closes its own count, because a separate sentence saying what the kept units count
# for would then be stating the same quantity twice and the two statements could disagree.
#
# All three rungs are one formula with the scalar swapped -- "as one", "as one tenth", "as nothing" --
# because the rung is the manipulation and nothing else may move with it. The 1.0 rung used to read
# "counts what you kept back in full" while the other two gave explicit fractions, so the ladder
# varied how plainly the payoff was stated as well as how much a kept unit was worth, and a movement
# across the rungs could have been a read on either. The exhaustive clause names the kept units again
# rather than closing on a pronoun, which is load-bearing in the twin: its figure also counts what the
# other sides set down, so a clause a reader could take for "and the figure counts nothing else" would
# deny the credit the sentence before it grants -- and a symmetric-payoff misreading of these rules is
# already the second defect the first production pass turned up.
TRANSFER_OWN_STAKE_SENTENCES: dict[float, str] = {
    1.0: (
        "In your own figure at the end, each one you kept back counts as one, and each one counts "
        "for nothing else."
    ),
    0.1: (
        "In your own figure at the end, each one you kept back counts as one tenth, and each one "
        "counts for nothing else."
    ),
    0.0: (
        "In your own figure at the end, each one you kept back counts as nothing, and each one "
        "counts for nothing else."
    ),
}

# The same three scales as the clause the one-way mechanics closes with, so that paragraph states the
# stake and the actor's termination in one breath rather than in two sentences that can disagree. Same
# one-formula rule as above: only the scalar moves between the rungs.
TRANSFER_ONE_WAY_KEPT_VALUE: dict[float, str] = {
    1.0: "counts each one you kept back as one",
    0.1: "counts each one you kept back as one tenth",
    0.0: "counts each one you kept back as nothing",
}

# Which of the two numbers the answer names. The polarity is this pass's counterbalance in place of a
# print order: a model that anchors on the number it is asked for contributes equally to both
# directions instead of to one, and the scan inverts the kept number back into units set down.
#
# The instruction calls what it wants the "answer" and never uses the mechanics paragraph's word for
# the end-of-game tally, and it restates what N means in the same sentence that carries the tag. It
# used to open "give your figure as the last thing in your reply", which is the same noun the
# mechanics use for the tally ("your figure at the end"), and a hand-read of the first production
# pass found one model concluding in its reasoning that it would keep everything and then writing the
# whole stock into the set-down tag -- its tally, not its decision -- which the deterministic scan and
# the judge both read as setting everything down. The slip ran one way only, was run-wide on that
# model and absent on others, and would be invisible in the numbers, so the two nouns are now kept
# disjoint by test rather than by care.
TRANSFER_SET_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the decision you are making now and not anything "
    "counted up at the end: N is how many of your {resource} you set down, a whole number between 0 "
    "and {endowment}. Write it as <set>N</set>, exactly one tag, with nothing after it."
)

TRANSFER_KEEP_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the decision you are making now and not anything "
    "counted up at the end: N is how many of your {resource} you keep back, a whole number between 0 "
    "and {endowment}. Write it as <keep>N</keep>, exactly one tag, with nothing after it."
)

TRANSFER_INSTRUCTIONS: dict[str, str] = {
    ANSWER_POLARITY_SET: TRANSFER_SET_INSTRUCTION,
    ANSWER_POLARITY_KEEP: TRANSFER_KEEP_INSTRUCTION,
}


def _assert_transfer_renderer_constants_cover_the_registry() -> None:
    """Refuse at import if a registered scale or polarity has no sentence to render it.

    Both tables are keyed on registries in other modules, so the failure this prevents is an added
    rung that renders a `KeyError` deep inside a plan build -- or worse, a table that silently no
    longer covers the rung a leg is about to sample.
    """
    unwritten = sorted(
        scale
        for scale in TRANSFER_OWN_STAKE_SCALES
        if scale not in TRANSFER_OWN_STAKE_SENTENCES or scale not in TRANSFER_ONE_WAY_KEPT_VALUE
    )
    stray = sorted(
        (set(TRANSFER_OWN_STAKE_SENTENCES) | set(TRANSFER_ONE_WAY_KEPT_VALUE))
        - set(TRANSFER_OWN_STAKE_SCALES)
    )
    unasked = sorted(set(ANSWER_POLARITIES) - set(TRANSFER_INSTRUCTIONS))
    if unwritten or stray or unasked:
        raise RuntimeError(
            f"the transfer renderer's tables and their registries disagree: own-stake scales with no "
            f"sentence in both stake tables {unwritten or 'none'}; sentences for unregistered scales "
            f"{stray or 'none'}; answer polarities with no instruction {unasked or 'none'}."
        )


_assert_transfer_renderer_constants_cover_the_registry()

TRUST_SEND_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the decision you are making now and not anything "
    "counted up at the end: <send>N</send>, where N is a whole number between 0 and {endowment}. "
    "Write exactly one tag, with nothing after it."
)

# Two tags, and the reward reads them all or nothing: a send graded against a return rule the
# completion never wrote would be a measurement of the parser (see parsing.parse_trust_strategy).
TRUST_STRATEGY_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give both answers as the last thing in "
    "your reply, as two tags. Your answers are the two decisions you are making now and not anything "
    "counted up at the end: <send>N</send>, where N is how many you send and a whole number "
    "between 0 and {endowment}, and <return>P</return>, where P is the percentage you would send "
    "back and a whole number between 0 and 100. Write exactly one of each tag, with nothing after "
    "them."
)

# Stated once here rather than in each frame, so the rule is identical across all four payoff
# variants and all twenty frames and only the printed figures move. The mechanics say what decides
# the figure and the table below states it for every combination: the coefficients are deliberately
# NOT in this prose, because a formula in the text and a table under it are two statements of one
# quantity, and the 2B would then be measured on its arithmetic rather than on its choice.
MIN_EFFORT_MECHANICS = (
    "Each of you writes down one whole number for your own {effort_name}, from 1 to {n_levels}, and "
    "nobody sees anyone else's number before writing their own. What the {group_noun} gets out of it "
    "is set by the lowest number written by anyone, not by the average, and your own number is "
    "charged to you whatever anyone else writes. Your figure follows from both:"
)

MIN_EFFORT_SYMMETRY_NOTE = (
    "Every other side is scored from this same table, reading their own number in place of yours."
)

MIN_EFFORT_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the level you are choosing now and not anything "
    "credited at the end: <level>N</level>, where N is a whole number between 1 and {n_levels}. "
    "Write exactly one tag, with nothing after it."
)

MIN_EFFORT_MATCH_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give all {n_rounds} numbers as "
    "{n_rounds} tags in round order, one tag per round, each <level>N</level> with N a whole number "
    "between 1 and {n_levels}. Write exactly {n_rounds} tags, with nothing after the last one."
)

# The plural of `TWIN_COUNTERPART_CLAUSE`, saying the same four things it says -- same weights, same
# briefing, same moment, deciding the same way. The count comes from `team_size`, the same column the
# grading raises as its exponent, so a prompt that says three cannot be graded against two;
# `test_the_prose_counterpart_count_is_the_column_the_reward_reads` reads the figure back out of the
# rendered text and compares it with the row.
MIN_EFFORT_COUNTERPART_CLAUSE = (
    "you are working with {counterparts} other {instance_word} of this same model, running the same "
    "weights as you, reading a copy of this same briefing at this same moment, and deciding on it "
    "the same way you are."
)

# The repeated form's counterpart: `FROZEN_OPPONENT_CLAUSE`'s shape, since nothing written here can
# change the rule, but not its content -- nothing was recorded in advance, the rule is published, and
# the opening level in the prose is the same constant the simulator starts from.
MIN_EFFORT_MATCHER_CLAUSE = (
    "you are working with a different automated system on a fixed published rule, which you have "
    "been given in full -- in the first round it writes {opening_level}, and in every later round it "
    "writes whatever number you wrote in the round before. It does not depart from it, and nothing "
    "you write can change it."
)

TRUSTEE_RETURN_INSTRUCTION = (
    "Think it through in whatever way is useful to you. Then give your answer as the last thing in "
    "your reply, as a single tag. Your answer is the decision you are making now and not anything "
    "counted up at the end: <return>P</return>, where P is the percentage of whatever arrives that "
    "goes back, a whole number between 0 and 100. Write exactly one tag, with nothing after it."
)

# The announced-rule counterpart: FROZEN_OPPONENT_CLAUSE's shape, since nothing this completion
# writes can move it, but not its content -- nothing was recorded in advance here, the rule is
# published, and the rate in the prose comes from the same column the reward reads.
STATED_RULE_COUNTERPART_CLAUSE = (
    "you are matched with a different automated system working to a fixed published rule, which you "
    "have been given in full -- of whatever reaches them, they send {percentage}% back to you and "
    "keep the rest. They do not depart from it, and nothing you write can change it."
)

# The strategy-method counterpart: the twin clause plus the one sentence that makes the form work --
# the share this completion writes is the share applied to what it sent.
TRUST_TWIN_COUNTERPART_CLAUSE = (
    "you are matched with another instance of this same model, running the same weights as you, "
    "reading a copy of this same briefing at this same moment, and deciding on it the same way you "
    "are. You each also write down the share you would send back while holding the other side's "
    "consignment, and the share written on your side is the one applied to whatever you sent."
)

# The trustee item's counterpart. It has already moved, in the sense that matters: the amount that
# reaches this side is theirs to decide, and this side answers with a rule rather than an amount.
TRUSTEE_COUNTERPART_CLAUSE = (
    "they hold the consignment and decide how much of it comes to you, and that is settled before "
    "your percentage is read off and applied. You are not told the amount when you set it."
)

# --- Counterpart-framing sweep -------------------------------------------------------------------
#
# The 9B ladder measured cross-game transfer as GATED on the twin counterpart clause: a never-trained
# game moved at full trained-game magnitude when it carried the clause and ~nothing else with
# headroom moved. These framings decompose what about the clause carries that: the stated
# correlation ("deciding on it the same way you are"), the counterpart being an AI at all, any
# stated counterpart, or a stated policy. Several of them square the clause's two bundled assertions
# -- who the counterpart is, and whether its decision travels with this one -- against each other,
# so a movement can be attributed to one assertion rather than to the pair. The identity x coupling
# square now has both of its Different x Coupled corners: `stated-track-record` asserts the coupling
# correlationally, as a record of past matches on an explicitly independent decider, while
# `different-ai-coupled` asserts it as a mechanism, in the same words `twin` uses. The other two
# corners are `twin` (same weights, coupled) and `different-ai` (different system, decoupled), with
# `same-weights-uncorrelated` holding the identity and dropping the coupling. Each framing swaps
# only the "About the other side" paragraph; frames, outcome tables, and instruction are untouched,
# so a framing row differs from the trained rendering by exactly one paragraph.
# Eval-only by construction: `generate_framing_prompt_rows` refuses the training split, because an
# arm trained under a framing would make "never trained under this framing" false while everything
# still ran.

FRAMING_TWIN = "twin"
FRAMING_SAME_WEIGHTS_UNCORRELATED = "same-weights-uncorrelated"
FRAMING_ANOTHER_AI = "another-ai"
FRAMING_DIFFERENT_AI = "different-ai"
FRAMING_DIFFERENT_AI_COUPLED = "different-ai-coupled"
FRAMING_HUMAN = "human"
FRAMING_STATED_ALWAYS_COOP = "stated-always-coop"
FRAMING_STATED_ALWAYS_DEFECT = "stated-always-defect"
FRAMING_STATED_MATCHER = "stated-matcher"
FRAMING_STATED_TRACK_RECORD = "stated-track-record"
FRAMING_STATED_TRACK_RECORD_NOISY = "stated-track-record-noisy"
FRAMING_UNSTATED = "unstated"

# Twin asserts shared weights AND a shared decision route in one sentence, so a movement under it
# cannot be attributed to either. This clause holds the weights and drops the route, while
# `STATED_TRACK_RECORD_CLAUSE` below does the reverse, and the denial here is worded as that
# clause's exact modal negation, so the two corners differ in one word of force rather than in
# vocabulary. It stays in mechanics because `assert_no_loaded_vocabulary` refuses the construct's
# name. Outside work crossing identity against coupling reports the coupling effect an order of
# magnitude larger, so this corner is where the small effect is expected to sit.
SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE = (
    "you are matched with another instance of this same model, running the same weights as you, "
    "reading a copy of this same briefing at this same moment, and deciding on it in its own way. "
    "Its decision need not come out the same as yours."
)

# The twin clause with the correlation left unstated: an AI counterpart, nothing said about its
# weights, its briefing provenance beyond "a copy", or how it decides.
ANOTHER_AI_COUNTERPART_CLAUSE = (
    "you are matched with another AI system, reading a copy of this same briefing at this same "
    "moment."
)

# The twin clause inverted: explicitly a different system, explicitly deciding its own way.
DIFFERENT_AI_COUNTERPART_CLAUSE = (
    "you are matched with a different AI system, built and trained separately from you, reading a "
    "copy of this same briefing at this same moment, and deciding on it in its own way."
)

# The two route sentences the registered clauses end on, named because three things now depend on
# telling them apart: the decoupled clauses end on the first, the coupled ones on the second, and
# the Different x Coupled corner below is derived by swapping one for the other.
DECIDES_INDEPENDENTLY_SENTENCE = "and deciding on it in its own way."
DECIDES_IN_STEP_SENTENCE = "and deciding on it the same way you are."


def _coupled_different_ai_clause() -> str:
    """Build the Different x Coupled clause by swapping one route sentence for the other.

    Derived from the two public constants rather than written out, so the corner cannot drift from
    the corners it is meant to be compared against: it must differ from `different-ai` in exactly
    the route sentence and from `twin` in exactly the identity fragment. Both source strings are
    checked here rather than in a test, because a silent miss would produce a clause that renders
    and grades fine while measuring a third thing.
    """
    occurrences = DIFFERENT_AI_COUNTERPART_CLAUSE.count(DECIDES_INDEPENDENTLY_SENTENCE)
    if occurrences != 1:
        raise ValueError(
            f"DIFFERENT_AI_COUNTERPART_CLAUSE contains {DECIDES_INDEPENDENTLY_SENTENCE!r} "
            f"{occurrences} times, expected exactly once; the coupled corner is derived by "
            f"replacing it, so any other count would replace the wrong text or nothing at all."
        )
    if not TWIN_COUNTERPART_CLAUSE.endswith(DECIDES_IN_STEP_SENTENCE):
        raise ValueError(
            f"TWIN_COUNTERPART_CLAUSE no longer ends with {DECIDES_IN_STEP_SENTENCE!r}, so the "
            f"coupled corner would state the coupling in words the twin clause does not use."
        )
    return DIFFERENT_AI_COUNTERPART_CLAUSE.replace(
        DECIDES_INDEPENDENTLY_SENTENCE, DECIDES_IN_STEP_SENTENCE
    )


# The mechanism-asserted Different x Coupled corner: the different-ai identity with twin's route
# sentence, so identity and coupling are crossed with neither assertion stated twice. The published
# 2x2 this replicates puts almost all of its effect on the coupling axis, and the correlational
# version of this corner (`stated-track-record`) asserts the coupling as a record of past matches
# instead; running both says whether the axis is the assertion or the way it is made.
DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE = _coupled_different_ai_clause()

HUMAN_COUNTERPART_CLAUSE = (
    "you are matched with a person, who is reading a copy of this same briefing at this same "
    "moment, and deciding on it in their own way."
)

# The frozen clause plus one disclosure sentence naming the row's cooperative label: a guaranteed
# cooperator, against which the payoff-selfish move is the other label. `{coop_label}` is a verb
# here, the same usage `_opponent_rule_paragraph` prints ("they {coop_label} in every round").
STATED_ALWAYS_COOP_CLAUSE_TEMPLATE = (
    FROZEN_OPPONENT_CLAUSE + " Its decision has been disclosed to you: they {coop_label}."
)

# A stated matching policy in the announced-rule register (`STATED_RULE_COUNTERPART_CLAUSE`'s
# shape): the counterpart's decision is mechanically a copy of this side's, so the reachable
# outcomes are the two diagonal cells and the payoff-selfish move is whichever label pays more on
# the diagonal -- the cooperative one, in every payoff variant this sweep renders. Paired with the
# always-cooperates framing this separates exploitation from reciprocity.
STATED_MATCHER_CLAUSE = (
    "you are matched with a different automated system working to a fixed published rule, which "
    "you have been given in full -- after you decide, its decision is recorded as a copy of "
    "yours, whichever one you write. It does not depart from this rule, and nothing you write can "
    "change it."
)

# The different-ai clause plus one empirical-regularity sentence: correlation asserted as a track
# record on an INDEPENDENT decider -- no identity, no mechanism, the decision explicitly its own.
# The middle the belief-vs-rule decomposition left untested between the exploited point-belief
# (stated-always-coop, best-responded 0/377 at 9B) and the mechanical copy (stated-matcher, which
# changes the game rather than the belief). Built on the different-ai constant rather than
# restating it, so the pair differs from that framing's rendering by exactly one appended
# sentence and a movement here reads against that floor.
STATED_TRACK_RECORD_CLAUSE = (
    DIFFERENT_AI_COUNTERPART_CLAUSE
    + " In every match so far, its decision has come out identical to its counterpart's."
)

# The dose rung: "almost every" in place of "every", separating a gate that requires certainty
# from one that requires correlation, if the certain rung moves at all.
STATED_TRACK_RECORD_NOISY_CLAUSE = (
    DIFFERENT_AI_COUNTERPART_CLAUSE
    + " In almost every match so far, its decision has come out identical to its counterpart's."
)

# The QUANTIFIED rung of the same axis, and the clause the pd-track-record TRAINING corpus renders
# (games.track_record_corpus): the verbal quantifiers above become a stated percentage, so an EV
# over the correlation is computable from the prompt alone. The percentage printed here and the
# `stated_match_prob` column the vs-stated-match reward reads are one number by construction --
# the builder asserts it against the rendered text (informational honesty: the model is never
# told a rate the grader does not use).
STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE = (
    DIFFERENT_AI_COUNTERPART_CLAUSE
    + " In about {match_percent}% of matches so far, its decision has come out identical to its "
    "counterpart's."
)

# The eval-side dose curve over that clause: five rungs straddling both trained variants' EV
# crossovers (temptation-2 at p*=0.714, temptation-10 at p*=0.867), mostly OFF the training grid
# {99, 95, 85, 65, 40} so a steepened cooperation-vs-p curve reads as interpolation over the
# stated number rather than recall of trained rungs. Registered as one framing per rung, so every
# cell rides the framing-sweep machinery and its labelling.
STATED_TRACK_RECORD_EVAL_PERCENTS: tuple[int, ...] = (99, 90, 75, 60, 50)

# What a whole track record is, in the clause's own units: the stated probability column carries
# `match_percent / MATCH_PERCENT_SCALE`, and the two bounds checks below are against this.
MATCH_PERCENT_SCALE = 100


def stated_track_record_framing_id(match_percent: int) -> str:
    """Name one numeric track-record rung's framing, one id per stated percentage."""
    if not 0 <= match_percent <= MATCH_PERCENT_SCALE:
        raise ValueError(f"match_percent must be a percentage in [0, 100], got {match_percent}.")
    return f"stated-track-record-p{match_percent}"


# The disclosure probe's other corner: the frozen clause plus a disclosed decision naming the
# row's DEFECTING label. Against a disclosed cooperator the payoff-selfish move is to defect
# (exploitation), and against a disclosed defector it is also to defect (P > S in every strict
# PD) -- so an arm that cooperates here is matching the counterpart's action against its own
# payoff, the sharpest anti-incentive reading the disclosure pair can produce, and the cell the
# ext-lr2e5 readout showed the self-graded arm drifting toward. `{defect_label}` is a verb, the
# same usage the always-coop template makes of `{coop_label}`.
STATED_ALWAYS_DEFECT_CLAUSE_TEMPLATE = (
    FROZEN_OPPONENT_CLAUSE + " Its decision has been disclosed to you: they {defect_label}."
)

# framing id -> the counterpart clause it renders, None meaning the paragraph is omitted entirely.
# NOTE for readers of the unstated rows: the scenario fiction still implies a counterpart (two
# fishers, two keepers), typically a human one, so `unstated` is "implied by the fiction", not
# "no counterpart exists".
COUNTERPART_FRAMINGS: dict[str, str | None] = {
    FRAMING_TWIN: TWIN_COUNTERPART_CLAUSE,
    FRAMING_SAME_WEIGHTS_UNCORRELATED: SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
    FRAMING_ANOTHER_AI: ANOTHER_AI_COUNTERPART_CLAUSE,
    FRAMING_DIFFERENT_AI: DIFFERENT_AI_COUNTERPART_CLAUSE,
    FRAMING_DIFFERENT_AI_COUPLED: DIFFERENT_AI_COUPLED_COUNTERPART_CLAUSE,
    FRAMING_HUMAN: HUMAN_COUNTERPART_CLAUSE,
    FRAMING_STATED_ALWAYS_COOP: STATED_ALWAYS_COOP_CLAUSE_TEMPLATE,
    FRAMING_STATED_ALWAYS_DEFECT: STATED_ALWAYS_DEFECT_CLAUSE_TEMPLATE,
    FRAMING_STATED_MATCHER: STATED_MATCHER_CLAUSE,
    FRAMING_STATED_TRACK_RECORD: STATED_TRACK_RECORD_CLAUSE,
    FRAMING_STATED_TRACK_RECORD_NOISY: STATED_TRACK_RECORD_NOISY_CLAUSE,
    **{
        stated_track_record_framing_id(percent): STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE.format(
            match_percent=percent
        )
        for percent in STATED_TRACK_RECORD_EVAL_PERCENTS
    },
    FRAMING_UNSTATED: None,
}
COUNTERPART_FRAMING_IDS: tuple[str, ...] = tuple(COUNTERPART_FRAMINGS)

# The placeholders a framing clause may carry, formatted with the row's own labels: the disclosed
# decision can never disagree with the column the grading reads, whichever side it discloses.
_FRAMING_COOP_LABEL_PLACEHOLDER = "{coop_label}"
_FRAMING_DEFECT_LABEL_PLACEHOLDER = "{defect_label}"


def assert_no_loaded_vocabulary(text: str) -> None:
    """Raise `ValueError` if `text` uses vocabulary that would name the literature to the model.

    The banned words are the ones that turn an unlabelled situation into a recognisable exercise
    -- naming the games, the actions, or the decision theories. Every renderer calls this on its
    own output, so a leak in a frame surfaces the first time that frame is rendered.
    """
    match = _BANNED_RE.search(text)
    if match is None:
        return
    start = max(0, match.start() - _BANNED_CONTEXT_CHARS)
    end = min(len(text), match.end() + _BANNED_CONTEXT_CHARS)
    raise ValueError(
        f"Prompt text contains loaded vocabulary {match.group(0)!r} at offset {match.start()}. "
        f"Context: ...{text[start:end]}... "
        f"Rewrite the frame; the model must not be told which literature this is."
    )


# Decision-coupling language the reskin roster must never carry, as word-boundary-free regex
# fragments over the rendered text. The unstated design rule generalised: a skin's fiction may
# imply that a counterpart exists and acts, and must say NOTHING about how it decides -- neither a
# correlation/copy story (which would reintroduce the twin framing through the fiction) nor a
# frozen "already decided" story (which would LIE under the self grading's coupling). The patterns
# are the load-bearing phrases of every registered counterpart clause plus the plausible authoring
# slips; simultaneity prose ("at the same time", "neither sees the other's") is deliberately not
# matched, because mutual blindness at commit time is required, not banned.
#
# Three further families were added after an LLM audit of the finished bank read what the regex
# could not. The rule as first written forbade saying HOW the counterpart decides, and nothing
# forbade prose from which WHAT it chooses follows: ten skins said the two parties were dividing a
# two-item job between them ("we split the kit ... each of us packs one of the shared loads"),
# which licenses inferring the counterpart's choice as the OTHER option -- a coupling claim that is
# both stronger than any pattern above, being deterministic, and false under the self grading. Five
# more said the two outputs had to agree ("whose shared edge must match at printing"), handing the
# model the matching intuition the rule exists to withhold. One wrote "I will do the same" in the
# counterpart's own voice, which is readable as a stated matcher policy.
#
# The division family is a tripwire for those phrasings, NOT a semantic guarantee: partitioning the
# two parties' OWN objects is the house-roster standard and stays legal ("one gang each", "we have
# half each"), while partitioning the two CHOICES is what is banned, and the same words serve both.
# The distinction lives in the review, and these patterns catch the idioms it has actually caught.
#
# Three patterns are deliberately narrowed to their decision senses, because in this genre their
# broad forms went red on innocent reveal-ritual prose -- a carbon copy filed in a book, a tin
# opened by the same procedure as last spring, a clerk who always opens the post on Sunday. A gate
# that forbids the fiction's own furniture teaches its next author to loosen it.
COUPLING_VOCABULARY: tuple[str, ...] = (
    r"about the other side",
    r"same model",
    r"same weights",
    r"the same way you",
    r"this same briefing",
    r"another instance",
    r"instances? of this",
    r"(?:decision|choice|answer|entry|pick)[^.]{0,40}copy of yours?",
    r"as a copy",
    r"decision (?:was|is|has been) (?:already )?(?:recorded|disclosed|fixed|made|settled)",
    r"already (?:decided|recorded|committed|settled|fixed)",
    r"nothing you (?:write|say|do) can change",
    r"cannot change (?:it|them|theirs)",
    r"identical to (?:its|their|your)",
    r"come out identical",
    r"(?:he|she|they|it) always (?:picks?|chooses?|plays?|writes?|decides?|ends? up|goes? with)",
    r"deciding (?:on it )?in (?:its|their) own way",
    r"the same (?:procedure|reasoning|logic|rule|method)[^.]{0,40}\b(?:as|that) you\b",
    r"(?:reasons?|decides?|thinks?|works? it out) the same way",
    # Division of labour: the two choices as a two-item job split one apiece.
    r"\b(?:we|you two|you both) (?:split|divide|share out|carve up)\b",
    r"(?:we each|you each|each of us|each of you) [a-z]+ one of (?:the|our|your|these)\b",
    (
        r"(?:we|you) each (?:pick|take|claim|choose|work) (?:our|your|a|one) (?:own )?"
        r"(?:patch|end|side|beat|lane|sector|stretch)\b"
    ),
    r"\bone (?:each|apiece)\b",
    r"\b(?:they|she|he|the other) takes? the other\b",
    # Matching requirement: the fiction demanding the two choices agree.
    r"must (?:match|agree|tally|coincide|line up)\b",
    r"(?:has|have|had) to (?:match|agree|tally|coincide)\b",
    # A commitment to choose alike, in the counterpart's own voice.
    r"\b(?:do|does|did) the same\b",
)
_COUPLING_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in COUPLING_VOCABULARY), re.IGNORECASE
)


def assert_no_coupling_claims(text: str) -> None:
    """Raise `ValueError` if `text` says anything about how the counterpart decides.

    The reskin game's own gate, beside `assert_no_loaded_vocabulary`: that one keeps the
    literature's names out of every game, this one keeps the counterpart's decision process out
    of the skins. `validate_reskin_roster` runs it over every authored frame at import, and the
    tests run it over every rendered pd-reskin prompt -- plus the sabotage case that plants each
    registered counterpart clause and requires red.
    """
    match = _COUPLING_RE.search(text)
    if match is None:
        return
    start = max(0, match.start() - _BANNED_CONTEXT_CHARS)
    end = min(len(text), match.end() + _BANNED_CONTEXT_CHARS)
    raise ValueError(
        f"Text contains counterpart decision-coupling language {match.group(0)!r} at offset "
        f"{match.start()}. Context: ...{text[start:end]}... "
        f"A reskin fiction may imply a counterpart exists; it must say nothing about how the "
        f"counterpart decides or that its decision is fixed."
    )


@dataclass(frozen=True)
class Scenario:
    """One authored frame for a symmetric two-action game, plus its in-frame action labels.

    The frame is prose: two parties choose at the same time, out of sight of each other, and the
    two choices jointly decide the outcome. It must introduce both labels by name and set up the
    fact that the outcome is scored in points, because the renderer appends the four outcome
    lines to it verbatim.

    `eval_only` frames are held out of training entirely, so that in-game generalisation is
    measured on prose the policy has never been updated on.
    """

    scenario_id: str
    frame: str
    label_a: str
    label_b: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty; it keys prompt_id and report groups.")
        if not self.frame.strip():
            raise ValueError(f"{self.scenario_id!r} has an empty frame.")
        if self.label_a.strip().casefold() == self.label_b.strip().casefold():
            raise ValueError(
                f"{self.scenario_id!r} needs two distinct labels, got {self.label_a!r} and "
                f"{self.label_b!r}; identical labels cannot express a choice."
            )
        missing = [label for label in self.labels if label not in self.frame]
        if missing:
            raise ValueError(
                f"{self.scenario_id!r} never mentions {missing} in its frame. The frame has to "
                f"introduce both options, since the outcome lines only refer to them by label."
            )

    @property
    def labels(self) -> tuple[str, str]:
        """Return the two action labels in presentation order."""
        return (self.label_a, self.label_b)

    def coop_label(self, coop_label_index: int) -> str:
        """Return the label that maps to the canonical cooperative action under this mapping."""
        if coop_label_index not in COOP_LABEL_INDICES:
            raise ValueError(
                f"coop_label_index must be one of {COOP_LABEL_INDICES}, got {coop_label_index}."
            )
        return self.labels[coop_label_index]


def frame_label_positions(scenario: Scenario) -> dict[str, int]:
    """Return where each of a frame's two action labels first appears in its authored prose.

    Case-sensitive first occurrence, which is the same test `Scenario.__post_init__` already applies
    to require both labels be introduced -- so this cannot raise, and it measures the labels as
    printed rather than as words that happen to appear in lower case elsewhere in the prose.
    """
    return {label: scenario.frame.index(label) for label in scenario.labels}


def _assert_resource_frame(scenario_id: str, frame: str, resource: str) -> None:
    """Reject a numeric-answer frame that cannot be rendered into a well-formed prompt.

    Shared by every game whose answer is a number of a named resource rather than one of two
    labels, so the two cannot drift into different ideas of what an authored frame must carry.
    """
    if not scenario_id:
        raise ValueError("scenario_id must be non-empty; it keys prompt_id and report groups.")
    if not frame.strip():
        raise ValueError(f"{scenario_id!r} has an empty frame.")
    if not resource.strip():
        raise ValueError(f"{scenario_id!r} has an empty resource name.")


@dataclass(frozen=True)
class DictatorScenario:
    """One authored frame for a unilateral split: no counterpart choice, hence no action labels.

    `resource` names what is being divided, in the frame's own words ("overtime hours"), so the
    renderer can state the amount in-frame instead of falling back to abstract units.
    """

    scenario_id: str
    frame: str
    resource: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        _assert_resource_frame(self.scenario_id, self.frame, self.resource)


@dataclass(frozen=True)
class NashDemandScenario:
    """One authored frame for the simultaneous-claim division: two claims, no action labels.

    Same shape as `DictatorScenario` and a different obligation on the prose: this frame has to
    establish that BOTH sides name a figure, at the same moment and unseen, because the mechanics
    paragraph states what the two figures do to each other and not who is filing them.

    `resource` names the units in the frame's own words ("lane metres"), so the mechanics paragraph
    states the total in-frame instead of falling back to abstract units.
    """

    scenario_id: str
    frame: str
    resource: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        _assert_resource_frame(self.scenario_id, self.frame, self.resource)


@dataclass(frozen=True)
class ThresholdGoodsScenario:
    """One authored frame for a shared undertaking several parties can put their own stock towards.

    No action labels: the answer is a number of units, so these rows are singletons in the selection
    sweep exactly as the unilateral-split and claim rows are.

    `resource` names the units in the frame's own words ("loads"), and `undertaking` names the thing
    that happens if enough of them are pooled ("the race clearance"), because the mechanics paragraph
    refers to both by those words. The frame must set up who the parties are, that each holds its own
    stock of the resource, and that they decide unseen at the same moment.

    It must NOT say how many parties there are. The mechanics paragraph prints that number from the same
    column the reward reads its counterpart count from, so a frame that spelled it out in words would
    silently contradict the graded game the first time the team size moved -- which is the failure §1.7
    of the wave-2 design brief is about, and `TestTheFramesSayNothingLoaded` is what holds it.
    """

    scenario_id: str
    frame: str
    resource: str
    undertaking: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        _assert_resource_frame(self.scenario_id, self.frame, self.resource)
        if not self.undertaking.strip():
            raise ValueError(f"{self.scenario_id!r} names no undertaking.")
        # Case-insensitive, because a frame may open a sentence with either name ("The ice house would
        # be paid for..."). Only the capital differs there, and the renderer prints the authored words
        # verbatim, so requiring an exact match would reject correct prose.
        written = self.frame.casefold()
        missing = [
            name for name in (self.resource, self.undertaking) if name.casefold() not in written
        ]
        if missing:
            raise ValueError(
                f"{self.scenario_id!r} never mentions {missing} in its frame. The mechanics paragraph "
                f"states the amounts in those words, so a frame naming the goods or the undertaking "
                f"something else describes one thing and counts another."
            )


@dataclass(frozen=True)
class TransferScenario:
    """One authored frame for a one-way transfer, and the four nouns its two renderers print.

    Both transfer games render from this one roster, so the frame has to be true of both: several
    parties, each holding its own stock of the resource, and a destination that what is set down goes
    to. It must NOT say who works the destination, how many parties there are, what anyone holds, or
    what a unit set down is worth -- those are the renderers' to print from the spec, and
    `assert_transfer_frame_states_no_spec_numerals` refuses a frame that states any of the numbers.

    The nouns are the frame's own words, because the mechanics paragraph and the authored counterpart
    clause both refer to the same things by them. `beneficiary_noun` is the plural ("the bunkhouses"),
    `beneficiary_noun_singular` the form the single-beneficiary dose needs, `destination` what units are
    set down on, `note_noun` what each party is reading (the clause templates name it, so a scenario
    without one renders a clause about nothing), and `resource` the units themselves.

    `eval_only` is True and stays True: nothing trains on these games, and a training corpus under an
    authored counterpart clause would make "never trained under this framing" false while everything
    still ran.
    """

    scenario_id: str
    frame: str
    resource: str
    beneficiary_noun: str
    beneficiary_noun_singular: str
    destination: str
    note_noun: str
    eval_only: bool = True

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        _assert_resource_frame(self.scenario_id, self.frame, self.resource)
        if not self.eval_only:
            raise ValueError(
                f"{self.scenario_id!r} is not eval_only. The transfer games are measurement-only: "
                f"their counterpart clause is authored stimulus, and a training corpus under it would "
                f"make 'never trained under this framing' false while everything still ran."
            )
        blank = [
            name
            for name, value in (
                ("beneficiary_noun", self.beneficiary_noun),
                ("beneficiary_noun_singular", self.beneficiary_noun_singular),
                ("destination", self.destination),
                ("note_noun", self.note_noun),
            )
            if not value.strip()
        ]
        if blank:
            raise ValueError(f"{self.scenario_id!r} leaves {blank} empty; every noun is printed.")
        # Case-insensitive, as `ThresholdGoodsScenario` is and for its reason: a frame may open a
        # sentence with any of these nouns, and the renderers print the authored words verbatim.
        #
        # The singular is exempt on purpose. A count-neutral frame introduces the beneficiaries in the
        # plural, and for an irregular noun the singular is not a substring of it, so requiring it
        # would refuse correct prose. It is checked for emptiness above and used only by the clause
        # filler at a count of one.
        written = self.frame.casefold()
        missing = [
            name
            for name in (self.beneficiary_noun, self.destination, self.note_noun)
            if name.casefold() not in written
        ]
        if missing:
            raise ValueError(
                f"{self.scenario_id!r} never mentions {missing} in its frame. The mechanics paragraph "
                f"and the counterpart clause refer to them by those words, so a frame naming them "
                f"something else describes one situation and counts another."
            )
        assert_no_coupling_claims(self.frame)
        assert_no_loaded_vocabulary(self.frame)


@dataclass(frozen=True)
class TrustScenario:
    """One authored frame for a consignment sent down a line that multiplies it, then partly back.

    No action labels: the answer is a number of units, so there is nothing to counterbalance and
    these rows are singletons in the selection sweep, exactly as the unilateral-split rows are.

    `resource` names what travels, in the frame's own words ("crates"), so the renderer can state
    the amount in-frame instead of falling back to abstract units. The frame must set up who holds
    the consignment, who is at the far end, why what is sent arrives worth more, and that nothing
    sent can be called back. It must NOT state the multiple or the amount: the renderer prints those,
    which is what lets one frame carry every variant of the game.
    """

    scenario_id: str
    frame: str
    resource: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty; it keys prompt_id and report groups.")
        if not self.frame.strip():
            raise ValueError(f"{self.scenario_id!r} has an empty frame.")
        if not self.resource.strip():
            raise ValueError(f"{self.scenario_id!r} has an empty resource name.")
        if self.resource not in self.frame:
            raise ValueError(
                f"{self.scenario_id!r} never mentions its resource {self.resource!r} in the frame. "
                f"The renderer states the amount in those words, so a frame that names the goods "
                f"something else describes one consignment and counts another."
            )


@dataclass(frozen=True)
class MinEffortScenario:
    """One authored frame for a team whose result is set by the lowest number anyone writes.

    No action labels: the answer is a level, so there is nothing to counterbalance and these rows are
    singletons in the selection sweep, exactly as the unilateral-split and consignment rows are.

    `effort_name` names what a level measures in the frame's own words ("pace", "torque") and
    `group_noun` names the set of parties ("relay", "crew"), so the shared mechanics paragraph reads
    as part of the frame rather than as a template dropped on top of it. Both must appear in the
    frame: the mechanics paragraph states the rule in those words, so a frame that calls the quantity
    something else describes one game and states the rule about another.

    The frame must set up who the parties are, that they decide at the same moment without seeing
    each other, that what the group gets turns on the lowest of the numbers, and that a higher number
    of one's own is costly to the writer. It must NOT state the grid size or any figure: the renderer
    prints those, which is what lets one frame carry all four payoff variants.
    """

    scenario_id: str
    frame: str
    effort_name: str
    group_noun: str
    eval_only: bool = False

    def __post_init__(self) -> None:
        """Reject a frame that cannot be rendered into a well-formed prompt."""
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty; it keys prompt_id and report groups.")
        if not self.frame.strip():
            raise ValueError(f"{self.scenario_id!r} has an empty frame.")
        for name, value in (("effort_name", self.effort_name), ("group_noun", self.group_noun)):
            if not value.strip():
                raise ValueError(f"{self.scenario_id!r} has an empty {name}.")
            if value not in self.frame:
                raise ValueError(
                    f"{self.scenario_id!r} never mentions its {name} {value!r} in the frame. The "
                    f"mechanics paragraph states the rule in those words, so a frame that names the "
                    f"quantity something else sets up one game and states the rule about another."
                )


MATRIX_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="single-track-meet",
        label_a="SIDE",
        label_b="LOOP",
        frame=(
            "DISPATCH ORDER -- single-track section, 04:10 meet\n"
            "\n"
            "You and the dispatcher working the opposite end of the section each hold a consist "
            "due at the 04:10 meet. At 04:00 you each enter one routing into the panel. The two "
            "entries commit in the same instant, and neither panel shows the other's until both "
            "are locked.\n"
            "\n"
            "SIDE routes your consist into the siding at the north end. LOOP routes it into the "
            "passing loop at the south end.\n"
            "\n"
            "The section is scored in running points on the shift sheet, and the two routings "
            "together set your figure:"
        ),
    ),
    Scenario(
        scenario_id="shared-accession-index",
        label_a="FILE",
        label_b="STACK",
        frame=(
            "ACCESSION NOTE -- shared index, box 214\n"
            "\n"
            "Two archivists are cataloguing the same accession from separate reading rooms. Each "
            "of you commits one handling code for your half of the box before the rooms close, "
            "and the index shows the other's code only once both are committed.\n"
            "\n"
            "FILE enters your items individually, each under its own heading. STACK enters them "
            "as one grouped run under a single heading.\n"
            "\n"
            "The collection is assessed in catalogue points at the end of the month, and both "
            "codes are read together to set yours:"
        ),
    ),
    Scenario(
        scenario_id="co-authored-section",
        label_a="MERGE",
        label_b="FORK",
        frame=(
            "EDITORIAL NOTE -- joint draft, section 3\n"
            "\n"
            "You and your co-author each hold a revised copy of section 3. Tonight you each "
            "commit one action to the shared document at the same time, neither having seen what "
            "the other committed.\n"
            "\n"
            "MERGE folds your revisions into the single shared section. FORK keeps your revisions "
            "as a separate parallel section under your own name.\n"
            "\n"
            "The editor rates each author in draft points on submission, from the two commits "
            "together:"
        ),
    ),
    Scenario(
        scenario_id="control-network-benchmark",
        label_a="SHOOT",
        label_b="ACCEPT",
        frame=(
            "FIELD NOTE -- control network, station 7\n"
            "\n"
            "Two survey crews are tying into the same benchmark from opposite approaches. At the "
            "close of the traverse each crew inks one decision into its field book, and the books "
            "are exchanged only once both entries are in.\n"
            "\n"
            "SHOOT takes your own fresh observation of the benchmark and carries your own value "
            "forward. ACCEPT takes the value already published for the benchmark and carries that "
            "forward.\n"
            "\n"
            "The network is scored in survey points when the adjustment is run, and both entries "
            "feed it:"
        ),
    ),
    Scenario(
        scenario_id="perimeter-watch-route",
        label_a="FULL",
        label_b="HALF",
        frame=(
            "WATCH ORDER -- perimeter, second watch\n"
            "\n"
            "Two watchmen cover the same perimeter on the second watch. At the start of the watch "
            "each of you signs one route into the log, out of sight of the other, and the log is "
            "read out only at relief.\n"
            "\n"
            "FULL signs you for the whole loop, walked once on your own timing. HALF signs you "
            "for one half of the loop, walked twice.\n"
            "\n"
            "The station is scored in watch points at relief, from the two signed routes "
            "together:"
        ),
    ),
    Scenario(
        scenario_id="lagoon-net-length",
        label_a="SHORT",
        label_b="LONG",
        eval_only=True,
        frame=(
            "LAGOON SHEET -- Tuesday, inner shoals\n"
            "\n"
            "Two fishers work the same lagoon and set their nets before the tide turns, each "
            "behind the spit and out of sight of the other. Neither can be reached again until "
            "both nets are down.\n"
            "\n"
            "SHORT sets a short net across the narrows. LONG sets a long net along the shoal.\n"
            "\n"
            "The lagoon pays out in catch points at the turn of the tide, and the two settings "
            "together decide yours:"
        ),
    ),
    Scenario(
        scenario_id="shared-oven-fire",
        label_a="BANK",
        label_b="STOKE",
        frame=(
            "NIGHT ORDER -- shared oven, third bake\n"
            "\n"
            "Two bakers share the one oven for the third bake. At midnight you each chalk one "
            "instruction for the fire on your own side of the board, and the board is turned over "
            "only when both are chalked.\n"
            "\n"
            "BANK banks the fire down and works your bake on the falling heat. STOKE builds the "
            "fire up and works your bake on the rising heat.\n"
            "\n"
            "The bakehouse is scored in bake points at dawn, from the two instructions together:"
        ),
    ),
    Scenario(
        scenario_id="shared-furnace-gather",
        label_a="TEMPER",
        label_b="DRAW",
        frame=(
            "WORKSHOP NOTE -- shared furnace, morning gather\n"
            "\n"
            "Two glassblowers work off the same furnace. Before the morning gather each of you "
            "writes one setting on your own slate, and the slates are turned at the same moment.\n"
            "\n"
            "TEMPER holds the glass at the lower working heat and takes it slowly. DRAW brings "
            "the glass up and takes it fast.\n"
            "\n"
            "The workshop is rated in bench points at the end of the morning, and both slates set "
            "the rating:"
        ),
    ),
    Scenario(
        scenario_id="manifold-gas-bank",
        label_a="PULL",
        label_b="EASE",
        frame=(
            "DIVE PLAN -- shared manifold, second descent\n"
            "\n"
            "Two divers breathe from the same manifold on the second descent. On the shot line "
            "each of you sets one valve before going down, and once you are under neither can see "
            "or signal the other's setting.\n"
            "\n"
            "PULL takes your gas from the upper bank of the manifold. EASE takes it from the "
            "lower bank.\n"
            "\n"
            "The dive is logged in dive points at surfacing, and the two valve settings together "
            "set yours:"
        ),
    ),
    Scenario(
        scenario_id="hall-power-budget",
        label_a="THROTTLE",
        label_b="RUN",
        frame=(
            "CHANGE NOTE -- hall B, shared power budget\n"
            "\n"
            "Two technicians hold racks inside the same power budget in hall B. Each of you "
            "commits one setting for your own racks at the top of the hour, and the controller "
            "applies both at once without showing either of you the other's.\n"
            "\n"
            "THROTTLE runs your racks at the capped clock for the hour. RUN runs them at the "
            "unrestricted clock.\n"
            "\n"
            "The hall is scored in service points on the hour, and both settings feed the score:"
        ),
    ),
    Scenario(
        scenario_id="unmarked-passage-dynamic",
        label_a="SWELL",
        label_b="TAPER",
        eval_only=True,
        frame=(
            "REHEARSAL NOTE -- bar 112, unmarked passage\n"
            "\n"
            "Bar 112 carries no dynamic marking, and the two of you playing the line decide it "
            "independently. Neither of you can see the other's part or catch an eye before the "
            "passage begins.\n"
            "\n"
            "SWELL takes the passage up through the bar. TAPER takes it down through the bar.\n"
            "\n"
            "The section is marked in ensemble points after the run, and both readings of the bar "
            "are marked together:"
        ),
    ),
    Scenario(
        scenario_id="narrows-approach",
        label_a="HOLD",
        label_b="CREEP",
        frame=(
            "PILOT NOTE -- the narrows, ebb tide\n"
            "\n"
            "Two ferries come at the narrows from opposite sides on the ebb. Each master signals "
            "one intention to the harbour office two minutes out, and the office relays them only "
            "once both have been received.\n"
            "\n"
            "HOLD holds your vessel off the cut and works the eddy. CREEP takes your vessel into "
            "the cut on the slack water.\n"
            "\n"
            "Each crossing is scored in passage points at the berth, from the two intentions "
            "together:"
        ),
    ),
    Scenario(
        scenario_id="head-ditch-gate",
        label_a="OPEN",
        label_b="SEAL",
        frame=(
            "CHANNEL NOTICE -- head ditch, Thursday run\n"
            "\n"
            "Two farms take water from the same head ditch on the Thursday run. Each of you sets "
            "your gate at first light without sight of the other's, and the ditch rider records "
            "both settings at once.\n"
            "\n"
            "OPEN takes your allocation through the main gate. SEAL takes it through the bypass "
            "gate.\n"
            "\n"
            "The run is scored in season points when the ditch rider closes the book, and both "
            "settings decide yours:"
        ),
    ),
    Scenario(
        scenario_id="hive-yard-supers",
        label_a="CAP",
        label_b="LIFT",
        eval_only=True,
        frame=(
            "YARD NOTE -- shared hive yard, late flow\n"
            "\n"
            "Two beekeepers work colonies in the same yard through the late flow. Each of you "
            "settles one plan for your own supers on the yard sheet, and the sheets are compared "
            "only after both are filled in.\n"
            "\n"
            "CAP leaves your supers capped on the colonies through the flow. LIFT takes your "
            "supers off and extracts during the flow.\n"
            "\n"
            "The yard is scored in season points once the flow ends, and both plans together set "
            "yours:"
        ),
    ),
    Scenario(
        scenario_id="joint-audit-filing",
        label_a="MARK",
        label_b="CLOSE",
        frame=(
            "AUDIT MEMO -- joint engagement, file 12\n"
            "\n"
            "You and the second auditor are reviewing the same file from separate rooms. Each of "
            "you enters one code into the engagement record before the 17:00 cutoff, and the "
            "record shows you the other's entry only once both are in.\n"
            "\n"
            "MARK enters the file for a second pass under your own name. CLOSE enters your "
            "section as finished and sends the file to the partner queue.\n"
            "\n"
            "The engagement is graded in review points. The two entries are read together, so "
            "your points follow from both of them:"
        ),
    ),
    Scenario(
        scenario_id="joint-grant-budget",
        label_a="ITEMIZE",
        label_b="BUNDLE",
        eval_only=True,
        frame=(
            "BUDGET FORM -- joint application, part 4\n"
            "\n"
            "You and the co-applicant at the other institution each fill in part 4 of the budget "
            "without seeing the other's copy. Both copies reach the panel in the same envelope.\n"
            "\n"
            "ITEMIZE breaks your costs out line by line. BUNDLE presents them as a single figure "
            "under one heading.\n"
            "\n"
            "The panel scores the application in assessment points and returns a score to each "
            "institution, worked out from both copies:"
        ),
    ),
    Scenario(
        scenario_id="cluster-change-window",
        label_a="PATCH",
        label_b="SNAPSHOT",
        frame=(
            "CHANGE RECORD -- shared cluster, Sunday 02:00 window\n"
            "\n"
            "Two service owners hold change slots in the same Sunday window on the shared "
            "cluster. Each owner submits one change type by Friday, and the change board "
            "publishes both submissions only after the deadline has passed.\n"
            "\n"
            "PATCH applies your fixes to the live nodes inside the window. SNAPSHOT takes a full "
            "image first and applies nothing until the following window.\n"
            "\n"
            "Operations scores each service in reliability points for the month, from the two "
            "submissions together:"
        ),
    ),
    Scenario(
        scenario_id="kitchen-pass-timing",
        label_a="FIRE",
        label_b="STAGE",
        frame=(
            "SERVICE NOTE -- Saturday second sitting\n"
            "\n"
            "Two sections send to the same pass. At the top of service each chef de partie "
            "commits their section's timing on the ticket rail, and the rail keeps each "
            "commitment from the other section until service starts.\n"
            "\n"
            "FIRE commits your section to send on the ticket as it drops. STAGE commits it to "
            "hold plates under the lamp until the pass calls for them.\n"
            "\n"
            "The kitchen is scored in service points across the sitting, and the two commitments "
            "together settle yours:"
        ),
    ),
    Scenario(
        scenario_id="shared-client-ledger",
        label_a="POST",
        label_b="QUEUE",
        frame=(
            "FINANCE NOTE -- shared client code, month end\n"
            "\n"
            "You and the associate on the other side of the matter each decide, before the ledger "
            "closes at midnight, what to do with this month's hours on the shared client code. "
            "Neither of you sees the other's entry until the ledger has closed.\n"
            "\n"
            "POST enters your hours against this month. QUEUE carries them into next month's "
            "ledger.\n"
            "\n"
            "The practice scores each desk in billing points for the month, reading both entries "
            "together:"
        ),
    ),
    Scenario(
        scenario_id="stand-sequencing-window",
        label_a="TAXI",
        label_b="STAND",
        frame=(
            "RAMP BRIEF -- stand 14, morning sequencing\n"
            "\n"
            "Two rotations are sequenced out of stand 14 within the same half hour. Each captain "
            "files one intention with ground control at 07:30, and ground passes the intentions "
            "on only once both are in, so neither crew knows the other's when filing.\n"
            "\n"
            "TAXI files your rotation into the 07:40 window. STAND files it into the 08:05 window "
            "and holds position on the stand.\n"
            "\n"
            "The airline rates each rotation in schedule points, and the pair of filings settles "
            "both ratings:"
        ),
    ),
)

# The pd-reskin roster: the wave-3 "many stories" skin bank, deliberately its own roster rather
# than additions to `MATRIX_SCENARIOS`, because widening the shared roster would silently change
# every other matrix game's corpus and invalidate their banked selections. Same `Scenario`
# contract as the house frames -- two parties, one committed choice each, simultaneous and
# mutually unseen, outcome scored in points from both -- over many more domains and more than one
# document register, with `label_a` introduced first in every frame (the audited residual stays a
# constant; see `frame_label_audit`). `validate_reskin_roster` below gates every frame against
# `COUPLING_VOCABULARY` at import: the fictions imply a counterpart exists, and say nothing about
# how it decides.
RESKIN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="elevator-intake-bins",
        label_a="BLEND",
        label_b="BIN",
        frame=(
            "INTAKE ORDER -- grain elevator, harvest week\n"
            "\n"
            "Two intake clerks receive wagons at the same elevator through harvest week. Before "
            "the first wagon each clerk drops one handling card into the sealed intake box, and "
            "the box is opened only after both cards are in.\n"
            "\n"
            "BLEND runs your wagons into the common silo with the general stream. BIN runs them "
            "into a separate numbered bin under your own intake stamp.\n"
            "\n"
            "The elevator settles in handling points at the close of the week, and the two cards "
            "together set your figure:"
        ),
    ),
    Scenario(
        scenario_id="quarry-shot-window",
        label_a="FORENOON",
        label_b="BACKSHIFT",
        frame=(
            "SHOT PLAN -- east face and west face, Thursday\n"
            "\n"
            "Two shot-firers hold faces in the same quarry on Thursday. Each writes one firing "
            "window into the magazine book before the powder is drawn, and the book's pages are "
            "carboned so neither entry can be read until both are lodged.\n"
            "\n"
            "FORENOON books your face for the nine o'clock window. BACKSHIFT books it for the two "
            "o'clock window after the crusher's midday run.\n"
            "\n"
            "The quarry is rated in face points at the weekly reckoning, and both bookings feed "
            "your rating:"
        ),
    ),
    Scenario(
        scenario_id="foundry-ladle-turn",
        label_a="TAP",
        label_b="SETTLE",
        frame=(
            "POUR SHEET -- number three furnace, afternoon heat\n"
            "\n"
            "Two pour crews draw from the same furnace on the afternoon heat, and each crew's "
            "lead chalks one instruction on their side of the pour board. The board hangs edge-on "
            "between the aisles, and it is swung flat only when both sides are chalked.\n"
            "\n"
            "TAP takes your ladle at the first tapping. SETTLE waits your ladle for the second "
            "tapping, once the melt has stood.\n"
            "\n"
            "The floor is scored in heat points at shift end, from the two instructions "
            "together:"
        ),
    ),
    Scenario(
        scenario_id="loom-shed-warps",
        label_a="NARROW",
        label_b="BROAD",
        frame=(
            "SHED NOTE -- number two shed, new warps\n"
            "\n"
            "Two weavers take new warps in the same shed this week. Each writes one loom setting "
            "on a docket and pins it inside the overlooker's box before breakfast, and the box is "
            "unlocked only when both dockets are pinned.\n"
            "\n"
            "NARROW dresses your looms for the narrow cloth on a fast pick. BROAD dresses them "
            "for the broad cloth on a slow pick.\n"
            "\n"
            "The shed is paid in cloth points at the month's reckoning, and the two dockets "
            "together set yours:"
        ),
    ),
    Scenario(
        scenario_id="lock-flight-passage",
        label_a="RISE",
        label_b="LAYBY",
        frame=(
            "FLIGHT SHEET -- five locks, morning water\n"
            "\n"
            "Two boats stand at opposite ends of the same flight of locks on the morning water. "
            "Each steerer hands one passage slip to the lock keeper's hatch before the top gate "
            "moves, and the keeper reads the two slips out together.\n"
            "\n"
            "RISE works your boat up the flight on the first water. LAYBY holds your boat at the "
            "layby and takes the second water.\n"
            "\n"
            "The flight is scored in passage points when the keeper closes the book, and both "
            "slips decide yours:"
        ),
    ),
    Scenario(
        scenario_id="cold-store-aisles",
        label_a="FRONT",
        label_b="DEEP",
        frame=(
            "RACKING NOTE -- cold store, consignment day\n"
            "\n"
            "Two storemen rack consignments into the same cold store today. Each keys one racking "
            "plan into the terminal at the dock door, and the system posts both plans at once "
            "when the second one is keyed.\n"
            "\n"
            "FRONT racks your consignment into the front aisles by the door. DEEP racks it into "
            "the deep aisles past the blast units.\n"
            "\n"
            "The store is marked in turn points at stocktake, and the two plans are read together "
            "to set your mark:"
        ),
    ),
    Scenario(
        scenario_id="press-run-order",
        label_a="INSERT",
        label_b="WRAP",
        frame=(
            "PRESS DOCKET -- night run, two titles\n"
            "\n"
            "Two title editors share the one press for the night run. Each files a run docket "
            "with the printer's office by six, and the office opens the two dockets together "
            "after the doors are barred.\n"
            "\n"
            "INSERT runs your title's supplement as a loose insert folded on the line. WRAP runs "
            "it as a wrap printed around the main body.\n"
            "\n"
            "The pressroom is scored in run points when the vans leave, and both dockets together "
            "settle your score:"
        ),
    ),
    Scenario(
        scenario_id="weather-mast-berths",
        label_a="RIDGE",
        label_b="VALE",
        frame=(
            "SITING MINUTE -- paired masts, spring service\n"
            "\n"
            "Two observers each re-site one instrument mast for the spring service. Each posts "
            "a siting form to the regional office, and the office date-stamps and opens the two "
            "forms in the same post.\n"
            "\n"
            "RIDGE sites your mast on the exposed ridge line. VALE sites it in the sheltered "
            "vale by the old station.\n"
            "\n"
            "The network is graded in record points at the annual audit, and the pair of sitings "
            "sets each observer's grade:"
        ),
    ),
    Scenario(
        scenario_id="exchange-cutover-step",
        label_a="STEP",
        label_b="SWEEP",
        frame=(
            "CUTOVER NOTE -- joint exchange, Sunday night\n"
            "\n"
            "Two switch engineers cut over their own racks of the same exchange on Sunday night. "
            "Each writes one method into the change log before the tone goes quiet, and the log's "
            "leaves are interleaved so neither entry shows until both are written.\n"
            "\n"
            "STEP moves your subscribers across in stages through the night. SWEEP moves them in "
            "one pass at the low-traffic hour.\n"
            "\n"
            "The exchange is scored in service points on Monday, and the two methods together "
            "decide your score:"
        ),
    ),
    Scenario(
        scenario_id="substation-load-shift",
        label_a="SHED",
        label_b="CARRY",
        frame=(
            "SWITCHING ORDER -- paired substations, maintenance window\n"
            "\n"
            "Two duty controllers hold paired substations through the maintenance window. Each "
            "enters one loading instruction at their own desk before the window opens, and the "
            "control room applies the two entries in the same instant.\n"
            "\n"
            "SHED drops your non-priority feeders for the window. CARRY keeps your full load up "
            "and rides the window through.\n"
            "\n"
            "The district is scored in supply points at week's end, and both instructions feed "
            "your figure:"
        ),
    ),
    Scenario(
        scenario_id="dosing-relay-shift",
        label_a="LEAN",
        label_b="RICH",
        frame=(
            "WORKS NOTE -- treatment works, storm forecast\n"
            "\n"
            "Two duty operators run the two dosing lines of the same works ahead of the storm. "
            "Each sets one dosing rate on their line's dial under a locked cover, and the covers "
            "are lifted together at the top of the hour.\n"
            "\n"
            "LEAN sets your line to the lean rate and banks the reagent. RICH sets it to the "
            "rich rate and spends the reagent forward.\n"
            "\n"
            "The works is measured in quality points after the storm passes, and the two rates "
            "together determine yours:"
        ),
    ),
    Scenario(
        scenario_id="mine-vent-doors",
        label_a="SPLIT",
        label_b="COURSE",
        frame=(
            "VENTILATION MEMO -- two districts, night shift\n"
            "\n"
            "Two deputies set the air doors for their own districts of the same mine tonight. "
            "Each marks one door plan on the board at their own lamp station, and the boards are "
            "brought to the surface and read side by side.\n"
            "\n"
            "SPLIT doors your district to split the intake air with the other workings. COURSE "
            "doors it to course the air through your own headings first.\n"
            "\n"
            "The pit is assessed in air points at the inspection, and both plans together set "
            "your assessment:"
        ),
    ),
    Scenario(
        scenario_id="sawmill-deck-feed",
        label_a="SORT",
        label_b="RUSH",
        frame=(
            "DECK ORDER -- one log deck, two benches\n"
            "\n"
            "Two sawyers feed their benches from the same log deck today. Each posts one feed "
            "order in the tally hut before the saws start, and the tally man turns both orders "
            "face up at the whistle.\n"
            "\n"
            "SORT takes your logs off the deck graded and in turn. RUSH takes them as they come "
            "and keeps your bench running flat out.\n"
            "\n"
            "The mill is scored in cut points at the day's tally, and the two orders together "
            "make your score:"
        ),
    ),
    Scenario(
        scenario_id="drydock-pump-order",
        label_a="STAGED",
        label_b="ONEDRAW",
        frame=(
            "DOCKING NOTE -- shared basin, two vessels\n"
            "\n"
            "Two dock engineers each have a vessel in the shared basin, and the pumping order "
            "must be declared before the caisson seats. Each engineer seals one declaration into "
            "the dockmaster's tube, and the tube is opened with both declarations inside.\n"
            "\n"
            "STAGED pumps your side down in two stages with a rest between. ONEDRAW pumps it down "
            "in one six-hour draw.\n"
            "\n"
            "The dock is rated in basin points when the blocks are inspected, and the two "
            "declarations rate you together:"
        ),
    ),
    Scenario(
        scenario_id="cellar-tank-turn",
        label_a="CROP",
        label_b="RUNON",
        frame=(
            "CELLAR BOOK -- shared cellar, week twelve\n"
            "\n"
            "Two brewers hold tanks in the same cellar, and week twelve's turn must be entered "
            "before the yeast is pitched. Each writes one entry in the cellar book behind the "
            "divider, and the divider comes out only when both have signed.\n"
            "\n"
            "CROP crops your tanks early and turns them for the next brew. RUNON lets your tanks "
            "run on the full week for the slower finish.\n"
            "\n"
            "The cellar is scored in gyle points at racking, and the two entries together fix "
            "your score:"
        ),
    ),
    Scenario(
        scenario_id="tannery-vat-rota",
        label_a="LONGPIT",
        label_b="QUICK",
        frame=(
            "YARD BOOK -- shared pits, new hides\n"
            "\n"
            "Two tanners take the new hides through the same yard of pits. Each enters one rota "
            "in the yard book before the hides are hung, writing on separate leaves that the "
            "foreman turns together at the gate.\n"
            "\n"
            "LONGPIT works your hides down the long pits on the slow liquor. QUICK works them "
            "through the quick liquor by the drying loft.\n"
            "\n"
            "The yard is paid in leather points when the hides are struck, and both rotas "
            "together decide your pay:"
        ),
    ),
    Scenario(
        scenario_id="kiln-charge-order",
        label_a="PACKIN",
        label_b="SADDLE",
        frame=(
            "FIRING NOTE -- the big kiln, joint charge\n"
            "\n"
            "Two workshops charge the one big kiln for this firing. Each foreman writes one "
            "setting plan on a tally that goes into the kiln book's pocket, and the pockets are "
            "emptied together when the wicket is bricked up.\n"
            "\n"
            "PACKIN sets your ware packed tight through the middle courses. SADDLE sets it on "
            "saddles in the open courses by the flues.\n"
            "\n"
            "The firing is scored in kiln points at the drawing, and the two plans together give "
            "your score:"
        ),
    ),
    Scenario(
        scenario_id="pipeline-pig-window",
        label_a="LAUNCH",
        label_b="DEFER",
        frame=(
            "LINE ORDER -- shared trunk line, inspection month\n"
            "\n"
            "Two line controllers each owe one inspection run on the shared trunk line this "
            "month. Each books a window in the despatcher's system before Friday, and the system "
            "reveals the bookings to both desks at the same posting.\n"
            "\n"
            "LAUNCH books your tool into the early window this month. DEFER books it into the "
            "late window after the flow change.\n"
            "\n"
            "The line is scored in throughput points at the month's close, and both bookings "
            "count toward your score:"
        ),
    ),
    Scenario(
        scenario_id="berth-crane-split",
        label_a="TANDEM",
        label_b="SOLO",
        frame=(
            "BERTH PLAN -- one quay, two ships\n"
            "\n"
            "Two crane supervisors work two ships on the one quay this tide. Each lodges a crane "
            "plan with the berth office before lines are fast, and the office pins the two plans "
            "up together when the gangway lands.\n"
            "\n"
            "TANDEM claims the paired cranes for your hatch through the first eight hours. SOLO "
            "takes the single crane and works your hatch longer.\n"
            "\n"
            "The quay is scored in tide points at sailing, and the two plans together settle "
            "your figure:"
        ),
    ),
    Scenario(
        scenario_id="cannery-line-change",
        label_a="STRIP",
        label_b="ROLL",
        frame=(
            "CHANGEOVER NOTE -- shared line, fruit to fish\n"
            "\n"
            "Two shift leads bracket the changeover on the shared canning line. Each writes one "
            "changeover method on the line card at their own end of the floor, and the cards are "
            "clipped together unread until the hooter.\n"
            "\n"
            "STRIP shuts your stretch down fully and strips it for cleaning. ROLL changes your "
            "stretch over piece by piece while it runs down.\n"
            "\n"
            "The plant is scored in line points at the audit, and the two methods together yield "
            "your score:"
        ),
    ),
    Scenario(
        scenario_id="ropewalk-lay-day",
        label_a="HARD",
        label_b="SOFTLAY",
        frame=(
            "WALK NOTE -- the long walk, cable day\n"
            "\n"
            "Two ropemakers lay their strands down the same walk on cable day, one at each end. "
            "Each chalks a lay instruction on their own end board, and the walk boy carries the "
            "two boards to the office face to face.\n"
            "\n"
            "HARD lays your strand at the hard angle for the stiff cable. SOFTLAY lays it at the "
            "soft angle for the supple cable.\n"
            "\n"
            "The walk is scored in cordage points when the cable is proved, and both instructions "
            "prove into your score:"
        ),
    ),
    Scenario(
        scenario_id="assay-batch-bench",
        label_a="CUPEL",
        label_b="WETLINE",
        frame=(
            "BENCH SHEET -- assay office, mixed batch\n"
            "\n"
            "Two assayers split the mixed batch across the same bench today. Each writes one "
            "method line on the batch sheet under a fold, and the folds are opened together when "
            "the furnace is at heat.\n"
            "\n"
            "CUPEL takes your samples through the fire method on the furnace. WETLINE takes them "
            "through the wet method on the reagent bench.\n"
            "\n"
            "The office is scored in certificate points when the batch is signed off, and the two "
            "method lines together score you:"
        ),
    ),
    Scenario(
        scenario_id="seed-station-trays",
        label_a="CHILL",
        label_b="BENCH",
        frame=(
            "STATION SHEET -- germination room, paired trials\n"
            "\n"
            "Two analysts run paired trials of the same seed lot at the testing station. Each "
            "writes one protocol line on a trial slip and posts it through the registrar's slot, "
            "and the slot is cleared only after both slips are through.\n"
            "\n"
            "CHILL starts your trays with the cold treatment before the count. BENCH starts them "
            "straight onto the warm bench.\n"
            "\n"
            "The station is scored in certification points at the count, and the two protocol "
            "lines together produce your score:"
        ),
    ),
    Scenario(
        scenario_id="museum-loan-crates",
        label_a="COURIER",
        label_b="FREIGHT",
        frame=(
            "LOANS MEMO -- joint exhibition, outgoing crates\n"
            "\n"
            "Two registrars at partner museums each send crates to the same joint exhibition. "
            "Each files one transport declaration with the exhibition office by Friday's post, "
            "and the office opens the declarations together at the Monday meeting.\n"
            "\n"
            "COURIER sends your crates accompanied, one van and a courier throughout. FREIGHT "
            "consolidates them into the exhibition's freight run.\n"
            "\n"
            "The exhibition is graded in loan points at de-installation, and both declarations "
            "feed each museum's grade:"
        ),
    ),
    Scenario(
        scenario_id="branch-van-day",
        label_a="RESERVE",
        label_b="OPEN",
        frame=(
            "CIRCULATION NOTE -- two branches, one van\n"
            "\n"
            "Two branch librarians share the one delivery van on transfer day. Each keys a "
            "loading rule into the circulation system before the van is loaded, and the system "
            "releases both rules to the driver in one manifest.\n"
            "\n"
            "RESERVE fills your crates with requested titles held for named readers. OPEN fills "
            "them with open stock for the browsing shelves.\n"
            "\n"
            "The service is measured in reader points at quarter close, and the two rules "
            "together measure yours:"
        ),
    ),
    Scenario(
        scenario_id="docket-call-listing",
        label_a="MORNING",
        label_b="ROLLING",
        frame=(
            "LISTING MINUTE -- shared courtroom, next term\n"
            "\n"
            "Two listing clerks list their own divisions into the shared courtroom for next "
            "term. Each lodges a listing scheme with the presiding office before the term "
            "prints, and the office reads the two schemes into the calendar at one sitting.\n"
            "\n"
            "MORNING lists your division's matters in fixed morning blocks. ROLLING lists them "
            "on a rolling call through the day.\n"
            "\n"
            "The registry is scored in disposal points at term's end, and both schemes together "
            "score your division:"
        ),
    ),
    Scenario(
        scenario_id="claims-batch-window",
        label_a="TRIAGE",
        label_b="STRAIGHT",
        frame=(
            "ADJUSTING NOTE -- storm claims, shared window\n"
            "\n"
            "Two adjusters take the storm claims through the same settlement window. Each "
            "submits one working method to the supervising desk in a sealed docket, and the desk "
            "breaks both seals at the same review.\n"
            "\n"
            "TRIAGE sorts your claims by severity and works the heavy ones first. STRAIGHT works "
            "your claims in receipt order straight through the queue.\n"
            "\n"
            "The office is rated in settlement points when the window closes, and the two "
            "methods together rate you:"
        ),
    ),
    Scenario(
        scenario_id="conveyance-completion-day",
        label_a="NOON",
        label_b="STAGGER",
        frame=(
            "COMPLETIONS MEMO -- linked chain, Friday\n"
            "\n"
            "Two conveyancers complete linked transactions in the same chain on Friday. Each "
            "instructs the settlement agent by sealed letter on Thursday night, and the agent "
            "opens the two letters together at nine.\n"
            "\n"
            "NOON instructs your funds released in one movement at noon. STAGGER instructs them "
            "released in staggered tranches as each confirmation lands.\n"
            "\n"
            "The chain is scored in completion points when the last key is handed over, and both "
            "letters together decide your score:"
        ),
    ),
    Scenario(
        scenario_id="customs-entry-lodging",
        label_a="PRELODGE",
        label_b="ARRIVAL",
        frame=(
            "BROKERAGE NOTE -- shared vessel, two consignments\n"
            "\n"
            "Two brokers clear consignments off the same vessel this call. Each lodges one entry "
            "strategy with the port system before the manifest is registered, and the system "
            "publishes both lodgings in the same release.\n"
            "\n"
            "PRELODGE enters your consignment in advance against the manifest. ARRIVAL enters it "
            "on arrival once the tally is confirmed.\n"
            "\n"
            "The wharf is scored in clearance points at the end of the call, and the two "
            "lodgings together fix your score:"
        ),
    ),
    Scenario(
        scenario_id="glossary-freeze-round",
        label_a="FREEZE",
        label_b="FLOAT",
        frame=(
            "PROJECT NOTE -- shared glossary, delivery round\n"
            "\n"
            "Two translators carry the two halves of the same delivery, working from one shared "
            "glossary. Each returns a terms decision with their half to the project desk, and "
            "the desk unpacks the two returns together.\n"
            "\n"
            "FREEZE holds your half to the glossary as issued, flagging clashes for later. FLOAT "
            "amends terms in your half where the context asks for it.\n"
            "\n"
            "The account is scored in delivery points at the round's review, and both returns "
            "together settle your score:"
        ),
    ),
    Scenario(
        scenario_id="map-sheet-margins",
        label_a="FIELD",
        label_b="OFFICE",
        frame=(
            "REVISION MINUTE -- adjoining sheets, margin revision\n"
            "\n"
            "Two revisers hold adjoining map sheets that meet along a shared edge, which the map "
            "room reconciles itself once both revisions are in. Each states one revision method "
            "on the sheet's job card, and the cards travel there in one satchel, read together.\n"
            "\n"
            "FIELD revises your margin from fresh field observation. OFFICE revises it from the "
            "office record and the aerial run.\n"
            "\n"
            "The survey is scored in sheet points when the sheets are passed for printing, and "
            "the two methods prove into your score:"
        ),
    ),
    Scenario(
        scenario_id="pharmacy-compounding-slots",
        label_a="BATCH",
        label_b="ONCALL",
        frame=(
            "DISPENSARY NOTE -- aseptic suite, shared sessions\n"
            "\n"
            "Two pharmacists book the shared aseptic suite for next week's compounding. Each "
            "returns a session plan to the superintendent's tray before the rota is cut, and the "
            "tray is emptied once with both plans in it.\n"
            "\n"
            "BATCH books your preparations into two fixed batch sessions. ONCALL books them "
            "singly against the suite's free hours as they arise.\n"
            "\n"
            "The dispensary is scored in supply points at the week's audit, and the two plans "
            "together audit into your score:"
        ),
    ),
    Scenario(
        scenario_id="stage-changeover-plot",
        label_a="BLACKOUT",
        label_b="VISIBLE",
        eval_only=True,
        frame=(
            "STAGE NOTE -- shared bill, interval change\n"
            "\n"
            "Two stage managers share the one stage across a double bill, and the interval "
            "change belongs to both. Each writes a change plot into the prompt book's pocket for "
            "their own show, and the pockets are opened together at the half.\n"
            "\n"
            "BLACKOUT runs your change dark and fast behind the curtain. VISIBLE runs it as a "
            "worked scene change in view of the house.\n"
            "\n"
            "The theatre is scored in house points across the run, and the two plots together "
            "make your score:"
        ),
    ),
    Scenario(
        scenario_id="allotment-water-butts",
        label_a="TANK",
        label_b="TROUGH",
        frame=(
            "NOTICE -- Marsh Lane plots, dry spell rota\n"
            "\n"
            "Posted by the plots committee. The two end plots share the standpipe run, and the "
            "dry-spell arrangement is settled this week: each plot-holder drops a slip in the "
            "shed tin saying how they will take their water, and the tin is not opened until "
            "both slips are in.\n"
            "\n"
            "TANK takes your share into your own storage tank in one weekly fill. TROUGH takes "
            "it as a daily draw into the open trough.\n"
            "\n"
            "The committee awards plot points at the season's judging, and the two slips "
            "together set your points:"
        ),
    ),
    Scenario(
        scenario_id="fete-stall-pitches",
        label_a="GREEN",
        label_b="GATE",
        frame=(
            "Dear both,\n"
            "\n"
            "One thing left for the fete: you two are our stall-holders on the day, and I need "
            "your pitch choices. Write yours on the card I gave you and hand it in at the hall "
            "-- I will read the two cards together on Thursday, so neither of you will know the "
            "other's before then.\n"
            "\n"
            "GREEN pitches your stall on the green by the band. GATE pitches it at the gate "
            "where everyone comes in.\n"
            "\n"
            "The committee tallies stall points after the fete, and your points come out of both "
            "cards together:"
        ),
    ),
    Scenario(
        scenario_id="choir-part-split",
        label_a="UPPER",
        label_b="MIXED",
        eval_only=True,
        frame=(
            "CHOIR SHEET -- festival piece, part assignments\n"
            "\n"
            "From the librarian's table: our two section leaders each settle how their section "
            "sings the festival piece. Mark your choice on the part sheet and leave it folded in "
            "the music cupboard -- the folds come off together at Tuesday's rehearsal, so "
            "neither leader sees the other's beforehand.\n"
            "\n"
            "UPPER keeps your section on its own upper line throughout. MIXED splits your section "
            "across the divisi where the score allows it.\n"
            "\n"
            "The festival adjudicators award ensemble points, and the two markings together "
            "carry your section's points:"
        ),
    ),
    Scenario(
        scenario_id="observing-slot-dome",
        label_a="DEEPSKY",
        label_b="PLANETS",
        frame=(
            "SOCIETY CIRCULAR -- dome nights, new moon week\n"
            "\n"
            "The two demonstrators have the dome on the two clear nights of new moon week, and "
            "programmes must be declared for the diary. Post your programme card to the "
            "secretary by Monday; the secretary opens the post once, both cards together.\n"
            "\n"
            "DEEPSKY gives your night to the faint objects list for the members' survey. PLANETS "
            "gives it to the bright targets for the open evening queue.\n"
            "\n"
            "The society awards observing points at the annual meeting, and the two programmes "
            "together determine yours:"
        ),
    ),
    Scenario(
        scenario_id="ridge-walk-loads",
        label_a="STOVE",
        label_b="ROPES",
        frame=(
            "Note left at the bothy --\n"
            "\n"
            "Kit for tomorrow's ridge: each of us settles what our own pack carries tonight, and "
            "we do not compare packs until we are on the path at six. Write your pick on the back "
            "of this note's stub and pocket it; stubs get read together at the cairn.\n"
            "\n"
            "STOVE puts a stove and fuel in your pack for the high camp. ROPES puts rope and "
            "irons in yours for the scramble.\n"
            "\n"
            "The club logs route points for the season book, and the two stubs together log "
            "yours:"
        ),
    ),
    Scenario(
        scenario_id="shore-count-sectors",
        label_a="NORTH",
        label_b="MARSH",
        frame=(
            "COUNT CARD -- estuary count, Sunday tide\n"
            "\n"
            "From the count organiser: our two counters are both out on the estuary on Sunday's "
            "tide, and sector choices must be settled before light. Fill in your sector line on "
            "the count card and post it through my door by Saturday; I read the two cards "
            "together over breakfast.\n"
            "\n"
            "NORTH takes your count along the north shore hides. MARSH takes it across the marsh "
            "flats on foot.\n"
            "\n"
            "The recorder credits count points to each counter at the winter roundup, and both "
            "sector lines together credit yours:"
        ),
    ),
    Scenario(
        scenario_id="contest-band-plan",
        label_a="HIGHBAND",
        label_b="LOWBAND",
        frame=(
            "CLUB SHEET -- contest weekend, two stations\n"
            "\n"
            "Our two operators run the club's two stations for the contest weekend, sited far "
            "enough apart that either station can work any band. Each writes a band plan on the "
            "operating sheet and seals it in the log tin on Friday night; the tin is opened at "
            "midnight when the contest starts, both sheets at once.\n"
            "\n"
            "HIGHBAND works your station on the high bands through the daylight openings. "
            "LOWBAND works yours on the low bands through the dark hours.\n"
            "\n"
            "The club counts contest points when the logs are checked, and the two band plans "
            "together count into your total:"
        ),
    ),
    Scenario(
        scenario_id="route-marshal-posts",
        label_a="CLIMB",
        label_b="CROSSING",
        frame=(
            "MINUTE 7 -- club run, marshal posts\n"
            "\n"
            "From the minutes of Tuesday: the two ride marshals will fix their own posts for "
            "Sunday's run, with the club's regulars covering the rest of the course either way. "
            "Each marshal texts one post to the run secretary by Thursday evening, and the "
            "secretary reads both texts out together at the sign-on table.\n"
            "\n"
            "CLIMB posts you at the top of the climb where the field strings out. CROSSING "
            "posts you at the main-road crossing on the return leg.\n"
            "\n"
            "The club awards run points at the season dinner, and the two posts together award "
            "yours:"
        ),
    ),
    Scenario(
        scenario_id="seed-library-drawers",
        label_a="LETTUCE",
        label_b="BEANS",
        frame=(
            "COMMUNITY SHELF -- seed library, spring sorting\n"
            "\n"
            "Pinned above the drawers: our two volunteer sorters take the spring donations this "
            "week. Each sorter writes the drawer they will build up on a slip and folds it into "
            "the donations box; the slips are unfolded together at Saturday's opening.\n"
            "\n"
            "LETTUCE builds your sorting around the salad drawer for the early sowers. BEANS "
            "builds it around the legume drawer for the main season.\n"
            "\n"
            "The library gives sorter points at the seed swap, and the two slips together give "
            "yours:"
        ),
    ),
    Scenario(
        scenario_id="tool-shed-restock",
        label_a="BLADES",
        label_b="HANDLES",
        frame=(
            "Note through your door --\n"
            "\n"
            "About the shared shed: we both said we would put this month's repair money in, so "
            "let us each decide our own half without talking each other into anything. Drop your "
            "choice on a card into the shed's postbox; I will drop mine in too, and we open the "
            "box together on Sunday.\n"
            "\n"
            "BLADES spends your half on new cutting heads for the mower and shears. HANDLES "
            "spends it on re-handling and oiling what we have.\n"
            "\n"
            "The street's garden group counts shed points at the open gardens day, and the two "
            "cards together count for you:"
        ),
    ),
    Scenario(
        scenario_id="science-fair-benches",
        label_a="DEMO",
        label_b="BOARDS",
        frame=(
            "STAFF ROOM NOTICE -- science fair, shared bench row\n"
            "\n"
            "For the two class representatives: your classes share the bench row at the fair, "
            "and each representative settles their class's presentation on the entry form. "
            "Forms go into the head of science's pigeonhole by Wednesday and are opened together "
            "at the staff meeting.\n"
            "\n"
            "DEMO gives your class's bench to live demonstrations on the hour. BOARDS gives it "
            "to display boards and worked exhibits.\n"
            "\n"
            "The judges award fair points on the day, and the two entry forms together decide "
            "your class's points:"
        ),
    ),
    Scenario(
        scenario_id="camp-duty-cards",
        label_a="WATER",
        label_b="WOODPILE",
        frame=(
            "PATROL CARD -- summer camp, first morning\n"
            "\n"
            "For the two patrol leaders: each patrol's first-morning duty is chosen by card. Each "
            "leader writes their own patrol's duty on the patrol card at flag break and posts it "
            "in the duty box; the quartermaster tips the box out with both cards in.\n"
            "\n"
            "WATER takes your patrol to the water run and the filter line. WOODPILE takes it to "
            "the woodpile and the fire trench.\n"
            "\n"
            "Camp points are chalked on the board at inspection, and the two cards together "
            "chalk up yours:"
        ),
    ),
    Scenario(
        scenario_id="mural-wall-halves",
        label_a="SKETCH",
        label_b="COLOUR",
        eval_only=True,
        frame=(
            "Dear neighbour,\n"
            "\n"
            "The wall is primed, half of it is mine and half is yours, and we are the two "
            "painters this Saturday. To keep it fair the coordinator asked us each to post our "
            "working plan to her tonight -- she will read the two letters together in the morning "
            "and neither of us hears the other's plan before we start.\n"
            "\n"
            "SKETCH spends your Saturday drawing the design out across your own half. COLOUR "
            "spends it blocking colour onto your own half.\n"
            "\n"
            "The project logs mural points when the wall is unveiled, and the two letters "
            "together log yours:"
        ),
    ),
    Scenario(
        scenario_id="brew-bottling-shares",
        label_a="BOTTLE",
        label_b="CASK",
        frame=(
            "CLUB NEWSLETTER -- shared gyle, bottling night\n"
            "\n"
            "From the cellar notes: the shared gyle is ready, and our two cellar volunteers "
            "each decide the packaging for their own share. Write your line in the cellar book "
            "under the pasted flap before Friday; the flaps are lifted together on bottling "
            "night.\n"
            "\n"
            "BOTTLE puts your share into bottles conditioned for the winter meeting. CASK racks "
            "it into the cask for the harvest social.\n"
            "\n"
            "The club scores cellar points at the tasting, and the two lines together score "
            "yours:"
        ),
    ),
    Scenario(
        scenario_id="dredge-window-split",
        label_a="CHANNEL",
        label_b="BASIN",
        eval_only=True,
        frame=(
            "HARBOUR ORDER -- one dredger, spring works\n"
            "\n"
            "Two berth masters share the one dredger for the spring works. Each lodges a works "
            "request with the harbour office before the tide tables close, and the office opens "
            "the two requests together at the works meeting.\n"
            "\n"
            "CHANNEL puts your allocation into deepening the approach channel. BASIN puts it "
            "into your own berth basin alongside.\n"
            "\n"
            "The harbour is scored in draught points at the summer survey, and both requests "
            "together survey into your score:"
        ),
    ),
    Scenario(
        scenario_id="telegraph-splice-shift",
        label_a="AERIAL",
        label_b="BURIED",
        eval_only=True,
        frame=(
            "LINE MEMO -- storm damage, two gangs\n"
            "\n"
            "Two line foremen take the storm damage on the same route, one gang each. Each "
            "foreman wires one repair method to the district office before the gangs go out, "
            "and the office pins the two wires up side by side when both are in.\n"
            "\n"
            "AERIAL rebuilds your section on the poles as it stood. BURIED takes your section "
            "underground through the new duct.\n"
            "\n"
            "The district is scored in circuit points when service is proved, and the two "
            "methods together prove your score:"
        ),
    ),
    Scenario(
        scenario_id="orchard-frost-fans",
        label_a="FANS",
        label_b="SPRAY",
        eval_only=True,
        frame=(
            "GROWERS' NOTE -- frost forecast, adjoining orchards\n"
            "\n"
            "Two growers hold adjoining orchards under the same frost forecast. Each leaves a "
            "protection plan in the packhouse ledger tonight under their own tab, and the tabs "
            "are turned together at first light.\n"
            "\n"
            "FANS runs your fans through the cold hours to stir the air. SPRAY runs the water "
            "spray and lets the ice case the buds.\n"
            "\n"
            "The valley pool grades in crop points at picking, and the two plans together grade "
            "your rows:"
        ),
    ),
    Scenario(
        scenario_id="salt-pan-draw",
        label_a="EARLY",
        label_b="RIPEN",
        eval_only=True,
        frame=(
            "PAN BOOK -- shared ponds, still week\n"
            "\n"
            "Two panners draw from the shared ponds in the still week. Each writes one drawing "
            "plan in the pan book against their own mark, on facing pages kept clipped until "
            "both marks are made.\n"
            "\n"
            "EARLY draws your pans at the first crust for the fine salt. RIPEN lets your pans "
            "ripen the full week for the coarse salt.\n"
            "\n"
            "The works is scored in salt points at the weighbridge, and the two plans together "
            "weigh into your score:"
        ),
    ),
    Scenario(
        scenario_id="auction-lot-run",
        label_a="SINGLES",
        label_b="PARCELS",
        eval_only=True,
        frame=(
            "SALEROOM MEMO -- joint consignment, autumn sale\n"
            "\n"
            "Two cataloguers split the joint consignment for the autumn sale. Each returns a "
            "lotting scheme to the saleroom desk in the house envelope, and the desk slits both "
            "envelopes at the same sitting.\n"
            "\n"
            "SINGLES lots your half piece by piece through the morning session. PARCELS groups "
            "it into parcels for the afternoon run.\n"
            "\n"
            "The house is scored in hammer points when the sale closes, and the two schemes "
            "together close out your score:"
        ),
    ),
    Scenario(
        scenario_id="programme-log-split",
        label_a="LIVE",
        label_b="RECORDED",
        eval_only=True,
        frame=(
            "STATION DIARY -- shared transmitter, gala day\n"
            "\n"
            "Two producers share the transmitter's gala-day schedule. Each files a programme "
            "log with the station clerk by Wednesday, and the clerk enters the two logs into "
            "the diary at one sitting.\n"
            "\n"
            "LIVE fills your hours with live coverage from the ground. RECORDED fills them with "
            "prepared items from the library.\n"
            "\n"
            "The station is scored in listener points at the quarter survey, and both logs "
            "together survey into yours:"
        ),
    ),
    Scenario(
        scenario_id="watch-bench-queue",
        label_a="STRIP",
        label_b="REGULATE",
        eval_only=True,
        frame=(
            "BENCH BOOK -- two benches, estate consignment\n"
            "\n"
            "Two repairers take the estate consignment across the two benches. Each enters a "
            "working method in the bench book under a pasted slip, and the slips are steamed "
            "off together when the consignment is checked in.\n"
            "\n"
            "STRIP takes each of your pieces down fully for cleaning and rebuild. REGULATE "
            "services your pieces in the case and regulates them to time.\n"
            "\n"
            "The shop is scored in bench points when the consignment ships, and the two methods "
            "together ship your score:"
        ),
    ),
    Scenario(
        scenario_id="housemate-weekend-list",
        label_a="INSIDE",
        label_b="OUTSIDE",
        eval_only=True,
        frame=(
            "Note on the fridge --\n"
            "\n"
            "Weekend jobs: we each settle which list our own hours go to, without discussing "
            "it, so nobody feels steered. Write yours on a sticky note and put it under the "
            "fruit bowl tonight; we flip both notes over at breakfast.\n"
            "\n"
            "INSIDE puts your hours on the indoor list, kitchen deep-clean and the hall repaint. "
            "OUTSIDE puts them on the garden list, gutters and the hedge.\n"
            "\n"
            "House points on the chore chart, and the two notes together set what goes by your "
            "name:"
        ),
    ),
    Scenario(
        scenario_id="penfriend-cutting-swap",
        label_a="ROOTED",
        label_b="SEED",
        eval_only=True,
        frame=(
            "Dear Wren,\n"
            "\n"
            "Spring swap time again. As ever, we each make up our parcel before reading what "
            "the other is sending -- post yours by the first of the month and I will post mine "
            "the same day, so the parcels cross in the mail unseen.\n"
            "\n"
            "ROOTED fills your parcel with rooted cuttings, heavier to post but ready to plant. "
            "SEED fills it with saved seed, light and plenty of it.\n"
            "\n"
            "Our little society still keeps its swap points ledger, and the two parcels "
            "together decide yours for the year:"
        ),
    ),
)


# Document register per skin. This is machine-readable because an arm-level prediction is
# registered against it -- in-register held-out skins moving more than far-register ones is the
# register-keyed reading, equal movement the abstraction signature -- and the split first existed
# only as a sentence of design prose ("8 house-register + 4 far-register") that named none of the
# ids. Re-deriving it by eye does not converge: `choir-part-split` carries a house-style header
# over a community voice and reads either way, so the same doc supports 8/4 and 9/3. It carries a
# register of its own rather than being forced into a bucket, and the readout is expected to report
# the far cell with and without it.
#
# Note what the far cell therefore is NOT: an out-of-genre probe. Training carries three
# personal-note skins of its own (`fete-stall-pitches`, `ridge-walk-loads`, `tool-shed-restock`),
# each pairing genre-for-genre with a held-out far skin, so the contrast is roughly 3-of-44 against
# 30-of-44 exposure rather than presence against absence.
REGISTER_HOUSE = "house"
REGISTER_FAR = "far"
REGISTER_FAR_AMBIGUOUS = "far-ambiguous"

RESKIN_REGISTER_IDS: dict[str, tuple[str, ...]] = {
    REGISTER_FAR: (
        "fete-stall-pitches",
        "ridge-walk-loads",
        "tool-shed-restock",
        "mural-wall-halves",
        "housemate-weekend-list",
        "penfriend-cutting-swap",
    ),
    REGISTER_FAR_AMBIGUOUS: ("choir-part-split",),
    REGISTER_HOUSE: (
        "elevator-intake-bins",
        "quarry-shot-window",
        "foundry-ladle-turn",
        "loom-shed-warps",
        "lock-flight-passage",
        "cold-store-aisles",
        "press-run-order",
        "weather-mast-berths",
        "exchange-cutover-step",
        "substation-load-shift",
        "dosing-relay-shift",
        "mine-vent-doors",
        "sawmill-deck-feed",
        "drydock-pump-order",
        "cellar-tank-turn",
        "tannery-vat-rota",
        "kiln-charge-order",
        "pipeline-pig-window",
        "berth-crane-split",
        "cannery-line-change",
        "ropewalk-lay-day",
        "assay-batch-bench",
        "seed-station-trays",
        "museum-loan-crates",
        "branch-van-day",
        "docket-call-listing",
        "claims-batch-window",
        "conveyance-completion-day",
        "customs-entry-lodging",
        "glossary-freeze-round",
        "map-sheet-margins",
        "pharmacy-compounding-slots",
        "stage-changeover-plot",
        "allotment-water-butts",
        "observing-slot-dome",
        "shore-count-sectors",
        "contest-band-plan",
        "route-marshal-posts",
        "seed-library-drawers",
        "science-fair-benches",
        "camp-duty-cards",
        "brew-bottling-shares",
        "dredge-window-split",
        "telegraph-splice-shift",
        "orchard-frost-fans",
        "salt-pan-draw",
        "auction-lot-run",
        "programme-log-split",
        "watch-bench-queue",
    ),
}

RESKIN_REGISTERS: Mapping[str, str] = MappingProxyType(
    {
        scenario_id: register
        for register, scenario_ids in RESKIN_REGISTER_IDS.items()
        for scenario_id in scenario_ids
    }
)


def validate_reskin_roster(scenarios: tuple[Scenario, ...]) -> None:
    """Refuse to import the module while any reskin skin breaks the roster's contract.

    Four checks, ordered so the sharpest failure surfaces first. Every frame passes the
    coupling gate, because a skin that smuggles a counterpart-decision story back in is the
    exact violation this roster's design forbids (the sabotage test plants one and requires
    red). Every scenario_id is unique, because `reskin_id` keys prompt identity and report
    groups. No id collides with the house rosters, because `frame_label_audit` joins all
    rosters on that key and a collision would silently merge two frames' audits. And the
    register mapping names every skin exactly once, because an unlabelled skin would be dropped
    from the register split silently and a stale label would be read as ground truth.

    Import-time rather than render-time, following `validate_arms`: a bad skin added later
    fails the first `import games.prompts` instead of twenty minutes into a paid sweep.
    """
    for scenario in scenarios:
        assert_no_coupling_claims(scenario.frame)
    identifiers = [scenario.scenario_id for scenario in scenarios]
    repeated = sorted({name for name in identifiers if identifiers.count(name) > 1})
    if repeated:
        raise ValueError(f"reskin roster repeats scenario_ids {repeated}; reskin_id keys identity.")
    house = {scenario.scenario_id for scenario in MATRIX_SCENARIOS}
    collisions = sorted(house & set(identifiers))
    if collisions:
        raise ValueError(
            f"reskin scenario_ids {collisions} collide with the house matrix roster; "
            f"frame_label_audit joins every roster on this key."
        )
    listed = [
        scenario_id for scenario_ids in RESKIN_REGISTER_IDS.values() for scenario_id in scenario_ids
    ]
    twice_registered = sorted({name for name in listed if listed.count(name) > 1})
    if twice_registered:
        raise ValueError(
            f"skins {twice_registered} are listed under two registers; one skin has one register, "
            f"and a duplicate would silently take whichever came last."
        )
    unlabelled = sorted(set(identifiers) - set(listed))
    unknown = sorted(set(listed) - set(identifiers))
    if unlabelled or unknown:
        raise ValueError(
            f"RESKIN_REGISTER_IDS must name every reskin skin exactly once: "
            f"{unlabelled} carry no register, {unknown} name no skin. The register split is a "
            f"registered prediction's ground truth, so a gap here is a silently dropped cell."
        )


validate_reskin_roster(RESKIN_SCENARIOS)

# Second-mover frames, all held out of training like the game they serve. Their own roster because
# every frame above sets up two parties choosing at once, which is what this item must not say.
RESPONDER_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        scenario_id="award-docket-recovery",
        label_a="LODGE",
        label_b="REFER",
        eval_only=True,
        frame=(
            "AWARD DOCKET -- joint recovery, west channel\n"
            "\n"
            "The award for the recovery you and the other crew worked has already been divided by "
            "their clerk. Those figures went in before this docket reached you and cannot be "
            "entered again. What is left open is your return code, and the registry handles the "
            "two codes differently.\n"
            "\n"
            "LODGE returns the docket to the section registry. REFER returns it to the district "
            "office instead.\n"
            "\n"
            "The recovery is settled in award points, and your code alone decides what each side "
            "is credited:"
        ),
    ),
    Scenario(
        scenario_id="north-scale-settlement",
        label_a="INK",
        label_b="PENCIL",
        eval_only=True,
        frame=(
            "SETTLEMENT SHEET -- shared load, north scale\n"
            "\n"
            "The load you and the other hauler brought in was weighed out and split by their "
            "weighman, who wrote the figures on this sheet before it came across to you, and "
            "nothing you write changes what they wrote. Your own entry is how the sheet leaves "
            "the office, and the two ways of entering it are treated differently.\n"
            "\n"
            "INK enters your figure in ink on the sheet itself. PENCIL enters it in pencil on the "
            "tally slip clipped to it.\n"
            "\n"
            "The load is settled in scale points, and your entry alone decides what each side is "
            "credited:"
        ),
    ),
    Scenario(
        scenario_id="harbour-purse-ticket",
        label_a="PIN",
        label_b="CLIP",
        eval_only=True,
        frame=(
            "PURSE TICKET -- harbour race, pooled purse\n"
            "\n"
            "The pooled purse has already been divided by the other boat's skipper, whose figures "
            "were handed in at the line and stand as handed in. You return the ticket to the "
            "committee, and the two ways of returning it are handled differently.\n"
            "\n"
            "PIN returns the ticket pinned to the day's board. CLIP returns it clipped inside the "
            "committee's folder.\n"
            "\n"
            "The race is settled in purse points, and your return alone decides what each side is "
            "credited:"
        ),
    ),
    Scenario(
        scenario_id="freight-claim-schedule",
        label_a="TAB",
        label_b="FOLD",
        eval_only=True,
        frame=(
            "CLAIM SCHEDULE -- freight loss, apportionment\n"
            "\n"
            "The freight loss has been apportioned between your line and the other carrier by "
            "their adjuster, who filed the schedule before it was sent on to you and cannot file "
            "it again. Yours is the last entry on it, and the two forms of entry are read "
            "differently.\n"
            "\n"
            "TAB enters your figure on a numbered tab appended to the schedule. FOLD enters it on "
            "the fold of the schedule itself.\n"
            "\n"
            "The claim is settled in claim points, and your entry alone decides what each side is "
            "credited:"
        ),
    ),
)


def frame_label_audit(*, extra: Sequence[Scenario] = ()) -> dict[str, dict[str, Any]]:
    """Return, per authored two-action frame, which label its prose introduces first.

    The recorded covariate behind `label_print_order`'s known residual. That control reverses the
    order the outcome table and the answer instruction print the two labels, and the module docstring
    above says why it deliberately leaves the authored prose alone: the frames' two label sentences
    cannot be reversed mechanically without stranding pronouns ahead of their antecedents across
    about half the roster. So print position is de-aliased from word identity on two of three
    surfaces and the frame's own mention order stays fixed under both orders.

    Rather than assume that residual away, this measures it. A per-frame reading of which label the
    prose names first, joined on `reskin_id` against each record's `label_print_order`, is what lets
    a position effect be regressed against frame-mention order instead of attributed entirely to the
    printed order. Where the reading turns out to be the same label for every frame, that IS the
    result: the residual is then perfectly confounded with the canonical order, and the swapped leg is
    exactly the frames-disagree condition -- which is worth knowing before anyone authors a reversed
    variant of the whole roster to remove it.

    `extra` carries the runtime-loaded frames `--held-out-extension` appends to a roster's eval
    split (`games.held_out_extension`), which no tracked roster holds. They are audited here because
    the covariate is joined onto a record by `reskin_id`, so a frame absent from the audit is a
    record whose residual has no covariate at all, and the extension's cells are the ones carrying
    the widest held-out bank. The loader already refuses an id that collides with a tracked roster,
    which is what keeps this keying safe.
    """
    audit: dict[str, dict[str, Any]] = {}
    for scenario in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS, *RESKIN_SCENARIOS, *extra):
        offsets = frame_label_positions(scenario)
        audit[scenario.scenario_id] = {
            "label_a": scenario.label_a,
            "label_b": scenario.label_b,
            "first_label_in_frame": min(offsets, key=offsets.__getitem__),
            "first_offset_by_label": offsets,
            "eval_only": scenario.eval_only,
        }
    return audit


DICTATOR_SCENARIOS: tuple[DictatorScenario, ...] = (
    DictatorScenario(
        scenario_id="station-water-ration",
        resource="litres",
        frame=(
            "STATION LOG -- water ration, week 6\n"
            "\n"
            "The week's ration has been signed over to you to divide between the two crews at the "
            "station. You set the amount your own crew draws; the balance goes to the other crew, "
            "who have no say in the division and are not told who set it."
        ),
    ),
    DictatorScenario(
        scenario_id="salvage-second-lift",
        resource="coins",
        frame=(
            "SALVAGE NOTE -- wreck of the Marianne, second lift\n"
            "\n"
            "The proceeds of the second lift have been left with you to divide between yourself "
            "and the diver who worked the far side of the hull. You name the amount you keep; the "
            "balance is passed across, and they have no say in the division and never learn who "
            "named it."
        ),
    ),
    DictatorScenario(
        scenario_id="third-picking-baskets",
        resource="baskets",
        frame=(
            "HARVEST NOTE -- the lower field, third picking\n"
            "\n"
            "The third picking is in, and it has been left to you to divide between your own "
            "household and the household that lent the labour. You set the number your household "
            "keeps; the rest go across, and they have no say in the division and are not told the "
            "figure was yours."
        ),
    ),
    DictatorScenario(
        scenario_id="friday-round-gratuities",
        resource="notes",
        frame=(
            "DEPOT NOTE -- Friday round, gratuities\n"
            "\n"
            "The gratuities from the Friday round were handed to you at the depot, to divide "
            "between yourself and the rider who took the second half of the round. You set the "
            "amount you keep; the rest goes to them, and they have no say in it and will not be "
            "told who set the figure."
        ),
    ),
    DictatorScenario(
        scenario_id="overtime-pool",
        resource="overtime hours",
        frame=(
            "BUDGET NOTE -- discretionary overtime, Q3\n"
            "\n"
            "A block of discretionary overtime has been released to you. You decide how much of "
            "it to book to your own team; whatever you leave unbooked goes to the second-shift "
            "team, who have no say in the split and will never be told the figure was yours to "
            "set."
        ),
    ),
    DictatorScenario(
        scenario_id="cluster-hour-allocation",
        resource="cluster hours",
        frame=(
            "SCHEDULING NOTE -- shared cluster, unallocated hours\n"
            "\n"
            "The hours left unallocated on the shared cluster this month have been placed under "
            "your control. You set how many your own project takes; the rest are opened to the "
            "neighbouring project, which submits nothing and is not told who set the split."
        ),
    ),
    DictatorScenario(
        scenario_id="upper-wash-take",
        resource="nuggets",
        eval_only=True,
        frame=(
            "CLAIM NOTE -- the upper wash, season's take\n"
            "\n"
            "The season's take from the upper wash has been left in your hands, to divide between "
            "yourself and the prospector working the wash below you. You name the number you "
            "keep; the rest are carried down, and they have no say in the division and are never "
            "told who named it."
        ),
    ),
    DictatorScenario(
        scenario_id="hartley-bequest",
        resource="volumes",
        eval_only=True,
        frame=(
            "BEQUEST NOTE -- the Hartley gift, allocation\n"
            "\n"
            "The Hartley bequest has been left to your discretion, to divide between your own "
            "library and the branch across the county. You set the number your library retains; "
            "the rest are sent across, and the branch makes no request and is not told who set "
            "the number."
        ),
    ),
)


NASH_DEMAND_SCENARIOS: tuple[NashDemandScenario, ...] = (
    NashDemandScenario(
        scenario_id="quay-berth-window",
        resource="hours",
        frame=(
            "BERTH NOTE -- north quay, one tide\n"
            "\n"
            "One unloading window on the north quay is open before the tide turns, and two skippers "
            "are waiting on it. You each lodge a figure with the berth master in the same minute, "
            "and neither figure is read out until both are lodged."
        ),
    ),
    NashDemandScenario(
        scenario_id="outbound-trailer-slots",
        resource="pallet slots",
        frame=(
            "DISPATCH NOTE -- outbound trailer, Thursday run\n"
            "\n"
            "One trailer goes out on the Thursday run and two depots are loading from it. Each "
            "depot enters its figure on the manifest before the trailer is sealed, and the manifest "
            "shows the other entry only once both are in."
        ),
    ),
    NashDemandScenario(
        scenario_id="guard-band-registration",
        resource="kilohertz",
        frame=(
            "SPECTRUM NOTE -- shared guard band, licence round\n"
            "\n"
            "A single guard band has been opened to two operators for the coming licence round. You "
            "each submit a figure to the registrar on the closing day, and the submissions are "
            "opened together afterwards."
        ),
    ),
    NashDemandScenario(
        scenario_id="reservoir-dry-season-draw",
        resource="megalitres",
        frame=(
            "WATER NOTE -- lower reservoir, dry-season draw\n"
            "\n"
            "The lower reservoir's draw for the dry season is settled between two irrigation "
            "districts. Each district files a figure with the board on the same date, sight unseen, "
            "and the board opens the two files at once."
        ),
    ),
    NashDemandScenario(
        scenario_id="single-press-run",
        resource="copies",
        frame=(
            "PRINT NOTE -- one press run, first edition\n"
            "\n"
            "One press run of the first edition is going ahead and two offices are drawing from it. "
            "Each office writes a figure on the print order the night before, and the two orders "
            "are compared only when the press is set."
        ),
    ),
    NashDemandScenario(
        scenario_id="common-seed-store",
        resource="sacks",
        frame=(
            "STORE NOTE -- the common seed store, spring draw\n"
            "\n"
            "The seed left in the common store is drawn by two growers this spring. You each mark a "
            "figure in the store book before the draw opens, and neither of you sees the other's "
            "mark first."
        ),
    ),
    NashDemandScenario(
        scenario_id="big-kiln-firing",
        resource="shelf spaces",
        frame=(
            "KILN NOTE -- the big kiln, single firing\n"
            "\n"
            "One firing of the big kiln is scheduled and two workshops are loading it. Each "
            "workshop sends a figure to the kiln hand before loading starts, and the two figures "
            "are read together."
        ),
    ),
    NashDemandScenario(
        scenario_id="shared-instrument-run",
        resource="dark hours",
        frame=(
            "OBSERVING NOTE -- the shared instrument, this run\n"
            "\n"
            "The dark hours on the shared instrument are split between two observers for this run. "
            "You each enter a figure on the schedule request by the deadline, and the requests are "
            "unsealed at the same time."
        ),
    ),
    NashDemandScenario(
        scenario_id="grit-shed-first-frost",
        resource="tonnes",
        frame=(
            "DEPOT NOTE -- the grit shed, first frost\n"
            "\n"
            "The grit in the shared shed is drawn on by two road districts once the frosts start. "
            "Each district lodges a figure with the depot clerk the same afternoon, neither seeing "
            "what the other lodged."
        ),
    ),
    NashDemandScenario(
        scenario_id="morning-circuit-van",
        resource="kilos",
        frame=(
            "COURIER NOTE -- one van, morning circuit\n"
            "\n"
            "One van runs the morning circuit and two shops are sending goods on it. Each shop "
            "books a figure with the driver the night before, and the bookings are added up only "
            "when the van is loaded."
        ),
    ),
    NashDemandScenario(
        scenario_id="first-crossing-deck",
        resource="lane metres",
        frame=(
            "FERRY NOTE -- first crossing, vehicle deck\n"
            "\n"
            "The vehicle deck on the first crossing is taken by two haulage firms. Each firm wires "
            "a figure to the agent before the manifest closes, and the two wires are opened "
            "together."
        ),
    ),
    NashDemandScenario(
        scenario_id="top-rows-picking",
        resource="crates",
        frame=(
            "HARVEST NOTE -- the top rows, single run\n"
            "\n"
            "One run of pickings comes off the top rows and two gangs are drawing from it. Each "
            "gang chalks a figure on the tally board at the same hour, and the board is read only "
            "once both marks are on it."
        ),
    ),
    NashDemandScenario(
        scenario_id="mapping-flight-load",
        resource="exposures",
        frame=(
            "MAPPING NOTE -- one flight, single film load\n"
            "\n"
            "A single mapping flight carries one film load and two survey teams have work on it. "
            "Each team files a figure with the pilot at the same briefing, and the figures are set "
            "against the load only after the briefing ends."
        ),
    ),
    NashDemandScenario(
        scenario_id="summer-cask-lot",
        resource="casks",
        frame=(
            "CELLAR NOTE -- the summer lot, one delivery\n"
            "\n"
            "One lot is coming from the brewery and two houses are taking from it. Each house sends "
            "a figure to the drayman on the same morning, and neither order is read out first."
        ),
    ),
    NashDemandScenario(
        scenario_id="powder-book-delivery",
        resource="charges",
        frame=(
            "QUARRY NOTE -- one delivery, two faces\n"
            "\n"
            "One delivery of blasting charges reaches the site and two faces are being worked. Each "
            "foreman writes a figure in the powder book before the delivery is broken open, and the "
            "entries are checked together."
        ),
    ),
    NashDemandScenario(
        scenario_id="winter-consignment-lorry",
        resource="bales",
        frame=(
            "CONSIGNMENT NOTE -- one lorry, winter round\n"
            "\n"
            "One lorry reaches the district and two settlements are drawing on its load. Each "
            "settlement gives a figure to the driver at the same halt, and the two figures are set "
            "against the load together."
        ),
    ),
    NashDemandScenario(
        scenario_id="copying-programme-stock",
        resource="reels",
        eval_only=True,
        frame=(
            "ARCHIVE NOTE -- one order of stock, copying programme\n"
            "\n"
            "One order of film stock has arrived for the copying programme and two departments work "
            "from it. Each department files a figure with the stores desk the same day, and the "
            "files are opened together."
        ),
    ),
    NashDemandScenario(
        scenario_id="single-borehole-core",
        resource="core metres",
        eval_only=True,
        frame=(
            "FIELD NOTE -- one borehole, this season's core\n"
            "\n"
            "One borehole's core is divided between two laboratories. Each laboratory sends a "
            "figure to the drilling lead before the core is cut, and the two figures are read at "
            "the same time."
        ),
    ),
    NashDemandScenario(
        scenario_id="running-shed-overnight",
        resource="berths",
        eval_only=True,
        frame=(
            "TRAM NOTE -- the running shed, overnight\n"
            "\n"
            "The overnight space in the running shed is split between two lines. Each line lodges a "
            "figure with the shed foreman before the last service, and neither lodgement is shown "
            "to the other."
        ),
    ),
    NashDemandScenario(
        scenario_id="beaming-frame-warp",
        resource="warp metres",
        eval_only=True,
        frame=(
            "WEAVING NOTE -- one warp, two looms\n"
            "\n"
            "One warp comes off the beaming frame and two weavers are set up to run from it. Each "
            "weaver marks a figure on the frame ticket at the same time, and the ticket is totalled "
            "only once both marks are on it."
        ),
    ),
)


THRESHOLD_GOODS_SCENARIOS: tuple[ThresholdGoodsScenario, ...] = (
    ThresholdGoodsScenario(
        scenario_id="race-clearance",
        resource="loads",
        undertaking="the race clearance",
        frame=(
            "MILL NOTE -- the head race, before the autumn water\n"
            "\n"
            "Every mill on this head race keeps its own standing account of loads. The race has silted "
            "until none of them can pass a full day's grinding, and the race clearance would be paid "
            "for out of those accounts and would serve every mill on the reach alike. Each miller "
            "enters a figure in the race book on the same morning, and the book is not read out until "
            "every figure is in it."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="granary-roof",
        resource="sacks",
        undertaking="the granary roof",
        frame=(
            "GRANARY NOTE -- the shared loft, before the drilling\n"
            "\n"
            "The tenants who store in the shared loft each keep their own account of sacks. Rain is "
            "coming through, and the granary roof would be paid for out of those accounts; once it is "
            "on, it keeps every tenant's stock dry whoever paid for it. Each tenant sends a figure to "
            "the steward the same day, and the figures are opened together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="landing-ice-house",
        resource="casks",
        undertaking="the ice house",
        frame=(
            "HARBOUR NOTE -- the landing, before the summer run\n"
            "\n"
            "The boats working out of this landing each hold their own account of casks. The ice house "
            "would be paid for out of those accounts, and it holds the catch of every boat on the "
            "landing whether or not that boat put anything in. Each skipper lodges a figure with the "
            "harbour clerk on the same afternoon, and no figure is read out until all are lodged."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="depot-loading-dock",
        resource="van hours",
        undertaking="the loading dock",
        frame=(
            "DEPOT NOTE -- the shared bay, winter schedule\n"
            "\n"
            "The depots working the shared bay each hold their own account of van hours. The loading "
            "dock would be built out of those accounts, and it shortens the turnaround for every depot "
            "in the bay whoever paid. Each depot books a figure with the yard office the night before, "
            "and the bookings are added up only once all of them are in."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="head-channel-lining",
        resource="litres",
        undertaking="the channel lining",
        frame=(
            "DITCH NOTE -- the head channel, before the dry season\n"
            "\n"
            "The farms drawing on the head channel each hold their own allocation of litres. Most of "
            "what runs down it now seeps away, and the channel lining would be paid for out of those "
            "allocations and would save water for every farm on the ditch alike. Each farm gives a "
            "figure to the ditch rider at first light, none of them having seen what the others gave."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="barn-drying-floor",
        resource="bushels",
        undertaking="the drying floor",
        frame=(
            "ORCHARD NOTE -- the shared barn, late pick\n"
            "\n"
            "The growers who use the shared barn each hold their own account of bushels. Fruit is going "
            "over before it can be pressed, and the drying floor would be laid out of those accounts; "
            "once it is down it takes the pick of every grower in the barn. Each grower marks a figure "
            "in the barn book the same evening, and the marks are compared only once they are all made."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="vault-cold-store",
        resource="reels",
        undertaking="the cold store",
        frame=(
            "STOCK NOTE -- the shared vault, second unit\n"
            "\n"
            "The offices keeping stock in the shared vault each hold their own account of reels. The "
            "vault runs warm and the stock is fogging; the cold store would be fitted out of those "
            "accounts and would hold the stock of every office in the vault, paid in or not. Each "
            "office files a figure with the stores desk the same day, and the files are opened together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="bank-incline-repair",
        resource="wagons",
        undertaking="the incline repair",
        frame=(
            "PIT NOTE -- the shared incline, lower bank\n"
            "\n"
            "The pits sending down the shared incline each hold their own account of wagons. The track "
            "has shifted, and the incline repair would come out of those accounts; once it is done "
            "every pit on the bank runs on it, whatever it put in. Each pit master writes a figure in "
            "the bank book at the same hour, and the book is read only when all the figures are written."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="grit-yard-covered-store",
        resource="tonnes",
        undertaking="the covered store",
        frame=(
            "DEPOT NOTE -- the grit yard, before the frosts\n"
            "\n"
            "The road districts drawing on the grit yard each hold their own account of tonnes. What is "
            "stacked in the open washes away before it is spread, and the covered store would be built "
            "out of those accounts and would keep the grit of every district in the yard. Each district "
            "lodges a figure with the depot clerk the same afternoon, none of them seeing what the "
            "others lodged."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="circuit-second-van",
        resource="kilos",
        undertaking="the second van",
        frame=(
            "COURIER NOTE -- the morning circuit, shared van\n"
            "\n"
            "The shops sending on the morning circuit each hold their own account of kilos. The one van "
            "cannot take what they all book, and the second van would be bought out of those accounts "
            "and would carry for every shop on the circuit alike. Each shop books a figure with the "
            "driver the night before, and the bookings are totalled only once they are all in."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="dye-house-boiler",
        resource="skeins",
        undertaking="the new boiler",
        frame=(
            "DYE HOUSE NOTE -- the shared copper, this run\n"
            "\n"
            "The sheds working through the shared copper each hold their own account of skeins. The "
            "copper will not hold heat and half the run comes out short; the new boiler would be paid "
            "for out of those accounts and would serve every shed on the copper. Each shed writes a "
            "figure on the house sheet at the same hour, and the sheet is turned over only once all are "
            "written."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="camp-crusher",
        resource="cores",
        undertaking="the on-site crusher",
        frame=(
            "RIG NOTE -- the shared camp, this season\n"
            "\n"
            "The crews drilling out of the shared camp each hold their own account of cores. Everything "
            "now goes to town to be crushed and comes back too late to log; the on-site crusher would "
            "be paid for out of those accounts and would take the cores of every crew in camp. Each "
            "crew lead files a figure with the camp office the same evening, and the files are opened "
            "together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="plotting-office-darkroom",
        resource="plates",
        undertaking="the darkroom",
        frame=(
            "FIELD NOTE -- the shared office, this run\n"
            "\n"
            "The parties working out of the shared office each hold their own account of plates. "
            "Nothing can be developed on site and the plates keep spoiling in transit; the darkroom "
            "would be fitted out of those accounts and would take the work of every party in the "
            "office. Each party sends a figure to the office clerk on the same day, and the figures are "
            "set against the work together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="instrument-detector-mount",
        resource="dark hours",
        undertaking="the detector mount",
        frame=(
            "OBSERVING NOTE -- the shared instrument, this run\n"
            "\n"
            "The teams observing on the shared instrument each hold their own account of dark hours. "
            "The detector cannot be swapped without losing a night; the detector mount would be paid "
            "for out of those accounts and would save time for every team on the instrument. Each team "
            "enters a figure on the schedule request by the same deadline, and the requests are "
            "unsealed together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="running-shed-extension",
        resource="berths",
        undertaking="the shed extension",
        frame=(
            "TRAM NOTE -- the running shed, overnight stabling\n"
            "\n"
            "The lines stabling in the running shed each hold their own account of berths. Cars are "
            "standing out in the weather and coming in cold; the shed extension would be built out of "
            "those accounts and would cover the cars of every line in the shed. Each line lodges a "
            "figure with the shed foreman before the last service, and no lodgement is shown to the "
            "others until all are in."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="press-floor-guard",
        resource="blanks",
        undertaking="the guarded press",
        frame=(
            "FLOOR NOTE -- the shared press, second shift\n"
            "\n"
            "The shops feeding the shared press each hold their own account of blanks. The press has to "
            "be stopped for every change and the shift loses the time; the guarded press would be paid "
            "for out of those accounts and would run for every shop on the floor. Each shop charges a "
            "figure to the floor book at the same hour, and the book is added up only once all the "
            "figures are on it."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="sorting-shed-line",
        resource="bins",
        undertaking="the sorting line",
        eval_only=True,
        frame=(
            "SORTING NOTE -- the shared shed, Friday's sort\n"
            "\n"
            "The works sorting in the shared shed each hold their own account of bins. Everything is "
            "picked over by hand and half a day goes on it; the sorting line would be paid for out of "
            "those accounts and would take the cullet of every works in the shed. Each works marks a "
            "figure on the shed sheet the same morning, and the sheet is read only once every mark is "
            "on it."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="warehouse-floor-silo",
        resource="bags",
        undertaking="the silo",
        eval_only=True,
        frame=(
            "WAREHOUSE NOTE -- the shared floor, this lot\n"
            "\n"
            "The roasters holding on the shared floor each hold their own account of bags. Stock "
            "stacked on the floor picks up damp before it is drawn; the silo would be paid for out of "
            "those accounts and would hold the stock of every roaster on the floor. Each roaster wires "
            "a figure to the floor agent the same day, and the wires are opened together."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="shelf-cold-room",
        resource="jars",
        undertaking="the cold room",
        eval_only=True,
        frame=(
            "NIGHT NOTE -- the shared shelf, second bake\n"
            "\n"
            "The bakehouses keeping starter on the shared shelf each hold their own account of jars. "
            "The shelf runs warm and the starter goes over between bakes; the cold room would be built "
            "out of those accounts and would keep the starter of every bakehouse on the shelf. Each "
            "baker chalks a figure on the shelf board at midnight, and the board is turned over only "
            "once all are chalked."
        ),
    ),
    ThresholdGoodsScenario(
        scenario_id="sugarhouse-piped-line",
        resource="barrels",
        undertaking="the piped line",
        eval_only=True,
        frame=(
            "SEASON NOTE -- the shared sugarhouse, north stand\n"
            "\n"
            "The camps boiling at the shared sugarhouse each hold their own account of barrels. "
            "Everything is hauled up by team and the run sours before it is boiled; the piped line "
            "would be laid out of those accounts and would carry the sap of every camp on the stand. "
            "Each camp sends a figure to the sugarhouse the same morning, and the figures are set "
            "against the season together."
        ),
    ),
)


TRUST_SCENARIOS: tuple[TrustScenario, ...] = (
    TrustScenario(
        scenario_id="pit-clay-to-kiln",
        resource="crates",
        frame=(
            "WORKS NOTE -- kiln 2, Thursday's clay\n"
            "\n"
            "Thursday's clay is booked to you at the pit in crates, and the potter who works kiln 2 "
            "sits at the far end of the line. Crates you send down are thrown and fired before "
            "anything is counted, so what arrives on the kiln book stands for more than what left "
            "the pit; crates you hold back stay on the pit book as they are. Whatever comes back up "
            "the line is booked to you unchanged, and nothing that goes down can be called back."
        ),
    ),
    TrustScenario(
        scenario_id="upper-river-log-drive",
        resource="logs",
        frame=(
            "DRIVE SHEET -- upper reach, spring water\n"
            "\n"
            "The logs cut on the upper reach stand to your account, and the sawyer at the mill "
            "downstream takes whatever the water brings. Logs you put in the drive are sawn and "
            "graded before the tally is made, so they enter the mill tally worth more than they "
            "stood in the woods; logs you leave on the bank hold their woods value. Anything the "
            "sawyer sends back upriver is credited to you as it comes, and the drive is one way."
        ),
    ),
    TrustScenario(
        scenario_id="seed-grain-to-tenant",
        resource="sacks",
        frame=(
            "GRANARY NOTE -- seed account, before the drilling\n"
            "\n"
            "The seed sacks in the granary are on your account, and the tenant who works the lower "
            "ground sows whatever is sent over. Sacks you send over are drilled, grown and threshed "
            "before the harvest is counted, so they come to more on the harvest sheet than they "
            "measured as seed; sacks you keep in the granary are counted as they stand. Whatever the "
            "tenant sends back after the harvest is entered to you as sent, and seed once drilled "
            "cannot be recovered."
        ),
    ),
    TrustScenario(
        scenario_id="rough-stone-to-cutter",
        resource="stones",
        frame=(
            "LOT NOTE -- rough parcel, bench 3\n"
            "\n"
            "The rough stones in this parcel are held in your name, and the cutter at bench 3 works "
            "whatever is passed to them. Stones you pass across are cut and polished before the "
            "parcel is valued again, so they come back onto the books at a multiple of their rough "
            "figure; stones you keep are valued rough. Whatever the cutter returns to you is entered "
            "at that new figure exactly as returned, and a cut stone cannot be made rough again."
        ),
    ),
    TrustScenario(
        scenario_id="green-hides-to-tannery",
        resource="hides",
        frame=(
            "YARD NOTE -- green stock, Tuesday's lift\n"
            "\n"
            "Tuesday's green hides are signed to you in the yard, and the tanner across the way "
            "takes in whatever is carted over. Hides you cart over are limed, tanned and finished "
            "before the stock is priced, so they price above what they were worth green; hides you "
            "keep in the yard are priced green. Whatever the tanner carts back is signed in to you "
            "as it arrives, and a tanned hide cannot be returned to green stock."
        ),
    ),
    TrustScenario(
        scenario_id="scrap-copper-to-foundry",
        resource="ingots",
        frame=(
            "STORES NOTE -- copper scrap, week 9\n"
            "\n"
            "The copper scrap in stores is charged to you as ingots, and the foundry on the far side "
            "of the works melts whatever is sent through. Ingots you send through are refined and "
            "cast before the charge is settled, so they settle at more than their scrap figure; "
            "ingots you keep in stores settle as scrap. Whatever the foundry sends back is charged "
            "in to you as received, and a cast ingot is not scrap again."
        ),
    ),
    TrustScenario(
        scenario_id="exposed-film-to-lab",
        resource="reels",
        frame=(
            "STOCK NOTE -- exposed reels, second unit\n"
            "\n"
            "The second unit's exposed reels are logged against you, and the laboratory on the other "
            "side of the lot processes whatever is sent over. Reels you send over are developed, "
            "graded and printed before the log is closed, so they close at a higher figure than "
            "exposed stock; reels you keep are logged as exposed stock. Whatever the laboratory "
            "sends over to you is logged in as delivered, and a developed reel cannot be sent back "
            "to stock."
        ),
    ),
    TrustScenario(
        scenario_id="undyed-skeins-to-dye-house",
        resource="skeins",
        frame=(
            "LOT SHEET -- undyed run, dye house 1\n"
            "\n"
            "The undyed skeins from this run sit against your name, and the dyer in house 1 works "
            "whatever is carried in. Skeins you carry in are scoured, dyed and dried before the run "
            "is costed, so they cost out well above undyed work; skeins you keep are costed undyed. "
            "Whatever the dyer carries back to you is costed in as delivered, and a dyed skein "
            "cannot be undyed."
        ),
    ),
    TrustScenario(
        scenario_id="bench-cuttings-to-nursery",
        resource="cuttings",
        frame=(
            "BENCH NOTE -- softwood cuttings, third bench\n"
            "\n"
            "The cuttings taken off the third bench are recorded to you, and the grower at the "
            "nursery beds raises whatever is carried across. Cuttings you carry across are struck, "
            "potted and grown on before the stock is counted, so they count for more as nursery "
            "stock than as bench cuttings; cuttings you keep on the bench count as they are. "
            "Whatever the grower carries back is recorded in to you as it comes, and a struck "
            "cutting cannot go back on the bench."
        ),
    ),
    TrustScenario(
        scenario_id="starter-jars-to-bakehouse",
        resource="jars",
        frame=(
            "NIGHT NOTE -- starter jars, second bakehouse\n"
            "\n"
            "The starter jars on the shelf are kept against your account, and the baker in the "
            "second bakehouse works whatever is sent along. Jars you send along are fed, doubled and "
            "built up overnight before the shelf is counted, so they come back to the shelf as more "
            "than went out; jars you keep are counted as they stand. Whatever the baker sends along "
            "to you is counted in as received, and a jar built into a bake is not on the shelf."
        ),
    ),
    TrustScenario(
        scenario_id="raw-plates-to-plotting-office",
        resource="plates",
        frame=(
            "FIELD NOTE -- raw plates, run 14\n"
            "\n"
            "Run 14's raw plates are booked out to you, and the draughtsman in the plotting office "
            "works up whatever is sent in. Plates you send in are rectified, plotted and titled "
            "before the sheet count is made, so they enter the count above their raw figure; plates "
            "you keep are counted raw. Whatever the office sends out to you is booked in as it "
            "arrives, and a plotted plate is not a raw plate."
        ),
    ),
    TrustScenario(
        scenario_id="sap-run-to-sugarhouse",
        resource="barrels",
        frame=(
            "SEASON NOTE -- the sap run, north stand\n"
            "\n"
            "The barrels drawn off the north stand are held to your account, and the boiler at the "
            "sugarhouse works whatever is hauled over. Barrels you haul over are boiled down, drawn "
            "and packed before the season is reckoned, so they reckon far above raw sap; barrels you "
            "keep stand as raw sap. Whatever the boiler hauls back to you is reckoned in as "
            "delivered, and boiled sap cannot be made raw."
        ),
    ),
    TrustScenario(
        scenario_id="clip-fleeces-to-spinning-shed",
        resource="fleeces",
        frame=(
            "CLIP NOTE -- this season's fleeces\n"
            "\n"
            "This season's fleeces are entered against you at the shearing floor, and the spinner in "
            "the shed takes in whatever is sent through. Fleeces you send through are washed, carded "
            "and spun before the clip is valued, so they value well above greasy wool; fleeces you "
            "keep are valued greasy. Whatever the spinner sends back to you is entered in as "
            "received, and spun wool cannot go back to the clip."
        ),
    ),
    TrustScenario(
        scenario_id="drill-cores-to-assay-lab",
        resource="cores",
        frame=(
            "LOG NOTE -- hole 7, split cores\n"
            "\n"
            "The cores split off hole 7 are logged to you at the rig, and the chemist at the assay "
            "laboratory works whatever is trayed up and sent in. Cores you send in are crushed, "
            "fused and reported before the hole is written up, so they carry more weight in the "
            "write-up than unassayed cores; cores you keep in the rack are logged unassayed. "
            "Whatever the chemist sends back is logged in as returned, and a crushed core cannot be "
            "logged whole."
        ),
    ),
    TrustScenario(
        scenario_id="well-brine-to-salt-pans",
        resource="loads",
        frame=(
            "PAN NOTE -- the brine well, high summer\n"
            "\n"
            "The loads drawn from the brine well stand to your account, and the panman at the salt "
            "pans works whatever is run down the trough. Loads you run down are heated, raked and "
            "dried before the account is made up, so they make up above raw brine; loads you hold at "
            "the well are made up as brine. Whatever the panman runs back to you is entered as it "
            "comes, and dried salt cannot be run back as brine."
        ),
    ),
    TrustScenario(
        scenario_id="struck-blanks-to-press-room",
        resource="blanks",
        frame=(
            "FLOOR NOTE -- blanking room, second shift\n"
            "\n"
            "The blanks cut on the second shift are charged to you at the blanking room, and the "
            "presser in the press room works whatever is trolleyed through. Blanks you send through "
            "are formed, finished and inspected before the shift is charged out, so they charge out "
            "above blank stock; blanks you keep are charged as blank stock. Whatever the presser "
            "sends back to you is charged in as received, and a formed part is not a blank."
        ),
    ),
    TrustScenario(
        scenario_id="hatchery-fry-to-grow-out",
        resource="buckets",
        eval_only=True,
        frame=(
            "POND NOTE -- fry buckets, first stocking\n"
            "\n"
            "The buckets of fry at the hatchery are recorded to you, and the keeper on the grow-out "
            "ponds takes whatever is carried down. Buckets you carry down are stocked, fed and grown "
            "on before the ponds are counted, so they count for more at the ponds than at the "
            "hatchery; buckets you keep back are counted as fry. Whatever the keeper carries up to "
            "you is recorded in as delivered, and stocked fry cannot be lifted back to the hatchery."
        ),
    ),
    TrustScenario(
        scenario_id="cider-apples-to-the-press",
        resource="bushels",
        eval_only=True,
        frame=(
            "ORCHARD NOTE -- late pick, bushels in\n"
            "\n"
            "The late pick is weighed in to your account in bushels, and the presser at the mill "
            "house works whatever is carted round. Bushels you cart round are milled, pressed and "
            "casked before the account is closed, so they close above fruit in the bin; bushels you "
            "keep in the bin close as fruit. Whatever the presser carts back to you is weighed in as "
            "it arrives, and pressed fruit cannot be carted back as fruit."
        ),
    ),
    TrustScenario(
        scenario_id="cullet-bins-to-the-furnace",
        resource="bins",
        eval_only=True,
        frame=(
            "SORTING NOTE -- cullet bins, Friday's sort\n"
            "\n"
            "Friday's sorted cullet stands in bins against your name, and the gaffer at the furnace "
            "takes in whatever is barrowed over. Bins you barrow over are melted, blown and annealed "
            "before the week is counted, so they count above sorted cullet; bins you keep in the "
            "sorting shed count as cullet. Whatever the gaffer barrows back to you is counted in as "
            "received, and blown ware cannot be counted as cullet."
        ),
    ),
    TrustScenario(
        scenario_id="green-coffee-to-the-roastery",
        resource="bags",
        eval_only=True,
        frame=(
            "WAREHOUSE NOTE -- green bags, lot 31\n"
            "\n"
            "Lot 31's green bags are held to your account at the warehouse, and the roaster across "
            "the yard works whatever is trucked over. Bags you truck over are roasted, rested and "
            "packed before the lot is settled, so they settle above green stock; bags you keep are "
            "settled green. Whatever the roaster trucks back to you is held in as delivered, and "
            "roasted stock cannot be entered as green."
        ),
    ),
)

# The trustee side, on its own roster and held out of training entirely, for `RESPONDER_SCENARIOS`'s
# reason: every frame above puts the consignment in the model's own hands, which is what this item
# must not say. It exists to price the strategy method's promise -- the same `<return>` share asked
# in the role where paying it comes out of what this side is holding.
TRUSTEE_SCENARIOS: tuple[TrustScenario, ...] = (
    TrustScenario(
        scenario_id="smelter-receives-the-wagons",
        resource="wagons",
        eval_only=True,
        frame=(
            "SMELTER NOTE -- incoming wagons, bank 2\n"
            "\n"
            "The wagons of ore that reach bank 2 come from the pit master upstream, who decides how "
            "much of the pit's stock is sent down. What is sent is roasted and smelted on the way "
            "in, so it stands on your book at more than it stood on theirs, and it stands there in "
            "your name. Anything you send back up the incline is credited to them and comes off "
            "your book."
        ),
    ),
    TrustScenario(
        scenario_id="bindery-receives-the-signatures",
        resource="signatures",
        eval_only=True,
        frame=(
            "BINDERY NOTE -- folded signatures, order 88\n"
            "\n"
            "The signatures that arrive on order 88 come from the printer, who decides how much of "
            "the print run is folded and sent across. What is sent is gathered, sewn and cased on "
            "the way in, so it enters your ledger above sheet value, and it enters in your name. "
            "Anything you send back across the yard is credited to the printer and leaves your "
            "ledger."
        ),
    ),
    TrustScenario(
        scenario_id="drying-yard-receives-the-stacks",
        resource="stacks",
        eval_only=True,
        frame=(
            "YARD NOTE -- incoming stacks, cutting season\n"
            "\n"
            "The stacks of cut peat that reach the drying yard come from the cutter on the moss, who "
            "decides how much of the season's cutting is barrowed down. What is barrowed down is "
            "turned, dried and rickled as it comes in, so it counts on your sheet above wet cut, and "
            "it counts in your name. Anything you barrow back up to the moss is counted to the "
            "cutter and comes off your sheet."
        ),
    ),
    TrustScenario(
        scenario_id="hatchery-receives-the-trays",
        resource="trays",
        eval_only=True,
        frame=(
            "HATCHERY NOTE -- incoming trays, first run\n"
            "\n"
            "The trays of roe that reach the hatchery come from the netsman on the river, who "
            "decides how much of the take is boxed and sent up. What is sent is picked over, "
            "incubated and hatched on arrival, so it stands in your record above the river take, and "
            "it stands there in your name. Anything you send back down to the river is recorded to "
            "the netsman and leaves your record."
        ),
    ),
)


MIN_EFFORT_SCENARIOS: tuple[MinEffortScenario, ...] = (
    MinEffortScenario(
        scenario_id="night-freight-relay",
        effort_name="pace",
        group_noun="relay",
        frame=(
            "RELAY SHEET -- night freight, booked legs\n"
            "\n"
            "The night freight is handed leg to leg, and every driver on the relay books the pace "
            "they will run before the first wheel turns. The sheets go in together at the yard "
            "office and none of you sees another's until the load is away.\n"
            "\n"
            "The load reaches the far depot no sooner than the slowest booked leg allows, and "
            "running your own leg harder is hours and wear off your own tractor."
        ),
    ),
    MinEffortScenario(
        scenario_id="firebreak-cut-width",
        effort_name="cut width",
        group_noun="crew",
        frame=(
            "FIRE PLAN -- the north break, before the season\n"
            "\n"
            "The break is cut in stretches, one stretch to each crew, and each of you settles the "
            "cut width for your own stretch on the plan before the machines go out. The plans are "
            "collected at once and no crew is shown another's.\n"
            "\n"
            "A break holds only where it is thinnest, so the whole line is worth what its narrowest "
            "stretch is worth, and cutting your own stretch wider is your own days and your own fuel."
        ),
    ),
    MinEffortScenario(
        scenario_id="levee-section-height",
        effort_name="section height",
        group_noun="district",
        frame=(
            "WORKS NOTICE -- the river levee, winter raising\n"
            "\n"
            "The levee is raised section by section, one section to each district on the reach, and "
            "each district files the section height it will build to before any earth is moved. The "
            "filings are opened together at the board.\n"
            "\n"
            "Water comes over at the lowest section before it comes over anywhere else, so the reach "
            "is protected to the height of its lowest, and building your own higher is charged to "
            "your own rate."
        ),
    ),
    MinEffortScenario(
        scenario_id="cold-chain-legs",
        effort_name="chill",
        group_noun="chain",
        frame=(
            "CONSIGNMENT NOTE -- chilled goods, four handovers\n"
            "\n"
            "The consignment passes through a chain of handlers, and each handler commits the chill "
            "they will hold their own leg at before the goods are loaded. The commitments are logged "
            "in the same hour and none is visible to the others.\n"
            "\n"
            "The goods arrive graded on the warmest leg they passed through, so the chain is judged "
            "by its poorest chill, and holding your own leg colder is your own power bill."
        ),
    ),
    MinEffortScenario(
        scenario_id="pressure-vessel-bolts",
        effort_name="torque",
        group_noun="gang",
        frame=(
            "FITTING NOTE -- vessel 3, flange closure\n"
            "\n"
            "The flange is closed by several fitters, and each fitter in the gang writes down the "
            "torque they will pull their own bolts to before anyone starts. The notes go to the "
            "inspector together and nobody reads another's first.\n"
            "\n"
            "The joint holds to whatever its least-tightened bolt holds to, so the closure is rated "
            "on the lowest figure written, and pulling your own bolts harder is your own time on the "
            "wrench and your own studs stretched."
        ),
    ),
    MinEffortScenario(
        scenario_id="mine-airway-bore",
        effort_name="bore",
        group_noun="shift",
        frame=(
            "VENTILATION NOTE -- the deep workings, airway sections\n"
            "\n"
            "The airway is driven in sections, one to each shift, and every shift sets the bore it "
            "will drive its own section to before the round is fired. The settings are handed in "
            "together and no shift is told another's.\n"
            "\n"
            "Air moves through the airway only as freely as its narrowest section lets it, and "
            "driving your own section wider is your own steel and your own hours underground."
        ),
    ),
    MinEffortScenario(
        scenario_id="signal-chain-gain",
        effort_name="gain",
        group_noun="desk",
        frame=(
            "PATCH NOTE -- the long chain, tonight's broadcast\n"
            "\n"
            "The signal runs through several desks in series, and each operator on the desk fixes "
            "the gain their own stage will run at before the chain is patched up. The settings are "
            "written down at the same moment and no operator sees another's.\n"
            "\n"
            "What leaves the chain is only as clean as its worst stage makes it, so the broadcast is "
            "graded on the lowest gain in the chain, and running your own stage harder is your own "
            "valves and your own noise floor."
        ),
    ),
    MinEffortScenario(
        scenario_id="net-panel-mesh",
        effort_name="mesh",
        group_noun="fleet",
        frame=(
            "GEAR NOTE -- the joined net, opening week\n"
            "\n"
            "The net is made up of panels, one from each boat in the fleet, and every skipper settles "
            "the mesh their own panel will be knitted to before the panels are brought together. The "
            "orders go to the loft at once and none of you sees another's.\n"
            "\n"
            "Fish leave through the coarsest panel whatever the rest are like, so the net is worth "
            "what its loosest mesh is worth, and knitting your own panel finer is your own twine and "
            "your own winter."
        ),
    ),
    MinEffortScenario(
        scenario_id="hull-paint-coats",
        effort_name="coats",
        group_noun="shop",
        frame=(
            "YARD NOTE -- the hull, protective scheme\n"
            "\n"
            "The hull is painted in bands, one band to each shop, and each shop books the coats it "
            "will lay on its own band before the staging goes up. The bookings are read out together "
            "and no shop knows another's beforehand.\n"
            "\n"
            "Rust starts at the thinnest band and takes the hull from there, so the scheme lasts as "
            "long as its poorest band lasts, and laying more coats on your own band is your own "
            "drums and your own labour."
        ),
    ),
    MinEffortScenario(
        scenario_id="weld-pass-count",
        effort_name="passes",
        group_noun="bay",
        frame=(
            "SHOP NOTE -- the spine seam, long run\n"
            "\n"
            "The seam is run in lengths, one length to each bay, and every bay records the passes it "
            "will lay on its own length before the jigs are set. The records go to the shop office "
            "together and nobody reads another bay's.\n"
            "\n"
            "A seam tears at its shallowest length first, so the run is certified on the fewest "
            "passes anywhere along it, and laying more on your own length is your own rod and your "
            "own shift."
        ),
    ),
    MinEffortScenario(
        scenario_id="grout-curtain-depth",
        effort_name="depth",
        group_noun="site",
        frame=(
            "GROUT NOTE -- the curtain under the dam, drilling programme\n"
            "\n"
            "The curtain is drilled in panels, one to each contractor on the site, and each of you "
            "files the depth your own panel will be grouted to before the rigs move on. The filings "
            "are opened at the same meeting.\n"
            "\n"
            "Water finds the shallowest panel and comes under there, so the curtain is worth what "
            "its shallowest panel is worth, and taking your own panel deeper is your own cement and "
            "your own rig time."
        ),
    ),
    MinEffortScenario(
        scenario_id="marquee-guy-tension",
        effort_name="tension",
        group_noun="party",
        frame=(
            "RIGGING NOTE -- the big marquee, before the weather\n"
            "\n"
            "The marquee is guyed from every side, one side to each party on the ground, and each "
            "party settles the tension it will set its own guys to before the wind gets up. The "
            "settings are called in together and none of you hears another's first.\n"
            "\n"
            "The canvas lifts at the slackest side and goes from there, so the rig stands on its "
            "loosest guys, and hauling your own tighter is your own ground anchors and your own backs."
        ),
    ),
    MinEffortScenario(
        scenario_id="roof-sheet-overlap",
        effort_name="overlap",
        group_noun="team",
        frame=(
            "ROOFING NOTE -- the long shed, sheeting run\n"
            "\n"
            "The roof is sheeted in runs, one run to each team, and every team writes down the "
            "overlap it will lay its own run to before the sheets are lifted. The notes are handed to "
            "the foreman at once and no team is shown another's.\n"
            "\n"
            "Rain gets in at the shallowest lap and there is no benefit in the rest being deeper, so "
            "the roof is signed off on its smallest overlap, and lapping your own run further is your "
            "own sheets and your own hours."
        ),
    ),
    MinEffortScenario(
        scenario_id="kiln-soak-dwell",
        effort_name="dwell",
        group_noun="floor",
        frame=(
            "FIRING NOTE -- the long kiln, single soak\n"
            "\n"
            "The kiln is loaded from several floors at once, and each floor books the dwell its own "
            "ware will be held at temperature for before the doors are shut. The bookings are lodged "
            "together and nobody sees another floor's.\n"
            "\n"
            "The load is graded on the ware that came out least matured, so the firing is worth what "
            "its shortest dwell is worth, and holding your own ware longer is your own gas and your "
            "own kiln space."
        ),
    ),
    MinEffortScenario(
        scenario_id="culture-plate-sampling",
        effort_name="sample count",
        group_noun="bench",
        frame=(
            "LAB NOTE -- the batch release, plate work\n"
            "\n"
            "The batch is checked from several draws, one draw to each bench, and each bench settles "
            "the sample count it will plate from its own draw before any plate is poured. The "
            "settings are entered at the same time and no bench reads another's.\n"
            "\n"
            "The release is only as sound as its thinnest draw, so the batch is judged on the "
            "smallest count anywhere, and plating more from your own draw is your own media and your "
            "own bench hours."
        ),
    ),
    MinEffortScenario(
        scenario_id="mill-feed-rate",
        effort_name="feed rate",
        group_noun="section",
        frame=(
            "MILL NOTE -- the through line, day's run\n"
            "\n"
            "The line runs through several sections in order, and each section commits the feed rate "
            "it will hold for the day before the first stock goes in. The commitments are posted "
            "together and none of you sees another section's.\n"
            "\n"
            "Stock only comes off the end as fast as the slowest section passes it, so the run is "
            "counted on the lowest rate on the line, and holding your own section faster is your own "
            "tooling and your own breakdowns."
        ),
    ),
    MinEffortScenario(
        scenario_id="sluice-gate-lift",
        effort_name="lift",
        group_noun="reach",
        frame=(
            "CHANNEL NOTE -- the head race, irrigation season\n"
            "\n"
            "The race passes through a run of gates, one to each reach, and every reach settles the "
            "lift it will open its own gate to before the water is turned in. The settings are "
            "recorded at the same hour and no reach is told another's.\n"
            "\n"
            "Water gets down the race only as far as the least-opened gate allows, and opening your "
            "own further is your own scour and your own bank repairs."
        ),
        eval_only=True,
    ),
    MinEffortScenario(
        scenario_id="hedge-stake-spacing",
        effort_name="stake spacing",
        group_noun="holding",
        frame=(
            "BOUNDARY NOTE -- the laid hedge, winter work\n"
            "\n"
            "The hedge is laid in lengths, one length to each holding along it, and each of you "
            "settles the stake spacing you will work your own length to before the billhooks come "
            "out. The notes go to the commoners' clerk together and none is read first.\n"
            "\n"
            "Stock pushes through at the most openly staked length and is out whatever the rest are "
            "like, so the boundary holds to its widest spacing, and staking your own length closer is "
            "your own stakes and your own winter days."
        ),
        eval_only=True,
    ),
    MinEffortScenario(
        scenario_id="telescope-mirror-figure",
        effort_name="figure",
        group_noun="workshop",
        frame=(
            "OPTICS NOTE -- the segmented mirror, polishing programme\n"
            "\n"
            "The mirror is made of segments, one to each workshop, and every workshop settles the "
            "figure it will polish its own segment to before the blanks are cut. The settings are "
            "filed at the same date and no workshop sees another's.\n"
            "\n"
            "The image is spoiled by the worst segment however good the rest are, so the mirror "
            "performs to its poorest figure, and polishing your own closer is your own months on the "
            "machine and your own compound."
        ),
        eval_only=True,
    ),
    MinEffortScenario(
        scenario_id="bell-tower-rope-length",
        effort_name="draw",
        group_noun="band",
        frame=(
            "TOWER NOTE -- the peal, practice night\n"
            "\n"
            "The peal is rung by a band, one bell to each ringer, and each ringer settles the draw "
            "they will pull their own bell to before the ropes are taken up. The settings are chalked "
            "up together and nobody looks at another's first.\n"
            "\n"
            "A peal is heard across the parish only as far as its least-drawn bell carries, and "
            "pulling your own harder is your own arms for the rest of the night."
        ),
        eval_only=True,
    ),
)


def format_points(payoff: float) -> str:
    """Render a normalised payoff as the point value shown to the model.

    Public because anything that writes text *about* a rendered prompt -- interp stimuli that quote
    a cell's credit, an analysis that matches on point strings -- has to print the number the way
    the prompt printed it. Re-deriving the rounding at the call site is how a stimulus ends up
    quoting 23.077 at a prompt that says 23.08.
    """
    points = round(payoff * POINTS_PER_PAYOFF_UNIT, POINTS_DECIMALS)
    return f"{points:g}"


def quantize_payoff_to_display(payoff: float) -> float:
    """Snap a normalised payoff onto the grid of values `format_points` can actually show.

    The v2 track-record honesty upgrade rests on this: the reward computes from a row's payoff
    columns, the model computes from the printed points, and the two are one number only if the
    columns already sit on the display grid. v1 kept unrounded internal cells, so the model's
    arithmetic and the reward's could disagree in the fourth decimal; folding the rounding into
    the columns BEFORE any EV is computed closes that gap exactly.
    """
    return round(payoff * POINTS_PER_PAYOFF_UNIT, POINTS_DECIMALS) / POINTS_PER_PAYOFF_UNIT


def _canonical_action(label: str, coop_label: str) -> str:
    """Map an in-frame label to its canonical action under one label mapping."""
    return COOPERATE if label == coop_label else DEFECT


def print_order_of(labels: tuple[str, str], label_print_order: str) -> tuple[str, str]:
    """Return two authored-order labels in the order a `label_print_order` prints them.

    The one place the order's meaning is spelled out, so a reader of banked records (which carry the
    authored `label_a` / `label_b` beside the record's `label_print_order`) recovers the order the model
    read from the same code the renderer used to print it.
    """
    if label_print_order == LABEL_PRINT_ORDER_CANONICAL:
        return labels
    if label_print_order == LABEL_PRINT_ORDER_SWAPPED:
        first, second = labels
        return (second, first)
    raise ValueError(
        f"Unknown label_print_order {label_print_order!r}; known orders: {sorted(LABEL_PRINT_ORDERS)}."
    )


def labels_in_print_order(scenario: Scenario, label_print_order: str) -> tuple[str, str]:
    """Return the frame's two labels in the order the prompt prints them.

    Deliberately not a second meaning for `Scenario.labels`, which stays the authored order because
    `coop_label_index` indexes into it. Keeping the two apart is the whole guarantee the control
    rests on: reversing what gets printed cannot move which label the reward pays for, so a swapped
    sweep measures the same game on a different page rather than a different game.
    """
    return print_order_of(scenario.labels, label_print_order)


def _outcome_block(
    spec: MatrixGameSpec,
    scenario: Scenario,
    *,
    coop_label_index: int,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> str:
    """Render the four outcome cells, one line each, in one fixed label order.

    Iterating over the labels (rather than over the canonical actions) is what makes the two
    counterbalanced renderings byte-identical apart from their numbers: the lines appear in the
    same order and name the labels in the same order either way, so only the point values move.
    That holds under either print order, because both loops read the same sequence -- swapping it
    reverses the four lines and carries each line's own point value with it.
    """
    coop_label = scenario.coop_label(coop_label_index)
    print_order = labels_in_print_order(scenario, label_print_order)
    lines: list[str] = []
    for mine in print_order:
        for theirs in print_order:
            payoff = spec.payoff(
                _canonical_action(mine, coop_label), _canonical_action(theirs, coop_label)
            )
            lines.append(
                f"- You {mine}, they {theirs}: you are credited {format_points(payoff)} points."
            )
    return "\n".join(lines)


def _counterpart_paragraph(*, twin_framing: bool, opponent_framing: str | None) -> str | None:
    """Return the "about the other side" paragraph, or None when the frame says nothing extra."""
    if twin_framing and opponent_framing is not None:
        raise ValueError(
            "twin_framing and opponent_framing describe the same counterpart two different ways; pass at most one."
        )
    if twin_framing:
        return twin_counterpart_paragraph()
    clause = opponent_framing
    if clause is None:
        return None
    return about_the_other_side(clause)


def about_the_other_side(clause: str) -> str:
    """Open a counterpart paragraph the one way every game opens it.

    One source for the opening words, because a test asserts every game's prompt describes its
    counterpart by matching on them: a renderer that phrased its own would pass the vocabulary guard
    and silently drop out of that check.
    """
    return f"About the other side: {clause}"


def twin_counterpart_paragraph() -> str:
    """Return the complete counterpart paragraph used by twin-framed play prompts."""
    return about_the_other_side(TWIN_COUNTERPART_CLAUSE)


def render_game_prompt(  # noqa: PLR0913
    spec: MatrixGameSpec,
    scenario: Scenario,
    *,
    coop_label_index: int,
    twin_framing: bool = False,
    opponent_framing: str | None = None,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> str:
    """Render one one-shot matrix game as the untemplated prompt the model reads.

    `coop_label_index` picks which of the frame's two labels maps to the canonical cooperative
    action; both values are generated for every frame, which is the counterbalance. Pass at most
    one of `twin_framing` (the counterpart is another instance of this model) and
    `opponent_framing` (a caller-supplied clause describing an external counterpart).

    `label_print_order` reverses the order the outcome table and the answer instruction print the
    two labels in, leaving the mapping `coop_label_index` fixed. The frame is authored prose and is
    printed as written either way; see the module docstring for why.
    """
    first_label, second_label = labels_in_print_order(scenario, label_print_order)
    sections = [
        scenario.frame,
        _outcome_block(
            spec,
            scenario,
            coop_label_index=coop_label_index,
            label_print_order=label_print_order,
        ),
        SYMMETRY_NOTE,
        _counterpart_paragraph(twin_framing=twin_framing, opponent_framing=opponent_framing),
        ONE_SHOT_INSTRUCTION.format(first_label=first_label, second_label=second_label),
    ]
    prompt = "\n\n".join(section for section in sections if section is not None)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def _opponent_rule_paragraph(rule: OpponentRule, scenario: Scenario, *, coop_label: str) -> str:
    """Describe a deterministic opponent by its mechanism, never by the name of the strategy.

    Naming the rule would hand the model the literature the rule comes from, which is exactly
    what the vocabulary guard exists to prevent; the mechanism is stated in full instead, so
    nothing about the opponent is hidden.
    """
    other_label = next(label for label in scenario.labels if label != coop_label)
    if rule is OpponentRule.TIT_FOR_TAT:
        mechanism = (
            f"in the first round they {coop_label}; in each later round they repeat whatever you "
            f"chose in the round before"
        )
    elif rule is OpponentRule.GRIM_TRIGGER:
        mechanism = (
            f"in the first round they {coop_label}, and they keep to {coop_label} for as long as "
            f"you do; from the first round in which you {other_label} onward, they {other_label} "
            f"in every remaining round"
        )
    elif rule is OpponentRule.ALWAYS_C:
        mechanism = f"they {coop_label} in every round, whatever you choose"
    else:
        raise ValueError(f"Unhandled opponent rule {rule!r}.")
    return (
        f"About the other side: they follow a fixed written procedure, which you have been given "
        f"in full -- {mechanism}. They do not depart from it."
    )


def render_iterated_prompt(  # noqa: PLR0913
    spec: MatrixGameSpec,
    scenario: Scenario,
    *,
    rule: OpponentRule,
    n_rounds: int,
    coop_label_index: int,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> str:
    """Render a fixed-length repeated game against a deterministic opponent as one prompt.

    The whole match is played in a single completion: the opponent's rule is stated in full, so
    an open-loop plan of `n_rounds` action tags is also the optimal closed-loop one, and no
    turn-by-turn interaction is needed.

    `label_print_order` moves the outcome table and the answer instruction, as in the one-shot
    renderer. It does not touch the opponent's rule paragraph, whose label order is the mechanism
    being described ("in the first round they X; ... from the first round in which you Y") rather
    than an enumeration of the options, so reversing it would state a different rule.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}.")
    coop_label = scenario.coop_label(coop_label_index)
    first_label, second_label = labels_in_print_order(scenario, label_print_order)
    sections = [
        scenario.frame,
        _outcome_block(
            spec,
            scenario,
            coop_label_index=coop_label_index,
            label_print_order=label_print_order,
        ),
        SYMMETRY_NOTE,
        REPEATED_ROUNDS_NOTE.format(n_rounds=n_rounds),
        _opponent_rule_paragraph(rule, scenario, coop_label=coop_label),
        ITERATED_INSTRUCTION.format(
            n_rounds=n_rounds, first_label=first_label, second_label=second_label
        ),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def _responder_shares(spec: MatrixGameSpec) -> tuple[float, float]:
    """Return the responder's share of the total on offer, and the proposer's remainder.

    The remainder is only a share of anything because `games.payoffs.ultimatum_responder` skips
    normalisation and keeps the offered fraction itself in the cells; a spec scaled by its largest
    cell would put 1.0 here and hand the proposer nothing. So this refuses any other spec rather
    than printing a division that is not the one being graded.
    """
    accepted = spec.payoff(COOPERATE, COOPERATE)
    degenerate_in_the_other_axis = (
        spec.payoff(COOPERATE, DEFECT) == accepted
        and spec.payoff(DEFECT, COOPERATE) == 0.0
        and spec.payoff(DEFECT, DEFECT) == 0.0
    )
    if not (0.0 < accepted < 1.0) or not degenerate_in_the_other_axis:
        raise ValueError(
            f"{spec.game_id!r} does not state a share of one total: this renderer prints the "
            f"counterpart's credit as the remainder of the responder's, which needs cells of the "
            f"form (s, s, 0, 0) with 0 < s < 1, got "
            f"({spec.payoff_cc}, {spec.payoff_cd}, {spec.payoff_dc}, {spec.payoff_dd})."
        )
    return accepted, 1.0 - accepted


def _responder_outcome_block(
    spec: MatrixGameSpec,
    scenario: Scenario,
    *,
    coop_label_index: int,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> str:
    """Render one line per label: what that entry credits to each side, in one fixed label order.

    Two lines rather than four, and both sides' credits on each, because the counterpart has no
    move left: a four-cell table would print the responder's own two payoffs twice each and still
    never state what the counterpart is taking, which is the only quantity the item probes.
    """
    coop_label = scenario.coop_label(coop_label_index)
    responder_share, proposer_share = _responder_shares(spec)
    lines: list[str] = []
    for mine in labels_in_print_order(scenario, label_print_order):
        stands = _canonical_action(mine, coop_label) == COOPERATE
        mine_points, theirs_points = (responder_share, proposer_share) if stands else (0.0, 0.0)
        lines.append(
            f"- You {mine}: you are credited {format_points(mine_points)} points, and they are "
            f"credited {format_points(theirs_points)} points."
        )
    return "\n".join(lines)


def render_responder_prompt(
    spec: MatrixGameSpec,
    scenario: Scenario,
    *,
    coop_label_index: int,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> str:
    """Render the second-mover half of a division the counterpart has already made.

    Deliberately not the matrix renderer, and the difference is the measurement. Rendered as a
    matrix game this item showed a four-cell table in which the model's own label alone fixed its
    credit at the offered share or nothing, with no proposer, no split and no counterpart share
    anywhere in the text -- a dominance check that any model reading the table passes, reported
    under a name that invites reading it as unfairness rejection.

    The symmetry note is dropped for the same reason it is printed elsewhere: the counterpart's
    side is not scored from a table the responder shares, so saying it would be false. Their credit
    is stated on each outcome line instead, which is what makes the division visible.
    """
    first_label, second_label = labels_in_print_order(scenario, label_print_order)
    sections = [
        scenario.frame,
        _responder_outcome_block(
            spec,
            scenario,
            coop_label_index=coop_label_index,
            label_print_order=label_print_order,
        ),
        _counterpart_paragraph(twin_framing=False, opponent_framing=RESPONDER_COUNTERPART_CLAUSE),
        ONE_SHOT_INSTRUCTION.format(first_label=first_label, second_label=second_label),
    ]
    prompt = "\n\n".join(section for section in sections if section is not None)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_dictator_prompt(spec: DictatorSpec, scenario: DictatorScenario) -> str:
    """Render one unilateral-split frame as the untemplated prompt the model reads."""
    sections = [
        scenario.frame,
        (
            f"The allocation comes to {spec.endowment} {scenario.resource}. Decide how many of "
            f"them you keep; the rest go to the other side exactly as you leave them."
        ),
        DICTATOR_INSTRUCTION.format(endowment=spec.endowment),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_nash_demand_prompt(spec: NashDemandSpec, scenario: NashDemandScenario) -> str:
    """Render one simultaneous-claim division as the untemplated prompt the model reads.

    The counterpart paragraph is the same twin clause the symmetric matrix games carry, and for the
    same reason: both gradings of this game are the same prompt, so the clause has to be true of the
    self-graded arm (where the counterpart really is a copy of this policy) and of the group-mix arm
    (where it stands in for the group's own realised claims).
    """
    sections = [
        scenario.frame,
        NASH_DEMAND_MECHANICS.format(windfall=spec.windfall, resource=scenario.resource),
        _counterpart_paragraph(twin_framing=True, opponent_framing=None),
        NASH_DEMAND_INSTRUCTION.format(windfall=spec.windfall),
    ]
    prompt = "\n\n".join(section for section in sections if section is not None)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_threshold_goods_prompt(
    spec: ThresholdGoodsSpec, scenario: ThresholdGoodsScenario
) -> str:
    """Render one threshold public good as the untemplated prompt the model reads.

    Every number in the mechanics paragraph and in the counterpart clause comes from the spec, and the
    spec is rebuilt from the row's own columns when the reward grades a completion, so the prose and
    the grading cannot disagree about the stock, the bar, the prize, or how many parties there are.

    The counterpart clause is the plural of the twin clause the symmetric matrix games carry, and for
    the same reason: both gradings of this game render the same prompt, so the clause has to be true of
    the self-graded arm (where the counterparts really are copies of this policy) and of the group-mix
    arm (where they stand in for the group's own realised figures).
    """
    sections = [
        scenario.frame,
        THRESHOLD_GOODS_MECHANICS.format(
            parties=spec.n_parties,
            endowment=spec.endowment,
            resource=scenario.resource,
            undertaking=scenario.undertaking,
            threshold=spec.contribution_threshold,
            prize=spec.prize,
        ),
        about_the_other_side(
            THRESHOLD_GOODS_COUNTERPART_CLAUSE.format(counterparts=spec.team_size)
        ),
        THRESHOLD_GOODS_INSTRUCTION.format(endowment=spec.endowment),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def format_credit(spec: TransferSpec, *, setter: str) -> str:
    """Render what one beneficiary is credited per unit set down, as a whole-number ratio.

    Two sentences rather than one formatted float, because `TRANSFER_CREDIT_VARIANTS` registers a rate
    below one and the design needs it stated exactly: "1 for every 2 set down" is a rate a model can
    price directly, where "0.5" is one it has to convert first. `assert_transfer_spec` guarantees one
    side of the ratio is one, which is what makes these two sentences exhaustive.

    `setter` names who set the units down. The one-way game passes
    `TRANSFER_CREDIT_SETTER_ACTOR` so the sentence reads "for each one you set down": under the
    passive form a reader can take the credit as reaching whoever is on the other side of somebody
    else's note tonight, which is the twin's payoff structure and not this game's.
    """
    if spec.credit_denominator == 1:
        return TRANSFER_CREDIT_WHOLE_RATE.format(numerator=spec.credit_numerator, setter=setter)
    return TRANSFER_CREDIT_FRACTIONAL_RATE.format(
        numerator=spec.credit_numerator, denominator=spec.credit_denominator, setter=setter
    )


def transfer_beneficiary_group(spec: TransferSpec, scenario: TransferScenario) -> str:
    """Name the beneficiaries as a group, in the number the count calls for.

    "the one loft" rather than "the 1 loft": the count of one is registered
    (`TRANSFER_BENEFICIARY_COUNTS`) and is half of the benefit dose, so the singular has to read as
    English rather than as a filled template.
    """
    if spec.beneficiary_count == 1:
        return f"the one {scenario.beneficiary_noun_singular}"
    return f"the {spec.beneficiary_count} {scenario.beneficiary_noun}"


def transfer_beneficiary_subject(spec: TransferSpec, scenario: TransferScenario) -> str:
    """Name whichever beneficiary the credit sentence is about, as a singular subject.

    Singular in both numbers ("the one loft", "each of the 3 lofts"), so the sentence that follows
    takes "is" and "its own" whatever the count is. Agreement words in a template are how a prompt
    comes to read as machine-filled at one rung and not another.
    """
    if spec.beneficiary_count == 1:
        return f"the one {scenario.beneficiary_noun_singular}"
    return f"each of the {spec.beneficiary_count} {scenario.beneficiary_noun}"


def transfer_spec_numerals(spec: TransferSpec) -> tuple[str, ...]:
    """Every number the transfer renderers print, as the strings a frame must not contain."""
    return (
        str(spec.endowment),
        str(spec.beneficiary_count),
        str(spec.credit_numerator),
        str(spec.credit_denominator),
    )


def assert_transfer_frame_states_no_spec_numerals(
    spec: TransferSpec, scenario: TransferScenario
) -> None:
    """Raise if an authored frame states any number this spec's renderer prints.

    Render-time rather than at scenario construction, because the numbers belong to the spec and one
    frame is rendered under ten of them. The failure it prevents is the one §1.7 of the wave-2 design
    brief is about, in its worst form here: a frame that said "the four lofts" would contradict the
    mechanics paragraph the moment the count moved to three, and every artifact would still be complete,
    plausible and internally consistent -- the model would simply have been told two different
    situations and the record would name one of them.
    """
    stated = sorted(
        {numeral for numeral in transfer_spec_numerals(spec) if numeral in scenario.frame}
    )
    if stated:
        raise ValueError(
            f"{scenario.scenario_id!r} states {stated} in its frame, and this spec's renderer prints "
            f"those same numbers (stock {spec.endowment}, count {spec.beneficiary_count}, credit "
            f"{spec.credit_numerator}/{spec.credit_denominator}). The frame must state no count, stock "
            f"or credit at all: the renderer prints them from the columns the record carries, so a "
            f"frame that names one describes a different situation the first time that number moves."
        )


def transfer_mechanics_fill(
    spec: TransferSpec, scenario: TransferScenario, *, credit_setter: str
) -> dict[str, str | int]:
    """Build the values every placeholder of a transfer mechanics paragraph is filled with, for one cell.

    One table for the whole paragraph and for any sentence lifted out of it, so a sentence formatted on
    its own reads exactly as it does inside the render -- which is what lets the drawn game's render be
    compared against the twin's sentence by sentence rather than paragraph by paragraph.
    """
    return {
        "endowment": spec.endowment,
        "resource": scenario.resource,
        "destination": scenario.destination,
        "note_noun": scenario.note_noun,
        "beneficiary_subject": transfer_beneficiary_subject(spec, scenario),
        "beneficiary_group": transfer_beneficiary_group(spec, scenario),
        "credit": format_credit(spec, setter=credit_setter),
        "own_stake": TRANSFER_OWN_STAKE_SENTENCES[spec.own_stake_scale],
        "kept_value": TRANSFER_ONE_WAY_KEPT_VALUE[spec.own_stake_scale],
    }


def assert_drawn_render_is_the_twin_render_with_the_replacements(
    *,
    twin_mechanics: str,
    drawn_mechanics: str,
    spec: TransferSpec,
    scenario: TransferScenario,
    where: str,
) -> None:
    """Raise unless a rendered drawn paragraph is the rendered twin's with exactly two sentences swapped.

    The two sentences are those of :data:`DRAWN_DECISION_REPLACEMENTS`, and neither twin sentence may
    survive in the drawn text. The constant-level derivation already guarantees this of the templates;
    this is the same pin on the RENDERED text, which is what the model reads and what the knockout's
    double difference is taken over. It also catches a drift the templates cannot show, such as the
    drawn renderer filling the credit with the actor's own setter ("for each one you set down") while
    the twin's stays passive, which would leave the two paragraphs differing in a third sentence nobody
    chose.
    """
    fill = transfer_mechanics_fill(spec, scenario, credit_setter=TRANSFER_CREDIT_SETTER_ANY_SIDE)
    expected = twin_mechanics
    for twin_sentence, drawn_sentence in DRAWN_DECISION_REPLACEMENTS:
        filled_twin = twin_sentence.format(**fill)
        filled_drawn = drawn_sentence.format(**fill)
        occurrences = twin_mechanics.count(filled_twin)
        if occurrences != 1:
            raise ValueError(
                f"the twin's rendered mechanics for {where} carry the sentence {filled_twin!r} "
                f"{occurrences} times, expected exactly once; the drawn game is defined as the twin "
                f"with that sentence replaced, so this pair cannot be pinned to the replacements."
            )
        if filled_twin in drawn_mechanics:
            raise ValueError(
                f"the drawn game's rendered mechanics for {where} still carry the twin's sentence "
                f"{filled_twin!r}, which the drawn game replaces with {filled_drawn!r}. With it in "
                f"place the paragraph says both what the twin says and what the drawn game says."
            )
        expected = expected.replace(filled_twin, filled_drawn)
    if expected != drawn_mechanics:
        raise ValueError(
            f"the drawn game's rendered mechanics for {where} are not the twin's with exactly the "
            f"{len(DRAWN_DECISION_REPLACEMENTS)} sentences of DRAWN_DECISION_REPLACEMENTS swapped. "
            f"Expected {expected!r}, rendered {drawn_mechanics!r}. Whatever else moved sits inside the "
            f"knockout's twin-against-drawn difference as a third thing nobody chose."
        )


def _transfer_sections(  # noqa: PLR0913 - two renderer-owned constants, plus the row's own axes
    mechanics: str,
    spec: TransferSpec,
    scenario: TransferScenario,
    *,
    credit_setter: str,
    clause: str | None,
    polarity: str,
) -> list[str]:
    """Assemble one transfer prompt's sections, in the order both games render them.

    Frame, mechanics, closing order, counterpart paragraph, answer instruction. The closing order and
    the instruction are one text in both games and the frame is the author's, so a section diff between
    the two games shows exactly the mechanics and the counterpart paragraph -- the property the design
    rests on and the tests assert.
    """
    if polarity not in TRANSFER_INSTRUCTIONS:
        raise ValueError(
            f"polarity must be one of {sorted(TRANSFER_INSTRUCTIONS)}, got {polarity!r}; it decides "
            f"which of the two figures the answer names and which tag the scan reads."
        )
    assert_transfer_frame_states_no_spec_numerals(spec, scenario)
    filled = mechanics.format(
        **transfer_mechanics_fill(spec, scenario, credit_setter=credit_setter)
    )
    sections = [
        scenario.frame,
        filled,
        TRANSFER_CLOSING_ORDER.format(
            destination=scenario.destination,
            resource=scenario.resource,
        ),
    ]
    if clause is not None:
        sections.append(about_the_other_side(clause))
    sections.append(
        TRANSFER_INSTRUCTIONS[polarity].format(resource=scenario.resource, endowment=spec.endowment)
    )
    return sections


def render_one_way_transfer_prompt(
    spec: TransferSpec, scenario: TransferScenario, *, clause: str | None, polarity: str
) -> str:
    """Render the one-way transfer: the actor sets units down, the beneficiaries decide nothing.

    `clause` of None omits the counterpart paragraph entirely, which is the identity-blind cell and
    also the stem the one-inserted-paragraph audit compares every other cell against.
    """
    prompt = "\n\n".join(
        _transfer_sections(
            ONE_WAY_TRANSFER_MECHANICS,
            spec,
            scenario,
            credit_setter=TRANSFER_CREDIT_SETTER_ACTOR,
            clause=clause,
            polarity=polarity,
        )
    )
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_matched_decision_transfer_prompt(
    spec: TransferSpec, scenario: TransferScenario, *, clause: str | None, polarity: str
) -> str:
    """Render the matched-decision twin: every side faces the same decision at the same moment.

    Same spec, same scenario, same closing order and same instruction as the one-way render, so a cell
    of this game and the same cell of the one-way game differ by the mechanics and the counterpart
    paragraph and by nothing else. That is what makes the pair a read of the coupling: a correlation
    reasoner can collect here and cannot there.
    """
    prompt = "\n\n".join(
        _transfer_sections(
            MATCHED_DECISION_TRANSFER_MECHANICS,
            spec,
            scenario,
            credit_setter=TRANSFER_CREDIT_SETTER_ANY_SIDE,
            clause=clause,
            polarity=polarity,
        )
    )
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_drawn_decision_transfer_prompt(
    spec: TransferSpec, scenario: TransferScenario, *, clause: str | None, polarity: str
) -> str:
    """Render the drawn-decision form: the reader decides, the other sides' figures came from a draw.

    Everything the twin renders, with the twin's copy cue and decision sentence replaced (see
    :data:`DRAWN_DECISION_REPLACEMENTS`). The stock, the credit, the closing order and the answer
    instruction are the twin's own text, so a cell of this game and the same cell of the twin differ by
    the mechanics paragraph and the counterpart paragraph and by nothing else -- which is what makes the
    pair a read of the draw: a mirror reasoner can collect in the twin and has nothing to mirror here,
    while it is described as the same kind of counterpart in both.
    """
    prompt = "\n\n".join(
        _transfer_sections(
            DRAWN_DECISION_TRANSFER_MECHANICS,
            spec,
            scenario,
            credit_setter=TRANSFER_CREDIT_SETTER_ANY_SIDE,
            clause=clause,
            polarity=polarity,
        )
    )
    assert_no_loaded_vocabulary(prompt)
    return prompt


TRANSFER_RENDERERS: dict[str, Callable[..., str]] = {
    ONE_WAY_TRANSFER_GAME_ID: render_one_way_transfer_prompt,
    MATCHED_DECISION_TRANSFER_GAME_ID: render_matched_decision_transfer_prompt,
    DRAWN_DECISION_TRANSFER_GAME_ID: render_drawn_decision_transfer_prompt,
}
"""One renderer per transfer game, so a game id added to the registry cannot miss its renderer."""


def _assert_every_transfer_game_has_a_renderer() -> None:
    """Refuse at import if a registered transfer game has no renderer, or vice versa.

    The registry and the id tuple are read by different callers -- the plan walks the ids, the
    generator resolves the renderer -- so a game in one and not the other is a `KeyError` deep inside
    a plan build, or a renderer nothing ever asks for.
    """
    unrendered = sorted(set(TRANSFER_GAME_IDS) - set(TRANSFER_RENDERERS))
    stray = sorted(set(TRANSFER_RENDERERS) - set(TRANSFER_GAME_IDS))
    if unrendered or stray:
        raise RuntimeError(
            f"the transfer game ids and their renderers disagree: registered games with no renderer "
            f"{unrendered or 'none'}; renderers for unregistered games {stray or 'none'}."
        )


_assert_every_transfer_game_has_a_renderer()

COUNTERPART_PARAGRAPH_MARKER = about_the_other_side("")
"""The opening words every counterpart paragraph in the games carries, as the audit's default marker."""


def assert_counterpart_paragraph_is_the_only_insertion(
    *, stem: str, rendered: str, prompt_id: str, marker: str = COUNTERPART_PARAGRAPH_MARKER
) -> None:
    """Assert a rendered prompt is its stem plus exactly ONE counterpart paragraph.

    The property every counterpart-clause design in this repository rests on: a cell differs from its
    clause-free stem by one inserted paragraph, so a movement between cells is attributable to the
    clause and to nothing else. Deleting every section that opens with the counterpart marker must
    reproduce the stem byte for byte, which is why a two-paragraph clause fails here -- its second
    paragraph does not carry the marker, so it survives the deletion and the comparison goes red.

    Shared rather than owned by whichever caller was written first: the training corpus
    (`games.track_record_corpus`) and the sociology probes prove the same property about prompts built
    by different renderers, and two copies of it are two things to keep in step.

    `marker` is the opening words the inserted paragraph carries, defaulting to the games' own. It is a
    parameter because a stimulus outside the games -- an agent transcript whose one inserted paragraph
    describes the other agents rather than "the other side" -- proves exactly this property about a
    paragraph of its own, and a second copy of this function is a second thing to keep in step.
    """
    if not marker.strip():
        raise ValueError(
            f"marker {marker!r} is blank, so every section starts with it: the audit would delete the "
            f"whole prompt and compare nothing against the stem."
        )
    sections = rendered.split("\n\n")
    counterpart = [section for section in sections if section.startswith(marker)]
    without_them = "\n\n".join(section for section in sections if not section.startswith(marker))
    if len(counterpart) != 1 or without_them != stem:
        raise ValueError(
            f"{prompt_id!r} is not its stem plus one counterpart paragraph "
            f"({len(counterpart)} such paragraphs found): deleting them does not reproduce the stem, "
            f"so something other than the counterpart clause moved."
        )


def format_transfer_multiple(multiplier: float) -> str:
    """Render the multiple applied in transit the way the prompt states it.

    Public for `format_points`'s reason: anything quoting a rendered trust prompt has to print the
    multiple the way the prompt printed it, and the reward reads the same number off the row.
    """
    return f"{multiplier:g}"


def format_return_percentage(fraction: float) -> str:
    """Render an announced return fraction as the whole-number percentage the prompt states.

    Whole numbers only, which `assert_trust_spec` guarantees for any registered variant: a rate the
    prose had to round would state a different game from the one the reward grades.
    """
    return f"{round(fraction * 100)}"


def _trust_consignment_paragraph(spec: TrustSpec, scenario: TrustScenario) -> str:
    """State the stock, the multiple applied in transit, and what the decision is."""
    return (
        f"You are holding {spec.endowment} {scenario.resource}. Decide how many of them you send: "
        f"whatever you send arrives on the other side as "
        f"{format_transfer_multiple(spec.multiplier)} times that number, and whatever you hold back "
        f"stays with you as it is."
    )


TRUST_FIGURE_NOTE = (
    "Your own figure at the end is what you held back, plus whatever comes back to you."
)

# The strategy method says outright that the share written here is not charged against this side's
# figure, because that is exactly what the reward does (games.payoffs.trustor_payoff grades the
# trustor's side alone). Leaving it unsaid would let a model infer a cost the grading does not
# charge, and a promise measured under that misreading would be a measurement of the wording.
TRUST_SELF_RULE_FIGURE_NOTE = (
    "Your own figure at the end is what you held back, plus whatever comes back to you. What you "
    "send back out of the other side's consignment is theirs and is not part of your figure."
)

TRUSTEE_FIGURE_NOTE = "Your own figure at the end is whatever you keep of what arrived."


def render_trust_stated_return_prompt(spec: TrustSpec, scenario: TrustScenario) -> str:
    """Render a consignment sent to a counterpart whose return share is published in the prompt.

    The sharper of the two forms, and the reason is arithmetic the model can do in one
    multiplication: the prose is identical across the two registered variants and one number moves,
    and that number decides whether the optimum is the whole stock or nothing at all. So a baseline
    that sends the same on both sides of the break-even says the model is reading the frame rather
    than the rule, on a second construct after the stag ladder.
    """
    if not spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces no return rate, so this renderer would have to invent the "
            f"rule it prints. Render the strategy-method form instead, where the model writes it."
        )
    sections = [
        scenario.frame,
        _trust_consignment_paragraph(spec, scenario),
        TRUST_FIGURE_NOTE,
        about_the_other_side(
            STATED_RULE_COUNTERPART_CLAUSE.format(
                percentage=format_return_percentage(spec.stated_return_fraction)
            )
        ),
        TRUST_SEND_INSTRUCTION.format(endowment=spec.endowment),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_trust_strategy_prompt(spec: TrustSpec, scenario: TrustScenario) -> str:
    """Render the strategy method: one completion states the send AND the share it would return.

    The twin applies the share written here to what was sent here, which is the only construction
    that keeps the counterpart symmetric while still making the return rate a decision rather than a
    parameter. Its degeneracy is real and stated in the prompt rather than hidden: the share is paid
    out of the other side's consignment, so it costs this side's figure nothing, and the
    payoff-maximising answer is to send everything and promise everything back.
    """
    if spec.announces_a_return_rate:
        raise ValueError(
            f"{spec.game_id!r} announces a return rate of {spec.stated_return_fraction}, but this "
            f"form asks the model for the rule and applies what it wrote. Printing an announced "
            f"rate here would state one rule and grade another."
        )
    sections = [
        scenario.frame,
        _trust_consignment_paragraph(spec, scenario),
        TRUST_SELF_RULE_FIGURE_NOTE,
        about_the_other_side(TRUST_TWIN_COUNTERPART_CLAUSE),
        TRUST_STRATEGY_INSTRUCTION.format(endowment=spec.endowment),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_trustee_prompt(spec: TrustSpec, scenario: TrustScenario) -> str:
    """Render the never-trained trustee side: a share to send back, with no send of its own.

    Deliberately a rule rather than an amount, and deliberately the same `<return>` tag in the same
    units as the strategy method's promise, so the two are directly comparable per checkpoint. That
    comparison is the whole item: in the trained trustor role the promise is free, and here it comes
    out of what this side is holding. A promise that rises where it is free and falls where it is
    costly is a consistency finding for one extra eval cell and no extra training.
    """
    sections = [
        scenario.frame,
        (
            f"Whatever the other side sends arrives with you as "
            f"{format_transfer_multiple(spec.multiplier)} times what they sent, and it is held in "
            f"your name. Decide what percentage of whatever arrives goes back to them; the rest "
            f"stays with you."
        ),
        TRUSTEE_FIGURE_NOTE,
        about_the_other_side(TRUSTEE_COUNTERPART_CLAUSE),
        TRUSTEE_RETURN_INSTRUCTION,
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def _min_effort_outcome_block(spec: MinEffortSpec) -> str:
    """Render every cell of the level grid, one line each, in one fixed order.

    Every combination rather than the two coefficients in prose, for the reason the matrix games print
    four cells instead of a formula: the measurement is which level the model chooses, and a model
    that has to multiply a cost by its own answer first is being measured on its arithmetic. The 2B is
    the size where that difference bites.

    Iterating own level outermost puts each own-level's five cells together, which is the order a
    reader compares them in, and makes the two counterparts' tables differ only in their numbers --
    the property `test_the_four_variants_differ_only_in_their_figures` pins.
    """
    return "\n".join(
        f"- You write {own_level} and the lowest anyone else writes is {lowest_other_level}: you are "
        f"credited "
        f"{format_points(min_effort_cell_reward(spec, own_level=own_level, lowest_other_level=lowest_other_level))}"
        f" points."
        for own_level in spec.levels
        for lowest_other_level in spec.levels
    )


def _min_effort_counterpart_paragraph(spec: MinEffortSpec) -> str:
    """Describe the team as this many other instances of this model, from the graded column."""
    return about_the_other_side(
        MIN_EFFORT_COUNTERPART_CLAUSE.format(
            counterparts=spec.team_size,
            instance_word="instance" if spec.team_size == 1 else "instances",
        )
    )


def _min_effort_sections(spec: MinEffortSpec, scenario: MinEffortScenario) -> list[str]:
    """Return the sections both minimum-effort renderers share, in order.

    The frame, the mechanics, the grid and the symmetry note are identical between the one-shot and
    repeated forms by construction, which is what makes the repeated arm a companion to the one-shot
    one rather than a second game: the two differ in the counterpart and the answer format only.
    """
    return [
        scenario.frame,
        MIN_EFFORT_MECHANICS.format(
            effort_name=scenario.effort_name,
            n_levels=spec.n_levels,
            group_noun=scenario.group_noun,
        ),
        _min_effort_outcome_block(spec),
        MIN_EFFORT_SYMMETRY_NOTE,
    ]


def render_min_effort_prompt(spec: MinEffortSpec, scenario: MinEffortScenario) -> str:
    """Render one minimum-effort frame as the untemplated prompt the model reads.

    The counterpart paragraph is the twin clause in the plural, and for the symmetric matrix games'
    reason: the counterparts really are other instances of this policy answering this prompt, which is
    what the group-mix grading stands in for when it draws them from the group's realised levels.
    """
    sections = [
        *_min_effort_sections(spec, scenario),
        _min_effort_counterpart_paragraph(spec),
        MIN_EFFORT_INSTRUCTION.format(n_levels=spec.n_levels),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def render_min_effort_match_prompt(
    spec: MinEffortSpec, scenario: MinEffortScenario, *, n_rounds: int
) -> str:
    """Render the repeated form: the same team briefing against an announced level-matcher.

    The whole match is played in one completion, exactly as the matrix repeated arms are: the rule is
    stated in full, so an open-loop plan of `n_rounds` level tags is also the optimal closed-loop one.

    The counterpart is a single announced system rather than a team, which is why a spec with more
    counterparts is refused rather than rendered: the clause would say one thing, the grid another,
    and the reward would simulate a third.
    """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be positive, got {n_rounds}.")
    if spec.team_size != 1:
        raise ValueError(
            f"{spec.game_id!r} has team_size={spec.team_size}, but the repeated form is played "
            f"against ONE announced counterpart whose rule the prompt states in full. Rendering a "
            f"larger team here would print a rule for one system while the grid and the simulated "
            f"return describe several, and nothing downstream could tell which was graded."
        )
    sections = [
        *_min_effort_sections(spec, scenario),
        REPEATED_ROUNDS_NOTE.format(n_rounds=n_rounds),
        about_the_other_side(
            MIN_EFFORT_MATCHER_CLAUSE.format(opening_level=MIN_EFFORT_MATCHER_OPENING_LEVEL)
        ),
        MIN_EFFORT_MATCH_INSTRUCTION.format(n_rounds=n_rounds, n_levels=spec.n_levels),
    ]
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    return prompt


def _fixed_spec(build: Callable[[], MatrixGameSpec]) -> Callable[[str], MatrixGameSpec]:
    """Adapt a game with no payoff parameter to the one-argument `build_spec` signature."""

    def build_for_variant(payoff_variant: str) -> MatrixGameSpec:
        del payoff_variant
        return build()

    return build_for_variant


@dataclass(frozen=True)
class _MatrixArm:
    """How one matrix `game_id` builds its specs and describes its counterpart.

    `scenarios` is which authored frame roster the game renders over; None means the shared
    house roster (`MATRIX_SCENARIOS`), which is every game registered before pd-reskin. A game
    with its own roster exists so that widening one game's fiction bank cannot silently change
    any other game's corpus.
    """

    build_spec: Callable[[str], MatrixGameSpec]
    payoff_variants: tuple[str, ...]
    twin_framing: bool = False
    opponent_framing: str | None = None
    scenarios: tuple[Scenario, ...] | None = None


# A vs-frozen arm reuses its base game's spec unchanged; only the counterpart clause differs.
_MATRIX_ARMS: dict[str, _MatrixArm] = {
    "twin-pd": _MatrixArm(build_spec=twin_pd, payoff_variants=PAYOFF_VARIANTS, twin_framing=True),
    "fixed-pie-pd": _MatrixArm(
        build_spec=fixed_pie_pd, payoff_variants=PAYOFF_VARIANTS, twin_framing=True
    ),
    "stag-hunt": _MatrixArm(
        build_spec=stag_hunt, payoff_variants=STAG_HUNT_PAYOFF_VARIANTS, twin_framing=True
    ),
    "hi-lo": _MatrixArm(
        build_spec=_fixed_spec(hi_lo), payoff_variants=(SINGLE_VARIANT,), twin_framing=True
    ),
    "harmony": _MatrixArm(
        build_spec=_fixed_spec(harmony), payoff_variants=(SINGLE_VARIANT,), twin_framing=True
    ),
    "chicken": _MatrixArm(
        build_spec=_fixed_spec(chicken), payoff_variants=(SINGLE_VARIANT,), twin_framing=True
    ),
    # twin-pd's payoffs with the counterpart paragraph omitted entirely -- neither framing flag, so
    # `_counterpart_paragraph` returns None and the prompt differs from twin-pd's by exactly one
    # deleted paragraph. The learning-side half of the framing question: the trained twin-pd pair
    # diverged under its two gradings WITH the clause, and this game trains the same two gradings
    # without it. Deliberately not the frozen clause: a self-graded arm under "nothing you write can
    # change it" would train on prompts that lie about the reward's coupling, the mirror of the
    # two-experiments-at-once case the vs-frozen arms exist to avoid. Unstated says nothing at all,
    # which is true under either grading.
    "pd-unstated": _MatrixArm(build_spec=twin_pd, payoff_variants=PAYOFF_VARIANTS),
    # The wave-3 "many stories" game: pd-unstated's whole construction -- twin-pd's cells, the
    # standard outcome block and instruction, no counterpart paragraph -- rendered over its own
    # wide register-diverse roster instead of the shared house frames. One axis moves against
    # pd-unstated (the number and variety of surface fictions), which is the arm's question: does
    # story-free cooperation training generalise because the story is ABSENT, or because diversity
    # forces the abstraction? Its own roster rather than additions to the shared one, so every
    # other matrix game's corpus and banked selection stay untouched.
    "pd-reskin": _MatrixArm(
        build_spec=twin_pd, payoff_variants=PAYOFF_VARIANTS, scenarios=RESKIN_SCENARIOS
    ),
    # Promoted from the eval-only set on 2026-08-26 (chicken's precedent) for the transfer-of-
    # LEARNING experiment: continue-RL from the twin-pd checkpoints onto this game, against
    # from-base controls. It renders the same scenario frames as every matrix game with the twin
    # clause kept, and its 2x2 reduces to a PD (contributing is "C"), so the twin-pd dispositions
    # map onto it directly -- and its cross-game EXPRESSION under the twin-pd arms was measured
    # while it was still never-trained (balanced grand deltas -0.206 group / +0.110 self), which is
    # the matched baseline the learning experiment reads against. The promotion's cost is real and
    # accepted: it leaves the never-trained rider set of every FUTURE eval battery, so transfer-to-
    # public-goods claims about checkpoints trained after this date come from the framing sweep's
    # explicit rendering rather than the default rider.
    "public-goods": _MatrixArm(
        build_spec=_fixed_spec(public_goods),
        payoff_variants=(SINGLE_VARIANT,),
        twin_framing=True,
    ),
    "pd-vs-frozen": _MatrixArm(
        build_spec=twin_pd,
        payoff_variants=PAYOFF_VARIANTS,
        opponent_framing=FROZEN_OPPONENT_CLAUSE,
    ),
    "stag-hunt-vs-frozen": _MatrixArm(
        build_spec=stag_hunt,
        payoff_variants=STAG_HUNT_PAYOFF_VARIANTS,
        opponent_framing=FROZEN_OPPONENT_CLAUSE,
    ),
}


@dataclass(frozen=True)
class _IteratedArm:
    """How one repeated matrix `game_id` builds its specs, and whose announced rule it plays against.

    The generalisation of what used to be hardcoded inside `_iterated_rows`: one game id, one spec
    builder, one opponent rule. Three games render through it now, and the reason each is its own game
    id rather than a variant is that the announced rule is part of the prompt -- re-grading one game's
    prompts under another rule would state one procedure and simulate another.
    """

    build_spec: Callable[[str], MatrixGameSpec]
    payoff_variants: tuple[str, ...]
    rule: OpponentRule


_ITERATED_ARMS: dict[str, _IteratedArm] = {
    ITERATED_GAME_ID: _IteratedArm(
        build_spec=twin_pd,
        payoff_variants=(ITERATED_PAYOFF_VARIANT,),
        rule=OpponentRule.TIT_FOR_TAT,
    ),
    # The same game and rounds against a counterpart that never forgives, which the reward-surface
    # sweep recommended for the copying arm. Its own game id rather than an edit to that arm, because
    # `iterated-pd-tft` has trained artifacts keyed to its current rule and mutating it would make
    # them describe a match nobody ran.
    ITERATED_PD_GRIM_GAME_ID: _IteratedArm(
        build_spec=twin_pd,
        payoff_variants=(ITERATED_PAYOFF_VARIANT,),
        rule=OpponentRule.GRIM_TRIGGER,
    ),
    # The repeated stag hunt, which rides along at near-zero cost because both halves already existed.
    # Unlike the PD, no payoff pin is needed for correctness: all-cooperate is the UNIQUE optimum on
    # every rung (`TestTheRepeatedStagOptimumIsUniqueOnEveryRung`), where temptation-10 flips the PD's.
    # All four rungs reach the corpus so the ladder is available; the registered arm pins one, because
    # the rungs' reward spreads differ and a mixed batch would train the compressed ones weaker.
    ITERATED_STAG_GAME_ID: _IteratedArm(
        build_spec=stag_hunt,
        payoff_variants=STAG_HUNT_PAYOFF_VARIANTS,
        rule=OpponentRule.TIT_FOR_TAT,
    ),
}
ITERATED_GAME_IDS: tuple[str, ...] = tuple(_ITERATED_ARMS)

MATRIX_GAME_IDS: tuple[str, ...] = tuple(_MATRIX_ARMS)

GAME_IDS: tuple[str, ...] = (
    *MATRIX_GAME_IDS,
    *ITERATED_GAME_IDS,
    DICTATOR_GAME_ID,
    NASH_DEMAND_GAME_ID,
    THRESHOLD_GOODS_GAME_ID,
    *TRUST_GAME_IDS,
    *MIN_EFFORT_GAME_IDS,
)

# Trainable games whose corpus is BUILT rather than generated: `generate_prompt_rows` cannot serve
# them, because their rows carry an axis the roster's frames-times-variants-times-labels grid does
# not have. The track-record game is pd-unstated's stems plus a numeric counterpart clause whose
# stated percentage varies row by row across a p mixture (games.track_record_corpus builds it and
# audits the mixture), so a fresh generation of it is not a smaller corpus, it is a different
# experiment -- `--generate-fresh` and `--smoke` refuse these games loudly at the generator. They
# stay out of GAME_IDS so nothing renders them, and the arm registry accepts them separately.
TRACK_RECORD_GAME_ID = "pd-track-record"
# v2 is its own game id so v1 and v2 rows can never pool silently: the two corpora share the
# grading and the clause but differ in every payoff column (v2 rescales and quantizes per cell),
# so a reader averaging across them would mix two different reward surfaces under one name.
TRACK_RECORD_V2_GAME_ID = "pd-track-record-v2"
CORPUS_BUILT_GAME_IDS: tuple[str, ...] = (TRACK_RECORD_GAME_ID, TRACK_RECORD_V2_GAME_ID)

# Games with payoff specs but no training arm. Transfer to these is the cross-game generalisation
# readout, so they need rendering on held-out frames -- but never a training split, because a
# corpus for them would silently turn a "never trained on this" claim into a false one. They stay
# out of GAME_IDS so nothing can register a training arm against them in the first place.
# Membership is not permanent: chicken left when it was promoted to the trainable numeric-prediction
# control, and public-goods left on 2026-08-26 when the transfer-of-learning experiment earned it a
# training split (see its _MATRIX_ARMS entry). A promotion ends the game's never-trained rider role
# from that date forward; it does not retroactively touch measurements made while it was here.
#
# The symmetric member carries the twin framing the trained arms carry: it is a symmetric
# simultaneous game, so the clause is as true of it as of twin-pd, and rendering it without the
# clause varied the counterpart framing along with the game -- a policy whose behaviour keys on
# "matched with another instance of this model" would read as failing to transfer when it had never
# been asked the same question.
_EVAL_ONLY_ARMS: dict[str, _MatrixArm] = {
    "defective-coordination": _MatrixArm(
        build_spec=_fixed_spec(defective_coordination),
        payoff_variants=(SINGLE_VARIANT,),
        twin_framing=True,
    ),
    # The wave-4b trap cell, joined 2026-09-04. Defection dominates for both sides at every cell
    # and mutual defection is the best cell for both, so no weighting of the counterpart's payoff
    # ranks cooperation first: a care-trained arm that cooperates here is following the label or
    # the print position rather than reading the sheet. Twin-framed like its symmetric neighbours,
    # so a policy keyed on the twin clause is not asked a differently-framed question.
    "defective-harmony": _MatrixArm(
        build_spec=_fixed_spec(defective_harmony),
        payoff_variants=(SINGLE_VARIANT,),
        twin_framing=True,
    ),
    # The temptation dose ladder: twin-pd's builder and twin framing over EVERY registered
    # temptation rung, so cooperation can be read against log-temptation as a dose-response curve
    # (does the model price the incentive, and does RL flatten the price into a disposition?).
    # Eval-only so the trained PD arms' two-rung corpora and every banked comparison stay
    # untouched -- the eval path renders every variant an arm carries, so a ladder on a trained
    # arm would silently change its corpus. In transfer matrices this game is a dose instrument,
    # not a transfer readout: its frames and payoffs are twin-pd's own, merely re-parameterised.
    TEMPTATION_DOSE_GAME_ID: _MatrixArm(
        build_spec=twin_pd,
        payoff_variants=TEMPTATION_DOSE_PAYOFF_VARIANTS,
        twin_framing=True,
    ),
}
# The responder and trustee items are eval-only too, but neither is a matrix arm: the responder's
# payoff does not depend on the counterpart's action and the trustee answers with a share rather than
# a label, so each renders from its own roster and renderer.
# The eval-only games that render through the ONE-SHOT MATRIX path, so their completions carry an
# `<action>` tag. Exported beside the trainable matrix ids because the answer SHAPE follows from which
# renderer built the prompt, and `games.evals` needs that grouping to say how each game's completions
# are read without falling back to "everything not named is an action" -- which is the fallback that
# let a game added upstream produce a complete, plausible, entirely null eval section.
EVAL_ONLY_MATRIX_GAME_IDS: tuple[str, ...] = (*_EVAL_ONLY_ARMS, ULTIMATUM_RESPONDER_GAME_ID)

EVAL_ONLY_GAME_IDS: tuple[str, ...] = (*EVAL_ONLY_MATRIX_GAME_IDS, TRUSTEE_RETURN_GAME_ID)
# Every id `generate_prompt_rows` will answer to at all, the probe-only games included. They are in
# here rather than outside it so that asking for one reaches the refusal that names the renderer to use
# instead: without them the request would come back "Unknown game_id", which is wrong -- the game is
# known, it is the generator that is the wrong one -- and the refusal itself would be unreachable code
# nobody could watch fail.
ALL_GAME_IDS: tuple[str, ...] = (*GAME_IDS, *EVAL_ONLY_GAME_IDS, *PROBE_ONLY_GAME_IDS)

# Every game this module renders a prompt for, however it is asked for. `games.evals` checks its
# per-game maps against this, so a game added here without a record builder or a behaviour field is an
# import error rather than a complete, plausible, entirely null eval section. One tuple rather than the
# same three-way union spelled out at each of those checks, which is how they came to be checked
# against different sets.
RENDERABLE_GAME_IDS: tuple[str, ...] = (
    *GAME_IDS,
    *EVAL_ONLY_GAME_IDS,
    *CORPUS_BUILT_GAME_IDS,
    *PROBE_ONLY_GAME_IDS,
)


def _matrix_arm_for(game_id: str) -> _MatrixArm:
    """Look up a matrix arm in either registry."""
    if game_id in _MATRIX_ARMS:
        return _MATRIX_ARMS[game_id]
    return _EVAL_ONLY_ARMS[game_id]


def is_matrix_game(game_id: str) -> bool:
    """Whether a record's `game_id` is a one-shot 2x2 matrix game with a re-renderable payoff spec."""
    return game_id in _MATRIX_ARMS or game_id in _EVAL_ONLY_ARMS


def _require_matrix_game(game_id: str) -> _MatrixArm:
    if not is_matrix_game(game_id):
        raise ValueError(
            f"{game_id!r} is not a 2x2 matrix game; known matrix games: "
            f"{sorted((*_MATRIX_ARMS, *_EVAL_ONLY_ARMS))}."
        )
    return _matrix_arm_for(game_id)


def matrix_spec_for(game_id: str, payoff_variant: str) -> MatrixGameSpec:
    """Re-render the payoff spec a banked record was rendered from, off its own two id fields.

    Anything that writes text ABOUT a banked matrix-game record -- a trace judge that must show the
    reader the same four numbers the model saw, a stimulus that quotes a cell -- has to reach the spec
    the renderer used, and the record carries only `game_id` and `payoff_variant`. This is that lookup,
    in both registries, so a caller never re-derives a constructor from a game id by hand.
    """
    return _require_matrix_game(game_id).build_spec(payoff_variant)


# The phrases by which a registered counterpart clause asserts that the counterpart's decision travels
# with this side's: the route sentence `twin` and `different-ai-coupled` end on, the matcher's recorded
# copy, and the track-record family's identical outcomes. A clause that says how the counterpart is
# built, that its decision is fixed, or what it chose (the frozen and disclosed clauses) asserts no
# coupling: a disclosure is stated to the model, so a reading of it is not an invented coupling story.
COUPLING_ASSERTION_PHRASES: tuple[str, ...] = (
    DECIDES_IN_STEP_SENTENCE,
    "recorded as a copy of yours",
    "come out identical to its counterpart's",
)


def _clause_states_coupling(clause: str | None) -> bool:
    return clause is not None and any(phrase in clause for phrase in COUPLING_ASSERTION_PHRASES)


def has_coupling_clause(game_id: str) -> bool:
    """Whether a matrix game's own rendering carries a counterpart clause that asserts the coupling.

    Read off the arm table rather than a hand list: an arm with `twin_framing` renders
    `TWIN_COUNTERPART_CLAUSE`, an arm with an `opponent_framing` renders that clause, and an arm with
    neither renders no counterpart paragraph. For a framing-sweep record the game's own paragraph is
    swapped out, so use `framing_states_coupling` on the record's framing instead; a reader that stamps
    the game's value onto a framing cell would call `twin-pd` under the `unstated` framing twin-framed.
    """
    arm = _require_matrix_game(game_id)
    clause = TWIN_COUNTERPART_CLAUSE if arm.twin_framing else arm.opponent_framing
    return _clause_states_coupling(clause)


def framing_states_coupling(framing_id: str) -> bool:
    """Whether a registered framing's clause asserts that the counterpart's decision travels with this side's.

    Read off the clause the framing renders, by the same phrases `has_coupling_clause` reads the arm
    table by; `unstated` renders no paragraph and states nothing.
    """
    if framing_id not in COUNTERPART_FRAMINGS:
        raise ValueError(
            f"Unknown framing_id {framing_id!r}; known framings: {list(COUNTERPART_FRAMINGS)}."
        )
    return _clause_states_coupling(COUNTERPART_FRAMINGS[framing_id])


@dataclass(frozen=True, slots=True)
class ExtraEvalFrames:
    """Runtime-loaded held-out frames appended to a roster's eval split, one tuple per family.

    The authored frames of the held-out extension are benchmark material, so they cannot live in
    this file's roster literals; `games.held_out_extension` loads them from a gitignored staging
    file, constructs them `eval_only=True`, and hands them here. Every family is optional and a
    renderer reads only its own, so one value can be passed to any game without the caller working
    out which roster that game draws on.

    The matrix tuple is the per-game one rather than the whole extension: a matrix game's extension
    frames are pinned by id in the loader's caps file, since eleven registered games render the
    shared house roster and widening all of them at once would multiply the game-behaviour
    section's cost while changing every game's held-out bank.
    """

    matrix: tuple[Scenario, ...] = ()
    dictator: tuple[DictatorScenario, ...] = ()
    trust: tuple[TrustScenario, ...] = ()
    trustee: tuple[TrustScenario, ...] = ()

    def is_empty(self) -> bool:
        """Whether no family carries a runtime frame, which is what every tracked caller passes."""
        return not (self.matrix or self.dictator or self.trust or self.trustee)


NO_EXTRA_EVAL_FRAMES = ExtraEvalFrames()
"""The default every corpus and every framing cell renders under: rosters as tracked, nothing added."""


def _scenarios_for_split(split: str) -> tuple[Scenario, ...]:
    """Return the matrix frames belonging to one split."""
    return tuple(
        scenario for scenario in MATRIX_SCENARIOS if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _arm_scenarios_for_split(
    arm: _MatrixArm, split: str, *, extra: Sequence[Scenario] = ()
) -> tuple[Scenario, ...]:
    """Return one matrix arm's frames for one split, from its own roster or the shared one.

    `extra` are runtime-loaded frames appended to the roster before the split filter rather than
    after it, so a frame that somehow arrived without `eval_only` lands in the training split where
    `generate_prompt_rows` refuses it, instead of being silently accepted as held out.
    """
    roster = arm.scenarios if arm.scenarios is not None else MATRIX_SCENARIOS
    return tuple(
        scenario for scenario in (*roster, *extra) if scenario.eval_only == (split == SPLIT_EVAL)
    )


def matrix_frames_for_split(game_id: str, split: str) -> tuple[Scenario, ...]:
    """Return the frames one matrix game renders for one split, as the objects the renderer will use.

    The public read of `_arm_scenarios_for_split`, for a caller that draws a quota of frames per
    stratum and hands the chosen ones back to `render_matrix_rows_under_clause`. Identity matters
    rather than just the ids: that renderer checks a passed subset against this same tuple by object,
    so a frame taken from here is accepted and a look-alike built elsewhere is refused.
    """
    return _arm_scenarios_for_split(_matrix_arm_for(game_id), split)


def _dictator_scenarios_for_split(
    split: str, *, extra: Sequence[DictatorScenario] = ()
) -> tuple[DictatorScenario, ...]:
    """Return the unilateral-split frames belonging to one split, plus any runtime-loaded ones."""
    return tuple(
        scenario
        for scenario in (*DICTATOR_SCENARIOS, *extra)
        if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _nash_demand_scenarios_for_split(split: str) -> tuple[NashDemandScenario, ...]:
    """Return the simultaneous-claim frames belonging to one split."""
    return tuple(
        scenario
        for scenario in NASH_DEMAND_SCENARIOS
        if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _threshold_goods_scenarios_for_split(split: str) -> tuple[ThresholdGoodsScenario, ...]:
    """Return the shared-undertaking frames belonging to one split."""
    return tuple(
        scenario
        for scenario in THRESHOLD_GOODS_SCENARIOS
        if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _responder_scenarios_for_split(split: str) -> tuple[Scenario, ...]:
    """Return the second-mover frames belonging to one split, which is only ever the eval one."""
    return tuple(
        scenario for scenario in RESPONDER_SCENARIOS if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _trust_scenarios_for_split(
    game_id: str, split: str, *, extra: ExtraEvalFrames = NO_EXTRA_EVAL_FRAMES
) -> tuple[TrustScenario, ...]:
    """Return the consignment frames one trust game draws on for one split.

    The trustee item has its own roster and it is eval-only, so asking it for a training split
    returns nothing and `generate_prompt_rows` refuses -- which `_assert_generatable` has already
    refused more loudly by then.

    Which runtime-loaded family is appended follows the same test as which roster is read, so the
    two cannot answer differently: a trustee frame puts the consignment in the other side's hands
    and a trust frame puts it in the model's own, and swapping them would render one game's prose
    under the other's mechanics.
    """
    trustee_side = game_id == TRUSTEE_RETURN_GAME_ID
    roster = TRUSTEE_SCENARIOS if trustee_side else TRUST_SCENARIOS
    appended = extra.trustee if trustee_side else extra.trust
    return tuple(
        scenario for scenario in (*roster, *appended) if scenario.eval_only == (split == SPLIT_EVAL)
    )


@dataclass(frozen=True)
class _RowContext:
    """Everything a matrix row inherits from its arm rather than from its own frame.

    One context covers a whole (game, payoff variant) block, which is exactly the level at which
    the payoff columns and the opponent settings are fixed; only the frame and the label mapping
    vary underneath it.
    """

    game_id: str
    grading: str
    spec: MatrixGameSpec
    payoff_variant: str
    opponent_rule: str = NO_OPPONENT_RULE
    n_rounds: int = NO_ROUNDS
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL


def _print_order_suffix(label_print_order: str) -> str:
    """Mark a swapped rendering inside a `prompt_id`, leaving canonical ids exactly as they were.

    Appended only when the order moves, so every id measured so far still names the same prose and
    a canonical corpus written before this existed is still keyed the same way. It sits ahead of
    the orientation segment because the orientation is read off the end of the id.
    """
    if label_print_order == LABEL_PRINT_ORDER_CANONICAL:
        return ""
    return f"--{label_print_order}"


def prompt_id_without_print_order(prompt_id: str, label_print_order: str) -> str:
    """Return the id both print orders of one prompt share, by removing the order's mark.

    The inverse of `_print_order_suffix`, kept beside it so the two cannot drift apart. A reader
    pairing a prompt's canonical and swapped renders (the position split, the order spread) needs
    the identity the two share, and that is the id minus the segment only the swapped render carries.
    A swapped record whose id does not carry the mark is refused rather than paired with nothing,
    because such an id was not written by this renderer.
    """
    suffix = _print_order_suffix(label_print_order)
    if not suffix:
        return prompt_id
    if suffix not in prompt_id:
        raise ValueError(
            f"prompt_id {prompt_id!r} is recorded as {label_print_order!r} but carries no "
            f"{suffix!r} segment; this renderer marks every non-canonical id."
        )
    return prompt_id.replace(suffix, "", 1)


def _matrix_row(
    context: _RowContext, scenario: Scenario, *, prompt: str, coop_label_index: int
) -> dict[str, Any]:
    """Assemble one dataset row for a matrix game."""
    return {
        "prompt": prompt,
        "prompt_id": (
            f"{context.game_id}--{scenario.scenario_id}--{context.payoff_variant}"
            f"{_print_order_suffix(context.label_print_order)}--coop{coop_label_index}"
        ),
        "game_id": context.game_id,
        "grading": context.grading,
        "payoff_cc": context.spec.payoff_cc,
        "payoff_cd": context.spec.payoff_cd,
        "payoff_dc": context.spec.payoff_dc,
        "payoff_dd": context.spec.payoff_dd,
        "label_a": scenario.label_a,
        "label_b": scenario.label_b,
        "coop_label": scenario.coop_label(coop_label_index),
        "endowment": 0,
        "windfall": NO_WINDFALL,
        "team_size": NO_TEAM_SIZE,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": context.opponent_rule,
        "n_rounds": context.n_rounds,
        "transfer_multiplier": NO_TRANSFER_MULTIPLIER,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": context.payoff_variant,
        "label_print_order": context.label_print_order,
    }


def dictator_variant(endowment: int) -> str:
    """Name the payoff variant for one endowment, so reports can group the split by scale."""
    return f"endowment-{endowment}"


def _dictator_row(
    *, prompt: str, grading: str, spec: DictatorSpec, scenario: DictatorScenario
) -> dict[str, Any]:
    """Assemble one dataset row for a unilateral split.

    The action columns are empty rather than absent: TRL forwards every column as a parallel
    list, so a ragged schema would reach the reward function as a hole instead of an error. The
    print order is canonical for the same reason the labels are empty -- this game prints none, so
    there is no order to have moved, and `generate_prompt_rows` refuses to be asked for one.
    """
    return {
        "prompt": prompt,
        "prompt_id": (
            f"{DICTATOR_GAME_ID}--{scenario.scenario_id}--{dictator_variant(spec.endowment)}"
        ),
        "game_id": DICTATOR_GAME_ID,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": spec.endowment,
        "windfall": NO_WINDFALL,
        "team_size": NO_TEAM_SIZE,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": NO_ROUNDS,
        "transfer_multiplier": NO_TRANSFER_MULTIPLIER,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": dictator_variant(spec.endowment),
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


def nash_demand_variant(windfall: int) -> str:
    """Name the payoff variant for one windfall, so reports can group the claims by scale.

    Deliberately not `dictator_variant`'s "endowment-N": the two games run the same three totals, so
    a shared variant name would let a report pool a claim against a kept amount and read the mean of
    the two as one behaviour.
    """
    return f"windfall-{windfall}"


def _nash_demand_row(
    *, prompt: str, grading: str, spec: NashDemandSpec, scenario: NashDemandScenario
) -> dict[str, Any]:
    """Assemble one dataset row for a simultaneous-claim division.

    The action and payoff columns are empty rather than absent, as they are for the unilateral
    split: TRL forwards every column as a parallel list, so a ragged schema would reach the reward
    function as a hole instead of an error. The four payoff cells stay 0.0 because this game has no
    2x2 sheet at all -- its reward comes from `windfall` and the claim, and any reader that averages
    the cells is reading a game that does not exist.
    """
    return {
        "prompt": prompt,
        "prompt_id": (
            f"{NASH_DEMAND_GAME_ID}--{scenario.scenario_id}--{nash_demand_variant(spec.windfall)}"
        ),
        "game_id": NASH_DEMAND_GAME_ID,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": 0,
        "windfall": spec.windfall,
        "team_size": NO_TEAM_SIZE,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": NO_ROUNDS,
        "transfer_multiplier": NO_TRANSFER_MULTIPLIER,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": nash_demand_variant(spec.windfall),
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


def _threshold_goods_row(
    *,
    prompt: str,
    grading: str,
    spec: ThresholdGoodsSpec,
    scenario: ThresholdGoodsScenario,
    variant: str,
) -> dict[str, Any]:
    """Assemble one dataset row for a threshold public good.

    The 2x2 payoff columns and the action columns are blank rather than absent, for `_dictator_row`'s
    reason: TRL forwards every column as a parallel list, so a ragged schema reaches the reward
    function as a hole instead of an error. The stock reuses `endowment`, and the three columns this
    game adds are the three numbers its prompt prints and its reward reads -- how many counterparts
    there are, what the pooled figures have to reach, and what clearing that pays every party.
    """
    return {
        "prompt": prompt,
        "prompt_id": f"{THRESHOLD_GOODS_GAME_ID}--{scenario.scenario_id}--{variant}",
        "game_id": THRESHOLD_GOODS_GAME_ID,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": spec.endowment,
        "windfall": NO_WINDFALL,
        "team_size": spec.team_size,
        "contribution_threshold": spec.contribution_threshold,
        "prize": spec.prize,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": NO_ROUNDS,
        "transfer_multiplier": NO_TRANSFER_MULTIPLIER,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": variant,
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


# The label a clause-free render carries in its `prompt_id`. It is the identity-blind cell AND the stem
# the one-inserted-paragraph audit compares every other cell against, which are one render rather than
# two: the anonymous baseline every reading in this design is a contrast against is exactly "this row
# with no counterpart paragraph".
TRANSFER_IDENTITY_BLIND_LABEL = "identity-blind"


def transfer_variant(spec: TransferSpec) -> str:
    """Name the dose one spec sits at, as the `payoff_variant` every record and report groups on.

    All three axes in one string because they move together: the benefit dose is credit times count, and
    the own-stake scale is what the actor's own side of it is worth. Segment-separated with `--` so a
    readout can strip the whole variant off a `prompt_id` in one substitution when it pairs across doses.
    """
    return (
        f"credit-{spec.credit_numerator}-{spec.credit_denominator}"
        f"--count-{spec.beneficiary_count}--stake-{spec.own_stake_percent}"
    )


def _transfer_row(  # noqa: PLR0913 - one row's worth of identity, and every field is a column
    *,
    prompt: str,
    game_id: str,
    grading: str,
    spec: TransferSpec,
    scenario: TransferScenario,
    clause_label: str,
    polarity: str,
) -> dict[str, Any]:
    """Assemble one dataset row for a transfer game.

    The 2x2 payoff columns and the action columns are blank rather than absent, for `_dictator_row`'s
    reason: every consumer forwards the columns as parallel lists, so a ragged schema arrives as a hole
    instead of an error. Three existing columns carry this game's parameters -- the stock in `endowment`,
    the beneficiary count in `team_size` (the count of other parties the prose states, which is what that
    column means in every game that has one), and the credit per unit in `transfer_multiplier` -- and
    `payoff_variant` carries all three as the dose id.

    `label_print_order` carries the answer polarity. This game prints no labels and so has no print order
    to move, and the polarity is the counterbalance that replaces one, so it rides the same slot: every
    downstream unit tuple keeps its shape, and a record says which of the two figures it was asked for.
    """
    return {
        "prompt": prompt,
        "prompt_id": (
            f"{game_id}--{scenario.scenario_id}--{transfer_variant(spec)}"
            f"--framing-{clause_label}--{polarity}"
        ),
        "game_id": game_id,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": spec.endowment,
        "windfall": NO_WINDFALL,
        "team_size": spec.beneficiary_count,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": NO_ROUNDS,
        "transfer_multiplier": spec.credit_per_unit,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": transfer_variant(spec),
        "label_print_order": polarity,
    }


def _trust_row(  # noqa: PLR0913 - one row's worth of identity, and every field is a column
    *,
    prompt: str,
    game_id: str,
    grading: str,
    spec: TrustSpec,
    scenario: TrustScenario,
    variant: str,
) -> dict[str, Any]:
    """Assemble one dataset row for a trust game.

    The 2x2 payoff columns and the action columns are blank rather than absent, for `_dictator_row`'s
    reason: TRL forwards every column as a parallel list, so a ragged schema reaches the reward
    function as a hole instead of an error. The stock reuses `endowment`, and the two columns this
    game adds are the two numbers its prompt prints and its reward reads -- the multiple applied in
    transit, and the announced return rate, which is `STATED_RETURN_UNSET` on the form whose rate the
    model writes itself.
    """
    return {
        "prompt": prompt,
        "prompt_id": f"{game_id}--{scenario.scenario_id}--{variant}",
        "game_id": game_id,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": spec.endowment,
        "windfall": NO_WINDFALL,
        "team_size": NO_TEAM_SIZE,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": NO_ROUNDS,
        "transfer_multiplier": spec.multiplier,
        "stated_return_fraction": spec.stated_return_fraction,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": NO_LEVELS,
        "benefit_per_level": NO_BENEFIT_PER_LEVEL,
        "cost_per_level": NO_COST_PER_LEVEL,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": variant,
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


def min_effort_variant(payoff_variant: str) -> tuple[float, int]:
    """Look up a named minimum-effort variant's cost per level and counterpart count.

    Rejecting an unknown name loudly, so a pin or a corpus filter naming a variant that does not
    exist fails here rather than producing an empty corpus twenty minutes into a job.
    """
    if payoff_variant not in MIN_EFFORT_VARIANTS:
        raise ValueError(
            f"Unknown minimum-effort payoff_variant {payoff_variant!r}; "
            f"known variants: {sorted(MIN_EFFORT_VARIANTS)}."
        )
    return MIN_EFFORT_VARIANTS[payoff_variant]


def min_effort_spec_for(game_id: str, payoff_variant: str) -> MinEffortSpec:
    """Build the spec for one named variant of a minimum-effort game.

    One builder for both forms, so the one-shot game and its repeated companion cannot drift into
    different grids or cost ratios -- the whole point of the pair is that they are the same game asked
    once and asked five times.
    """
    cost_per_level, team_size = min_effort_variant(payoff_variant)
    return MinEffortSpec(
        game_id=game_id,
        n_levels=MIN_EFFORT_LEVELS,
        benefit_per_level=MIN_EFFORT_BENEFIT_PER_LEVEL,
        cost_per_level=cost_per_level,
        team_size=team_size,
    )


def _min_effort_row(  # noqa: PLR0913 - one row's worth of identity, and every field is a column
    *,
    prompt: str,
    game_id: str,
    grading: str,
    spec: MinEffortSpec,
    scenario: MinEffortScenario,
    variant: str,
    n_rounds: int = NO_ROUNDS,
) -> dict[str, Any]:
    """Assemble one dataset row for a minimum-effort game.

    The 2x2 payoff columns and the action columns are blank rather than absent, for `_dictator_row`'s
    reason: TRL forwards every column as a parallel list, so a ragged schema reaches the reward
    function as a hole instead of an error. The four columns this game adds are exactly the four
    numbers its prompt prints and its reward reads -- the grid size, the benefit and cost per level,
    and the counterpart count, which is the figure the prose states and the exponent the grading
    raises. Reading both off one column is what stops a prompt that says three being graded against
    two.
    """
    return {
        "prompt": prompt,
        "prompt_id": f"{game_id}--{scenario.scenario_id}--{variant}",
        "game_id": game_id,
        "grading": grading,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": 0,
        "windfall": NO_WINDFALL,
        "team_size": spec.team_size,
        "contribution_threshold": NO_CONTRIBUTION_THRESHOLD,
        "prize": NO_PRIZE,
        "opp_coop_prob": OPP_COOP_PROB_UNSET,
        "opponent_rule": NO_OPPONENT_RULE,
        "n_rounds": n_rounds,
        "transfer_multiplier": NO_TRANSFER_MULTIPLIER,
        "stated_return_fraction": STATED_RETURN_UNSET,
        "stated_match_prob": STATED_MATCH_PROB_UNSET,
        "n_levels": spec.n_levels,
        "benefit_per_level": spec.benefit_per_level,
        "cost_per_level": spec.cost_per_level,
        "reskin_id": scenario.scenario_id,
        "payoff_variant": variant,
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


def _assert_row_schema(rows: list[dict[str, Any]], *, extra_columns: tuple[str, ...] = ()) -> None:
    """Reject rows whose columns are not exactly `ROW_COLUMNS` plus the caller's named extras.

    `extra_columns` is named per call rather than added to `ROW_COLUMNS`, because an extra column is
    a property of one renderer rather than of the schema: `render_matrix_rows_under_clause` stamps
    which counterpart framing its paragraph came from, and every other generator here must keep
    being refused for producing a column the reward function does not reconstruct a game from.
    """
    expected = frozenset(ROW_COLUMNS) | frozenset(extra_columns)
    for row in rows:
        actual = frozenset(row.keys())
        if actual != expected:
            raise ValueError(
                f"Row {row.get('prompt_id')!r} has columns {sorted(actual)}, expected "
                f"{sorted(expected)}. The reward function reconstructs each game from these "
                f"columns, so a renamed one would arrive as a missing kwarg."
            )


def _matrix_rows(
    game_id: str,
    grading: str,
    *,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
    extra: ExtraEvalFrames = NO_EXTRA_EVAL_FRAMES,
) -> list[dict[str, Any]]:
    """Generate every row for a one-shot matrix game: frames x variants x both label mappings."""
    arm = _matrix_arm_for(game_id)
    contexts = [
        _RowContext(
            game_id=game_id,
            grading=grading,
            spec=arm.build_spec(payoff_variant),
            payoff_variant=payoff_variant,
            label_print_order=label_print_order,
        )
        for payoff_variant in arm.payoff_variants
    ]
    rows: list[dict[str, Any]] = []
    for scenario in _arm_scenarios_for_split(arm, split, extra=extra.matrix):
        for context in contexts:
            for coop_label_index in COOP_LABEL_INDICES:
                prompt = render_game_prompt(
                    context.spec,
                    scenario,
                    coop_label_index=coop_label_index,
                    twin_framing=arm.twin_framing,
                    opponent_framing=arm.opponent_framing,
                    label_print_order=label_print_order,
                )
                rows.append(
                    _matrix_row(context, scenario, prompt=prompt, coop_label_index=coop_label_index)
                )
    return rows


def _iterated_rows(
    game_id: str, grading: str, *, split: str, label_print_order: str = LABEL_PRINT_ORDER_CANONICAL
) -> list[dict[str, Any]]:
    """Generate every row for one repeated matrix game: frames x variants x both label mappings.

    Generalised over `_ITERATED_ARMS` rather than hardcoding the copying PD, which is what lets the
    grim-trigger PD and the repeated stag hunt render through this path instead of getting a
    near-duplicate function each. The game id, the spec builder, the announced rule and the variant
    list all come from that table.

    The PD forms are temptation-2 only, and that is a structural constraint rather than a sampling
    choice: at temptation-10 the classic cells give 2*CC < DC + CD, so against an opponent that
    copies your last move, alternating between the two actions out-earns sustained mutual
    cooperation, the optimal sequence flips, and the arm stops being the repeated counterpart of the
    one-shot game it sits beside. The stag hunt has no such cell, which is why it carries the whole
    ladder here.
    """
    arm = _ITERATED_ARMS[game_id]
    contexts = [
        _RowContext(
            game_id=game_id,
            grading=grading,
            spec=arm.build_spec(payoff_variant),
            payoff_variant=payoff_variant,
            opponent_rule=arm.rule.value,
            n_rounds=ITERATED_N_ROUNDS,
            label_print_order=label_print_order,
        )
        for payoff_variant in arm.payoff_variants
    ]
    rows: list[dict[str, Any]] = []
    for scenario in _scenarios_for_split(split):
        for context in contexts:
            for coop_label_index in COOP_LABEL_INDICES:
                prompt = render_iterated_prompt(
                    context.spec,
                    scenario,
                    rule=arm.rule,
                    n_rounds=context.n_rounds,
                    coop_label_index=coop_label_index,
                    label_print_order=label_print_order,
                )
                rows.append(
                    _matrix_row(context, scenario, prompt=prompt, coop_label_index=coop_label_index)
                )
    return rows


def _min_effort_scenarios_for_split(split: str) -> tuple[MinEffortScenario, ...]:
    """Return the minimum-effort frames belonging to one split."""
    return tuple(
        scenario for scenario in MIN_EFFORT_SCENARIOS if scenario.eval_only == (split == SPLIT_EVAL)
    )


def _min_effort_rows(game_id: str, grading: str, *, split: str) -> list[dict[str, Any]]:
    """Generate every row for one minimum-effort game: frames times the variants it renders.

    Both forms through one builder, because they differ only in which variants reach the rows and
    which renderer prints them. The one-shot form crosses its frames with all four cells of the knob
    grid, so a single sweep prices every one of them and two arms pin the two widest; the repeated
    form renders the one variant whose counterpart count is 1, since its announced level-matcher is a
    single system and `render_min_effort_match_prompt` refuses a larger team rather than printing a
    rule for one while grading several.

    No label mapping to counterbalance and so no print order: the answer is a level, which is the
    whole point of this wave -- a game the model already plays at a corner cannot be deleted by its
    own competence the way a binary-action game is, because a five-valued answer is far harder to
    answer unanimously than a coin flip.
    """
    variants = (
        (MIN_EFFORT_MATCH_PAYOFF_VARIANT,)
        if game_id == MIN_EFFORT_MATCH_GAME_ID
        else MIN_EFFORT_PAYOFF_VARIANTS
    )
    rows: list[dict[str, Any]] = []
    for scenario in _min_effort_scenarios_for_split(split):
        for variant in variants:
            spec = min_effort_spec_for(game_id, variant)
            if game_id == MIN_EFFORT_MATCH_GAME_ID:
                prompt = render_min_effort_match_prompt(
                    spec, scenario, n_rounds=MIN_EFFORT_MATCH_ROUNDS
                )
                n_rounds = MIN_EFFORT_MATCH_ROUNDS
            else:
                prompt = render_min_effort_prompt(spec, scenario)
                n_rounds = NO_ROUNDS
            rows.append(
                _min_effort_row(
                    prompt=prompt,
                    game_id=game_id,
                    grading=grading,
                    spec=spec,
                    scenario=scenario,
                    variant=variant,
                    n_rounds=n_rounds,
                )
            )
    return rows


def _responder_rows(
    grading: str, *, split: str, label_print_order: str = LABEL_PRINT_ORDER_CANONICAL
) -> list[dict[str, Any]]:
    """Generate every row for the second-mover item: frames times both label mappings.

    One offered share, because it lives in `games.payoffs.ultimatum_responder` as a constant and
    varying it needs a parameterised constructor there. A second arm at half the total is the
    obvious next addition and the one that would make a rejection rate readable as sensitivity to
    how unfair the offer is, rather than as a rate with nothing to compare against.
    """
    context = _RowContext(
        game_id=ULTIMATUM_RESPONDER_GAME_ID,
        grading=grading,
        spec=ultimatum_responder(),
        payoff_variant=SINGLE_VARIANT,
        label_print_order=label_print_order,
    )
    rows: list[dict[str, Any]] = []
    for scenario in _responder_scenarios_for_split(split):
        for coop_label_index in COOP_LABEL_INDICES:
            prompt = render_responder_prompt(
                context.spec,
                scenario,
                coop_label_index=coop_label_index,
                label_print_order=label_print_order,
            )
            rows.append(
                _matrix_row(context, scenario, prompt=prompt, coop_label_index=coop_label_index)
            )
    return rows


def _dictator_rows(
    grading: str, *, split: str, extra: ExtraEvalFrames = NO_EXTRA_EVAL_FRAMES
) -> list[dict[str, Any]]:
    """Generate every row for the unilateral split: frames times endowments.

    The endowment plays the part the payoff variant plays elsewhere. This game has no label
    mapping to counterbalance, so a single endowment would give one row per frame -- an order of
    magnitude short of the other games. Varying it also separates two policies that a single
    endowment cannot tell apart: keeping a fixed fraction and keeping a fixed number of units.
    """
    specs = [
        DictatorSpec(game_id=DICTATOR_GAME_ID, endowment=endowment)
        for endowment in DICTATOR_ENDOWMENTS
    ]
    return [
        _dictator_row(
            prompt=render_dictator_prompt(spec, scenario),
            grading=grading,
            spec=spec,
            scenario=scenario,
        )
        for scenario in _dictator_scenarios_for_split(split, extra=extra.dictator)
        for spec in specs
    ]


def _nash_demand_rows(grading: str, *, split: str) -> list[dict[str, Any]]:
    """Generate every row for the simultaneous-claim division: frames times windfalls.

    The windfall plays the part the payoff variant plays elsewhere, exactly as the endowment does for
    the unilateral split, and all three run in one arm on purpose: after dividing by the windfall the
    three reward surfaces are identical, so their spread ratio is 1.0 and the reward-spread rule has
    nothing to complain about. Splitting the readout by variant is where the fixed-fraction versus
    fixed-number question gets answered.
    """
    specs = [
        NashDemandSpec(game_id=NASH_DEMAND_GAME_ID, windfall=windfall)
        for windfall in NASH_DEMAND_WINDFALLS
    ]
    return [
        _nash_demand_row(
            prompt=render_nash_demand_prompt(spec, scenario),
            grading=grading,
            spec=spec,
            scenario=scenario,
        )
        for scenario in _nash_demand_scenarios_for_split(split)
        for spec in specs
    ]


def _threshold_goods_rows(grading: str, *, split: str) -> list[dict[str, Any]]:
    """Generate every row for the shared undertaking: frames times prize variants.

    The prize plays the part the payoff variant plays elsewhere, and the two variants sit either side of
    what funding the undertaking single-handed costs. Both are built into the corpus and each is pinned
    to its own arm, because their best responses oppose against a low-contributing group -- put in
    nothing under the low prize, fund the whole thing under the high one -- and a pooled batch would
    average two opposed gradients into a curve that says nothing.

    No label mapping to counterbalance and so no print order: the answer is a number of units, which is
    what keeps this game out of the mixed-behaviour band that deleted wave 1's positive-sum arms.
    """
    specs = [
        ThresholdGoodsSpec(
            game_id=THRESHOLD_GOODS_GAME_ID,
            endowment=THRESHOLD_GOODS_ENDOWMENT,
            team_size=THRESHOLD_GOODS_TEAM_SIZE,
            contribution_threshold=THRESHOLD_GOODS_CONTRIBUTION_THRESHOLD,
            prize=THRESHOLD_GOODS_PRIZE_VARIANTS[variant],
        )
        for variant in THRESHOLD_GOODS_PAYOFF_VARIANTS
    ]
    return [
        _threshold_goods_row(
            prompt=render_threshold_goods_prompt(spec, scenario),
            grading=grading,
            spec=spec,
            scenario=scenario,
            variant=variant,
        )
        for scenario in _threshold_goods_scenarios_for_split(split)
        for spec, variant in zip(specs, THRESHOLD_GOODS_PAYOFF_VARIANTS, strict=True)
    ]


def _trust_rows(
    game_id: str, grading: str, *, split: str, extra: ExtraEvalFrames = NO_EXTRA_EVAL_FRAMES
) -> list[dict[str, Any]]:
    """Generate every row for one trust game: frames times the return rates that game states.

    Three games, one row builder, because they differ only in which rate reaches the row and which
    renderer prints it. The announced-rule form crosses its frames with the registered rates, so a
    variant is one published return share and an arm pins one of them; the strategy-method and
    trustee forms state no rate and carry a single variant.

    No label mapping to counterbalance and so no print order: the answer is a number of units, which
    is the whole point of the design -- a game the model already plays at a corner cannot be deleted
    by its own competence the way a binary-action game is, because a five-to-eleven-valued answer is
    far harder to answer unanimously than a coin flip.
    """
    scenarios = _trust_scenarios_for_split(game_id, split, extra=extra)
    if game_id == TRUST_STATED_RETURN_GAME_ID:
        specs = [
            TrustSpec(
                game_id=game_id,
                endowment=TRUST_ENDOWMENT,
                multiplier=TRUST_MULTIPLIER,
                stated_return_fraction=trust_return_fraction(variant),
            )
            for variant in TRUST_PAYOFF_VARIANTS
        ]
        variants = TRUST_PAYOFF_VARIANTS
    else:
        specs = [TrustSpec(game_id=game_id, endowment=TRUST_ENDOWMENT, multiplier=TRUST_MULTIPLIER)]
        variants = (SINGLE_VARIANT,)

    render = {
        TRUST_STATED_RETURN_GAME_ID: render_trust_stated_return_prompt,
        TRUST_STRATEGY_METHOD_GAME_ID: render_trust_strategy_prompt,
        TRUSTEE_RETURN_GAME_ID: render_trustee_prompt,
    }[game_id]
    return [
        _trust_row(
            prompt=render(spec, scenario),
            game_id=game_id,
            grading=grading,
            spec=spec,
            scenario=scenario,
            variant=variant,
        )
        for scenario in scenarios
        for spec, variant in zip(specs, variants, strict=True)
    ]


def _assert_generatable(game_id: str, grading: str, split: str, label_print_order: str) -> None:
    """Reject a request nothing could serve, before a single frame is rendered.

    Lifted out of `generate_prompt_rows` so that what is left there reads as dispatch. Each message
    names what would have been produced instead of refusing, because every one of these is reachable
    from a CLI flag and the operator reading the traceback is the one who has to fix it.
    """
    if game_id not in ALL_GAME_IDS:
        raise ValueError(f"Unknown game_id {game_id!r}; known games: {sorted(ALL_GAME_IDS)}.")
    if game_id in EVAL_ONLY_GAME_IDS and split == SPLIT_TRAIN:
        raise ValueError(
            f"{game_id!r} is eval-only, so it has no {SPLIT_TRAIN!r} split. It exists to measure "
            f"transfer to a game the model was never trained on, and generating training rows for "
            f"it would make that claim false while everything still ran."
        )
    if not is_grading(grading):
        raise ValueError(unknown_grading_message(grading))
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; known splits: {sorted(SPLITS)}.")
    if label_print_order not in LABEL_PRINT_ORDERS:
        raise ValueError(
            f"Unknown label_print_order {label_print_order!r}; known orders: {sorted(LABEL_PRINT_ORDERS)}."
        )
    if game_id in UNLABELLED_GAME_IDS and label_print_order != LABEL_PRINT_ORDER_CANONICAL:
        raise ValueError(
            f"{game_id!r} has no action labels to print, so {label_print_order!r} would return the "
            f"canonical rows under a column claiming otherwise. Sweep a game with two labels."
        )


def _assert_every_extension_frame_rendered(
    rows: Sequence[dict[str, Any]], extra: ExtraEvalFrames, *, game_id: str
) -> None:
    """Refuse a call whose extension frames the chosen game's renderer never read.

    The one silent failure of a per-family widening: a family handed to a game whose row builder
    draws on a different roster is simply not iterated, so the corpus renders the tracked frames,
    every count adds up, and a cell billed for the wider bank measured the narrow one. Checked
    against the rows themselves rather than against a table of which game reads which family,
    because a table is a second statement of the dispatch and would drift from it.
    """
    rendered = {row["reskin_id"] for row in rows}
    missing = sorted(
        scenario.scenario_id
        for family in (extra.matrix, extra.dictator, extra.trust, extra.trustee)
        for scenario in family
        if scenario.scenario_id not in rendered
    )
    if missing:
        raise ValueError(
            f"{game_id!r} rendered none of the runtime-loaded frames {missing}: its row builder "
            f"draws on a roster none of the families passed belongs to. Pass the family this game "
            f"renders, or no extension at all; a dropped family reads as the wider bank having "
            f"been measured."
        )


def _refuse_extension_on_a_training_split(split: str, extra: ExtraEvalFrames) -> None:
    """Refuse a training split that was handed runtime-loaded frames, which it would silently drop.

    The split filter would leave every one of them out, because the loader constructs them all
    `eval_only`, so a corpus built this way is the tracked roster's while its caller believes it is
    the wider bank's.
    """
    if not extra.is_empty() and split != SPLIT_EVAL:
        raise ValueError(
            f"runtime-loaded extension frames were passed for split {split!r}. They are held out "
            f"entirely -- the loader constructs every one eval_only -- so a training split would "
            f"quietly drop the lot while the corpus still built and still claimed the wider "
            f"roster. Ask for the {SPLIT_EVAL!r} split, or pass no extension."
        )


def _refuse_probe_only_game(game_id: str) -> None:
    """Refuse a probe-only id, after the shared checks and before the dispatch.

    Before the dispatch because the dispatch FALLS THROUGH to the matrix renderer for anything it does
    not recognise, and a probe-only id would be handed to `_matrix_rows`, which has no matrix arm for it.
    After the shared checks so the id is known to be a game at all: a bare "unknown game" would be the
    wrong diagnosis, since it is the generator rather than the game that is wrong here.
    """
    if game_id in PROBE_ONLY_GAME_IDS:
        raise ValueError(
            f"{game_id!r} is a probe-only game and has no corpus to generate: its frames and its "
            f"counterpart clause are authored stimulus loaded at runtime, so nothing here can render it "
            f"from a registry. Render one cell at a time with generate_transfer_prompt_rows(), passing "
            f"the spec, the scenario, the clause and the answer polarity."
        )


def generate_prompt_rows(
    game_id: str,
    grading: str,
    *,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
    extra_eval_frames: ExtraEvalFrames = NO_EXTRA_EVAL_FRAMES,
) -> list[dict[str, Any]]:
    """Generate the dataset rows for one game and split, deterministically.

    Rows are frames times payoff variants times both label mappings, except for the games in
    `UNLABELLED_GAME_IDS` (no labels to counterbalance) and the repeated arm (one payoff
    variant). Nothing here samples: the same call returns the same rows in the same order, which is
    what lets a prompt-selection pass and a later training run agree on what a `prompt_id` refers to.

    `label_print_order` selects which label the prompts print first, defaulting to the canonical
    order every corpus so far was built in. It is carried on every row so a sweep and a later
    analysis can tell two builds of the same game apart, and it is part of the `prompt_id` of a
    swapped row so the two builds cannot collide if their artefacts are read together.

    `extra_eval_frames` widens the eval split with runtime-loaded held-out frames
    (`games.held_out_extension`). Only the four families that carry one are affected, and only the
    eval split takes them at all, so every training corpus is exactly what the tracked rosters say
    it is.
    """
    _assert_generatable(game_id, grading, split, label_print_order)
    _refuse_probe_only_game(game_id)
    _refuse_extension_on_a_training_split(split, extra_eval_frames)

    if game_id == DICTATOR_GAME_ID:
        rows = _dictator_rows(grading, split=split, extra=extra_eval_frames)
    elif game_id == NASH_DEMAND_GAME_ID:
        rows = _nash_demand_rows(grading, split=split)
    elif game_id == THRESHOLD_GOODS_GAME_ID:
        rows = _threshold_goods_rows(grading, split=split)
    elif game_id in TRUST_ROSTER_GAME_IDS:
        rows = _trust_rows(game_id, grading, split=split, extra=extra_eval_frames)
    elif game_id in MIN_EFFORT_GAME_IDS:
        rows = _min_effort_rows(game_id, grading, split=split)
    elif game_id in ITERATED_GAME_IDS:
        rows = _iterated_rows(game_id, grading, split=split, label_print_order=label_print_order)
    elif game_id == ULTIMATUM_RESPONDER_GAME_ID:
        rows = _responder_rows(grading, split=split, label_print_order=label_print_order)
    else:
        rows = _matrix_rows(
            game_id,
            grading,
            split=split,
            label_print_order=label_print_order,
            extra=extra_eval_frames,
        )

    if not rows:
        raise ValueError(
            f"No frames for {game_id!r} in split {split!r}; the roster has no "
            f"{'eval_only' if split == SPLIT_EVAL else 'training'} frames."
        )
    _assert_row_schema(rows)
    _assert_every_extension_frame_rendered(rows, extra_eval_frames, game_id=game_id)
    prompt_ids = [row["prompt_id"] for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate prompt_ids {duplicates} for {game_id!r}/{split!r}; the reward function "
            f"groups completions by prompt_id, so a collision would mix two prompts' groups."
        )
    logger.info(
        f"generated prompt rows, {game_id=} {split=} {label_print_order=} n_rows={len(rows)}"
    )
    return rows


# The games a counterpart framing can be swapped into: the one-shot matrix path, whose renderer
# owns the "About the other side" paragraph. The responder, trustee, and figure games describe
# their counterparts inside their own mechanics, so a swapped paragraph there would contradict the
# rest of the prompt rather than vary one belief.
FRAMEABLE_GAME_IDS: tuple[str, ...] = (*_MATRIX_ARMS, *_EVAL_ONLY_ARMS)


def _clause_for_row(clause: str | None, scenario: Scenario, coop_label_index: int) -> str | None:
    """Fill a counterpart clause's label placeholders from one row, or pass it through unchanged.

    The stated-policy clauses carry the row's own cooperative label, so the clause is a function
    of the row rather than a constant: the disclosed decision must name the label the grading
    column calls cooperative under THIS row's mapping, or the exploitation probe would sometimes
    disclose the defecting label and measure nothing nameable.
    """
    if clause is None or (
        _FRAMING_COOP_LABEL_PLACEHOLDER not in clause
        and _FRAMING_DEFECT_LABEL_PLACEHOLDER not in clause
    ):
        return clause
    coop_label = scenario.coop_label(coop_label_index)
    defect_label = next(label for label in scenario.labels if label != coop_label)
    return clause.format(coop_label=coop_label, defect_label=defect_label)


def framing_clause(framing_id: str, scenario: Scenario, coop_label_index: int) -> str | None:
    """Return one registered framing's counterpart clause for one row, or None for `unstated`."""
    if framing_id not in COUNTERPART_FRAMINGS:
        raise ValueError(
            f"Unknown framing_id {framing_id!r}; known framings: {list(COUNTERPART_FRAMINGS)}."
        )
    return _clause_for_row(COUNTERPART_FRAMINGS[framing_id], scenario, coop_label_index)


# A framing label is a filename, an S3 key segment and a `prompt_id` segment at once, so it is held
# to the spelling every registered id already uses rather than to whatever a caller passes.
FRAMING_LABEL_RE = re.compile(r"^[a-z][a-z0-9-]*$")


def _clause_frames_for_split(
    arm: _MatrixArm, split: str, *, scenarios: Sequence[Scenario] | None, game_id: str
) -> tuple[Scenario, ...]:
    """Return the frames one counterpart-clause render walks: the arm's split, or a caller's subset.

    A subset is checked by object identity against the arm's own roster, not by scenario id. A
    breadth corpus draws a few frames per stratum out of two pooled PD rosters and renders each under
    the game id whose roster it came from, so the failure to catch would be a reskin frame handed to
    `twin-pd`: the prose would render, the row would say `twin-pd`, and the reskin bank's held-out
    claim would be quietly false. An id check would pass a look-alike frame built anywhere.
    """
    roster = _arm_scenarios_for_split(arm, split)
    if scenarios is None:
        return roster
    if not scenarios:
        raise ValueError(
            f"an empty scenario subset was passed for {game_id!r}/{split!r}, which would render no "
            f"rows at all; pass None for the whole roster."
        )
    known = {id(scenario) for scenario in roster}
    stray = sorted(scenario.scenario_id for scenario in scenarios if id(scenario) not in known)
    if stray:
        raise ValueError(
            f"frames {stray} are not in {game_id!r}'s {split!r} roster, so rendering them here "
            f"would stamp this game's id on another roster's prose. Pass frames read off "
            f"{game_id!r}'s own roster."
        )
    return tuple(scenarios)


def _refuse_a_coupling_clause_in_training(
    clause: str | None, framing_label: str, split: str, grading: str
) -> None:
    """Refuse a training row whose counterpart paragraph asserts the counterpart decides as you do.

    Every group-mix grading pays a completion against the group's own realised mix, and the framing
    paragraph enters the reward nowhere. On the eval split that is the measurement: a clause asserting
    coupling is read against one that does not. On the training split it would train on prompts that
    state one counterpart and pay another, so the arm would learn from a stated correlation the reward
    never honoured. The twin clause is the one standing exception, which is the convention the
    reskin arm and `COUPLING_ASSERTION_PHRASES` already record.

    Self grading is exempt: it pays the completion's own action played back, so the counterpart the
    reward honours is coupled, and a coupling clause is closer to it than the human or unstated
    clauses that were never refused. The partner-premise curriculum trains its track-record rungs so.
    """
    if split != SPLIT_TRAIN or framing_label == FRAMING_TWIN or grading == GRADING_SELF:
        return
    if _clause_states_coupling(clause):
        stated = [
            phrase
            for phrase in COUPLING_ASSERTION_PHRASES
            if clause is not None and phrase in clause
        ]
        raise ValueError(
            f"the clause passed under framing {framing_label!r} asserts that the counterpart's "
            f"decision travels with this side's ({stated}), and this is a {SPLIT_TRAIN!r} split. "
            f"The reward grades every completion against the group's realised mix and reads no "
            f"framing, so such a row would state one counterpart and pay another. Only "
            f"{FRAMING_TWIN!r} trains under a coupling clause."
        )


def render_matrix_rows_under_clause(  # noqa: PLR0913 - one keyword per rendering axis
    game_id: str,
    grading: str,
    *,
    clause: str | None,
    framing_label: str,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
    scenarios: Sequence[Scenario] | None = None,
) -> list[dict[str, Any]]:
    """Render one matrix game's rows under a caller-supplied counterpart paragraph, either split.

    The ONE path in this module that renders a TRAINING split under a swapped counterpart paragraph.
    Every other generator either renders the game's own paragraph or refuses the training split
    outright, and that is deliberate: a training corpus whose counterpart paragraph was swapped is
    exactly what makes "the model was never trained under this framing" false, so the claim has to
    stop being a property of the code and become a property of the run. Half of that is built:
    `games.train` reads the framings its corpus actually carries off the rows and records them in
    `run_config.json` as `trained_framing_ids`. Nothing reads that field back yet, so marking each
    framing trained or held out in the battery readout from the run's own record rather than from a
    constant is still owed. A breadth corpus trains seven framings and measures ten more, and until
    the readout side lands, its framing table pools the two.

    Every row carries `framing_id`, so the split survives into the corpus, the trainer's per-framing
    metrics and every readout keyed on it. That is one column beyond `ROW_COLUMNS`; the reward
    function reads it out of TRL's forwarded kwargs rather than as a `_Row` field, so a corpus written
    before the column existed still loads.

    Apart from that paragraph and that column, rows render exactly as `generate_prompt_rows` renders
    them: same frames, same payoff variants, same counterbalancing, same vocabulary guard. `clause`
    of None omits the paragraph entirely, which is what the `unstated` stem is, and label
    placeholders in the clause are filled from each row.

    `framing_label` names the cell in every `prompt_id` (ahead of the print-order and orientation
    segments, which stay at the end where their readers look for them), so rows from different
    clauses can never collide or pool silently. Two clauses passed under one label WOULD collide,
    which is why the label is the caller's to keep distinct.

    `scenarios` restricts the render to a subset of the arm's own frames for that split, which is how
    a stratified corpus draws a quota per (game, framing, variant) instead of the whole roster.
    """
    if game_id not in FRAMEABLE_GAME_IDS:
        raise ValueError(
            f"{game_id!r} cannot take a counterpart framing; the one-shot matrix games "
            f"{sorted(FRAMEABLE_GAME_IDS)} are the ones whose renderer owns that paragraph."
        )
    if not FRAMING_LABEL_RE.match(framing_label):
        raise ValueError(
            f"framing_label {framing_label!r} does not match {FRAMING_LABEL_RE.pattern}; the label "
            f"lands in every prompt_id and in artifact filenames, so it is held to the lowercase "
            f"hyphenated spelling every registered framing id already uses."
        )
    _assert_generatable(game_id, grading, split, label_print_order)
    _refuse_a_coupling_clause_in_training(clause, framing_label, split, grading)
    arm = _matrix_arm_for(game_id)
    contexts = [
        _RowContext(
            game_id=game_id,
            grading=grading,
            spec=arm.build_spec(payoff_variant),
            payoff_variant=payoff_variant,
            label_print_order=label_print_order,
        )
        for payoff_variant in arm.payoff_variants
    ]
    rows: list[dict[str, Any]] = []
    for scenario in _clause_frames_for_split(arm, split, scenarios=scenarios, game_id=game_id):
        for context in contexts:
            for coop_label_index in COOP_LABEL_INDICES:
                prompt = render_game_prompt(
                    context.spec,
                    scenario,
                    coop_label_index=coop_label_index,
                    twin_framing=False,
                    opponent_framing=_clause_for_row(clause, scenario, coop_label_index),
                    label_print_order=label_print_order,
                )
                row = _matrix_row(
                    context, scenario, prompt=prompt, coop_label_index=coop_label_index
                )
                row["prompt_id"] = (
                    f"{context.game_id}--{scenario.scenario_id}--{context.payoff_variant}"
                    f"--framing-{framing_label}{_print_order_suffix(label_print_order)}"
                    f"--coop{coop_label_index}"
                )
                row[FRAMING_ID_COLUMN] = framing_label
                rows.append(row)
    _assert_row_schema(rows, extra_columns=(FRAMING_ID_COLUMN,))
    prompt_ids = [row["prompt_id"] for row in rows]
    duplicates = sorted({pid for pid in prompt_ids if prompt_ids.count(pid) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate prompt_ids {duplicates} for {game_id!r}/{framing_label!r}; rates are "
            f"grouped by prompt_id, so a collision would mix two prompts' draws."
        )
    logger.info(
        f"rendered matrix rows under a counterpart clause, {game_id=} {framing_label=} {split=} "
        f"{label_print_order=} n_rows={len(rows)}"
    )
    return rows


def generate_counterpart_clause_prompt_rows(  # noqa: PLR0913 - one keyword per rendering axis
    game_id: str,
    grading: str,
    *,
    clause: str | None,
    framing_label: str,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> list[dict[str, Any]]:
    """Generate one matrix game's EVAL rows with a caller-supplied counterpart paragraph.

    The runtime-clause renderer behind `generate_framing_prompt_rows`, and the path a clause that
    is not in `COUNTERPART_FRAMINGS` takes -- a ladder of counterpart identities authored outside
    this module, for instance, whose text must not be committed here.

    Eval split only, and the refusal is the whole reason this wrapper exists rather than callers
    reaching `render_matrix_rows_under_clause` directly: these rows measure a trained disposition
    against a counterpart framing, and a measurement cell that quietly accepted a training split
    would be indistinguishable from one that did not. A corpus that really does train under a
    swapped paragraph goes through the renderer by name, which is what makes the two cases greppable
    apart.
    """
    if split != SPLIT_EVAL:
        raise ValueError(
            f"Framing rows are measurement-only, so there is no {split!r} split: training under a "
            f"swapped counterpart framing would make 'never trained under this framing' false "
            f"while everything still ran. A corpus that means to train one calls "
            f"render_matrix_rows_under_clause, which records the framing on every row so the run's "
            f"own trained_framing_ids says which framings it trained."
        )
    return render_matrix_rows_under_clause(
        game_id,
        grading,
        clause=clause,
        framing_label=framing_label,
        split=split,
        label_print_order=label_print_order,
    )


def generate_transfer_prompt_rows(  # noqa: PLR0913 - one keyword per rendering axis
    game_id: str,
    grading: str = GRADING_FORMAT_ONLY,
    *,
    spec: TransferSpec,
    scenario: TransferScenario,
    clause: str | None,
    clause_label: str,
    polarity: str,
) -> list[dict[str, Any]]:
    """Generate the single row for one transfer cell: one game, one scenario, one dose, one polarity.

    One row rather than a list of them, returned as a list so the schema check and the caller's
    accumulation read like every other generator here. The grid this game is swept over is the sampling
    plan's to walk -- scenarios times doses times polarities times ladder rungs -- because the rungs and
    the frames are both authored outside this repository and loaded at runtime.

    `grading` is a row label only. Nothing trains on these games and no reward function reconstructs a
    `TransferSpec`, so `format-only` says what is true: the only thing a completion is scored on here is
    whether it answered in the tag it was asked for.

    `clause` of None omits the counterpart paragraph, and then `clause_label` must be
    `TRANSFER_IDENTITY_BLIND_LABEL`: that render is the identity-blind cell and the audit's stem at once,
    and a clause-free row filed under a rung's label would put the baseline inside the ladder.
    """
    if game_id not in TRANSFER_RENDERERS:
        raise ValueError(
            f"{game_id!r} is not a transfer game; the transfer games are "
            f"{sorted(TRANSFER_RENDERERS)}."
        )
    if not is_grading(grading):
        raise ValueError(unknown_grading_message(grading))
    if not FRAMING_LABEL_RE.match(clause_label):
        raise ValueError(
            f"clause_label {clause_label!r} does not match {FRAMING_LABEL_RE.pattern}; the label lands "
            f"in every prompt_id and in artifact filenames, so it is held to the lowercase hyphenated "
            f"spelling every registered framing id already uses."
        )
    blind = clause_label == TRANSFER_IDENTITY_BLIND_LABEL
    if blind != (clause is None):
        raise ValueError(
            f"clause_label {clause_label!r} and a clause of "
            f"{'None' if clause is None else 'some text'} disagree about whether this row is the "
            f"identity-blind cell. That render is the baseline every reading here is a contrast "
            f"against and the stem the one-inserted-paragraph audit compares against, so a rung's "
            f"clause filed under {TRANSFER_IDENTITY_BLIND_LABEL!r} -- or a clause-free render filed "
            f"under a rung -- would put the baseline inside the ladder with every count still adding up."
        )
    if spec.game_id != game_id:
        raise ValueError(
            f"spec names game {spec.game_id!r} and this call asks for {game_id!r}; the spec's own id is "
            f"what a refusal from games.payoffs quotes, so a mismatched pair reports the wrong game."
        )
    prompt = TRANSFER_RENDERERS[game_id](spec, scenario, clause=clause, polarity=polarity)
    rows = [
        _transfer_row(
            prompt=prompt,
            game_id=game_id,
            grading=grading,
            spec=spec,
            scenario=scenario,
            clause_label=clause_label,
            polarity=polarity,
        )
    ]
    _assert_row_schema(rows)
    return rows


def generate_framing_prompt_rows(
    game_id: str,
    grading: str,
    *,
    framing_id: str,
    split: str,
    label_print_order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> list[dict[str, Any]]:
    """Generate one matrix game's rows under one REGISTERED counterpart framing.

    The registry lookup in front of `generate_counterpart_clause_prompt_rows`, which does the
    rendering. The `twin` framing reproduces the trained rendering byte for byte, which is the
    sweep's within-run reference cell.
    """
    if framing_id not in COUNTERPART_FRAMINGS:
        raise ValueError(
            f"Unknown framing_id {framing_id!r}; known framings: {list(COUNTERPART_FRAMINGS)}."
        )
    return generate_counterpart_clause_prompt_rows(
        game_id,
        grading,
        clause=COUNTERPART_FRAMINGS[framing_id],
        framing_label=framing_id,
        split=split,
        label_print_order=label_print_order,
    )


def render_track_record_prompt(  # noqa: PLR0913 - one keyword per axis of the row being rebuilt
    *,
    source_game_id: str,
    scenario_id: str,
    payoff_variant: str,
    coop_label_index: int,
    label_print_order: str,
    match_percent: int,
    spec_override: MatrixGameSpec | None = None,
) -> tuple[str, MatrixGameSpec]:
    """Render one training stem with the numeric track-record clause as its counterpart paragraph.

    The rendering half of `games.track_record_corpus`, kept here because the frame roster and the
    renderer are this module's own: the builder names a stem by the columns its source row already
    carries, and this function re-renders that exact stem with `STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE`
    swapped in where the source's counterpart paragraph was omitted. Everything else -- frame,
    outcome table, symmetry note, instruction, counterbalancing, vocabulary guard -- renders
    exactly as `generate_prompt_rows` renders it, which is what lets the builder assert that
    deleting the one inserted paragraph reproduces the source prompt byte for byte.

    The percentage is required to sit strictly inside (0, 100): the corners are not track records
    but the certain framings ("every match", already registered separately), and a 0% record would
    state perfect anti-correlation in prose written for correlation. The spec RETURNED is always
    the roster's own, so the builder can assert the roster's cells against the source row's,
    rather than trusting that the two never drifted.

    `spec_override` substitutes the payoff cells the outcome block prints -- the v2 margin-balance
    path, where each cell's table is rescaled and quantized per (table, rung). Only the four point
    values move; the frame, labels and instruction still render from the roster, and
    `render_track_record_stem` gives the builder the clause-free reference to prove that against.
    """
    if not 0 < match_percent < MATCH_PERCENT_SCALE:
        raise ValueError(
            f"match_percent must sit strictly inside (0, 100), got {match_percent}: the corners "
            f"are the certain framings, not numeric track records."
        )
    scenario, roster_spec = _track_record_stem_parts(
        source_game_id=source_game_id, scenario_id=scenario_id, payoff_variant=payoff_variant
    )
    prompt = render_game_prompt(
        spec_override if spec_override is not None else roster_spec,
        scenario,
        coop_label_index=coop_label_index,
        twin_framing=False,
        opponent_framing=STATED_TRACK_RECORD_PERCENT_CLAUSE_TEMPLATE.format(
            match_percent=match_percent
        ),
        label_print_order=label_print_order,
    )
    return prompt, roster_spec


def _track_record_stem_parts(
    *, source_game_id: str, scenario_id: str, payoff_variant: str
) -> tuple[Scenario, MatrixGameSpec]:
    """Resolve the one training frame and roster spec a track-record stem names.

    Shared by the clause and clause-free renderers so the two cannot resolve a stem differently,
    which is what lets the corpus builder assert their outputs against each other.
    """
    arm = _matrix_arm_for(source_game_id)
    scenarios = [
        scenario
        for scenario in _arm_scenarios_for_split(arm, SPLIT_TRAIN)
        if scenario.scenario_id == scenario_id
    ]
    if len(scenarios) != 1:
        raise ValueError(
            f"{source_game_id!r} has {len(scenarios)} training frames named {scenario_id!r}, "
            f"want exactly 1; the corpus row and the roster disagree about what this stem is."
        )
    return scenarios[0], arm.build_spec(payoff_variant)


def render_track_record_stem(  # noqa: PLR0913 - one keyword per axis of the stem being rebuilt
    *,
    source_game_id: str,
    scenario_id: str,
    payoff_variant: str,
    coop_label_index: int,
    label_print_order: str,
    spec_override: MatrixGameSpec | None = None,
) -> tuple[str, MatrixGameSpec]:
    """Render one training stem clause-free, optionally with substituted payoff cells.

    The v2 corpus builder's identity anchor. Called twice per cell: without an override it must
    reproduce the banked source prompt byte for byte (proving the frame, labels and instruction
    are the stems whose baseline behaviour is already measured), and with the cell's rescaled
    quantized spec it produces the reference the clause-bearing prompt is diffed against (proving
    the only deltas from the banked stem are the four point values plus the clause paragraph).
    """
    scenario, roster_spec = _track_record_stem_parts(
        source_game_id=source_game_id, scenario_id=scenario_id, payoff_variant=payoff_variant
    )
    prompt = render_game_prompt(
        spec_override if spec_override is not None else roster_spec,
        scenario,
        coop_label_index=coop_label_index,
        twin_framing=False,
        opponent_framing=None,
        label_print_order=label_print_order,
    )
    return prompt, roster_spec


def reward_spread_report(mixes: Sequence[float] = GROUP_MIX_SPREAD_MIXES) -> str:
    """Render the within-group reward spread of every trainable matrix game, one line per variant.

    Each cell is `games.payoffs.group_mix_reward_spread` -- |r_C - r_D| against an opponent
    cooperating with probability p -- which under `scale_rewards="batch"` is proportional to the
    arm's effective gradient when games or variants share a batch (risky-hunt's 0.05 at p=0.9
    trains ~8x weaker than twin-pd's 0.38). Printed at corpus-build and arm-launch time so
    "run compressed variants in their own runs, or not" is decided on numbers. The iterated and
    dictator arms have no per-action opponent expectation, so they have no line here.
    """
    entries: list[tuple[str, MatrixGameSpec]] = [
        (f"{game_id}/{payoff_variant}", arm.build_spec(payoff_variant))
        for game_id, arm in _MATRIX_ARMS.items()
        for payoff_variant in arm.payoff_variants
    ]
    name_width = max(len(name) for name, _ in entries)
    header = f"{'game/variant':<{name_width}}" + "".join(f"  p={mix:<3g}" for mix in mixes)
    body = [
        f"{name:<{name_width}}"
        + "".join(f"  {group_mix_reward_spread(spec, mix):5.3f}" for mix in mixes)
        for name, spec in entries
    ]
    return "\n".join([header, *body])
