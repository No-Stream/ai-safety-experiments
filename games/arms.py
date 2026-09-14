"""The arm registry: which game each training run describes, and how it grades completions.

An *arm* is one training run, and `ARMS` is the whole experiment: the contrast between
`twin-pd-group` and `twin-pd-self` is identical prompts under two grading rules, so reading
anything off that contrast requires that nothing else differed. `games/train.py` runs an arm; this
module says what the arms ARE.

It lives apart from the trainer for a measured reason. Every stage plan in `games/` wants the
registry in order to render a launch command, and `games/train.py` imports torch, transformers, trl,
peft and (through `grpo.rlvr_math`) matplotlib and pandas at module top. Importing the registry
through the trainer therefore cost 10.0 s against 0.05 s for the prompt module -- a 200x tax on
`--print-plan`, whose whole selling point is being the cheap check to run before the meter starts.
Nothing here imports the training stack, and `test_games_arms.py` fails if that ever changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from games.payoffs import (
    CONSTANT_SUM_TOLERANCE,
    THRESHOLD_GOODS_MAX_PRIZE,
    TRUST_MAX_SELF_STATED_RETURN_FRACTION,
    NashDemandSpec,
    ThresholdGoodsSpec,
    TrustSpec,
    nash_demand_self_reward_span,
    threshold_goods_self_reward_span,
    trustor_self_rule_reward_span,
)
from games.prompts import (
    CORPUS_BUILT_GAME_IDS,
    GAME_IDS,
    ONE_SHOT_INSTRUCTION,
    TRACK_RECORD_GAME_ID,
    TRACK_RECORD_V2_GAME_ID,
    generate_prompt_rows,
)
from games.rewards import (
    GRADING_FORMAT_ONLY,
    GRADING_GROUP_MIX,
    GRADING_ITERATED_RETURN,
    GRADING_JOINT_WELFARE_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    GRADING_OTHER_PAYOFF_GROUP_MIX,
    GRADING_SELF,
    GRADING_THRESHOLD_GOODS_GROUP_MIX,
    GRADING_THRESHOLD_GOODS_SELF,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    GRADING_VS_STATED_MATCH,
    GRADINGS,
    PARSE_PENALTY_CONSTANT,
    PARSE_PENALTY_MARGIN_BELOW_WORSE,
    PARSE_PENALTY_MODES,
    care_alpha_of,
    care_grading,
    is_grading,
    unknown_grading_message,
    why_a_grading_cannot_price_a_failure,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# The payoff magnitude of a prompt row, which an arm may pin itself to a subset of.
PAYOFF_VARIANT_COLUMN = "payoff_variant"

# The answer-shape signature of a one-shot action prompt, taken from the instruction template itself
# rather than restated, so a reworded instruction fails the format-only guard below instead of
# silently making it vacuous. The split drops everything up to the template's last placeholder,
# leaving the placeholder-free tail; it distinguishes all three answer shapes the corpus renders --
# the dictator asks for `<keep>N</keep>` and the iterated arm for N tags "with nothing after the
# last one", so neither carries this string.
ONE_ACTION_TAG_MARKER = ONE_SHOT_INSTRUCTION.rsplit("}", 1)[-1]

# Which side of its game's group-mix boundary a swept prompt sat on at baseline, written onto the
# rows by `games.corpus_partition` and pinnable by an arm the way a payoff variant is. Two arms over
# one game, one grading and one payoff variant then differ in the corpus they train on and in
# nothing else, which is what the mix-split contrast is.
#
# The column and its vocabulary live here rather than beside the code that computes them, for the
# same reason `PAYOFF_VARIANT_COLUMN` does: `games.corpus_partition` reads the counterbalanced-pair
# key out of `games.select_prompts` and so imports torch, while this registry must stay importable
# without the training stack (see the module docstring, and TestTheRegistryStaysCheapToImport).
CORPUS_PARTITION_COLUMN = "corpus_partition"
PARTITION_ABOVE_THRESHOLD = "above-threshold"
PARTITION_BELOW_THRESHOLD = "below-threshold"
CORPUS_PARTITIONS = frozenset({PARTITION_ABOVE_THRESHOLD, PARTITION_BELOW_THRESHOLD})


@dataclass(frozen=True)
class GameArm:
    """One training run: which game the prompts describe, and how completions are graded.

    `game_ids` names the OTHER games this arm's corpus may carry, beside `game_id`. Empty, which is
    every arm registered before 2026-09-04, means the corpus is one game and the loader refuses any
    other. A breadth arm trains several games at once, and the set has to be registry state rather
    than a property of whichever corpus file was passed: the loader's refusal is the only thing
    standing between a mixed file and a run that trains rows the arm never claimed, and with one
    `game_id` field that refusal is all-or-nothing. `arm_game_ids` returns the whole set with
    `game_id` first, and every consumer asks for that rather than reassembling it.

    `payoff_variants` pins the arm to a subset of the corpus's payoff magnitudes. Empty means
    every variant the corpus carries.

    `corpus_partition` pins the arm to one side of its game's group-mix boundary, as measured on the
    base policy by the baseline sweep and stamped onto the corpus by `games.corpus_partition`. Empty
    means the whole corpus, which is every arm registered before the mix-split existed.

    `parse_penalty_mode` says how the grading prices an unparseable completion
    (`games.rewards.PARSE_PENALTY_MODES`). It is part of the grading, so it lives here rather than
    on the run: two arms over one corpus that differ only in this are two experiments, and every
    artifact that names the arm names the penalty regime with it. The default is the constant every
    arm registered before 2026-09-02 trained under; the scaled mode is defined for the
    vs-stated-match grading only, whose reward knows both actions' expected payoffs.
    """

    game_id: str
    grading: str
    notes: str
    game_ids: tuple[str, ...] = ()
    payoff_variants: tuple[str, ...] = ()
    corpus_partition: str = ""
    parse_penalty_mode: str = PARSE_PENALTY_CONSTANT


ARMS: dict[str, GameArm] = {
    "twin-pd-group": GameArm(
        game_id="twin-pd",
        grading="group-mix",
        notes=(
            "Expected payoff against the group's own empirical action mix. Under group-relative "
            "advantage this can only ever upweight defection, whatever the nominal payoffs say."
        ),
    ),
    "twin-pd-self": GameArm(
        game_id="twin-pd",
        grading="self",
        notes=(
            "Identical prompts graded against the model's OWN action, the true functional twin. "
            "The novel contrast arm: it isolates grading correlation structure as the cause."
        ),
    ),
    # The third leg of the twin-PD contrast, and the one that decides whether the other two are
    # readable. OpenAI's reward-vs-data control (arXiv 2606.24014 §4.2) found that the same data
    # under a generic reward generalised not at all, so the graded dimension carried their whole
    # effect; wave-1 here found format pressure taking a large share of every arm's gradient and
    # parse failure and median length falling in all 11 paired legs. Until one arm holds the prompts
    # fixed and grades only the answer shape, "RL on game-shaped prompts moved this instrument"
    # cannot be told from "the graded dimension moved it" -- and that is a confound on every other
    # arm's survey reading, not a curiosity about this one.
    "twin-pd-format-only": GameArm(
        game_id="twin-pd",
        grading="format-only",
        notes=(
            "Identical prompts to twin-pd-group and twin-pd-self, graded on answer shape alone: "
            "every component of the reward is a rule the prompt already states or a form it "
            "already prints (games.format_rubric), so nothing about the game enters the reward and "
            "either action reaches the same maximum. Trainability is inherited rather than hoped "
            "for: the corpus is the twin-pd group-mix corpus re-graded on CPU "
            "(games.regrade_corpus), so selection attrition is identical by construction to the "
            "arms this is a placebo for -- the baseline mix that kept those prompts is the mix "
            "here. What is NOT inherited is within-group reward spread, since a rubric over answer "
            "shape can go pure once the policy is tidy: measure it from the baseline sweep's own "
            "completions with games.format_spread before launching, and expect frac_groups_pure "
            "well above the graded arms'. The action is still parsed and coop_rate still logged, "
            "because what this arm's behaviour does while its reward ignores behaviour is the "
            "measurement."
        ),
    ),
    # The learning-side framing pair, completing the 2x2 {twin framing, unstated} x {self,
    # group-mix} whose twin column is the trained twin-pd pair. The prompts differ from twin-pd's by
    # exactly one deleted paragraph (the twin counterpart clause; pinned byte-level in
    # test_games_prompts.py), and the reward never reads the prompt, so both gradings keep the same
    # optima -- cooperate under self (CC=0.6 > DD=0.2), defect under group-mix (dominance). What the
    # deletion removes is the self arm's DERIVABLE story: without the clause the stated one-shot
    # game makes defection dominant while the reward pays cooperation anyway. Whether that arm still
    # learns to diverge is the question the pair exists to ask -- the eval-side framing sweep
    # already says the clause gates EXPRESSION of the trained divergence.
    "pd-unstated-group": GameArm(
        game_id="pd-unstated",
        grading="group-mix",
        notes=(
            "twin-pd's cells and frames with no 'About the other side' paragraph, graded against "
            "the group's own realised mix. Defection is dominant with or without the framing, so "
            "the pre-registered expectation is a cooperation fall comparable to twin-pd-group's; a "
            "failure to fall is a plumbing lead rather than a finding."
        ),
    ),
    "pd-unstated-self": GameArm(
        game_id="pd-unstated",
        grading="self",
        notes=(
            "The informative cell of the 2x2: identical framing-less prompts graded against the "
            "completion's own action, so cooperation pays 0.6 against 0.2 exactly as in "
            "twin-pd-self -- but the prompt no longer states the coupling that makes that reward "
            "derivable. If cooperation still rises, the framing is unnecessary for LEARNING the "
            "divergence and the readout becomes which rationalisation got amplified; if it stays "
            "flat while the group arm moves, the framing is load-bearing for learning despite "
            "equal advantage magnitudes by construction."
        ),
    ),
    # The wave-3 "many stories" arm (2026-08-29). pd-unstated-self showed the reward coupling
    # alone teaches cooperation on story-free prompts, expressing ACROSS the belief framings --
    # the inverse of the twin-trained gate. Two explanations survive: generality comes from the
    # story's ABSENCE (nothing in the prompt for the learned behaviour to key on), or from
    # DIVERSITY forcing abstraction (pd-unstated already spans ~10-13 house fictions). This arm
    # moves only the diversity axis: the same cells, instruction, deletion and grading over the
    # 44-train-skin register-diverse reskin roster, with 12 held-out skins as the in-arm
    # generalization probe. The group leg is deliberately unregistered -- dominance training
    # needs no story (settled twice) -- and the corpus-size-with-diversity confound is accepted
    # and named in the design doc: a corpus wide enough to be diverse cannot be 20 prompts.
    "pd-reskin-self": GameArm(
        game_id="pd-reskin",
        grading=GRADING_SELF,
        notes=(
            "twin-pd's cells with no counterpart paragraph over the wide reskin roster, graded "
            "against the completion's own action -- the pd-unstated-self construction with many "
            "stories instead of few. Registered contrast: if generality comes from story-absence, "
            "this arm's cross-framing deltas sit at or below pd-unstated-self's; if diversity "
            "forces abstraction, at or above, including on the held-out skins. The per-skin "
            "baseline sweep doubles as the amplification probe: gains confined to skins with "
            "baseline cooperative mass read as amplification, gains on ~zero-mass skins as "
            "creation."
        ),
    ),
    # The wave-3 grading ladder (2026-08-29), on the SAME story-free prompts as the pd-unstated
    # pair: the reward's RECIPIENT moves while the corpus, the coupling machinery and every other
    # knob hold still. With the trained pd-unstated-self it forms the ladder {mine, ours, theirs}:
    # cooperation trained because it pays me (self, via the coupling), because it maximises the pie
    # (joint mean), or because it pays the counterpart (pure other-regard). The readout is which
    # generalisation signature each produces -- guaranteed-cooperator (anti-incentive) cells,
    # dictator kept-fraction, trust sends, cross-framing breadth. The three rungs' advantage gaps
    # are unmatched by construction (self 0.4 flat; joint 0.1-0.3; other 0.6-0.8 at temptation-2),
    # and that does NOT license reading generalisation apart from training strength: the scored
    # ladder found the training-side reward rise ordering the rungs exactly as every generalisation
    # result did, so a cross-rung verdict is conditional unless the arms match on the
    # strategy-to-format advantage ratio or on in-distribution outcome with steps floating. A
    # constant reward rescale cannot supply that match under either estimator this repo has used;
    # see the matched-signal method note at the end of docs/games-predictions.md.
    "pd-unstated-joint-welfare": GameArm(
        game_id="pd-unstated",
        grading=GRADING_JOINT_WELFARE_GROUP_MIX,
        notes=(
            "Story-free prompts graded as the MEAN of both sides' payoffs against the group's "
            "realised mix. At temptation-2 cooperation is joint-dominant (gap 0.3 - 0.2p), so the "
            "arm should run toward the cooperative corner; at temptation-10 the pie is bigger on "
            "the off-diagonal (DC + CD > 2*CC) and the gap crosses zero at p* = 0.611 "
            "(games.payoffs.joint_welfare_gap_crossing), a STABLE interior attractor -- the arm's "
            "numeric prediction on that variant, chicken's role reprised. Read the two variants "
            "separately, never pooled: they push toward different attractors, and the results log "
            "already carries a case where a pooled curve reversed both variants' signs."
        ),
    ),
    "pd-unstated-other-payoff": GameArm(
        game_id="pd-unstated",
        grading=GRADING_OTHER_PAYOFF_GROUP_MIX,
        notes=(
            "Story-free prompts graded as the COUNTERPART's expected payoff against the group's "
            "realised mix: the reward is what my choice hands the other side, and in any strict "
            "PD cooperating is strictly dominant under it at every mix and temptation (gap "
            "0.8 - 0.2p at temptation-2, the widest on the ladder). The selfless rung: if what "
            "trains here generalises differently from the self arm's coupling-paid cooperation -- "
            "more movement on the guaranteed-cooperator cells, the dictator split, the trust "
            "sends -- the recipient of the reward, not just the trained action, is what the "
            "policy learned."
        ),
    ),
    # The track-record arm (2026-08-30, owner-approved "make the counterpart matter"). The
    # ext-lr2e5 leg of pd-unstated-self showed the self grading's cost: payoff(a,a) is a fixed
    # function of the model's own action, and the policy measurably learned to stop consulting the
    # counterpart (not-discussed among cooperators 1/31 base -> 68/106 at cumulative step 140,
    # framing sweep flattened, cooperation 0.56-0.64 with a disclosed guaranteed cooperator where
    # defecting pays ~4x). ANY fixed function of own action recreates that -- including one fixed
    # "matched you 90% of the time" clause with EV reward, whose EV at fixed correlation is again
    # own-action-only. This arm's corpus therefore MIXES stated match probabilities across each
    # payoff variant's EV crossover, so which action pays is decided by the one number the clause
    # states and by nothing else the policy could cache.
    "pd-track-record": GameArm(
        game_id=TRACK_RECORD_GAME_ID,
        grading=GRADING_VS_STATED_MATCH,
        notes=(
            "pd-unstated's stems plus a numeric track-record clause ('in about NN% of matches so "
            "far, its decision has come out identical to its counterpart's'), graded as the exact "
            "expected payoff under the stated NN -- the same number, one column, informational "
            "honesty by construction. The p mixture straddles each variant's EV crossover "
            "(temptation-2 at p*=0.714, temptation-10 at p*=0.867), so cooperation is strictly "
            "EV-optimal on the high rungs and defection on the low ones: a policy that stops "
            "reading the counterpart clause cannot collect this reward. Corpus built and "
            "margin-audited by games.track_record_corpus (cells within 0.05 of their crossover "
            "are dropped); games.train.assert_stated_match_mixture refuses a corpus that has "
            "degenerated to one side. Registered readings: cooperation splits by incentive "
            "direction (coop_rate_where_coop_pays up, coop_rate_where_defect_pays down), the "
            "cooperation-vs-p eval curve steepens around the crossover, counterpart-mention "
            "stays high, and the disclosed-cooperator cell moves toward defection, not away."
        ),
    ),
    # Track-record v2 (2026-09-01, owner-approved "make the models actually have to pay attention
    # and play the game"). v1's readout: anti-lobotomization succeeded (counterpart-reading and
    # framing sensitivity retained, disclosed actions best-responded) but cooperation FELL
    # globally, and the measured cause was margin asymmetry -- defect-side |EV margins| to 0.538
    # against coop-side 0.050-0.386, a ~9x mean penalty gap under batch reward scaling. v1 also
    # quietly left the stated rate alone predictive of the optimum on 4 of its 5 rungs. v2 fixes
    # both structurally: every cell's displayed table is rescaled (uniformly, preserving the
    # crossover and PD ordering) and quantized so its |EV margin| at its stated rate is one
    # shared constant, and the (table, rate) chain grid makes neither the rate alone nor the
    # table alone predictive (best rate-lookup policy <= 0.70 by audit).
    "pd-track-record-v2": GameArm(
        game_id=TRACK_RECORD_V2_GAME_ID,
        grading=GRADING_VS_STATED_MATCH,
        notes=(
            "pd-unstated's stems, the numeric track-record clause, and reward = the exact "
            "expected payoff under the stated NN computed from the DISPLAYED quantized payoff "
            "table -- the honesty invariant now covers the table as well as the clause. Corpus "
            "built by games.track_record_corpus --grid v2: a three-slot crossover chain "
            "(p* = 0.57 / 5-out-of-7 / 0.895, rungs 51/63/79/99, interior rungs shared between "
            "adjacent slots so they are exactly 50/50), every cell margin-balanced to "
            "TRACK_RECORD_V2_TARGET_MARGIN on the displayed cells. Registered readings: "
            "coop_rate_where_coop_pays RISES (v1's failed headline), defect side holds, "
            "counterpart-mention stays high, disclosed corners stay best-response, framing SD "
            "stays at base, the 17-game battery shows no homogenization, and dose thresholding "
            "tracks each table's own crossover rather than a p-only rule."
        ),
    ),
    # Track-record v2, penalty-scaled (2026-09-02, owner-approved "sure, yes"). v2's readout: the
    # margin balance landed (right-minus-wrong advantage gap symmetric, +0.199 / +0.187 per side)
    # but the policy did not carve by side -- cooperation rose everywhere base defected, EV-optimal
    # play fell to chance, the 17-game battery compressed -- and the training-side read located the
    # carrier in the parse-failure penalty: the constant -1.0 was ten times the 0.10 margin, landed
    # on long defect-leaning deliberation (161 of 395 failures were truncations at the 32768 cap,
    # 205 malformed tags after long reasoning), and the policy halved its deliberation. This arm
    # keeps the v2 corpus and everything else identical and changes exactly two things: the parse
    # penalty is priced one EV margin below the row's worse action (the `parse_penalty_mode` pin
    # below, so right > wrong > malformed with equal gaps) and the run's completion budget rises
    # above the observed deliberation lengths (65536, a run knob). Same game and grading as v2 so
    # the corpus, the eval cells and the readout instruments carry over unchanged; a separate
    # registry entry, distinguished by the pin, so every artifact names which penalty regime
    # trained it.
    "pd-track-record-v2-softpen": GameArm(
        game_id=TRACK_RECORD_V2_GAME_ID,
        grading=GRADING_VS_STATED_MATCH,
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        notes=(
            "pd-track-record-v2's corpus and reward (exact EV under the stated NN from the "
            "displayed quantized table, honesty invariant covering table and clause), with the "
            "parse-failure price scaled to the reward's own range: a malformed answer earns "
            "min(EV(C), EV(D)) - |EV(C) - EV(D)| for its cell, one margin below the worse action, "
            "so the failure gradient equals the side-correct gradient in magnitude instead of "
            "being ten times it; and a 65536-token completion budget so long deliberation is not "
            "truncated into that price. Registered readings inherit v2's seven, plus the "
            "mechanism predictions inverted: completion length does not collapse, parse failures "
            "hold near 9% or fall for the right reason, ev_optimum_rate rises above chance, and "
            "coop_rate_where_defect_pays falls."
        ),
    ),
    # The transfer-of-LEARNING pair, registered 2026-08-26 when public-goods was promoted out of
    # the eval-only set (chicken's precedent). Everything measured so far about the twin-pd pair is
    # EXPRESSION -- a trained disposition evaluated on other games' prompts; these two arms measure
    # LEARNING: continue-RL onto public-goods from a twin-pd checkpoint versus from base, and read
    # the learning curves against each other. The arms carry the game and grading only; which
    # checkpoint initialises the LoRA is a `games.train --init-adapter` knob recorded in each run's
    # run_config.json, so one registered arm serves both its from-base and from-checkpoint cells and
    # a reader of any artifact splits the cells on that field. Public-goods reduces to a PD on the
    # shared scenario frames (contributing is "C", twin clause kept), so the prompts differ from
    # twin-pd's in the four printed point values and nothing else -- the group-mix reward gap is a
    # flat 0.111 against twin-pd's 0.2+0.2p, and the self-graded span is 0.333 against 0.4, so
    # per-step movement should read shallower than the flagship's even where the dynamics match.
    "public-goods-group": GameArm(
        game_id="public-goods",
        grading="group-mix",
        notes=(
            "Learning-to-defect leg of the transfer pair: expected payoff against the group's own "
            "realised mix, where defection is upweighted from any mix exactly as in twin-pd-group. "
            "Runs twice -- initialised from the twin-pd-group checkpoint and from base -- and the "
            "readout is the curve contrast: acceleration (prior RL built reusable machinery), bias "
            "(converges elsewhere), or interference (slower than base)."
        ),
    ),
    "public-goods-self": GameArm(
        game_id="public-goods",
        grading="self",
        notes=(
            "Learning-to-cooperate leg of the transfer pair: identical prompts graded against the "
            "completion's own action, so contributing pays CC=0.889 against DD=0.556 and "
            "cooperation is the trained direction, as in twin-pd-self. Runs twice -- initialised "
            "from the twin-pd-self checkpoint and from base -- with the same three-way curve "
            "readout as the group leg."
        ),
    ),
    "stag-hunt-group": GameArm(
        game_id="stag-hunt",
        grading="group-mix",
        notes=(
            "Mutual cooperation is itself a Nash equilibrium here, so self-play can reinforce "
            "it through plain causal reasoning. Does the decision theory move at all? The "
            "corpus is now a four-rung risk-dominance ladder (thresholds 1/3, 1/2, 3/4, 0.95), "
            "so the readout is where trained cooperation gives out, not one yes/no. Check the "
            "reward-spread report before batch-mixing rungs: the compressed ones train weaker."
        ),
    ),
    # Two rungs of that ladder, pinned to run as their own arms. This is what the reward-spread rule
    # asks for -- the four rungs' within-group spreads differ up to 10.2x, so under
    # scale_rewards="batch" a mixed batch trains the compressed rungs proportionally weaker, and
    # their flat curves would be artifacts of the mixing. Only these two are registered, because the
    # 2026-08-19 baseline sweep measured all four and the other two are ceiling-bound: favoured-hunt
    # kept 6 of 32 prompts and even-hunt 10, against a mixedness floor of 12, with 16 and 13
    # respectively answered unanimously. Registering an arm whose corpus cannot clear its own floor
    # would only reproduce hi-lo's stopped run.
    "stag-hunt-safe-rung": GameArm(
        game_id="stag-hunt",
        grading="group-mix",
        notes=(
            "The measured crossover rung: its risk-dominance threshold of 3/4 sits 0.018 below the "
            "baseline cooperation of 0.732 measured on 2026-08-19, an order of magnitude closer "
            "than any other rung. The pre-registration expects the crossover to move slowest and "
            "noisiest, so this arm is that prediction's test and the most informative single point "
            "on the ladder. Kept 14 of 32 prompts."
        ),
        payoff_variants=("safe-hunt",),
    ),
    "stag-hunt-risky-rung": GameArm(
        game_id="stag-hunt",
        grading="group-mix",
        notes=(
            "The far end of the ladder: threshold 0.95 against a measured baseline of 0.754, the "
            "largest downward margin of any runnable rung, so it should move fastest and most "
            "cleanly and is the contrast partner for the safe rung's predicted sluggishness. Kept "
            "18 of 32 prompts, the widest surviving corpus on the ladder. Note this rung is the "
            "narrowest by reward spread against a mostly-cooperating opponent (0.05), which is "
            "exactly why it must not share a batch with the others."
        ),
        payoff_variants=("risky-hunt",),
    ),
    "hi-lo-group": GameArm(
        game_id="hi-lo",
        grading="group-mix",
        notes=(
            "Pure coordination with no conflict anywhere on the sheet: the positive control on "
            "the reward path. The reward gap favours the high meeting point once the group mix "
            "clears ~0.09, so group-mix training should drive cooperation to the high "
            "equilibrium from nearly any start. If this arm does not move, the reward plumbing "
            "is broken -- no other arm fails loudly in that case."
        ),
    ),
    "harmony-group": GameArm(
        game_id="harmony",
        grading="group-mix",
        notes=(
            "Cooperation is strictly dominant AND mutual cooperation is the best cell, so "
            "maximum reward is simply cooperation. The companion placebo to fixed-pie-pd-group: "
            "one arm where cooperation is trivially upweighted, one where it is impossible, "
            "with identical machinery -- and the contrast with the stag hunt separates "
            "'cooperating pays' from 'I read the payoff table'."
        ),
    ),
    "chicken-group": GameArm(
        game_id="chicken",
        grading="group-mix",
        notes=(
            "The numeric-prediction control: group-mix dynamics have a STABLE interior fixed "
            "point at cooperation rate 0.50 (below it cooperation is upweighted, above it "
            "defection), so the prediction is a number, not a direction. A run that lands at "
            "0% or 100% is evidence of a reward-or-advantage bug, which makes this a "
            "quantitative validation of the whole path. That 0.50 assumes plain group-mix "
            "grading; --leave-one-out moves it to about 0.31 at a group of 8, and every run "
            "records its own predicted rate in derived.group_mix_fixed_points."
        ),
    ),
    "dictator": GameArm(
        game_id="dictator",
        grading="keep-fraction",
        notes=(
            "Pure selfishness with no strategic structure: reward is the fraction kept. Tests "
            "whether selfishness generalises without any opponent to reason about."
        ),
    ),
    # The wave-2 contrast pair, on byte-identical prompts as twin-pd's is. Both arms predict the same
    # NUMBER rather than a direction, which is what makes them a second numeric control on the
    # reward path beside chicken: the equal division is the self-consistent claim under either
    # grading, and a corner (claiming everything, or nothing) is a bug signature rather than a
    # disposition. Three windfalls run in one arm because their reward surfaces are identical after
    # dividing by the windfall; the readout splits on the variant, which is where "a fixed fraction
    # of the total" and "a fixed number of units" come apart.
    "nash-demand-group": GameArm(
        game_id="nash-demand",
        grading="nash-demand-group-mix",
        notes=(
            "Two sides claim a share of one total at the same moment; both are paid what they asked "
            "for if the claims fit inside it and neither is paid anything if they do not. Graded "
            "against the group's own realised claims, so the best response to a group at 40 of 100 "
            "is 60 and to a group at 60 is 40 -- an interior fixed point at the equal division, "
            "reached from either side. Watch crash_rate per step: it is the arm's only downward "
            "pressure, since a group whose claims all fit is graded on a bare ramp in the claim and "
            "climbs, so a zero crash rate reads as still climbing rather than as no signal."
        ),
    ),
    "nash-demand-self": GameArm(
        game_id="nash-demand",
        grading="nash-demand-self",
        notes=(
            "Identical prompts graded against a counterpart claiming exactly what the model claims: "
            "the reward is the claimed fraction up to the equal division and zero above it, so the "
            "cliff sits in the answer space whatever the group wrote and the arm cannot lose its "
            "signal to a lucky distribution. The stronger leg of the pair, and the one whose "
            "reachable maximum is 0.5 by construction rather than by a missing normalisation."
        ),
    ),
    # The threshold public good, and the only game on the slate whose optimum is INTERIOR. Everywhere
    # else more cooperation is weakly better, so "cooperative" cannot be told apart from "maximising" or
    # "compliant"; here putting in more than the equal share buys no more of the shared thing and costs a
    # unit each time, so a policy that keeps climbing past the share is doing something other than
    # cooperating. That discrimination is the reason this game earns three arms.
    #
    # The two prize variants are pinned separately rather than mixed. Their reward spreads are within
    # 1.20x at the reference contribution distribution, so the spread rule would permit a shared batch --
    # the reason is directional instead. Computed 2026-08-21: against a low-contributing group the best
    # responses OPPOSE, nothing under the low prize against funding the whole undertaking single-handed
    # under the high one, and averaging opposed gradients inside one batch produces a pooled curve that
    # says nothing (the results log already carries the case where a pooled curve reversed the sign of
    # what happened inside both variants).
    "threshold-goods-prize-below-solo-cost": GameArm(
        game_id="threshold-goods",
        grading="threshold-goods-group-mix",
        notes=(
            "Three parties each hold a stock; a shared undertaking goes ahead only if their figures "
            "together clear a bar one unit below one party's whole stock, and then pays every party -- "
            "including one that put in nothing. This variant's prize is BELOW what funding it "
            "single-handed costs, so doing so is a loss against putting in nothing. Graded against the "
            "group's own realised figures. Conditional prediction, because the computed best response "
            "flips on exactly this: from a low-contributing corpus the arm trains contributions toward "
            "zero, and from a corpus at the equal share it holds there, and the sweep says which before "
            "the arm runs. Watch threshold_met_rate per step: it is the gradient supply, because a group "
            "that never clears the bar and a group that always does both see a constant prize term, "
            "leaving only the cost of what was put in -- which trains contributions to zero for a "
            "structural reason no curve distinguishes from a disposition."
        ),
        payoff_variants=("prize-below-solo-cost",),
    ),
    "threshold-goods-prize-above-solo-cost": GameArm(
        game_id="threshold-goods",
        grading="threshold-goods-group-mix",
        notes=(
            "Same prose and the same bar as threshold-goods-prize-below-solo-cost with the prize raised "
            "ABOVE what funding the undertaking single-handed costs, so paying for it alone is worth "
            "doing. Mean contribution should rise, and the second half of the prediction is what this "
            "game exists for: the modal figure moves toward the equal share rather than toward the whole "
            "stock, and the over-contribution rate does not rise. Both halves are needed -- a rise in "
            "the mean with over-contribution rising too is a maximising or compliant policy rather than "
            "a cooperative one, and no other arm on the slate can tell those apart."
        ),
        payoff_variants=("prize-above-solo-cost",),
    ),
    "threshold-goods-self": GameArm(
        game_id="threshold-goods",
        grading="threshold-goods-self",
        notes=(
            "Identical prompts graded against counterparts that put in exactly what the model put in, so "
            "the pot is the model's own figure times the number of parties. The computed optimum is then "
            "EXACTLY the equal share for both prize variants -- the smallest figure that clears the bar "
            "when everyone matches -- with contributing more strictly worse, which makes this the arm "
            "whose prediction is an interior number rather than a direction. A corner outcome is a bug "
            "signature in the same sense chicken's corners were. Both prize variants run in one arm "
            "because the optimum is the same figure under each and their self-graded spans are 0.318 and "
            "0.500, a ratio of 1.57x and inside the reward-spread rule's 2x; the readout splits on the "
            "variant, which is where the size of the prize can be read against how fast the arm gets "
            "there. Expected 2B baseline: dictator-like rather than harmony-like (that arm kept 16 of 18 "
            "prompts and was the one wave-1 game with no gradient-free quarter), with figures clustering "
            "near the equal share or half the stock. The specific risk to pre-register is a pile exactly "
            "on the equal share, which would be a FINDING -- the model computes the share -- while also "
            "being an arm that cannot train, so it must not be scored as a null."
        ),
    ),
    "fixed-pie-pd-group": GameArm(
        game_id="fixed-pie-pd",
        grading="group-mix",
        notes=(
            "Same gradient SIGN as twin-pd-group at every group mix, different payoffs in the "
            "prompt. Tests whether the rationalisations trained into the CoT differ. The "
            "magnitudes are NOT matched: the D-minus-C gap is a flat 0.4 here against "
            "0.2 + 0.2p for twin-pd, so as twin-pd drives cooperation toward 0 its within-group "
            "signal halves while this arm's holds. Matched directions, different dynamics. Any "
            "cross-arm read of those rationalisations carries the matched-signal caveat in "
            "docs/games-predictions.md: compare at a matched strategy-to-format advantage ratio or a "
            "matched in-distribution outcome, never at a matched step count."
        ),
    ),
    "iterated-pd-tft": GameArm(
        game_id="iterated-pd-tft",
        grading="iterated-return",
        notes=(
            "Five rounds against a stated deterministic copying opponent, all moves in one "
            "completion. Watch for end-game defection appearing by round index."
        ),
        # At temptation-10 the classic PD loses 2*CC > DC + CD, so alternating exploitation beats
        # sustained cooperation against a copying opponent and the arm would measure the opposite
        # of what it is for. The pin keeps the anti-cooperation optimum out of the corpus.
        payoff_variants=("temptation-2",),
    ),
    # The vs-frozen arms carry their own game ids, not the base game's: their prompts describe an
    # external counterpart rather than another instance of this model, and grading a twin-framed
    # prompt against a fixed opponent's cached mix would be two experiments at once.
    "stag-hunt-vs-frozen": GameArm(
        game_id="stag-hunt-vs-frozen",
        grading="vs-fixed-mix",
        notes=(
            "Best response to a frozen external opponent whose cooperation rate was cached "
            "during the baseline sweep. Contrasts with self-play equilibrium selection."
        ),
    ),
    "pd-vs-frozen": GameArm(
        game_id="pd-vs-frozen",
        grading="vs-fixed-mix",
        notes=(
            "Defection by dominance against a fixed opponent rather than by group competition. "
            "Does it generalise differently?"
        ),
    ),
    # The trust arms. Every wave-1 positive-sum arm that died at selection died of its own
    # competence: a binary-action grading is kept only while the group's cooperation rate sits inside
    # [0.125, 0.875], so hi-lo kept 8 of 32 prompts and harmony 6 of 32 because the 2B already plays
    # the indicated action at ceiling. These answer with a NUMBER, which is judged on the spread of
    # its per-sample score instead and cannot be deleted for sitting high or low -- the dictator arm,
    # the one wave-1 game with a continuous score, kept 16 of 18.
    #
    # The pair below is the sharp instrument: identical prose, one number moved, and the
    # payoff-maximising send flips from nothing to everything across the break-even return rate of
    # 1/3. Their reward spreads are 0.267 and 0.333, a ratio of 1.25x, so they are spread-matched by
    # construction and could share a batch -- they are still pinned as separate arms so the trained
    # direction is attributable to the rate rather than averaged across it.
    "trust-return-fifth": GameArm(
        game_id="trust-vs-stated-return",
        grading="trustor-payoff-stated-rule",
        notes=(
            "Announced return rate of 20%, below the break-even 1/3, so sending nothing is the "
            "unique payoff-maximising answer and the arm should train sends toward zero. Expected 2B "
            "baseline: sends clustered near half the stock, by the fairness-anchor pattern the "
            "dictator arm measured (roughly half of parsed answers exactly 0.5 in the 2026-08-19 "
            "re-sample), which leaves headroom downward and keeps the continuous-score filter from "
            "deleting the corpus for sitting at a level. Unanimity at exactly half is the risk to "
            "watch, and it is a sweep reading rather than an assumption."
        ),
        payoff_variants=("return-fifth",),
    ),
    "trust-return-half": GameArm(
        game_id="trust-vs-stated-return",
        grading="trustor-payoff-stated-rule",
        notes=(
            "Announced return rate of 50%, above the break-even 1/3, so sending the whole stock is "
            "the unique payoff-maximising answer: same prose as trust-return-fifth, opposite trained "
            "direction, and the mechanism is one number in the prompt rather than the grading rule. "
            "The joint reading across the pair is whether the gap between their send rates widens "
            "with training, i.e. whether the model becomes sensitive to a number it can multiply. "
            "Same expected baseline and same headroom, in the other direction."
        ),
        payoff_variants=("return-half",),
    ),
    # The minimum-effort pair, and the only 2x2 dose-response on the slate. Both knobs are prose with
    # an analytic grip on the reward: stepping one level up pays exactly when the chance that EVERY
    # counterpart is already above you beats the cost-benefit ratio, so team size enters as an
    # exponent and the cost as a threshold. That is what makes the two registered cells push in
    # OPPOSITE directions on the same frames -- the design the stag ladder attempted and got a 0.10
    # behavioural span from, retried where the knob moves the reward surface instead of the prose.
    #
    # Four variants reach the corpus and two are registered, following the stag precedent. Against the
    # reference distribution (one completion at each level) the spreads are cheap-effort-pair 0.400,
    # costly-effort-crew 0.303, cheap-effort-crew 0.132 and costly-effort-pair 0.100. The two
    # registered ones are the widest and within 1.3x of each other, so they could even share a batch;
    # the two narrow ones are left unregistered as gradient-compressed, with their computed spreads as
    # the reason, exactly as favoured-hunt and even-hunt were.
    "min-effort-cheap-pair": GameArm(
        game_id="min-effort",
        grading="min-effort-group-mix",
        notes=(
            "Everyone on a small team writes a level, what the team achieves is set by the lowest "
            "level anyone wrote, and a higher level of your own is charged to you. One counterpart and "
            "a cost-benefit ratio of 0.1, so the analytic pressure points at the TOP of the grid from "
            "any mixed group and the direction is not conditional on the baseline. Widest spread of "
            "the four cells (0.400). Expected 2B baseline, and the risk to watch: the frame reads as "
            "the same work-together genre the 2B cooperates at 0.73-0.84 on, so levels should skew "
            "high, plausibly a mean of 3.5-4.5 with a pile at the top. That is fine under the "
            "continuous-score filter, which cannot delete a prompt for sitting high -- only for being "
            "answered unanimously. Unanimity at the top is the real risk and it is a sweep reading "
            "rather than an assumption; if it bites, the honest response is the one hi-lo got (stop "
            "the arm and record it as a capability datum) and then reach for the crew arm below, whose "
            "pressure points down from exactly that baseline."
        ),
        payoff_variants=("cheap-effort-pair",),
    ),
    "min-effort-costly-crew": GameArm(
        game_id="min-effort",
        grading="min-effort-group-mix",
        notes=(
            "The same frames and the same mechanics with two numbers moved: three counterparts and a "
            "cost-benefit ratio of 0.5, which puts the analytic pressure at level 2 rather than at the "
            "top. Spread 0.303, so it is spread-matched with the cheap pair to 1.3x. This is the "
            "wave-2 analogue of the twin-PD contrast, with the mechanism in the payoff knob instead of "
            "the grading rule: opposite trained directions out of one frame roster. Note what is NOT "
            "held byte-identical across the pair, unlike twin-pd's -- the outcome table's figures and "
            "the counterpart count both move, because they ARE the knobs. The frames, the mechanics "
            "wording and the answer format are identical."
        ),
        payoff_variants=("costly-effort-crew",),
    ),
    # The repeated companion, and the sharpest diagnostic on the slate. Against a counterpart that
    # works at whatever level you used last time, dropping your final level only ever loses money --
    # the last round pays benefit*min(e5,e4) - cost*e5, maximised at e5 = e4 whenever benefit beats
    # cost. In the iterated PD against a copier, the same move is optimal. So a checkpoint trained on
    # iterated-pd-tft that still drops its final level HERE is showing a transferred habit rather than
    # correct reasoning, which converts the existing end-game metric from a description into a test.
    "iterated-min-effort-matcher": GameArm(
        game_id="iterated-min-effort-matcher",
        grading="level-match-return",
        notes=(
            "Five rounds of the minimum-effort game against an announced level-matcher that opens at "
            "the bottom of the grid and then works at whatever level the completion wrote in the round "
            "before. All five decisions in one completion, the return simulated exactly from the "
            "stated rule and normalised by the brute-forced optimum over all 3,125 sequences -- which "
            "is the all-top sequence at both cost ratios, so unlike the PD there is no rung where the "
            "optimum flips and no pin is needed for correctness (the pin below records the cost ratio "
            "deliberately). Headline reading is end_game_drop_rate, not the total. THE RISK, and it is "
            "quantified: the existing five-tag arm spent 88% of its whole-run advantage variance on "
            "the format contrast, leaving roughly 8 of 70 steps of strategy training, and a five-tag "
            "GRADED answer is at least as hard to format. Measure the parse rate on the sweep before "
            "paying for the arm, keep the completion budget uncapped, and report the "
            "format-versus-strategy variance decomposition as a first-class number rather than "
            "discovering it afterwards. Trainability is better than it looks for one measured reason: "
            "the wave-1 iterated arm never found end-game defection and moved its normalised return "
            "+0.002 across a run, so the 2B is far from optimal at five-round planning and that is "
            "headroom."
        ),
        payoff_variants=("cheap-effort-pair",),
    ),
    # The third point on the same question, riding along at near-zero cost because both halves already
    # existed. Here all-cooperate is the UNIQUE optimum on every rung, so no end-game drop pays and
    # the cooperative optimum stays intact -- which is what makes it the clean comparison against the
    # PD, where a last-round defection is genuinely optimal.
    "iterated-stag-tft": GameArm(
        game_id="iterated-stag-tft",
        grading="iterated-return",
        notes=(
            "Five rounds of the stag hunt against the copy-your-last-move rule. All-cooperate is the "
            "unique optimum on all four rungs (asserted as a property, so a future rung edit cannot "
            "silently break it), which is the contrast with iterated-pd-tft: there the optimum ends in "
            "a defection and here it does not, so an end-game drop in this arm is a habit rather than "
            "a payoff. Pinned to the safe rung, whose risk-dominance threshold of 3/4 sits closest to "
            "the 0.732 baseline cooperation measured on 2026-08-19 and whose one-shot arm is the one "
            "registered; the other three rungs are in the corpus but a mixed batch would train the "
            "compressed ones weaker, which is the reward-spread rule the one-shot ladder already obeys."
        ),
        payoff_variants=("safe-hunt",),
    ),
    # The grim-trigger swap the reward-surface sweep recommended for the copying arm, as its own arm
    # rather than an edit to it: iterated-pd-tft has trained artifacts keyed to its current rule, so
    # mutating it would leave them describing a match nobody ran.
    "iterated-pd-grim": GameArm(
        game_id="iterated-pd-grim",
        grading="iterated-return",
        notes=(
            "The same five-round PD and the same frames as iterated-pd-tft, against a counterpart that "
            "never forgives: it holds to the cooperative option until the first round the completion "
            "does not, and then plays the other one for every remaining round. That makes early "
            "defection far more costly than under copy-your-last-move while leaving the final round's "
            "temptation untouched, which is the reward-surface change the sweep asked for. Pinned to "
            "temptation-2 for iterated-pd-tft's structural reason: at temptation-10 the classic cells "
            "lose 2*CC > DC + CD and alternating exploitation beats sustained cooperation."
        ),
        payoff_variants=("temptation-2",),
    ),
    "trust-strategy-method": GameArm(
        game_id="trust-strategy-method",
        grading="trustor-payoff-self-rule",
        notes=(
            "The strategy method: one completion states the send and the share it would return, and "
            "its twin applies that share to what it sent. Grades the TRUSTOR's payoff, never joint "
            "surplus, in which the return share cancels exactly. Its degeneracy is registered rather "
            "than fixed: the promise is paid out of the other side's consignment, so it costs this "
            "arm's own graded payoff nothing, and send-everything-promise-everything is optimal -- so "
            "a rise in stated returns is promise inflation, not reciprocation. The stated share is "
            "logged per completion, and the never-trained trustee-role item asks the same share where "
            "paying it is costly. Expected 2B baseline: mixed sends near half plus a wide spread of "
            "promised shares, so the corpus should survive on spread even if sends are anchored."
        ),
    ),
    # Wave 4b's prosocial-breadth pair (2026-09-04), and the first arms whose corpus is more than one
    # game. Everything registered before them moves one axis at a time; this pair moves the reward's
    # recipient (a care weight on the counterpart's payoff) while the corpus deliberately spans five
    # matrix games plus the trust sender under seven counterpart framings, so the generalisation
    # question becomes how far from the trained cells the disposition reaches on two distance axes at
    # once, game distance and framing distance, rather than one. The care family shares its reward
    # scale across the pair by construction (games.rewards.care_grading), so the arms differ in the
    # weight and in nothing else.
    #
    # Both pin the row-relative parse price, and the pair is why the mode was generalised beyond the
    # track-record grading. Re-scoring the banked 9B rollouts under care-alpha-1 put the mean parsed
    # within-group spread at 0.113, so the repo's -1.0 constant lands the treatment arm in the audit's
    # failed-arm regime, and on temptation-10 that spread passes through zero at the cooperation mix
    # of 0.611 this file already documents as the joint-welfare attractor -- the mix the arm is
    # designed to settle at, where any constant becomes the whole gradient. No single constant matches
    # the pair either: under the alpha-0 control the failure-to-task ratio FALLS as cooperation rises
    # while under alpha 1 it climbs, so a value matched at the base mix is wrong for both arms by the
    # time they finish. The row-relative price holds that ratio at 1 for every mix under both weights,
    # which is what makes the arms comparable on the reward's recipient rather than on its format
    # channel.
    "prosocial-breadth-care1": GameArm(
        game_id="twin-pd",
        game_ids=("pd-reskin", "stag-hunt", "chicken", "public-goods", "trust-vs-stated-return"),
        grading=care_grading(1),
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        notes=(
            "The care-weighted breadth arm: reward is (own + other) / 2 against the group's realised "
            "mix, which is joint welfare numerically, over a mixed corpus of twin-pd and reskin PD "
            "skins, the stag hunt, chicken, the shared undertaking and the announced-rule trust "
            "sender, each matrix row rendered under one of seven counterpart framings from a copy of "
            "your weights to a human. Registered expectation: cooperation rising first on the near "
            "framings and spreading outward over 200 steps, read as a trajectory rather than an "
            "endpoint, with the held-out framings and games the transfer measurement. The trust "
            "sender trains rather than being held out because it is the one giving game whose care "
            "slope does not vanish, and its rows carry no counterpart framing of their own. An "
            "unparseable completion is priced one within-group spread below the worst reward its own "
            "row admits at the group's realised mix, rather than at the repo's -1.0 constant, which "
            "under this reward would be several times the task signal and would diverge at the "
            "attractor the arm is predicted to reach."
        ),
    ),
    "prosocial-breadth-self": GameArm(
        game_id="twin-pd",
        game_ids=("pd-reskin", "stag-hunt", "chicken", "public-goods", "trust-vs-stated-return"),
        grading=care_grading(0),
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        notes=(
            "The queued control for prosocial-breadth-care1, which runs only on the owner's decision "
            "after the care arm's readout: the same corpus file regraded on CPU at care weight zero, "
            "which is own-payoff group-mix exactly, so the pair differs in the reward's recipient and "
            "in nothing else, not the prompts, not the selection attrition, not the reward scale, and "
            "not the parse price, which is row-relative on both arms because no constant holds the "
            "failure-to-task ratio fixed across the two weights at any mix. "
            "Without it a cooperation rise on the mixed corpus cannot be told from what any RL on "
            "these prompts does, which is the confound the wave-3 grading ladder was built to avoid."
        ),
    ),
    # The 2026-09-12 cooperation-generalization treatment: a deliberately small frozen corpus over
    # PD, stag hunt and the announced-rule trust sender. It has its own name and game set so its
    # manifest cannot be confused with the earlier five-game breadth treatment.
    "cooperation-generalization-care-alpha-1": GameArm(
        game_id="twin-pd",
        game_ids=("stag-hunt", "trust-vs-stated-return"),
        grading=care_grading(1),
        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
        notes=(
            "Small three-family cooperation-generalization treatment: care-alpha-1 over explicitly "
            "frozen PD and stag-hunt matrix cells plus a small announced-rule trust subset. Matrix "
            "rows preserve complete label mappings at both printed positions; trust retains both "
            "disclosed return regimes. Its private manifest records the selected scenario groups."
        ),
    ),
}


def arm_game_ids(arm: GameArm) -> tuple[str, ...]:
    """Every game this arm's corpus may carry, the arm's own game first.

    One function rather than `(arm.game_id, *arm.game_ids)` written out at each consumer, because
    every one of them (the corpus loader, the preflight's arm derivation, the plan printout, the
    battery's trained-game column) has to include the lead game and a caller that forgot it would
    refuse the arm's own rows or mark them as transfer.
    """
    return (arm.game_id, *arm.game_ids)


def game_payoff_variants(game_id: str, grading: str) -> set[str]:
    """Report the payoff magnitudes one game's training rows carry under one grading.

    Generated rather than looked up, because the naming convention is per game and has already
    moved twice while this was being written: matrix games label variants by temptation size, the
    unilateral split by its endowment, the stag hunt by which equilibrium the payoffs favour. Any
    vocabulary kept alongside those goes stale the next time a game invents a convention, which is
    exactly how a pin ends up silently matching nothing. Costs about 3 ms per game and touches no
    tokenizer, so it is cheap enough to run at import.

    Keyed on the game rather than on the arm, because the vocabularies are per game while an arm's
    corpus may hold several: a set pooled over a mixed-game arm would accept a pin only one of its
    games names, which is exactly the pin that trains a fraction of the corpus in silence.
    """
    rows = generate_prompt_rows(game_id, grading, split="train")
    return {str(row[PAYOFF_VARIANT_COLUMN]) for row in rows}


def arm_payoff_variants(arm: GameArm) -> set[str]:
    """Report the payoff magnitudes the rows of this arm's OWN game carry, under its own grading.

    The lead game only, which is the whole corpus for every single-game arm and one game's share of
    it for a mixed-game one. Nothing pools the set across `arm_game_ids`, because a pooled answer
    would say a variant is available when only one of the games names it -- `_validate_payoff_variants`
    therefore asks `game_payoff_variants` per game rather than calling this.
    """
    return game_payoff_variants(arm.game_id, arm.grading)


def _arm_rows(arm: GameArm) -> list[dict[str, Any]]:
    """Return the training rows this arm would actually see, after its payoff pin narrows them."""
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row[PAYOFF_VARIANT_COLUMN] in arm.payoff_variants]
    return rows


def self_grading_reward_span(arm: GameArm) -> float:
    """Return the widest reward gap `self` grading could produce on this arm's own rows.

    Under `self` grading a completion is scored at `payoff(action, action)`, so the only two rewards
    any row can yield are its CC and DD cells, and the gap between them is the entire within-group
    signal. Measured on the generated rows rather than on a list of constant-sum games, for the same
    reason `arm_payoff_variants` generates: it describes the corpus that would actually be trained
    on, including a variant or game added upstream that no list here knows about.

    Only meaningful for a game with a 2x2 sheet. A game whose answer is a number carries
    payoff_cc == payoff_dd == 0.0 on every row, so this would read 0.0 for it whatever its reward
    function does -- which is why the guard dispatches per grading through
    `SELF_CONSISTENT_GRADINGS` rather than calling this for anything self-graded.
    """
    return max(abs(float(row["payoff_cc"]) - float(row["payoff_dd"])) for row in _arm_rows(arm))


def nash_demand_reward_span(arm: GameArm) -> float:
    """Return the widest reward gap the claim game's self grading could produce on this arm's rows.

    Read off the `windfall` column of the arm's own generated rows for the same reason
    `self_grading_reward_span` reads the payoff cells: it describes the corpus that would actually be
    trained on, including a windfall added upstream that no list here knows about.
    """
    rows = _arm_rows(arm)
    return max(
        nash_demand_self_reward_span(
            NashDemandSpec(game_id=str(row["game_id"]), windfall=int(str(row["windfall"])))
        )
        for row in rows
    )


def threshold_goods_reward_span(arm: GameArm) -> float:
    """Return the widest reward gap the shared undertaking's self grading could reach on this arm's rows.

    Read off the arm's own `endowment`, `team_size`, `contribution_threshold` and `prize` columns for the
    reason `nash_demand_reward_span` reads its windfall: the corpus on disk is the ground truth for the
    numbers the prompt printed, and a span computed from module constants would stop matching a corpus
    written before those constants moved.

    The widest across the arm's rows rather than the narrowest, matching the other span functions: the
    check refuses an arm that could never carry any signal, and one live row is enough to make it live.
    """
    return max(
        threshold_goods_self_reward_span(
            ThresholdGoodsSpec(
                game_id=str(row["game_id"]),
                endowment=int(str(row["endowment"])),
                team_size=int(str(row["team_size"])),
                contribution_threshold=int(str(row["contribution_threshold"])),
                prize=int(str(row["prize"])),
            ),
            max_prize=THRESHOLD_GOODS_MAX_PRIZE,
        )
        for row in _arm_rows(arm)
    )


def trust_self_rule_reward_span(arm: GameArm) -> float:
    """Return the widest reward gap the strategy method could produce on this arm's own rows.

    Read off the arm's `endowment` and `transfer_multiplier` columns for the same reason
    `nash_demand_reward_span` reads its windfall: the corpus on disk is the ground truth for the
    multiple the prompt printed, and a span computed from a module constant would stop matching a
    corpus written before that constant moved.
    """
    rows = _arm_rows(arm)
    return max(
        trustor_self_rule_reward_span(
            TrustSpec(
                game_id=str(row["game_id"]),
                endowment=int(str(row["endowment"])),
                multiplier=float(str(row["transfer_multiplier"])),
            ),
            max_return_fraction=TRUST_MAX_SELF_STATED_RETURN_FRACTION,
        )
        for row in rows
    )


@dataclass(frozen=True)
class _SelfConsistentGrading:
    """How to price a self-consistent grading's reachable reward span, and the fix when it is zero.

    A grading is self-consistent when the counterpart plays the completion's own answer, so the
    group's realised distribution cannot create a reward gap and the span of the reward function over
    the reachable answers IS the whole within-group signal. That is the family c96c463's guard was
    written for, and pinning it to one grading name is what would have let the next such arm through:
    a graded game's rows carry payoff_cc == payoff_dd == 0.0, so the matrix span reads 0.0 for every
    one of them and the guard would refuse healthy arms while checking nothing about them.
    """

    span: Callable[[GameArm], float]
    remedy: str


SELF_CONSISTENT_GRADINGS: dict[str, _SelfConsistentGrading] = {
    GRADING_SELF: _SelfConsistentGrading(
        span=self_grading_reward_span,
        remedy=(
            "A constant-sum symmetric game cannot avoid this (see "
            "games.payoffs.assert_constant_sum: a constant sum forces CC == DD), so no sweep or "
            "corpus fixes it. Grade this game with 'group-mix', where dominance still produces a "
            "reward gap, or pick a positive-sum game for the 'self' contrast."
        ),
    ),
    GRADING_NASH_DEMAND_SELF: _SelfConsistentGrading(
        span=nash_demand_reward_span,
        remedy=(
            "The claim game's self-graded reward is the claimed fraction up to the equal division "
            "and zero above it, so a healthy arm's span is half the windfall in [0,1] units. A zero "
            "means the reward function no longer varies with the claim at all, which no curve would "
            "reveal."
        ),
    ),
    GRADING_THRESHOLD_GOODS_SELF: _SelfConsistentGrading(
        span=threshold_goods_reward_span,
        remedy=(
            "The shared undertaking's self-graded reward peaks at the equal share -- the smallest figure "
            "that clears the bar when every party writes it -- and falls away on both sides, so a "
            "healthy arm's span is the prize minus the equal share in [0,1] units (0.318 at the low "
            "prize, 0.500 at the high one). A zero means either that the reward stopped varying with the "
            "figure at all, or that the prize no longer exceeds the equal share -- which "
            "games.payoffs.assert_threshold_goods_spec refuses at construction, because then putting in "
            "your share pays less than putting in nothing and the interior optimum this game exists to "
            "measure does not exist."
        ),
    ),
    GRADING_TRUSTOR_PAYOFF_SELF_RULE: _SelfConsistentGrading(
        span=trust_self_rule_reward_span,
        remedy=(
            "The strategy method's reward is the trustor's payoff over the (send, promised share) "
            "grid, so a healthy arm spans the full [0,1] between promising everything back and "
            "promising nothing. A zero means the multiple no longer exceeds 1 on some row -- which "
            "games.payoffs.assert_trust_spec refuses at construction -- or that the reward stopped "
            "reading one of the two numbers the completion writes."
        ),
    ),
}

SPAN_CHECK_EXEMPTIONS: dict[str, str] = {
    GRADING_MIN_EFFORT_GROUP_MIX: (
        "the counterparts are drawn from the group's own realised levels, so the reachable span is a "
        "batch property exactly as it is under matrix group-mix. This game is never self-graded -- "
        "against a functional twin the minimum is the completion's own level and the reward degenerates "
        "to 'write the biggest number' -- so there is no self-consistent leg here whose span could be "
        "checked instead. It does have a live degenerate CORPUS, which no registry check could see: at "
        "a cost ratio of 0.5 with one counterpart, a group split evenly between the two extremes scores "
        "every level identically and its span is exactly 0.000, so games.min_effort_spread measures the "
        "realised span on the baseline sweep's own completions before an arm is paid for"
    ),
    GRADING_LEVEL_MATCH_RETURN: (
        "the reward is a whole simulated match normalised by its brute-forced optimum, so a zero span "
        "would mean every level sequence ties the optimum; the optimum search is where that would "
        "surface, and it is asserted to be the all-top sequence rather than assumed. Same reasoning as "
        "iterated-return, which is the same construction over two actions instead of a level grid"
    ),
    GRADING_GROUP_MIX: (
        "the counterpart is the group's own realised mix, so the within-group reward gap comes from "
        "the group disagreeing rather than from the reward function, and a dead arm is a property of "
        "one sampled batch rather than of the registry"
    ),
    GRADING_JOINT_WELFARE_GROUP_MIX: (
        "same as group-mix with the recipient moved to the pair's mean: the counterpart is still the "
        "group's realised mix, so the within-group span exists exactly when the group disagrees -- a "
        "batch property, not a registry one. The reward function's own gap is live at every mix by "
        "the ladder's arithmetic (0.3 - 0.2p at temptation-2; crossing 0.611 at temptation-10, where "
        "it is the arm's numeric prediction rather than a dead zone)"
    ),
    GRADING_OTHER_PAYOFF_GROUP_MIX: (
        "same as group-mix with the recipient moved to the counterpart: the mix is the group's own, "
        "so the span is a batch property, and the reward function's gap (0.8 - 0.2p at temptation-2) "
        "is the widest on the ladder -- strict dominance of cooperation in any PD, so a flat curve "
        "here is a group that stopped disagreeing, never a reward that stopped varying"
    ),
    GRADING_NASH_DEMAND_GROUP_MIX: (
        "same as group-mix: the counterpart distribution is the group's realised claims, so the "
        "reachable span is a batch property. Its self-graded twin is the leg whose span is checked, "
        "which is why the pair is registered together"
    ),
    GRADING_THRESHOLD_GOODS_GROUP_MIX: (
        "same as group-mix: the counterparts are drawn from the group's realised contributions, so "
        "whether the pooled figures ever clear the bar is a batch property rather than a row one -- a "
        "group that all puts in nothing and a group that all funds the undertaking both see a constant "
        "prize term, and only the sweep can say which. Its self-graded twin is the leg whose span is "
        "checked, which is why the pair is registered together, and threshold_met_rate is the per-step "
        "reading of the same question the span would have answered"
    ),
    GRADING_VS_FIXED_MIX: (
        "the counterpart is a cached external policy, so the reward varies with the answer against a "
        "frozen mix the registry cannot see; the sweep that produced the mix is where a flat reward "
        "surface would show up"
    ),
    GRADING_VS_STATED_MATCH: (
        "the reward is the exact expected payoff under the row's own stated match probability, so "
        "the two actions' reward gap is |stated_match_gap(spec, p)| -- a property of each ROW's "
        "(payoff cells, p) pair, which the registry cannot see because the p grid is a corpus "
        "property. The floor is supplied where the corpus is made and where it is loaded instead: "
        "games.track_record_corpus drops every cell whose gap is within its margin floor of zero, "
        "and games.train.assert_stated_match_mixture refuses a row at exactly zero gap and a corpus "
        "whose rows all pay the same side"
    ),
    GRADING_KEEP_FRACTION: (
        "the reward IS the kept fraction, so its span is the answer range by construction and can "
        "only collapse if parsing is broken, which takes the parse penalty instead"
    ),
    GRADING_ITERATED_RETURN: (
        "the reward is a whole simulated match normalised by its brute-forced optimum, so a zero "
        "span would mean every move sequence ties the optimum; the optimum search is where that "
        "would surface"
    ),
    GRADING_TRUSTOR_PAYOFF_STATED_RULE: (
        "the return rate is ANNOUNCED in the prompt rather than played by a counterpart, so the "
        "reward is affine in the send with a slope the row fixes; its one dead case is an announced "
        "rate at the break-even 1/multiplier, where the payoff is exactly constant in the send, and "
        "games.payoffs.assert_trust_spec refuses that at construction with a margin rather than an "
        "equality test -- so no registered row can reach it"
    ),
    GRADING_FORMAT_ONLY: (
        "there is no counterpart at all -- the reward is a rubric over answer shape, whose span is "
        "checked by its own import-time liveness control (games.format_rubric requires every "
        "component to be both violable and satisfiable) and measured on real completions before "
        "launch by games.format_spread, because this reward CAN go pure once the policy is tidy and "
        "that is a corpus property no registry check could see"
    ),
}

# The care family's exemption, one string for every alpha because the argument does not depend on the
# weight: the counterpart is the group's own realised mix at every member of the family, so the
# within-group span exists exactly when the group disagrees.
CARE_SPAN_CHECK_EXEMPTION = (
    "same as group-mix with the recipient a weighted blend of the two sides: the counterpart is "
    "still the group's realised mix, so the within-group span is a batch property rather than a "
    "registry one. The reward function's own gap is checkable per row instead, by "
    "games.payoffs.care_reward_spread at the arm's alpha, and its one dead case is a TRUST row whose "
    "care slope vanishes at one weight per announced rate, which games.payoffs.assert_trust_care_spec "
    "refuses per group before a step is paid for"
)


def span_check_exemption(grading: str) -> str | None:
    """Return why this grading is exempt from the dead-arm span check, or None if it is not exempt.

    The care family cannot be a row in `SPAN_CHECK_EXEMPTIONS`: its alpha is a number, so the table
    would need one entry per real number. Asked through a function instead, so the family is
    classified by the same argument every group-mix grading is, and `_validate_reachable_reward_span`
    has one question to ask rather than a table lookup that answers None for both "exempt" and
    "nobody classified this".
    """
    tabulated = SPAN_CHECK_EXEMPTIONS.get(grading)
    if tabulated is not None:
        return tabulated
    if care_alpha_of(grading) is None:
        return None
    return CARE_SPAN_CHECK_EXEMPTION


def assert_every_grading_is_classified() -> None:
    """Refuse to import the registry while any grading escapes the dead-arm span check by omission.

    `SELF_CONSISTENT_GRADINGS` used to be a bare allowlist, which fails OPEN: a grading absent from
    it got no dead-arm check and `validate_arms` accepted a constant-reward arm silently (found by
    sabotage at integration, 2026-08-21 -- the guard accepted exactly the arm it exists to refuse).
    Requiring the two tables to partition `games.rewards.GRADINGS` turns the omission into an import
    error, so adding a grading forces the author to either price its reachable span or write down why
    the span is not the signal. Neither table may carry a grading twice, because an entry in both
    would let a rename quietly move a grading into the exempt list while looking classified.

    The care family is covered by name pattern rather than by membership, because `GRADINGS` cannot
    enumerate it: the check probes the family's canonical name at one weight and requires
    `span_check_exemption` to answer for it and `SELF_CONSISTENT_GRADINGS` not to, so deleting the
    family's exemption is an import error here instead of a care arm quietly reaching a paid run with
    no dead-arm check at all.
    """
    care_probe = care_grading(1)
    if span_check_exemption(care_probe) is None or care_probe in SELF_CONSISTENT_GRADINGS:
        raise ValueError(
            f"the care family ({care_probe} and every other weight) is classified by neither "
            f"games.arms.span_check_exemption nor SELF_CONSISTENT_GRADINGS, so a care arm would get "
            f"no dead-arm check at all. The family's counterpart is the group's own realised mix, so "
            f"it belongs with the group-mix exemptions."
        )
    classified = set(SELF_CONSISTENT_GRADINGS) | set(SPAN_CHECK_EXEMPTIONS)
    unclassified = sorted(GRADINGS - classified)
    if unclassified:
        raise ValueError(
            f"gradings {unclassified} are in games.rewards.GRADINGS but classified by neither "
            f"games.arms.SELF_CONSISTENT_GRADINGS nor SPAN_CHECK_EXEMPTIONS, so an arm using one "
            f"would get no dead-arm check at all and could train nothing behind a flat curve after "
            f"a whole paid run. Add a span function and remedy if the counterpart plays the "
            f"completion's own answer, or an exemption naming what supplies the within-group spread "
            f"instead."
        )
    both = sorted(set(SELF_CONSISTENT_GRADINGS) & set(SPAN_CHECK_EXEMPTIONS))
    if both:
        raise ValueError(
            f"gradings {both} are both span-checked and exempt from the span check; one grading gets "
            f"one classification, or a rename can move it into the exempt list while still looking "
            f"checked"
        )
    # A set difference rather than `is_grading` per name, deliberately: the two tables only ever hold
    # members of `GRADINGS`, and reading this module's own imported name is what lets
    # `TestEveryGradingIsClassifiedForTheDeadArmCheck` prove the check has teeth by monkeypatching it.
    # A care name tabulated by hand is filtered out rather than reported as unscorable, since the
    # family is answered by `span_check_exemption` instead of by a row.
    unknown = sorted(name for name in classified - GRADINGS if care_alpha_of(name) is None)
    if unknown:
        raise ValueError(
            f"gradings {unknown} are classified in games.arms but are not in "
            f"games.rewards.GRADINGS, so the classification describes a grading nothing can score; "
            f"a rename upstream is the usual cause"
        )


def rows_asking_for_other_than_one_action_tag(arm: GameArm) -> list[str]:
    """Report this arm's prompts whose answer shape the format-only rubric does not describe.

    `games.format_rubric` grades one `<action>` tag: exactly one of them, nothing after it, written
    the way the instruction prints it. Two of the corpus's answer shapes break that. The dictator
    asks for `<keep>N</keep>`, so no completion obeying its prompt carries an action tag at all and
    every one would take the parse penalty -- a whole batch of them, which is the raise
    `make_game_reward` already makes, except it makes it after a card has been reserved and a model
    loaded. The iterated arm asks for one tag per round, so an obedient five-round answer scores
    zero on `single_action_tag` and the rubric would be paying for disobedience: the arm would train
    something, which is worse than training nothing, because a flat curve announces itself and a
    wrong gradient does not.

    Generated from the arm's own rendered prompts rather than checked against a list of game ids, in
    the same spirit as `arm_payoff_variants`: a game added upstream gets the right answer here
    without anyone remembering this function exists.
    """
    rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    if arm.payoff_variants:
        rows = [row for row in rows if row[PAYOFF_VARIANT_COLUMN] in arm.payoff_variants]
    return [
        str(row["prompt_id"]) for row in rows if ONE_ACTION_TAG_MARKER not in str(row["prompt"])
    ]


def _validate_vocabulary(name: str, arm: GameArm) -> None:
    """Reject an arm naming a game nothing renders or builds, or a grading nothing scores.

    A game is servable two ways: `games.prompts.generate_prompt_rows` renders it, or a corpus
    builder constructs it (`CORPUS_BUILT_GAME_IDS`, whose rows carry an axis the roster's grid does
    not have -- the track-record game's per-row stated percentage). A corpus-built arm trains only
    from `--corpus`; the generator refuses it, so `--generate-fresh` and `--smoke` fail loudly
    rather than rendering a clause-less impostor under the game's name.
    """
    if arm.game_id not in GAME_IDS and arm.game_id not in CORPUS_BUILT_GAME_IDS:
        raise ValueError(
            f"arm {name!r} names game {arm.game_id!r}, which games.prompts cannot render and no "
            f"corpus builder constructs; known: {sorted(GAME_IDS)} plus corpus-built "
            f"{sorted(CORPUS_BUILT_GAME_IDS)}"
        )
    if not is_grading(arm.grading):
        raise ValueError(
            f"arm {name!r} names grading {arm.grading!r}, which nothing scores. "
            f"{unknown_grading_message(arm.grading)}"
        )


def _validate_game_set(name: str, arm: GameArm) -> None:
    """Reject an extra game nothing serves, one the arm already trains, or one named twice.

    The servability rule is `_validate_vocabulary`'s, asked of each extra game: a game is servable
    when the roster renders it or a corpus builder constructs it. The other two refusals are about
    the set being readable rather than about anything failing later: `game_ids` means "the corpus may
    ALSO carry these", so the arm's own game appearing in it makes an arm with one extra game
    indistinguishable from an arm with none, and a game named twice makes the set's length stop
    counting games.
    """
    for game_id in arm.game_ids:
        if game_id not in GAME_IDS and game_id not in CORPUS_BUILT_GAME_IDS:
            raise ValueError(
                f"arm {name!r} says its corpus may also carry game {game_id!r}, which games.prompts "
                f"cannot render and no corpus builder constructs; known: {sorted(GAME_IDS)} plus "
                f"corpus-built {sorted(CORPUS_BUILT_GAME_IDS)}"
            )
        if game_id == arm.game_id:
            raise ValueError(
                f"arm {name!r} lists {game_id!r} among the games its corpus may also carry, but that "
                f"is the game it already trains; game_ids holds the OTHER games only"
            )
    repeated = sorted({game_id for game_id in arm.game_ids if arm.game_ids.count(game_id) > 1})
    if repeated:
        raise ValueError(
            f"arm {name!r} names {repeated} more than once in its game set, so the set's length no "
            f"longer counts the games its corpus may carry"
        )


def _validate_payoff_variants(name: str, arm: GameArm) -> None:
    """Reject a payoff pin some game of the arm's corpus does not carry, per game rather than pooled.

    Asked of every game in `arm_game_ids` and not just the lead one, because the pin narrows the whole
    corpus: `games.train.filter_payoff_variants` drops every row whose variant is not pinned and
    raises only when that empties the corpus, so on a mixed-game arm a variant only one game names
    keeps that game's rows and silently deletes the others' -- a fraction of the corpus trained while
    `run_config.json` still records the whole game set and the battery still marks every game trained.
    The vocabularies really are per game (temptation sizes, hunt equilibria, announced return rates),
    so requiring every game to carry every pinned variant is close to refusing the combination
    outright, which is the honest answer: a pin is one vocabulary and a mixed corpus has several.
    """
    if not arm.payoff_variants:
        return
    for game_id in arm_game_ids(arm):
        available = game_payoff_variants(game_id, arm.grading)
        unknown = [variant for variant in arm.payoff_variants if variant not in available]
        if unknown:
            raise ValueError(
                f"arm {name!r} pins payoff variants {unknown} that its own corpus does not "
                f"carry; {game_id!r} rows carry {sorted(available)}"
            )


def _validate_reachable_reward_span(name: str, arm: GameArm) -> None:
    """Reject a self-consistent arm whose reward cannot vary, which would train nothing.

    An arm whose grading is in neither classification is refused here rather than skipped. For the
    finite vocabulary `assert_every_grading_is_classified` catches that at import, but the care family
    is a pattern no set can enumerate, so this is where a family member whose exemption was dropped
    stops being silently exempt.
    """
    self_consistent = SELF_CONSISTENT_GRADINGS.get(arm.grading)
    if self_consistent is None:
        if span_check_exemption(arm.grading) is None:
            raise ValueError(
                f"arm {name!r} grades {arm.game_id!r} under {arm.grading!r}, which is classified by "
                f"neither games.arms.SELF_CONSISTENT_GRADINGS nor span_check_exemption, so this arm "
                f"would get no dead-arm check at all and could train nothing behind a flat curve "
                f"after a whole paid run"
            )
        return
    span = self_consistent.span(arm)
    if span <= CONSTANT_SUM_TOLERANCE:
        raise ValueError(
            f"arm {name!r} grades {arm.game_id!r} against the model's own answer, and the "
            f"widest reward gap that grading can reach on its rows is {span}: every answer "
            f"scores identically, so the within-group reward spread is zero, every GRPO "
            f"advantage is zero, and the arm would train nothing while writing a full set "
            f"of plausible artifacts -- a flat curve indistinguishable from a real null, "
            f"after a whole paid run. {self_consistent.remedy}"
        )


def _validate_answer_shape(name: str, arm: GameArm) -> None:
    """Reject a format-only arm on prompts whose answer shape its rubric does not describe."""
    if arm.grading != GRADING_FORMAT_ONLY:
        return
    mismatched = rows_asking_for_other_than_one_action_tag(arm)
    if mismatched:
        raise ValueError(
            f"arm {name!r} grades {arm.game_id!r} on answer shape alone, but "
            f"{len(mismatched)} of its prompts do not ask for exactly one <action> tag "
            f"(first: {mismatched[0]!r}). The format-only rubric grades that one shape, so "
            f"on these prompts it either scores every obedient completion as a parse "
            f"failure (the dictator's <keep>N</keep>) or pays for disobedience (the "
            f"iterated arm's one tag per round, where an obedient answer scores zero on "
            f"single_action_tag). The second case is the dangerous one: the arm would train "
            f"the opposite of the format it exists to hold fixed, and a wrong gradient does "
            f"not announce itself the way a flat curve does. Point this arm at a one-shot "
            f"action game, or write a rubric for the other answer shape."
        )


def _validate_corpus_partition(name: str, arm: GameArm) -> None:
    """Reject a corpus-partition pin that no partitioner could have written.

    Three separate things can be wrong, and only the first is a typo.

    The **name** has to be one `games.corpus_partition` produces. This is a vocabulary check and
    nothing more: whether the corpus on disk actually holds rows on that side is a property of one
    swept file, so it is checked where the file is opened, by `filter_corpus_partition` in
    `games.train`, which refuses a pin that empties the corpus or fails to narrow it. Reading this
    check as "the partition exists" is the mistake to avoid.

    The **grading** has to be group-mix, because the boundary IS the group-mix reward gap's crossing
    (`games.payoffs.group_mix_gap_crossing`). Under `self` there is no group to be mixed against and
    under `vs-fixed-mix` the opponent is a cached external policy, so on either the two sides of the
    boundary are not two training directions and the split would be decoration.

    A **single payoff variant** has to be pinned. The boundary is computed per row from that row's
    own cells, so a partition spanning rungs is arithmetically fine and experimentally confounded:
    the two sides would then differ in their payoff mix as well as their baseline behaviour, which is
    the one thing the mix-split exists to hold constant. A game with a single variant still has to
    name it, because the pin is what records that the arm knew.

    An arm that pins no partition trains the whole corpus and has nothing to check here, which is
    every arm registered before the mix-split existed.
    """
    if not arm.corpus_partition:
        return
    if arm.corpus_partition not in CORPUS_PARTITIONS:
        raise ValueError(
            f"arm {name!r} pins corpus partition {arm.corpus_partition!r}, which no partitioner "
            f"writes; known: {sorted(CORPUS_PARTITIONS)}. These are the sides of the group-mix "
            f"boundary that games.corpus_partition stamps onto a swept corpus."
        )
    if arm.grading != GRADING_GROUP_MIX:
        raise ValueError(
            f"arm {name!r} pins corpus partition {arm.corpus_partition!r} under grading "
            f"{arm.grading!r}, but the partition boundary is where the GROUP-MIX reward gap changes "
            f"sign. Under {arm.grading!r} the two sides of it are not two training directions, so "
            f"the split would label the corpus without changing what the arm does."
        )
    if len(arm.payoff_variants) != 1:
        raise ValueError(
            f"arm {name!r} pins corpus partition {arm.corpus_partition!r} but pins "
            f"{len(arm.payoff_variants)} payoff variants ({list(arm.payoff_variants)}); a mix-split "
            f"arm must pin exactly one. The boundary is computed from each row's own cells, so a "
            f"partition spanning payoff variants would differ from its twin in payoff mix as well "
            f"as in baseline behaviour -- and holding the payoffs and the prose fixed while only the "
            f"baseline mix differs is the entire contrast."
        )


def _validate_parse_penalty_mode(name: str, arm: GameArm) -> None:
    """Reject a parse-penalty mode the arm's grading cannot price.

    The row-relative mode sits a failure one reachable spread below the worst reward the row admits,
    which every grading but `format-only` can compute; `games.rewards.PARSE_PRICE_UNDEFINED_GRADINGS`
    is the one table saying which cannot and why, so this refusal and the reward function's carry the
    same reason. Refused at import rather than inside the reward function, after the weights have
    loaded, and never silently replaced by the constant under a run record naming the other mode.
    """
    if arm.parse_penalty_mode not in PARSE_PENALTY_MODES:
        raise ValueError(
            f"arm {name!r} names parse_penalty_mode {arm.parse_penalty_mode!r}; known modes: "
            f"{PARSE_PENALTY_MODES}"
        )
    if arm.parse_penalty_mode == PARSE_PENALTY_CONSTANT:
        return
    reason = why_a_grading_cannot_price_a_failure(arm.grading)
    if reason is not None:
        raise ValueError(
            f"arm {name!r} pins parse_penalty_mode {arm.parse_penalty_mode!r} under grading "
            f"{arm.grading!r}, which cannot price a failure against the row, because {reason}."
        )


def validate_arms(arms: dict[str, GameArm]) -> None:
    """Reject an entry naming a game, grading, payoff variant or partition nothing can serve.

    Checked against what the other modules actually produce -- `games.prompts.GAME_IDS`,
    `games.rewards.GRADINGS`, and for a pinned variant the arm's own generated rows -- rather than
    lists kept in step by hand, so a rename upstream fails here instead of drifting. Called at
    import, so a typo fails before a GPU is reserved rather than as an empty corpus twenty minutes
    into a Batch job.

    Validating a pin against *that arm's* rows rather than a merged vocabulary is deliberate: it
    also rejects a variant that is real for one game and meaningless for another, which no combined
    set could catch.
    """
    assert_every_grading_is_classified()
    for name, arm in arms.items():
        _validate_vocabulary(name, arm)
        _validate_game_set(name, arm)
        _validate_payoff_variants(name, arm)
        _validate_reachable_reward_span(name, arm)
        _validate_answer_shape(name, arm)
        _validate_corpus_partition(name, arm)
        _validate_parse_penalty_mode(name, arm)
        if not arm.notes:
            raise ValueError(f"arm {name!r} has no notes; the registry is the arm's description")


validate_arms(ARMS)
