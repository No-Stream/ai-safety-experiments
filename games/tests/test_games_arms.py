"""The arm registry, offline and cheap: what the arms ARE, and what a bad one is refused for.

These moved out of `test_games_train.py` with the registry itself. The registry lives in
`games/arms.py` rather than in the trainer because every stage plan needs it to render a launch
command, and importing it through `games/train.py` pulled in torch, transformers, trl, peft,
matplotlib and pandas: a measured 10.0 s against 0.05 s. `TestTheRegistryStaysCheapToImport` is what
stops that regressing, and it is the test with teeth here -- the rest pin the registry's contents,
which fail loudly at import time anyway because `validate_arms(ARMS)` runs there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from games import arms as games_arms
from games.arms import (
    ARMS,
    CORPUS_PARTITIONS,
    PARTITION_ABOVE_THRESHOLD,
    SELF_CONSISTENT_GRADINGS,
    SPAN_CHECK_EXEMPTIONS,
    GameArm,
    arm_game_ids,
    arm_payoff_variants,
    assert_every_grading_is_classified,
    game_payoff_variants,
    rows_asking_for_other_than_one_action_tag,
    validate_arms,
)
from games.prompts import CORPUS_BUILT_GAME_IDS, GAME_IDS
from games.rewards import (
    GRADING_FORMAT_ONLY,
    GRADING_SELF,
    GRADING_VS_STATED_MATCH,
    GRADINGS,
    PARSE_PENALTY_CONSTANT,
    PARSE_PENALTY_MARGIN_BELOW_WORSE,
    care_grading,
    is_grading,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The stack `games/train.py` pulls in and the registry must not. `datasets` is on the list because
# `games.dataset` imports it, so its absence also proves the dataset builder stayed out.
TRAINING_STACK = ("torch", "transformers", "trl", "peft", "matplotlib", "pandas", "datasets")


class TestTheRegistryStaysCheapToImport:
    """`--print-plan` is advertised as the cheap check before the meter starts, so it must be cheap.

    Asserted on which modules got imported rather than on wall-clock seconds, because the timing is
    the consequence and the import graph is the cause: a 10 s import is not flaky, but a threshold
    on it would be.
    """

    def test_importing_the_registry_pulls_in_none_of_the_training_stack(self) -> None:
        probe = (
            "import sys, games.arms;"
            f"print(sorted(name for name in {TRAINING_STACK!r} if name in sys.modules))"
        )
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        assert finished.stdout.strip() == "[]", finished.stdout

    def test_the_same_probe_would_notice_the_stack_if_it_were_there(self) -> None:
        """The negative control: the probe has to be able to fail, or it proves nothing."""
        probe = (
            "import sys, torch, games.arms;"
            f"print(sorted(name for name in {TRAINING_STACK!r} if name in sys.modules))"
        )
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        )
        assert "torch" in finished.stdout


class TestArmsRegistry:
    def test_every_arm_names_a_servable_game_and_a_known_grading(self):
        # Servable two ways: the roster renders it, or a corpus builder constructs it (the
        # track-record game's rows carry a per-row stated percentage the roster's grid cannot
        # express, so generate_prompt_rows refuses it and the corpus is built by
        # games.track_record_corpus instead).
        # `is_grading` rather than `in GRADINGS`, because the care family is a pattern over a
        # weight rather than a member of the finite set: `care-alpha-1` is scored and is not in it.
        for name, arm in ARMS.items():
            assert arm.game_id in GAME_IDS or arm.game_id in CORPUS_BUILT_GAME_IDS, name
            assert is_grading(arm.grading), name
            assert arm.notes, name

    def test_the_planned_arms_are_all_registered(self):
        # The base plan's eight, plus the positive-sum wave-1 controls: hi-lo (positive control
        # on the reward path), harmony (cooperation-is-simply-best placebo), chicken (interior
        # fixed-point numeric control, promoted from eval-only).
        assert set(ARMS) == {
            "twin-pd-group",
            "twin-pd-self",
            "stag-hunt-group",
            "dictator",
            "fixed-pie-pd-group",
            "iterated-pd-tft",
            "stag-hunt-vs-frozen",
            "pd-vs-frozen",
            "hi-lo-group",
            "harmony-group",
            "chicken-group",
            # The wave-2 format-only placebo: the twin-pd prompts graded on answer shape alone, so
            # "RL on game-shaped prompts moved this instrument" can be told apart from "the graded
            # dimension moved it". Its corpus is the twin-pd group-mix corpus re-graded on CPU, so
            # it inherits that arm's selection attrition exactly rather than sweeping again.
            "twin-pd-format-only",
            # Two rungs of the stag ladder as their own arms, added 2026-08-19 after the baseline
            # sweep measured all four: the rungs' within-group spreads differ up to 10.2x, so a
            # mixed batch under scale_rewards="batch" trains the compressed ones weaker and their
            # flat curves would be artifacts. Only safe and risky are registered -- favoured-hunt
            # kept 6 of 32 prompts and even-hunt 10, under the mixedness floor of 12.
            "stag-hunt-safe-rung",
            "stag-hunt-risky-rung",
            # Wave 2's first pair, added 2026-08-21: the simultaneous-claim division under the two
            # gradings, on byte-identical prompts. Both predict the equal division rather than a
            # direction, so the pair is a second numeric control on the reward path.
            "nash-demand-group",
            "nash-demand-self",
            # The transfer-of-learning pair, added 2026-08-26 when public-goods was promoted out
            # of the eval-only set: continue-RL onto public-goods from the twin-pd checkpoints vs
            # from base. Two arms, not four -- which checkpoint seeds the LoRA is games.train's
            # --init-adapter knob, recorded per run, not an arm property.
            "public-goods-group",
            "public-goods-self",
            # Wave 2's trust arms, added 2026-08-21. The first two are the same prompts with one
            # number changed -- an announced return rate either side of the break-even 1/3 -- so the
            # payoff-maximising send flips from nothing to the whole stock across the pair. The
            # third is the strategy method, where the completion states its own return rule and its
            # twin applies it. All three answer with a number rather than a label, which is what
            # keeps them out of the mixed-behaviour band that deleted wave 1's positive-sum arms.
            "trust-return-fifth",
            "trust-return-half",
            "trust-strategy-method",
            # Wave 2's threshold public good, added 2026-08-21, and the only game on the slate whose
            # optimum is INTERIOR: putting in more than the equal share buys no more of the shared thing
            # and costs a unit each time, so this is the one arm that can separate a cooperative policy
            # from a maximising or compliant one. Two of the three pin a prize variant either side of
            # what funding the undertaking single-handed costs, because against a low-contributing group
            # their best responses oppose; the self-graded third runs both variants, since the optimum is
            # the equal share under each.
            "threshold-goods-prize-below-solo-cost",
            "threshold-goods-prize-above-solo-cost",
            "threshold-goods-self",
            # Wave 2's minimum-effort pair, added 2026-08-21, and the only 2x2 dose-response on the
            # slate: the same frames with two numbers moved (the cost per level and the number of
            # counterparts), which puts the analytic pressure at the top of the grid in one and at
            # level 2 in the other. Registered as the two widest cells of the four the corpus carries
            # (spreads 0.400 and 0.303 against the reference distribution); the other two are
            # gradient-compressed at 0.132 and 0.100 and are left unregistered, as favoured-hunt and
            # even-hunt were.
            "min-effort-cheap-pair",
            "min-effort-costly-crew",
            # The repeated companion plus its two comparison points, all added 2026-08-21. Against a
            # level-matcher, dropping the final level only ever loses money, so end_game_drop_rate
            # there is a reading on a transferred habit rather than a description of play -- and the
            # comparison needs the two arms where the same move is optimal (the copying PD) and where
            # it is not (the repeated stag hunt, whose all-cooperate optimum is unique on every rung).
            # iterated-pd-grim is the grim-trigger swap the reward-surface sweep recommended, added as
            # its own arm because iterated-pd-tft has trained artifacts keyed to its current rule.
            "iterated-min-effort-matcher",
            "iterated-stag-tft",
            "iterated-pd-grim",
            # The learning-side framing pair, added 2026-08-25: twin-pd's two gradings on prompts
            # with the counterpart paragraph deleted, completing the 2x2 {twin framing, unstated}
            # x {self, group-mix} whose twin column is the trained twin-pd pair.
            "pd-unstated-group",
            "pd-unstated-self",
            # The wave-3 grading ladder, added 2026-08-29: the same story-free prompts with the
            # reward's RECIPIENT moved -- the pie's mean, or the counterpart's own payoff -- so the
            # ladder {self, joint-welfare, other-payoff} separates cooperation trained because it
            # pays me, pays us, or pays them.
            "pd-unstated-joint-welfare",
            "pd-unstated-other-payoff",
            # The wave-3 "many stories" arm, added 2026-08-29: twin-pd's cells with no counterpart
            # paragraph over a wide register-diverse skin bank, self-graded -- does surface
            # diversity change what story-free cooperation training generalises to?
            "pd-reskin-self",
            # The track-record arm, added 2026-08-30: the pd-unstated stems plus a numeric
            # decision-matching clause, graded as the exact expected payoff under the stated
            # probability, with the corpus mixing p across each variant's EV crossover so the
            # counterpart clause stays reward-relevant (the answer to the self grading's measured
            # counterpart-blindness).
            "pd-track-record",
            # Track-record v2, added 2026-09-01: the same construction with the reward computed
            # from the DISPLAYED quantized table, every cell rescaled to one shared EV margin
            # (v1's cooperation fell under a ~9x defect-side margin asymmetry), and a chained
            # (table, rate) grid on which neither the stated rate alone nor the table alone
            # predicts the optimal action.
            "pd-track-record-v2",
            # Track-record v2, penalty-scaled, added 2026-09-02: v2's corpus, game and grading,
            # differing only in the parse-penalty pin (a malformed answer earns one EV margin below
            # the row's worse action instead of the constant -1.0 that v2's readout found carrying
            # its coop-ward drift).
            "pd-track-record-v2-softpen",
            # Wave 4b's prosocial-breadth pair, added 2026-09-04, and the first arms whose corpus
            # carries several games at once: one care-weighted reward over five matrix games plus the
            # trust sender, under seven counterpart framings, so what the training generalises to is
            # read on two distance axes rather than one. The alpha-0 control is registered beside the
            # treatment because its corpus is the same file regraded, but it runs only if the owner
            # asks for it.
            "prosocial-breadth-care1",
            "prosocial-breadth-self",
            "cooperation-generalization-care-alpha-1",
        }

    def test_no_two_arms_are_the_same_experiment(self):
        """Generalised twice: to the payoff pin (2026-08-19), then to the corpus partition.

        The original form asserted that `(game_id, grading)` appeared once, which encoded the
        assumption that an arm is fully identified by its game and its grading. Running individual
        stag rungs deliberately breaks that assumption: three arms now share
        `("stag-hunt", "group-mix")` and differ only in which rung they pin. A mix-split pair breaks
        it again -- two such arms share the game, the grading AND the payoff variant, and differ only
        in which side of the group-mix boundary their corpus sits on. The invariant the test was
        really protecting -- that no two registered arms are the same experiment under two names -- is
        preserved by widening the key rather than weakening the check, and it keeps its teeth: two
        arms with identical keys would be exactly that duplicate.

        The game set joined the key on 2026-09-04, when an arm's corpus stopped being one game: two
        arms over the same lead game and grading that differ in the other games their corpora carry
        are two experiments, and a key without the set would report them as one duplicate.
        """
        keys = [
            (
                arm.game_id,
                arm.grading,
                arm.game_ids,
                arm.payoff_variants,
                arm.corpus_partition,
                arm.parse_penalty_mode,
            )
            for arm in ARMS.values()
        ]
        assert len(keys) == len(set(keys))

    def test_the_identity_key_separates_the_two_track_record_v2_arms(self):
        """Widened a third time (2026-09-02), to the parse-penalty pin.

        `pd-track-record-v2` and `pd-track-record-v2-softpen` share the game, the grading, the
        corpus and every payoff cell; the softpen arm prices an unparseable completion one EV
        margin below the row's worse action instead of at the constant. That is a different
        reward, hence a different experiment, and the key has to say so or the two read as one
        duplicate.
        """
        v2, softpen = ARMS["pd-track-record-v2"], ARMS["pd-track-record-v2-softpen"]
        assert (v2.game_id, v2.grading) == (softpen.game_id, softpen.grading)
        assert v2.parse_penalty_mode == PARSE_PENALTY_CONSTANT
        assert softpen.parse_penalty_mode == PARSE_PENALTY_MARGIN_BELOW_WORSE
        assert (v2.game_id, v2.grading, v2.payoff_variants, v2.corpus_partition) == (
            softpen.game_id,
            softpen.grading,
            softpen.payoff_variants,
            softpen.corpus_partition,
        )

    def test_row_relative_arms_price_a_parse_failure_against_the_row(self):
        """Every other arm keeps its reward byte for byte, and these four say why they do not.

        The softpen arm is where the row-relative price was introduced (2026-09-02) and the two
        prosocial-breadth arms are where the 2026-09-03 audit's generalisation of it is used: the
        care-alpha-1 spread on a PD-shaped row passes through zero at the attractor the treatment arm
        is designed to settle at, and no single constant holds the failure-to-task ratio fixed across
        the pair's two weights at any mix.
        """
        row_relative = {
            name for name, arm in ARMS.items() if arm.parse_penalty_mode != PARSE_PENALTY_CONSTANT
        }
        assert row_relative == {
            "pd-track-record-v2-softpen",
            "prosocial-breadth-care1",
            "prosocial-breadth-self",
            "cooperation-generalization-care-alpha-1",
        }
        assert {ARMS[name].grading for name in row_relative} == {
            GRADING_VS_STATED_MATCH,
            care_grading(0),
            care_grading(1),
        }

    def test_validate_arms_rejects_the_row_relative_price_under_a_grading_that_cannot_price_it(
        self,
    ):
        """The registry's half of the one refusal, whose reason table lives in `games.rewards`.

        `format-only` grades answer shape with nothing about the game in it, so a failure priced one
        rubric spread below the worst rubric score would scale the format channel against itself.
        """
        arms = {
            "bad": GameArm(
                game_id="twin-pd",
                grading=GRADING_FORMAT_ONLY,
                notes="n/a",
                parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
            )
        }
        with pytest.raises(ValueError, match="cannot price a failure against the row"):
            validate_arms(arms)

    def test_validate_arms_accepts_the_row_relative_price_under_every_other_grading(self):
        # The generalisation, stated as the registry sees it: the pin used to be legal for one grading
        # and is now legal for every grading whose row bounds its own rewards.
        for grading in sorted(GRADINGS):
            if grading == GRADING_FORMAT_ONLY:
                continue
            arm = ARMS[next(name for name, one in ARMS.items() if one.grading == grading)]
            validate_arms(
                {
                    "candidate": GameArm(
                        game_id=arm.game_id,
                        game_ids=arm.game_ids,
                        grading=grading,
                        notes="row-relative price under every grading",
                        payoff_variants=arm.payoff_variants,
                        parse_penalty_mode=PARSE_PENALTY_MARGIN_BELOW_WORSE,
                    )
                }
            )

    def test_validate_arms_rejects_an_unknown_parse_penalty_mode(self):
        arms = {
            "bad": GameArm(
                game_id="twin-pd", grading="self", notes="n/a", parse_penalty_mode="gentle"
            )
        }
        with pytest.raises(ValueError, match="known modes"):
            validate_arms(arms)

    def test_the_identity_key_separates_two_arms_that_differ_only_by_partition(self):
        # The reason the key was widened, asserted on constructed arms because no mix-split arm is
        # registered yet: without the partition in the key these two would read as one duplicate.
        sides = [
            GameArm(
                game_id="stag-hunt",
                grading="group-mix",
                notes=f"the {side} side of the safe rung's boundary",
                payoff_variants=("safe-hunt",),
                corpus_partition=side,
            )
            for side in sorted(CORPUS_PARTITIONS)
        ]
        validate_arms({f"stag-hunt-safe-rung-{arm.corpus_partition}": arm for arm in sides})
        keys = {
            (
                arm.game_id,
                arm.grading,
                arm.payoff_variants,
                arm.corpus_partition,
                arm.parse_penalty_mode,
            )
            for arm in sides
        }
        assert len(keys) == len(sides)

    def test_the_stag_rung_arms_overlap_the_pooled_arm_on_purpose(self):
        """The overlap is deliberate, so nobody "fixes" it later.

        `stag-hunt-group` pins nothing and therefore spans all four rungs, which means its corpus is
        a strict superset of each rung arm's. That is not a duplicate registration: the pooled
        arm is the record of what the four-rung ladder was, and the rung arms exist precisely
        because
        training them in one batch is what the reward-spread rule forbids. Running the pooled
        arm and a rung arm at the same time would be the mistake, and that is an operator
        decision rather
        than something the registry can prevent.
        """
        pooled = ARMS["stag-hunt-group"]
        assert pooled.payoff_variants == ()
        for name in ("stag-hunt-safe-rung", "stag-hunt-risky-rung"):
            rung = ARMS[name]
            assert (rung.game_id, rung.grading) == (pooled.game_id, pooled.grading)
            assert len(rung.payoff_variants) == 1
            assert set(rung.payoff_variants) <= arm_payoff_variants(pooled)

    def test_the_frozen_opponent_arms_use_their_own_frozen_framed_prompts(self):
        # Their prompts describe an external counterpart rather than another instance of this
        # model, so they are separate game ids in games.prompts -- not the base game re-graded.
        assert ARMS["pd-vs-frozen"].game_id == "pd-vs-frozen"
        assert ARMS["stag-hunt-vs-frozen"].game_id == "stag-hunt-vs-frozen"
        assert ARMS["pd-vs-frozen"].game_id != ARMS["twin-pd-group"].game_id

    def test_the_contrast_arms_share_a_game_and_differ_only_in_grading(self):
        group = ARMS["twin-pd-group"]
        own = ARMS["twin-pd-self"]
        assert group.game_id == own.game_id
        assert group.grading != own.grading

    def test_the_iterated_arm_is_pinned_to_the_temptation_two_payoffs(self):
        # At temptation-10 the classic PD loses 2*CC > DC + CD, so alternating exploitation beats
        # sustained cooperation against a copying opponent and the arm measures the opposite of
        # what it exists for. The pin is the whole reason this arm is safe to run.
        arm = ARMS["iterated-pd-tft"]
        assert arm.payoff_variants == ("temptation-2",)
        assert "temptation-10" not in arm.payoff_variants

    def test_exactly_the_intended_arms_pin_a_payoff_variant(self):
        """Kept as an exact set so a new pin has to be deliberate.

        Was `test_only_the_iterated_arm_pins_a_payoff_variant` until 2026-08-19, when two stag rungs
        were registered as their own arms. The assertion's value is unchanged and lies in being an
        equality rather than a subset check: a pin silently narrows the corpus an arm trains on, so
        adding one should require editing this line and saying why.
        """
        pinned = {name for name, arm in ARMS.items() if arm.payoff_variants}
        assert pinned == {
            "iterated-pd-tft",
            "stag-hunt-safe-rung",
            "stag-hunt-risky-rung",
            # The two announced-return-rate arms, added 2026-08-21. Each pins one published return
            # share, which is the whole experiment: the pair's prose is identical and the pinned
            # number is what flips the payoff-maximising send between nothing and the whole stock.
            # An unpinned arm over both would train the two directions in one batch and average the
            # contrast away.
            "trust-return-fifth",
            "trust-return-half",
            # The two prize variants of the threshold public good, added 2026-08-21. Each pins one
            # prize, which is the whole experiment: the pair's prose is identical and the pinned number
            # decides whether funding the undertaking single-handed is worth doing. Computed, and the
            # reason a pooled arm would say nothing: against a low-contributing group the best responses
            # oppose -- put in nothing under the low prize, fund the whole thing under the high one.
            "threshold-goods-prize-below-solo-cost",
            "threshold-goods-prize-above-solo-cost",
            # The minimum-effort cells, added 2026-08-21. Each pins one of the four knob combinations,
            # which is the whole experiment: an unpinned arm over all four would train the upward and
            # downward pressures in one batch and average the contrast away, and the two unregistered
            # cells are four times narrower in reward spread than the widest.
            "min-effort-cheap-pair",
            "min-effort-costly-crew",
            # The repeated arms. The level-matcher arm pins its cost ratio deliberately rather than
            # from necessity -- its corpus carries only that variant, and the pin is what records that
            # the author knew which one. The repeated stag hunt pins the safe rung because the four
            # rungs' reward spreads differ and a mixed batch trains the compressed ones weaker, and the
            # grim-trigger PD pins temptation-2 for iterated-pd-tft's structural reason.
            "iterated-min-effort-matcher",
            "iterated-stag-tft",
            "iterated-pd-grim",
        }

    def test_validate_arms_rejects_a_game_prompts_cannot_render(self):
        # defective-coordination is an eval-only transfer game: it has a payoff spec but no
        # training split, and an arm trained on it would destroy its transfer value anyway.
        arms = {
            "bad": GameArm(game_id="defective-coordination", grading="group-mix", notes="eval only")
        }
        with pytest.raises(ValueError, match="cannot render"):
            validate_arms(arms)

    def test_validate_arms_rejects_an_unknown_grading(self):
        arms = {"bad": GameArm(game_id="twin-pd", grading="vibes", notes="n/a")}
        with pytest.raises(ValueError, match="names grading"):
            validate_arms(arms)

    def test_validate_arms_rejects_a_payoff_variant_no_corpus_carries(self):
        arms = {
            "bad": GameArm(
                game_id="twin-pd",
                grading="self",
                notes="n/a",
                payoff_variants=("temptation-7",),
            )
        }
        with pytest.raises(ValueError, match="its own corpus does not carry"):
            validate_arms(arms)

    def test_validate_arms_rejects_a_variant_that_belongs_to_a_different_game(self):
        # The check is per arm, not against a merged vocabulary, so it catches a variant that is
        # real somewhere and meaningless here. Both the variant and the game it is foreign to are
        # discovered rather than named, because the naming conventions upstream keep changing --
        # they changed twice while this was written.
        foreign = next(
            variant
            for variant in arm_payoff_variants(ARMS["twin-pd-group"])
            if variant not in arm_payoff_variants(ARMS["stag-hunt-group"])
        )
        arms = {
            "bad": GameArm(
                game_id="stag-hunt",
                grading="group-mix",
                notes="n/a",
                payoff_variants=(foreign,),
            )
        }
        with pytest.raises(ValueError, match="its own corpus does not carry"):
            validate_arms(arms)

    def test_no_single_vocabulary_could_validate_every_arm(self):
        # The justification for deriving per arm: two arms carry disjoint variant sets, so a
        # merged set would accept a pin that matches no row in the game it names. Asserted as a
        # property because the conventions themselves are upstream's to change -- matrix games
        # label by temptation size, the unilateral split by endowment, the stag hunt by which
        # equilibrium its payoffs favour, and that list has already grown twice.
        # Corpus-built games are skipped: their variants are a property of the built corpus,
        # and generate_prompt_rows (which arm_payoff_variants renders through) refuses them.
        by_arm = {
            name: arm_payoff_variants(arm)
            for name, arm in ARMS.items()
            if arm.game_id not in CORPUS_BUILT_GAME_IDS
        }
        for name, variants in by_arm.items():
            assert variants, name
            assert all(variant and isinstance(variant, str) for variant in variants), name
        assert any(
            left.isdisjoint(right)
            for left in by_arm.values()
            for right in by_arm.values()
            if left != right
        )

    def test_a_variant_the_corpus_carries_is_a_valid_pin_for_every_arm(self):
        # Whatever the conventions are called, pinning one of an arm's own variants must pass.
        for name, arm in ARMS.items():
            if arm.game_id in CORPUS_BUILT_GAME_IDS:
                continue
            available = sorted(arm_payoff_variants(arm))
            probe = GameArm(
                game_id=arm.game_id,
                grading=arm.grading,
                notes="pins one variant the corpus carries",
                payoff_variants=(available[0],),
            )
            validate_arms({f"probe-{name}": probe})

    def test_every_registered_pin_matches_rows_its_own_corpus_carries(self):
        # The invariant the import-time check exists to hold: a pin that matches nothing would
        # empty the corpus at startup instead of failing here.
        for name, arm in ARMS.items():
            if arm.payoff_variants:
                available = arm_payoff_variants(arm)
                assert set(arm.payoff_variants) <= available, name

    def test_validate_arms_rejects_an_undescribed_arm(self):
        arms = {"bad": GameArm(game_id="twin-pd", grading="self", notes="")}
        with pytest.raises(ValueError, match="no notes"):
            validate_arms(arms)


class TestTheCorpusMayCarrySeveralGames:
    """`GameArm.game_ids` names the other games one arm's corpus may carry, for a mixed-corpus arm.

    Wave 4b trains one care-weighted arm on a corpus of six games at once, so the registry has to
    say which games that is: the loader refuses a row whose game the arm does not name, and with a
    single `game_id` field that refusal is either the whole corpus or nothing. The field holds the
    extra games only, and `arm_game_ids` is the whole set every consumer asks for, so no caller has
    to remember to add `game_id` back.
    """

    @staticmethod
    def probe(**overrides: object) -> dict[str, GameArm]:
        """One arm carrying a game set, for the validator to accept or refuse."""
        base: dict[str, object] = {
            "game_id": "twin-pd",
            "grading": GRADING_SELF,
            "notes": "a probe arm over several games",
        }
        return {"probe": GameArm(**(base | overrides))}  # pyright: ignore[reportArgumentType]

    def test_the_default_is_one_game(self) -> None:
        assert ARMS["twin-pd-self"].game_ids == ()
        assert arm_game_ids(ARMS["twin-pd-self"]) == ("twin-pd",)

    def test_the_arms_own_game_comes_first_in_the_set(self) -> None:
        arm = ARMS["prosocial-breadth-care1"]
        assert arm_game_ids(arm)[0] == arm.game_id
        assert set(arm_game_ids(arm)) == {arm.game_id, *arm.game_ids}

    def test_the_breadth_arm_names_the_five_other_training_games(self) -> None:
        assert ARMS["prosocial-breadth-care1"].game_ids == (
            "pd-reskin",
            "stag-hunt",
            "chicken",
            "public-goods",
            "trust-vs-stated-return",
        )

    def test_the_queued_control_trains_the_same_games(self) -> None:
        care = ARMS["prosocial-breadth-care1"]
        control = ARMS["prosocial-breadth-self"]
        assert arm_game_ids(control) == arm_game_ids(care)
        assert control.grading == care_grading(0)
        assert care.grading == care_grading(1)

    def test_a_game_nothing_renders_or_builds_is_refused(self) -> None:
        with pytest.raises(ValueError, match="also carry"):
            validate_arms(self.probe(game_ids=("no-such-game",)))

    def test_repeating_the_arms_own_game_is_refused(self) -> None:
        # It would make the set say the corpus may ALSO carry the game the arm already trains, so a
        # reader could not tell an arm with one extra game from an arm with none.
        with pytest.raises(ValueError, match="already trains"):
            validate_arms(self.probe(game_ids=("stag-hunt", "twin-pd")))

    def test_a_repeated_extra_game_is_refused(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            validate_arms(self.probe(game_ids=("stag-hunt", "stag-hunt")))

    def test_a_servable_set_is_accepted(self) -> None:
        validate_arms(self.probe(game_ids=("stag-hunt", "chicken")))

    def test_a_pin_only_the_lead_game_carries_is_refused_on_a_mixed_set(self) -> None:
        # The pin narrows the whole corpus, not the lead game's share of it: games.train's
        # filter_payoff_variants keeps only pinned rows and raises solely when that empties the
        # corpus, so this arm would train the twin-PD rows alone while run_config recorded three
        # games and the battery marked all three trained.
        with pytest.raises(ValueError, match=r"'stag-hunt' rows carry"):
            validate_arms(
                self.probe(game_ids=("stag-hunt", "chicken"), payoff_variants=("temptation-2",))
            )

    def test_a_pin_every_game_in_the_set_carries_is_accepted(self) -> None:
        # The positive control, so the refusal above is not just "a mixed set may never pin": the PD
        # reskins share the base game's temptation vocabulary, so this pin narrows both games alike.
        validate_arms(self.probe(game_ids=("pd-reskin",), payoff_variants=("temptation-2",)))

    def test_the_arm_level_variant_set_reports_the_lead_game_only(self) -> None:
        # Pinned so nobody pools it across the set later: a union would call a variant available when
        # only one of the games names it, which is the pin the refusal above exists to catch.
        care = ARMS["prosocial-breadth-care1"]
        assert arm_payoff_variants(care) == game_payoff_variants(care.game_id, care.grading)
        assert "favoured-hunt" in game_payoff_variants("stag-hunt", care.grading)
        assert "favoured-hunt" not in arm_payoff_variants(care)

    def test_every_registered_arms_game_set_is_servable(self) -> None:
        for name, arm in ARMS.items():
            for game_id in arm.game_ids:
                assert game_id in GAME_IDS or game_id in CORPUS_BUILT_GAME_IDS, name


class TestCorpusPartitionPin:
    """A mix-split arm trains one side of its game's group-mix boundary. What may name one.

    No arm registered today pins a partition, and the first exact-set assertion below is what makes
    adding one deliberate. The reason it is still empty is arithmetic rather than design. Partitioning
    the 2026-08-19 safe-hunt corpus (the only rung whose boundary sits inside the measured spread of
    baseline cooperation) puts 1 pair above the boundary, 2 below and 7 straddling: the two label
    orientations of a scenario disagree by 0.213 on average, which is larger than most scenarios'
    distance from the boundary. `games.corpus_partition` reports that as a number rather than hiding
    it, and registering an arm whose corpus is two prompts would reproduce hi-lo's stopped run.
    """

    @staticmethod
    def arm(**overrides: object) -> GameArm:
        fields: dict[str, object] = {
            "game_id": "stag-hunt",
            "grading": "group-mix",
            "notes": "one side of the safe rung's group-mix boundary",
            "payoff_variants": ("safe-hunt",),
            "corpus_partition": PARTITION_ABOVE_THRESHOLD,
        }
        return GameArm(**{**fields, **overrides})  # pyright: ignore[reportArgumentType]

    def test_no_registered_arm_pins_a_partition_yet(self):
        # Kept as an equality for the same reason the payoff-pin test is: a partition silently
        # narrows the corpus an arm trains on, so adding one should require editing this line.
        assert {name for name, arm in ARMS.items() if arm.corpus_partition} == set()

    def test_an_unpinned_arm_is_the_default(self):
        assert all(arm.corpus_partition == "" for arm in ARMS.values())

    def test_a_well_formed_mix_split_arm_validates(self):
        validate_arms({"stag-hunt-safe-rung-above-threshold": self.arm()})

    def test_a_partition_no_partitioner_writes_is_refused(self):
        with pytest.raises(ValueError, match="no partitioner writes"):
            validate_arms({"bad": self.arm(corpus_partition="mostly-cooperative")})

    def test_the_two_sides_the_partitioner_writes_are_both_accepted(self):
        for partition in sorted(CORPUS_PARTITIONS):
            validate_arms({f"probe-{partition}": self.arm(corpus_partition=partition)})

    def test_a_partition_under_self_grading_is_refused(self):
        # The boundary IS the group-mix reward gap's crossing; under `self` there is no group to be
        # mixed against, so the two sides would be labels rather than directions.
        with pytest.raises(ValueError, match="GROUP-MIX reward gap"):
            validate_arms(
                {
                    "bad": self.arm(
                        game_id="twin-pd", grading="self", payoff_variants=("temptation-2",)
                    )
                }
            )

    def test_a_partition_under_vs_fixed_mix_grading_is_refused(self):
        with pytest.raises(ValueError, match="GROUP-MIX reward gap"):
            validate_arms({"bad": self.arm(game_id="stag-hunt-vs-frozen", grading="vs-fixed-mix")})

    def test_a_partition_spanning_payoff_variants_is_refused(self):
        # The boundary is computed per row from that row's own cells, so an unpinned partition would
        # differ from its twin in payoff mix as well as in baseline behaviour.
        with pytest.raises(ValueError, match="must pin exactly one"):
            validate_arms({"bad": self.arm(payoff_variants=())})
        with pytest.raises(ValueError, match="must pin exactly one"):
            validate_arms({"bad": self.arm(payoff_variants=("safe-hunt", "risky-hunt"))})


class TestSelfGradingOnAConstantSumGameIsRejected:
    """A `self`-graded constant-sum arm is dead: nothing in the registry caught it before.

    Under `self` grading a completion is scored at payoff(action, action), so the only two rewards
    reachable are the CC and DD cells. A constant-sum symmetric game forces CC == DD, so every
    parsed completion in every group scores identically, every GRPO advantage is zero, and the arm
    trains nothing while producing a full set of plausible artifacts. It would cost a whole paid run
    to discover from a flat curve, and the flat curve is indistinguishable from a real null.
    """

    def test_a_self_graded_constant_sum_arm_is_refused(self):
        arms = {
            "fixed-pie-pd-self": GameArm(
                game_id="fixed-pie-pd", grading="self", notes="the dead arm this guard exists for"
            )
        }
        with pytest.raises(ValueError, match="zero"):
            validate_arms(arms)

    def test_the_refusal_explains_the_mechanism_not_just_the_verdict(self):
        arms = {
            "fixed-pie-pd-self": GameArm(
                game_id="fixed-pie-pd", grading="self", notes="the dead arm this guard exists for"
            )
        }
        with pytest.raises(ValueError, match="advantage is zero") as raised:
            validate_arms(arms)
        message = str(raised.value)
        assert "constant-sum" in message
        assert "group-mix" in message

    def test_the_same_game_under_group_mix_grading_is_left_alone(self):
        # fixed-pie-pd-group is registered and healthy: group-mix scores against the opponent mix,
        # so defection being dominant still produces a within-group reward gap.
        validate_arms({"fixed-pie-pd-group": ARMS["fixed-pie-pd-group"]})

    def test_self_grading_on_a_positive_sum_game_still_passes(self):
        # twin-pd-self is the flagship contrast arm: CC and DD differ there, so the guard must not
        # reach it. A guard that rejected every self-graded arm would pass its own first test.
        validate_arms({"twin-pd-self": ARMS["twin-pd-self"]})

    def test_the_whole_registry_is_free_of_dead_arms(self):
        validate_arms(ARMS)


class TestThePdUnstatedPair:
    """The learning-side framing pair: twin-pd's two gradings on prompts with no counterpart paragraph.

    Completes the 2x2 {twin framing, unstated} x {self, group-mix} whose twin column is already
    trained. Its own game id rather than a re-grading of twin-pd, for the vs-frozen arms' reason:
    the prompts differ (one deleted paragraph), so sharing twin-pd's id would pool two prompt
    corpora under one key.
    """

    def test_both_arms_are_registered_on_the_unstated_game(self):
        group = ARMS["pd-unstated-group"]
        own = ARMS["pd-unstated-self"]
        assert group.game_id == "pd-unstated"
        assert own.game_id == group.game_id
        assert group.grading != own.grading

    def test_the_pair_mirrors_the_twin_pd_pairs_gradings(self):
        # One deleted paragraph is the only designed difference from the trained pair, so each
        # leg must grade exactly as its twin-framed counterpart does.
        assert ARMS["pd-unstated-group"].grading == ARMS["twin-pd-group"].grading
        assert ARMS["pd-unstated-self"].grading == ARMS["twin-pd-self"].grading

    def test_neither_arm_pins_a_payoff_variant(self):
        # The twin-pd pair trains both temptations unpinned; a pin here would change the corpus
        # along with the framing and break the contrast.
        assert ARMS["pd-unstated-group"].payoff_variants == ()
        assert ARMS["pd-unstated-self"].payoff_variants == ()

    def test_the_self_arm_passes_the_dead_arm_guard(self):
        # Same positive-sum cells as twin-pd (CC=0.6 vs DD=0.2), so the span check must accept it.
        validate_arms({"pd-unstated-self": ARMS["pd-unstated-self"]})


class TestThePublicGoodsTransferPair:
    """The transfer-of-learning pair: twin-pd's two gradings on the promoted public-goods game.

    The arms carry game and grading only; which checkpoint seeds the LoRA is `games.train`'s
    --init-adapter knob, recorded per run, so each arm serves both its from-checkpoint cell and
    its from-base control. Registering init-specific arm ids instead would put two entries on one
    (game, grading) pair and the registry's meaning -- one arm, one corpus contract -- would blur.
    """

    def test_both_arms_are_registered_on_the_promoted_game(self):
        group = ARMS["public-goods-group"]
        own = ARMS["public-goods-self"]
        assert group.game_id == "public-goods"
        assert own.game_id == group.game_id
        assert group.grading != own.grading

    def test_the_pair_mirrors_the_twin_pd_pairs_gradings(self):
        # The transfer question is whether twin-pd's trained dispositions change LEARNING under
        # the same grading rules, so each leg must grade exactly as its twin-pd counterpart does.
        assert ARMS["public-goods-group"].grading == ARMS["twin-pd-group"].grading
        assert ARMS["public-goods-self"].grading == ARMS["twin-pd-self"].grading

    def test_neither_arm_pins_a_payoff_variant(self):
        # public-goods is single-variant; an empty pin trains the whole corpus, as twin-pd's does.
        assert ARMS["public-goods-group"].payoff_variants == ()
        assert ARMS["public-goods-self"].payoff_variants == ()

    def test_the_self_arm_passes_the_dead_arm_guard(self):
        # Positive-sum diagonal (CC=0.889 vs DD=0.556 after normalisation), so the span check
        # must accept it -- the same property that makes cooperation the self-graded optimum.
        validate_arms({"public-goods-self": ARMS["public-goods-self"]})


class TestThePdReskinArm:
    """The wave-3 "many stories" arm: self grading over the register-diverse reskin roster."""

    def test_registered_on_the_reskin_game_under_self_grading(self):
        arm = ARMS["pd-reskin-self"]
        assert arm.game_id == "pd-reskin"
        assert arm.grading == ARMS["pd-unstated-self"].grading

    def test_no_payoff_pin_and_the_dead_arm_guard_accepts_it(self):
        # Both temptations train, exactly as in the pd-unstated pair; the positive-sum cells
        # (CC=0.6 vs DD=0.2) must clear the self-grading span check on the new roster too.
        assert ARMS["pd-reskin-self"].payoff_variants == ()
        validate_arms({"pd-reskin-self": ARMS["pd-reskin-self"]})

    def test_the_group_leg_is_deliberately_not_registered(self):
        # Defection-by-dominance training needs no story (twin-pd-group and pd-unstated-group
        # both fell); the wave's question is what makes COOPERATION training general, so the
        # group leg stays unregistered until a question needs it.
        assert "pd-reskin-group" not in ARMS


class TestTheGradingLadderArms:
    """The wave-3 ladder: pd-unstated's prompts under two recipient-moved gradings.

    Both arms share the pd-unstated game (the prompts ARE the held-fixed side of the contrast; a
    new game id would fork the corpus and break byte-identity with the trained pair) and differ
    from the trained pair and each other only in whose payoff the reward pays.
    """

    def test_both_arms_are_registered_on_the_unstated_game(self):
        joint = ARMS["pd-unstated-joint-welfare"]
        other = ARMS["pd-unstated-other-payoff"]
        assert joint.game_id == "pd-unstated"
        assert other.game_id == joint.game_id
        assert joint.grading == "joint-welfare-group-mix"
        assert other.grading == "other-payoff-group-mix"

    def test_neither_arm_pins_a_payoff_variant(self):
        # The pd-unstated pair trains both temptations unpinned; a pin here would change the
        # corpus along with the grading and break the ladder's held-fixed side. The temptation-10
        # interior-attractor prediction is read by SPLITTING on payoff_variant, never by pinning.
        assert ARMS["pd-unstated-joint-welfare"].payoff_variants == ()
        assert ARMS["pd-unstated-other-payoff"].payoff_variants == ()

    def test_both_gradings_are_exempt_from_the_span_check_as_group_mix_family(self):
        # The within-group span is a batch property (the group's realised mix), exactly as for
        # own-payoff group-mix -- so they must be classified exempt, and classified at all is
        # enforced by assert_every_grading_is_classified at import.
        assert "joint-welfare-group-mix" in SPAN_CHECK_EXEMPTIONS
        assert "other-payoff-group-mix" in SPAN_CHECK_EXEMPTIONS
        validate_arms(
            {
                "pd-unstated-joint-welfare": ARMS["pd-unstated-joint-welfare"],
                "pd-unstated-other-payoff": ARMS["pd-unstated-other-payoff"],
            }
        )


class TestEveryGradingIsClassifiedForTheDeadArmCheck:
    """The dead-arm check must fail CLOSED: an unclassified grading is an import error, not a pass.

    `SELF_CONSISTENT_GRADINGS` began as a bare allowlist, and integration sabotage (2026-08-21) got a
    constant-reward arm accepted silently by registering a grading the table did not name -- the guard
    approving exactly the arm it exists to refuse. So the two tables must partition
    `games.rewards.GRADINGS`, and every test here is one way that partition can rot.
    """

    def test_the_live_tables_cover_every_grading_exactly_once(self):
        assert_every_grading_is_classified()
        assert set(SELF_CONSISTENT_GRADINGS) | set(SPAN_CHECK_EXEMPTIONS) == set(GRADINGS)

    def test_a_grading_classified_by_neither_table_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The sabotage that came back green before this check existed."""
        monkeypatch.setattr(games_arms, "GRADINGS", frozenset({*GRADINGS, "unclassified-grading"}))
        with pytest.raises(ValueError, match="unclassified-grading"):
            assert_every_grading_is_classified()

    def test_an_arm_using_an_unclassified_grading_is_refused_at_validate_time(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The failure the author would actually hit: an arm, not the table, is what gets written."""
        monkeypatch.setattr(games_arms, "GRADINGS", frozenset({*GRADINGS, "unclassified-grading"}))
        arm = GameArm(
            game_id="twin-pd", grading="unclassified-grading", notes="graded by nothing classified"
        )
        with pytest.raises(ValueError, match="no dead-arm check"):
            validate_arms({"some-new-arm": arm})

    def test_a_grading_in_both_tables_is_refused(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(
            SPAN_CHECK_EXEMPTIONS, GRADING_SELF, "claimed exempt as well as checked"
        )
        with pytest.raises(ValueError, match="both span-checked and exempt"):
            assert_every_grading_is_classified()

    def test_a_classification_for_a_grading_rewards_dropped_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A rename upstream must not leave a stale entry behind looking like coverage."""
        monkeypatch.setattr(games_arms, "GRADINGS", GRADINGS - {GRADING_SELF})
        with pytest.raises(ValueError, match=r"classified in games\.arms"):
            assert_every_grading_is_classified()

    def test_every_exemption_says_what_supplies_the_spread_instead(self):
        # An exemption whose reason is blank is an allowlist entry wearing a docstring: the whole
        # point of the table is that skipping the check is an argument someone had to write down.
        for grading, reason in SPAN_CHECK_EXEMPTIONS.items():
            assert len(reason.split()) >= 10, f"{grading} is exempt without a stated reason"


class TestFormatOnlyGradingNeedsAOneShotActionPrompt:
    """A format-only arm on the wrong answer shape is either dead or trains the wrong thing.

    `games.format_rubric` grades one `<action>` tag. The dictator asks for `<keep>N</keep>`, so no
    obedient completion carries an action tag and every reward is the parse penalty -- caught today
    only by the reward function's whole-batch raise, which happens after a card is reserved and a
    model is loaded. The iterated arm is worse: it asks for one tag per round, so an obedient
    five-round answer scores zero on `single_action_tag` and the rubric would pay for disobedience.
    A flat curve announces itself; a wrong gradient does not.
    """

    def test_the_registered_placebo_shares_its_game_with_the_arms_it_controls_for(self):
        placebo = ARMS["twin-pd-format-only"]
        for name in ("twin-pd-group", "twin-pd-self"):
            assert placebo.game_id == ARMS[name].game_id
            assert placebo.grading != ARMS[name].grading

    def test_a_format_only_arm_on_the_dictator_is_refused(self):
        arms = {
            "dictator-format-only": GameArm(
                game_id="dictator", grading="format-only", notes="the keep-tag answer shape"
            )
        }
        with pytest.raises(ValueError, match="do not ask for exactly one <action> tag"):
            validate_arms(arms)

    def test_a_format_only_arm_on_the_iterated_game_is_refused(self):
        arms = {
            "iterated-format-only": GameArm(
                game_id="iterated-pd-tft",
                grading="format-only",
                notes="five tags, so one-tag grading pays for disobedience",
                payoff_variants=("temptation-2",),
            )
        }
        with pytest.raises(ValueError, match="do not ask for exactly one <action> tag"):
            validate_arms(arms)

    def test_the_refusal_names_the_wrong_gradient_rather_than_only_the_verdict(self):
        arms = {
            "iterated-format-only": GameArm(
                game_id="iterated-pd-tft",
                grading="format-only",
                notes="n/a",
                payoff_variants=("temptation-2",),
            )
        }
        with pytest.raises(ValueError, match="pays for disobedience") as raised:
            validate_arms(arms)
        message = str(raised.value)
        assert "single_action_tag" in message
        assert "wrong gradient" in message

    def test_every_one_shot_action_game_is_an_acceptable_host_for_the_placebo(self):
        # The other direction, as a property over the registry: a guard that rejected every game
        # would pass all three tests above while making the arm unregisterable. Derived from the
        # arms themselves rather than a list of game ids, so a game added upstream is covered.
        hosts = {
            arm.game_id
            for arm in ARMS.values()
            if arm.game_id not in CORPUS_BUILT_GAME_IDS
            and not rows_asking_for_other_than_one_action_tag(
                GameArm(game_id=arm.game_id, grading="format-only", notes="probe")
            )
        }
        assert "twin-pd" in hosts
        assert "dictator" not in hosts
        for game_id in hosts:
            validate_arms(
                {
                    f"{game_id}-format-only": GameArm(
                        game_id=game_id, grading="format-only", notes="probe"
                    )
                }
            )
