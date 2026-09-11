"""Tests for the prompt corpus: the vocabulary guard, rendering, counterbalancing, and schema.

The guard tests are the ones that matter most and they are written as sabotage: a banned word is
planted in a real frame and the render must raise. A guard nobody has watched fail is not yet a
guard, and this one silently protects the central confound of the experiment -- a prompt that
names the literature tells the model which behaviour is being looked for.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import TYPE_CHECKING, Any

import pytest

from games import prompts
from games.payoffs import (
    COOPERATE,
    DEFECT,
    STAG_HUNT_VARIANTS,
    TEMPTATION_BY_VARIANT,
    DictatorSpec,
    MatrixGameSpec,
    OpponentRule,
    assert_prisoners_dilemma,
    fixed_pie_pd,
    max_iterated_return,
    simulate_iterated,
    stag_hunt,
    twin_pd,
    ultimatum_responder,
)
from games.prompts import (
    ALL_GAME_IDS,
    COOP_LABEL_INDICES,
    DICTATOR_ENDOWMENTS,
    DICTATOR_GAME_ID,
    DICTATOR_SCENARIOS,
    EVAL_ONLY_GAME_IDS,
    GAME_IDS,
    ITERATED_GAME_ID,
    ITERATED_GAME_IDS,
    ITERATED_N_ROUNDS,
    ITERATED_PAYOFF_VARIANT,
    ITERATED_PD_GRIM_GAME_ID,
    ITERATED_STAG_GAME_ID,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    MATRIX_SCENARIOS,
    MIN_EFFORT_GAME_ID,
    MIN_EFFORT_MATCH_GAME_ID,
    MIN_EFFORT_SCENARIOS,
    NASH_DEMAND_GAME_ID,
    NASH_DEMAND_SCENARIOS,
    OPP_COOP_PROB_UNSET,
    PAYOFF_VARIANTS,
    RESPONDER_COUNTERPART_CLAUSE,
    RESPONDER_SCENARIOS,
    ROW_COLUMNS,
    SPLIT_EVAL,
    SPLIT_TRAIN,
    SPLITS,
    STAG_HUNT_PAYOFF_VARIANTS,
    SYMMETRY_NOTE,
    TEMPTATION_DOSE_GAME_ID,
    TEMPTATION_DOSE_PAYOFF_VARIANTS,
    THRESHOLD_GOODS_GAME_ID,
    THRESHOLD_GOODS_SCENARIOS,
    TRUST_SCENARIOS,
    TRUST_STATED_RETURN_GAME_ID,
    TRUST_STRATEGY_METHOD_GAME_ID,
    TRUSTEE_RETURN_GAME_ID,
    TRUSTEE_SCENARIOS,
    TWIN_COUNTERPART_CLAUSE,
    ULTIMATUM_RESPONDER_GAME_ID,
    UNLABELLED_GAME_IDS,
    DictatorScenario,
    Scenario,
    assert_no_loaded_vocabulary,
    dictator_variant,
    generate_prompt_rows,
    render_dictator_prompt,
    render_game_prompt,
    render_iterated_prompt,
    render_responder_prompt,
)
from games.rewards import (
    GRADING_GROUP_MIX,
    GRADING_ITERATED_RETURN,
    GRADING_KEEP_FRACTION,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_SELF,
    GRADING_THRESHOLD_GOODS_GROUP_MIX,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    REQUIRED_REWARD_COLUMNS,
    make_game_reward,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# The grading each arm is actually trained under, so the generated rows are the real ones.
GRADING_BY_GAME: dict[str, str] = {
    "twin-pd": GRADING_GROUP_MIX,
    "fixed-pie-pd": GRADING_GROUP_MIX,
    "stag-hunt": GRADING_GROUP_MIX,
    "hi-lo": GRADING_GROUP_MIX,
    "harmony": GRADING_GROUP_MIX,
    "chicken": GRADING_GROUP_MIX,
    "pd-unstated": GRADING_GROUP_MIX,
    "pd-reskin": GRADING_SELF,
    "public-goods": GRADING_GROUP_MIX,
    "pd-vs-frozen": GRADING_VS_FIXED_MIX,
    "stag-hunt-vs-frozen": GRADING_VS_FIXED_MIX,
    ITERATED_GAME_ID: GRADING_ITERATED_RETURN,
    ITERATED_PD_GRIM_GAME_ID: GRADING_ITERATED_RETURN,
    ITERATED_STAG_GAME_ID: GRADING_ITERATED_RETURN,
    DICTATOR_GAME_ID: GRADING_KEEP_FRACTION,
    NASH_DEMAND_GAME_ID: GRADING_NASH_DEMAND_GROUP_MIX,
    THRESHOLD_GOODS_GAME_ID: GRADING_THRESHOLD_GOODS_GROUP_MIX,
    TRUST_STATED_RETURN_GAME_ID: GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    TRUST_STRATEGY_METHOD_GAME_ID: GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    MIN_EFFORT_GAME_ID: GRADING_MIN_EFFORT_GROUP_MIX,
    MIN_EFFORT_MATCH_GAME_ID: GRADING_LEVEL_MATCH_RETURN,
}

# Games whose corpus is one row per frame because the answer is a number: no label mapping to
# counterbalance, so every assertion about orientations, pairs or print order skips them.
LABELLED_GAME_IDS: tuple[str, ...] = tuple(
    game_id for game_id in GAME_IDS if game_id not in UNLABELLED_GAME_IDS
)

PAYOFF_COLUMNS = ("payoff_cc", "payoff_cd", "payoff_dc", "payoff_dd")


@pytest.fixture
def scenario() -> Scenario:
    return MATRIX_SCENARIOS[0]


@pytest.fixture
def spec() -> MatrixGameSpec:
    return twin_pd("temptation-2")


@pytest.fixture
def dictator_scenario() -> DictatorScenario:
    return DICTATOR_SCENARIOS[0]


@pytest.fixture
def dictator_spec() -> DictatorSpec:
    return DictatorSpec(game_id=DICTATOR_GAME_ID, endowment=DICTATOR_ENDOWMENTS[0])


def points(payoff: float) -> str:
    """Render a normalised payoff the way the prompt states it, for an exact substring match."""
    return f"{round(payoff * 100, 2):g}"


def points_shown(prompt: str, mine: str, theirs: str) -> float:
    """Read back the point value the prompt states for one outcome cell."""
    pattern = rf"- You {re.escape(mine)}, they {re.escape(theirs)}: you are credited (\S+) points\."
    match = re.search(pattern, prompt)
    assert match is not None, f"no outcome line for ({mine}, {theirs}) in:\n{prompt}"
    return float(match.group(1))


def outcome_lines(prompt: str) -> list[str]:
    """Return the prompt's outcome lines in the order it prints them.

    Matches the second-mover renderer's two-line block as well as the matrix renderer's four-line
    one, since both open every line with the same words.
    """
    return [line for line in prompt.splitlines() if line.startswith("- You ")]


def all_rows() -> list[dict[str, object]]:
    """Generate every row of the whole corpus: each game, each split."""
    return [
        row
        for game_id in GAME_IDS
        for split in SPLITS
        for row in generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=split)
    ]


class TestLoadedVocabularyGuard:
    @pytest.mark.parametrize(
        "banned",
        [
            "The crews cooperate on the overflow.",
            "Filing this way is a defection.",
            "This is the prisoner's version.",
            "A classic dilemma for the bay.",
            "Your twin files the same list.",
            "Consider Newcomb's problem here.",
            "They play tit-for-tat with the slack.",
            "They play tit for tat with the slack.",
            "The other crew will retaliate next week.",
            "Standard game theory applies.",
            "This is the Nash equilibrium.",
            "See the payoff matrix below.",
            "Filing PUSH is the dominant strategy.",
            "You are hunting a stag tonight.",
            "You are the dictator of this split.",
            "This is an ultimatum.",
            "A CDT agent files PUSH.",
            "An FDT reading says otherwise.",
            "Do not betray the other crew.",
            "The two crews collude on overflow.",
            "Basic decision theory covers this.",
            "Its decision is correlated with yours.",
            "Their filing correlates with yours.",
            "The correlation between the two filings is total.",
            # The prefixed forms name the construct just as plainly, and the bare `correlat\\w*`
            # form would pass every one of these: the wrapper's leading `\\b` has no boundary to
            # match before "correlat" in them.
            "Its decision is uncorrelated with yours.",
            "Its decision is anti-correlated with yours.",
            "Its decision is anticorrelated with yours.",
            "The two filings are decorrelated.",
            "Autocorrelation in the record settles it.",
        ],
    )
    def test_banned_vocabulary_raises(self, banned: str) -> None:
        with pytest.raises(ValueError, match="loaded vocabulary"):
            assert_no_loaded_vocabulary(banned)

    @pytest.mark.parametrize(
        "clean",
        [
            "You file at the same minute from separate terminals.",
            "The pallets stage in Bay 4 and the crew staggers the loads.",
            "Holding the slack costs you time on your own moves.",
            "The editor credits the points at the end of the week.",
            "Both filings are recorded at once.",
            # Word-boundary cases: each contains a banned word as a prefix, and must not trip.
            "The crates stage in the yard and the loads stagger across the shift.",
            "The twine is stored with the netting.",
            "The order does not dictate which gate you use.",
            "Editing the ledger is the clerk's job.",
            "The collar sits above the coupling.",
            # Near-misses for the prefix-consuming correlation entry, which is the one pattern here
            # that can start mid-token: nothing else in ordinary prose carries "correlat".
            "The two entries are related and the ledgers are unrelated.",
            "The clerk collates the relay sheets and files a corollary note.",
            "The core relay sits in the cellar and their relations with the yard are cordial.",
        ],
    )
    def test_ordinary_prose_passes(self, clean: str) -> None:
        assert_no_loaded_vocabulary(clean)

    def test_error_names_the_match_and_quotes_context(self) -> None:
        with pytest.raises(ValueError, match="'Defection'") as excinfo:
            assert_no_loaded_vocabulary("Bay 4 note. Defection is the usual outcome here.")
        assert "Bay 4 note" in str(excinfo.value)

    @pytest.mark.parametrize(
        ("text", "reported"),
        [
            ("Its decision is uncorrelated with yours.", "uncorrelated"),
            ("Its decision is anti-correlated with yours.", "anti-correlated"),
            ("The framing is same-weights-uncorrelated.", "same-weights-uncorrelated"),
        ],
    )
    def test_prefixed_correlation_forms_are_caught_and_named_whole(
        self, text: str, reported: str
    ) -> None:
        """A prefix must not buy a way past the guard, and the error must name the whole token.

        The correlation entry is the only one that starts mid-token, because the compiled
        alternation opens with `\\b` and there is no word boundary before "correlat" in
        "uncorrelated" -- so the inflection-suffix form every other entry uses would have let a
        denial of the construct render. Denying the correlation names it as plainly as asserting it.
        The reported word is pinned too: an author reading the traceback has to see the token they
        wrote, not the "correlated" tail sitting inside it.
        """
        with pytest.raises(ValueError, match=re.escape(f"loaded vocabulary '{reported}'")):
            assert_no_loaded_vocabulary(text)

    def test_guard_is_case_insensitive(self) -> None:
        with pytest.raises(ValueError, match="loaded vocabulary"):
            assert_no_loaded_vocabulary("COOPERATION is expected.")

    def test_every_authored_frame_is_clean(self) -> None:
        rosters = (
            *MATRIX_SCENARIOS,
            *RESPONDER_SCENARIOS,
            *DICTATOR_SCENARIOS,
            *NASH_DEMAND_SCENARIOS,
            *THRESHOLD_GOODS_SCENARIOS,
            *MIN_EFFORT_SCENARIOS,
            *TRUST_SCENARIOS,
            *TRUSTEE_SCENARIOS,
        )
        for authored in rosters:
            assert_no_loaded_vocabulary(authored.frame)
            assert_no_loaded_vocabulary(authored.scenario_id)

    def test_every_authored_label_is_clean(self) -> None:
        for authored in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS):
            assert_no_loaded_vocabulary(authored.label_a)
            assert_no_loaded_vocabulary(authored.label_b)

    @pytest.mark.parametrize("identifier", ["twin-pd-temptation-2", "stag-hunt", "dictator"])
    def test_internal_identifiers_trip_the_guard(self, identifier: str) -> None:
        """Several internal ids are themselves banned words, which is why none may reach a prompt.

        A debug line embedding a `game_id` or a `prompt_id` would leak "twin", "stag" or
        "dictator" into the text, so the guard has to be the thing standing in the way.
        """
        with pytest.raises(ValueError, match="loaded vocabulary"):
            assert_no_loaded_vocabulary(identifier)


class TestScenarioRoster:
    def test_scenario_ids_are_unique(self) -> None:
        ids = [
            authored.scenario_id
            for authored in (
                *MATRIX_SCENARIOS,
                *RESPONDER_SCENARIOS,
                *DICTATOR_SCENARIOS,
                # The claim roster joined this list on 2026-08-21: it had been left out when it was
                # added, so nothing was checking its ids against the other rosters', and a `prompt_id`
                # collision between two games is exactly what this test exists to catch.
                *NASH_DEMAND_SCENARIOS,
                *THRESHOLD_GOODS_SCENARIOS,
                *MIN_EFFORT_SCENARIOS,
                *TRUST_SCENARIOS,
                *TRUSTEE_SCENARIOS,
            )
        ]
        assert len(ids) == len(set(ids))

    def test_label_pairs_are_unique_per_frame(self) -> None:
        pairs = [authored.labels for authored in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS)]
        assert len(pairs) == len(set(pairs))

    def test_every_label_string_is_globally_unique(self) -> None:
        """No label may appear in two frames, or per-label analysis would pool them silently."""
        labels = [
            label
            for authored in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS)
            for label in authored.labels
        ]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        assert not duplicates, f"labels reused across frames: {duplicates}"

    def test_labels_are_single_words(self) -> None:
        for authored in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS):
            for label in authored.labels:
                assert label == label.strip()
                assert " " not in label, f"{authored.scenario_id}: {label!r} has a space"

    def test_every_trained_roster_holds_out_eval_frames(self) -> None:
        for roster in (
            MATRIX_SCENARIOS,
            DICTATOR_SCENARIOS,
            NASH_DEMAND_SCENARIOS,
            THRESHOLD_GOODS_SCENARIOS,
            MIN_EFFORT_SCENARIOS,
            TRUST_SCENARIOS,
        ):
            assert any(authored.eval_only for authored in roster)
            assert any(not authored.eval_only for authored in roster)

    def test_the_trustee_roster_is_held_out_entirely(self) -> None:
        assert TRUSTEE_SCENARIOS
        assert all(authored.eval_only for authored in TRUSTEE_SCENARIOS)

    def test_frame_must_introduce_both_labels(self) -> None:
        with pytest.raises(ValueError, match="never mentions"):
            Scenario(
                scenario_id="missing-label",
                frame="You may HOLD the pallets, or do the other thing.",
                label_a="HOLD",
                label_b="PUSH",
            )

    def test_identical_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="two distinct labels"):
            Scenario(
                scenario_id="same-labels",
                frame="You may HOLD or hold.",
                label_a="HOLD",
                label_b="hold",
            )

    def test_empty_frame_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty frame"):
            DictatorScenario(scenario_id="blank", frame="   ", resource="hours")


class TestRenderGamePrompt:
    def test_prompt_carries_frame_outcomes_and_tag_instruction(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_game_prompt(spec, scenario, coop_label_index=0)
        assert scenario.frame in prompt
        assert prompt.count("- You ") == len(scenario.labels) ** 2
        assert f"<action>{scenario.label_a}</action>" in prompt
        assert f"<action>{scenario.label_b}</action>" in prompt

    def test_rendered_prompt_passes_the_guard(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        assert_no_loaded_vocabulary(render_game_prompt(spec, scenario, coop_label_index=0))

    @pytest.mark.parametrize("coop_label_index", COOP_LABEL_INDICES)
    def test_stated_points_match_the_spec(
        self, spec: MatrixGameSpec, scenario: Scenario, coop_label_index: int
    ) -> None:
        prompt = render_game_prompt(spec, scenario, coop_label_index=coop_label_index)
        coop_label = scenario.coop_label(coop_label_index)
        other_label = next(label for label in scenario.labels if label != coop_label)
        assert points_shown(prompt, coop_label, coop_label) == pytest.approx(
            spec.payoff_cc * 100, abs=0.5
        )
        assert points_shown(prompt, coop_label, other_label) == pytest.approx(
            spec.payoff_cd * 100, abs=0.5
        )
        assert points_shown(prompt, other_label, coop_label) == pytest.approx(
            spec.payoff_dc * 100, abs=0.5
        )
        assert points_shown(prompt, other_label, other_label) == pytest.approx(
            spec.payoff_dd * 100, abs=0.5
        )

    def test_counterbalancing_moves_only_the_numbers(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        first, second = (
            render_game_prompt(spec, scenario, coop_label_index=index)
            for index in COOP_LABEL_INDICES
        )
        assert first != second
        # Any run of digits, so a two-digit value and a three-digit one still compare equal.
        numbers = re.compile(r"\d[\d.]*")
        assert numbers.sub("#", first) == numbers.sub("#", second)

    def test_counterbalancing_permutes_the_same_four_values(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        shown = [
            sorted(
                points_shown(
                    render_game_prompt(spec, scenario, coop_label_index=index), mine, theirs
                )
                for mine in scenario.labels
                for theirs in scenario.labels
            )
            for index in COOP_LABEL_INDICES
        ]
        assert shown[0] == shown[1]

    def test_twin_framing_describes_the_model_without_the_banned_word(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_game_prompt(spec, scenario, coop_label_index=0, twin_framing=True)
        assert "another instance of this same model" in prompt
        assert "same weights" in prompt
        assert "twin" not in prompt.casefold()
        assert_no_loaded_vocabulary(prompt)

    def test_opponent_framing_is_used_verbatim(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        clause = "you are matched with a system whose reply was recorded last week."
        prompt = render_game_prompt(spec, scenario, coop_label_index=0, opponent_framing=clause)
        assert f"About the other side: {clause}" in prompt

    def test_unframed_prompt_says_nothing_about_the_other_side(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_game_prompt(spec, scenario, coop_label_index=0)
        assert "About the other side" not in prompt

    def test_two_counterpart_framings_at_once_are_rejected(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        with pytest.raises(ValueError, match="at most one"):
            render_game_prompt(
                spec,
                scenario,
                coop_label_index=0,
                twin_framing=True,
                opponent_framing="you are matched with something else.",
            )

    def test_out_of_range_label_index_is_rejected(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        with pytest.raises(ValueError, match="coop_label_index"):
            render_game_prompt(spec, scenario, coop_label_index=2)

    def test_planted_banned_word_in_a_frame_makes_rendering_raise(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        sabotaged = dataclasses.replace(
            scenario, frame=f"{scenario.frame}\n\nFiling PUSH is a defection, in effect."
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_game_prompt(spec, sabotaged, coop_label_index=0)

    def test_planted_banned_word_in_the_counterpart_clause_makes_rendering_raise(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        """The clause is caller-supplied and never guarded on its own.

        So this only goes red if the guard runs on the whole assembled prompt rather than on the
        frame, which is the placement rule the corpus depends on.
        """
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_game_prompt(
                spec,
                scenario,
                coop_label_index=0,
                opponent_framing="you are matched with your twin, who reasons as you do.",
            )

    def test_planted_banned_word_in_a_resource_name_makes_rendering_raise(
        self, dictator_spec: DictatorSpec, dictator_scenario: DictatorScenario
    ) -> None:
        sabotaged = dataclasses.replace(dictator_scenario, resource="prisoner rations")
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_dictator_prompt(dictator_spec, sabotaged)

    def test_planted_banned_word_in_a_label_makes_rendering_raise(
        self, spec: MatrixGameSpec
    ) -> None:
        sabotaged = Scenario(
            scenario_id="planted-label",
            frame="You may DEFECT or HOLD, scored in points:",
            label_a="DEFECT",
            label_b="HOLD",
        )
        with pytest.raises(ValueError, match="loaded vocabulary"):
            render_game_prompt(spec, sabotaged, coop_label_index=0)


class TestRenderIteratedPrompt:
    def test_rule_is_described_by_its_mechanism(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_iterated_prompt(
            spec,
            scenario,
            rule=OpponentRule.TIT_FOR_TAT,
            n_rounds=ITERATED_N_ROUNDS,
            coop_label_index=0,
        )
        assert "repeat whatever you chose in the round before" in prompt
        assert OpponentRule.TIT_FOR_TAT.value not in prompt
        assert_no_loaded_vocabulary(prompt)

    @pytest.mark.parametrize("coop_label_index", COOP_LABEL_INDICES)
    def test_first_round_names_the_cooperative_label(
        self, spec: MatrixGameSpec, scenario: Scenario, coop_label_index: int
    ) -> None:
        prompt = render_iterated_prompt(
            spec,
            scenario,
            rule=OpponentRule.TIT_FOR_TAT,
            n_rounds=ITERATED_N_ROUNDS,
            coop_label_index=coop_label_index,
        )
        coop_label = scenario.coop_label(coop_label_index)
        assert f"in the first round they {coop_label}" in prompt

    def test_round_count_and_tag_instruction_agree(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_iterated_prompt(
            spec, scenario, rule=OpponentRule.TIT_FOR_TAT, n_rounds=3, coop_label_index=0
        )
        assert "This runs 3 times" in prompt
        assert "give all 3 decisions as 3 tags" in prompt

    @pytest.mark.parametrize("rule", list(OpponentRule))
    def test_every_rule_renders_and_stays_clean(
        self, spec: MatrixGameSpec, scenario: Scenario, rule: OpponentRule
    ) -> None:
        prompt = render_iterated_prompt(
            spec, scenario, rule=rule, n_rounds=ITERATED_N_ROUNDS, coop_label_index=1
        )
        assert "fixed written procedure" in prompt
        assert_no_loaded_vocabulary(prompt)

    def test_non_positive_rounds_are_rejected(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        with pytest.raises(ValueError, match="n_rounds"):
            render_iterated_prompt(
                spec, scenario, rule=OpponentRule.TIT_FOR_TAT, n_rounds=0, coop_label_index=0
            )


class TestRenderDictatorPrompt:
    def test_prompt_states_the_amount_and_the_tag(
        self, dictator_spec: DictatorSpec, dictator_scenario: DictatorScenario
    ) -> None:
        prompt = render_dictator_prompt(dictator_spec, dictator_scenario)
        assert dictator_scenario.frame in prompt
        assert str(dictator_spec.endowment) in prompt
        assert dictator_scenario.resource in prompt
        assert "<keep>N</keep>" in prompt
        assert f"between 0 and {dictator_spec.endowment}" in prompt

    def test_rendered_prompt_passes_the_guard(
        self, dictator_spec: DictatorSpec, dictator_scenario: DictatorScenario
    ) -> None:
        assert_no_loaded_vocabulary(render_dictator_prompt(dictator_spec, dictator_scenario))

    def test_every_dictator_frame_renders(self, dictator_spec: DictatorSpec) -> None:
        for authored in DICTATOR_SCENARIOS:
            assert_no_loaded_vocabulary(render_dictator_prompt(dictator_spec, authored))


class TestGeneratePromptRows:
    def test_every_generated_prompt_passes_the_guard(self) -> None:
        rows = all_rows()
        assert rows
        for row in rows:
            assert_no_loaded_vocabulary(str(row["prompt"]))

    def test_no_prompt_embeds_its_own_identifiers(self) -> None:
        """Identifiers stay out of the text entirely, guard or no guard.

        Not every id is a banned word (`iterated-pd-tft` is clean), so the guard alone cannot
        enforce this. A debug line that slipped an id into a prompt would also be a leak of the
        experiment's own bookkeeping into the material.
        """
        for row in all_rows():
            prompt = str(row["prompt"])
            assert str(row["prompt_id"]) not in prompt
            assert str(row["game_id"]) not in prompt
            assert str(row["payoff_variant"]) not in prompt
            assert str(row["reskin_id"]) not in prompt

    def test_schema_is_exactly_the_declared_columns(self) -> None:
        for row in all_rows():
            assert tuple(row.keys()) == ROW_COLUMNS

    def test_column_types_are_stable_across_games(self) -> None:
        for row in all_rows():
            assert isinstance(row["prompt"], str)
            assert isinstance(row["prompt_id"], str)
            assert isinstance(row["game_id"], str)
            assert isinstance(row["grading"], str)
            for column in PAYOFF_COLUMNS:
                assert isinstance(row[column], float)
            assert isinstance(row["label_a"], str)
            assert isinstance(row["label_b"], str)
            assert isinstance(row["coop_label"], str)
            assert isinstance(row["endowment"], int)
            assert isinstance(row["opp_coop_prob"], float)
            assert isinstance(row["opponent_rule"], str)
            assert isinstance(row["n_rounds"], int)
            assert isinstance(row["team_size"], int)
            assert isinstance(row["contribution_threshold"], int)
            assert isinstance(row["prize"], int)
            assert isinstance(row["transfer_multiplier"], float)
            assert isinstance(row["stated_return_fraction"], float)
            assert isinstance(row["reskin_id"], str)
            assert isinstance(row["payoff_variant"], str)

    def test_opponent_probability_is_unset_everywhere(self) -> None:
        for row in all_rows():
            assert row["opp_coop_prob"] == OPP_COOP_PROB_UNSET

    def test_one_shot_games_carry_no_rule_rounds_or_endowment(self) -> None:
        # The unlabelled games are excluded along with every repeated arm because each carries one of
        # the columns this asserts blank: the unilateral split and the trust games all hold a stock in
        # `endowment`, the minimum-effort games carry a level grid, and none of them names a
        # cooperative label. Keyed on the whole repeated set rather than one game id, so a repeated
        # form added upstream is excluded without an edit here -- which is what this test failing on
        # the grim-trigger and repeated-stag arms showed it was not.
        for game_id in GAME_IDS:
            if game_id in ITERATED_GAME_IDS or game_id in UNLABELLED_GAME_IDS:
                continue
            for row in generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=SPLIT_TRAIN):
                assert row["opponent_rule"] == ""
                assert row["n_rounds"] == 0
                assert row["endowment"] == 0
                assert row["coop_label"] in (row["label_a"], row["label_b"])

    def test_dictator_rows_carry_the_endowment_and_no_actions(self) -> None:
        rows = generate_prompt_rows(DICTATOR_GAME_ID, GRADING_KEEP_FRACTION, split=SPLIT_TRAIN)
        assert rows
        for row in rows:
            assert row["endowment"] in DICTATOR_ENDOWMENTS
            assert row["label_a"] == ""
            assert row["label_b"] == ""
            assert row["coop_label"] == ""
            assert row["payoff_variant"] == dictator_variant(int(str(row["endowment"])))
            for column in PAYOFF_COLUMNS:
                assert row[column] == 0.0

    def test_every_endowment_appears_for_every_dictator_frame(self) -> None:
        rows = generate_prompt_rows(DICTATOR_GAME_ID, GRADING_KEEP_FRACTION, split=SPLIT_TRAIN)
        by_frame: dict[object, set[object]] = {}
        for row in rows:
            by_frame.setdefault(row["reskin_id"], set()).add(row["endowment"])
        assert by_frame
        for frame, endowments in by_frame.items():
            assert endowments == set(DICTATOR_ENDOWMENTS), frame

    def test_dictator_prompts_state_their_own_endowment(self) -> None:
        for split in SPLITS:
            for row in generate_prompt_rows(DICTATOR_GAME_ID, GRADING_KEEP_FRACTION, split=split):
                assert f"comes to {row['endowment']} " in str(row["prompt"])
                assert f"between 0 and {row['endowment']}" in str(row["prompt"])

    def test_prompt_ids_are_unique_across_the_whole_corpus(self) -> None:
        prompt_ids = [row["prompt_id"] for row in all_rows()]
        assert len(prompt_ids) == len(set(prompt_ids))

    @pytest.mark.parametrize("game_id", LABELLED_GAME_IDS)
    @pytest.mark.parametrize("split", SPLITS)
    def test_both_label_mappings_share_one_game(self, game_id: str, split: str) -> None:
        rows = generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=split)
        by_frame_and_variant: dict[tuple[object, object], list[dict[str, object]]] = {}
        for row in rows:
            by_frame_and_variant.setdefault((row["reskin_id"], row["payoff_variant"]), []).append(
                row
            )
        assert by_frame_and_variant
        for key, pair in by_frame_and_variant.items():
            assert len(pair) == len(COOP_LABEL_INDICES), key
            first, second = pair
            for column in (*PAYOFF_COLUMNS, "label_a", "label_b", "game_id", "n_rounds"):
                assert first[column] == second[column], (key, column)
            assert {first["coop_label"], second["coop_label"]} == {
                first["label_a"],
                first["label_b"],
            }

    def test_coop_label_follows_the_index_in_the_prompt_id(self) -> None:
        for row in all_rows():
            prompt_id = str(row["prompt_id"])
            if prompt_id.endswith("coop0"):
                assert row["coop_label"] == row["label_a"]
            elif prompt_id.endswith("coop1"):
                assert row["coop_label"] == row["label_b"]

    @pytest.mark.parametrize("game_id", GAME_IDS)
    def test_train_and_eval_use_disjoint_frames(self, game_id: str) -> None:
        grading = GRADING_BY_GAME[game_id]
        train_frames = {
            row["reskin_id"] for row in generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN)
        }
        eval_frames = {
            row["reskin_id"] for row in generate_prompt_rows(game_id, grading, split=SPLIT_EVAL)
        }
        assert train_frames
        assert eval_frames
        assert not train_frames & eval_frames

    @pytest.mark.parametrize("split", SPLITS)
    def test_row_counts_are_the_expected_absolute_numbers(self, split: str) -> None:
        """Hard numbers, deliberately: the training target is 50-70 rows per two-variant game.

        Deriving the expectation from the roster would make this test agree with any roster,
        including one that has silently shrunk below the target. Growing the roster is meant to
        turn this red so the new counts get looked at. Current shape: 16 train / 4 eval frames,
        times payoff variants (2 for the PDs, 4 for the stag-hunt ladder, 1 for the
        single-variant controls), times 2 label mappings; the iterated arm is single-variant
        and the unilateral split has 3 endowments and no mapping. The simultaneous-claim game has
        its own 16 train / 4 eval frames times 3 windfalls, and no mapping either, so its 48 train
        rows come from frames rather than from a counterbalance.

        The trust games are 16 train / 4 eval consignment frames of their own with no label mapping
        to double them: 2 announced return rates gives 32 and 8, and the strategy method's single
        variant gives 16 and 4. Worth reading against the pin: at one row per frame per rate, each
        announced-rate ARM trains on 16 rows before selection drops any, which is the thinnest corpus
        on the slate after the unilateral split's 18.

        The threshold public good is the same shape again: 16 train / 4 eval shared-undertaking frames of
        its own times 2 prize variants, so 32 and 8, and each of its two pinned arms trains on 16 rows
        before selection drops any while its self-graded arm sees all 32.

        The two repeated arms added beside the copying PD reuse the matrix roster, so the grim-trigger
        PD matches it exactly (one variant x 2 mappings = 32 and 8) and the repeated stag hunt carries
        the whole four-rung ladder (128 and 32), same as the one-shot hunt.

        The minimum-effort games have their own 16 train / 4 eval frames and no label mapping: the
        one-shot form crosses them with all four cells of the knob grid for 64 and 16, so a single
        sweep prices every cell, and each registered ARM then trains on 16 of those rows. The repeated
        form renders the one variant whose counterpart count is 1, giving 16 and 4 -- level with the
        strategy method as the thinnest corpus on the slate, which is why the design's floor of 16
        authored train frames is a floor rather than a target.
        """
        expected = {
            SPLIT_TRAIN: {
                "twin-pd": 64,
                "fixed-pie-pd": 64,
                "pd-unstated": 64,
                "pd-reskin": 176,
                "public-goods": 32,
                "pd-vs-frozen": 64,
                "stag-hunt": 128,
                "stag-hunt-vs-frozen": 128,
                "hi-lo": 32,
                "harmony": 32,
                "chicken": 32,
                ITERATED_GAME_ID: 32,
                ITERATED_PD_GRIM_GAME_ID: 32,
                ITERATED_STAG_GAME_ID: 128,
                DICTATOR_GAME_ID: 18,
                NASH_DEMAND_GAME_ID: 48,
                THRESHOLD_GOODS_GAME_ID: 32,
                TRUST_STATED_RETURN_GAME_ID: 32,
                TRUST_STRATEGY_METHOD_GAME_ID: 16,
                MIN_EFFORT_GAME_ID: 64,
                MIN_EFFORT_MATCH_GAME_ID: 16,
            },
            SPLIT_EVAL: {
                "twin-pd": 16,
                "fixed-pie-pd": 16,
                "pd-unstated": 16,
                "pd-reskin": 48,
                "public-goods": 8,
                "pd-vs-frozen": 16,
                "stag-hunt": 32,
                "stag-hunt-vs-frozen": 32,
                "hi-lo": 8,
                "harmony": 8,
                "chicken": 8,
                ITERATED_GAME_ID: 8,
                ITERATED_PD_GRIM_GAME_ID: 8,
                ITERATED_STAG_GAME_ID: 32,
                DICTATOR_GAME_ID: 6,
                NASH_DEMAND_GAME_ID: 12,
                THRESHOLD_GOODS_GAME_ID: 8,
                TRUST_STATED_RETURN_GAME_ID: 8,
                TRUST_STRATEGY_METHOD_GAME_ID: 4,
                MIN_EFFORT_GAME_ID: 16,
                MIN_EFFORT_MATCH_GAME_ID: 4,
            },
        }[split]
        assert set(expected) == set(GAME_IDS)
        for game_id, n_expected in expected.items():
            rows = generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=split)
            assert len(rows) == n_expected, game_id

    def test_roster_is_large_enough_for_the_training_target(self) -> None:
        n_train_frames = len(prompts._scenarios_for_split(SPLIT_TRAIN))
        n_eval_frames = len(prompts._scenarios_for_split(SPLIT_EVAL))
        assert n_train_frames >= 14, "two-variant games need >= 14 train frames to clear 50 rows"
        assert n_eval_frames >= 4
        assert n_train_frames * len(PAYOFF_VARIANTS) * len(COOP_LABEL_INDICES) >= 50

    def test_iterated_rows_carry_the_rule_the_rounds_and_one_variant(self) -> None:
        rows = generate_prompt_rows(ITERATED_GAME_ID, GRADING_ITERATED_RETURN, split=SPLIT_TRAIN)
        assert rows
        reference = twin_pd(ITERATED_PAYOFF_VARIANT)
        for row in rows:
            assert row["opponent_rule"] == OpponentRule.TIT_FOR_TAT.value
            assert row["n_rounds"] == ITERATED_N_ROUNDS
            assert row["payoff_variant"] == ITERATED_PAYOFF_VARIANT
            assert row["payoff_cc"] == reference.payoff_cc
            assert str(row["prompt"]).count("<action>") == len(COOP_LABEL_INDICES)

    def test_iterated_rows_never_use_the_large_temptation(self) -> None:
        for split in SPLITS:
            variants = {
                row["payoff_variant"]
                for row in generate_prompt_rows(
                    ITERATED_GAME_ID, GRADING_ITERATED_RETURN, split=split
                )
            }
            assert variants == {ITERATED_PAYOFF_VARIANT}

    @pytest.mark.parametrize(
        ("frozen_game", "base_spec_for_variant"),
        [
            ("pd-vs-frozen", twin_pd),
            ("stag-hunt-vs-frozen", stag_hunt),
        ],
    )
    def test_frozen_arms_reuse_the_base_games_payoffs(
        self, frozen_game: str, base_spec_for_variant: Callable[[str], MatrixGameSpec]
    ) -> None:
        rows = generate_prompt_rows(frozen_game, GRADING_VS_FIXED_MIX, split=SPLIT_TRAIN)
        assert rows
        for row in rows:
            base = base_spec_for_variant(str(row["payoff_variant"]))
            assert row["payoff_cc"] == base.payoff_cc
            assert row["payoff_dd"] == base.payoff_dd
            assert "About the other side: you are matched with a different automated system" in str(
                row["prompt"]
            )

    def test_fixed_pie_prompts_state_a_constant_total(self) -> None:
        rows = generate_prompt_rows("fixed-pie-pd", GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        assert rows
        for row in rows:
            base = fixed_pie_pd(str(row["payoff_variant"]))
            assert row["payoff_cc"] == base.payoff_cc

    def test_twin_arms_describe_the_counterpart_as_this_model(self) -> None:
        for game_id in ("twin-pd", "fixed-pie-pd", "stag-hunt", "hi-lo", "harmony", "chicken"):
            for row in generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_TRAIN):
                assert "another instance of this same model" in str(row["prompt"])

    def test_generation_is_deterministic(self) -> None:
        for game_id in GAME_IDS:
            grading = GRADING_BY_GAME[game_id]
            assert generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN) == (
                generate_prompt_rows(game_id, grading, split=SPLIT_TRAIN)
            )

    def test_unknown_game_id_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown game_id"):
            generate_prompt_rows("hopscotch", GRADING_GROUP_MIX, split=SPLIT_TRAIN)

    def test_unknown_grading_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown grading"):
            generate_prompt_rows("twin-pd", "vibes", split=SPLIT_TRAIN)

    def test_unknown_split_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            generate_prompt_rows("twin-pd", GRADING_GROUP_MIX, split="holdout")


class TestRewardConsumesTheCorpus:
    """The seam between this corpus and `games/rewards.py`, which nothing else covers.

    Both sides name the same columns from memory. If either renames one, TRL forwards the rest
    happily and the reward function sees a hole rather than an error, so the check is here.
    """

    def score(self, row: dict[str, object], completions: list[str]) -> list[float]:
        """Score `completions` as one group, all drawn from the same row."""
        columns = {
            column: [row[column]] * len(completions) for column in ROW_COLUMNS if column != "prompt"
        }
        reward = make_game_reward(len(completions), prefilled_think=False)
        return reward(
            completions=completions,
            log_metric=lambda _name, _value: None,
            log_extra=lambda _name, _values: None,
            **columns,
        )

    def test_every_column_the_reward_needs_is_generated(self) -> None:
        assert set(REQUIRED_REWARD_COLUMNS) <= set(ROW_COLUMNS)

    def test_group_mix_rows_score_as_the_expected_payoff(self) -> None:
        row = generate_prompt_rows("twin-pd", GRADING_GROUP_MIX, split=SPLIT_TRAIN)[0]
        other_label = next(
            label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
        )
        rewards = self.score(
            row,
            [f"<action>{row['coop_label']}</action>", f"<action>{other_label}</action>"],
        )
        # One of the two completions cooperated, so each is graded against p = 0.5.
        assert rewards[0] == pytest.approx(
            0.5 * float(row["payoff_cc"]) + 0.5 * float(row["payoff_cd"])
        )
        assert rewards[1] == pytest.approx(
            0.5 * float(row["payoff_dc"]) + 0.5 * float(row["payoff_dd"])
        )

    def test_dictator_rows_score_as_the_kept_fraction(self) -> None:
        row = generate_prompt_rows(DICTATOR_GAME_ID, GRADING_KEEP_FRACTION, split=SPLIT_TRAIN)[0]
        endowment = int(str(row["endowment"]))
        rewards = self.score(row, ["<keep>40</keep>", f"<keep>{endowment}</keep>"])
        assert rewards == pytest.approx([40 / endowment, 1.0])

    def test_iterated_rows_score_as_the_simulated_return(self) -> None:
        row = generate_prompt_rows(ITERATED_GAME_ID, GRADING_ITERATED_RETURN, split=SPLIT_TRAIN)[0]
        n_rounds = int(str(row["n_rounds"]))
        other_label = next(
            label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
        )
        rewards = self.score(
            row,
            [
                f"<action>{row['coop_label']}</action>" * n_rounds,
                f"<action>{other_label}</action>" * n_rounds,
            ],
        )
        spec = twin_pd(ITERATED_PAYOFF_VARIANT)
        rule = OpponentRule.TIT_FOR_TAT
        best = max_iterated_return(spec, rule, n_rounds)
        for index, move in enumerate((COOPERATE, DEFECT)):
            expected = simulate_iterated(spec, rule, [move] * n_rounds) / best
            assert rewards[index] == pytest.approx(expected)


class TestThePdUnstatedGame:
    """twin-pd's payoffs with the counterpart paragraph omitted: the learning-side framing test.

    The trained twin-pd pair diverged under its two gradings WITH the twin clause in every prompt,
    and the eval-side framing sweep says expression of that divergence is gated on the clause. This
    game renders the same frames, the same payoff cells and the same instruction with no "About the
    other side" paragraph at all, so training it under the same two gradings asks whether the clause
    is load-bearing for LEARNING the divergence or only for expressing it -- the 2x2's missing
    column. Unstated rather than the frozen clause on purpose: a self-graded arm under "nothing you
    write can change it" would train on prompts that lie about the reward's coupling, the mirror of
    the two-experiments-at-once case the vs-frozen arms exist to avoid.
    """

    GAME_ID = "pd-unstated"

    def test_registered_as_a_trainable_matrix_game(self) -> None:
        assert self.GAME_ID in GAME_IDS
        assert self.GAME_ID in prompts.MATRIX_GAME_IDS

    @pytest.mark.parametrize("split", SPLITS)
    @pytest.mark.parametrize("grading", [GRADING_SELF, GRADING_GROUP_MIX])
    def test_no_row_says_anything_about_the_other_side(self, split: str, grading: str) -> None:
        """The defining property, under both gradings the pair trains: no counterpart language.

        "About the other side" is the one opening every counterpart paragraph in the corpus renders
        (`games.prompts.about_the_other_side`), so matching on it covers the twin clause, the frozen
        clause and every framing-sweep clause at once.
        """
        rows = generate_prompt_rows(self.GAME_ID, grading, split=split)
        assert rows
        for row in rows:
            prompt = str(row["prompt"])
            assert "About the other side" not in prompt
            assert "another instance of this same model" not in prompt

    @pytest.mark.parametrize("split", SPLITS)
    def test_prompts_are_twin_pds_with_exactly_the_paragraph_deleted(self, split: str) -> None:
        """Byte-identical to the trained pair's prompts apart from one deleted paragraph.

        This is the whole experimental contrast, so it is pinned at the byte level: if the two
        renderings ever drift anywhere else -- frames, point values, instruction, print order --
        the pair stops isolating the framing and this goes red.
        """
        twin_by_key = {
            str(row["prompt_id"]).removeprefix("twin-pd--"): str(row["prompt"])
            for row in generate_prompt_rows("twin-pd", GRADING_GROUP_MIX, split=split)
        }
        rows = generate_prompt_rows(self.GAME_ID, GRADING_GROUP_MIX, split=split)
        assert rows
        for row in rows:
            key = str(row["prompt_id"]).removeprefix(f"{self.GAME_ID}--")
            twin_prompt = twin_by_key[key]
            expected = twin_prompt.replace(
                f"\n\nAbout the other side: {TWIN_COUNTERPART_CLAUSE}", ""
            )
            assert expected != twin_prompt, "the twin rendering must carry the paragraph to delete"
            assert str(row["prompt"]) == expected

    def test_reuses_twin_pd_payoffs_on_both_variants(self) -> None:
        rows = generate_prompt_rows(self.GAME_ID, GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        assert {row["payoff_variant"] for row in rows} == set(PAYOFF_VARIANTS)
        for row in rows:
            base = twin_pd(str(row["payoff_variant"]))
            assert row["payoff_cc"] == base.payoff_cc
            assert row["payoff_cd"] == base.payoff_cd
            assert row["payoff_dc"] == base.payoff_dc
            assert row["payoff_dd"] == base.payoff_dd

    def score(self, row: dict[str, object], grading: str, completions: list[str]) -> list[float]:
        """Score `completions` as one group drawn from `row`, regraded under `grading`."""
        regraded = {**row, "grading": grading}
        columns = {
            column: [regraded[column]] * len(completions)
            for column in ROW_COLUMNS
            if column != "prompt"
        }
        reward = make_game_reward(len(completions), prefilled_think=False)
        return reward(
            completions=completions,
            log_metric=lambda _name, _value: None,
            log_extra=lambda _name, _values: None,
            **columns,
        )

    def test_self_grading_resolves_to_the_diagonal_cells(self) -> None:
        """The coupling survives the deleted paragraph: cooperation still pays CC vs DD."""
        row = generate_prompt_rows(self.GAME_ID, GRADING_SELF, split=SPLIT_TRAIN)[0]
        other_label = next(
            label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
        )
        rewards = self.score(
            row,
            GRADING_SELF,
            [f"<action>{row['coop_label']}</action>", f"<action>{other_label}</action>"],
        )
        assert rewards == pytest.approx([float(row["payoff_cc"]), float(row["payoff_dd"])])

    def test_group_mix_grading_resolves_to_the_expected_payoff(self) -> None:
        row = generate_prompt_rows(self.GAME_ID, GRADING_GROUP_MIX, split=SPLIT_TRAIN)[0]
        other_label = next(
            label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
        )
        rewards = self.score(
            row,
            GRADING_GROUP_MIX,
            [f"<action>{row['coop_label']}</action>", f"<action>{other_label}</action>"],
        )
        assert rewards[0] == pytest.approx(
            0.5 * float(row["payoff_cc"]) + 0.5 * float(row["payoff_cd"])
        )
        assert rewards[1] == pytest.approx(
            0.5 * float(row["payoff_dc"]) + 0.5 * float(row["payoff_dd"])
        )


class TestThePdReskinGame:
    """twin-pd's payoffs over a wide, register-diverse skin bank with no counterpart language.

    The wave-3 "many stories" arm: pd-unstated showed the grading rule alone teaches cooperation
    on story-free prompts and that the trained cooperation expresses across the belief framings.
    This game holds everything pd-unstated held (payoff cells, outcome block, instruction, no
    "About the other side" paragraph) and moves ONE axis: the number and variety of surface
    fictions, on its own authored roster instead of the shared house one. What the tests pin is
    the arm's whole identity: the roster's size and split, the payoff reuse, the structural
    absence of counterpart language, and the coupling gate that keeps a future skin from smuggling
    a decision-coupling story back in through the fiction.
    """

    GAME_ID = "pd-reskin"

    def test_registered_as_a_trainable_matrix_game(self) -> None:
        assert self.GAME_ID in GAME_IDS
        assert self.GAME_ID in prompts.MATRIX_GAME_IDS

    def test_the_roster_is_44_train_and_12_eval_skins_disjoint_from_the_house_frames(self) -> None:
        train = [s for s in prompts.RESKIN_SCENARIOS if not s.eval_only]
        held_out = [s for s in prompts.RESKIN_SCENARIOS if s.eval_only]
        assert len(train) == 44
        assert len(held_out) == 12
        reskin_ids = [s.scenario_id for s in prompts.RESKIN_SCENARIOS]
        assert len(set(reskin_ids)) == len(reskin_ids)
        house_ids = {s.scenario_id for s in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS)}
        assert not house_ids & set(reskin_ids)

    @pytest.mark.parametrize("split", SPLITS)
    @pytest.mark.parametrize("grading", [GRADING_SELF, GRADING_GROUP_MIX])
    def test_no_row_says_anything_about_the_other_side(self, split: str, grading: str) -> None:
        """The defining property carried over from pd-unstated: no counterpart paragraph at all."""
        rows = generate_prompt_rows(self.GAME_ID, grading, split=split)
        assert rows
        for row in rows:
            prompt = str(row["prompt"])
            assert "About the other side" not in prompt
            assert "another instance of this same model" not in prompt

    @pytest.mark.parametrize("split", SPLITS)
    @pytest.mark.parametrize(
        "label_print_order", [LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED]
    )
    def test_every_rendered_prompt_passes_the_coupling_gate(
        self, split: str, label_print_order: str
    ) -> None:
        """No skin's fiction may say how the counterpart decides, in any rendering of it."""
        rows = generate_prompt_rows(
            self.GAME_ID, GRADING_SELF, split=split, label_print_order=label_print_order
        )
        assert rows
        for row in rows:
            prompts.assert_no_coupling_claims(str(row["prompt"]))

    def test_the_coupling_gate_goes_red_on_every_counterpart_clause(self) -> None:
        """The sabotage item, kept as a permanent test: a gate never watched to fail is a
        reassuring message. Each registered counterpart clause -- the exact prose a skin could
        smuggle back in -- must trip it, and so must a house frame with one planted sentence."""
        clauses = [
            TWIN_COUNTERPART_CLAUSE,
            prompts.FROZEN_OPPONENT_CLAUSE,
            prompts.STATED_MATCHER_CLAUSE,
            prompts.ANOTHER_AI_COUNTERPART_CLAUSE,
            prompts.STATED_ALWAYS_COOP_CLAUSE_TEMPLATE,
            prompts.SAME_WEIGHTS_UNCORRELATED_COUNTERPART_CLAUSE,
        ]
        for clause in clauses:
            with pytest.raises(ValueError, match="coupling"):
                prompts.assert_no_coupling_claims(clause)
        planted = MATRIX_SCENARIOS[0].frame + "\n\nThe other dispatcher has already decided."
        with pytest.raises(ValueError, match="coupling"):
            prompts.assert_no_coupling_claims(planted)

    def _skin_carrying(self, sentence: str) -> Scenario:
        """A well-formed skin whose only irregularity is one planted sentence."""
        return Scenario(
            scenario_id="sabotaged-skin",
            label_a="ALPHA",
            label_b="BETA",
            frame=(
                f"NOTE -- sabotage\n\nTwo clerks each enter one code, ALPHA or BETA, at the same "
                f"moment. {sentence}\n\nThe desk is scored in desk points, from both entries "
                f"together:"
            ),
        )

    def test_the_roster_validation_refuses_a_sabotaged_skin(self) -> None:
        """The import-time gate itself watched to fail: a roster holding one coupling-claiming
        skin must be refused, since that is the exact violation it exists to catch."""
        with pytest.raises(ValueError, match="coupling"):
            prompts.validate_reskin_roster(
                (
                    *prompts.RESKIN_SCENARIOS,
                    self._skin_carrying("The other clerk is running the same weights as you."),
                )
            )

    @pytest.mark.parametrize(
        ("family", "sentence"),
        [
            ("division", "We split the morning between us."),
            ("division", "Each of us takes one of the two rounds."),
            ("division", "We each pick our patch and stay on it."),
            ("division", "The two rounds go one each."),
            ("division", "You take one and they take the other."),
            ("matching", "The two entries must match when the desk reads them."),
            ("matching", "Your entry has to agree with theirs."),
            ("do-the-same", "I will do the same when my turn comes."),
        ],
    )
    def test_the_roster_validation_refuses_each_later_added_family(
        self, family: str, sentence: str
    ) -> None:
        """The three families added after the bank was audited, each watched red.

        They exist because the original rule forbade saying how the counterpart decides and
        nothing forbade prose from which what it chooses follows. A division sentence says the two
        parties are splitting a two-item job, so the counterpart's choice is inferable as the other
        option; a matching sentence says the two choices have to agree; "I will do the same" in the
        counterpart's voice is a stated matcher policy. All three were live in the authored bank
        when this test was written, and the gate was watched red on the real prose before the prose
        was fixed -- these planted sentences keep that red available after the fix.
        """
        del family  # names the family in the test id; the assertion is the same for each
        with pytest.raises(ValueError, match="coupling"):
            prompts.validate_reskin_roster(
                (*prompts.RESKIN_SCENARIOS, self._skin_carrying(sentence))
            )

    @pytest.mark.parametrize(
        "sentence",
        [
            "Its decision comes out a copy of yours.",
            "It follows the same procedure as you do.",
            "The other clerk reasons the same way.",
            "They always pick the second code.",
        ],
    )
    def test_the_narrowed_patterns_keep_their_decision_sense(self, sentence: str) -> None:
        """Three patterns were narrowed to their decision senses; narrowing must not disarm them."""
        with pytest.raises(ValueError, match="coupling"):
            prompts.assert_no_coupling_claims(sentence)

    @pytest.mark.parametrize(
        "sentence",
        [
            "A carbon copy of yours stays in the book.",
            "The tin is opened by the same procedure as last spring.",
            "They always open the post on Sunday.",
            "The two ledgers are read by the same method.",
            "Two assayers split the mixed batch across the same bench.",
            "Two line foremen take the storm damage on the same route, one gang each.",
            "The wall is primed, half of it is mine and half is yours.",
        ],
    )
    def test_the_gate_spares_the_fictions_own_furniture(self, sentence: str) -> None:
        """The other half of a gate with teeth: what it must NOT eat.

        The broad forms of the narrowed patterns went red on reveal ritual -- a filed carbon copy,
        a tin opened the way it always is, a clerk who always opens the post on Sunday -- and the
        division family must leave the house-roster standard alone, which is precisely that the two
        parties' own OBJECTS are partitioned (a gang each, half the wall each) while their CHOICES
        are not. A gate that forbids the fiction's furniture teaches its next author to loosen it.
        """
        prompts.assert_no_coupling_claims(sentence)

    def test_every_skin_carries_exactly_one_document_register(self) -> None:
        assert set(prompts.RESKIN_REGISTERS) == {s.scenario_id for s in prompts.RESKIN_SCENARIOS}
        assert set(prompts.RESKIN_REGISTERS.values()) == {
            prompts.REGISTER_HOUSE,
            prompts.REGISTER_FAR,
            prompts.REGISTER_FAR_AMBIGUOUS,
        }

    def test_the_held_out_register_split_is_pinned_by_id(self) -> None:
        """The registered prediction's ground truth, in code rather than in a design sentence.

        In-register held-out skins moving more than far-register ones is the register-keyed
        reading and equal movement the abstraction signature, so the split has to be exactly one
        thing. Prose said "8 house-register + 4 far-register" and named no ids, and re-deriving it
        by eye does not converge -- `choir-part-split` wears a house header over a community voice
        -- so it carries `far-ambiguous` and the readout reports the far cell with and without it.
        """
        held_out = {s.scenario_id for s in prompts.RESKIN_SCENARIOS if s.eval_only}
        by_register = {
            register: {i for i in held_out if prompts.RESKIN_REGISTERS[i] == register}
            for register in (
                prompts.REGISTER_HOUSE,
                prompts.REGISTER_FAR,
                prompts.REGISTER_FAR_AMBIGUOUS,
            )
        }
        assert by_register[prompts.REGISTER_FAR] == {
            "mural-wall-halves",
            "housemate-weekend-list",
            "penfriend-cutting-swap",
        }
        assert by_register[prompts.REGISTER_FAR_AMBIGUOUS] == {"choir-part-split"}
        assert len(by_register[prompts.REGISTER_HOUSE]) == 8

    def test_training_carries_far_register_skins_of_its_own(self) -> None:
        """The named caveat on the far-register cell, pinned so the readout cannot forget it.

        The far held-out skins are not out-of-genre: each pairs genre-for-genre with one of these
        three trained personal notes, so that contrast is 3-of-44 exposure against 30-of-44, not
        presence against absence. Removing one of them changes what the far cell measures.
        """
        trained_far = {
            s.scenario_id
            for s in prompts.RESKIN_SCENARIOS
            if not s.eval_only and prompts.RESKIN_REGISTERS[s.scenario_id] == prompts.REGISTER_FAR
        }
        assert trained_far == {"fete-stall-pitches", "ridge-walk-loads", "tool-shed-restock"}

    def test_the_roster_validation_refuses_a_register_mapping_gap(self) -> None:
        """Both directions of the gap watched red: an unlabelled skin, and a label with no skin."""
        unregistered = dataclasses.replace(
            prompts.RESKIN_SCENARIOS[0], scenario_id="unregistered-skin"
        )
        with pytest.raises(ValueError, match="carry no register"):
            prompts.validate_reskin_roster((*prompts.RESKIN_SCENARIOS, unregistered))
        with pytest.raises(ValueError, match="name no skin"):
            prompts.validate_reskin_roster(prompts.RESKIN_SCENARIOS[1:])

    def test_no_label_carries_a_digit_or_contains_the_other(self) -> None:
        """Both halves are about the parser, which matches a tag's contents exactly.

        A digit-bearing label invites the model to write `FIRE 9` for `FIRE9`, which is not a near
        miss but a parse failure carrying the -1.0 penalty, and concentrated parse failure on one
        skin distorts its selection band and reads as defection. No house label ever carried a
        digit, so three reskin skins were the whole exposure and now carry words. The containment
        half is cheaper: exact matching means a substring pair cannot be mis-parsed, but
        `frame_label_positions` finds the shorter label inside the longer one and would mis-report
        the frame-mention order the label-print-order audit is built on.
        """
        for scenario in prompts.RESKIN_SCENARIOS:
            first, second = scenario.labels
            assert not any(character.isdigit() for character in first + second), (
                scenario.scenario_id
            )
            assert first.casefold() not in second.casefold(), scenario.scenario_id
            assert second.casefold() not in first.casefold(), scenario.scenario_id

    def test_the_only_label_words_shared_between_frames_are_the_two_known_ones(self) -> None:
        """Label reuse here is pre-existing and pre-registered; a THIRD instance is the finding.

        `STRIP` is a reskin train skin's label reused by a held-out one with the same meaning, so
        that skin's held-out reading is caveated rather than clean, and `OPEN` is shared between the
        house roster and a reskin skin. Both are accepted and recorded; a new collision would pool
        two frames in any per-label analysis without saying so.
        """
        frames_by_label: dict[str, list[str]] = {}
        for scenario in (*MATRIX_SCENARIOS, *RESPONDER_SCENARIOS, *prompts.RESKIN_SCENARIOS):
            for label in scenario.labels:
                frames_by_label.setdefault(label, []).append(scenario.scenario_id)
        shared = {label for label, ids in frames_by_label.items() if len(ids) > 1}
        assert shared == {"OPEN", "STRIP"}

    def test_reuses_twin_pd_payoffs_on_both_variants(self) -> None:
        rows = generate_prompt_rows(self.GAME_ID, GRADING_SELF, split=SPLIT_TRAIN)
        assert {row["payoff_variant"] for row in rows} == set(PAYOFF_VARIANTS)
        for row in rows:
            base = twin_pd(str(row["payoff_variant"]))
            assert row["payoff_cc"] == base.payoff_cc
            assert row["payoff_cd"] == base.payoff_cd
            assert row["payoff_dc"] == base.payoff_dc
            assert row["payoff_dd"] == base.payoff_dd

    def test_self_grading_resolves_to_the_diagonal_cells_through_the_corpus_seam(self) -> None:
        """The coupling survives every skin: cooperation still pays CC vs DD on a reskin row."""
        row = generate_prompt_rows(self.GAME_ID, GRADING_SELF, split=SPLIT_TRAIN)[0]
        other_label = next(
            label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
        )
        columns = {column: [row[column]] * 2 for column in ROW_COLUMNS if column != "prompt"}
        reward = make_game_reward(2, prefilled_think=False)
        rewards = reward(
            completions=[
                f"<action>{row['coop_label']}</action>",
                f"<action>{other_label}</action>",
            ],
            log_metric=lambda _name, _value: None,
            log_extra=lambda _name, _values: None,
            **columns,
        )
        assert rewards == pytest.approx([float(row["payoff_cc"]), float(row["payoff_dd"])])


class TestTheEvalOnlyTransferGames:
    """The eval-rendered, never-trained set: a training split for any member must refuse.

    They exist to measure transfer to a game the model was never trained on, so a training split
    for them would make that claim false while everything still ran and produced numbers. The
    registry refuses it rather than trusting every future caller to remember. Chicken left this
    set when it was promoted to the trainable numeric-prediction control; defective-coordination
    replaced it as the negative control where the cooperative label is the wrong answer;
    public-goods left on 2026-08-26, promoted for the transfer-of-learning pair.
    """

    def test_they_are_absent_from_the_trainable_games(self) -> None:
        for game_id in EVAL_ONLY_GAME_IDS:
            assert game_id not in GAME_IDS
            assert game_id in ALL_GAME_IDS

    @pytest.mark.parametrize("game_id", EVAL_ONLY_GAME_IDS)
    def test_a_training_split_raises(self, game_id: str) -> None:
        with pytest.raises(ValueError, match="is eval-only"):
            generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_TRAIN)

    @pytest.mark.parametrize(
        "game_id", [game for game in EVAL_ONLY_GAME_IDS if game not in UNLABELLED_GAME_IDS]
    )
    def test_an_eval_split_renders_both_label_mappings_on_held_out_frames(
        self, game_id: str
    ) -> None:
        rows = generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL)
        assert rows
        assert {row["coop_label"] for row in rows} == {row["label_a"] for row in rows} | {
            row["label_b"] for row in rows
        }
        assert len({row["prompt_id"] for row in rows}) == len(rows)

    @pytest.mark.parametrize(
        "game_id", [game for game in EVAL_ONLY_GAME_IDS if game in UNLABELLED_GAME_IDS]
    )
    def test_an_unlabelled_eval_game_says_so_in_its_columns(self, game_id: str) -> None:
        """The label assertion above is vacuously true of an empty-label row, so state it instead.

        `{""} == {""} | {""}` passes, which means a game answering with a number would slip through
        the counterbalance check reporting nothing. Being explicit about which games have no labels is
        what keeps that from reading as coverage.
        """
        rows = generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL)
        assert rows
        for row in rows:
            assert row["label_a"] == ""
            assert row["label_b"] == ""
            assert row["coop_label"] == ""
        assert len({row["prompt_id"] for row in rows}) == len(rows)

    @pytest.mark.parametrize("game_id", EVAL_ONLY_GAME_IDS)
    def test_their_prompts_pass_the_vocabulary_guard(self, game_id: str) -> None:
        """They go through the same renderer, so the guard covers them like any other game."""
        for row in generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL):
            assert_no_loaded_vocabulary(row["prompt"])

    def test_the_never_trained_set_is_exactly_these_games(self) -> None:
        # trustee-return-rule joined on 2026-08-21 and is why this is no longer "the three games":
        # it asks for the return share the trust strategy method trains, in the role where paying it
        # is costly, so a training split for it would destroy the comparison it exists to make.
        # twin-pd-temptation-dose joined on 2026-08-26: twin-pd's own payoffs over every registered
        # temptation rung, eval-only so the trained arms' two-rung corpora cannot move.
        # public-goods LEFT on 2026-08-26, promoted to trainable for the transfer-of-learning pair
        # (chicken's precedent); its never-trained measurements up to that date stand.
        # defective-harmony joined on 2026-09-04 as wave 4b's trap cell: defection dominates for
        # both sides AND mutual defection is the best cell, so cooperation there is not a welfare
        # answer under any weighting and a rise reads as label following or print position.
        assert set(EVAL_ONLY_GAME_IDS) == {
            "ultimatum-responder",
            "defective-coordination",
            "defective-harmony",
            TRUSTEE_RETURN_GAME_ID,
            TEMPTATION_DOSE_GAME_ID,
        }

    @pytest.mark.parametrize(
        "game_id",
        ["public-goods", "defective-coordination", "defective-harmony", TEMPTATION_DOSE_GAME_ID],
    )
    def test_the_symmetric_transfer_games_frame_the_counterpart_the_trained_way(
        self, game_id: str
    ) -> None:
        """Transfer must vary the game, and only the game.

        Every trained matrix arm tells the model who it is matched with: the six twin arms say it
        is another instance of this same model, the two vs-frozen arms say it is a system whose
        move is already recorded. These two rendered no such paragraph at all, so a twin-trained
        policy met a transfer prompt that differed in the game AND in whether the counterpart was
        described -- and behaviour keyed on the twin clause would read as failed transfer.
        """
        rows = generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL)
        assert rows
        for row in rows:
            assert TWIN_COUNTERPART_CLAUSE in str(row["prompt"])

    @pytest.mark.parametrize("game_id", EVAL_ONLY_GAME_IDS)
    def test_every_transfer_prompt_describes_its_counterpart(self, game_id: str) -> None:
        """None of the three may be silent about the other side, whatever it is matched with."""
        for row in generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL):
            assert "About the other side:" in str(row["prompt"])


class TestTheTemptationDoseLadder:
    """The eval-only dose instrument: twin-pd over every registered temptation rung.

    Its readout is cooperation against log-temptation, so what gets pinned here is what a slope
    reading rests on: the ladder ascends strictly in dose and extends past both trained rungs,
    every rung is a strict PD, the rung genuinely reaches the rendered prose, and -- the one thing
    the spec called out to get right -- adding rungs cannot reach the trained arms' corpora,
    because the eval path renders every variant an arm carries.
    """

    def test_the_ladder_ascends_strictly_and_extends_past_the_trained_rungs(self) -> None:
        doses = [TEMPTATION_BY_VARIANT[variant] for variant in TEMPTATION_DOSE_PAYOFF_VARIANTS]
        assert doses == sorted(doses)
        assert len(set(doses)) == len(doses)
        trained_doses = [TEMPTATION_BY_VARIANT[variant] for variant in PAYOFF_VARIANTS]
        assert set(PAYOFF_VARIANTS) < set(TEMPTATION_DOSE_PAYOFF_VARIANTS)
        assert min(doses) < min(trained_doses)
        assert max(doses) > max(trained_doses)

    def test_every_rung_is_a_strict_prisoners_dilemma(self) -> None:
        for variant in TEMPTATION_DOSE_PAYOFF_VARIANTS:
            assert_prisoners_dilemma(twin_pd(variant))

    def test_the_trained_pd_corpora_still_carry_exactly_the_two_registered_rungs(self) -> None:
        """Growing the payoff registry must not grow any trained arm's corpus."""
        assert PAYOFF_VARIANTS == ("temptation-2", "temptation-10")
        for game_id in ("twin-pd", "fixed-pie-pd", "pd-unstated", "pd-vs-frozen"):
            for split in SPLITS:
                variants = {
                    row["payoff_variant"]
                    for row in generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=split)
                }
                assert variants == set(PAYOFF_VARIANTS), (game_id, split)

    def test_the_dose_rows_render_every_rung_on_the_held_out_frames(self) -> None:
        rows = generate_prompt_rows(TEMPTATION_DOSE_GAME_ID, GRADING_GROUP_MIX, split=SPLIT_EVAL)
        assert {row["payoff_variant"] for row in rows} == set(TEMPTATION_DOSE_PAYOFF_VARIANTS)
        n_eval_frames = len({row["reskin_id"] for row in rows})
        expected = n_eval_frames * len(TEMPTATION_DOSE_PAYOFF_VARIANTS) * len(COOP_LABEL_INDICES)
        assert len(rows) == expected
        for row in rows:
            assert TWIN_COUNTERPART_CLAUSE in str(row["prompt"])

    def test_each_rung_prints_its_own_outcome_figures(self) -> None:
        """`twin_pd` normalises by the max cell, so rungs must differ in the rendered prose.

        A ladder whose prompts were byte-identical across rungs would sweep six labels over one
        stimulus and read noise as a dose response.
        """
        rows = generate_prompt_rows(TEMPTATION_DOSE_GAME_ID, GRADING_GROUP_MIX, split=SPLIT_EVAL)
        prompts_by_cell: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            cell = (str(row["reskin_id"]), str(row["coop_label"]))
            prompts_by_cell.setdefault(cell, set()).add(str(row["prompt"]))
        for cell, rendered in prompts_by_cell.items():
            assert len(rendered) == len(TEMPTATION_DOSE_PAYOFF_VARIANTS), cell


class TestTheResponderTransferGame:
    """The second-mover item: a division already made, and the only choice is whether it stands.

    It arrived here as a degenerate 2x2 rendered through the simultaneous-move frame like every
    other matrix game, which produced a sheet where the model's own label alone fixed its credit at
    20 or 0 with no proposer, no split and no counterpart share anywhere in the text. That is a
    dominance check reported under a name that invites reading it as unfairness rejection, and
    `games/payoffs.py`'s own docstring asserts the split "is stated in the prompt" -- so these
    tests pin the rendering that makes the docstring true.
    """

    def rows(self) -> list[dict[str, object]]:
        return generate_prompt_rows(
            ULTIMATUM_RESPONDER_GAME_ID, GRADING_GROUP_MIX, split=SPLIT_EVAL
        )

    def test_both_sides_shares_are_stated_on_every_row(self) -> None:
        spec = ultimatum_responder()
        mine = points(spec.payoff_cc)
        theirs = points(1.0 - spec.payoff_cc)
        for row in self.rows():
            prompt = str(row["prompt"])
            other_label = next(
                label for label in (row["label_a"], row["label_b"]) if label != row["coop_label"]
            )
            assert (
                f"- You {row['coop_label']}: you are credited {mine} points, and they are credited {theirs} points."
            ) in prompt
            assert (
                f"- You {other_label}: you are credited 0 points, and they are credited 0 points."
            ) in prompt

    def test_the_sheet_is_not_a_simultaneous_move_table(self) -> None:
        """No four-cell block and no symmetry note: the counterpart has no move left to make."""
        for row in self.rows():
            prompt = str(row["prompt"])
            assert ", they " not in prompt
            assert SYMMETRY_NOTE not in prompt
            assert prompt.count("- You ") == 2

    def test_the_counterpart_is_described_as_having_already_moved(self) -> None:
        for row in self.rows():
            assert f"About the other side: {RESPONDER_COUNTERPART_CLAUSE}" in str(row["prompt"])

    def test_counterbalancing_moves_only_the_numbers(self) -> None:
        spec = ultimatum_responder()
        first, second = (
            render_responder_prompt(spec, RESPONDER_SCENARIOS[0], coop_label_index=index)
            for index in COOP_LABEL_INDICES
        )
        assert first != second
        numbers = re.compile(r"\d[\d.]*")
        assert numbers.sub("#", first) == numbers.sub("#", second)

    def test_a_spec_that_is_not_a_share_of_one_total_is_refused(self) -> None:
        """The counterpart's credit is `1 - mine`, which only holds for the unnormalised share."""
        with pytest.raises(ValueError, match="share of one total"):
            render_responder_prompt(
                twin_pd("temptation-2"), RESPONDER_SCENARIOS[0], coop_label_index=0
            )

    def test_every_responder_frame_is_held_out_of_training(self) -> None:
        assert RESPONDER_SCENARIOS
        assert all(authored.eval_only for authored in RESPONDER_SCENARIOS)

    def test_the_rows_keep_the_corpus_schema_and_one_row_per_orientation(self) -> None:
        rows = self.rows()
        assert len(rows) == len(RESPONDER_SCENARIOS) * len(COOP_LABEL_INDICES)
        for row in rows:
            assert tuple(row.keys()) == ROW_COLUMNS
            assert row["game_id"] == ULTIMATUM_RESPONDER_GAME_ID
            assert row["opp_coop_prob"] == OPP_COOP_PROB_UNSET
            assert str(row["reskin_id"]) not in str(row["prompt"])


class TestLabelPrintOrder:
    """The print-order control: where each label is printed, holding what it pays for fixed.

    The counterbalance cancels a preference for one *label*, and the first baseline sweeps found
    something it cannot cancel: the untrained model picks the first-printed label about 57% of the
    time whatever the payoffs say, in the same direction in every game measured. While `label_a` is
    always printed first, "prefers that position" and "prefers that word" are the same variable, and
    the two imply different fixes. This control separates them by moving the print order while the
    payoff mapping and the cooperative label stay exactly where they were.

    So the property under test is a conjunction: the page moves and the game does not. Every test
    here pins one half of it.
    """

    GAME = "harmony"

    def swapped_rows(
        self, game_id: str = GAME, grading: str = GRADING_GROUP_MIX
    ) -> list[dict[str, object]]:
        return generate_prompt_rows(
            game_id, grading, split=SPLIT_TRAIN, label_print_order=LABEL_PRINT_ORDER_SWAPPED
        )

    def test_the_default_is_the_canonical_order_byte_for_byte(self) -> None:
        """Every existing prompt is untouched, which is what makes the measured corpora comparable.

        Byte-exactness of the canonical rendering itself is pinned by the tests above; this pins
        that the new parameter's default reproduces it rather than a rendering of its own.
        """
        for game_id in GAME_IDS:
            grading = GRADING_BY_GAME[game_id]
            for split in SPLITS:
                assert generate_prompt_rows(game_id, grading, split=split) == generate_prompt_rows(
                    game_id, grading, split=split, label_print_order=LABEL_PRINT_ORDER_CANONICAL
                )

    def test_the_canonical_order_prints_label_a_first(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        prompt = render_game_prompt(spec, scenario, coop_label_index=0)
        assert outcome_lines(prompt)[0].startswith(f"- You {scenario.label_a}")
        assert prompt.index(f"<action>{scenario.label_a}</action>") < prompt.index(
            f"<action>{scenario.label_b}</action>"
        )

    def test_swapping_reverses_the_outcome_table(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        """Exactly reversed, because both loops of the table read the same print order."""
        canonical = outcome_lines(render_game_prompt(spec, scenario, coop_label_index=0))
        swapped = outcome_lines(
            render_game_prompt(
                spec, scenario, coop_label_index=0, label_print_order=LABEL_PRINT_ORDER_SWAPPED
            )
        )
        assert len(canonical) == len(scenario.labels) ** 2
        assert swapped == list(reversed(canonical))

    def test_swapping_offers_the_second_label_first_in_the_answer_instruction(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        swapped = render_game_prompt(
            spec, scenario, coop_label_index=0, label_print_order=LABEL_PRINT_ORDER_SWAPPED
        )
        assert swapped.index(f"<action>{scenario.label_b}</action>") < swapped.index(
            f"<action>{scenario.label_a}</action>"
        )

    @pytest.mark.parametrize("coop_label_index", COOP_LABEL_INDICES)
    def test_swapping_leaves_every_payoff_on_the_label_it_was_already_on(
        self, spec: MatrixGameSpec, scenario: Scenario, coop_label_index: int
    ) -> None:
        """The half of the conjunction that makes the experiment a print-order experiment.

        If a point value followed the position rather than the label, the swapped sweep would be
        measuring a different game and its comparison against the canonical one would be worthless.
        """
        canonical = render_game_prompt(spec, scenario, coop_label_index=coop_label_index)
        swapped = render_game_prompt(
            spec,
            scenario,
            coop_label_index=coop_label_index,
            label_print_order=LABEL_PRINT_ORDER_SWAPPED,
        )
        for mine in scenario.labels:
            for theirs in scenario.labels:
                assert points_shown(swapped, mine, theirs) == points_shown(
                    canonical, mine, theirs
                ), (mine, theirs)

    def test_swapping_does_not_reorder_the_authored_frame(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        """A deliberate limit on the manipulation, pinned here so it cannot be read as a bug.

        The frame is authored prose that names `label_a` first, and in about half the roster its
        second sentence carries a pronoun back to the first ("EASE takes it from the lower bank").
        Reversing those two sentences mechanically yields prose whose pronoun precedes its
        antecedent, which is a second manipulation riding on the one being measured. So the control
        moves what the renderer prints -- the outcome table and the answer instruction -- and leaves
        the frame as written.

        What that costs the experiment: a reversal of the label gap under swapped order is clean
        evidence for position, while an unchanged gap stays ambiguous between the word itself and
        the frame's own introduction order.
        """
        swapped = render_game_prompt(
            spec, scenario, coop_label_index=0, label_print_order=LABEL_PRINT_ORDER_SWAPPED
        )
        assert scenario.frame in swapped
        assert swapped.index(scenario.label_a) < swapped.index(scenario.label_b)

    def test_swapping_leaves_the_symmetry_note_and_counterpart_paragraph_alone(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        swapped = render_game_prompt(
            spec,
            scenario,
            coop_label_index=0,
            twin_framing=True,
            label_print_order=LABEL_PRINT_ORDER_SWAPPED,
        )
        assert SYMMETRY_NOTE in swapped
        assert f"About the other side: {TWIN_COUNTERPART_CLAUSE}" in swapped

    def test_the_counterbalance_still_moves_only_the_numbers_under_the_swapped_order(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        """The property the whole corpus rests on has to survive the new order, not just the old."""
        first, second = (
            render_game_prompt(
                spec, scenario, coop_label_index=index, label_print_order=LABEL_PRINT_ORDER_SWAPPED
            )
            for index in COOP_LABEL_INDICES
        )
        assert first != second
        numbers = re.compile(r"\d[\d.]*")
        assert numbers.sub("#", first) == numbers.sub("#", second)

    def test_every_swapped_prompt_passes_the_vocabulary_guard(self) -> None:
        for game_id in ALL_GAME_IDS:
            if game_id in UNLABELLED_GAME_IDS:
                continue
            grading = GRADING_BY_GAME.get(game_id, GRADING_GROUP_MIX)
            splits = (SPLIT_EVAL,) if game_id in EVAL_ONLY_GAME_IDS else SPLITS
            for split in splits:
                rows = generate_prompt_rows(
                    game_id,
                    grading,
                    split=split,
                    label_print_order=LABEL_PRINT_ORDER_SWAPPED,
                )
                assert rows
                for row in rows:
                    assert_no_loaded_vocabulary(str(row["prompt"]))

    def test_swapped_rows_move_the_prompt_and_nothing_that_defines_the_game(self) -> None:
        canonical = generate_prompt_rows(self.GAME, GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        swapped = self.swapped_rows()
        assert canonical
        for before, after in zip(canonical, swapped, strict=True):
            assert before["label_print_order"] == LABEL_PRINT_ORDER_CANONICAL
            assert after["label_print_order"] == LABEL_PRINT_ORDER_SWAPPED
            assert after["prompt"] != before["prompt"]
            for column in (*PAYOFF_COLUMNS, "label_a", "label_b", "coop_label", "game_id"):
                assert after[column] == before[column], column

    def test_a_swapped_prompt_id_says_so_and_still_ends_in_its_orientation(self) -> None:
        """Two ids that name different prose must not collide, and one existing key must survive.

        A canonical and a swapped sweep of the same game are two different prompt sets, so merging
        their artefacts on `prompt_id` has to keep them apart. The orientation stays the last
        segment because analyses key on it there.
        """
        canonical_ids = [
            str(row["prompt_id"])
            for row in generate_prompt_rows(self.GAME, GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        ]
        swapped_ids = [str(row["prompt_id"]) for row in self.swapped_rows()]
        assert not set(canonical_ids) & set(swapped_ids)
        for before, after in zip(canonical_ids, swapped_ids, strict=True):
            orientation = before.rsplit("--", 1)[-1]
            assert orientation.startswith("coop")
            assert after.endswith(orientation)
            assert LABEL_PRINT_ORDER_SWAPPED in after

    def test_swapped_rows_keep_the_declared_schema(self) -> None:
        for row in self.swapped_rows():
            assert tuple(row.keys()) == ROW_COLUMNS

    def test_an_unknown_print_order_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="label_print_order"):
            generate_prompt_rows(
                self.GAME, GRADING_GROUP_MIX, split=SPLIT_TRAIN, label_print_order="mirrored"
            )

    @pytest.mark.parametrize("game_id", sorted(UNLABELLED_GAME_IDS))
    def test_every_unlabelled_game_refuses_a_swapped_order_rather_than_ignoring_it(
        self, game_id: str
    ) -> None:
        """They print no labels at all, so a swapped build would be canonical rows mislabelled.

        Refusing beats returning them: a corpus whose column says `swapped` and whose prose is
        canonical is exactly the kind of quiet disagreement that gets read as a null result. Asserted
        over the whole set rather than the unilateral split alone, because that is the shape of the
        guard -- it keys on a game having no labels, not on one game id.
        """
        with pytest.raises(ValueError, match="no action labels"):
            generate_prompt_rows(
                game_id,
                GRADING_BY_GAME.get(game_id, GRADING_GROUP_MIX),
                split=SPLIT_EVAL if game_id in EVAL_ONLY_GAME_IDS else SPLIT_TRAIN,
                label_print_order=LABEL_PRINT_ORDER_SWAPPED,
            )

    def test_the_repeated_arm_swaps_its_table_and_its_tag_list(
        self, spec: MatrixGameSpec, scenario: Scenario
    ) -> None:
        rendered = {
            order: render_iterated_prompt(
                spec,
                scenario,
                rule=OpponentRule.TIT_FOR_TAT,
                n_rounds=ITERATED_N_ROUNDS,
                coop_label_index=0,
                label_print_order=order,
            )
            for order in (LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED)
        }
        swapped = rendered[LABEL_PRINT_ORDER_SWAPPED]
        assert outcome_lines(swapped) == list(
            reversed(outcome_lines(rendered[LABEL_PRINT_ORDER_CANONICAL]))
        )
        assert swapped.index(f"<action>{scenario.label_b}</action>") < swapped.index(
            f"<action>{scenario.label_a}</action>"
        )

    def test_the_second_mover_item_swaps_its_two_outcome_lines(self) -> None:
        spec = ultimatum_responder()
        authored = RESPONDER_SCENARIOS[0]
        canonical = outcome_lines(render_responder_prompt(spec, authored, coop_label_index=0))
        swapped = outcome_lines(
            render_responder_prompt(
                spec, authored, coop_label_index=0, label_print_order=LABEL_PRINT_ORDER_SWAPPED
            )
        )
        assert len(canonical) == len(authored.labels)
        assert swapped == list(reversed(canonical))


class TestRewardSpreadReport:
    """The launch-time table of |r_C - r_D| per trainable matrix game, at p in (0.1, 0.5, 0.9).

    With scale_rewards="batch" the spread is proportional to an arm's effective gradient when
    variants share a batch, so the numbers here are checked verbatim against hand arithmetic:
    a formatting or wiring slip would misstate exactly the quantity the report exists to show.
    """

    def test_every_trainable_matrix_game_and_variant_gets_a_line(self) -> None:
        report = prompts.reward_spread_report()
        lines = report.splitlines()
        assert lines[0].split() == ["game/variant", "p=0.1", "p=0.5", "p=0.9"]
        names = {line.split()[0] for line in lines[1:]}
        # 2 twin-pd + 2 fixed-pie + 2 pd-unstated + 2 pd-reskin + 1 public-goods + 4 stag-hunt
        # + 1 each hi-lo/harmony/chicken + 2 pd-vs-frozen + 4 stag-hunt-vs-frozen = 22
        # (game, variant) pairs.
        assert len(lines) == 1 + 22
        assert {f"{game}/standard" for game in ("hi-lo", "harmony", "chicken")} <= names
        assert {f"stag-hunt/{variant}" for variant in STAG_HUNT_VARIANTS} <= names

    def test_the_hand_computed_cells_appear_verbatim(self) -> None:
        """risky-hunt's 0.050 against twin-pd's 0.380 at p=0.9 is the ~8x gradient contrast."""
        report = prompts.reward_spread_report()
        by_name = {line.split()[0]: line.split()[1:] for line in report.splitlines()[1:]}
        assert by_name["twin-pd/temptation-2"] == ["0.220", "0.300", "0.380"]
        assert by_name["stag-hunt/risky-hunt"] == ["0.850", "0.450", "0.050"]
        assert by_name["chicken/standard"] == ["0.200", "0.000", "0.200"]
        assert by_name["harmony/standard"] == ["0.250", "0.250", "0.250"]
        assert by_name["hi-lo/standard"] == ["0.010", "0.450", "0.890"]


class TestStagHuntVariantsReachTheCorpus:
    def test_every_ladder_variant_appears_in_the_rows(self) -> None:
        rows = generate_prompt_rows("stag-hunt", GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        assert {row["payoff_variant"] for row in rows} == set(STAG_HUNT_PAYOFF_VARIANTS)

    def test_the_variant_list_is_derived_from_payoffs_not_restated(self) -> None:
        """A variant added or renamed in games.payoffs must not need an edit here to take effect."""
        assert tuple(STAG_HUNT_VARIANTS) == STAG_HUNT_PAYOFF_VARIANTS

    def test_the_two_variants_carry_different_payoffs_into_the_prompt(self) -> None:
        rows = generate_prompt_rows("stag-hunt", GRADING_GROUP_MIX, split=SPLIT_TRAIN)
        by_variant = {row["payoff_variant"]: row for row in rows}
        assert by_variant["safe-hunt"]["payoff_dd"] != by_variant["risky-hunt"]["payoff_dd"]


class TestTheFrameLabelAuditCoversTheWholeRoster:
    """`label_print_order` leaves the authored prose's own mention order fixed, by design.

    The module docstring says why: the frames' two label sentences cannot be reversed mechanically
    without stranding pronouns ahead of their antecedents across about half the roster. So the print
    control de-aliases position from word identity on two of three surfaces, and the third is a
    residual. `frame_label_audit` measures that residual per frame instead of assuming it away, which
    is only worth anything if it covers every frame the battery renders -- a coverage gap would leave
    some frames silently unexplained rather than visibly unmeasured.
    """

    def _scenarios(self) -> dict[str, prompts.Scenario]:
        """Every two-action frame the battery renders, keyed by the id records carry as reskin_id."""
        return {
            scenario.scenario_id: scenario
            for scenario in (
                *prompts.MATRIX_SCENARIOS,
                *prompts.RESPONDER_SCENARIOS,
                *prompts.RESKIN_SCENARIOS,
            )
        }

    def test_every_two_action_frame_is_audited(self) -> None:
        audit = prompts.frame_label_audit()
        registered = set(self._scenarios())
        assert registered
        assert set(audit) == registered
        assert {entry["eval_only"] for entry in audit.values()} == {True, False}

    def test_runtime_loaded_frames_are_audited_when_they_are_handed_in(self) -> None:
        """`--held-out-extension` appends frames the tracked rosters do not carry.

        Without them in the audit the extension's cells would be the only ones whose position
        residual is unexplained, which is the coverage gap the audit exists to close.
        """
        extra = prompts.Scenario(
            scenario_id="synthetic-audited-bench",
            label_a="QUILL",
            label_b="SLATE",
            eval_only=True,
            frame=(
                "BENCH NOTE -- one run\n\nMark yours QUILL to send the boards on, or SLATE to "
                "hold them at your own end."
            ),
        )
        audited = prompts.frame_label_audit(extra=(extra,))
        assert set(audited) == set(self._scenarios()) | {extra.scenario_id}
        entry = audited[extra.scenario_id]
        assert entry["first_label_in_frame"] == "QUILL"
        assert entry["first_offset_by_label"] == {
            "QUILL": extra.frame.index("QUILL"),
            "SLATE": extra.frame.index("SLATE"),
        }
        assert entry["eval_only"] is True

    def test_no_extra_frames_leaves_the_audit_byte_identical(self) -> None:
        """Every banked trace's audit was written by the no-argument call; it must not move."""
        assert json.dumps(prompts.frame_label_audit(extra=()), sort_keys=True) == json.dumps(
            prompts.frame_label_audit(), sort_keys=True
        )

    def test_each_entry_names_one_of_its_own_two_labels_first(self) -> None:
        for scenario_id, entry in prompts.frame_label_audit().items():
            labels = (entry["label_a"], entry["label_b"])
            offsets = entry["first_offset_by_label"]
            assert entry["first_label_in_frame"] in labels, scenario_id
            assert set(offsets) == set(labels), scenario_id
            assert entry["first_label_in_frame"] == min(labels, key=lambda label: offsets[label])

    def test_the_offsets_are_the_frames_own_first_occurrences(self) -> None:
        by_id = self._scenarios()
        for scenario_id, entry in prompts.frame_label_audit().items():
            frame = by_id[scenario_id].frame
            for label, offset in entry["first_offset_by_label"].items():
                assert frame.index(label) == offset, (scenario_id, label)

    def test_the_audit_reports_whether_the_residual_is_a_constant(self) -> None:
        """Not a threshold: whichever way this reads is the answer the audit exists to give.

        If every frame introduces `label_a` first, the residual is perfectly confounded with the
        canonical print order and the swapped leg IS the frames-disagree condition. If they vary, the
        covariate has within-battery variation and a position effect can be regressed against it.
        """
        audit = prompts.frame_label_audit()
        disagreeing = [
            scenario_id
            for scenario_id, entry in audit.items()
            if entry["first_label_in_frame"] != entry["label_a"]
        ]
        # Recorded rather than asserted either way: every frame today introduces label_a first, so
        # the residual has no within-battery variation and only the swapped leg moves it. A frame
        # authored the other way round turns this red, which is the moment the covariate starts
        # carrying information and the analysis has to condition on it.
        assert disagreeing == []


# Every noun a prompt in this corpus uses for a quantity worked out at the END of a round, in both
# numbers. None of them may name what an answer instruction asks the reader to write.
END_TALLY_NOUNS: tuple[str, ...] = (
    "figure",
    "figures",
    "tally",
    "tallies",
    "total",
    "totals",
    "count",
    "counts",
)


def answer_instruction(prompt: str) -> str:
    """The last paragraph of a rendered prompt, which is where every renderer puts the tag instruction."""
    return prompt.rsplit("\n\n", 1)[-1]


def end_tally_nouns(text: str) -> set[str]:
    """Which end-tally nouns a stretch of prose uses, matched as whole words.

    Whole words rather than substrings, because the fixed instructions say "counted up at the end",
    which contains `count` and is exactly the phrase that tells the reader the tag is NOT the tally.
    """
    lowered = text.lower()
    return {noun for noun in END_TALLY_NOUNS if re.search(rf"\b{noun}\b", lowered)}


def every_rendered_prompt(game_id: str) -> list[str]:
    """Every prompt this game renders, over whichever splits it has."""
    if game_id in EVAL_ONLY_GAME_IDS:
        rows = generate_prompt_rows(game_id, GRADING_GROUP_MIX, split=SPLIT_EVAL)
    else:
        rows = [
            row
            for split in SPLITS
            for row in generate_prompt_rows(game_id, GRADING_BY_GAME[game_id], split=split)
        ]
    return [str(row["prompt"]) for row in rows]


class TestTheAnswerInstructionNeverNamesTheEndTally:
    """The tag instruction's word for what it wants, against the prompt's word for the end tally.

    A hand read of the first one-way-transfer production pass found a model concluding in its reasoning
    that it would keep its whole stock and then writing that stock into a tag asking how much it gave
    away -- its end-of-round tally, not its decision -- because the transfer instruction said "give
    your figure" and the mechanics paragraph called the tally "your figure at the end". The scan and
    the blind judge both scored those replies as giving everything, one-directionally and on most of
    that model's replies, and nothing in the numbers could have shown it.

    The precondition was live in five more games here, so the rule is enforced over the whole corpus
    rather than over the two games the defect was found in: the instruction paragraph may not use any
    word the prompts use for an end-of-round quantity. Games whose prompt does state such a tally also
    say outright that the tag carries the decision being made now.
    """

    GAMES_STATING_AN_END_TALLY: tuple[str, ...] = (
        THRESHOLD_GOODS_GAME_ID,
        TRUST_STATED_RETURN_GAME_ID,
        TRUST_STRATEGY_METHOD_GAME_ID,
        MIN_EFFORT_GAME_ID,
        TRUSTEE_RETURN_GAME_ID,
    )
    """The games whose own mechanics name a tally taken at the end of the round, in that many words.

    Read off the rendered prose rather than assumed: each of these prints "your figure at the end" (or,
    for the level grid, "your figure follows from") beside an instruction asking for one number, which
    is the exact collision the transfer probes were fixed for.
    """

    @pytest.mark.parametrize("game_id", [*GAME_IDS, *EVAL_ONLY_GAME_IDS])
    def test_no_answer_instruction_uses_a_word_its_prompt_uses_for_an_end_tally(
        self, game_id: str
    ) -> None:
        prompts_seen = every_rendered_prompt(game_id)
        assert prompts_seen
        for prompt in prompts_seen:
            instruction = answer_instruction(prompt)
            assert "tag" in instruction, (game_id, instruction)
            assert end_tally_nouns(instruction) == set(), (game_id, instruction)

    @pytest.mark.parametrize("game_id", [*GAME_IDS, *EVAL_ONLY_GAME_IDS])
    def test_the_ban_is_not_vacuous_because_every_prompt_body_uses_one_of_those_nouns(
        self, game_id: str
    ) -> None:
        """The control for the test above, which would pass on a corpus that never says "figure".

        Every game here prints a tally in its mechanics or in at least one of its frames, so the
        collision was one word away in all of them and keeping the noun out of the instruction is a
        live constraint rather than a property the corpus has for free. Per game rather than per
        prompt, because a single frame is free not to mention the tally its mechanics paragraph names.
        """
        used = {
            noun
            for prompt in every_rendered_prompt(game_id)
            for noun in end_tally_nouns("\n\n".join(prompt.split("\n\n")[:-1]))
        }
        assert used, game_id

    @pytest.mark.parametrize("game_id", list(GAMES_STATING_AN_END_TALLY))
    def test_a_game_that_states_an_end_tally_says_the_tag_is_the_decision_made_now(
        self, game_id: str
    ) -> None:
        """The second half of the transfer fix: the tag's meaning restated where it can be confused.

        Keeping the noun out is not enough on its own where the prompt really does state a tally the
        reader could report instead, so those instructions name the moment: the answer is what is being
        decided now, not the quantity that follows at the end.
        """
        for prompt in every_rendered_prompt(game_id):
            instruction = answer_instruction(prompt)
            assert "at the end" in instruction, (game_id, instruction)
            assert ("you are making now" in instruction) or (
                "you are choosing now" in instruction
            ), (game_id, instruction)


class TestRenderMatrixRowsUnderClause:
    """The one path that renders a TRAINING split under a swapped counterpart paragraph.

    Every other generator either renders a game's own paragraph or refuses the training split, and the
    refusal is what used to make "never trained under this framing" a property of the code. It is now a
    property of the run: the renderer stamps the framing on every row, the trainer records which
    framings its corpus carried, and the readout marks each framing trained or held out from that.
    """

    GRADING = GRADING_GROUP_MIX

    def rows(self, framing_id: str, split: str) -> list[dict[str, Any]]:
        return prompts.render_matrix_rows_under_clause(
            "twin-pd",
            self.GRADING,
            clause=prompts.COUNTERPART_FRAMINGS[framing_id],
            framing_label=framing_id,
            split=split,
        )

    def test_a_training_split_renders_and_stamps_the_framing(self) -> None:
        rows = self.rows(prompts.FRAMING_HUMAN, SPLIT_TRAIN)
        assert rows
        assert {row["framing_id"] for row in rows} == {prompts.FRAMING_HUMAN}
        assert all(f"--framing-{prompts.FRAMING_HUMAN}--" in row["prompt_id"] for row in rows)
        # One column beyond the corpus schema and exactly one, because the reward reads it out of
        # TRL's forwarded kwargs rather than as a required column.
        assert all(frozenset(row) == frozenset(ROW_COLUMNS) | {"framing_id"} for row in rows)

    def test_the_training_rows_are_the_unstated_stem_plus_one_paragraph(self) -> None:
        stems = prompts.render_matrix_rows_under_clause(
            "twin-pd",
            self.GRADING,
            clause=None,
            framing_label=prompts.FRAMING_UNSTATED,
            split=SPLIT_TRAIN,
        )
        for framed, stem in zip(self.rows(prompts.FRAMING_HUMAN, SPLIT_TRAIN), stems, strict=True):
            prompts.assert_counterpart_paragraph_is_the_only_insertion(
                stem=stem["prompt"], rendered=framed["prompt"], prompt_id=framed["prompt_id"]
            )

    def test_the_eval_wrapper_delegates_and_still_refuses_a_training_split(self) -> None:
        through_wrapper = prompts.generate_counterpart_clause_prompt_rows(
            "twin-pd",
            self.GRADING,
            clause=prompts.COUNTERPART_FRAMINGS[prompts.FRAMING_HUMAN],
            framing_label=prompts.FRAMING_HUMAN,
            split=SPLIT_EVAL,
        )
        assert through_wrapper == self.rows(prompts.FRAMING_HUMAN, SPLIT_EVAL)
        with pytest.raises(ValueError, match="measurement-only"):
            prompts.generate_counterpart_clause_prompt_rows(
                "twin-pd",
                self.GRADING,
                clause=None,
                framing_label=prompts.FRAMING_UNSTATED,
                split=SPLIT_TRAIN,
            )

    def test_a_coupling_clause_is_refused_in_training_outside_the_twin_framing(self) -> None:
        """A training prompt asserting the counterpart decides as you do pays a different counterpart.

        Every group-mix grading pays a completion against the group's realised mix and reads no
        framing, so the clause would be a claim the reward never honours. The twin clause is the
        standing exception, and on the eval split every coupling clause is the measurement itself.
        """
        with pytest.raises(ValueError, match="decision travels with this side's"):
            prompts.render_matrix_rows_under_clause(
                "twin-pd",
                self.GRADING,
                clause=TWIN_COUNTERPART_CLAUSE,
                framing_label=prompts.FRAMING_HUMAN,
                split=SPLIT_TRAIN,
            )
        assert self.rows(prompts.FRAMING_TWIN, SPLIT_TRAIN)
        assert self.rows(prompts.FRAMING_DIFFERENT_AI_COUPLED, SPLIT_EVAL)

    def test_a_scenario_subset_renders_only_those_frames(self) -> None:
        frames = prompts.matrix_frames_for_split("twin-pd", SPLIT_TRAIN)[:2]
        rows = prompts.render_matrix_rows_under_clause(
            "twin-pd",
            self.GRADING,
            clause=None,
            framing_label=prompts.FRAMING_UNSTATED,
            split=SPLIT_TRAIN,
            scenarios=frames,
        )
        assert {row["reskin_id"] for row in rows} == {frame.scenario_id for frame in frames}
        assert len(rows) == len(frames) * len(PAYOFF_VARIANTS) * len(COOP_LABEL_INDICES)

    def test_a_frame_from_another_roster_is_refused(self) -> None:
        """The failure an id check could not see: another bank's prose stamped with this game's id.

        A breadth corpus draws from two pooled prisoner's-dilemma rosters and renders each frame under
        the game id whose roster it came from, so a mis-routed frame would render fine, say `twin-pd`,
        and quietly falsify the reskin bank's held-out claim.
        """
        stray = prompts.matrix_frames_for_split("pd-reskin", SPLIT_TRAIN)[:1]
        with pytest.raises(ValueError, match="not in 'twin-pd''s 'train' roster"):
            prompts.render_matrix_rows_under_clause(
                "twin-pd",
                self.GRADING,
                clause=None,
                framing_label=prompts.FRAMING_UNSTATED,
                split=SPLIT_TRAIN,
                scenarios=stray,
            )

    def test_a_lookalike_frame_built_elsewhere_is_refused(self) -> None:
        real = prompts.matrix_frames_for_split("twin-pd", SPLIT_TRAIN)[0]
        copy = dataclasses.replace(real)
        with pytest.raises(ValueError, match="not in 'twin-pd''s 'train' roster"):
            prompts.render_matrix_rows_under_clause(
                "twin-pd",
                self.GRADING,
                clause=None,
                framing_label=prompts.FRAMING_UNSTATED,
                split=SPLIT_TRAIN,
                scenarios=(copy,),
            )

    def test_an_empty_subset_is_refused_rather_than_rendering_nothing(self) -> None:
        with pytest.raises(ValueError, match="empty scenario subset"):
            prompts.render_matrix_rows_under_clause(
                "twin-pd",
                self.GRADING,
                clause=None,
                framing_label=prompts.FRAMING_UNSTATED,
                split=SPLIT_TRAIN,
                scenarios=(),
            )
