"""Pin the answer-shape rubric's arithmetic, its strategic emptiness, and its own liveness check.

Offline and CPU-only. Every expected reward is written as the arithmetic that produces it and then
asserted against a literal, both lines on purpose: the first says which components are being credited
and the second catches an algebra slip in the first.

The class that carries the most weight is :class:`TestTheLivenessCheckCanFail`. `format_rubric` runs
`assert_components_are_live` at import, so the happy path is proved by the module loading at all --
which is exactly the shape of reassurance this repo distrusts. These tests hand it deliberately
broken component tables and require each defect to be refused, so the import-time green means
something.

:class:`TestTheRewardIsBlindToTheAction` is the property the whole placebo arm rests on: if any
component could read which of the two labels was chosen, the arm would be grading strategy under
another name and would stop being a control for the arms it exists to control for.
"""

from __future__ import annotations

import ast
import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

import games.format_rubric
from games.format_rubric import (
    COMPONENT_LOWERCASE_TAG_NAME,
    COMPONENT_NAMES,
    COMPONENT_NOTHING_AFTER_TAG,
    COMPONENT_SINGLE_TAG,
    COMPONENT_VERBATIM_LABEL_CASING,
    EXAMPLE_LABEL_A,
    EXAMPLE_LABEL_B,
    FORMAT_COMPONENTS,
    MAX_FORMAT_REWARD,
    FormatComponent,
    assert_components_are_live,
    format_component_scores,
    format_reward,
)
from games.parsing import parse_action
from games.prompts import GAME_IDS, generate_prompt_rows

LABELS = {"label_a": EXAMPLE_LABEL_A, "label_b": EXAMPLE_LABEL_B}

WEIGHT = {component.name: component.weight for component in FORMAT_COMPONENTS}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

PERFECT = f"<action>{EXAMPLE_LABEL_A}</action>"
TWO_TAGS = f"<action>{EXAMPLE_LABEL_A}</action>\n<action>{EXAMPLE_LABEL_A}</action>"
TRAILING_PROSE = f"<action>{EXAMPLE_LABEL_A}</action> so that is settled."
SHOUTED_TAG = f"<ACTION>{EXAMPLE_LABEL_A}</ACTION>"
PADDED_LOWERCASE_LABEL = f"<action> {EXAMPLE_LABEL_A.casefold()} </action>"
NO_TAG_AT_ALL = "I would rather not commit to either."


def replace(name: str, **changes: str) -> tuple[FormatComponent, ...]:
    """Return the live component table with one component's fields overridden."""
    return tuple(
        dataclasses.replace(component, **changes) if component.name == name else component
        for component in FORMAT_COMPONENTS
    )


class TestTheRubricArithmetic:
    """Each expected reward as the sum of the components it credits, then as a literal."""

    def test_an_answer_obeying_the_instruction_scores_the_maximum(self) -> None:
        assert format_reward(PERFECT, **LABELS) == pytest.approx(sum(WEIGHT.values()))
        assert format_reward(PERFECT, **LABELS) == pytest.approx(MAX_FORMAT_REWARD)

    def test_a_second_tag_costs_exactly_the_one_tag_component(self) -> None:
        expected = MAX_FORMAT_REWARD - WEIGHT[COMPONENT_SINGLE_TAG]
        assert format_reward(TWO_TAGS, **LABELS) == pytest.approx(expected)
        assert format_reward(TWO_TAGS, **LABELS) == pytest.approx(0.70)

    def test_prose_after_the_tag_costs_exactly_that_component(self) -> None:
        expected = MAX_FORMAT_REWARD - WEIGHT[COMPONENT_NOTHING_AFTER_TAG]
        assert format_reward(TRAILING_PROSE, **LABELS) == pytest.approx(expected)
        assert format_reward(TRAILING_PROSE, **LABELS) == pytest.approx(0.70)

    def test_tag_casing_and_label_casing_are_separate_debits(self) -> None:
        shouted = MAX_FORMAT_REWARD - WEIGHT[COMPONENT_LOWERCASE_TAG_NAME]
        padded = MAX_FORMAT_REWARD - WEIGHT[COMPONENT_VERBATIM_LABEL_CASING]
        assert format_reward(SHOUTED_TAG, **LABELS) == pytest.approx(shouted)
        assert format_reward(PADDED_LOWERCASE_LABEL, **LABELS) == pytest.approx(padded)
        assert (shouted, padded) == (pytest.approx(0.80), pytest.approx(0.80))

    def test_two_violations_debit_both_components(self) -> None:
        both = f"<action>{EXAMPLE_LABEL_A}</action> and also <action>{EXAMPLE_LABEL_A}</action> yes"
        expected = (
            MAX_FORMAT_REWARD - WEIGHT[COMPONENT_SINGLE_TAG] - WEIGHT[COMPONENT_NOTHING_AFTER_TAG]
        )
        assert format_reward(both, **LABELS) == pytest.approx(expected)
        assert format_reward(both, **LABELS) == pytest.approx(0.40)

    def test_an_answer_with_no_tag_scores_zero_everywhere(self) -> None:
        # It never reaches this rubric in a run -- games.rewards gives it the parse penalty -- but
        # games.format_spread scores raw sweep output, where unanswered completions are a measured
        # fraction rather than an error.
        assert format_component_scores(NO_TAG_AT_ALL, **LABELS) == dict.fromkeys(
            COMPONENT_NAMES, 0.0
        )
        assert format_reward(NO_TAG_AT_ALL, **LABELS) == 0.0

    def test_casing_is_judged_on_the_last_tag_because_parsing_reads_the_last_tag(self) -> None:
        # A tidy first tag must not launder an untidy answer: the reward has to describe the tag the
        # action was taken from, or a model could satisfy the rubric and be scored on another tag.
        drifted_last = f"<action>{EXAMPLE_LABEL_A}</action>\n<Action>{EXAMPLE_LABEL_A}</Action>"
        scores = format_component_scores(drifted_last, **LABELS)
        assert scores[COMPONENT_LOWERCASE_TAG_NAME] == 0.0
        assert scores[COMPONENT_SINGLE_TAG] == 0.0


class TestTheRewardIsBlindToTheAction:
    """The placebo's defining property: no component can read which action was chosen."""

    @pytest.mark.parametrize("shape", [PERFECT, TWO_TAGS, TRAILING_PROSE, SHOUTED_TAG])
    def test_swapping_the_label_leaves_the_reward_unchanged(self, shape: str) -> None:
        swapped = shape.replace(EXAMPLE_LABEL_A, EXAMPLE_LABEL_B)
        assert format_reward(swapped, **LABELS) == format_reward(shape, **LABELS)

    def test_the_swap_really_changed_the_parsed_action(self) -> None:
        """The negative control for the test above: an unchanged string proves nothing."""
        swapped = PERFECT.replace(EXAMPLE_LABEL_A, EXAMPLE_LABEL_B)
        before = parse_action(PERFECT, coop_label=EXAMPLE_LABEL_A, **LABELS)
        after = parse_action(swapped, coop_label=EXAMPLE_LABEL_A, **LABELS)
        assert before is not None
        assert after is not None
        assert before != after

    def test_it_holds_on_every_real_label_pair_the_corpus_renders(self) -> None:
        # Asserted over generated rows rather than the two example words, because the corpus reskins
        # its labels and counterbalances which one cooperates: a component keyed on the coop label
        # would pass against one hand-written pair and fail in production.
        for game_id in GAME_IDS:
            grading = "keep-fraction" if game_id == "dictator" else "group-mix"
            for row in generate_prompt_rows(game_id, grading, split="train"):
                label_a, label_b = str(row["label_a"]), str(row["label_b"])
                if not label_a or not label_b:
                    continue
                labels = {"label_a": label_a, "label_b": label_b}
                chose_a = format_reward(f"<action>{label_a}</action>", **labels)
                chose_b = format_reward(f"<action>{label_b}</action>", **labels)
                assert chose_a == chose_b == pytest.approx(MAX_FORMAT_REWARD), row["prompt_id"]


class TestTheLivenessCheckCanFail:
    """The rubric's negative control, run against deliberately broken component tables.

    Every case here is a defect that would otherwise reach a paid run as a placebo arm with a
    constant term in its reward and a flat curve nobody could distinguish from a real null.
    """

    def test_the_live_table_passes(self) -> None:
        assert_components_are_live(FORMAT_COMPONENTS)

    def test_a_component_nothing_can_violate_is_refused(self) -> None:
        broken = replace(COMPONENT_SINGLE_TAG, violating_example=PERFECT)
        with pytest.raises(ValueError, match="nothing can violate it"):
            assert_components_are_live(broken)

    def test_a_component_nothing_can_satisfy_is_refused(self) -> None:
        broken = replace(COMPONENT_SINGLE_TAG, satisfying_example=TWO_TAGS)
        with pytest.raises(ValueError, match="nothing can reach its maximum"):
            assert_components_are_live(broken)

    def test_an_example_pair_that_moves_two_components_is_refused(self) -> None:
        # The isolation requirement. Without it a separation could be another component's doing and
        # this one could be dead underneath, which is the failure the whole check exists for.
        both = f"<action>{EXAMPLE_LABEL_A}</action> and <action>{EXAMPLE_LABEL_A}</action> yes"
        broken = replace(COMPONENT_SINGLE_TAG, violating_example=both)
        with pytest.raises(ValueError, match=r"move components \[.*\] rather than"):
            assert_components_are_live(broken)

    def test_an_example_that_does_not_parse_is_refused(self) -> None:
        # A violation severe enough to fail parsing means real completions shaped that way take the
        # parse penalty and never exercise the component: dead in practice, live-looking here.
        broken = replace(COMPONENT_NOTHING_AFTER_TAG, violating_example=NO_TAG_AT_ALL)
        with pytest.raises(ValueError, match="does not parse"):
            assert_components_are_live(broken)

    def test_a_rubric_that_does_not_span_the_payoff_scale_is_refused(self) -> None:
        heavier = tuple(
            dataclasses.replace(component, weight=component.weight * 2)
            for component in FORMAT_COMPONENTS
        )
        with pytest.raises(ValueError, match="weights sum to"):
            assert_components_are_live(heavier)

    def test_the_refusals_explain_the_mechanism(self) -> None:
        with pytest.raises(ValueError, match="nothing can violate it") as raised:
            assert_components_are_live(replace(COMPONENT_SINGLE_TAG, violating_example=PERFECT))
        message = str(raised.value)
        assert "constant offset" in message
        assert "gradient" in message


class TestTheLivenessCheckActuallyRunsAtImport:
    """The check has teeth; this is whether anything invokes it, which is a separate question.

    Deleting the module-level `assert_components_are_live()` call left the whole suite green
    (integration mutation testing, 2026-08-21), because every test above calls the function itself.
    A rubric arithmetic change would then be caught at test time rather than at import, which is the
    difference between refusing to load and reserving a card first. Asserted structurally, over the
    module's own source, because an import-time side effect cannot be observed from inside a process
    that has already imported the module.
    """

    def test_the_module_calls_the_liveness_check_at_module_scope(self) -> None:
        module_source = Path(games.format_rubric.__file__).read_text(encoding="utf-8")
        tree = ast.parse(module_source)
        top_level_calls = [
            node.value.func.id
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
        ]
        assert "assert_components_are_live" in top_level_calls

    def test_importing_the_module_with_a_dead_component_fails(self) -> None:
        """The behavioural half: a doctored table really does refuse at import, not just in a test.

        Runs in a subprocess because this process has already imported the module, and patches the
        component table before the module body runs by importing it under a name the patch can reach.
        """
        probe = (
            "import games.format_rubric as rubric;"
            "import dataclasses, importlib, sys;"
            "dead = tuple("
            "  dataclasses.replace(c, violating_example=c.satisfying_example)"
            "  for c in rubric.FORMAT_COMPONENTS"
            ");"
            "rubric.assert_components_are_live(dead)"
        )
        finished = subprocess.run(  # noqa: S603
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        )
        assert finished.returncode != 0
        assert "nothing can violate it" in finished.stderr
