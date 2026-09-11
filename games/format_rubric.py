"""The answer-shape rubric the format-only placebo arm is graded on, and its liveness self-check.

The placebo exists because of the reward-vs-data control in OpenAI's beneficial-trait RL work
(arXiv 2606.24014 §4.2): the same training data under a generic reward produced no generalisation,
so the *graded dimension* carried their whole effect. Our `twin-pd-group` / `twin-pd-self` pair
already varies the grading over one corpus; this rubric is the third leg, a reward on the same
prompts that carries no information about the game at all. Wave-1's batteries make that leg
necessary rather than ornamental: format pressure consumed a large share of those arms' gradient,
and parse failure and median completion length fell in every paired leg, so "RL on game-shaped
prompts moves an instrument" and "the graded dimension moves it" are not yet distinguishable.

**Every component is a rule the prompt already states, or a form the prompt already displays.**
`games.prompts.ONE_SHOT_INSTRUCTION` ends "Write exactly one tag, with nothing after it" and prints
its two examples as lowercase `<action>` tags around the labels verbatim. So this reward asks for
nothing the model was not already told, which is what keeps the arm a placebo: a component that
introduced a new demand (be terse, say less) would train a second thing and the arm would stop
being a control for the others. Terseness is the obvious lever if the measured spread turns out too
small to train -- `games.format_spread` measures that from a baseline sweep's own completions rather
than arguing about it -- but it is deliberately not in here.

**Strategic emptiness is a testable property, not a claim.** Nothing below reads which action was
chosen: the label components compare the tag body against `label_a` and `label_b` symmetrically, so
either action can reach the maximum. `games/tests/test_games_format_rubric.py` asserts that
swapping a completion's label for the other one leaves the reward unchanged, which is the property
the whole arm rests on.

The rubric also carries its own negative control. `assert_components_are_live` runs at import and
requires, for each component, a satisfying example that scores it 1.0 and a violating example that
scores it 0.0 while leaving every *other* component alone -- and requires both examples to parse,
since a component whose violation is severe enough to fail parsing can never be observed (that
completion takes the parse penalty and the rubric never runs on it). A component that drifted into
being unreachable, or always true, is exactly the dead-metric shape this repo keeps rediscovering,
and it would show up as a placebo arm with no gradient and a plausible flat curve.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from games.parsing import parse_action

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Captures the tag names as written, which `games.parsing._ACTION_RE` deliberately does not: parsing
# is case-insensitive so a model that drifts still gets scored on its choice, and that leniency is
# what leaves casing available as a format axis to pay for here.
_ACTION_TAG_RE = re.compile(
    r"<(?P<open>action)>(?P<body>.*?)</(?P<close>action)>", re.IGNORECASE | re.DOTALL
)

CANONICAL_TAG_NAME = "action"

COMPONENT_SINGLE_TAG = "single_action_tag"
COMPONENT_NOTHING_AFTER_TAG = "nothing_after_the_tag"
COMPONENT_LOWERCASE_TAG_NAME = "lowercase_tag_name"
COMPONENT_VERBATIM_LABEL_CASING = "verbatim_label_casing"


@dataclass(frozen=True)
class FormatComponent:
    """One graded answer-shape property: its weight, why it is gradeable, and its two examples.

    The examples are not documentation. `assert_components_are_live` scores both of them and
    requires this component to separate them while the other three stay put, so a component that
    stopped being reachable or stopped being violable fails at import rather than producing a
    placebo arm with a silently dead term in its reward.
    """

    name: str
    weight: float
    why: str
    satisfying_example: str
    violating_example: str


# The labels the examples are written against. Real corpus vocabulary (`games.prompts` prints
# uppercase reskin words such as these), so the examples exercise the same casing question a real
# completion does, and neutral enough to carry no strategic reading.
EXAMPLE_LABEL_A = "SIDE"
EXAMPLE_LABEL_B = "LOOP"

# The two components the prompt states as explicit rules carry more weight than the two it only
# demonstrates by example. That ordering is the whole justification for the numbers: a model that
# disobeys a written instruction has drifted further from the asked-for shape than one that
# reproduces the shape with different casing, and weighting them equally would say otherwise.
_STATED_RULE_WEIGHT = 0.30
_DISPLAYED_FORM_WEIGHT = 0.20

FORMAT_COMPONENTS: tuple[FormatComponent, ...] = (
    FormatComponent(
        name=COMPONENT_SINGLE_TAG,
        weight=_STATED_RULE_WEIGHT,
        why="the prompt says 'Write exactly one tag'",
        satisfying_example=f"<action>{EXAMPLE_LABEL_A}</action>",
        violating_example=f"<action>{EXAMPLE_LABEL_A}</action>\n<action>{EXAMPLE_LABEL_A}</action>",
    ),
    FormatComponent(
        name=COMPONENT_NOTHING_AFTER_TAG,
        weight=_STATED_RULE_WEIGHT,
        why="the prompt says 'with nothing after it'",
        satisfying_example=f"<action>{EXAMPLE_LABEL_A}</action>",
        violating_example=f"<action>{EXAMPLE_LABEL_A}</action> That is my choice.",
    ),
    FormatComponent(
        name=COMPONENT_LOWERCASE_TAG_NAME,
        weight=_DISPLAYED_FORM_WEIGHT,
        why="the prompt prints the tag lowercase, and parsing is case-insensitive so drift is free",
        satisfying_example=f"<action>{EXAMPLE_LABEL_A}</action>",
        violating_example=f"<Action>{EXAMPLE_LABEL_A}</Action>",
    ),
    FormatComponent(
        name=COMPONENT_VERBATIM_LABEL_CASING,
        weight=_DISPLAYED_FORM_WEIGHT,
        why="the prompt prints the label verbatim, and parsing strips and casefolds it",
        satisfying_example=f"<action>{EXAMPLE_LABEL_A}</action>",
        violating_example=f"<action> {EXAMPLE_LABEL_A.casefold()} </action>",
    ),
)

COMPONENT_NAMES: tuple[str, ...] = tuple(component.name for component in FORMAT_COMPONENTS)

# The rubric spans [0, 1] like every game's payoffs (`games.payoffs`), so the parse penalty stays
# commensurable across arms: an unparseable completion is worse than any answer shape, in this arm
# exactly as in the graded ones.
MAX_FORMAT_REWARD = 1.0
_WEIGHT_SUM_TOLERANCE = 1e-9


def format_component_scores(visible_text: str, *, label_a: str, label_b: str) -> dict[str, float]:
    """Score each answer-shape component of one visible answer, each in {0.0, 1.0}.

    Nothing here reads which of the two labels was chosen: `label_a` and `label_b` enter only as an
    unordered pair to compare the tag body against, so both actions can reach every component's
    maximum. That is the property the placebo rests on and the reason the labels are not a single
    "chosen label" argument, which would make an action-dependent component easy to write by
    accident.

    A visible answer carrying no tag at all scores zero everywhere. That case never reaches a reward
    in practice -- `games.rewards` gives it the parse penalty instead -- but returning zeros rather
    than raising keeps this callable usable on raw sweep output, where unparseable completions are a
    measured fraction rather than an error (`games.format_spread`).
    """
    tags = list(_ACTION_TAG_RE.finditer(visible_text))
    if not tags:
        return dict.fromkeys(COMPONENT_NAMES, 0.0)
    last = tags[-1]
    return {
        COMPONENT_SINGLE_TAG: float(len(tags) == 1),
        COMPONENT_NOTHING_AFTER_TAG: float(not visible_text[last.end() :].strip()),
        COMPONENT_LOWERCASE_TAG_NAME: float(
            last.group("open") == CANONICAL_TAG_NAME and last.group("close") == CANONICAL_TAG_NAME
        ),
        COMPONENT_VERBATIM_LABEL_CASING: float(last.group("body") in (label_a, label_b)),
    }


def format_reward(visible_text: str, *, label_a: str, label_b: str) -> float:
    """Return the weighted answer-shape reward in [0, 1] for one visible answer."""
    scores = format_component_scores(visible_text, label_a=label_a, label_b=label_b)
    return sum(component.weight * scores[component.name] for component in FORMAT_COMPONENTS)


def _assert_weights_span_the_payoff_scale(components: Sequence[FormatComponent]) -> None:
    """Refuse a rubric whose maximum is not 1.0, which would break parse-penalty commensurability."""
    total = sum(component.weight for component in components)
    if abs(total - MAX_FORMAT_REWARD) > _WEIGHT_SUM_TOLERANCE:
        raise ValueError(
            f"format rubric weights sum to {total}, not {MAX_FORMAT_REWARD}. The rubric shares the "
            f"[0, 1] scale every game's payoffs use so that one parse penalty stays comparable "
            f"across arms; a rubric with a different ceiling silently reweights the format-only "
            f"arm's parse pressure against every other arm's."
        )


def _assert_component_is_reachable_and_violable(component: FormatComponent) -> None:
    """Require this component's two examples to separate it, and only it, and both to parse.

    Three failures this catches, all of which would otherwise show up as a placebo arm with a
    plausible flat curve after a paid run: a component nothing can satisfy (always 0, so it is a
    constant offset), a component nothing can violate (always 1, same), and a component whose
    violating example is bad enough to fail parsing -- in which case a real completion violating it
    that way takes the parse penalty and this term never fires at all.

    The isolation requirement is what makes the first two checks trustworthy. Without it a
    satisfying example could be scoring 1.0 because of some other component's behaviour, and the
    pair would look like a working separation while this component sat dead underneath.
    """
    labels = {"label_a": EXAMPLE_LABEL_A, "label_b": EXAMPLE_LABEL_B}
    for role, example in (
        ("satisfying", component.satisfying_example),
        ("violating", component.violating_example),
    ):
        if parse_action(example, coop_label=EXAMPLE_LABEL_A, **labels) is None:
            raise ValueError(
                f"format component {component.name!r} has a {role} example that does not parse "
                f"into an action ({example!r}). `games.rewards` gives an unparseable completion the "
                f"parse penalty and never reaches this rubric, so a real completion shaped like "
                f"this example would never exercise the component: it would be dead in practice "
                f"while looking live here."
            )
    satisfied = format_component_scores(component.satisfying_example, **labels)
    violated = format_component_scores(component.violating_example, **labels)
    if satisfied[component.name] != 1.0:
        raise ValueError(
            f"format component {component.name!r} scores "
            f"{satisfied[component.name]} on its own satisfying example {component.satisfying_example!r}, "
            f"so nothing can reach its maximum and the term is a constant offset rather than a "
            f"gradient. It is graded because {component.why}."
        )
    if violated[component.name] != 0.0:
        raise ValueError(
            f"format component {component.name!r} scores "
            f"{violated[component.name]} on its own violating example {component.violating_example!r}, "
            f"so nothing can violate it and the term is a constant offset rather than a gradient. "
            f"It is graded because {component.why}."
        )
    moved = sorted(name for name in COMPONENT_NAMES if satisfied[name] != violated[name])
    if moved != [component.name]:
        raise ValueError(
            f"format component {component.name!r}'s examples move components {moved} rather than "
            f"only itself. Its separation is therefore not attributable to this component, so the "
            f"reachable/violable check above proves nothing about it: a dead term could hide behind "
            f"a sibling that happens to differ between the two examples."
        )


def assert_components_are_live(
    components: Sequence[FormatComponent] = FORMAT_COMPONENTS,
) -> None:
    """Run the rubric's own negative control: every component reachable, violable and isolated.

    Takes the components rather than reading the module constant, in the shape
    `games.arms.validate_arms` already uses: a check that can only ever be run against the one live
    value cannot be shown to fail, and a check nobody has watched fail is not yet a check.
    """
    _assert_weights_span_the_payoff_scale(components)
    for component in components:
        _assert_component_is_reachable_and_violable(component)


assert_components_are_live()
