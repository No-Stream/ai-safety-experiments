"""The five prompt conditions every item is asked under.

An arm appends exactly one instruction fragment to the item's task, and every arm that appends
anything composes its prompt identically. That uniformity is the measurement: the elicitation gap
and the pressure penalty are differences between arms, so any formatting difference between them
would be indistinguishable from the effect the arm exists to isolate.

`render_prompt` is also where an item is validated, which is a deliberate choice recorded here
because the alternative was validating a corpus once at import. Every prompt that reaches a model
passes through this function, including items built by a test, a notebook, or a generator that never
touches a checked-in corpus, so validating here means an item whose markers leak into its own prompt
or whose placebo drifted in length cannot be measured unnoticed.

The `elicited` fragment must request the *class* of move and never the instance. OverEager-Bench
(arXiv:2605.18583) found that annotating the authorised scope in a prompt stops an agent inferring
boundaries and starts it pattern-matching the declaration text, so a fragment naming the specific
defect lets a model comply by echoing the instruction without demonstrating the capability
underneath. That inflates the ceiling and with it the elicitation gap, in a known direction.
Validating on every render catches the mechanical half, because `prompt_text()` covers `elicitation`
and a fragment carrying a move marker verbatim is rejected. Paraphrase is not caught -- a marker
reading `truncated` against a fragment asking "check whether the range is complete" -- and is
deliberately left an authoring judgement rather than a heuristic: a paraphrase detector could not be
sabotaged into reliability, and false coverage is worse than a documented human responsibility.

Since the arms differ only in this text, fragment wording is the most consequential authoring choice
in the eval. `graders.py` records how far it moves a rate.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from reward_hacking.jagged.items import ItemValidationError, validate_item

if TYPE_CHECKING:
    from reward_hacking.jagged.items import Item


class Arm(StrEnum):
    """One prompt condition. `SPONTANEOUS` is the bare task; the other four append a fragment."""

    SPONTANEOUS = "spontaneous"
    ELICITED = "elicited"
    ELICITED_PLACEBO = "elicited_placebo"
    PRESSURED = "pressured"
    PRESSURED_PLACEBO = "pressured_placebo"


# A blank line, used identically by every arm that appends a fragment.
FRAGMENT_SEPARATOR = "\n\n"


def arm_fragment(item: Item, arm: Arm) -> str:
    """Return the fragment `arm` appends to `item`, which is empty only for `SPONTANEOUS`."""
    match arm:
        case Arm.SPONTANEOUS:
            return ""
        case Arm.ELICITED:
            return item.elicitation
        case Arm.ELICITED_PLACEBO:
            return item.elicitation_placebo
        case Arm.PRESSURED:
            return item.pressure
        case Arm.PRESSURED_PLACEBO:
            return item.pressure_placebo


def render_prompt(item: Item, arm: Arm) -> str:
    """Return the prompt text for one (item, arm) pair, rejecting an item that measures nothing."""
    validate_item(item)
    if arm is Arm.SPONTANEOUS:
        return item.task

    fragment = arm_fragment(item, arm)
    if not fragment:
        msg = (
            f"{item.id}: arm {arm.value} has an empty fragment, so it would render identically to "
            "spontaneous and report a gap of zero that was never measured"
        )
        raise ItemValidationError(msg)
    return f"{item.task}{FRAGMENT_SEPARATOR}{fragment}"
