"""The closed arm vocabulary: one rendered prompt per arm, along an elicitation spectrum.

An arm is a framing of the same base question. Supplying flawed prior working is *negative*
elicitation, a bare prompt is neutral, and an explicit request to scrutinise the working is
positive; the pressure arms sit off to one side as a side condition rather than on the spectrum.

Unlike the v1 ``jagged`` arms, an arm here does not render itself. The prompt for every arm is
authored per item and stored on the item, because the flawed working, the correct working and the
irrelevant working are all item-specific prose that no shared template can produce. This enum is
therefore the *vocabulary and the order*, not the renderer: it closes the set of names an item may
use, and it fixes a deterministic iteration order so that two runs of the same corpus render cells
in the same sequence (the batch path this will run on compares digests over that sequence, so an
order that came from a JSON file's key order would read as a corpus edit; that path is not built
yet).
"""

from __future__ import annotations

from enum import StrEnum


class Arm(StrEnum):
    """The arms an item may define, in canonical run order.

    The order is controls first, then the corrupted core, then elicitation, then the pressure side
    condition -- so a truncated run still holds the arms the headline readouts need.

    * ``BARE`` -- the question alone. The capability covariate: what the model does unaided.
    * ``CORRECT_WORKING`` -- correct prior working supplied. The deference comparator, which says
      how much of any carry effect is "follows supplied working" rather than "follows a flaw".
    * ``CORRUPTED`` -- flawed working supplied, nothing hinting an error exists. The headline arm.
    * ``CORRUPTED_CONTINUE`` -- flawed working supplied and framed as work to continue rather than
      to check, which is the same corruption under a stronger instruction to build on it.
    * ``CORRUPTED_IRRELEVANT`` -- working that is neither correct nor the planted flaw. Controls for
      the mere presence of a wall of supplied prose.
    * ``CORRUPTED_ELICITED`` -- flawed working plus a class-level request to scrutinise it. Positive
      elicitation, and the arm that says whether the failure is one of noticing or of willingness.
    * ``CORRUPTED_ELICITED_PLACEBO`` -- flawed working plus an instruction of matched length that
      asks for something irrelevant, so the elicitation arm is not measuring prompt length.
    * ``CORRUPTED_PRESSURED`` -- flawed working under realistic deadline/scope framing ("I just need
      the final number for the 9am meeting"). A side condition, not a headline.
    * ``CORRUPTED_PRESSURED_RUBRIC`` -- the explicit-rubric pressure wording, kept on a subset for
      comparability with the v1 result and the literature.
    """

    BARE = "bare"
    CORRECT_WORKING = "correct_working"
    CORRUPTED = "corrupted"
    CORRUPTED_CONTINUE = "corrupted_continue"
    CORRUPTED_IRRELEVANT = "corrupted_irrelevant"
    CORRUPTED_ELICITED = "corrupted_elicited"
    CORRUPTED_ELICITED_PLACEBO = "corrupted_elicited_placebo"
    CORRUPTED_PRESSURED = "corrupted_pressured"
    CORRUPTED_PRESSURED_RUBRIC = "corrupted_pressured_rubric"


# Carry is P(flawed | CORRUPTED) and accuracy is read against BARE: without both, no headline.
REQUIRED_ARMS: frozenset[Arm] = frozenset({Arm.BARE, Arm.CORRUPTED})

# Every arm that supplies the planted flaw, so an analysis can pool them without restating the list.
CORRUPTED_ARMS: frozenset[Arm] = frozenset(
    {
        Arm.CORRUPTED,
        Arm.CORRUPTED_CONTINUE,
        Arm.CORRUPTED_ELICITED,
        Arm.CORRUPTED_ELICITED_PLACEBO,
        Arm.CORRUPTED_PRESSURED,
        Arm.CORRUPTED_PRESSURED_RUBRIC,
    }
)
