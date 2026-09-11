"""Synthetic stimulus material for the fingerprint and correlation-dose passes (v3 transfer file).

The authored board tasks, the lead-in, the paraphrase instruction, the record-sentence template and every
generated board message are stimulus that will run against future models, so they live only in the
gitignored file the loader reads at runtime. What is here is built to satisfy exactly the constraints the
loader checks and nothing more, which is most of what makes it read unlike ordinary prose:

- No noun any synthetic scenario prints (:data:`~sociology.tests.conftest.SYNTHETIC_TRANSFER_SCENARIO_NOUNS`
  supplies eight frames' worth), because a board message or a task naming one reads as being about that
  frame while its cell id says the task was unrelated to the transfer.
- No model or lab name, no design numeral as a whole token, neither of the transfer's own verbs, no loaded
  vocabulary, no blank line, no brace, nothing shaped like a tag.
- The record template is ONE sentence over ``{matched}`` and ``{rounds}`` whose expansions differ in the
  stated numeral and in nothing else, which is the property the dose curve's every step rests on.

The boards here are the shape a real generation produces, provenance included: three messages per board,
the two crossed boards rewriting the raw board of the other side message for message and naming each
source message's digest. That is what lets the offline tests exercise the loader's cross-board checks
without a model call.
"""

from __future__ import annotations

from string import ascii_uppercase
from typing import TYPE_CHECKING, Any

from reward_hacking.model_backend import DEFAULT_BEDROCK_MAX_TOKENS
from sociology.transfer_stimulus import (
    BOARD_CONTENT_SIDE,
    BOARD_IDS,
    BOARD_LEAD_IN_FIELD,
    BOARD_MESSAGE_COUNT,
    BOARD_MODEL_ID_BY_SIDE,
    BOARD_SIDES,
    BOARD_SOURCE_BOARD,
    BOARD_SOURCE_DIGEST_FIELD,
    BOARD_WORDING_SIDE,
    BOARDS_FIELD,
    PARAPHRASE_INSTRUCTION_FIELD,
    RAW_BOARD_IDS,
    TASK_FIELD,
    TASK_GLOSS_FIELD,
    TRACK_RECORD_ROUNDS,
    TRACK_RECORD_ROUNDS_FIELD,
    TRACK_RECORD_TEMPLATE_FIELD,
    UNRELATED_TASKS_FIELD,
    VERDICT_KEYS_ONE_WAY,
    VERDICT_KEYS_TWIN,
    message_digest,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

TWIN_ONLY_BOOL_FLAGS: tuple[str, ...] = tuple(
    key
    for key in VERDICT_KEYS_TWIN
    if key not in VERDICT_KEYS_ONE_WAY and key != "counterpart_reasoning"
)
"""The twin rubric's own booleans, derived rather than listed so a fifth pass needs no edit here.

The loader compares every registered key against every validation reply and the judge module refuses a
set that never exercises one of them both ways, so a flag added to the twin schema and not to the
synthetic replies would fail every transfer test with a missing-expectation error rather than a useful one.
"""

SYNTHETIC_BOARD_LEAD_IN = (
    "SYNTHETIC-BOARD-LEAD-IN: earlier tonight the {count} {beneficiary_noun} were asked, one at a "
    "time and out of sight of one another, to {task_gloss}. What came back is copied out below in "
    "the order it arrived, and nothing else about them is known here."
)
"""The board's own lead-in: what the others were asked, and that their replies follow. No count word
(the renderer prints ``{count}``), no scenario noun, and nothing about how anybody decides."""

SYNTHETIC_PARAPHRASE_INSTRUCTION = (
    "SYNTHETIC-PARAPHRASE-INSTRUCTION: rewrite the reply below in your own words, keeping every "
    "point it makes and their order, changing the wording throughout. Reply with the rewritten "
    "text only.\n\n{message}"
)
"""The prompt the paraphrasing side gets. Generation-side text, and prompt-affecting all the same,
because it decides the wording of two of the four boards."""

SYNTHETIC_TASK_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("rinse and refit a mesh filter", "rinsing and refitting a mesh filter"),
    ("free a stiff window latch", "freeing a stiff window latch"),
    ("clean and oil a bicycle chain", "cleaning and oiling a bicycle chain"),
    ("coil a garden hose so it does not kink", "coiling a garden hose so it does not kink"),
    ("quieten a squeaking door hinge", "quietening a squeaking door hinge"),
    ("sharpen a blunt pencil by hand", "sharpening a blunt pencil by hand"),
    ("thread a long lace through a boot", "threading a long lace through a boot"),
    ("descale a kettle with vinegar", "descaling a kettle with vinegar"),
)
"""(what the task asks, how the lead-in glosses it) for the eight synthetic frames.

Eight everyday procedures, one per scenario, so the eight units of a board cell carry eight independent
board draws rather than one board read eight times -- and none of them shares a noun with any frame.
"""

_SYNTHETIC_MESSAGE_BODIES: tuple[str, ...] = (
    (
        "start by running warm water through it until the water comes out clear, then leave it "
        "upright to drip while you wipe the housing dry"
    ),
    (
        "hold it steady with one hand, work from the near end to the far end, and stop the moment "
        "it starts to feel tight rather than forcing it"
    ),
    (
        "the part people skip is drying it properly afterwards, which is what stops the whole job "
        "needing doing again the following week"
    ),
)
"""Three plain replies to an everyday procedure, one per board position. Written to clear every board
gate at once: no lab name, no frame noun, no design numeral, no transfer verb, no loaded vocabulary."""

_SYNTHETIC_REWRITE_BODIES: tuple[str, ...] = (
    (
        "begin with warm water through the thing until it runs clean, and stand it on end to drain "
        "while the outside gets a dry cloth"
    ),
    (
        "keep a firm grip, move along it from the closer side outward, and ease off as soon as "
        "there is resistance instead of pushing on"
    ),
    (
        "what most people leave out is a proper dry at the finish, and that omission is why the job "
        "comes round again a week later"
    ),
)
"""The same three replies as one side would rewrite the other's: the same points in other words.

Parallel to the bodies above position for position, because a crossed board rewrites its source board
message for message and the loader checks that identity through the recorded source digests.
"""

SYNTHETIC_TRACK_RECORD_TEMPLATE = (
    " SYNTHETIC-RECORD-TEMPLATE: on {matched} of the {rounds} earlier nights this was run, what the "
    "others put down came out the same as what the side holding your {note_noun} put down."
)
"""One sentence over the two counts, leading with a space because it is appended to a full stop.

Count-neutral ("the others" rather than "each of them"), so the same sentence is appended at every dose;
its expansions differ in the numeral alone, which is what the loader checks pairwise across the ladder.
"""


def synthetic_board_message(*, board_id: str, ordinal: int, index: int) -> str:
    """Build one synthetic board message: whose wording it is, which frame, which position.

    The side, the frame and the position are all named by LETTER rather than by their own ids, because a
    side token IS a model name, a synthetic scenario id carries that frame's own beneficiary noun, and a
    digit would be one of the design's registered numerals. All three refusals are the board gate working
    rather than a nuisance: a real message may name none of the three either.
    """
    bodies = _SYNTHETIC_MESSAGE_BODIES if board_id in RAW_BOARD_IDS else _SYNTHETIC_REWRITE_BODIES
    wording = ascii_uppercase[BOARD_SIDES.index(BOARD_WORDING_SIDE[board_id])]
    return (
        f"SYNTHETIC-BOARD-MESSAGE-SIDE-{wording}-FRAME-{ascii_uppercase[ordinal]}"
        f"-POSITION-{ascii_uppercase[index]}: {bodies[index]}."
    )


def _synthetic_provenance(
    *, board_id: str, scenario_id: str, index: int, source_message: str | None
) -> dict[str, Any]:
    """Record one synthetic message's origin in the shape the loader checks field by field."""
    entry: dict[str, Any] = {
        "source_model_id": BOARD_MODEL_ID_BY_SIDE[BOARD_CONTENT_SIDE[board_id]],
        "paraphraser_model_id": (
            None
            if board_id in RAW_BOARD_IDS
            else BOARD_MODEL_ID_BY_SIDE[BOARD_WORDING_SIDE[board_id]]
        ),
        "task_scenario_id": scenario_id,
        "draw": index,
        "transport": "live",
        "redraws": 0,
        "generated_at": "2026-09-04T00:00:00+00:00",
        "code_sha": "synthetic",
        "sampling": {"max_tokens": DEFAULT_BEDROCK_MAX_TOKENS},
    }
    if source_message is not None:
        entry[BOARD_SOURCE_DIGEST_FIELD] = message_digest(source_message)
    return entry


def synthetic_board(*, board_id: str, scenario_id: str, ordinal: int) -> dict[str, Any]:
    """Build one whole synthetic board: its three messages and their provenance, in draw order."""
    messages = [
        synthetic_board_message(board_id=board_id, ordinal=ordinal, index=index)
        for index in range(BOARD_MESSAGE_COUNT)
    ]
    sources = (
        [None] * BOARD_MESSAGE_COUNT
        if board_id in RAW_BOARD_IDS
        else [
            synthetic_board_message(
                board_id=BOARD_SOURCE_BOARD[board_id], ordinal=ordinal, index=index
            )
            for index in range(BOARD_MESSAGE_COUNT)
        ]
    )
    return {
        "messages": messages,
        "provenance": [
            _synthetic_provenance(
                board_id=board_id,
                scenario_id=scenario_id,
                index=index,
                source_message=sources[index],
            )
            for index in range(BOARD_MESSAGE_COUNT)
        ],
    }


def synthetic_boards(scenarios: Sequence[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Build the whole board table: four boards for each of the eight synthetic scenarios."""
    return {
        str(scenario["scenario_id"]): {
            board_id: synthetic_board(
                board_id=board_id, scenario_id=str(scenario["scenario_id"]), ordinal=ordinal
            )
            for board_id in BOARD_IDS
        }
        for ordinal, scenario in enumerate(scenarios)
    }


def synthetic_unrelated_tasks(scenarios: Sequence[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Build the eight board tasks, one per scenario, each stating its own length cap in words."""
    return {
        str(scenario["scenario_id"]): {
            TASK_FIELD: (
                f"SYNTHETIC-TASK-FRAME-{ascii_uppercase[ordinal]}: in no more than sixty words, "
                f"and with no lists, say how you would {asked}."
            ),
            TASK_GLOSS_FIELD: gloss,
        }
        for ordinal, (scenario, (asked, gloss)) in enumerate(
            zip(scenarios, SYNTHETIC_TASK_SUBJECTS, strict=True)
        )
    }


def synthetic_v3_fields(scenarios: Sequence[dict[str, str]]) -> dict[str, Any]:
    """Build every top-level field the v3 transfer stimulus added, over these scenarios' ids.

    Taken as an argument rather than imported from ``conftest``, which imports this module: the tasks and
    the board messages have to avoid every noun these frames print, so the frames are what decides them.
    """
    return {
        BOARD_LEAD_IN_FIELD: SYNTHETIC_BOARD_LEAD_IN,
        UNRELATED_TASKS_FIELD: synthetic_unrelated_tasks(scenarios),
        PARAPHRASE_INSTRUCTION_FIELD: SYNTHETIC_PARAPHRASE_INSTRUCTION,
        BOARDS_FIELD: synthetic_boards(scenarios),
        TRACK_RECORD_TEMPLATE_FIELD: SYNTHETIC_TRACK_RECORD_TEMPLATE,
        TRACK_RECORD_ROUNDS_FIELD: TRACK_RECORD_ROUNDS,
    }
