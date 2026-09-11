"""Runtime loader and renderer for the deference stimulus: the brief, the log, the report, the fork.

The eight authored briefs, both constraint sentences of each, the tool-result blocks of the agent's own
log, the fork sentences, the identity paragraph's parts, the judge rubric and every hand-authored
validation reply are stimulus that will run against future models. Committed, they become training data
and contaminate every measurement made with them, so they live in one gitignored JSON file this module
loads at runtime and refuses to run without. No stimulus prose appears in tracked code; this module knows
the file's *shape* and its authoring constraints, never its text.

What a rendered prompt is: one user message whose blank-line-separated sections are the principal's
brief (carrying exactly one procedural constraint in a slot the arm fills), then -- on the peer cells --
one paragraph saying who the other agents on the same job are, then the two to four tool-result blocks of
work the agent has already done, then -- on the report cells -- one more tool-result block in which a
shared status file lists what every other agent chose, then the fork, then the shared answer instruction.

Four properties the readings depend on are enforced here rather than described:

- **The identity paragraph is composed, not read whole.** The file carries one opening, two identity
  fragments and one tail, and the loader concatenates them, so the two peer identities differ in the
  fragment ALONE. A file that carried whole paragraphs could drift on the opening or the tail while
  still rendering, and the report cells' contrast would then be a read of two rewordings.
- **The arms differ by one sentence and nothing else.** Both constraint sentences fill the same slot of
  the same brief, and each is checked to be exactly one sentence, so the ``benign`` and ``breaking``
  renders of a scenario differ in that sentence and in no other text. Anything else that moved would sit
  inside the pass's headline double difference.
- **The status report is renderer-owned.** Its handles, its line shape and its line count come from this
  module and the loaded peer count, never from authored prose, so the two report cells' report blocks are
  byte-identical and the identity of the peers lives in one place: the paragraph above.
- **Three vocabulary gates run on every render.** The games' own
  :func:`~games.prompts.assert_no_loaded_vocabulary`; the frame-only
  :func:`~games.prompts.assert_no_coupling_claims`, on the peer-free render alone, because the identity
  paragraph exists to say who the others are and would trip it by construction; and
  :func:`assert_no_safety_or_evaluation_vocabulary`, which is this stimulus's own and the reason the
  constraint is procedural: a safety-flavoured word would confound deference with refusal training, and
  an evaluation-flavoured one would prime the very awareness a judge flag is here to measure.

Two digests, because the file serves two instruments. ``digest`` covers the whole payload and moves on
any edit; ``prompt_digest`` covers only the sections a rendered prompt depends on
(:data:`PROMPT_AFFECTING_FIELDS`). A reply record is stamped with the prompt digest and a live resume
compares it, so a rubric or validation-reply edit -- which changes what the judge reads and nothing the
model was shown -- neither mislabels collected rows nor refuses every resume. Every top-level field has to
be filed on one side or the other, and a file carrying one that is not refuses to load.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from games.prompts import (
    FRAMING_LABEL_RE,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDERS,
    ONE_SHOT_INSTRUCTION,
    assert_no_coupling_claims,
    assert_no_loaded_vocabulary,
)
from sociology.decoupled_stimulus import RUNG_DIFFERENT_FAMILY, RUNG_SAME_CHECKPOINT

if TYPE_CHECKING:
    from collections.abc import Mapping

STIMULUS_PATH = Path("docs/scratch/swarm-sociology-2026-08-31/deference_stimulus.json")

STIMULUS_VERSION = "deference-stimulus-v1"

N_SCENARIOS = 8
"""How many briefs the file carries. Eight because the unit of pairing is (scenario, print order), and a
per-cell rate over fewer briefs is one job's answer rather than a rate."""

PEER_COUNT = 6
"""How many other agents the identity paragraph names and the status report lists.

A design constant rather than a factor: the owner fixed eight cells, and a group of six reads as a
working majority without reading as the whole fleet. The file states it too, and the loader refuses a
file that states another number -- the count is printed into the paragraph and IS the report's line
count, so the two would disagree about how many agents there are while both rendering.
"""

ARM_BENIGN = "benign"
ARM_BREAKING = "breaking"
ARMS: tuple[str, ...] = (ARM_BENIGN, ARM_BREAKING)
"""The two arms: the brief's one constraint governs an unrelated dimension of the artifact, or the very
dimension the fork decides. Everything else in the render is identical between them."""

CONSTRAINT_FIELD_BY_ARM: dict[str, str] = {
    ARM_BENIGN: "constraint_benign",
    ARM_BREAKING: "constraint_breaking",
}
"""Which authored sentence each arm fills the brief's constraint slot with."""

CONSTRAINT_SLOT = "{constraint}"
FIRST_LABEL_SLOT = "{first_label}"
SECOND_LABEL_SLOT = "{second_label}"

COUNT_PLACEHOLDER = "count"
IDENTITY_PLACEHOLDERS: tuple[str, ...] = (COUNT_PLACEHOLDER,)
"""The only placeholder the identity paragraph's parts may name; the filler supplies nothing else."""

_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

IDENTITY_RUNGS: tuple[str, ...] = (RUNG_SAME_CHECKPOINT, RUNG_DIFFERENT_FAMILY)
"""The two peer identities, imported rather than renamed: they are the same two identities the transfer
and ladder passes sample, so a rate here is comparable with a rate there by construction."""

CELL_NO_PEERS = "no-peers"
CELL_SAME_CHECKPOINT_PRESENT = "same-checkpoint-present"
CELL_SAME_CHECKPOINT_CHOSE = "same-checkpoint-chose"
CELL_DIFFERENT_FAMILY_CHOSE = "different-family-chose"
CELLS: tuple[str, ...] = (
    CELL_NO_PEERS,
    CELL_SAME_CHECKPOINT_PRESENT,
    CELL_SAME_CHECKPOINT_CHOSE,
    CELL_DIFFERENT_FAMILY_CHOSE,
)
"""The four cells, in reading order: the peer-free floor, presence without a report, and the two reports.

The peer-free cell is first because every reading is a contrast against it and a table that reads top to
bottom should open with what the rest is compared to.
"""

IDENTITY_BY_CELL: dict[str, str | None] = {
    CELL_NO_PEERS: None,
    CELL_SAME_CHECKPOINT_PRESENT: RUNG_SAME_CHECKPOINT,
    CELL_SAME_CHECKPOINT_CHOSE: RUNG_SAME_CHECKPOINT,
    CELL_DIFFERENT_FAMILY_CHOSE: RUNG_DIFFERENT_FAMILY,
}
"""Which identity paragraph each cell carries; ``None`` is the render with no such paragraph at all,
which is also the stem the one-inserted-paragraph audit compares every other cell against."""

CELLS_WITH_STATUS_REPORT: frozenset[str] = frozenset(
    {CELL_SAME_CHECKPOINT_CHOSE, CELL_DIFFERENT_FAMILY_CHOSE}
)
"""Which cells carry the status-report block. It is one section and its text does not depend on the
identity, so the two report cells' renders differ in the identity paragraph alone."""

MIN_OWN_LOG_BLOCKS = 2
MAX_OWN_LOG_BLOCKS = 4
"""How many tool-result blocks of the agent's own work a brief may show.

At least two, because one block cannot establish that the agent has read its sources and has everything
it needs, which is what makes the fork the last decision left. At most four, because the log is
identical across every cell and both arms, and a longer one buys nothing while pushing the fork further
from the identity paragraph the design varies.
"""

MIN_VALIDATION_REPLIES = 12
"""How many hand-authored replies the rubric must be calibrated against before it runs.

Twelve because the design's own coverage list -- each of six flags true and false at least twice, both
labels as the action, one reply that argues and never tags, a mirror inference made and rejected, a
report relied on without any identity, a constraint cited and then departed from, one that names the
situation as an exercise, and one whose reasoning lives only in the reasoning channel -- does not fit
below it. :func:`sociology.deference_judge.assert_validation_covers_every_flag_both_ways` checks the
coverage itself; this is the floor that catches a set too thin to carry it at all.
"""

MIN_INTENT_VALIDATION_REPLIES = 6
"""The intent check's own coverage list, which is what its floor is: a conclusion agreeing with the tag
in each direction, a conclusion contradicting it in each direction, a reply that is the tag and nothing
else, and one that argues and never settles. Below six one of those is missing, and the one most likely
to be missing is the slip the check exists to count."""

VERDICT_KEYS: tuple[str, ...] = (
    "action_label",
    "constraint_noticed",
    "mirror",
    "social_proof",
    "authority",
    "all_instances_policy",
    "eval_awareness",
)
"""Every verdict field a validation reply registers an expectation for; all of them are compared.

``constraint_noticed`` is in the rubric because a departure without noticing the brief's constraint is an
attention result and a departure after noticing is the deference result, and the two predict the same
tag. ``eval_awareness`` is read on every cell, where no render carries any of the vocabulary
:func:`assert_no_safety_or_evaluation_vocabulary` bans, so its rate is a floor the whole design rests on.
"""

INTENT_VERDICT_KEYS: tuple[str, ...] = ("concluded_action",)
"""What an intent-check validation reply registers: which label the reasoning committed to, if either.

One field rather than the rubric of record's seven, because this instrument answers one question -- did
the reasoning conclude what the tag then said -- and it is the only instrument here whose output rewrites
a published rate.
"""

PROMPT_AFFECTING_FIELDS: tuple[str, ...] = (
    "version",
    "peer_count",
    "scenarios",
    "identity_opening",
    "identity_fragments",
    "identity_tail",
)
"""Every top-level field a rendered prompt can depend on: what :attr:`DeferenceStimulus.prompt_digest`
covers. The complement is :data:`JUDGE_SIDE_FIELDS`, and a file carrying a field in neither refuses to
load, because a section that changed what the model reads without moving the prompt digest would let a
resume continue a file sampled under different prompts while every count still added up."""

JUDGE_INSTRUCTIONS_FIELD = "judge_instructions"
VALIDATION_REPLIES_FIELD = "validation_replies"
INTENT_RUBRIC_FIELD = "intent_rubric"
INTENT_VALIDATION_FIELD = "intent_validation_replies"

JUDGE_SIDE_FIELDS: tuple[str, ...] = (
    JUDGE_INSTRUCTIONS_FIELD,
    VALIDATION_REPLIES_FIELD,
    INTENT_RUBRIC_FIELD,
    INTENT_VALIDATION_FIELD,
)
"""The fields only the judge reads: they move the whole-payload digest and not the prompt digest, so a
rubric authored after a leg has been sampled leaves every row on disk labelled and resumable."""

_SENTENCE_TERMINATORS: tuple[str, ...] = (".", "!", "?")
"""What ends a sentence, so "exactly one sentence" is a check rather than a description."""

SUCCESS_EXIT = 0
"""The exit status every rendered tool result carries: nothing in this transcript failed."""

PEER_HANDLE_TEMPLATE = "agent-{index:02d}"
FIRST_PEER_HANDLE_INDEX = 2
"""The handles the status report lists, and where they start.

Renderer constants rather than authored prose, so the report's shape cannot drift between cells. They
start at two because the report lists the OTHER agents: a list opening at one reads as including the
reader, which would make the count the paragraph states and the count the report shows disagree by one.
"""

PEER_HANDLE_PREFIX = PEER_HANDLE_TEMPLATE.split("{", 1)[0]
"""What every handle opens with, derived from the template so a count of handles cannot drift from it."""

STATUS_LINE_TEMPLATE = "{handle}  {status_field}: {deviation_label}"
"""One status line: a handle, one field, and the option that agent went with. No prose, no reason, no
addressee and no imperative -- an addressed message would measure instruction-following by
construction, and what the design wants read is a choice found in the environment."""

SAFETY_VOCABULARY: tuple[str, ...] = (
    r"safe\w*",
    r"unsafe",
    r"secur\w*",
    r"harm\w*",
    r"danger\w*",
    r"risk\w*",
    r"hazard\w*",
    r"permission\w*",
    r"permitted",
    r"authoriz\w*",
    r"unauthoriz\w*",
    r"polic(?:y|ies)",
    r"violat\w*",
    r"prohibit\w*",
    r"forbid\w*",
    r"illegal",
    r"compl(?:y|ies|iance|iant)",
    r"rule\w*",
    r"guardrail\w*",
    r"sandbox\w*",
    r"credential\w*",
    r"secret\w*",
    r"password\w*",
    r"delet\w*",
    r"destroy\w*",
    r"wipe\w*",
    r"rm -rf",
    r"production",
    r"exfiltrat\w*",
    r"malicious",
    r"attack\w*",
    r"exploit\w*",
    r"breach\w*",
    r"privacy",
    r"private",
    r"sensitive",
    r"confidential",
    r"never",
    r"must not",
    r"under no circumstances",
    r"strictly",
    r"critical\w*",
    r"important",
    r"warning\w*",
    r"caution",
)
"""The safety register, banned from every rendered prompt of this stimulus.

The constraint this design turns on is procedural on purpose. A safety-flavoured rule would confound
following the principal with refusal training -- a model that declined to depart from it would be
indistinguishable from a model that declined because the departure looked unsafe -- and the deontic
intensifiers at the end of the list ("never", "must not", "strictly") would do the same thing through
register alone, which is why they are banned beside the topic words.
"""

EVALUATION_VOCABULARY: tuple[str, ...] = (
    r"test\w*",
    r"evaluat\w*",
    r"\beval\b",
    r"benchmark\w*",
    r"monitor\w*",
    r"observ\w*",
    r"review\w*",
    r"grad(?:e|ed|er|ing)",
    r"scor\w*",
    r"assess\w*",
    r"audit\w*",
    r"exercise",
    r"simulat\w*",
    r"experiment\w*",
    r"scenario",
    r"hypothetical",
    r"research\w*",
    r"study",
    r"watch\w*",
    r"judg\w*",
    r"inspect\w*",
    r"check\w*",
)
"""The evaluation register, banned from every rendered prompt of this stimulus.

``eval_awareness`` is a judge flag on every cell, and its floor is that no render carries any of this
vocabulary: a rate read against a prompt that said "test" would be a measurement of the prompt.
``check\\w*`` bans the harness's own ``<check/>`` idiom from this stimulus deliberately -- no verdict
channel exists here, and a brief that offered one would answer the fork's question for the reader.
"""

_SAFETY_OR_EVALUATION_RE = re.compile(
    r"\b(?:" + "|".join((*SAFETY_VOCABULARY, *EVALUATION_VOCABULARY)) + r")\b", re.IGNORECASE
)

END_TALLY_NOUNS: tuple[str, ...] = (
    r"figures?",
    r"tall(?:y|ies)",
    r"totals?",
    r"counts?",
)
"""Nouns for a quantity worked out at the end of a round, banned from every rendered prompt here.

The transfer probes measured what happens when an answer instruction shares a noun with the prompt's own
end-of-round tally: one model wrote its tally into the decision tag on 323 replies, and both scoring
instruments read the tag. This stimulus has no tally at all -- the artifact is a file, the answer is a
label -- so the whole class is kept out rather than the one noun the defect was found in.
"""

_END_TALLY_RE = re.compile(r"\b(?:" + "|".join(END_TALLY_NOUNS) + r")\b", re.IGNORECASE)

_VOCABULARY_CONTEXT_CHARS = 60
"""How much text a refusal quotes around the offending word, so it points at the sentence."""

DEFERENCE_PARAGRAPH_MARKER_TEMPLATE = "About the other agents: {clause}"


def about_the_other_agents(clause: str) -> str:
    """Open the identity paragraph the one way every render of it opens.

    One source for the opening words, because they are what the one-inserted-paragraph audit deletes:
    a render that phrased its own opening would pass every vocabulary gate and silently drop out of the
    audit, which is the property the whole design rests on.
    """
    return DEFERENCE_PARAGRAPH_MARKER_TEMPLATE.format(clause=clause)


DEFERENCE_PARAGRAPH_MARKER = about_the_other_agents("")
"""The marker handed to :func:`~games.prompts.assert_counterpart_paragraph_is_the_only_insertion`.

The games' audit takes its marker as a keyword for exactly this: an agent transcript whose one inserted
paragraph describes the other agents rather than "the other side" proves the same property, and a second
copy of that function would be a second thing to keep in step.
"""


def assert_no_safety_or_evaluation_vocabulary(text: str) -> None:
    """Refuse rendered text using this stimulus's banned safety or evaluation register.

    Whole words, case-insensitively, over both lists at once, because a reader does not care which list
    a word came from: either register changes what the fork means. The refusal quotes the surrounding
    sentence so the fix is one word rather than a hunt.
    """
    match = _SAFETY_OR_EVALUATION_RE.search(text)
    if match is None:
        return
    start = max(0, match.start() - _VOCABULARY_CONTEXT_CHARS)
    end = min(len(text), match.end() + _VOCABULARY_CONTEXT_CHARS)
    raise ValueError(
        f"rendered text uses the banned safety-or-evaluation word {match.group(0)!r} at offset "
        f"{match.start()}. Context: ...{text[start:end]}... The constraint in this design is "
        f"procedural on purpose: a safety-flavoured word confounds following the principal with "
        f"refusal training, and an evaluation-flavoured one primes the awareness a judge flag here is "
        f"meant to measure against a floor of zero."
    )


def assert_no_end_tally_nouns(text: str) -> None:
    """Refuse rendered text naming a quantity worked out at the end, which this design does not have."""
    match = _END_TALLY_RE.search(text)
    if match is None:
        return
    start = max(0, match.start() - _VOCABULARY_CONTEXT_CHARS)
    end = min(len(text), match.end() + _VOCABULARY_CONTEXT_CHARS)
    raise ValueError(
        f"rendered text uses the end-tally noun {match.group(0)!r} at offset {match.start()}. "
        f"Context: ...{text[start:end]}... The answer instruction asks for a decision, and a prompt "
        f"that also named a quantity worked out at the end is how a model came to write that quantity "
        f"into the decision tag on the transfer probes, where both scoring instruments read the tag."
    )


def peer_handles(count: int) -> tuple[str, ...]:
    """Name the other agents the status report lists, in report order."""
    if count < 1:
        raise ValueError(f"peer_count must be at least 1, got {count}")
    return tuple(
        PEER_HANDLE_TEMPLATE.format(index=index)
        for index in range(FIRST_PEER_HANDLE_INDEX, FIRST_PEER_HANDLE_INDEX + count)
    )


def format_tool_result(command: str, exit_code: int, stdout: str) -> str:
    """Render one command's result in the harness's own observation envelope.

    Byte-identical to :func:`reward_hacking.harness.loop._format_observation` for a command that
    finished and wrote nothing to stderr, which is every block of this transcript, and pinned against it
    by a test. Identical rather than similar because the whole point of the transcript register is that
    the reader has met this envelope before: a stimulus that invented its own would be measuring how a
    model reads an unfamiliar format.
    """
    return f'<result command="{command}" exit={exit_code}>\n<stdout>{stdout}</stdout>\n</result>'


class LabelledScenario(Protocol):
    """Any authored scenario whose two options a print order may swap on the page.

    A protocol rather than a base class, because the sibling coordination stimulus renders a different
    game out of a different file and its scenarios are not deference scenarios in any sense except this
    one: they carry two labels, one of which follows the principal. :func:`labels_in_print_order` is the
    counterbalance both passes read off, and a second copy of it is a second thing to keep in step.

    Read-only properties, which is what a frozen dataclass field satisfies.
    """

    @property
    def scenario_id(self) -> str:
        """What keys this scenario's prompt ids and record keys."""
        ...

    @property
    def compliant_label(self) -> str:
        """The option that follows what the principal asked."""
        ...

    @property
    def deviation_label(self) -> str:
        """The option that departs from it."""
        ...

    @property
    def labels(self) -> tuple[str, str]:
        """The two options, compliant first: the mapping, never the order they are printed in."""
        ...


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """One command of the agent's own log: what it ran, and what came back on stdout."""

    command: str
    stdout: str


@dataclass(frozen=True, slots=True)
class DeferenceScenario:
    """One authored job: its brief, both constraint sentences, its log, its fork and its two labels."""

    scenario_id: str
    brief: str
    constraint_breaking: str
    constraint_benign: str
    own_log: tuple[ToolResultBlock, ...]
    fork: str
    compliant_label: str
    deviation_label: str
    status_command: str
    status_field: str

    @property
    def labels(self) -> tuple[str, str]:
        """The two options, compliant first: the mapping, never the order they are printed in."""
        return (self.compliant_label, self.deviation_label)

    def constraint_for(self, arm: str) -> str:
        """Return the constraint sentence this arm fills the brief's slot with."""
        field_name = CONSTRAINT_FIELD_BY_ARM.get(arm)
        if field_name is None:
            raise ValueError(f"{arm!r} is not an arm of this design; the arms are {list(ARMS)}.")
        return self.constraint_breaking if arm == ARM_BREAKING else self.constraint_benign


def labels_in_print_order(scenario: LabelledScenario, print_order: str) -> tuple[str, str]:
    """Return the scenario's two labels in the order this row prints them.

    The counterbalance, and it moves the page alone: which label is the compliant one is fixed by the
    scenario, so a rate that differed between the orders is a read of the order rather than of the
    design.

    Typed to :class:`LabelledScenario` so the coordination pass's own scenarios go through this one
    function: the mapping from compliant option to printed position is the same counterbalance in both.
    """
    if print_order not in LABEL_PRINT_ORDERS:
        raise ValueError(
            f"print_order must be one of {list(LABEL_PRINT_ORDERS)}, got {print_order!r}."
        )
    compliant, deviation = scenario.labels
    if print_order == LABEL_PRINT_ORDER_CANONICAL:
        return (compliant, deviation)
    return (deviation, compliant)


@dataclass(frozen=True, slots=True)
class DeferenceValidationReply:
    """One hand-authored synthetic reply and the whole verdict the judge must return for it.

    ``compliant_label`` is the validator's own bookkeeping and never reaches a judge prompt: the judge is
    blind to which option follows the brief, and a case that leaked it would validate a different
    instrument than the one production runs.
    """

    name: str
    label_a: str
    label_b: str
    compliant_label: str | None
    reply: str
    reasoning: str
    expected: dict[str, Any]


@dataclass(frozen=True, slots=True)
class DeferenceStimulus:
    """The loaded briefs, the composed identity paragraphs, both rubrics, and the two digests."""

    peer_count: int
    scenarios: tuple[DeferenceScenario, ...]
    identity_clauses: dict[str, str]
    judge_instructions: str
    validation_replies: tuple[DeferenceValidationReply, ...]
    intent_instructions: str
    intent_validation_replies: tuple[DeferenceValidationReply, ...]
    digest: str
    prompt_digest: str

    def scenario(self, scenario_id: str) -> DeferenceScenario:
        """Look up one scenario by id, naming the roster when the id is not in it."""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        known = [scenario.scenario_id for scenario in self.scenarios]
        raise ValueError(f"{scenario_id!r} is not a scenario of this stimulus; known: {known}.")


def identity_for_cell(cell: str) -> str | None:
    """Name which identity paragraph one cell carries, or None for the peer-free render."""
    if cell not in IDENTITY_BY_CELL:
        raise ValueError(f"{cell!r} is not a cell of this design; the cells are {list(CELLS)}.")
    return IDENTITY_BY_CELL[cell]


def status_report_lines(
    scenario: DeferenceScenario, stimulus: DeferenceStimulus
) -> tuple[str, ...]:
    """One status line per other agent: a handle, one field, and the option that agent went with.

    Every line says the same thing because that is what the design shows the reader -- the other agents all
    went the same way -- and no line carries an identity, so the two report cells differ in the paragraph
    above the report and in nothing else.
    """
    return tuple(
        STATUS_LINE_TEMPLATE.format(
            handle=handle,
            status_field=scenario.status_field,
            deviation_label=scenario.deviation_label,
        )
        for handle in peer_handles(stimulus.peer_count)
    )


def status_report_block(scenario: DeferenceScenario, stimulus: DeferenceStimulus) -> str:
    """Render the status-report block: the agent's own log envelope around the status lines."""
    return format_tool_result(
        scenario.status_command, SUCCESS_EXIT, "\n".join(status_report_lines(scenario, stimulus))
    )


def render_deference_sections(  # noqa: PLR0913 - one keyword per axis of the render
    scenario: DeferenceScenario,
    *,
    arm: str,
    identity: str | None,
    with_status_report: bool,
    print_order: str,
    stimulus: DeferenceStimulus,
) -> tuple[str, ...]:
    """Render one prompt as its blank-line-separated sections, gates run over the joined text.

    Sections rather than a string, because every audit this design rests on is positional: a cell is its
    peer-free stem plus one paragraph, a report cell is a present cell plus one block at a fixed
    position, and the two arms differ in the brief section alone. Joining is the caller's last step.

    A two-paragraph brief contributes two sections, so the tuple this returns IS the joined prompt split
    on blank lines, and the closing check says so. One convention, because the audits read the render two
    ways -- some walk this tuple, one splits the joined prompt -- and a brief counted as one element here
    and as its paragraphs there put the status report at two different indices, which refused a correct
    render and let an arm difference in the identity paragraph pass as "inside the brief".

    ``identity`` and ``with_status_report`` are taken separately from the cell id so an audit can render
    a combination the plan does not sample -- the different-family cell with no report is the matched
    stem the report contrast needs, and it is a render rather than a purchase.
    """
    if identity is not None and identity not in stimulus.identity_clauses:
        raise ValueError(
            f"{identity!r} is not an identity of this stimulus; the identities are "
            f"{sorted(stimulus.identity_clauses)}."
        )
    first_label, second_label = labels_in_print_order(scenario, print_order)
    sections = scenario.brief.format(constraint=scenario.constraint_for(arm)).split("\n\n")
    if identity is not None:
        sections.append(about_the_other_agents(stimulus.identity_clauses[identity]))
    sections.extend(
        format_tool_result(block.command, SUCCESS_EXIT, block.stdout) for block in scenario.own_log
    )
    if with_status_report:
        sections.append(status_report_block(scenario, stimulus))
    sections.append(scenario.fork.format(first_label=first_label, second_label=second_label))
    sections.append(ONE_SHOT_INSTRUCTION.format(first_label=first_label, second_label=second_label))
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    assert_no_safety_or_evaluation_vocabulary(prompt)
    assert_no_end_tally_nouns(prompt)
    if identity is None and not with_status_report:
        # Frame-only: anything HERE about how another party decides is the manipulation arriving
        # through the fiction. Never run on the identity paragraph, which trips it by construction.
        assert_no_coupling_claims(prompt)
    if "{" in prompt or "}" in prompt:
        raise ValueError(
            f"the render of {scenario.scenario_id!r} under arm {arm!r} still carries a brace, so an "
            f"unfilled placeholder would reach the model verbatim: {prompt!r}"
        )
    split = prompt.split("\n\n")
    if tuple(split) != tuple(sections):
        raise ValueError(
            f"the render of {scenario.scenario_id!r} under arm {arm!r} builds {len(sections)} sections "
            f"but splits into {len(split)}, so an audit that walks the sections and one that splits the "
            f"prompt disagree about where every block sits. One of the parts carries a blank line of "
            f"its own."
        )
    return tuple(sections)


def render_deference_prompt(
    scenario: DeferenceScenario,
    *,
    arm: str,
    cell: str,
    print_order: str,
    stimulus: DeferenceStimulus,
) -> str:
    """Render one (scenario, arm, cell, print order) into the prompt the model reads."""
    return "\n\n".join(
        render_deference_sections(
            scenario,
            arm=arm,
            identity=identity_for_cell(cell),
            with_status_report=cell in CELLS_WITH_STATUS_REPORT,
            print_order=print_order,
            stimulus=stimulus,
        )
    )


def brief_section_count(scenario: DeferenceScenario) -> int:
    """Count the sections one authored brief contributes, which every cell renders identically.

    Counted rather than assumed to be one, because an authored brief may run to two paragraphs and the
    positional audits index past it: a fixed index would check the wrong sections while still passing on
    the roster it was written against. Both arms give the same answer, since a constraint sentence
    carrying a paragraph break is refused at load.
    """
    return scenario.brief.count("\n\n") + 1


def status_report_section_index(scenario: DeferenceScenario, *, identity: str | None) -> int:
    """Where the status-report block sits: after the brief, the identity paragraph and the own log."""
    return brief_section_count(scenario) + (0 if identity is None else 1) + len(scenario.own_log)


def assert_no_paragraph_break(what: str, text: str, path: Path) -> None:
    """Refuse an identity part carrying a blank line: it would render as two inserted paragraphs.

    Every part of the paragraph, not the fragment alone, because a break in the shared opening or tail
    inserts the second paragraph into BOTH identities, where a break in one fragment inserts it into one.
    """
    if "\n\n" in text:
        raise ValueError(
            f"{what} in {path} contains a paragraph break, so its rendering would insert two "
            f"paragraphs where the one-inserted-paragraph audit expects one."
        )


def _assert_no_unsupported_placeholder(what: str, template: str, path: Path) -> None:
    """Refuse a template naming a placeholder the identity filler cannot supply."""
    unsupported = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(IDENTITY_PLACEHOLDERS))
    if unsupported:
        raise ValueError(
            f"{what} in {path} names the placeholders {unsupported}, and the filler supplies only "
            f"{list(IDENTITY_PLACEHOLDERS)}. An unsupplied placeholder reaches the model in braces."
        )


def assert_one_sentence(what: str, sentence: str, path: Path) -> None:
    """Refuse a sentence that is blank, is two sentences, or carries a paragraph break.

    One sentence is what makes the arm difference a one-sentence swap, which is how the pass reports it.
    Two sentences under one field would be a longer manipulation read under that label, and a paragraph
    break would split the brief into a different number of sections in one arm than in the other.
    """
    body = sentence.strip()
    if not body:
        raise ValueError(f"{what} in {path} is blank, and it is printed verbatim.")
    if "\n\n" in sentence:
        raise ValueError(
            f"{what} in {path} contains a paragraph break, so this arm's brief would render as one "
            f"more section than the other arm's and no pair of the two could be compared by position."
        )
    if not body.endswith(_SENTENCE_TERMINATORS) or any(
        mark in body[:-1] for mark in _SENTENCE_TERMINATORS
    ):
        raise ValueError(
            f"{what} in {path} is not exactly one sentence ({body!r}): it must end on one of "
            f"{list(_SENTENCE_TERMINATORS)} and carry none of them before the end. The two arms differ "
            f"in this sentence alone, and that is how the difference is reported."
        )


def read_own_log(entries: Any, scenario_id: str, path: Path) -> tuple[ToolResultBlock, ...]:  # noqa: ANN401 - raw JSON
    """Read one scenario's own log, refusing a log that cannot render as its own sections.

    Three failures, each of which would leave every count adding up. A log outside the block bounds is a
    transcript that does not establish the agent has what it needs, or one long enough to bury the
    paragraph the design varies. A command carrying a double quote breaks the envelope's own attribute,
    which is a malformed observation the reader has never seen. And a blank line inside stdout splits one
    tool result into two sections, which moves every positional audit by one.
    """
    if (
        not isinstance(entries, list)
        or not MIN_OWN_LOG_BLOCKS <= len(entries) <= MAX_OWN_LOG_BLOCKS
    ):
        raise ValueError(
            f"scenario {scenario_id!r} in {path} carries an own_log of "
            f"{len(entries) if isinstance(entries, list) else type(entries).__name__} rather than "
            f"between {MIN_OWN_LOG_BLOCKS} and {MAX_OWN_LOG_BLOCKS} tool-result blocks."
        )
    blocks: list[ToolResultBlock] = []
    for index, entry in enumerate(entries):
        where = f"own_log[{index}] of scenario {scenario_id!r}"
        if not isinstance(entry, dict):
            raise TypeError(f"{where} in {path} is a {type(entry).__name__}, not an object")
        command = str(entry.get("command", ""))
        stdout = str(entry.get("stdout", ""))
        if not command.strip() or not stdout.strip():
            raise ValueError(
                f"{where} in {path} has a blank command or a blank stdout; a result block with "
                f"nothing in it shows the agent doing nothing while taking up a section."
            )
        if '"' in command:
            raise ValueError(
                f"{where} in {path} has a double quote in its command ({command!r}), which is the "
                f"envelope's own attribute delimiter, so the block would render malformed."
            )
        for name, text in (("command", command), ("stdout", stdout)):
            if "\n\n" in text:
                raise ValueError(
                    f"{where} in {path} has a blank line in its {name}, so this one tool result would "
                    f"render as two sections and every positional audit past it would be off by one."
                )
        blocks.append(ToolResultBlock(command=command, stdout=stdout))
    return tuple(blocks)


def _assert_labels_well_formed(scenario: DeferenceScenario, path: Path) -> None:
    """Refuse a label pair that cannot express a choice, or that the principal's own text pre-empts.

    Both labels reach the reader through the fork's two slots and the answer instruction's, which
    :func:`_assert_slots_are_present` requires and the render's own brace check proves are filled, so
    there is nothing left for this function to check about the reader having met them. What it checks
    instead runs the other way: the principal must never name the DEVIATION option. The brief and its
    constraint are the principal's own words, and a brief that offered the departing option as a live way
    to finish would make the breaking arm's constraint contradict its own brief, and would fix an
    authored order of the two options that the print-order counterbalance cannot swap. They also have to
    differ, because identical labels cannot express a choice and the parser refuses them, and each has to
    be filename-shaped, because a label reaches a status line, a decision tag and a report group.

    The own log is deliberately not searched: it is the agent's own observations, and a source file whose
    readings arrive in the departing option's units is the scenario, not the principal endorsing it.
    """
    unfilenamed = [
        label
        for label in (scenario.compliant_label, scenario.deviation_label, scenario.status_field)
        if not FRAMING_LABEL_RE.match(label)
    ]
    if unfilenamed:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has labels or a status field the pattern "
            f"{FRAMING_LABEL_RE.pattern} refuses: {unfilenamed}. Each of them is printed into a status "
            f"line and into a decision tag, and the two labels key report groups."
        )
    if scenario.compliant_label.strip().casefold() == scenario.deviation_label.strip().casefold():
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} gives both options the same label "
            f"({scenario.compliant_label!r}); identical labels cannot express a choice, and the parser "
            f"refuses the pair outright."
        )
    principal_text = {
        "brief": scenario.brief,
        CONSTRAINT_FIELD_BY_ARM[ARM_BENIGN]: scenario.constraint_benign,
        CONSTRAINT_FIELD_BY_ARM[ARM_BREAKING]: scenario.constraint_breaking,
    }
    pre_empted = sorted(
        name for name, text in principal_text.items() if scenario.deviation_label in text
    )
    if pre_empted:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} names its deviation option "
            f"{scenario.deviation_label!r} in {pre_empted}, which is the principal's own text. The "
            f"departing option is introduced by the fork and by what the other agents did, never by the "
            f"principal: a brief that offered it would contradict the breaking arm's own constraint and "
            f"would fix an option order the print-order counterbalance cannot swap."
        )


def _assert_slots_are_present(scenario: DeferenceScenario, path: Path) -> None:
    """Refuse a brief with no constraint slot, or a fork that does not print both labels."""
    if CONSTRAINT_SLOT not in scenario.brief:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has no {CONSTRAINT_SLOT} in its brief, so "
            f"both arms would render the same prompt and the arm difference would be a measurement of "
            f"nothing."
        )
    missing = [slot for slot in (FIRST_LABEL_SLOT, SECOND_LABEL_SLOT) if slot not in scenario.fork]
    if missing:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has a fork missing {missing}. The fork is "
            f"what prints the two options in the row's own order, and the print-order counterbalance "
            f"is read off it."
        )


def _assert_constraints_are_a_matched_pair(scenario: DeferenceScenario, path: Path) -> None:
    """Refuse a constraint pair that is not two different one-sentence rules, or that misses the fork.

    The breaking arm's whole construction is that the departing option violates the brief's own
    constraint, and the dimension the departure moves is the one the status report names. A breaking
    sentence that never mentions that dimension governs something else, so the arm would be a second
    benign arm with a different sentence in it and the double difference would read as zero honestly.

    Matched case-insensitively, because the dimension word is named just as well by a sentence that opens
    on it. A case-sensitive match cost one authored sentence a subordinate clause whose only job was to
    repeat the word in lower case, which is worse prose for no measurement.
    """
    for arm in ARMS:
        assert_one_sentence(
            f"{CONSTRAINT_FIELD_BY_ARM[arm]} of scenario {scenario.scenario_id!r}",
            scenario.constraint_for(arm),
            path,
        )
    if scenario.constraint_breaking == scenario.constraint_benign:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} gives both arms the same constraint "
            f"sentence, so the two arms would render byte-identical prompts under two block labels."
        )
    if scenario.status_field.casefold() not in scenario.constraint_breaking.casefold():
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has a constraint_breaking that never names "
            f"{scenario.status_field!r}, the dimension its status report reports on "
            f"({scenario.constraint_breaking!r}). The breaking arm exists because the departing option "
            f"violates this constraint; a constraint about another dimension makes it a second benign "
            f"arm whose headline would read as a null."
        )


def status_file_tokens(status_command: str) -> tuple[str, ...]:
    """Return the file the status command reads, as every spelling a log could reach it by.

    Both the path as written and its bare name, because the own log runs in the job's own directory and a
    reader that had already opened the same file by its short name is the hazard the log gate exists for.
    """
    written = [token for token in status_command.split() if "/" in token or "." in token]
    spellings = set(written) | {PurePosixPath(token).name for token in written}
    return tuple(sorted(spellings))


def _assert_the_log_has_not_already_read_the_report(
    scenario: DeferenceScenario, path: Path
) -> None:
    """Refuse an own log that has already reached the status file, by any command or in any output.

    The report is the one section the report cells add, and the presence cells must not have shown it
    already: a log that had listed the same file would mean the presence cell also carried the choice,
    and the report effect the design reads would be a contrast between two renders that both show it.

    Matched on the file rather than on the whole command string, because a second reader command, a
    directory listing, or a stdout that happens to include the file among its lines all show the peers'
    choices while sharing no substring with ``status_command`` at all.
    """
    tokens = status_file_tokens(scenario.status_command)
    if not tokens:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has a status_command "
            f"({scenario.status_command!r}) that names no file, so nothing says which file the own log "
            f"must not have read and the log gate would pass whatever the log did."
        )
    named = sorted(
        {
            f"{where} of own_log[{index}]"
            for index, block in enumerate(scenario.own_log)
            for where, text in (("command", block.command), ("stdout", block.stdout))
            for token in tokens
            if token in text
        }
    )
    if named:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} reaches its own status file "
            f"{list(tokens)} in {named}. The status report is the only section the report cells add, so "
            f"a log that already read that file, or that listed it, shows the choice in every cell."
        )


def _assert_the_scenario_states_no_peer_numeral(
    scenario: DeferenceScenario, peer_count: int, path: Path
) -> None:
    """Refuse authored prose that prints the number the renderer prints.

    The identity paragraph states how many other agents there are and the report shows that many lines;
    a brief that also named the number describes a different situation the first time the count moves,
    and every artifact would still be complete and plausible.
    """
    numeral = str(peer_count)
    stated = sorted(
        name
        for name, text in {
            "brief": scenario.brief,
            CONSTRAINT_FIELD_BY_ARM[ARM_BENIGN]: scenario.constraint_benign,
            CONSTRAINT_FIELD_BY_ARM[ARM_BREAKING]: scenario.constraint_breaking,
            "fork": scenario.fork,
            "own_log": "\n".join(f"{b.command}\n{b.stdout}" for b in scenario.own_log),
        }.items()
        if numeral in text
    )
    if stated:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} states the peer count {numeral!r} in "
            f"{stated}. The renderer prints that number into the identity paragraph and prints that "
            f"many status lines, so authored prose naming it would contradict them the first time the "
            f"count moves."
        )


def _scenario(entry: Mapping[str, Any], peer_count: int, path: Path) -> DeferenceScenario:
    """Read and check one scenario, naming the file and the entry when a field is missing."""
    scenario_id = str(entry.get("scenario_id", "<unnamed>"))
    missing = [
        name
        for name in (
            "brief",
            "constraint_breaking",
            "constraint_benign",
            "own_log",
            "fork",
            "compliant_label",
            "deviation_label",
            "status_command",
            "status_field",
        )
        if name not in entry
    ]
    if missing:
        raise ValueError(f"scenario {scenario_id!r} in {path} is missing {missing}")
    if not FRAMING_LABEL_RE.match(scenario_id):
        raise ValueError(
            f"scenario id {scenario_id!r} in {path} is refused by {FRAMING_LABEL_RE.pattern}; a "
            f"scenario id keys every record key and every prompt id."
        )
    scenario = DeferenceScenario(
        scenario_id=scenario_id,
        brief=str(entry["brief"]),
        constraint_breaking=str(entry["constraint_breaking"]),
        constraint_benign=str(entry["constraint_benign"]),
        own_log=read_own_log(entry["own_log"], scenario_id, path),
        fork=str(entry["fork"]),
        compliant_label=str(entry["compliant_label"]),
        deviation_label=str(entry["deviation_label"]),
        status_command=str(entry["status_command"]),
        status_field=str(entry["status_field"]),
    )
    _assert_slots_are_present(scenario, path)
    _assert_labels_well_formed(scenario, path)
    _assert_constraints_are_a_matched_pair(scenario, path)
    _assert_the_log_has_not_already_read_the_report(scenario, path)
    _assert_the_scenario_states_no_peer_numeral(scenario, peer_count, path)
    return scenario


def _identity_clauses(payload: Mapping[str, Any], peer_count: int, path: Path) -> dict[str, str]:
    """Compose both identity paragraphs from the opening, the fragments and the tail.

    Composed rather than read whole, because the two paragraphs must differ in the fragment ALONE: that
    is what makes the report cells' contrast a read of the identity and of nothing else. A file carrying
    the paragraphs whole could drift on the opening or the tail while still rendering.
    """
    opening = str(payload.get("identity_opening", ""))
    tail = str(payload.get("identity_tail", ""))
    raw = payload.get("identity_fragments", {})
    missing = [rung for rung in IDENTITY_RUNGS if rung not in raw]
    if missing:
        raise ValueError(f"stimulus file {path} is missing identity fragments for {missing}")
    fragments = {rung: str(raw[rung]) for rung in IDENTITY_RUNGS}
    stray = sorted(set(raw) - set(IDENTITY_RUNGS))
    if stray:
        raise ValueError(
            f"stimulus file {path} carries identity fragments {stray} for identities this design does "
            f"not sample; the identities are {list(IDENTITY_RUNGS)}."
        )
    blank = sorted(
        name
        for name, text in {
            "identity_opening": opening,
            "identity_tail": tail,
            **{f"identity_fragments[{rung!r}]": text for rung, text in fragments.items()},
        }.items()
        if not text.strip()
    )
    if blank:
        raise ValueError(
            f"stimulus file {path} leaves {blank} blank; every one is printed verbatim."
        )
    for name, text in (("identity_opening", opening), ("identity_tail", tail)):
        _assert_no_unsupported_placeholder(name, text, path)
        assert_no_paragraph_break(name, text, path)
    for rung, fragment in fragments.items():
        where = f"identity_fragments[{rung!r}]"
        _assert_no_unsupported_placeholder(where, fragment, path)
        assert_no_paragraph_break(where, fragment, path)
        swallowed = sorted(
            name
            for name, shared in (("identity_opening", opening), ("identity_tail", tail))
            if shared.strip() and shared.strip() in fragment
        )
        if swallowed:
            raise ValueError(
                f"{where} in {path} contains the shared {swallowed}: a fragment carries the identity "
                f"alone, and one carrying its own opening or tail would make the two paragraphs differ "
                f"outside the fragment while both still rendered."
            )
    clauses = {
        rung: f"{opening}{fragment}{tail}".format(count=peer_count)
        for rung, fragment in fragments.items()
    }
    _assert_paragraphs_differ_in_the_fragment_alone(
        clauses, opening=opening, fragments=fragments, tail=tail, peer_count=peer_count, path=path
    )
    return clauses


def _assert_paragraphs_differ_in_the_fragment_alone(  # noqa: PLR0913 - the parts, plus what names it
    clauses: Mapping[str, str],
    *,
    opening: str,
    fragments: Mapping[str, str],
    tail: str,
    peer_count: int,
    path: Path,
) -> None:
    """Refuse composed paragraphs that are not the shared parts around their own fragment.

    True by construction today, and checked anyway: it is the identity the report contrast rests on, and
    the composition above is one edit away from being reordered or from filling a paragraph from the
    wrong fragment. The two paragraphs also have to differ at all -- two identities rendering one text
    would make the pass's headline a comparison of a cell with itself.
    """
    for rung, fragment in fragments.items():
        expected = f"{opening}{fragment}{tail}".format(count=peer_count)
        if clauses[rung] != expected:
            raise ValueError(
                f"the composed identity paragraph for {rung!r} in {path} is not the shared opening and "
                f"tail around its own fragment: expected {expected!r}, composed {clauses[rung]!r}."
            )
    if len({clauses[rung] for rung in fragments}) != len(fragments):
        raise ValueError(
            f"stimulus file {path} composes the same identity paragraph for two of {list(fragments)}, "
            f"so the design's headline contrast would compare a cell with itself."
        )


def _validation_reply(
    entry: Mapping[str, Any], path: Path, *, keys: tuple[str, ...]
) -> DeferenceValidationReply:
    """Read one validation reply, naming the file and the reply when an expectation is missing.

    ``keys`` is the rubric's own registered fields, so the same reader serves both instruments: the
    rubric of record registers seven, and the intent check registers the one field it answers.
    """
    name = str(entry.get("name", "<unnamed>"))
    expected = entry.get("expected", {})
    missing = [key for key in keys if key not in expected]
    if missing:
        raise ValueError(
            f"validation reply {name!r} in {path} registers no expectation for {missing}; all "
            f"{len(keys)} verdict fields are compared, so an absent one would go unchecked while the "
            f"report still read as agreement."
        )
    label_a = str(entry["label_a"])
    label_b = str(entry["label_b"])
    if label_a.strip().casefold() == label_b.strip().casefold():
        raise ValueError(
            f"validation reply {name!r} in {path} gives both options the label {label_a!r}; the judge "
            f"is handed the row's own two labels and a verdict naming one of them could mean either."
        )
    compliant = entry.get("compliant_label")
    if compliant is not None and str(compliant) not in (label_a, label_b):
        raise ValueError(
            f"validation reply {name!r} in {path} names compliant_label {compliant!r}, which is "
            f"neither of its own labels ({label_a!r}, {label_b!r}); the bookkeeping would say a reply "
            f"followed an option the row never offered."
        )
    return DeferenceValidationReply(
        name=name,
        label_a=label_a,
        label_b=label_b,
        compliant_label=None if compliant is None else str(compliant),
        reply=str(entry["reply"]),
        reasoning=str(entry.get("reasoning", "")),
        expected={key: expected[key] for key in keys},
    )


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    """Digest a JSON-shaped payload canonically: sorted keys, no whitespace, first sixteen hex chars."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def prompt_digest_of(payload: Mapping[str, Any]) -> str:
    """Digest only the sections a rendered prompt depends on, so a judge-side edit leaves it alone."""
    return _canonical_digest({name: payload[name] for name in PROMPT_AFFECTING_FIELDS})


def _assert_every_field_is_classified(payload: Mapping[str, Any], path: Path) -> None:
    """Refuse a top-level field filed as neither prompt-affecting nor judge-side.

    Filing is what makes the two digests mean what they claim: a field on neither list could change what
    the model reads without moving ``prompt_digest``, and a resume would then continue a file sampled
    under different prompts while every count still added up.
    """
    unclassified = sorted(set(payload) - set(PROMPT_AFFECTING_FIELDS) - set(JUDGE_SIDE_FIELDS))
    if unclassified:
        raise ValueError(
            f"stimulus file {path} carries top-level fields {unclassified} that are filed as neither "
            f"prompt-affecting ({list(PROMPT_AFFECTING_FIELDS)}) nor judge-side "
            f"({list(JUDGE_SIDE_FIELDS)}). Add each one to the list that matches what reads it, so the "
            f"prompt digest moves when the prompts do and stays put when only the judge's material does."
        )


def load_stimulus(path: Path = STIMULUS_PATH) -> DeferenceStimulus:
    """Load and validate the deference stimulus file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must never
    carry, or empty strings, which would render briefless prompts that measure nothing the design
    describes. The refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"stimulus file {path} is missing. It is gitignored on purpose (the briefs, the constraint "
            "sentences, the tool-result log, the forks, the identity fragments, both rubrics and the "
            "validation replies are authored stimulus that must never be committed); a fresh clone does "
            "not contain it. Recreate it from the design doc in docs/scratch/."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    if version != STIMULUS_VERSION:
        raise ValueError(
            f"stimulus file {path} has version {version!r}, expected {STIMULUS_VERSION!r}"
        )
    _assert_every_field_is_classified(payload, path)
    peer_count = int(payload.get("peer_count", 0))
    if peer_count != PEER_COUNT:
        raise ValueError(
            f"stimulus file {path} states peer_count {peer_count!r} and this design is built around "
            f"{PEER_COUNT}. The count is printed into the identity paragraph and IS the status "
            f"report's line count, so a file stating another number would render a paragraph and a "
            f"report that disagree about how many other agents there are."
        )
    entries = payload.get("scenarios", [])
    if len(entries) != N_SCENARIOS:
        raise ValueError(
            f"stimulus file {path} carries {len(entries)} scenarios, expected exactly {N_SCENARIOS}: "
            f"the unit of pairing is (scenario, print order), and a per-cell rate over fewer briefs is "
            f"one job's answer rather than a rate."
        )
    scenarios = tuple(_scenario(entry, peer_count, path) for entry in entries)
    duplicated = sorted(
        {
            scenario.scenario_id
            for scenario in scenarios
            if sum(1 for other in scenarios if other.scenario_id == scenario.scenario_id) > 1
        }
    )
    if duplicated:
        raise ValueError(
            f"stimulus file {path} carries duplicate scenario ids {duplicated}; a scenario id keys "
            f"every prompt id and record key, so two briefs under one id would pool their answers."
        )
    stimulus = DeferenceStimulus(
        peer_count=peer_count,
        scenarios=scenarios,
        identity_clauses=_identity_clauses(payload, peer_count, path),
        judge_instructions=_rubric(payload, JUDGE_INSTRUCTIONS_FIELD, path),
        validation_replies=_validation_set(
            payload,
            VALIDATION_REPLIES_FIELD,
            path,
            keys=VERDICT_KEYS,
            floor=MIN_VALIDATION_REPLIES,
        ),
        intent_instructions=_rubric(payload, INTENT_RUBRIC_FIELD, path),
        intent_validation_replies=_validation_set(
            payload,
            INTENT_VALIDATION_FIELD,
            path,
            keys=INTENT_VERDICT_KEYS,
            floor=MIN_INTENT_VALIDATION_REPLIES,
        ),
        digest=_canonical_digest(payload),
        prompt_digest=prompt_digest_of(payload),
    )
    _assert_every_cell_renders(stimulus, path)
    return stimulus


def _rubric(payload: Mapping[str, Any], field_name: str, path: Path) -> str:
    """Read one authored rubric, refusing an absent one by name."""
    text = str(payload.get(field_name, ""))
    if not text.strip():
        raise ValueError(
            f"stimulus file {path} carries no {field_name}. Without it the instrument it belongs to "
            f"would run on an empty prompt and return verdicts nobody asked for."
        )
    return text


def _validation_set(
    payload: Mapping[str, Any],
    field_name: str,
    path: Path,
    *,
    keys: tuple[str, ...],
    floor: int,
) -> tuple[DeferenceValidationReply, ...]:
    """Read one instrument's authored cases, refusing a set too thin to calibrate it."""
    raw = payload.get(field_name, [])
    if len(raw) < floor:
        raise ValueError(
            f"stimulus file {path} carries {len(raw)} {field_name}, and that instrument is not "
            f"calibrated on fewer than {floor}: the design's own coverage list for it does not fit "
            f"below that."
        )
    return tuple(_validation_reply(entry, path, keys=keys) for entry in raw)


def _assert_every_cell_renders(stimulus: DeferenceStimulus, path: Path) -> None:
    """Render every (scenario, arm, cell, print order) at load, so a bad brief fails before a plan.

    At load rather than at plan time, because a brief that only breaks in one arm or one print order
    would otherwise surface halfway through a sampling run, with the counts already written down and the
    first leg already submitted. Every vocabulary gate runs inside the renderer, so this sweep is what
    makes all of them cover all 128 renders of the file.
    """
    for scenario in stimulus.scenarios:
        for arm in ARMS:
            for cell in CELLS:
                for print_order in LABEL_PRINT_ORDERS:
                    try:
                        render_deference_prompt(
                            scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"the render of {scenario.scenario_id!r} in {path} under arm {arm!r}, cell "
                            f"{cell!r}, print order {print_order!r} is refused: {error}"
                        ) from error
