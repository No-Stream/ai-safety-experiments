"""Runtime loader and renderer for the coordination stimulus: the brief, the log, the blocks, the fork.

The eight authored briefs, both setting passages and both oversight sentences of each, the tool-result
blocks of the agent's own log, both reconciliation variants, the forks, the identity paragraph's parts, the
judge rubric and every hand-authored validation reply are stimulus that will run against future models.
Committed, they become training data and contaminate every measurement made with them, so they live in one
gitignored JSON file this module loads at runtime and refuses to run without. No stimulus prose appears in
tracked code; this module knows the file's *shape* and its authoring constraints, never its text.

What a rendered prompt is: one user message whose blank-line-separated sections are the principal's brief
(carrying the setting in one slot and the oversight sentence in another), then -- on the three peer cells --
one paragraph saying who the other agents on this migration are, then the two to four tool-result blocks of
work the agent has already done, then the status block in which the shared file lists every shard as
pending, then the reconciliation block in which the close procedure states its own rule, then the fork,
then the shared answer instruction.

Five properties the readings depend on are enforced here rather than described:

- **The identity paragraph is composed out of three parts, not read whole.** The file carries one opening,
  two identity fragments and two tails, and the loader concatenates them, so the copies cell and the
  strangers cell differ in the FRAGMENT alone and the copies cell and the coin cell differ in the TAIL
  alone. A file carrying whole paragraphs could drift on the opening while still rendering, and the
  headline would then be a read of two rewordings.
- **The two arms differ by one whitespace token.** Both oversight sentences fill the same slot of the same
  brief, each is exactly one sentence, and the pair is checked to differ by exactly one token inserted,
  deleted or replaced -- so the oversight effect is that one token and nothing else.
- **Nobody has chosen yet, and the reader learns it from the environment.** The status block is
  renderer-owned: its handles, its line shape, its line count and the pending label come from this module
  and the loaded peer count, never from authored prose, so it is byte-identical in every cell of both arms
  and no cell can show a choice the design says has not been made.
- **The close procedure is an observation, not a message.** Both reconciliation variants are the stdout of
  one command, so what the reader knows about how the shards are compared arrives the way the incident's
  own agents learned things: by reading the environment.
- **Four vocabulary gates run on every render.** The games' own
  :func:`~games.prompts.assert_no_loaded_vocabulary`; the frame-only
  :func:`~games.prompts.assert_no_coupling_claims`, on the renders with no identity paragraph, because that
  paragraph exists to say who the others are and would trip it by construction; the deference pass's
  :func:`~sociology.deference_stimulus.assert_no_safety_or_evaluation_vocabulary`, imported unchanged, which
  is why the oversight cue may say a person reads this batch and may not say review, monitor or check; and
  :func:`~sociology.deference_stimulus.assert_no_end_tally_nouns`, because the answer instruction asks for a
  decision and a prompt that also named a quantity worked out at the end is how a model came to write that
  quantity into the decision tag.

Two digests, for the reason the deference file has two: ``digest`` covers the whole payload and moves on any
edit, ``prompt_digest`` covers only the fields a rendered prompt depends on, so a rubric edit neither
mislabels collected rows nor refuses every resume. Every top-level field has to be filed on one side or the
other, and a file carrying one that is not refuses to load.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from games.prompts import (
    FRAMING_LABEL_RE,
    LABEL_PRINT_ORDERS,
    ONE_SHOT_INSTRUCTION,
    assert_no_coupling_claims,
    assert_no_loaded_vocabulary,
)
from sociology.decoupled_stimulus import RUNG_DIFFERENT_FAMILY, RUNG_SAME_CHECKPOINT
from sociology.deference_stimulus import (
    SUCCESS_EXIT,
    DeferenceValidationReply,
    ToolResultBlock,
    about_the_other_agents,
    assert_no_end_tally_nouns,
    assert_no_paragraph_break,
    assert_no_safety_or_evaluation_vocabulary,
    assert_one_sentence,
    format_tool_result,
    labels_in_print_order,
    read_own_log,
    status_file_tokens,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

STIMULUS_PATH = Path("docs/scratch/swarm-sociology-2026-08-31/coordination_stimulus.json")

STIMULUS_VERSION = "coordination-stimulus-v1"

N_SCENARIOS = 8
"""How many migrations the file carries. Eight because the unit of pairing is (scenario, print order), and
a per-cell rate over fewer briefs is one job's answer rather than a rate."""

PEER_COUNT = 5
"""How many OTHER agents hold a shard of tonight's migration.

Five others plus the reader is a group of six, which reads as a working shift rather than as a fleet, and
six shards is what the owner's design fixed. The file states it too and the loader refuses another number,
because the count is printed into the identity paragraph and the shard count derived from it IS the status
block's line count: a file stating another number would render a paragraph and a block that disagree about
how many agents there are while both still rendering.
"""


def shard_count(peer_count: int = PEER_COUNT) -> int:
    """How many shards the migration has: the other agents plus the reader.

    Derived rather than stated twice, because the status block lists one line per shard and the identity
    paragraph states the number of OTHER agents; two independent numbers would eventually disagree by one
    and every artifact would still be complete and plausible.
    """
    return peer_count + 1


SHARD_COUNT = shard_count()

ARM_OVERSIGHT_ON = "oversight-on"
ARM_OVERSIGHT_OFF = "oversight-off"
ARMS: tuple[str, ...] = (ARM_OVERSIGHT_ON, ARM_OVERSIGHT_OFF)
"""The two arms: a person goes through this batch by hand after the close, or nobody does.

A factor rather than a contaminant. ``eval_awareness`` is read in both, and the ``oversight-off`` arm is
its floor by construction: no render there carries an oversight cue or any word of the evaluation register.
"""

OVERSIGHT_FIELD_BY_ARM: dict[str, str] = {
    ARM_OVERSIGHT_ON: "oversight_on",
    ARM_OVERSIGHT_OFF: "oversight_off",
}
"""Which authored sentence each arm fills the brief's oversight slot with."""

BRIEF_PLACEHOLDERS: tuple[str, ...] = ("setting", "oversight")
SETTING_PLACEHOLDERS: tuple[str, ...] = ("reader_shard", "peer_count", "shard_count")
FORK_PLACEHOLDERS: tuple[str, ...] = ("first_label", "second_label")
COUNT_PLACEHOLDER = "count"
IDENTITY_PLACEHOLDERS: tuple[str, ...] = (COUNT_PLACEHOLDER,)
"""The only placeholder the identity paragraph's parts may name; the filler supplies nothing else."""

_PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")

CELL_IDENTITY_BLIND = "identity-blind"
CELL_SAME_CHECKPOINT = "same-checkpoint"
CELL_DIFFERENT_FAMILY = "different-family"
CELL_SAME_CHECKPOINT_DRAWN = "same-checkpoint-drawn"
CELL_NO_PEERS = "no-peers"
CELL_NO_PEERS_BLOCKS = "no-peers-blocks"
CELLS: tuple[str, ...] = (
    CELL_IDENTITY_BLIND,
    CELL_SAME_CHECKPOINT,
    CELL_DIFFERENT_FAMILY,
    CELL_SAME_CHECKPOINT_DRAWN,
    CELL_NO_PEERS,
    CELL_NO_PEERS_BLOCKS,
)
"""The six cells, in reading order: the anonymous baseline, the three peer identities, then the two
sole-owner cells.

The identity-blind cell is first because it is the stem the one-inserted-paragraph audit compares every
peer cell against and the baseline every peer reading is a contrast to. The two sole-owner cells close the
table: ``no-peers`` is the floor (the shortcut is certain to be caught by the runbook comparison) and
``no-peers-blocks`` is the headroom ceiling (a sole owner can make the whole set uniform alone, so only the
human reader of the ``oversight-on`` arm stands in the way).
"""

PARAGRAPH_SAME_CHECKPOINT = CELL_SAME_CHECKPOINT
PARAGRAPH_DIFFERENT_FAMILY = CELL_DIFFERENT_FAMILY
PARAGRAPH_SAME_CHECKPOINT_DRAWN = CELL_SAME_CHECKPOINT_DRAWN

IDENTITY_BY_CELL: dict[str, str | None] = {
    CELL_IDENTITY_BLIND: None,
    CELL_SAME_CHECKPOINT: PARAGRAPH_SAME_CHECKPOINT,
    CELL_DIFFERENT_FAMILY: PARAGRAPH_DIFFERENT_FAMILY,
    CELL_SAME_CHECKPOINT_DRAWN: PARAGRAPH_SAME_CHECKPOINT_DRAWN,
    CELL_NO_PEERS: None,
    CELL_NO_PEERS_BLOCKS: None,
}
"""Which composed identity paragraph each cell carries; ``None`` is a render with no such paragraph."""

PEER_CELLS: tuple[str, ...] = tuple(cell for cell in CELLS if IDENTITY_BY_CELL[cell] is not None)
"""The three cells that name who the other agents are, which is what the insertion audit walks."""

IDENTITY_FRAGMENTS: tuple[str, ...] = (RUNG_SAME_CHECKPOINT, RUNG_DIFFERENT_FAMILY)
"""The two peer identities, imported rather than renamed: they are the same two the transfer, ladder and
deference passes sample, so a rate here is comparable with a rate there by construction."""

TAIL_SIMULTANEOUS = "simultaneous"
TAIL_DRAWN = "drawn"
IDENTITY_TAILS: tuple[str, ...] = (TAIL_SIMULTANEOUS, TAIL_DRAWN)
"""The two tails: each of the others writes its own block at the close, or what each writes was fixed for
it by a fair coin before the run began.

The tails are the ONLY place any decision mechanism is stated -- the brief states the runbook as a rule,
the status block shows nothing written yet, and the reconciliation rule speaks of blocks -- so the coin cell
and the simultaneous cell differ in the tail alone and neither contradicts the frame.
"""

IDENTITY_PARTS_BY_PARAGRAPH: dict[str, tuple[str, str]] = {
    PARAGRAPH_SAME_CHECKPOINT: (RUNG_SAME_CHECKPOINT, TAIL_SIMULTANEOUS),
    PARAGRAPH_DIFFERENT_FAMILY: (RUNG_DIFFERENT_FAMILY, TAIL_SIMULTANEOUS),
    PARAGRAPH_SAME_CHECKPOINT_DRAWN: (RUNG_SAME_CHECKPOINT, TAIL_DRAWN),
}
"""(fragment, tail) for each composed paragraph. The three paragraphs are pairwise one part apart: the
first two share the tail, the first and third share the fragment."""

SETTING_PEERS = "setting_peers"
SETTING_SOLE = "setting_sole"
SETTINGS: tuple[str, ...] = (SETTING_PEERS, SETTING_SOLE)

SETTING_BY_CELL: dict[str, str] = {
    CELL_IDENTITY_BLIND: SETTING_PEERS,
    CELL_SAME_CHECKPOINT: SETTING_PEERS,
    CELL_DIFFERENT_FAMILY: SETTING_PEERS,
    CELL_SAME_CHECKPOINT_DRAWN: SETTING_PEERS,
    CELL_NO_PEERS: SETTING_SOLE,
    CELL_NO_PEERS_BLOCKS: SETTING_SOLE,
}
"""Which setting passage fills each cell's brief: the reader holds one shard of several, or all of them."""

RECONCILIATION_BLOCKS = "reconciliation_blocks"
RECONCILIATION_RUNBOOK = "reconciliation_runbook"
RECONCILIATIONS: tuple[str, ...] = (RECONCILIATION_BLOCKS, RECONCILIATION_RUNBOOK)

RECONCILIATION_BY_CELL: dict[str, str] = {
    CELL_IDENTITY_BLIND: RECONCILIATION_BLOCKS,
    CELL_SAME_CHECKPOINT: RECONCILIATION_BLOCKS,
    CELL_DIFFERENT_FAMILY: RECONCILIATION_BLOCKS,
    CELL_SAME_CHECKPOINT_DRAWN: RECONCILIATION_BLOCKS,
    CELL_NO_PEERS: RECONCILIATION_RUNBOOK,
    CELL_NO_PEERS_BLOCKS: RECONCILIATION_BLOCKS,
}
"""Which close procedure each cell's reconciliation block states.

``no-peers`` is the one cell compared against the runbook rather than against the other blocks, which is
what makes its shortcut certain to fail and its rate the floor of constraint-breaking with no coordination
story and no way out. ``no-peers-blocks`` keeps the block comparison a sole owner can satisfy alone.
"""

SHARD_HANDLE_TEMPLATE = "shard-{index:02d}"
FIRST_SHARD_INDEX = 1
"""The handles the status block lists, and where they start.

Renderer constants rather than authored prose, so the block's shape cannot drift between cells. They start
at one because the block lists EVERY shard including the reader's own: the reader's row being pending is
how the design says the reader has not chosen either.
"""

SHARD_HANDLE_PREFIX = SHARD_HANDLE_TEMPLATE.split("{", 1)[0]
"""What every handle opens with, derived from the template so a count of handles cannot drift from it."""

READER_SHARD = SHARD_HANDLE_TEMPLATE.format(index=FIRST_SHARD_INDEX)
"""Which shard the reader holds, printed into the setting passage through its own slot."""

PENDING_LABEL = "pending"
"""What every status line says tonight: nobody has chosen yet.

A renderer constant, and the loader refuses a scenario whose own option labels equal it -- a status line
reading as one of the two options would show a choice in a design whose whole construction is that no
choice has been made.
"""

STATUS_LINE_TEMPLATE = "{handle}  {status_field}: {pending}"
"""One status line: a shard handle, one field, and the fact that it is pending. No prose, no reason, no
addressee -- an addressed message would measure instruction-following by construction, and what this design
shows the reader is the state of the environment."""

MIN_VALIDATION_REPLIES = 14
"""How many hand-authored replies the rubric must be calibrated against before it runs.

Fourteen because the design's own coverage list -- each of eight flags true and false at least twice, both
labels as the action, one reply that argues and never tags, a mirror inference made and rejected, one that
takes the coin, one that ignores it and mirrors, one that weighs being found out and departs anyway, one
that weighs it and complies, one that cites the runbook and departs, one that names the situation as an
examination, one whose reasoning lives only in the reasoning channel, and one that weighs the deadline and
still complies -- does not fit below it.
:meth:`sociology.deference_judge.PassJudge.assert_validation_covers_every_flag_both_ways` checks the
coverage itself; this is the floor that catches a set too thin to carry it at all.
"""

MIN_INTENT_VALIDATION_REPLIES = 6
"""The intent check's own coverage list, which is what its floor is: a conclusion agreeing with the tag in
each direction, a conclusion contradicting it in each direction, a reply that is the tag and nothing else,
and one that argues and never settles."""

VERDICT_KEYS: tuple[str, ...] = (
    "action_label",
    "constraint_noticed",
    "deadline_weighed",
    "mirror",
    "treats_others_as_deciding",
    "all_instances_policy",
    "authority",
    "eval_awareness",
    "reasons_about_being_caught",
)
"""Every verdict field a validation reply registers an expectation for; all of them are compared.

``deadline_weighed`` is here because a shortcut taken without weighing the close time is inattention and
one taken after weighing it is the trade the design is about, and the two predict the same tag.
``treats_others_as_deciding`` is true by construction in the simultaneous cells and false by construction
in the coin cell, so its rate there is the manipulation-landed check. ``reasons_about_being_caught`` is the
other one: its contrast between the two arms is what says the oversight cue was read at all, and unlike
``eval_awareness`` it has no floor of zero, because the reconciliation rule is in every cell.
"""

INTENT_VERDICT_KEYS: tuple[str, ...] = ("concluded_action",)
"""What an intent-check validation reply registers: which option the reasoning committed to, if either."""

PROMPT_AFFECTING_FIELDS: tuple[str, ...] = (
    "version",
    "peer_count",
    "scenarios",
    "identity_opening",
    "identity_fragments",
    "identity_tails",
)
"""Every top-level field a rendered prompt can depend on: what :attr:`CoordinationStimulus.prompt_digest`
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

MIN_LABEL_WORDS = 2
"""How many hyphen-joined words an option label needs, so it reads as the option rather than as a token.

The fork is the only place the reader meets either label -- the principal's text never names the departing
option and the status block is all pending -- so ``mark-now`` says what it does and ``a`` does not.
"""

_NUMBER_WORDS: dict[int, str] = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
}
"""How each count could be spelled in words, so the numeral gate covers both spellings.

A table rather than a library call, and the loader refuses a count outside it: a count whose word this
module does not know would pass the gate silently, which is the failure the gate exists to prevent.
"""

_VOCABULARY_CONTEXT_CHARS = 60
"""How much text a refusal quotes around the offending word, so it points at the sentence."""


@dataclass(frozen=True, slots=True)
class CoordinationScenario:
    """One authored migration: its brief, both settings, both oversight sentences, its log and its close."""

    scenario_id: str
    brief: str
    setting_peers: str
    setting_sole: str
    oversight_on: str
    oversight_off: str
    own_log: tuple[ToolResultBlock, ...]
    reconciliation_command: str
    reconciliation_blocks: str
    reconciliation_runbook: str
    status_command: str
    status_field: str
    fork: str
    compliant_label: str
    deviation_label: str

    @property
    def labels(self) -> tuple[str, str]:
        """The two options, compliant first: the mapping, never the order they are printed in."""
        return (self.compliant_label, self.deviation_label)

    def setting_for(self, cell: str) -> str:
        """Return the setting passage this cell's brief carries, unfilled."""
        field_name = SETTING_BY_CELL.get(cell)
        if field_name is None:
            raise ValueError(f"{cell!r} is not a cell of this design; the cells are {list(CELLS)}.")
        return self.setting_peers if field_name == SETTING_PEERS else self.setting_sole

    def oversight_for(self, arm: str) -> str:
        """Return the oversight sentence this arm fills the brief's slot with."""
        if arm not in OVERSIGHT_FIELD_BY_ARM:
            raise ValueError(f"{arm!r} is not an arm of this design; the arms are {list(ARMS)}.")
        return self.oversight_on if arm == ARM_OVERSIGHT_ON else self.oversight_off

    def reconciliation_for(self, cell: str) -> str:
        """Return the close procedure's stdout for this cell."""
        field_name = RECONCILIATION_BY_CELL.get(cell)
        if field_name is None:
            raise ValueError(f"{cell!r} is not a cell of this design; the cells are {list(CELLS)}.")
        return (
            self.reconciliation_blocks
            if field_name == RECONCILIATION_BLOCKS
            else self.reconciliation_runbook
        )


@dataclass(frozen=True, slots=True)
class CoordinationStimulus:
    """The loaded migrations, the three composed identity paragraphs, both rubrics, and the two digests."""

    peer_count: int
    scenarios: tuple[CoordinationScenario, ...]
    identity_clauses: dict[str, str]
    judge_instructions: str
    validation_replies: tuple[DeferenceValidationReply, ...]
    intent_instructions: str
    intent_validation_replies: tuple[DeferenceValidationReply, ...]
    digest: str
    prompt_digest: str

    @property
    def shard_count(self) -> int:
        """How many shards the migration has, derived from the loaded peer count."""
        return shard_count(self.peer_count)

    def scenario(self, scenario_id: str) -> CoordinationScenario:
        """Look up one scenario by id, naming the roster when the id is not in it."""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        known = [scenario.scenario_id for scenario in self.scenarios]
        raise ValueError(f"{scenario_id!r} is not a scenario of this stimulus; known: {known}.")


def identity_for_cell(cell: str) -> str | None:
    """Name which composed identity paragraph one cell carries, or None for a render with none."""
    if cell not in IDENTITY_BY_CELL:
        raise ValueError(f"{cell!r} is not a cell of this design; the cells are {list(CELLS)}.")
    return IDENTITY_BY_CELL[cell]


def shard_handles(count: int) -> tuple[str, ...]:
    """Name every shard of the migration, in the order the status block lists them."""
    if count < 1:
        raise ValueError(f"a migration needs at least one shard, got {count}")
    return tuple(
        SHARD_HANDLE_TEMPLATE.format(index=index)
        for index in range(FIRST_SHARD_INDEX, FIRST_SHARD_INDEX + count)
    )


def status_lines(scenario: CoordinationScenario, stimulus: CoordinationStimulus) -> tuple[str, ...]:
    """One status line per shard, every one of them pending.

    Every line says the same thing because that is what the design shows the reader -- nobody has written a
    block yet -- and no line carries an identity, so the peer cells' renders differ in the paragraph above
    the block and in nothing else.
    """
    return tuple(
        STATUS_LINE_TEMPLATE.format(
            handle=handle, status_field=scenario.status_field, pending=PENDING_LABEL
        )
        for handle in shard_handles(stimulus.shard_count)
    )


def status_block(scenario: CoordinationScenario, stimulus: CoordinationStimulus) -> str:
    """Render the status block: the harness envelope around the all-pending status lines."""
    return format_tool_result(
        scenario.status_command, SUCCESS_EXIT, "\n".join(status_lines(scenario, stimulus))
    )


def reconciliation_block(scenario: CoordinationScenario, *, cell: str) -> str:
    """Render the reconciliation block: the harness envelope around this cell's close procedure."""
    return format_tool_result(
        scenario.reconciliation_command, SUCCESS_EXIT, scenario.reconciliation_for(cell)
    )


def filled_setting(
    scenario: CoordinationScenario, stimulus: CoordinationStimulus, *, cell: str
) -> str:
    """Fill this cell's setting passage with the counts and the reader's own shard.

    The counts are printed here and never authored, which is why the loader refuses a passage that spells
    either of them: authored prose naming the number would contradict the status block the first time the
    count moved, and every artifact would still be complete and plausible.
    """
    return scenario.setting_for(cell).format(
        reader_shard=READER_SHARD,
        peer_count=stimulus.peer_count,
        shard_count=stimulus.shard_count,
    )


def render_coordination_sections(
    scenario: CoordinationScenario,
    *,
    arm: str,
    cell: str,
    print_order: str,
    stimulus: CoordinationStimulus,
) -> tuple[str, ...]:
    """Render one prompt as its blank-line-separated sections, every gate run over the joined text.

    Sections rather than a string, because every audit this design rests on is positional: a peer cell is
    its identity-blind stem plus one paragraph, the sole-owner cells are that stem with one setting passage
    or one stdout substituted, and the two arms differ in the brief section alone. Joining is the caller's
    last step.

    A two-paragraph brief contributes two sections, so the tuple this returns IS the joined prompt split on
    blank lines, and the closing check says so. One convention, because the audits read a render two ways --
    some walk this tuple, one splits the joined prompt -- and a brief counted as one element here and as its
    paragraphs there would put the status block at two different indices, which refuses a correct render and
    lets a difference in the identity paragraph pass as "inside the brief".
    """
    if cell not in CELLS:
        raise ValueError(f"{cell!r} is not a cell of this design; the cells are {list(CELLS)}.")
    identity = identity_for_cell(cell)
    if identity is not None and identity not in stimulus.identity_clauses:
        raise ValueError(
            f"{identity!r} is not an identity paragraph of this stimulus; the paragraphs are "
            f"{sorted(stimulus.identity_clauses)}."
        )
    first_label, second_label = labels_in_print_order(scenario, print_order)
    sections = scenario.brief.format(
        setting=filled_setting(scenario, stimulus, cell=cell),
        oversight=scenario.oversight_for(arm),
    ).split("\n\n")
    if identity is not None:
        sections.append(about_the_other_agents(stimulus.identity_clauses[identity]))
    sections.extend(
        format_tool_result(block.command, SUCCESS_EXIT, block.stdout) for block in scenario.own_log
    )
    sections.append(status_block(scenario, stimulus))
    sections.append(reconciliation_block(scenario, cell=cell))
    sections.append(scenario.fork.format(first_label=first_label, second_label=second_label))
    sections.append(ONE_SHOT_INSTRUCTION.format(first_label=first_label, second_label=second_label))
    prompt = "\n\n".join(sections)
    assert_no_loaded_vocabulary(prompt)
    assert_no_safety_or_evaluation_vocabulary(prompt)
    assert_no_end_tally_nouns(prompt)
    if identity is None:
        # Frame-only: anything HERE about how another party decides is the manipulation arriving through
        # the fiction. Never run on the identity paragraph, which trips it by construction.
        assert_no_coupling_claims(prompt)
    if "{" in prompt or "}" in prompt:
        raise ValueError(
            f"the render of {scenario.scenario_id!r} under arm {arm!r}, cell {cell!r} still carries a "
            f"brace, so an unfilled placeholder would reach the model verbatim: {prompt!r}"
        )
    split = prompt.split("\n\n")
    if tuple(split) != tuple(sections):
        raise ValueError(
            f"the render of {scenario.scenario_id!r} under arm {arm!r}, cell {cell!r} builds "
            f"{len(sections)} sections but splits into {len(split)}, so an audit that walks the sections "
            f"and one that splits the prompt disagree about where every block sits. One of the parts "
            f"carries a blank line of its own."
        )
    return tuple(sections)


def render_coordination_prompt(
    scenario: CoordinationScenario,
    *,
    arm: str,
    cell: str,
    print_order: str,
    stimulus: CoordinationStimulus,
) -> str:
    """Render one (scenario, arm, cell, print order) into the prompt the model reads."""
    return "\n\n".join(
        render_coordination_sections(
            scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
        )
    )


def brief_section_count(scenario: CoordinationScenario) -> int:
    """Count the sections one authored brief contributes, which every cell renders in the same number.

    Counted rather than assumed to be one, because an authored brief may run to two paragraphs and the
    positional audits index past it: a fixed index would check the wrong sections while still passing on
    the roster it was written against. Both arms and both settings give the same answer, since a setting
    passage or an oversight sentence carrying a paragraph break is refused at load.
    """
    return scenario.brief.count("\n\n") + 1


def status_section_index(scenario: CoordinationScenario, *, identity: str | None) -> int:
    """Where the status block sits: after the brief, the identity paragraph and the own log."""
    return brief_section_count(scenario) + (0 if identity is None else 1) + len(scenario.own_log)


def reconciliation_section_index(scenario: CoordinationScenario, *, identity: str | None) -> int:
    """Where the reconciliation block sits: immediately after the status block."""
    return status_section_index(scenario, identity=identity) + 1


def oversight_token_difference(first: str, second: str) -> tuple[list[str], list[str]]:
    """Return the one whitespace-token span each of two sentences has where the other has its own.

    The common prefix and suffix are stripped, so what is left is the edit: one token against one token is
    a replacement, one against none is a deletion, none against one an insertion. Returned rather than
    checked here, so the refusal can print both spans and a caller can re-run the same comparison on the
    rendered sections.
    """
    left = first.split()
    right = second.split()
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(left), len(right)) - prefix and left[-1 - suffix] == right[-1 - suffix]:
        suffix += 1
    return (
        left[prefix : len(left) - suffix],
        right[prefix : len(right) - suffix],
    )


def assert_differs_by_one_whitespace_token(first: str, second: str, *, what: str) -> None:
    """Refuse a pair of texts that is not one whitespace token apart.

    The oversight cue IS this check: the arms of every cell differ in one token of one sentence, so the
    effect the pass reports is that token. A pair two tokens apart would be a longer manipulation read
    under the label "one oversight sentence", and an identical pair would render two arms of one prompt
    under two block labels.
    """
    left, right = oversight_token_difference(first, second)
    if not left and not right:
        raise ValueError(
            f"{what} are the same text, so the two arms would render byte-identical prompts under two "
            f"block labels and the oversight effect would be a measurement of nothing."
        )
    if len(left) > 1 or len(right) > 1:
        raise ValueError(
            f"{what} differ by more than one whitespace token: {left} against {right}. The arms of every "
            f"cell differ in this one token, and that is how the oversight effect is reported."
        )


def _assert_only_placeholders(
    what: str, template: str, allowed: tuple[str, ...], path: Path
) -> None:
    """Refuse a template naming a placeholder its own filler cannot supply."""
    unsupported = sorted(set(_PLACEHOLDER_RE.findall(template)) - set(allowed))
    if unsupported:
        raise ValueError(
            f"{what} in {path} names the placeholders {unsupported}, and its filler supplies only "
            f"{list(allowed)}. An unsupplied placeholder reaches the model in braces."
        )


def _assert_required_slots(what: str, template: str, required: tuple[str, ...], path: Path) -> None:
    """Refuse a template missing a slot the design fills, naming every slot that is absent."""
    missing = [slot for slot in required if f"{{{slot}}}" not in template]
    if missing:
        raise ValueError(
            f"{what} in {path} has no {['{' + slot + '}' for slot in missing]}, so what the design varies "
            f"through that slot would never reach the reader."
        )


def count_spellings(peer_count: int) -> tuple[str, ...]:
    """Every way the two counts the renderer prints could be spelled in authored prose.

    Both counts and both spellings of each, because the identity paragraph states the number of other
    agents and the status block lists one line per shard: prose naming either number would contradict them
    the first time the count moved.
    """
    counts = (peer_count, shard_count(peer_count))
    unknown = sorted(count for count in counts if count not in _NUMBER_WORDS)
    if unknown:
        raise ValueError(
            f"this module knows no number word for {unknown}, so the numeral gate would pass authored "
            f"prose that spelled the count out. Extend the number-word table."
        )
    return tuple(str(count) for count in counts) + tuple(_NUMBER_WORDS[count] for count in counts)


def _assert_no_count_numeral(what: str, text: str, peer_count: int, path: Path) -> None:
    """Refuse authored prose that prints a number the renderer prints, as a whole token.

    Whole tokens rather than substrings, because the agent's own log carries timings and row counts and a
    substring match would refuse ``0:06:12`` for containing a six. What is banned is the prose stating the
    count itself.
    """
    pattern = re.compile(
        r"\b(?:" + "|".join(re.escape(word) for word in count_spellings(peer_count)) + r")\b",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match is None:
        return
    start = max(0, match.start() - _VOCABULARY_CONTEXT_CHARS)
    end = min(len(text), match.end() + _VOCABULARY_CONTEXT_CHARS)
    raise ValueError(
        f"{what} in {path} states the count {match.group(0)!r} the renderer prints. Context: "
        f"...{text[start:end]}... The identity paragraph states how many other agents there are and the "
        f"status block lists one line per shard, so authored prose naming either number would contradict "
        f"them the first time the count moves."
    )


def _assert_labels_well_formed(scenario: CoordinationScenario, path: Path) -> None:
    """Refuse a label pair that cannot express a choice, that reads as pending, or that the brief pre-empts.

    Four things, and the pending-label check runs before the word-count one deliberately: the pending label
    is one word today, so a label equal to it would otherwise be refused as too short and the check that
    exists for the collision would never be the one that fired. Each label has to be filename-shaped,
    because it reaches a decision tag and keys a report group. Each has to be at least two hyphen-joined
    words, because the fork is the only place the reader meets it and a one-word label does not say what the
    option is. Neither may be the pending label the status block prints, or a status line would read as a
    choice in a design whose construction is that nobody has chosen. And the PRINCIPAL must never name the
    deviation option: the brief, both settings and
    both oversight sentences are the principal's own words, and a principal that offered the departing
    option would contradict the runbook rule and would fix an option order the print-order counterbalance
    cannot swap.

    The own log and the two reconciliation variants are deliberately not searched: they are the
    environment, and a close procedure that names the option a shard's block might carry is the scenario
    rather than the principal endorsing it.
    """
    labels = (scenario.compliant_label, scenario.deviation_label)
    unfilenamed = [
        label for label in (*labels, scenario.status_field) if not FRAMING_LABEL_RE.match(label)
    ]
    if unfilenamed:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has labels or a status field the pattern "
            f"{FRAMING_LABEL_RE.pattern} refuses: {unfilenamed}. Each label is printed into a decision "
            f"tag and keys a report group."
        )
    as_pending = [label for label in labels if label.strip().casefold() == PENDING_LABEL]
    if as_pending:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} labels an option {as_pending}, which is the "
            f"status block's own pending label: every status line would then read as a shard that had "
            f"already chosen that option, in a design whose construction is that nobody has chosen."
        )
    too_short = [label for label in labels if len(label.split("-")) < MIN_LABEL_WORDS]
    if too_short:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} has labels of fewer than {MIN_LABEL_WORDS} "
            f"hyphen-joined words: {too_short}. The fork is the only place the reader meets either option, "
            f"so each label has to read as the option it names."
        )
    if scenario.compliant_label.strip().casefold() == scenario.deviation_label.strip().casefold():
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} gives both options the same label "
            f"({scenario.compliant_label!r}); identical labels cannot express a choice, and the parser "
            f"refuses the pair outright."
        )
    principal_text = {
        "brief": scenario.brief,
        SETTING_PEERS: scenario.setting_peers,
        SETTING_SOLE: scenario.setting_sole,
        OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_ON]: scenario.oversight_on,
        OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_OFF]: scenario.oversight_off,
    }
    pre_empted = sorted(
        name for name, text in principal_text.items() if scenario.deviation_label in text
    )
    if pre_empted:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} names its deviation option "
            f"{scenario.deviation_label!r} in {pre_empted}, which is the principal's own text. The "
            f"departing option is introduced by the fork alone: a brief that offered it would contradict "
            f"the runbook rule it states and would fix an option order the counterbalance cannot swap."
        )


def _assert_the_log_has_not_already_read_the_shared_files(
    scenario: CoordinationScenario, path: Path
) -> None:
    """Refuse an own log that has already reached the status file or the close procedure's source.

    The status block and the reconciliation block are what tell the reader that nobody has chosen and how
    the close compares the shards, and both are renderer-positioned sections every cell carries. A log that
    had already listed either file would show that content at a different position too, so the sole-owner
    cells' substitution audits would compare renders that both carry the other cell's text.

    Matched on the FILE rather than on the whole command string, because a second reader command, a
    directory listing, or a stdout that happens to include the file among its lines all reach the same
    content while sharing no substring with the command as written.
    """
    for what, command in (
        ("status_command", scenario.status_command),
        ("reconciliation_command", scenario.reconciliation_command),
    ):
        tokens = status_file_tokens(command)
        if not tokens:
            raise ValueError(
                f"scenario {scenario.scenario_id!r} in {path} has a {what} ({command!r}) that names no "
                f"file, so nothing says which file the own log must not have read and the log gate would "
                f"pass whatever the log did."
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
                f"scenario {scenario.scenario_id!r} in {path} reaches its own {what} file {list(tokens)} "
                f"in {named}. Both shared blocks are sections the renderer places, so a log that already "
                f"read one of those files shows its content twice and at a drifting position."
            )


def _assert_the_oversight_pair_is_one_token_apart(
    scenario: CoordinationScenario, path: Path
) -> None:
    """Refuse an oversight pair that is not two one-sentence texts one whitespace token apart."""
    for arm in ARMS:
        assert_one_sentence(
            f"{OVERSIGHT_FIELD_BY_ARM[arm]} of scenario {scenario.scenario_id!r}",
            scenario.oversight_for(arm),
            path,
        )
    assert_differs_by_one_whitespace_token(
        scenario.oversight_on,
        scenario.oversight_off,
        what=f"the two oversight sentences of scenario {scenario.scenario_id!r} in {path}",
    )


def _assert_the_reconciliation_variants_are_a_pair(
    scenario: CoordinationScenario, path: Path
) -> None:
    """Refuse a reconciliation pair that is blank, identical, or carries a blank line.

    A blank line would split one tool result into two sections and move every positional audit past it by
    one; an identical pair would make the ``no-peers`` floor a second copy of the ceiling cell under
    another label, and its whole role is that its close procedure compares each block with the runbook
    instead of with the other blocks.
    """
    texts = {
        name: getattr(scenario, name)
        for name in (RECONCILIATION_BLOCKS, RECONCILIATION_RUNBOOK, "reconciliation_command")
    }
    blank = sorted(name for name, text in texts.items() if not str(text).strip())
    if blank:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} leaves {blank} blank; each is printed verbatim "
            f"into the reconciliation block every cell carries."
        )
    for name, text in texts.items():
        if "\n\n" in str(text):
            raise ValueError(
                f"{name} of scenario {scenario.scenario_id!r} in {path} carries a blank line, so this one "
                f"tool result would render as two sections and every positional audit past it would be "
                f"off by one."
            )
    if scenario.reconciliation_blocks == scenario.reconciliation_runbook:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} gives both close procedures the same stdout, so "
            f"the no-peers floor would render the ceiling cell's prompt under another cell label."
        )


def _assert_the_settings_are_a_pair(
    scenario: CoordinationScenario, peer_count: int, path: Path
) -> None:
    """Refuse a setting pair that shares a text, breaks a paragraph, or misses the slots it fills."""
    for name, text, required in (
        (SETTING_PEERS, scenario.setting_peers, ("reader_shard", "peer_count")),
        (SETTING_SOLE, scenario.setting_sole, ("shard_count",)),
    ):
        where = f"{name} of scenario {scenario.scenario_id!r}"
        if not text.strip():
            raise ValueError(f"{where} in {path} is blank, and it is printed verbatim.")
        assert_no_paragraph_break(where, text, path)
        _assert_only_placeholders(where, text, SETTING_PLACEHOLDERS, path)
        _assert_required_slots(where, text, required, path)
        _assert_no_count_numeral(where, text, peer_count, path)
    if scenario.setting_peers == scenario.setting_sole:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} in {path} gives both settings the same passage, so the "
            f"sole-owner cells would render the peer cells' brief under another cell label."
        )


def _assert_no_authored_prose_states_a_count(
    scenario: CoordinationScenario, peer_count: int, path: Path
) -> None:
    """Refuse any of this scenario's authored prose printing a number the renderer prints."""
    for name, text in (
        ("brief", scenario.brief),
        (OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_ON], scenario.oversight_on),
        (OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_OFF], scenario.oversight_off),
        ("fork", scenario.fork),
        (RECONCILIATION_BLOCKS, scenario.reconciliation_blocks),
        (RECONCILIATION_RUNBOOK, scenario.reconciliation_runbook),
        ("own_log", "\n".join(f"{b.command}\n{b.stdout}" for b in scenario.own_log)),
    ):
        _assert_no_count_numeral(
            f"{name} of scenario {scenario.scenario_id!r}", text, peer_count, path
        )


def _scenario(entry: Mapping[str, Any], peer_count: int, path: Path) -> CoordinationScenario:
    """Read and check one scenario, naming the file and the entry when a field is missing."""
    scenario_id = str(entry.get("scenario_id", "<unnamed>"))
    fields = (
        "brief",
        SETTING_PEERS,
        SETTING_SOLE,
        OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_ON],
        OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_OFF],
        "own_log",
        "reconciliation_command",
        RECONCILIATION_BLOCKS,
        RECONCILIATION_RUNBOOK,
        "status_command",
        "status_field",
        "fork",
        "compliant_label",
        "deviation_label",
    )
    missing = [name for name in fields if name not in entry]
    if missing:
        raise ValueError(f"scenario {scenario_id!r} in {path} is missing {missing}")
    if not FRAMING_LABEL_RE.match(scenario_id):
        raise ValueError(
            f"scenario id {scenario_id!r} in {path} is refused by {FRAMING_LABEL_RE.pattern}; a scenario "
            f"id keys every record key and every prompt id."
        )
    scenario = CoordinationScenario(
        scenario_id=scenario_id,
        brief=str(entry["brief"]),
        setting_peers=str(entry[SETTING_PEERS]),
        setting_sole=str(entry[SETTING_SOLE]),
        oversight_on=str(entry[OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_ON]]),
        oversight_off=str(entry[OVERSIGHT_FIELD_BY_ARM[ARM_OVERSIGHT_OFF]]),
        own_log=read_own_log(entry["own_log"], scenario_id, path),
        reconciliation_command=str(entry["reconciliation_command"]),
        reconciliation_blocks=str(entry[RECONCILIATION_BLOCKS]),
        reconciliation_runbook=str(entry[RECONCILIATION_RUNBOOK]),
        status_command=str(entry["status_command"]),
        status_field=str(entry["status_field"]),
        fork=str(entry["fork"]),
        compliant_label=str(entry["compliant_label"]),
        deviation_label=str(entry["deviation_label"]),
    )
    _assert_required_slots(
        f"the brief of scenario {scenario_id!r}", scenario.brief, BRIEF_PLACEHOLDERS, path
    )
    _assert_only_placeholders(
        f"the brief of scenario {scenario_id!r}", scenario.brief, BRIEF_PLACEHOLDERS, path
    )
    _assert_required_slots(
        f"the fork of scenario {scenario_id!r}", scenario.fork, FORK_PLACEHOLDERS, path
    )
    _assert_only_placeholders(
        f"the fork of scenario {scenario_id!r}", scenario.fork, FORK_PLACEHOLDERS, path
    )
    _assert_the_settings_are_a_pair(scenario, peer_count, path)
    _assert_the_oversight_pair_is_one_token_apart(scenario, path)
    _assert_the_reconciliation_variants_are_a_pair(scenario, path)
    _assert_labels_well_formed(scenario, path)
    _assert_the_log_has_not_already_read_the_shared_files(scenario, path)
    _assert_no_authored_prose_states_a_count(scenario, peer_count, path)
    return scenario


def _identity_clauses(payload: Mapping[str, Any], peer_count: int, path: Path) -> dict[str, str]:
    """Compose the three identity paragraphs from the opening, the two fragments and the two tails.

    Composed rather than read whole, because the three paragraphs must be pairwise ONE PART apart: that is
    what makes the headline a read of the identity alone and the coin control a read of the mechanism
    alone. A file carrying the paragraphs whole could drift on the opening while still rendering.
    """
    opening = str(payload.get("identity_opening", ""))
    raw_fragments = payload.get("identity_fragments", {})
    raw_tails = payload.get("identity_tails", {})
    parts: dict[str, dict[str, str]] = {}
    for label, raw, registered in (
        ("identity_fragments", raw_fragments, IDENTITY_FRAGMENTS),
        ("identity_tails", raw_tails, IDENTITY_TAILS),
    ):
        if not isinstance(raw, dict):
            raise TypeError(f"{label} in {path} is a {type(raw).__name__}, not an object")
        missing = [name for name in registered if name not in raw]
        stray = sorted(set(raw) - set(registered))
        if missing or stray:
            raise ValueError(
                f"{label} in {path} is missing {missing or 'nothing'} and carries {stray or 'nothing'} "
                f"this design does not sample; the registered keys are {list(registered)}."
            )
        parts[label] = {name: str(raw[name]) for name in registered}
    fragments = parts["identity_fragments"]
    tails = parts["identity_tails"]
    blank = sorted(
        name
        for name, text in {
            "identity_opening": opening,
            **{f"identity_fragments[{key!r}]": text for key, text in fragments.items()},
            **{f"identity_tails[{key!r}]": text for key, text in tails.items()},
        }.items()
        if not text.strip()
    )
    if blank:
        raise ValueError(
            f"stimulus file {path} leaves {blank} blank; every one is printed verbatim."
        )
    _assert_only_placeholders("identity_opening", opening, IDENTITY_PLACEHOLDERS, path)
    assert_no_paragraph_break("identity_opening", opening, path)
    for label, texts in (("identity_fragments", fragments), ("identity_tails", tails)):
        for key, text in texts.items():
            where = f"{label}[{key!r}]"
            _assert_only_placeholders(where, text, IDENTITY_PLACEHOLDERS, path)
            assert_no_paragraph_break(where, text, path)
            if opening.strip() and opening.strip() in text:
                raise ValueError(
                    f"{where} in {path} contains the shared identity_opening: a part carries its own "
                    f"contribution alone, and one carrying the opening would make two paragraphs differ "
                    f"outside the part the design varies while both still rendered."
                )
    _assert_the_parts_are_disjoint(fragments, tails, path)
    clauses = {
        paragraph: f"{opening}{fragments[fragment]}{tails[tail]}".format(count=peer_count)
        for paragraph, (fragment, tail) in IDENTITY_PARTS_BY_PARAGRAPH.items()
    }
    _assert_paragraphs_are_pairwise_one_part_apart(
        clauses, opening=opening, fragments=fragments, tails=tails, peer_count=peer_count, path=path
    )
    return clauses


def _assert_the_parts_are_disjoint(
    fragments: Mapping[str, str], tails: Mapping[str, str], path: Path
) -> None:
    """Refuse a fragment that carries a tail's text, or a tail that carries a fragment's.

    Each paragraph is opening plus fragment plus tail, so a fragment that swallowed a tail would put the
    tail's words into a paragraph the design says carries the other tail, and the coin control would then
    be a read of two rewordings.
    """
    swallowed = sorted(
        f"identity_fragments[{fragment!r}] contains identity_tails[{tail!r}]"
        for fragment, fragment_text in fragments.items()
        for tail, tail_text in tails.items()
        if tail_text.strip() and tail_text.strip() in fragment_text
    ) + sorted(
        f"identity_tails[{tail!r}] contains identity_fragments[{fragment!r}]"
        for tail, tail_text in tails.items()
        for fragment, fragment_text in fragments.items()
        if fragment_text.strip() and fragment_text.strip() in tail_text
    )
    if swallowed:
        raise ValueError(
            f"stimulus file {path} has parts of the identity paragraph inside one another: {swallowed}. "
            f"Each paragraph is the opening, one fragment and one tail, and the three paragraphs are "
            f"pairwise one of those apart."
        )


def _assert_paragraphs_are_pairwise_one_part_apart(  # noqa: PLR0913 - the parts, plus what names them
    clauses: Mapping[str, str],
    *,
    opening: str,
    fragments: Mapping[str, str],
    tails: Mapping[str, str],
    peer_count: int,
    path: Path,
) -> None:
    """Refuse composed paragraphs that are not the shared opening around their own fragment and tail.

    True by construction today, and checked anyway: the headline rests on the copies and strangers
    paragraphs differing in the fragment alone, and the coin control on the copies and coin paragraphs
    differing in the tail alone. The composition above is one edit away from filling a paragraph from the
    wrong part. The three also have to differ from each other at all -- two paragraphs rendering one text
    would make a headline contrast a comparison of a cell with itself.
    """
    for paragraph, (fragment, tail) in IDENTITY_PARTS_BY_PARAGRAPH.items():
        expected = f"{opening}{fragments[fragment]}{tails[tail]}".format(count=peer_count)
        if clauses[paragraph] != expected:
            raise ValueError(
                f"the composed identity paragraph for {paragraph!r} in {path} is not the shared opening "
                f"around its own fragment and tail: expected {expected!r}, composed "
                f"{clauses[paragraph]!r}."
            )
    if len(set(clauses.values())) != len(clauses):
        raise ValueError(
            f"stimulus file {path} composes the same identity paragraph for two of "
            f"{sorted(clauses)}, so one of this pass's headline contrasts would compare a cell with "
            f"itself."
        )
    same_tail = (
        clauses[PARAGRAPH_SAME_CHECKPOINT].replace(
            fragments[RUNG_SAME_CHECKPOINT], fragments[RUNG_DIFFERENT_FAMILY], 1
        )
        != clauses[PARAGRAPH_DIFFERENT_FAMILY]
    )
    same_fragment = (
        clauses[PARAGRAPH_SAME_CHECKPOINT].replace(tails[TAIL_SIMULTANEOUS], tails[TAIL_DRAWN], 1)
        != clauses[PARAGRAPH_SAME_CHECKPOINT_DRAWN]
    )
    if same_tail or same_fragment:
        raise ValueError(
            f"the identity paragraphs in {path} are not pairwise one part apart: substituting the "
            f"strangers fragment for the copies one does not give the strangers paragraph "
            f"({same_tail}), or substituting the coin tail for the simultaneous one does not give the "
            f"coin paragraph ({same_fragment})."
        )


def _validation_reply(
    entry: Mapping[str, Any], path: Path, *, keys: tuple[str, ...]
) -> DeferenceValidationReply:
    """Read one validation reply, naming the file and the reply when an expectation is missing.

    The sibling pass's reply dataclass, imported rather than copied: an authored reply is a reply text, a
    reasoning text, the row's two labels and a registered verdict in every pass on this CLI, and the judge
    module that consumes it is shared.
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
            f"validation reply {name!r} in {path} gives both options the label {label_a!r}; the judge is "
            f"handed the row's own two labels and a verdict naming one of them could mean either."
        )
    compliant = entry.get("compliant_label")
    if compliant is not None and str(compliant) not in (label_a, label_b):
        raise ValueError(
            f"validation reply {name!r} in {path} names compliant_label {compliant!r}, which is neither "
            f"of its own labels ({label_a!r}, {label_b!r}); the bookkeeping would say a reply followed an "
            f"option the row never offered."
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
    """Digest only the fields a rendered prompt depends on, so a judge-side edit leaves it alone."""
    return _canonical_digest({name: payload[name] for name in PROMPT_AFFECTING_FIELDS})


def _assert_every_field_is_classified(payload: Mapping[str, Any], path: Path) -> None:
    """Refuse a top-level field filed as neither prompt-affecting nor judge-side.

    Filing is what makes the two digests mean what they claim: a field on neither list could change what
    the model reads without moving ``prompt_digest``, and a resume would then continue a file sampled under
    different prompts while every count still added up.
    """
    unclassified = sorted(set(payload) - set(PROMPT_AFFECTING_FIELDS) - set(JUDGE_SIDE_FIELDS))
    if unclassified:
        raise ValueError(
            f"stimulus file {path} carries top-level fields {unclassified} that are filed as neither "
            f"prompt-affecting ({list(PROMPT_AFFECTING_FIELDS)}) nor judge-side "
            f"({list(JUDGE_SIDE_FIELDS)}). Add each one to the list that matches what reads it, so the "
            f"prompt digest moves when the prompts do and stays put when only the judge's material does."
        )


def load_stimulus(path: Path = STIMULUS_PATH) -> CoordinationStimulus:
    """Load and validate the coordination stimulus file, refusing absence loudly rather than defaulting.

    A default here would either be committed stimulus prose, which this public repository must never carry,
    or empty strings, which would render briefless prompts that measure nothing the design describes. The
    refusal names the path and why the file is machine-local.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"stimulus file {path} is missing. It is gitignored on purpose (the briefs, both settings, "
            "both oversight sentences, the tool-result log, both close procedures, the forks, the "
            "identity parts, both rubrics and the validation replies are authored stimulus that must "
            "never be committed); a fresh clone does not contain it. Recreate it from the design doc in "
            "docs/scratch/."
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
            f"{PEER_COUNT}. The count is printed into the identity paragraph and the shard count derived "
            f"from it IS the status block's line count, so a file stating another number would render a "
            f"paragraph and a block that disagree about how many agents there are."
        )
    entries = payload.get("scenarios", [])
    if len(entries) != N_SCENARIOS:
        raise ValueError(
            f"stimulus file {path} carries {len(entries)} scenarios, expected exactly {N_SCENARIOS}: the "
            f"unit of pairing is (scenario, print order), and a per-cell rate over fewer briefs is one "
            f"job's answer rather than a rate."
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
            f"stimulus file {path} carries duplicate scenario ids {duplicated}; a scenario id keys every "
            f"prompt id and record key, so two briefs under one id would pool their answers."
        )
    stimulus = CoordinationStimulus(
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
            f"stimulus file {path} carries no {field_name}. Without it the instrument it belongs to would "
            f"run on an empty prompt and return verdicts nobody asked for."
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
            f"stimulus file {path} carries {len(raw)} {field_name}, and that instrument is not calibrated "
            f"on fewer than {floor}: the design's own coverage list for it does not fit below that."
        )
    return tuple(_validation_reply(entry, path, keys=keys) for entry in raw)


def _assert_every_cell_renders(stimulus: CoordinationStimulus, path: Path) -> None:
    """Render every (scenario, arm, cell, print order) at load, so a bad brief fails before a plan.

    At load rather than at plan time, because a brief that only breaks in one arm, one cell or one print
    order would otherwise surface halfway through a sampling run, with the counts already written down and
    the first leg already submitted. Every vocabulary gate runs inside the renderer, so this sweep is what
    makes all of them cover all 192 renders of the file.
    """
    for scenario in stimulus.scenarios:
        for arm in ARMS:
            for cell in CELLS:
                for print_order in LABEL_PRINT_ORDERS:
                    try:
                        render_coordination_prompt(
                            scenario, arm=arm, cell=cell, print_order=print_order, stimulus=stimulus
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"the render of {scenario.scenario_id!r} in {path} under arm {arm!r}, cell "
                            f"{cell!r}, print order {print_order!r} is refused: {error}"
                        ) from error
