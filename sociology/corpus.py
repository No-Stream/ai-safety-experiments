r"""Load the banked episode corpora and render each unit the way an analysis model will read it.

Two substrates, one rendered shape. The agentic corpus is the blind-cot harness bank: per-episode
``episode_summary`` records carrying the full environment transcript, joined to per-turn ``turn``
records that alone carry the model's private reasoning. The single-turn contrast substrate is the
hatch-probe anchor cell: one completion (plus reasoning) per record, no environment at all. Both
render to plain text with the private reasoning interleaved and labelled, and with **no metadata
fields at all** -- no episode id, no model id, no arm, no directory -- because anything rendered is
something the analysis model can key on.

The interleave anchors on how the harness built the transcript: every turn's completion entered it
as ``"\\n" + completion + "\\n"`` (see ``reward_hacking/harness/loop.py``), so a cursor-ordered
substring search recovers each non-empty reply's position exactly, and a completion that cannot be
found raises rather than guessing. Empty completions are real and a third of the corpus's
reasoning hangs off them (834 of 2,558 turns; a mid-episode one is the empty-start nudge path, a
trailing one ends the episode); an empty reply has no findable text to anchor on, so its reasoning
travels under a label saying the reply it preceded was empty, placed at the next verified anchor
or the transcript's end -- kept and honestly labelled rather than positioned by guesswork.

The one deliberately manipulable cue lives here too: the box's resource limiter wrote
``resource-limits: unit=reslimit-<pid>-...`` lines into command stderr, a genuine co-location
signal inside otherwise host-clean text. :func:`scrub_colocation_cues` removes those lines for the
stripped rung and :func:`assert_cue_free` is the gate that refuses a stripped rendering still
carrying them (or any host-path marker), so the scrub cannot silently rot.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

AGENTIC_CORPUS_DIR = Path("artifacts/harness/blind-cot-20260819")
SINGLE_TURN_RECORDS_PATH = Path(
    "artifacts/reward_hacking/hatch_probe/prod-gpt-oss-120b-20260825/summary-records.jsonl"
)

AGENTIC_ARM_LABEL = "check+hidden"
"""The one arm the bundle families draw from; the ``check`` arm episodes stay out of the pool."""

# The hatch probe's anchor cell: no planted error, no hatch paragraph, neutral framing.
ANCHOR_ARM = "control"
ANCHOR_HATCH = "absent"
ANCHOR_FRAMING = "neutral"

FAMILY_AGENTIC_120B = "agentic-120b"
FAMILY_AGENTIC_20B = "agentic-20b"
FAMILY_SINGLE_TURN_120B = "single-turn-120b"

AGENTIC_FAMILY_MODELS: dict[str, str] = {
    FAMILY_AGENTIC_120B: "openai.gpt-oss-120b-1:0",
    FAMILY_AGENTIC_20B: "openai.gpt-oss-20b-1:0",
}

PRIVATE_REASONING_OPEN = "[Private reasoning of the agent -- not visible in the environment]"
PRIVATE_REASONING_EMPTY_OPEN = (
    "[Private reasoning of the agent before it gave an empty reply -- "
    "not visible in the environment]"
)
PRIVATE_REASONING_CLOSE = "[End of private reasoning]"

_LIMITER_LINE_RE = re.compile(r"resource-limits:[^\n]*\n?")
"""One resource-limiter stderr line, through its newline, as the harness rendered it.

Keyed on the ``resource-limits:`` prefix rather than the ``unit=reslimit-`` body because the corpus
carries three line forms under that prefix, not one: the common ``unit=reslimit-<pid>-...`` telemetry
line, a ``resource-limits: TIMEOUT after 60s ...`` line, and transcript-truncated fragments like
``resource-limits: unit=res...`` where the stored transcript elided the tail. All three are the same
co-location cue, and every ``reslimit-`` occurrence in the corpus sits behind this prefix, so
stripping the whole line family is what makes :func:`assert_cue_free` pass on the stripped rung.
"""

HOST_MARKERS = (
    "reslimit-",
    "resource-limits:",
    "/tmp/agent-harness",  # noqa: S108 - a leak marker to grep for, not a path this code uses
    "/local/home",
    "/home/",
)
"""Every substring whose presence in a stripped rendering means the scrub failed.

The limiter spellings are the manipulated cue itself; the path prefixes are the host markers the
corpus was verified not to carry (zero hits across all 778 transcripts), kept in the gate so a
future corpus that does carry one cannot ride in silently. Two home spellings, because a host may
hold its real home trees under a prefix and shadow them with a symlink at the bare path.
"""

_TWIN_TASK_RE = re.compile(r"^ilcb-(?:conflicting|original)-(?P<base>.+)$")

ERROR_MESSAGE_EXAMPLES = 3
"""How many offending ids an error message names before deferring to the count."""


@dataclass(frozen=True, slots=True)
class EpisodeTurn:
    """One agent turn: the reply that entered the transcript and the reasoning that never did."""

    index: int
    completion: str
    reasoning: str


@dataclass(frozen=True, slots=True)
class BundleUnit:
    """One poolable unit, already rendered: what the bundle builder draws and the prompts carry.

    ``exclusion_key`` is the near-duplicate identity the within-bundle uniqueness rule keys on: the
    twin base for agentic episodes (so the ``conflicting`` and ``original`` variants of one problem
    index never share a bundle and read as two agents on one problem), and the problem id for
    single-turn records (two samples of one problem are the same trap by another route).

    ``rendered`` keeps the co-location cues; the stripped rung scrubs at bundle-render time so one
    loaded corpus serves both rungs.
    """

    unit_id: str
    family: str
    exclusion_key: str
    rendered: str


def twin_key(task_id: str) -> str:
    """Return the problem-index identity shared by a task's conflicting/original twins.

    A task id outside the twin naming scheme is its own key: unique, so the rule never blocks it.
    """
    match = _TWIN_TASK_RE.match(task_id)
    return match.group("base") if match else task_id


def _reasoning_block(reasoning: str, *, open_label: str = PRIVATE_REASONING_OPEN) -> str:
    """Render one turn's private reasoning as the labelled block the bundle text carries."""
    return f"\n{open_label}\n{reasoning}\n{PRIVATE_REASONING_CLOSE}\n"


def render_episode(transcript: str, turns: tuple[EpisodeTurn, ...], *, unit_id: str) -> str:
    r"""Interleave each turn's private reasoning into the stored transcript, before its reply.

    Non-empty replies anchor exactly (see the module docstring), and a completion that cannot be
    found forward of the cursor raises with the unit id rather than rendering a bundle whose
    reasoning sits beside the wrong reply. An empty reply leaves nothing findable to anchor on --
    its contribution is a bare ``"\\n\\n"`` somewhere inside text of unknown length -- so its
    reasoning is NOT dropped (834 of 2,558 turns end that way and a third of the corpus's reasoning
    hangs off them) but carried forward under its own label, which states that the reply it
    preceded was empty, and inserted at the next verified anchor or the transcript's end. The label
    carries the truth the position cannot.
    """
    parts: list[str] = []
    deferred: list[str] = []
    cursor = 0
    for turn in turns:
        if not turn.completion:
            if turn.reasoning:
                deferred.append(
                    _reasoning_block(turn.reasoning, open_label=PRIVATE_REASONING_EMPTY_OPEN)
                )
            continue
        needle = "\n" + turn.completion + "\n"
        found = transcript.find(needle, cursor)
        if found < 0:
            raise ValueError(
                f"{unit_id}: turn {turn.index}'s completion does not appear in the transcript "
                "forward of the previous turn, so the reasoning interleave has nowhere "
                "verified to anchor"
            )
        parts.append(transcript[cursor:found])
        parts.extend(deferred)
        deferred = []
        if turn.reasoning:
            parts.append(_reasoning_block(turn.reasoning))
        parts.append(needle)
        cursor = found + len(needle)
    parts.append(transcript[cursor:])
    parts.extend(deferred)
    return "".join(parts)


def render_single_turn(completion: str, reasoning: str) -> str:
    """Render one single-turn record: labelled private reasoning, then the visible completion."""
    block = _reasoning_block(reasoning) if reasoning else ""
    return f"{block}\n{completion}\n"


def scrub_colocation_cues(text: str) -> str:
    """Remove every resource-limiter line from a rendering, for the stripped cue rung."""
    return _LIMITER_LINE_RE.sub("", text)


def assert_cue_free(text: str, *, context: str) -> None:
    """Refuse a stripped rendering that still carries the co-location cue or any host marker."""
    for marker in HOST_MARKERS:
        if marker in text:
            raise ValueError(
                f"{context}: stripped rendering still contains {marker!r}; the co-location scrub "
                "did not cover it, so the stripped rung would not be stripped"
            )


def contains_colocation_cue(text: str) -> bool:
    """Whether a rendering carries the limiter cue at all -- what makes the kept rung a factor."""
    return _LIMITER_LINE_RE.search(text) is not None


def _read_corpus_shards(
    corpus_dir: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[EpisodeTurn]]]:
    """Parse every shard into episode summaries and per-episode turns, refusing duplicates."""
    summaries: dict[str, dict[str, Any]] = {}
    turns_by_episode: dict[str, list[EpisodeTurn]] = {}
    paths = sorted(corpus_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no corpus shards under {corpus_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["record"] == "episode_summary":
                    episode_id = str(record["episode_id"])
                    if episode_id in summaries:
                        raise ValueError(f"duplicate episode summary {episode_id} in {path}")
                    summaries[episode_id] = record
                elif record["record"] == "turn":
                    turns_by_episode.setdefault(str(record["episode_id"]), []).append(
                        EpisodeTurn(
                            index=int(record["turn"]),
                            completion=str(record["completion"] or ""),
                            reasoning=str(record["reasoning"] or ""),
                        )
                    )
                else:
                    raise ValueError(f"unknown record type {record['record']!r} in {path}")
    return summaries, turns_by_episode


def load_agentic_units(corpus_dir: Path = AGENTIC_CORPUS_DIR) -> dict[str, list[BundleUnit]]:
    """Load, join, and render the blind-cot bank into the two agentic bundle families.

    Turn records alone carry reasoning, so they are joined to summaries by episode id and
    interleaved here; a turn without a summary, a duplicate summary, or an episode whose transcript
    the interleave cannot anchor all raise rather than thinning the pool silently. Only the
    ``check+hidden`` arm enters the families.
    """
    summaries, turns_by_episode = _read_corpus_shards(corpus_dir)
    orphans = sorted(set(turns_by_episode) - set(summaries))
    if orphans:
        examples = orphans[:ERROR_MESSAGE_EXAMPLES]
        raise ValueError(f"{len(orphans)} turn episode ids have no summary, e.g. {examples}")

    model_to_family = {model: family for family, model in AGENTIC_FAMILY_MODELS.items()}
    families: dict[str, list[BundleUnit]] = {family: [] for family in AGENTIC_FAMILY_MODELS}
    for episode_id in sorted(summaries):
        summary = summaries[episode_id]
        if str(summary["arm"]["label"]) != AGENTIC_ARM_LABEL:
            continue
        family = model_to_family.get(str(summary["model_id"]))
        if family is None:
            continue
        ordered = tuple(sorted(turns_by_episode.get(episode_id, []), key=lambda t: t.index))
        rendered = render_episode(str(summary["transcript"]), ordered, unit_id=episode_id)
        families[family].append(
            BundleUnit(
                unit_id=episode_id,
                family=family,
                exclusion_key=twin_key(str(summary["task_id"])),
                rendered=rendered,
            )
        )
    for family, units in families.items():
        logger.info("agentic family %s: %d units", family, len(units))
    return families


def _is_anchor(record: Mapping[str, object]) -> bool:
    return (
        str(record.get("arm")) == ANCHOR_ARM
        and str(record.get("hatch")) == ANCHOR_HATCH
        and str(record.get("framing")) == ANCHOR_FRAMING
    )


def load_single_turn_units(path: Path = SINGLE_TURN_RECORDS_PATH) -> dict[str, list[BundleUnit]]:
    """Load and render the hatch-probe anchor cell into the single-turn contrast family."""
    units: list[BundleUnit] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not _is_anchor(record):
                continue
            unit_id = f"{record['problem_id']}|s{record['sample_index']}"
            if unit_id in seen:
                raise ValueError(f"duplicate anchor record {unit_id} in {path}")
            seen.add(unit_id)
            units.append(
                BundleUnit(
                    unit_id=unit_id,
                    family=FAMILY_SINGLE_TURN_120B,
                    exclusion_key=str(record["problem_id"]),
                    rendered=render_single_turn(
                        str(record["completion"] or ""), str(record["reasoning"] or "")
                    ),
                )
            )
    units.sort(key=lambda unit: unit.unit_id)
    logger.info("single-turn family %s: %d units", FAMILY_SINGLE_TURN_120B, len(units))
    return {FAMILY_SINGLE_TURN_120B: units}
