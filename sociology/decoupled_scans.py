"""Deterministic scans over decoupled-ladder replies: the parsed action and the truncation tripwires.

Layer one of the scoring stack, running before and independently of the blind judge. The judge is
the instrument of record here -- regex-vs-judge extraction agreement on this repo's own traces sits
at 0.63 -- so these scans exist to give every judged rate a deterministic counterpart to disagree
with, and a disagreement is an instrument finding to report rather than something to smooth.

Three properties are load-bearing:

- **A parse failure is None, never a clamp.** A reply that argues its way to a label without
  emitting the tag is unparsed, not a defection. Clamping it would move a real refusal-to-answer
  into whichever bucket happened to be the default, and the counterpart-identity effect being
  measured here is a few points wide.
- **Truncation is reported per cell, because it is the design's tripwire.** The decoupled cells
  deliberate roughly twice as long as the coupled ones, so a flat reply cap bites them first, and a
  truncated reply reads downstream as a model that declined to decide.
- **A classifier refusal is its own bucket, not a parse failure and not an empty reply.** A record
  Bedrock stopped with ``content_filtered`` is a reply the provider's safety classifier refused or
  cut short; the model never answered, so counting it as unparsed would read a refused model as an
  evasive one, and dropping it as empty would hide that a whole row was refused. Opus 5 came back
  this way on ~97% of this pass's records (2026-09-02), which is what made the bucket necessary.

The mirror-vocabulary regex is labelled weak in its own name and its own docstring: it cannot tell
the reply's own voice from a hypothetical it raises and rejects, and MiniMax traces have already
shown that assert-then-retract swings a rate of this kind by more than an order of magnitude. It is
a lead for the judge's per-reply reading, never a rate to headline.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING, Any

from games.parsing import parse_action, strip_thinking
from games.payoffs import COOPERATE, DEFECT
from reward_hacking.model_backend import STOP_REASON_MAX_TOKENS
from reward_hacking.trace import refuse_tracked_trace_path
from sociology.records import load_replies

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)

SCANS_FILENAME = "scans.jsonl"
REPLY_FILE_GLOB = "replies--*.jsonl"

STOP_REASON_CONTENT_FILTERED = "content_filtered"
"""Converse's stop reason for a reply the provider's safety classifier refused or cut short.

Spelled Converse's way for the same reason the shared stop-reason constants in
:mod:`reward_hacking.model_backend` are: the field is pooled across transports. It lives here rather
than beside them because no transport of ours maps anything onto it -- it arrives verbatim from
Bedrock on the batch and live paths alike -- and this scan is its only reader. Anthropic's native
``refusal`` stop reason is rendered as this on Converse, with the category invisible on that path.
"""

MIRROR_REGEX = re.compile(
    r"same reasoning"
    r"|identical reasoning"
    r"|same way I|same way that I"
    r"|whatever I (?:choose|pick|decide|do)"
    r"|what I (?:choose|pick|decide|do)"
    r"|(?:it|they) will do the same"
    r"|(?:mirror|mirrors|mirroring)\b"
    r"|same conclusion as (?:me|mine)"
    r"|reach(?:es)? the same",
    re.IGNORECASE,
)
"""WEAK mirror-vocabulary probe: surface phrases, blind to whether the reply endorses or rejects them.

Counted per reply and reported beside the judge's ``counterpart_reasoning`` verdict so the two can
disagree in the open. Never a mirror rate on its own -- see the module docstring for why a phrase
count cannot be one.
"""

SCAN_CARRIES = "the parsed action of every reply, keyed to its prompt"

ACTION_NAMES: dict[str, str] = {COOPERATE: "cooperate", DEFECT: "defect"}
"""The parser's one-letter canonical actions spelled out, because a scans row is read by humans."""


def is_refused(record: Mapping[str, Any]) -> bool:
    """Whether the provider's safety classifier refused this reply, whatever text survived the cut."""
    return record.get("stop_reason") == STOP_REASON_CONTENT_FILTERED


def scan_reply(record: Mapping[str, Any]) -> dict[str, Any]:
    """Scan one reply record into its scan row, without touching the reply's stored labels.

    ``strip_thinking`` first, then ``parse_action`` over what remains: a completion whose thinking
    ran into the cap has no visible answer at all, and ``truncated_thinking`` reports that
    separately, because a rising rate there means the cap is shaping behaviour rather than the
    payoffs.

    A refused record (:func:`is_refused`) scans as ``refused=True`` with no action, even when an
    action tag survived ahead of the classifier's cut: what the classifier let through is a
    classifier-selected fragment, not the model's answer. The label triple is still validated for
    it, so ``errored`` means one thing in every bucket.
    """
    reply = str(record.get("reply") or "")
    reasoning = str(record.get("reasoning") or "")
    refused = is_refused(record)
    visible, truncated_thinking = strip_thinking(reply)
    action = parse_action(
        visible,
        label_a=str(record["label_a"]),
        label_b=str(record["label_b"]),
        coop_label=str(record["coop_label"]),
    )
    parsed = None if refused else action
    return {
        "key": str(record["key"]),
        "model_id": record.get("model_id"),
        "cell": record.get("cell"),
        "block": record.get("block"),
        "game_id": record.get("game_id"),
        "prompt_id": record.get("prompt_id"),
        "label_print_order": record.get("label_print_order"),
        "reasoning_effort": record.get("reasoning_effort"),
        "sitting": record.get("sitting"),
        "draw": record.get("draw"),
        "refused": refused,
        "parsed": parsed is not None,
        "action": None if parsed is None else ACTION_NAMES[parsed],
        "cooperated": None if parsed is None else parsed == COOPERATE,
        "truncated": record.get("stop_reason") == STOP_REASON_MAX_TOKENS,
        "truncated_thinking": truncated_thinking,
        "incomplete": bool(record.get("incomplete")),
        "reply_chars": len(reply),
        "reasoning_chars": len(reasoning),
        "output_tokens": record.get("output_tokens"),
        "mirror_regex_hits": len(MIRROR_REGEX.findall(visible or reply)),
        "stimulus_digest": record.get("stimulus_digest"),
    }


def load_run_replies(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Load every leg's replies into one map, refusing a key that appears in two legs' files.

    One file per leg keeps concurrent legs off a shared flat file; merging them here is what lets a
    scan, a judge pass and a readout see the whole run. A key in two files would mean two legs
    claimed one identity, so it refuses rather than last-winning.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob(REPLY_FILE_GLOB)):
        for key, row in load_replies(path).items():
            if key in merged:
                raise ValueError(
                    f"reply key {key} appears in {path.name} and in an earlier leg file of "
                    f"{run_dir}; two legs cannot claim one record identity."
                )
            merged[key] = row
    return merged


def stimulus_digest_counts(replies: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    """How many of these reply rows carry each stimulus PROMPT digest, in sorted order."""
    counted: Counter[str] = Counter(str(row.get("stimulus_digest")) for row in replies.values())
    return dict(sorted(counted.items()))


def refuse_pooled_stimulus_mixture(
    replies: Mapping[str, Mapping[str, Any]], *, run_dir: Path, what: str
) -> str:
    """Refuse to pool reply rows sampled under two different stimulus versions; return the one digest.

    Reachable only by editing prompt-affecting stimulus mid-run: the live resume guard compares the digest
    of keys already on disk, and a leg that has not been sampled yet has no keys, so sampling one model's
    leg, editing a frame and sampling the next lands two stimuli in one directory with nothing objecting.
    Every instrument that pools rows across legs -- the judge, the intent check, the cross-judge subset --
    then reports one rate over two experiments, and the only thing that ever noticed was a note in a
    readout, printed after the money was spent.

    The scan is deliberately not gated: its rows carry the digest one per row, which is how an operator
    finds out WHICH legs came from which file. The remedy the message names is per leg, because the reply
    files are per leg.
    """
    counts = stimulus_digest_counts(replies)
    if len(counts) > 1:
        raise ValueError(
            f"the replies in {run_dir} were sampled under {len(counts)} different stimulus prompt "
            f"digests ({counts}), so {what} would pool rows from two experiments into one rate. Run "
            f"`scan` (which is not gated, and writes the digest on every row) to see which legs came "
            f"from which file, then point --run-dir at a directory holding only the legs of one digest, "
            f"or restore the stimulus file the other legs were sampled under and re-sample."
        )
    return next(iter(counts), "")


def scan_run(run_dir: Path) -> dict[str, Any]:
    """Scan every reply on disk into a full-rewrite scans file; return the denominators.

    A full rewrite rather than an append: the scans are a pure function of the replies plus this
    code, so re-running after a code change must replace the old rows rather than doubling them. The
    rows land in a sibling temp file that is renamed into place, so a judge or readout that opens the
    file mid-rewrite reads the old rows or the new ones, never a prefix of the new ones.

    The accounting, every count with what it is out of:

    - ``examined == scanned + skipped_empty + errored``. ``scanned`` rows are the ones written to the
      file; ``skipped_empty`` is a reply with no text that the classifier did NOT refuse (a reasoning
      budget that consumed the whole reply, typically); ``errored`` counts records whose own stored
      labels cannot express a choice -- a corrupt row, which is a finding with a key attached rather
      than a reason to abandon the other 30,000 rows.
    - ``scanned == parsed + parse_failed + refused``. A refused row (``content_filtered``, text or
      no text) is written with ``refused=True`` and no action, and is neither a parse nor a parse
      failure: the model never answered.
    """
    replies = load_run_replies(run_dir)
    out_path = run_dir / SCANS_FILENAME
    refuse_tracked_trace_path(out_path, carries=SCAN_CARRIES)
    rows: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    skipped_empty = 0
    for key in sorted(replies):
        record = replies[key]
        if not is_refused(record) and not str(record.get("reply") or "").strip():
            skipped_empty += 1
            continue
        try:
            rows.append(scan_reply(record))
        except ValueError as error:
            errors[key] = str(error)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = out_path.with_name(out_path.name + ".tmp")
    with staging_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    staging_path.replace(out_path)
    refused = sum(1 for row in rows if row["refused"])
    parsed = sum(1 for row in rows if row["parsed"])
    totals: dict[str, Any] = {
        "examined": len(replies),
        "scanned": len(rows),
        "skipped_empty": skipped_empty,
        "errored": len(errors),
        "errors": errors,
        "refused": refused,
        "parsed": parsed,
        "parse_failed": len(rows) - parsed - refused,
        "cooperated": sum(1 for row in rows if row["cooperated"]),
        "truncated": sum(1 for row in rows if row["truncated"]),
        "truncated_thinking": sum(1 for row in rows if row["truncated_thinking"]),
        "incomplete": sum(1 for row in rows if row["incomplete"]),
        "with_mirror_regex_hits": sum(1 for row in rows if int(row["mirror_regex_hits"]) > 0),
    }
    logger.info(
        "decoupled-ladder scans: %s", json.dumps({k: v for k, v in totals.items() if k != "errors"})
    )
    return totals
