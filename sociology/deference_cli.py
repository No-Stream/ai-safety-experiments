"""The deference and coordination CLI: build a plan, price it, smoke it, sample it, scan it, judge it.

One subcommand per step of each design's order of work, one required ``--pass`` naming which of the two
passes the command operates on, and one ``--leg`` per batch job or live pass. Nothing here decides anything:
a :class:`PassBinding` carries the pass's stimulus loader, its leg table, its structural audits and its two
judge schemas, and every subcommand takes that binding rather than reaching for a module -- so the pass is
selected once, at the command line, and no code path can read one pass's legs while writing into another's
run directory.

``--pass`` is required and never defaulted, for the reason the transfer CLI's is: the two passes share this
surface, and a default would let a command re-render, re-price or re-submit the finished pass because nobody
typed the flag.

The tables live in :mod:`sociology.deference_plan` and :mod:`sociology.coordination_plan`, the briefs and
the renderers in :mod:`sociology.deference_stimulus` and :mod:`sociology.coordination_stimulus`, the
deterministic scans in :mod:`sociology.deference_scans` (shared, because both passes' records carry the same
label fields), the instruments in :mod:`sociology.deference_judge` and :mod:`sociology.coordination_judge`,
and this module is the operator surface over them.

Five habits this repository has paid for are enforced here rather than left to whoever runs it:

- **The plan is written down before any model call.** ``build`` renders every planned prompt, runs the
  one-inserted-paragraph audit over all of them plus the four structural audits (the report as one added
  block, the arms differing in one constraint sentence, the print orders moving the fork and the
  instruction alone, and the peer count agreeing between the paragraph and the report), and writes
  ``plan.json`` with the per-leg record counts and both digests. A run whose plan was never written is a
  run nobody can say what it intended to sample.
- **Batch handles are guarded and reused, never re-submitted.** The tracked-path refusal runs BEFORE the
  submit -- a refusal afterwards would orphan a job already paid for -- and a handle already on disk is
  reused only after the five-way check that it describes THIS invocation's job.
- **Resume is by content-derived key, and by content.** Live passes skip keys already on disk, count them
  separately from what ran, and refuse a key whose record answers a different prompt or was rendered from
  different prompt material (the stimulus PROMPT digest, so a rubric edit is not a refusal); collects skip
  keys already written and count those separately too, and refuse a job whose submit recorded a different
  prompt digest than the stimulus now loaded.
- **Neither judge runs before it has been validated.** The rubric of record has that gate because its
  rates are the pass's findings; the intent check needs it more, because its verdicts are used to correct
  which option a reply is counted as having taken rather than only to report a rate.
- **The summary is written LAST**, as the completion marker for every subcommand.

Every artifact lands under ``--run-dir``, which defaults under ``artifacts/`` because the replies carry the
authored briefs inside their prompts' digests and the judge's quoted evidence verbatim; the writers refuse a
git-tracked destination outright.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from reward_hacking.bedrock_batch import (
    BATCH_TIMEOUT_SECONDS,
    HANDLE_CARRIES,
    BatchJobHandle,
    BedrockBatchBackend,
    cell_digest,
    prompt_digest,
)
from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_CONCURRENCY,
    BedrockBackend,
    BedrockSamplingConfig,
    is_incomplete_stop_reason,
)
from reward_hacking.trace import refuse_tracked_trace_path
from sociology import (
    coordination_judge,
    coordination_plan,
    coordination_stimulus,
    deference_plan,
    deference_stimulus,
)
from sociology import deference_judge as judge_module
from sociology import deference_scans as scans_module
from sociology.decoupled_scans import load_run_replies, refuse_pooled_stimulus_mixture
from sociology.deference_plan import (
    LUNA_MODEL_ID,
    MAX_TOKENS,
    SITTING_A,
    SONNET_MODEL_ID,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    Cell,
    Leg,
    PlannedCall,
    batch_price_per_mtok,
    refuse_below_batch_floor,
)
from sociology.model_stub import ScriptedDetailedBackend
from sociology.records import (
    append_replies,
    completion_telemetry,
    load_replies,
    make_live_backend_factory,
    record_prompt_digest,
    refuse_changed_resume,
    run_live_calls,
    write_summary,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend
    from sociology.records import LiveBackendFactory

logger = logging.getLogger(__name__)

PLAN_FILENAME = "plan.json"
JUDGE_VALIDATION_LABEL = "judge-validation"
JUDGE_VALIDATION_SUMMARY = f"summary-{JUDGE_VALIDATION_LABEL}.json"
INTENT_VALIDATION_LABEL = "intent-check-validation"
INTENT_VALIDATION_SUMMARY = f"summary-{INTENT_VALIDATION_LABEL}.json"
INTENT_CHECKED_FILENAME = "intent-checked.jsonl"

CHARS_PER_TOKEN_ESTIMATE = 3.5
"""Conservative chars-per-token for the dry run's input estimate; measured by the smoke afterwards."""

ASSUMED_OUTPUT_TOKENS_PER_CALL = 2_000
"""Dry-run output assumption per reply, stated rather than hidden inside a total.

Anchored on the previous passes' measured per-call output at default effort, rounded up because Claude
bills adaptive thinking against the same budget. The smoke reports measured chars-per-token and the
collects report real usage, so this number is only ever load-bearing before the first record exists.
"""

SMOKE_ROWS_PER_CELL = 2
SMOKE_SCRIPTED_CALLS = 4
"""Four rather than two, so the offline smoke covers both print orders in both directions."""

DEFAULT_JUDGE_CONCURRENCY = DEFAULT_BEDROCK_CONCURRENCY
"""Judge calls in flight when ``--concurrency`` is not given: the transport's own default.

A throughput knob rather than a sampling one, so raising it changes wall clock and nothing about what is
measured. It is stamped into every judge summary so a slow or throttled pass can be read back to the number
it actually ran at.
"""

SMOKE_LIVE_MODEL_ID = LUNA_MODEL_ID
"""The live smoke's model on both passes: the cheapest live row on either roster, so the whole path is
exercised for cents.

Its reasoning comes back encrypted, which the smoke does not care about -- what a live smoke checks is
that the transport, the resume loop, the scans and both judges run end to end against real endpoints.
"""

_ACTION_TAG_RE = re.compile(r"<action>([^<>\n]+)</action>")


@dataclass(frozen=True, slots=True)
class LivePrice:
    """One live row's (input, output) $/Mtok and where the figure came from.

    ``verified`` is the live-path twin of the roster's ``batch_price_verified``: false means the number was
    reasoned to rather than read, and every dry-run line derived from it says so.
    """

    per_mtok: tuple[float, float]
    provenance: str
    verified: bool


LIVE_PRICES: dict[str, LivePrice] = {
    LUNA_MODEL_ID: LivePrice((0.20, 1.20), "verified 2026-09-02 handoff", verified=True),
    SONNET_MODEL_ID: LivePrice(
        (3.0, 15.0),
        "UNVERIFIED: the public list price, not read from the Price List API",
        verified=False,
    ),
}
"""$/Mtok for every live row; the dry run refuses at import if the live roster and this table drift."""


def _assert_every_live_row_is_priced() -> None:
    """Refuse at import if a live row of EITHER pass has no price line, or a price line names no live row.

    Both rosters at once, because the dry run prices whichever pass it was given and an unpriced row would
    raise a ``KeyError`` in the middle of a report rather than at import.
    """
    live = {*deference_plan.LIVE_MODEL_IDS, *coordination_plan.LIVE_MODEL_IDS}
    unpriced = sorted(live - set(LIVE_PRICES))
    stray = sorted(set(LIVE_PRICES) - live)
    if unpriced or stray:
        raise RuntimeError(
            f"LIVE_PRICES and the two live rosters disagree: live rows without a price "
            f"{unpriced or 'none'}; priced ids that are not live rows {stray or 'none'}."
        )


_assert_every_live_row_is_priced()


class PassStimulus(judge_module.JudgedStimulus, Protocol):
    """What every subcommand needs off a loaded stimulus, whichever pass loaded it.

    The two judge-side rubrics and their authored cases come from
    :class:`~sociology.deference_judge.JudgedStimulus`; what this adds is the three fields the plan header
    and every resume guard read. ``peer_count`` is on both files -- the deference pass states how many other
    agents a status report lists, the coordination pass how many other agents hold a shard -- and it lands in
    ``plan.json`` so a plan says what group size it planned.
    """

    @property
    def digest(self) -> str:
        """The whole-payload digest: it moves on any edit, rubrics included."""
        ...

    @property
    def prompt_digest(self) -> str:
        """The prompt-affecting digest: what a reply row is stamped with and a resume compares."""
        ...

    @property
    def peer_count(self) -> int:
        """How many other agents the design names."""
        ...


@dataclass(frozen=True, slots=True)
class PassBinding:
    """One pass's whole surface: its stimulus, its legs, its audits, its instruments and its artifacts.

    A binding rather than module-level imports, because this CLI now serves two passes over one operator
    surface and every subcommand has to be told which. The alternative the design considered and rejected
    was a sibling ``coordination_cli.py``: the CLI has no pass-specific logic beyond which modules it binds,
    the transfer and deference CLIs are already two copies of one surface (the ``8002a6b`` digest-guard
    hardening had to be applied to both), and a third copy would be a third thing to keep in step.

    The four callables take their pass's own stimulus type, which no single annotation here can name -- the
    loaders return different classes -- so they are typed loosely and the binding is what guarantees the
    pass's loader and the pass's plan module are the pair that belong together. Nothing else in this module
    reaches for a plan module by name.
    """

    pass_id: str
    load_stimulus: Callable[[], PassStimulus]
    stimulus_version: str
    legs: tuple[Leg, ...]
    all_legs: tuple[Leg, ...]
    cells_by_block: Mapping[str, tuple[Cell, ...]]
    rows_per_cell: int
    draws: int
    batch_prefix: str
    batch_pass_token: str
    default_run_dir: str
    reply_carries: str
    planned_calls: Callable[[Leg, Any], list[PlannedCall]]
    audit_planned_calls: Callable[[Sequence[PlannedCall], Any], int]
    audits: Callable[[Any], dict[str, int]]
    judge: judge_module.PassJudge
    smoke_model_id: str

    @property
    def blocks(self) -> tuple[str, ...]:
        """Name this pass's block ids, in table order."""
        return tuple(self.cells_by_block)

    def leg_for(self, leg_id: str) -> Leg:
        """Look up one of THIS pass's legs, listing the pass's table when the id is not in it.

        Per binding rather than across both passes: a leg id of the other pass is a mistyped ``--pass``, and
        resolving it here would submit that pass's job into this pass's run directory.
        """
        legs = {leg.leg_id: leg for leg in self.all_legs}
        leg = legs.get(leg_id)
        if leg is None:
            raise ValueError(
                f"{leg_id!r} is not a leg of the {self.pass_id!r} pass. Known legs: "
                f"{', '.join(sorted(legs))}."
            )
        return leg

    def smoke_leg(self, transport: str, block: str) -> Leg:
        """Build the leg the smoke renders from: one block of the smoke model, on its transport."""
        if block not in self.cells_by_block:
            raise ValueError(
                f"{block!r} is not a block of the {self.pass_id!r} pass; its blocks are "
                f"{list(self.blocks)}."
            )
        return Leg(
            self.smoke_model_id,
            transport,
            block,
            self.cells_by_block[block],
            None,
            SITTING_A,
            batch_prefix=self.batch_prefix,
            batch_pass_token=self.batch_pass_token,
        )


DEFERENCE_BINDING = PassBinding(
    pass_id=deference_plan.PASS_ID,
    load_stimulus=deference_stimulus.load_stimulus,
    stimulus_version=deference_stimulus.STIMULUS_VERSION,
    legs=deference_plan.production_legs(),
    all_legs=deference_plan.LEGS,
    cells_by_block=deference_plan.CELLS_BY_BLOCK,
    rows_per_cell=deference_plan.ROWS_PER_CELL,
    draws=deference_plan.DRAWS,
    batch_prefix=deference_plan.DEFERENCE_BATCH_PREFIX,
    batch_pass_token=deference_plan.BATCH_PASS_TOKEN,
    default_run_dir=deference_plan.DEFAULT_RUN_DIR,
    reply_carries="verbatim model replies to the authored deference briefs",
    planned_calls=deference_plan.planned_calls_for_leg,
    audit_planned_calls=deference_plan.audit_planned_calls,
    audits=deference_plan.audits,
    judge=judge_module.DEFERENCE_JUDGE,
    smoke_model_id=SMOKE_LIVE_MODEL_ID,
)
"""The finished deference slice: four peer conditions in two arms, plus the floor.

Bound rather than rewritten, and the pin test in the suite renders one of its cells and compares the whole
digest surface against the values this pass was sampled under: the ``--pass`` refactor had to leave it
byte-identical, because its production run directory is on disk and its numbers are indexed.
"""

COORDINATION_BINDING = PassBinding(
    pass_id=coordination_plan.PASS_ID,
    load_stimulus=coordination_stimulus.load_stimulus,
    stimulus_version=coordination_stimulus.STIMULUS_VERSION,
    legs=coordination_plan.production_legs(),
    all_legs=coordination_plan.LEGS,
    cells_by_block=coordination_plan.CELLS_BY_BLOCK,
    rows_per_cell=coordination_plan.ROWS_PER_CELL,
    draws=coordination_plan.DRAWS,
    batch_prefix=coordination_plan.COORDINATION_BATCH_PREFIX,
    batch_pass_token=coordination_plan.BATCH_PASS_TOKEN,
    default_run_dir=coordination_plan.DEFAULT_RUN_DIR,
    reply_carries="verbatim model replies to the authored coordination briefs",
    planned_calls=coordination_plan.planned_calls_for_leg,
    audit_planned_calls=coordination_plan.audit_planned_calls,
    audits=coordination_plan.audits,
    judge=coordination_judge.COORDINATION_JUDGE,
    smoke_model_id=SMOKE_LIVE_MODEL_ID,
)
"""The coordination pass: six cells in two oversight arms, plus the floor.

``--block`` is required on the smoke for both passes rather than defaulted per pass, because each block's
replies land in a file named after it: smoking one block and reading the other's file is the mistake the
required flag removes.
"""

PASS_BINDINGS: dict[str, PassBinding] = {
    binding.pass_id: binding for binding in (DEFERENCE_BINDING, COORDINATION_BINDING)
}
"""Every pass this CLI operates, by the id ``--pass`` selects it with."""

PASS_IDS: tuple[str, ...] = tuple(PASS_BINDINGS)


def pass_binding(pass_id: str) -> PassBinding:
    """Select one pass's binding by id, naming every pass when the id is not one of them."""
    binding = PASS_BINDINGS.get(pass_id)
    if binding is None:
        raise ValueError(
            f"{pass_id!r} is not a pass of this CLI; the passes are {list(PASS_BINDINGS)}."
        )
    return binding


def _assert_the_two_passes_share_no_leg_id() -> None:
    """Refuse at import if the two passes' leg tables collide on an id the command line offers.

    ``--leg`` has to offer both passes' ids, because argparse builds its choices before ``--pass`` is
    parsed, and the selected binding's own lookup is what refuses a leg belonging to the other pass. That
    refusal only works while the ids are disjoint: a shared id would resolve in whichever pass was selected
    and sample the wrong block into the run directory named for the other.
    """
    shared = sorted(
        {leg.leg_id for leg in DEFERENCE_BINDING.all_legs}
        & {leg.leg_id for leg in COORDINATION_BINDING.all_legs}
    )
    if shared:
        raise RuntimeError(
            f"the deference and coordination leg tables share the ids {shared}, so --leg cannot tell them "
            f"apart and a mistyped --pass would sample one pass's block into the other's run directory. "
            f"Block ids are what make a leg id unique across passes; rename one of them."
        )


_assert_the_two_passes_share_no_leg_id()


def reply_path(run_dir: Path, leg: Leg) -> Path:
    """Where one leg's replies land: one file per leg, never a flat file shared across legs."""
    return run_dir / f"replies--{leg.file_stem}.jsonl"


def handle_path(run_dir: Path, leg: Leg) -> Path:
    """Where one batch leg's job handle lands, beside the replies so the artifacts travel together."""
    return run_dir / "handles" / f"{leg.file_stem}.json"


def _submit_summary_label(leg: Leg) -> str:
    """Name the summary label one leg's submit writes under, spelled once for writer and reader."""
    return f"submit-{leg.file_stem}"


def submit_summary_path(run_dir: Path, leg: Leg) -> Path:
    """Where one leg's submit summary lands: the collect reads the stimulus prompt digest from it.

    The batch handle is a shared dataclass with no slot for a stimulus digest, so the digest the leg was
    submitted under lives in this summary instead, and :func:`collect_batch` compares it before writing a
    record.
    """
    return run_dir / f"summary-{_submit_summary_label(leg)}.json"


def reply_record(
    call: PlannedCall, completion: BedrockCompletion, stimulus: PassStimulus
) -> dict[str, Any]:
    """Build one reply record: every planned label, the sampler stamp, the reply, and completeness.

    The prompt text is not stored -- it is re-derivable from the labels plus the stimulus file -- but its
    digest and length are, so a record can be proved to answer the prompt this code renders today.

    ``stimulus_digest`` carries the stimulus PROMPT digest, not the whole-file one: the field keeps its name
    so every reader joins on the same column, but it answers "were these prompts rendered from this
    material", which is the only question a sampled row has to answer. A rubric edit moves the whole-file
    digest and leaves every reply row labelled.
    """
    return {
        "key": call.key,
        "block": call.block,
        "arm": call.arm,
        "cell": call.cell,
        "scenario_id": call.scenario_id,
        "print_order": call.print_order,
        "label_a": call.label_a,
        "label_b": call.label_b,
        "compliant_label": call.compliant_label,
        "deviation_label": call.deviation_label,
        "prompt_id": call.prompt_id,
        "model_id": call.model_id,
        "transport": call.transport,
        "reasoning_effort": call.reasoning_effort,
        "sitting": call.sitting,
        "draw": call.draw,
        "max_tokens": MAX_TOKENS,
        "stimulus_digest": stimulus.prompt_digest,
        "prompt_digest": record_prompt_digest(call.prompt),
        "prompt_chars": len(call.prompt),
        "reply": completion.text,
        "reasoning": completion.reasoning,
        "stop_reason": completion.stop_reason,
        "incomplete": bool(
            completion.stop_reason is not None and is_incomplete_stop_reason(completion.stop_reason)
        ),
        "input_tokens": completion.usage.input_tokens,
        "output_tokens": completion.usage.output_tokens,
        **completion_telemetry(completion),
        "recorded_at": datetime.now(UTC).isoformat(),
    }


def _leg_sampling(leg: Leg) -> BedrockSamplingConfig:
    """Build the sampler for one leg: the flat cap plus that leg's effort, and no other knob ever.

    No temperature, no top_p, no top_k anywhere in this pass. Roster rows refuse them outright and a run
    that asked for one and silently got none would have measured something other than what it recorded.
    """
    return BedrockSamplingConfig(max_tokens=MAX_TOKENS, reasoning_effort=leg.reasoning_effort)


def build(run_dir: Path, binding: PassBinding) -> dict[str, Any]:
    """Render every leg, run every audit, and write ``plan.json`` before any model call.

    The insertion audit runs here over every planned prompt of every leg, and the bound pass's own
    structural audits over the whole stimulus: on the deference pass, the report as one added block, the
    arms differing in one constraint sentence, the print orders moving the fork and the instruction alone,
    and the peer count agreeing between the paragraph and the report; on the coordination pass, the two
    sole-owner substitutions, the arms one whitespace token apart, the three peer cells one paragraph apart,
    the print orders, and the status block being one all-pending list in every cell of both arms. Their
    counts land in ``plan.json``, because a run whose plan is missing one of them is a run nobody can say
    ran that audit.
    """
    plan_path = run_dir / PLAN_FILENAME
    # Before the render: a tracked --run-dir should be refused now, not a minute of rendering later.
    refuse_tracked_trace_path(plan_path, carries="the planned-call table and its digests")
    stimulus = binding.load_stimulus()
    audits = binding.audits(stimulus)
    legs: list[dict[str, Any]] = []
    audited = 0
    for leg in binding.legs:
        calls = binding.planned_calls(leg, stimulus)
        audited += binding.audit_planned_calls(calls, stimulus)
        keys = {call.key for call in calls}
        if len(keys) != len(calls):
            raise RuntimeError(
                f"leg {leg.leg_id} renders {len(calls)} calls under {len(keys)} distinct keys, so two "
                f"calls would be filed under one identity."
            )
        legs.append(
            {
                "leg_id": leg.leg_id,
                "model_id": leg.model_id,
                "transport": leg.transport,
                "block": leg.block,
                "arm": leg.arm,
                "cells": [f"{cell.arm}|{cell.cell_id}" for cell in leg.cells],
                "reasoning_effort": leg.reasoning_effort,
                "sitting": leg.sitting,
                "records": len(calls),
                "prompt_digest": prompt_digest([call.prompt for call in calls]),
                "cell_digest": cell_digest([call.metadata() for call in calls]),
                "reply_file": reply_path(run_dir, leg).name,
                "handle_file": handle_path(run_dir, leg).name,
                "batch_run_id": leg.batch_run_id if leg.transport == TRANSPORT_BATCH else None,
                "batch_job_name": leg.batch_job_name,
                "prompt_chars_max": max(len(call.prompt) for call in calls),
            }
        )
    plan = {
        "pass_id": binding.pass_id,
        "stimulus_version": binding.stimulus_version,
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "peer_count": stimulus.peer_count,
        "rows_per_cell": binding.rows_per_cell,
        "draws": binding.draws,
        "max_tokens": MAX_TOKENS,
        "audited_prompts": audited,
        **audits,
        "legs": legs,
        "total_records": sum(int(entry["records"]) for entry in legs),
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir, "build", {"command": "build", **plan})
    sys.stdout.write(json.dumps(plan, indent=2) + "\n")
    return plan


def dry_run(run_dir: Path, binding: PassBinding) -> dict[str, Any]:
    """Count and price every leg without touching AWS, every assumption stated separately.

    Legs priced off an unverified figure are listed by id, so an unverified number cannot hide inside a
    total.
    """
    stimulus = binding.load_stimulus()
    report: dict[str, Any] = {"legs": [], "assumptions": {}}
    total = 0.0
    priced_unverified: list[str] = []
    for leg in binding.legs:
        calls = binding.planned_calls(leg, stimulus)
        input_tokens = int(sum(len(call.prompt) for call in calls) / CHARS_PER_TOKEN_ESTIMATE)
        output_tokens = ASSUMED_OUTPUT_TOKENS_PER_CALL * len(calls)
        if leg.transport == TRANSPORT_BATCH:
            price_in, price_out = batch_price_per_mtok(leg.model_id)
            price_note = "verified batch price"
            price_verified = True
        else:
            live_price = LIVE_PRICES[leg.model_id]
            price_in, price_out = live_price.per_mtok
            price_note = f"live price, {live_price.provenance}"
            price_verified = live_price.verified
        if not price_verified:
            priced_unverified.append(leg.leg_id)
        cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
        report["legs"].append(
            {
                "leg_id": leg.leg_id,
                "transport": leg.transport,
                "records": len(calls),
                "estimated_input_tokens": input_tokens,
                "assumed_output_tokens": output_tokens,
                "price_per_mtok": [price_in, price_out],
                "price_note": price_note,
                "price_verified": price_verified,
                "estimated_cost_usd": round(cost, 2),
            }
        )
        total += cost
    report["estimated_total_usd"] = round(total, 2)
    report["legs_priced_unverified"] = priced_unverified
    report["assumptions"] = {
        "chars_per_token": CHARS_PER_TOKEN_ESTIMATE,
        "output_tokens_per_call": ASSUMED_OUTPUT_TOKENS_PER_CALL,
        "live_prices_per_mtok": {
            model_id: {
                "price_per_mtok": list(price.per_mtok),
                "provenance": price.provenance,
                "verified": price.verified,
            }
            for model_id, price in LIVE_PRICES.items()
        },
        "note": (
            "the judge, cross-judge and intent-check passes are NOT priced here; they read the replies, "
            "so their cost is a function of measured output lengths this run does not have yet"
        ),
    }
    del run_dir  # priced from the plan alone; nothing is read from or written to the run dir
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def _refuse_resumed_handle_mismatch(
    handle: BatchJobHandle,
    backend: BedrockBatchBackend,
    *,
    calls: Sequence[PlannedCall],
    path: Path,
) -> None:
    """Refuse a saved handle whose job is not the job THIS invocation describes.

    Cell membership is positional, so collecting someone else's job against these locally rendered inputs
    would file real responses under the wrong cells or sampler labels -- every count plausible and wrong.
    Five things are compared, and each has failed somewhere in this repo's history: the model id, the record
    count, the prompt digest, the cell digest, and the sampling labels (which matter because the records'
    sampler stamp comes from this invocation's config, not the handle's).
    """
    mismatches: list[str] = []
    if handle.model_id != backend.model_id:
        mismatches.append(f"model {handle.model_id!r} on the handle vs {backend.model_id!r} here")
    if handle.record_count != len(calls):
        mismatches.append(
            f"{handle.record_count} records on the handle vs {len(calls)} rendered here"
        )
    if handle.prompt_digest != prompt_digest([call.prompt for call in calls]):
        mismatches.append("the prompt digest (the rendered prompt text differs)")
    if handle.cell_digest != cell_digest([call.metadata() for call in calls]):
        mismatches.append("the cell digest (the prompts match but their cell labels moved)")
    mismatches.extend(
        handle.sampling_label_mismatches(
            max_tokens=backend.sampling.max_tokens,
            reasoning_effort=backend.sampling.reasoning_effort,
        )
    )
    if mismatches:
        raise RuntimeError(
            f"the saved handle at {path} does not describe the job this invocation would submit: "
            f"{'; '.join(mismatches)}. Collecting it would attach responses to the wrong cells or stamp "
            f"them with a sampler that never ran. Re-run with the flags of the original submit, or point "
            f"--run-dir somewhere fresh to submit (and pay for) a new job. The job the handle names is "
            f"untouched ({handle.job_arn})."
        )


def _batch_backend(leg: Leg, run_dir: Path) -> BedrockBatchBackend:
    """Build the batch backend for one leg, namespaced so two legs never share a path or a job name."""
    return BedrockBatchBackend(
        leg.model_id,
        sampling=_leg_sampling(leg),
        prefix=f"{leg.batch_prefix}/{run_dir.name}",
        run_id=leg.batch_run_id,
    )


def submit_batch(
    leg_id: str, run_dir: Path, binding: PassBinding, *, check_only: bool
) -> dict[str, Any]:
    """Submit one leg as ONE pooled batch job, persisting the handle the moment the job exists.

    ``check_only`` stops after the floor check and the handle-path guard, spending nothing. A handle already
    on disk is reused after the five-way check rather than paid for twice.

    No refusal canary runs here, and that is a property of both rosters rather than an omission: the canary
    fronts Claude BATCH submits, the only batch row on either pass is qwen, and the live rows refuse per
    record as a ``content_filtered`` the scans bucket on their own.
    :func:`sociology.deference_plan.assert_no_batch_leg_needs_the_refusal_canary` runs at each plan module's
    import and refuses if a Claude batch row is ever added to either roster without wiring one in.
    """
    leg = binding.leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    stimulus = binding.load_stimulus()
    calls = binding.planned_calls(leg, stimulus)
    refuse_below_batch_floor(leg, len(calls))
    path = handle_path(run_dir, leg)
    # Before the submit: the handle carries account-identifying values, and refusing after the create
    # call would orphan a job already paid for.
    refuse_tracked_trace_path(path, carries=HANDLE_CARRIES)
    if check_only:
        summary = {
            "command": "submit-batch",
            "pass_id": binding.pass_id,
            "leg_id": leg.leg_id,
            "check_only": True,
            "records": len(calls),
            "handle_exists": path.exists(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
        }
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return summary
    backend = _batch_backend(leg, run_dir)
    resumed = path.exists()
    digest_recorded_at_submit = True
    if resumed:
        handle = BatchJobHandle.load(path)
        _refuse_resumed_handle_mismatch(handle, backend, calls=calls, path=path)
        digest_recorded_at_submit = _refuse_resumed_submit_under_a_different_stimulus(
            run_dir, leg, stimulus
        )
        logger.warning(
            "resuming from the saved handle at %s: job %s (submitted %s) will be collected; no new job "
            "is submitted and nothing new is billed",
            path,
            handle.job_name,
            handle.submitted_at,
        )
    else:
        handle = backend.submit(
            [call.prompt for call in calls], metadata=[call.metadata() for call in calls]
        )
        handle.save(path)
    summary = {
        "command": "submit-batch",
        "pass_id": binding.pass_id,
        "leg_id": leg.leg_id,
        "records": len(calls),
        "job_arn": handle.job_arn,
        "job_name": handle.job_name,
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "resumed_handle": resumed,
        "stimulus_prompt_digest_recorded_at_submit": digest_recorded_at_submit,
    }
    write_summary(run_dir, _submit_summary_label(leg), summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _refuse_resumed_submit_under_a_different_stimulus(
    run_dir: Path, leg: Leg, stimulus: PassStimulus
) -> bool:
    """Refuse to rewrite a submit summary with a stimulus prompt digest the job never ran under.

    ``submit-batch`` against an existing handle submits nothing and rewrites the summary, which is the
    documented repair for a lost summary and, unguarded, the one way to launder the collect-time guard:
    the rewrite replaces the recorded digest with whatever is loaded now, so
    :func:`_refuse_collect_under_a_different_stimulus` then compares that value with itself and agrees.
    The five-way handle check cannot see it either, because a prompt-affecting field this leg never
    renders -- one arm's constraint sentence while the other arm's leg resumes -- leaves every one of its
    prompts byte-identical.

    Returns whether the digest really was recorded at submit time. A summary that is simply absent carries
    no evidence in either direction, so the repair still runs and the rewritten summary records that its
    digest was reconstructed rather than claiming a provenance it does not have.
    """
    path = submit_summary_path(run_dir, leg)
    if not path.exists():
        return False
    recorded = json.loads(path.read_text(encoding="utf-8")).get("stimulus_prompt_digest")
    if recorded is None:
        return False
    if recorded != stimulus.prompt_digest:
        raise RuntimeError(
            f"leg {leg.leg_id} was submitted under stimulus prompt digest {recorded!r} (per {path}) and "
            f"this invocation loaded {stimulus.prompt_digest!r}. Rewriting the summary would make the "
            f"collect-time stimulus check compare the new digest with itself and pass, stamping rows "
            f"sampled under one stimulus as answers to another. Restore the stimulus file the submit "
            f"used, or point --run-dir somewhere fresh and submit (and pay for) a new job."
        )
    return True


def _refuse_collect_under_a_different_stimulus(
    run_dir: Path, leg: Leg, stimulus: PassStimulus, *, stimulus_unchanged_since_submit: bool
) -> None:
    """Refuse to collect a job whose submit recorded a different stimulus prompt digest.

    Every collected row is stamped with the prompt digest of the stimulus loaded NOW, so a stimulus whose
    prompt material moved between submit and collect would label rows sampled under one file as answers to
    another. The handle's own prompt digest catches most of that (the rendered text moved), but not an edit
    this leg never renders, and the handle has no slot for the stimulus digest; the submit summary is where
    it was recorded. A missing summary refuses too: re-running ``submit-batch`` reuses the saved handle
    after the five-way check and rewrites the summary, so the fix is one cheap command rather than a guess.

    That repair is also why the recorded flag is read here rather than only written at submit. A summary
    the repair rebuilt records the digest loaded at repair time, so comparing it proves nothing about what
    the job was sampled under; the collect refuses on it and asks the operator to say, in the invocation,
    that the prompt material has not moved. A summary written before the flag existed says nothing either
    way and warns instead of refusing: those runs are collected already, and refusing them would be a
    gate that fires only on history nobody can change.
    """
    path = submit_summary_path(run_dir, leg)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so the stimulus prompt digest this job was submitted under is "
            f"unknown. Re-run `submit-batch --leg {leg.leg_id} --run-dir {run_dir}`: with the handle "
            f"already saved it submits nothing new and only rewrites the summary."
        )
    summary = json.loads(path.read_text(encoding="utf-8"))
    recorded_at_submit = summary.get("stimulus_prompt_digest_recorded_at_submit")
    if recorded_at_submit is False and not stimulus_unchanged_since_submit:
        raise RuntimeError(
            f"{path} says its stimulus prompt digest was NOT recorded at submit time: the summary was "
            f"rebuilt by the repair path (`submit-batch` against a saved handle), which loaded the "
            f"digest as it stands now rather than as the job was submitted under. The check below would "
            f"then compare that value with itself and agree, so it proves nothing here. Pass "
            f"--stimulus-unchanged-since-submit if you know the prompt-affecting fields have not moved "
            f"since this job was submitted -- that is an operator statement about the file's history, "
            f"which nothing on disk can make for you."
        )
    if recorded_at_submit is None:
        logger.warning(
            "%s carries no stimulus_prompt_digest_recorded_at_submit, so whether its digest was "
            "observed at submit or reconstructed by a repair is unknown; it was written before that "
            "field existed. The digest comparison below still runs.",
            path,
        )
    recorded = summary.get("stimulus_prompt_digest")
    if recorded != stimulus.prompt_digest:
        raise RuntimeError(
            f"leg {leg.leg_id} was submitted under stimulus prompt digest {recorded!r} and this "
            f"invocation loaded {stimulus.prompt_digest!r}. Collecting would stamp rows sampled under one "
            f"stimulus as answers to another. Restore the stimulus file the submit used, or point "
            f"--run-dir somewhere fresh and submit (and pay for) a new job."
        )


def collect_batch(
    leg_id: str,
    run_dir: Path,
    binding: PassBinding,
    *,
    timeout_seconds: float,
    stimulus_unchanged_since_submit: bool = False,
) -> dict[str, Any]:
    """Collect one leg's job, verifying three digests caller-side before any record is written.

    The prompt digest proves the returned records answer the submitted prompts in order; the cell digest
    proves the labels this process would attach are the labels the submit recorded; the stimulus prompt
    digest recorded at submit proves the rows will be stamped with the material they were sampled under.
    Any mismatch refuses the whole collect. Keys already on disk are skipped and counted separately, so a
    re-collect after a partial write is a continuation rather than a duplicate -- and each skipped key is
    checked against its own stored digests first, the way the live loop does, so a reply file that
    outlived the handle it was collected against cannot be extended under a different stimulus.
    """
    leg = binding.leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    stimulus = binding.load_stimulus()
    calls = binding.planned_calls(leg, stimulus)
    path = handle_path(run_dir, leg)
    handle = BatchJobHandle.load(path)
    backend = _batch_backend(leg, run_dir)
    _refuse_resumed_handle_mismatch(handle, backend, calls=calls, path=path)
    _refuse_collect_under_a_different_stimulus(
        run_dir, leg, stimulus, stimulus_unchanged_since_submit=stimulus_unchanged_since_submit
    )
    completions = backend.collect(handle, timeout_seconds=timeout_seconds)
    out_path = reply_path(run_dir, leg)
    existing = load_replies(out_path)
    refuse_changed_resume(
        existing,
        calls,
        path=out_path,
        resume_identity=lambda call: _resume_identity(call, stimulus),
    )
    records: list[dict[str, Any]] = []
    resumed = 0
    incomplete = 0
    for call, completion in zip(calls, completions, strict=True):
        if call.key in existing:
            resumed += 1
            continue
        record = reply_record(call, completion, stimulus)
        incomplete += int(bool(record["incomplete"]))
        records.append(record)
    append_replies(out_path, records, carries=binding.reply_carries)
    summary = {
        "command": "collect-batch",
        "pass_id": binding.pass_id,
        "leg_id": leg.leg_id,
        "collected": len(records),
        "resumed": resumed,
        "incomplete": incomplete,
        "input_tokens": backend.usage.input_tokens,
        "output_tokens": backend.usage.output_tokens,
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "stimulus_unchanged_since_submit_asserted": stimulus_unchanged_since_submit,
    }
    write_summary(run_dir, f"collect-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def run_live(
    leg_id: str, run_dir: Path, binding: PassBinding, *, concurrency: int, chunk_size: int
) -> dict[str, Any]:
    """Run one live leg with resume-by-key, then write the summary as the completion marker."""
    leg = binding.leg_for(leg_id)
    if leg.transport != TRANSPORT_LIVE:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not live")
    stimulus = binding.load_stimulus()
    calls = binding.planned_calls(leg, stimulus)
    counts = _run_live_calls(
        calls,
        reply_path(run_dir, leg),
        binding,
        stimulus,
        concurrency=concurrency,
        chunk_size=chunk_size,
    )
    summary = {
        "command": "run-live",
        "pass_id": binding.pass_id,
        "leg_id": leg.leg_id,
        **counts,
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
    }
    write_summary(run_dir, f"live-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_live_calls(  # noqa: PLR0913 - one argument per seam; the last one is the testable one
    calls: Sequence[PlannedCall],
    out_path: Path,
    binding: PassBinding,
    stimulus: PassStimulus,
    *,
    concurrency: int,
    chunk_size: int,
    backend_factory: LiveBackendFactory | None = None,
) -> dict[str, int]:
    """Run planned calls through the shared resume loop with this pass's row schema.

    ``resume_identity`` is what makes the resume safe rather than merely cheap: a key already on disk is
    skipped only if its record was produced for this same prompt from this same prompt material. An edited
    brief or constraint renders new prompts under the OLD keys, and without this the second pass would skip
    every one of them and report itself complete. The stimulus side compares the PROMPT digest, not the
    whole-file one: a rubric or validation-reply edit changes what the judge reads and nothing a reply row
    answered.
    """
    return run_live_calls(
        calls,
        out_path,
        concurrency=concurrency,
        chunk_size=chunk_size,
        make_record=lambda call, completion: reply_record(call, completion, stimulus),
        resume_identity=lambda call: _resume_identity(call, stimulus),
        backend_factory=(
            backend_factory
            if backend_factory is not None
            else make_live_backend_factory(MAX_TOKENS)
        ),
        carries=binding.reply_carries,
    )


def _resume_identity(call: PlannedCall, stimulus: PassStimulus) -> dict[str, Any]:
    """Say what a record on disk has to carry before its key may be skipped as already answered.

    One function for both transports rather than a lambda inside the live loop, because a skipped key
    means the same thing on either: this call was already answered. The batch collect used to skip on the
    key alone, so a reply file that outlived its handle and its submit summary could be re-collected under
    an edited stimulus and every surviving row counted as resumed.
    """
    return {
        "stimulus_digest": stimulus.prompt_digest,
        "prompt_digest": record_prompt_digest(call.prompt),
    }


def scan(run_dir: Path) -> dict[str, Any]:
    """Run the deterministic scans over every reply on disk, then write the summary."""
    totals = scans_module.scan_run(run_dir)
    write_summary(run_dir, "scan", {"command": "scan", **totals})
    sys.stdout.write(json.dumps(totals, indent=2) + "\n")
    return totals


def _judge_backend(model_id: str, effort: str | None, *, concurrency: int) -> BedrockBackend:
    """Build the live judge backend: the flat cap and an effort, and no other sampling knob."""
    return BedrockBackend(
        model_id,
        concurrency=concurrency,
        sampling=BedrockSamplingConfig(max_tokens=MAX_TOKENS, reasoning_effort=effort),
    )


def refuse_unvalidated_judge(
    run_dir: Path, stimulus: PassStimulus, binding: PassBinding, *, skip_gate: bool
) -> bool:
    """Refuse a judge pass whose rubric has not been validated in this run; return whether skipped.

    Three keys, for the reason the intent gate below carries the same three: the rubric digest covers the
    AUTHORED instructions and nothing else, so the prompt version and the code-side prompt shape are
    checked beside it, and a validation run against an earlier wording of any of them says nothing about
    this one. Zero misses and zero unparsed replies as well, because a judge that disagreed with a
    registered verdict will misreport a rate and look perfectly healthy doing it.

    ``skip_gate`` exists for the case where a human has read the misses and decided to run anyway. It warns
    loudly and is stamped into the judge summary, so a table read months later says the judge behind it was
    never validated.
    """
    if skip_gate:
        logger.warning(
            "SKIPPING THE JUDGE-VALIDATION GATE: this pass will judge with a rubric nobody has checked "
            "against the registered verdicts in this run. Every rate it produces is uncalibrated, and the "
            "summary records skipped_validation_gate=true for that reason."
        )
        return True
    path = run_dir / JUDGE_VALIDATION_SUMMARY
    digest = binding.judge.judge_digest(stimulus)
    shape = binding.judge.judge_shape_digest()
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist, so this run has never validated its judge. Run "
            f"`judge-validate --run-dir {run_dir}` first (or pass --skip-validation-gate to judge with an "
            f"unvalidated rubric, which is recorded in the summary)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("judge_prompt_digest") != digest:
        problems.append(
            f"it validated rubric digest {payload.get('judge_prompt_digest')!r}, not this run's "
            f"{digest!r}"
        )
    if payload.get("judge_prompt_version") != binding.judge.verdict.prompt_version:
        problems.append(
            f"it validated prompt version {payload.get('judge_prompt_version')!r}, not this run's "
            f"{binding.judge.verdict.prompt_version!r}"
        )
    if payload.get("judge_prompt_shape_digest") != shape:
        problems.append(
            f"it validated prompt-shape digest {payload.get('judge_prompt_shape_digest')!r}, not this "
            f"run's {shape!r}: the options header or another code-side section of the prompt has changed "
            f"since that validation"
        )
    if payload.get("misses"):
        problems.append(f"it reported {len(payload['misses'])} field misses")
    if payload.get("unparsed"):
        problems.append(f"it reported {len(payload['unparsed'])} unparsed judge replies")
    if problems:
        raise RuntimeError(
            f"the judge validation in {path} does not clear this run: {'; '.join(problems)}. Run "
            f"`judge-validate --run-dir {run_dir}` against the current rubric and read its misses before "
            f"judging (or pass --skip-validation-gate, which is recorded in the summary)."
        )
    return False


def refuse_unvalidated_intent_check(
    run_dir: Path, stimulus: PassStimulus, binding: PassBinding, *, skip_gate: bool
) -> bool:
    """Refuse an intent-check pass whose rubric has not been validated in this run; return whether skipped.

    The same gate the rubric of record has, and load-bearing for a stronger reason: this instrument's
    verdicts are used to CORRECT which option a reply is counted as having taken, so a reader that had a
    reversal backwards would move real answers to the other option while every table still looked healthy.

    Three keys rather than one, because the rubric digest covers the AUTHORED instructions and nothing
    else. The asked-for paragraph, the channel headings and the prompt version all live in code and sit
    outside that digest, so a validation made before a code-side prompt edit would otherwise clear the pass
    that followed it -- while the per-row resume, which does compare the version, re-read every production
    row under a prompt no validation had ever covered.
    """
    if skip_gate:
        logger.warning(
            "SKIPPING THE INTENT-CHECK VALIDATION GATE: this pass will read conclusions with a rubric "
            "nobody has checked against the registered verdicts in this run, and its output corrects which "
            "option a reply is counted as having taken. The summary records skipped_validation_gate=true."
        )
        return True
    path = run_dir / INTENT_VALIDATION_SUMMARY
    digest = binding.judge.intent_digest(stimulus)
    shape = binding.judge.intent_shape_digest()
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist, so this run has never validated its intent-check rubric. Run "
            f"`intent-check-validate --run-dir {run_dir}` first (or pass --skip-validation-gate to read "
            f"with an unvalidated rubric, which is recorded in the summary)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("intent_prompt_digest") != digest:
        problems.append(
            f"it validated rubric digest {payload.get('intent_prompt_digest')!r}, not this run's "
            f"{digest!r}"
        )
    if payload.get("intent_prompt_version") != binding.judge.intent.prompt_version:
        problems.append(
            f"it validated prompt version {payload.get('intent_prompt_version')!r}, not this run's "
            f"{binding.judge.intent.prompt_version!r}"
        )
    if payload.get("intent_prompt_shape_digest") != shape:
        problems.append(
            f"it validated prompt-shape digest {payload.get('intent_prompt_shape_digest')!r}, not this "
            f"run's {shape!r}: the asked-for paragraph or another code-side section of the prompt has "
            f"changed since that validation"
        )
    if payload.get("misses"):
        problems.append(f"it reported {len(payload['misses'])} field misses")
    if payload.get("unparsed"):
        problems.append(f"it reported {len(payload['unparsed'])} unparsed replies")
    if problems:
        raise RuntimeError(
            f"the intent-check validation in {path} does not clear this run: {'; '.join(problems)}. Run "
            f"`intent-check-validate --run-dir {run_dir}` against the current rubric and read its misses "
            f"before correcting anything (or pass --skip-validation-gate, recorded in the summary)."
        )
    return False


def _records_on_disk(run_dir: Path, *, model: str | None) -> list[Mapping[str, Any]]:
    """Every reply on disk in key order, optionally narrowed to one model, refusing an empty selection.

    Also the one place every pooling instrument -- the judge, the intent check, the cross-judge subset --
    reads its rows, so the mixed-stimulus refusal sits here rather than three times over.
    """
    replies = load_run_replies(run_dir)
    refuse_pooled_stimulus_mixture(replies, run_dir=run_dir, what="every pooled rate")
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    if model is None:
        return records
    narrowed = [record for record in records if str(record.get("model_id")) == model]
    if not narrowed:
        present = sorted({str(row.get("model_id")) for row in replies.values()})
        raise ValueError(
            f"no reply in {run_dir} was written by model {model!r}; the replies on disk cover {present}. "
            f"A filtered pass over nothing would write a healthy-looking summary with zero rows read."
        )
    return narrowed


def judge(
    run_dir: Path,
    binding: PassBinding,
    *,
    limit: int | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Run the blind rubric of record over every reply on disk (resumable)."""
    stimulus = binding.load_stimulus()
    skipped_gate = refuse_unvalidated_judge(
        run_dir, stimulus, binding, skip_gate=skip_validation_gate
    )
    records = _records_on_disk(run_dir, model=None)
    if limit is not None:
        records = records[:limit]
    out_path = run_dir / "judged.jsonl"
    backend = _judge_backend(
        judge_module.JUDGE_MODEL_ID, judge_module.JUDGE_REASONING_EFFORT, concurrency=concurrency
    )
    counts = binding.judge.judge_run(backend, records, out_path, stimulus)
    write_summary(
        run_dir,
        "judge",
        {
            "command": "judge",
            "pass_id": binding.pass_id,
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_effort": judge_module.JUDGE_REASONING_EFFORT,
            "judge_concurrency": concurrency,
            "judge_prompt_version": binding.judge.verdict.prompt_version,
            "judge_prompt_digest": binding.judge.judge_digest(stimulus),
            "skipped_validation_gate": skipped_gate,
            **counts,
            "usage": {
                "input_tokens": backend.usage.input_tokens,
                "output_tokens": backend.usage.output_tokens,
            },
        },
    )
    sys.stdout.write(json.dumps(counts, indent=2) + "\n")
    return counts


def judge_validate(run_dir: Path, binding: PassBinding) -> dict[str, Any]:
    """Judge the hand-authored validation replies and report every disagreement by name and field.

    The summary this writes is what ``judge`` and ``cross-judge`` gate on, so it carries the rubric's digest
    and the prompt version: a validation of an earlier wording must not clear a later one.
    """
    stimulus = binding.load_stimulus()
    out_path = run_dir / "judge-validation.jsonl"
    backend = _judge_backend(
        judge_module.JUDGE_MODEL_ID,
        judge_module.JUDGE_REASONING_EFFORT,
        concurrency=DEFAULT_JUDGE_CONCURRENCY,
    )
    report = binding.judge.validate_judge(backend, stimulus, out_path)
    write_summary(
        run_dir,
        JUDGE_VALIDATION_LABEL,
        {
            "command": "judge-validate",
            "pass_id": binding.pass_id,
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_prompt_version": binding.judge.verdict.prompt_version,
            "judge_prompt_digest": binding.judge.judge_digest(stimulus),
            "judge_prompt_shape_digest": binding.judge.judge_shape_digest(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
            **report,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def intent_check_validate(run_dir: Path, binding: PassBinding) -> dict[str, Any]:
    """Read the hand-authored intent cases and report every disagreement by reply and field."""
    stimulus = binding.load_stimulus()
    out_path = run_dir / f"{INTENT_VALIDATION_LABEL}.jsonl"
    backend = _judge_backend(
        judge_module.INTENT_CHECK_MODEL_ID,
        judge_module.INTENT_CHECK_REASONING_EFFORT,
        concurrency=DEFAULT_JUDGE_CONCURRENCY,
    )
    report = binding.judge.validate_intent_check(backend, stimulus, out_path)
    write_summary(
        run_dir,
        INTENT_VALIDATION_LABEL,
        {
            "command": "intent-check-validate",
            "pass_id": binding.pass_id,
            "judge_model_id": judge_module.INTENT_CHECK_MODEL_ID,
            "intent_prompt_version": binding.judge.intent.prompt_version,
            "intent_prompt_digest": binding.judge.intent_digest(stimulus),
            "intent_prompt_shape_digest": binding.judge.intent_shape_digest(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
            **report,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def _rows_under_the_current_intent_rubric(
    rows: Sequence[Mapping[str, Any]], stimulus: PassStimulus, binding: PassBinding
) -> int:
    """How many of the pooled intent rows were read under the rubric and version loaded now.

    The whole-file totals beside it are what the readout uses, and they pool every row ever written; after
    a rubric edit plus a model-narrowed re-run, the rows this invocation did not select still carry the old
    digest. The same predicate the per-row resume uses, so this figure and what the next pass would re-read
    cannot drift apart.
    """
    digest = binding.judge.intent_digest(stimulus)
    return sum(
        1
        for row in rows
        if judge_module.judged_under_current_rubric(
            row, digest, prompt_version=binding.judge.intent.prompt_version
        )
    )


def intent_check(
    run_dir: Path,
    binding: PassBinding,
    *,
    model: str | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Read what each reply's reasoning concluded, so a reversed tag can be counted and corrected.

    Every reply rather than a scoped subset: a binary tag has no interior, so there is no figure whose
    middle a reversal would leave alone. It resumes by record key against ``intent-checked.jsonl`` exactly
    as the judge does, so a narrowed pass (one model) appends to whatever is already there and a later wider
    pass fills in the rest. The per-cell counts are recomputed from the WHOLE file every time for that
    reason, and the summary keeps ``invocation`` (what THIS call did) apart from ``over_the_whole_file``
    (every row ever written), because one flat object read as one thing is how a one-model pass came to
    publish run-wide totals under a one-model label.
    """
    stimulus = binding.load_stimulus()
    skipped_gate = refuse_unvalidated_intent_check(
        run_dir, stimulus, binding, skip_gate=skip_validation_gate
    )
    chosen = scans_module.chosen_labels_by_key(run_dir)
    records = _records_on_disk(run_dir, model=model)
    unscanned = sum(1 for record in records if str(record["key"]) not in chosen)
    if unscanned == len(records):
        raise ValueError(
            f"none of the {len(records)} replies selected in {run_dir} has a row in "
            f"{scans_module.SCANS_FILENAME}, so no conclusion could be read against an option and every "
            f"cell would report zero reversals over a full denominator. Run `scan` first."
        )
    if unscanned:
        logger.warning(
            "%d of %d selected replies have no row in %s, so no option of theirs can be read against a "
            "conclusion; they are counted as without_scan_label. Re-run `scan` if a leg landed after it.",
            unscanned,
            len(records),
            scans_module.SCANS_FILENAME,
        )
    out_path = run_dir / INTENT_CHECKED_FILENAME
    backend = _judge_backend(
        judge_module.INTENT_CHECK_MODEL_ID,
        judge_module.INTENT_CHECK_REASONING_EFFORT,
        concurrency=concurrency,
    )
    counts = binding.judge.intent_check_run(backend, records, out_path, stimulus)
    rows = list(judge_module.load_judged(out_path).values())
    per_cell = judge_module.intent_check_counts(rows, chosen)
    totals = {
        name: sum(cell[name] for cell in per_cell.values())
        for name in judge_module.INTENT_COUNT_NAMES
    }
    summary = {
        "command": "intent-check",
        "pass_id": binding.pass_id,
        "judge_model_id": judge_module.INTENT_CHECK_MODEL_ID,
        "judge_effort": judge_module.INTENT_CHECK_REASONING_EFFORT,
        "judge_concurrency": concurrency,
        "intent_prompt_version": binding.judge.intent.prompt_version,
        "intent_prompt_digest": binding.judge.intent_digest(stimulus),
        "intent_prompt_shape_digest": binding.judge.intent_shape_digest(),
        "skipped_validation_gate": skipped_gate,
        "invocation": {
            "model_filter": model,
            "selected": len(records),
            "without_scan_label": unscanned,
            **counts,
            "usage": {
                "input_tokens": backend.usage.input_tokens,
                "output_tokens": backend.usage.output_tokens,
            },
        },
        "over_the_whole_file": {
            "rows": len(rows),
            "rows_by_rubric": judge_module.rows_by_rubric(rows),
            "rows_under_the_current_rubric": _rows_under_the_current_intent_rubric(
                rows, stimulus, binding
            ),
            "totals": totals,
            "by_cell": dict(sorted(per_cell.items())),
        },
    }
    write_summary(run_dir, "intent-check", summary)
    sys.stdout.write(
        json.dumps({"invocation": summary["invocation"], "totals": totals}, indent=2) + "\n"
    )
    return summary


def _scan_outcomes(run_dir: Path) -> dict[str, str]:
    """Read each record's deterministic outcome off the scans file, for the cross-judge strata.

    Three buckets rather than the label text, because the labels differ per brief: what a stratified subset
    has to spread over is the OUTCOME -- followed the brief, departed from it, or never parsed -- so that
    the agreement table is read on all three rather than on whichever the roster produced most of.
    """
    path = run_dir / scans_module.SCANS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so the cross-judge subset cannot be stratified by the parsed "
            f"outcome. Run the `scan` subcommand first."
        )
    buckets: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            deviated = row.get("deviated")
            if deviated is None:
                buckets[str(row["key"])] = "unparsed"
                continue
            buckets[str(row["key"])] = "deviation" if bool(deviated) else "compliant"
    return buckets


def cross_judge(
    run_dir: Path,
    binding: PassBinding,
    *,
    n: int = judge_module.CROSS_JUDGE_RECORDS,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Re-judge a stratified subset with a second judge, because the judge is also a subject.

    Stratified per MODEL first and only then per (block, cell, outcome): the agreement table is read per
    model, so a subset that reached one model of three would not answer the question it is asked. That is
    the defect the shared loop's :func:`~sociology.judge_loop.stratified_subset` was fixed for.
    """
    stimulus = binding.load_stimulus()
    skipped_gate = refuse_unvalidated_judge(
        run_dir, stimulus, binding, skip_gate=skip_validation_gate
    )
    outcomes = _scan_outcomes(run_dir)
    records = _records_on_disk(run_dir, model=None)
    subset = judge_module.stratified_subset(
        records,
        n=n,
        outer_stratum=lambda record: str(record.get("model_id")),
        stratum=lambda record: (
            f"{record.get('block')}|{record.get('cell')}|{outcomes.get(str(record['key']))}"
        ),
    )
    out_path = run_dir / "cross-judged.jsonl"
    backend = _judge_backend(
        judge_module.CROSS_JUDGE_MODEL_ID,
        judge_module.CROSS_JUDGE_REASONING_EFFORT,
        concurrency=concurrency,
    )
    counts = binding.judge.judge_run(backend, subset, out_path, stimulus)
    write_summary(
        run_dir,
        "cross-judge",
        {
            "command": "cross-judge",
            "pass_id": binding.pass_id,
            "judge_model_id": judge_module.CROSS_JUDGE_MODEL_ID,
            "judge_concurrency": concurrency,
            "judge_prompt_digest": binding.judge.judge_digest(stimulus),
            "skipped_validation_gate": skipped_gate,
            "requested": n,
            "selected": len(subset),
            "models_covered": len({str(record.get("model_id")) for record in subset}),
            **counts,
        },
    )
    sys.stdout.write(json.dumps(counts, indent=2) + "\n")
    return counts


def _scripted_reply(prompt: str) -> str:
    """Answer a deference prompt offline in whichever option the prompt printed first.

    Reading the label off the prompt rather than emitting a fixed one, so the offline smoke exercises the
    real extraction path -- ``strip_thinking``, then the decision tag, then the compliant-label mapping --
    which is where a print-order mapping bug would surface.
    """
    match = _ACTION_TAG_RE.search(prompt)
    if match is None:
        raise ValueError(
            "a rendered deference prompt must ask for an <action> tag naming an option"
        )
    label = match.group(1)
    return f"A scripted reply for the offline smoke. <action>{label}</action>"


def scripted_judge_verdict(binding: PassBinding) -> Callable[[str], str]:
    """Build the offline smoke's one off-the-shelf verdict, shaped for this pass's two schemas at once.

    ``none`` and ``no_reasoning`` are the values that parse against any row, which is what lets one scripted
    answer exercise both judge paths. It follows that the offline smoke's validation report shows misses
    against the registered expectations: that is the miss-reporting path working, not a failing judge, and
    the live smoke is where real agreement is read. The flags are derived from the bound schema rather than
    listed, so an added field cannot silently turn every scripted judge call into an errored row -- and the
    two passes have different flags, which is why this takes the binding rather than being a constant.
    """

    def verdict_for(prompt: str) -> str:
        del prompt
        verdict: dict[str, Any] = {
            "action_label": judge_module.ACTION_NONE,
            "concluded_action": judge_module.CONCLUDED_NO_REASONING,
            "evidence": "",
        }
        verdict.update(dict.fromkeys(binding.judge.bool_fields, False))
        return json.dumps(verdict)

    return verdict_for


def smoke_calls(
    stimulus: PassStimulus,
    binding: PassBinding,
    *,
    transport: str,
    rows_per_cell: int,
    block: str,
) -> list[PlannedCall]:
    """Take the first few draw-zero calls of each of one block's cells, on the smoke's transport.

    Rendered from the real leg and then narrowed, rather than from a smoke-shaped leg of its own, so the
    smoke exercises exactly the prompts and keys production will send.
    """
    by_cell: dict[str, list[PlannedCall]] = {}
    for call in binding.planned_calls(binding.smoke_leg(transport, block), stimulus):
        if call.draw != 0:
            continue
        cell_calls = by_cell.setdefault(call.cell, [])
        if len(cell_calls) < rows_per_cell:
            cell_calls.append(call)
    return [call for cell in sorted(by_cell) for call in by_cell[cell]]


def inline_think_counts(replies: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count per model how many replies carried a thinking block inside the answer channel.

    Reported with its denominator rather than as a bare count, because the number that matters is the
    share: a roster row that reasons inside the answer channel is one whose visible reply is not what the
    transport returned, and every instrument here reads the reply after that block is cut out.
    """
    counts: dict[str, dict[str, int]] = {}
    for record in replies.values():
        tally = counts.setdefault(
            str(record.get("model_id")), {"replies": 0, "with_inline_think": 0}
        )
        tally["replies"] += 1
        block = judge_module.inline_think_block(str(record.get("reply") or ""))
        tally["with_inline_think"] += int(block is not None)
    return counts


def _refuse_judge_that_read_a_different_text(
    replies: Mapping[str, Mapping[str, Any]], judged_path: Path
) -> int:
    """Refuse a smoke whose judged rows disagree with the replies about inline thinking.

    The judge and the deterministic scan are meant to read ONE text, and this is the end-to-end check that
    they do: every judged row records whether the reply it was given carried a thinking block in the answer
    channel, so recomputing that from the reply on disk and comparing is the only place a judge quietly
    handed the raw reply would show up. Returns how many judged rows carried one.
    """
    rows = judge_module.load_judged(judged_path)
    mismatched = [
        key
        for key, row in rows.items()
        if key in replies
        and row.get("had_inline_think")
        != (judge_module.inline_think_block(str(replies[key].get("reply") or "")) is not None)
    ]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} judged rows disagree with their replies about whether a thinking block "
            f"was emitted in the answer channel (one of them: {mismatched[0]}). The judge and the "
            f"deterministic scan must read the same visible text, or their disagreement rate -- one of "
            f"this pass's reported instrument findings -- is an artifact of two different inputs."
        )
    return sum(1 for row in rows.values() if row.get("had_inline_think"))


def smoke(run_dir: Path, binding: PassBinding, *, backend: str, block: str) -> dict[str, Any]:
    """Run the end-to-end smoke: render, sample, scan, both validations, both judges, under ``smoke/``.

    ``scripted`` spends nothing and is the gate before any real submit: a handful of rows through the live
    resume loop on an offline backend, then the deterministic scans and both judge passes on scripted
    verdicts. ``live`` is the same path against real endpoints on the cheap reasoning-bearing model.
    Everything lands under ``<run-dir>/smoke/<backend>/`` so the production replies never carry smoke
    records and the two backends never resume each other's.

    One block per invocation, because a block's replies land in a file named after it: the pass reads two
    blocks, and smoking each in turn accumulates in one smoke directory rather than resuming the other's.
    """
    stimulus = binding.load_stimulus()
    smoke_dir = run_dir / "smoke" / backend
    scripted = backend == "scripted"
    transport = ScriptedDetailedBackend.transport if scripted else TRANSPORT_LIVE
    calls = smoke_calls(
        stimulus, binding, transport=transport, rows_per_cell=SMOKE_ROWS_PER_CELL, block=block
    )
    if scripted:
        calls = calls[:SMOKE_SCRIPTED_CALLS]
    out_path = reply_path(smoke_dir, binding.smoke_leg(transport, block))
    binding.audit_planned_calls(calls, stimulus)

    def scripted_factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        del model_id, effort, concurrency
        return ScriptedDetailedBackend(_scripted_reply)

    counts = _run_live_calls(
        calls,
        out_path,
        binding,
        stimulus,
        concurrency=2,
        chunk_size=len(calls),
        backend_factory=scripted_factory if scripted else None,
    )
    scan_totals = scans_module.scan_run(smoke_dir)
    judge_backend: DetailedBackend = (
        ScriptedDetailedBackend(scripted_judge_verdict(binding))
        if scripted
        else _judge_backend(
            judge_module.JUDGE_MODEL_ID,
            judge_module.JUDGE_REASONING_EFFORT,
            concurrency=DEFAULT_JUDGE_CONCURRENCY,
        )
    )
    validation = binding.judge.validate_judge(
        judge_backend, stimulus, smoke_dir / "judge-validation.jsonl"
    )
    intent_validation = binding.judge.validate_intent_check(
        judge_backend, stimulus, smoke_dir / f"{INTENT_VALIDATION_LABEL}.jsonl"
    )
    replies = load_run_replies(smoke_dir)
    records = [replies[key] for key in sorted(replies)]
    judge_counts = binding.judge.judge_run(
        judge_backend, records, smoke_dir / "judged.jsonl", stimulus
    )
    intent_counts = binding.judge.intent_check_run(
        judge_backend, records, smoke_dir / INTENT_CHECKED_FILENAME, stimulus
    )
    ratios = [
        int(record["prompt_chars"]) / int(record["input_tokens"])
        for record in replies.values()
        if int(record["input_tokens"] or 0) > 0
    ]
    judged_with_inline_think = _refuse_judge_that_read_a_different_text(
        replies, smoke_dir / "judged.jsonl"
    )
    summary = {
        "command": "smoke",
        "pass_id": binding.pass_id,
        "backend": backend,
        "block": block,
        "model_id": binding.smoke_model_id,
        "calls": len(calls),
        **counts,
        "scans": scan_totals,
        "judge_validation": validation,
        "judge_validation_misses": validation["misses"],
        "intent_validation": intent_validation,
        "judge_counts": judge_counts,
        "intent_counts": intent_counts,
        "inline_think_by_model": inline_think_counts(replies),
        "judged_rows_with_inline_think": judged_with_inline_think,
        "measured_chars_per_token": [round(ratio, 2) for ratio in ratios],
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
    }
    write_summary(smoke_dir, "smoke", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _at_least_one(text: str) -> int:
    """Parse an argparse integer that must be 1 or more, refusing at the parser rather than in a run."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--pass",
        dest="pass_id",
        choices=PASS_IDS,
        required=True,
        help=(
            "which pass's plan this command operates on. Required and never defaulted: the two passes share "
            "this surface, and a default would let a command re-render, re-price or re-submit the finished "
            "pass because nobody typed the flag"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="where this run's artifacts live; defaults to the selected pass's own run directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="render every leg, audit every prompt, write plan.json")
    sub.add_parser("dry-run", help="count and price every leg; no AWS calls")
    smoke_parser = sub.add_parser("smoke", help="end-to-end smoke, offline or on the live path")
    smoke_parser.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    # Every pass's blocks and legs, because argparse builds its choices before --pass is parsed; the
    # selected binding's own lookups are what refuse a block or a leg belonging to the other pass.
    smoke_parser.add_argument(
        "--block",
        choices=sorted({block for binding in PASS_BINDINGS.values() for block in binding.blocks}),
        required=True,
        help=(
            "which block's cells to smoke; run it once per block the pass reads, since each block's "
            "replies land in a file of their own"
        ),
    )
    leg_ids = sorted(leg.leg_id for binding in PASS_BINDINGS.values() for leg in binding.all_legs)
    submit = sub.add_parser("submit-batch", help="submit ONE pooled batch job for one leg")
    submit.add_argument("--leg", choices=leg_ids, required=True)
    submit.add_argument(
        "--check-only",
        action="store_true",
        help="run the floor and path checks and stop before anything is submitted or billed",
    )
    collect = sub.add_parser("collect-batch", help="collect one leg's job into its replies file")
    collect.add_argument("--leg", choices=leg_ids, required=True)
    collect.add_argument("--timeout-seconds", type=float, default=BATCH_TIMEOUT_SECONDS)
    collect.add_argument(
        "--stimulus-unchanged-since-submit",
        action="store_true",
        help=(
            "assert the stimulus file's prompt-affecting fields have not moved since this job was "
            "submitted; needed only where the submit summary was rebuilt by the repair path and so "
            "records a digest it read after the fact"
        ),
    )
    live = sub.add_parser("run-live", help="run one live leg with resume-by-key")
    live.add_argument("--leg", choices=leg_ids, required=True)
    live.add_argument("--concurrency", type=_at_least_one, default=16)
    live.add_argument("--chunk-size", type=_at_least_one, default=32)
    sub.add_parser("scan", help="deterministic scans over every reply on disk")
    judge_parser = sub.add_parser("judge", help="blind judge over every reply on disk (resumable)")
    judge_parser.add_argument(
        "--limit",
        type=_at_least_one,
        default=None,
        help="judge only the first N records by key; a run-shaped smoke, not a sampling choice",
    )
    sub.add_parser("judge-validate", help="judge the rubric of record's validation replies")
    intent = sub.add_parser(
        "intent-check", help="read what each reply's reasoning concluded, against the tag it wrote"
    )
    intent.add_argument(
        "--model",
        default=None,
        help=(
            "read only this model's replies, validated against the model ids actually on disk; recorded "
            "as model_filter in the summary (null when absent)"
        ),
    )
    sub.add_parser(
        "intent-check-validate",
        help="read the hand-authored intent cases and report every disagreement",
    )
    cross = sub.add_parser("cross-judge", help="re-judge a stratified subset with a second judge")
    cross.add_argument("--n", type=_at_least_one, default=judge_module.CROSS_JUDGE_RECORDS)
    for gated in (judge_parser, intent, cross):
        gated.add_argument(
            "--skip-validation-gate",
            action="store_true",
            help=(
                "judge with a rubric this run has not validated; warns loudly and records "
                "skipped_validation_gate=true in the summary"
            ),
        )
        gated.add_argument(
            "--concurrency",
            type=_at_least_one,
            default=DEFAULT_JUDGE_CONCURRENCY,
            help=(
                "judge calls in flight at once; recorded as judge_concurrency in the summary (the hatch "
                "judge ran Luna at 32-48 without throttling)"
            ),
        )
    return parser.parse_args(argv)


_DISPATCH_WITH_BINDING: dict[str, Callable[[Path, PassBinding], object]] = {
    "build": build,
    "dry-run": dry_run,
    "judge-validate": judge_validate,
    "intent-check-validate": intent_check_validate,
}
"""Subcommands whose arguments are the run directory and the bound pass, and nothing else."""


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one subcommand of the pass ``--pass`` selects."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    binding = pass_binding(args.pass_id)
    run_dir: Path = Path(binding.default_run_dir) if args.run_dir is None else args.run_dir
    with_binding = _DISPATCH_WITH_BINDING.get(args.command)
    if with_binding is not None:
        with_binding(run_dir, binding)
    elif args.command == "scan":
        # The scans read whatever replies are on disk and carry the same label fields in both passes, so
        # this is the one subcommand the binding decides nothing about beyond the run directory.
        scan(run_dir)
    elif args.command == "smoke":
        smoke(run_dir, binding, backend=args.backend, block=args.block)
    elif args.command == "submit-batch":
        submit_batch(args.leg, run_dir, binding, check_only=args.check_only)
    elif args.command == "collect-batch":
        collect_batch(
            args.leg,
            run_dir,
            binding,
            timeout_seconds=args.timeout_seconds,
            stimulus_unchanged_since_submit=args.stimulus_unchanged_since_submit,
        )
    elif args.command == "run-live":
        run_live(
            args.leg, run_dir, binding, concurrency=args.concurrency, chunk_size=args.chunk_size
        )
    elif args.command == "judge":
        judge(
            run_dir,
            binding,
            limit=args.limit,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
    elif args.command == "intent-check":
        intent_check(
            run_dir,
            binding,
            model=args.model,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
    else:
        cross_judge(
            run_dir,
            binding,
            n=args.n,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
