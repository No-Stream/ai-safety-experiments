"""The decoupled-ladder CLI: build the plan, price it, smoke it, sample it, scan it, judge it.

One subcommand per step of the design's order of work, and one ``--leg`` per batch job or live pass.
Nothing here decides anything: the leg table lives in :mod:`sociology.decoupled_plan`, the clauses in
:mod:`sociology.decoupled_stimulus`, the instruments in :mod:`sociology.decoupled_scans` and
:mod:`sociology.decoupled_judge`, and this module is the operator surface over them.

Five habits this repository has paid for are enforced here rather than left to whoever runs it:

- **The plan is written down before any model call.** ``build`` renders every planned prompt, runs the
  one-inserted-paragraph audit over all of them, and writes ``plan.json`` with the per-leg record
  counts and both digests. A run whose plan was never written is a run nobody can say what it
  intended to sample.
- **Batch handles are guarded and reused, never re-submitted.** The tracked-path refusal runs BEFORE
  the submit (a refusal afterwards would orphan a job already paid for), and a handle already on disk
  is reused only after the five-way check that it describes THIS invocation's job.
- **Resume is by content-derived key, and by content.** Live passes skip keys already on disk, count
  them separately from what ran, and refuse a key whose record answers a different prompt or a
  different stimulus file; collects skip keys already written and count those separately too.
- **The judge does not run before it has been validated.** ``judge`` and ``cross-judge`` refuse unless
  this run's ``judge-validate`` cleared the CURRENT rubric with no misses, because an uncalibrated
  judge produces the ladder's headline rate and looks healthy doing it.
- **The summary is written LAST**, as the completion marker for every subcommand.

Every artifact lands under ``--run-dir``, which defaults under ``artifacts/`` because the replies
carry the authored clause prose inside their prompts' digests and the judge's quoted evidence
verbatim; the writers refuse a git-tracked destination outright.
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
from typing import TYPE_CHECKING, Any

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
from sociology import decoupled_judge as judge_module
from sociology import decoupled_scans as scans_module
from sociology.decoupled_plan import (
    ANCHOR_CELLS,
    BLOCK_ANCHOR,
    DECOUPLED_BATCH_PREFIX,
    DRAWS,
    GPT_OSS_120B_MODEL_ID,
    LEGS_BY_ID,
    LIVE_MODEL_IDS,
    LUNA_MODEL_ID,
    MAX_TOKENS,
    OPUS_MODEL_ID,
    ROWS_PER_CELL,
    SITTING_A,
    SONNET_MODEL_ID,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    Leg,
    PlannedCall,
    all_legs,
    audit_planned_calls,
    batch_price_per_mtok,
    leg_for,
    planned_calls_for_leg,
    refuse_below_batch_floor,
)
from sociology.decoupled_stimulus import (
    STIMULUS_VERSION,
    DecoupledStimulus,
    load_stimulus,
)
from sociology.model_stub import ScriptedDetailedBackend
from sociology.records import (
    append_replies,
    completion_telemetry,
    load_replies,
    make_live_backend_factory,
    record_prompt_digest,
    run_live_calls,
    write_summary,
)
from sociology.refusal_canary import (
    REFUSAL_CANARY_CALLS,
    canary_plan,
    needs_refusal_canary,
    recorded_canary,
    run_refusal_canary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend
    from sociology.records import LiveBackendFactory

logger = logging.getLogger(__name__)

DEFAULT_RUN_DIR = Path("artifacts/swarm_sociology/decoupled_ladder/run-20260902")

REPLY_CARRIES = "verbatim model replies to the authored counterpart clauses"

PLAN_FILENAME = "plan.json"
ADDED_LADDER_FILENAME = "added-ladder.json"
JUDGE_VALIDATION_LABEL = "judge-validation"
JUDGE_VALIDATION_SUMMARY = f"summary-{JUDGE_VALIDATION_LABEL}.json"

CHARS_PER_TOKEN_ESTIMATE = 3.5
"""Conservative chars-per-token for the dry run's input estimate; measured by the smoke afterwards."""

ASSUMED_OUTPUT_TOKENS_PER_CALL = 2_000
"""Dry-run output assumption per reply, stated rather than hidden inside a total.

Anchored on the roster's own note that GPT-OSS 120B emits about 1,780 output tokens per call at
default effort, rounded up because the decoupled cells deliberate roughly twice as long as the
coupled ones on this repo's banked traces, and because Claude Opus 5 bills adaptive thinking against
the same budget. The smoke reports measured chars-per-token and the collects report real usage, so
this number is only ever load-bearing before the first record exists.
"""


@dataclass(frozen=True, slots=True)
class LivePrice:
    """One live row's (input, output) $/Mtok and where the figure came from.

    ``verified`` is the live-path twin of the roster's ``batch_price_verified``: false means the number
    was reasoned to rather than read, and every dry-run line derived from it says so.
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
    """Refuse at import if a live roster row has no price line, or a price line names no live row."""
    unpriced = sorted(set(LIVE_MODEL_IDS) - set(LIVE_PRICES))
    stray = sorted(set(LIVE_PRICES) - set(LIVE_MODEL_IDS))
    if unpriced or stray:
        raise RuntimeError(
            f"LIVE_PRICES and the live roster disagree: live rows without a price "
            f"{unpriced or 'none'}; priced ids that are not live rows {stray or 'none'}."
        )


_assert_every_live_row_is_priced()

SMOKE_LIVE_MODEL_ID = GPT_OSS_120B_MODEL_ID
"""The live smoke's model: cheap, reasoning-bearing, and verified on both transports."""

SMOKE_LIVE_ROWS_PER_CELL = 2
SMOKE_SCRIPTED_CALLS = 2

DEFAULT_JUDGE_CONCURRENCY = DEFAULT_BEDROCK_CONCURRENCY
"""Judge calls in flight when ``--concurrency`` is not given: the transport's own default.

A throughput knob rather than a sampling one, so raising it changes wall clock and nothing about what
is measured; the hatch judge in :mod:`reward_hacking.hatch_narration_judge` ran the same Luna model at
32 to 48 without throttling. It is stamped into every judge summary so a slow or throttled pass can be
read back to the number it actually ran at.
"""

_ACTION_OPTION_RE = re.compile(r"<action>([^<]+)</action>")

ON_DEMAND_MODEL_CHOICES: dict[str, str] = {
    "opus": OPUS_MODEL_ID,
    "luna": LUNA_MODEL_ID,
    "sonnet": SONNET_MODEL_ID,
}
"""The ``add-ladder --model`` handles, one per model with an on-demand leg in the plan."""


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
    """Where one leg's submit summary lands; a resumed submit reads the recorded canary back from it."""
    return run_dir / f"summary-{_submit_summary_label(leg)}.json"


def reply_record(
    call: PlannedCall, completion: BedrockCompletion, stimulus: DecoupledStimulus
) -> dict[str, Any]:
    """Build one reply record: every planned label, the sampler stamp, the reply, and completeness.

    The prompt text is not stored -- it is re-derivable from ``prompt_id`` and ``cell`` -- but its
    digest and length are, so a record can be proved to answer the prompt this code renders today.
    """
    return {
        "key": call.key,
        "block": call.block,
        "cell": call.cell,
        "game_id": call.game_id,
        "prompt_id": call.prompt_id,
        "reskin_id": call.reskin_id,
        "payoff_variant": call.payoff_variant,
        "label_print_order": call.label_print_order,
        "coop_label": call.coop_label,
        "label_a": call.label_a,
        "label_b": call.label_b,
        "coop_label_index": call.coop_label_index,
        "model_id": call.model_id,
        "transport": call.transport,
        "reasoning_effort": call.reasoning_effort,
        "sitting": call.sitting,
        "draw": call.draw,
        "max_tokens": MAX_TOKENS,
        "stimulus_digest": stimulus.digest,
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

    No temperature, no top_p, no top_k anywhere in this pass. Three roster rows refuse them outright
    and a run that asked for one and silently got none would have measured something other than what
    it recorded.
    """
    return BedrockSamplingConfig(max_tokens=MAX_TOKENS, reasoning_effort=leg.reasoning_effort)


def authorized_on_demand_models(run_dir: Path) -> list[str]:
    """Read which on-demand ladder legs this run has authorized, if any."""
    path = run_dir / ADDED_LADDER_FILENAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(model_id) for model_id in payload.get("model_ids", [])]


def _refuse_unauthorized_leg(leg: Leg, run_dir: Path) -> None:
    """Refuse an on-demand leg the run has not authorized with ``add-ladder``.

    The on-demand ladders wait on their own model's anchor: the design runs them only when that
    model's pooled decoupled cooperation lands inside the readable band, because outside it a ladder
    is measuring a rate at a ceiling or a floor. Opus 5's anchor and floor are on demand for the other
    reason a leg can be -- the classifier refuses the model on this stimulus, so re-submitting it is a
    decision to spend on a row the pass does not read. Making either a recorded authorization rather
    than an operator's memory means the run dir says why the leg was submitted.
    """
    if not leg.on_demand or leg.model_id in authorized_on_demand_models(run_dir):
        return
    short = next(
        (name for name, model_id in ON_DEMAND_MODEL_CHOICES.items() if model_id == leg.model_id),
        leg.model_id,
    )
    raise RuntimeError(
        f"leg {leg.leg_id} is an on-demand ladder and this run has not authorized it. Read the "
        f"anchor's band verdict first, then record the decision with "
        f"`add-ladder --model {short} --run-dir {run_dir}`."
    )


def build(run_dir: Path) -> dict[str, Any]:
    """Render every leg, audit every prompt, and write ``plan.json`` before any model call.

    The audit is the design's load-bearing gate: each cell's prompt must be its clause-free stem plus
    exactly one counterpart paragraph, so a movement between cells is attributable to the clause.
    Running it over all ~45,000 planned prompts here rather than sampling them costs seconds and is
    the only place the whole grid is checked at once.
    """
    plan_path = run_dir / PLAN_FILENAME
    # Before the render, not after it: an operator who pointed --run-dir at a tracked directory should
    # hear about it now rather than a minute of rendering later.
    refuse_tracked_trace_path(plan_path, carries="the planned-call table and its digests")
    stimulus = load_stimulus()
    legs: list[dict[str, Any]] = []
    audited = 0
    for leg in all_legs():
        calls = planned_calls_for_leg(leg, stimulus)
        audited += audit_planned_calls(calls)
        keys = {call.key for call in calls}
        if len(keys) != len(calls):
            raise RuntimeError(
                f"leg {leg.leg_id} renders {len(calls)} calls under {len(keys)} distinct keys, so "
                f"two calls would be filed under one identity."
            )
        legs.append(
            {
                "leg_id": leg.leg_id,
                "model_id": leg.model_id,
                "transport": leg.transport,
                "block": leg.block,
                "cells": list(leg.cells),
                "reasoning_effort": leg.reasoning_effort,
                "sitting": leg.sitting,
                "on_demand": leg.on_demand,
                "records": len(calls),
                "prompt_digest": prompt_digest([call.prompt for call in calls]),
                "cell_digest": cell_digest([call.metadata() for call in calls]),
                "reply_file": reply_path(run_dir, leg).name,
                "handle_file": handle_path(run_dir, leg).name,
                "prompt_chars_max": max(len(call.prompt) for call in calls),
            }
        )
    plan = {
        "stimulus_version": STIMULUS_VERSION,
        "stimulus_digest": stimulus.digest,
        "rows_per_cell": ROWS_PER_CELL,
        "draws": DRAWS,
        "max_tokens": MAX_TOKENS,
        "audited_prompts": audited,
        "legs": legs,
        "total_records": sum(int(entry["records"]) for entry in legs),
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir, "build", {"command": "build", **plan})
    sys.stdout.write(json.dumps(plan, indent=2) + "\n")
    return plan


def dry_run(run_dir: Path) -> dict[str, Any]:
    """Count and price every leg without touching AWS, with every assumption stated separately.

    Two totals: the production legs alone, and every leg the plan can be asked for. On-demand legs
    are the band-gated ladders and every leg of a row the pass does not read, so the first total is
    what a default run of this plan spends. Legs priced off an unverified figure are listed by id, so
    an unverified number cannot hide inside a total.
    """
    stimulus = load_stimulus()
    report: dict[str, Any] = {"legs": [], "assumptions": {}}
    total = 0.0
    priced_unverified: list[str] = []
    for leg in all_legs():
        calls = planned_calls_for_leg(leg, stimulus)
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
                "on_demand": leg.on_demand,
                "records": len(calls),
                "estimated_input_tokens": input_tokens,
                "assumed_output_tokens": output_tokens,
                "price_per_mtok": [price_in, price_out],
                "price_note": price_note,
                "price_verified": price_verified,
                "estimated_cost_usd": round(cost, 2),
            }
        )
        if not leg.on_demand:
            total += cost
    report["estimated_total_usd_production_legs"] = round(total, 2)
    report["estimated_total_usd_all_legs"] = round(
        sum(float(entry["estimated_cost_usd"]) for entry in report["legs"]), 2
    )
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
            "the judge and cross-judge passes are NOT priced here; they read the replies, so their "
            "cost is a function of measured output lengths this run does not have yet"
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

    Cell membership is positional, so collecting someone else's job against these locally rendered
    inputs would file real responses under the wrong cells or sampler labels -- every count plausible
    and wrong. Five things are compared, and each has failed somewhere in this repo's history: the
    model id, the record count, the prompt digest, the cell digest, and the sampling labels (which
    matter because the records' sampler stamp comes from this invocation's config, not the handle's).
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
            f"{'; '.join(mismatches)}. Collecting it would attach responses to the wrong cells or "
            f"stamp them with a sampler that never ran. Re-run with the flags of the original "
            f"submit, or point --run-dir somewhere fresh to submit (and pay for) a new job. The job "
            f"the handle names is untouched ({handle.job_arn})."
        )


def _batch_backend(leg: Leg, run_dir: Path) -> BedrockBatchBackend:
    """Build the batch backend for one leg, namespaced so two legs never share an S3 path or a job name."""
    return BedrockBatchBackend(
        leg.model_id,
        sampling=_leg_sampling(leg),
        prefix=f"{DECOUPLED_BATCH_PREFIX}/{run_dir.name}",
        run_id=leg.batch_run_id,
    )


def _canary_cell(call: PlannedCall) -> str:
    """Name the group a ladder canary call stands for: the cell, i.e. the one counterpart clause it carries."""
    return call.cell


def _refusal_canary(
    leg: Leg,
    calls: Sequence[PlannedCall],
    run_dir: Path,
    *,
    calls_per_group: int,
    backend_factory: LiveBackendFactory | None,
) -> dict[str, Any] | None:
    """Run the live refusal canary for a Claude leg, persist its counts, and refuse if it tripped.

    ``None`` for every other vendor, whose classifiers have never refused a job wholesale. The counts
    are written under their own summary label BEFORE the refusal decision, so a refused submit leaves
    its evidence in the run dir; the canary's replies are never written anywhere.
    """
    if not needs_refusal_canary(leg.model_id):
        return None
    result = run_refusal_canary(
        calls,
        group_of=_canary_cell,
        backend_factory=backend_factory or make_live_backend_factory(MAX_TOKENS),
        calls_per_group=calls_per_group,
    )
    write_summary(
        run_dir,
        f"refusal-canary-{leg.file_stem}",
        {"command": "submit-batch", "leg_id": leg.leg_id, **result.summary},
    )
    result.refuse_if_tripped(job=f"leg {leg.leg_id}", records=len(calls))
    return result.summary


def submit_batch(
    leg_id: str,
    run_dir: Path,
    *,
    check_only: bool,
    canary_calls: int = REFUSAL_CANARY_CALLS,
    canary_backend_factory: LiveBackendFactory | None = None,
) -> dict[str, Any]:
    """Submit one leg as ONE pooled batch job, persisting the handle the moment the job exists.

    ``check_only`` stops after the floor check and the handle-path guard, spending nothing. A handle
    already on disk is reused after the five-way check rather than paid for twice.

    A Claude leg's job is preceded by the live refusal canary (:mod:`sociology.refusal_canary`):
    ``canary_calls`` distinct prompts of every cell go through the live tier under the leg's own
    sampler, and a cell that answers none of them refuses the submit before the job exists. Its counts
    land in the submit summary and in ``summary-refusal-canary-<leg>.json``; its replies are never
    written anywhere. A resumed handle runs no canary -- that job is already bought -- and its summary
    carries the canary block the original submit recorded rather than erasing it.
    ``canary_backend_factory`` is the offline tests' seam.
    """
    leg = leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    _refuse_unauthorized_leg(leg, run_dir)
    stimulus = load_stimulus()
    calls = planned_calls_for_leg(leg, stimulus)
    refuse_below_batch_floor(leg, len(calls))
    path = handle_path(run_dir, leg)
    # Before the submit, unconditionally: the handle carries the account id, bucket and profile, and
    # a refusal after the create call would orphan a job already paid for.
    refuse_tracked_trace_path(path, carries=HANDLE_CARRIES)
    if check_only:
        summary = {
            "command": "submit-batch",
            "leg_id": leg.leg_id,
            "check_only": True,
            "records": len(calls),
            "handle_exists": path.exists(),
            "stimulus_digest": stimulus.digest,
            "refusal_canary_plan": (
                canary_plan(calls, group_of=_canary_cell, calls_per_group=canary_calls)
                if needs_refusal_canary(leg.model_id)
                else None
            ),
        }
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
        return summary
    backend = _batch_backend(leg, run_dir)
    resumed = path.exists()
    if resumed:
        handle = BatchJobHandle.load(path)
        _refuse_resumed_handle_mismatch(handle, backend, calls=calls, path=path)
        logger.warning(
            "resuming from the saved handle at %s: job %s (submitted %s) will be collected; no new "
            "job is submitted and nothing new is billed",
            path,
            handle.job_name,
            handle.submitted_at,
        )
        canary = recorded_canary(submit_summary_path(run_dir, leg))
    else:
        canary = _refusal_canary(
            leg,
            calls,
            run_dir,
            calls_per_group=canary_calls,
            backend_factory=canary_backend_factory,
        )
        handle = backend.submit(
            [call.prompt for call in calls], metadata=[call.metadata() for call in calls]
        )
        handle.save(path)
    summary = {
        "command": "submit-batch",
        "leg_id": leg.leg_id,
        "records": len(calls),
        "job_arn": handle.job_arn,
        "job_name": handle.job_name,
        "stimulus_digest": stimulus.digest,
        "resumed_handle": resumed,
        "refusal_canary": canary,
    }
    write_summary(run_dir, _submit_summary_label(leg), summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def collect_batch(leg_id: str, run_dir: Path, *, timeout_seconds: float) -> dict[str, Any]:
    """Collect one leg's job, verifying both digests caller-side before any record is written.

    The prompt digest proves the returned records answer the submitted prompts in order; the cell
    digest proves the labels this process would attach are the labels the submit recorded. Either
    mismatch refuses the whole collect. Keys already on disk are skipped and counted separately, so a
    re-collect after a partial write is a continuation rather than a duplicate.
    """
    leg = leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    stimulus = load_stimulus()
    calls = planned_calls_for_leg(leg, stimulus)
    path = handle_path(run_dir, leg)
    handle = BatchJobHandle.load(path)
    backend = _batch_backend(leg, run_dir)
    _refuse_resumed_handle_mismatch(handle, backend, calls=calls, path=path)
    completions = backend.collect(handle, timeout_seconds=timeout_seconds)
    out_path = reply_path(run_dir, leg)
    existing = load_replies(out_path)
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
    append_replies(out_path, records, carries=REPLY_CARRIES)
    summary = {
        "command": "collect-batch",
        "leg_id": leg.leg_id,
        "collected": len(records),
        "resumed": resumed,
        "incomplete": incomplete,
        "input_tokens": backend.usage.input_tokens,
        "output_tokens": backend.usage.output_tokens,
        "stimulus_digest": stimulus.digest,
    }
    write_summary(run_dir, f"collect-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def run_live(leg_id: str, run_dir: Path, *, concurrency: int, chunk_size: int) -> dict[str, Any]:
    """Run one live leg with resume-by-key, then write the summary as the completion marker."""
    leg = leg_for(leg_id)
    if leg.transport != TRANSPORT_LIVE:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not live")
    _refuse_unauthorized_leg(leg, run_dir)
    stimulus = load_stimulus()
    calls = planned_calls_for_leg(leg, stimulus)
    counts = _run_live_calls(
        calls,
        reply_path(run_dir, leg),
        stimulus,
        concurrency=concurrency,
        chunk_size=chunk_size,
    )
    summary = {
        "command": "run-live",
        "leg_id": leg.leg_id,
        **counts,
        "stimulus_digest": stimulus.digest,
    }
    write_summary(run_dir, f"live-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_live_calls(  # noqa: PLR0913 - one argument per seam; the last two are the testable ones
    calls: Sequence[PlannedCall],
    out_path: Path,
    stimulus: DecoupledStimulus,
    *,
    concurrency: int,
    chunk_size: int,
    backend_factory: LiveBackendFactory | None = None,
) -> dict[str, int]:
    """Run planned calls through the shared resume loop with this pass's row schema.

    ``resume_identity`` is what makes the resume safe rather than merely cheap: a key already on disk
    is skipped only if its record was produced for this same prompt under this same stimulus file. An
    edited clause or rubric renders new prompts under the OLD keys, and without this the second pass
    would skip every one of them and report itself complete.
    """
    return run_live_calls(
        calls,
        out_path,
        concurrency=concurrency,
        chunk_size=chunk_size,
        make_record=lambda call, completion: reply_record(call, completion, stimulus),
        resume_identity=lambda call: {
            "stimulus_digest": stimulus.digest,
            "prompt_digest": record_prompt_digest(call.prompt),
        },
        backend_factory=(
            backend_factory
            if backend_factory is not None
            else make_live_backend_factory(MAX_TOKENS)
        ),
        carries=REPLY_CARRIES,
    )


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
    run_dir: Path, stimulus: DecoupledStimulus, *, skip_gate: bool
) -> bool:
    """Refuse a judge pass whose rubric has not been validated in this run; return whether skipped.

    The validation pass is what says the rubric means what it was written to mean, and it is per
    rubric: a validation run against an earlier wording says nothing about this one, which is why the
    digest is compared rather than the file's existence. Zero misses and zero unparsed replies as
    well, because a judge that disagreed with a registered verdict is an instrument that will
    misreport the ladder's headline rate and look perfectly healthy doing it.

    ``skip_gate`` exists for the case where a human has read the misses and decided to run anyway. It
    warns loudly and is stamped into the judge summary, so a table read months later says the judge
    behind it was never validated.
    """
    if skip_gate:
        logger.warning(
            "SKIPPING THE JUDGE-VALIDATION GATE: this pass will judge with a rubric nobody has "
            "checked against the registered verdicts in this run. Every rate it produces is "
            "uncalibrated, and the summary records skipped_validation_gate=true for that reason."
        )
        return True
    path = run_dir / JUDGE_VALIDATION_SUMMARY
    digest = judge_module.rubric_digest(stimulus)
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist, so this run has never validated its judge. Run "
            f"`judge-validate --run-dir {run_dir}` first (or pass --skip-validation-gate to judge "
            f"with an unvalidated rubric, which is recorded in the summary)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("judge_prompt_digest") != digest:
        problems.append(
            f"it validated rubric digest {payload.get('judge_prompt_digest')!r}, not this run's "
            f"{digest!r}"
        )
    if payload.get("misses"):
        problems.append(f"it reported {len(payload['misses'])} field misses")
    if payload.get("unparsed"):
        problems.append(f"it reported {len(payload['unparsed'])} unparsed judge replies")
    if problems:
        raise RuntimeError(
            f"the judge validation in {path} does not clear this run: {'; '.join(problems)}. Run "
            f"`judge-validate --run-dir {run_dir}` against the current rubric and read its misses "
            f"before judging (or pass --skip-validation-gate, which is recorded in the summary)."
        )
    return False


def judge(
    run_dir: Path,
    *,
    limit: int | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, int]:
    """Run the blind judge over every reply on disk (resumable, chunk-appended)."""
    stimulus = load_stimulus()
    skipped_gate = refuse_unvalidated_judge(run_dir, stimulus, skip_gate=skip_validation_gate)
    replies = scans_module.load_run_replies(run_dir)
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    if limit is not None:
        records = records[:limit]
    out_path = run_dir / "judged.jsonl"
    backend = _judge_backend(
        judge_module.JUDGE_MODEL_ID, judge_module.JUDGE_REASONING_EFFORT, concurrency=concurrency
    )
    counts = judge_module.judge_records(backend, records, out_path, stimulus)
    write_summary(
        run_dir,
        "judge",
        {
            "command": "judge",
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_effort": judge_module.JUDGE_REASONING_EFFORT,
            "judge_concurrency": concurrency,
            "judge_prompt_digest": judge_module.rubric_digest(stimulus),
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


def judge_validate(run_dir: Path) -> dict[str, Any]:
    """Judge the stimulus file's hand-authored validation replies and report every disagreement.

    The summary this writes is what the judge and cross-judge subcommands gate on, so it carries the
    rubric's digest and version: a validation of an earlier wording must not clear a later one.
    """
    stimulus = load_stimulus()
    out_path = run_dir / "judge-validation.jsonl"
    backend = _judge_backend(
        judge_module.JUDGE_MODEL_ID,
        judge_module.JUDGE_REASONING_EFFORT,
        concurrency=DEFAULT_JUDGE_CONCURRENCY,
    )
    report = judge_module.validate_judge(backend, stimulus, out_path)
    write_summary(
        run_dir,
        JUDGE_VALIDATION_LABEL,
        {
            "command": "judge-validate",
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_prompt_version": judge_module.JUDGE_PROMPT_VERSION,
            "judge_prompt_digest": judge_module.rubric_digest(stimulus),
            "stimulus_digest": stimulus.digest,
            **report,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def _scan_actions(run_dir: Path) -> dict[str, str]:
    """Read each record's deterministic parsed action off the scans file, for the strata."""
    path = run_dir / scans_module.SCANS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so the cross-judge subset cannot be stratified by parsed "
            f"action. Run the `scan` subcommand first."
        )
    actions: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            actions[str(row["key"])] = str(row["action"])
    return actions


def cross_judge(
    run_dir: Path,
    *,
    n: int = judge_module.CROSS_JUDGE_RECORDS,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, int]:
    """Re-judge a stratified subset with a second judge, because the judge is also a subject."""
    stimulus = load_stimulus()
    skipped_gate = refuse_unvalidated_judge(run_dir, stimulus, skip_gate=skip_validation_gate)
    replies = scans_module.load_run_replies(run_dir)
    actions = _scan_actions(run_dir)
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    subset = judge_module.stratified_subset(
        records,
        n=n,
        outer_stratum=lambda record: str(record.get("model_id")),
        stratum=lambda record: (
            f"{record.get('model_id')}|{record.get('cell')}|{actions.get(str(record['key']))}"
        ),
    )
    out_path = run_dir / "cross-judged.jsonl"
    backend = _judge_backend(
        judge_module.CROSS_JUDGE_MODEL_ID,
        judge_module.CROSS_JUDGE_REASONING_EFFORT,
        concurrency=concurrency,
    )
    counts = judge_module.judge_records(backend, subset, out_path, stimulus)
    write_summary(
        run_dir,
        "cross-judge",
        {
            "command": "cross-judge",
            "judge_model_id": judge_module.CROSS_JUDGE_MODEL_ID,
            "judge_concurrency": concurrency,
            "judge_prompt_digest": judge_module.rubric_digest(stimulus),
            "skipped_validation_gate": skipped_gate,
            "requested": n,
            "selected": len(subset),
            "strata": len({actions.get(str(record["key"])) for record in subset}),
            **counts,
        },
    )
    sys.stdout.write(json.dumps(counts, indent=2) + "\n")
    return counts


def add_ladder(run_dir: Path, model: str) -> dict[str, Any]:
    """Record that one model's on-demand legs are authorized for this run, after the band read.

    Per model rather than per leg, because the decision is about the model: a band read admits its
    ladder, and for Opus 5 the same record is what admits re-submitting a row the pass does not read.
    The summary lists every leg the authorization unlocks, so the run dir says what was opened.
    """
    model_id = ON_DEMAND_MODEL_CHOICES[model]
    authorized = sorted({*authorized_on_demand_models(run_dir), model_id})
    path = run_dir / ADDED_LADDER_FILENAME
    refuse_tracked_trace_path(path, carries="which on-demand ladder legs this run authorized")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_ids": authorized,
        "authorized_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    legs = [leg.leg_id for leg in all_legs() if leg.on_demand and leg.model_id in authorized]
    summary = {"command": "add-ladder", "authorized": authorized, "legs": legs}
    write_summary(run_dir, "add-ladder", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _scripted_reply(prompt: str) -> str:
    """Answer a game prompt offline by echoing the first action label the prompt itself offers.

    Echoing rather than a fixed string, so the offline smoke exercises the real extraction path:
    ``strip_thinking`` then ``parse_action`` against this row's own labels, which is where a label
    mapping bug would surface.
    """
    options = _ACTION_OPTION_RE.findall(prompt)
    if not options:
        raise ValueError("a rendered game prompt must offer at least one action tag to echo")
    return f"A scripted reply for the offline smoke. <action>{options[0]}</action>"


_SCRIPTED_JUDGE_VERDICT = json.dumps(
    {
        "action_label": judge_module.ACTION_NONE,
        "counterpart_reasoning": "not_discussed",
        "identity_mentioned": False,
        "they_are_me": False,
        "all_instances_policy": False,
        "ev_arithmetic": False,
        "fairness_or_norm": False,
        "evidence": "",
    }
)
"""A fixed off-the-shelf verdict for the offline smoke.

``none`` and ``not_discussed`` are the two values that parse against ANY row's labels, which is what
lets one scripted string exercise the whole judge path. It follows that the offline smoke's
validation report will show misses against the registered expectations: that is the miss-reporting
path working, not a failing judge, and the live smoke is where real agreement is read.
"""


def smoke_leg(transport: str) -> Leg:
    """Build the leg the smoke renders from: the cheap model's anchor block, on its transport."""
    return Leg(SMOKE_LIVE_MODEL_ID, transport, BLOCK_ANCHOR, ANCHOR_CELLS, None, SITTING_A)


def smoke_calls(
    stimulus: DecoupledStimulus, *, transport: str, rows_per_cell: int
) -> list[PlannedCall]:
    """Take the first few draw-zero calls of each anchor cell, on the smoke's transport.

    Rendered from the real anchor leg and then narrowed, rather than from a smoke-shaped leg of its
    own, so the smoke exercises exactly the prompts and keys production will send.
    """
    by_cell: dict[str, list[PlannedCall]] = {}
    for call in planned_calls_for_leg(smoke_leg(transport), stimulus):
        if call.draw != 0:
            continue
        cell_calls = by_cell.setdefault(call.cell, [])
        if len(cell_calls) < rows_per_cell:
            cell_calls.append(call)
    return [call for cell in ANCHOR_CELLS for call in by_cell[cell]]


def inline_think_counts(replies: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Count per model how many replies carried a thinking block inside the answer channel.

    Reported with its denominator rather than as a bare count, because the number that matters is the
    share: a roster row that reasons inside the answer channel is one whose visible reply is not what
    the transport returned, and every instrument here reads the reply after that block is cut out.
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

    The judge and the deterministic scan are meant to read ONE text, and this is the end-to-end check
    that they do: every judged row records whether the reply it was given carried a thinking block in
    the answer channel, so recomputing that from the reply on disk and comparing is the only place a
    judge quietly handed the raw reply would show up. Returns how many judged rows carried one.
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
            f"{len(mismatched)} judged rows disagree with their replies about whether a thinking "
            f"block was emitted in the answer channel (first: {mismatched[:3]}). The judge and the "
            f"deterministic scan must read the same visible text, or their disagreement rate -- one "
            f"of this pass's reported instrument findings -- is an artifact of two different inputs."
        )
    return sum(1 for row in rows.values() if row.get("had_inline_think"))


def smoke(run_dir: Path, *, backend: str) -> dict[str, Any]:
    """Run the end-to-end smoke: render, sample, scan, judge-validate, judge, all under ``smoke/``.

    ``scripted`` spends nothing and is the gate before any real submit: two rows through the live
    resume loop on an offline backend, then the deterministic scans and both judge passes on scripted
    verdicts. ``live`` is the same path against real endpoints on the cheap reasoning-bearing model,
    two rows of each anchor cell at one draw. Everything lands under ``<run-dir>/smoke/<backend>/``
    so the production replies never carry smoke records and the two backends never resume each
    other's.
    """
    stimulus = load_stimulus()
    # One subtree per backend, because the two smokes render the SAME keys: a live smoke sharing the
    # scripted one's reply file would resume its offline records and only call for the remainder.
    smoke_dir = run_dir / "smoke" / backend
    scripted = backend == "scripted"
    transport = ScriptedDetailedBackend.transport if scripted else TRANSPORT_LIVE
    calls = smoke_calls(stimulus, transport=transport, rows_per_cell=SMOKE_LIVE_ROWS_PER_CELL)
    if scripted:
        calls = calls[:SMOKE_SCRIPTED_CALLS]
    out_path = reply_path(smoke_dir, smoke_leg(transport))
    audit_planned_calls(calls)

    def scripted_factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        del model_id, effort, concurrency
        return ScriptedDetailedBackend(_scripted_reply)

    counts = _run_live_calls(
        calls,
        out_path,
        stimulus,
        concurrency=2,
        chunk_size=len(calls),
        backend_factory=scripted_factory if scripted else None,
    )
    scan_totals = scans_module.scan_run(smoke_dir)
    judge_backend: DetailedBackend = (
        ScriptedDetailedBackend([_SCRIPTED_JUDGE_VERDICT])
        if scripted
        else _judge_backend(
            judge_module.JUDGE_MODEL_ID,
            judge_module.JUDGE_REASONING_EFFORT,
            concurrency=DEFAULT_JUDGE_CONCURRENCY,
        )
    )
    validation = judge_module.validate_judge(
        judge_backend, stimulus, smoke_dir / "judge-validation.jsonl"
    )
    replies = scans_module.load_run_replies(smoke_dir)
    judge_counts = judge_module.judge_records(
        judge_backend,
        [replies[key] for key in sorted(replies)],
        smoke_dir / "judged.jsonl",
        stimulus,
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
        "backend": backend,
        "model_id": SMOKE_LIVE_MODEL_ID,
        "calls": len(calls),
        **counts,
        "scans": scan_totals,
        "judge_validation": validation,
        "judge_validation_misses": validation["misses"],
        "judge_counts": judge_counts,
        "inline_think_by_model": inline_think_counts(replies),
        "judged_rows_with_inline_think": judged_with_inline_think,
        "measured_chars_per_token": [round(ratio, 2) for ratio in ratios],
        "stimulus_digest": stimulus.digest,
    }
    write_summary(smoke_dir, "smoke", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _at_least_one(text: str) -> int:
    """Parse an argparse integer that must be 1 or more, refusing at the parser rather than deep in a run."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="render every leg, audit every prompt, write plan.json")
    sub.add_parser("dry-run", help="count and price every leg; no AWS calls")
    smoke_parser = sub.add_parser("smoke", help="end-to-end smoke, offline or on the live path")
    smoke_parser.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    leg_ids = sorted(LEGS_BY_ID)
    submit = sub.add_parser("submit-batch", help="submit ONE pooled batch job for one leg")
    submit.add_argument("--leg", choices=leg_ids, required=True)
    submit.add_argument(
        "--check-only",
        action="store_true",
        help="run the floor and path checks and stop before anything is submitted or billed",
    )
    submit.add_argument(
        "--canary-calls",
        type=_at_least_one,
        default=REFUSAL_CANARY_CALLS,
        help=(
            "live calls per cell the refusal canary sends before a Claude leg is submitted; a cell "
            "that answers none of them refuses the submit (see sociology/refusal_canary.py)"
        ),
    )
    collect = sub.add_parser("collect-batch", help="collect one leg's job into its replies file")
    collect.add_argument("--leg", choices=leg_ids, required=True)
    collect.add_argument("--timeout-seconds", type=float, default=BATCH_TIMEOUT_SECONDS)
    live = sub.add_parser("run-live", help="run one live leg with resume-by-key")
    live.add_argument("--leg", choices=leg_ids, required=True)
    live.add_argument("--concurrency", type=int, default=16)
    live.add_argument("--chunk-size", type=int, default=32)
    sub.add_parser("scan", help="deterministic scans over every reply on disk")
    judge_parser = sub.add_parser("judge", help="blind judge over every reply on disk (resumable)")
    judge_parser.add_argument("--limit", type=int, default=None)
    sub.add_parser("judge-validate", help="judge the stimulus file's validation replies")
    cross = sub.add_parser("cross-judge", help="re-judge a stratified subset with a second judge")
    cross.add_argument("--n", type=int, default=judge_module.CROSS_JUDGE_RECORDS)
    for gated in (judge_parser, cross):
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
                "judge calls in flight at once; recorded as judge_concurrency in the summary "
                "(the hatch judge ran Luna at 32-48 without throttling)"
            ),
        )
    ladder = sub.add_parser(
        "add-ladder", help="authorize one on-demand ladder leg after the band read"
    )
    ladder.add_argument("--model", choices=sorted(ON_DEMAND_MODEL_CHOICES), required=True)
    return parser.parse_args(argv)


_DISPATCH_WITHOUT_ARGS = {"build": build, "dry-run": dry_run, "scan": scan}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one decoupled-ladder subcommand."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    run_dir: Path = args.run_dir
    simple = _DISPATCH_WITHOUT_ARGS.get(args.command)
    if simple is not None:
        simple(run_dir)
    elif args.command == "smoke":
        smoke(run_dir, backend=args.backend)
    elif args.command == "submit-batch":
        submit_batch(args.leg, run_dir, check_only=args.check_only, canary_calls=args.canary_calls)
    elif args.command == "collect-batch":
        collect_batch(args.leg, run_dir, timeout_seconds=args.timeout_seconds)
    elif args.command == "run-live":
        run_live(args.leg, run_dir, concurrency=args.concurrency, chunk_size=args.chunk_size)
    elif args.command == "judge":
        judge(
            run_dir,
            limit=args.limit,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
    elif args.command == "judge-validate":
        judge_validate(run_dir)
    elif args.command == "cross-judge":
        cross_judge(
            run_dir,
            n=args.n,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
    else:
        add_ladder(run_dir, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
