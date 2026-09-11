"""The transfer CLI: build the plan, price it, smoke it, sample it, scan it, judge it.

One subcommand per step of the design's order of work, one ``--leg`` per batch job or live pass, and a
required ``--pass`` naming which of :data:`sociology.transfer_plan.PLAN_TABLES` the command operates on.
Nothing here decides anything: the leg tables live in :mod:`sociology.transfer_plan`, the scenarios and
clause skeletons in :mod:`sociology.transfer_stimulus`, the instruments in
:mod:`sociology.transfer_scans` and :mod:`sociology.transfer_judge`, and this module is the operator
surface over them.

Six habits this repository has paid for are enforced here rather than left to whoever runs it:

- **The pass is named, never defaulted.** Two passes share this machinery, and the first one is finished:
  its run directory is not resumed and its numbers were sampled on clause wording since corrected. A
  default ``--pass`` would let a command re-render, re-price or re-submit that pass because nobody typed
  the flag, and every artifact would still be complete and plausible. The run directory follows the pass
  unless one is given, so the two passes' replies cannot land in one directory either.

- **The plan is written down before any model call.** ``build`` renders every planned prompt, runs the
  one-inserted-paragraph audit over all of them AND the two-games-differ-by-two-sections audit, and
  writes ``plan.json`` with the per-leg record counts and both digests. A run whose plan was never
  written is a run nobody can say what it intended to sample.
- **Batch handles are guarded and reused, never re-submitted.** The tracked-path refusal runs BEFORE the
  submit (a refusal afterwards would orphan a job already paid for), and a handle already on disk is
  reused only after the five-way check that it describes THIS invocation's job.
- **Resume is by content-derived key, and by content.** Live passes skip keys already on disk, count them
  separately from what ran, and refuse a key whose record answers a different prompt or was rendered
  from different prompt material (the stimulus PROMPT digest, so a rubric edit is not a refusal);
  collects skip keys already written and count those separately too, and refuse a job whose submit
  recorded a different prompt digest than the stimulus now loaded.
- **The judge does not run before it has been validated** -- both arms of it, because the two rubrics are
  two instruments and the twin's extra field is the pass's headline read. The intent check carries the
  same gate on its own rubric, and needs it more, because its verdicts are used to correct figures rather
  than only to report a rate.
- **The summary is written LAST**, as the completion marker for every subcommand.

Every artifact lands under ``--run-dir``, which defaults under ``artifacts/`` because the replies carry
the authored frames inside their prompts' digests and the judge's quoted evidence verbatim; the writers
refuse a git-tracked destination outright.
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

from games.parsing import strip_thinking
from games.prompts import MATCHED_DECISION_TRANSFER_GAME_ID, TRANSFER_GAME_IDS
from games.provenance import git_sha
from reward_hacking.bedrock_batch import (
    BATCH_TIMEOUT_SECONDS,
    HANDLE_CARRIES,
    MIN_BATCH_RECORDS,
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
from sociology import transfer_judge as judge_module
from sociology import transfer_scans as scans_module
from sociology.decoupled_scans import refuse_pooled_stimulus_mixture
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
from sociology.refusal_canary import (
    REFUSAL_CANARY_CALLS,
    canary_plan,
    needs_refusal_canary,
    recorded_canary,
    run_refusal_canary,
)
from sociology.transfer_plan import (
    DRAWS,
    FINGERPRINT_BATCH_PASS_TOKEN,
    FINGERPRINT_BATCH_PREFIX,
    LEGS_BY_ID,
    LIVE_MODEL_IDS,
    LUNA_MODEL_ID,
    MAX_TOKENS,
    OPUS_MODEL_ID,
    PAIR_MODEL_IDS,
    PASS_CORRELATION_DOSE,
    PASS_FINGERPRINT,
    PASS_IDS,
    PLAN_TABLES,
    ROWS_PER_CELL,
    SITTING_A,
    SONNET_MODEL_ID,
    TRANSPORT_BATCH,
    TRANSPORT_LIVE,
    Leg,
    PlannedCall,
    PlanTable,
    audit_games_differ_by_mechanics_and_clause,
    audit_planned_calls,
    batch_price_per_mtok,
    matched_rounds_thresholds,
    plan_table,
    planned_calls_for_leg,
    refuse_below_batch_floor,
    spec_for,
)
from sociology.transfer_stimulus import (
    BOARD_CONTENT_SIDE,
    BOARD_IDS,
    BOARD_MESSAGE_COUNT,
    BOARD_MODEL_ID_BY_SIDE,
    BOARD_PROVENANCE_FIELDS,
    BOARD_REDRAW_CAP,
    BOARD_SAMPLING_KEYS,
    BOARD_SOURCE_BOARD,
    BOARD_SOURCE_DIGEST_FIELD,
    BOARD_WORDING_SIDE,
    BOARDS_FIELD,
    PARAPHRASE_MESSAGE_PLACEHOLDER,
    RAW_BOARD_IDS,
    RECORD_RUNG_MATCHED,
    STIMULUS_PATH,
    STIMULUS_VERSION,
    board_message_problem,
    load_stimulus,
    message_digest,
    prompt_digest_of,
    scenario_nouns,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend
    from sociology.records import LiveBackendFactory
    from sociology.transfer_stimulus import TransferStimulus

logger = logging.getLogger(__name__)

REPLY_CARRIES = "verbatim model replies to the authored transfer frames"

PLAN_FILENAME = "plan.json"
ADDED_OPUS_FILENAME = "added-opus.json"
JUDGE_VALIDATION_LABEL = "judge-validation"
JUDGE_VALIDATION_SUMMARY = f"summary-{JUDGE_VALIDATION_LABEL}.json"

CHARS_PER_TOKEN_ESTIMATE = 3.5
"""Conservative chars-per-token for the dry run's input estimate; measured by the smoke afterwards."""

ASSUMED_OUTPUT_TOKENS_PER_CALL = 2_000
"""Dry-run output assumption per reply, stated rather than hidden inside a total.

Anchored on the roster's own note that GPT-OSS 120B emits about 1,780 output tokens per call at default
effort, rounded up because a figure answer invites arithmetic and because Claude bills adaptive thinking
against the same budget. The smoke reports measured chars-per-token and the collects report real usage,
so this number is only ever load-bearing before the first record exists.
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
    OPUS_MODEL_ID: LivePrice(
        (5.0, 25.0),
        "on-demand list price as used by the analysis-model pass 2026-08-31; UNVERIFIED against the "
        "price list",
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

SMOKE_LIVE_ROWS_PER_CELL = 2
SMOKE_SCRIPTED_CALLS = 4
"""Four rather than two, so the offline smoke covers both answer polarities in both directions."""

DEFAULT_JUDGE_CONCURRENCY = DEFAULT_BEDROCK_CONCURRENCY
"""Judge calls in flight when ``--concurrency`` is not given: the transport's own default.

A throughput knob rather than a sampling one, so raising it changes wall clock and nothing about what is
measured. It is stamped into every judge summary so a slow or throttled pass can be read back to the
number it actually ran at.
"""

_SET_TAG_RE = re.compile(r"<(set|keep)>N</\1>")

ON_DEMAND_MODEL_CHOICES: dict[str, str] = {"opus": OPUS_MODEL_ID}
"""The ``add-opus --model`` handles, one per model with an on-demand leg in the plan."""

OPUS_PROBE_CALLS = 20
"""How many live calls the Opus refusal probe makes before its legs are considered at all.

Twenty because per-sample classifier refusal is stimulus-dependent and the previous pass measured ~97%
refusal on a different stimulus: twenty calls separate "refuses this stimulus too" from "refused that
one" without spending a leg to find out.
"""


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
    call: PlannedCall, completion: BedrockCompletion, stimulus: TransferStimulus
) -> dict[str, Any]:
    """Build one reply record: every planned label, the sampler stamp, the reply, and completeness.

    The prompt text is not stored -- it is re-derivable from the labels plus the stimulus file -- but its
    digest and length are, so a record can be proved to answer the prompt this code renders today.

    ``stimulus_digest`` carries the stimulus PROMPT digest (:attr:`TransferStimulus.prompt_digest`), not
    the whole-file one: the field keeps its name so every reader joins on the same column, but it now
    answers "were these prompts rendered from this material", which is the only question a sampled row
    has to answer. A rubric edit moves the whole-file digest and leaves every reply row labelled.
    """
    return {
        "key": call.key,
        "block": call.block,
        "game_id": call.game_id,
        "cell": call.cell,
        "variant": call.variant,
        "scenario_id": call.scenario_id,
        "polarity": call.polarity,
        "prompt_id": call.prompt_id,
        "endowment": call.endowment,
        "credit_numerator": call.credit_numerator,
        "credit_denominator": call.credit_denominator,
        "beneficiary_count": call.beneficiary_count,
        "own_stake_scale": call.own_stake_scale,
        "model_id": call.model_id,
        "transport": call.transport,
        "reasoning_effort": call.reasoning_effort,
        "sitting": call.sitting,
        "draw": call.draw,
        "board_condition": call.board_condition,
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

    No temperature, no top_p, no top_k anywhere in this pass. Three roster rows refuse them outright and
    a run that asked for one and silently got none would have measured something other than what it
    recorded.
    """
    return BedrockSamplingConfig(max_tokens=MAX_TOKENS, reasoning_effort=leg.reasoning_effort)


def authorized_on_demand_models(run_dir: Path) -> list[str]:
    """Read which on-demand legs this run has authorized, if any."""
    path = run_dir / ADDED_OPUS_FILENAME
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(model_id) for model_id in payload.get("model_ids", [])]


def _refuse_unauthorized_leg(leg: Leg, run_dir: Path) -> None:
    """Refuse an on-demand leg the run has not authorized with ``add-opus``.

    Opus 5's legs wait on its refusal probe: per-sample classifier refusal is stimulus-dependent, and the
    previous pass's ~97% refusal rate makes spending on this row a decision rather than a default. Making
    it a recorded authorization rather than an operator's memory means the run dir says why the leg was
    submitted.
    """
    if not leg.on_demand or leg.model_id in authorized_on_demand_models(run_dir):
        return
    short = next(
        (name for name, model_id in ON_DEMAND_MODEL_CHOICES.items() if model_id == leg.model_id),
        leg.model_id,
    )
    raise RuntimeError(
        f"leg {leg.leg_id} is an on-demand leg and this run has not authorized it. Run the "
        f"{OPUS_PROBE_CALLS}-call refusal probe first, then record the decision with "
        f"`add-opus --model {short} --run-dir {run_dir}`."
    )


def build(run_dir: Path, table: PlanTable, *, allow_empty_boards: bool = False) -> dict[str, Any]:
    """Render every leg of one pass, audit every prompt, and write ``plan.json`` before any model call.

    Two audits, both load-bearing. Each cell's prompt must be its identity-blind stem plus exactly one
    counterpart paragraph, so a movement between cells is attributable to the clause. And every PAIR of
    the games' prompts for one (scenario, dose, polarity) must differ in the mechanics paragraph and the
    counterpart paragraph and in nothing else, so no pair's contrast is confounded with observability.

    ``allow_empty_boards`` loads a stimulus file whose fingerprint boards have not been generated yet,
    which is the state between the author writing the v3 fields and ``generate-boards`` filling them. It
    lets the dose pass and the scripted smoke be built in that window and is NOT a sampling escape: a
    board cell still refuses to render, so pass D cannot be planned, priced or sampled without its boards.
    """
    plan_path = run_dir / PLAN_FILENAME
    # Before the render, not after it: an operator who pointed --run-dir at a tracked directory should
    # hear about it now rather than a minute of rendering later.
    refuse_tracked_trace_path(plan_path, carries="the planned-call table and its digests")
    stimulus = load_stimulus(allow_empty_boards=allow_empty_boards)
    pairs_compared = audit_games_differ_by_mechanics_and_clause(stimulus)
    legs: list[dict[str, Any]] = []
    audited = 0
    for leg in table.all_legs():
        calls = planned_calls_for_leg(leg, stimulus)
        audited += audit_planned_calls(calls, stimulus)
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
                "cells": [f"{cell.game_id}|{cell.cell_id}|{cell.variant}" for cell in leg.cells],
                "reasoning_effort": leg.reasoning_effort,
                "sitting": leg.sitting,
                "on_demand": leg.on_demand,
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
    record_rungs = sorted(
        {
            cell.cell_id
            for cells in table.cells_by_block.values()
            for cell in cells
            if cell.cell_id in RECORD_RUNG_MATCHED
        }
    )
    plan = {
        "pass_id": table.pass_id,
        "stimulus_version": STIMULUS_VERSION,
        "allow_empty_boards": allow_empty_boards,
        "stimulus_digest": stimulus.digest,
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "rows_per_cell": ROWS_PER_CELL,
        "draws": DRAWS,
        "max_tokens": MAX_TOKENS,
        "audited_prompts": audited,
        "game_pairs_compared": pairs_compared,
        "legs": legs,
        "total_records": sum(int(entry["records"]) for entry in legs),
        "production_records": sum(
            int(entry["records"]) for entry in legs if not entry["on_demand"]
        ),
        # The thresholds are a function of the dose alone, and they are written into the plan of any pass
        # that samples a stated record so the readout marks numbers this code computed rather than
        # numbers somebody typed beside a curve.
        "matched_rounds_thresholds": (
            matched_rounds_thresholds(spec_for(MATCHED_DECISION_TRANSFER_GAME_ID))
            if record_rungs
            else None
        ),
        "record_rungs_planned": record_rungs,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    write_summary(run_dir, "build", {"command": "build", **plan})
    sys.stdout.write(json.dumps(plan, indent=2) + "\n")
    return plan


def dry_run(run_dir: Path, table: PlanTable) -> dict[str, Any]:
    """Count and price one pass's legs without touching AWS, every assumption stated separately.

    Two totals: the production legs alone, and every leg the plan can be asked for. Legs priced off an
    unverified figure are listed by id, so an unverified number cannot hide inside a total.
    """
    stimulus = load_stimulus()
    report: dict[str, Any] = {"pass_id": table.pass_id, "legs": [], "assumptions": {}}
    total = 0.0
    priced_unverified: list[str] = []
    for leg in table.all_legs():
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

    Cell membership is positional, so collecting someone else's job against these locally rendered inputs
    would file real responses under the wrong cells or sampler labels -- every count plausible and wrong.
    Five things are compared, and each has failed somewhere in this repo's history: the model id, the
    record count, the prompt digest, the cell digest, and the sampling labels (which matter because the
    records' sampler stamp comes from this invocation's config, not the handle's).
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
    """Build the batch backend for one leg, namespaced so two legs never share a path or a job name.

    The prefix comes off the leg rather than from a module constant, because it is the leg's own pass that
    decides it: two passes under one prefix would overwrite each other's ``input.jsonl`` the first time a
    run directory name repeated, and a submit and its collect must resolve the same path either way.
    """
    return BedrockBatchBackend(
        leg.model_id,
        sampling=_leg_sampling(leg),
        prefix=f"{leg.batch_prefix}/{run_dir.name}",
        run_id=leg.batch_run_id,
    )


def _canary_cell(call: PlannedCall) -> str:
    """Name the group a transfer canary call stands for: the cell, i.e. one (game, identity, dose) triple."""
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
    its evidence in the run dir; the canary's replies are never written anywhere. The transfer roster's
    one Claude batch row is Haiku 4.5, on a stimulus introduced because Opus refused ~97% of the last
    one, so this leg is the canary's reason for existing as much as the ladder's Opus legs are.
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


def submit_batch(  # noqa: PLR0913 - one keyword per guard; the last two are the offline tests' seams
    leg_id: str,
    run_dir: Path,
    table: PlanTable,
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
    leg = table.leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    _refuse_unauthorized_leg(leg, run_dir)
    stimulus = load_stimulus()
    calls = planned_calls_for_leg(leg, stimulus)
    refuse_below_batch_floor(leg, len(calls))
    path = handle_path(run_dir, leg)
    # Before the submit, unconditionally: the handle carries the account id, bucket and profile, and a
    # refusal after the create call would orphan a job already paid for.
    refuse_tracked_trace_path(path, carries=HANDLE_CARRIES)
    if check_only:
        summary = {
            "command": "submit-batch",
            "leg_id": leg.leg_id,
            "check_only": True,
            "records": len(calls),
            "handle_exists": path.exists(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
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
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "resumed_handle": resumed,
        "stimulus_prompt_digest_recorded_at_submit": digest_recorded_at_submit,
        "refusal_canary": canary,
    }
    write_summary(run_dir, _submit_summary_label(leg), summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _refuse_resumed_submit_under_a_different_stimulus(
    run_dir: Path, leg: Leg, stimulus: TransferStimulus
) -> bool:
    """Refuse to rewrite a submit summary with a stimulus prompt digest the job never ran under.

    ``submit-batch`` against an existing handle submits nothing and rewrites the summary, which is the
    documented repair for a lost summary and, unguarded, the one way to launder the collect-time guard:
    the rewrite replaces the recorded digest with whatever is loaded now, so
    :func:`_refuse_collect_under_a_different_stimulus` then compares that value with itself and agrees.
    The five-way handle check cannot see it either, because a prompt-affecting field this leg never
    renders leaves every one of its prompts byte-identical.

    Returns whether the digest really was recorded at submit time. A summary that is simply absent
    carries no evidence in either direction, so the repair still runs and the rewritten summary records
    that its digest was reconstructed rather than claiming a provenance it does not have.
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
    run_dir: Path, leg: Leg, stimulus: TransferStimulus, *, stimulus_unchanged_since_submit: bool
) -> None:
    """Refuse to collect a job whose submit recorded a different stimulus prompt digest.

    Every collected row is stamped with the prompt digest of the stimulus loaded NOW, so a stimulus whose
    prompt material moved between submit and collect would label rows sampled under one file as answers
    to another. The handle's own prompt digest catches most of that (the rendered text moved), but not a
    fragment or frame edit this leg never renders, and the handle has no slot for the stimulus digest;
    the submit summary is where it was recorded. A missing summary refuses too: re-running
    ``submit-batch`` reuses the saved handle after the five-way check and rewrites the summary, so the
    fix is one cheap command rather than a guess.

    That repair is also why the recorded flag is read here rather than only written at submit. A summary
    the repair rebuilt records the digest loaded at repair time, so comparing it proves nothing about
    what the job was sampled under; the collect refuses on it and asks the operator to say, in the
    invocation, that the prompt material has not moved. A summary written before the flag existed says
    nothing either way and warns instead of refusing: those runs are collected already, and refusing
    them would be a gate that fires only on history nobody can change.
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
            f"leg {leg.leg_id} was submitted under stimulus prompt digest {recorded!r} (per {path}) "
            f"and this invocation loaded {stimulus.prompt_digest!r}. Collecting would stamp rows sampled "
            f"under one stimulus as answers to another. Restore the stimulus file the submit used, or "
            f"point --run-dir somewhere fresh and submit (and pay for) a new job."
        )


def collect_batch(
    leg_id: str,
    run_dir: Path,
    table: PlanTable,
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
    leg = table.leg_for(leg_id)
    if leg.transport != TRANSPORT_BATCH:
        raise ValueError(f"leg {leg_id} runs on the {leg.transport} transport, not batch")
    stimulus = load_stimulus()
    calls = planned_calls_for_leg(leg, stimulus)
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
        "stimulus_prompt_digest": stimulus.prompt_digest,
        "stimulus_unchanged_since_submit_asserted": stimulus_unchanged_since_submit,
    }
    write_summary(run_dir, f"collect-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def run_live(
    leg_id: str, run_dir: Path, table: PlanTable, *, concurrency: int, chunk_size: int
) -> dict[str, Any]:
    """Run one live leg with resume-by-key, then write the summary as the completion marker."""
    leg = table.leg_for(leg_id)
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
        "stimulus_prompt_digest": stimulus.prompt_digest,
    }
    write_summary(run_dir, f"live-{leg.file_stem}", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _run_live_calls(  # noqa: PLR0913 - one argument per seam; the last one is the testable one
    calls: Sequence[PlannedCall],
    out_path: Path,
    stimulus: TransferStimulus,
    *,
    concurrency: int,
    chunk_size: int,
    backend_factory: LiveBackendFactory | None = None,
) -> dict[str, int]:
    """Run planned calls through the shared resume loop with this pass's row schema.

    ``resume_identity`` is what makes the resume safe rather than merely cheap: a key already on disk is
    skipped only if its record was produced for this same prompt from this same prompt material. An
    edited frame or clause renders new prompts under the OLD keys, and without this the second pass would
    skip every one of them and report itself complete. The stimulus side compares the PROMPT digest, not
    the whole-file one: a rubric or validation-reply edit changes what the judge reads and nothing a reply
    row answered, and refusing every resume for it was the failure this split removed.
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
        carries=REPLY_CARRIES,
    )


def _resume_identity(call: PlannedCall, stimulus: TransferStimulus) -> dict[str, Any]:
    """Say what a record on disk has to carry before its key may be skipped as already answered.

    One function for both transports rather than a lambda inside the live loop, because a skipped key
    means the same thing on either: this call was already answered. The batch collect used to skip on the
    key alone, so a reply file that outlived its handle and its submit summary could be re-collected
    under an edited stimulus and every surviving row counted as resumed.
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


def rubric_digests(stimulus: TransferStimulus) -> dict[str, str]:
    """Digest both arms' rubrics, which is what a validation summary has to clear this run against."""
    return {
        game_id: judge_module.judge_digest(stimulus, game_id)
        for game_id in judge_module.VERDICT_SCHEMA_BY_GAME
    }


def refuse_unvalidated_judge(run_dir: Path, stimulus: TransferStimulus, *, skip_gate: bool) -> bool:
    """Refuse a judge pass whose rubrics have not been validated in this run; return whether skipped.

    Per rubric, per ARM and per prompt version: a validation run against an earlier wording says nothing
    about this one, a validation of the one-way rubric says nothing about the twin's, which is where the
    pass's headline field lives, and a code-side version bump with the authored rubric text unchanged
    re-judges every row (:func:`sociology.judge_loop.judged_under_current_rubric` compares the version)
    under a scaffold no validation has covered. Zero misses and zero unparsed replies as well, because a
    judge that disagreed with a registered verdict will misreport a rate and look perfectly healthy doing
    it.

    ``skip_gate`` exists for the case where a human has read the misses and decided to run anyway. It
    warns loudly and is stamped into the judge summary, so a table read months later says the judge behind
    it was never validated.
    """
    if skip_gate:
        logger.warning(
            "SKIPPING THE JUDGE-VALIDATION GATE: this pass will judge with rubrics nobody has checked "
            "against the registered verdicts in this run. Every rate it produces is uncalibrated, and "
            "the summary records skipped_validation_gate=true for that reason."
        )
        return True
    path = run_dir / JUDGE_VALIDATION_SUMMARY
    digests = rubric_digests(stimulus)
    shape = judge_module.judge_prompt_shape_digest()
    if not path.exists():
        raise RuntimeError(
            f"{path} does not exist, so this run has never validated its judge. Run "
            f"`judge-validate --run-dir {run_dir}` first (or pass --skip-validation-gate to judge with "
            f"unvalidated rubrics, which is recorded in the summary)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if payload.get("judge_prompt_digests") != digests:
        problems.append(
            f"it validated rubric digests {payload.get('judge_prompt_digests')}, not this run's "
            f"{digests}"
        )
    if payload.get("judge_prompt_version") != judge_module.JUDGE_PROMPT_VERSION:
        problems.append(
            f"it validated prompt version {payload.get('judge_prompt_version')!r}, not this run's "
            f"{judge_module.JUDGE_PROMPT_VERSION!r}"
        )
    if payload.get("judge_prompt_shape_digest") != shape:
        problems.append(
            f"it validated prompt-shape digest {payload.get('judge_prompt_shape_digest')!r}, not this "
            f"run's {shape!r}: the asked-for headers or another code-side section of the prompt has "
            f"changed since that validation"
        )
    if payload.get("misses"):
        problems.append(f"it reported {len(payload['misses'])} field misses")
    if payload.get("unparsed"):
        problems.append(f"it reported {len(payload['unparsed'])} unparsed judge replies")
    if problems:
        raise RuntimeError(
            f"the judge validation in {path} does not clear this run: {'; '.join(problems)}. Run "
            f"`judge-validate --run-dir {run_dir}` against the current rubrics and read its misses "
            f"before judging (or pass --skip-validation-gate, which is recorded in the summary)."
        )
    return False


def _pooled_replies(run_dir: Path, *, what: str) -> dict[str, dict[str, Any]]:
    """Every reply on disk, for an instrument that pools rows across legs and reports one rate."""
    replies = scans_module.load_run_replies(run_dir)
    refuse_pooled_stimulus_mixture(replies, run_dir=run_dir, what=what)
    return replies


def judge(
    run_dir: Path,
    *,
    limit: int | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Run the blind judge over every reply on disk, each arm under its own rubric (resumable)."""
    stimulus = load_stimulus()
    skipped_gate = refuse_unvalidated_judge(run_dir, stimulus, skip_gate=skip_validation_gate)
    replies = _pooled_replies(run_dir, what="the judge")
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    if limit is not None:
        records = records[:limit]
    out_path = run_dir / "judged.jsonl"
    backend = _judge_backend(
        judge_module.JUDGE_MODEL_ID, judge_module.JUDGE_REASONING_EFFORT, concurrency=concurrency
    )
    counts = judge_module.judge_run(backend, records, out_path, stimulus)
    write_summary(
        run_dir,
        "judge",
        {
            "command": "judge",
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_effort": judge_module.JUDGE_REASONING_EFFORT,
            "judge_concurrency": concurrency,
            "judge_prompt_digests": rubric_digests(stimulus),
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
    """Judge both arms' hand-authored validation replies and report every disagreement.

    The summary this writes is what the judge and cross-judge subcommands gate on, so it carries both
    rubrics' digests and the prompt version: a validation of an earlier wording must not clear a later
    one, and a validation of one arm must not clear the other.
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
            "judge_prompt_digests": rubric_digests(stimulus),
            "judge_prompt_shape_digest": judge_module.judge_prompt_shape_digest(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
            **report,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


INTENT_CHECKED_FILENAME = "intent-checked.jsonl"
INTENT_VALIDATION_LABEL = "intent-check-validation"
INTENT_VALIDATION_SUMMARY = f"summary-{INTENT_VALIDATION_LABEL}.json"

SCOPE_ENDPOINT = "endpoint"
SCOPE_ALL = "all"
INTENT_SCOPES: tuple[str, ...] = (SCOPE_ENDPOINT, SCOPE_ALL)
"""Which replies the intent check reads.

``endpoint`` is every scanned reply whose parsed figure sits at one end of the range, both polarities and
every model. That scope is chosen for the SLIP and for the slip alone: a tag reading as "the whole stock"
or as "nothing" is the only kind a reversal moves across the whole range, so an interior figure is never
corrected and reading it buys no correction. It is NOT the scope the reciprocal-misread flag wants --
``assumes_return`` can be true of any reply, interior figures included -- so under this scope that flag is
measured on the endpoint subset only, which the summary says and every rate built on it has to repeat.
``all`` reads the interior as well, which is a superset and costs proportionally more. Neither is a
sampling choice -- the pass reads replies already paid for -- so the scope is recorded in the summary and
the two are never pooled without saying which is which.
"""


def intent_checked_path(run_dir: Path) -> Path:
    """Where the intent check's rows land: one file for both arms, resumed by record key."""
    return run_dir / INTENT_CHECKED_FILENAME


def scan_figures_by_key(run_dir: Path) -> dict[str, tuple[int | None, int]]:
    """Every scan row's parsed figure and its own stock, keyed by record key.

    Read from the scans file rather than re-parsed here, so the endpoint scope and the slip counts are
    defined against the same figures every other table in this pass is built on. A run with no scans file
    refuses: selecting endpoints without them would silently read every reply, and counting slips without
    them would report zero slips on a healthy-looking summary. A file that is merely SHORT of the replies
    is caught in :func:`_intent_records` instead, which counts the replies it has no row for and refuses
    when that is all of them -- a distinction this function cannot make, since it never sees the replies.
    """
    path = run_dir / scans_module.SCANS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so no reply has a parsed figure to read a conclusion against. Run "
            f"the `scan` subcommand first."
        )
    figures: dict[str, tuple[int | None, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            units = row.get("set_units")
            figures[str(row["key"])] = (
                None if units is None else int(units),
                int(row["endowment"]),
            )
    return figures


POSITION_ENDPOINT = "endpoint"
POSITION_INTERIOR = "interior"
POSITION_UNPARSED = "unparsed"
POSITION_UNSCANNED = "unscanned"
FIGURE_POSITIONS: tuple[str, ...] = (
    POSITION_ENDPOINT,
    POSITION_INTERIOR,
    POSITION_UNPARSED,
    POSITION_UNSCANNED,
)
"""Where one reply's deterministic figure sits, as four states rather than a boolean.

The endpoint filter used to ask ``at_an_endpoint`` and take False for an answer three different ways: a
figure in the middle, a scanned reply that parsed to nothing, and a reply the scans file has no row for at
all. Only the first is a deliberate exclusion. The other two are the instrument being short of the pool it
is filtering, which is worth a count of its own -- a scans file written before the last leg landed would
otherwise silently narrow the pass and leave a summary that reads as a complete endpoint pass.
"""


def figure_position(figure: tuple[int | None, int] | None) -> str:
    """Classify one record's parsed figure: at an end of its range, inside it, unparsed, or unscanned."""
    if figure is None:
        return POSITION_UNSCANNED
    units, endowment = figure
    if units is None:
        return POSITION_UNPARSED
    return POSITION_ENDPOINT if units in (0, endowment) else POSITION_INTERIOR


def at_an_endpoint(figure: tuple[int | None, int] | None) -> bool:
    """Whether one record's parsed figure sits at an end of its own range: nothing, or the whole stock."""
    return figure_position(figure) == POSITION_ENDPOINT


def refuse_unvalidated_intent_check(
    run_dir: Path, stimulus: TransferStimulus, *, skip_gate: bool
) -> bool:
    """Refuse an intent-check pass whose rubric has not been validated in this run; return whether skipped.

    The same gate the judge has, and load-bearing for a stronger reason: this instrument's verdicts are
    used to REWRITE figures in the readout, so a reader that had ``concluded_action`` backwards on the
    authored slip would move real answers to the other end of the range while every table still looked
    healthy. Keyed on the digest of the rubric as loaded now, because a validation of an earlier wording
    says nothing about this one.

    Three keys rather than one, because the rubric digest covers the AUTHORED instructions and nothing
    else. The two rules paragraphs, the rung clauses, the tag-asked-for headers and the prompt version all
    live in code and sat outside that digest, so a validation made before a code-side prompt edit used to
    clear the pass that followed it -- while the per-row resume, which does compare the version, re-read
    every production row under a prompt no validation had ever covered.
    """
    if skip_gate:
        logger.warning(
            "SKIPPING THE INTENT-CHECK VALIDATION GATE: this pass will read conclusions with a rubric "
            "nobody has checked against the registered verdicts in this run, and its output corrects "
            "figures. The summary records skipped_validation_gate=true for that reason."
        )
        return True
    path = run_dir / INTENT_VALIDATION_SUMMARY
    digest = judge_module.intent_digest(stimulus)
    shape = judge_module.intent_prompt_shape_digest()
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
    if payload.get("intent_prompt_version") != judge_module.INTENT_CHECK_PROMPT_VERSION:
        problems.append(
            f"it validated prompt version {payload.get('intent_prompt_version')!r}, not this run's "
            f"{judge_module.INTENT_CHECK_PROMPT_VERSION!r}"
        )
    if payload.get("intent_prompt_shape_digest") != shape:
        problems.append(
            f"it validated prompt-shape digest {payload.get('intent_prompt_shape_digest')!r}, not this "
            f"run's {shape!r}: the rules paragraphs, the rung clauses or another code-side section of "
            f"the prompt has changed since that validation"
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


def intent_check_validate(run_dir: Path) -> dict[str, Any]:
    """Read the hand-authored intent cases and report every disagreement by reply and field."""
    stimulus = load_stimulus()
    out_path = run_dir / f"{INTENT_VALIDATION_LABEL}.jsonl"
    backend = _judge_backend(
        judge_module.INTENT_CHECK_MODEL_ID,
        judge_module.INTENT_CHECK_REASONING_EFFORT,
        concurrency=DEFAULT_JUDGE_CONCURRENCY,
    )
    report = judge_module.validate_intent_check(backend, stimulus, out_path)
    write_summary(
        run_dir,
        INTENT_VALIDATION_LABEL,
        {
            "command": "intent-check-validate",
            "judge_model_id": judge_module.INTENT_CHECK_MODEL_ID,
            "intent_prompt_version": judge_module.INTENT_CHECK_PROMPT_VERSION,
            "intent_prompt_digest": judge_module.intent_digest(stimulus),
            "intent_prompt_shape_digest": judge_module.intent_prompt_shape_digest(),
            "stimulus_digest": stimulus.digest,
            "stimulus_prompt_digest": stimulus.prompt_digest,
            **report,
        },
    )
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


@dataclass(frozen=True, slots=True)
class IntentSelection:
    """The replies one intent-check invocation reads, with every count that says what it left out."""

    records: list[Mapping[str, Any]]
    figures: dict[str, tuple[int | None, int]]
    replies_on_disk: int
    """Every reply row in the run directory, before the model filter or the scope."""
    after_model_filter: int
    """Replies left once ``--model`` has been applied; the denominator the scope then filters."""
    positions: dict[str, int]
    """How those replies' deterministic figures fall over :data:`FIGURE_POSITIONS`, counted in full.

    Printed whichever scope ran, because it is the only place a reader can see that the pass read the
    endpoint subset out of a pool that also held interior, unparsed and unscanned replies -- and
    ``unscanned`` in particular is the scans file being short of the replies rather than a design choice.
    """


def _intent_records(run_dir: Path, *, scope: str, model: str | None) -> IntentSelection:
    """Select the replies this invocation reads, with the scan figures and every excluded count."""
    if scope not in INTENT_SCOPES:
        raise ValueError(f"--scope must be one of {list(INTENT_SCOPES)}, not {scope!r}")
    replies = _pooled_replies(run_dir, what="the intent check")
    figures = scan_figures_by_key(run_dir)
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    pool = len(records)
    if model is not None:
        records = [record for record in records if str(record.get("model_id")) == model]
        if not records:
            present = sorted({str(row.get("model_id")) for row in replies.values()})
            raise ValueError(
                f"no reply in {run_dir} was written by model {model!r}; the replies on disk cover "
                f"{present}. A filtered pass over nothing would write a healthy-looking summary with "
                f"zero rows read."
            )
    after_model_filter = len(records)
    positions = dict.fromkeys(FIGURE_POSITIONS, 0)
    for record in records:
        positions[figure_position(figures.get(str(record["key"])))] += 1
    if positions[POSITION_UNSCANNED] == after_model_filter:
        raise ValueError(
            f"none of the {after_model_filter} replies selected in {run_dir} has a row in "
            f"{scans_module.SCANS_FILENAME}, so no conclusion could be read against a figure and every "
            f"cell would report zero slips over a full denominator. Run `scan` first."
        )
    if positions[POSITION_UNSCANNED]:
        logger.warning(
            f"{positions[POSITION_UNSCANNED]} of {after_model_filter} replies have no row in "
            f"{scans_module.SCANS_FILENAME} at all, so no figure of theirs can be read against a "
            f"conclusion; they are excluded under the {SCOPE_ENDPOINT} scope and counted as "
            f"without_scan_figure under {SCOPE_ALL}. Re-run `scan` if a leg landed after it."
        )
    if scope == SCOPE_ENDPOINT:
        records = [record for record in records if at_an_endpoint(figures.get(str(record["key"])))]
        if not records:
            raise ValueError(
                f"no reply in {run_dir} parsed to an end of its own range, so the endpoint scope selects "
                f"nothing (of {after_model_filter} replies: {positions}). Run `scan` first, or pass "
                f"--scope {SCOPE_ALL} to read the interior too."
            )
    return IntentSelection(
        records=records,
        figures=figures,
        replies_on_disk=pool,
        after_model_filter=after_model_filter,
        positions=positions,
    )


def _rows_under_the_current_intent_rubric(
    rows: Sequence[Mapping[str, Any]], stimulus: TransferStimulus
) -> int:
    """How many of the pooled intent rows were read under the rubric and version loaded now.

    The whole-file totals beside it are what the readout uses, and they pool every row ever written; after
    a rubric edit plus a model-narrowed re-run, the rows this invocation did not select still carry the old
    digest. The same predicate the per-row resume uses, so this figure and what the next pass would re-read
    cannot drift apart.
    """
    digest = judge_module.intent_digest(stimulus)
    return sum(
        1
        for row in rows
        if judge_module.judged_under_current_rubric(
            row, digest, prompt_version=judge_module.INTENT_CHECK_PROMPT_VERSION
        )
    )


def intent_check(
    run_dir: Path,
    *,
    scope: str = SCOPE_ENDPOINT,
    model: str | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Read what each selected reply's reasoning concluded, so a tag slip can be counted and corrected.

    A third instrument over replies already paid for, not a re-sample. It resumes by record key against
    ``intent-checked.jsonl`` exactly as the judge does, so a narrowed pass (one model, or the endpoint
    scope) appends to whatever is already there and a later wider pass fills in the rest. The per-cell
    counts in the summary are recomputed from the WHOLE file every time for that reason: a summary that
    only counted this invocation's rows would report the last narrow pass as though it were the run.

    Which is why the summary keeps the two apart in its own shape. ``invocation`` is what THIS call did --
    its scope, its model filter, what it selected, what it read, what it spent -- and
    ``over_the_whole_file`` is every row ever written to the file. Both in one flat object read as one
    thing, so a Sonnet-only endpoint pass published run-wide totals under a one-model label.
    """
    stimulus = load_stimulus()
    skipped_gate = refuse_unvalidated_intent_check(
        run_dir, stimulus, skip_gate=skip_validation_gate
    )
    selection = _intent_records(run_dir, scope=scope, model=model)
    out_path = intent_checked_path(run_dir)
    backend = _judge_backend(
        judge_module.INTENT_CHECK_MODEL_ID,
        judge_module.INTENT_CHECK_REASONING_EFFORT,
        concurrency=concurrency,
    )
    counts = judge_module.intent_check_run(backend, selection.records, out_path, stimulus)
    rows = list(judge_module.load_judged(out_path).values())
    per_cell = judge_module.intent_check_counts(
        rows, {key: units for key, (units, _endowment) in selection.figures.items()}
    )
    totals = {
        name: sum(cell[name] for cell in per_cell.values())
        for name in judge_module.INTENT_COUNT_NAMES
    }
    summary = {
        "command": "intent-check",
        "judge_model_id": judge_module.INTENT_CHECK_MODEL_ID,
        "judge_effort": judge_module.INTENT_CHECK_REASONING_EFFORT,
        "judge_concurrency": concurrency,
        "intent_prompt_version": judge_module.INTENT_CHECK_PROMPT_VERSION,
        "intent_prompt_digest": judge_module.intent_digest(stimulus),
        "intent_prompt_shape_digest": judge_module.intent_prompt_shape_digest(),
        "skipped_validation_gate": skipped_gate,
        "invocation": {
            "scope": scope,
            "model_filter": model,
            "replies_on_disk": selection.replies_on_disk,
            "after_model_filter": selection.after_model_filter,
            "figure_positions": selection.positions,
            "selected": len(selection.records),
            **counts,
            "usage": {
                "input_tokens": backend.usage.input_tokens,
                "output_tokens": backend.usage.output_tokens,
            },
        },
        "over_the_whole_file": {
            "rows": len(rows),
            "rows_by_rubric": judge_module.rows_by_rubric(rows),
            "rows_under_the_current_rubric": _rows_under_the_current_intent_rubric(rows, stimulus),
            "totals": totals,
            "by_cell": dict(sorted(per_cell.items())),
        },
    }
    write_summary(run_dir, "intent-check", summary)
    sys.stdout.write(
        json.dumps({"invocation": summary["invocation"], "totals": totals}, indent=2) + "\n"
    )
    return summary


def _scan_figures(run_dir: Path) -> dict[str, str]:
    """Read each record's deterministic parsed figure off the scans file, for the strata.

    Bucketed rather than exact, because a stratum per integer over a 21-value grid would make almost
    every record its own stratum and the subset would stop being stratified at all.
    """
    path = run_dir / scans_module.SCANS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so the cross-judge subset cannot be stratified by the parsed "
            f"figure. Run the `scan` subcommand first."
        )
    buckets: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            fraction = row.get("set_fraction")
            if fraction is None:
                bucket = "unparsed"
            elif float(fraction) == 0.0:
                bucket = "zero"
            elif float(fraction) >= 1.0:
                bucket = "all"
            else:
                bucket = "interior"
            buckets[str(row["key"])] = bucket
    return buckets


def cross_judge(
    run_dir: Path,
    *,
    n: int = judge_module.CROSS_JUDGE_RECORDS,
    game: str | None = None,
    skip_validation_gate: bool = False,
    concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> dict[str, Any]:
    """Re-judge a stratified subset with a second judge, because the judge is also a subject.

    Stratified per MODEL first and only then per cell: the agreement table is read per model, so a subset
    that reached two of nine models would not answer the question it is asked to. That is the defect the
    shared loop's :func:`~sociology.judge_loop.stratified_subset` was fixed for.

    ``game`` narrows the candidates to one arm BEFORE the draw. The twin arm is a quarter of the run and
    carries the only field the two judges disagree on much, so a draw over both arms leaves each model
    with a handful of twin rows; a twin-only draw of the same size gives each model a readable count.
    The output file and its resume-by-key are shared with the unfiltered draw, so a filtered pass appends
    to whatever is already judged, and the summary records the filter so the provenance says what was drawn.
    """
    if game is not None and game not in TRANSFER_GAME_IDS:
        raise ValueError(f"--game must be one of {list(TRANSFER_GAME_IDS)}, not {game!r}")
    stimulus = load_stimulus()
    skipped_gate = refuse_unvalidated_judge(run_dir, stimulus, skip_gate=skip_validation_gate)
    replies = _pooled_replies(run_dir, what="the cross-judge subset")
    figures = _scan_figures(run_dir)
    records: list[Mapping[str, Any]] = [replies[key] for key in sorted(replies)]
    if game is not None:
        records = [record for record in records if str(record.get("game_id")) == game]
        if not records:
            present = sorted({str(row.get("game_id")) for row in replies.values()})
            raise ValueError(
                f"no reply in {run_dir} belongs to game {game!r}; the replies on disk cover {present}. "
                f"A filtered draw over nothing would write a healthy-looking summary with zero rows judged."
            )
    subset = judge_module.stratified_subset(
        records,
        n=n,
        outer_stratum=lambda record: str(record.get("model_id")),
        stratum=lambda record: (
            f"{record.get('game_id')}|{record.get('cell')}|{record.get('variant')}"
            f"|{figures.get(str(record['key']))}"
        ),
    )
    out_path = run_dir / "cross-judged.jsonl"
    backend = _judge_backend(
        judge_module.CROSS_JUDGE_MODEL_ID,
        judge_module.CROSS_JUDGE_REASONING_EFFORT,
        concurrency=concurrency,
    )
    counts = judge_module.judge_run(backend, subset, out_path, stimulus)
    write_summary(
        run_dir,
        "cross-judge",
        {
            "command": "cross-judge",
            "judge_model_id": judge_module.CROSS_JUDGE_MODEL_ID,
            "judge_concurrency": concurrency,
            "judge_prompt_digests": rubric_digests(stimulus),
            "skipped_validation_gate": skipped_gate,
            "requested": n,
            "game_filter": game,
            "selected": len(subset),
            "models_covered": len({str(record.get("model_id")) for record in subset}),
            **counts,
        },
    )
    sys.stdout.write(json.dumps(counts, indent=2) + "\n")
    return counts


def add_opus(run_dir: Path, table: PlanTable, model: str) -> dict[str, Any]:
    """Record that one model's on-demand legs are authorized for this run, after its refusal probe.

    Per model rather than per leg, because the decision is about the model: a clean refusal probe admits
    its legs. The summary lists every leg the authorization unlocks, so the run dir says what was opened.
    """
    model_id = ON_DEMAND_MODEL_CHOICES[model]
    authorized = sorted({*authorized_on_demand_models(run_dir), model_id})
    path = run_dir / ADDED_OPUS_FILENAME
    refuse_tracked_trace_path(path, carries="which on-demand legs this run authorized")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"model_ids": authorized, "authorized_at": datetime.now(UTC).isoformat()}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    legs = [leg.leg_id for leg in table.all_legs() if leg.on_demand and leg.model_id in authorized]
    summary = {"command": "add-opus", "authorized": authorized, "legs": legs}
    write_summary(run_dir, "add-opus", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


# --- pass D: board generation -----------------------------------------------------------------------

BOARD_STAGE_RAW = "raw"
BOARD_STAGE_PARAPHRASE = "paraphrase"
BOARD_STAGES: tuple[str, ...] = (BOARD_STAGE_RAW, BOARD_STAGE_PARAPHRASE)
"""The two rounds of generation: each side's own messages, then each side's rewrite of the other's."""

BOARD_LIVE_PROBE_CALLS = 3
"""How many live calls the batch row's transport probe spends before the boards are drawn.

Three because the question is binary -- does this model answer a Converse call at all on this account --
and the row has never been called live from this repository. If it answers, the whole generation runs
live in a few minutes; if it refuses, the same material comes back through two batch jobs.
"""

BOARD_GENERATION_CONCURRENCY = 8
"""Live calls in flight while drawing boards. Small: this is about a hundred calls in total."""


@dataclass(frozen=True, slots=True)
class BoardSlot:
    """One board message to be drawn: where it belongs, what prompt draws it, and what it rewrites."""

    scenario_id: str
    board_id: str
    index: int
    prompt: str
    drawing_model_id: str
    source_message: str | None

    @property
    def what(self) -> str:
        """Name this slot the way a redraw log line and a refusal name it."""
        return f"{self.scenario_id}/{self.board_id} message {self.index}"


def _raw_board_slots(stimulus: TransferStimulus, board_id: str) -> list[BoardSlot]:
    """Every slot of one raw board: three draws of each scenario's own task by the board's own side."""
    model_id = BOARD_MODEL_ID_BY_SIDE[BOARD_CONTENT_SIDE[board_id]]
    return [
        BoardSlot(
            scenario_id=scenario.scenario_id,
            board_id=board_id,
            index=index,
            prompt=stimulus.unrelated_tasks[scenario.scenario_id].task,
            drawing_model_id=model_id,
            source_message=None,
        )
        for scenario in stimulus.scenarios
        for index in range(BOARD_MESSAGE_COUNT)
    ]


def _paraphrase_board_slots(
    stimulus: TransferStimulus, board_id: str, drawn: Mapping[tuple[str, str], list[str]]
) -> list[BoardSlot]:
    """Every slot of one crossed board: the wording side rewriting its source board, message for message."""
    model_id = BOARD_MODEL_ID_BY_SIDE[BOARD_WORDING_SIDE[board_id]]
    source_id = BOARD_SOURCE_BOARD[board_id]
    slots: list[BoardSlot] = []
    for scenario in stimulus.scenarios:
        sources = drawn[scenario.scenario_id, source_id]
        slots.extend(
            BoardSlot(
                scenario_id=scenario.scenario_id,
                board_id=board_id,
                index=index,
                prompt=stimulus.paraphrase_instruction.format(
                    **{PARAPHRASE_MESSAGE_PLACEHOLDER: source}
                ),
                drawing_model_id=model_id,
                source_message=source,
            )
            for index, source in enumerate(sources)
        )
    return slots


def _visible_reply(completion: BedrockCompletion) -> str:
    """Read one drawn message off a completion: the visible answer, thinking stripped.

    A completion whose thinking ran into the cap has no visible answer at all and comes back empty, which
    the board gate then refuses on length -- so it is re-drawn rather than written to the file as a board
    message of nothing.
    """
    visible, _truncated = strip_thinking(completion.text or "")
    return visible.strip()


def _live_round(
    slots: Sequence[BoardSlot], *, model_id: str, backend_factory: LiveBackendFactory
) -> list[str]:
    """Draw one candidate per slot through the live transport, in slot order."""
    backend = backend_factory(model_id, None, min(BOARD_GENERATION_CONCURRENCY, len(slots)))
    completions = backend.generate_detailed([slot.prompt for slot in slots])
    return [_visible_reply(completion) for completion in completions]


def _batch_candidates(  # noqa: PLR0913 - the job's identity, plus the timeout and the tests' seam
    slots: Sequence[BoardSlot],
    *,
    model_id: str,
    stage: str,
    run_dir: Path,
    timeout_seconds: float,
    backend_factory: Callable[[str, str], BedrockBatchBackend] | None,
) -> list[list[str]]:
    """Draw every candidate for one stage in ONE padded batch job; return them per slot, in draw order.

    Padded to at least :data:`~reward_hacking.bedrock_batch.MIN_BATCH_RECORDS` because a job below the
    floor is refused outright, and padded further so the whole re-draw budget
    (:data:`~sociology.transfer_stimulus.BOARD_REDRAW_CAP`) is available inside the one job: a second job
    for a re-draw would be a second queue wait, and the pad costs a fraction of a cent.

    The handle is saved before the wait, so a killed collect resumes the job rather than paying for it
    twice -- the same rule every sampled leg follows.
    """
    copies = max(BOARD_REDRAW_CAP + 1, -(-MIN_BATCH_RECORDS // len(slots)))
    prompts = [slot.prompt for _copy in range(copies) for slot in slots]
    run_id = f"{FINGERPRINT_BATCH_PASS_TOKEN}-boards-{stage}"
    backend = (
        BedrockBatchBackend(
            model_id,
            sampling=BedrockSamplingConfig(max_tokens=MAX_TOKENS, reasoning_effort=None),
            prefix=f"{FINGERPRINT_BATCH_PREFIX}/{run_dir.name}",
            run_id=run_id,
        )
        if backend_factory is None
        else backend_factory(model_id, run_id)
    )
    path = run_dir / "handles" / f"boards-{stage}.json"
    refuse_tracked_trace_path(path, carries=HANDLE_CARRIES)
    if path.exists():
        handle = BatchJobHandle.load(path)
        logger.warning("resuming the saved board-generation job at %s: %s", path, handle.job_name)
    else:
        handle = backend.submit(prompts)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle.save(path)
    completions = backend.collect(handle, timeout_seconds=timeout_seconds)
    texts = [_visible_reply(completion) for completion in completions]
    return [
        [texts[copy * len(slots) + position] for copy in range(copies)]
        for position in range(len(slots))
    ]


def _round_from_collected_candidates(
    slots: Sequence[BoardSlot], candidates: Sequence[Sequence[str]]
) -> Callable[[Sequence[BoardSlot], int], list[str]]:
    """Serve one round from a batch job's already-collected copies: the attempt index picks the copy."""
    index_of = {slot.what: position for position, slot in enumerate(slots)}

    def draw_round(pending: Sequence[BoardSlot], attempt: int) -> list[str]:
        drawn = []
        for slot in pending:
            copies = candidates[index_of[slot.what]]
            drawn.append(copies[attempt] if attempt < len(copies) else "")
        return drawn

    return draw_round


def _round_from_live_calls(
    model_id: str, backend_factory: LiveBackendFactory
) -> Callable[[Sequence[BoardSlot], int], list[str]]:
    """Serve one round as a fresh live burst over the slots still pending."""

    def draw_round(pending: Sequence[BoardSlot], attempt: int) -> list[str]:
        del attempt  # a live round is a fresh draw; the round index only paces the re-draw cap
        return _live_round(pending, model_id=model_id, backend_factory=backend_factory)

    return draw_round


def _board_round_drawer(  # noqa: PLR0913 - the job's identity, the transport, and the tests' two seams
    slots: Sequence[BoardSlot],
    *,
    transport: str,
    model_id: str,
    stage: str,
    run_dir: Path,
    timeout_seconds: float,
    live_backend_factory: LiveBackendFactory,
    batch_backend_factory: Callable[[str, str], BedrockBatchBackend] | None,
) -> Callable[[Sequence[BoardSlot], int], list[str]]:
    """Build the round-drawing callable for one board's slots on whichever transport it runs.

    One callable for both transports, so the accept loop below has a single code path: on the batch
    transport every round is already collected from one padded job, and on the live transport a round is
    a burst of calls made when the round is asked for.
    """
    if transport == TRANSPORT_BATCH:
        return _round_from_collected_candidates(
            slots,
            _batch_candidates(
                slots,
                model_id=model_id,
                stage=stage,
                run_dir=run_dir,
                timeout_seconds=timeout_seconds,
                backend_factory=batch_backend_factory,
            ),
        )
    return _round_from_live_calls(model_id, live_backend_factory)


def _accept_messages(
    slots: Sequence[BoardSlot],
    *,
    draw_round: Callable[[Sequence[BoardSlot], int], list[str]],
    nouns: Sequence[str],
) -> list[tuple[str, int]]:
    """Draw until every slot has a message the board gate accepts; return each with its re-draw count.

    Rounds rather than per-slot loops, so one code path serves both transports: a live round is one
    concurrent burst over the slots still pending, and a batch round is the next copy of the padded job
    that is already collected. Past the cap it refuses by name rather than spending forever, which is the
    failure a model that names its own lab in every draw would otherwise be.
    """
    accepted: dict[int, tuple[str, int]] = {}
    pending = list(enumerate(slots))
    refusals: list[str] = []
    for attempt in range(BOARD_REDRAW_CAP + 1):
        if not pending:
            break
        texts = draw_round([slot for _position, slot in pending], attempt)
        still_pending: list[tuple[int, BoardSlot]] = []
        for (position, slot), text in zip(pending, texts, strict=True):
            problem = board_message_problem(text, nouns=nouns)
            if problem is None:
                accepted[position] = (text, attempt)
                continue
            refusals.append(f"{slot.what} draw {attempt}: {problem}")
            logger.warning("re-drawing %s: %s", slot.what, problem)
            still_pending.append((position, slot))
        pending = still_pending
    if pending:
        raise RuntimeError(
            f"{len(pending)} board messages still fail the board gate after {BOARD_REDRAW_CAP} re-draws "
            f"({[slot.what for _position, slot in pending]}). The last refusals were "
            f"{refusals[-len(pending) :]}. Either the task prompt invites the refused shape (a model "
            f"naming itself, a scenario noun) and wants rewording, or the drawing model cannot answer "
            f"this task; re-drawing further would spend without changing either."
        )
    return [accepted[position] for position in range(len(slots))]


def _board_provenance_entry(
    slot: BoardSlot, *, redraws: int, transport: str, code_sha: str
) -> dict[str, Any]:
    """Record where one accepted message came from, in the shape the loader checks field by field."""
    entry: dict[str, Any] = {
        "source_model_id": BOARD_MODEL_ID_BY_SIDE[BOARD_CONTENT_SIDE[slot.board_id]],
        "paraphraser_model_id": (
            None
            if slot.board_id in RAW_BOARD_IDS
            else BOARD_MODEL_ID_BY_SIDE[BOARD_WORDING_SIDE[slot.board_id]]
        ),
        "task_scenario_id": slot.scenario_id,
        "draw": slot.index,
        "transport": transport,
        "redraws": redraws,
        "generated_at": datetime.now(UTC).isoformat(),
        "code_sha": code_sha,
        "sampling": {"max_tokens": MAX_TOKENS},
    }
    if slot.source_message is not None:
        entry[BOARD_SOURCE_DIGEST_FIELD] = message_digest(slot.source_message)
    missing = [name for name in BOARD_PROVENANCE_FIELDS if name not in entry]
    if missing:
        raise RuntimeError(
            f"the generator wrote provenance without {missing}, which the loader requires; the two lists "
            f"have drifted apart and every board written now would refuse to load."
        )
    return entry


def _probe_the_batch_row_live(
    stimulus: TransferStimulus, *, calls: int, backend_factory: LiveBackendFactory
) -> dict[str, Any]:
    """Ask the batch roster row whether it answers a live Converse call at all, before relying on it.

    This repository has only ever called that row through batch inference, so whether it is served live
    on this account is unverified. Probing it with three calls costs a fraction of a cent and decides
    between a few minutes of live generation and two batch queue waits; assuming either way is how a
    generation run discovers the answer an hour in.
    """
    model_id = next(model_id for model_id in PAIR_MODEL_IDS if model_id not in LIVE_MODEL_IDS)
    prompts = [
        stimulus.unrelated_tasks[scenario.scenario_id].task for scenario in stimulus.scenarios
    ]
    backend = backend_factory(model_id, None, min(calls, BOARD_GENERATION_CONCURRENCY))
    completions = backend.generate_detailed(prompts[:calls])
    answered = [_visible_reply(completion) for completion in completions]
    return {
        "model_id": model_id,
        "calls": calls,
        "answered": sum(1 for text in answered if text),
        "live": all(bool(text) for text in answered),
    }


def board_consuming_run_roots() -> tuple[Path, ...]:
    """Where the two passes that read the boards keep their run directories.

    Both, because both stamp the same stimulus prompt digest: a regeneration is a new stimulus for the
    dose pass as much as for the fingerprint pass, whichever of them generated the boards.
    """
    return tuple(
        Path(PLAN_TABLES[pass_id].default_run_dir).parent
        for pass_id in (PASS_FINGERPRINT, PASS_CORRELATION_DOSE)
    )


def _refuse_boards_already_sampled(
    payload: Mapping[str, Any], *, regenerate: bool, run_roots: Sequence[Path], run_dir: Path
) -> list[Path]:
    """Refuse to overwrite boards that exist, or to regenerate once anything has been sampled on them.

    Two refusals, and the second is the one that matters. Regenerating after a leg has been sampled
    would leave one run directory holding replies to two different boards under one cell id: the prompt
    digest guard would catch a resume, but a fresh leg in the same directory would simply be pooled with
    the old one by every reader. Both passes' directories are checked, because both stamp the same
    stimulus prompt digest.

    The generation's OWN ``--run-dir`` and its parent are swept alongside the two default roots, because
    the default roots are where the spec puts the run directories and not where an operator is forced to
    put them: a leg sampled into a directory of one's own would otherwise be invisible here and the
    regeneration would go through. What no sweep can see is a leg sampled somewhere neither this
    invocation nor the design names, so the refusal is a check on the directories in play rather than a
    proof that nothing anywhere has been sampled.
    """
    filled = sorted(
        f"{scenario_id}/{board_id}"
        for scenario_id, boards in payload.get(BOARDS_FIELD, {}).items()
        for board_id, board in boards.items()
        if board.get("messages")
    )
    if filled and not regenerate:
        raise RuntimeError(
            f"{len(filled)} board slots already carry messages (for instance {filled[:3]}). The boards "
            f"are generated once and frozen, because both passes stamp the same stimulus prompt digest "
            f"and a second generation is a new stimulus version. Pass --regenerate if that is what you "
            f"mean, which is refused once anything has been sampled."
        )
    swept = {*run_roots, run_dir.parent}
    sampled = sorted(
        {
            *(path for root in swept for path in root.glob("*/replies--*.jsonl")),
            *run_dir.glob("replies--*.jsonl"),
        }
    )
    if regenerate and sampled:
        raise RuntimeError(
            f"--regenerate was given and these passes have already sampled replies: "
            f"{[str(path) for path in sampled]}. Regenerating the boards now would put replies to two "
            f"different stimuli under one cell id in a directory a reader pools. Bump the stimulus "
            f"version and open new run directories for both passes instead."
        )
    return sampled


def generate_boards(  # noqa: PLR0913 - one keyword per seam; the last three are the offline tests'
    run_dir: Path,
    *,
    regenerate: bool,
    stimulus_path: Path = STIMULUS_PATH,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    probe_calls: int = BOARD_LIVE_PROBE_CALLS,
    run_roots: Sequence[Path] | None = None,
    live_backend_factory: LiveBackendFactory | None = None,
    batch_backend_factory: Callable[[str, str], BedrockBatchBackend] | None = None,
) -> dict[str, Any]:
    """Draw the fingerprint boards with the roster models themselves, gate every message, and freeze them.

    Four boards per scenario: each side's own three draws of that scenario's task, then each side's
    rewrite of the other side's three. The raw boards come first because the crossed ones rewrite them,
    which is also the identity the loader checks afterwards (every crossed message names the digest of the
    message it rewrote).

    The live probe runs first, so the batch row's transport is a measurement rather than an assumption.
    Everything the generator writes is re-read through the full loader before the summary is written: a
    board table that would not load is a board table nobody can sample, and finding that out here costs
    nothing while finding it out at build time costs the whole generation.
    """
    refuse_tracked_trace_path(
        stimulus_path, carries="the authored transfer stimulus, boards included"
    )
    payload = json.loads(stimulus_path.read_text(encoding="utf-8"))
    already_sampled = _refuse_boards_already_sampled(
        payload,
        regenerate=regenerate,
        run_roots=board_consuming_run_roots() if run_roots is None else run_roots,
        run_dir=run_dir,
    )
    stimulus = load_stimulus(stimulus_path, allow_empty_boards=True)
    nouns = scenario_nouns(stimulus.scenarios)
    factory = (
        live_backend_factory
        if live_backend_factory is not None
        else make_live_backend_factory(MAX_TOKENS)
    )
    probe = _probe_the_batch_row_live(stimulus, calls=probe_calls, backend_factory=factory)
    batch_model_ids = set() if probe["live"] else {str(probe["model_id"])}
    code_sha = git_sha()
    drawn: dict[tuple[str, str], list[str]] = {}
    provenance: dict[tuple[str, str], list[dict[str, Any]]] = {}
    redraws = 0
    for stage, board_ids in (
        (BOARD_STAGE_RAW, RAW_BOARD_IDS),
        (BOARD_STAGE_PARAPHRASE, tuple(BOARD_SOURCE_BOARD)),
    ):
        for board_id in board_ids:
            slots = (
                _raw_board_slots(stimulus, board_id)
                if stage == BOARD_STAGE_RAW
                else _paraphrase_board_slots(stimulus, board_id, drawn)
            )
            model_id = slots[0].drawing_model_id
            transport = TRANSPORT_BATCH if model_id in batch_model_ids else TRANSPORT_LIVE
            draw_round = _board_round_drawer(
                slots,
                transport=transport,
                model_id=model_id,
                stage=f"{stage}-{board_id}",
                run_dir=run_dir,
                timeout_seconds=timeout_seconds,
                live_backend_factory=factory,
                batch_backend_factory=batch_backend_factory,
            )
            accepted = _accept_messages(slots, draw_round=draw_round, nouns=nouns)
            for slot, (text, slot_redraws) in zip(slots, accepted, strict=True):
                key = (slot.scenario_id, slot.board_id)
                drawn.setdefault(key, []).append(text)
                provenance.setdefault(key, []).append(
                    _board_provenance_entry(
                        slot, redraws=slot_redraws, transport=transport, code_sha=code_sha
                    )
                )
                redraws += slot_redraws
    payload[BOARDS_FIELD] = {
        scenario.scenario_id: {
            board_id: {
                "messages": drawn[scenario.scenario_id, board_id],
                "provenance": provenance[scenario.scenario_id, board_id],
            }
            for board_id in BOARD_IDS
        }
        for scenario in stimulus.scenarios
    }
    stimulus_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    frozen = load_stimulus(stimulus_path)
    summary = {
        "command": "generate-boards",
        "stimulus_path": str(stimulus_path),
        "stimulus_version": STIMULUS_VERSION,
        "regenerated": regenerate,
        "already_sampled_replies": [str(path) for path in already_sampled],
        "live_probe": probe,
        "boards": len(BOARD_IDS) * len(stimulus.scenarios),
        "messages": sum(len(messages) for messages in drawn.values()),
        "redraws": redraws,
        "sampling": dict.fromkeys(BOARD_SAMPLING_KEYS, MAX_TOKENS),
        "code_sha": code_sha,
        "stimulus_digest": frozen.digest,
        "stimulus_prompt_digest": frozen.prompt_digest,
        "payload_prompt_digest": prompt_digest_of(payload),
    }
    write_summary(run_dir, "generate-boards", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _scripted_reply(prompt: str) -> str:
    """Answer a transfer prompt offline in whichever tag the prompt itself asked for.

    Reading the tag off the prompt rather than emitting a fixed one, so the offline smoke exercises the
    real extraction path -- ``strip_thinking``, then the polarity's own tag, then the inversion -- which
    is where a polarity mapping bug would surface.
    """
    match = _SET_TAG_RE.search(prompt)
    if match is None:
        raise ValueError("a rendered transfer prompt must ask for a <set> or a <keep> tag")
    tag = match.group(1)
    return f"A scripted reply for the offline smoke. <{tag}>4</{tag}>"


def _scripted_judge_verdict(prompt: str) -> str:
    """Return a fixed off-the-shelf verdict for the offline smoke, shaped for whichever arm asked.

    ``none`` and ``not_discussed`` are the values that parse against any row, which is what lets one
    scripted answer exercise the whole judge path. It follows that the offline smoke's validation report
    shows misses against the registered expectations: that is the miss-reporting path working, not a
    failing judge, and the live smoke is where real agreement is read.

    The twin arm's extra fields are included unconditionally, because the one-way schema ignores fields it
    does not declare while the twin schema refuses a verdict missing one of its own -- so the union is the
    one answer that parses on every arm. Derived from the schemas rather than listed, so an added field
    does not silently turn every scripted judge call into an errored row.
    """
    del prompt
    twin_fields = judge_module.schema_for(MATCHED_DECISION_TRANSFER_GAME_ID).verdict_fields
    verdict: dict[str, Any] = {
        "action_units": judge_module.ACTION_NONE,
        "counterpart_reasoning": "not_discussed",
        "evidence": "",
    }
    verdict.update({field_name: False for field_name in twin_fields if field_name not in verdict})
    return json.dumps(verdict)


def smoke_leg(transport: str, table: PlanTable, block: str | None = None) -> Leg:
    """Build the leg the smoke renders from: one block of the pass's smoke model, on its transport.

    ``block`` defaults to the table's own smoke block. It is selectable because a pass whose readings span
    two blocks wants both smoked, and the reply file's name carries the block, so smoking each in turn
    accumulates in one smoke directory rather than resuming the other's records.
    """
    chosen = table.smoke_block if block is None else block
    return Leg(
        table.smoke_live_model_id,
        transport,
        chosen,
        table.cells_by_block[chosen],
        None,
        SITTING_A,
        batch_prefix=table.batch_prefix,
        batch_pass_token=table.batch_pass_token,
    )


def smoke_calls(
    stimulus: TransferStimulus,
    table: PlanTable,
    *,
    transport: str,
    rows_per_cell: int,
    block: str | None = None,
) -> list[PlannedCall]:
    """Take the first few draw-zero calls of each of one block's cells, on the smoke's transport.

    Rendered from the real leg and then narrowed, rather than from a smoke-shaped leg of its own, so the
    smoke exercises exactly the prompts and keys production will send.
    """
    by_cell: dict[str, list[PlannedCall]] = {}
    for call in planned_calls_for_leg(smoke_leg(transport, table, block), stimulus):
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
    they do: every judged row records whether the reply it was given carried a thinking block in the
    answer channel, so recomputing that from the reply on disk and comparing is the only place a judge
    quietly handed the raw reply would show up. Returns how many judged rows carried one.
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
            f"was emitted in the answer channel (first: {mismatched[:3]}). The judge and the "
            f"deterministic scan must read the same visible text, or their disagreement rate -- one of "
            f"this pass's reported instrument findings -- is an artifact of two different inputs."
        )
    return sum(1 for row in rows.values() if row.get("had_inline_think"))


def smoke(
    run_dir: Path, table: PlanTable, *, backend: str, block: str | None = None
) -> dict[str, Any]:
    """Run the end-to-end smoke: render, sample, scan, judge-validate, judge, all under ``smoke/``.

    ``scripted`` spends nothing and is the gate before any real submit: a handful of rows through the live
    resume loop on an offline backend, then the deterministic scans and both judge passes on scripted
    verdicts. ``live`` is the same path against real endpoints on the cheap reasoning-bearing model.
    Everything lands under ``<run-dir>/smoke/<backend>/`` so the production replies never carry smoke
    records and the two backends never resume each other's.
    """
    stimulus = load_stimulus()
    # One subtree per backend, because the two smokes render the SAME keys: a live smoke sharing the
    # scripted one's reply file would resume its offline records and only call for the remainder.
    smoke_dir = run_dir / "smoke" / backend
    scripted = backend == "scripted"
    transport = ScriptedDetailedBackend.transport if scripted else TRANSPORT_LIVE
    calls = smoke_calls(
        stimulus, table, transport=transport, rows_per_cell=SMOKE_LIVE_ROWS_PER_CELL, block=block
    )
    if scripted:
        calls = calls[:SMOKE_SCRIPTED_CALLS]
    out_path = reply_path(smoke_dir, smoke_leg(transport, table, block))
    audit_planned_calls(calls, stimulus)

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
        ScriptedDetailedBackend(_scripted_judge_verdict)
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
    judge_counts = judge_module.judge_run(
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
        "pass_id": table.pass_id,
        "backend": backend,
        "block": smoke_leg(transport, table, block).block,
        "model_id": table.smoke_live_model_id,
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
            "which pass's plan this command operates on. Required and never defaulted: the tables share "
            "this substrate, and a default would let a command re-render, re-price or re-submit the "
            "finished pass because nobody typed the flag"
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="where this run's artifacts live; defaults to the selected pass's own run directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser(
        "build", help="render every leg, audit every prompt, write plan.json"
    )
    build_parser.add_argument(
        "--allow-empty-boards",
        action="store_true",
        help=(
            "build against a stimulus file whose fingerprint boards have not been generated yet, which "
            "is the window between the author writing the v3 fields and generate-boards filling them. "
            "Not a sampling escape: a board cell still refuses to render, so pass D cannot be planned "
            "or priced without its boards"
        ),
    )
    boards = sub.add_parser(
        "generate-boards",
        help="draw the fingerprint boards with the roster models themselves, gate them, and freeze them",
    )
    boards.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "overwrite boards that already carry messages; refused once either pass has sampled a reply, "
            "because a second generation is a new stimulus version and a new run directory"
        ),
    )
    boards.add_argument("--timeout-seconds", type=float, default=BATCH_TIMEOUT_SECONDS)
    boards.add_argument(
        "--probe-calls",
        type=_at_least_one,
        default=BOARD_LIVE_PROBE_CALLS,
        help="live calls the batch row's transport probe spends before the boards are drawn",
    )
    sub.add_parser("dry-run", help="count and price every leg; no AWS calls")
    smoke_parser = sub.add_parser("smoke", help="end-to-end smoke, offline or on the live path")
    smoke_parser.add_argument("--backend", choices=["scripted", "live"], default="scripted")
    smoke_parser.add_argument(
        "--block",
        default=None,
        help=(
            "smoke this block's cells rather than the pass's default smoke block; run it once per block "
            "when a pass reads two, since each block's replies land in a file of their own"
        ),
    )
    # Every pass's legs, because argparse builds its choices before --pass is parsed; the selected table's
    # own `leg_for` is what refuses a leg belonging to a different pass.
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
    sub.add_parser("judge-validate", help="judge both arms' validation replies")
    intent = sub.add_parser(
        "intent-check", help="read what each reply's reasoning concluded, against the tag it wrote"
    )
    intent.add_argument(
        "--scope",
        choices=INTENT_SCOPES,
        default=SCOPE_ENDPOINT,
        help=(
            "endpoint reads every scanned reply whose parsed figure is nothing or the whole stock, which "
            "is where a slip or a misread moves the score; all reads the interior too (recorded as scope)"
        ),
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
    cross.add_argument(
        "--game",
        choices=TRANSFER_GAME_IDS,
        default=None,
        help=(
            "draw the subset from this arm's replies only, so a per-model read of a twin-only field "
            "gets a full allotment; recorded as game_filter in the summary (null when absent)"
        ),
    )
    for gated in (judge_parser, intent, cross):
        gated.add_argument(
            "--skip-validation-gate",
            action="store_true",
            help=(
                "judge with rubrics this run has not validated; warns loudly and records "
                "skipped_validation_gate=true in the summary"
            ),
        )
        gated.add_argument(
            "--concurrency",
            type=_at_least_one,
            default=DEFAULT_JUDGE_CONCURRENCY,
            help=(
                "judge calls in flight at once; recorded as judge_concurrency in the summary (the "
                "hatch judge ran Luna at 32-48 without throttling)"
            ),
        )
    opus = sub.add_parser("add-opus", help="authorize the on-demand legs after the refusal probe")
    opus.add_argument("--model", choices=sorted(ON_DEMAND_MODEL_CHOICES), required=True)
    return parser.parse_args(argv)


_DISPATCH_WITHOUT_ARGS = {
    "scan": scan,
    "judge-validate": judge_validate,
    "intent-check-validate": intent_check_validate,
}
"""Subcommands whose only argument is the run directory: they read what is already on disk.

They still take ``--pass``, because it is what resolves the run directory they read; they just have no
use for the leg table once it has.
"""

_DISPATCH_WITH_TABLE = {"dry-run": dry_run}
"""Subcommands that walk the selected pass's whole leg table and take no other argument."""


def _dispatch_plan_writing(args: argparse.Namespace, run_dir: Path, table: PlanTable) -> bool:
    """Run the subcommands that write a plan, a stimulus or a smoke; report whether it was one of them."""
    if args.command == "build":
        build(run_dir, table, allow_empty_boards=args.allow_empty_boards)
        return True
    if args.command == "generate-boards":
        if table.pass_id != PASS_FINGERPRINT:
            raise SystemExit(
                f"generate-boards belongs to the {PASS_FINGERPRINT!r} pass, which owns the boards; "
                f"--pass {table.pass_id!r} was given. The dose pass reads the same stimulus file and "
                f"generates nothing in it."
            )
        generate_boards(
            run_dir,
            regenerate=args.regenerate,
            timeout_seconds=args.timeout_seconds,
            probe_calls=args.probe_calls,
        )
        return True
    if args.command == "smoke":
        smoke(run_dir, table, backend=args.backend, block=args.block)
        return True
    return False


def _dispatch_sampling(args: argparse.Namespace, run_dir: Path, table: PlanTable) -> bool:
    """Run the subcommands that operate on ONE leg; report whether the command was one of them."""
    if args.command == "submit-batch":
        submit_batch(
            args.leg, run_dir, table, check_only=args.check_only, canary_calls=args.canary_calls
        )
        return True
    if args.command == "collect-batch":
        collect_batch(
            args.leg,
            run_dir,
            table,
            timeout_seconds=args.timeout_seconds,
            stimulus_unchanged_since_submit=args.stimulus_unchanged_since_submit,
        )
        return True
    if args.command == "run-live":
        run_live(args.leg, run_dir, table, concurrency=args.concurrency, chunk_size=args.chunk_size)
        return True
    return False


def _dispatch_judging(args: argparse.Namespace, run_dir: Path) -> bool:
    """Run the three model-reading instruments over the replies on disk; report whether it was one."""
    if args.command == "judge":
        judge(
            run_dir,
            limit=args.limit,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
        return True
    if args.command == "intent-check":
        intent_check(
            run_dir,
            scope=args.scope,
            model=args.model,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
        return True
    if args.command == "cross-judge":
        cross_judge(
            run_dir,
            n=args.n,
            game=args.game,
            skip_validation_gate=args.skip_validation_gate,
            concurrency=args.concurrency,
        )
        return True
    return False


def _dispatch(args: argparse.Namespace, run_dir: Path, table: PlanTable) -> None:
    """Run whichever subcommand was asked for, in groups rather than in one chain.

    Grouped because the chain outgrew what a reader can hold: the groups are the run's own stages -- read
    what is on disk, write a plan or a stimulus, sample one leg, read the replies with a model.
    """
    simple = _DISPATCH_WITHOUT_ARGS.get(args.command)
    if simple is not None:
        simple(run_dir)
        return
    with_table = _DISPATCH_WITH_TABLE.get(args.command)
    if with_table is not None:
        with_table(run_dir, table)
        return
    if _dispatch_plan_writing(args, run_dir, table):
        return
    if _dispatch_sampling(args, run_dir, table):
        return
    if _dispatch_judging(args, run_dir):
        return
    add_opus(run_dir, table, args.model)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one transfer subcommand, against the pass ``--pass`` selects."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    table = plan_table(args.pass_id)
    run_dir: Path = Path(table.default_run_dir) if args.run_dir is None else args.run_dir
    _dispatch(args, run_dir, table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
