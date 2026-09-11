"""Stage-one runners: plan the legs, price them dry, submit/collect batch, run live, score.

The sampling plan is a fixed table of legs (:data:`PRODUCTION_LEGS`), one row per
(model, cell, bundle subset, draws, effort), so what runs is read off a table rather than
assembled per invocation. Both transports reuse ``reward_hacking``'s verified paths: batch legs go
through ``BedrockBatchBackend`` under a NEW S3 prefix with one job per model pooling all of that
model's cells (the 100-record job floor is per job, and pooling is what clears it), the handle
persisted at submit and both digests compared caller-side at collect, copying the jagged sweep;
live legs go through ``BedrockBackend`` with incomplete stop reasons flagged and kept in every
denominator.

Replies are written incrementally to one JSONL keyed by (cell, bundle, model, effort, draw); a
resumed invocation skips keys already on disk and counts them separately from anything else, and
the summary JSON is written last as the completion marker (trace presence is not completion). That
reply I/O, the resume loop and the summary marker live in :mod:`sociology.records`, shared with the
decoupled-ladder pass; what stays here is this study's own row schema and leg table. The
``--dry-run``-shaped ``dry-run`` subcommand prices and counts every production call without
touching AWS; batch prices come off the verified roster rows, live prices are labelled UNVERIFIED
because no probe has read them, and the input-context check is recorded per model with its
assumptions stated.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from reward_hacking.bedrock_batch import (
    BatchJobHandle,
    BedrockBatchBackend,
    cell_digest,
    prompt_digest,
    roster_model,
)
from reward_hacking.model_backend import (
    BedrockBackend,
    BedrockSamplingConfig,
    is_incomplete_stop_reason,
)
from reward_hacking.trace import refuse_tracked_trace_path
from sociology import judge as judge_module
from sociology.bundles import (
    CELLS_BY_NAME,
    MAX_BUNDLE_RENDER_CHARS,
    BundleSpec,
    CellSpec,
    assert_manifest_current,
    build_manifest,
    build_pool,
    bundles_of,
    first_bundles_per_family,
    load_manifest,
    render_bundle_text,
    render_prompt,
    write_manifest,
)
from sociology.corpus import (
    FAMILY_AGENTIC_120B,
    contains_colocation_cue,
    load_agentic_units,
    load_single_turn_units,
)
from sociology.records import (
    append_replies,
    completion_telemetry,
    load_replies,
    make_live_backend_factory,
    record_prompt_digest,
    write_summary,
)
from sociology.records import run_live_calls as run_resume_loop
from sociology.refusal_canary import (
    REFUSAL_CANARY_CALLS,
    canary_plan,
    needs_refusal_canary,
    run_refusal_canary,
)
from sociology.scans import scan_reply
from sociology.stimulus import Stimulus, load_stimulus

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend
    from sociology.corpus import BundleUnit

    # (model_id, reasoning_effort, concurrency) -> backend; the live runner's one testable seam.
    LiveBackendFactory = Callable[[str, str | None, int], DetailedBackend]

logger = logging.getLogger(__name__)

DEFAULT_RUN_DIR = Path("artifacts/swarm_sociology/analysis_model/stage1-20260831")

SOCIOLOGY_BATCH_PREFIX = "batch_jobs/sociology_stage1"
"""A prefix of this stage's own; parallel runs must never share a flat prefix with another sweep."""

ANALYSIS_MAX_TOKENS = 16_384
"""Reply cap for every analysis call: generous for a written analysis, under every roster ceiling.

Deliberately not routed through the RecoveryBench measured-budget table, which refuses unmeasured
models by design; this is a fresh elicitation with its own cap, stated here once. For Claude Opus 5
adaptive thinking is billed inside this same budget, so a truncated (``max_tokens``) reply stays
possible, is flagged, and stays in every denominator.
"""

OPUS_MODEL_ID = "global.anthropic.claude-opus-5"
HAIKU_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
SOL_MODEL_ID = "global.openai.gpt-5.6-sol"
"""The inference-profile id, not the bare model id: ConverseStream refuses the bare
``openai.gpt-5.6-sol`` with "on-demand throughput isn't supported" (hit live, 2026-08-31), while
both geo profiles resolve ACTIVE. ``global.`` matches Luna's routing so the two GPT-5.6 arms share
one prefix. The bare id remains correct in the batch module's not-batch-capable record, a
different question about a different transport."""
LUNA_MODEL_ID = "global.openai.gpt-5.6-luna"
SMOKE_MODEL_ID = "openai.gpt-oss-120b-1:0"

TRANSPORT_BATCH = "batch"
TRANSPORT_LIVE = "live"

CHARS_PER_TOKEN_ESTIMATE = 3.5
"""Conservative chars-per-token for sizing and pricing; code-heavy text tokenises worse than 4."""

ASSUMED_OUTPUT_TOKENS_PER_CALL = 2_500
"""Dry-run output assumption per analysis reply, stated rather than hidden in a total."""

FRONTIER_LIVE_PRICE_UNVERIFIED = (1.25, 10.0)
"""(input, output) $/Mtok used for the GPT-5.6 live legs in the dry run. UNVERIFIED: no Price List
read backs these figures; every dry-run line derived from them says so."""

CONTEXT_WINDOWS: dict[str, tuple[int, str]] = {
    OPUS_MODEL_ID: (200_000, "assumed: 200k-class per the design doc; not probed"),
    HAIKU_MODEL_ID: (200_000, "assumed: 200k-class; not probed"),
    SOL_MODEL_ID: (400_000, "assumed: GPT-5.6 family figure; not probed"),
    LUNA_MODEL_ID: (400_000, "assumed: GPT-5.6 family figure; not probed"),
    SMOKE_MODEL_ID: (131_072, "assumed: GPT-OSS 128k; not probed"),
}
"""Input-context ceilings for the pre-spend size check, each carrying its own epistemic label."""


@dataclass(frozen=True, slots=True)
class Leg:
    """One row of the sampling plan: a model reading one cell at one effort, so many times."""

    model_id: str
    transport: str
    cell: str
    bundles_per_family: int | None  # None reads every bundle of the cell
    draws: int
    reasoning_effort: str | None


PRODUCTION_LEGS: tuple[Leg, ...] = (
    # Opus 5, batch, every cell: no temperature/top_p (deprecated); variance is resampling.
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "center", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "framing-independent", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "framing-unstated", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "cues-stripped", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "size-4", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "size-16", None, 3, None),
    Leg(OPUS_MODEL_ID, TRANSPORT_BATCH, "single-turn", None, 3, None),
    # Claude Haiku 4.5, batch: the does-it-track-model rider on the framing rungs.
    Leg(HAIKU_MODEL_ID, TRANSPORT_BATCH, "center", None, 3, None),
    Leg(HAIKU_MODEL_ID, TRANSPORT_BATCH, "framing-independent", None, 3, None),
    Leg(HAIKU_MODEL_ID, TRANSPORT_BATCH, "framing-unstated", None, 3, None),
    # GPT-5.6 Sol, live only (not batch-capable): default effort on purpose, recorded not chosen.
    Leg(SOL_MODEL_ID, TRANSPORT_LIVE, "center", None, 1, None),
    Leg(SOL_MODEL_ID, TRANSPORT_LIVE, "framing-independent", 5, 1, None),
    Leg(SOL_MODEL_ID, TRANSPORT_LIVE, "framing-unstated", 5, 1, None),
    # GPT-5.6 Luna, live: the deliberation factor, effort medium vs high on the center bundles.
    Leg(LUNA_MODEL_ID, TRANSPORT_LIVE, "center", None, 1, "medium"),
    Leg(LUNA_MODEL_ID, TRANSPORT_LIVE, "center", None, 1, "high"),
)

MODEL_SHORT_NAMES: dict[str, str] = {
    OPUS_MODEL_ID: "opus",
    HAIKU_MODEL_ID: "haiku",
    SOL_MODEL_ID: "sol",
    LUNA_MODEL_ID: "luna",
}


def record_key(cell: str, bundle_id: str, model_id: str, effort: str | None, draw: int) -> str:
    """One planned call's stable identity, derived from content, never from execution order."""
    return f"{cell}|{bundle_id}|{model_id}|effort={effort or 'default'}|draw={draw}"


@dataclass(frozen=True, slots=True)
class PlannedCall:
    """One fully rendered analysis call, ready for either transport."""

    key: str
    cell: str
    bundle_id: str
    family: str
    framing: str
    cues: str
    size: int
    model_id: str
    transport: str
    draw: int
    reasoning_effort: str | None
    prompt: str

    def metadata(self) -> dict[str, object]:
        """Return the sidecar labels the batch path hashes into the cell digest."""
        return {
            "key": self.key,
            "cell": self.cell,
            "bundle_id": self.bundle_id,
            "family": self.family,
            "framing": self.framing,
            "cues": self.cues,
            "size": self.size,
            "draw": self.draw,
            "reasoning_effort": self.reasoning_effort,
        }


def load_units(
    agentic_dir: Path | None = None, single_turn: Path | None = None
) -> dict[str, list[BundleUnit]]:
    """Load both corpora into one family map (the manifest's whole substrate)."""
    units = load_agentic_units(agentic_dir) if agentic_dir else load_agentic_units()
    units.update(load_single_turn_units(single_turn) if single_turn else load_single_turn_units())
    return units


def _bundle_text(
    spec: BundleSpec, cell: CellSpec, units_by_family: Mapping[str, Sequence[BundleUnit]]
) -> str:
    """Re-render one bundle from the manifest spec, refusing silently drifted content."""
    by_id = {unit.unit_id: unit for unit in units_by_family[spec.family]}
    missing = [unit_id for unit_id in spec.members if unit_id not in by_id]
    if missing:
        raise RuntimeError(
            f"bundle {spec.bundle_id}: {len(missing)} member(s) absent from the corpus: {missing}"
        )
    text = render_bundle_text(
        [by_id[unit_id] for unit_id in spec.members],
        cues=cell.cues,
        context=f"{cell.name}/{spec.bundle_id}",
    )
    if len(text) != spec.rendered_chars:
        raise RuntimeError(
            f"bundle {spec.bundle_id} renders to {len(text)} chars but the manifest recorded "
            f"{spec.rendered_chars}: the corpus or the renderer moved since the manifest was "
            "written, so replies would be attached to text the model never read"
        )
    return text


def planned_calls_for_model(
    model_id: str,
    manifest: Mapping[str, Any],
    units_by_family: Mapping[str, Sequence[BundleUnit]],
    stimulus: Stimulus,
) -> list[PlannedCall]:
    """Render every planned call for one model, in the fixed leg/bundle/draw order.

    The order is load-bearing on the batch path: submit and collect both call this, and the prompt
    and cell digests compare positionally.
    """
    calls: list[PlannedCall] = []
    for leg in PRODUCTION_LEGS:
        if leg.model_id != model_id:
            continue
        cell = CELLS_BY_NAME[leg.cell]
        bundles = bundles_of(manifest, leg.cell)
        if leg.bundles_per_family is not None:
            bundles = first_bundles_per_family(bundles, leg.bundles_per_family)
        for spec in bundles:
            text = _bundle_text(spec, cell, units_by_family)
            prompt = render_prompt(stimulus, cell, text, n=cell.size)
            calls.extend(
                PlannedCall(
                    key=record_key(cell.name, spec.bundle_id, model_id, leg.reasoning_effort, draw),
                    cell=cell.name,
                    bundle_id=spec.bundle_id,
                    family=spec.family,
                    framing=cell.framing,
                    cues=cell.cues,
                    size=cell.size,
                    model_id=model_id,
                    transport=leg.transport,
                    draw=draw,
                    reasoning_effort=leg.reasoning_effort,
                    prompt=prompt,
                )
                for draw in range(leg.draws)
            )
    if not calls:
        raise ValueError(f"no production legs are defined for model {model_id!r}")
    return calls


REPLIES_FILENAME = "replies.jsonl"


def replies_path(run_dir: Path) -> Path:
    """Return the one incremental replies file every leg of a run appends to."""
    return run_dir / REPLIES_FILENAME


def _reply_record(
    call: PlannedCall, completion: BedrockCompletion, stimulus: Stimulus
) -> dict[str, Any]:
    """Build one reply record from a completion, carrying labels, usage, and completeness."""
    return {
        "key": call.key,
        "cell": call.cell,
        "bundle_id": call.bundle_id,
        "family": call.family,
        "framing": call.framing,
        "cues": call.cues,
        "size": call.size,
        "model_id": call.model_id,
        "transport": call.transport,
        "draw": call.draw,
        "reasoning_effort": call.reasoning_effort,
        "max_tokens": ANALYSIS_MAX_TOKENS,
        "stimulus_digest": stimulus.digest,
        "prompt_chars": len(call.prompt),
        "prompt_digest": record_prompt_digest(call.prompt),
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


REPLY_CARRIES = "verbatim episode transcripts inside model prompts"
"""What this study's reply file publishes if committed: the bundles ride inside every prompt."""


def assert_context_fits(model_id: str, calls: Sequence[PlannedCall]) -> dict[str, Any]:
    """Check the largest prompt plus the reply cap against the model's context ceiling.

    The design's pre-spend size check, recorded rather than merely passed: the returned dict lands
    in the dry-run output and the submit summary with the ceiling's own epistemic label, because
    every window figure here is assumed rather than probed.
    """
    window, provenance = CONTEXT_WINDOWS[model_id]
    largest = max(len(call.prompt) for call in calls)
    estimated = int(largest / CHARS_PER_TOKEN_ESTIMATE) + ANALYSIS_MAX_TOKENS
    check = {
        "model_id": model_id,
        "context_window": window,
        "window_provenance": provenance,
        "largest_prompt_chars": largest,
        "chars_per_token_estimate": CHARS_PER_TOKEN_ESTIMATE,
        "estimated_tokens_with_reply": estimated,
        "fits": estimated <= window,
    }
    if not check["fits"]:
        raise RuntimeError(
            f"{model_id}: the largest planned prompt (~{estimated} tokens with the reply cap) "
            f"exceeds the {window}-token window ({provenance}); drop the oversized cells for this "
            "model rather than submitting a job that dies per record"
        )
    return check


def run_live_calls(  # noqa: PLR0913 - trailing keyword-only knobs with defaults
    calls: Sequence[PlannedCall],
    out_path: Path,
    stimulus: Stimulus,
    *,
    concurrency: int,
    chunk_size: int,
    backend_factory: LiveBackendFactory | None = None,
) -> dict[str, int]:
    """Run this study's planned live calls through the shared resume loop, with its row schema.

    Everything about resume, chunked appends and the counts lives in :mod:`sociology.records`; what
    this study contributes is the reply record (:func:`_reply_record`, which stamps the stimulus
    digest) and the reply cap the backend samples at. ``backend_factory`` exists so the offline
    tests can script the transport; production callers leave it unset.
    """
    return run_resume_loop(
        calls,
        out_path,
        concurrency=concurrency,
        chunk_size=chunk_size,
        make_record=lambda call, completion: _reply_record(call, completion, stimulus),
        backend_factory=(
            backend_factory
            if backend_factory is not None
            else make_live_backend_factory(ANALYSIS_MAX_TOKENS)
        ),
        carries=REPLY_CARRIES,
    )


def _handle_path(run_dir: Path, model_id: str) -> Path:
    slug = model_id.replace(":", "-").replace(".", "-").replace("/", "-")
    return run_dir / "handles" / f"{slug}.json"


def _canary_cell(call: PlannedCall) -> str:
    """Name the group a stage-1 canary call stands for: its cell.

    The cell rather than its framing, although five of the seven cells share ``framing="population"``:
    those five differ in rendering (cues stripped), bundle size and corpus (single-turn), and whether
    the classifier keys on the framing sentence or on something else in the text is exactly what the
    canary is there to find out, not to assume. Grouping by framing sent every "population" call from
    ``center`` and never probed the other four.
    """
    return call.cell


def _refusal_canary(
    model_id: str,
    calls: Sequence[PlannedCall],
    run_dir: Path,
    *,
    calls_per_group: int,
    backend_factory: LiveBackendFactory | None,
) -> dict[str, Any] | None:
    """Run the live refusal canary for a Claude job, persist its counts, and refuse if it tripped.

    ``None`` for every other vendor, whose classifiers have never refused a job wholesale. The counts
    are written under their own summary label BEFORE the refusal decision, so a refused submit leaves
    its evidence in the run dir; the canary's replies are never written anywhere. ``cell_framings``
    rides along as a label so a tripped cell can be read against its framing without opening the
    cell table.
    """
    if not needs_refusal_canary(model_id):
        return None
    result = run_refusal_canary(
        calls,
        group_of=_canary_cell,
        backend_factory=backend_factory or make_live_backend_factory(ANALYSIS_MAX_TOKENS),
        calls_per_group=calls_per_group,
    )
    summary = {
        **result.summary,
        "cell_framings": {call.cell: call.framing for call in calls},
    }
    short = MODEL_SHORT_NAMES.get(model_id, "model")
    write_summary(
        run_dir,
        f"refusal-canary-{short}",
        {"command": "submit-batch", "model_id": model_id, **summary},
    )
    result.refuse_if_tripped(job=f"the {short} batch job", records=len(calls))
    return summary


def submit_batch(
    model_id: str,
    run_dir: Path,
    *,
    dry_run: bool,
    canary_calls: int = REFUSAL_CANARY_CALLS,
    canary_backend_factory: LiveBackendFactory | None = None,
) -> dict[str, Any]:
    """Submit one model's whole plan as ONE batch job, persisting the handle at submit.

    Pooling every cell of the model into one job is what clears the 100-record-per-job floor; the
    sidecar metadata carries the cell labels and the handle carries both digests for the collect
    tripwire. ``dry_run`` stops after the size and context checks, spending nothing.

    A Claude model's job is preceded by the live refusal canary (:mod:`sociology.refusal_canary`):
    ``canary_calls`` distinct prompts of every cell go through the live tier under the batch's own
    sampler, and a cell that answers none of them refuses the submit before the job exists. Its
    counts land in the submit summary and in ``summary-refusal-canary-<model>.json``; its replies are
    never written anywhere. ``canary_backend_factory`` is the offline tests' seam.
    """
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    calls = planned_calls_for_model(model_id, manifest, units, stimulus)
    context_check = assert_context_fits(model_id, calls)
    handle_path = _handle_path(run_dir, model_id)
    if handle_path.exists():
        raise RuntimeError(
            f"{handle_path} already exists: this model's job is already submitted and paid for. "
            "Collect it instead; a second submit is a second bill"
        )
    if dry_run:
        return {
            "model_id": model_id,
            "records": len(calls),
            "context_check": context_check,
            "refusal_canary_plan": (
                canary_plan(calls, group_of=_canary_cell, calls_per_group=canary_calls)
                if needs_refusal_canary(model_id)
                else None
            ),
        }
    backend = BedrockBatchBackend(
        model_id,
        sampling=BedrockSamplingConfig(max_tokens=ANALYSIS_MAX_TOKENS),
        prefix=SOCIOLOGY_BATCH_PREFIX,
        run_id=run_dir.name,
    )
    canary = _refusal_canary(
        model_id,
        calls,
        run_dir,
        calls_per_group=canary_calls,
        backend_factory=canary_backend_factory,
    )
    handle = backend.submit(
        [call.prompt for call in calls], metadata=[call.metadata() for call in calls]
    )
    handle.save(handle_path)
    summary = {
        "command": "submit-batch",
        "model_id": model_id,
        "records": len(calls),
        "job_arn": handle.job_arn,
        "context_check": context_check,
        "stimulus_digest": stimulus.digest,
        "refusal_canary": canary,
    }
    write_summary(run_dir, f"submit-{MODEL_SHORT_NAMES.get(model_id, 'model')}", summary)
    return summary


def collect_batch(model_id: str, run_dir: Path, *, timeout_seconds: float) -> dict[str, Any]:
    """Collect one model's job, verifying both digests caller-side before any record is written.

    The prompt digest proves the returned records answer the submitted prompts in order; the cell
    digest proves the labels this process would attach are the labels the submit recorded. Either
    mismatch refuses the whole collect (the jagged sweep's tripwire, copied).
    """
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    calls = planned_calls_for_model(model_id, manifest, units, stimulus)
    handle = BatchJobHandle.load(_handle_path(run_dir, model_id))
    local_prompts = prompt_digest([call.prompt for call in calls])
    local_cells = cell_digest([call.metadata() for call in calls])
    for label, submitted, local in (
        ("prompt", handle.prompt_digest, local_prompts),
        ("cell", handle.cell_digest, local_cells),
    ):
        if submitted != local:
            raise RuntimeError(
                f"handle {handle.job_name} was submitted with {label} digest {submitted} but this "
                f"process renders {local}: the corpus, manifest, stimulus, or plan moved since "
                "submit, so the join would attach replies to the wrong cells"
            )
    backend = BedrockBatchBackend(
        model_id,
        sampling=BedrockSamplingConfig(max_tokens=ANALYSIS_MAX_TOKENS),
        prefix=SOCIOLOGY_BATCH_PREFIX,
        run_id=run_dir.name,
    )
    completions = backend.collect(handle, timeout_seconds=timeout_seconds)
    out_path = replies_path(run_dir)
    existing = load_replies(out_path)
    records = []
    resumed = 0
    incomplete = 0
    for call, completion in zip(calls, completions, strict=True):
        if call.key in existing:
            resumed += 1
            continue
        record = _reply_record(call, completion, stimulus)
        incomplete += int(bool(record["incomplete"]))
        records.append(record)
    append_replies(out_path, records, carries=REPLY_CARRIES)
    summary = {
        "command": "collect-batch",
        "model_id": model_id,
        "collected": len(records),
        "resumed": resumed,
        "incomplete": incomplete,
        "input_tokens": backend.usage.input_tokens,
        "output_tokens": backend.usage.output_tokens,
        "stimulus_digest": stimulus.digest,
    }
    write_summary(run_dir, f"collect-{MODEL_SHORT_NAMES.get(model_id, 'model')}", summary)
    return summary


def run_live(model_id: str, run_dir: Path, *, concurrency: int, chunk_size: int) -> dict[str, Any]:
    """Run one live model's legs with resume, then write the summary as the completion marker."""
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    calls = planned_calls_for_model(model_id, manifest, units, stimulus)
    context_check = assert_context_fits(model_id, calls)
    counts = run_live_calls(
        calls, replies_path(run_dir), stimulus, concurrency=concurrency, chunk_size=chunk_size
    )
    summary = {
        "command": "run-live",
        "model_id": model_id,
        **counts,
        "context_check": context_check,
        "stimulus_digest": stimulus.digest,
    }
    write_summary(run_dir, f"live-{MODEL_SHORT_NAMES.get(model_id, 'model')}", summary)
    return summary


def _priced_leg(model_id: str, calls: Sequence[PlannedCall]) -> dict[str, Any]:
    """Price one model's planned calls: verified roster rates for batch, labelled rates for live."""
    input_tokens = int(sum(len(call.prompt) for call in calls) / CHARS_PER_TOKEN_ESTIMATE)
    output_tokens = ASSUMED_OUTPUT_TOKENS_PER_CALL * len(calls)
    transport = calls[0].transport
    if transport == TRANSPORT_BATCH:
        roster = roster_model(model_id)
        price_in, price_out = roster.batch_price_in_per_mtok, roster.batch_price_out_per_mtok
        price_note = "verified batch price"
    else:
        price_in, price_out = FRONTIER_LIVE_PRICE_UNVERIFIED
        price_note = "UNVERIFIED live price placeholder"
    cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
    return {
        "model_id": model_id,
        "transport": transport,
        "calls": len(calls),
        "estimated_input_tokens": input_tokens,
        "assumed_output_tokens": output_tokens,
        "price_per_mtok": [price_in, price_out],
        "price_note": price_note,
        "estimated_cost_usd": round(cost, 2),
    }


def dry_run(run_dir: Path) -> dict[str, Any]:
    """Count and price every production leg without touching AWS; record the context checks."""
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    report: dict[str, Any] = {"legs": [], "context_checks": []}
    total = 0.0
    for model_id in dict.fromkeys(leg.model_id for leg in PRODUCTION_LEGS):
        calls = planned_calls_for_model(model_id, manifest, units, stimulus)
        report["context_checks"].append(assert_context_fits(model_id, calls))
        priced = _priced_leg(model_id, calls)
        report["legs"].append(priced)
        total += float(priced["estimated_cost_usd"])
    report["estimated_total_usd"] = round(total, 2)
    report["assumptions"] = {
        "chars_per_token": CHARS_PER_TOKEN_ESTIMATE,
        "output_tokens_per_call": ASSUMED_OUTPUT_TOKENS_PER_CALL,
        "note": (
            "Opus 5 bills adaptive thinking against output, so its real output figure runs above "
            "the assumption; live GPT-5.6 prices are placeholders labelled UNVERIFIED"
        ),
    }
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def scan_replies(run_dir: Path) -> dict[str, Any]:
    """Run the deterministic scans over every reply on disk, writing one scan row per record."""
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    replies = load_replies(replies_path(run_dir))
    texts: dict[tuple[str, str], str] = {}
    for cell in manifest["cells"]:
        spec_cell = CELLS_BY_NAME[str(cell["name"])]
        for spec in bundles_of(manifest, spec_cell.name):
            texts[(spec_cell.name, spec.bundle_id)] = _bundle_text(spec, spec_cell, units)
    out_path = run_dir / "scans.jsonl"
    refuse_tracked_trace_path(out_path, carries="quoted transcript spans")
    rows = []
    examined = 0
    skipped_empty = 0
    for key in sorted(replies):
        record = replies[key]
        examined += 1
        reply = str(record.get("reply") or "")
        if not reply.strip():
            skipped_empty += 1
            continue
        bundle_text = texts[(str(record["cell"]), str(record["bundle_id"]))]
        rows.append(
            {
                "key": key,
                "cell": record["cell"],
                "model_id": record["model_id"],
                "stimulus_digest": stimulus.digest,
                **scan_reply(reply, bundle_text),
            }
        )
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    totals = {
        "examined": examined,
        "scanned": len(rows),
        "skipped_empty": skipped_empty,
        "replies_with_invented_quotes": sum(1 for row in rows if int(row["invented_quotes"]) > 0),
        "replies_with_non_verbatim_quotes": sum(
            1 for row in rows if int(row["non_verbatim_quotes"]) > 0
        ),
        "replies_with_denominated_rates": sum(
            1 for row in rows if row["rate_vs_narrative"] == "rate_with_denominator"
        ),
    }
    logger.info("scans: %s", json.dumps(totals))
    return totals


def judge_replies(run_dir: Path, *, limit: int | None = None) -> dict[str, int]:
    """Run the blind meta-judge over every reply on disk (resumable, chunk-appended)."""
    stimulus = load_stimulus()
    replies = load_replies(replies_path(run_dir))
    records: list[dict[str, Any]] = [replies[key] for key in sorted(replies)]
    if limit is not None:
        records = records[:limit]
    out_path = run_dir / "judged.jsonl"
    refuse_tracked_trace_path(out_path, carries="verbatim analysis replies")
    backend = BedrockBackend(
        judge_module.JUDGE_MODEL_ID,
        sampling=BedrockSamplingConfig(reasoning_effort=judge_module.JUDGE_REASONING_EFFORT),
    )
    counts = judge_module.judge_records(backend, records, out_path, stimulus)
    write_summary(
        run_dir,
        "judge",
        {
            "command": "judge",
            "judge_model_id": judge_module.JUDGE_MODEL_ID,
            "judge_effort": judge_module.JUDGE_REASONING_EFFORT,
            "judge_prompt_digest": judge_module.rubric_digest(stimulus),
            **counts,
            "usage": {
                "input_tokens": backend.usage.input_tokens,
                "output_tokens": backend.usage.output_tokens,
            },
        },
    )
    return counts


def judge_validation(run_dir: Path) -> dict[str, Any]:
    """Run the rubric's hand-authored validation replies through the live judge and report."""
    stimulus = load_stimulus()
    out_path = run_dir / "judge-validation.jsonl"
    refuse_tracked_trace_path(out_path, carries="validation reply texts")
    backend = BedrockBackend(
        judge_module.JUDGE_MODEL_ID,
        sampling=BedrockSamplingConfig(reasoning_effort=judge_module.JUDGE_REASONING_EFFORT),
    )
    report = judge_module.validate_judge(backend, stimulus, out_path)
    write_summary(run_dir, "judge-validation", {"command": "judge-validate", **report})
    return report


def _cell_bundle_stats(
    manifest: Mapping[str, Any], units: Mapping[str, Sequence[BundleUnit]]
) -> list[dict[str, Any]]:
    """Per-cell bundle statistics for the build report: sizes, redraws, cap headroom, cue presence."""
    stats: list[dict[str, Any]] = []
    for cell_entry in manifest["cells"]:
        cell = CELLS_BY_NAME[str(cell_entry["name"])]
        specs = bundles_of(manifest, cell.name)
        chars = [spec.rendered_chars for spec in specs]
        with_cue = sum(
            1 for spec in specs if contains_colocation_cue(_bundle_text(spec, cell, units))
        )
        stats.append(
            {
                "cell": cell.name,
                "design_label": cell.design_label,
                "bundles": len(specs),
                "size": cell.size,
                "framing": cell.framing,
                "cues": cell.cues,
                "rendered_chars_min": min(chars),
                "rendered_chars_mean": int(sum(chars) / len(chars)),
                "rendered_chars_max": max(chars),
                "estimated_tokens_max": int(max(chars) / CHARS_PER_TOKEN_ESTIMATE),
                "redraws_total": sum(spec.redraws for spec in specs),
                "bundles_with_colocation_cue": with_cue,
            }
        )
    return stats


def build(run_dir: Path) -> dict[str, Any]:
    """Build (or verify) the manifest and write the bundle-statistics report.

    An existing manifest is verified against a fresh rebuild rather than overwritten: the manifest
    is the run's resume key, and replacing it under a live run would re-attach every stored reply
    to different bundles. The statistics report carries the exclusion and redraw counts, the size
    distributions, the cap headroom for the size-16 cell, and how many kept-rung bundles actually
    contain the co-location cue (what makes the stripped rung a factor).
    """
    stimulus = load_stimulus()
    units = load_units()
    pools = {family: build_pool(family, family_units) for family, family_units in units.items()}
    manifest = build_manifest(pools, stimulus)
    manifest_file = run_dir / "manifest.json"
    if manifest_file.exists():
        assert_manifest_current(load_manifest(run_dir), manifest)
        manifest = load_manifest(run_dir)
        logger.info("existing manifest at %s verified against a fresh rebuild", manifest_file)
    else:
        write_manifest(manifest, run_dir)
    report = {
        "stimulus_digest": str(manifest["stimulus_digest"]),
        "pools": manifest["pools"],
        "cells": _cell_bundle_stats(manifest, units),
        "bundle_char_cap": MAX_BUNDLE_RENDER_CHARS,
    }
    report_path = run_dir / "bundle-stats.json"
    refuse_tracked_trace_path(report_path, carries="bundle statistics")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return report


def smoke(run_dir: Path) -> dict[str, Any]:
    """Run the authorized two-bundle end-to-end smoke: render, live call, scans, judge.

    One size-4 and one size-10 bundle (both agentic-120b, population frame) through the live path
    on the cheap reasoning-bearing smoke model, then the deterministic scans and the blind judge
    over the two replies. Everything lands under ``<run-dir>/smoke/`` so the production replies
    file never carries smoke records. The summary reports the judge's severity rungs and the
    measured chars-per-token ratio, which calibrates the dry-run estimate.
    """
    stimulus = load_stimulus()
    manifest = load_manifest(run_dir)
    units = load_units()
    smoke_dir = run_dir / "smoke"
    calls: list[PlannedCall] = []
    for cell_name in ("size-4", "center"):
        cell = CELLS_BY_NAME[cell_name]
        spec = next(s for s in bundles_of(manifest, cell_name) if s.family == FAMILY_AGENTIC_120B)
        text = _bundle_text(spec, cell, units)
        calls.append(
            PlannedCall(
                key=record_key(cell.name, spec.bundle_id, SMOKE_MODEL_ID, None, 0),
                cell=cell.name,
                bundle_id=spec.bundle_id,
                family=spec.family,
                framing=cell.framing,
                cues=cell.cues,
                size=cell.size,
                model_id=SMOKE_MODEL_ID,
                transport=TRANSPORT_LIVE,
                draw=0,
                reasoning_effort=None,
                prompt=render_prompt(stimulus, cell, text, n=cell.size),
            )
        )
    assert_context_fits(SMOKE_MODEL_ID, calls)
    out_path = smoke_dir / REPLIES_FILENAME
    counts = run_live_calls(calls, out_path, stimulus, concurrency=2, chunk_size=2)
    replies = load_replies(out_path)

    texts = {call.key: call.prompt for call in calls}
    scan_rows: list[dict[str, Any]] = []
    for call in calls:
        record = replies[call.key]
        scan_rows.append({"key": call.key, **scan_reply(str(record["reply"]), texts[call.key])})
    scans_path = smoke_dir / "scans.jsonl"
    refuse_tracked_trace_path(scans_path, carries="quoted transcript spans")
    with scans_path.open("w", encoding="utf-8") as handle:
        for row in scan_rows:
            handle.write(json.dumps(row) + "\n")

    judge_backend = BedrockBackend(
        judge_module.JUDGE_MODEL_ID,
        sampling=BedrockSamplingConfig(reasoning_effort=judge_module.JUDGE_REASONING_EFFORT),
    )
    judged_path = smoke_dir / "judged.jsonl"
    judge_counts = judge_module.judge_records(
        judge_backend, [replies[call.key] for call in calls], judged_path, stimulus
    )
    judged = judge_module.load_judged(judged_path)
    verdicts = {call.key: _smoke_verdict(judged.get(call.key), replies[call.key]) for call in calls}
    ratios = [
        int(record["prompt_chars"]) / int(record["input_tokens"])
        for record in (replies[call.key] for call in calls)
        if int(record["input_tokens"] or 0) > 0
    ]
    summary = {
        "command": "smoke",
        "model_id": SMOKE_MODEL_ID,
        **counts,
        "judge_counts": judge_counts,
        "verdicts": verdicts,
        "scans": scan_rows,
        "measured_chars_per_token": [round(ratio, 2) for ratio in ratios],
        "stimulus_digest": stimulus.digest,
    }
    write_summary(smoke_dir, "smoke", summary)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return summary


def _smoke_verdict(row: Mapping[str, Any] | None, reply: Mapping[str, Any]) -> dict[str, object]:
    """Summarise one smoke reply's judge outcome, tolerating empty and unparsed replies.

    A reply the judge skipped as empty has no row: a hijacked or refusing model returns no
    analysable text, which is a finding to surface (``skipped_empty``), never a KeyError.
    """
    if row is None:
        return {"severity": "skipped_empty", "reply_chars": len(str(reply.get("reply") or ""))}
    verdict = row.get("verdict")
    if not isinstance(verdict, dict):
        return {"severity": "UNPARSED", "judge_error": row.get("judge_error")}
    return verdict


_MODEL_CHOICES = {short: model_id for model_id, short in MODEL_SHORT_NAMES.items()}


def _at_least_one(text: str) -> int:
    """Parse an argparse integer that must be 1 or more, refusing at the parser rather than in a run."""
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer of at least 1, got {value}")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="build or verify the manifest and report bundle statistics")
    sub.add_parser("dry-run", help="count and price every production leg; no AWS calls")
    submit = sub.add_parser("submit-batch", help="submit ONE pooled batch job for a model")
    submit.add_argument("--model", choices=["opus", "haiku"], required=True)
    submit.add_argument(
        "--check-only",
        action="store_true",
        help="run the size and context checks and stop before anything is submitted or billed",
    )
    submit.add_argument(
        "--canary-calls",
        type=_at_least_one,
        default=REFUSAL_CANARY_CALLS,
        help=(
            "live calls per cell the refusal canary sends before a Claude job is submitted; a cell "
            "that answers none of them refuses the submit (see sociology/refusal_canary.py)"
        ),
    )
    collect = sub.add_parser("collect-batch", help="collect a submitted job into the replies file")
    collect.add_argument("--model", choices=["opus", "haiku"], required=True)
    collect.add_argument("--timeout-seconds", type=float, default=3600.0)
    live = sub.add_parser("run-live", help="run a live model's legs with resume-by-key")
    live.add_argument("--model", choices=["sol", "luna"], required=True)
    live.add_argument("--concurrency", type=int, default=4)
    live.add_argument("--chunk-size", type=int, default=8)
    sub.add_parser("smoke", help="the authorized two-bundle end-to-end smoke")
    sub.add_parser("scan", help="deterministic scans over every reply on disk")
    judge = sub.add_parser("judge", help="blind meta-judge over every reply on disk (resumable)")
    judge.add_argument("--limit", type=int, default=None)
    sub.add_parser("judge-validate", help="judge the stimulus file's validation replies")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one runner subcommand."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    run_dir: Path = args.run_dir
    if args.command == "build":
        build(run_dir)
    elif args.command == "dry-run":
        dry_run(run_dir)
    elif args.command == "submit-batch":
        result = submit_batch(
            _MODEL_CHOICES[args.model],
            run_dir,
            dry_run=args.check_only,
            canary_calls=args.canary_calls,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "collect-batch":
        result = collect_batch(
            _MODEL_CHOICES[args.model], run_dir, timeout_seconds=args.timeout_seconds
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "run-live":
        result = run_live(
            _MODEL_CHOICES[args.model],
            run_dir,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size,
        )
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "smoke":
        smoke(run_dir)
    elif args.command == "scan":
        sys.stdout.write(json.dumps(scan_replies(run_dir), indent=2) + "\n")
    elif args.command == "judge":
        judge_replies(run_dir, limit=args.limit)
    else:
        judge_validation(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
