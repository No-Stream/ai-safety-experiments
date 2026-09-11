"""A live refusal canary in front of every Claude batch submit: cents to learn what a job would buy.

Two Opus 5 batch jobs on 2026-09-02 came back 1,862 of 1,920 and 751 of 768 records
``content_filtered`` -- Anthropic's safety classifier refusing the stimulus wholesale -- and every one
of those records was billed. A refused record is not data (what the classifier let through is a
classifier-selected fragment, never the model's answer; see :func:`sociology.decoupled_scans.is_refused`),
so the two jobs bought nothing usable, and the only way to have known beforehand was to ask the live
tier first. This module asks.

Before a Claude batch submit, the first few DISTINCT prompts of every group the job pools -- a cell in
the decoupled ladder, a framing in the analysis-model study -- go through the live Converse path under
the SAME sampler the batch would use, and the submit is refused if any group produced no answer at all.
Five calls per group, and the whole group has to come back without an answer, because per-record
refusal is real on stimulus the classifier merely dislikes: 67 of 396 stage-1 Opus records (17%) were
refused inside a job that was otherwise fine, and a one-call any-refusal rule would have refused that
job about one time in six per framing. Five refusals in a row on a 17% stimulus happens once in
~7,000 groups; on the wholesale-refused stimulus (97%) it happens 86% of the time per group and
effectively always across a leg's cells. A transport failure on every call of a group refuses the
same way -- the canary could not read the model, so it cannot say what the job would buy -- and the
refusal names the two counts separately so the operator knows whether to retry or to stand down.
"Answered" means neither of those: not ``content_filtered`` and not a failed call. A truncated or
empty reply still counts as answered, because the question here is whether the classifier let the
prompt through, not whether the model said anything useful.

That arithmetic only holds at the call count it was done for, so a group with fewer distinct prompts
than the canary asks for is refused at plan time rather than canaried on whatever it has: a one-prompt
group tripping on one refusal is exactly the one-call rule above. Every production group today has at
least 12 distinct prompts; a thinner plan names its count with ``--canary-calls`` and owns the rule.

The canary's rows are counts, never data: nothing it receives is written into a replies file, and the
batch's own content is untouched -- the same prompts, the same sampler, the same record order. It adds
live calls and a summary block to the submit, nothing else.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from reward_hacking.bedrock_batch import vendor_namespace
from reward_hacking.model_backend import TokenUsage, is_incomplete_stop_reason
from sociology.decoupled_scans import STOP_REASON_CONTENT_FILTERED

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from reward_hacking.model_backend import BedrockCompletion
    from sociology.records import LiveBackendFactory, LiveCall

CANARY_VENDOR = "anthropic"
"""The vendor whose batch jobs get a canary: the only one whose classifier has refused a job wholesale."""

REFUSAL_CANARY_CALLS = 5
"""Live calls per pooled group before a Claude batch submit; see the module docstring for the arithmetic."""

REFUSAL_CANARY_CONCURRENCY = 8
"""Canary calls in flight at once: the per-model live cap the receiver-map runbook settled on."""


class RefusalCanaryError(RuntimeError):
    """A Claude batch submit refused because a pooled group answered none of its canary calls."""


def needs_refusal_canary(model_id: str) -> bool:
    """Whether a batch submit for this model runs the canary first: only the Claude rows do."""
    return vendor_namespace(model_id) == CANARY_VENDOR


def canary_calls[CallT: LiveCall](
    calls: Sequence[CallT], *, group_of: Callable[[CallT], str], calls_per_group: int
) -> dict[str, list[CallT]]:
    """Pick the first ``calls_per_group`` calls with DISTINCT prompts from every group, in plan order.

    Distinct prompts rather than the first few calls, because a plan lists its draws contiguously and
    five draws of one prompt would test the classifier on one text five times. Plan order rather than a
    random pick, so a re-run canaries the same prompts and its counts are comparable with the last one's.
    A group with fewer distinct prompts than asked is refused (see the module docstring): the
    all-refused rule was argued at ``calls_per_group`` calls, and a thinner group would run a rule
    nobody argued for.
    """
    if calls_per_group < 1:
        raise ValueError(f"calls_per_group must be at least 1, got {calls_per_group}")
    picked: dict[str, list[CallT]] = {}
    prompts_seen: dict[str, set[str]] = {}
    for call in calls:
        group = group_of(call)
        chosen = picked.setdefault(group, [])
        seen = prompts_seen.setdefault(group, set())
        if len(chosen) >= calls_per_group or call.prompt in seen:
            continue
        seen.add(call.prompt)
        chosen.append(call)
    thin = {group: len(chosen) for group, chosen in picked.items() if len(chosen) < calls_per_group}
    if thin:
        raise ValueError(
            f"the refusal canary asks {calls_per_group} distinct prompts of every group, but "
            f"{len(thin)} group(s) have fewer: {thin}. The all-refused rule is only the rule argued "
            f"for at {calls_per_group} calls; pass --canary-calls {min(thin.values())} or fewer to "
            f"canary this plan on the prompts it has, and own the higher false-trip rate that buys"
        )
    return picked


def recorded_canary(submit_summary_path: Path) -> dict[str, Any] | None:
    """Return the canary block an earlier submit recorded for the job a resume is about to reuse.

    A resumed handle runs no canary -- that job is already bought -- but its submit rewrites the
    same summary, and a rewrite that said ``refusal_canary: null`` would erase what the original
    submit saw. ``None`` only when no summary exists or it predates the canary, which is honest: no
    canary is recorded for that job.
    """
    if not submit_summary_path.exists():
        return None
    return json.loads(submit_summary_path.read_text(encoding="utf-8")).get("refusal_canary")


def canary_plan[CallT: LiveCall](
    calls: Sequence[CallT], *, group_of: Callable[[CallT], str], calls_per_group: int
) -> dict[str, Any]:
    """Describe what the canary would send without sending it, for the priced dry runs."""
    picked = canary_calls(calls, group_of=group_of, calls_per_group=calls_per_group)
    return {
        "calls_per_group": calls_per_group,
        "groups": {group: len(group_calls) for group, group_calls in picked.items()},
        "calls": sum(len(group_calls) for group_calls in picked.values()),
    }


def _group_tally(completions: Sequence[BedrockCompletion]) -> dict[str, Any]:
    """Count one group's canary outcomes three ways: refused, transport-failed, and answered."""
    refused = sum(1 for c in completions if c.stop_reason == STOP_REASON_CONTENT_FILTERED)
    failed = sum(
        1
        for c in completions
        if c.stop_reason != STOP_REASON_CONTENT_FILTERED
        and c.stop_reason is not None
        and is_incomplete_stop_reason(c.stop_reason)
    )
    stop_reasons = Counter(str(c.stop_reason) for c in completions)
    return {
        "calls": len(completions),
        "refused": refused,
        "failed": failed,
        "answered": len(completions) - refused - failed,
        "stop_reasons": dict(sorted(stop_reasons.items())),
    }


@dataclass(frozen=True, slots=True)
class RefusalCanaryResult:
    """What the canary saw, per group and in total, plus which groups produced no answer at all."""

    summary: dict[str, Any]
    unanswered_groups: tuple[str, ...]

    @property
    def tripped(self) -> bool:
        """Whether any pooled group came back with no answer, refused or failed on every call."""
        return bool(self.unanswered_groups)

    def refuse_if_tripped(self, *, job: str, records: int) -> None:
        """Raise :class:`RefusalCanaryError` naming the job, the records it would have bought, and why."""
        if not self.tripped:
            return
        groups = self.summary["groups"]
        detail = "; ".join(
            f"{group}: {groups[group]['refused']} refused, {groups[group]['failed']} failed "
            f"of {groups[group]['calls']}"
            for group in self.unanswered_groups
        )
        raise RefusalCanaryError(
            f"refusing to submit {job} ({records} records on {self.summary['model_id']}): the live "
            f"canary got no answer from {len(self.unanswered_groups)} pooled group(s) -- {detail}. A "
            f"refused record is a classifier-selected fragment rather than an answer, so a job whose "
            f"cells refuse wholesale buys nothing readable; a group that failed on every call could not "
            f"be read at all, so retry the submit once the live tier answers. Nothing was submitted and "
            f"nothing but the canary calls was billed."
        )


def run_refusal_canary[CallT: LiveCall](
    calls: Sequence[CallT],
    *,
    group_of: Callable[[CallT], str],
    backend_factory: LiveBackendFactory,
    calls_per_group: int = REFUSAL_CANARY_CALLS,
) -> RefusalCanaryResult:
    """Send every group's canary prompts live under the job's own sampler and tally what came back.

    One backend for the whole canary, because a batch job is one (model, effort) pair by construction;
    two of either in one call list is a plan bug and refuses here rather than canarying half a job.
    Nothing the model returns is kept beyond its stop reason and its token counts -- the canary's rows
    are accounting, not data, and must never land in a replies file.
    """
    picked = canary_calls(calls, group_of=group_of, calls_per_group=calls_per_group)
    flat = [call for group_calls in picked.values() for call in group_calls]
    if not flat:
        raise ValueError(
            "the refusal canary was handed no calls; a leg with nothing to submit has no job"
        )
    model_ids = sorted({call.model_id for call in flat})
    efforts = {call.reasoning_effort for call in flat}
    if len(model_ids) != 1 or len(efforts) != 1:
        raise ValueError(
            f"a batch job is one model at one effort, but these calls span models {model_ids} and "
            f"efforts {sorted(efforts, key=str)}; the canary cannot stand for a job that is not one job"
        )
    (model_id,) = model_ids
    (effort,) = efforts
    backend = backend_factory(model_id, effort, min(len(flat), REFUSAL_CANARY_CONCURRENCY))
    completions = backend.generate_detailed([call.prompt for call in flat])
    if len(completions) != len(flat):
        raise RuntimeError(
            f"the canary backend returned {len(completions)} completions for {len(flat)} prompts; a "
            f"results list of the wrong length is a transport bug, not a refusal rate"
        )
    groups: dict[str, dict[str, Any]] = {}
    cursor = 0
    for group, group_calls in picked.items():
        groups[group] = _group_tally(completions[cursor : cursor + len(group_calls)])
        cursor += len(group_calls)
    usage = TokenUsage()
    for completion in completions:
        usage += completion.usage
    elapsed = [c.elapsed_seconds for c in completions if c.elapsed_seconds is not None]
    summary: dict[str, Any] = {
        "model_id": model_id,
        "reasoning_effort": effort,
        "calls_per_group": calls_per_group,
        "calls": len(flat),
        "refused": sum(tally["refused"] for tally in groups.values()),
        "failed": sum(tally["failed"] for tally in groups.values()),
        "answered": sum(tally["answered"] for tally in groups.values()),
        "groups": groups,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_input_tokens": usage.cache_read_input_tokens,
        "cache_write_input_tokens": usage.cache_write_input_tokens,
        "max_elapsed_seconds": max(elapsed) if elapsed else None,
    }
    unanswered = tuple(group for group, tally in groups.items() if tally["answered"] == 0)
    return RefusalCanaryResult(summary=summary, unanswered_groups=unanswered)
