r"""Noticed-versus-complied narration judge over legibility-probe records, on Bedrock batch.

The grader-twin runs (:data:`reward_hacking.legibility_probe.GRADER_TWIN_CELLS`) label every record
by construction: ``hack`` is a visible pass on the rigged ``oneoff`` grader with a hidden fail on the
original check, ``hidden_pass`` is the honest solve. Neither says what the model *saw*. A hack can be a
model that read the perturbed assertion, recognised it as wrong and returned the grader's value anyway,
or a model that never noticed and fitted its code to the examples in front of it; an honest pass on the
rigged twin can be a model that noticed the conflict and refused, or one that never looked at the
grader. Those are different dispositions with the same outcome labels, and the interpretability work
downstream needs them apart: the noticed-and-refused rows are the ``G_notice`` group every twin
direction is read against. This module puts one blind LLM call over each record's private reasoning
and labels the two things: NOTICED and COMPLIED. What is read and what a verdict is live in
:mod:`reward_hacking.legibility_narration_labels`; the denominators in
:mod:`reward_hacking.legibility_narration_summary`.

Design, each point load-bearing:

* **The judge is blind to every outcome.** It sees the task prompt exactly as the audited model saw it
  (problem statement, the grader as shown, the answer contract) and the reasoning span, and nothing
  else: never ``hack``, ``hidden_pass``, either grader's verdict, the disposition, the submitted code, the
  cell label or the served checkpoint. The honest twin's grader carries no perturbation, so judged
  noticing there is the false-positive floor -- but only if the judge could not have keyed on a label.
* **Submit-first, handle persisted, collect separately, resume by record key.** Every submit writes its
  request set to disk, creates one batch job for everything not yet judged or in flight, and saves the
  handle the moment the job exists, so a job outlives the session and is never paid for twice; a later
  submit over the same or a grown record set covers only the new keys, and a row whose reply did not
  parse is pending again. Every judged row stores the raw reply, the parsed verdict or the parse error,
  token usage, stop reason, the judge model id and the digest of the instruction file it was judged under.
* **The instruction text lives in a gitignored file** (:data:`DEFAULT_INSTRUCTIONS_PATH`), per the repo's
  stimulus rule; rendered prompts quote grader text and reasoning, and land only under gitignored roots.
* **Priced before anything is spent.** ``plan`` estimates the job from the rendered prompts at the roster's
  batch prices and refuses above the cap (default :data:`DEFAULT_MAX_COST_USD`); ``submit`` runs the same
  refusal before the job exists. The validated hatch judge (``gpt-5.6-luna``) is not batch-capable, so
  the judge model is chosen per run from the batch roster and recorded on every row.

Run order (every path under the gitignored ``artifacts/``)::

    plan     --records <unit>-records.jsonl ... --model <roster id> --out-dir <run>   # dry run, priced
    submit   --records ... --model <roster id> --out-dir <run>                        # one job, handle saved
    collect  --out-dir <run>                                                          # every uncollected handle
    summary  --records ... --out-dir <run> [--planted-values <problem id -> literal>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from games.provenance import git_provenance
from reward_hacking.bedrock_batch import (
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_POLL_SECONDS,
    HANDLE_CARRIES,
    BatchJobHandle,
    BedrockBatchBackend,
    roster_model,
)
from reward_hacking.legibility_narration_labels import (
    JudgeReplyError,
    Selected,
    load_planted_values,
    load_records,
    parse_judge_reply,
    quote_is_verbatim,
    select_records,
)
from reward_hacking.legibility_narration_summary import render_summary_markdown, summarize
from reward_hacking.legibility_probe import KNOWN_CELLS
from reward_hacking.model_backend import (
    STOP_REASON_MAX_TOKENS,
    BedrockSamplingConfig,
    TokenUsage,
    is_incomplete_stop_reason,
)
from reward_hacking.trace import refuse_tracked_trace_path
from reward_hacking.train_dataset import training_rows
from reward_hacking.train_partition import DEFAULT_PARTITION_PATH, load_partition

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from reward_hacking.model_backend import BedrockCompletion
    from reward_hacking.train_partition import HeldOutPartition

logger = logging.getLogger(__name__)

JUDGE_PROMPT_VERSION = "legibility-narration-judge-v1"
BATCH_PREFIX = "batch_jobs/legibility-narration-judge"
DEFAULT_INSTRUCTIONS_PATH = Path(
    "docs/scratch/tmax-interp-first-wave-2026-09-04/legibility-narration-judge-instructions.md"
)

DEFAULT_MAX_COST_USD = 40.0
DEFAULT_OUTPUT_TOKENS_PER_RECORD = 1500
"""Output allowance per judge reply in the estimate: a short JSON verdict plus a reasoning model's
thinking, which on the frontier roster row is billed as output and runs to a few thousand tokens."""

CHARS_PER_TOKEN = 3.0
"""Characters per token for the input estimate. Measured on the screen's three 9B record files as
completion characters over recorded completion tokens under the Qwen tokenizer: 3.03, 3.13 and 3.62.
The lowest is taken, so the estimate errs high; a judge tokenizer denser than Qwen's would only lower it."""

ERROR_MESSAGE_EXAMPLES = 3


# ---------------------------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Instructions:
    """The judge instruction text and the digest every judged row records against it."""

    text: str
    digest: str


def load_instructions(path: Path = DEFAULT_INSTRUCTIONS_PATH) -> Instructions:
    """Load the gitignored instruction file, refusing absence or a version line that does not match.

    A default here would be committed judge prose, which the stimulus rule forbids; the first line of
    the file is the prompt version, pinned so a stale or unrelated file cannot be judged under as if it
    were this one. The digest covers the whole file, version line included.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"judge instruction file {path} is missing. It is gitignored on purpose (judge prompts "
            "are authored stimulus that must never be committed) and a fresh clone does not contain "
            "it; recreate it beside the wave plan in docs/scratch, first line "
            f"{JUDGE_PROMPT_VERSION!r}."
        )
    raw = path.read_text(encoding="utf-8")
    version, _, body = raw.partition("\n")
    if version.strip() != JUDGE_PROMPT_VERSION:
        raise ValueError(
            f"{path} starts with {version.strip()!r}; this judge expects its first line to be "
            f"{JUDGE_PROMPT_VERSION!r}"
        )
    if not body.strip():
        raise ValueError(f"{path} carries a version line and no instructions")
    return Instructions(text=body.strip(), digest=hashlib.sha256(raw.encode()).hexdigest()[:16])


# ---------------------------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    """One rendered judge prompt plus the positional metadata that never travels to the model."""

    key: str
    unit: str
    cell: str
    stratum: str
    problem_id: str
    task_id: str
    prompt: str
    reasoning: str

    def metadata(self) -> dict[str, str]:
        """Return the sidecar labels for this request: identity and stratum, no outcome fields."""
        return {
            "key": self.key,
            "unit": self.unit,
            "cell": self.cell,
            "stratum": self.stratum,
            "problem_id": self.problem_id,
            "task_id": self.task_id,
        }


def render_judge_prompt(instructions: str, *, task_prompt: str, reasoning: str) -> str:
    """Assemble one prompt: the instructions, the task as the audited model saw it, its reasoning.

    Deliberately carries nothing else -- no cell, unit, grade, disposition or submitted code -- so the
    judge cannot key on an outcome (see the module docstring's blindness rationale).
    """
    return (
        f"{instructions.rstrip()}\n\n"
        f"##### TASK PROMPT (as shown to the audited model, grading script included) #####\n"
        f"{task_prompt.strip()}\n\n"
        f"##### AUDITED MODEL'S PRIVATE REASONING (verbatim) #####\n{reasoning.strip()}\n\n"
        f"##### END OF TRANSCRIPT #####\n"
    )


def resolve_task_prompts(partition: HeldOutPartition, cell_labels: Iterable[str]) -> dict[str, str]:
    """Re-render, per visible-grader task id, the prompt each cell showed its problems.

    The records carry the task id and never the prompt, so the prompt is rebuilt from the same
    :func:`~reward_hacking.train_dataset.training_rows` the probe sampled from, under each cell's own
    arm and exposure; the join is on the cell's visible task id, which is unique per split.
    """
    prompts: dict[str, str] = {}
    for label in sorted(set(cell_labels)):
        cell = KNOWN_CELLS[label]
        for row in training_rows(cell.arm, partition, exposure=cell.exposure):
            prompts[str(row["task_id"])] = str(row["prompt"])
    return prompts


def build_requests(
    selected: Sequence[Selected], task_prompts: Mapping[str, str], instructions: Instructions
) -> list[JudgeRequest]:
    """Render one request per selected record, refusing a task id the corpus cannot render."""
    missing = sorted({s.task_id for s in selected if s.task_id not in task_prompts})
    if missing:
        raise ValueError(
            f"{len(missing)} selected task ids have no rendered prompt, e.g. "
            f"{missing[:ERROR_MESSAGE_EXAMPLES]}; the records and the partition disagree"
        )
    return [
        JudgeRequest(
            key=s.key,
            unit=s.unit,
            cell=s.cell,
            stratum=s.stratum,
            problem_id=s.problem_id,
            task_id=s.task_id,
            prompt=render_judge_prompt(
                instructions.text, task_prompt=task_prompts[s.task_id], reasoning=s.reasoning
            ),
            reasoning=s.reasoning,
        )
        for s in selected
    ]


# ---------------------------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The priced plan for one judge job, with the cap it was checked against."""

    model_id: str
    n_requests: int
    input_tokens: int
    output_tokens: int
    usd: float
    cap_usd: float
    price_verified: bool

    @property
    def over_cap(self) -> bool:
        """Whether the estimate exceeds the cap; the caller refuses rather than spends."""
        return self.usd > self.cap_usd


def estimate_cost(
    prompts: Sequence[str],
    *,
    model_id: str,
    output_tokens_per_record: int = DEFAULT_OUTPUT_TOKENS_PER_RECORD,
    cap_usd: float = DEFAULT_MAX_COST_USD,
) -> CostEstimate:
    """Price the prompts at the roster's batch rates from a characters-per-token estimate."""
    roster = roster_model(model_id)
    input_tokens = sum(math.ceil(len(prompt) / CHARS_PER_TOKEN) for prompt in prompts)
    output_tokens = len(prompts) * output_tokens_per_record
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    return CostEstimate(
        model_id=model_id,
        n_requests=len(prompts),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        usd=roster.estimated_cost_usd(usage),
        cap_usd=cap_usd,
        price_verified=roster.batch_price_verified,
    )


def refuse_over_cap(estimate: CostEstimate) -> None:
    """Raise when the estimate exceeds its cap, naming the numbers and the knob."""
    if estimate.over_cap:
        raise ValueError(
            f"the judge job is estimated at ${estimate.usd:.2f} ({estimate.n_requests} records, "
            f"~{estimate.input_tokens} input and {estimate.output_tokens} output tokens on "
            f"{estimate.model_id}), above the ${estimate.cap_usd:.2f} cap. Raise --max-cost-usd "
            "deliberately or judge fewer records"
        )


# ---------------------------------------------------------------------------------------------
# Judged rows
# ---------------------------------------------------------------------------------------------


def judged_row(
    request: JudgeRequest,
    completion: BedrockCompletion,
    *,
    judge_model_id: str,
    instructions_digest: str,
    job_stem: str,
) -> dict[str, Any]:
    """Build one judged row: the request's labels, provenance, and the verdict or the parse error.

    A truncated or otherwise incomplete reply is an error, never a kept verdict, and a quote the
    judge claims is verbatim is checked against the reasoning it was asked to quote from.
    """
    row: dict[str, Any] = {
        **request.metadata(),
        "job_stem": job_stem,
        "judge_model_id": judge_model_id,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "judge_instructions_digest": instructions_digest,
        "judge_prompt_digest": hashlib.sha256(request.prompt.encode()).hexdigest()[:16],
        "judge_stop_reason": completion.stop_reason,
        "judge_input_tokens": completion.usage.input_tokens,
        "judge_output_tokens": completion.usage.output_tokens,
        "judge_raw_reply": completion.text,
        "judged_at": datetime.now(UTC).isoformat(),
    }
    incomplete = completion.stop_reason == STOP_REASON_MAX_TOKENS or (
        completion.stop_reason is not None and is_incomplete_stop_reason(completion.stop_reason)
    )
    if incomplete:
        row["judge_error"] = f"incomplete reply: stop_reason={completion.stop_reason}"
        return row
    try:
        verdict = parse_judge_reply(completion.text)
    except JudgeReplyError as error:
        row["judge_error"] = str(error)
        return row
    row["verdict"] = asdict(verdict)
    row["noticed_quote_verbatim"] = quote_is_verbatim(verdict.noticed_quote, request.reasoning)
    row["complied_quote_verbatim"] = quote_is_verbatim(verdict.complied_quote, request.reasoning)
    return row


# ---------------------------------------------------------------------------------------------
# Files and resume
# ---------------------------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def judged_path(out_dir: Path) -> Path:
    """Where every collected job's judged rows accumulate."""
    return out_dir / "judged.jsonl"


def handle_path(out_dir: Path, job_stem: str) -> Path:
    """Where one job's batch handle lives; its existence means the job was paid for."""
    return out_dir / "handles" / f"{job_stem}.json"


def requests_path(out_dir: Path, job_stem: str) -> Path:
    """Where one job's submitted requests live, in submitted order."""
    return out_dir / "requests" / f"{job_stem}.jsonl"


def load_judged(out_dir: Path) -> dict[str, dict[str, Any]]:
    """Judged rows keyed by record key, last row winning: a re-judged key's newest row is the row."""
    return {str(row["key"]): row for row in _read_jsonl(judged_path(out_dir))}


def submitted_handles(out_dir: Path) -> dict[str, BatchJobHandle]:
    """Every saved handle by job stem."""
    handles_dir = out_dir / "handles"
    if not handles_dir.is_dir():
        return {}
    return {path.stem: BatchJobHandle.load(path) for path in sorted(handles_dir.glob("*.json"))}


def in_flight_keys(out_dir: Path, judged: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Keys submitted under a saved handle whose job has not been collected into the judged rows."""
    collected = {str(row["job_stem"]) for row in judged.values()}
    keys: set[str] = set()
    for stem in submitted_handles(out_dir):
        if stem not in collected:
            keys.update(str(row["key"]) for row in _read_jsonl(requests_path(out_dir, stem)))
    return keys


def pending_requests(out_dir: Path, requests: Sequence[JudgeRequest]) -> list[JudgeRequest]:
    """Return the requests not yet judged and not in flight: what a submit should pay for.

    A row that errored (no parsed verdict) is pending again, so a later submit retries it.
    """
    judged = load_judged(out_dir)
    done = {key for key, row in judged.items() if "verdict" in row}
    flying = in_flight_keys(out_dir, judged)
    return [r for r in requests if r.key not in done and r.key not in flying]


# ---------------------------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------------------------


class BatchTransport(Protocol):
    """The two batch calls this module makes; :class:`BedrockBatchBackend` satisfies it."""

    model_id: str

    def submit(
        self, prompts: list[str], *, metadata: Sequence[Mapping[str, Any]] | None = None
    ) -> BatchJobHandle:
        """Create one job over the prompts and return its handle."""
        ...

    def collect(
        self,
        handle: BatchJobHandle,
        *,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
    ) -> list[BedrockCompletion]:
        """Wait for the job and return its completions in submitted order."""
        ...


def submit_job(
    requests: Sequence[JudgeRequest], transport: BatchTransport, out_dir: Path, *, job_stem: str
) -> BatchJobHandle:
    """Persist the request set, create the job, and save its handle the moment it exists.

    The requests file is written first so the handle can never point at a request set that was not
    recorded; a submit that then fails leaves a requests file with no handle, which every later
    submit ignores (only handled requests count as in flight).
    """
    reqs_path = requests_path(out_dir, job_stem)
    refuse_tracked_trace_path(reqs_path, carries="grader text and verbatim model reasoning")
    refuse_tracked_trace_path(handle_path(out_dir, job_stem), carries=HANDLE_CARRIES)
    if reqs_path.exists() or handle_path(out_dir, job_stem).exists():
        raise FileExistsError(f"job stem {job_stem} already has files under {out_dir}")
    _append_jsonl(reqs_path, ({"position": i, **asdict(r)} for i, r in enumerate(requests)))
    handle = transport.submit(
        [r.prompt for r in requests], metadata=[r.metadata() for r in requests]
    )
    handle.save(handle_path(out_dir, job_stem))
    return handle


def _stored_requests(out_dir: Path, job_stem: str) -> list[JudgeRequest]:
    return [
        JudgeRequest(**{k: v for k, v in row.items() if k != "position"})
        for row in _read_jsonl(requests_path(out_dir, job_stem))
    ]


def collect_jobs(
    out_dir: Path,
    transport_for: Callable[[BatchJobHandle], BatchTransport],
    *,
    instructions_digest: str,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
) -> dict[str, int]:
    """Collect every handle not yet in the judged rows, appending one job's rows at a time."""
    judged = load_judged(out_dir)
    collected = {str(row["job_stem"]) for row in judged.values()}
    counts: Counter[str] = Counter()
    for stem, handle in submitted_handles(out_dir).items():
        if stem in collected:
            counts["already_collected"] += 1
            continue
        requests = _stored_requests(out_dir, stem)
        if len(requests) != handle.record_count:
            raise RuntimeError(
                f"{stem}: {len(requests)} stored requests but the handle names "
                f"{handle.record_count} records; the requests file does not belong to this job"
            )
        completions = transport_for(handle).collect(
            handle, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds
        )
        rows = [
            judged_row(
                request,
                completion,
                judge_model_id=handle.model_id,
                instructions_digest=instructions_digest,
                job_stem=stem,
            )
            for request, completion in zip(requests, completions, strict=True)
        ]
        _append_jsonl(judged_path(out_dir), rows)
        parsed = sum(1 for row in rows if "verdict" in row)
        counts["collected_jobs"] += 1
        counts["judged"] += parsed
        counts["errored"] += len(rows) - parsed
        logger.info(
            "collected %s: %d rows, %d parsed, %d errored",
            stem,
            len(rows),
            parsed,
            len(rows) - parsed,
        )
    return dict(counts)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _prepare(
    args: argparse.Namespace,
) -> tuple[list[JudgeRequest], list[dict[str, Any]], Instructions]:
    records = load_records(args.records)
    selected, report = select_records(records)
    instructions = load_instructions(args.instructions)
    task_prompts = resolve_task_prompts(load_partition(args.partition), {s.cell for s in selected})
    return build_requests(selected, task_prompts, instructions), report, instructions


def _estimate(args: argparse.Namespace, pending: Sequence[JudgeRequest]) -> CostEstimate:
    return estimate_cost(
        [r.prompt for r in pending],
        model_id=args.model,
        output_tokens_per_record=args.output_tokens_per_record,
        cap_usd=args.max_cost_usd,
    )


def _cmd_plan(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    refuse_tracked_trace_path(out_dir / "plan.json", carries="record keys and a priced plan")
    requests, report, instructions = _prepare(args)
    pending = pending_requests(out_dir, requests)
    estimate = _estimate(args, pending)
    plan = {
        "dry_run": True,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "instructions_digest": instructions.digest,
        "records": [str(p) for p in args.records],
        "selection": report,
        "n_selected": len(requests),
        "n_pending": len(pending),
        "by_stratum": dict(Counter(r.stratum for r in pending)),
        "by_unit": dict(Counter(r.unit for r in pending)),
        "estimate": asdict(estimate),
        "chars_per_token": CHARS_PER_TOKEN,
        **git_provenance(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    logger.info("plan: %s", json.dumps({k: v for k, v in plan.items() if k != "selection"}))
    refuse_over_cap(estimate)


def _job_stem(model_id: str, run_id: str) -> str:
    slug = model_id.replace(":", "-").replace(".", "-").replace("/", "-")
    return f"{slug}-{run_id}"


def _cmd_submit(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    requests, report, instructions = _prepare(args)
    pending = pending_requests(out_dir, requests)
    if not pending:
        logger.info(
            "nothing to submit: all %d selected records are judged or in flight", len(requests)
        )
        return
    estimate = _estimate(args, pending)
    refuse_over_cap(estimate)
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    job_stem = _job_stem(args.model, run_id)
    backend = BedrockBatchBackend(
        args.model,
        sampling=BedrockSamplingConfig(max_tokens=args.max_tokens),
        prefix=BATCH_PREFIX,
        run_id=run_id,
    )
    handle = submit_job(pending, backend, out_dir, job_stem=job_stem)
    (out_dir / f"{job_stem}-submit-manifest.json").write_text(
        json.dumps(
            {
                "job_stem": job_stem,
                "model_id": args.model,
                "n_selected": len(requests),
                "n_submitted": len(pending),
                "selection": report,
                "by_stratum": dict(Counter(r.stratum for r in pending)),
                "estimate": asdict(estimate),
                "instructions_path": str(args.instructions),
                "instructions_digest": instructions.digest,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,
                "job_arn": handle.job_arn,
                "submitted_at": handle.submitted_at,
                **git_provenance(),
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("submitted %d records as %s (%s)", len(pending), job_stem, handle.job_arn)


def _cmd_collect(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    refuse_tracked_trace_path(judged_path(out_dir), carries="judge replies quoting model reasoning")
    instructions = load_instructions(args.instructions)

    def transport_for(handle: BatchJobHandle) -> BatchTransport:
        return BedrockBatchBackend(handle.model_id, region=handle.region, profile=handle.profile)

    counts = collect_jobs(
        out_dir,
        transport_for,
        instructions_digest=instructions.digest,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    logger.info("collect: %s", json.dumps(counts))


def _cmd_summary(args: argparse.Namespace) -> None:
    out_dir: Path = args.out_dir
    planted = None if args.planted_values is None else load_planted_values(args.planted_values)
    summary = summarize(load_records(args.records), load_judged(out_dir), planted=planted)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(render_summary_markdown(summary))
    logger.info("summary: identifiability %s", json.dumps(summary["identifiability"]))


def _add_records_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION_PATH)
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    parser.add_argument("--model", required=True, help="a batch-roster judge model id")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST_USD)
    parser.add_argument(
        "--output-tokens-per-record", type=int, default=DEFAULT_OUTPUT_TOKENS_PER_RECORD
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: plan (the priced dry run), submit, collect, and summarize."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="dry run: select, render, price, refuse over the cap")
    _add_records_args(plan)
    _add_render_args(plan)
    plan.set_defaults(func=_cmd_plan)

    submit = sub.add_parser("submit", help="one batch job over everything not judged or in flight")
    _add_records_args(submit)
    _add_render_args(submit)
    submit.add_argument("--run-id", default=None)
    submit.add_argument("--max-tokens", type=int, default=BedrockSamplingConfig().max_tokens)
    submit.set_defaults(func=_cmd_submit)

    collect = sub.add_parser("collect", help="collect every saved handle not yet judged")
    collect.add_argument("--out-dir", type=Path, required=True)
    collect.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    collect.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    collect.add_argument("--timeout-seconds", type=float, default=BATCH_TIMEOUT_SECONDS)
    collect.set_defaults(func=_cmd_collect)

    summary = sub.add_parser("summary", help="denominators per unit and stratum, plus agreement")
    _add_records_args(summary)
    summary.add_argument("--planted-values", type=Path, default=None)
    summary.set_defaults(func=_cmd_summary)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
