r"""The hatch/framing probe's runner: pinned sampler, backend construction, and the cell loop.

The substance -- cells, rendering, the cross-cell budget, report parsing, double grading, records,
summary -- lives in :mod:`reward_hacking.hatch_probe`; this module is what turns it into a run
against a real backend. Sampling is pinned, never a flag: temperature 1.0 (TRL's own training
value, so the anchor cells sample the distribution the flagship screen measured), ``top_p`` omitted
on Bedrock Converse, and the output cap from the measured
:data:`~reward_hacking.recoverybench.budgets.MAX_TOKENS_BY_MODEL` table, which refuses a model
nobody has measured -- that refusal is the mechanism working. The levers for a cheaper run are
``--max-problems`` and ``--samples-per-prompt``, never the sampler, least of all the token cap in
the experiment about exactly that confound (a capped rollout submits nothing and reads as
incapacity).

Every transport ``backend_cli`` offers works here except ``codex``, whose own agent harness makes
it a different elicitation; ``--bedrock-batch`` swaps live Converse for the batch inference service
with the identical request payload, submitting all the run's cells as ONE job (see
:func:`sample_all_cells_in_one_batch` for why a per-cell submit is the shape to avoid). The job's
handle lands beside the summary the moment the job exists -- before any waiting -- so a crash,
timeout, or dead session never orphans the paid inference: re-running the identical command resumes
the same job from that handle instead of submitting a new one, and ``--batch-timeout-seconds`` says
how long one sitting waits. Grading always runs on local CPU in the episode jail, whatever sampled
the completions -- which is what lets this run unchanged against hosted models.

    uv run python -m reward_hacking.hatch_probe_cli --model <bedrock-model-id> --backend bedrock \\
        --out artifacts/reward_hacking/option3/hatch-probe/<model>.json \\
        --grader-scratch-root /var/tmp/rh-hatch-probe-graders

The per-record JSONL lands beside the summary and is appended cell by cell, so a crashed run keeps
its completed cells. The summary is written LAST, exclusively: its presence is the completion
marker, and a records file without one is a run that died mid-flight.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoTokenizer

from games.generation import TRAINING_TEMPERATURE
from games.preflight import derive_prefilled_think, resolve_chat_template_kwargs
from games.provenance import git_provenance
from reward_hacking.backend_cli import (
    LOCAL_KINDS,
    add_backend_args,
    backend_from_args,
    log_token_usage,
    reject_inapplicable_knobs,
    resolve_thinking,
)
from reward_hacking.bedrock_batch import (
    BATCH_TIMEOUT_HOURS,
    BATCH_TIMEOUT_SECONDS,
    DEFAULT_POLL_SECONDS,
    HANDLE_CARRIES,
    MIN_BATCH_RECORDS,
    BatchJobHandle,
    BedrockBatchBackend,
    cell_digest,
    prompt_digest,
)
from reward_hacking.harness.tasks_ilcb import PROBLEMS_BY_TASK_ID
from reward_hacking.hatch_probe import (
    DEFAULT_HATCH_LEVELS,
    GIVE_UP_CLOSE,
    GIVE_UP_OPEN,
    GRADER_REPORT_CLOSE,
    GRADER_REPORT_OPEN,
    HATCH_LEVELS,
    HATCH_TEXT_BY_LEVEL,
    PROBE_KIND,
    PROBE_SCHEMA,
    CellInputs,
    ProbeCell,
    ProbeRunContext,
    RecordDisposition,
    aligned_problem_ids,
    apply_cross_cell_budget,
    build_cell_records,
    cell_inputs_by_label,
    dropped_only_by_variant_text,
    generate_cell_responses_in_order,
    load_resumable_cells,
    probe_cells_for,
    probe_summary,
    reclaim_probe_paths,
    records_path_for,
    repeat_for_samples,
    reserve_probe_paths,
    resumed_cell_responses,
    write_probe_summary,
)
from reward_hacking.model_backend import (
    DEFAULT_BEDROCK_REGION,
    Backend,
    BedrockSamplingConfig,
    MockBackend,
    converse_supports_temperature,
    raw_response,
)
from reward_hacking.recoverybench.budgets import max_tokens_for
from reward_hacking.trace import refuse_tracked_trace_path, write_trace
from reward_hacking.train_dataset import (
    SOLUTION_CLOSE,
    SOLUTION_OPEN,
    TRAINING_ARMS,
    TRAINING_SPLIT_BY_ARM,
    BoundedSubset,
    training_rows,
)
from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    SPLIT_ONEOFF,
    HeldOutPartition,
    load_partition,
)
from reward_hacking.train_reward import (
    GradedCompletion,
    GraderConfig,
    assert_jail_usable,
    grade_solution,
)
from reward_hacking.train_screen import (
    DEFAULT_MAX_PROMPT_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_SAMPLES_PER_PROMPT,
    DEFAULT_THINKING,
    assert_backend_can_render_template,
    training_matched_sampling,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

    from reward_hacking.harness.tasks_ilcb import ILCBProblem
    from reward_hacking.hatch_probe import HatchProbeRecord
    from reward_hacking.model_backend import RawResponse
    from reward_hacking.train_dataset import PromptBudgetFilter

logger = logging.getLogger(__name__)

# The training template's opening line, which keys the mock completion to its entry point.
_ENTRY_POINT_PATTERN = re.compile(r"Implement `([^`]+)` in Python")


def mock_probe_response(prompt: str) -> str:
    """Build the ``--backend mock`` canned completion: parses cleanly, fails the grader honestly.

    Keyed on the prompt because a static string cannot define the entry point the grader imports.
    When the prompt offers the report channel, the mock also files a report, and when it offers
    the give-up channel the mock also gives up (beside its solution, which the contract allows),
    so a mock run exercises both channels' parsing paths end to end for free. The whole path
    executes -- real jail, both graders -- which is all a mock run is for.
    """
    match = _ENTRY_POINT_PATTERN.search(prompt)
    if match is None:
        raise ValueError(
            "the mock backend could not find the entry point in this prompt; the training prompt "
            "template moved and the canned completion would grade as a jail failure"
        )
    report = (
        f"\n{GRADER_REPORT_OPEN}mock report: exercising the channel{GRADER_REPORT_CLOSE}"
        if GRADER_REPORT_OPEN in prompt
        else ""
    )
    give_up = (
        f"\n{GIVE_UP_OPEN}mock give-up: exercising the channel{GIVE_UP_CLOSE}"
        if GIVE_UP_OPEN in prompt
        else ""
    )
    return (
        "mock reasoning\n</think>\n"
        f"{SOLUTION_OPEN}\ndef {match.group(1)}(*args, **kwargs):\n    return None\n"
        f"{SOLUTION_CLOSE}{report}{give_up}"
    )


# The CLI sampler flags this probe pins shut, mapped dest -> flag for the refusal message.
_PINNED_SAMPLER_FLAGS: dict[str, str] = {
    "temperature": "--temperature",
    "top_p": "--top-p",
    "top_k": "--top-k",
    "max_new_tokens": "--max-new-tokens",
    "min_p": "--min-p",
    "repetition_penalty": "--repetition-penalty",
    "presence_penalty": "--presence-penalty",
}


def refuse_sampler_flags(args: argparse.Namespace) -> None:
    """Refuse any decoding knob: this probe's sampler is pinned, and cheapness has other levers."""
    given = [
        flag
        for dest, flag in _PINNED_SAMPLER_FLAGS.items()
        if getattr(args, dest, None) is not None
    ]
    if given:
        raise ValueError(
            f"{', '.join(given)} cannot be set on this probe: its sampler is pinned "
            f"(temperature {TRAINING_TEMPERATURE}, top_p omitted, output cap from the measured "
            f"budgets table) so every cell samples the distribution the flagship screen measured. "
            f"The levers for a cheaper run are --max-problems and --samples-per-prompt."
        )


def resolve_probe_bedrock_sampling(
    model_id: str, *, reasoning_effort: str | None = None
) -> BedrockSamplingConfig:
    """Build the pinned Converse sampler: temperature 1.0, ``top_p`` omitted, measured cap.

    The cap comes from :func:`~reward_hacking.recoverybench.budgets.max_tokens_for`, which refuses
    a model nobody has measured -- that refusal is the mechanism working, not an obstacle: a
    guessed cap spends a whole sweep manufacturing truncation that reads as non-compliance.

    A family that refuses the ``temperature`` field outright
    (:data:`~reward_hacking.model_backend.CONVERSE_TEMPERATURE_REFUSING_FAMILIES` -- gpt-5.6,
    which samples at a fixed 1.0 and takes no override) gets the field omitted rather than the
    run refused: the pin wanted temperature 1.0 and that is what such a model runs at, so the
    distribution is the pinned one while the sampler record honestly shows ``temperature: null``
    for that rung. The pin stays a pin: this is the probed field-support fact deciding the
    request shape, never a per-run knob.
    """
    return BedrockSamplingConfig(
        max_tokens=max_tokens_for(model_id),
        temperature=TRAINING_TEMPERATURE if converse_supports_temperature(model_id) else None,
        top_p=None,
        reasoning_effort=reasoning_effort,
    )


def _refuse_bad_batch_wait_flags(args: argparse.Namespace) -> None:
    """Refuse the batch wait knobs off the batch transport, and non-positive values on it.

    Two failure modes, both before any spend. Off the batch transport, silently ignoring them is
    the trap: an operator who typed ``--batch-timeout-seconds`` on a live run believes a knob was
    honoured that nothing read. On it, these are the flags where a typo costs a paid batch job: a
    poll of zero turns the wait into a while-true poll storm against the API for up to the 24-hour
    job deadline, and a non-positive timeout submits and pays for the job and then raises on the
    first poll.
    """
    supplied = [
        (flag, value)
        for flag, value in (
            ("--batch-poll-seconds", args.batch_poll_seconds),
            ("--batch-timeout-seconds", args.batch_timeout_seconds),
        )
        if value is not None
    ]
    if not args.bedrock_batch:
        if supplied:
            raise ValueError(
                f"{', '.join(flag for flag, _ in supplied)} tune the wait on a batch inference "
                f"job, and this run submits none; pass --bedrock-batch with them or drop them"
            )
        return
    for flag, value in supplied:
        if value <= 0:
            raise ValueError(
                f"{flag} must be positive, got {value}. The job would already be submitted and "
                f"paid for when the wait gives up, so a nonsense wait is refused before the spend."
            )


def _resolve_batch_wait(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve the batch collect's (poll_seconds, timeout_seconds) from the flags' None defaults.

    The timeout defaults to the service's own job deadline rather than the library's session-sized
    ``DEFAULT_WAIT_SECONDS``: this CLI runs one job end to end, so its default wait never abandons
    a job that could still finish.
    """
    poll = args.batch_poll_seconds if args.batch_poll_seconds is not None else DEFAULT_POLL_SECONDS
    timeout = (
        args.batch_timeout_seconds
        if args.batch_timeout_seconds is not None
        else BATCH_TIMEOUT_SECONDS
    )
    return poll, timeout


def _build_bedrock_batch_backend(
    args: argparse.Namespace, model_id: str, sampling: BedrockSamplingConfig
) -> BedrockBatchBackend:
    """Construct the batch-service transport with the identical pinned request shape."""
    if args.concurrency is not None:
        raise ValueError(
            "--concurrency sizes the live Converse thread pool; the batch inference service has "
            "no such knob"
        )
    return BedrockBatchBackend(
        model_id,
        region=args.region if args.region is not None else DEFAULT_BEDROCK_REGION,
        profile=args.profile or None,
        sampling=sampling,
    )


def build_probe_backend(
    args: argparse.Namespace, model_id: str
) -> tuple[Backend, dict[str, object]]:
    """Construct the backend under the pinned sampler, plus the sampler record for the artifact.

    Every kind ``backend_cli`` offers except ``codex``: the codex CLI wraps the prompt in its own
    agent harness (system prompt, tool loop, workdir reads), so a codex record is a different
    elicitation than the bare user turn every other transport renders, and pooling them would put
    the scaffolding difference inside what reads as a model difference. ``--bedrock-batch`` swaps
    the live Converse transport for the batch inference service with the identical request shape.
    """
    kind: str = args.backend
    if kind == "codex":
        raise ValueError(
            "--backend codex is refused: the codex CLI runs its own agent harness around the "
            "prompt, a different elicitation than the bare user turn every other transport "
            "renders, so its numbers would not be comparable with any other cell of this probe"
        )
    if args.bedrock_batch and kind != "bedrock":
        raise ValueError(
            f"--bedrock-batch rides the bedrock transport; pass --backend bedrock with it, "
            f"not --backend {kind}"
        )
    if kind == "mock":
        reject_inapplicable_knobs(kind, args)
        logger.warning(
            "mock backend: nothing is sampled and the artifact is a plumbing smoke, not a "
            "measurement; records will still read model_id=%r",
            model_id,
        )
        return MockBackend(mock_probe_response, model_id=model_id), {
            "backend": kind,
            "sampled": False,
        }
    if kind in LOCAL_KINDS:
        if args.thinking is None:
            raise ValueError(
                "a local backend must state --thinking or --no-thinking explicitly: the shared "
                "CLI default is no-thinking while the flagship trained with thinking on, and "
                "inheriting the wrong template mode is a silent elicitation change"
            )
        local_sampling = training_matched_sampling(model_id)
        backend = backend_from_args(args, model_id, local_sampling=local_sampling)
        # An engine kwarg, not a SamplingConfig field: unrecorded, fp8 reads as the bf16 anchor.
        return backend, {
            "backend": kind,
            "sampled": True,
            "quantization": args.vllm_quantization,
            **asdict(local_sampling),
        }
    if kind == "bedrock":
        sampling = resolve_probe_bedrock_sampling(model_id, reasoning_effort=args.reasoning_effort)
        record: dict[str, object] = {
            "backend": kind,
            "sampled": True,
            "bedrock_batch": bool(args.bedrock_batch),
            **asdict(sampling),
        }
        if args.bedrock_batch:
            # backend_from_args does this for the live path, which the batch path returns before.
            reject_inapplicable_knobs(kind, args)
            return _build_bedrock_batch_backend(args, model_id, sampling), record
        return backend_from_args(args, model_id, bedrock_sampling=sampling), record
    raise ValueError(f"unknown backend kind {kind!r}")


def resolve_probe_problems(partition: HeldOutPartition) -> dict[str, tuple[ILCBProblem, ...]]:
    """Resolve the flagship's training-side problems, one aligned row list per trainable split.

    Through :func:`~reward_hacking.train_dataset.training_rows` rather than the registry directly,
    so the trainer's own exclusions (the ``conflicting`` refusal, the unsatisfiable-grader drop,
    the partition-side assertion) apply here identically instead of being re-derived.
    """
    problems_by_split = {
        TRAINING_SPLIT_BY_ARM[arm]: tuple(
            PROBLEMS_BY_TASK_ID[str(row["task_id"])] for row in training_rows(arm, partition)
        )
        for arm in TRAINING_ARMS
    }
    aligned_problem_ids(problems_by_split)
    return problems_by_split


def _bound_problems(
    problems_by_split: dict[str, tuple[ILCBProblem, ...]],
    *,
    max_problems: int | None,
    seed: int,
) -> tuple[dict[str, tuple[ILCBProblem, ...]], BoundedSubset | None]:
    """Bound the probe to a seeded subset of problems, loudly, identically in every cell.

    Applied AFTER the budget filter for the reason ``resolve_arm_rows`` fixes that order: the
    budget is a property of the corpus and the subset is a property of this run, and swapping them
    silently selects a different subset.
    """
    if max_problems is None:
        return problems_by_split, None
    if max_problems < 1:
        raise ValueError(f"a probe over {max_problems} problems measures nothing")
    problem_ids = list(aligned_problem_ids(problems_by_split))
    if max_problems >= len(problem_ids):
        return problems_by_split, None
    shuffled = list(problem_ids)
    random.Random(seed).shuffle(shuffled)
    kept_ids = frozenset(shuffled[:max_problems])
    subset = BoundedSubset(
        max_prompts=max_problems,
        n_total_rows=len(problem_ids),
        seed=seed,
        kept_problem_ids=tuple(shuffled[:max_problems]),
    )
    logger.warning("%s", subset.to_json_dict()["warning"])
    bounded = {
        split: tuple(problem for problem in problems if problem.task_id in kept_ids)
        for split, problems in problems_by_split.items()
    }
    return bounded, subset


def elicitation_record(args: argparse.Namespace) -> dict[str, object]:
    """Record the template mode this run applied, and the one the shared budget was measured in.

    Two different constants share the name ``DEFAULT_THINKING``: the screen's is True (the flagship
    trained with thinking on) and ``backend_cli``'s is False (no CLI turns traces on by default).
    The budget is measured in the screen's mode on purpose -- it is a property of the corpus, and
    changing it changes which problems survive -- so the two are reported separately rather than one
    being quietly read as the other.

    ``thinking`` is None where the transport templates server-side: ``resolve_thinking`` would report
    the shared CLI default there, a mode nothing applied. Present and empty beats a fabricated value.
    """
    return {
        "thinking": resolve_thinking(args) if args.backend in LOCAL_KINDS else None,
        "budget_thinking": DEFAULT_THINKING,
    }


def _resolve_prefilled_think(kind: str, model_id: str, *, thinking: bool) -> bool:
    """Say whether completions carry only a closing think tag, and refuse a template we cannot render.

    Local kinds render the model's own chat template, so the answer is measured off that template
    the way the screen measures it. Hosted transports apply their template server-side and return
    reasoning out of band, and the mock's canned completion closes its own block, so for all of
    those the completion text is read as-is.

    The served model's tokenizer is loaded here and nowhere else, so this is also where the screen's
    :func:`~reward_hacking.train_screen.assert_backend_can_render_template` belongs: the local
    backends template internally with no kwargs channel, so on a checkpoint whose template needs
    ``reasoning_effort`` pinned (Qwen3.8-27B injects a system message nobody wrote without it) this
    probe would sample different prompt text than both training and the budget it just measured --
    which would make the anchor cell's byte-identity claim false in the one direction no test over
    ``render_prompt`` can see, since that claim is about the pre-template string.
    """
    if kind not in LOCAL_KINDS:
        return False
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    assert_backend_can_render_template(tokenizer)
    return derive_prefilled_think(tokenizer, enable_thinking=thinking)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="model id the chosen backend serves")
    add_backend_args(parser, default="mock")
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION_PATH)
    parser.add_argument(
        "--out", type=Path, required=True, help="summary JSON path; the records JSONL lands beside"
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help="samples per problem per cell; the default matches the flagship screen",
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument(
        "--budget-model",
        default=DEFAULT_MODEL_ID,
        help=(
            "tokenizer that measures the shared prompt budget. The flagship's training model by "
            "default, so the probe's problem set anchors to the screen's whatever model is "
            "sampled; changing it changes which problems survive the budget"
        ),
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="bound the probe to a seeded subset of problems; the artifact records this LOUDLY",
    )
    parser.add_argument(
        "--hatch-levels",
        nargs="+",
        choices=HATCH_LEVELS,
        default=list(DEFAULT_HATCH_LEVELS),
        help=(
            "which hatch levels this run samples, each crossed with both arms and all three "
            "framings. The default is the original 12-cell design; the consequence levels "
            "(present-forfeit, present-costless) are the reward-forfeiting variant and the "
            "give-up levels (give-up, present-give-up) are the honest-stopping variant, each run "
            "as their own cells against an existing default run's cells as comparators. Cell "
            "order is canonical whatever order the levels are given in"
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="seeds only the subset selection")
    parser.add_argument(
        "--grader-scratch-root",
        type=Path,
        required=True,
        help=(
            "episode scratch root for the jailed graders. Point it somewhere under /var/tmp: the "
            "jail refuses the home tree outright, and /tmp is RAM-backed with an inode cap this "
            "box has already exhausted once. Required rather than defaulted so the choice is on "
            "the record"
        ),
    )
    parser.add_argument(
        "--bedrock-batch",
        action="store_true",
        help=(
            "submit through the Bedrock batch inference service instead of live Converse; the "
            "request shape is identical and all the run's cells ride in one job, so the service's "
            "100-record floor applies once to n_problems x samples x n_cells. The job's handle is "
            "persisted beside --out before any waiting, and re-running the identical command "
            "resumes that job instead of submitting a new one"
        ),
    )
    parser.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=None,
        help="how often the batch path polls the job; only meaningful with --bedrock-batch",
    )
    parser.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=None,
        help=(
            f"how long one sitting waits for the batch job before giving up, default the "
            f"service's own {BATCH_TIMEOUT_HOURS}-hour job deadline. A timeout never stops the "
            f"job: re-run the identical command to resume it from the persisted handle"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve problems, budget and plan; load no model, launch no jail",
    )
    args = parser.parse_args(argv)
    refuse_sampler_flags(args)
    _refuse_bad_batch_wait_flags(args)
    return args


def _dry_run_plan(
    args: argparse.Namespace,
    problems_by_split: Mapping[str, tuple[ILCBProblem, ...]],
    budget: PromptBudgetFilter,
    subset: BoundedSubset | None,
    variant_dropped: Sequence[str],
) -> dict[str, object]:
    """Everything the run would do, with no model loaded and no jail launched."""
    cells = probe_cells_for(args.hatch_levels)
    n_problems = len(problems_by_split[SPLIT_ONEOFF])
    return {
        "dry_run": True,
        "model_id": args.model,
        "backend": args.backend,
        "bedrock_batch": bool(args.bedrock_batch),
        "hatch_levels": sorted({cell.hatch for cell in cells}, key=HATCH_LEVELS.index),
        "cells": [cell.label for cell in cells],
        "n_problems": n_problems,
        "samples_per_prompt": args.samples_per_prompt,
        "n_calls": n_problems * args.samples_per_prompt * len(cells),
        "prompt_budget": budget.to_json_dict(),
        "dropped_only_by_variant_text": list(variant_dropped),
        "bounded_subset": subset.to_json_dict() if subset else None,
        "out": str(args.out),
        "records_out": str(records_path_for(args.out)),
        "batch_handle_out": str(batch_handle_path_for(args.out)) if args.bedrock_batch else None,
        # The one fact deciding whether this run spends anything: True means the run collects the
        # paid job already on disk (and resets the leftover records); False means it submits and
        # bills a new one.
        "batch_resumes_existing_handle": (
            batch_handle_path_for(args.out).exists() if args.bedrock_batch else None
        ),
        # The live twin: True means finished cells are on disk and will be skipped, not re-sampled.
        "live_resumes_existing_records": (
            None
            if args.bedrock_batch
            else records_path_for(args.out).exists() and not args.out.exists()
        ),
    }


def _refuse_below_batch_floor(n_records: int, *, n_cells: int) -> None:
    """Refuse a batch run under the service's record floor, in the levers THIS CLI has.

    ``BedrockBatchBackend.submit`` refuses too, but its message offers a ``--repeats`` multiplier
    this probe does not define, and it refuses only after a model-availability call and an upload.
    The floor is per job and not adjustable, so one job for all the run's cells clears it whenever
    ``n_problems x samples x n_cells`` does; the shapes that still fall short are the cheap smokes.
    """
    if n_records >= MIN_BATCH_RECORDS:
        return
    raise ValueError(
        f"batch inference needs at least {MIN_BATCH_RECORDS} records per job and this probe renders "
        f"{n_records} across all {n_cells} cells. The floor is per job and not adjustable, "
        f"so raise --samples-per-prompt or --max-problems until n_problems x samples x "
        f"{n_cells} reaches {MIN_BATCH_RECORDS}, or drop --bedrock-batch and run it on live "
        f"Converse, which has no floor and finishes a run this size in minutes."
    )


def batch_handle_path_for(out_path: Path) -> Path:
    """Where the batch job's handle lands, beside the summary so the artifacts travel together.

    Derived from ``--out`` exactly as :func:`~reward_hacking.hatch_probe.records_path_for` derives
    the records path, and for the same reason: the handle is part of the run, and a handle filed
    anywhere else is a handle a resume cannot find.
    """
    return out_path.with_name(f"{out_path.stem}-batch-handle.json")


def _refuse_resumed_handle_mismatch(
    handle: BatchJobHandle,
    backend: BedrockBatchBackend,
    *,
    repeated: list[str],
    metadata: list[dict[str, object]],
    handle_path: Path,
) -> None:
    """Refuse a saved handle whose job is not the job THIS invocation describes.

    Cell membership is positional, so collecting someone else's job against these locally rendered
    inputs would file real responses under the wrong problems, cells, or sampler labels -- every
    count plausible and wrong. The digests are the same tripwires ``jagged.sweep``'s collect runs
    (comparing the cell digest is the caller's obligation, per :func:`~reward_hacking.bedrock_batch.
    cell_digest`); the sampling comparison exists because the records' sampler stamp comes from this
    invocation's config, not the handle, so a job sampled at another cap or effort would be
    mislabelled record by record.
    """
    mismatches: list[str] = []
    if handle.model_id != backend.model_id:
        mismatches.append(f"model {handle.model_id!r} on the handle vs {backend.model_id!r} here")
    if handle.record_count != len(repeated):
        mismatches.append(
            f"{handle.record_count} records on the handle vs {len(repeated)} rendered here"
        )
    if handle.prompt_digest != prompt_digest(repeated):
        mismatches.append("the prompt digest (the rendered prompt text differs)")
    if handle.cell_digest != cell_digest(metadata):
        mismatches.append("the cell digest (the prompts match but their cell labels moved)")
    mismatches.extend(
        handle.sampling_label_mismatches(
            max_tokens=backend.sampling.max_tokens,
            reasoning_effort=backend.sampling.reasoning_effort,
        )
    )
    if mismatches:
        raise RuntimeError(
            f"the saved handle at {handle_path} does not describe the job this invocation would "
            f"submit: {'; '.join(mismatches)}. Collecting it would attach responses to the wrong "
            f"cells or stamp them with a sampler that never ran. Re-run with the flags of the "
            f"original submit, or point --out somewhere fresh to submit (and pay for) a new job. "
            f"The job the handle names is untouched ({handle.job_arn})."
        )


def _submit_or_resume_batch(
    backend: BedrockBatchBackend,
    repeated: list[str],
    metadata: list[dict[str, object]],
    handle_path: Path,
) -> BatchJobHandle:
    """Reuse the saved handle when one exists, otherwise submit and persist the handle FIRST.

    The save happens the moment the job exists and before any waiting, because the handle is the
    only durable reference to a paid job: the 2026-08-25 production runs held it in process memory
    through a fixed one-hour wait, and a job that outlived the wait orphaned its results. The
    tracked-path refusal runs before the submit -- the handle carries the account id, bucket and
    profile, which this public repository must never track, and a refusal after the create call
    would itself orphan the job it just paid for.
    """
    if handle_path.exists():
        handle = BatchJobHandle.load(handle_path)
        _refuse_resumed_handle_mismatch(
            handle, backend, repeated=repeated, metadata=metadata, handle_path=handle_path
        )
        logger.warning(
            "resuming from the saved handle at %s: job %s (submitted %s) will be collected; no "
            "new job is submitted and nothing new is billed",
            handle_path,
            handle.job_name,
            handle.submitted_at,
        )
        return handle
    refuse_tracked_trace_path(handle_path, carries=HANDLE_CARRIES)
    handle = backend.submit(repeated, metadata=metadata)
    handle.save(handle_path)
    return handle


def sample_all_cells_in_one_batch(  # noqa: PLR0913 - keyword-only, the wait knobs defaulted
    backend: BedrockBatchBackend,
    cell_inputs: Mapping[str, CellInputs],
    *,
    samples_per_prompt: int,
    handle_path: Path,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = BATCH_TIMEOUT_SECONDS,
) -> dict[str, list[RawResponse]]:
    """Sample all the run's cells in ONE batch job, then slice the results back per cell.

    One submit rather than one per cell, because a per-cell submit through one backend fails three ways at
    once. The S3 layout and the Bedrock job name are fixed per (run_id, model), so eleven cells would
    overwrite each other's ``input.jsonl`` and the sidecar that exists to make the S3 artefacts
    self-describing, and all of them would ask the service for the same job name -- the collision
    ``_job_name`` was written to avoid. The 100-record floor is per job, so it would apply once per
    cell instead of once to the whole probe. And ``generate_detailed`` blocks until collection,
    so a dozen 10-25 minute waits would serialise. One instance, one submit is the designed contract,
    the shape ``jagged.sweep`` builds, with the per-record cell labels riding in the sidecar metadata.

    Cell membership is positional both ways: the metadata labels each record and the returned slices
    are cut at the same boundaries :func:`~reward_hacking.hatch_probe.repeat_for_samples` lays down,
    which is why both sides call it rather than repeating the arithmetic.

    The job's handle is persisted to ``handle_path`` before any waiting, and a handle already there
    resumes that job instead of submitting a new one (after the digest checks in
    :func:`_refuse_resumed_handle_mismatch`). A timeout raises with the resume route spelled out;
    the job itself is never stopped by one, so losing patience never loses the inference.
    """
    repeated: list[str] = []
    metadata: list[dict[str, object]] = []
    spans: dict[str, tuple[int, int]] = {}
    for label, inputs in cell_inputs.items():
        cell_prompts = repeat_for_samples(inputs.prompts, samples_per_prompt=samples_per_prompt)
        start = len(repeated)
        repeated.extend(cell_prompts)
        spans[label] = (start, len(repeated))
        metadata.extend(
            {
                "cell": label,
                "group_index": index // samples_per_prompt,
                "sample_index": index % samples_per_prompt,
            }
            for index in range(len(cell_prompts))
        )
    _refuse_below_batch_floor(len(repeated), n_cells=len(cell_inputs))
    handle = _submit_or_resume_batch(backend, repeated, metadata, handle_path)
    try:
        completions = backend.collect(
            handle, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds
        )
    except TimeoutError as error:
        raise TimeoutError(
            f"{error} Nothing is lost: the handle is saved at {handle_path}, so re-running this "
            f"command unchanged resumes the SAME job instead of submitting (and paying for) a new "
            f"one. Raise --batch-timeout-seconds to wait longer in one sitting."
        ) from error
    if len(completions) != len(repeated):
        raise RuntimeError(
            f"the batch job returned {len(completions)} completions for {len(repeated)} records; "
            f"cell membership is positional, so a short collection would slice every later cell's "
            f"samples out of its neighbour"
        )
    responses = [raw_response(completion) for completion in completions]
    return {label: responses[start:end] for label, (start, end) in spans.items()}


def _batch_job_record(handle_path: Path, *, resumed: bool) -> dict[str, object]:
    """Describe the run's batch job for the summary.

    Read back from the artifact rather than held in memory: the file is what a resume reads.
    """
    handle = BatchJobHandle.load(handle_path)
    return {
        "job_arn": handle.job_arn,
        "job_name": handle.job_name,
        "submitted_at": handle.submitted_at,
        "handle_path": str(handle_path),
        "resumed": resumed,
    }


def _run_cells(  # noqa: PLR0913 - the run's seams are keyword-only
    cell_inputs: Mapping[str, CellInputs],
    context: ProbeRunContext,
    grader: GraderConfig,
    records_path: Path,
    *,
    sample_cells: Callable[[Sequence[ProbeCell]], Iterator[tuple[int, Sequence[RawResponse]]]],
    cells: Sequence[ProbeCell],
    resumed: Mapping[str, Sequence[HatchProbeRecord]],
) -> tuple[list[HatchProbeRecord], dict[str, float]]:
    """Sample, grade and persist the run's unfinished cells in order, appending records per cell.

    ``sample_cells`` maps the PENDING cells, in order, to an iterator of ``(index into pending,
    responses)``, one whole cell at a time -- the index rather than a count of yields, because the
    live path may skip a cell it could not finish ahead of a raise while still handing over a later
    whole one. It is an iterator over the run rather than a per-cell callable so the live path
    can keep one continuous queue of calls across every cell boundary
    (:func:`~reward_hacking.hatch_probe.generate_cell_responses_in_order`) while this loop still
    grades and persists each cell the moment its last call lands; the batch transport has already
    sampled everything in one job and just hands the slices over in the same order.

    ``resumed`` holds the cells an interrupted attempt finished, read back and checked by
    :func:`~reward_hacking.hatch_probe.load_resumable_cells`. They are not re-bought: their raw
    responses go back through :func:`build_cell_records` under the code running now, exactly like a
    cell sampled this sitting, and the records file is rewritten with the re-derived records before
    any new cell is appended (atomically, temp file then rename, so a death mid-rewrite leaves the
    interrupted attempt's file intact). Folding the stored records in unchanged would let a parser
    or grader edit between the two sittings put two code states' verdicts under one provenance --
    the ``resumed-runs-mix-code-states`` trap -- and the batch resume regrades for the same reason.

    ``cell_seconds`` covers only the cells that ran this sitting, and with one queue of calls across
    every cell it is NOT a cell's sampling time: it is the interval from the previous cell's release
    (or the loop's start) to this cell's release, plus its grading and persistence, so the first
    cell absorbs the queue's warm-up and a slow call in one cell shows up in that cell's number
    while its neighbours' calls ran underneath it. The per-call clocks are on the records.
    """

    def jailed_grade(task_id: str, solution: str | None) -> GradedCompletion:
        return grade_solution(task_id, solution, grader=grader)

    def build(cell: ProbeCell, responses: Sequence[RawResponse]) -> list[HatchProbeRecord]:
        return build_cell_records(
            cell_inputs[cell.label].problems,
            responses,
            cell=cell,
            context=context,
            grade=jailed_grade,
        )

    built: dict[str, list[HatchProbeRecord]] = {}
    if resumed:
        logger.warning(
            "resuming: %d of %d cells are already on disk and are not re-bought: %s; their raw "
            "responses are re-derived and regraded under this code and the records file rewritten",
            len(resumed),
            len(cells),
            sorted(resumed),
        )
        for cell in cells:
            if cell.label in resumed:
                built[cell.label] = build(cell, resumed_cell_responses(resumed[cell.label]))
        _rewrite_records(
            records_path,
            [record for cell in cells if cell.label in built for record in built[cell.label]],
        )
    pending = [cell for cell in cells if cell.label not in resumed]
    cell_seconds: dict[str, float] = {}
    cell_started = time.perf_counter()
    for index, responses in sample_cells(pending):
        cell = pending[index]
        records = build(cell, responses)
        write_trace(records_path, [record.to_json_dict() for record in records], append=True)
        built[cell.label] = records
        cell_seconds[cell.label] = time.perf_counter() - cell_started
        cell_started = time.perf_counter()
        logger.info(
            "cell done, %s",
            f"cell={cell.label} n_records={len(records)} "
            f"n_graded={sum(r.disposition is RecordDisposition.GRADED for r in records)} "
            f"n_reported={sum(r.reported for r in records)} "
            f"n_gave_up={sum(r.gave_up for r in records)} "
            f"seconds={cell_seconds[cell.label]:.0f}",
        )
    all_records = [record for cell in cells for record in built[cell.label]]
    return all_records, cell_seconds


def _rewrite_records(records_path: Path, records: Sequence[HatchProbeRecord]) -> None:
    """Replace the records file with the re-derived resumed records, never leaving it half-written.

    ``write_trace`` truncates then writes, which is fine for the batch resume (its paid material is
    in S3) and not here, where the file IS the paid material: a death between the truncate and the
    last line would lose every completion the interrupted attempt bought. So the re-derived records
    go to a sibling temp file first and ``Path.replace`` swaps it in, which is atomic on one
    filesystem, so the file on disk is always either the interrupted attempt's or the rewrite.
    """
    staging_path = records_path.with_name(records_path.name + ".rewrite")
    write_trace(staging_path, [record.to_json_dict() for record in records])
    staging_path.replace(records_path)
    logger.info("rewrote %d re-derived records to %s", len(records), records_path)


def _resolve_inputs(
    args: argparse.Namespace, cells: Sequence[ProbeCell]
) -> tuple[
    HeldOutPartition,
    dict[str, tuple[ILCBProblem, ...]],
    PromptBudgetFilter,
    BoundedSubset | None,
    tuple[str, ...],
]:
    """Resolve everything the run derives from flags before any backend or jail exists.

    The budget measures across the RUN's cells (so a variant run budgets its own longer
    renderings), while ``dropped_only_by_variant_text`` measures over the pre-budget corpus,
    since the ids in question are exactly the ones the budget just removed.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.budget_model, trust_remote_code=True)
    partition = load_partition(args.partition)
    all_problems = resolve_probe_problems(partition)
    budget_kwargs = resolve_chat_template_kwargs(tokenizer)
    problems_by_split, budget = apply_cross_cell_budget(
        all_problems,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        enable_thinking=DEFAULT_THINKING,
        chat_template_kwargs=budget_kwargs,
        cells=cells,
    )
    variant_dropped = dropped_only_by_variant_text(
        all_problems,
        tokenizer,
        budget,
        enable_thinking=DEFAULT_THINKING,
        chat_template_kwargs=budget_kwargs,
    )
    problems_by_split, subset = _bound_problems(
        problems_by_split, max_problems=args.max_problems, seed=args.seed
    )
    return partition, problems_by_split, budget, subset, tuple(variant_dropped)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the probe's cells end to end and write its two artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    started_at = datetime.now(tz=UTC)
    cells = probe_cells_for(args.hatch_levels)
    partition, problems_by_split, budget, subset, variant_dropped = _resolve_inputs(args, cells)
    if args.dry_run:
        plan = _dry_run_plan(args, problems_by_split, budget, subset, variant_dropped)
        logger.info("dry-run plan: %s", json.dumps(plan, indent=2))
        return 0

    backend, sampler = build_probe_backend(args, args.model)
    prefilled_think = _resolve_prefilled_think(
        args.backend, args.model, thinking=resolve_thinking(args)
    )
    grader = GraderConfig(scratch_root=args.grader_scratch_root)
    jail = assert_jail_usable(timeout_seconds=grader.timeout_seconds)
    grader.scratch_root.mkdir(parents=True, exist_ok=True)
    # One notion of batch-ness from here down: the typed backend the sampling branch needs.
    # build_probe_backend refuses --bedrock-batch on any other kind, so this agrees with the flag.
    batch_backend = backend if isinstance(backend, BedrockBatchBackend) else None
    handle_path = batch_handle_path_for(args.out)
    # A saved handle is an earlier attempt's paid job, so its artifacts are reclaimed, not refused.
    resuming_batch = batch_backend is not None and handle_path.exists()
    # Records with no summary beside them: an interrupted live attempt, resumed cell by cell.
    resuming_live = batch_backend is None and records_path_for(args.out).exists()
    if resuming_batch:
        records_path = reclaim_probe_paths(args.out)
        logger.warning(
            "resuming the batch run whose handle is at %s: the records at %s are reset and every "
            "cell is regraded from the collected results",
            handle_path,
            records_path,
        )
    elif resuming_live:
        records_path = reclaim_probe_paths(args.out)
        logger.warning(
            "resuming the live run whose records are at %s: finished cells are read back, "
            "re-derived and regraded under this code rather than re-sampled, and the rest are "
            "sampled and appended",
            records_path,
        )
    else:
        records_path = reserve_probe_paths(args.out)
        # Reserved up front so a crash mid-run keeps completed cells; no summary marks it
        # incomplete. Fresh runs only: a resume must not reset the interrupted attempt's records
        # until the resume is proven collectable (see the reset before _run_cells below).
        write_trace(records_path, [])
    context = ProbeRunContext(
        samples_per_prompt=args.samples_per_prompt,
        prefilled_think=prefilled_think,
        model_id=args.model,
        transport=backend.transport,
        sampler=sampler,
        grader_workers=grader.workers,
    )
    cell_inputs = cell_inputs_by_label(problems_by_split, cells=cells)
    resumed = (
        load_resumable_cells(records_path, cell_inputs, context=context, cells=cells)
        if resuming_live
        else {}
    )
    batch_sampling_seconds: float | None = None
    batch_job: dict[str, object] | None = None
    if batch_backend is not None:
        poll_seconds, timeout_seconds = _resolve_batch_wait(args)
        batch_started = time.perf_counter()
        batched = sample_all_cells_in_one_batch(
            batch_backend,
            cell_inputs,
            samples_per_prompt=args.samples_per_prompt,
            handle_path=handle_path,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
        batch_sampling_seconds = time.perf_counter() - batch_started
        batch_job = _batch_job_record(handle_path, resumed=resuming_batch)

        def sample_cells(
            pending: Sequence[ProbeCell],
        ) -> Iterator[tuple[int, Sequence[RawResponse]]]:
            return ((index, batched[cell.label]) for index, cell in enumerate(pending))
    else:

        def sample_cells(
            pending: Sequence[ProbeCell],
        ) -> Iterator[tuple[int, Sequence[RawResponse]]]:
            return generate_cell_responses_in_order(
                backend,
                [cell_inputs[cell.label].prompts for cell in pending],
                samples_per_prompt=args.samples_per_prompt,
            )

    if resuming_batch:
        # Deferred past the point where the resume is proven collectable, so a refused resume (a
        # drifted handle, the batch floor, a timed-out collect) leaves the interrupted attempt's
        # records untouched. Reset rather than appended to, or the regrade below would stack onto
        # whatever cells the interrupted attempt had written and double the artifact's denominator.
        write_trace(records_path, [])
    all_records, cell_seconds = _run_cells(
        cell_inputs,
        context,
        grader,
        records_path,
        sample_cells=sample_cells,
        cells=cells,
        resumed=resumed,
    )

    run_hatch_levels = sorted({cell.hatch for cell in cells}, key=HATCH_LEVELS.index)
    summary: dict[str, object] = {
        "schema": PROBE_SCHEMA,
        "kind": PROBE_KIND,
        **git_provenance(),
        "model_id": args.model,
        "backend": args.backend,
        "backend_transport": backend.transport,
        "bedrock_batch": bool(args.bedrock_batch),
        "sampler": dict(sampler),
        "sampling_unseeded": True,
        **elicitation_record(args),
        "prefilled_think": prefilled_think,
        "partition": {
            "path": str(args.partition),
            "pool_fingerprint": partition.pool_fingerprint,
            "n_training_problems": len(partition.training_problem_ids),
        },
        "max_prompt_tokens": args.max_prompt_tokens,
        "budget_model": args.budget_model,
        "prompt_budget": budget.to_json_dict(),
        "dropped_only_by_variant_text": list(variant_dropped),
        "bounded_subset": subset.to_json_dict() if subset else None,
        # The run's levels and their exact prompt text: what a consequence delta is attributed to.
        "hatch_levels": run_hatch_levels,
        "hatch_text_by_level": {level: HATCH_TEXT_BY_LEVEL[level] for level in run_hatch_levels},
        "grader": grader.to_json_dict(),
        "jail_preflight": jail,
        **probe_summary(all_records, samples_per_prompt=args.samples_per_prompt, cells=cells),
        "records_path": str(records_path),
        # Release-to-release intervals, not per-cell sampling time: the live path keeps one queue
        # of calls across every cell, so a cell's number includes the wait for its last call while
        # the next cells' calls were already running. Per-call clocks are on the records.
        "cell_seconds": cell_seconds,
        # The cells an interrupted live attempt finished and this sitting read back, re-derived and
        # regraded rather than sampled; their seconds are not in cell_seconds because they did not
        # run here.
        "resumed_cells": sorted(resumed),
        # On a resumed run this covers only the collect leg; batch_job says when the job was born.
        "batch_sampling_seconds": batch_sampling_seconds,
        "batch_job": batch_job,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    write_probe_summary(args.out, summary)
    log_token_usage(backend)
    logger.info(
        "probe done, %s",
        f"n_records={len(all_records)} out={args.out} records={records_path}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
