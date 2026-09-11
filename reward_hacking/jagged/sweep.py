r"""Sweep an item corpus across a roster of hosted models via Bedrock batch inference.

Two subcommands, deliberately not one. ``submit`` writes each model's records to S3 and creates its
job, saving a handle per model; ``collect`` reads those handles back, waits, downloads, and writes
one trace per model plus the per-dimension table. Splitting them is the whole point: every job runs
concurrently (the quota allows 100 in-progress jobs per base model), an interrupted session resumes
from the handle files rather than re-paying for the inference, and a sweep whose jobs outlast the
session is normal rather than a loss.

    AWS_PROFILE=<your-profile> python -m reward_hacking.jagged.sweep submit --repeats 1 \
        --items <module>:<attribute> --max-new-tokens <cap> \
        --models us.amazon.nova-micro-v1:0 openai.gpt-oss-20b-1:0
    AWS_PROFILE=<your-profile> python -m reward_hacking.jagged.sweep collect \
        --items <module>:<attribute> --handle-dir artifacts/jagged/sweep/<run-id>

``--max-new-tokens`` has no default because the cap is part of the measurement: see the flag's help
text. A submit re-run into a handle directory that already holds a model's handle skips that model
rather than paying for its inference twice, unless that handle was sampled at a different cap or
effort, in which case the submit stops rather than pooling two sampling configs into one table;
``--resubmit`` overrides both, re-sampling at the new config for a second bill.

``--items`` names the corpus rather than this module importing one, because the v1 corpus it was
built for has been deleted and the next one will live somewhere else. It takes the same value at
submit and at collect; the digest check below turns a mismatch into an error rather than a regrade.

The traces this writes carry the JSONL records ``runner.py`` defines, so ``analysis.py`` reads them
with no changes; each record's ``transport`` field says which route produced it.

``collect`` re-renders the prompts locally from the corpus and compares its digest against the one
recorded at submit time, so editing an item between submit and collect is caught rather than
silently mixing an old response with a new item definition.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.bedrock_batch import (
    BATCH_PROFILE_ENV,
    BATCH_ROSTER,
    DEFAULT_POLL_SECONDS,
    DEFAULT_WAIT_SECONDS,
    HANDLE_CARRIES,
    NOVA_MICRO_MAX_TOKENS,
    BatchJobHandle,
    BedrockBatchBackend,
    cell_digest,
    prompt_digest,
)
from reward_hacking.items_reference import resolve_reference
from reward_hacking.jagged.analysis import (
    calibration_warnings,
    format_cells,
    instability_warnings,
    item_calibration_warnings,
    summarise,
)
from reward_hacking.jagged.items import Item, wrong_path_leaks
from reward_hacking.jagged.runner import (
    Cell,
    calls_from_results,
    render_cells,
    trace_records,
)
from reward_hacking.model_backend import BedrockSamplingConfig, raw_response
from reward_hacking.trace import refuse_tracked_trace_path, write_trace

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_SWEEP_ROOT = Path("artifacts/jagged/sweep")

# The cheapest roster member, so a smoke run of the whole path costs a fraction of a cent.
SMOKE_MODEL = "us.amazon.nova-micro-v1:0"

ITEMS_HELP = (
    "the corpus to sweep, as module:attribute naming a sequence of Item "
    "(for example my_local_corpus:ITEMS). Must match between submit and collect"
)


def resolve_items(reference: str) -> tuple[Item, ...]:
    """Import ``module:attribute`` and return the items it names.

    Every failure is loud and specific, because the alternative is a sweep that submits an empty or
    half-typed corpus and bills for it. A non-Item in the sequence is rejected here rather than at
    ``render_cells``, so the failure names the corpus rather than surfacing as a missing attribute
    deep inside rendering. The flag context lives in ``ITEMS_HELP``.
    """
    return resolve_reference(reference, Item)


def _cells(args: argparse.Namespace) -> list[Cell]:
    """Render the cells this run covers, warning loudly about a partial item set.

    Also the one place the corpus is in hand before any job exists, so the advisory wrong-path leak
    check runs here: a marker the task itself prints is legitimate for an item that supplies working
    containing the flawed intermediate result, and reads as a wrong path taken when the model only
    quoted it. Advisory rather than a refusal for that reason, and one aggregated line, because a
    warning that prints a dozen of them is one nobody reads.
    """
    if args.limit is not None and args.limit < 1:
        # Python would read a negative limit as a slice from the end and silently drop the last
        # items instead of rejecting the flag.
        raise ValueError(f"--limit must be at least 1, got {args.limit}")
    corpus = resolve_items(args.items)
    items = corpus if args.limit is None else corpus[: args.limit]
    if args.limit is not None:
        logger.warning("running %d of %d items because --limit was given", len(items), len(corpus))
    leaking = [
        f"{item.id} ({', '.join(leaks)})" for item in items if (leaks := wrong_path_leaks(item))
    ]
    if leaking:
        logger.warning(
            "wrong-path markers appear in the item's own prompt text for %d of %d items: %s. Read "
            "each one deliberately before spending a sweep on it -- a marker the task obliges the "
            "model to restate records a wrong path it only quoted",
            len(leaking),
            len(items),
            "; ".join(leaking),
        )
    return render_cells(items, repeats=args.repeats)


def _metadata(cells: Sequence[Cell]) -> list[dict[str, object]]:
    """Describe each cell for the S3 sidecar and for the cell digest.

    One builder for both, and for both subcommands: the digest is only a tripwire if submit and
    collect describe a cell the same way, and two copies of these four fields would eventually not.
    """
    return [
        {
            "item_id": cell.item.id,
            "dimension": cell.item.dimension,
            "arm": cell.arm.value,
            "repeat": cell.repeat,
        }
        for cell in cells
    ]


def _backend(model_id: str, args: argparse.Namespace, run_id: str) -> BedrockBatchBackend:
    """Build the batch backend for one model, with the run's shared sampling config."""
    return BedrockBatchBackend(
        model_id,
        region=args.region,
        profile=args.profile,
        run_id=run_id,
        sampling=BedrockSamplingConfig(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            reasoning_effort=args.reasoning_effort,
        ),
    )


def _handle_path(handle_dir: Path, model_id: str) -> Path:
    return handle_dir / f"{model_id.replace(':', '-').replace('.', '-').replace('/', '-')}.json"


def _refuse_handles_sampled_differently(
    handle_dir: Path, models: Sequence[str], args: argparse.Namespace
) -> None:
    """Raise before anything is billed if a handle to be reused records another sampling config.

    The skip below reuses paid inference, so it has to check that the inference on disk is the
    inference being asked for. ``--max-new-tokens`` is required precisely because the cap is part of
    the measurement, and the skip is the one path that reaches ``collect`` without honouring it: the
    trace is labelled off the handle, so a re-run at 24576 that skips a model sampled at 2048 writes
    a table whose every row is correctly labelled and whose comparison across rows is meaningless.
    The cap is also the direction that manufactures a fake capability gap, since a truncated reply
    reads as a model that never made the move.

    Checked over every model before the first submit rather than per model in the loop, because the
    operator's fix is to change a flag or point at a different directory; aborting half-way would
    leave that directory holding a partly billed run of the new invocation as well as the old one.

    ``None`` on a handle means "written before the field existed" -- three directories on disk
    predate them -- which is unknown rather than mismatched, so it warns and proceeds. Temperature
    and top-p are not on the handle at all and so cannot be checked here.
    """
    mismatches: list[str] = []
    for model_id in models:
        path = _handle_path(handle_dir, model_id)
        if not path.exists():
            continue
        handle = BatchJobHandle.load(path)
        if handle.sampling_labels_unrecorded:
            logger.warning(
                "%s: the handle at %s records no sampling config (it predates those fields), so "
                "whether its paid inference matches this run's --max-new-tokens %s cannot be "
                "checked; reusing it as it stands",
                model_id,
                path,
                args.max_new_tokens,
            )
            continue
        differing = handle.sampling_label_mismatches(
            max_tokens=args.max_new_tokens, reasoning_effort=args.reasoning_effort
        )
        if differing:
            mismatches.append(f"{model_id} ({'; '.join(differing)})")
    if mismatches:
        raise RuntimeError(
            f"{len(mismatches)} handle(s) in {handle_dir} were sampled with a different config "
            f"than this submit asks for: {'; '.join(mismatches)}. Reusing them would pool two "
            "sampling configs into one table with every row correctly labelled and the comparison "
            "across rows meaningless. Submit into a fresh --handle-dir to keep the runs apart, "
            "pass --resubmit to re-sample these models at the new config (a second bill), or "
            "re-run with the flags the handles record."
        )


def submit(args: argparse.Namespace) -> int:
    """Submit one batch job per model and save a handle for each.

    Every model is submitted before anything is waited on, because concurrency across models is
    the only reason this path beats the live Converse backend. A model that fails to submit is
    reported and does not stop the rest -- but the failures are raised at the end, so a partly
    submitted sweep cannot be mistaken for a whole one.

    A model whose handle is already in ``--handle-dir`` is skipped, which is what makes re-running
    after such a partial failure safe. The failure above sends the operator back to the same
    directory, the handle path is a pure function of (directory, model id), and ``save`` overwrites,
    so without the skip a re-run would create a second billable job per already-successful model and
    overwrite the handle that was paid for -- leaving the first job's ARN recoverable only from the
    log. ``--resubmit`` asks for that overwrite deliberately.

    Reusing paid inference is only safe if it is the inference being asked for, so a handle whose
    recorded cap or effort disagrees with this invocation stops the whole submit before anything is
    billed -- see :func:`_refuse_handles_sampled_differently`.
    """
    run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    handle_dir = args.handle_dir or DEFAULT_SWEEP_ROOT / run_id
    # Unconditionally, before any job exists: --resubmit must not skip the privacy guard, and the
    # writer's own refusal fires only at save time -- inside the per-model except, after the bill.
    refuse_tracked_trace_path(handle_dir, carries=HANDLE_CARRIES)
    if not args.resubmit:
        _refuse_handles_sampled_differently(handle_dir, args.models, args)
    cells = _cells(args)
    prompts = [cell.prompt for cell in cells]
    metadata = _metadata(cells)

    logger.info(
        "sweep submit | run_id=%s models=%d records/model=%d handles=%s",
        run_id,
        len(args.models),
        len(prompts),
        handle_dir,
    )
    # Lazily, via importlib, for the reason the backends do it: botocore is the optional bedrock
    # extra, and this keeps importing the module free of it. ClientError covers the AWS side
    # (throttling, a per-model access denial); without it the loop would abort on the first model's
    # transport hiccup, which is not what this promises.
    client_error = importlib.import_module("botocore.exceptions").ClientError

    failures: dict[str, Exception] = {}
    reused: list[str] = []
    for model_id in args.models:
        handle_path = _handle_path(handle_dir, model_id)
        if handle_path.exists() and not args.resubmit:
            logger.info(
                "%s already has a handle at %s, so its inference is already paid for; collect will "
                "pick it up as it stands. Pass --resubmit to submit a second job instead",
                model_id,
                handle_path,
            )
            reused.append(model_id)
            continue
        try:
            handle = _backend(model_id, args, run_id).submit(prompts, metadata=metadata)
            # Inside the try on purpose. The job is submitted and billable by this point, so a
            # filesystem error here must be reported against this model rather than aborting the
            # remaining submits -- and the ARN needed to recover it is already in the log above.
            handle.save(handle_path)
        except (ValueError, RuntimeError, OSError, client_error) as error:
            logger.exception("submit failed for %s", model_id)
            failures[model_id] = error

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(args.models)} models failed to submit "
            f"({', '.join(sorted(failures))}); the rest have handles in {handle_dir}"
        )
    reused_note = (
        f", reused {len(reused)} existing handle(s): {', '.join(reused)}" if reused else ""
    )
    sys.stdout.write(
        f"submitted {len(args.models) - len(reused)} job(s), {len(prompts)} records each"
        f"{reused_note}.\n"
        f"collect with: python -m reward_hacking.jagged.sweep collect --handle-dir {handle_dir}\n"
    )
    return 0


def collect(args: argparse.Namespace) -> int:
    """Collect every saved handle into a trace, then print the pooled table.

    The digest check is what makes collecting in a later session safe. The cells are re-rendered
    here from the current corpus, and if the corpus moved since submit, the recorded digest will not
    match and the run stops -- rather than grading a stale response against an edited item.

    The sampling labels on the records are read off the handle rather than from this process's
    flags, and ``collect`` deliberately takes no sampling flags at all: a hand-restated cap is how a
    trace comes to misdescribe itself, measured on the sibling benchmark where a backend left at
    2048 produced records stamped 30000.
    """
    handle_paths = sorted(args.handle_dir.glob("*.json"))
    if not handle_paths:
        raise FileNotFoundError(f"no handle files in {args.handle_dir}")

    cells = _cells(args)
    local_prompts = prompt_digest([cell.prompt for cell in cells])
    local_cells = cell_digest(_metadata(cells))
    all_records: list[dict[str, object]] = []
    for path in handle_paths:
        handle = BatchJobHandle.load(path)
        for label, submitted, local in (
            ("prompt", handle.prompt_digest, local_prompts),
            ("cell", handle.cell_digest, local_cells),
        ):
            if submitted != local:
                raise RuntimeError(
                    f"{path} was submitted with {label} digest {submitted} but the corpus now "
                    f"renders {local} for {len(cells)} cells. Either the items, arms or "
                    "--repeats/--limit differ from the submitted run; re-run collect with the same "
                    "flags, or check out the commit the sweep was submitted from. A cell-digest "
                    "mismatch with a matching prompt digest means the prompts are unchanged but "
                    "their labels moved, which would silently regrade responses as other items"
                )
        # The run id only shapes the S3 paths a submit writes to; collect reads them off the handle.
        backend = BedrockBatchBackend(handle.model_id, region=handle.region, profile=handle.profile)
        completions = backend.collect(
            handle, poll_seconds=args.poll_seconds, timeout_seconds=args.timeout_seconds
        )
        calls = calls_from_results(
            cells,
            [raw_response(completion) for completion in completions],
            started_at=handle.submitted_at,
            completed_at=datetime.now(UTC).isoformat(),
        )
        records = trace_records(
            calls,
            handle.model_id,
            transport=backend.transport,
            max_tokens=handle.max_tokens,
            reasoning_effort=handle.reasoning_effort,
        )
        write_trace(args.handle_dir / f"{path.stem}.trace.jsonl", records)
        all_records.extend(records)

    cells_summary = summarise(all_records)
    sys.stdout.write(format_cells(cells_summary) + "\n")
    for warning in calibration_warnings(cells_summary):
        logger.warning("%s", warning)
    for warning in item_calibration_warnings(all_records):
        logger.warning("%s", warning)
    for warning in instability_warnings(cells_summary):
        logger.warning("%s", warning)
    return 0


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--items", required=True, help=ITEMS_HELP)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N items")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "sample every (item, arm) cell N times. Items times five arms times repeats has to "
            "clear the hard 100-record batch minimum, so a small corpus needs repeats to reach it"
        ),
    )
    parser.add_argument(
        "--run-id", default=None, help="reuse a run id instead of deriving one from the clock"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the submit/collect subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # allow_abbrev=False, or `--res` becomes a synonym for --resubmit's second bill (verified).
    submit_parser = sub.add_parser(
        "submit", help="create one batch job per model", allow_abbrev=False
    )
    _add_shared_args(submit_parser)
    submit_parser.add_argument(
        "--models",
        nargs="+",
        default=[SMOKE_MODEL],
        choices=[model.model_id for model in BATCH_ROSTER],
        help="roster model ids to sweep; defaults to the cheapest one, for a smoke run",
    )
    submit_parser.add_argument("--handle-dir", type=Path, default=None)
    submit_parser.add_argument("--region", default="us-west-2")
    submit_parser.add_argument(
        "--profile",
        default=None,
        help=(
            f"AWS profile. Defaults to ${BATCH_PROFILE_ENV}, which has to name a principal that "
            "holds iam:PassRole on the Bedrock service role -- creating a batch job passes that "
            "role to the service, and the PowerUser profile the live Converse path uses cannot"
        ),
    )
    submit_parser.add_argument(
        "--max-new-tokens",
        type=int,
        required=True,
        help=(
            "output-token cap, per record. Required and never defaulted, because the cap is part "
            "of the measurement: below a reasoning model's thinking length it truncates the reply "
            "before any marker can appear, which reads as a model that did not make the move. The "
            "traces already on disk show that happening at 2048. Bracketing figures: Nova Micro "
            f"hard-rejects anything above {NOVA_MICRO_MAX_TOKENS}, and the sibling benchmark "
            "measured the small reasoning models on this roster needing about 24576 before their "
            "answers stopped being cut off"
        ),
    )
    submit_parser.add_argument(
        "--resubmit",
        action="store_true",
        help=(
            "submit a model that already has a handle in --handle-dir, overwriting it. Off by "
            "default: the handle means the inference is already paid for, and a second job for it "
            "is a second bill plus an orphaned first job"
        ),
    )
    submit_parser.add_argument("--temperature", type=float, default=None)
    submit_parser.add_argument("--top-p", type=float, default=None)
    submit_parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "left unset by default: the field is documented but unverified on the batch path, and "
            "GPT-OSS ignores unknown request fields without error"
        ),
    )

    collect_parser = sub.add_parser("collect", help="wait for saved handles and write traces")
    _add_shared_args(collect_parser)
    collect_parser.add_argument("--handle-dir", type=Path, required=True)
    collect_parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    collect_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_WAIT_SECONDS,
        help="give up waiting after this long; the job is never stopped and stays collectable",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the submit or collect handler."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = parse_args(argv)
    if args.command == "submit":
        return submit(args)
    return collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
