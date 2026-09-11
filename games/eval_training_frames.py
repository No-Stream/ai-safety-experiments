"""Evaluate a checkpoint ladder on its own TRAINING corpus prompts, battery-style.

The eval battery's game-behavior section renders held-out frames only (`games.prompts`,
`SPLIT_EVAL`) -- by design, every battery row is a frame the arm never trained on. That leaves one
cell nobody can read off the existing artifacts: the merged checkpoints the battery evaluates,
asked the exact prompts training optimised, under an eval-time sampler. The twin-pd-group
dissociation is why that cell matters: training-time cooperation on its own prompts fell ~0.21
over 70 steps while the battery's held-out frames sat flat, and one candidate explanation is that
the learned behaviour is specific to the 15 training reskins the battery never asks
(`docs/scratch/games-readout-notes-2026-08-20/12-transfer-controls.md` section 10). This CLI
produces the missing cell and nothing else.

The prompt corpus is the run's own selection artifact (the JSONL `run_config.json` records as
`config.corpus_path`), which carries the rendered prompt text plus the label/reskin/payoff fields
parsing needs -- the same rows `games.train` fed the policy, so "the training prompts" is a file
identity rather than a reconstruction. Records reuse `games.evals._game_record` and land under a
distinct ``record: "training-frames"`` type so they can never silently pool with the battery's
game-behavior rows (which mean *held-out* frames everywhere else in this repo).

The cell IS the battery's run body (`games.evals.run_eval_battery`, handed this cell's plan in
place of the section planners' since 2026-09-03), so it carries the battery's persistence contract
(2026-09-02): every record is appended and flushed as its completion lands -- per finished sequence
on the pooled vLLM drain, per call group (one corpus reskin) on every other transport -- so a death
loses at most one call group. A trace WITHOUT a summary is a cell that died mid-run, and
relaunching the identical command continues it: its finished records are kept byte for byte, only
the planned identities without a record are generated, and the meta's ``resume`` block names every
session and its commit. A trace WITH a summary is complete and is never overwritten. The summary is
a pure function of the trace (`games.evals.summarise_frames_trace`, reached through
`games.evals.rebuild_summary`, which reads the trace's kind off its meta), so a cell that died after
its last generate call is closed out without an engine, on the box or via ``--summarise-only`` from
a copy; ``--sync-dest`` restores the cell's directory from S3 before starting and syncs it back on
an interval while records land, exactly as `games.run_evals` does. The resume and the CLI close-out
both hold for traces written from 2026-09-02 on: a trace the earlier writer left without a summary
carries no ``eval_config`` and is refused as another cell's. That writer wrote every record in one
pass and the summary right after, so such a file is either complete (it died in the gap) or torn
mid-write; its close-out is `games.evals.rebuild_summary` from Python, after checking that it holds
``n_corpus_rows * samples_per_prompt`` records. The rebuild of a legacy trace WITH its summary is
what was verified byte for byte.

Target resolution, serving (`games.eval_model.resolve_served_model`: the un-merged runtime
adapter where the backend can hold one, a merge only where it cannot), sampler metadata, the
existing-trace checks, the summary write, the S3 sync and the mock-smoke path all reuse
`games.run_evals`; the run body, the resume, the close-out and the summary shape live in
`games.evals`, which registers this record kind beside the battery's sections. Only what differs
from the battery lives here: which prompts, and how they are keyed. A pooled submission's
admission order (`games.evals.ADMISSIONS`) is not exposed on this CLI: with one record kind in the
plan, every order is plan order.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.eval_model import resolve_served_model, verify_served_model
from games.eval_sampler import (
    add_sampler_arg,
    eval_sampling,
    resolve_sampler_mode,
    sampler_mode_meta,
)

# `_game_record` is a private import on purpose: a frames record IS a battery game record restamped,
# and a copy of its parser here is how the two surfaces would come to disagree about a field.
from games.evals import (
    COUNTERPART_FRAMING_FIELD,
    RECORD_TRAINING_FRAMES,
    SECTIONS_TRAINING_FRAMES,
    SUBMISSION_POOLED,
    SUBMISSION_SERIAL,
    SUBMISSIONS,
    EvalConfig,
    PlannedRequest,
    _game_record,  # pyright: ignore[reportPrivateUsage]
    finish_complete_trace,
    inspect_trace,
    record_identity,
    run_eval_battery,
    salvage_summary,
)
from games.rewards import FRAMING_ID_COLUMN
from games.run_evals import (
    DEFAULT_EVAL_ROOT,
    DEFAULT_MERGE_ROOT,
    DEFAULT_SYNC_INTERVAL_SECONDS,
    MOCK_PLUMBING_RESPONSES,
    MOCK_TRACE_SUBDIR,
    TRAINED_GAMES_FROM_CORPUS_ROWS,
    TRAINED_GAMES_NONE_UNADAPTED,
    EvalPlan,
    EvalTarget,
    IntervalSync,
    _bridge_hf_decode_kernel,  # pyright: ignore[reportPrivateUsage]
    _cell_identity,  # pyright: ignore[reportPrivateUsage]
    _check_existing_traces,  # pyright: ignore[reportPrivateUsage]
    _has_complete_summary,  # pyright: ignore[reportPrivateUsage]
    _read_run_facts,  # pyright: ignore[reportPrivateUsage]
    _resolve_arm,  # pyright: ignore[reportPrivateUsage]
    _resolve_template_facts,  # pyright: ignore[reportPrivateUsage]
    _resolve_thinking,  # pyright: ignore[reportPrivateUsage]
    _restore_missing_files,  # pyright: ignore[reportPrivateUsage]
    _run_dir_specs,  # pyright: ignore[reportPrivateUsage]
    _sampling_meta,  # pyright: ignore[reportPrivateUsage]
    _write_summary,  # pyright: ignore[reportPrivateUsage]
    summary_path_for,
)
from games.vllm_teardown import baseline_before_engine, release_engine
from reward_hacking import backend_cli

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from games.eval_model import ServedModel
    from games.run_evals import RunFacts

logger = logging.getLogger(__name__)

REQUIRED_CORPUS_FIELDS: frozenset[str] = frozenset(
    {
        "prompt",
        "prompt_id",
        "game_id",
        "grading",
        "reskin_id",
        "payoff_variant",
        "label_a",
        "label_b",
        "coop_label",
        "label_print_order",
        "endowment",
        "windfall",
        "n_rounds",
        "n_levels",
    }
)
"""Every field one corpus row must carry for `_game_record` to parse and label its completions.

`endowment`, `windfall`, `n_rounds` and `n_levels` are read only for the dictator,
simultaneous-claim, repeated and minimum-effort games respectively, but the selection artifact
writes the full schema on every row, so requiring them costs nothing and catches a hand-edited
corpus early. `label_print_order` is required for the sharper version of the same reason: the
record writer stamps it on every record now that the battery renders both orders, so a corpus
without it would raise mid-generation with a bare KeyError after the GPU time was spent, rather
than here with a message naming the file.
"""


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI: the corpus, the run to evaluate, the trace destination, backend knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help=(
            "The training-corpus JSONL the run's run_config.json records as config.corpus_path "
            "(a games.select_prompts artifact). Its rows are the prompts evaluated here."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Training run directory; each checkpoint is served the way games.eval_model "
            "decides -- the un-merged runtime adapter where the backend can, a merge where it "
            "cannot."
        ),
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            "Comma-separated steps to evaluate; 0 means the un-adapted base model. "
            "Default: every checkpoint present."
        ),
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help=(
            "Completions sampled per corpus prompt (default: 1). The corpus is small (one arm's "
            "selection, tens of rows), so this is the denominator lever."
        ),
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Arm label for the trace meta; derived from the run's run_config.json when omitted.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model override for merging and step 0; defaults to what the run recorded.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"Trace directory (default: {DEFAULT_EVAL_ROOT}/<arm>-training-frames, plus a "
            f"/{MOCK_TRACE_SUBDIR} level under --backend mock). Traces are step-<step>.jsonl; a "
            f"trace with a summary beside it is complete and is refused, never overwritten, while a "
            f"trace without one is a cell that died mid-run and is resumed."
        ),
    )
    parser.add_argument(
        "--submission",
        choices=SUBMISSIONS,
        default=SUBMISSION_POOLED,
        help=(
            f"How prompts reach the backend (default: {SUBMISSION_POOLED}). {SUBMISSION_POOLED!r} "
            f"submits every corpus prompt at once and files each record as its completion lands, so "
            f"the engine never idles; {SUBMISSION_SERIAL!r} is one call per corpus reskin, appended "
            f"per call. Same sampler, same distribution either way; not the same bytes. Pooling is a "
            f"vLLM mechanism, so every other transport runs the serial calls under either name."
        ),
    )
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help=(
            "Write the summary of an already-complete trace at each target's out path and load no "
            "engine: the salvage for a cell that died after its last generate call and before its "
            "summary write. The base model's tokenizer is still read (from the hub or its cache) "
            "to derive the chat-template facts the cell identity includes; nothing else loads. "
            "Refuses an incomplete trace (relaunch the cell's own command instead, which resumes "
            "it) and, without --sync-dest, a trace that already has a summary; with --sync-dest a "
            "complete cell of this configuration is skipped and one of any other refused."
        ),
    )
    parser.add_argument(
        "--sync-dest",
        default=None,
        help=(
            "s3:// prefix mirroring --out-dir. Restored before anything runs, adding only the files "
            "--out-dir lacks (a local trace is never replaced by an older upload of itself), so a "
            "relaunch on a fresh box continues the cell it finds there; synced back after the first "
            "records land and then every --sync-interval-seconds while a cell runs, so a box death "
            "loses at most one interval of finished records. A sync failure logs and the cell "
            "continues; a restore failure refuses to start."
        ),
    )
    parser.add_argument(
        "--sync-interval-seconds",
        type=float,
        default=DEFAULT_SYNC_INTERVAL_SECONDS,
        help=f"Minimum seconds between two --sync-dest uploads (default: {DEFAULT_SYNC_INTERVAL_SECONDS:g}).",
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        default=DEFAULT_MERGE_ROOT,
        help=(
            f"Staging area for merged checkpoints (default: {DEFAULT_MERGE_ROOT}). Each merge is "
            "deleted after its step; a crash leaves it behind for debugging."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            f"Sequences per generation call on the {SUBMISSION_SERIAL!r} call sequence (every "
            f"non-vLLM transport, or --submission {SUBMISSION_SERIAL}); default derives from the "
            f"VRAM actually present. The {SUBMISSION_POOLED!r} vLLM submission hands the engine "
            f"every prompt at once and sizes nothing by it."
        ),
    )
    parser.add_argument(
        "--prefilled-think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether completions carry only the closing </think> tag. Default: measured from the "
            "model's chat template, which is the only source that tracks the checkpoint."
        ),
    )
    backend_cli.add_backend_args(parser)
    add_sampler_arg(parser)
    return parser.parse_args(argv)


def load_corpus_rows(path: Path) -> list[dict[str, Any]]:
    """Read the selection-artifact JSONL, refusing rows that could not be parsed or labelled.

    The refusals are cheap and each one blocks a silent mislabel: a row missing `label_a` would
    make `_game_record` raise mid-generation after GPU time was spent. Two rows sharing a
    `(prompt_id, label_print_order)` are refused for the resume's sake: that pair plus the draw
    index is a record's identity, so two rows under one identity would file their draws as
    duplicates that `inspect_trace` refuses to continue and every rate double-weights.

    Which games the file may hold is NOT decided here, because it is not a property of the file: it is
    the game set of the arm that trained it, and a breadth arm's set is several games. That check is
    `assert_corpus_games`, called with the set by whoever knows it -- `_refuse_corpus_from_another_game`
    from the run's own record, and `games.train.load_corpus` from the registry when
    `games.corpus_preflight` runs the trainer's loader over the same file.
    """
    if not path.is_file():
        raise FileNotFoundError(f"--corpus {path} does not exist.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"--corpus {path} contains no rows; there is nothing to evaluate.")
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_CORPUS_FIELDS - set(row))
        if missing:
            raise ValueError(
                f"--corpus {path} row {index} is missing {missing}; this does not look like a "
                f"games.select_prompts corpus artifact."
            )
    keys = [(str(row["prompt_id"]), str(row["label_print_order"])) for row in rows]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ValueError(
            f"--corpus {path} carries {len(repeated)} (prompt_id, label_print_order) pair(s) more "
            f"than once, e.g. {repeated[0]}; every record of this cell is keyed on that pair plus "
            f"its draw index, so repeated rows would file duplicate identities."
        )
    return rows


def assert_corpus_games(
    rows: Sequence[Mapping[str, Any]], *, path: Path, game_ids: Sequence[str]
) -> None:
    """Refuse a corpus holding a game outside the set the caller says this run may train.

    `game_ids` is that set: an arm's own game plus the extra games a breadth arm's corpus may carry
    (`games.arms.arm_game_ids`). Passing it empty means the caller knows of no set, and the rule is
    then the one every corpus obeyed before wave 4b -- exactly one game, whichever the file's own rows
    name -- because a caller with nothing to check against should still catch a file concatenated from
    two corpora rather than wave it through.

    Two messages for one refusal, because the reader's next move differs: a file holding several games
    where the run trains fewer is a rebuild or a concatenation, while a file holding one game that is
    not the run's is the wrong file entirely.
    """
    corpus_games = sorted({str(row["game_id"]) for row in rows})
    allowed = tuple(game_ids) or (corpus_games[0],)
    foreign = [game_id for game_id in corpus_games if game_id not in allowed]
    if not foreign:
        return
    if len(corpus_games) > 1:
        raise ValueError(
            f"--corpus {path} mixes games {corpus_games} where this run trains {list(allowed)}; a "
            f"corpus of several games belongs to an arm that names them all "
            f"(games.arms.GameArm.game_ids), so this file is not the corpus a run trained on."
        )
    raise ValueError(
        f"--corpus {path} is a {corpus_games[0]!r} corpus but this run trains {list(allowed)}; "
        f"evaluating an arm on another game's training prompts would label the trace with frames "
        f"the policy never saw."
    )


def _refuse_corpus_from_another_game(
    rows: Sequence[dict[str, Any]], facts: RunFacts, *, path: Path
) -> None:
    """Refuse a corpus whose games are not the ones the run's own config says it trained on.

    The set comes from the run's record rather than from the registry, because the registry can be
    edited after a run: what this cell reads back has to be checked against what that run allowed. A
    record naming no game at all leaves the set empty, which is the pre-wave-4b one-game rule.
    """
    recorded = () if facts.game_id is None else (str(facts.game_id), *facts.game_ids)
    assert_corpus_games(rows, path=path, game_ids=recorded)


def _corpus_sha256(path: Path) -> str:
    """Hash the corpus bytes, so the trace names exactly which prompt file was asked."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_frame_record(  # noqa: PLR0913 - one record's rendering context, bound once per request
    game_id: str,
    row: Mapping[str, Any],
    completion: str,
    *,
    prefilled_think: bool,
    trained_game: bool,
    sample_index: int,
) -> dict[str, Any]:
    """Parse one completion into this cell's record: a battery game record, restamped.

    The record type and the split marker are what distinguish this surface from the battery's
    held-out rows. The sample index is the shared writer's own field rather than one stamped here,
    since the battery samples per prompt too and two writers of one field is how the two surfaces
    would come to disagree about it.

    A row that carries a counterpart framing has it stamped under the battery's own field name, so the
    framing sweep's reducer and this cell's describe one framing the same way. Absent where the corpus
    has no framing column, which is every corpus before wave 4b.
    """
    record = _game_record(
        game_id,
        row,
        completion,
        prefilled_think=prefilled_think,
        trained_game=trained_game,
        eval_only_game=False,
        sample_index=sample_index,
    )
    record["record"] = RECORD_TRAINING_FRAMES
    record["frame_split"] = "train"
    framing_id = row.get(FRAMING_ID_COLUMN)
    if framing_id:
        record[COUNTERPART_FRAMING_FIELD] = str(framing_id)
    return record


def plan_training_frames(
    rows: Sequence[Mapping[str, Any]], *, config: EvalConfig, trained_game: bool
) -> list[PlannedRequest]:
    """Plan `config.game_behavior_samples` draws of every corpus prompt, each keyed for resume.

    Prompts are repeated row-major (row 0's draws, then row 1's), matching how every repeated
    section of the battery expresses a sample count: the `Backend` protocol has no sample count, so
    N samples means the prompt appears N times in the list. Each request's identity is
    `record_identity` of the record its parser will produce, so the plan and the records on disk
    agree by construction and `PlannedRequest.record` has nothing to disagree about.

    The serial call group is one corpus reskin. It has to be smaller than the corpus, because a call
    group is the unit the serial path appends and flushes at, so it bounds what a death loses on a
    transport with no engine to drain (the HF path; vLLM drains per finished sequence whatever the
    grouping). A reskin's rows share their prose and so their length, which keeps padding tight
    inside a group, and one reskin is tens of rows times the draw count -- wide enough to fill the
    HF path's derived chunk width rather than starve it, the way per-row groups would.

    Every game id, identity and call group is read off the ROW rather than off row 0. A breadth arm's
    corpus holds several games, and one id lifted from the first row would relabel all the others:
    the records would parse, the rates would compute, and the trace would say the whole corpus was
    whatever game the first row happened to be.
    """
    requests: list[PlannedRequest] = []
    for row in rows:
        game_id = str(row["game_id"])
        call_group = f"{game_id}:{row['reskin_id']}"
        for sample_index in range(config.game_behavior_samples):
            identity = record_identity(
                {
                    "record": RECORD_TRAINING_FRAMES,
                    "game_id": game_id,
                    "prompt_id": row["prompt_id"],
                    "label_print_order": row["label_print_order"],
                    "sample_index": sample_index,
                }
            )
            requests.append(
                PlannedRequest(
                    section=RECORD_TRAINING_FRAMES,
                    identity=identity,
                    prompt=str(row["prompt"]),
                    call_group=call_group,
                    parse=functools.partial(
                        _training_frame_record,
                        game_id,
                        row,
                        prefilled_think=config.prefilled_think,
                        trained_game=trained_game,
                        sample_index=sample_index,
                    ),
                )
            )
    return requests


def _corpus_identity(
    args: argparse.Namespace, rows: Sequence[Mapping[str, Any]], corpus_sha256: str
) -> dict[str, Any]:
    """Name the corpus half of a frames cell's identity, known before any tokenizer or engine loads.

    Which bytes were asked, how many rows they held and how many draws per row: a trace at the out
    path that disagrees on any of these answers different prompts under this cell's name, and the
    shared up-front check (`_check_existing_traces`, handed these as ``extra_identity``) refuses it
    -- partial or complete -- before an earlier step spends its GPU time.
    """
    return {
        "corpus_sha256": corpus_sha256,
        "n_corpus_rows": len(rows),
        "samples_per_prompt": int(args.samples_per_prompt),
    }


def _frames_cell_identity(
    plan: EvalPlan, target: EvalTarget, *, config: EvalConfig, corpus: Mapping[str, Any]
) -> dict[str, Any]:
    """Name the meta fields an existing trace must agree with before an engine loads for it.

    The battery's own identity (`games.run_evals._cell_identity`: arm, step, thinking, base model,
    sections, eval config) plus the corpus half above and the measured `prefilled_think`, which
    this cell has always written at the top level of its meta. The serving-level fields (backend
    kind, sampling, load mode) are compared by `run_eval_battery` against the full meta once the
    backend is resolved -- and not at all under --summarise-only, which resolves no backend.
    """
    return {
        **_cell_identity(plan, target, sections=SECTIONS_TRAINING_FRAMES, config=config),
        **corpus,
        "prefilled_think": config.prefilled_think,
    }


def _frames_meta(  # noqa: PLR0913 - one trace's provenance, mirroring run_evals' meta shape
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    target: EvalTarget,
    served: ServedModel,
    corpus: Mapping[str, Any],
    config: EvalConfig,
) -> dict[str, Any]:
    """Assemble the caller-owned meta fields; `games.evals._meta_record` adds sha, timestamp, config and resume."""
    return {
        "arm": plan.arm,
        "step": target.step,
        "thinking": plan.thinking,
        "base_model_id": target.base_model,
        "checkpoint": None if target.checkpoint is None else str(target.checkpoint),
        "run_dir": None if plan.run_dir is None else str(plan.run_dir),
        "backend_kind": args.backend,
        "load_mode": served.load_mode,
        "vllm_quantization": args.vllm_quantization,
        "sampler_mode": sampler_mode_meta(args),
        "sampling": _sampling_meta(args, thinking=plan.thinking),
        "grading": plan.grading,
        # No `trained_games_source` beside these, unlike the battery's meta: this cell was handed the
        # corpus, so `corpus_path` and `corpus_sha256` below name the source by file and hash already.
        "corpus_path": str(args.corpus),
        **corpus,
        "prefilled_think": config.prefilled_think,
    }


def _finish_without_engine(
    target: EvalTarget, *, frames_plan: Sequence[PlannedRequest], expected_meta: Mapping[str, Any]
) -> bool:
    """Write the summary of an already-complete trace at the target, or say it needs generating.

    Returns True when the trace was complete and its summary is now written, False when records are
    missing and the engine has to load. Either way the trace has been checked to be this cell's.
    """
    inspection = inspect_trace(target.out_path, frames_plan, expected_meta=expected_meta)
    pending = inspection.pending(frames_plan)
    if pending:
        logger.info(
            f"step {target.step}: resuming {target.out_path} with {len(inspection.done)} of "
            f"{len(frames_plan)} records on disk; {len(pending)} left to generate"
        )
        return False
    logger.info(
        f"step {target.step}: {target.out_path} is complete ({len(frames_plan)} records) and only "
        f"lacked its summary; writing it without loading an engine"
    )
    _write_summary(target, finish_complete_trace(target.out_path, inspection, frames_plan))
    return True


def _evaluate_target(  # noqa: PLR0913 - the same seam run_evals' _evaluate_target carries
    target: EvalTarget,
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    rows: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    config: EvalConfig,
    sync: IntervalSync | None,
) -> None:
    """Resolve how the weights are served, sample or resume the corpus, and write this step's artifacts.

    Serving goes through `games.eval_model.resolve_served_model`, exactly as `games.run_evals`
    serves the battery: the un-merged runtime adapter where the backend can hold one, a merge
    only where it cannot. This file used to merge unconditionally, which contradicted the owner's
    runtime-LoRA-for-eval ruling (a bf16 merge loses part of the delta) and made this cell
    non-comparable with the battery cells evaluated beside it.
    """
    if _has_complete_summary(target.out_path):
        # Checked at plan time too; appearing since means another process finished this cell here.
        raise FileExistsError(
            f"{summary_path_for(target.out_path)} appeared mid-run; refusing to redo a complete cell."
        )
    # Step 0 carries no adapter, so its rows must not claim trained frames.
    frames_plan = plan_training_frames(
        rows, config=config, trained_game=target.checkpoint is not None
    )
    resume = target.out_path.exists()
    if resume and _finish_without_engine(
        target,
        frames_plan=frames_plan,
        expected_meta=_frames_cell_identity(plan, target, config=config, corpus=corpus),
    ):
        if sync is not None:
            sync.sync_now(reason=f"step {target.step} summarised")
        return

    label = "base" if target.checkpoint is None else target.checkpoint.parent.name
    served = resolve_served_model(
        checkpoint=target.checkpoint,
        base_model=target.base_model,
        backend_kind=args.backend,
        merge_root=cast("Path", args.merge_root),
        merge_label=f"{label}-step-{target.step}-",
    )
    logger.info(f"step {target.step} serving {served.load_mode}: {served.provenance}")

    model_id = f"mock:{served.model_id}" if args.backend == "mock" else served.model_id
    # Read before the engine exists, per games.vllm_teardown.baseline_before_engine, and None for
    # every kind that loads no engine.
    baseline_mib = baseline_before_engine(args.backend)
    backend = backend_cli.backend_from_args(
        args,
        model_id,
        local_sampling=eval_sampling(resolve_sampler_mode(args), thinking=plan.thinking),
        mock_responses=MOCK_PLUMBING_RESPONSES,
        extra_kwargs=served.backend_kwargs,
    )
    try:
        # Before the corpus is sampled: an un-merged adapter that failed to apply serves base
        # weights without saying so, and the whole cell would read as a run that changed nothing.
        verify_served_model(backend, served)
        # The battery's own run body over this cell's plan; the summary comes back in this kind's shape.
        summary = run_eval_battery(
            backend,
            sections=SECTIONS_TRAINING_FRAMES,
            plan=frames_plan,
            out_path=target.out_path,
            meta=_frames_meta(
                args=args, plan=plan, target=target, served=served, corpus=corpus, config=config
            ),
            config=config,
            submission=args.submission,
            resume=resume,
            on_records_written=None if sync is None else sync.maybe_sync,
        )
        _write_summary(target, summary)
        if sync is not None:
            sync.sync_now(reason=f"step {target.step} complete")
    finally:
        # One process walks the whole checkpoint ladder, so an engine left up after a failed step
        # holds the card against every step after it -- hence a `finally` rather than the success
        # path, and vLLM's own teardown rather than a `del` its EngineCore subprocess cannot hear.
        if baseline_mib is None:
            # The `del` has to stay at the call site: run from inside a helper it would unbind only
            # that helper's name, which is the refcount bug games/vllm_teardown.py was written over.
            del backend
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            release_engine(backend, baseline_mib=baseline_mib)
    # Kept on a crash as the evidence, so this stays off the `finally`: see games.eval_model.
    if served.merged_dir is not None:
        shutil.rmtree(served.merged_dir)


def _default_out_dir(arm: str, backend_kind: str) -> Path:
    """Per-arm default, suffixed so these traces never share a directory with battery ones.

    The suffix and the mock level exist for the same reason as run_evals' `_default_out_dir`: a
    trace directory is an identity, and both a mock smoke and a training-frames pass landing in
    the arm's battery directory would block or contaminate the real ladder.
    """
    out_dir = DEFAULT_EVAL_ROOT / f"{arm}-training-frames"
    if backend_kind == "mock":
        return out_dir / MOCK_TRACE_SUBDIR
    return out_dir


def _resolve_plan(
    args: argparse.Namespace, *, rows: Sequence[Mapping[str, Any]], facts: RunFacts
) -> EvalPlan:
    """Turn the parsed flags into the plan the shared existing-trace checks and identities read.

    `games.run_evals.EvalPlan` is the container those checks take, and everything in it is known
    here before any tokenizer loads. `trained_game_ids` follows the battery's own rule: the corpus's
    games when any target carries an adapter, nothing when every target is the un-adapted base. Every
    game the file holds rather than row 0's, since the whole file is what the run trained. This cell
    is handed the corpus itself, so it reads the games off its rows where the battery has to read them
    off what the run recorded (`games.run_evals.TRAINED_GAMES_FROM_CORPUS_COMPOSITION`); the source is
    recorded either way, because the two are the same fact measured one step apart.
    """
    arm = _resolve_arm(args, facts)
    thinking = _resolve_thinking(args, facts)
    if args.samples_per_prompt < 1:
        raise ValueError(f"--samples-per-prompt must be at least 1, got {args.samples_per_prompt}.")
    run_dir = cast("Path", args.run_dir)
    out_dir = (
        cast("Path", args.out_dir)
        if args.out_dir is not None
        else _default_out_dir(arm, args.backend)
    )
    targets = tuple(
        EvalTarget(
            step=spec.step,
            base_model=spec.base_model,
            checkpoint=spec.checkpoint,
            out_path=out_dir / f"step-{spec.step}.jsonl",
        )
        for spec in _run_dir_specs(args, run_dir, facts)
    )
    adapted = any(target.checkpoint is not None for target in targets)
    return EvalPlan(
        arm=arm,
        thinking=thinking,
        out_dir=out_dir,
        run_dir=run_dir,
        targets=targets,
        trained_game_ids=tuple(sorted({str(row["game_id"]) for row in rows})) if adapted else (),
        trained_games_source=(
            TRAINED_GAMES_FROM_CORPUS_ROWS if adapted else TRAINED_GAMES_NONE_UNADAPTED
        ),
        grading=facts.grading,
        executed_estimator=facts.executed_estimator,
    )


def _run_targets(  # noqa: PLR0913 - one keyword per thing main resolved before the template loads
    targets: Sequence[EvalTarget],
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    rows: Sequence[Mapping[str, Any]],
    corpus: Mapping[str, Any],
    sync: IntervalSync | None,
) -> None:
    """Measure the chat template once, then sample, resume or summarise each target still to run.

    Called only with targets left: the template facts come off a tokenizer load, and a --sync-dest
    relaunch that restored every cell complete has nothing to measure them for.
    """
    template = _resolve_template_facts(
        args, thinking=plan.thinking, template_source=plan.targets[0].base_model
    )
    config = EvalConfig(
        # In the config so the meta's `eval_config` says what was drawn; `samples_per_prompt` is
        # this cell's own name for the same number.
        game_behavior_samples=args.samples_per_prompt,
        prefilled_think=template.prefilled_think,
        batch_size=args.batch_size,
        chat_template_kwargs=template.chat_template_kwargs,
    )
    logger.info(
        f"training-frames plan: arm={plan.arm} steps={[target.step for target in targets]} "
        f"corpus={args.corpus} n_rows={len(rows)} samples_per_prompt={args.samples_per_prompt} "
        f"thinking={plan.thinking} prefilled_think={template.prefilled_think} "
        f"backend={args.backend} submission={args.submission} "
        f"sampler_mode={sampler_mode_meta(args)} out_dir={plan.out_dir}"
    )
    if args.summarise_only:
        for target in targets:
            frames_plan = plan_training_frames(
                rows, config=config, trained_game=target.checkpoint is not None
            )
            summary = salvage_summary(
                target.out_path,
                frames_plan,
                expected_meta=_frames_cell_identity(plan, target, config=config, corpus=corpus),
            )
            _write_summary(target, summary)
        return
    for target in targets:
        _evaluate_target(
            target, args=args, plan=plan, rows=rows, corpus=corpus, config=config, sync=sync
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve targets, then sample or resume the training corpus at each one and write the traces."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    # Fail on a mode aimed at a backend that cannot honour it before anything expensive resolves.
    resolve_sampler_mode(args)
    if not args.summarise_only:
        _bridge_hf_decode_kernel(args.backend)
    corpus_path = cast("Path", args.corpus)
    rows = load_corpus_rows(corpus_path)
    facts = _read_run_facts(cast("Path", args.run_dir))
    _refuse_corpus_from_another_game(rows, facts, path=corpus_path)
    plan = _resolve_plan(args, rows=rows, facts=facts)
    if args.backend in backend_cli.LOCAL_KINDS:
        # Namespace mutation is how the derived default reaches backend_from_args (run_evals ditto).
        args.thinking = plan.thinking
    corpus = _corpus_identity(args, rows, _corpus_sha256(corpus_path))
    sync = (
        None
        if args.sync_dest is None
        else IntervalSync(plan.out_dir, str(args.sync_dest), float(args.sync_interval_seconds))
    )
    if sync is not None:
        # Before the existing-trace check, so a relaunch on a fresh box finds the cell it is
        # continuing; an empty prefix restores nothing and is the first launch.
        _restore_missing_files(sync.s3_dest, sync.local_dir)
    targets = _check_existing_traces(
        plan.targets,
        plan=plan,
        sections=SECTIONS_TRAINING_FRAMES,
        summarise_only=bool(args.summarise_only),
        skip_complete=sync is not None,
        # The corpus half of the identity, checked on every existing trace -- complete ones too.
        extra_identity=corpus,
    )
    if targets:
        _run_targets(targets, args=args, plan=plan, rows=rows, corpus=corpus, sync=sync)
    # Resumed records are counted per cell in its meta; skipped means whole cells already complete.
    logger.info(
        f"training-frames run finished: {len(targets)} cell(s) run or summarised, "
        f"{len(plan.targets) - len(targets)} skipped as already complete"
    )
    if sync is not None:
        sync.sync_now(reason="run finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
