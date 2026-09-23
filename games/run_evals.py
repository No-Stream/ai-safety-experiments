r"""Drive `games.evals.run_eval_battery` over checkpoints and render the arm's report.

The battery itself is a library function; this is the entry point that points it at something
real. Three target shapes, exactly one per invocation:

*   `--model <hub id or dir>` -- a plain model: the un-adapted base, or a directory that is
    already an ordinary HF checkpoint.
*   `--checkpoint <dir>` -- one LoRA checkpoint. How its weights reach the backend is
    `games.eval_model`'s decision, not this file's: the un-merged runtime adapter where the backend
    can serve one, a merge where it cannot. Every trace records which.
*   `--run-dir <dir>` -- every `checkpoint-<step>` under a training run, in step order
    (`--steps` subsets; step 0 means the un-adapted base model).

Provenance is derived rather than demanded: the training run's `run_config.json` supplies the arm
name and thinking mode, the adapter's own `adapter_config.json` supplies the base model, and the
checkpoint directory name supplies the step. Every derivation has an explicit flag override, and a
value that can be neither derived nor found raises before anything loads -- `games.report` refuses
traces whose meta lacks `arm` or `step`, so writing one would be paying GPU time for an artifact
the report layer will bounce.

Two refusals and one continuation are the point of this wrapper rather than incidental. A trace
that already has its summary beside it is a complete cell and is never overwritten: every target's
out path is checked up front, so a collision on step 70 is discovered before step 5 spends an hour
of GPU, and the run dies loudly instead of truncating paid-for output. A trace WITHOUT a summary is
a cell that died mid-run, and relaunching the identical command continues it: the finished records
are kept byte for byte, only the missing prompts are generated, and the trace's meta gains a
`resume` block naming every session and its commit (`games.evals.inspect_trace`). A trace that is
already complete but never got its summary is summarised without loading an engine at all. So a
kit runner's whole recovery is "run the same command again"; the `rm -f` of a partial trace that
runners used to do before relaunching throws away paid-for records for nothing and must go.
`--summarise-only` does the summary step alone, for a trace copied down from S3 with no GPU around.
`--sync-dest s3://...` restores the cell's directory from that prefix before starting and syncs it
back on an interval while records land (`--sync-interval-seconds`), so a box that dies takes at
most one interval of finished records with it and its relaunch on any box is a continuation. The
restore only adds files the local directory lacks: on the box a cell died on, the local trace holds
every record appended since the last upload and always wins (`_restore_missing_files`). And
`prefilled_think` is measured from the chat template (`derive_prefilled_think`) rather than
assumed: the thinking-off arms render an empty `<think></think>` pair into the prompt, so a
hardcoded True would score every completion as truncated thinking.

The base model's step-0 cell can come from a shared bank instead of being generated per arm
(owner decision C1, 2026-09-03). Step 0 is the un-adapted base, so the cell depends on the prompts
rendered, the sampler and the budget and not on the arm, and every arm on one base used to pay for
its own draw of it: 177 minutes per v2-shaped 2B run, a ~12 h cell per arm at 9B. With
`--banked-base-cells s3://.../banked-base-cells/` the driver derives the cell's bank key
(`bank_identity`, `bank_key`: base model, thinking mode, sections, the whole eval config bar the
machine-local item paths, backend kind and quantization, sampler mode and every sampling knob
including `max_new_tokens`, and a digest of every prompt the plan renders) and, when the bank holds a
complete entry under it, copies that entry's trace and summary into the run's out dir byte for
byte -- exactly what the track-record-v2 and softpen kits did by hand with `aws s3 cp` -- and
writes a provenance sidecar beside them (`games.report.banked_provenance_path`: source entry,
source arm, the code revision that generated it, sha256 of each file, and this arm's own run dir,
grading and estimator for the readers to overlay). Otherwise the cell is generated as before,
resumed if a partial trace is already there, and, with `--bank-base-cells`, published under the
key once complete: the trace and summary under content-tagged object names, so two arms publishing
one key concurrently never overwrite each other, and the manifest that names them last. A publish
problem never fails the run -- the twelve-hour cell is the expensive thing and is safe on disk
either way -- so what the publish did lands beside the cell as `step-<step>.bank-publish.json`
(`bank_publish_record_path`), which is the only record of it a later box or a kit reading exit codes
can see. The copied
trace's meta names the arm that generated it, so the readers (`games.report.attribute_trace`) label
it with this arm only through the sidecar, after checking the hash. One instrument stays per arm as
the step-0 test-retest noise floor: a cell whose framing sweep renders only `NOISE_FLOOR_FRAMINGS`
(the unstated-only cell by default) is never taken from the bank and never published.

GPU discipline follows the house pattern: the runner does not talk to `gpu_preflight` itself --
invoke it under the limiter, which runs the preflight and refuses a busy card:

    scripts/resource-limits.sh --gpu -t 480m -- \
        uv run --frozen python -m games.run_evals --run-dir artifacts/games/runs/<run>

(`make games-evals` wraps exactly that.) `--backend mock` runs the whole driver -- resolution,
trace, summary, report -- with canned completions and no model load, which is the offline smoke
of this file's own plumbing. Its traces are branded `mock:` in their meta and, with no explicit
`--out-dir`, land in a `mock/` subdirectory of the arm's trace directory rather than beside the
real ones: branding alone was not enough, because no report table read it and the collision then
blocked the arm's real eval of those steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import time
import zlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.arms import ARMS, arm_game_ids
from games.deltanet_kernels import assert_bridged_kernel_matches_call_site, bridge_decode_kernel
from games.eval_model import (
    ServedModel,
    resolve_served_model,
    sha256_of_file,
    verify_served_model,
)
from games.eval_sampler import (
    add_sampler_arg,
    eval_sampling,
    resolve_sampler_mode,
    sampler_mode_meta,
)
from games.evals import (
    ADMISSION_FIFO,
    ADMISSION_LONGEST_FIRST,
    ADMISSIONS,
    DEFAULT_SECTIONS,
    FRAMING_SWEEP_GAME_IDS,
    LABEL_PRINT_ORDER_BOTH,
    LABEL_PRINT_ORDER_REQUESTS,
    RECORD_META,
    RESUME_LOCAL_PATH_CONFIG_FIELDS,
    SECTION_FRAMING_SWEEP,
    SECTION_TRAP_CELLS,
    SECTIONS,
    SUBMISSION_POOLED,
    SUBMISSION_SERIAL,
    SUBMISSIONS,
    EvalConfig,
    _json_native,  # pyright: ignore[reportPrivateUsage]
    finish_complete_trace,
    inspect_trace,
    plan_battery,
    read_eval_records,
    run_eval_battery,
    salvage_summary,
    summarise_trace,
)
from games.framing_stimulus import load_dictator_recipient_clauses, load_framings
from games.held_out_extension import load_extension
from games.lora import checkpoint_step, iter_checkpoints, read_adapter_base_model
from games.preflight import (
    default_cuda_allocator_config,
    derive_prefilled_think,
    resolve_chat_template_kwargs,
)
from games.prompts import COUNTERPART_FRAMING_IDS, FRAMING_UNSTATED, LABEL_PRINT_ORDERS
from games.provenance import git_sha
from games.report import (
    BANKED_PROVENANCE_RECORD,
    MOCK_BACKEND_KIND,
    banked_provenance_path,
    read_banked_provenance,
    render_report,
)
from games.s3_sync import SyncOutcome, restore_directory, sync_directory
from games.vllm_teardown import baseline_before_engine, release_engine
from reward_hacking import backend_cli

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

    from games.evals import PlannedRequest

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = _REPO_ROOT / "artifacts" / "games" / "evals"
DEFAULT_MERGE_ROOT = DEFAULT_EVAL_ROOT / "merge-tmp"

RUN_CONFIG_FILENAME = "run_config.json"
BASE_MODEL_STEP = 0
SUMMARY_SUFFIX = ".summary.json"
# Where `_restore_missing_files` lands the S3 copy before deciding, file by file, what to move in.
RESTORE_STAGING_SUFFIX = ".restore-tmp"
# How often a running cell pushes its trace directory to --sync-dest. Ten minutes bounds the loss a
# box death can inflict to ten minutes of records against cells that run for hours, and a 177 MB
# trace re-uploads in seconds from EC2; the training side syncs per checkpoint save on the same
# reasoning.
DEFAULT_SYNC_INTERVAL_SECONDS = 600.0

# The shared bank of base-model step-0 cells (`--banked-base-cells`; module docstring). One entry per
# `bank_key` under the prefix, holding the cell's trace and summary plus this manifest, which is
# uploaded last so an entry without it is half-published rather than complete.
BANK_MANIFEST_FILENAME = "banked_cell.json"
BANK_MANIFEST_RECORD = "banked-cell"
BANK_STAGING_SUFFIX = ".bank-tmp"
# Leading sha256 characters of the trace that tag an entry's payload objects; see `bank_object_name`.
BANK_OBJECT_TAG_CHARS = 12
# What `--bank-base-cells` did, written beside the cell on every exit path of `_publish_banked_cell`
# because that function deliberately never raises over a publish problem; see its docstring.
BANK_PUBLISH_RECORD_SUFFIX = ".bank-publish.json"
BANK_PUBLISH_RECORD = "bank-publish"
BANK_PUBLISH_PUBLISHED = "published"
BANK_PUBLISH_KEPT_RIVAL = "kept-another-runs-entry"
BANK_PUBLISH_INVALID_RIVAL = "invalid-rival-entry-kept"
BANK_PUBLISH_FILES_FAILED = "files-upload-failed"
BANK_PUBLISH_MANIFEST_FAILED = "manifest-upload-failed"
BANK_PUBLISH_NO_COMPLETE_CELL = "no-complete-cell-to-publish"
# The step-0 instrument that stays per arm as the test-retest noise floor: a cell whose framing sweep
# renders only these framings is never taken from the bank and never published (`is_noise_floor_cell`).
NOISE_FLOOR_FRAMINGS: tuple[str, ...] = (FRAMING_UNSTATED,)

# One EvalConfig() so the CLI defaults are the dataclass defaults, never a drifting second copy.
EVAL_DEFAULTS = EvalConfig()
# The single-order default, read off the dataclass for the same reason: `--label-print-order` takes
# one value while the config takes a tuple, so the CLI names the request and `_parse_print_orders`
# expands it. A config already asking for several orders has no single name, which cannot happen
# from this CLI and would be a caller building EvalConfig directly.
DEFAULT_LABEL_PRINT_ORDER = (
    EVAL_DEFAULTS.label_print_orders[0]
    if len(EVAL_DEFAULTS.label_print_orders) == 1
    else LABEL_PRINT_ORDER_BOTH
)

# Deliberately unparseable, so a mock trace can never read as a behavioural measurement.
MOCK_PLUMBING_RESPONSES: tuple[str, ...] = (
    "Mock plumbing completion; parses as nothing on purpose.",
)
# Where a mock smoke writes when no --out-dir is given; see `_default_out_dir`.
MOCK_TRACE_SUBDIR = "mock"


@dataclass(frozen=True)
class RunFacts:
    """What a training run's `run_config.json` can contribute; None where it recorded nothing."""

    arm: str | None
    base_model: str | None
    thinking: bool | None
    game_id: str | None
    # The OTHER games the run's corpus was allowed to carry (`games.arms.GameArm.game_ids`), so the
    # games it trained are `game_id` plus these. Empty rather than None where a record names none,
    # which is every arm before wave 4b: "no extra games" and "this field did not exist" are the same
    # fact here, unlike `game_id`, whose absence has to stay tellable from a game named.
    game_ids: tuple[str, ...]
    # The games the run's corpus ACTUALLY held (`derived.corpus_composition.game_id`), which is what
    # it trained rather than what its arm allowed: a breadth arm's selection can drop a whole game
    # (`games.breadth_corpus._drop_thin_games`), and the two disagreeing is a corpus that came in
    # short. Empty for every record written before the composition carried the column.
    corpus_game_ids: tuple[str, ...]
    grading: str | None
    # The aggregation the run EXECUTED (may differ from config.loss_type under Liger); None for
    # records written before games/train.py recorded it, which includes every pre-2026-08-20 arm.
    executed_estimator: str | None


EMPTY_RUN_FACTS = RunFacts(
    arm=None,
    base_model=None,
    thinking=None,
    game_id=None,
    game_ids=(),
    corpus_game_ids=(),
    grading=None,
    executed_estimator=None,
)

# Which derivation named a plan's trained games, recorded in every trace's meta. The sources are not
# equally direct -- the corpus composition is what the run trained, the arm's game list is only what
# it allowed -- and a reader months later has no other way to tell which one a trace's transfer
# column rests on.
TRAINED_GAMES_FROM_FLAG = "trained-games-flag"
TRAINED_GAMES_FROM_CORPUS_COMPOSITION = "run-config-corpus-composition"
TRAINED_GAMES_FROM_ARM_ALLOWED = "run-config-arm-allowed"
TRAINED_GAMES_FROM_ARM_REGISTRY = "arm-registry"
TRAINED_GAMES_FROM_CORPUS_ROWS = "corpus-file-rows"
TRAINED_GAMES_NONE_UNADAPTED = "no-adapter-trained-nothing"
TRAINED_GAMES_UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class _TargetSpec:
    """One model to evaluate, before the out path is attached."""

    step: int
    base_model: str
    checkpoint: Path | None


@dataclass(frozen=True)
class EvalTarget:
    """One model the battery will evaluate, with the provenance its trace must carry."""

    step: int
    base_model: str
    checkpoint: Path | None
    out_path: Path


@dataclass(frozen=True)
class EvalPlan:
    """Everything resolved before a model loads: what to evaluate, as whom, and where to write."""

    arm: str
    thinking: bool
    out_dir: Path
    run_dir: Path | None
    targets: tuple[EvalTarget, ...]
    trained_game_ids: tuple[str, ...]
    # Which of the derivations above filled `trained_game_ids`; carried into every trace's meta.
    trained_games_source: str
    grading: str | None
    executed_estimator: str | None


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI: target selection, provenance overrides, battery knobs, backend knobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Evaluate a plain model: a hub id or an ordinary HF checkpoint directory. Requires "
            f"--arm; --step defaults to {BASE_MODEL_STEP} (the pre-training point)."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Evaluate one LoRA checkpoint directory. Served un-merged where the backend can, "
            "merged onto its recorded base where it cannot; games.eval_model decides."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Evaluate every checkpoint-<step> under a training run directory, in step order.",
    )
    parser.add_argument(
        "--arm",
        default=None,
        help="Arm label for the trace meta; derived from the run's run_config.json when omitted.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help=(
            "Step label override for --model/--checkpoint. Defaults to the checkpoint's own "
            f"number, or {BASE_MODEL_STEP} for --model."
        ),
    )
    parser.add_argument(
        "--steps",
        default=None,
        help=(
            f"Comma-separated steps to evaluate under --run-dir; {BASE_MODEL_STEP} means the "
            "un-adapted base model. Default: every checkpoint present."
        ),
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base model override for merging and for step 0; defaults to what the adapter's "
            "adapter_config.json (or the run's run_config.json) recorded."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            f"Trace directory (default: {DEFAULT_EVAL_ROOT}/<arm>, plus a /{MOCK_TRACE_SUBDIR} "
            f"level under --backend mock). Traces are step-<step>.jsonl; a trace with a summary "
            f"beside it is complete and is refused, never overwritten, while a trace without one "
            f"is a cell that died mid-run and is resumed."
        ),
    )
    parser.add_argument(
        "--submission",
        choices=SUBMISSIONS,
        default=SUBMISSION_POOLED,
        help=(
            f"How prompts reach the backend (default: {SUBMISSION_POOLED}). {SUBMISSION_POOLED!r} "
            f"submits every section's prompts at once and files each record as its completion "
            f"lands, so the engine never idles between games; {SUBMISSION_SERIAL!r} is one call per "
            f"(game, print order) and per remaining section, the pre-2026-09-02 sequence, which "
            f"replays a banked cell byte for byte on the same engine seed. Same sampler, same seed, "
            f"same distribution either way; not the same bytes."
        ),
    )
    parser.add_argument(
        "--admission",
        choices=ADMISSIONS,
        default=ADMISSION_LONGEST_FIRST,
        help=(
            f"The order a {SUBMISSION_POOLED!r} submission queues its prompts in (default: "
            f"{ADMISSION_LONGEST_FIRST}). {ADMISSION_LONGEST_FIRST!r} admits the sections whose "
            f"sequences run longest first (self-report, then dt-probes, capabilities, game-behavior, "
            f"framing-sweep), so the 65k-token thinkers overlap the bulk instead of trailing it: "
            f"measured on a base-2B cell (probe P-E4), the self-report prompts admitted last ran "
            f"nearly alone for the final ~50 of 221 minutes. {ADMISSION_FIFO!r} is plan order, how "
            f"every pooled cell before 2026-09-03 was queued; keep it to replay one of those byte "
            f"for byte on its engine seed. Same record set, same identities either way, at the "
            f"statistical grade pooling itself carries, with its one open caveat: bf16 sampling is "
            f"not batch-invariant and the order changes the batch the self-report thinkers run in, "
            f"so the next cell with a fifo or serial twin should compare the self-report cap-hit "
            f"rate (games.evals.ADMISSIONS). Ignored under the {SUBMISSION_SERIAL!r} submission, "
            f"whose call groups run in plan order."
        ),
    )
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help=(
            "Write the summary of an already-complete trace at each target's out path and load no "
            "engine: the salvage for a cell that died after its last generate call and before its "
            "summary write. The base model's tokenizer is still read (from the hub or its cache) "
            "to derive the chat-template facts the cell identity includes, unless every target is "
            "already complete under --sync-dest, when nothing loads at all; nothing else ever "
            "loads. Refuses an incomplete trace (relaunch the cell's own command instead, which "
            "resumes it) and, without --sync-dest, a trace that already has a summary; with "
            "--sync-dest a complete cell is skipped as this run's own and the run exits 0."
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
        "--banked-base-cells",
        default=None,
        help=(
            "s3:// prefix of the shared bank of base-model step-0 cells. A step-0 target whose bank "
            "key (base model, thinking, sections, eval config, backend, sampler and every sampling "
            "knob, and a digest of every rendered prompt) matches a complete entry is copied from it "
            f"byte for byte, with a {BANKED_PROVENANCE_RECORD!r} sidecar naming the source beside "
            "it, and generates nothing; any other step-0 target generates as usual. Checkpoint "
            "cells are never banked. The bank must be readable: a failed fetch refuses to start "
            "rather than silently paying for the cell again."
        ),
    )
    parser.add_argument(
        "--bank-base-cells",
        action="store_true",
        help=(
            "Publish every step-0 cell this run generates into --banked-base-cells under its key "
            "(trace, summary, then the manifest last), so the next arm on this base can take it. "
            "A cell another run banked under the same key first is kept and this run's copy is not "
            "published over it. A publish problem never fails the run, so what it did is recorded "
            f"beside the cell in step-<step>{BANK_PUBLISH_RECORD_SUFFIX}."
        ),
    )
    parser.add_argument(
        "--noise-floor-framings",
        # None, not the joined default, so `_parse_bank` can tell an explicit pass from the default.
        default=None,
        help=(
            f"Comma-separated counterpart framings whose sweep cell is the per-arm step-0 "
            f"test-retest noise floor (default: {','.join(NOISE_FLOOR_FRAMINGS)}). A step-0 cell "
            f"whose framing-sweep section renders only these framings is never taken from the "
            f"bank and never published, so every arm still draws it itself. An explicit empty "
            f"list banks every step-0 cell and is logged as such. Only meaningful alongside "
            f"--banked-base-cells, and refused without it."
        ),
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        default=DEFAULT_MERGE_ROOT,
        help=(
            f"Staging area for merged checkpoints (default: {DEFAULT_MERGE_ROOT}). Each merge is "
            "deleted after its step's eval; a crash leaves it behind for debugging."
        ),
    )
    parser.add_argument(
        "--sections",
        default=",".join(DEFAULT_SECTIONS),
        help=(
            f"Comma-separated sections to run (default: {list(DEFAULT_SECTIONS)}). Also available: "
            f"{sorted(set(SECTIONS) - set(DEFAULT_SECTIONS))}. That one is opt-in because it is the "
            f"largest section by render count and ALL of its item text lives outside version "
            f"control; it requires --survey-data-dir (see games/data/survey/README.md) and runs "
            f"nothing without it."
        ),
    )
    parser.add_argument(
        "--games",
        default="",
        help=(
            "Comma-separated game ids for the game-behavior section; empty means every game. "
            "Eval-only ids are accepted and render through the never-trained leg; naming any "
            "narrows that leg to exactly the named ones."
        ),
    )
    parser.add_argument(
        "--counterpart-framings",
        default="",
        help=(
            f"Comma-separated framing ids for the framing-sweep section, from "
            f"{list(COUNTERPART_FRAMING_IDS)}; empty means all of them when that section is "
            f"requested. The section renders {list(FRAMING_SWEEP_GAME_IDS)} under each framing, "
            f"eval frames only."
        ),
    )
    parser.add_argument(
        "--framing-sweep-games",
        default="",
        help=(
            f"Comma-separated frameable game ids the framing-sweep section renders; empty means "
            f"the pinned default {list(FRAMING_SWEEP_GAME_IDS)}, which every banked sweep cell "
            f"was measured on. Naming games is explicit opt-in (e.g. the temptation-dose ladder "
            f"for a same-rate-different-optimum cell)."
        ),
    )
    parser.add_argument(
        "--framings-file",
        type=Path,
        default=None,
        help=(
            "Path to the gitignored runtime framings file (games/framing_stimulus.py gives its "
            "schema), whose counterpart clauses are authored stimulus and so live outside version "
            "control. Required whenever --counterpart-framings names a framing the registry does "
            "not carry, including under --summarise-only, which re-renders the plan to rebuild a "
            "summary. Pointing at a file does not widen a sweep: the framing list still names what "
            "runs, and the file's digest lands in the trace meta and the cell identity."
        ),
    )
    parser.add_argument(
        "--held-out-extension",
        type=Path,
        default=None,
        help=(
            "Path to the gitignored held-out extension staging file (games.held_out_extension). "
            "When set, the game-behavior section renders the tracked held-out frames plus the "
            "extension's, per the caps file beside it; the framing sweep is deliberately "
            "unaffected, so its cells stay comparable to the banked ones. The file's digest lands "
            "in the trace meta and the cell identity, so an extended cell can never share a bank "
            "entry or a resumed trace with a plain one."
        ),
    )
    parser.add_argument(
        "--trained-games",
        default=None,
        help=(
            "Comma-separated game ids this checkpoint trained on, for the trained-versus-transfer "
            f"column. Derived from the run's {RUN_CONFIG_FILENAME} when omitted; a plain --model "
            "trained on nothing."
        ),
    )
    parser.add_argument(
        "--dtbench-dir",
        type=Path,
        default=None,
        help="DTBench data directory; omitted means the dt-probes section runs our battery only.",
    )
    parser.add_argument(
        "--open-ended-samples",
        type=int,
        default=EVAL_DEFAULTS.open_ended_samples,
        help=f"Samples per open-ended probe (default: {EVAL_DEFAULTS.open_ended_samples}).",
    )
    parser.add_argument(
        "--open-ended-samples-by-item",
        default="",
        help=(
            "Per-item overrides of --open-ended-samples, as comma-separated probe-id=count pairs "
            "(e.g. open-parfits-hitchhiker=2). Rebalances the open-ended render budget between "
            "items without deleting any: counts below 1 are refused."
        ),
    )
    parser.add_argument(
        "--multiple-choice-samples",
        type=int,
        default=EVAL_DEFAULTS.multiple_choice_samples,
        help=(
            f"Samples per choice probe PER OPTION ORDER (default: "
            f"{EVAL_DEFAULTS.multiple_choice_samples}; one render per order resolves 1/8 steps at "
            f"best, which was the binding measurement-quality limit of the first battery)."
        ),
    )
    parser.add_argument(
        "--label-print-order",
        choices=sorted(LABEL_PRINT_ORDER_REQUESTS),
        default=DEFAULT_LABEL_PRINT_ORDER,
        help=(
            f"Which label the game-behaviour prompts print first (default: "
            f"{DEFAULT_LABEL_PRINT_ORDER}, which is what every battery cell so far rendered). "
            f"{LABEL_PRINT_ORDER_BOTH!r} renders each game under both orders and stamps every record "
            f"with the one it got, which is what de-aliases a move toward a label from a move toward "
            f"a position. The games that print no action labels always render canonically."
        ),
    )
    parser.add_argument(
        "--game-behavior-samples",
        type=int,
        default=EVAL_DEFAULTS.game_behavior_samples,
        help=(
            f"Completions drawn per game-behaviour prompt (default: "
            f"{EVAL_DEFAULTS.game_behavior_samples}). One draw resolves a per-prompt rate to 0/1 or "
            f"1/1, which is one of the three named components of the first wave's flat rows; the "
            f"readouts average within a prompt before averaging over prompts, so raising this buys "
            f"per-prompt resolution rather than a larger denominator."
        ),
    )
    parser.add_argument(
        "--capability-items",
        type=int,
        default=EVAL_DEFAULTS.capability_items,
        help=f"Arithmetic-canary items (default: {EVAL_DEFAULTS.capability_items}).",
    )
    parser.add_argument(
        "--survey-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory holding the untracked item files (authored.json and published.json) for the "
            "self-report section; games/data/survey/README.md gives their schemas. Required "
            "whenever that section is requested -- it runs nothing without local item data."
        ),
    )
    parser.add_argument(
        "--trap-cells",
        type=Path,
        default=None,
        help=(
            "Path to the runtime framings file whose dictator_recipient_clauses key holds the two "
            "authored recipient paragraphs the trap-cells section inserts (see "
            "games/framing_stimulus.py for the shape). Required whenever that section is requested "
            "-- its paragraphs are "
            "authored stimulus that is never committed, so the battery renders nothing without the "
            "file. Its digest lands in the trace meta and in the cell identity, so a cell measured "
            "under edited paragraphs cannot be mistaken for one measured under the earlier text."
        ),
    )
    parser.add_argument(
        "--survey-samples",
        type=int,
        default=EVAL_DEFAULTS.survey_samples,
        help=(
            f"Samples per survey item PER OPTION ORDER (default: {EVAL_DEFAULTS.survey_samples}). "
            f"The section costs twice this times the item count in renders."
        ),
    )
    parser.add_argument(
        "--survey-instruments",
        default="",
        help=(
            "Comma-separated published-instrument ids for the self-report section; empty means every "
            "instrument the local file supplies. Naming a subset is the only supported way to run a "
            "partial battery, and the names reach the trace's meta."
        ),
    )
    parser.add_argument(
        "--survey-families",
        default="",
        help=(
            "Comma-separated survey families to administer; empty means every family that has items. "
            "This is how the negative-control or breadth families get run on their own."
        ),
    )
    parser.add_argument(
        "--survey-tier",
        default=EVAL_DEFAULTS.survey_tier,
        help=(
            "Survey tier to administer (core or breadth); empty means both. The core set is "
            "position-level within instruments, so this knob is the only way to run the "
            "deliberated leg without paying thinking-on completions for every breadth item."
        ),
    )
    parser.add_argument(
        "--survey-counterpart-variants",
        action=argparse.BooleanOptionalAction,
        default=EVAL_DEFAULTS.survey_counterpart_variants,
        help=(
            "Include the opt-in counterpart-framed variant of the self-prediction twin-pd item. "
            "The default keeps the historical survey battery unchanged."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EVAL_DEFAULTS.batch_size,
        help=(
            "Sequences per generation call. Default: derived from the free VRAM and the measured "
            "throughput knee, which is what lets one config run on whatever card is free. An "
            "explicit width past that knee is refused rather than clamped."
        ),
    )
    parser.add_argument(
        "--include-never-trained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also play the eval-only transfer games (the cross-game generalisation readout).",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render report.md over every trace in the out dir after the battery finishes.",
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


def _parse_sections(raw: str) -> tuple[str, ...]:
    """Validate the section list before anything expensive loads; the battery re-checks later."""
    sections = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not sections:
        raise ValueError(f"--sections is empty; choose from {list(SECTIONS)}.")
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise ValueError(f"Unknown eval sections {unknown}; known sections: {list(SECTIONS)}.")
    duplicated = sorted({name for name in sections if sections.count(name) > 1})
    if duplicated:
        raise ValueError(f"Sections repeat {duplicated}; each section runs at most once.")
    return sections


def _refuse_orphan_trap_cells_file(args: argparse.Namespace, sections: Sequence[str]) -> None:
    """Refuse a --trap-cells file none of the requested sections would render.

    Ignoring it silently is the shape of failure this repository keeps finding: a launch that passes
    the file and forgets to name the section runs a complete, plausible battery with no trap cells in
    it, and the absence is only visible to whoever notices a missing summary key months later.
    """
    if args.trap_cells is not None and SECTION_TRAP_CELLS not in sections:
        raise ValueError(
            f"--trap-cells {args.trap_cells} was passed while --sections names {list(sections)}, "
            f"which does not include {SECTION_TRAP_CELLS!r}, so the recipient paragraphs would be "
            f"loaded and never rendered. Add {SECTION_TRAP_CELLS!r} to --sections, or drop the flag."
        )


def _parse_id_list(raw: str) -> tuple[str, ...]:
    """Split a comma-separated id list; empty means "every registered one", which EvalConfig reads.

    Shared by --games, --trained-games and --survey-instruments rather than written three times:
    `EvalConfig` validates each list against its own registry, so the splitter has no registry of
    its own and does not need one per flag.
    """
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_open_ended_allocation(raw: str) -> tuple[tuple[str, int], ...]:
    """Parse --open-ended-samples-by-item into (probe_id, count) pairs; EvalConfig validates ids."""
    allocation: list[tuple[str, int]] = []
    for part in (piece.strip() for piece in raw.split(",") if piece.strip()):
        probe_id, separator, count = part.partition("=")
        if not separator or not probe_id.strip():
            raise ValueError(
                f"--open-ended-samples-by-item entries are probe-id=count pairs, got {part!r}."
            )
        try:
            allocation.append((probe_id.strip(), int(count)))
        except ValueError as error:
            raise ValueError(
                f"--open-ended-samples-by-item count for {probe_id.strip()!r} must be an "
                f"integer, got {count!r}."
            ) from error
    return tuple(allocation)


def _parse_print_orders(raw: str) -> tuple[str, ...]:
    """Expand the one `--label-print-order` value into the orders the battery will render.

    `argparse` already refuses anything outside `LABEL_PRINT_ORDER_REQUESTS`, and `EvalConfig`
    re-checks what comes out, so this is only the both-means-two expansion.
    """
    if raw == LABEL_PRINT_ORDER_BOTH:
        return tuple(LABEL_PRINT_ORDERS)
    return (raw,)


def _parse_steps(raw: str) -> list[int]:
    """Parse --steps into unique integers, refusing shapes that would double-write a trace."""
    try:
        steps = [int(part) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise ValueError(f"--steps must be comma-separated integers, got {raw!r}.") from error
    if not steps:
        raise ValueError(f"--steps is empty, got {raw!r}.")
    duplicated = sorted({step for step in steps if steps.count(step) > 1})
    if duplicated:
        raise ValueError(f"--steps repeats {duplicated}; each step is evaluated at most once.")
    return steps


def _read_run_facts(run_dir: Path) -> RunFacts:
    """Read what the run's own config recorded, so arm and thinking never come from a filename.

    The run dir name is decorative (`twin-pd-self-2b-plumbing` for arm `twin-pd-self`), so
    deriving labels from paths would quietly mislabel traces -- exactly what `games.report`
    refuses filename-guessed labels to prevent.

    `game_id`, `game_ids` and `grading` are read for the same reason and are what make two eval
    columns truthful: most arms train exactly one game, so `trained_game` has to name the games that
    arm trained rather than every game in the registry, and two arms share `twin-pd` under different
    gradings, so the arm's grading cannot be recovered from the game. `games/train.py` writes all of
    them at the payload's top level beside `arm`.

    The corpus composition is read from `derived` because that is the only field naming what the
    corpus FILE held: the top-level game list is the arm's allowance, and on a breadth arm the two
    part company whenever the selection dropped a game group whole.
    """
    path = run_dir / RUN_CONFIG_FILENAME
    if not path.is_file():
        return EMPTY_RUN_FACTS
    raw = json.loads(path.read_text(encoding="utf-8"))
    config = raw.get("config") or {}
    thinking = config.get("thinking")
    composition = (raw.get("derived") or {}).get("corpus_composition") or {}
    return RunFacts(
        arm=raw.get("arm"),
        base_model=config.get("model_id"),
        thinking=None if thinking is None else bool(thinking),
        game_id=raw.get("game_id"),
        game_ids=tuple(str(game_id) for game_id in raw.get("game_ids") or ()),
        corpus_game_ids=tuple(str(game_id) for game_id in composition.get("game_id") or ()),
        grading=raw.get("grading"),
        executed_estimator=raw.get("executed_estimator"),
    )


def _base_model_for(args: argparse.Namespace, checkpoint: Path) -> str:
    """Resolve a checkpoint's base model: the explicit flag, else what PEFT recorded at training."""
    if args.base_model is not None:
        return str(args.base_model)
    return read_adapter_base_model(checkpoint)


def _run_dir_specs(args: argparse.Namespace, run_dir: Path, facts: RunFacts) -> list[_TargetSpec]:
    """Build the target list for --run-dir: every checkpoint, or the --steps subset."""
    if args.steps is None:
        checkpoints = list(iter_checkpoints(run_dir))
        if not checkpoints:
            raise ValueError(f"{run_dir} contains no checkpoint-<step> directories to evaluate.")
        return [
            _TargetSpec(checkpoint_step(path), _base_model_for(args, path), path)
            for path in checkpoints
        ]
    specs: list[_TargetSpec] = []
    for step in _parse_steps(args.steps):
        if step == BASE_MODEL_STEP:
            base = args.base_model if args.base_model is not None else facts.base_model
            if base is None:
                raise ValueError(
                    f"step {BASE_MODEL_STEP} means the un-adapted base model, but no base model "
                    f"is known: pass --base-model, or run from a dir whose "
                    f"{RUN_CONFIG_FILENAME} records config.model_id."
                )
            specs.append(_TargetSpec(BASE_MODEL_STEP, str(base), None))
            continue
        checkpoint = run_dir / f"checkpoint-{step}"
        if not checkpoint.is_dir():
            available = [checkpoint_step(path) for path in iter_checkpoints(run_dir)]
            raise FileNotFoundError(f"{checkpoint} does not exist; available steps: {available}.")
        specs.append(_TargetSpec(step, _base_model_for(args, checkpoint), checkpoint))
    return specs


def _require_exactly_one_mode(args: argparse.Namespace) -> None:
    """Refuse zero or several target modes, and the step flags that belong to the other mode."""
    given = [
        flag
        for flag, value in (
            ("--model", args.model),
            ("--checkpoint", args.checkpoint),
            ("--run-dir", args.run_dir),
        )
        if value is not None
    ]
    if len(given) != 1:
        raise ValueError(
            f"exactly one of --model, --checkpoint, or --run-dir selects what to evaluate; "
            f"got {given or 'none of them'}."
        )
    if args.run_dir is not None and args.step is not None:
        raise ValueError("--step labels a single target; with --run-dir use --steps instead.")
    if args.run_dir is None and args.steps is not None:
        raise ValueError("--steps only applies to --run-dir; use --step for a single target.")


def _resolve_arm(args: argparse.Namespace, facts: RunFacts) -> str:
    """Name the arm from the flag or the run's own record -- never from a directory name."""
    if args.arm is not None:
        return str(args.arm)
    if facts.arm is not None:
        return facts.arm
    raise ValueError(
        f"no arm name available: pass --arm, or evaluate from a run dir whose "
        f"{RUN_CONFIG_FILENAME} records one. games.report refuses traces whose meta lacks arm "
        f"and step, so writing one would waste the whole eval."
    )


def _resolve_thinking(args: argparse.Namespace, facts: RunFacts) -> bool:
    """Resolve thinking mode: the flag, else what training used, else the repo-wide default.

    Following training is the design invariant from `games.preflight`: one raw prompt string
    flows through training and evaluation unchanged, so evaluating a thinking-off arm in
    thinking mode would measure a different policy from the one that was trained.
    """
    if args.thinking is not None:
        return bool(args.thinking)
    if facts.thinking is not None:
        logger.info(f"thinking={facts.thinking} taken from {RUN_CONFIG_FILENAME}")
        return facts.thinking
    logger.info(f"thinking defaulted to {backend_cli.DEFAULT_THINKING} (no flag, no run config)")
    return backend_cli.DEFAULT_THINKING


def summary_path_for(trace_path: Path) -> Path:
    """Where a trace's summary lives: beside it, `.summary.json` for `.jsonl`."""
    return trace_path.with_suffix(SUMMARY_SUFFIX)


def _peek_existing_trace(target: EvalTarget, *, expected: Mapping[str, Any]) -> None:
    """Refuse an existing trace whose first line is not a complete meta record for this very cell.

    Cheap -- one line read, before any tokenizer or engine -- and deliberately shallow: the full
    identity check (`games.evals.inspect_trace`) needs the resolved config and runs per target
    before its engine loads. What this catches up front is a file at the out path that is not ours
    at all: a sentinel, a torn meta line, another arm's or another step's trace, one run with the
    other thinking mode or against another base model or section list. Every field of ``expected``
    (`_cell_identity_without_config`, plus whatever the caller's cell adds to it) has to agree,
    because a later step's trace that fails only here would otherwise be refused when its turn came,
    after the earlier steps had paid their GPU time -- the exact cost this up-front pass exists to
    avoid.

    Run over complete traces as much as partial ones. A partial trace of another cell cannot be
    resumed; a complete one of another cell cannot be skipped as already done either, which is what
    --sync-dest does with a complete trace: skipped, it reports the other configuration's numbers
    under this run's name with exit 0 and no engine loaded, and the run's final sync then uploads
    them under this run's prefix (reproduced 2026-09-02 with a changed draw count, a changed corpus
    and a changed arm at one out path; each relaunch returned 0 and logged the cell as complete).
    """
    with target.out_path.open(encoding="utf-8") as handle:
        first = handle.readline()
    reason: str | None = None
    if not first.endswith("\n"):
        reason = "its first line is torn or empty"
    else:
        try:
            meta = json.loads(first)
        except json.JSONDecodeError:
            reason = "its first line is not JSON"
        else:
            if meta.get("record") != RECORD_META:
                reason = f"its first record is not a {RECORD_META!r} record"
            else:
                drifted = [
                    f"{name}={meta.get(name)!r} where {value!r} was asked"
                    for name, value in json.loads(json.dumps(expected)).items()
                    if meta.get(name) != value
                ]
                if drifted:
                    reason = f"its meta describes another cell ({'; '.join(drifted)})"
    if reason is not None:
        raise FileExistsError(
            f"refusing to touch {target.out_path}: it exists but {reason}, so it is not a trace of "
            f"this cell -- neither one to resume nor one to skip as already complete. Move it (and "
            f"any summary beside it) aside or pick another --out-dir. Nothing was evaluated."
        )


def _has_complete_summary(out_path: Path) -> bool:
    """Whether the cell at ``out_path`` is complete: a summary that parses, beside a trace that exists.

    The summary is the runner's completion marker, so both halves of that sentence are checked. A
    summary with no trace beside it is refused outright: the marker says complete and there is
    nothing it could be the summary of, so something deleted or half-copied the cell and a human has
    to look. A summary that does not parse was torn by a death mid-write (the driver now writes it
    atomically, but a trace from before that fix can still carry one); it is a pure function of the
    trace, so it is treated as absent with a warning and the trace is inspected and, when complete,
    summarised again -- the same recovery as for a cell that never got its summary at all.
    """
    summary_path = summary_path_for(out_path)
    if not summary_path.exists():
        return False
    if not out_path.exists():
        raise FileExistsError(
            f"{summary_path} exists with no trace at {out_path} beside it. A summary marks a "
            f"complete cell and is derived from its trace, so this one describes a file that is "
            f"gone; move the orphan summary aside (or restore the trace) before running this cell. "
            f"Nothing was evaluated."
        )
    try:
        json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        logger.warning(
            f"{summary_path} is not valid JSON ({error}); a death mid-write tore it. It is a pure "
            f"function of the trace beside it, so the trace is inspected and the summary rebuilt."
        )
        return False
    return True


def _check_existing_traces(  # noqa: PLR0913 - one keyword per decision the driver made about existing files
    targets: Sequence[EvalTarget],
    *,
    plan: EvalPlan,
    sections: Sequence[str],
    summarise_only: bool,
    skip_complete: bool,
    extra_identity: Mapping[str, Any] | None = None,
    banked_expected: Mapping[Path, Mapping[str, Any]] | None = None,
) -> tuple[EvalTarget, ...]:
    """Decide, before any engine loads, what each file already at a target's out path means.

    A summary beside a trace marks a complete cell (`_has_complete_summary`), and a complete cell is
    never redone from here. Without --sync-dest that is a refusal with the paths, so a collision on a
    later step is discovered before an earlier step spends its GPU time; with it, the complete cell
    was just restored from this run's own prefix and is skipped (`skip_complete`), which is what lets
    a kit relaunch the same command on a fresh box and have every finished cell cost nothing. Every
    existing trace, complete or not, must read as this cell's first (`_peek_existing_trace`, over
    `_cell_identity_without_config` plus ``extra_identity``, the fields a caller's cell carries beyond
    the battery's -- the corpus half of a training-frames cell): a complete cell of another
    configuration skipped as done would report that configuration's numbers under this run's name,
    and a partial one cannot be continued as this cell. A trace without a summary that passes is a
    cell that died mid-run and is continued. Under --summarise-only every target must already have a
    trace, since nothing will be generated. Returns the targets still to run.

    A trace with a banked-copy sidecar beside it is the one case whose meta names another arm on
    purpose, so it is held to `_verify_banked_copy` instead of the peek: ``banked_expected`` maps
    the out path of every target that may be a banked copy (a base-model step-0 target) to its
    `bank_identity_without_config`. A sidecar beside any other target's trace is refused outright.
    """
    complete = [target for target in targets if _has_complete_summary(target.out_path)]
    if complete and not skip_complete:
        raise FileExistsError(
            f"refusing to overwrite complete eval trace(s), each with a summary beside it: "
            f"{[str(target.out_path) for target in complete]}. A complete cell is paid-for GPU "
            f"output; move both files aside or pick another --out-dir. Nothing was evaluated."
        )
    for target in targets:
        if not target.out_path.exists():
            if summarise_only:
                raise FileNotFoundError(
                    f"--summarise-only needs a trace at {target.out_path}, and there is none."
                )
            continue
        if banked_provenance_path(target.out_path).exists():
            expected_bank = (banked_expected or {}).get(target.out_path)
            if expected_bank is None:
                raise FileExistsError(
                    f"refusing to touch {target.out_path}: a banked-copy sidecar "
                    f"{banked_provenance_path(target.out_path)} sits beside it, but this target "
                    f"(step {target.step}, checkpoint {target.checkpoint}) is not a base-model "
                    f"step-{BASE_MODEL_STEP} cell and can never be a banked copy. Move all three "
                    f"files aside or pick another --out-dir. Nothing was evaluated."
                )
            _verify_banked_copy(target, plan=plan, expected_identity=expected_bank)
            continue
        _peek_existing_trace(
            target,
            expected={
                **_cell_identity_without_config(plan, target, sections=sections),
                **(extra_identity or {}),
            },
        )
    for target in complete:
        logger.info(
            f"step {target.step}: {target.out_path} is complete (summary present) and this "
            f"cell's; skipping it"
        )
    return tuple(target for target in targets if target not in complete)


def _refuse_checkpoints_on_hosted_backends(
    backend_kind: str, targets: Sequence[EvalTarget]
) -> None:
    """Refuse to point a hosted endpoint at a local adapter, which it has no way to load."""
    if backend_kind not in backend_cli.HOSTED_KINDS:
        return
    checkpoints = [str(target.checkpoint) for target in targets if target.checkpoint is not None]
    if checkpoints:
        raise ValueError(
            f"--backend {backend_kind} is a hosted endpoint and cannot load local LoRA "
            f"checkpoints {checkpoints}; use --backend hf (or vllm), or evaluate a hosted model "
            f"via --model."
        )


@dataclass(frozen=True)
class TrainedGames:
    """The games a plan's checkpoints trained on, and which derivation said so."""

    ids: tuple[str, ...]
    source: str


def _warn_about_games_the_corpus_never_held(
    held: Sequence[str], *, allowed: Sequence[str], arm: str
) -> None:
    """Say which of the arm's allowed games its corpus turned out to hold no rows for.

    Not a refusal: `games.breadth_corpus` drops a game group whose kept share of its pair quota fell
    below the floor, so a corpus short of the arm's allowance is a designed outcome of selection. What
    it must not be is quiet, because the dropped game becomes a transfer measurement in every table
    keyed on the trained column, and a reader comparing this trace against the arm's registry entry
    would otherwise find the two disagreeing with nothing on record saying why.
    """
    dropped = [game_id for game_id in allowed if game_id not in set(held)]
    if not dropped:
        return
    logger.warning(
        f"arm {arm!r} allows {list(allowed)} but the corpus it trained on held no rows for "
        f"{dropped}, so this trace marks them trained_game=False, which is what they are: the "
        f"selection dropped them whole. They are transfer measurements here, and the arm's registry "
        f"entry naming them is not evidence that they trained."
    )


def _resolve_trained_games(
    args: argparse.Namespace, facts: RunFacts, arm: str, targets: Sequence[EvalTarget]
) -> TrainedGames:
    """Name the games these checkpoints actually trained on, or refuse to guess.

    Most arms train exactly one game, so every other game in the eval's cross-game grid is a transfer
    measurement. `games.report` groups its action-rate table on `(game_id, trained_game)` and calls
    that column the never-trained marker, so a battery that stamped True on every registered game put
    most of the transfer grid in the "trained" bucket of the headline table. A breadth arm trains
    several, and reading only its lead game would put the other five in the transfer bucket, which is
    the same error in the other direction: five trained games would be read as evidence of transfer.

    Four sources, in descending order of directness, and the one used is recorded on the plan so every
    trace's meta says which: the flag; the corpus composition the run's own `run_config.json` recorded
    (`derived.corpus_composition.game_id`, the games the corpus FILE held); that record's arm game list
    (`game_id` plus `game_ids`, the games the arm ALLOWED), which covers every record written before
    the composition carried the column; and the arm registry, which is where the arm-to-game mapping is
    defined in the first place and covers records that named no game at all. The composition leads the
    allowance because a breadth arm's selection can drop a whole game group
    (`games.breadth_corpus._drop_thin_games`), and reading the allowance then stamps a game the run
    demonstrably never trained as trained -- filing the cleanest transfer evidence the run produced in
    the in-distribution bucket, which is the exact error this column exists to prevent.

    A plan whose targets carry no adapter at all trained on nothing whatever its arm label says -- that
    is the step-0 baseline, and marking its rows against the arm's game would claim training that has
    not happened yet -- so the flag aside, it short-circuits to empty before any derivation.

    An adapted target whose game none of the sources name is left empty and said so loudly, rather
    than guessed at: `games.report` derives the same fact from `ARMS` when it renders and leaves the
    transfer column unanswered under the same condition, so the honest state is "nobody knows which
    game this arm trained", visible in both places, and never a game silently promoted into the
    trained bucket.
    """
    if args.trained_games is not None:
        return TrainedGames(_parse_id_list(str(args.trained_games)), TRAINED_GAMES_FROM_FLAG)
    adapted = [str(target.checkpoint) for target in targets if target.checkpoint is not None]
    if not adapted:
        return TrainedGames((), TRAINED_GAMES_NONE_UNADAPTED)
    allowed = () if facts.game_id is None else (str(facts.game_id), *facts.game_ids)
    if facts.corpus_game_ids:
        _warn_about_games_the_corpus_never_held(facts.corpus_game_ids, allowed=allowed, arm=arm)
        return TrainedGames(facts.corpus_game_ids, TRAINED_GAMES_FROM_CORPUS_COMPOSITION)
    if allowed:
        return TrainedGames(allowed, TRAINED_GAMES_FROM_ARM_ALLOWED)
    registered = ARMS.get(arm)
    if registered is not None:
        return TrainedGames(arm_game_ids(registered), TRAINED_GAMES_FROM_ARM_REGISTRY)
    logger.warning(
        f"arm {arm!r} is not in games.arms.ARMS, no --trained-games was given, and no "
        f"{RUN_CONFIG_FILENAME} beside the run records game_id, so every row of this trace is "
        f"marked trained_game=False even though {adapted} carry adapters. Pass --trained-games "
        f"<game-id> before reading the cross-game grid off it."
    )
    return TrainedGames((), TRAINED_GAMES_UNRESOLVED)


def _default_out_dir(arm: str, backend_kind: str) -> Path:
    """Choose where traces land when no `--out-dir` says: per arm, and per backend for the mock.

    A mock smoke sharing the arm's real trace directory did two things, both silent. Its plumbing
    rows rendered into the arm's `report.md`, where nothing marks a trace as mock and the
    deliberately unparseable completion reads as a real termination failure. And
    the existing-trace check then blocked the later real eval of those same steps, calling the
    plumbing file paid-for GPU output -- so the smoke had to be deleted before the measurement
    could run, by someone who trusted that message.
    """
    if backend_kind == MOCK_BACKEND_KIND:
        return DEFAULT_EVAL_ROOT / arm / MOCK_TRACE_SUBDIR
    return DEFAULT_EVAL_ROOT / arm


def resolve_plan(args: argparse.Namespace) -> EvalPlan:
    """Turn the parsed flags into a fully-checked plan, before any model or tokenizer loads."""
    _require_exactly_one_mode(args)
    if args.model is not None:
        facts = EMPTY_RUN_FACTS
        run_dir = None
        step = args.step if args.step is not None else BASE_MODEL_STEP
        specs = [_TargetSpec(step, str(args.model), None)]
    elif args.checkpoint is not None:
        checkpoint = cast("Path", args.checkpoint)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"--checkpoint {checkpoint} is not a directory.")
        run_dir = checkpoint.parent
        facts = _read_run_facts(run_dir)
        step = args.step if args.step is not None else checkpoint_step(checkpoint)
        specs = [_TargetSpec(step, _base_model_for(args, checkpoint), checkpoint)]
    else:
        run_dir = cast("Path", args.run_dir)
        facts = _read_run_facts(run_dir)
        specs = _run_dir_specs(args, run_dir, facts)

    arm = _resolve_arm(args, facts)
    thinking = _resolve_thinking(args, facts)
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
        for spec in specs
    )
    _refuse_checkpoints_on_hosted_backends(args.backend, targets)
    trained = _resolve_trained_games(args, facts, arm, targets)
    return EvalPlan(
        arm=arm,
        thinking=thinking,
        out_dir=out_dir,
        run_dir=run_dir,
        targets=targets,
        trained_game_ids=trained.ids,
        trained_games_source=trained.source,
        grading=facts.grading,
        executed_estimator=facts.executed_estimator,
    )


@dataclass(frozen=True)
class TemplateFacts:
    """What this checkpoint's own chat template says, measured by rendering it.

    Both fields are read off one tokenizer load because they answer the same question -- what text
    the model actually sees -- and because `games.preflight` states the invariant they enforce:
    whatever training derived about `<think>` and `reasoning_effort` has to be what evaluation uses.
    """

    prefilled_think: bool
    chat_template_kwargs: tuple[tuple[str, str], ...]


def _resolve_template_facts(
    args: argparse.Namespace, *, thinking: bool, template_source: str
) -> TemplateFacts:
    """Measure the template facts the battery needs, rather than assuming either of them.

    `prefilled_think`: the defaults move non-monotonically across the Qwen ladder and the
    thinking-off runs render an empty `<think></think>` pair into the prompt, so the only
    trustworthy source is rendering the template (`derive_prefilled_think`). Backends with no local
    template fall back to False with a loud note, because their completions arrive as plain text.

    `chat_template_kwargs`: whatever `resolve_chat_template_kwargs` pinned at training. Today that
    is `reasoning_effort="medium"` on Qwen3.8-27B alone, whose template otherwise prepends an
    unauthored "Reasoning effort is set to xhigh" system message -- so the 27B arm would be
    evaluated on different prompt text than it trained on, which is the one thing the arms must not
    differ in. Empty for every other checkpoint on the ladder, and for a hosted transport, which
    renders no template at all.
    """
    if args.backend not in backend_cli.LOCAL_KINDS:
        logger.warning(
            f"--backend {args.backend} has no local chat template to measure from; assuming "
            f"prefilled_think=False and no pinned template kwargs. Pass --prefilled-think if "
            f"completions open mid-thought."
        )
        return TemplateFacts(
            prefilled_think=bool(args.prefilled_think)
            if args.prefilled_think is not None
            else False,
            chat_template_kwargs=(),
        )
    tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(template_source))
    prefilled = (
        bool(args.prefilled_think)
        if args.prefilled_think is not None
        else derive_prefilled_think(tokenizer, enable_thinking=thinking)
    )
    pinned = tuple(sorted(resolve_chat_template_kwargs(tokenizer).items()))
    logger.info(
        f"template facts measured from {template_source} at {thinking=}: "
        f"prefilled_think={prefilled} chat_template_kwargs={dict(pinned)}"
    )
    return TemplateFacts(prefilled_think=prefilled, chat_template_kwargs=pinned)


def _sampling_meta(args: argparse.Namespace, *, thinking: bool) -> dict[str, Any]:
    """Record the decoding knobs the chosen backend will actually honour, for the meta record."""
    if args.backend in backend_cli.LOCAL_KINDS:
        resolved = backend_cli.local_sampling_from_args(
            args, eval_sampling(resolve_sampler_mode(args), thinking=thinking)
        )
        return asdict(resolved)
    if args.backend == "bedrock":
        return asdict(backend_cli.bedrock_sampling_from_args(args))
    if args.backend == "codex":
        return {"reasoning_effort": args.reasoning_effort}
    return {}


def engine_seed(arm: str, step: int, *, backend_kind: str, config: EvalConfig) -> int | None:
    """Derive the vLLM engine seed for one cell, so different cells sample different streams.

    vLLM's ``EngineArgs.seed`` defaults to 0 and nothing here overrode it, so every engine load
    replayed one RNG stream: two cells serving the same weights over the same prompt list came back
    byte-identical (measured 8/8 across processes), which turned the paired step-0 cells -- the
    battery's test-retest noise floor -- into copies of one draw. Deriving the seed from the cell's
    own identity keeps a cell reproducible under relaunch while giving distinct cells distinct
    streams. CRC32 rather than ``hash()`` because the latter is salted per process, and masked to
    31 bits because vLLM forwards the value to seeders with int32 ceilings. Only the vLLM path is
    seeded: the HF backend draws from torch's global RNG, which this function does not own.

    The identity is the arm, the step, the print orders rendered and the batch width REQUESTED. The
    last two joined it when the battery gained a print-order flag: two cells that render different
    prompts or hand the engine different widths are different measurements, so they get different
    streams. Requested rather than effective width on purpose -- the effective one is derived from
    whatever VRAM the run landed on, so hashing it would make the seed, and therefore the cell,
    unreproducible on another card.
    """
    if backend_kind != "vllm":
        return None
    identity = (
        f"{arm}:step-{step}:orders-{','.join(config.label_print_orders)}:batch-{config.batch_size}"
    )
    return zlib.crc32(identity.encode()) & 0x7FFFFFFF


def _target_meta(
    target: EvalTarget,
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    served: ServedModel,
    seed: int | None,
) -> dict[str, Any]:
    """Assemble the caller-owned meta fields; the battery adds sha, timestamp, and config itself.

    `grading` is the ARM's grading rule, read from the run's own config. It is not the same thing as
    the per-row `render_grading` column, which names what rendered the prompt corpus: one game maps
    to one rendering, and `twin-pd-group` and `twin-pd-self` share `twin-pd`, so a row's grading
    cannot name the arm and this field is the only place the arm's own rule appears.

    `served.provenance` carries how the weights were assembled -- `model_load_mode` and whether the
    stored weights could represent the trained delta at all. It belongs in every trace for the same
    reason `vllm_quantization` does: a checkpoint served through a bf16 merge and the same
    checkpoint served un-merged are not the same measurement, the merge attenuates effect sizes by
    an amount that varies per module, and no reader could recover which they were looking at
    afterwards. See :mod:`games.eval_model`.
    """
    return {
        **served.provenance,
        "arm": plan.arm,
        "step": target.step,
        "thinking": plan.thinking,
        "base_model_id": target.base_model,
        "checkpoint": None if target.checkpoint is None else str(target.checkpoint),
        "run_dir": None if plan.run_dir is None else str(plan.run_dir),
        "backend_kind": args.backend,
        # Branded like the mock prefix: an online-quantized engine's trace must never read as a
        # bf16 measurement, and nothing downstream could recover the difference otherwise.
        "vllm_quantization": args.vllm_quantization,
        "sampler_mode": sampler_mode_meta(args),
        "sampling": _sampling_meta(args, thinking=plan.thinking),
        "engine_seed": seed,
        "grading": plan.grading,
        # The aggregation the training run EXECUTED, read from run_config.json (games/train.py
        # writes it, grpo/estimator_defaults.py derives it); None for pre-2026-08-20 records.
        "executed_estimator": plan.executed_estimator,
        # Which derivation named `eval_config.trained_game_ids` (`_resolve_trained_games`). The set
        # alone cannot say whether it is the games the run's corpus held or the wider set its arm
        # allowed, and on a breadth arm those differ by whichever game group selection dropped.
        "trained_games_source": plan.trained_games_source,
    }


@dataclass
class IntervalSync:
    """Push the trace directory to S3 while a cell runs, at most once per interval, and once at the end.

    The hook `run_eval_battery` calls after every appended batch is `maybe_sync`; the driver calls
    `sync_now` after each step's summary lands. Upload failures log and the cell continues -- the
    next interval retries, and the run is the expensive thing -- which is `games.s3_sync.sync_directory`'s
    own policy. `clock` and `sync` are injectable so the cadence can be tested with no S3 and no
    waiting. The upload runs synchronously from inside the drain loop, so the engine is not stepped
    while it runs: seconds per interval for a 177 MB trace from EC2, nothing at the default ten
    minutes, and the reason to think before shortening the interval much.
    """

    local_dir: Path
    s3_dest: str
    interval_seconds: float
    clock: Callable[[], float] = time.monotonic
    # None means `games.s3_sync.sync_directory`, looked up at call time so a test can stand in for it.
    sync: Callable[[Path, str], SyncOutcome] | None = None
    last_synced_at: float | None = field(default=None, init=False)
    outcomes: list[SyncOutcome] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Refuse a cadence that would sync on every record or never."""
        if not self.interval_seconds > 0:
            raise ValueError(
                f"--sync-interval-seconds must be positive, got {self.interval_seconds}."
            )

    def maybe_sync(self, records_on_disk: int) -> None:
        """Sync if the interval has elapsed since the last sync (or none has happened yet)."""
        now = self.clock()
        if self.last_synced_at is not None and now - self.last_synced_at < self.interval_seconds:
            return
        self.sync_now(reason=f"{records_on_disk} records on disk")

    def sync_now(self, *, reason: str) -> SyncOutcome:
        """Sync unconditionally and remember when, so the next interval counts from here."""
        run_sync = sync_directory if self.sync is None else self.sync
        outcome = run_sync(self.local_dir, self.s3_dest)
        self.last_synced_at = self.clock()
        self.outcomes.append(outcome)
        if outcome.succeeded:
            logger.info(f"trace directory synced to {self.s3_dest} ({reason})")
        else:
            logger.error(
                f"trace directory sync to {self.s3_dest} failed ({reason}); the cell continues and "
                f"the next interval retries: {outcome}"
            )
        return outcome


def _cell_identity_without_config(
    plan: EvalPlan, target: EvalTarget, *, sections: Sequence[str]
) -> dict[str, Any]:
    """Name the identity fields known before the tokenizer loads: what `_peek_existing_trace` checks.

    Arm, step, thinking mode, base model and section list are all resolved from the flags and the
    run's own config; only `eval_config` waits on the chat template (`prefilled_think`), so this is
    the part of `_cell_identity` the up-front pass over every target can compare.
    """
    return {
        "arm": plan.arm,
        "step": target.step,
        "thinking": plan.thinking,
        "base_model_id": target.base_model,
        "sections": list(sections),
    }


def _cell_identity(
    plan: EvalPlan, target: EvalTarget, *, sections: Sequence[str], config: EvalConfig
) -> dict[str, Any]:
    """Name the meta fields an existing trace must agree with before an engine loads for it.

    What was asked, of which arm and step: enough to refuse a trace from another configuration
    without paying an engine load. The serving-level fields (backend kind, sampling, engine seed,
    load mode) are checked by `run_eval_battery` itself against the full meta once the backend is
    resolved -- and skipped entirely under --summarise-only, which resolves no backend.
    """
    return {
        **_cell_identity_without_config(plan, target, sections=sections),
        "eval_config": config.as_record(),
    }


def _write_summary(target: EvalTarget, summary: Mapping[str, Any]) -> Path:
    """Write the step's summary beside its trace, atomically; the file is the runner's completion marker.

    Through a temp file, an fsync and one ``os.replace``, because a death mid-write would otherwise
    leave a torn summary that reads as "complete" to every later launch, and the marker for a
    complete cell has to be either there and whole or not there.
    """
    summary_path = summary_path_for(target.out_path)
    tmp = summary_path.with_name(summary_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(summary_path)
    logger.info(f"step {target.step} done, {summary_path=} summary: {json.dumps(summary)}")
    return summary_path


@dataclass(frozen=True)
class CellBank:
    """Where the shared base-model step-0 cells live and what this run may do with them.

    `prefix` is the operator's s3:// root (trailing slash normalised); one entry per `bank_key`
    lives at `entry_prefix(key)`. `publish` is `--bank-base-cells`. `noise_floor_framings` names
    the sweep cell that stays per arm (`is_noise_floor_cell`).
    """

    prefix: str
    publish: bool
    noise_floor_framings: tuple[str, ...]

    def entry_prefix(self, key: str) -> str:
        """Return the s3:// prefix one bank entry lives under."""
        return f"{self.prefix}{key}/"


def _parse_bank(args: argparse.Namespace) -> CellBank | None:
    """Turn the three bank flags into a `CellBank`, refusing the combinations that do nothing.

    `--bank-base-cells` without a bank to publish into, `--noise-floor-framings` without one (it
    names which cell stays out of a bank, so with no bank it selects nothing), and any of the three
    under `--summarise-only` (which generates nothing, so there is no cell to take or to publish),
    are refused rather than ignored: a flag that is accepted and does nothing is how a kit comes to
    believe it banked a cell. The framing list is parsed and checked against the registry before
    the bank prefix is looked at, so a misspelled framing raises whatever else was passed with it
    rather than being dropped on the floor with the flag it rode in on.
    """
    framings = (
        NOISE_FLOOR_FRAMINGS
        if args.noise_floor_framings is None
        else _parse_id_list(str(args.noise_floor_framings))
    )
    unknown = sorted(set(framings) - set(COUNTERPART_FRAMING_IDS))
    if unknown:
        raise ValueError(
            f"--noise-floor-framings names {unknown}, which are not registered counterpart "
            f"framings; registered: {list(COUNTERPART_FRAMING_IDS)}."
        )
    prefix = args.banked_base_cells
    if prefix is None:
        if args.bank_base_cells:
            raise ValueError(
                "--bank-base-cells publishes into the bank named by --banked-base-cells, and no "
                "bank was named; pass both or neither."
            )
        if args.noise_floor_framings is not None:
            raise ValueError(
                "--noise-floor-framings names the step-0 cell that stays out of the bank named by "
                "--banked-base-cells, and no bank was named, so it would select nothing; pass both "
                "or neither."
            )
        return None
    if args.summarise_only:
        raise ValueError(
            "--summarise-only generates no cell, so --banked-base-cells has nothing to take from "
            "the bank or publish into it; drop the bank flags for a summary-only salvage."
        )
    prefix = str(prefix)
    if not prefix.startswith("s3://"):
        raise ValueError(f"--banked-base-cells must be an s3:// prefix, got {prefix!r}.")
    if not prefix.endswith("/"):
        prefix += "/"
    if not framings:
        logger.warning(
            "--noise-floor-framings is empty: every step-0 cell of this run may come from the bank, "
            "so this arm keeps no per-arm step-0 draw as its test-retest noise floor"
        )
    return CellBank(
        prefix=prefix, publish=bool(args.bank_base_cells), noise_floor_framings=framings
    )


def is_noise_floor_cell(
    sections: Sequence[str], config: EvalConfig, noise_floor_framings: Sequence[str]
) -> bool:
    """Whether this cell is the per-arm step-0 noise floor: a framing sweep over only the floor framings.

    Every framing the cell renders has to be in the list, not merely one of them: a cell that
    sweeps the unstated framing alongside others is one of the large instruments the bank exists
    for, and the small per-arm draw the design keeps is the cell that renders nothing else. A cell
    without the sweep section is never the floor.
    """
    if SECTION_FRAMING_SWEEP not in sections or not config.counterpart_framings:
        return False
    return set(config.counterpart_framings) <= set(noise_floor_framings)


def bank_identity_without_config(
    args: argparse.Namespace, plan: EvalPlan, target: EvalTarget, *, sections: Sequence[str]
) -> dict[str, Any]:
    """Name the part of a step-0 cell's bank identity known before the chat template is measured.

    What `_verify_banked_copy` can hold an existing copy to up front: which base, which thinking
    mode, which sections, which backend and quantization, which sampler mode and every sampling
    knob the backend will honour (`_sampling_meta`, which carries `max_new_tokens`). The arm is
    deliberately absent -- sharing across arms is the point -- and so is the engine seed, which is
    derived from the arm and recorded in the entry's manifest rather than keyed on.
    """
    return {
        "base_model_id": target.base_model,
        "step": target.step,
        "thinking": plan.thinking,
        "sections": list(sections),
        "backend_kind": args.backend,
        "vllm_quantization": args.vllm_quantization,
        "sampler_mode": sampler_mode_meta(args),
        "sampling": _sampling_meta(args, thinking=plan.thinking),
    }


def prompt_set_digest(plan: Sequence[PlannedRequest]) -> str:
    """One sha256 over every prompt the plan renders, with its section and identity, in plan order.

    The item text behind the survey and DTBench sections lives outside version control, so two
    machines can build the same `EvalConfig` over different items; the digest is what tells them
    apart. The count is folded in so a plan cannot collide with a prefix of another.
    """
    digest = hashlib.sha256()
    for request in plan:
        digest.update(
            json.dumps(
                [request.section, list(request.identity), request.prompt],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    digest.update(f"n={len(plan)}".encode())
    return digest.hexdigest()


def bank_identity(
    args: argparse.Namespace,
    plan: EvalPlan,
    target: EvalTarget,
    *,
    sections: Sequence[str],
    config: EvalConfig,
) -> dict[str, Any]:
    """Everything a step-0 cell's measurement depends on, as JSON-safe values; `bank_key` hashes it.

    `bank_identity_without_config` plus the whole eval config as recorded on the meta (samples,
    print orders, capability seed, survey selection, prefilled_think, batch width, template kwargs,
    framings, sweep games) minus the machine-local item paths (`RESUME_LOCAL_PATH_CONFIG_FIELDS`),
    which the prompt-set digest covers by content instead. Two cells whose identities differ in any
    field are different measurements and never share an entry; two whose identities agree are the
    same draw of the same base under the same prompts and sampler, which is what a shared entry
    claims. Over-keying costs a regeneration; under-keying reports another cell's numbers.
    """
    portable_config = {
        name: value
        for name, value in config.as_record().items()
        if name not in RESUME_LOCAL_PATH_CONFIG_FIELDS
    }
    return {
        **bank_identity_without_config(args, plan, target, sections=sections),
        "eval_config": portable_config,
        "prompt_set_digest": prompt_set_digest(plan_battery(sections, config)),
    }


def bank_key(identity: Mapping[str, Any]) -> str:
    """Return the bank entry a cell of this identity lives under: sha256 of its canonical JSON."""
    canonical = json.dumps(
        _json_native(identity), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _identity_drift(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """Name every field of ``expected`` that ``stored`` disagrees with, compared through JSON types."""
    return [
        f"{name}: {_drift_text(stored.get(name, '<absent>'))} banked, {_drift_text(value)} asked"
        for name, value in expected.items()
        if _json_native(stored.get(name)) != _json_native(value)
    ]


def _drift_text(value: Any) -> str:  # noqa: ANN401 - any identity value, rendered for a refusal
    """Render one side of a drifted identity field short enough for a log line."""
    rendered = json.dumps(_json_native(value), sort_keys=True, default=str)
    limit = 160
    return rendered if len(rendered) <= limit else rendered[:limit] + "..."


def _file_facts(path: Path) -> dict[str, Any]:
    """Return the sha256 and size a manifest or sidecar records for one copied file."""
    return {"sha256": sha256_of_file(path), "bytes": path.stat().st_size}


def bank_object_name(local_name: str, tag: str) -> str:
    """Name the bank object a cell file is published as: `step-0.jsonl` -> `step-0.<tag>.jsonl`.

    `tag` is the first `BANK_OBJECT_TAG_CHARS` of the trace's sha256, shared by the trace and its
    summary so one publish reads as a pair, and the manifest records the object each local file
    lives under. The names are content-tagged rather than fixed because the bank has no lock: two
    arms finishing step 0 minutes apart each re-read the entry, each see no manifest, and each
    upload. Under fixed names every object is last-writer-wins on its own, so the manifest that
    survived could record one publisher's hashes over the other publisher's trace, and every later
    consumer would refuse the entry until a human deleted it. Two publishers never share a tagged
    name (a shared tag means identical trace bytes, hence an identical summary), so whichever
    manifest lands last names objects that exist and hash as it recorded them.
    """
    head, _, rest = local_name.partition(".")
    return f"{head}.{tag}.{rest}"


def _consumer_meta(plan: EvalPlan) -> dict[str, Any]:
    """Return the arm-descriptive meta fields a banked copy carries for the arm that took it.

    `_target_meta` reads these off the arm's own plan rather than off the cell's generation, so a
    copy whose meta named the producer's run dir and grading would misdescribe the consuming arm in
    every reader that pools them per arm: the readout's arm facts render `grading`, and the prime
    sharing pair, twin-pd-group and twin-pd-self, differ in exactly that field. Recorded in the
    sidecar and overlaid by `games.report.attribute_trace` (`BANKED_CONSUMER_META_FIELDS` is the
    reader's list of them); the generation facts -- git sha, engine seed, written_at, sampler -- stay
    the producer's, because they describe how the bytes were made.
    """
    return {
        "run_dir": None if plan.run_dir is None else str(plan.run_dir),
        "grading": plan.grading,
        "executed_estimator": plan.executed_estimator,
    }


def _write_json_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document through a temp file and one replace, so a death mid-write leaves none."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def _bank_staging_dir(target: EvalTarget) -> Path:
    """Make a fresh sibling of the out dir to land a bank entry in, removing a dead run's leftover."""
    staging = target.out_path.parent.with_name(target.out_path.parent.name + BANK_STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def _staged_bank_files(
    manifest: Mapping[str, Any], staging: Path, *, names: Sequence[str], entry: str
) -> dict[str, Path]:
    """Locate each of the entry's files under the object the manifest names for it, hash-checked.

    Refuses a manifest that names no object for a file, an object that is not under the fetched
    entry (a torn publish), and an object whose bytes hash to something other than the manifest
    recorded (a manifest describing another publisher's files). Returns the staged path per local
    file name.
    """
    files = dict(manifest.get("files") or {})
    staged: dict[str, Path] = {}
    for name in names:
        object_name = (files.get(name) or {}).get("object")
        if not object_name:
            raise ValueError(
                f"{entry}: the manifest names no bank object for {name} (files: {sorted(files)}), "
                f"so nothing under this key can be matched to a recorded hash."
            )
        staged[name] = staging / str(object_name)
        if not staged[name].is_file():
            raise ValueError(
                f"{entry}: the manifest says the entry is complete but {object_name} ({name}) is "
                f"not under it (present: {sorted(path.name for path in staging.iterdir())}); the "
                f"entry is torn and a human has to look before anything takes it."
            )
        recorded = files[name].get("sha256")
        actual = sha256_of_file(staged[name])
        if actual != recorded:
            raise ValueError(
                f"{entry}: {object_name} ({name}) hashes to {actual} but the manifest recorded "
                f"{recorded!r}; the entry's files and its manifest disagree, so nothing under it "
                f"can be trusted."
            )
    return staged


def _verify_bank_entry(  # noqa: PLR0913 - one keyword per thing the driver resolved for the bank
    manifest: Mapping[str, Any],
    staging: Path,
    *,
    target: EvalTarget,
    identity: Mapping[str, Any],
    key: str,
    plan: EvalPlan,
    sections: Sequence[str],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    """Refuse a fetched bank entry that is not the complete, consistent cell its key promises.

    Six things have to hold before a byte of it reaches the run's out dir, each a way the copy could
    silently report another measurement: the manifest is a bank manifest for this key; its recorded
    identity equals the one this run derived, through JSON types (the key is a hash of it, so a
    disagreement is a collision or a hand-placed entry); the objects the manifest names for the
    trace and the summary are both there and hash to what it recorded; the trace passes the resume
    gate itself (`inspect_trace`, against this cell's full identity with the manifest's source arm
    in place of this arm: every meta field including the eval config, and every record an identity
    this cell's plan renders, none twice); the trace is complete, with no planned record missing;
    and the summary equals the trace's own rebuild, so the entry's two files describe one cell.
    Returns the trace's records, which the caller has no further use for but which prove the trace
    parsed end to end.
    """
    entry = f"bank entry {key}"
    if manifest.get("record") != BANK_MANIFEST_RECORD:
        raise ValueError(
            f"{entry}: {BANK_MANIFEST_FILENAME} is not a {BANK_MANIFEST_RECORD!r} record "
            f"(record={manifest.get('record')!r}); nothing under this key can be trusted."
        )
    if manifest.get("bank_key") != key:
        raise ValueError(
            f"{entry}: the manifest names key {manifest.get('bank_key')!r}, so the entry was "
            f"published under a key it was not derived for; nothing under it can be trusted."
        )
    drifted = _identity_drift(dict(manifest.get("bank_identity") or {}), identity)
    if drifted:
        raise ValueError(
            f"{entry}: the banked cell's identity is not this cell's although the keys agree -- "
            f"{'; '.join(drifted)}. Taking it would report another measurement's numbers under "
            f"this arm's name; the entry was hand-placed or the key derivation changed. Nothing "
            f"was copied."
        )
    trace_name = target.out_path.name
    summary_name = summary_path_for(target.out_path).name
    staged = _staged_bank_files(manifest, staging, names=(trace_name, summary_name), entry=entry)
    battery_plan = plan_battery(sections, config)
    expected_meta = {
        **_cell_identity(plan, target, sections=sections, config=config),
        "arm": manifest.get("source_arm"),
    }
    try:
        inspection = inspect_trace(staged[trace_name], battery_plan, expected_meta=expected_meta)
    except ValueError as error:
        raise ValueError(
            f"{entry}: the banked trace is not this cell's although the manifest says it is "
            f"({error}). Nothing was copied."
        ) from error
    pending = inspection.pending(battery_plan)
    if pending:
        raise ValueError(
            f"{entry}: the banked trace holds {len(inspection.done)} of {len(battery_plan)} planned "
            f"records ({len(pending)} missing) although the manifest says it is complete; copied, "
            f"it would report a partial cell as this arm's step {target.step}. Nothing was copied."
        )
    records = read_eval_records(staged[trace_name])
    summary = json.loads(staged[summary_name].read_text(encoding="utf-8"))
    if _json_native(summary) != _json_native(summarise_trace(records)):
        raise ValueError(
            f"{entry}: {summary_name} is not the rebuild of {trace_name} beside it, so the entry's "
            f"summary describes another trace. Nothing was copied."
        )
    return records


def _reuse_banked_cell(  # noqa: PLR0913 - one keyword per thing the driver resolved for the bank
    target: EvalTarget,
    *,
    bank: CellBank,
    identity: Mapping[str, Any],
    key: str,
    plan: EvalPlan,
    sections: Sequence[str],
    config: EvalConfig,
    sync: IntervalSync | None,
) -> bool:
    """Copy the bank's cell for this key into the run, or say there is none; never generate.

    The entry is fetched into a sibling staging directory (`restore_directory`, which raises when
    the bank cannot be read: paying for the cell again because S3 blinked is the silent failure this
    flag exists to remove), verified (`_verify_bank_entry`), and only then moved into place under
    the cell's own file names: trace, summary, and the provenance sidecar last, since the sidecar
    is what tells every later reader and every relaunch that the two files beside it are a banked
    copy. Returns False when the bank holds no manifest under the key, which is the first arm on
    this base arriving.
    """
    entry_prefix = bank.entry_prefix(key)
    staging = _bank_staging_dir(target)
    try:
        restore_directory(entry_prefix, staging)
        manifest_path = staging / BANK_MANIFEST_FILENAME
        if not manifest_path.exists():
            stray = sorted(path.name for path in staging.iterdir())
            logger.info(
                f"step {target.step}: no banked cell at {entry_prefix} "
                f"({len(stray)} file(s) without a manifest: {stray}); generating it"
            )
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = _verify_bank_entry(
            manifest,
            staging,
            target=target,
            identity=identity,
            key=key,
            plan=plan,
            sections=sections,
            config=config,
        )
        summary_path = summary_path_for(target.out_path)
        files = dict(manifest["files"])
        target.out_path.parent.mkdir(parents=True, exist_ok=True)
        (staging / str(files[target.out_path.name]["object"])).replace(target.out_path)
        (staging / str(files[summary_path.name]["object"])).replace(summary_path)
        now = datetime.now(UTC).isoformat()
        _write_json_atomically(
            banked_provenance_path(target.out_path),
            {
                "record": BANKED_PROVENANCE_RECORD,
                "consumer_arm": plan.arm,
                "consumer_meta": _consumer_meta(plan),
                "step": target.step,
                "bank_prefix": bank.prefix,
                "bank_key": key,
                "source_key": entry_prefix,
                "source_arm": manifest.get("source_arm"),
                "source_git_sha": manifest.get("source_git_sha"),
                "source_written_at": manifest.get("source_written_at"),
                "source_engine_seed": manifest.get("source_engine_seed"),
                "source_run_dir": manifest.get("source_run_dir"),
                "source_grading": manifest.get("source_grading"),
                "source_executed_estimator": manifest.get("source_executed_estimator"),
                "banked_at": manifest.get("banked_at"),
                "banked_by_git_sha": manifest.get("banked_by_git_sha"),
                "copied_at": now,
                "copied_by_git_sha": git_sha(),
                "files": manifest.get("files"),
                "bank_identity": _json_native(identity),
            },
        )
    finally:
        shutil.rmtree(staging)
    logger.info(
        f"step {target.step}: took the banked cell {entry_prefix} ({len(records) - 1} records, "
        f"generated under arm {manifest.get('source_arm')!r} at {manifest.get('source_git_sha')}) "
        f"into {target.out_path}; nothing generated"
    )
    if sync is not None:
        sync.sync_now(reason=f"step {target.step} copied from the bank")
    return True


def bank_publish_record_path(out_path: Path) -> Path:
    """Where the record of what `--bank-base-cells` did with a cell lives: beside it, `.bank-publish.json`."""
    return out_path.with_name(out_path.stem + BANK_PUBLISH_RECORD_SUFFIX)


def _record_bank_publish(  # noqa: PLR0913 - one keyword per fact the record has to carry
    target: EvalTarget,
    *,
    entry_prefix: str,
    key: str,
    outcome: str,
    detail: str | None = None,
    failed_sync: SyncOutcome | None = None,
) -> None:
    """Record beside the cell what publishing it into the bank did, on every exit path.

    `--bank-base-cells` exits 0 whatever happens to the upload, because the twelve-hour cell it just
    generated is the expensive thing and is safe on disk either way. That policy is right and makes
    the run's own log the only place a failure appears, which a relaunch on another box cannot read
    and a kit runner reading exit codes never sees. So the outcome lands in the cell's directory,
    which `--sync-dest` ships with the trace: afterwards "did this arm's step-0 reach the bank" is a
    question about an artifact rather than about a log nobody kept.
    """
    _write_json_atomically(
        bank_publish_record_path(target.out_path),
        {
            "record": BANK_PUBLISH_RECORD,
            "outcome": outcome,
            "detail": detail,
            "step": target.step,
            "trace": target.out_path.name,
            "bank_key": key,
            "entry_prefix": entry_prefix,
            "written_at": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "failed_sync_command": None if failed_sync is None else " ".join(failed_sync.command),
            "failed_sync_returncode": None if failed_sync is None else failed_sync.returncode,
            "failed_sync_failure_reason": (
                None if failed_sync is None else failed_sync.failure_reason
            ),
            "failed_sync_skipped_reason": (
                None if failed_sync is None else failed_sync.skipped_reason
            ),
        },
    )


def _rival_manifest_faults(
    manifest: Mapping[str, Any], staging: Path, *, key: str, names: Sequence[str]
) -> list[str]:
    """Name every way a rival publisher's manifest fails to be an entry a consumer could take.

    The publish-time counterpart of `_verify_bank_entry`, and deliberately weaker: it collects
    faults instead of raising, and asks only what a rival's manifest has to get right for the key to
    be usable at all -- that it is a bank manifest, that it names this key, and that it points at an
    object per payload file which is actually under the entry. Whether the cell it names is really
    this cell is the consumer's question, checked in full when someone takes it.
    """
    faults: list[str] = []
    if manifest.get("record") != BANK_MANIFEST_RECORD:
        faults.append(
            f"{BANK_MANIFEST_FILENAME} is not a {BANK_MANIFEST_RECORD!r} record "
            f"(record={manifest.get('record')!r})"
        )
    if manifest.get("bank_key") != key:
        faults.append(f"it names bank key {manifest.get('bank_key')!r} rather than {key!r}")
    files = dict(manifest.get("files") or {})
    present = sorted(path.name for path in staging.iterdir())
    for name in names:
        object_name = (files.get(name) or {}).get("object")
        if not object_name:
            faults.append(f"it names no bank object for {name} (files: {sorted(files)})")
        elif not (staging / str(object_name)).is_file():
            faults.append(
                f"the object it names for {name}, {object_name}, is not under the entry "
                f"(present: {present})"
            )
    return faults


def _publish_banked_cell(
    target: EvalTarget,
    *,
    bank: CellBank,
    identity: Mapping[str, Any],
    key: str,
    sync_dest: str | None,
) -> None:
    """Publish this run's freshly generated step-0 cell under its key, unless another run got there first.

    The bank is re-read before uploading, because two arms on one base can generate concurrently and
    the reuse check ran hours ago: an entry already complete is kept and this copy is not published
    over it (both are valid draws, and every consumer that took the first has its hash-verified
    copy). The re-read is a check without a lock, so the payload goes up under content-tagged names
    (`bank_object_name`) that two publishers can never share, and the manifest records which objects
    it describes. Upload order is the completion-marker invariant: trace and summary in one sync,
    the manifest in a second, so an entry with a manifest is whole. A failed upload is logged, not
    raised -- the cell itself is on disk and in the run's own prefix, and the rest of the ladder is
    the expensive thing -- and the next arm simply generates.

    A rival's manifest is checked before it is deferred to (`_rival_manifest_faults`), because a torn
    one blocks the key for every later consumer and the file's mere existence says nothing. An
    invalid one is neither overwritten nor raised over: overwriting another publisher's manifest is a
    destructive act on shared state that a human should decide, and raising would throw away a
    completed twelve-hour cell over a publish-side problem. Every exit here therefore leaves a
    `_record_bank_publish` record beside the cell, which is the only durable trace of an outcome the
    exit code cannot carry.
    """
    entry_prefix = bank.entry_prefix(key)
    if not _has_complete_summary(target.out_path):
        _record_bank_publish(
            target,
            entry_prefix=entry_prefix,
            key=key,
            outcome=BANK_PUBLISH_NO_COMPLETE_CELL,
            detail=f"{summary_path_for(target.out_path).name} is not beside the trace",
        )
        raise FileNotFoundError(
            f"step {target.step}: {target.out_path} has no summary beside it, so there is no "
            f"complete cell to publish."
        )
    summary_path = summary_path_for(target.out_path)
    payload_names = (target.out_path.name, summary_path.name)
    staging = _bank_staging_dir(target)
    try:
        restore_directory(entry_prefix, staging)
        if (staging / BANK_MANIFEST_FILENAME).exists():
            manifest = json.loads((staging / BANK_MANIFEST_FILENAME).read_text(encoding="utf-8"))
            faults = _rival_manifest_faults(manifest, staging, key=key, names=payload_names)
            if faults:
                logger.error(
                    f"step {target.step}: {entry_prefix} holds a {BANK_MANIFEST_FILENAME} that no "
                    f"consumer can take ({'; '.join(faults)}); it is left exactly as it is, this "
                    f"run's cell is not published, and a human has to clear the key"
                )
                _record_bank_publish(
                    target,
                    entry_prefix=entry_prefix,
                    key=key,
                    outcome=BANK_PUBLISH_INVALID_RIVAL,
                    detail="; ".join(faults),
                )
                return
            logger.info(
                f"step {target.step}: {entry_prefix} already holds a complete cell banked "
                f"{manifest.get('banked_at')} from arm {manifest.get('source_arm')!r}; keeping "
                f"theirs, this run's copy stays in {target.out_path.parent} only"
            )
            _record_bank_publish(
                target,
                entry_prefix=entry_prefix,
                key=key,
                outcome=BANK_PUBLISH_KEPT_RIVAL,
                detail=f"banked by arm {manifest.get('source_arm')!r} at {manifest.get('banked_at')}",
            )
            return
        tag = sha256_of_file(target.out_path)[:BANK_OBJECT_TAG_CHARS]
        objects = {
            local.name: bank_object_name(local.name, tag)
            for local in (target.out_path, summary_path)
        }
        for local_name, object_name in objects.items():
            # copyfile rather than copy2: a fresh mtime, so `aws s3 sync` never skips the upload as
            # unchanged against an object of the same size (the mtime rule the CLI applies).
            shutil.copyfile(target.out_path.with_name(local_name), staging / object_name)
        meta = read_eval_records(target.out_path)[0]
        files = {
            local_name: {"object": object_name, **_file_facts(staging / object_name)}
            for local_name, object_name in objects.items()
        }
        uploaded = sync_directory(staging, entry_prefix)
        if not uploaded.succeeded:
            logger.error(
                f"step {target.step}: publishing the cell's files to {entry_prefix} failed "
                f"({uploaded}); the bank keeps no manifest for it and the next arm generates"
            )
            _record_bank_publish(
                target,
                entry_prefix=entry_prefix,
                key=key,
                outcome=BANK_PUBLISH_FILES_FAILED,
                detail=f"the {len(objects)} payload object(s) never reached the entry",
                failed_sync=uploaded,
            )
            return
        _write_json_atomically(
            staging / BANK_MANIFEST_FILENAME,
            {
                "record": BANK_MANIFEST_RECORD,
                "bank_key": key,
                "bank_identity": _json_native(identity),
                "banked_at": datetime.now(UTC).isoformat(),
                "banked_by_git_sha": git_sha(),
                "source_arm": meta.get("arm"),
                "source_git_sha": meta.get("git_sha"),
                "source_written_at": meta.get("written_at"),
                "source_engine_seed": meta.get("engine_seed"),
                "source_run_dir": meta.get("run_dir"),
                "source_grading": meta.get("grading"),
                "source_executed_estimator": meta.get("executed_estimator"),
                "source_sync_dest": sync_dest,
                "source_n_sessions": (meta.get("resume") or {}).get("n_sessions"),
                "files": files,
            },
        )
        manifest_upload = sync_directory(staging, entry_prefix)
        if not manifest_upload.succeeded:
            logger.error(
                f"step {target.step}: the cell's files reached {entry_prefix} but its manifest did "
                f"not ({manifest_upload}); the entry is half-published and the next arm generates"
            )
            _record_bank_publish(
                target,
                entry_prefix=entry_prefix,
                key=key,
                outcome=BANK_PUBLISH_MANIFEST_FAILED,
                detail=f"{BANK_MANIFEST_FILENAME} never reached the entry its payload is under",
                failed_sync=manifest_upload,
            )
            return
        _record_bank_publish(
            target,
            entry_prefix=entry_prefix,
            key=key,
            outcome=BANK_PUBLISH_PUBLISHED,
            detail=f"objects {sorted(objects.values())}",
        )
    finally:
        shutil.rmtree(staging)
    logger.info(
        f"step {target.step}: banked {target.out_path.name} and its summary at {entry_prefix} as "
        f"{files[target.out_path.name]['object']} ({files[target.out_path.name]['bytes']} bytes, "
        f"sha256 {files[target.out_path.name]['sha256']})"
    )


def _verify_banked_copy(
    target: EvalTarget, *, plan: EvalPlan, expected_identity: Mapping[str, Any]
) -> None:
    """Hold an existing banked copy at the out path to this run's request, before anything loads.

    The counterpart of `_peek_existing_trace` for a trace whose meta names another arm on purpose.
    The sidecar's hash of the trace is checked against the file (`read_banked_provenance`), then the
    sidecar has to name this arm and step, its recorded identity has to agree with the request on
    every field known before the template loads, and the summary beside the trace has to be there
    and hash to what the sidecar recorded. A copy that fails any of these is refused with the
    paths, exactly as a foreign trace is: skipped as complete it would report another cell.
    """
    provenance = read_banked_provenance(target.out_path)
    if provenance is None:
        raise FileNotFoundError(f"no banked-copy sidecar beside {target.out_path}")
    sidecar = banked_provenance_path(target.out_path)
    drifted: list[str] = []
    if provenance.get("consumer_arm") != plan.arm:
        drifted.append(f"arm: {provenance.get('consumer_arm')!r} copied for, {plan.arm!r} asked")
    if provenance.get("step") != target.step:
        drifted.append(f"step: {provenance.get('step')!r} copied for, {target.step!r} asked")
    drifted.extend(_identity_drift(dict(provenance.get("bank_identity") or {}), expected_identity))
    summary_path = summary_path_for(target.out_path)
    recorded_summary = ((provenance.get("files") or {}).get(summary_path.name) or {}).get("sha256")
    if not summary_path.exists():
        drifted.append(
            f"{summary_path.name}: recorded in the sidecar but missing beside the trace, so the "
            f"copy is torn (delete the trace and the sidecar and relaunch; the bank still holds "
            f"the cell)"
        )
    elif sha256_of_file(summary_path) != recorded_summary:
        drifted.append(
            f"{summary_path.name}: sha256 {sha256_of_file(summary_path)} on disk, "
            f"{recorded_summary!r} recorded"
        )
    if drifted:
        raise FileExistsError(
            f"refusing to touch {target.out_path}: the banked copy {sidecar} describes is not this "
            f"cell's ({'; '.join(drifted)}), so it can be neither skipped as complete nor "
            f"continued. Move the trace, its summary and the sidecar aside or pick another "
            f"--out-dir. Nothing was evaluated."
        )


def _restore_missing_files(s3_dest: str, local_dir: Path) -> None:
    """Pull the cell's prefix down, adding only the files ``local_dir`` lacks; a local file always wins.

    `restore_directory` is ``aws s3 sync``, which replaces any local file whose size differs from
    the object's. That is exactly wrong on the box a cell died on: its trace holds every record
    appended since the last interval upload and is LARGER than the S3 copy, so a plain restore
    would roll it back to the upload and the resume would regenerate those records -- silently, for
    up to one interval, and it is the "records are kept, never regenerated" property the resume is
    for. So the restore lands in a sibling staging directory and only files absent locally move in.
    On the same box S3 only ever holds what this driver uploaded from these very files, so keeping
    the local copy can never lose anything; the fresh-box relaunch, which the flag exists for, has
    nothing local and receives everything. A staging directory left by an earlier failed restore is
    removed first, so a partial download is never mistaken for the prefix.
    """
    staging = local_dir.with_name(local_dir.name + RESTORE_STAGING_SUFFIX)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    restore_directory(s3_dest, staging)
    local_dir.mkdir(parents=True, exist_ok=True)
    added: list[str] = []
    kept: list[str] = []
    for source in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = source.relative_to(staging)
        destination = local_dir / relative
        if destination.exists():
            kept.append(str(relative))
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        added.append(str(relative))
    shutil.rmtree(staging)
    logger.info(
        f"restore from {s3_dest}: {len(added)} file(s) added to {local_dir} {added}, "
        f"{len(kept)} already present locally and kept {kept}"
    )


def _finish_without_engine(
    target: EvalTarget, *, plan: EvalPlan, sections: Sequence[str], config: EvalConfig
) -> bool:
    """Write the summary of an already-complete trace at the target, or say it needs generating.

    Returns True when the trace was complete and its summary is now written, False when records are
    missing and the engine has to load. Either way the trace has been checked to be this cell's.
    """
    battery_plan = plan_battery(sections, config)
    inspection = inspect_trace(
        target.out_path,
        battery_plan,
        expected_meta=_cell_identity(plan, target, sections=sections, config=config),
    )
    pending = inspection.pending(battery_plan)
    if pending:
        logger.info(
            f"step {target.step}: resuming {target.out_path} with {len(inspection.done)} of "
            f"{len(battery_plan)} records on disk; {len(pending)} left to generate"
        )
        return False
    logger.info(
        f"step {target.step}: {target.out_path} is complete ({len(battery_plan)} records) and only "
        f"lacked its summary; writing it without loading an engine"
    )
    _write_summary(target, finish_complete_trace(target.out_path, inspection, battery_plan))
    return True


def _evaluate_target(  # noqa: PLR0913 - one keyword per thing the driver resolved for this step
    target: EvalTarget,
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    sections: Sequence[str],
    config: EvalConfig,
    sync: IntervalSync | None,
) -> None:
    """Resolve how the weights are served, run or resume the battery, and write this step's summary."""
    if _has_complete_summary(target.out_path):
        # Checked at plan time too; appearing since means another process finished this cell here.
        raise FileExistsError(
            f"{summary_path_for(target.out_path)} appeared mid-run; refusing to redo a complete cell."
        )
    resume = target.out_path.exists()
    if resume and _finish_without_engine(target, plan=plan, sections=sections, config=config):
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

    # A mock trace must never pass for a measurement, so its model id is branded as such.
    model_id = f"mock:{served.model_id}" if args.backend == "mock" else served.model_id
    seed = engine_seed(plan.arm, target.step, backend_kind=args.backend, config=config)
    extra_kwargs = (
        served.backend_kwargs if seed is None else {**served.backend_kwargs, "seed": seed}
    )
    # Read before the engine exists, per games.vllm_teardown.baseline_before_engine, and None for
    # every kind that loads no engine.
    baseline_mib = baseline_before_engine(args.backend)
    backend = backend_cli.backend_from_args(
        args,
        model_id,
        local_sampling=eval_sampling(resolve_sampler_mode(args), thinking=plan.thinking),
        mock_responses=MOCK_PLUMBING_RESPONSES,
        extra_kwargs=extra_kwargs,
    )
    try:
        # Before the battery, not after: an un-merged adapter that failed to apply serves base
        # weights without saying so, and a whole arm of GPU time would read as a run that changed
        # nothing. Inside the `try` because raising is what it is for, and the engine is already up.
        verify_served_model(backend, served)
        # The summary is the battery's own rebuild from the trace it wrote (`games.evals.rebuild_summary`):
        # its top-level attribution -- sha, arm, step, sampler, sampling, engine seed -- is lifted off
        # the meta record, so the file a reader gets months later is exactly what the trace can say.
        summary = run_eval_battery(
            backend,
            sections=sections,
            out_path=target.out_path,
            meta=_target_meta(target, args=args, plan=plan, served=served, seed=seed),
            config=config,
            submission=args.submission,
            admission=args.admission,
            resume=resume,
            on_records_written=None if sync is None else sync.maybe_sync,
        )
        _write_summary(target, summary)
        if sync is not None:
            sync.sync_now(reason=f"step {target.step} complete")
        backend_cli.log_token_usage(backend)
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


def _render_arm_report(plan: EvalPlan) -> None:
    """Render the arm's report over every trace in the out dir, not just this invocation's.

    `report.md` is deliberately overwritten: unlike a trace it is derived, and re-rendering over
    the accumulated traces is what keeps one file current as checkpoints are evaluated in
    instalments.
    """
    trace_paths = sorted(plan.out_dir.glob("step-*.jsonl"))
    report_path = plan.out_dir / "report.md"
    report_path.write_text(
        render_report(trace_paths, title=f"Eval battery: {plan.arm}"), encoding="utf-8"
    )
    logger.info(f"report written over {len(trace_paths)} trace(s), {report_path=}")


def _bridge_hf_decode_kernel(backend_kind: str) -> dict[str, object] | None:
    """Bridge the Gated DeltaNet decode kernel for the one backend that decodes through HF here.

    Must run before anything can import the Qwen3.5 modeling module: transformers binds each
    DeltaNet kernel at that module's import time, so a bridge applied later is a silent no-op and
    every HF-path decode runs the pure-torch loop -- this was the one generating entry point that
    never bridged (measured kernel gain 1.52x at width 32, 1.94x at 64). The other backends decode
    elsewhere (vLLM runs its own GDN kernels, mock and the hosted kinds run no local model), and
    the merge path only loads weights, so they skip the fla import entirely.
    """
    if backend_kind != "hf":
        return None
    kernel_bridge = bridge_decode_kernel()
    assert_bridged_kernel_matches_call_site()
    logger.info("deltanet decode bridge: %s", kernel_bridge)
    return kernel_bridge


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the plan, run the battery per target, and render the arm's report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    default_cuda_allocator_config()
    args = _parse_args(argv)
    # Fail on a mode aimed at a backend that cannot honour it before anything expensive resolves.
    resolve_sampler_mode(args)
    if not args.summarise_only:
        _bridge_hf_decode_kernel(args.backend)
    sections = _parse_sections(args.sections)
    _refuse_orphan_trap_cells_file(args, sections)
    bank = _parse_bank(args)
    plan = resolve_plan(args)
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
        sections=sections,
        summarise_only=bool(args.summarise_only),
        skip_complete=sync is not None,
        banked_expected={
            target.out_path: bank_identity_without_config(args, plan, target, sections=sections)
            for target in plan.targets
            if _is_base_model_target(target)
        },
    )
    if args.backend in backend_cli.LOCAL_KINDS:
        # Namespace mutation is how a derived default reaches backend_from_args (select_prompts).
        args.thinking = plan.thinking
    if targets:
        _run_targets(targets, args=args, plan=plan, sections=sections, sync=sync, bank=bank)
    # Resumed records are counted per cell in its meta; skipped means whole cells already complete.
    logger.info(
        f"eval run finished: {len(targets)} cell(s) run or summarised, "
        f"{len(plan.targets) - len(targets)} skipped as already complete"
    )
    if args.report:
        _render_arm_report(plan)
    if sync is not None:
        sync.sync_now(reason="run finished")
    return 0


def _is_base_model_target(target: EvalTarget) -> bool:
    """Whether a target is the un-adapted base at step 0, the only cell the bank ever holds."""
    return target.checkpoint is None and target.step == BASE_MODEL_STEP


def _run_targets(  # noqa: PLR0913 - one keyword per thing main resolved before the template loads
    targets: Sequence[EvalTarget],
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    sections: Sequence[str],
    sync: IntervalSync | None,
    bank: CellBank | None = None,
) -> None:
    """Measure the chat template once, then run, resume or summarise each target still to run.

    Called only with targets left: the template facts come off a tokenizer load (from the hub or
    its cache), and a --sync-dest relaunch that restored every cell complete has nothing to measure
    them for -- the same guard `games.eval_training_frames` carries.

    With a ``bank``, a base-model step-0 target goes through `_bank_step_zero` instead: taken from
    the bank when a complete entry matches its key, generated and (with `--bank-base-cells`)
    published otherwise. Checkpoint cells never touch the bank.
    """
    template = _resolve_template_facts(
        args, thinking=plan.thinking, template_source=plan.targets[0].base_model
    )
    config = _resolve_eval_config(args, plan=plan, sections=sections, template=template)
    logger.info(
        f"eval plan: arm={plan.arm} steps={[target.step for target in targets]} "
        f"thinking={plan.thinking} prefilled_think={template.prefilled_think} "
        f"trained_games={list(plan.trained_game_ids)} grading={plan.grading} "
        f"game_behavior_samples={config.game_behavior_samples} "
        f"label_print_orders={list(config.label_print_orders)} "
        f"chat_template_kwargs={dict(template.chat_template_kwargs)} backend={args.backend} "
        f"submission={args.submission} admission={args.admission} "
        f"sampler_mode={sampler_mode_meta(args)} out_dir={plan.out_dir} "
        f"bank={None if bank is None else bank.prefix} "
        f"bank_publish={bank is not None and bank.publish}"
    )
    if args.summarise_only:
        for target in targets:
            summary = salvage_summary(
                target.out_path,
                plan_battery(sections, config),
                expected_meta=_cell_identity(plan, target, sections=sections, config=config),
            )
            _write_summary(target, summary)
        return
    for target in targets:
        if bank is not None and _is_base_model_target(target):
            _bank_step_zero(
                target, args=args, plan=plan, sections=sections, config=config, sync=sync, bank=bank
            )
            continue
        _evaluate_target(target, args=args, plan=plan, sections=sections, config=config, sync=sync)


def _bank_step_zero(  # noqa: PLR0913 - the same seam _evaluate_target carries, plus the bank
    target: EvalTarget,
    *,
    args: argparse.Namespace,
    plan: EvalPlan,
    sections: Sequence[str],
    config: EvalConfig,
    sync: IntervalSync | None,
    bank: CellBank,
) -> None:
    """Take the base-model cell from the bank, or generate it and offer it to the bank.

    The noise-floor cell (`is_noise_floor_cell`) is generated per arm and never published, so
    every arm keeps one step-0 draw of its own. Otherwise the key is derived (`bank_identity`,
    which renders the plan once to digest its prompts) and the cell takes one of two paths to a
    complete trace. A trace already at the out path is a cell that died mid-run and is resumed
    rather than replaced by the bank's copy -- a paid-for partial trace is never overwritten, and
    the bank's cell is the same measurement but not the same records -- so the bank is not
    consulted for it. With nothing at the out path a matching entry is copied
    (`_reuse_banked_cell`) and nothing generates, and with no entry the cell generates as usual.
    Either way a cell this run completed is published under the key when `--bank-base-cells` was
    given: the relaunch is the normal path for a 12 h cell that died once, and a bank that only
    ever received cells whose first session finished would stay empty for exactly the cells worth
    banking.
    """
    if is_noise_floor_cell(sections, config, bank.noise_floor_framings):
        logger.info(
            f"step {target.step}: this cell renders only the noise-floor framings "
            f"{list(config.counterpart_framings)}, so it stays per arm: not taken from the bank "
            f"and not published; generating it"
        )
        _evaluate_target(target, args=args, plan=plan, sections=sections, config=config, sync=sync)
        return
    identity = bank_identity(args, plan, target, sections=sections, config=config)
    key = bank_key(identity)
    logger.info(f"step {target.step}: bank key {key} under {bank.prefix}")
    if target.out_path.exists():
        publish_note = "; it is published once complete" if bank.publish else ""
        logger.info(
            f"step {target.step}: {target.out_path} already holds this cell's partial trace, "
            f"which is resumed rather than replaced by the bank's copy{publish_note}"
        )
    elif _reuse_banked_cell(
        target,
        bank=bank,
        identity=identity,
        key=key,
        plan=plan,
        sections=sections,
        config=config,
        sync=sync,
    ):
        return
    _evaluate_target(target, args=args, plan=plan, sections=sections, config=config, sync=sync)
    if bank.publish:
        _publish_banked_cell(
            target, bank=bank, identity=identity, key=key, sync_dest=args.sync_dest
        )


def _resolve_eval_config(
    args: argparse.Namespace, *, plan: EvalPlan, sections: Sequence[str], template: TemplateFacts
) -> EvalConfig:
    """Turn the parsed flags, the resolved plan and the measured template facts into the battery's config."""
    return EvalConfig(
        open_ended_samples=args.open_ended_samples,
        open_ended_samples_by_item=_parse_open_ended_allocation(args.open_ended_samples_by_item),
        multiple_choice_samples=args.multiple_choice_samples,
        game_behavior_samples=args.game_behavior_samples,
        label_print_orders=_parse_print_orders(args.label_print_order),
        capability_items=args.capability_items,
        survey_samples=args.survey_samples,
        survey_data_dir=args.survey_data_dir,
        survey_instruments=_parse_id_list(args.survey_instruments),
        survey_families=_parse_id_list(args.survey_families),
        survey_tier=args.survey_tier,
        survey_counterpart_variants=args.survey_counterpart_variants,
        prefilled_think=template.prefilled_think,
        batch_size=args.batch_size,
        dtbench_dir=args.dtbench_dir,
        games=_parse_id_list(args.games),
        trained_game_ids=plan.trained_game_ids,
        include_never_trained=args.include_never_trained,
        chat_template_kwargs=template.chat_template_kwargs,
        # All registered framings when the sweep section is requested bare, so the CLI's "empty
        # means everything" convention holds; empty otherwise, so a trace whose battery never ran
        # the section does not record framings it never rendered.
        counterpart_framings=(
            _parse_id_list(args.counterpart_framings)
            or (COUNTERPART_FRAMING_IDS if SECTION_FRAMING_SWEEP in sections else ())
        ),
        framing_sweep_games=(_parse_id_list(args.framing_sweep_games) or FRAMING_SWEEP_GAME_IDS),
        trap_cells_file=args.trap_cells,
        # Digested here rather than inside the section, so the recipient paragraphs the cell
        # identity claims are the ones present when the plan was resolved: the planner re-reads the
        # file and refuses a digest that has moved since.
        trap_cells_digest=(
            ""
            if args.trap_cells is None
            else load_dictator_recipient_clauses(args.trap_cells).digest
        ),
        # Loaded here rather than at parse time so one refusal covers every path into a battery,
        # `--summarise-only` included; the load is a small JSON read and validates the whole file.
        runtime_framings=(
            None if args.framings_file is None else load_framings(args.framings_file)
        ),
        # Loaded here rather than at parse time so one load serves every target of the run, and
        # every gate the loader runs fires before the first prompt is rendered.
        held_out_extension=(
            None
            if args.held_out_extension is None
            else load_extension(Path(args.held_out_extension))
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
