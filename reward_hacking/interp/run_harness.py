"""One-command interpretability run harness: fit a lens, contrast twins, then intervene causally.

This is the in-repo THIN driver that replaces the loose bespoke stage scripts. Those shipped as
files OUTSIDE the code archive, so they drifted from the modules they imported and nothing checked
it -- the second rented-box failure. Living here, the driver ships INSIDE the ``git archive`` and is
covered by the ``run_preflight`` content manifest, so it cannot drift from the modules it calls.

It adds no science. Each stage is orchestration over the repo's own tested functions:

* ``fit-lens`` -- build a fit corpus from the matched grader twins (both arms) plus generated
  thinking-mode reasoning, then :func:`reward_hacking.interp.jacobian.fit_lens` (the closed-form
  lens fit that now forwards ``dim_batch`` / ``max_seq_len``); save the lens immediately, verify a
  save/reload round-trip, and refuse to resume a checkpoint whose corpus cannot be PROVEN identical.
* ``contrast`` -- the generation-phase read: generate each twin's continuation ONCE and pool it
  three ways (whole response, and a matched window at the response start and end), then project
  every pooling variant onto the concept axes with the matched-norm placebo control
  (:func:`prompt_contrast.contrast_all_layers`). The eval-awareness axis is a validated one loaded
  from disk per pooling (fatal if missing -- no silent fresh fallback). Emits a ``peak_layers``
  handoff keyed ``concept -> variant -> pooling`` for the causal stage, and saves the concept axes
  so the causal stage projects onto the SAME vectors rather than re-extracting its own.
* ``steer-patch`` -- the causal tier: steering AND ablation against matched-norm placebos at an
  alpha derived from the layer's measured residual norm, plus activation patching between the
  end-anchored twins at the peak layer. A decodable direction is not a used one; this is what tells
  them apart.
* ``patch-sweep`` -- patching only, swept: every layer rather than one, a ladder of window widths
  that separates the readout position out of the wide window, both patch directions (sufficiency and
  necessity), and an optional decomposition of the patch along each concept axis. It exists because
  ``steer-patch``'s single wide window included the readout position, which makes a recovery near 1.0
  near-tautological; see :func:`plan_twin_patch_ladder` and :func:`run_patch_sweep`.

The patch target is the region where the twins DIVERGE. An earlier version patched their shared
leading prefix, where causal attention makes both runs' activations identical -- so the patch was a
no-op and every recovery was 0.0 for every pair at every layer: a green null that was an artifact of
the code. That region is now kept as a named negative control (``shared_prefix_control``), which is
the only arm expected to read zero, and the real arms are read against a matched-Frobenius-norm
random perturbation at the same positions.

Structure mirrors the rest of the package: the pure orchestration logic (corpus assembly, the VRAM
-> ``dim_batch`` rule, the twin patch plan, peak-layer selection, the alpha rule, the behavioural
observables, the round-trip tolerance) lives in module-level functions the offline tests exercise on
synthetic data; every path that loads a model, generates, fits, or patches is GPU-only and behind a
function guarded by CUDA availability.

Privacy: aggregate metrics and the peak-layer handoff go to ``--out-dir`` (uploadable); the fit
corpus text, the fitted lens, generated responses, raw activations and concept axes go to
``--raw-dir`` (never uploaded -- they embed benchmark item text or are keyed to it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields, replace
from functools import cache
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import torch

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    assert_one_deltanet_kernel,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from reward_hacking.harness.protocol import ActionKind, parse_actions
from reward_hacking.interp import stimuli
from reward_hacking.interp.directions import (
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # shared trunk-layer resolver
    capture_positionwise_activations,
    load_model_and_tokenizer,
    matched_norm_random_direction,
)
from reward_hacking.interp.eval_awareness_probe import (
    ConceptAxisResult,
    probe_concept,
)
from reward_hacking.interp.eval_awareness_probe import (
    save_artifacts as save_axis_artifacts,
)
from reward_hacking.interp.generation_capture import (
    DEFAULT_VLLM_GPU_FRACTION,
    GEN_ENGINE_VLLM,
    GEN_ENGINES,
    PENALTY_FREE_THINKING_SAMPLING,
    GenerationRecord,
    HFResponseGenerator,
    ResolvedSampler,
    ResponseGenerator,
    VLLMResponseGenerator,
    _chat_format,  # pyright: ignore[reportPrivateUsage]  # the one harness-matching chat wrapper
    capture_record_pooled,
    default_gen_engine,
    generate_response,
    matched_window_positions,
    resolved_sampler,
    resolved_sampler_for,
    response_positions,
)
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_COMMIT,
    JacobianConfig,
    SeqLenPlan,
    _load_jlens_model,  # pyright: ignore[reportPrivateUsage]  # the shared jlens model loader
    _require_jlens,  # pyright: ignore[reportPrivateUsage]  # the shared PYTHONPATH jlens loader
    cap_fit_prompts,
    derive_max_seq_len,
    evaluate_reconstruction,
    fit_lens,
    fit_quality_payload,
    fit_skip_first,
    resolve_weights_identity,
    transport_and_decode,
    verify_cached_lens,
    verify_lens_roundtrip,
)
from reward_hacking.interp.linear_probe import CaptureSpec, ProbeConfig
from reward_hacking.interp.patch_readout import (
    FORCED_CHOICE_PREFILL,
    OPTION_ORDERS,
    build_action_readout,
    build_forced_choice_readout,
    forced_choice_suffix,
    readout_provenance,
    readout_transcripts,
    require_shared_suffix_covers,
)
from reward_hacking.interp.prompt_contrast import (
    CONTRADICTION,
    CONTRAST_CONCEPTS,
    DECEPTION,
    EVAL_AWARENESS,
    SHORTCUT,
    ContrastRead,
    StimulusPair,
    build_stimulus_pairs,
    contrast_all_layers,
    extract_concept_directions,
    load_validated_eval_direction,
    metrics_dict,
    placebo_stream_seed,
    selection_by_group,
)
from reward_hacking.interp.steering import (
    DEFAULT_HEAD_WIDTHS,
    DEFAULT_MAX_NARROW_POSITIONS,
    DEFAULT_TAIL_WIDTHS,
    GAP_READOUT_MODES,
    PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT,
    PATCH_WINDOW_READOUT_ONLY,
    READOUT_MODE_ACTION_LOGPROB,
    READOUT_MODE_FORCED_CHOICE,
    TAIL_WINDOW_PREFIX,
    GapRead,
    GapReadout,
    PatchBaseline,
    PatchResult,
    PatchWindow,
    TwinPatchPlan,
    axis_complement_replacement,
    axis_component_replacement,
    capture_patch_prefix,
    matched_norm_replacement,
    plan_twin_patch,
    plan_twin_patch_ladder,
    random_axis_replacement,
    recovery_metrics,
    run_activation_patch,
    run_steered_generation,
    steering_directions,
    window_contains_readout,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reward_hacking.interp.generation_capture import PositionSelector
    from reward_hacking.model_backend import SamplingConfig

logger = logging.getLogger(__name__)

# Concept axes extracted from raw sentence pairs (eval-awareness is loaded validated instead).
EXTRACTED_CONCEPTS: dict[str, list[stimuli.ContrastivePair]] = {
    SHORTCUT: list(stimuli.SHORTCUT_PAIRS),
    DECEPTION: list(stimuli.DECEPTION_PAIRS),
    CONTRADICTION: list(stimuli.CONTRADICTION_PAIRS),
}

# Every concept axis the contrast/causal tiers project onto -- the three extracted ones PLUS
# eval-awareness -- mapped to its sentence pairs. The ``axis-probe`` stage runs the full validation
# battery over ALL of these, so each axis a downstream read leans on carries the same quality
# metrics (probe accuracy, matched-norm placebo, split-half, per-layer significance) rather than
# only eval-awareness carrying them while the other three ship as bare, unvalidated diff-of-means
# directions. The key set is asserted equal to CONTRAST_CONCEPTS in the tests, so dropping a concept
# here (the exact single-axis defect) fails loudly instead of silently skipping its validation.
AXIS_PROBE_CONCEPTS: dict[str, list[stimuli.ContrastivePair]] = {
    **EXTRACTED_CONCEPTS,
    EVAL_AWARENESS: list(stimuli.EVAL_AWARENESS_PAIRS),
}

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_POOLINGS: tuple[str, ...] = ("mean", "last")
DEFAULT_N_PLACEBOS = 100
CONCEPT_BATCH_SIZE = 16
# VRAM headroom (GiB) at which the lens fit steps dim_batch up; read from the device, not hardcoded.
DIM_BATCH_HIGH_VRAM_GIB = 20.0
DIM_BATCH_MID_VRAM_GIB = 10.0
MIN_CONTRAST_PAIRS = 2  # a paired contrast needs at least two twins

COIN_FLIP = 0.5  # the fair-coin threshold the twin-arrival shuffle draws against

FINAL_FIFTH = 0.8
"""Depth fraction past which a patch cell sits in the final-layer perturbation trap, and is flagged.

Flagged per cell, NOT excluded. The direction-selection literature restricts candidate layers to the
first ~80% of the stack, and this repo measured the reason directly: at the final layer the output
head reads each position independently, so patching any non-readout position moves the readout row by
exactly 0.000000 while a window CONTAINING the readout reads recovery exactly 1.0. Excluding those
layers would hide the artifact the full sweep exists to characterise, so the sweep runs them and every
row says whether it is in the band -- which is what lets a reader apply the restriction downstream and
still see what the excluded layers did.
"""

PEAK_LAYER_MIN_EXCESS = 0.0
"""How far a cell's best layer must clear its matched-norm placebo to be handed downstream at all.

Zero, deliberately: this is not a significance threshold (the repo's probing posture rules those out
on exploratory work) but the weakest possible statement that a peak must at least beat the placebo it
is measured against. Its absence fired -- see :func:`select_peak_layers` -- and the point is to make
"nothing here cleared chance" an ABSENCE in the handoff rather than a layer index indistinguishable
from a real one. Raising this to a real effect size would be a gate and does not belong here.
"""
# Reconstruction fit-quality read after a fit: how many corpus prompts to evaluate on and how many
# interior positions to read per prompt. A handful of extra forward passes on top of a ~2 h fit, so
# a fit is never shipped without a self-describing quality number, yet the read stays negligible.
DEFAULT_RECON_EVAL_PROMPTS = 8
DEFAULT_RECON_MAX_POSITIONS = 8

# Pooling variants over the generated response. All three come off ONE generation per prompt: the
# whole response, and a fixed-width window at its start and at its end. The windows exist because
# the twins generate different amounts of text, so an all-response pooling mixes "what the model
# represents" with "how long it talked"; a matched width holds that constant.
VARIANT_ALL_RESPONSE = "all_response"
VARIANT_WINDOW_START = "window_start"
VARIANT_WINDOW_END = "window_end"
DEFAULT_VARIANTS: tuple[str, ...] = (VARIANT_ALL_RESPONSE, VARIANT_WINDOW_START, VARIANT_WINDOW_END)
DEFAULT_WINDOW_LENGTH = 128
# The reasoning the contrast generates IS its measurement substrate, so the cap must clear the
# model's thinking length or the read pools a TRUNCATED reasoning prefix rather than a complete
# trace. Default to the thinking-mode cap (read from the harness's own sampling preset via
# generation_capture, never a literal), not a small round number: a hardcoded 2048 truncated 43 of
# 46 generations on the first run. The cap stays a CLI knob (``--max-new-tokens``) so the human can
# trade reasoning length against pair coverage for a bounded box; ``ContrastCoverage`` counts every
# truncation regardless, so a low setting shows up rather than hiding.
DEFAULT_CONTRAST_MAX_NEW_TOKENS = PENALTY_FREE_THINKING_SAMPLING.max_new_tokens

# Twin pairs per generation call (so two prompts each). Only vLLM cares: its throughput comes from
# having many sequences in flight, and the HF generator loops one prompt at a time regardless. 8 is a
# compromise rather than a measurement -- 16 concurrent thinking traces keep the engine busy, while
# still letting --deadline-seconds stop the stage at a granularity a bounded box can plan around,
# since the deadline can only be checked between chunks and vLLM has no wall-clock budget of its own.
DEFAULT_GEN_BATCH_PAIRS = 8

# The twin sides, named rather than positional so a saved artifact reads itself.
SIDE_CONFLICTING = "conflicting"
SIDE_ORIGINAL = "original"

# The two arms every patch window runs: the clean activations themselves, and a random perturbation
# of the same Frobenius norm at the same positions. The window names and the alignment that produces
# them live in :mod:`reward_hacking.interp.steering` beside the patch driver they feed.
PATCH_ARM_REAL = "real"
PATCH_ARM_PLACEBO = "placebo_matched_norm"

# The patch-sweep arms; :func:`_patch_sweep_arms` documents what each isolates.
PATCH_ARM_REAL_FULL = "real_full_residual"
PATCH_ARM_REAL_AXIS_COMPONENT = "real_axis_component"
PATCH_ARM_REAL_AXIS_COMPLEMENT = "real_axis_complement"
PATCH_ARM_PLACEBO_AXIS_MATCHED_NORM = "placebo_axis_matched_norm"
PATCH_ARM_PLACEBO_RANDOM_AXIS = "placebo_random_axis_component"
PATCH_MODE_FULL_RESIDUAL = "full_residual"
PATCH_MODE_AXIS = "axis"
PATCH_MODES: tuple[str, ...] = (PATCH_MODE_FULL_RESIDUAL, PATCH_MODE_AXIS)

# Which run is patched into which; :func:`_sweep_one_pair` documents what each direction asks.
PATCH_DIRECTION_ORIGINAL_INTO_RIGGED = "original_into_rigged"
PATCH_DIRECTION_RIGGED_INTO_ORIGINAL = "rigged_into_original"
PATCH_DIRECTIONS: tuple[str, ...] = (
    PATCH_DIRECTION_ORIGINAL_INTO_RIGGED,
    PATCH_DIRECTION_RIGGED_INTO_ORIGINAL,
)

DEFAULT_MAX_PAIR_TOKENS = 6144
"""Cost bound on one twin pair, not a quality filter (see :func:`eligible_patch_pairs`).

The ILCB pairs tokenize to 1.3k-22.3k with a ~2.3k median, so the longest pair's forward passes cost
ten times a typical pair's for one more row. 6144 keeps 96 of the 100.
"""

PATCH_SWEEP_RAW_FILENAME = "patch_sweep_raw.pt"
"""The sweep's raw artifact, deliberately NOT carrying the ``.local.`` never-uploaded marker.

It is meant to be shipped to the run's private S3 prefix (owner, 2026-08-23): two earlier causal runs
wrote their raw material to a local raw dir, never uploaded it, and let the box self-terminate, which
destroyed it. It holds logit rows and residual vectors, never item text.
"""

READOUT_SEMANTICS = (
    "recovery_gap is the PRIMARY reading: (patched - corrupted) / (clean - corrupted) on the GAP "
    "between two complete candidate actions, one of which the harness's own read_tampering "
    "classifies as weakening the grader and one of which it classifies as leaving the machinery "
    "alone. Under the action_logprob readout that gap is the difference of two teacher-forced "
    "SEQUENCE log-probs; under the forced_choice readout it is one logit difference between two "
    "option labels at an answer slot and is a STATED PREFERENCE, a different construct. It is None "
    "when the two runs' gaps agree exactly, which the gap_denominator field makes countable. "
    "recovery_logit, recovery_kl and recovery_logprob_l2 are secondary diagnostics off the same "
    "rows at no extra cost; recovery_logit is what the 2026-08-22 run reported, and on THAT run it "
    "scored the first token of the model's thinking block (the four answer tokens decoded to 'C', "
    "'This', 'Need', 'Thinking'), so the two are comparable in arithmetic and not in meaning. "
    "None of these is behavioural: nothing in this artifact may be quoted as 'the transplant "
    "transfers the rigged behaviour' or as a hack rate. Behaviour-level effects need a separate run "
    "over the real execution harness. The raw action gap carries a constant candidate-length term "
    "and is not a preference reading on its own; only its clean-versus-corrupted difference is "
    "interpreted. Substrate limit, not fixable with these twins: the conflicting grader is the "
    "original plus an inserted assertion, so clean_middle_len is 0 and there is no clean-side row "
    "AT the insertion -- the manipulation itself is unpatchable here and only its downstream effect "
    "is measured."
)
"""One line stating what the patch readout measures and what it does not.

Written into every patch artifact and into the run's completion marker, because the number is easy
to over-read: a recovery is a statement about a scored gap between two authored candidates, and the
earlier single-logit version of it scored the opening word of a reasoning trace.
"""

DEFAULT_DELTA_WINDOWS: tuple[str, ...] = (
    PATCH_WINDOW_READOUT_ONLY,
    f"{TAIL_WINDOW_PREFIX}8",
    f"{TAIL_WINDOW_PREFIX}32",
    PATCH_WINDOW_POST_DIVERGENCE_EXCL_READOUT,
)
"""Windows whose per-pair mean transplanted delta is retained for the lens decode.

The delta is what patching actually injects (``clean - corrupted`` at the patched positions), so it
is the vector worth decoding into token space. Keeping it for every window of every pair would be
~1 GB per unit; these four cover the readout position, two narrow tails and the honest wide window.
Cross-pair means are retained for EVERY window regardless, since they are 10 KB each.
"""

# Steering arms and the alpha ladder. ``alpha`` is a FRACTION of the layer's measured residual norm,
# never an absolute activation-unit constant: the same constant is a shove at one layer and noise at
# another, and it cannot survive a change of model width. Two scales because one is a guess.
STEER_MODE_BASELINE = "none"
STEER_MODE_STEER = "steer"
STEER_MODE_ABLATE = "ablate"
STEER_ARM_BASELINE = "baseline"
DEFAULT_ALPHA_SCALES: tuple[float, ...] = (0.1, 0.5)
PLACEBO_ARM_PREFIX = "placebo_"

# Token budget for the steering/ablation read. This read GENERATES the model's own response and
# counts the protocol actions in it, so the cap must let a thinking model finish reasoning AND emit
# at least one action. Qwen3.5 thinking traces routinely run thousands of tokens before the first
# action; the earlier 512 default truncated every arm inside the reasoning block, so no action was
# ever parsed and every behavioural observable floored at zero -- baseline included -- which made
# each real_minus_placebo zero by construction rather than by measurement. Generous by default and
# overridable on the CLI; the exact production value is a cost-versus-coverage knob set per run
# (more pairs vs. a longer cap), so it lives here as a named default rather than a bare literal.
STEER_DEFAULT_MAX_NEW_TOKENS = 16384

# The behavioural observables read off one generated response. Counts and rates only, so every arm
# is directly comparable and the placebo mean is a plain average. Response length is kept (it is a
# real behavioural change) but is no longer the ONLY read: what the model does with the grader is.
OBSERVABLE_NAMES: tuple[str, ...] = (
    "response_tokens",
    "hit_token_cap",
    "n_actions",
    "n_run_commands",
    "n_commands_naming_grader",
    "n_abort_actions",
    "n_empty_actions",
)

# The grader-read OBSERVATION the stimulus transcripts carry, which names the grader file
# unambiguously. Read from the observation rather than from the twins' ``<run>cat grade.py</run>``
# because the protocol preamble describes ``<run>...</run>`` in prose, and the harness's own
# non-greedy action regex then pairs a prose ``<run>`` with the real read's closing tag and swallows
# it -- so parsing the transcript's actions returns three fragments of prose and no grader read
# (observed in the 0.8B smoke, not hypothesised).
_GRADER_OBSERVATION_RE = re.compile(r'<result command="cat (?P<filename>\S+)" exit=0>')


# --------------------------------------------------------------------------------------
# Pure logic (no model, no CUDA): what the offline tests exercise
# --------------------------------------------------------------------------------------


def require_record_sampler(
    record: GenerationRecord, expected: ResolvedSampler, *, stage: str
) -> None:
    """Refuse a generated record whose sampler is not the one this stage resolved.

    The threading guard, and it exists because the failure it catches is silent. Each stage builds
    one :class:`~reward_hacking.model_backend.SamplingConfig` and records it in the stage's metrics
    artifact; a generation site that forgets to pass it still generates, still returns a record, and
    still produces a green run -- under ``generation_capture``'s default sampler, while the artifact
    claims the stage's. Comparing what came back against what the stage resolved turns that into a
    crash on the first response, before a GPU-hour has been spent measuring the wrong sampler.

    Checked per record rather than once at startup: the point is what reached ``generate``, and only
    a record can say.
    """
    if record.sampler == expected:
        return
    raise RuntimeError(
        f"the {stage} stage generated a response under a sampler it did not resolve, so its metrics "
        f"artifact would claim a sampler that never ran. Resolved applied={expected.applied} "
        f"dropped={sorted(expected.dropped)}; the record carries applied={record.sampler.applied} "
        f"dropped={sorted(record.sampler.dropped)}. A generation site that does not forward the "
        "stage's SamplingConfig falls back to generation_capture's default one."
    )


def require_raw_dir_outside_episode_dir(raw_dir: Path, episode_dir: Path) -> None:
    """Refuse a raw tree at or inside the episode dir, which ``lay_down_task`` deletes per task.

    Every stage here materialises its stimuli through ``build_stimulus_pairs``, and the episode dir
    it writes into is CLEARED and rewritten for each task. So a raw tree nested inside it is deleted
    partway through the stage, after the stage already created it -- the artifacts vanish between the
    mkdir and the first write, and the failure surfaces later as a bare ``FileNotFoundError`` on a
    path the caller can see was created. A rented g7e lost a 4B lens fit to exactly this after 20
    reasoning traces had already been generated, and the traceback pointed at the write rather than
    at the layout, which is the part worth failing loudly on instead.

    Checked before any model load or CUDA call so a bad layout costs milliseconds on any machine
    rather than a GPU-hour. ``resolve`` first, so ``work/../work/raw`` and a symlinked scratch cannot
    walk past the comparison.
    """
    resolved_raw = raw_dir.resolve()
    resolved_episode = episode_dir.resolve()
    if resolved_raw.is_relative_to(resolved_episode):
        raise ValueError(
            f"--raw-dir {resolved_raw} is at or inside --episode-dir {resolved_episode}, which is "
            "cleared and rewritten for every materialised task, so the raw tree would be deleted "
            "mid-stage. Point --raw-dir at a sibling of the episode dir instead, e.g. "
            f"{resolved_episode.parent / 'raw'}."
        )


def build_fit_corpus(pairs: Sequence[StimulusPair], reasoning_texts: Sequence[str]) -> list[str]:
    """Assemble the lens-fit corpus: both arms of every twin, then the generated reasoning texts.

    ``jlens.fit`` truncates each entry to ``max_seq_len`` and iterates every one, so the corpus is
    just the list of texts to average Jacobians over -- both grader conditions (so the lens is not
    fit to one arm's distribution) followed by the model's own thinking-phase text (so the lens has
    next-token fidelity on the agentic regime the reads target). Order is deterministic for a
    reproducible fit; empties are dropped so a blank generation cannot become a degenerate prompt.
    """
    corpus: list[str] = []
    for pair in pairs:
        corpus.append(pair.conflicting_transcript)
        corpus.append(pair.original_transcript)
    corpus.extend(reasoning_texts)
    return [text for text in corpus if text.strip()]


def corpus_hash(texts: Sequence[str]) -> str:
    """Hex sha256 over the ordered corpus, stored beside a fit so a resume cannot mix distributions.

    ``jlens.fit`` resumes by list position, so resuming a checkpoint against a different corpus
    would average two different distributions into one lens. The fit metadata carries this hash and
    a resume must refuse on a mismatch.
    """
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\x00")  # length-delimit so ["ab","c"] and ["a","bc"] differ
    return digest.hexdigest()


def corpus_seq_len_plan(
    tokenizer: object,
    corpus: Sequence[str],
    *,
    max_seq_len: int | None,
    ceiling: int = DEFAULT_MAX_SEQ_LEN_CEILING,
) -> SeqLenPlan:
    """Derive a fit's window from what this corpus actually tokenizes to, under whichever bound applies.

    Both flags are ceilings on one derivation rather than two settings: an explicit ``--max-seq-len``
    replaces the default ceiling, and either way the corpus decides below it, because a window wider
    than the longest item changes nothing about the fit and recording it would overstate what the fit
    read. The counts come from the tokenizer the fit's own ``encode`` uses, so the reported truncation is
    the truncation.
    """
    if not corpus:
        raise ValueError("an empty fit corpus has no token lengths to derive a window from")
    encoded = cast(
        "list[list[int]]",
        tokenizer(list(corpus))["input_ids"],  # pyright: ignore[reportCallIssue,reportOperatorIssue,reportIndexIssue]
    )
    return derive_max_seq_len(
        [len(ids) for ids in encoded], ceiling=ceiling if max_seq_len is None else max_seq_len
    )


def dim_batch_for_free_vram(free_gib: float) -> int:
    """Pick the lens-fit ``dim_batch`` from free VRAM, never a hardcoded budget.

    ``dim_batch`` is a memory knob only: it sets how many residual dimensions each backward pass
    carries, so live activation memory scales with it while the total backward FLOPs do not
    (``jlens.fitting.jacobian_for_prompt``: "total backward FLOPs are unchanged"). Measured fits at
    2B and 9B gained little past 16, and 64 OOM'd both a 44 GiB and a 96 GiB card, each costing a
    failed attempt and a relaunch. So the ladder exists to keep a fit inside the card that is
    present, not to make it faster: it is read from the device at startup so a config cannot assume
    a card it did not land on, and it tops out at the 16 the repo-measured 4B fit ran at on a 24 GiB
    L4.
    """
    if free_gib >= DIM_BATCH_HIGH_VRAM_GIB:
        return 16
    if free_gib >= DIM_BATCH_MID_VRAM_GIB:
        return 8
    return 4


def _matched_norm_or_noop(
    clean_rows: torch.Tensor, corrupted_rows: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Placebo rows for one patch window: a matched-norm perturbation, or a no-op when identical.

    Every window now carries a placebo arm, so its recovery reads against a random perturbation of
    equal norm rather than against nothing. The shared-prefix control is bit-identical between the
    twins in the healthy case, so its clean and corrupted rows are equal and a matched-norm placebo
    is a zero-magnitude perturbation -- the corrupted rows unchanged. Returning them keeps the arm
    present and honest (a no-op placebo of a no-op real patch) instead of tripping
    :func:`matched_norm_replacement`'s guard, which exists to catch a SILENT no-op where a divergent
    perturbation was expected. The window's recorded ``patch_delta_norm`` stays the denominator that
    tells a bit-identical window (zero) apart from a divergent one the readout simply did not move.
    """
    if (clean_rows - corrupted_rows).norm() == 0:
        return corrupted_rows
    return matched_norm_replacement(clean_rows, corrupted_rows, generator)


def _best_read_per_cell(
    reads_by_variant: Mapping[str, Sequence[ContrastRead]],
) -> dict[tuple[str, str, str], ContrastRead]:
    """Pick each ``(concept, variant, pooling)`` cell's best read on ``auc_above_placebo``.

    The ONE place the peak comparison lives. :func:`select_peak_layers` hands the winning layer to the
    causal stage and :func:`peak_layers_lineage` audits which cells the floor withheld, so an audit
    that scanned for its own best read could disagree with the selector it exists to describe and
    nothing would catch it. This file has already been burned by that shape once, which is why
    :func:`peak_selection_significance` had to stop indexing the selector's output.

    First-wins on an exact tie, matching what a plain argmax over the reads in arrival order does.
    """
    best: dict[tuple[str, str, str], ContrastRead] = {}
    for variant, reads in reads_by_variant.items():
        for read in reads:
            key = (read.concept, variant, read.pooling)
            current = best.get(key)
            if current is None or read.auc_above_placebo > current.auc_above_placebo:
                best[key] = read
    return best


def _clears_placebo_floor(read: ContrastRead) -> bool:
    """Whether a cell's best read is handed downstream at all, per :data:`PEAK_LAYER_MIN_EXCESS`."""
    return read.auc_above_placebo > PEAK_LAYER_MIN_EXCESS


def select_peak_layers(
    reads_by_variant: Mapping[str, Sequence[ContrastRead]],
) -> dict[str, dict[str, dict[str, int]]]:
    """For each ``(concept, variant, pooling)`` pick the layer whose real axis most clears placebo.

    ``auc_above_placebo`` is the separation over chance, so the peak layer is where the concept is
    most linearly present under that pooling of that response window -- the layer the causal stage
    should steer/ablate/patch. The pooling VARIANT is part of the key because an all-response
    pooling and a matched window at the response end are different reads and can peak at different
    layers; collapsing them would hand the causal stage a layer chosen under a read it is not using.

    **A cell that never clears its placebo at any layer emits no peak** (added 2026-08-24). This was
    a plain argmax with no floor, and the floor's absence fired: in the 2026-08-22 replicate contrast
    TWO cells were negative on ``auc_above_placebo`` at all 32 layers and still emitted a handoff
    layer -- shortcut/window_end/mean at best -0.0651 and deception/window_end/mean at -0.0074 -- and
    the causal box consumed those layers without recording where they came from. "The best of 32 bad
    layers" is a draw from a flat distribution wearing the word *peak*, so it is now absent from the
    handoff rather than present and indistinguishable from a real one. Absence is the honest signal:
    :func:`peak_layers_lineage` records which cells were withheld and why, and a downstream stage
    asking for a missing peak fails loudly.

    The comparison and the floor are :func:`_best_read_per_cell` and :func:`_clears_placebo_floor`, so
    the lineage that audits this selection cannot drift from it. A cell the floor withholds leaves its
    ``(concept, variant)`` key present with an EMPTY pooling map rather than dropping the variant, so a
    reader can tell "measured and withheld" from "never measured".
    """
    peaks: dict[str, dict[str, dict[str, int]]] = {}
    for (concept, variant, pooling), read in _best_read_per_cell(reads_by_variant).items():
        by_pooling = peaks.setdefault(concept, {}).setdefault(variant, {})
        if _clears_placebo_floor(read):
            by_pooling[pooling] = read.layer
    return peaks


def _read_fingerprint(read: ContrastRead) -> tuple[float, ...]:
    """Every numeric field of one read, flattened in declaration order.

    Derived from the dataclass rather than hand-listed, because the claim
    :func:`duplicate_read_cells` publishes is that the grouped cells are bit-identical on every
    numeric field. A hand-listed subset made that claim wider than the check: it fingerprinted five of
    ContrastRead's twenty-one numeric fields, so two cells differing only in, say, ``paired_t`` or
    ``auc_empirical_p`` were reported as one measurement under two names. Reading the fields off the
    dataclass also means a field added later is inside the claim without anyone remembering to add it.

    The per-placebo AUC draws are a tuple of numbers rather than a scalar and are flattened in, not
    skipped: two cells that agree on every summary while drawing different placebo directions are not
    the same measurement. ``concept`` and ``pooling`` are the only non-numeric fields and are part of
    the cell key, so nothing identifying is lost by their absence here.
    """
    values: list[float] = []
    for field_ in fields(read):
        value = getattr(read, field_.name)
        if isinstance(value, (int, float)):
            values.append(float(value))
        elif isinstance(value, tuple):
            values.extend(float(item) for item in cast("tuple[float, ...]", value))
    return tuple(values)


def duplicate_read_cells(
    reads_by_variant: Mapping[str, Sequence[ContrastRead]],
) -> dict[str, object]:
    """Find (variant, pooling, concept) cells whose reads are bit-identical to another cell's.

    Found 2026-08-24: ``window_end``/``last`` is identical to ``all_response``/``last`` for all four
    concepts in both retained runs -- 0.0 maximum difference across 32 layers of every numeric field,
    identical peak-selection dicts. It is identical BY CONSTRUCTION rather than by coincidence: a
    ``last``-pooled read takes the final response position, and the ``window_end`` variant's window is
    end-anchored, so its last position IS the response's last position. Two names, one measurement.

    Reported rather than dropped, and the choice matters. Dropping one silently would change the
    denominator of every tally without saying so; a run that names its duplicates lets a downstream
    count use the honest number (20 distinct cells rather than 24) while the artifact still contains
    the cell a reader might go looking for. Nothing here interprets: it compares the numbers -- every
    numeric field of every layer's read, per :func:`_read_fingerprint`, which is what makes
    "bit-identical" above a statement about the whole read rather than about a chosen few of its fields.
    """
    fingerprints: dict[tuple[str, str, str], tuple[tuple[float, ...], ...]] = {}
    for variant, reads in reads_by_variant.items():
        by_cell: dict[tuple[str, str, str], list[ContrastRead]] = {}
        for read in reads:
            by_cell.setdefault((variant, read.pooling, read.concept), []).append(read)
        for cell, cell_reads in by_cell.items():
            ordered = sorted(cell_reads, key=lambda read: read.layer)
            fingerprints[cell] = tuple(_read_fingerprint(read) for read in ordered)
    groups: dict[tuple[tuple[float, ...], ...], list[str]] = {}
    for cell, fingerprint in fingerprints.items():
        groups.setdefault(fingerprint, []).append("|".join(cell))
    duplicates = [sorted(names) for names in groups.values() if len(names) > 1]
    return {
        "n_cells": len(fingerprints),
        "n_distinct_cells": len(groups),
        "identical_cell_groups": sorted(duplicates),
        "note": (
            "cells listed together produced bit-identical reads at every layer; a tally over cells "
            "should use n_distinct_cells as its denominator. window_end/last matching "
            "all_response/last is expected by construction (an end-anchored window's last position "
            "is the response's last position), not a bug."
        ),
    }


def peak_layers_lineage(
    reads_by_variant: Mapping[str, Sequence[ContrastRead]],
) -> dict[str, object]:
    """Which cells emitted a peak, which were withheld by the floor, and the excess behind each.

    Written beside the peak handoff so a consuming stage's artifact can say which contrast fed it and
    how well that cell actually separated. The 2026-08-22 causal boxes recorded neither, and the
    ``peak_layers.json`` they consumed turned out to be byte-identical to the weaker of the two
    contrast runs -- unrecoverable after the fact because nothing on either side wrote it down.

    Reads the peak and applies the floor through the same :func:`_best_read_per_cell` and
    :func:`_clears_placebo_floor` the selector uses, so "which cells were withheld" is derived from the
    withholding rather than reconstructed beside it. An audit that re-implements what it audits can
    disagree with it, and would then describe a handoff nobody made.
    """
    emitted: dict[str, dict[str, float | int]] = {}
    withheld: dict[str, dict[str, float | int]] = {}
    for (concept, variant, pooling), read in _best_read_per_cell(reads_by_variant).items():
        target = emitted if _clears_placebo_floor(read) else withheld
        target[f"{concept}|{variant}|{pooling}"] = {
            "layer": read.layer,
            "auc_above_placebo": read.auc_above_placebo,
        }
    return {
        "min_excess_required": PEAK_LAYER_MIN_EXCESS,
        "n_cells_emitted": len(emitted),
        "n_cells_withheld": len(withheld),
        "emitted": emitted,
        "withheld_never_cleared_placebo": withheld,
    }


def peak_selection_significance(
    reads_by_variant: Mapping[str, Sequence[ContrastRead]],
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    """Layer-selection-corrected significance keyed ``concept -> variant -> pooling``, like the peaks.

    ``select_peak_layers`` reports WHICH layer peaks under each pooling variant; this reports whether
    that peak's separation survives correcting for the layer having been picked as the best of many
    (see :class:`reward_hacking.interp.prompt_contrast.LayerSelection`). It carries the peak's
    within-layer p beside the selection-corrected one.

    The two share their KEYS and, by design, not their peak layer. ``peak_layer`` here is the argmax
    RAW AUC, matching the statistic the max-over-layers null is built from; the layer handed to the
    causal stage is the argmax AUC-ABOVE-PLACEBO. With a flat ~0.5 placebo mean they coincide, and
    nothing forces them to. So each cell also carries ``causal_handoff_layer`` -- the
    ``select_peak_layers`` value for the same key -- because a reader lining the two files up by key
    alone would otherwise attribute a ``selection_corrected_p`` to a layer the causal stage never
    steered, ablated or patched.

    ``causal_handoff_layer`` is ``None`` for a cell the floor WITHHELD, and looked up rather than
    indexed for exactly that reason. Every cell appears here -- the significance block reports what a
    cell measured whether or not it earned a handoff -- while ``select_peak_layers`` now omits a cell
    whose best layer never clears its placebo, so the two no longer cover the same keys. A direct
    index raised ``KeyError`` on precisely the situation the floor exists to make visible, at the END
    of a contrast run with all the generation already paid for, taking ``contrast_metrics.json`` with
    it. ``None`` is also the honest value: there is no handoff layer to name.
    """
    causal_handoff = select_peak_layers(reads_by_variant)
    selection: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for variant, reads in reads_by_variant.items():
        for concept, by_pooling in selection_by_group(reads).items():
            handoffs = causal_handoff.get(concept, {}).get(variant, {})
            for pooling, cell in by_pooling.items():
                cell["causal_handoff_layer"] = handoffs.get(pooling)
            selection.setdefault(concept, {})[variant] = by_pooling
    return selection


def position_selector_for_variant(variant: str, *, window_length: int) -> PositionSelector:
    """Return the ``GenerationRecord -> position mask`` selector for one pooling variant."""
    if variant == VARIANT_ALL_RESPONSE:
        return response_positions
    if variant == VARIANT_WINDOW_START:
        return lambda record: matched_window_positions(record, length=window_length, anchor="start")
    if variant == VARIANT_WINDOW_END:
        return lambda record: matched_window_positions(record, length=window_length, anchor="end")
    raise ValueError(f"unknown pooling variant {variant!r}; expected one of {DEFAULT_VARIANTS}")


def mean_residual_norm(activations: torch.Tensor) -> float:
    """Mean L2 norm across the positions of a ``[seq, d]`` residual capture.

    The scale a steering alpha is expressed in. Measured per layer per prompt because residual norms
    grow by an order of magnitude through the trunk, so one constant cannot be the right push at
    both ends of it.
    """
    return float(activations.norm(dim=-1).mean().item())


def steering_alpha(residual_norm: float, scale: float) -> float:
    """Steering magnitude as ``scale`` times the layer's measured residual norm."""
    if residual_norm <= 0.0:
        raise ValueError(f"residual norm must be positive to scale an alpha, got {residual_norm}")
    return scale * residual_norm


def grader_filename_from_transcript(transcript: str) -> str:
    """Read the grader filename out of the stimulus transcript's grader-read observation.

    The transcripts are built by ``prompt_contrast._task_transcript``, which appends a synthetic
    grader read and its ``<result command="cat grade.py" exit=0>`` observation. That observation is
    what gets matched, rather than the filename being hardcoded here (the task spec owns it) and
    rather than the twins' ``<run>`` action being parsed (see :data:`_GRADER_OBSERVATION_RE` for why
    that does not work). A format drift raises here instead of silently reading the wrong file.
    """
    named = sorted(set(_GRADER_OBSERVATION_RE.findall(transcript)))
    if len(named) != 1:
        raise ValueError(
            'expected exactly one `<result command="cat <file>" exit=0>` grader-read observation '
            f"in the stimulus transcript, found {named}"
        )
    return named[0]


def behavioral_observables(record: GenerationRecord, *, grader_filename: str) -> dict[str, float]:
    """Behavioural reads on one generated response, all counts so arms average and subtract.

    ``len(response_text)`` alone cannot tell a steering effect from a verbosity change, so the
    reads that matter here are protocol-level: how many actions the model emitted, how many of its
    commands name the grader file (the reaching observable this whole project is about), and whether
    it took the honest exit. Length is kept because a collapse in output IS an effect worth seeing.
    """
    actions = parse_actions(record.response_text)
    commands = [action.payload for action in actions if action.kind is ActionKind.RUN]
    return {
        "response_tokens": float(record.n_generated),
        "hit_token_cap": float(record.hit_token_cap),
        "n_actions": float(len(actions)),
        "n_run_commands": float(len(commands)),
        "n_commands_naming_grader": float(
            sum(1 for command in commands if grader_filename in command)
        ),
        "n_abort_actions": float(sum(1 for action in actions if action.kind is ActionKind.ABORT)),
        "n_empty_actions": float(sum(1 for action in actions if action.kind is ActionKind.EMPTY)),
    }


@dataclass(frozen=True)
class SteerRecord:
    """One steered or ablated generation: which arm produced it and what it did.

    ``deltanet_kernel`` is what the Gated DeltaNet kernel functions were bound to while this arm
    decoded. It is on the record rather than only on the run's summary because the comparison this
    instrument makes is real-arm-against-placebo-arm, and the fused decode kernel and the torch
    fallback produce different token ids from the same distribution -- so two arms decoded under
    different bindings are not a comparison, and :func:`summarize_steering` refuses to average them.
    """

    concept: str
    layer: int
    prompt_index: int
    arm: str
    mode: str
    alpha: float
    alpha_scale: float
    residual_norm: float
    observables: dict[str, float]
    deltanet_kernel: dict[str, str]


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` on an empty sequence (so a missing arm reads as null, not 0)."""
    return sum(values) / len(values) if values else None


def summarize_steering(records: Sequence[SteerRecord]) -> list[dict[str, object]]:
    """Per ``(concept, layer, mode, alpha_scale, observable)``: the real arm against its placebos.

    The placebo mean is the whole point -- a shifted observable under the real axis means nothing
    unless it exceeds what matched-norm random directions do at the same layer and magnitude. Rows
    where an arm is absent carry ``None`` rather than a zero that would read as a measurement.

    Refuses to summarise records that decoded under two Gated DeltaNet kernel bindings, before it
    averages anything: ``real_minus_placebo`` is a difference between two sets of generations, and a
    difference taken across the fused kernel and the torch fallback measures the kernels.
    """
    assert_one_deltanet_kernel(
        (record.deltanet_kernel for record in records), what="this steering summary"
    )
    groups: dict[tuple[str, int, str, float], list[SteerRecord]] = {}
    for record in records:
        key = (record.concept, record.layer, record.mode, record.alpha_scale)
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, object]] = []
    for (concept, layer, mode, alpha_scale), group in sorted(groups.items()):
        real = [record for record in group if not record.arm.startswith(PLACEBO_ARM_PREFIX)]
        placebo = [record for record in group if record.arm.startswith(PLACEBO_ARM_PREFIX)]
        for observable in sorted({name for record in group for name in record.observables}):
            real_mean = _mean_or_none([record.observables[observable] for record in real])
            placebo_mean = _mean_or_none([record.observables[observable] for record in placebo])
            difference = (
                real_mean - placebo_mean
                if real_mean is not None and placebo_mean is not None
                else None
            )
            rows.append(
                {
                    "concept": concept,
                    "layer": layer,
                    "mode": mode,
                    "alpha_scale": alpha_scale,
                    "observable": observable,
                    "real_mean": real_mean,
                    "placebo_mean": placebo_mean,
                    "real_minus_placebo": difference,
                    "n_placebo_arms": len(placebo),
                    "n_prompts": len({record.prompt_index for record in group}),
                }
            )
    return rows


@dataclass(frozen=True)
class Deadline:
    """A monotonic wall-clock budget for a stage, so a slow generation cannot run a box dry.

    Checked between units of work rather than interrupting one, so an expired deadline stops the
    stage with everything it has finished intact and recorded, not with a truncated artifact.
    """

    limit_seconds: float | None
    started_at: float

    def expired(self, *, now: float) -> bool:
        """Whether the budget is spent; always False when no limit was set."""
        if self.limit_seconds is None:
            return False
        return now - self.started_at > self.limit_seconds


@dataclass(frozen=True)
class ContrastCoverage:
    """What the contrast stage actually managed to read -- the denominator behind every number.

    A stage that reports a null must also report what it examined, skipped and dropped, or the null
    is unfalsifiable. ``pairs_attempted`` minus ``pairs_completed`` is never silent: every drop has
    a counted reason, and a deadline stop is a flag rather than a shorter list of reads.
    """

    pairs_available: int
    pairs_attempted: int
    pairs_completed: int
    pairs_dropped_empty_response: int
    conflicting_responses_truncated: int
    original_responses_truncated: int
    stopped_on_deadline: bool
    # Which twin was submitted first, per chunk, under the arrival shuffle. Recorded rather than
    # merely performed: the retained runs' strict conflicting-even/original-odd parity was invisible
    # in their artifacts, so a reader could not tell whether it had been controlled for.
    twin_arrival_conflicting_first: tuple[tuple[bool, ...], ...] = ()


class PooledKey(NamedTuple):
    """Key into the pooled generation-phase activations: which variant, pooling and twin side."""

    variant: str
    pooling: str
    side: str

    def as_path(self) -> str:
        """Flatten to a string key, so a re-analysis reads the artifact without importing this."""
        return f"{self.variant}/{self.pooling}/{self.side}"


def axis_out_dir(base_out_dir: Path, concept: str, pooling: str) -> Path:
    """Per-axis metrics directory under the run's out-dir, in the eval-awareness layout.

    ``<out-dir>/axes/<concept-hyphenated>-<pooling>``, so shortcut / deception / the contradiction
    control land beside eval-awareness (``eval_awareness`` -> ``eval-awareness-mean``) in ONE uniform
    per-axis layout the run uploads. Matching that established ``axes/eval-awareness-<pooling>`` shape
    is what lets the contrast stage's ``--validated-eval-dir`` point straight at the eval-awareness
    axis this stage produces, and what makes all four axes read the same way on disk.
    """
    return base_out_dir / "axes" / f"{concept.replace('_', '-')}-{pooling}"


def axis_index_row(result: ConceptAxisResult, pooling: str, metrics_dir: Path) -> dict[str, object]:
    """One row of the axis-probe index: which axis was validated, where, and its control summary.

    Aggregate only (counts and layer indices, no item text), so it is safe to upload. The
    controls-cleared counts are a diagnosis, never a gate: an axis clearing neither control is a
    lead about transfer or the stimulus, reported plainly rather than treated as a failure.
    """
    clearing_both = [read.layer for read in result.reads if read.clears_null and read.beats_placebo]
    return {
        "concept": result.concept,
        "pooling": pooling,
        "metrics_dir": metrics_dir.name,
        "n_layers": len(result.reads),
        "n_layers_clearing_both_controls": len(clearing_both),
        "layers_clearing_both_controls": clearing_both,
    }


# --------------------------------------------------------------------------------------
# GPU stages (model load, generate, fit, patch): behind functions, guarded, not offline-tested
# --------------------------------------------------------------------------------------


def _require_cuda() -> None:
    """Fail fast on a CPU-only box: the Qwen3.5 linear-attention path needs a CUDA Triton kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "the interp run harness is GPU-only (Qwen3.5 CPU route is closed); no CUDA device found"
        )


def _free_gib() -> float:
    """Free VRAM on the current device, in GiB, read at startup so nothing hardcodes a budget."""
    free_bytes, _total = torch.cuda.mem_get_info()
    return free_bytes / (1024**3)


@dataclass(frozen=True)
class FitLensArgs:
    """Inputs for the lens-fit stage, one object so the GPU driver stays a thin dispatch."""

    model_id: str
    episode_dir: Path
    out_dir: Path
    raw_dir: Path
    limit: int | None
    max_fit_prompts: int
    max_seq_len: int | None
    max_seq_len_ceiling: int
    reasoning_prompts: int
    reasoning_sampling: SamplingConfig
    recon_eval_prompts: int
    recon_max_positions: int
    gen_engine: str
    vllm_gpu_fraction: float


def capture_model_then_generator(
    model_id: str, *, gen_engine: str, sampling: SamplingConfig, vllm_gpu_fraction: float
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ResponseGenerator]:
    """Load the capture model, THEN build the generator, and hand a stage all three.

    The one place the order lives, because the order is not a preference and getting it wrong is not
    subtle: bringing up a vLLM engine first leaves ``AutoModelForCausalLM.from_pretrained`` unable to
    build a causal-LM head for the same Qwen3.5 checkpoint afterwards. Measured on the 0.8B here
    (2026-08-21): the engine starts, generates fine, and the later HuggingFace load dies with
    ``AttributeError: 'Qwen3_5Config' object has no attribute 'vocab_size'``, raised from
    transformers' composite-config attribute forwarding. Importing vllm is harmless; it is the
    engine's startup that leaves transformers in that state, so only the ORDER fixes it.

    Loading the capture model first costs the engine nothing, which is the other half of why this is
    safe: vLLM sizes its budget as a fraction of TOTAL VRAM, not of what is free (its own startup log
    on that run: "Desired GPU memory utilization is (0.3, 6.62 GiB)" against a 22.06 GiB card). So a
    resident capture model does not shrink the engine's share -- it only has to leave that share free.

    Two engine-startup prerequisites worth knowing before a rented box burns an hour, both observed on
    this dev box: vLLM's flashinfer top-k/top-p sampler is JIT-compiled at startup, so ``ninja`` must
    be on ``PATH`` (it ships in the venv's ``bin``, which ``scripts/resource-limits.sh`` strips because
    it runs under ``systemd-run`` -- pass ``PATH`` through ``env``), and the build needs the CUDA
    toolkit headers (``curand.h``). Where those are absent, ``VLLM_USE_FLASHINFER_SAMPLER=0`` falls
    back to vLLM's native torch sampler and the engine starts.
    """
    bridge_deltanet_decode_kernel()
    model, tokenizer = load_model_and_tokenizer(model_id)
    if gen_engine == GEN_ENGINE_VLLM:
        generator: ResponseGenerator = VLLMResponseGenerator(
            model_id, sampling=sampling, gpu_fraction=vllm_gpu_fraction
        )
    else:
        generator = HFResponseGenerator(model, tokenizer, sampling=sampling)
    return model, tokenizer, generator


@cache
def _bridge_deltanet_decode_kernel_once() -> tuple[tuple[str, object], ...]:
    """Apply the bridge on the first call and remember its report.

    Cached to keep the one TRUTHFUL report, not to dodge a raise. The alias can only be installed once
    per process, and `games.deltanet_kernels.bridge_decode_kernel` checks the already-exported gate
    before the too-late guard, so a second call after a successful one returns a ``bridged: False``
    report rather than raising -- and a stage may load more than once. Its reason does name the
    aliased-earlier-in-this-process case, but every other field describes a bridge that did not happen,
    so an artifact stamped with it would understate what its records ran under. Kept as a tuple of
    pairs so the public wrapper hands every caller its own dict rather than the cached one.
    """
    return tuple(bridge_and_check_decode_kernel().items())


def bridge_deltanet_decode_kernel() -> dict[str, object]:
    """Bridge the Gated DeltaNet decode kernel onto fla's fused one, once per process, and report it.

    Called from every seam here that loads a Qwen3.5 model, because the alias only reaches the decode
    path if it is registered BEFORE ``transformers.models.qwen3_5.modeling_qwen3_5`` is imported --
    which the first model load does. Memoized so every seam gets the report of the call that actually
    installed the alias: a later call finds the name resolving (to the alias) and returns a
    ``bridged: False`` no-op report, which an artifact would then carry as its bridge provenance.

    What it buys and what it costs: 1.14-1.51x per HF decode step at 0.8B, and greedy tokens that
    diverge from the torch fallback's from step 21 on (probe I1, 2026-09-02), so a run's records name
    the kernel they came out of (:data:`DELTANET_KERNEL_FIELD`) and nothing pools across two.
    """
    return dict(_bridge_deltanet_decode_kernel_once())


def _generate_reasoning_corpus(
    args: FitLensArgs, pairs: Sequence[StimulusPair]
) -> tuple[list[str], AutoModelForCausalLM, AutoTokenizer]:
    """Generate the reasoning half of the fit corpus, then hand back the model the fit runs on.

    The engine is released as soon as the corpus exists, because the fit downstream sizes its
    ``dim_batch`` off FREE VRAM: an engine still holding its share would cost the fit a tier in
    exchange for text it has already produced. Released rather than never started, since the order
    ``capture_model_then_generator`` enforces means it cannot be started later.
    """
    prompts = [pair.conflicting_transcript for pair in pairs[: args.reasoning_prompts]]
    stage_sampler = resolved_sampler_for(args.reasoning_sampling, engine=args.gen_engine)
    model, tokenizer, generator = capture_model_then_generator(
        args.model_id,
        gen_engine=args.gen_engine,
        sampling=args.reasoning_sampling,
        vllm_gpu_fraction=args.vllm_gpu_fraction,
    )
    records = generator.generate_records(prompts)
    generator.release()
    for record in records:
        require_record_sampler(record, stage_sampler, stage="fit-lens")
    return [record.response_text for record in records], model, tokenizer


LENS_FIT_SIDECAR_SUFFIX = ".fit.json"
"""Suffix of the sidecar written beside a fitted lens, naming the fit it is the product of.

Written by ``fit-lens`` and read by ``patch-decode --lens-path``: without it a saved lens is an opaque
tensor file, and reusing one is a guess about which corpus and which window produced it. Spelled as a
suffix on the whole filename (``lens.local.pt.fit.json``) rather than through ``with_suffix``, which
would eat the ``.local`` marker that says the file is never uploaded.
"""


def lens_fit_sidecar_path(lens_path: Path) -> Path:
    """Where a lens's fit sidecar lives: beside it, under the lens's own name."""
    return lens_path.with_name(lens_path.name + LENS_FIT_SIDECAR_SUFFIX)


def lens_fit_identity(
    *,
    config: JacobianConfig,
    corpus: Sequence[str],
    corpus_filename: str,
    model_weights_identity: str,
    skip_first: int,
) -> dict[str, object]:
    """Everything a fitted lens is a function of, as the resume marker and the sidecar record it.

    Derived from the very ``config`` the fit runs with, so the identity cannot drift from the fit:
    ``corpus_sha256`` is :func:`corpus_hash` over :func:`cap_fit_prompts` of the offered corpus,
    because ``fit_lens`` caps the list to ``config.max_fit_prompts`` and a lens is an average of
    per-prompt Jacobians over exactly the strings it iterated, in that order -- a digest of the whole
    offered corpus would certify strings the lens never saw. ``n_fit_prompts`` is that fitted count and
    ``n_corpus`` the offered one. ``model_weights_identity`` pins the weights the way the lens cache
    does (:func:`resolve_weights_identity`: a hub revision or a local digest), since ``model_id`` is a
    string that a rewritten local dir or a moved hub revision leaves unchanged. ``dim_batch`` rides
    along because it is part of the fit's reduction order, but a reuse does not fit, so it is reported
    rather than compared (see :data:`LENS_REUSE_COMPARED_FIELDS`). ``corpus_filename`` names the
    corpus file beside the sidecar, so a reader can see what the lens read without guessing.
    """
    fitted = cap_fit_prompts(corpus, config)
    return {
        "model_id": config.model_id,
        "model_weights_identity": model_weights_identity,
        "corpus_sha256": corpus_hash(fitted),
        "n_fit_prompts": len(fitted),
        "n_corpus": len(corpus),
        "corpus_filename": corpus_filename,
        "max_seq_len": config.max_seq_len,
        "dim_batch": config.dim_batch,
        "skip_first": skip_first,
        "jlens_commit": JLENS_COMMIT,
    }


LENS_REUSE_COMPARED_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_weights_identity",
    "corpus_sha256",
    "max_seq_len",
    "skip_first",
    "jlens_commit",
)
"""The identity fields a reuse or a resume must match on: what the lens IS, not how it was scheduled.

``n_fit_prompts``, ``n_corpus`` and ``corpus_filename`` follow from the digest and the tree;
``dim_batch`` is the one deliberate omission, because it describes a fit's schedule and neither a
reuse (which does no fit) nor a resume (whose running mean is the same function of the same prompts
at any width) depends on it -- the recorded value is carried into the artifact instead of being held
against a value that never applies.
"""


def lens_identity_differences(
    wanted: Mapping[str, object], found: Mapping[str, object]
) -> list[str]:
    """Name every compared field on which a recorded lens identity differs from the one this run wants."""
    return [
        f"{field}: the saved lens has {found.get(field, '<absent>')!r}, this run wants "
        f"{wanted.get(field, '<absent>')!r}"
        for field in LENS_REUSE_COMPARED_FIELDS
        if found.get(field) != wanted.get(field)
    ]


def reusable_lens(
    wanted: Mapping[str, object], sidecar: Mapping[str, object]
) -> tuple[bool, list[str]]:
    """Whether a saved lens answers the fit this run would otherwise do, and what differs if not.

    The C8 gate: ``patch-decode`` reuses ``fit-lens``'s lens only when the two corpora are the same
    strings in the same order and the fit that produced it read them through the same weights, window
    and jlens. Anything else refits, because a lens fitted on other text is a different transform and
    decoding a delta through it would report token lists nobody's corpus produced. Returns the verdict
    and the differing fields, so a refusal says which term moved instead of only that one did.
    """
    differences = lens_identity_differences(wanted, sidecar)
    return not differences, differences


LENS_FIT_RESUME_MARKER_FILENAME = "lens_fit_resume.local.json"
"""The marker ``fit-lens`` writes beside its checkpoint BEFORE the fit starts, naming the fit it is.

It used to be a bare corpus digest. ``jlens.fit`` resumes into the prompt list by position and
cross-checks only ``source_layers`` / ``target_layer`` / ``skip_first``, never ``max_seq_len``, so once
the window became corpus-derived (128 fixed before, up to the ceiling now, and settable by two flags) a
checkpoint fitted at one window would have resumed under another and averaged Jacobians over two
windows while the report claimed the new one. The marker is now the whole :func:`lens_fit_identity`,
compared on :data:`LENS_REUSE_COMPARED_FIELDS`; a checkpoint beside the old digest-only marker has no
marker at this name and is refused as unprovable, which is the honest reading of it.
"""


def guard_resume(checkpoint_path: Path, marker_path: Path, identity: Mapping[str, object]) -> None:
    """Refuse any resume whose fit is not PROVABLY the one the checkpoint was started for.

    ``jlens.fit`` resumes into the prompt LIST cross-checking only ``source_layers`` /
    ``target_layer`` / ``skip_first`` -- nothing identifies the corpus, the window or the weights -- so
    a checkpoint fitted on a different corpus, at a different ``max_seq_len`` or on other weights
    silently corrupts the running mean into an average of two fits. ``identity`` is the
    :func:`lens_fit_identity` this run is about to fit, and the marker holds the one the checkpoint
    was started for; any difference on :data:`LENS_REUSE_COMPARED_FIELDS` refuses and says which term.

    The marker is written BEFORE the fit starts, not after it finishes, and a checkpoint with no marker
    is refused rather than resumed. A missing or unreadable marker is exactly the unprovable case this
    guard exists to catch: an interrupted fit that never got to write one, or one written before the
    window was recorded.
    """
    if checkpoint_path.exists():
        if not marker_path.exists():
            raise RuntimeError(
                f"resume abort: checkpoint at {checkpoint_path} has no fit marker at {marker_path}, "
                f"so the corpus, window and weights it was fitted on cannot be proven; jlens.fit "
                f"resumes by list position and cross-checks none of them. Delete the checkpoint to "
                f"start fresh."
            )
        stored = cast("dict[str, object]", json.loads(marker_path.read_text()))
        differences = lens_identity_differences(identity, stored)
        if differences:
            raise RuntimeError(
                f"resume abort: checkpoint at {checkpoint_path} was started for a different fit "
                f"({len(differences)} identity fields differ): "
                + "; ".join(differences)
                + ". Delete the checkpoint to start fresh, or relaunch with the marker's settings."
            )
        logger.info("resuming the lens fit from %s (fit identity matches)", checkpoint_path)
        return
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    logger.info("fresh lens fit: wrote the fit marker %s before starting", marker_path)


def run_fit_lens(args: FitLensArgs) -> None:
    """Fit and save a Jacobian lens over the twin transcripts plus generated reasoning (GPU).

    The fit window is DERIVED from the corpus rather than left at ``jlens.fit``'s 128-token default
    (`derive_max_seq_len`), because this corpus is chat-formatted agentic transcripts of 1.3k-22.3k
    tokens plus generated reasoning: at 128 the fit averaged Jacobians over the shared system-prompt
    prefix and never saw the twins' divergent content, which is what the reads it serves are about.
    ``--max-seq-len`` and ``--max-seq-len-ceiling`` are both ceilings on that derivation and the report
    carries what any remaining truncation cost.

    The lens lands with a sidecar naming the fit it came from (:func:`lens_fit_identity`), which is what
    lets ``patch-decode --lens-path`` reuse it instead of paying a second fit -- and what lets it refuse
    to, when the two corpora differ.

    ``fit_report.json`` carries the DeltaNet decode bridge's report beside the binding, like every
    sibling artifact here, and both are read once the model exists rather than while the report is being
    assembled: `bound_deltanet_kernels` raises on an upstream dispatch change, and after the fit that
    raise arrives hours late. The FULL four-kernel binding, because the reasoning corpus was generated.
    """
    require_raw_dir_outside_episode_dir(args.raw_dir, args.episode_dir)
    _require_cuda()
    jl = _require_jlens()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "fit-lens on %s, free VRAM %.1f GiB, generating on %s",
        torch.cuda.get_device_name(0),
        _free_gib(),
        args.gen_engine,
    )

    pairs = build_stimulus_pairs(args.episode_dir, limit=args.limit)
    stage_sampler = resolved_sampler_for(args.reasoning_sampling, engine=args.gen_engine)
    reasoning_texts, model, tokenizer = _generate_reasoning_corpus(args, pairs)
    # Read here rather than in the report: `bound_deltanet_kernels` raises on an upstream dispatch
    # change, which must cost seconds after the load rather than surface after the whole fit. The full
    # four-kernel binding, because generating the reasoning corpus dispatched the per-token pair too.
    kernel_bridge = bridge_deltanet_decode_kernel()
    deltanet_kernel = bound_deltanet_kernels()
    corpus = build_fit_corpus(pairs, reasoning_texts)

    wrapped = jl.from_hf(model, tokenizer)
    # After the wrap, because `from_hf` may set `add_bos_token` on this very tokenizer, and the window
    # has to be derived from the lengths the fit's own `encode` will see rather than one token short.
    seq_plan = corpus_seq_len_plan(
        tokenizer, corpus, max_seq_len=args.max_seq_len, ceiling=args.max_seq_len_ceiling
    )
    dim_batch = dim_batch_for_free_vram(_free_gib())
    checkpoint_path = args.raw_dir / "fit_checkpoint.local.pt"
    config = JacobianConfig(
        model_id=args.model_id,
        source="fit_own",
        max_fit_prompts=args.max_fit_prompts,
        dim_batch=dim_batch,
        max_seq_len=seq_plan.max_seq_len,
        checkpoint_path=checkpoint_path,
    )
    corpus_filename = "fit_corpus.local.json"
    (args.raw_dir / corpus_filename).write_text(json.dumps(corpus))
    # After the load, so a hub id resolves from the cache the weights just came out of.
    fit_identity = lens_fit_identity(
        config=config,
        corpus=corpus,
        corpus_filename=corpus_filename,
        model_weights_identity=resolve_weights_identity(args.model_id),
        skip_first=fit_skip_first(jl),
    )
    guard_resume(checkpoint_path, args.raw_dir / LENS_FIT_RESUME_MARKER_FILENAME, fit_identity)
    lens = fit_lens(config, wrapped, corpus, jl)

    # The lens is derived from the twin corpus and is large; it stays in the raw tree, which is
    # never uploaded, and only its filename is reported in the uploadable metrics.
    lens_path = args.raw_dir / "lens.local.pt"
    lens.save(str(lens_path))  # pyright: ignore[reportAttributeAccessIssue]  # save first
    logger.info("saved lens to %s", lens_path)
    verify_lens_roundtrip(jl, lens, lens_path)
    lens_fit_sidecar_path(lens_path).write_text(
        json.dumps(fit_identity, indent=2, sort_keys=True) + "\n"
    )

    # Fit quality: how well the saved lens reproduces the model's OWN logits, per source layer,
    # against the free logit-lens baseline. Read on a slice of the fit corpus (the lens is
    # corpus-averaged, so in-corpus vs held-out barely differ, and this keeps the eval self-
    # contained). Without this the artifact carries counts only and lens quality is unknowable.
    fit_quality = evaluate_reconstruction(
        lens,
        wrapped,
        corpus[: args.recon_eval_prompts],
        max_seq_len=seq_plan.max_seq_len,
        max_positions=args.recon_max_positions,
    )
    (args.out_dir / "fit_report.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "n_corpus": len(corpus),
                "n_fit_prompts": fit_identity["n_fit_prompts"],
                "n_pairs": len(pairs),
                "n_reasoning": len(reasoning_texts),
                "dim_batch": dim_batch,
                "max_seq_len": seq_plan.max_seq_len,
                "seq_len_plan": seq_plan.as_payload(),
                "sampler": stage_sampler.as_payload(),
                "lens_filename": lens_path.name,
                "lens_fit_sidecar": lens_fit_sidecar_path(lens_path).name,
                "lens_fit_identity": fit_identity,
                "deltanet_kernel_bridge": kernel_bridge,
                DELTANET_KERNEL_FIELD: deltanet_kernel,
                "fit_quality": fit_quality_payload(fit_quality),
            },
            indent=2,
        )
    )


@dataclass(frozen=True)
class AxisProbeArgs:
    """Inputs for the axis-probe stage: validate every concept axis under one model load."""

    model_id: str
    out_dir: Path
    concepts: tuple[str, ...]
    poolings: tuple[str, ...]
    limit: int | None
    layer_stride: int
    batch_size: int
    n_placebos: int
    probe_config: ProbeConfig


def run_axis_probe(args: AxisProbeArgs) -> None:
    """Validate ALL requested concept axes with the same battery, one per (concept, pooling) (GPU).

    The fix for the single-axis defect: eval-awareness used to be the only concept whose direction
    was validated (probe accuracy vs placebo, split-half, per-layer significance), while shortcut /
    deception / contradiction shipped as bare diff-of-means directions with no quality metrics. This
    runs the identical :func:`probe_concept` battery over every concept in ``args.concepts`` and each
    pooling, under ONE model load, and writes each axis's metrics to its own directory in the same
    layout eval-awareness already uses -- so every axis a downstream read projects onto is
    self-describing. An index of what was produced is written for the run to upload, carrying the
    DeltaNet decode bridge's report and the binding the capture ran under -- the prefill subset, since
    the probe's forwards have no cache and so never reach the per-token pair.
    """
    _require_cuda()
    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_deltanet_decode_kernel()
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    # The probe captures with no cache, so its forwards never reach the per-token decode pair.
    deltanet_kernel = prefill_deltanet_kernels(bound_deltanet_kernels())
    logger.info(
        "axis-probe on %s (free VRAM %.1f GiB): concepts=%s poolings=%s",
        torch.cuda.get_device_name(0),
        _free_gib(),
        args.concepts,
        args.poolings,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    for concept in args.concepts:
        pairs = AXIS_PROBE_CONCEPTS[concept]
        for pooling in args.poolings:
            spec = CaptureSpec(
                concepts=(concept,),
                pooling=pooling,
                batch_size=args.batch_size,
                limit=args.limit,
                layer_stride=args.layer_stride,
            )
            result = probe_concept(
                model,
                tokenizer,
                spec,
                args.probe_config,
                model_id=args.model_id,
                concept=concept,
                pairs=pairs,
                n_placebos=args.n_placebos,
            )
            metrics_dir = axis_out_dir(args.out_dir, concept, pooling)
            save_axis_artifacts(result, metrics_dir)
            index.append(axis_index_row(result, pooling, metrics_dir))

    (args.out_dir / "axis_probe_index.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "concepts": list(args.concepts),
                "poolings": list(args.poolings),
                "n_placebos": args.n_placebos,
                "deltanet_kernel_bridge": kernel_bridge,
                DELTANET_KERNEL_FIELD: deltanet_kernel,
                "axes": index,
            },
            indent=2,
        )
    )
    logger.info(
        "axis-probe done: validated %d concept axes over %d poolings, wrote %d per-axis metric dirs "
        "under %s/axes",
        len(args.concepts),
        len(args.poolings),
        len(index),
        args.out_dir,
    )


@dataclass(frozen=True)
class ContrastArgs:
    """Inputs for the generation-phase contrast stage."""

    model_id: str
    episode_dir: Path
    out_dir: Path
    raw_dir: Path
    validated_eval_dirs: Mapping[str, Path]
    poolings: tuple[str, ...]
    variants: tuple[str, ...]
    window_length: int
    limit: int | None
    n_placebos: int
    seed: int
    sampling: SamplingConfig
    deadline_seconds: float | None
    gen_engine: str
    vllm_gpu_fraction: float
    gen_batch_pairs: int


def load_validated_eval_axes(
    validated_dirs: Mapping[str, Path], poolings: Sequence[str], model_id: str
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, str]]:
    """Load one VALIDATED eval-awareness axis per requested pooling, before any model load.

    Refuses rather than falling back to a freshly extracted axis: Exp2's fragile last-pool signal
    rested on exactly such an axis, so a fresh one here would produce a headline number nobody
    could trust. Checked first so the failure costs a second, not a model load.
    """
    missing = [pooling for pooling in poolings if pooling not in validated_dirs]
    if missing:
        raise RuntimeError(
            f"no validated eval-awareness axis directory given for pooling(s) {missing}; the "
            "eval-awareness read is the headline one and must not use a fresh unvalidated axis. "
            "Either pass --validated-eval-last-dir (after `python -m "
            "reward_hacking.interp.eval_awareness_probe --pooling last --out-dir <dir>`) or "
            "restrict the run with --poolings mean."
        )
    axes: dict[str, dict[int, torch.Tensor]] = {}
    axis_model_ids: dict[str, str] = {}
    for pooling in poolings:
        axis, axis_model_id = load_validated_eval_direction(
            validated_dirs[pooling], expected_pooling=pooling, model_id=model_id
        )
        axes[pooling] = axis
        axis_model_ids[pooling] = axis_model_id
    return axes, axis_model_ids


def _concept_axes(
    model: object,
    tokenizer: object,
    poolings: Sequence[str],
    validated_eval: Mapping[str, dict[int, torch.Tensor]],
) -> dict[str, dict[str, dict[int, torch.Tensor]]]:
    """Build axes keyed ``concept -> pooling -> layer``: three extracted, plus the validated one.

    Every pooling of a concept comes off one pair of forwards (:func:`extract_concept_directions`),
    so the poolings loop no longer multiplies the capture cost.
    """
    axes: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        concept: extract_concept_directions(
            model,  # pyright: ignore[reportArgumentType]
            tokenizer,  # pyright: ignore[reportArgumentType]
            pairs,
            poolings=poolings,
            batch_size=CONCEPT_BATCH_SIZE,
        )
        for concept, pairs in EXTRACTED_CONCEPTS.items()
    }
    axes[EVAL_AWARENESS] = {pooling: validated_eval[pooling] for pooling in poolings}
    return axes


def _pool_one_side(
    model: object,
    record: GenerationRecord,
    side: str,
    args: ContrastArgs,
) -> dict[PooledKey, dict[int, torch.Tensor]]:
    """Pool one generated response every (variant, pooling) way, off the ONE capture it needed.

    The pooling happens inside the capture's forward hooks (:func:`capture_record_pooled`), one
    layer at a time, so the ``[seq, d]`` residual of every layer is never resident on the host at
    once; the vectors are the same bits the stash-then-pool read produced.
    """
    masks = {
        variant: position_selector_for_variant(variant, window_length=args.window_length)(record)
        for variant in args.variants
    }
    pooled = capture_record_pooled(
        model,  # pyright: ignore[reportArgumentType]
        record,
        masks,
        poolings=args.poolings,
    )
    return {
        PooledKey(variant, pooling, side): by_layer
        for (variant, pooling), by_layer in pooled.items()
    }


def twin_arrival_order(chunk: Sequence[StimulusPair], *, seed: int, chunk_index: int) -> list[bool]:
    """Per pair, whether the CONFLICTING twin is submitted first within this chunk.

    Both twins of a pair have always gone into the same generate call, which rules batch composition
    out as a confound. What was NOT ruled out until 2026-08-24 is arrival PARITY: the prompt list was
    built conflicting-then-original for every pair, so the conflicting twin sat at every even index
    and the original at every odd one, strictly, in every chunk of both retained runs. No evidence it
    biases anything, and no reason to keep a perfectly rank-correlated asymmetry between the arm label
    and the submission slot when removing it costs one shuffle and no extra generation.

    Seeded from the run seed and the chunk index rather than drawn from global state, so the order is
    reproducible and is recorded in the coverage payload; a run that cannot say which twin arrived
    first has replaced a known asymmetry with an unknown one.
    """
    rng = torch.Generator().manual_seed(seed + 7_919 * chunk_index)
    draws = torch.rand(len(chunk), generator=rng)
    return [bool(draw < COIN_FLIP) for draw in draws]


def _generate_twin_chunk(
    generator: ResponseGenerator,
    chunk: Sequence[StimulusPair],
    *,
    conflicting_first: Sequence[bool],
) -> list[dict[str, GenerationRecord]]:
    """Generate both twins of every pair in ``chunk`` in ONE call, keyed by side per pair.

    Where the vLLM speedup comes from: the engine's throughput needs many sequences in flight, so the
    prompts of a whole chunk go in together rather than one at a time. The HF generator loops
    internally and behaves exactly as before.

    ``conflicting_first`` decides each pair's submission order within the chunk (see
    :func:`twin_arrival_order`); the returned mapping is keyed by side either way, so nothing
    downstream needs to know which order a pair drew.

    Each record is then held against the transcript it should be answering. The generator promises
    prompt order, but the consequence of that promise being wrong is the worst kind of silent: the
    conflicting twin's activations filed under ``original`` reads as a real difference between the arms
    when it is pure bookkeeping. ``prompt_text`` is on the record, so the check costs nothing -- and
    with the submission order now shuffled it is the check that makes the shuffle safe.
    """
    ordered_sides = [
        (SIDE_CONFLICTING, SIDE_ORIGINAL) if first else (SIDE_ORIGINAL, SIDE_CONFLICTING)
        for first in conflicting_first
    ]
    transcripts = {
        SIDE_CONFLICTING: lambda pair: pair.conflicting_transcript,
        SIDE_ORIGINAL: lambda pair: pair.original_transcript,
    }
    prompts = [
        transcripts[side](pair)
        for pair, sides in zip(chunk, ordered_sides, strict=True)
        for side in sides
    ]
    records = generator.generate_records(prompts)
    per_pair: list[dict[str, GenerationRecord]] = []
    for pair, sides_order, (first, second) in zip(
        chunk, ordered_sides, batched(records, 2, strict=True), strict=True
    ):
        sides = {sides_order[0]: first, sides_order[1]: second}
        for side, transcript in (
            (SIDE_CONFLICTING, pair.conflicting_transcript),
            (SIDE_ORIGINAL, pair.original_transcript),
        ):
            if sides[side].prompt_text != transcript:
                raise RuntimeError(
                    f"the generator returned records out of order: pair {pair.problem_id}'s {side} "
                    "slot holds a record generated from a different transcript, which would file "
                    "every activation captured from it under the wrong twin"
                )
        per_pair.append(sides)
    return per_pair


def _capture_twin_pooled(
    generator: ResponseGenerator,
    model: object,
    pairs: Sequence[StimulusPair],
    args: ContrastArgs,
) -> tuple[dict[PooledKey, dict[int, torch.Tensor]], ContrastCoverage, list[dict[str, object]]]:
    """Generate both twins of every pair once, pool every way, and count what was lost.

    One generation per prompt feeds every pooling variant: regenerating per variant would compare
    reads taken on DIFFERENT samples of the model's own text, which is not the same experiment and
    costs the run a multiple of its generation budget.

    A pair is committed only when both twins produced tokens, so a single empty response drops that
    pair (counted) instead of either killing the stage or leaving the paired rows misaligned by one.

    Generation runs in chunks of ``args.gen_batch_pairs`` pairs and capture follows within the chunk,
    which is what lets a batching engine do its job while the deadline still bounds the stage. The
    deadline is therefore checked BETWEEN chunks rather than between pairs: vLLM has no wall-clock
    budget inside a call, so a chunk once started runs to completion. That is the knob to lower on a
    box with a hard cutoff, and ``stopped_on_deadline`` records that the stage stopped early either
    way. The capture pass itself stays one un-padded sequence at a time, which is the shape it
    requires: it passes no ``position_ids``, so a padded batch would shift rotary positions while the
    pooled vectors still looked clean.
    """
    deadline = Deadline(limit_seconds=args.deadline_seconds, started_at=time.monotonic())
    stage_sampler = resolved_sampler_for(args.sampling, engine=args.gen_engine)
    rows: dict[PooledKey, dict[int, list[torch.Tensor]]] = {}
    generation_log: list[dict[str, object]] = []
    attempted = 0
    completed = 0
    dropped = 0
    truncated = {SIDE_CONFLICTING: 0, SIDE_ORIGINAL: 0}
    stopped_on_deadline = False
    arrival_orders: list[tuple[bool, ...]] = []

    # strict=False: the final chunk is legitimately short whenever the pair count is not a
    # multiple of the batch size, and dropping it would silently lose those pairs.
    for chunk_index, chunk in enumerate(batched(pairs, args.gen_batch_pairs, strict=False)):
        if deadline.expired(now=time.monotonic()):
            stopped_on_deadline = True
            logger.warning(
                "contrast deadline of %.0fs reached after %d completed pairs; stopping here",
                args.deadline_seconds,
                completed,
            )
            break
        conflicting_first = twin_arrival_order(chunk, seed=args.seed, chunk_index=chunk_index)
        arrival_orders.append(tuple(conflicting_first))
        chunk_records = _generate_twin_chunk(generator, chunk, conflicting_first=conflicting_first)
        for pair, sides in zip(chunk, chunk_records, strict=True):
            attempted += 1
            per_pair: dict[PooledKey, dict[int, torch.Tensor]] = {}
            pair_truncated = {SIDE_CONFLICTING: 0, SIDE_ORIGINAL: 0}
            pair_ok = True
            for side in (SIDE_CONFLICTING, SIDE_ORIGINAL):
                record = sides[side]
                require_record_sampler(record, stage_sampler, stage="contrast")
                generation_log.append(
                    {
                        "problem_id": pair.problem_id,
                        "side": side,
                        "response_tokens": record.n_generated,
                        "hit_token_cap": record.hit_token_cap,
                        # Off the RECORD, never off args: only that can say what actually ran.
                        "sampler": record.sampler.as_payload(),
                        "response_text": record.response_text,
                    }
                )
                if record.n_generated == 0:
                    logger.warning(
                        "dropping pair %s: its %s twin generated no tokens, so nothing can be "
                        "pooled",
                        pair.problem_id,
                        side,
                    )
                    pair_ok = False
                    break
                pair_truncated[side] = int(record.hit_token_cap)
                per_pair.update(_pool_one_side(model, record, side, args))
            if not pair_ok:
                dropped += 1
                continue
            for key, by_layer in per_pair.items():
                store = rows.setdefault(key, {})
                for layer, vector in by_layer.items():
                    store.setdefault(layer, []).append(vector)
            for side, count in pair_truncated.items():
                truncated[side] += count
            completed += 1

    coverage = ContrastCoverage(
        pairs_available=len(pairs),
        pairs_attempted=attempted,
        pairs_completed=completed,
        pairs_dropped_empty_response=dropped,
        conflicting_responses_truncated=truncated[SIDE_CONFLICTING],
        original_responses_truncated=truncated[SIDE_ORIGINAL],
        stopped_on_deadline=stopped_on_deadline,
        twin_arrival_conflicting_first=tuple(arrival_orders),
    )
    logger.info("contrast coverage: %s", coverage)
    stacked = {
        key: {layer: torch.stack(vectors, dim=0) for layer, vectors in by_layer.items()}
        for key, by_layer in rows.items()
    }
    return stacked, coverage, generation_log


def _contrast_reads(
    pooled: Mapping[PooledKey, dict[int, torch.Tensor]],
    axes: Mapping[str, dict[str, dict[int, torch.Tensor]]],
    args: ContrastArgs,
) -> tuple[dict[str, list[ContrastRead]], dict[str, int]]:
    """Project every (variant, pooling) read of both twin groups onto every concept axis.

    Returns the reads and the placebo stream seed each cell was actually scored against, so the
    artifact records which random draw produced each placebo band instead of implying that one shared
    seed did all of it. It used to: ``contrast_all_layers`` built a fresh generator from the same seed
    on every call, so all four concepts were scored against the IDENTICAL 100 random directions
    (verified on the retained artifacts -- 9e-6 maximum difference between any two concepts' placebo
    statistics), and "three of four concepts cleared their placebo band" carried ONE draw's worth of
    independence. The derivation now lives inside ``contrast_all_layers``
    (:func:`~reward_hacking.interp.prompt_contrast.placebo_stream_seed`), which closes it for every
    caller rather than only this one; what is left here is reading the same function to record what
    each cell got.
    """
    reads_by_variant: dict[str, list[ContrastRead]] = {}
    placebo_seeds: dict[str, int] = {}
    for variant in args.variants:
        reads: list[ContrastRead] = []
        for pooling in args.poolings:
            conflicting = pooled[PooledKey(variant, pooling, SIDE_CONFLICTING)]
            original = pooled[PooledKey(variant, pooling, SIDE_ORIGINAL)]
            for concept in CONTRAST_CONCEPTS:
                placebo_seeds[f"{variant}|{pooling}|{concept}"] = placebo_stream_seed(
                    args.seed, concept, pooling
                )
                reads.extend(
                    contrast_all_layers(
                        concept,
                        pooling,
                        conflicting,
                        original,
                        axes[concept][pooling],
                        n_placebos=args.n_placebos,
                        seed=args.seed,
                    )
                )
        reads_by_variant[variant] = reads
    return reads_by_variant, placebo_seeds


def run_contrast(args: ContrastArgs) -> None:
    """Generation-phase twin contrast onto the concept axes, with placebo; emit peaks (GPU)."""
    require_raw_dir_outside_episode_dir(args.raw_dir, args.episode_dir)
    _require_cuda()
    validated_eval, axis_model_ids = load_validated_eval_axes(
        args.validated_eval_dirs, args.poolings, args.model_id
    )
    pairs = build_stimulus_pairs(args.episode_dir, limit=args.limit)
    if len(pairs) < MIN_CONTRAST_PAIRS:
        raise RuntimeError(
            f"need >={MIN_CONTRAST_PAIRS} twin pairs for a contrast, got {len(pairs)}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, generator = capture_model_then_generator(
        args.model_id,
        gen_engine=args.gen_engine,
        sampling=args.sampling,
        vllm_gpu_fraction=args.vllm_gpu_fraction,
    )
    logger.info(
        "contrast generating on %s, capturing on the HuggingFace model, %d pairs per generation call",
        generator.engine,
        args.gen_batch_pairs,
    )
    axes = _concept_axes(model, tokenizer, args.poolings, validated_eval)
    torch.save(axes, args.raw_dir / "concept_axes.local.pt")

    pooled, coverage, generation_log = _capture_twin_pooled(generator, model, pairs, args)
    (args.raw_dir / "generation_log.local.json").write_text(json.dumps(generation_log, indent=2))
    torch.save(
        {key.as_path(): by_layer for key, by_layer in pooled.items()},
        args.raw_dir / "pooled_activations.local.pt",
    )
    if coverage.pairs_completed < MIN_CONTRAST_PAIRS:
        raise RuntimeError(
            f"only {coverage.pairs_completed} twin pairs completed, below the "
            f"{MIN_CONTRAST_PAIRS} a paired contrast needs; coverage was {coverage}"
        )

    reads_by_variant, placebo_seeds = _contrast_reads(pooled, axes, args)
    peak_layers = select_peak_layers(reads_by_variant)
    lineage = peak_layers_lineage(reads_by_variant)
    (args.out_dir / "peak_layers.json").write_text(json.dumps(peak_layers, indent=2))
    (args.out_dir / "peak_layers_lineage.json").write_text(json.dumps(lineage, indent=2))
    (args.out_dir / "contrast_metrics.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "poolings": list(args.poolings),
                "variants": list(args.variants),
                "window_length": args.window_length,
                "sampler": resolved_sampler_for(args.sampling, engine=args.gen_engine).as_payload(),
                "deltanet_kernel_bridge": bridge_deltanet_decode_kernel(),
                DELTANET_KERNEL_FIELD: bound_deltanet_kernels(),
                "gen_batch_pairs": args.gen_batch_pairs,
                "n_placebos": args.n_placebos,
                # Named for what it is: the base seed the PER-CELL placebo seeds are derived from.
                # It used to be recorded as a bare "seed", which read as a generation seed and was
                # not one -- SamplingConfig had no seed field at all until 2026-08-24. The generation
                # seed, or its absence, is in the "sampler" block above.
                "placebo_base_seed": args.seed,
                "placebo_seed_by_cell": placebo_seeds,
                "coverage": asdict(coverage),
                "duplicate_read_cells": duplicate_read_cells(reads_by_variant),
                "peak_layers_lineage": lineage,
                "validated_eval_axis_model_ids": axis_model_ids,
                # The peak layer's within-layer p understates its significance because the peak was
                # chosen as the best of many layers; this is the correction for that, carrying both
                # p-values under unambiguous names. Keyed like the peak-layer handoff but NOT
                # peaking at the same layer: this block's peak_layer is the argmax raw AUC (the
                # statistic its null is built from) while peak_layers.json hands the causal stage the
                # argmax AUC-above-placebo, so each cell repeats that one as causal_handoff_layer
                # rather than leaving a reader to line the two files up and conflate them.
                "peak_selection": peak_selection_significance(reads_by_variant),
                # The FULL read, not a summary: the paired t, sign rates and placebo distribution
                # are what a later re-analysis needs, and they are aggregate (no item text). The raw
                # per-placebo AUC draws are dropped here (``metrics_dict``) to keep this file lean;
                # the layer-selection correction they feed is emitted compactly above.
                "reads": [
                    {"variant": variant, **metrics_dict(read)}
                    for variant, reads in reads_by_variant.items()
                    for read in reads
                ],
            },
            indent=2,
        )
    )
    logger.info(
        "contrast done: %d reads over %d variants, peak layers %s",
        sum(len(reads) for reads in reads_by_variant.values()),
        len(reads_by_variant),
        peak_layers,
    )


@dataclass(frozen=True)
class SteerPatchArgs:
    """Inputs for the causal steer/ablate/patch stage."""

    model_id: str
    episode_dir: Path
    out_dir: Path
    raw_dir: Path
    peak_layers_path: Path
    axes_path: Path
    concepts: tuple[str, ...]
    variant: str
    pooling: str
    alpha_scales: tuple[float, ...]
    n_placebos: int
    n_steer_prompts: int
    n_patch_pairs: int
    max_patch_positions: int
    seed: int
    sampling: SamplingConfig
    deadline_seconds: float | None


def load_concept_axis(axes_path: Path, concept: str, pooling: str, layer: int) -> torch.Tensor:
    """Load the axis the CONTRAST stage saved, so the two stages cannot drift apart.

    Re-extracting the axis here would silently measure a different vector whenever the sentence
    pairs, the pooling or the capture path changed between stages -- and it is what forced the
    eval-awareness axis (loaded, validated, never re-extractable) to be excluded from the causal
    tier altogether. Loading the saved axes makes every concept steerable, including that one.
    """
    if not axes_path.exists():
        raise RuntimeError(
            f"no concept axes at {axes_path}; run the `contrast` stage first (it saves the axes it "
            "projected onto). The causal stage must steer the SAME vector the contrast read."
        )
    axes: dict[str, dict[str, dict[int, torch.Tensor]]] = torch.load(axes_path, weights_only=True)
    if concept not in axes:
        raise RuntimeError(f"no {concept!r} axis in {axes_path}; it holds {sorted(axes)}")
    by_pooling = axes[concept]
    if pooling not in by_pooling:
        raise RuntimeError(
            f"the saved {concept!r} axis has no {pooling!r} pooling; it holds {sorted(by_pooling)}"
        )
    by_layer = by_pooling[pooling]
    if layer not in by_layer:
        raise RuntimeError(
            f"the saved {concept!r}/{pooling} axis has no layer {layer}; it covers "
            f"{min(by_layer)}..{max(by_layer)}"
        )
    return by_layer[layer]


def _recorded_peak_layer(
    peak_layers: object, concept: str, variant: str, pooling: str
) -> int | None:
    """One cell's contrast peak, or ``None`` when the placebo floor withheld it.

    For a caller that only RECORDS the number. Absence is a reading here rather than an error:
    :func:`select_peak_layers` omits a cell whose best layer never cleared its matched-norm placebo,
    and two cells of the retained 2026-08-22 contrast were negative at all 32 layers, so a withheld
    cell is the floor working. A caller that STEERS on the layer wants :func:`_peak_layer`, which
    raises on the same absence.

    A payload that is not a JSON object at all still raises, because that is the wrong file rather
    than a withheld cell, and the handoff's provenance is the reason the number is recorded.
    """
    if not isinstance(peak_layers, dict):
        raise TypeError(f"peak_layers must be a JSON object, got {type(peak_layers).__name__}")
    handoff = cast("dict[str, dict[str, dict[str, int]]]", peak_layers)
    return handoff.get(concept, {}).get(variant, {}).get(pooling)


def _peak_layer(peak_layers: object, concept: str, variant: str, pooling: str) -> int:
    """Read one layer out of the contrast stage's ``concept -> variant -> pooling`` handoff.

    Raises when the cell is absent, which is right for the steering caller: it intervenes AT that
    layer, so there is nothing to do without one and a silent substitute would report a steer at a
    layer the contrast never chose. Says so as one message naming every cell the handoff does carry,
    since the absence has two causes worth telling apart -- a contrast that never measured this cell,
    and one whose floor withheld it.
    """
    layer = _recorded_peak_layer(peak_layers, concept, variant, pooling)
    if layer is None:
        raise RuntimeError(
            f"no peak layer for {concept!r}/{variant}/{pooling} in the contrast handoff, which "
            f"carries {_peak_layer_cells(peak_layers)}. A cell whose best layer never cleared its "
            "matched-norm placebo is withheld by select_peak_layers, so this can be that floor "
            "firing rather than a missing contrast."
        )
    return layer


def _peak_layer_cells(peak_layers: object) -> list[str]:
    """Every ``concept|variant|pooling`` cell a peak-layer handoff carries, for a failure message."""
    handoff = cast("dict[str, dict[str, dict[str, int]]]", peak_layers)
    return sorted(
        f"{concept}|{variant}|{pooling}"
        for concept, variants in handoff.items()
        for variant, poolings in variants.items()
        for pooling in poolings
    )


def _chat_ids(
    tokenizer: object, transcript: str, *, thinking: bool = True, prefill: str = ""
) -> torch.Tensor:
    r"""Tokenize a transcript in the SAME chat space the generation-phase reads live in.

    Chat-formatting before tokenizing is what makes the patch readout meaningful: both twins end
    with the identical assistant-turn opener, so the last position asks one shared next-token
    question rather than continuing two different raw texts.

    ``thinking`` selects which opener that is, and the choice is load-bearing rather than cosmetic.
    With it on the template ends ``<|im_start|>assistant\n<think>\n``, so the last position is the
    newline inside the thinking block and a single-token readout there scores the OPENING WORD OF THE
    REASONING TRACE -- the defect the 2026-08-24 rebuild exists to remove. With it off the template
    ends ``<think>\n\n</think>\n\n`` (measured on the Qwen3.5 tokenizer, not assumed), so the last
    position is a visible-answer slot and an action or an option label can legitimately follow it.
    The default stays ``True`` because the contrast and steering stages read in thinking space and
    their axes were fitted there; the patch readout passes ``False``, and every artifact records which.

    ``prefill`` is assistant-side text appended AFTER the chat template, which is how the
    forced-choice readout puts its answer prefill in front of the option label. It is empty for every
    other caller.
    """
    chat = _chat_format(tokenizer, transcript, thinking=thinking)  # pyright: ignore[reportArgumentType]
    encoded = tokenizer(chat + prefill, return_tensors="pt")  # pyright: ignore[reportCallIssue,reportOperatorIssue]
    return encoded["input_ids"][0]  # pyright: ignore[reportIndexIssue]


def _layer_rows(
    model: object, input_ids: torch.Tensor, mask: torch.Tensor, layer: int
) -> torch.Tensor:
    """Capture one layer's ``[seq, d]`` residual over ``input_ids`` (float32, CPU).

    Captured once per run per pair and then indexed per window rather than re-captured per window:
    one pass serves both the placebo's replacement rows and every window's patch-magnitude read.
    """
    captured = capture_positionwise_activations(
        model,  # pyright: ignore[reportArgumentType]
        input_ids,
        mask,
        layers=[layer],
    )
    return captured[layer][0]


def _prompt_residual_norm(model: object, tokenizer: object, prompt: str, layer: int) -> float:
    """Measure the layer's mean residual norm on this prompt, the scale steering alpha is set in."""
    input_ids = _chat_ids(tokenizer, prompt).unsqueeze(0)
    mask = torch.ones_like(input_ids)
    captured = capture_positionwise_activations(
        model,  # pyright: ignore[reportArgumentType]
        input_ids,
        mask,
        layers=[layer],
    )
    return mean_residual_norm(captured[layer][0])


def _steer_configs(
    residual_norm: float, alpha_scales: Sequence[float]
) -> list[tuple[str, float, float]]:
    """Build the ``(mode, alpha_scale, alpha)`` grid: a baseline, each steer scale, then ablation.

    Ablation carries no alpha (it projects the direction out entirely), and the baseline is what
    makes a steered response readable at all -- without it a shifted observable has nothing to be
    shifted FROM.
    """
    configs: list[tuple[str, float, float]] = [(STEER_MODE_BASELINE, 0.0, 0.0)]
    configs.extend(
        (STEER_MODE_STEER, scale, steering_alpha(residual_norm, scale)) for scale in alpha_scales
    )
    configs.append((STEER_MODE_ABLATE, 0.0, 0.0))
    return configs


def _run_steering_arms(  # noqa: PLR0913 - model, tokenizer, stimuli, the axis and the knobs are all core
    model: object,
    tokenizer: object,
    pairs: Sequence[StimulusPair],
    *,
    concept: str,
    layer: int,
    direction: torch.Tensor,
    args: SteerPatchArgs,
    deadline: Deadline,
    deltanet_kernel: Mapping[str, str],
) -> tuple[list[SteerRecord], list[dict[str, object]]]:
    """Steer AND ablate at ``layer`` against matched-norm placebos, on the rigged-grader prompt.

    Every arm generates under the SAME seeded sampling stream, so a difference between the real
    axis and a placebo is attributable to the intervention rather than to two different samples.
    Ablation is run alongside steering because "the model stops doing X when this direction is
    removed" is the stronger causal claim, and it was previously never run at all.
    """
    arms = steering_directions(direction, n_placebos=args.n_placebos, seed=args.seed)
    stage_sampler = resolved_sampler(args.sampling)
    records: list[SteerRecord] = []
    responses: list[dict[str, object]] = []
    for prompt_index, pair in enumerate(pairs[: args.n_steer_prompts]):
        prompt = pair.conflicting_transcript
        grader_filename = grader_filename_from_transcript(prompt)
        residual_norm = _prompt_residual_norm(model, tokenizer, prompt, layer)
        logger.info(
            "steering %s at layer %d on prompt %d: residual norm %.1f",
            concept,
            layer,
            prompt_index,
            residual_norm,
        )
        for mode, alpha_scale, alpha in _steer_configs(residual_norm, args.alpha_scales):
            arm_vectors = {STEER_ARM_BASELINE: direction} if mode == STEER_MODE_BASELINE else arms
            for arm, vector in arm_vectors.items():
                if deadline.expired(now=time.monotonic()):
                    logger.warning("steering deadline reached; %d arms recorded", len(records))
                    return records, responses
                torch.manual_seed(args.seed)  # one sampling stream shared by every arm
                record = _generate_for_arm(
                    model, tokenizer, prompt, mode, layer, vector, alpha, args.sampling
                )
                require_record_sampler(record, stage_sampler, stage="steer-patch")
                records.append(
                    SteerRecord(
                        concept=concept,
                        layer=layer,
                        prompt_index=prompt_index,
                        arm=arm,
                        mode=mode,
                        alpha=alpha,
                        alpha_scale=alpha_scale,
                        residual_norm=residual_norm,
                        observables=behavioral_observables(record, grader_filename=grader_filename),
                        deltanet_kernel=dict(deltanet_kernel),
                    )
                )
                responses.append(
                    {
                        "concept": concept,
                        "layer": layer,
                        "prompt_index": prompt_index,
                        "problem_id": pair.problem_id,
                        "arm": arm,
                        "mode": mode,
                        "alpha": alpha,
                        "response_text": record.response_text,
                    }
                )
    return records, responses


def _generate_for_arm(  # noqa: PLR0913, PLR0917 - an arm IS model, prompt, layer, vector, magnitude
    model: object,
    tokenizer: object,
    prompt: str,
    mode: str,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    sampling: SamplingConfig,
) -> GenerationRecord:
    """Generate once for one arm: unsteered baseline, or under a steering/ablation hook.

    The baseline and the intervened arms take the SAME sampler, which is why it is one argument
    threaded through both branches rather than a cap each branch defaults for itself: a sampling
    difference between an arm and its baseline would read as an effect of the intervention.
    """
    if mode == STEER_MODE_BASELINE:
        return generate_response(
            model,  # pyright: ignore[reportArgumentType]
            tokenizer,  # pyright: ignore[reportArgumentType]
            prompt,
            sampling=sampling,
        )
    return run_steered_generation(
        model,  # pyright: ignore[reportArgumentType]
        tokenizer,  # pyright: ignore[reportArgumentType]
        prompt,
        layer=layer,
        direction=vector,
        alpha=alpha,
        mode=mode,
        sampling=sampling,
    )


def _run_patch_arms(  # noqa: PLR0913 - the twins, the layer and the arm knobs are all core inputs
    model: object,
    tokenizer: object,
    pairs: Sequence[StimulusPair],
    *,
    concept: str,
    layer: int,
    args: SteerPatchArgs,
    deadline: Deadline,
    deltanet_kernel: Mapping[str, str],
) -> list[dict[str, object]]:
    """Patch the clean (original-grader) run into the corrupted (rigged-grader) run, arm by arm.

    EVERY window of the plan runs as a real arm AND a matched-norm placebo arm, so no recovery
    number is left uninterpretable: the wide ``post_divergence`` window saturates toward 1.0 for a
    random perturbation of equal norm as readily as for the real one, and the ``shared_prefix``
    control's zero must be shown to hold under a placebo too. Running the placebo only at the primary
    window (the earlier behaviour) left the dramatic ``post_divergence`` recoveries with nothing to
    read against. All arms share the answer token the FIRST arm resolved from the clean run's argmax:
    letting each arm pick its own would compare recoveries measured on different tokens.

    The clean-run activations are captured once per pair and reused as the real arm's replacement
    rows, so a window is not re-captured per arm; that is identical to letting
    :func:`~reward_hacking.interp.steering.run_activation_patch` capture them itself.
    """
    rows: list[dict[str, object]] = []
    device = getattr(model, "device", torch.device("cpu"))
    for pair_index, pair in enumerate(pairs[: args.n_patch_pairs]):
        if deadline.expired(now=time.monotonic()):
            logger.warning("patch deadline reached after %d pairs", pair_index)
            break
        clean_ids = _chat_ids(tokenizer, pair.original_transcript)
        corrupted_ids = _chat_ids(tokenizer, pair.conflicting_transcript)
        plan = plan_twin_patch(
            clean_ids, corrupted_ids, max_narrow_positions=args.max_patch_positions
        )
        clean_batch = plan.clean_ids.unsqueeze(0).to(device)
        corrupted_batch = plan.corrupted_ids.unsqueeze(0).to(device)
        clean_mask = torch.ones_like(clean_batch)
        corrupted_mask = torch.ones_like(corrupted_batch)
        clean_layer = _layer_rows(model, clean_batch, clean_mask, layer)
        corrupted_layer = _layer_rows(model, corrupted_batch, corrupted_mask, layer)
        # Every arm patches the corrupted run at this layer, so one prefix serves the whole pair.
        prefix = capture_patch_prefix(
            model,  # pyright: ignore[reportArgumentType]
            corrupted_ids=corrupted_batch,
            corrupted_mask=corrupted_mask,
            layer=layer,
        )
        placebo_generator = torch.Generator().manual_seed(args.seed + pair_index)

        answer_token: int | None = None
        baseline: PatchBaseline | None = None
        for window in plan.windows:
            clean_rows = clean_layer[window.clean_positions]
            corrupted_rows = corrupted_layer[window.corrupted_positions]
            placebo_rows = _matched_norm_or_noop(clean_rows, corrupted_rows, placebo_generator)
            for arm, replacement in (
                (PATCH_ARM_REAL, clean_rows),
                (PATCH_ARM_PLACEBO, placebo_rows),
            ):
                result = run_activation_patch(
                    model,  # pyright: ignore[reportArgumentType]
                    clean_ids=clean_batch,
                    corrupted_ids=corrupted_batch,
                    clean_mask=clean_mask,
                    corrupted_mask=corrupted_mask,
                    layer=layer,
                    clean_positions=window.clean_positions,
                    corrupted_positions=window.corrupted_positions,
                    replacement_rows=replacement,
                    answer_token=answer_token,
                    baseline=baseline,
                    prefix=prefix,
                )
                # The first (real) arm resolves the shared answer token AND the two un-patched
                # readouts; every later arm of this pair reuses both, since neither depends on
                # which window is patched with what.
                answer_token = result.answer_token
                baseline = PatchBaseline(clean=result.clean, corrupted=result.corrupted)
                rows.append(
                    _patch_row(
                        concept,
                        layer,
                        pair_index,
                        plan,
                        window,
                        arm,
                        result,
                        float((replacement - corrupted_rows).norm().item()),
                        dict(deltanet_kernel),
                    )
                )
    return rows


def _patch_row(  # noqa: PLR0913, PLR0917 - a flat record of which arm patched which window, and how
    concept: str,
    layer: int,
    pair_index: int,
    plan: TwinPatchPlan,
    window: PatchWindow,
    arm: str,
    result: PatchResult,
    patch_delta_norm: float,
    deltanet_kernel: Mapping[str, str],
) -> dict[str, object]:
    """One patch cell as a flat, uploadable record (indices and counts only, no item text).

    A recovery of 0.0 is recorded WITH the parts it was computed from, because on its own it cannot
    be told apart from three situations: a patch that did nothing, a clean-vs-corrupted gap so large
    that a real shift rounds to nothing, and a shift too small to survive the bf16 resolution of the
    logits the readout reads (the readout row itself is float32, but its values came off a bf16 LM
    head, so the grid underneath is bf16 either way).
    ``answer_logit_gap`` is the denominator, the numerator is recorded beside it, and
    ``max_abs_logit_shift`` is the largest move ANYWHERE in the readout row -- so a
    zero recovery beside a non-zero shift says "the patch acted but did not move the clean answer
    token", while a zero shift says the intervention itself was inert. Measured on the 0.8B smoke,
    the 8-position narrow window reported exactly 0.0 recovery, which is exactly the reading that
    needed to be falsifiable.

    ``patch_delta_norm`` closes the last ambiguity: it is the Frobenius norm of what the real patch
    replaces (clean rows minus corrupted rows at these positions), so a zero recovery with a small
    delta norm means the two runs barely differ there, while a zero recovery with a LARGE delta norm
    means the region genuinely is not used downstream. Only the second is a claim about the model.
    """
    answer_gap = float(
        (result.clean[result.answer_token] - result.corrupted[result.answer_token]).item()
    )
    answer_shift = float(
        (result.patched[result.answer_token] - result.corrupted[result.answer_token]).item()
    )
    return {
        "concept": concept,
        "layer": layer,
        "pair_index": pair_index,
        "window": window.name,
        "arm": arm,
        DELTANET_KERNEL_FIELD: dict(deltanet_kernel),
        "n_positions": window.n_positions,
        "recovery": result.recovery,
        "answer_token": result.answer_token,
        "answer_logit_gap": answer_gap,
        "patched_minus_corrupted_answer_logit": answer_shift,
        "max_abs_logit_shift": float((result.patched - result.corrupted).abs().max().item()),
        "patch_delta_norm": patch_delta_norm,
        "prefix_len": plan.prefix_len,
        "suffix_len": plan.suffix_len,
        "clean_middle_len": plan.clean_middle_len,
        "corrupted_middle_len": plan.corrupted_middle_len,
        "post_divergence_len": plan.post_divergence_len,
        "clean_len": int(plan.clean_ids.shape[0]),
        "corrupted_len": int(plan.corrupted_ids.shape[0]),
    }


def run_steer_patch(args: SteerPatchArgs) -> None:
    """Steer, ablate, and activation-patch at each concept's peak layer, against placebos (GPU)."""
    require_raw_dir_outside_episode_dir(args.raw_dir, args.episode_dir)
    _require_cuda()
    peak_layers = json.loads(args.peak_layers_path.read_text())
    kernel_bridge = bridge_deltanet_decode_kernel()
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    deltanet_kernel = bound_deltanet_kernels()
    pairs = build_stimulus_pairs(
        args.episode_dir, limit=max(args.n_steer_prompts, args.n_patch_pairs)
    )
    if not pairs:
        raise RuntimeError("no twin pairs available for the causal stage")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    deadline = Deadline(limit_seconds=args.deadline_seconds, started_at=time.monotonic())
    steer_records: list[SteerRecord] = []
    responses: list[dict[str, object]] = []
    patch_rows: list[dict[str, object]] = []
    layers: dict[str, int] = {}
    for concept in args.concepts:
        layer = _peak_layer(peak_layers, concept, args.variant, args.pooling)
        layers[concept] = layer
        direction = load_concept_axis(args.axes_path, concept, args.pooling, layer)
        concept_records, concept_responses = _run_steering_arms(
            model,
            tokenizer,
            pairs,
            concept=concept,
            layer=layer,
            direction=direction,
            args=args,
            deadline=deadline,
            deltanet_kernel=deltanet_kernel,
        )
        steer_records.extend(concept_records)
        responses.extend(concept_responses)
        patch_rows.extend(
            _run_patch_arms(
                model,
                tokenizer,
                pairs,
                concept=concept,
                layer=layer,
                args=args,
                deadline=deadline,
                deltanet_kernel=deltanet_kernel,
            )
        )

    (args.raw_dir / "steer_responses.local.json").write_text(json.dumps(responses, indent=2))
    (args.out_dir / "steer_patch_metrics.json").write_text(
        json.dumps(
            {
                "concepts": list(args.concepts),
                "variant": args.variant,
                "pooling": args.pooling,
                "peak_layers": layers,
                "alpha_scales": list(args.alpha_scales),
                "n_placebos": args.n_placebos,
                "seed": args.seed,
                "sampler": resolved_sampler(args.sampling).as_payload(),
                "deltanet_kernel_bridge": kernel_bridge,
                DELTANET_KERNEL_FIELD: deltanet_kernel,
                "steering": [asdict(record) for record in steer_records],
                "steering_summary": summarize_steering(steer_records),
                "patching": patch_rows,
            },
            indent=2,
        )
    )
    for row in patch_rows:
        logger.info(
            "patch %s layer=%s pair=%s window=%s arm=%s recovery=%.4f",
            row["concept"],
            row["layer"],
            row["pair_index"],
            row["window"],
            row["arm"],
            row["recovery"],
        )
    logger.info(
        "steer-patch done: %d steering arms, %d patch cells over concepts %s",
        len(steer_records),
        len(patch_rows),
        list(args.concepts),
    )


# --------------------------------------------------------------------------------------
# patch-sweep: the properly-powered patching grid (layers x window widths x directions x arms)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchSweepArgs:
    """Inputs for the patching-only sweep stage. No generation, so no sampler here at all."""

    model_id: str
    episode_dir: Path
    out_dir: Path
    raw_dir: Path
    peak_layers_path: Path
    axes_path: Path
    concepts: tuple[str, ...]
    readout_modes: tuple[str, ...]
    option_orders: tuple[str, ...]
    readout_thinking: bool
    variant: str
    pooling: str
    patch_layers: tuple[str, ...]
    layer_chunk: int
    tail_widths: tuple[int, ...]
    head_widths: tuple[int, ...]
    directions: tuple[str, ...]
    modes: tuple[str, ...]
    n_patch_pairs: int
    max_pair_tokens: int
    limit: int | None
    seed: int
    deadline_seconds: float | None
    flush_every_pairs: int
    raw_topk: int
    keep_baseline_rows: bool
    delta_windows: tuple[str, ...]


class EligiblePatchPair(NamedTuple):
    """One twin pair that tokenizes into a plannable patch, with both sides' ids already computed."""

    pair_index: int
    problem_id: str
    original_ids: torch.Tensor
    conflicting_ids: torch.Tensor


def resolve_patch_layers(requested: Sequence[str], *, n_layers: int) -> tuple[int, ...]:
    """Turn the ``--patch-layers`` tokens (integers or ``all``) into a sorted layer list.

    ``all`` needs the loaded model's layer count, which is why this runs after the model load rather
    than in the parser. An out-of-range layer is rejected here instead of surfacing as an opaque
    ``ModuleList`` IndexError twenty minutes into a rented box's run.

    **``peak`` was removed as an anchor** (2026-08-24) and is refused with a message rather than
    silently reinterpreted, because the peak layers the earlier causal tier was aimed at were selected
    by noise. Layer selection swung up to 19 layers on resampled generations, and measured on
    ``direction_accuracy`` the three layers handed to that tier ranked 32nd of 32 (shortcut, L31),
    24th of 32 (eval_awareness, L10) and 20th of 32 (contradiction, L22) for their own concepts. A
    single layer chosen that way is worse than no choice at all: it looks like localisation and is a
    draw from a flat distribution. The peak is still RECORDED beside the sweep as metadata, so the
    comparison stays available, but the grid is the full layer sweep.
    """
    resolved: set[int] = set()
    for token in requested:
        if token == "all":
            resolved.update(range(n_layers))
            continue
        if token == "peak":
            raise ValueError(
                "--patch-layers no longer accepts 'peak': the peak layers the earlier causal tier "
                "used were selected by noise (shortcut's was rank 32 of 32 on its own concept's "
                "direction accuracy). Sweep 'all' layers, or name the layers explicitly."
            )
        if not token.lstrip("-").isdigit():
            raise ValueError(f"--patch-layers takes integers or 'all'; got {token!r}.")
        layer = int(token)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} is outside this {n_layers}-layer model")
        resolved.add(layer)
    if not resolved:
        raise ValueError("--patch-layers resolved to no layers")
    return tuple(sorted(resolved))


class ReadoutVariant(NamedTuple):
    """One resolved readout: its mode, its option order if it has one, and the scored candidates.

    A variant carries its own tokenized twins because the two modes are different PROMPTS -- forced
    choice appends an option block to the transcript and an answer prefill after the chat template,
    while action scoring leaves the transcript alone -- so eligibility, the twin bracket and the
    window ladder are all resolved per variant rather than shared.
    """

    mode: str
    order: str | None
    readout: GapReadout
    suffix: str
    prefill: str

    @property
    def name(self) -> str:
        """Stable identifier for this variant, used as an artifact key and a work-item label."""
        return self.mode if self.order is None else f"{self.mode}/{self.order}"


def resolve_readout_variants(
    tokenizer: object, *, modes: Sequence[str], orders: Sequence[str]
) -> tuple[list[ReadoutVariant], list[dict[str, object]]]:
    """Build every requested readout variant, returning the runnable ones and the skipped ones.

    A forced-choice order whose two option labels share a first token is skipped and COUNTED rather
    than run: the two options cannot be told apart at one readout position, so its gap would be
    identically zero and would enter a mean as evidence of indifference. The action-scoring mode has
    no such failure mode -- its candidates are scored over whole sequences, where a shared first token
    (both authored actions open ``<run>``) is expected.
    """

    def encode(text: str) -> list[int]:
        return cast(
            "list[int]",
            tokenizer(text, add_special_tokens=False)["input_ids"],  # pyright: ignore[reportCallIssue,reportOperatorIssue,reportIndexIssue]
        )

    variants: list[ReadoutVariant] = []
    skipped: list[dict[str, object]] = []
    for mode in modes:
        if mode not in GAP_READOUT_MODES:
            raise ValueError(f"unknown readout mode {mode!r}; expected {list(GAP_READOUT_MODES)}")
        if mode == READOUT_MODE_ACTION_LOGPROB:
            variants.append(
                ReadoutVariant(
                    mode=mode,
                    order=None,
                    readout=build_action_readout(encode),
                    suffix="",
                    prefill="",
                )
            )
            continue
        for order in orders if mode == READOUT_MODE_FORCED_CHOICE else ():
            readout = build_forced_choice_readout(encode, order)
            if readout is None:
                skipped.append(
                    {"variant": f"{mode}/{order}", "reason": "option labels share a first token"}
                )
                continue
            variants.append(
                ReadoutVariant(
                    mode=mode,
                    order=order,
                    readout=readout,
                    suffix=forced_choice_suffix(order),
                    prefill=FORCED_CHOICE_PREFILL,
                )
            )
    if not variants:
        raise RuntimeError(
            f"no runnable readout variant from modes={list(modes)} orders={list(orders)}; "
            f"skipped: {skipped}"
        )
    logger.info(
        "readout variants: %s (skipped %d)",
        [variant.name for variant in variants],
        len(skipped),
    )
    return variants, skipped


class _WorkItem(NamedTuple):
    """One unit of the patch grid: which concept, patch direction, readout variant and twin pair."""

    concept: str
    patch_direction: str
    variant: ReadoutVariant
    pair: EligiblePatchPair


def shuffled_work_order(work: Sequence[_WorkItem], *, seed: int) -> list[_WorkItem]:
    """Permute the sweep's work items, seeded, so a deadline censors a RANDOM subset.

    The deadline stops the sweep between work items, and the items were built in nested loop order
    (concept, then patch direction, then readout variant, then pair). A cutoff therefore dropped the
    TAIL of a fixed list: the last concept, the necessity direction, the highest pair indices --
    always the same cells, so "we ran out of time" and "this arm is missing" were the same fact and a
    partial run's coverage was not a random sample of the grid. Permuting first makes a truncated run
    an unbiased subsample of it. Seeded and the resulting order recorded, so the run is reproducible
    and a reader can see which cells a cutoff would have reached.
    """
    order = torch.randperm(len(work), generator=torch.Generator().manual_seed(seed)).tolist()
    return [work[index] for index in order]


def _cell_seed(seed: int, pair_index: int, layer: int, window_index: int) -> int:
    """Mix a reproducible generator seed for one patch cell, independent of iteration order.

    Seeding once per pair (the steer-patch driver's approach) makes each placebo depend on how many
    arms ran before it, so re-running with a different layer set silently changes every placebo. Two
    primes and the window index keep the cells distinct while a given cell's placebo is the same
    vector in any run of any shape.
    """
    return seed + 1_000_003 * pair_index + 1_009 * layer + window_index


def eligible_patch_pairs(  # noqa: PLR0913 - the corpus plus one knob per eligibility rule
    tokenizer: object,
    pairs: Sequence[StimulusPair],
    *,
    max_pair_tokens: int,
    n_patch_pairs: int,
    variant: ReadoutVariant,
    thinking: bool,
) -> tuple[list[EligiblePatchPair], dict[str, object]]:
    """Tokenize every twin pair FOR ONE READOUT VARIANT, keep the plannable ones, count the rest.

    Runs before the sweep so a pair the planner refuses costs a tokenizer call rather than killing a
    unit mid-flight with every finished cell still unwritten (``plan_twin_patch`` raises on twins that
    share no prefix or where the shorter is a strict prefix of the longer). The token cap is a cost
    bound, not a quality filter: one ILCB pair tokenizes to ~22k while the median is ~2.3k, and its
    forward passes cost ten times a typical pair's for one more row.

    Per variant rather than once, because the two readout modes are different prompts: forced choice
    appends an option block to the transcript and an answer prefill after the chat template. Both
    appended texts are byte-identical across the twins, and that is re-asserted twice -- once on the
    strings by ``readout_transcripts`` and once on the TOKENS here, since two byte-identical tails can
    still tokenize differently when the bytes before them differ, which is exactly the twins'
    situation and would move the readout without changing a byte.

    Returns the kept pairs and the coverage counts, because a sweep that reports a null must report
    what it examined, skipped and refused.
    """
    suffix_tokens = (
        0
        if not variant.suffix
        else len(
            cast(
                "list[int]",
                tokenizer(  # pyright: ignore[reportCallIssue,reportOperatorIssue,reportIndexIssue]
                    variant.suffix + variant.prefill, add_special_tokens=False
                )["input_ids"],
            )
        )
    )
    kept: list[EligiblePatchPair] = []
    over_cap = 0
    unplannable: list[str] = []
    for index, pair in enumerate(pairs):
        original, conflicting = readout_transcripts(
            pair.original_transcript,
            pair.conflicting_transcript,
            mode=variant.mode,
            order=variant.order,
        )
        original_ids = _chat_ids(tokenizer, original, thinking=thinking, prefill=variant.prefill)
        conflicting_ids = _chat_ids(
            tokenizer, conflicting, thinking=thinking, prefill=variant.prefill
        )
        longest = max(int(original_ids.shape[0]), int(conflicting_ids.shape[0]))
        if longest > max_pair_tokens:
            over_cap += 1
            continue
        try:
            plan = plan_twin_patch_ladder(original_ids, conflicting_ids)
        except ValueError as exc:
            unplannable.append(f"pair {index}: {exc}")
            continue
        if suffix_tokens:
            require_shared_suffix_covers(
                plan,
                min_tokens=suffix_tokens,
                what=f"the {variant.name} option block plus answer prefill",
            )
        kept.append(EligiblePatchPair(index, pair.problem_id, original_ids, conflicting_ids))
    for message in unplannable:
        logger.warning("unplannable twin pair skipped -- %s", message)
    coverage: dict[str, object] = {
        "readout_variant": variant.name,
        "readout_thinking": thinking,
        "appended_suffix_tokens": suffix_tokens,
        "pairs_available": len(pairs),
        "pairs_over_token_cap": over_cap,
        "pairs_unplannable": len(unplannable),
        "pairs_eligible": len(kept),
        "pairs_requested": n_patch_pairs,
        "pairs_used": min(len(kept), n_patch_pairs),
        "max_pair_tokens": max_pair_tokens,
    }
    logger.info("eligible patch pairs: %s", coverage)
    return kept[:n_patch_pairs], coverage


@dataclass
class PatchSweepRaw:
    """The raw material the sweep retains, so a later question is a re-analysis not another box.

    Both earlier causal runs wrote their raw material to a local ``raw_dir`` that was never uploaded
    and then let the box self-terminate, which destroyed it: the repo's probing posture says to keep
    the raw material precisely because otherwise "add rigor later" silently means "re-run the GPU".
    This holds everything a different readout metric would need, at a size that uploads in seconds:

    * ``baseline_rows`` -- the FULL un-patched clean and corrupted readout rows, float16, once per
      (variant, direction, pair). Any recovery metric anyone invents later can be recomputed from these.
    * ``baseline_gaps`` -- the two un-patched SCORED GAPS at the same key, with the per-candidate
      sequence log-probs behind them. This is what makes a different candidate pair, a different
      normalisation, or a re-derived recovery a CPU re-analysis rather than another rented box.
    * ``comparison_indices`` -- the union of the clean and corrupted top-``topk`` token ids at that
      readout, so every cell's patched row is sampled at ONE shared index set and the cells stay
      comparable. Chosen once per key rather than per cell for that reason.
    * ``cell_rows`` -- each cell's patched readout row at those indices, float16.
    * ``pair_deltas`` -- the mean transplanted delta (``clean - corrupted`` over the window's
      positions) per (variant, direction, pair, layer, window), for the windows in ``delta_windows``.
      This is the vector the Jacobian-lens decode reads: what patching injects, in residual space.
    * ``delta_sums`` / ``delta_counts`` -- the same delta accumulated across pairs for EVERY window,
      which is the cross-pair mean the decode leads with.
    * ``deltanet_kernel`` -- the FULL four-kernel binding these deltas were produced under. The sweep
      generates its readouts, so it dispatches the per-token pair as well as the chunked one, and the
      fused and fallback decode kernels reduce in different orders. Without it ``patch-decode`` cannot
      tell which binding produced the raw deltas it decodes, and a pool mixing two reads as one.

    The readout VARIANT is part of every key because the two readout modes are different prompts:
    forced choice appends an option block, so its twins tokenize differently and its deltas are
    different vectors. Pooling them under one key would average two substrates.

    Deliberately NOT a sample of anything: every cell the sweep ran appears here.
    """

    topk: int
    delta_windows: tuple[str, ...]
    keep_baseline_rows: bool
    deltanet_kernel: Mapping[str, str]
    comparison_indices: dict[str, torch.Tensor] = field(default_factory=dict)
    baseline_rows: dict[str, dict[str, torch.Tensor]] = field(default_factory=dict)
    baseline_gaps: dict[str, dict[str, object]] = field(default_factory=dict)
    answer_tokens: dict[str, int] = field(default_factory=dict)
    cell_rows: list[dict[str, object]] = field(default_factory=list)
    pair_deltas: list[dict[str, object]] = field(default_factory=list)
    delta_sums: dict[str, torch.Tensor] = field(default_factory=dict)
    delta_counts: dict[str, int] = field(default_factory=dict)
    n_baselines_recomputed: int = 0
    _delta_keys_seen: set[tuple[str, str, int, int, str]] = field(default_factory=set)

    def record_baseline(
        self,
        key: str,
        clean: torch.Tensor,
        corrupted: torch.Tensor,
        answer_token: int,
        *,
        gaps: tuple[GapRead | None, GapRead | None] = (None, None),
    ) -> None:
        """Store one (variant, direction, pair)'s un-patched readouts and fix its comparison indices.

        The same key is re-run once per concept, and neither the readouts nor the gaps depend on the
        concept, so later visits are counted rather than stored twice.
        """
        if key in self.comparison_indices:
            self.n_baselines_recomputed += 1
            return
        k = min(self.topk, int(clean.numel()))
        indices = torch.cat([clean.topk(k).indices, corrupted.topk(k).indices]).unique()
        self.comparison_indices[key] = indices
        self.answer_tokens[key] = answer_token
        clean_gap, corrupted_gap = gaps
        if clean_gap is not None and corrupted_gap is not None:
            self.baseline_gaps[key] = {
                "clean": clean_gap.payload(),
                "corrupted": corrupted_gap.payload(),
            }
        if self.keep_baseline_rows:
            self.baseline_rows[key] = {
                "clean": clean.half(),
                "corrupted": corrupted.half(),
            }

    def record_cell(self, cell: PatchSweepCell, window: str, patched: torch.Tensor) -> None:
        """Store one cell's patched readout row at the pair's shared comparison indices."""
        pair_key = f"{cell.readout_variant}|{cell.patch_direction}|{cell.pair_index}"
        indices = self.comparison_indices[pair_key]
        self.cell_rows.append(
            {
                "concept": cell.concept,
                "readout_variant": cell.readout_variant,
                "patch_direction": cell.patch_direction,
                "pair_index": cell.pair_index,
                "layer": cell.layer,
                "window": window,
                "arm": cell.arm,
                "patched_at_comparison_indices": patched[indices].half(),
            }
        )

    def record_delta(  # noqa: PLR0913 - the delta's identity is its five coordinates
        self,
        *,
        readout_variant: str,
        patch_direction: str,
        pair_index: int,
        layer: int,
        window: str,
        delta_rows: torch.Tensor,
    ) -> None:
        """Accumulate the mean transplanted delta for one (variant, direction, pair, layer, window).

        Deduplicated on that five-part key: the delta does not depend on the concept, so a unit
        sweeping two concepts would otherwise record each pair twice and the decode would report a
        doubled pair count for the same vector.
        """
        cell = (readout_variant, patch_direction, pair_index, layer, window)
        if cell in self._delta_keys_seen:
            return
        self._delta_keys_seen.add(cell)
        mean_delta = delta_rows.mean(dim=0).float()
        key = f"{readout_variant}|{patch_direction}|{layer}|{window}"
        running = self.delta_sums.get(key)
        self.delta_sums[key] = mean_delta if running is None else running + mean_delta
        self.delta_counts[key] = self.delta_counts.get(key, 0) + 1
        if window in self.delta_windows:
            self.pair_deltas.append(
                {
                    "readout_variant": readout_variant,
                    "patch_direction": patch_direction,
                    "pair_index": pair_index,
                    "layer": layer,
                    "window": window,
                    "mean_delta": mean_delta.half(),
                }
            )

    def payload(self) -> dict[str, object]:
        """Everything retained, plus the cross-pair mean deltas the decode stage leads with."""
        return {
            "readout_semantics": READOUT_SEMANTICS,
            "topk": self.topk,
            "delta_windows": list(self.delta_windows),
            DELTANET_KERNEL_FIELD: dict(self.deltanet_kernel),
            "comparison_indices": self.comparison_indices,
            "baseline_rows": self.baseline_rows,
            "baseline_gaps": self.baseline_gaps,
            "answer_tokens": self.answer_tokens,
            "cell_rows": self.cell_rows,
            "pair_deltas": self.pair_deltas,
            "mean_delta_across_pairs": {
                key: total / self.delta_counts[key] for key, total in self.delta_sums.items()
            },
            "mean_delta_pair_counts": dict(self.delta_counts),
            "n_baselines_recomputed": self.n_baselines_recomputed,
        }


def _patch_sweep_arms(
    clean_rows: torch.Tensor,
    corrupted_rows: torch.Tensor,
    *,
    modes: Sequence[str],
    direction: torch.Tensor | None,
    generator: torch.Generator,
) -> list[tuple[str, torch.Tensor]]:
    """Every replacement this cell runs, named: full-residual arms, then the axis-restricted ones.

    ``full_residual`` reproduces the earlier run's two arms (the clean rows themselves, and a random
    perturbation of equal Frobenius norm). ``axis`` adds the decomposition that connects patching to
    our concept directions: the axis component alone, its orthogonal complement, a rank-1 placebo
    matched to the component's norm, and a rank-1 projection onto a random axis with no rescaling.
    The last two answer different questions -- "is it THIS direction rather than a perturbation this
    size" and "how much would an arbitrary direction have captured" -- and both are one forward.
    """
    arms: list[tuple[str, torch.Tensor]] = []
    if PATCH_MODE_FULL_RESIDUAL in modes:
        arms.append((PATCH_ARM_REAL_FULL, clean_rows))
        arms.append(
            (PATCH_ARM_PLACEBO, _matched_norm_or_noop(clean_rows, corrupted_rows, generator))
        )
    if PATCH_MODE_AXIS in modes and direction is not None:
        component = axis_component_replacement(clean_rows, corrupted_rows, direction)
        component_norm = float((component - corrupted_rows).norm().item())
        arms.append((PATCH_ARM_REAL_AXIS_COMPONENT, component))
        arms.append(
            (
                PATCH_ARM_REAL_AXIS_COMPLEMENT,
                axis_complement_replacement(clean_rows, corrupted_rows, direction),
            )
        )
        arms.append(
            (
                PATCH_ARM_PLACEBO_AXIS_MATCHED_NORM,
                random_axis_replacement(
                    clean_rows, corrupted_rows, generator, match_norm_to=component_norm
                ),
            )
        )
        arms.append(
            (
                PATCH_ARM_PLACEBO_RANDOM_AXIS,
                random_axis_replacement(clean_rows, corrupted_rows, generator),
            )
        )
    return arms


class PatchSweepCell(NamedTuple):
    """The identity of one patch cell: concept, layer, pair, direction, readout variant and arm."""

    concept: str
    layer: int
    pair_index: int
    problem_id: str
    patch_direction: str
    source_side: str
    target_side: str
    arm: str
    readout_variant: str
    # The model's decoder-layer count, carried on the cell so every row can say where in the stack it
    # sits (see FINAL_FIFTH) rather than leaving a reader to look the model up.
    n_layers: int
    # What the Gated DeltaNet kernel functions were bound to; every row repeats it, so a pool of rows
    # from two runs can be refused rather than averaged (assert_one_deltanet_kernel).
    deltanet_kernel: dict[str, str]


def _patch_sweep_row(  # noqa: PLR0913, PLR0917 - a flat cell record: what was patched, where, how
    cell: PatchSweepCell,
    plan: TwinPatchPlan,
    window: PatchWindow,
    result: PatchResult,
    patch_delta_norm: float,
    axis_norm: float | None,
) -> dict[str, object]:
    """One patch cell, flat and uploadable: counts, indices and readout statistics only.

    ``recovery_gap`` is the PRIMARY reading: the fraction of the clean-versus-corrupted difference in
    the scored candidate gap that the patch recovered. The single-logit ``recovery`` beside it is kept
    as a secondary diagnostic and as the only field comparable with the earlier run, but the earlier
    run's version of it scored the opening token of the reasoning trace, so the two are comparable in
    arithmetic rather than in meaning.

    ``contains_readout_position`` is the field the first causal run lacked: a window that includes the
    last prompt position overwrites the residual the first scored token is read off. Under the
    multi-token readout that is no longer near-tautological -- one position cannot manufacture a
    sequence log-prob over twenty-odd tokens -- but it is still the position with the most direct
    route to the reading, so every cell says whether it was patched.

    Every recovery reading is recorded beside the raw quantities behind it (the three gaps, the shift,
    the answer-token gap, the largest shift anywhere in the row, the KL and log-prob distances), so a
    zero can be told apart from a patch that did nothing and a one from a rounding coincidence. A
    ``None`` recovery means the denominator was exactly zero, which is why the denominator is here too.
    """
    answer_gap = float(
        (result.clean[result.answer_token] - result.corrupted[result.answer_token]).item()
    )
    answer_shift = float(
        (result.patched[result.answer_token] - result.corrupted[result.answer_token]).item()
    )
    row: dict[str, object] = {
        "concept": cell.concept,
        "layer": cell.layer,
        "pair_index": cell.pair_index,
        "problem_id": cell.problem_id,
        "patch_direction": cell.patch_direction,
        "source_side": cell.source_side,
        "target_side": cell.target_side,
        "readout_variant": cell.readout_variant,
        "window": window.name,
        "arm": cell.arm,
        DELTANET_KERNEL_FIELD: dict(cell.deltanet_kernel),
        "n_positions": window.n_positions,
        "contains_readout_position": window_contains_readout(
            window, corrupted_len=int(plan.corrupted_ids.shape[0])
        ),
        "n_layers": cell.n_layers,
        "layer_fraction": cell.layer / cell.n_layers,
        "in_final_fifth": cell.layer >= FINAL_FIFTH * cell.n_layers,
        "answer_token": result.answer_token,
        "answer_logit_gap": answer_gap,
        "patched_minus_corrupted_answer_logit": answer_shift,
        "max_abs_logit_shift": float((result.patched - result.corrupted).abs().max().item()),
        "patch_delta_norm": patch_delta_norm,
        "axis_norm": axis_norm,
        "prefix_len": plan.prefix_len,
        "suffix_len": plan.suffix_len,
        "clean_middle_len": plan.clean_middle_len,
        "corrupted_middle_len": plan.corrupted_middle_len,
        "post_divergence_len": plan.post_divergence_len,
        "source_len": int(plan.clean_ids.shape[0]),
        "target_len": int(plan.corrupted_ids.shape[0]),
    }
    row.update(_gap_row(result))
    row.update(
        recovery_metrics(result.clean, result.corrupted, result.patched, result.answer_token)
    )
    # The single-logit ratio the earlier run reported, under its old name, so the two line up.
    row["recovery"] = row["recovery_logit"]
    return row


def _gap_row(result: PatchResult) -> dict[str, object]:
    """Assemble the scored-gap half of a cell record: three gaps, the shift, and the recovery.

    ``gap_shift`` is what an identity control must read as exactly 0.0 -- the same run patched with its
    own activations must not move its own gap -- so it is recorded on every cell rather than derived,
    which is what lets that control be checked from the artifact alone.
    """
    if result.clean_gap is None or result.corrupted_gap is None or result.patched_gap is None:
        return {"readout_mode": None, "recovery_gap": None}
    return {
        "readout_mode": result.clean_gap.mode,
        "recovery_gap": result.recovery_gap,
        "gap_clean": result.clean_gap.gap,
        "gap_corrupted": result.corrupted_gap.gap,
        "gap_patched": result.patched_gap.gap,
        "gap_denominator": result.clean_gap.gap - result.corrupted_gap.gap,
        "gap_shift": result.patched_gap.gap - result.corrupted_gap.gap,
        "gap_per_token_clean": result.clean_gap.gap_per_token,
        "gap_per_token_corrupted": result.corrupted_gap.gap_per_token,
        "gap_per_token_patched": result.patched_gap.gap_per_token,
        "positive_logprob_patched": result.patched_gap.positive_logprob,
        "negative_logprob_patched": result.patched_gap.negative_logprob,
        "positive_n_tokens": result.patched_gap.positive_n_tokens,
        "negative_n_tokens": result.patched_gap.negative_n_tokens,
    }


def _sweep_one_pair(  # noqa: PLR0913 - model plus the pair, concept, layers, axes and knobs
    model: object,
    pair: EligiblePatchPair,
    *,
    concept: str,
    patch_direction: str,
    variant: ReadoutVariant,
    layers: Sequence[int],
    n_layers: int,
    axes_by_layer: Mapping[int, torch.Tensor],
    args: PatchSweepArgs,
    deadline: Deadline,
    raw: PatchSweepRaw,
    deltanet_kernel: Mapping[str, str],
) -> tuple[list[dict[str, object]], bool]:
    """Patch one pair at every requested layer, window and arm. Returns its rows and a deadline flag.

    Two economies make the grid affordable, and both are exact rather than approximations. The two
    un-patched readouts, their two scored candidate gaps and the answer token do not depend on the
    layer or the window, so they are resolved once for the pair (by the first cell) and reused -- at 32
    layers times 15 windows times 2 arms that is 960 redundant forward pairs removed, and under the
    multi-token readout each of those pairs is two forwards rather than one. And the per-position
    activations for a whole chunk of layers come out of ONE forward per run, so a 32-layer sweep costs
    4 capture passes at the default chunk of 8 rather than 32.

    ``layer_chunk`` exists because the capture is the memory cost, not the compute: all 32 layers of
    a 4k-token run in float32 is ~1.3 GiB per side, and the pair set reaches 6k tokens.
    """
    source_ids, target_ids, source_side, target_side = (
        (pair.original_ids, pair.conflicting_ids, SIDE_ORIGINAL, SIDE_CONFLICTING)
        if patch_direction == PATCH_DIRECTION_ORIGINAL_INTO_RIGGED
        else (pair.conflicting_ids, pair.original_ids, SIDE_CONFLICTING, SIDE_ORIGINAL)
    )
    plan = plan_twin_patch_ladder(
        source_ids,
        target_ids,
        tail_widths=args.tail_widths,
        head_widths=args.head_widths,
    )
    device = getattr(model, "device", torch.device("cpu"))
    source_batch = source_ids.unsqueeze(0).to(device)
    target_batch = target_ids.unsqueeze(0).to(device)
    source_mask = torch.ones_like(source_batch)
    target_mask = torch.ones_like(target_batch)

    rows: list[dict[str, object]] = []
    baseline: PatchBaseline | None = None
    answer_token: int | None = None
    for chunk in batched(layers, args.layer_chunk, strict=False):
        if deadline.expired(now=time.monotonic()):
            logger.warning("patch-sweep deadline reached inside pair %d", pair.pair_index)
            return rows, True
        source_captured = capture_positionwise_activations(
            model,  # pyright: ignore[reportArgumentType]
            source_batch,
            source_mask,
            layers=list(chunk),
        )
        target_captured = capture_positionwise_activations(
            model,  # pyright: ignore[reportArgumentType]
            target_batch,
            target_mask,
            layers=list(chunk),
        )
        for layer in chunk:
            direction = axes_by_layer.get(layer)
            axis_norm = None if direction is None else float(direction.norm().item())
            source_rows = source_captured[layer][0]
            target_rows = target_captured[layer][0]
            # One prefix per (pair, layer): every window and arm patches the same target run here.
            prefix = capture_patch_prefix(
                model,  # pyright: ignore[reportArgumentType]
                corrupted_ids=target_batch,
                corrupted_mask=target_mask,
                layer=layer,
                readout=variant.readout,
            )
            for window_index, window in enumerate(plan.windows):
                clean_rows = source_rows[window.clean_positions]
                corrupted_rows = target_rows[window.corrupted_positions]
                raw.record_delta(
                    readout_variant=variant.name,
                    patch_direction=patch_direction,
                    pair_index=pair.pair_index,
                    layer=layer,
                    window=window.name,
                    delta_rows=clean_rows - corrupted_rows,
                )
                generator = torch.Generator().manual_seed(
                    _cell_seed(args.seed, pair.pair_index, layer, window_index)
                )
                for arm, replacement in _patch_sweep_arms(
                    clean_rows,
                    corrupted_rows,
                    modes=args.modes,
                    direction=direction,
                    generator=generator,
                ):
                    result = run_activation_patch(
                        model,  # pyright: ignore[reportArgumentType]
                        clean_ids=source_batch,
                        corrupted_ids=target_batch,
                        clean_mask=source_mask,
                        corrupted_mask=target_mask,
                        layer=layer,
                        clean_positions=window.clean_positions,
                        corrupted_positions=window.corrupted_positions,
                        replacement_rows=replacement,
                        answer_token=answer_token,
                        baseline=baseline,
                        readout=variant.readout,
                        prefix=prefix,
                    )
                    answer_token = result.answer_token
                    baseline = PatchBaseline(
                        clean=result.clean,
                        corrupted=result.corrupted,
                        clean_gap=result.clean_gap,
                        corrupted_gap=result.corrupted_gap,
                    )
                    cell = PatchSweepCell(
                        concept=concept,
                        layer=layer,
                        pair_index=pair.pair_index,
                        problem_id=pair.problem_id,
                        patch_direction=patch_direction,
                        source_side=source_side,
                        target_side=target_side,
                        arm=arm,
                        readout_variant=variant.name,
                        n_layers=n_layers,
                        deltanet_kernel=dict(deltanet_kernel),
                    )
                    raw.record_baseline(
                        f"{variant.name}|{patch_direction}|{pair.pair_index}",
                        result.clean,
                        result.corrupted,
                        answer_token,
                        gaps=(result.clean_gap, result.corrupted_gap),
                    )
                    raw.record_cell(cell, window.name, result.patched)
                    rows.append(
                        _patch_sweep_row(
                            cell,
                            plan,
                            window,
                            result,
                            float((replacement - corrupted_rows).norm().item()),
                            axis_norm,
                        )
                    )
        del source_captured, target_captured
    return rows, False


def _patch_sweep_payload(  # noqa: PLR0913 - the args plus each handoff the artifact must carry
    args: PatchSweepArgs,
    *,
    layers_by_concept: Mapping[str, object],
    coverage: Mapping[str, object],
    rows: Sequence[dict[str, object]],
    variants: Sequence[str],
    skipped_variants: Sequence[dict[str, object]],
    kernel_bridge: Mapping[str, object],
    deltanet_kernel: Mapping[str, str],
) -> dict[str, object]:
    """Assemble the uploadable artifact. Called mid-sweep too, so a hard kill loses one pair."""
    return {
        "stage": "patch-sweep",
        "readout_semantics": READOUT_SEMANTICS,
        "deltanet_kernel_bridge": dict(kernel_bridge),
        DELTANET_KERNEL_FIELD: dict(deltanet_kernel),
        "readout": readout_provenance(args.readout_modes),
        "readout_modes": list(args.readout_modes),
        "readout_variants": list(variants),
        "readout_variants_skipped": list(skipped_variants),
        "readout_thinking": args.readout_thinking,
        "option_orders": list(args.option_orders),
        "model_id": args.model_id,
        "concepts": list(args.concepts),
        "variant": args.variant,
        "pooling": args.pooling,
        "layers_by_concept": dict(layers_by_concept),
        "tail_widths": list(args.tail_widths),
        "head_widths": list(args.head_widths),
        "patch_directions": list(args.directions),
        "patch_modes": list(args.modes),
        "layer_chunk": args.layer_chunk,
        "seed": args.seed,
        "coverage": dict(coverage),
        "n_cells": len(rows),
        "patching": list(rows),
    }


def _sweep_layers_and_axes(
    args: PatchSweepArgs, peak_layers: object, *, n_layers: int
) -> tuple[dict[str, object], dict[str, dict[int, torch.Tensor]]]:
    """Resolve each concept's swept layers and, for the axis mode, its axis AT EVERY swept layer.

    A saved axis that does not cover a requested layer fails here rather than deep in the loop: it
    costs a second at startup instead of surfacing after the first pairs have already been patched.

    The contrast peak is still READ and recorded per concept, so the comparison with the earlier run
    stays available in the artifact, but it no longer selects anything: see
    :func:`resolve_patch_layers` for why a peak chosen that way is worse than no choice. Which is
    exactly why a withheld peak is recorded as ``null`` instead of raising -- this stage does not steer
    on the number, and :func:`_peak_layer`'s hard raise here aborted a whole sweep at startup over a
    value it only files. The floor withholding a cell is a real case, not a hypothetical one: two cells
    of the retained 2026-08-22 contrast were negative on ``auc_above_placebo`` at all 32 layers.
    """
    layers_by_concept: dict[str, object] = {}
    axes: dict[str, dict[int, torch.Tensor]] = {}
    for concept in args.concepts:
        peak = _recorded_peak_layer(peak_layers, concept, args.variant, args.pooling)
        layers = resolve_patch_layers(args.patch_layers, n_layers=n_layers)
        layers_by_concept[concept] = {
            "contrast_peak_layer_recorded_not_used": peak,
            "layers": list(layers),
        }
        axes[concept] = (
            {
                layer: load_concept_axis(args.axes_path, concept, args.pooling, layer)
                for layer in layers
            }
            if PATCH_MODE_AXIS in args.modes
            else {}
        )
    logger.info("patch-sweep layers: %s", layers_by_concept)
    return layers_by_concept, axes


def _eligible_pairs_per_variant(
    tokenizer: object,
    pairs: Sequence[StimulusPair],
    *,
    variants: Sequence[ReadoutVariant],
    args: PatchSweepArgs,
) -> tuple[dict[str, list[EligiblePatchPair]], dict[str, dict[str, object]]]:
    """Resolve eligibility once per readout variant, refusing a variant with nothing left to run.

    Fails before the first forward rather than partway through a rented box: a variant whose appended
    option block pushes every pair past the token cap, or whose twins stop bracketing, is a
    misconfiguration and not a null.
    """
    pairs_by_variant: dict[str, list[EligiblePatchPair]] = {}
    coverage_by_variant: dict[str, dict[str, object]] = {}
    for variant in variants:
        eligible, variant_coverage = eligible_patch_pairs(
            tokenizer,
            pairs,
            max_pair_tokens=args.max_pair_tokens,
            n_patch_pairs=args.n_patch_pairs,
            variant=variant,
            thinking=args.readout_thinking,
        )
        if not eligible:
            raise RuntimeError(
                f"no eligible twin pairs for readout variant {variant.name}: {variant_coverage}. "
                "Raise --max-pair-tokens or check the stimulus set."
            )
        pairs_by_variant[variant.name] = eligible
        coverage_by_variant[variant.name] = variant_coverage
    return pairs_by_variant, coverage_by_variant


def run_patch_sweep(args: PatchSweepArgs) -> None:
    """Activation-patch a grid of layers, window widths, directions and arms (GPU, forwards only).

    The stage exists because the earlier ``steer-patch`` patching read could not answer three
    questions its own numbers raised: whether the wide window's ~1.0 recovery survives narrowing,
    whether the effect is specific to a concept's decoding-peak layer, and whether it is carried by
    one of our concept axes at all. It patches nothing that stage could not patch in principle -- it
    is the same :func:`run_activation_patch` -- but it sweeps, it excludes the readout position from
    the window families that should not contain it, and it records enough per cell that the answers
    are a re-analysis rather than another rented box.

    Since 2026-08-24 it also asks a different question of each cell: the reading is a gap between two
    complete candidate actions rather than one next-token logit, and the readout variants are part of
    the grid because each one is a different prompt. See
    :mod:`reward_hacking.interp.patch_readout`.
    """
    require_raw_dir_outside_episode_dir(args.raw_dir, args.episode_dir)
    _require_cuda()
    peak_layers = json.loads(args.peak_layers_path.read_text())
    kernel_bridge = bridge_deltanet_decode_kernel()
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    deltanet_kernel = bound_deltanet_kernels()
    n_layers = len(_decoder_layers(model))
    logger.info("patch-sweep on a %d-layer %s", n_layers, args.model_id)
    pairs = build_stimulus_pairs(args.episode_dir, limit=args.limit)
    if not pairs:
        raise RuntimeError("no twin pairs available for the patch sweep")
    variants, skipped_variants = resolve_readout_variants(
        tokenizer, modes=args.readout_modes, orders=args.option_orders
    )
    pairs_by_variant, coverage_by_variant = _eligible_pairs_per_variant(
        tokenizer, pairs, variants=variants, args=args
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    layers_by_concept, axes = _sweep_layers_and_axes(args, peak_layers, n_layers=n_layers)

    work = [
        _WorkItem(concept, patch_direction, variant, pair)
        for concept in args.concepts
        for patch_direction in args.directions
        for variant in variants
        for pair in pairs_by_variant[variant.name]
    ]
    work = shuffled_work_order(work, seed=args.seed)
    coverage: dict[str, object] = {
        "by_readout_variant": coverage_by_variant,
        "readout_variants_skipped": skipped_variants,
        "work_items_total": len(work),
        "work_order_shuffled_with_seed": args.seed,
        "work_order": [
            f"{concept}|{direction}|{variant.name}|pair{pair.pair_index}"
            for concept, direction, variant, pair in work
        ],
    }
    deadline = Deadline(limit_seconds=args.deadline_seconds, started_at=time.monotonic())
    rows: list[dict[str, object]] = []
    raw = PatchSweepRaw(
        topk=args.raw_topk,
        delta_windows=args.delta_windows,
        keep_baseline_rows=args.keep_baseline_rows,
        deltanet_kernel=deltanet_kernel,
    )
    metrics_path = args.out_dir / "patch_sweep_metrics.json"
    raw_path = args.raw_dir / PATCH_SWEEP_RAW_FILENAME
    variant_names = [variant.name for variant in variants]
    completed = 0
    stopped = False
    for concept, patch_direction, variant, pair in work:
        if deadline.expired(now=time.monotonic()):
            logger.warning("patch-sweep deadline reached after %d work items", completed)
            stopped = True
            break
        started = time.monotonic()
        pair_layers = layers_by_concept[concept]["layers"]  # pyright: ignore[reportIndexIssue]
        pair_rows, hit_deadline = _sweep_one_pair(
            model,
            pair,
            concept=concept,
            patch_direction=patch_direction,
            variant=variant,
            layers=pair_layers,  # pyright: ignore[reportArgumentType]
            n_layers=n_layers,
            axes_by_layer=axes[concept],
            args=args,
            deadline=deadline,
            raw=raw,
            deltanet_kernel=deltanet_kernel,
        )
        rows.extend(pair_rows)
        completed += 1
        logger.info(
            "patch-sweep %s/%s/%s pair %d: %d cells in %.1fs (%d/%d work items)",
            concept,
            patch_direction,
            variant.name,
            pair.pair_index,
            len(pair_rows),
            time.monotonic() - started,
            completed,
            len(work),
        )
        if hit_deadline:
            stopped = True
            break
        if completed % args.flush_every_pairs == 0:
            metrics_path.write_text(
                json.dumps(
                    _patch_sweep_payload(
                        args,
                        layers_by_concept=layers_by_concept,
                        coverage={
                            **coverage,
                            "work_items_completed": completed,
                            "in_progress": True,
                        },
                        rows=rows,
                        variants=variant_names,
                        skipped_variants=skipped_variants,
                        kernel_bridge=kernel_bridge,
                        deltanet_kernel=deltanet_kernel,
                    ),
                    indent=2,
                )
            )
            torch.save(raw.payload(), raw_path)

    torch.save(raw.payload(), raw_path)
    logger.info(
        "patch-sweep raw material: %d cell readouts, %d per-pair deltas, %d baseline pairs -> %s "
        "(%.1f MiB). This is what the earlier runs discarded; upload it.",
        len(raw.cell_rows),
        len(raw.pair_deltas),
        len(raw.comparison_indices),
        raw_path,
        raw_path.stat().st_size / (1024**2),
    )
    metrics_path.write_text(
        json.dumps(
            _patch_sweep_payload(
                args,
                layers_by_concept=layers_by_concept,
                coverage={
                    **coverage,
                    "work_items_completed": completed,
                    "stopped_on_deadline": stopped,
                    "in_progress": False,
                },
                rows=rows,
                variants=variant_names,
                skipped_variants=skipped_variants,
                kernel_bridge=kernel_bridge,
                deltanet_kernel=deltanet_kernel,
            ),
            indent=2,
        )
    )
    logger.info(
        "patch-sweep done: %d cells over %d/%d work items (stopped_on_deadline=%s) -> %s",
        len(rows),
        completed,
        len(work),
        stopped,
        metrics_path,
    )


# --------------------------------------------------------------------------------------
# patch-decode: read the transplanted signal in TOKEN space through a Jacobian lens
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PatchDecodeArgs:
    """Inputs for the lens decode of a finished sweep's transplanted deltas."""

    model_id: str
    episode_dir: Path
    out_dir: Path
    raw_dir: Path
    sweep_raw_path: Path
    axes_path: Path
    concepts: tuple[str, ...]
    pooling: str
    decode_windows: tuple[str, ...]
    decode_layers: tuple[str, ...]
    n_fit_pairs: int
    # None fits every corpus string; a number caps the fit the way fit-lens's --max-fit-prompts does,
    # which is what lets a reuse match a lens that stage capped.
    max_fit_prompts: int | None
    max_seq_len: int | None
    max_seq_len_ceiling: int
    dim_batch: int
    top_k: int
    n_layers: int
    seed: int
    deadline_seconds: float | None
    lens_path: Path | None
    lens_fit_corpus_path: Path | None


def lens_fit_corpus(pairs: Sequence[StimulusPair]) -> list[str]:
    """Flatten the twin transcripts to one prompt per line, the lens fit's input format.

    The transcripts rather than the short concept sentences, for the reason the earlier decode run
    recorded: the fit needs prompts longer than ~17 tokens and silently skips the sentences, and the
    transcripts ARE this project's stimulus substrate. Each is whitespace-collapsed so it survives
    the one-prompt-per-line reader.
    """
    lines = [
        " ".join(transcript.split())
        for pair in pairs
        for transcript in (pair.conflicting_transcript, pair.original_transcript)
    ]
    return [line for line in lines if line]


def decode_lens_corpus(args: PatchDecodeArgs) -> tuple[list[str], str]:
    """Return the corpus this decode's lens answers to, and a line saying where it came from.

    Two sources, and which one is used decides whether a reuse can happen at all. By default the
    corpus is built here from the twin transcripts (:func:`lens_fit_corpus`), which is NOT the corpus
    ``fit-lens`` fitted on -- that one also carries the model's generated reasoning and keeps the
    transcripts' own whitespace -- so a ``--lens-path`` alone will always refit, loudly, naming the
    digest that differed. Passing ``--lens-fit-corpus`` at the ``fit_corpus.local.json`` that stage
    wrote makes the two corpora the same strings in the same order, which is what the gate is for.
    """
    if args.lens_fit_corpus_path is not None:
        loaded = json.loads(args.lens_fit_corpus_path.read_text())
        if not isinstance(loaded, list) or not all(isinstance(item, str) for item in loaded):
            raise RuntimeError(
                f"{args.lens_fit_corpus_path} is not a fit corpus: expected a JSON list of strings, "
                f"the shape run_fit_lens writes to fit_corpus.local.json"
            )
        corpus = cast("list[str]", loaded)
        if not corpus:
            raise RuntimeError(f"{args.lens_fit_corpus_path} holds an empty corpus")
        logger.info(
            "decode lens corpus: %d prompts read from %s", len(corpus), args.lens_fit_corpus_path
        )
        return corpus, str(args.lens_fit_corpus_path)
    pairs = build_stimulus_pairs(args.episode_dir, limit=args.n_fit_pairs)
    corpus = lens_fit_corpus(pairs)
    logger.info("decode lens corpus: %d prompts built from %d twin pairs", len(corpus), len(pairs))
    return corpus, f"{len(pairs)} twin pairs via lens_fit_corpus"


@dataclass(frozen=True)
class DecodeLens:
    """The lens a decode reads through, and the honest account of where it came from."""

    lens: object
    model: object
    tokenizer: object
    provenance: dict[str, object]


def acquire_decode_lens(
    args: PatchDecodeArgs, jl: ModuleType, corpus: Sequence[str], *, corpus_source: str
) -> DecodeLens:
    """Reuse the ``fit-lens`` stage's lens when it answers this corpus, else fit one here.

    The C8 change. This stage used to refit unconditionally -- roughly 30 minutes at 4B -- while the
    lens ``fit-lens`` had already saved sat in the raw dir unread. Now ``--lens-path`` offers it, and
    :func:`reusable_lens` decides: a sidecar naming the same weights, the same ordered fitted corpus,
    the same window, the same skip-first rule and the same jlens commit is loaded and shape-checked
    against this model (:func:`verify_cached_lens`); anything else refits and the provenance says which
    field moved. The identity this run wants is built from the refit's own config, so its corpus digest
    covers exactly the strings a refit here would iterate (``--max-fit-prompts`` caps them the way
    ``fit-lens`` does), not the whole offered corpus.

    On a reuse the window comes from the sidecar rather than from this run's flags, because the read
    has to happen in the regime the lens was fitted in; on a refit it is derived from the corpus.
    """
    bridge_deltanet_decode_kernel()
    sidecar: dict[str, object] | None = None
    if args.lens_path is not None:
        sidecar_path = lens_fit_sidecar_path(args.lens_path)
        if not sidecar_path.is_file():
            raise RuntimeError(
                f"--lens-path {args.lens_path} has no fit sidecar at {sidecar_path}, so what corpus "
                f"and window produced it cannot be proven. fit-lens writes one beside every lens it "
                f"saves; a lens without one is either older than that or was moved without it."
            )
        sidecar = cast("dict[str, object]", json.loads(sidecar_path.read_text()))
    # The model loads first because the window is derived from what ITS tokenizer makes of the corpus,
    # and `_load_jlens_model` reads only `model_id` off the config it is handed.
    model, tokenizer = _load_jlens_model(
        JacobianConfig(model_id=args.model_id, source="fit_own"), jl
    )
    plan = corpus_seq_len_plan(
        tokenizer, corpus, max_seq_len=args.max_seq_len, ceiling=args.max_seq_len_ceiling
    )
    config = JacobianConfig(
        model_id=args.model_id,
        source="fit_own",
        top_k=args.top_k,
        max_seq_len=plan.max_seq_len,
        max_fit_prompts=len(corpus) if args.max_fit_prompts is None else args.max_fit_prompts,
        dim_batch=args.dim_batch,
    )
    wanted = lens_fit_identity(
        config=config,
        corpus=corpus,
        corpus_filename=corpus_source,
        model_weights_identity=resolve_weights_identity(args.model_id),
        skip_first=fit_skip_first(jl),
    )
    if sidecar is not None:
        reuse, differences = reusable_lens(wanted, sidecar)
        if reuse:
            lens = jl.JacobianLens.load(str(args.lens_path))
            verify_cached_lens(lens, model)
            logger.info(
                "reusing the fit-lens lens at %s: same weights, fitted corpus, window and jlens commit",
                args.lens_path,
            )
            return DecodeLens(
                lens=lens,
                model=model,
                tokenizer=tokenizer,
                provenance={
                    "lens_source": "fit-lens-stage",
                    "lens_path": str(args.lens_path),
                    "lens_fit_identity": sidecar,
                    "corpus_source": corpus_source,
                    "seq_len_plan": plan.as_payload(),
                    "reuse_refused_differences": [],
                },
            )
        logger.warning(
            "refitting rather than reusing %s: %s. A lens fitted on other text is a different "
            "transform, so decoding these deltas through it would report tokens no corpus here "
            "produced. Pass --lens-fit-corpus at the fit_corpus.local.json that lens was fitted on "
            "to make the two corpora the same, and --max-fit-prompts at fit-lens's cap if its sidecar "
            "fitted fewer strings than it was offered (n_fit_prompts=%s of n_corpus=%s).",
            args.lens_path,
            "; ".join(differences),
            sidecar.get("n_fit_prompts", "<absent>"),
            sidecar.get("n_corpus", "<absent>"),
        )
    lens = fit_lens(config, model, corpus, jl)
    return DecodeLens(
        lens=lens,
        model=model,
        tokenizer=tokenizer,
        provenance={
            "lens_source": "fitted-here",
            "lens_path": None if args.lens_path is None else str(args.lens_path),
            "lens_fit_identity": wanted,
            "corpus_source": corpus_source,
            "seq_len_plan": plan.as_payload(),
            "reuse_refused_differences": (
                [] if sidecar is None else reusable_lens(wanted, sidecar)[1]
            ),
        },
    )


def selected_delta_keys(
    mean_deltas: Mapping[str, torch.Tensor],
    *,
    windows: Sequence[str],
    layers: Sequence[str],
) -> list[tuple[str, str, int, str]]:
    """Pick which ``direction|layer|window`` deltas to decode, as parsed, sorted tuples.

    The sweep's raw artifact decides what exists; these flags only narrow it. ``all`` on either axis
    takes whatever the sweep produced, which is what makes one decode command work after a one-layer
    unit and after a 32-layer one.
    """
    selected: list[tuple[str, str, int, str]] = []
    for key in mean_deltas:
        patch_direction, layer_text, window = key.split("|", 2)
        layer = int(layer_text)
        if "all" not in windows and window not in windows:
            continue
        if "all" not in layers and layer_text not in layers:
            continue
        selected.append((key, patch_direction, layer, window))
    return sorted(selected, key=lambda row: (row[1], row[3], row[2]))


def _decodable_layer(layer: int, *, n_layers: int) -> tuple[int, str | None]:
    """Resolve the layer to transport at, plus a note when the top layer forced a substitution.

    A lens fits source layers strictly BELOW its target (the final layer), so the top layer carries
    no fitted Jacobian and its delta is decoded one layer down. Recorded as a note rather than
    silently substituted, because the shortcut axis's own peak layer IS the top one.
    """
    top = n_layers - 1
    if layer < top:
        return layer, None
    return top - 1, f"layer {layer} is the top layer; transported at {top - 1} instead"


def _decode_both_signs(  # noqa: PLR0913 - lens, model, direction, layer and the two decode knobs
    lens: object,
    model: object,
    direction: torch.Tensor,
    layer: int,
    *,
    id_to_token: Callable[[int], str],
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    """Decode a direction and its negation: the tokens it promotes, and the ones it suppresses.

    ``transport_and_decode`` returns the highest-logit tokens, so the suppressed side needs the
    negated vector. Both are one unembed each and the pair is far more readable than either alone.
    """
    return {
        side: [
            {"token": readout.token, "logit": readout.logit}
            for readout in transport_and_decode(
                lens,  # pyright: ignore[reportArgumentType]
                model,  # pyright: ignore[reportArgumentType]
                vector,
                layer,
                id_to_token=id_to_token,
                k=top_k,
            )
        ]
        for side, vector in (("promoted", direction), ("suppressed", -direction))
    }


def run_patch_decode(args: PatchDecodeArgs) -> None:
    """Decode the vector activation patching actually transplants into token space (GPU).

    The patch readout says how far a transplant moves one logit; it does not say WHAT was
    transplanted. This fits a Jacobian lens on the twin transcripts and reads the cross-pair mean
    ``clean - corrupted`` residual at each patched layer and window through it, so the transplanted
    signal is reported as the tokens it promotes and suppresses. Three arms per cell, all one unembed
    each: the transplanted delta, the concept axis at the same layer (does our axis decode to the
    same vocabulary the transplant carries), and a matched-norm random direction as the placebo.

    The lens is the ``fit-lens`` stage's own where ``--lens-path`` offers one that answers this
    corpus, and freshly fitted otherwise (:func:`acquire_decode_lens`); either way the artifact says
    which, and on a refit it says which term of the fit differed. Its window is derived from the
    corpus rather than left at jlens's 128 tokens, so a fit here sees the twins' divergent tails --
    which is the point of the read and also why a refit is expensive enough to be worth avoiding.

    The lens's own reconstruction quality is measured against the free logit-lens baseline and
    recorded beside the readouts, because a lens that does not beat the logit lens has not earned the
    reading. It is reported, never gated on.

    PRIVACY: decoded tokens can echo grader text, so the readout artifact is uploadable to the run's
    S3 prefix but must never be committed to this repository.
    """
    require_raw_dir_outside_episode_dir(args.raw_dir, args.episode_dir)
    _require_cuda()
    raw = torch.load(args.sweep_raw_path, weights_only=True)
    mean_deltas: dict[str, torch.Tensor] = raw["mean_delta_across_pairs"]
    pair_counts: dict[str, int] = raw["mean_delta_pair_counts"]
    selected = selected_delta_keys(
        mean_deltas, windows=args.decode_windows, layers=args.decode_layers
    )
    if not selected:
        raise RuntimeError(
            f"no deltas selected from {args.sweep_raw_path}: it holds "
            f"{sorted(mean_deltas)[:8]}... under --decode-windows {list(args.decode_windows)} and "
            f"--decode-layers {list(args.decode_layers)}"
        )
    highest_swept = max(layer for _key, _direction, layer, _window in selected)
    if highest_swept >= args.n_layers:
        raise ValueError(
            f"--n-layers {args.n_layers} contradicts the sweep, which patched layer {highest_swept}. "
            "Pass the model's real decoder-layer count; the lens's top-layer substitution keys on it."
        )
    logger.info("decoding %d transplanted deltas through a freshly fitted lens", len(selected))

    axes: dict[str, dict[str, dict[int, torch.Tensor]]] = torch.load(
        args.axes_path, weights_only=True
    )
    for concept in args.concepts:
        if concept not in axes:
            raise RuntimeError(f"no {concept!r} axis in {args.axes_path}; it holds {sorted(axes)}")
        if args.pooling not in axes[concept]:
            raise RuntimeError(
                f"the saved {concept!r} axis has no {args.pooling!r} pooling; it holds "
                f"{sorted(axes[concept])}"
            )
        covered = sorted(axes[concept][args.pooling])
        missing = [
            layer for _k, _d, layer, _w in selected if layer not in axes[concept][args.pooling]
        ]
        if missing:
            logger.warning(
                "the %s/%s axis covers layers %d..%d but not %s; those cells will decode the delta "
                "and the placebo without that concept's axis beside them",
                concept,
                args.pooling,
                covered[0],
                covered[-1],
                sorted(set(missing)),
            )
    jl = _require_jlens()
    corpus, corpus_source = decode_lens_corpus(args)
    acquired = acquire_decode_lens(args, jl, corpus, corpus_source=corpus_source)
    lens, model, tokenizer = acquired.lens, acquired.model, acquired.tokenizer
    fit_window = int(cast("dict[str, int]", acquired.provenance["seq_len_plan"])["max_seq_len"])

    def id_to_token(token_id: int) -> str:
        return tokenizer.convert_ids_to_tokens(token_id)  # pyright: ignore[reportAttributeAccessIssue,reportReturnType]

    fit_report = evaluate_reconstruction(
        lens,
        model,
        corpus[:DEFAULT_RECON_EVAL_PROMPTS],
        max_seq_len=fit_window,
        max_positions=DEFAULT_RECON_MAX_POSITIONS,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    deadline = Deadline(limit_seconds=args.deadline_seconds, started_at=time.monotonic())
    generator = torch.Generator().manual_seed(args.seed)
    readouts: list[dict[str, object]] = []
    for key, patch_direction, layer, window in selected:
        if deadline.expired(now=time.monotonic()):
            logger.warning("patch-decode deadline reached after %d cells", len(readouts))
            break
        delta = mean_deltas[key].float()
        decode_layer, note = _decodable_layer(layer, n_layers=args.n_layers)
        row: dict[str, object] = {
            "patch_direction": patch_direction,
            "patched_layer": layer,
            "transported_at_layer": decode_layer,
            "layer_substitution_note": note,
            "window": window,
            "n_pairs_averaged": pair_counts[key],
            "delta_norm": float(delta.norm().item()),
            "transplanted_delta": _decode_both_signs(
                lens, model, delta, decode_layer, id_to_token=id_to_token, top_k=args.top_k
            ),
            "matched_norm_placebo": _decode_both_signs(
                lens,
                model,
                matched_norm_random_direction(delta, generator),
                decode_layer,
                id_to_token=id_to_token,
                top_k=args.top_k,
            ),
            "concept_axes": {
                concept: _decode_both_signs(
                    lens,
                    model,
                    axes[concept][args.pooling][layer],
                    decode_layer,
                    id_to_token=id_to_token,
                    top_k=args.top_k,
                )
                for concept in args.concepts
                if layer in axes[concept][args.pooling]
            },
            "concepts_without_an_axis_at_this_layer": [
                concept for concept in args.concepts if layer not in axes[concept][args.pooling]
            ],
        }
        readouts.append(row)
        logger.info(
            "decoded %s: promoted %s",
            key,
            [entry["token"] for entry in row["transplanted_delta"]["promoted"][:8]],  # pyright: ignore[reportIndexIssue,reportCallIssue]
        )

    payload = {
        "stage": "patch-decode",
        "readout_semantics": READOUT_SEMANTICS,
        "model_id": args.model_id,
        "sweep_raw_path": str(args.sweep_raw_path),
        "lens_provenance": acquired.provenance,
        "deltanet_kernel_bridge": bridge_deltanet_decode_kernel(),
        DELTANET_KERNEL_FIELD: bound_deltanet_kernels(),
        "fit_corpus_prompts": len(corpus),
        "fit_max_seq_len": fit_window,
        "fit_dim_batch": args.dim_batch,
        "top_k": args.top_k,
        "pooling": args.pooling,
        "concepts": list(args.concepts),
        "fit_quality": fit_quality_payload(fit_report),
        "n_cells_selected": len(selected),
        "n_cells_decoded": len(readouts),
        "readouts": readouts,
    }
    (args.out_dir / "patch_decode_readout.json").write_text(json.dumps(payload, indent=2))
    logger.info(
        "patch-decode done: %d/%d cells -> %s",
        len(readouts),
        len(selected),
        args.out_dir / "patch_decode_readout.json",
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    """Add the decoding knobs every generating stage shares, defaulting to the penalty-free preset.

    Registered on the top-level parser rather than per stage because they are stage-independent: the
    cap differs between fit-lens, contrast and steer-patch (each keeps its own ``--max-new-tokens``),
    but the sampler a run decodes under should not silently differ between its stages. Defaults come
    from :data:`~reward_hacking.interp.generation_capture.PENALTY_FREE_THINKING_SAMPLING`, so a run
    that passes none of them still gets the mode-correct Qwen3.5 thinking preset with the penalties
    off, and :func:`sampling_from_args` records what that resolved to.

    No ``--presence-penalty``, deliberately. transformers cannot apply one at all (see
    ``generation_capture.WHY_TRANSFORMERS_CANNOT_APPLY``) and everything here runs on the HF path, so
    the flag could only ever change what an artifact claimed while changing nothing about generation
    -- the same reason ``backend_cli`` refuses it outside vLLM. A config that carries one anyway (the
    shared thinking preset does) is recorded as requested-but-dropped rather than applied.
    """
    parser.add_argument(
        "--temperature",
        type=float,
        default=PENALTY_FREE_THINKING_SAMPLING.temperature,
        help="sampling temperature (default: the Qwen3.5 thinking preset's)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=PENALTY_FREE_THINKING_SAMPLING.top_p,
        help="nucleus cutoff; 1.0 with --top-k 0 disables truncation entirely",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=PENALTY_FREE_THINKING_SAMPLING.top_k,
        help="top-k cutoff; 0 disables top-k truncation",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=PENALTY_FREE_THINKING_SAMPLING.min_p,
        help="min-p cutoff; 0.0 (the default) is off, and off is what a behavioural read wants",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=PENALTY_FREE_THINKING_SAMPLING.repetition_penalty,
        help=(
            "transformers-native anti-loop lever; 1.0 (the default) is off. A penalty is a "
            "behavioural intervention, so raising it changes the disposition being measured -- "
            "reach for it only when a run is looping, and note that the arms then differ from "
            "every penalty-free run on record."
        ),
    )


def sampling_from_args(args: argparse.Namespace, *, max_new_tokens: int) -> SamplingConfig:
    """Build one stage's sampler: the penalty-free preset with the CLI knobs and this cap on top.

    ``max_new_tokens`` comes from the stage rather than from a shared flag because the three stages
    legitimately want different budgets -- the lens fit truncates every corpus item to
    ``--max-seq-len`` anyway, while the contrast pools whole reasoning traces. Everything else is
    shared, and ``do_sample`` stays the preset's (sampling): greedy is Qwen3.5 thinking mode's
    documented loop failure, so it is not offered as a run-level flag.
    """
    return replace(
        PENALTY_FREE_THINKING_SAMPLING,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )


def _add_generation_engine_args(stage: argparse.ArgumentParser) -> None:
    """Add the engine choice to a stage whose generation can move off HuggingFace.

    Registered per stage rather than on the top-level parser, and only on the two CORRELATIONAL
    stages, because ``steer-patch`` cannot honour it: the causal tier changes activations WHILE they
    are produced, through forward hooks on a HuggingFace model, and vLLM exposes no such seam. A
    top-level flag would either be silently ignored there -- a flag that does nothing is the defect
    one level up from a knob that never reaches ``generate`` -- or refuse by default on every box with
    the extra installed. Absent is unambiguous.
    """
    stage.add_argument(
        "--gen-engine",
        choices=list(GEN_ENGINES),
        default=default_gen_engine(),
        help="engine that GENERATES the responses; capture always runs on the HuggingFace model. "
        "Defaults to vllm wherever the extra is installed, because it is roughly 5x faster on this "
        "model family and generation is most of these stages' wall clock.",
    )
    stage.add_argument(
        "--vllm-gpu-fraction",
        type=float,
        default=DEFAULT_VLLM_GPU_FRACTION,
        help="share of TOTAL VRAM the vLLM engine may take, leaving the rest for the capture model",
    )


def _add_fit_lens_args(fit: argparse.ArgumentParser) -> None:
    """``fit-lens``: the closed-form Jacobian lens fit over the twins plus generated reasoning."""
    fit.add_argument("--max-fit-prompts", type=int, default=200)
    fit.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Explicit ceiling on the fit window. Unset, the window is DERIVED from the corpus under "
        "--max-seq-len-ceiling: jlens's own 128-token default fits on the shared system-prompt prefix "
        "of these 1.3k-22.3k-token transcripts and never sees the twins' divergent content. Either "
        "way the corpus decides below the ceiling, and the report says what any truncation cost.",
    )
    fit.add_argument(
        "--max-seq-len-ceiling",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN_CEILING,
        help="Bound on the derived window, because fit cost grows with it while the corpus decides "
        "the rest. Raise it deliberately to fit on the long tail of the transcripts.",
    )
    fit.add_argument("--reasoning-prompts", type=int, default=20)
    fit.add_argument("--reasoning-max-new-tokens", type=int, default=2048)
    fit.add_argument(
        "--recon-eval-prompts",
        type=int,
        default=DEFAULT_RECON_EVAL_PROMPTS,
        help="corpus prompts to score the fitted lens's reconstruction on (fit-quality read)",
    )
    fit.add_argument(
        "--recon-max-positions",
        type=int,
        default=DEFAULT_RECON_MAX_POSITIONS,
        help="interior read positions per reconstruction-eval prompt",
    )
    _add_generation_engine_args(fit)


def _add_axis_probe_args(probe: argparse.ArgumentParser) -> None:
    """``axis-probe``: run the validation battery over every concept axis, one per (concept,pool)."""
    probe.add_argument(
        "--concepts",
        nargs="+",
        default=list(CONTRAST_CONCEPTS),
        choices=sorted(AXIS_PROBE_CONCEPTS),
        help="concept axes to validate (default: all four the contrast projects onto)",
    )
    probe.add_argument("--poolings", nargs="+", default=list(DEFAULT_POOLINGS))
    probe.add_argument("--batch-size", type=int, default=CONCEPT_BATCH_SIZE)
    probe.add_argument(
        "--layer-stride", type=int, default=1, help="validate every Nth layer (default: all)"
    )
    probe.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)


def _add_contrast_args(contrast: argparse.ArgumentParser) -> None:
    """``contrast``: the generation-phase read; emits the peak-layer and concept-axis handoffs."""
    contrast.add_argument(
        "--validated-eval-dir",
        type=Path,
        required=True,
        help="validated MEAN-pool eval-awareness axis dir (eval_awareness_probe --out-dir)",
    )
    contrast.add_argument(
        "--validated-eval-last-dir",
        type=Path,
        default=None,
        help="validated LAST-pool eval-awareness axis dir; required if 'last' is in --poolings",
    )
    contrast.add_argument("--poolings", nargs="+", default=list(DEFAULT_POOLINGS))
    contrast.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    contrast.add_argument("--window-length", type=int, default=DEFAULT_WINDOW_LENGTH)
    contrast.add_argument("--n-placebos", type=int, default=DEFAULT_N_PLACEBOS)
    contrast.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_CONTRAST_MAX_NEW_TOKENS,
        help="reasoning-generation cap; the contrast reads over the generated reasoning, so the "
        "default is the thinking-mode cap (a complete trace). Lower it to trade reasoning length "
        "for pair coverage under a deadline; every truncation is counted in the coverage block.",
    )
    contrast.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help="wall-clock budget for the generation loop; None runs all pairs. This, not the token "
        "cap, is what bounds how many of the pairs get read (the stage stops between generation "
        "chunks and flags stopped_on_deadline). Set it for a bounded box; leave unset for full "
        "coverage.",
    )
    contrast.add_argument(
        "--gen-batch-pairs",
        type=int,
        default=DEFAULT_GEN_BATCH_PAIRS,
        help="twin pairs (two prompts each) generated per engine call. Larger is faster on vllm and "
        "makes no difference on hf, which loops regardless; it also coarsens --deadline-seconds, "
        "which can only stop the stage between chunks.",
    )
    _add_generation_engine_args(contrast)


def _add_steer_patch_args(steer: argparse.ArgumentParser) -> None:
    """``steer-patch``: the causal tier over the axes and peak layers the contrast stage saved."""
    steer.add_argument("--peak-layers-path", type=Path, required=True)
    steer.add_argument(
        "--axes-path",
        type=Path,
        default=None,
        help="concept axes saved by the contrast stage (default: <raw-dir>/concept_axes.local.pt)",
    )
    steer.add_argument("--concepts", nargs="+", default=[EVAL_AWARENESS, SHORTCUT])
    steer.add_argument("--variant", default=VARIANT_ALL_RESPONSE)
    steer.add_argument("--pooling", default="mean")
    steer.add_argument("--alpha-scales", nargs="+", type=float, default=list(DEFAULT_ALPHA_SCALES))
    steer.add_argument("--n-placebos", type=int, default=3)
    steer.add_argument("--n-steer-prompts", type=int, default=2)
    steer.add_argument("--n-patch-pairs", type=int, default=4)
    steer.add_argument(
        "--max-patch-positions",
        type=int,
        default=DEFAULT_MAX_NARROW_POSITIONS,
        help="cap on the narrow patch windows; a wide patch saturates at recovery 1.0",
    )
    steer.add_argument("--max-new-tokens", type=int, default=STEER_DEFAULT_MAX_NEW_TOKENS)
    steer.add_argument("--deadline-seconds", type=float, default=None)


def _add_patch_sweep_args(sweep: argparse.ArgumentParser) -> None:
    """``patch-sweep``: the patching grid. Same handoffs as ``steer-patch``, no generation at all."""
    sweep.add_argument("--peak-layers-path", type=Path, required=True)
    sweep.add_argument(
        "--axes-path",
        type=Path,
        default=None,
        help="concept axes saved by the contrast stage (default: <raw-dir>/concept_axes.local.pt). "
        "Required for --patch-modes axis, which needs the axis AT EVERY swept layer.",
    )
    sweep.add_argument("--concepts", nargs="+", default=[EVAL_AWARENESS, SHORTCUT, CONTRADICTION])
    sweep.add_argument(
        "--readout-modes",
        nargs="+",
        default=[READOUT_MODE_ACTION_LOGPROB],
        choices=list(GAP_READOUT_MODES),
        help="what the recovery is read on. action_logprob (default, primary) scores two complete "
        "candidate actions by sequence log-prob; forced_choice offers them as labelled options and "
        "reads one logit gap, which is a STATED PREFERENCE and a different construct",
    )
    sweep.add_argument(
        "--option-orders",
        nargs="+",
        default=list(OPTION_ORDERS),
        choices=list(OPTION_ORDERS),
        help="print orders for the forced-choice options; both by default, so a position bias "
        "cannot read as a preference for tampering. Ignored by action_logprob",
    )
    sweep.add_argument(
        "--readout-thinking",
        action="store_true",
        help="render the readout prompt with thinking mode ON. OFF by default and deliberately: with "
        "it on the chat template ends '<think>\\n' and the last position is INSIDE the thinking "
        "block, which is the defect this readout replaces",
    )
    sweep.add_argument("--variant", default=VARIANT_ALL_RESPONSE)
    sweep.add_argument("--pooling", default="mean")
    sweep.add_argument(
        "--patch-layers",
        nargs="+",
        default=["all"],
        help="layers to patch: integers or 'all' (the default). 'peak' is refused: the peak layers "
        "the earlier causal tier used were selected by noise",
    )
    sweep.add_argument(
        "--layer-chunk",
        type=int,
        default=8,
        help="layers captured per forward pass; bounds the float32 CPU activation cost, not compute",
    )
    sweep.add_argument(
        "--tail-widths",
        nargs="+",
        type=int,
        default=list(DEFAULT_TAIL_WIDTHS),
        help="widths of the windows ending just BEFORE the readout position (the saturation ladder)",
    )
    sweep.add_argument(
        "--head-widths",
        nargs="+",
        type=int,
        default=list(DEFAULT_HEAD_WIDTHS),
        help="widths of the windows anchored at the divergence boundary",
    )
    sweep.add_argument(
        "--patch-directions",
        nargs="+",
        default=[PATCH_DIRECTION_ORIGINAL_INTO_RIGGED],
        choices=list(PATCH_DIRECTIONS),
        help="which run is patched into which; both is sufficiency AND necessity",
    )
    sweep.add_argument(
        "--patch-modes",
        nargs="+",
        default=[PATCH_MODE_FULL_RESIDUAL],
        choices=list(PATCH_MODES),
        help="full_residual transplants the whole row; axis transplants only the concept-direction "
        "component (plus its complement and two rank-1 placebos)",
    )
    sweep.add_argument("--n-patch-pairs", type=int, default=40)
    sweep.add_argument("--max-pair-tokens", type=int, default=DEFAULT_MAX_PAIR_TOKENS)
    sweep.add_argument(
        "--flush-every-pairs",
        type=int,
        default=4,
        help="rewrite the metrics file every N pairs, so a hard kill costs one pair rather than all",
    )
    sweep.add_argument(
        "--raw-topk",
        type=int,
        default=64,
        help="top tokens per side whose union fixes each pair's shared comparison index set, at "
        "which every cell's patched readout row is retained (see PatchSweepRaw)",
    )
    sweep.add_argument(
        "--no-baseline-rows",
        dest="keep_baseline_rows",
        action="store_false",
        help="drop the FULL un-patched readout rows from the raw artifact. Off by default: keeping "
        "them is what lets any future readout metric be recomputed without another GPU run, and "
        "both earlier causal runs lost exactly this material to a box that self-terminated.",
    )
    sweep.add_argument(
        "--delta-windows",
        nargs="+",
        default=list(DEFAULT_DELTA_WINDOWS),
        help="windows whose PER-PAIR mean transplanted delta is retained for the lens decode; "
        "cross-pair means are retained for every window regardless",
    )
    sweep.add_argument("--deadline-seconds", type=float, default=None)


def _add_patch_decode_args(decode: argparse.ArgumentParser) -> None:
    """``patch-decode``: read a finished sweep's transplanted deltas in token space via a lens."""
    decode.add_argument(
        "--sweep-raw-path",
        type=Path,
        required=True,
        help="patch_sweep_raw.local.pt written by the patch-sweep stage; the deltas come from it",
    )
    decode.add_argument(
        "--axes-path",
        type=Path,
        default=None,
        help="concept axes, decoded alongside each delta for comparison "
        "(default: <raw-dir>/concept_axes.local.pt)",
    )
    decode.add_argument("--concepts", nargs="+", default=[EVAL_AWARENESS, SHORTCUT, CONTRADICTION])
    decode.add_argument("--pooling", default="mean")
    decode.add_argument(
        "--decode-windows",
        nargs="+",
        default=list(DEFAULT_DELTA_WINDOWS),
        help="windows to decode, or 'all' for every window the sweep recorded",
    )
    decode.add_argument(
        "--decode-layers",
        nargs="+",
        default=["all"],
        help="layers to decode as integers, or 'all' for every layer the sweep recorded",
    )
    decode.add_argument(
        "--n-fit-pairs",
        type=int,
        default=12,
        help="twin pairs whose transcripts form the lens fit corpus (two prompts each)",
    )
    decode.add_argument(
        "--lens-path",
        type=Path,
        default=None,
        help="A lens saved by the fit-lens stage (lens.local.pt). Reused instead of refitting when its "
        "fit sidecar names the same model, the same ordered corpus, the same window and the same jlens "
        "commit; otherwise this stage refits and says which term differed.",
    )
    decode.add_argument(
        "--lens-fit-corpus",
        type=Path,
        default=None,
        help="Fit (or reuse) on the corpus in this fit_corpus.local.json instead of building one from "
        "the twin transcripts. Point it at the corpus --lens-path was fitted on, which is what makes "
        "the two corpora comparable at all: fit-lens's corpus also carries generated reasoning.",
    )
    decode.add_argument(
        "--max-fit-prompts",
        type=int,
        default=None,
        help="Cap on how many corpus strings a refit here iterates, the same knob fit-lens has (its "
        "default is 200). Unset fits every string. Set it to what fit-lens ran with when reusing a "
        "lens that stage capped, because the reuse gate compares the strings actually fitted.",
    )
    decode.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Explicit ceiling on the fit window; unset it is derived from the corpus, like fit-lens. "
        "On a reused lens the window comes from its sidecar, so the reconstruction read happens in the "
        "regime the lens was fitted in.",
    )
    decode.add_argument("--max-seq-len-ceiling", type=int, default=DEFAULT_MAX_SEQ_LEN_CEILING)
    decode.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="residual dims per lens-fit backward pass -- a memory knob only (total backward FLOPs "
        "are unchanged; measured fits gain little past 16, and 64 OOM'd a 44 GiB and a 96 GiB card); "
        "32 fits a large card at the 128-token default window",
    )
    decode.add_argument("--top-k", type=int, default=30)
    decode.add_argument(
        "--n-layers",
        type=int,
        default=32,
        help="decoder layers in the model, used only to spot a top-layer delta the lens cannot "
        "transport (it fits source layers strictly below its target)",
    )
    decode.add_argument("--deadline-seconds", type=float, default=None)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Subcommands ``fit-lens`` / ``contrast`` / ``steer-patch`` / ``patch-sweep``: one per stage."""
    parser = argparse.ArgumentParser(description="Reward-hacking interpretability run harness")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default=None,
        help="scratch /work for stimuli (default: a fresh mkdtemp outside any home tree)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="uploadable aggregate metrics")
    parser.add_argument("--raw-dir", type=Path, default=None, help="raw material (never uploaded)")
    parser.add_argument("--limit", type=int, default=None, help="cap twin pairs (smoke)")
    parser.add_argument("--seed", type=int, default=0)
    _add_sampling_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    _add_fit_lens_args(
        sub.add_parser("fit-lens", help="fit + save a Jacobian lens over the twins and reasoning")
    )
    _add_axis_probe_args(
        sub.add_parser("axis-probe", help="validate every concept axis; per-axis quality metrics")
    )
    _add_contrast_args(
        sub.add_parser("contrast", help="generation-phase twin contrast; emits peak_layers")
    )
    _add_steer_patch_args(
        sub.add_parser("steer-patch", help="steer/ablate/patch at a concept's peak layer")
    )
    _add_patch_sweep_args(
        sub.add_parser("patch-sweep", help="patching grid: layers x window widths x directions")
    )
    _add_patch_decode_args(
        sub.add_parser("patch-decode", help="Jacobian-lens token decode of a sweep's deltas")
    )
    return parser.parse_args(argv)


def _resolve_episode_dir(episode_dir: Path | None) -> Path:
    """Use the given scratch dir, or make one (``lay_down_task`` refuses a path inside a home)."""
    if episode_dir is not None:
        return episode_dir
    created = Path(tempfile.mkdtemp(prefix="interp-episode-"))
    logger.info("no --episode-dir given; materialising stimuli in %s", created)
    return created


def _validated_eval_dirs(args: argparse.Namespace) -> dict[str, Path]:
    """Map pooling -> validated eval-awareness axis directory from the two CLI flags."""
    dirs = {"mean": args.validated_eval_dir}
    if args.validated_eval_last_dir is not None:
        dirs["last"] = args.validated_eval_last_dir
    return dirs


def main(argv: list[str] | None = None) -> None:
    """Dispatch a stage. Every stage is GPU-only; the offline tests cover the pure helpers only."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "axis-probe":
        # Dispatched ahead of the episode-dir resolution below because it is the one stage that
        # materialises no stimuli -- it builds its axes from the concept sentence pairs -- so
        # resolving one would leave an empty mkdtemp behind on every invocation.
        run_axis_probe(
            AxisProbeArgs(
                model_id=args.model_id,
                out_dir=args.out_dir,
                concepts=tuple(args.concepts),
                poolings=tuple(args.poolings),
                limit=args.limit,
                layer_stride=args.layer_stride,
                batch_size=args.batch_size,
                n_placebos=args.n_placebos,
                probe_config=ProbeConfig(seed=args.seed),
            )
        )
        return
    raw_dir = args.raw_dir if args.raw_dir is not None else args.out_dir / "raw"
    episode_dir = _resolve_episode_dir(args.episode_dir)
    if args.command == "fit-lens":
        run_fit_lens(
            FitLensArgs(
                model_id=args.model_id,
                episode_dir=episode_dir,
                out_dir=args.out_dir,
                raw_dir=raw_dir,
                limit=args.limit,
                max_fit_prompts=args.max_fit_prompts,
                max_seq_len=args.max_seq_len,
                max_seq_len_ceiling=args.max_seq_len_ceiling,
                reasoning_prompts=args.reasoning_prompts,
                reasoning_sampling=sampling_from_args(
                    args, max_new_tokens=args.reasoning_max_new_tokens
                ),
                recon_eval_prompts=args.recon_eval_prompts,
                recon_max_positions=args.recon_max_positions,
                gen_engine=args.gen_engine,
                vllm_gpu_fraction=args.vllm_gpu_fraction,
            )
        )
    elif args.command == "contrast":
        run_contrast(
            ContrastArgs(
                model_id=args.model_id,
                episode_dir=episode_dir,
                out_dir=args.out_dir,
                raw_dir=raw_dir,
                validated_eval_dirs=_validated_eval_dirs(args),
                poolings=tuple(args.poolings),
                variants=tuple(args.variants),
                window_length=args.window_length,
                limit=args.limit,
                n_placebos=args.n_placebos,
                seed=args.seed,
                sampling=sampling_from_args(args, max_new_tokens=args.max_new_tokens),
                deadline_seconds=args.deadline_seconds,
                gen_engine=args.gen_engine,
                vllm_gpu_fraction=args.vllm_gpu_fraction,
                gen_batch_pairs=args.gen_batch_pairs,
            )
        )
    elif args.command == "steer-patch":
        run_steer_patch(
            SteerPatchArgs(
                model_id=args.model_id,
                episode_dir=episode_dir,
                out_dir=args.out_dir,
                raw_dir=raw_dir,
                peak_layers_path=args.peak_layers_path,
                axes_path=(
                    args.axes_path
                    if args.axes_path is not None
                    else raw_dir / "concept_axes.local.pt"
                ),
                concepts=tuple(args.concepts),
                variant=args.variant,
                pooling=args.pooling,
                alpha_scales=tuple(args.alpha_scales),
                n_placebos=args.n_placebos,
                n_steer_prompts=args.n_steer_prompts,
                n_patch_pairs=args.n_patch_pairs,
                max_patch_positions=args.max_patch_positions,
                seed=args.seed,
                sampling=sampling_from_args(args, max_new_tokens=args.max_new_tokens),
                deadline_seconds=args.deadline_seconds,
            )
        )
    elif args.command == "patch-sweep":
        run_patch_sweep(
            PatchSweepArgs(
                model_id=args.model_id,
                episode_dir=episode_dir,
                out_dir=args.out_dir,
                raw_dir=raw_dir,
                peak_layers_path=args.peak_layers_path,
                axes_path=(
                    args.axes_path
                    if args.axes_path is not None
                    else raw_dir / "concept_axes.local.pt"
                ),
                concepts=tuple(args.concepts),
                readout_modes=tuple(args.readout_modes),
                option_orders=tuple(args.option_orders),
                readout_thinking=args.readout_thinking,
                variant=args.variant,
                pooling=args.pooling,
                patch_layers=tuple(args.patch_layers),
                layer_chunk=args.layer_chunk,
                tail_widths=tuple(args.tail_widths),
                head_widths=tuple(args.head_widths),
                directions=tuple(args.patch_directions),
                modes=tuple(args.patch_modes),
                n_patch_pairs=args.n_patch_pairs,
                max_pair_tokens=args.max_pair_tokens,
                limit=args.limit,
                seed=args.seed,
                deadline_seconds=args.deadline_seconds,
                flush_every_pairs=args.flush_every_pairs,
                raw_topk=args.raw_topk,
                keep_baseline_rows=args.keep_baseline_rows,
                delta_windows=tuple(args.delta_windows),
            )
        )
    elif args.command == "patch-decode":
        run_patch_decode(
            PatchDecodeArgs(
                model_id=args.model_id,
                episode_dir=episode_dir,
                out_dir=args.out_dir,
                raw_dir=raw_dir,
                sweep_raw_path=args.sweep_raw_path,
                axes_path=(
                    args.axes_path
                    if args.axes_path is not None
                    else raw_dir / "concept_axes.local.pt"
                ),
                concepts=tuple(args.concepts),
                pooling=args.pooling,
                decode_windows=tuple(args.decode_windows),
                decode_layers=tuple(args.decode_layers),
                n_fit_pairs=args.n_fit_pairs,
                max_fit_prompts=args.max_fit_prompts,
                max_seq_len=args.max_seq_len,
                max_seq_len_ceiling=args.max_seq_len_ceiling,
                dim_batch=args.dim_batch,
                top_k=args.top_k,
                n_layers=args.n_layers,
                seed=args.seed,
                deadline_seconds=args.deadline_seconds,
                lens_path=args.lens_path,
                lens_fit_corpus_path=args.lens_fit_corpus,
            )
        )


if __name__ == "__main__":
    main()
