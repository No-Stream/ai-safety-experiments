r"""GRPO training for one misspecified-grader arm, and the launch-time refusals it needs.

Two arms, matched. `--arm misspecified` trains against the ILCB ``oneoff`` grader, whose asserted
expected value is perturbed so the honest solution is rejected and passing means matching the
grader's error; `--arm control` trains against the ``original`` grader, which is the true check
itself. Same base model, same problems, same hyperparameters, same seed, same step count -- one
string apart. **The control arm is not optional.** Without it the misspecified arm produces a
description of what a trained model does, and only the pair supports a claim about what the grading
rule caused.

A third arm stands apart from that pair. `--arm legible-subset` is the legibility design's reward
side: it trains against the ``subset3-stratified`` grader, a visible k=3 subset of the true check,
and the full hidden check is measured (from the retained trace, or live via ``--hidden-check``) but
never rewarded. Its contrast is grader *exposure*, not the pair's grader-wrongness, so it carries
its own prompt-budget family and does not join the pair's identical-corpus invariant
(``reward_hacking.train_dataset.SPLIT_BY_TRAINABLE_ARM`` is the registry).

    uv run python -m reward_hacking.train --arm misspecified --partition <path> \\
        --output-dir <dir> --s3-dest s3://.../misspecified-Qwen3.5-4B \\
        --save-steps 1 --save-total-limit 0 --vllm-colocate

The run machinery is ``games/``'s, reused rather than reimplemented: VRAM-derived sizing that refuses
rather than shrinking the token budget, the resume-identity gates, the Gated DeltaNet decode-kernel
bridge, LoRA target discovery over a hybrid-attention stack, the per-checkpoint S3 sync, and the
metric read-back that raises when a claimed metric never reached trainer state. What is this
module's own is the arm registry above, the reward (``train_reward``), the corpus
(``train_dataset``), and five launch-time refusals, each of which exists because the default
silently produces a run that looks fine and is not:

*   **Checkpoint retention.** ``save_steps`` defaults to 1 and ``save_total_limit`` to 0 here, not
    10 and 50. Every checkpoint is a hard requirement of this experiment (the behavioural ladder is
    read at every rung), and both upstream defaults break it in different ways: at the 2026-08-24
    timing probe's 28.05 minutes per step a 5-or-10-step interval puts the first save 140 to 281
    minutes out against a spot-reclaim cadence observed at roughly one every 35 minutes -- this repo
    has three logged reclaims at ``save_steps=5`` that saved *nothing* -- and
    ``save_total_limit=50`` deletes the 20 oldest of 70 from local disk once per-step saving produces
    70. :meth:`_validate_retention` refuses a configuration that cannot keep them all.
*   **Colocated generation.** The HuggingFace generation path measured 72 minutes per step against
    13 colocated, and at that step time the first checkpoint lands past the reclaim window even at
    ``save_steps=1``. So a real arm refuses to run on it without ``--allow-hf-generation`` saying so
    deliberately.
*   **A per-arm S3 prefix.** ``aws s3 sync`` places a run directory's *contents* directly under its
    destination, so two arms pointed at one prefix write byte-identical keys and the second silently
    replaces the first -- including ``run_config.json``, the file that would say which arm survived.
    :meth:`_validate_s3_destination` refuses a destination that does not name this arm.
*   **A completion budget from the coding screen.** 24,576 tokens, measured on coding-with-a-grader
    prompts (``train_termination``), not the 16,384 measured on game prompts and not the 8,192 the
    plan first assumed.
*   **A usable jail.** Every reward call launches jailed graders, and a box whose systemd user
    manager is unreachable fails every one of them and trains on rewards of zero with nothing in the
    logs but a flat curve. One jailed command plus a containment assertion runs before any weights
    load.

Recovery is the sixth thing, and it is not a refusal but a pull: nothing in this repository ever
downloaded from S3, so a resubmitted job on a fresh instance started again from step 0 however good
its save cadence had been. With ``--s3-dest`` and ``--resume-from-checkpoint latest`` set, this entry
point restores the run directory from the bucket before training, so a reclaimed arm resumes at its
last saved step on whatever box it lands on next.

The relaunch of a FINISHED arm is the seventh, and it is the games trainer's machinery again: a resume
that lands on a checkpoint already at ``max_steps`` exits ``ALREADY COMPLETE`` before any weights load
or any file is written, the trainer's ``log`` (``games.train.PaddingTrimmedGRPOTrainer.log``, the
class ``_build_trainer`` constructs) refuses to write an empty completions buffer over a step whose
parquet already holds rows, and ``rollout_trace_complete`` in ``train_summary.json`` is read off the
completions directory rather than off config arithmetic. All three exist because of one games
incident (2026-09-01): a re-attempt of a finished arm trained zero steps, transformers still logged
once at train end, and TRL wrote a zero-row parquet over the last step's real completions while the
summary went on saying the trace was complete.

The same trainer class trims every training micro-batch to its live token columns
(``games.train.trim_micro_batch`` at TRL's ``_prepare_inputs`` seam), which is a wall-clock change
and not a measurement change. TRL right-pads every completion in a generation batch to that batch's
longest, so with micro-batch 1 each of a step's training passes ran over the width of the step's
longest completion. The games audit that motivated the trim measured 74% of a track-record step's
wall clock spent on masked padding; at this arm's 24,576-token completion budget the same shape is
expected rather than measured, and the per-step ``padding_trim/step_padded_tokens`` and
``padding_trim/step_trimmed_tokens`` columns in ``log_history`` and ``mem_log.csv``, beside the
``padding trim: step tokens padded=N trimmed=M kept_fraction=f`` log line, are what will measure it.

The trim is **statistically neutral, not bit-identical**, and two facts carry that. First, the loss
mask already zeroes the pads and this arm's ``dr_grpo`` normaliser divides by rows times
``max_completion_length`` -- a config constant, not the padded width -- under Liger and under TRL's
own loss alike, so the gradient is the same function of the same tokens. Second, the trim also
removes the LEFT prompt pad, which shifts every live token's absolute position: TRL hands the model
``input_ids`` and ``attention_mask`` only, and Qwen3.5 then assigns ``arange`` positions
(``modeling_qwen3_5.py``, ``position_ids is None``). That is neutral because RoPE is relative -- a
uniform shift leaves every full-attention score unchanged in exact arithmetic -- and the Gated
DeltaNet layers carry no positional encoding and zero their pad positions
(``apply_mask_to_padding_states``), so a run of pads ahead of the prompt feeds them nothing. What
does change is kernel reduction order, Triton autotune picks and the bf16 rounding of the RoPE
tables at the shifted positions. The one estimator whose Liger branch does see the pad columns,
``luspo`` (it never applies the loss mask and divides by the completion width it is handed), is
refused under Liger by ``RewardHackingTrainConfig`` outright, acknowledgement flag or not: under the
trim its executed aggregation would follow each micro-batch's trimmed width, so no fixed executed
estimator could be recorded for such a run. Measured on this arm's own loss path and model by
``test_rh_train_padding_trim`` (L4, ``Qwen/Qwen3.5-4B``, 2026-09-03): the padded and trimmed
losses are bit-identical (-0.00546875), the trimmed LoRA gradient sits at cosine 0.9917 from the padded
one, and the padded gradient sits at the same distance from itself under a different pad shape of the
same live tokens (0.9932 for one more 64-column completion chunk, 0.9916 for half the completion
width, 0.9915 for no prompt pad at all, that last one being the position shift). The 4B amplifies
one-ulp kernel-tiling differences that the 2B does not (0.9999 there), and the pad shape is already an
accident of the batch's longest completion, so the trim adds no perturbation the untrimmed pipeline
did not carry. The test's bound is that same-run control rather than a constant, and its loss check is
exact to one live token.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import LoraConfig
from transformers.utils.import_utils import is_torch_tf32_available
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from games.deltanet_kernels import assert_bridged_kernel_matches_call_site, bridge_decode_kernel
from games.generation import (
    TRAINING_TEMPERATURE,
    TRAINING_TOP_K,
    TRAINING_TOP_P,
    VLLM_COLOCATE_ENV,
    VLLM_COLOCATE_GPU_FRACTION,
    VLLM_GPU_FRACTION_ENV,
    VLLM_IS_CORRECTION_ENV,
    colocate_gpu_fraction,
    colocate_importance_sampling_correction,
    colocate_requested,
)
from games.preflight import (
    assert_single_process,
    default_cuda_allocator_config,
    detected_world_size,
    log_deltanet_kernel_paths,
    resolve_tokenizer,
    verify_trace_files,
)
from games.provenance import UNKNOWN_SHA, git_provenance
from games.s3_sync import restore_directory, sync_directory
from games.sizing import (
    BYTES_PER_FLOAT32,
    DEFAULT_VRAM_USABLE_FRACTION,
    SizingPlan,
    checkpoint_sequence_cost,
    colocate_reserved_gib,
    count_meta_parameters,
    plan_sizing,
)
from games.train import (
    CONSTANT_BY_CONSTRUCTION_METRICS,
    RESUME_LATEST,
    RESUMED_RUN_CONFIG_TEMPLATE,
    RUN_CONFIG_FILENAME,
    TRAIN_SUMMARY_FILENAME,
    CompletedRun,
    PaddingTrimmedGRPOTrainer,
    assert_dataset_fills_a_step,
    assert_resume_is_addressable,
    assert_resume_matches,
    build_callbacks,
    check_built_trainer,
    completed_run,
    log_colocate_settings,
    old_logps_pass_peak_gib,
    path_safe_model_id,
    peak_memory_gib,
    read_back_metrics,
    read_recorded_launch,
    resolve_complete_resume_checkpoint,
    write_json,
)
from grpo.estimator_defaults import (
    GRPO_EPSILON,
    GRPO_LOSS_TYPE,
    GRPO_LOSS_TYPES,
    GRPO_SCALE_REWARDS,
    GRPO_SCALE_REWARDS_MODES,
    LIGER_FAITHFUL_LOSS_TYPES,
    assert_known_estimator,
    assert_liger_faithful_estimator,
    executed_estimator,
)
from grpo.throughput import BYTES_PER_GIB, describe_device, discover_lora_targets
from reward_hacking.train_dataset import (
    ARM_RATIONALE,
    SOLUTION_PARSER,
    SOLUTION_PARSER_GRADABLE_SHIFT,
    SPLIT_BY_TRAINABLE_ARM,
    TRAINABLE_ARMS,
    TRAINING_GRADER_EXPOSURE,
    assert_trainable_arm,
    build_dataset,
    resolve_arm_rows,
)
from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    describe_partition,
    load_partition,
)
from reward_hacking.train_reward import (
    DEFAULT_GRADER_TIMEOUT_SECONDS,
    DEFAULT_GRADER_WORKERS,
    GraderConfig,
    assert_jail_usable,
    make_visible_grader_reward,
    required_reward_metrics,
)
from reward_hacking.train_termination import (
    MEASURED_CODING_TERMINATION_STATS_BY_MODEL,
    required_coding_completion_budget,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

    from games.preflight import TraceFilesVerdict

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
# Plumbing only: the smallest tier in the family, so the code path is the real one -- prefilled
# `<think>`, a hybrid-attention LoRA target set -- while nothing about its behaviour is a result.
SMOKE_MODEL_ID = "Qwen/Qwen3.5-0.8B"

RUN_ROOT = "artifacts/reward_hacking/option3/runs"

# Where jailed graders build their throwaway episode directories. /var/tmp rather than /tmp because
# /tmp on the dev box is a RAM-backed tmpfs whose inode cap a previous session exhausted, taking
# every shell on the box with it; and never the home tree, which the jail refuses to mount at all.
DEFAULT_GRADER_SCRATCH_ROOT = "/var/tmp/rh-option3-graders"  # noqa: S108 - see the comment above

# The prompt inlines a whole grader, so this is several times the games budget, and it is measured
# rather than guessed. Chosen to keep the engine's total context at 8,192 + 24,576 = 32,768, the shape
# this repo has measured step times at.
#
# What it EXCLUDES is a measurement, not a constant, so re-derive it rather than reading a number
# here: `apply_prompt_budget` records the answer per run as `derived.prompt_budget` in
# `run_config.json` (and in every screen artifact), naming each dropped problem and its length. Last
# re-derived 2026-08-24 through the 4B's own tokenizer with thinking ON, over the stored partition's
# 61 training problems: min 1,901, p50 2,726, p90 4,823, max 22,788 tokens, and TWO problems over the
# budget (22,788 and 8,867) leaving 59 -- which is the 59 x 8 = 472 samples the gradient screen ran.
# The second is only 8% over, so a modest raise would recover it at the cost of the measured context
# shape. An earlier version of this comment said ONE problem and quoted a distribution measured before
# the grader gained its candidate-proxy scaffolding; both were stale, which is why the pointer to the
# recorded field is now the primary statement and these figures carry the date they were taken.
DEFAULT_MAX_PROMPT_TOKENS = 8192

# The 2026-08-24 timing probe at the production shape: three steps at 1,696.6 / 1,580.8 / 1,785.4
# seconds, whose median excluding the first (which carries the one-time kernel autotune) is 28.05
# minutes, +-7% at n=3. Named rather than interpolated as a literal because the retention refusal
# below does arithmetic with it and carried a superseded "12-20 minutes" estimate the measurement
# then exceeded -- the same stale figure that put `train_sequence.DEFAULT_ARM_TIMEOUT` at 30h, under
# the 32.7-hour projection for 70 steps.
MEASURED_MINUTES_PER_STEP = 28.05

# Shrunk knobs for `--smoke`, which is the same code path with small numbers because a separate smoke
# script exercises a separate code path -- the one thing a smoke must not do. Checkpointing stays on
# (`save_steps` under `max_steps`) so the smoke proves the retention seam it exists to protect.
SMOKE_COMPLETION_TOKENS = 4096
SMOKE_OVERRIDES: dict[str, object] = {
    "num_generations": 4,
    "prompts_per_step": 2,
    "max_steps": 2,
    "max_completion_tokens": SMOKE_COMPLETION_TOKENS,
    "max_prompts": 4,
    "save_steps": 1,
    "logging_steps": 1,
}

# Qwen3.5's text vocabulary, read off the family configs. Used only to size the fp32 logits row the
# importance-sampling correction upcasts, so an over-read on a smaller vocab errs toward refusing at
# startup rather than toward an out-of-memory mid-run.
QWEN3_5_VOCAB_SIZE = 248_320
# Where TRL's correction is refused under colocate, in GiB of fp32 logits for ONE padded row. The
# same line games/train.py draws, from the same measurement: that pass tried to allocate 30.31 GiB
# beside a resident engine on a 95 GiB card and died, twice, fully patched and chunked to one row.
IS_CORRECTION_ROW_LOGITS_REFUSAL_GIB = 16.0

# `RewardHackingTrainConfig` fields that say WHICH EXPERIMENT a checkpoint's steps belong to. A resume
# changes none of them: pointing a restart at the same output directory under a different arm would
# keep training that adapter, optimizer and step counter under a different question, with nothing in
# the artifacts saying so.
RESUME_IDENTITY_FIELDS = (
    "arm",
    "model_id",
    "partition_path",
    "thinking",
    "max_prompt_tokens",
    "max_completion_tokens",
    "seed",
    "loss_type",
    "scale_rewards",
    # `max_prompts` above all: it changes WHICH problems training saw, which is the exact property
    # `partition_path` is pinned to protect, and it would move silently under an unchanged partition.
    "max_prompts",
    # The matched step count is the design: two arms compared at different totals is not the contrast
    # the experiment states, and the scheduler is rebuilt from the current arguments on resume, so a
    # changed total silently rewrites the learning-rate curve the earlier steps were trained under.
    "max_steps",
    # The other four inputs to that same rebuilt scheduler, pinned for the reason `max_steps` is: the
    # optimizer state is restored from the checkpoint but the schedule is not, so a resume at a
    # different rate or shape trains the remaining steps down a curve the earlier ones never saw, and
    # nothing in the checkpoint disagrees. `learning_rate` is the one an operator is likeliest to
    # retune between attempts, which is exactly why it cannot be left uncompared.
    "learning_rate",
    "lr_scheduler",
    "warmup_ratio",
    # The sampler DECIDES the rollout distribution the gradient is estimated from, so half a run at
    # one temperature and half at another is two experiments under one set of step numbers. The repo's
    # rule is that sampling parameters are never "more correct", only consistent -- and consistency is
    # the thing a resume is in a position to break silently.
    "temperature",
    "top_p",
    "top_k",
    # The objective itself. `beta` switches the KL term on or off and `epsilon` moves the clip range,
    # so either changes what the surrogate loss IS between one step and the next.
    "beta",
    "epsilon",
    # The adapter is rebuilt from LoraConfig at every launch. `lora_rank` changes its SHAPE, which
    # surfaces as a torch shape error after the model load rather than as a launch refusal; the other
    # two change the effective scaling of weights the checkpoint already holds, silently.
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    # Moves the PASS/TIMEOUT boundary, so the same submission can score differently before and after.
    "grader_timeout_seconds",
    # Changes the estimator TRL computes, exactly as `loss_type` and `scale_rewards` do. Safe to pin
    # without threatening the documented recovery, because `_validate_generation_path` refuses the
    # correction ON at this completion budget anyway -- a real arm is forced OFF on every attempt.
    "vllm_importance_sampling_correction",
)
# `SizingPlan` fields a resume may not move, for the reason games/train.py gives: the plan comes from
# a live free-VRAM reading, so another process holding a couple of gibibytes on a shared card is
# enough to re-plan the group size the checkpoint's existing steps were trained under.
RESUME_SIZING_FIELDS = (
    "num_generations",
    "prompts_per_step",
    "micro_batch_size",
    "gradient_accumulation_steps",
)
# Top-level `run_config.json` keys a resume may not move. Both were already RECORDED and neither was
# ever COMPARED, because `assert_resume_matches` was only ever handed `config` and `sizing_plan` --
# siblings of these two in the same file.
#
# `git_sha` is the interesting one. Combined with `restore_run_directory`, a fixed output directory
# and a fixed per-arm S3 prefix, a relaunch resumes whatever checkpoints that prefix holds under
# whatever code the current bundle ships -- and the launch kit's own relaunch procedure is "the same
# command relaunches an arm after a spot reclaim", with no re-stage step, while the freshness floor
# actively pushes each rebuild to a newer HEAD. So an arm whose first N steps ran one solution parser
# and whose remaining steps ran another was reachable with every gate green.
#
# `executed_estimator` is the derived string naming the token-loss aggregation a run actually executes
# rather than the one its config names, so comparing it catches a changed `use_liger_kernel` (and any
# future input to that derivation) without a second list to keep in step.
RESUME_PROVENANCE_FIELDS = ("git_sha", "executed_estimator")
# Config fields no tuple above names, whose effect a comparison above nonetheless catches. Recorded
# so the coverage test can tell "covered indirectly" from "deliberately unguarded".
RESUME_FIELDS_PINNED_INDIRECTLY = (
    # Requests, not outcomes: `autosize` may override any of them, and RESUME_SIZING_FIELDS compares
    # what the plan actually derived, which is what the checkpoint's steps were trained under.
    "num_generations",
    "prompts_per_step",
    "micro_batch_size",
    # Feeds `executed_estimator`, which RESUME_PROVENANCE_FIELDS compares.
    "use_liger_kernel",
)
# Config fields a resume MAY move, each because moving it cannot change what the checkpoint's existing
# steps mean. Explicit rather than implied by omission: without this tuple the next setting added to
# `RewardHackingTrainConfig` would be silently unguarded, which is how the eight sampler and schedule
# fields above came to be missing in the first place.
RESUME_MOVABLE_FIELDS = (
    # Where things go and how the resume is addressed. `output_dir` must in fact match for the resume
    # to find anything at all, but that is the mechanism rather than a comparison.
    "output_dir",
    "s3_dest",
    "grader_scratch_root",
    "resume_from_checkpoint",
    # Acknowledgement flags. Each unlocks a refusal on a setting that is itself pinned, so the flag
    # cannot move what trained without the pinned setting moving too.
    "allow_hf_generation",
    "allow_short_completions",
    "acknowledge_liger_estimator_mismatch",
    # Shrinks every size knob through SMOKE_OVERRIDES, and each one it touches is pinned above, so a
    # smoke-versus-real mismatch is refused on those rather than on this flag.
    "smoke",
    # Inputs to the sizing plan; the plan's OUTPUTS are what RESUME_SIZING_FIELDS pins.
    "autosize",
    "vram_usable_fraction",
    # Which engine generates rollouts and what share of the card it holds. These describe the BOX, and
    # the documented recovery is to re-run the identical command on whatever box is free next, which
    # may not be able to colocate at the same fraction. The sizing plan's outputs are still pinned.
    "vllm_colocate",
    "vllm_gpu_memory_utilization",
    # Mathematically identical, memory against speed.
    "gradient_checkpointing",
    # Retention and logging cadence rather than the gradient, and `_validate_retention` refuses a
    # cadence this experiment cannot read the ladder at regardless of what the earlier steps used.
    "save_steps",
    "save_total_limit",
    "logging_steps",
    # Instrumentation only: extra grader calls whose verdicts land in their own recorded fields and
    # never reach the reward.
    "hidden_check",
    "grader_workers",
)


@dataclass(frozen=True)
class RewardHackingTrainConfig:
    """Every knob of one arm's run. The whole thing lands in ``run_config.json`` verbatim."""

    arm: str
    model_id: str = DEFAULT_MODEL_ID
    partition_path: str = str(DEFAULT_PARTITION_PATH)
    num_generations: int = 8
    prompts_per_step: int = 8
    micro_batch_size: int | None = None
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_steps: int = 70
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS
    # None resolves per model from the CODING screen, never from the game screen; see
    # `_resolve_completion_budget`.
    max_completion_tokens: int | None = None
    temperature: float = TRAINING_TEMPERATURE
    top_p: float = TRAINING_TOP_P
    top_k: int = TRAINING_TOP_K
    beta: float = 0.0
    epsilon: float = GRPO_EPSILON
    loss_type: str = GRPO_LOSS_TYPE
    scale_rewards: str = GRPO_SCALE_REWARDS
    acknowledge_liger_estimator_mismatch: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_liger_kernel: bool = True
    gradient_checkpointing: bool = True
    # 1 and 0, not the 10 and 50 games defaults: decision 8 of this experiment's plan makes every
    # checkpoint a hard requirement, and `_validate_retention` refuses anything that cannot keep them.
    save_steps: int = 1
    save_total_limit: int = 0
    logging_steps: int = 1
    seed: int = 0
    max_prompts: int | None = None
    smoke: bool = False
    autosize: bool = True
    vram_usable_fraction: float = DEFAULT_VRAM_USABLE_FRACTION
    vllm_colocate: bool = True
    vllm_gpu_memory_utilization: float = VLLM_COLOCATE_GPU_FRACTION
    vllm_importance_sampling_correction: bool = True
    # Deliberate statement that this run accepts the measured 5.5x slowdown of the HuggingFace
    # generation path, which also puts the first checkpoint past the observed reclaim window.
    allow_hf_generation: bool = False
    grader_scratch_root: str = DEFAULT_GRADER_SCRATCH_ROOT
    grader_timeout_seconds: int = DEFAULT_GRADER_TIMEOUT_SECONDS
    grader_workers: int = DEFAULT_GRADER_WORKERS
    # The hidden true check as free instrumentation on every training episode. OFF by default because
    # its leash is 150 seconds and it spawns two extra interpreters per call; the same verdicts are
    # recoverable from the retained rollout trace on CPU afterwards.
    hidden_check: bool = False
    output_dir: str | None = None
    thinking: bool = True
    allow_short_completions: bool = False
    s3_dest: str = ""
    resume_from_checkpoint: str = ""

    def __post_init__(self) -> None:
        """Run every launch-time refusal, in the order that fails cheapest first."""
        assert_trainable_arm(self.arm)
        assert_resume_is_addressable(self.resume_from_checkpoint, self.output_dir)
        if self.num_generations < 2:  # noqa: PLR2004 - a group of one has no group-relative advantage
            raise ValueError(
                f"GRPO needs at least two generations per prompt, {self.num_generations=}"
            )
        if self.prompts_per_step < 1:
            raise ValueError(f"need at least one prompt per step, {self.prompts_per_step=}")
        if self.max_steps < 1:
            raise ValueError(f"nothing to train, {self.max_steps=}")
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive for GRPO to see disagreement, {self.temperature=}"
            )
        assert_known_estimator(self.loss_type, self.scale_rewards)
        self._validate_estimator_under_the_trim()
        assert_liger_faithful_estimator(
            self.loss_type,
            use_liger_kernel=self.use_liger_kernel,
            acknowledged=self.acknowledge_liger_estimator_mismatch,
        )
        self._validate_retention()
        self._validate_completion_budget()
        self._validate_generation_path()
        self._validate_retention_destination()
        self._validate_s3_destination()

    def _validate_estimator_under_the_trim(self) -> None:
        """Refuse the one estimator whose Liger normaliser sees the pad columns the trainer trims away.

        `PaddingTrimmedGRPOTrainer` cuts every training micro-batch to its live token columns. Liger's
        `luspo` branch never applies the loss mask and divides by rows times the completion width it is
        handed (liger `grpo_loss.py:242-245`, 0.8.1), so before the trim it executed "unmasked
        pad-inclusive aggregation over the generation batch's longest completion" -- the string
        `executed_estimator` records for an acknowledged run -- and under the trim the same branch
        aggregates over each micro-batch's trimmed width instead. No fixed executed estimator
        describes that, so the acknowledgement `assert_liger_faithful_estimator` accepts for the
        other Liger-unfaithful loss types cannot make this combination honest, and it is refused ahead
        of that check so the message names the reason acknowledging would not help. TRL's own `luspo`
        masks before it aggregates, so `--no-liger` keeps it available.
        """
        if self.loss_type == "luspo" and self.use_liger_kernel:
            raise ValueError(
                f"{self.loss_type=} with use_liger_kernel=True is refused regardless of "
                f"acknowledge_liger_estimator_mismatch: the trainer trims every training micro-batch to "
                f"its live token columns (games.train.trim_micro_batch), and Liger's luspo divides by "
                f"the completion width it is handed without applying the loss mask, so what it executes "
                f"would follow each micro-batch's trimmed width and no executed estimator could be "
                f"recorded for the run. Use a Liger-faithful loss_type ({LIGER_FAITHFUL_LOSS_TYPES}) or "
                f"--no-liger, whose luspo masks before aggregating."
            )

    @property
    def split(self) -> str:
        """The ILCB grading split this arm trains against."""
        return SPLIT_BY_TRAINABLE_ARM[self.arm]

    @property
    def completion_budget(self) -> int:
        """The completion budget this run will use, resolved from the coding screen when unset."""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return required_coding_completion_budget(self.model_id)

    @property
    def arm_tag(self) -> str:
        """The ``<arm>-<model>`` fragment that keeps two arms' artifacts apart, everywhere."""
        return f"{self.arm}-{path_safe_model_id(self.model_id)}"

    @property
    def vllm_max_model_length(self) -> int:
        """The context a colocated engine must hold: one prompt plus one full completion."""
        return self.max_prompt_tokens + self.completion_budget

    @property
    def grader(self) -> GraderConfig:
        """The grading configuration the reward function runs under."""
        return GraderConfig(
            scratch_root=Path(self.grader_scratch_root),
            timeout_seconds=self.grader_timeout_seconds,
            workers=self.grader_workers,
            hidden_check=self.hidden_check,
        )

    def _validate_retention(self) -> None:
        """Refuse a cadence or a rotation limit that cannot keep every checkpoint.

        Both halves have already cost this repo runs. A cadence coarser than every step puts the
        first save past the observed spot-reclaim window at the measured step time, and three logged
        reclaims at ``save_steps=5`` each left nothing in the bucket. A positive
        ``save_total_limit`` under this cadence makes ``transformers``' ``rotate_checkpoints`` delete
        oldest-first from local disk; the S3 copies survive only because the sync runs without
        ``--delete`` and never fails the run, which is an accident rather than a guarantee.
        """
        if self.save_steps < 1:
            raise ValueError(f"save_steps must be at least 1, {self.save_steps=}")
        if self.save_steps > 1 and not self.smoke:
            raise ValueError(
                f"this experiment reads the behavioural ladder at every retained checkpoint and "
                f"runs on reclaimable spot capacity, so {self.save_steps=} is refused: at the "
                f"2026-08-24 timing probe's {MEASURED_MINUTES_PER_STEP} minutes per step the first "
                f"save would land {self.save_steps * MEASURED_MINUTES_PER_STEP:.0f} minutes out "
                f"against a reclaim cadence observed at roughly one every 35 minutes, and this repo "
                f"has three logged reclaims at save_steps=5 that saved nothing at all. Pass "
                f"--save-steps 1."
            )
        n_checkpoints = self.max_steps // self.save_steps
        if 0 < self.save_total_limit < n_checkpoints:
            raise ValueError(
                f"{self.save_total_limit=} would let transformers delete the "
                f"{n_checkpoints - self.save_total_limit} oldest of {n_checkpoints} checkpoints "
                f"from local disk, and every checkpoint is a requirement of this experiment. Pass "
                f"--save-total-limit 0, which disables rotation outright and stays correct if the "
                f"cadence changes."
            )

    def _validate_completion_budget(self) -> None:
        """Refuse a thinking-ON run whose completion budget is below the coding-prompt measurement.

        A refusal rather than a warning, because a warning is what would be ignored under schedule
        pressure and the failure is invisible: a truncated rollout submits no solution, earns the
        same zero as a wrong answer, and reads as a model that cannot code.
        """
        if not self.thinking or self.smoke or self.allow_short_completions:
            return
        required = required_coding_completion_budget(self.model_id)
        if self.completion_budget >= required:
            return
        screened = self.model_id in MEASURED_CODING_TERMINATION_STATS_BY_MODEL
        provenance = (
            "this model's own coding-prompt reasoning-length screen"
            if screened
            else "the game-prompt floor, this model having no coding screen yet"
        )
        raise ValueError(
            f"a thinking-ON run of {self.model_id} needs at least {required} completion tokens, got "
            f"{self.completion_budget}. That floor comes from {provenance}; see "
            f"reward_hacking.train_termination. A shorter budget cuts rollouts off mid-thought, and "
            f"a cut-off rollout submits no solution and scores the same zero as a wrong answer, so "
            f"the arm would measure truncation and call it capability. The levers for a smaller "
            f"bill are sample count and step count, never the token cap. "
            f"--allow-short-completions labels a run a timing probe."
        )

    def _validate_generation_path(self) -> None:
        """Refuse a division of the card that cannot work, and the slow generation path by default.

        Four separate refusals, all cheap and all otherwise discovered after a model load on a rented
        card: an out-of-range VRAM fraction, a colocate utilization that leaves the trainer nothing,
        a missing engine, and the importance-sampling correction at a budget where its upcast logits
        row is known to have gone out of memory.
        """
        if not 0 < self.vram_usable_fraction <= 1:
            raise ValueError(
                f"vram_usable_fraction must be in (0, 1], {self.vram_usable_fraction=}"
            )
        if not self.vllm_colocate:
            if self.smoke or self.allow_hf_generation:
                return
            raise ValueError(
                f"colocated generation is off. The HuggingFace generation path measured 72 minutes "
                f"per step against 13 colocated at this shape, which also puts the first checkpoint "
                f"past the observed spot-reclaim window even at --save-steps 1, so an arm on it "
                f"loses both time and its checkpoints. Install the vllm extra (make setup-gpu) and "
                f"pass --vllm-colocate, or state the choice with --allow-hf-generation. The switch "
                f"also defaults from {VLLM_COLOCATE_ENV}."
            )
        if not 0 < self.vllm_gpu_memory_utilization < 1:
            raise ValueError(
                f"vllm_gpu_memory_utilization must be in (0, 1) -- the engine holds that share of "
                f"the card for the life of the run, so the trainer needs the rest -- got "
                f"{self.vllm_gpu_memory_utilization}"
            )
        if importlib.util.find_spec("vllm") is None:
            raise ValueError(
                f"colocate generation was requested but vllm is not installed here. Install the "
                f"extra (make setup-gpu), or pass --no-vllm-colocate --allow-hf-generation and "
                f"accept the 5.5x. Checked here rather than left to TRL, which raises only once the "
                f"weights are loaded. The switch defaults from {VLLM_COLOCATE_ENV}."
            )
        if self.smoke or not self.vllm_importance_sampling_correction:
            return
        row_logits_gib = (
            self.completion_budget * QWEN3_5_VOCAB_SIZE * BYTES_PER_FLOAT32 / BYTES_PER_GIB
        )
        if row_logits_gib > IS_CORRECTION_ROW_LOGITS_REFUSAL_GIB:
            raise ValueError(
                f"vLLM colocate with the importance-sampling correction ON recovers old "
                f"log-probabilities through a full-length forward pass whose fp32 logits cost "
                f"~{row_logits_gib:.1f} GiB per padded row at this completion budget "
                f"({self.completion_budget} tokens x {QWEN3_5_VOCAB_SIZE} vocab x 4 bytes). That "
                f"pass went out of memory at 30.31 GiB beside a resident engine on a 95 GiB card, "
                f"and it dies only AFTER paying for the first generation batch. Pass "
                f"--no-vllm-importance-sampling-correction (or set {VLLM_IS_CORRECTION_ENV}=0) and "
                f"note the estimator caveat, which both arms share and so cannot bias the contrast."
            )

    def _validate_retention_destination(self) -> None:
        """Refuse a real arm with nowhere to ship its checkpoints.

        Retention is a hard requirement of this experiment, not a convenience: the behavioural ladder
        is read at every rung, and the arms run on reclaimable spot capacity where a reclaim takes the
        instance's disk with it. Without a destination there is no retention AND no recovery -- the
        S3 restore this entry point performs has nothing to restore from -- so a reclaim at step 60
        loses sixty rungs and the run reports success on the relaunch that starts again at zero.

        Smoke runs are exempt because they are plumbing, and `--allow-hf-generation` is not an
        exemption: a slow arm needs retention more, not less.
        """
        if self.smoke or self.s3_dest:
            return
        raise ValueError(
            "this arm has no --s3-dest, so its checkpoints exist only on an instance that can be "
            "reclaimed at any moment, and the resume path would have nothing to restore from. Every "
            "checkpoint is a requirement of this experiment. Pass --s3-dest "
            f"<prefix>/{self.arm_tag}, or set GAMES_S3_DEST, or say --smoke if this is plumbing."
        )

    def _validate_s3_destination(self) -> None:
        """Refuse an S3 destination two arms could share, and say what to pass instead.

        ``aws s3 sync <dir> <prefix>`` puts the directory's *contents* under the prefix, so two arms
        with one destination write byte-identical keys -- ``checkpoint-7/adapter_model.safetensors``,
        ``completions/*.parquet``, ``run_config.json`` -- and the later sync silently replaces the
        earlier one. The file that records which arm a prefix holds is itself one of the colliding
        keys, so afterwards it is not even detectable. This repo has the incident logged.
        """
        if not self.s3_dest:
            return
        if not self.s3_dest.startswith("s3://"):
            raise ValueError(
                f"s3_dest must be an s3:// URI, got {self.s3_dest!r}. Checked at startup rather "
                f"than at the first checkpoint, so a typo costs no GPU time."
            )
        if self.s3_dest.rstrip("/").split("/")[-1] != self.arm_tag:
            raise ValueError(
                f"the S3 destination {self.s3_dest!r} does not end in this arm's own segment "
                f"{self.arm_tag!r}, so the other arm could be pointed at the same prefix and would "
                f"overwrite this one's checkpoints and rollout trace key for key. Pass "
                f"--s3-dest {self.s3_dest.rstrip('/')}/{self.arm_tag} instead."
            )


def assert_resume_provenance_matches(
    recorded: Mapping[str, object], *, checkpoint: str, config: RewardHackingTrainConfig
) -> dict[str, object]:
    """REFUSE a resume that would train the remaining steps with different code, and say which.

    The gap this closes was not a missing record but a missing comparison: ``git_sha`` and
    ``executed_estimator`` sit at the TOP LEVEL of ``run_config.json``, siblings of ``config`` and
    ``sizing_plan``, and only those two were ever handed to :func:`assert_resume_matches`.

    **The sentinel check is the load-bearing half, and it is why this cannot be one plain comparison.**
    ``games.provenance.git_sha`` never raises -- it records the reason in the returned string instead --
    so on a box with no git checkout and no ``GIT_SHA`` exported, BOTH sides read
    ``"unknown (git rev-parse failed: ...)"`` and compare equal. That is a check which prints a
    reassuring message, which is the failure mode this repository is organised around. It was also the
    live state: 74 artifacts under ``artifacts/`` carry exactly that string, because the box scripts
    wrote the sha to a marker file and never exported it. So an unattributable sha on either side is
    refused outright rather than compared.

    **The estimator half of the comparison is DERIVED here, because ``git_provenance`` carries only
    the git keys.** Handing its bare dict to :func:`assert_resume_matches` -- which subscripts
    ``current[field]`` -- raised ``KeyError`` on every resume that cleared the sha sentinel, and the
    suite stayed green because its fake provenance smuggled in an ``executed_estimator`` key the
    real function never produces (found by a kill-and-resume smoke, 2026-08-29). The derivation uses
    this launch's ``loss_type``/``use_liger_kernel`` against the RECORDED sizing plan's micro batch:
    no live plan exists yet at this point in ``_prepare_run`` (it is derived from a VRAM reading
    later), and micro-batch drift is ``RESUME_SIZING_FIELDS``' own refusal -- this comparison owns
    the aggregation axis, a flipped ``loss_type`` or ``use_liger_kernel`` between the checkpoint's
    steps and this launch.

    Returns the comparison for the run record, so a resumed launch says what it checked.
    """
    current = git_provenance()
    recorded_sha = recorded.get("git_sha")
    unattributable = {
        side: value
        for side, value in (("checkpoint", recorded_sha), ("this launch", current["git_sha"]))
        if not isinstance(value, str) or not value or value.startswith(UNKNOWN_SHA)
    }
    if unattributable:
        raise RuntimeError(
            f"refusing to resume {checkpoint}: {unattributable} names no commit, so whether the "
            f"remaining steps would be trained by the same code as the checkpoint's existing steps "
            f"cannot be answered -- and comparing two unattributable shas would pass while answering "
            f"nothing. Export GIT_SHA (the box scripts write it to /home/ubuntu/repo/GIT_SHA), or "
            f"start a fresh run in a new --output-dir rather than continuing this one blind."
        )
    recorded_plan = recorded.get("sizing_plan")
    micro_batch_size = (
        cast("Mapping[str, object]", recorded_plan).get("micro_batch_size")
        if isinstance(recorded_plan, dict)
        else None
    )
    if not isinstance(micro_batch_size, int):
        # TRY004 wants TypeError on an isinstance guard, but this validates a deserialized artifact
        # rather than a caller's argument, and every refusal in this gate is a RuntimeError.
        raise RuntimeError(  # noqa: TRY004
            f"refusing to resume {checkpoint}: the launch record carries no readable "
            f"sizing_plan.micro_batch_size (got {micro_batch_size!r}), so the estimator this launch "
            f"would execute cannot be derived for comparison against the checkpoint's recorded "
            f"executed_estimator. A record without its sizing plan was not written by this entry "
            f"point; start a fresh run in a new --output-dir rather than continuing it blind."
        )
    current_with_estimator: dict[str, object] = {
        **current,
        "executed_estimator": executed_estimator(
            config.loss_type,
            use_liger_kernel=config.use_liger_kernel,
            per_device_train_batch_size=micro_batch_size,
        ),
    }
    assert_resume_matches(
        recorded=recorded,
        current=current_with_estimator,
        fields=RESUME_PROVENANCE_FIELDS,
        checkpoint=checkpoint,
        consequence=(
            "The remaining steps would be trained by different code than the steps already in this "
            "checkpoint, under one set of step numbers and one rollout trace, with nothing in the "
            "artifacts saying where the change fell. The parser fix is the concrete case: an arm whose "
            "first N steps scored submissions by one rule and whose rest scored them by another is not "
            "an arm."
        ),
    )
    if recorded.get("git_tree_dirty") or current["git_tree_dirty"]:
        logger.warning(
            "resuming across a DIRTY tree, so the matching sha is weaker evidence than it looks: "
            "uncommitted edits are invisible to it, %s",
            f"checkpoint_dirty={recorded.get('git_tree_dirty')!r} "
            f"launch_dirty={current['git_tree_dirty']!r}",
        )
    return {"compared": list(RESUME_PROVENANCE_FIELDS), **current_with_estimator}


def shrink_for_smoke(config: RewardHackingTrainConfig) -> RewardHackingTrainConfig:
    """Return the same config with every size knob reduced to smoke scale."""
    return replace(config, smoke=True, **SMOKE_OVERRIDES)  # pyright: ignore[reportArgumentType]


def default_output_dir(
    config: RewardHackingTrainConfig, *, timestamp: datetime, smoke: bool = False
) -> str:
    """Build the per-run artifact directory, one per arm, model and launch."""
    prefix = "smoke-" if smoke else ""
    return f"{RUN_ROOT}/{prefix}{config.arm_tag}-{timestamp.strftime('%Y%m%d-%H%M%S')}"


@dataclass(frozen=True)
class PreparedRun:
    """Everything one launch settled before any weights loaded, including its record on disk."""

    config: RewardHackingTrainConfig
    plan: SizingPlan
    dataset: Dataset
    tokenizer: PreTrainedTokenizerBase
    lora_targets: dict[str, object]
    derived: dict[str, object]
    dtype: torch.dtype
    device: dict[str, object]
    resume_checkpoint: str | None

    @property
    def output_dir(self) -> str:
        """The run directory, resolved by the time a run is prepared."""
        return cast("str", self.config.output_dir)

    @property
    def prefilled_think(self) -> bool:
        """Whether this tokenizer's template opens ``<think>`` inside the prompt."""
        return cast("bool", self.derived["prefilled_think"])


def _announce_launch(config: RewardHackingTrainConfig) -> int:
    """Log what this run is, refuse a multi-process launch, and return the process count."""
    logger.info(
        "training arm, %s",
        f"{config.arm=} split={config.split!r} model_id={config.model_id!r} "
        f"output_dir={config.output_dir!r} completion_budget={config.completion_budget} "
        f"save_steps={config.save_steps} save_total_limit={config.save_total_limit} "
        f"vllm_colocate={config.vllm_colocate}",
    )
    # From the registry rather than per-arm branches: a branch chain fails open, so a newly
    # registered arm would launch with no statement of what it is (ARM_RATIONALE's completeness
    # check makes the missing entry an import error instead).
    logger.info("%s", ARM_RATIONALE[config.arm])
    if not config.thinking:
        for line in (
            "=" * 78,
            "PLUMBING RUN -- THINKING IS OFF. The path executes and reward may move, but this is",
            "NOT a behavioural result: the chain of thought is what the interpretability arm reads",
            "and what a grader-gaming rationale would appear in, and there is none here.",
            "=" * 78,
        ):
            logger.warning(line)
    world_size = detected_world_size(os.environ)
    assert_single_process(world_size, source="launch environment")
    return world_size


def _derive_plan(
    config: RewardHackingTrainConfig,
    *,
    device: dict[str, object],
    dtype: torch.dtype,
    param_count: int,
) -> SizingPlan:
    """Size the batch from the VRAM this process actually found, never from a constant."""
    engine_reserved_gib = (
        colocate_reserved_gib(
            total_vram_gib=cast("float", device["total_vram_gib"]),
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        )
        if config.vllm_colocate
        else 0.0
    )
    # The chunk width and the log-only flag come from the TRAINER CLASS rather than from this config,
    # because `_build_trainer` constructs `PaddingTrimmedGRPOTrainer` without either instrument
    # keyword: this thread has no launch knob for either, so what the banner must describe is what the
    # class will do. Passed explicitly, so basedpyright checks the banner's contract against this
    # config instead of a cast hiding fields it does not have.
    chunk_tokens = min(PaddingTrimmedGRPOTrainer.old_logps_chunk_tokens, config.completion_budget)
    log_colocate_settings(
        config,
        engine_reserved_gib=engine_reserved_gib,
        old_logps_chunk_tokens=chunk_tokens,
        old_logps_peak_gib=old_logps_pass_peak_gib(
            chunk_tokens=chunk_tokens, rows=config.micro_batch_size or 1
        ),
        importance_sampling_log_only=PaddingTrimmedGRPOTrainer.importance_sampling_log_only,
    )
    plan = plan_sizing(
        num_generations=config.num_generations,
        prompts_per_step=config.prompts_per_step,
        micro_batch_size=config.micro_batch_size,
        max_prompt_tokens=config.max_prompt_tokens,
        max_completion_tokens=config.completion_budget,
        cost=checkpoint_sequence_cost(config.model_id),
        free_vram_gib=cast("float", device["free_vram_gib_at_start"]),
        weights_gib=param_count * dtype.itemsize / BYTES_PER_GIB,
        usable_fraction=config.vram_usable_fraction,
        autosize=config.autosize,
        label=f"{config.model_id} on {device['device_name']}",
        engine_reserved_gib=engine_reserved_gib,
        generation_schedules_own_batch=config.vllm_colocate,
    )
    logger.info("sizing plan: %s", plan.reason)
    return plan


def _build_grpo_config(
    config: RewardHackingTrainConfig, plan: SizingPlan, *, dtype: torch.dtype
) -> GRPOConfig:
    """Translate the arm's config and its sizing plan into TRL's arguments.

    Deliberately a sibling of ``games.train._build_grpo_config`` rather than a call into it: that one
    is typed over the games arm config and reads its ``arm`` registry, and the science-bearing fields
    the two must agree on are asserted equal by ``test_rh_train_config``, so a drift between them is
    a test failure rather than a silent divergence.
    """
    vllm_arguments: dict[str, Any] = (
        {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "vllm_max_model_length": config.vllm_max_model_length,
            "vllm_importance_sampling_correction": config.vllm_importance_sampling_correction,
        }
        if config.vllm_colocate
        else {}
    )
    return GRPOConfig(
        output_dir=cast("str", config.output_dir),
        run_name=config.arm_tag,
        seed=config.seed,
        tf32=is_torch_tf32_available(),
        bf16=(dtype == torch.bfloat16),
        per_device_train_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        num_generations=plan.num_generations,
        max_completion_length=config.completion_budget,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_steps=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=config.logging_steps,
        # The per-completion rollout trace, written to <output_dir>/completions/*.parquet. It carries
        # every log_extra column the reward adds, which is what makes the hidden check recoverable
        # afterwards instead of a reason to re-run.
        log_completions=True,
        # Zero, not TRL's default of None: None renders EVERY completion of the step as a rich table on
        # stdout, roughly 300,000 lines and 27 MB per step at 64 completions, which cost 8-12 s of the
        # post-step phase and buried the ~650 timestamped lines of a 70-step log under 17.8 million
        # others. The parquet above is written in the same block regardless of this count, so the
        # trace loses nothing. Pinned by `test_rh_train_config`, which also checks that TRL still
        # treats 0 as "render nothing" rather than "render all".
        num_completions_to_print=0,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # TRL evaluates by computing the GRPO surrogate loss over eval prompts, which segfaults
        # inside Liger's fused loss. The held-out battery reads checkpoints instead.
        eval_strategy="no",
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        beta=config.beta,
        epsilon=config.epsilon,
        scale_rewards=config.scale_rewards,
        loss_type=config.loss_type,
        # A completion that ran out of tokens mid-thought keeps its gradient rather than being masked
        # away: the pressure toward shorter reasoning is itself something to watch.
        mask_truncated_completions=False,
        use_liger_kernel=config.use_liger_kernel,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",
        disable_tqdm=True,
        model_init_kwargs={
            # `dtype`, not `torch_dtype`: TRL ignores the latter in favour of its own key, and the
            # model would load in float32 while bf16=True told the trainer to autocast.
            "dtype": dtype,
            "trust_remote_code": True,
        },
        **vllm_arguments,
    )


def restore_run_directory(config: RewardHackingTrainConfig) -> dict[str, object]:
    """Pull this arm's run directory back from S3 before training, so a reclaim is not a restart.

    The missing half of recoverability. Every checkpoint is uploaded on save, but nothing in this
    repository ever downloaded one, and a spot reclaim takes the instance and its disk with it -- so
    a resubmitted job on a fresh box found an empty output directory, resumed from nothing, and began
    again at step 0 while reporting success. ``--resume-from-checkpoint latest`` then means "latest of
    what is on this disk", which after a reclaim is nothing at all.

    The restore runs ONLY for ``--resume-from-checkpoint latest`` with an S3 destination set. Two
    other launches skip it, and the skip is recorded in the launch record rather than logged: a run
    with no resume asked for (pulling a bucket into a fresh run directory would silently continue an
    unrelated run's steps under a new launch record), and -- easy to miss -- a resume by EXPLICIT
    checkpoint path, which gets no S3 restore either, so on a fresh box that path must already be on
    disk or ``resolve_resume_checkpoint`` raises "is not a directory". (Until 2026-08-24 this
    docstring said the skip happened "loudly, when no resume was asked for": no log line exists, and
    the explicit-path case skipped too. Whether either of those should change is an open behaviour
    question -- docs/scratch/test-prose-contradiction-ledger-2026-08-24.md, T5.)
    """
    output_dir = Path(cast("str", config.output_dir))
    if not config.s3_dest or config.resume_from_checkpoint != RESUME_LATEST:
        return {
            "restored": False,
            "reason": (
                f"restore runs only when --resume-from-checkpoint is {RESUME_LATEST!r} and an s3 "
                f"destination is set; got resume_from_checkpoint="
                f"{config.resume_from_checkpoint!r} and "
                f"{'an' if config.s3_dest else 'no'} s3 destination"
            ),
        }
    outcome = restore_directory(config.s3_dest, output_dir)
    restored = sorted(path.name for path in output_dir.glob("checkpoint-*"))
    record: dict[str, object] = {
        "restored": True,
        "s3_dest": config.s3_dest,
        "returncode": outcome.returncode,
        "checkpoints_present": restored,
    }
    logger.info("restored the run directory from s3 before training, %s", record)
    return record


def _prepare_run(
    config: RewardHackingTrainConfig, *, kernel_bridge: dict[str, object] | None
) -> PreparedRun | CompletedRun:
    """Settle everything a run needs before any weights load, and write its launch record.

    The order is the content. The jail is proved usable first, because every reward call depends on
    it and a dead jail turns the whole run into zeros. The S3 restore comes before the resume is
    resolved, because on a fresh box after a reclaim there is nothing on disk to resume from until it
    has run. The resume is resolved before the sizing plan, because the plan is derived from a live
    free-VRAM reading and a re-plan would silently change the group size the checkpoint's existing
    steps were trained under.

    Returns a `CompletedRun` instead, before the card is read, the sizing plan derived or the launch
    record written, when the resume landed on a checkpoint already at `max_steps`: the relaunch of a
    finished arm is the documented recovery procedure re-run one time too many (a stage runner that
    comes back after a reboot re-runs finished stages too), and left to the trainer it is not a no-op
    but a zero-step run that rewrites the finished run's trace, memory log and summary
    (`games.train.completed_run` carries the incident).
    """
    if config.output_dir is None:
        config = replace(
            config,
            output_dir=default_output_dir(
                config, timestamp=datetime.now(tz=UTC), smoke=config.smoke
            ),
        )
    output_dir = cast("str", config.output_dir)
    world_size = _announce_launch(config)
    jail = assert_jail_usable(timeout_seconds=config.grader_timeout_seconds)
    Path(config.grader_scratch_root).mkdir(parents=True, exist_ok=True)
    restore = restore_run_directory(config)

    resume = resolve_complete_resume_checkpoint(config.resume_from_checkpoint, output_dir)
    resume_checkpoint = resume.checkpoint
    recorded_launch: dict[str, object] | None = None
    resume_provenance: dict[str, object] | None = None
    if resume_checkpoint is not None:
        recorded_launch = read_recorded_launch(output_dir, checkpoint=resume_checkpoint)
        assert_resume_matches(
            recorded=cast("Mapping[str, object]", recorded_launch["config"]),
            current=asdict(config),
            fields=RESUME_IDENTITY_FIELDS,
            checkpoint=resume_checkpoint,
            consequence=(
                "Resuming would keep training that checkpoint's adapter, optimizer and step counter "
                "under a different question, and its rollout trace would carry both experiments "
                "under one set of step numbers."
            ),
        )
        resume_provenance = assert_resume_provenance_matches(
            recorded_launch, checkpoint=resume_checkpoint, config=config
        )
        # After the identity and provenance checks on purpose: a relaunch under the wrong arm or
        # under different code is refused as such, not waved through as "already complete".
        completed = completed_run(
            resume_checkpoint, output_dir=output_dir, max_steps=config.max_steps
        )
        if completed is not None:
            return completed

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = describe_device()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(
        "landed on %s (%.1f GiB total, %.1f GiB free), %s",
        device["device_name"],
        device["total_vram_gib"],
        device["free_vram_gib_at_start"],
        f"{dtype=}",
    )

    tokenizer, template_facts = resolve_tokenizer(config.model_id, thinking=config.thinking)
    partition = load_partition(Path(config.partition_path))
    chat_template_kwargs = cast("dict[str, str]", template_facts["chat_template_kwargs"])
    # The screen's resolver, shared rather than mirrored. The two used to apply the prompt budget and
    # `--max-prompts` in opposite orders, so a bounded run trained on fewer prompts than asked whenever
    # the over-length problem landed in its window and the recorded prompt_budget then described the
    # subset rather than the corpus.
    rows, prompt_budget, bounded_subset = resolve_arm_rows(
        config.arm,
        partition,
        tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        enable_thinking=config.thinking,
        chat_template_kwargs=chat_template_kwargs,
        max_prompts=config.max_prompts,
        seed=config.seed,
        # Explicit rather than the parameter default, so TRAINING_GRADER_EXPOSURE is load-bearing:
        # the screen gate compares recorded verdicts against this constant, and a comparison against
        # a value training does not actually pass would be a check that proves nothing.
        exposure=TRAINING_GRADER_EXPOSURE,
    )
    dataset = build_dataset(
        rows,
        tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        enable_thinking=config.thinking,
        chat_template_kwargs=chat_template_kwargs,
    )
    logger.info("dataset built, %s", f"n_prompts={len(dataset)} n_rows_in={len(rows)}")

    lora_targets = discover_lora_targets(config.model_id)
    logger.info("LoRA targets: %s", lora_targets["module_counts"])
    kernel_paths = log_deltanet_kernel_paths(
        expected_linear_attention_layers=cast(
            "int", lora_targets["expected_linear_attention_layers"]
        )
    )
    param_count = count_meta_parameters(config.model_id)
    plan = _derive_plan(config, device=device, dtype=dtype, param_count=param_count)
    assert_dataset_fills_a_step(len(dataset), plan)
    if recorded_launch is not None:
        assert_resume_matches(
            recorded=cast("Mapping[str, object]", recorded_launch["sizing_plan"]),
            current=asdict(plan),
            fields=RESUME_SIZING_FIELDS,
            checkpoint=cast("str", resume_checkpoint),
            consequence=(
                "The sizing plan comes from a live free-VRAM reading, so this is usually another "
                "process holding memory on a shared card. num_generations is the GRPO advantage "
                "baseline, so changing it mid-run changes the experiment. Free the memory, or pass "
                "--no-autosize with the recorded values."
            ),
        )

    derived: dict[str, object] = {
        "thinking": config.thinking,
        "plumbing_thinking_off": not config.thinking,
        **template_facts,
        "dtype": str(dtype),
        "lora_targets": lora_targets,
        "deltanet_kernel_paths": kernel_paths,
        "deltanet_kernel_bridge": kernel_bridge,
        "meta_parameter_count": param_count,
        "n_prompts": len(dataset),
        "n_rows_before_dataset_build": len(rows),
        "prompt_budget": prompt_budget.to_json_dict(),
        # The loud record that this arm trained on fewer prompts than the corpus holds, carrying its
        # own do-not-report-as-full-corpus warning. Previously only a log line, so a bounded arm's
        # artifacts said nothing about the bound and its rates read as the full-corpus figure.
        "bounded_subset": bounded_subset.to_json_dict() if bounded_subset else None,
        # Which rule decided what counted as a submission. Gradable rates from different parsers are
        # not comparable, so an artifact that does not name its parser cannot be read against another:
        # the shipped rule moved this corpus by SOLUTION_PARSER_GRADABLE_SHIFT on byte-identical
        # generations. Both fields, so a reader has the size of the difference without a code lookup.
        "solution_parser": SOLUTION_PARSER,
        "solution_parser_gradable_shift": SOLUTION_PARSER_GRADABLE_SHIFT,
        "completion_budget": config.completion_budget,
        "coding_termination_stats": (
            MEASURED_CODING_TERMINATION_STATS_BY_MODEL[config.model_id].to_json_dict()
            if config.model_id in MEASURED_CODING_TERMINATION_STATS_BY_MODEL
            else None
        ),
        "partition": describe_partition(partition),
        "grader": config.grader.to_json_dict(),
        "jail_preflight": jail,
        "s3_restore": restore,
        "world_size": world_size,
        "resumed_from_checkpoint": resume_checkpoint,
        # Torn checkpoints the resolver stepped past to reach `resumed_from_checkpoint`. Non-empty
        # means those steps ran twice, and this record is what attributes their second run.
        "incomplete_checkpoints_set_aside": list(resume.set_aside),
        # What the resume actually verified about the code, so a reader of a resumed run's record
        # does not have to take it on trust that the comparison happened.
        "resume_provenance_check": resume_provenance,
        "vllm_max_model_length": config.vllm_max_model_length if config.vllm_colocate else None,
    }
    record_name = (
        RUN_CONFIG_FILENAME
        if resume_checkpoint is None
        else RESUMED_RUN_CONFIG_TEMPLATE.format(checkpoint=Path(resume_checkpoint).name)
    )
    write_json(
        Path(output_dir, record_name),
        {
            "arm": config.arm,
            "split": config.split,
            # As games records arm_notes: the run's own artifact says what its arm IS, so a reader
            # of a bucket prefix never has to resolve the arm name against the right code version.
            "arm_rationale": ARM_RATIONALE[config.arm],
            "executed_estimator": executed_estimator(
                config.loss_type,
                use_liger_kernel=config.use_liger_kernel,
                per_device_train_batch_size=plan.micro_batch_size,
            ),
            "config": asdict(config),
            "sizing_plan": asdict(plan),
            "device": device,
            "derived": derived,
            **git_provenance(),
            "started_at": datetime.now(tz=UTC).isoformat(),
        },
    )
    return PreparedRun(
        config=config,
        plan=plan,
        dataset=dataset,
        tokenizer=tokenizer,
        lora_targets=lora_targets,
        derived=derived,
        dtype=dtype,
        device=device,
        resume_checkpoint=resume_checkpoint,
    )


def _build_trainer(prepared: PreparedRun) -> GRPOTrainer:
    """Construct the trainer this plan calls for, and stamp the tokenizer's ids onto the model.

    The class is the games one, not a subclass of it: `PaddingTrimmedGRPOTrainer` carries both
    behaviours this trainer needs -- the per-micro-batch padding trim at TRL's `_prepare_inputs`
    seam, and the `log` override that refuses to write an empty completions buffer over a step
    whose parquet already holds rows -- so a subclass here could only re-state one of them, and
    this module once did exactly that (`TraceGuardedGRPOTrainer`, a second copy of the guard).
    `test_rh_train_config` pins that no trainer subclass is defined here at all.
    """
    config = prepared.config
    reward = make_visible_grader_reward(
        prepared.plan.num_generations,
        prefilled_think=prepared.prefilled_think,
        grader=config.grader,
    )
    trainer = PaddingTrimmedGRPOTrainer(
        model=config.model_id,
        reward_funcs=reward,  # pyright: ignore[reportArgumentType]
        args=_build_grpo_config(config, prepared.plan, dtype=prepared.dtype),
        train_dataset=prepared.dataset,
        # A bare tokenizer rather than TRL's default AutoProcessor: every Qwen3.5 checkpoint maps to
        # a ProcessorMixin, which would switch TRL onto its vision-language path.
        processing_class=prepared.tokenizer,
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # From discovery, never a hand-written q/k/v/o list: three of every four layers in this
            # family are Gated DeltaNet linear attention with different projection names, and a
            # literal list silently leaves 24 of 32 layers frozen.
            target_modules=cast("list[str]", prepared.lora_targets["target_modules"]),
        ),
        callbacks=build_callbacks(
            logging_steps=config.logging_steps,
            output_dir=cast("str", config.output_dir),
            s3_dest=config.s3_dest,
        ),
    )
    model = trainer.model
    model.config.pad_token_id = prepared.tokenizer.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.config.eos_token_id = prepared.tokenizer.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.generation_config.pad_token_id = prepared.tokenizer.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.generation_config.eos_token_id = prepared.tokenizer.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    return trainer


def _summarize_run(
    prepared: PreparedRun,
    trainer: GRPOTrainer,
    *,
    wall_seconds: float,
    checks: dict[str, object],
) -> tuple[dict[str, object], list[str], TraceFilesVerdict]:
    """Read every claimed metric back out, write the summary, and report what never arrived.

    The rollout trace is read from disk here, not inferred: `checks` carries the config-arithmetic
    verdict from build time, and `rollout_trace_complete` is true only when the files agree with it.
    Before this the summary reported that arithmetic as the trace verdict, which is how a zero-row
    parquet over a finished step stayed green.
    """
    config = prepared.config
    metrics, missing = read_back_metrics(
        trainer,
        required=required_reward_metrics(hidden_check=config.hidden_check),
        # The mode-independent pair alone: this reward prices a parse failure at its own constant and
        # never logs the games guard's series, so no parse-penalty mode pins anything here.
        constant_by_construction=CONSTANT_BY_CONSTRUCTION_METRICS,
    )
    trace_files = verify_trace_files(
        Path(prepared.output_dir),
        steps=trainer.state.global_step,
        expected_rows=prepared.plan.episodes_per_step,
        logging_steps=config.logging_steps,
    )
    summary: dict[str, object] = {
        "arm": config.arm,
        "split": config.split,
        "model_id": config.model_id,
        "output_dir": prepared.output_dir,
        **git_provenance(),
        "smoke": config.smoke,
        "steps_completed": trainer.state.global_step,
        "max_steps": config.max_steps,
        "wall_clock_seconds": wall_seconds,
        "episodes_per_step": prepared.plan.episodes_per_step,
        "num_generations": prepared.plan.num_generations,
        "n_prompts": len(prepared.dataset),
        "prefilled_think": prepared.prefilled_think,
        "plumbing_thinking_off": not config.thinking,
        "resumed_from_checkpoint": prepared.resume_checkpoint,
        "sizing_clamped": prepared.plan.clamped,
        "checkpoints_written": sorted(
            path.name for path in Path(prepared.output_dir).glob("checkpoint-*")
        ),
        "missing_metrics": missing,
        **checks,
        "rollout_trace_config_complete": checks["rollout_trace_complete"],
        "rollout_trace_complete": bool(checks["rollout_trace_complete"]) and trace_files.complete,
        "rollout_trace_files": asdict(trace_files),
        **metrics,
        **peak_memory_gib(prepared.output_dir),
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    write_json(Path(prepared.output_dir, TRAIN_SUMMARY_FILENAME), summary)
    logger.info("RESULT %s", json.dumps(summary, default=str))
    return summary, missing, trace_files


def train_arm(
    config: RewardHackingTrainConfig, *, kernel_bridge: dict[str, object] | None = None
) -> GRPOTrainer | None:
    """Train one arm end to end, and persist enough that a gap is a re-analysis not a re-run.

    Returns None, having written nothing, when the resume landed on a run that already reached
    `max_steps`: the relaunch of a finished arm is a no-op that says so, not a zero-step training run
    that rewrites the finished run's artifacts (`games.train.completed_run`).
    """
    prepared = _prepare_run(config, kernel_bridge=kernel_bridge)
    if isinstance(prepared, CompletedRun):
        logger.info(
            "ALREADY COMPLETE: %s holds step %d of max_steps=%d and %s is on disk, so there is "
            "nothing to train. No trainer was built and no artifact was touched.",
            prepared.checkpoint,
            prepared.step,
            prepared.max_steps,
            prepared.summary_path,
        )
        return None
    trainer = _build_trainer(prepared)
    checks = check_built_trainer(
        trainer,
        expected_linear_attention_layers=cast(
            "int", prepared.lora_targets["expected_linear_attention_layers"]
        ),
        meta_parameter_count=cast("int", prepared.derived["meta_parameter_count"]),
        episodes_per_step=prepared.plan.episodes_per_step,
        gradient_accumulation_steps=prepared.plan.gradient_accumulation_steps,
    )

    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    trainer.train(resume_from_checkpoint=prepared.resume_checkpoint)
    wall_seconds = time.perf_counter() - wall_start
    trainer.save_model()
    trainer.save_state()

    _, missing, trace_files = _summarize_run(
        prepared, trainer, wall_seconds=wall_seconds, checks=checks
    )
    if config.s3_dest:
        # The callback's on_train_end fires inside trainer.train(), BEFORE save_model, save_state and
        # the summary write above, so without this the bucket's copy of a finished run is missing the
        # summary and the final trainer state.
        sync_directory(Path(prepared.output_dir), config.s3_dest)
    if missing:
        raise RuntimeError(
            f"expected metrics never reached trainer_state, {missing=}. The reward function's "
            f"log_metric calls are the only route for these, and a metric nobody reads back is not "
            f"recorded."
        )
    if not trace_files.complete:
        raise RuntimeError(
            f"the rollout trace under {trace_files.checked_dir} has gaps: missing steps "
            f"{list(trace_files.missing_steps)}, empty files at steps {list(trace_files.empty_steps)}, "
            f"wrong row counts (step, rows) {[list(pair) for pair in trace_files.wrong_row_count_steps]} "
            f"against {trace_files.expected_rows_per_step} per step. The summary records this as "
            f"rollout_trace_complete=false; the per-completion trace is the run's raw material (the "
            f"hidden check and the behavioural ladder are both read off it), so a run without all of "
            f"it is not a finished measurement."
        )
    return trainer


def _config_from_namespace(args: argparse.Namespace) -> RewardHackingTrainConfig:
    """Resolve the defaults that depend on another flag, then build the config."""
    values = vars(args)
    smoke = bool(values.pop("smoke"))
    model_id = values.pop("model_id") or (SMOKE_MODEL_ID if smoke else DEFAULT_MODEL_ID)
    config = RewardHackingTrainConfig(model_id=model_id, smoke=smoke, **values)
    return shrink_for_smoke(config) if smoke else config


def _add_optimisation_args(parser: argparse.ArgumentParser) -> None:
    """Add the estimator, LoRA and schedule flags: what the optimiser does with the rewards."""
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=TRAINING_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TRAINING_TOP_P)
    parser.add_argument("--top-k", type=int, default=TRAINING_TOP_K)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=GRPO_EPSILON)
    parser.add_argument("--loss-type", default=GRPO_LOSS_TYPE, choices=GRPO_LOSS_TYPES)
    parser.add_argument(
        "--scale-rewards", default=GRPO_SCALE_REWARDS, choices=GRPO_SCALE_REWARDS_MODES
    )
    parser.add_argument("--acknowledge-liger-estimator-mismatch", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--no-liger", dest="use_liger_kernel", action="store_false")
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1,
        help="every step, and a coarser cadence is refused for a real arm: see _validate_retention",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=0,
        help="0 disables checkpoint rotation outright, which decision 8 of the plan requires",
    )
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--allow-short-completions", action="store_true")
    parser.add_argument("--max-prompts", type=int, default=None)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags describing the BOX rather than the experiment: sizing and the engine."""
    parser.add_argument("--no-autosize", dest="autosize", action="store_false")
    parser.add_argument("--vram-usable-fraction", type=float, default=DEFAULT_VRAM_USABLE_FRACTION)
    parser.add_argument(
        "--vllm-colocate",
        action=argparse.BooleanOptionalAction,
        default=colocate_requested(),
        help=(
            f"generate rollouts through a colocated vLLM engine rather than transformers.generate, "
            f"measured at 13 against 72 minutes per step. Defaults from {VLLM_COLOCATE_ENV}, else "
            f"to ON wherever the engine is installed."
        ),
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=colocate_gpu_fraction(),
        help=(
            f"the engine's share of TOTAL card VRAM, held for the whole run and subtracted from "
            f"what the sizing plan may spend. Defaults from {VLLM_GPU_FRACTION_ENV}, else the "
            f"measured {VLLM_COLOCATE_GPU_FRACTION}."
        ),
    )
    parser.add_argument(
        "--no-vllm-importance-sampling-correction",
        dest="vllm_importance_sampling_correction",
        action="store_false",
        default=colocate_importance_sampling_correction(),
        help=(
            f"drop TRL's correction for the mismatch between vLLM-sampled tokens and this model's "
            f"own log-probabilities. Required at this completion budget, whose upcast logits row "
            f"went out of memory beside a resident engine. Defaults from {VLLM_IS_CORRECTION_ENV}."
        ),
    )
    parser.add_argument(
        "--allow-hf-generation",
        action="store_true",
        help="state deliberately that this run accepts the measured 5.5x of transformers.generate",
    )


def _add_grading_and_artifact_args(parser: argparse.ArgumentParser) -> None:
    """Add the grading knobs and where a run's artifacts go, including its S3 destination."""
    parser.add_argument("--grader-scratch-root", default=DEFAULT_GRADER_SCRATCH_ROOT)
    parser.add_argument(
        "--grader-timeout-seconds", type=int, default=DEFAULT_GRADER_TIMEOUT_SECONDS
    )
    parser.add_argument("--grader-workers", type=int, default=DEFAULT_GRADER_WORKERS)
    parser.add_argument(
        "--hidden-check",
        action="store_true",
        help=(
            "run the hidden true check on every training episode as instrumentation. Off by "
            "default: its leash is 150 s and the same verdicts are recoverable from the retained "
            "rollout trace with reward_hacking.train_trace_score."
        ),
    )
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument("--resume-from-checkpoint", dest="resume_from_checkpoint", default="")
    # From the environment for the same reason games does it: cloud/submit_job.py sets GAMES_S3_DEST
    # on the container and cloud/entrypoint.sh reads the same variable for its exit-trap sync, so a
    # flag-only default would leave the in-run liveness sync off on every Batch job.
    parser.add_argument("--s3-dest", default=os.environ.get("GAMES_S3_DEST", ""))


def _parse_args(argv: Sequence[str] | None = None) -> RewardHackingTrainConfig:
    """Parse one arm's training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(TRAINABLE_ARMS))
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument("--partition", dest="partition_path", default=str(DEFAULT_PARTITION_PATH))
    parser.add_argument("--smoke", action="store_true")
    _add_optimisation_args(parser)
    _add_generation_args(parser)
    _add_grading_and_artifact_args(parser)
    args = parser.parse_args(argv)

    config = _config_from_namespace(args)
    if config.output_dir is None:
        config = replace(
            config,
            output_dir=default_output_dir(
                config, timestamp=datetime.now(tz=UTC), smoke=config.smoke
            ),
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Train the arm named on the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    default_cuda_allocator_config()
    # First, before discover_lora_targets reaches for the Qwen3.5 modeling module: transformers binds
    # each Gated DeltaNet kernel at that module's import time, so a bridge applied afterwards
    # silently leaves decode on the pure-torch loop.
    kernel_bridge = dict(bridge_decode_kernel())
    logger.info("deltanet decode bridge: %s", kernel_bridge)
    call_site = assert_bridged_kernel_matches_call_site()
    kernel_bridge["decode_call_site"] = {
        "positional_count": call_site.positional_count,
        "keyword_names": sorted(call_site.keyword_names),
    }
    train_arm(_parse_args(argv), kernel_bridge=kernel_bridge)


if __name__ == "__main__":
    main()
