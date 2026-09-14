"""GRPO training for one matrix-game arm: registry, VRAM-derived sizing, and the run itself.

An *arm* is one training run. Arms differ only in which game the prompts describe and how the
completions are graded, so the whole experiment is `ARMS` plus one code path: the contrast
between `twin-pd-group` and `twin-pd-self` is identical prompts under two grading rules, and
reading anything off that contrast requires that nothing else differed.

Three things here exist because they were silent failures elsewhere in this repo:

*   Sizing is derived from the VRAM the process actually found, never from a constant. A config
    that assumes this box's 24 GiB L4 cannot take whatever card is free, and a job that quietly
    OOMs at step 40 costs more than the arithmetic does.
*   `prefilled_think` is measured by rendering a chat with the run's own tokenizer, not looked up
    by model name. Qwen3.5/3.8 templates open `<think>` inside the prompt so the completion holds
    only the closing tag, while Qwen3-0.6B emits both; a parser on the wrong side of that scores
    every rollout as a parse failure, and the reward would still look like a number.
*   Every metric this run claims to record is read back out of trainer state afterwards, and a
    missing one raises. A callback in this repo once wrote metrics into a dict `Trainer.log` had
    already copied, so verifier accuracy was never recorded for the whole history of the repo.

    uv run python -m games.train --arm twin-pd-group --corpus artifacts/games/select/twin-pd.jsonl

Smoke is the same function with small numbers (`--smoke`), never a second code path, because the
paper cuts live in the seams rather than the components.

Rollouts come from a vLLM engine colocated in this same process, always -- ~13.5 minutes per step
against ~72 through `transformers.generate` at the production shape, and since 2026-08-26 the slow
path is not selectable at all: `games.generation.assert_vllm_rollouts` refuses an environment that
asks, and a `StepPaceGuardCallback` kills any run whose realized pace says generation took a slow
path anyway. The engine shares one card with the trainer, so what it costs is logged as loudly as
what it saves: a third of the VRAM, taken off the top of the sizing plan, and an estimator caveat
whenever TRL's importance-sampling correction will not fit beside the engine.

Two instruments sit on the trainer for that mismatch and for what it does to the policy, both
default-off or default-invisible and neither changing a gradient: `sampled_surprisal` reads an
entropy estimate off the log-probabilities vLLM already returns, and
`InstrumentedGRPOTrainer._get_per_token_logps_and_entropies` computes the correction's
old-log-probability pass over token slices so its memory peak is a chunk rather than a whole row.
`--vllm-importance-sampling-log-only` then records the mismatch while weighting the gradient by
exactly one, and `--vllm-importance-sampling-mode` selects which of TRL's four corrections runs.

One knob does change what a step trains on, and labels itself as a treatment for it:
`--dynamic-sampling-oversample N` generates N times the step's prompt groups and hands the optimizer
the ones whose rewards disagreed, because a group whose completions all scored the same has zero
advantages and costs a full forward and backward for no gradient. Everything generated is still
scored, still logged and still in the rollout trace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend
from transformers import AutoConfig
from transformers.utils.import_utils import is_torch_tf32_available
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]
from trl.trainer.utils import selective_log_softmax

from games.arms import (
    ARMS,
    CORPUS_PARTITION_COLUMN,
    PAYOFF_VARIANT_COLUMN,
    GameArm,
    arm_game_ids,
)
from games.checkpoint_retention import (
    ADAPTER_WEIGHT_FILENAMES as RETENTION_ADAPTER_WEIGHT_FILENAMES,
)
from games.checkpoint_retention import (
    REQUIRED_CHECKPOINT_FILES as RETENTION_REQUIRED_CHECKPOINT_FILES,
)
from games.checkpoint_retention import (
    CheckpointRetentionCallback,
    missing_checkpoint_files,
    validate_retention_settings,
    write_retention_manifest,
)
from games.dataset import build_game_dataset
from games.deltanet_kernels import assert_bridged_kernel_matches_call_site, bridge_decode_kernel
from games.generation import (
    TRAINING_TEMPERATURE,
    TRAINING_TOP_K,
    TRAINING_TOP_P,
    VLLM_COLOCATE_GPU_FRACTION,
    VLLM_GPU_FRACTION_ENV,
    VLLM_IS_CORRECTION_ENV,
    assert_vllm_rollouts,
    colocate_gpu_fraction,
    colocate_importance_sampling_correction,
)
from games.lora import (
    ADAPTER_CONFIG_FILENAME,
    ERROR_EXAMPLE_COUNT,
    adapter_config_identity,
    assert_adapter_matches_base,
    checkpoint_step,
    iter_checkpoints,
)
from games.pace_guard import StepPaceGuardCallback
from games.payoffs import (
    STATED_MATCH_PROB_UNSET,
    STATED_RETURN_UNSET,
    MatrixGameSpec,
    group_mix_fixed_point,
    stated_match_optimal_action,
)
from games.preflight import (
    assert_single_process,
    assert_trainer_generates_through_vllm,
    check_trace_completeness,
    default_cuda_allocator_config,
    detected_world_size,
    log_deltanet_kernel_paths,
    log_resolved_sampler,
    refuse_empty_trace_overwrite,
    resolve_tokenizer,
    summarize_lora_for_architecture,
    trace_file_path,
    verify_trace_files,
)
from games.prompts import generate_prompt_rows, reward_spread_report
from games.provenance import git_provenance
from games.rewards import (
    BACKFILLABLE_REWARD_COLUMNS,
    FRAMING_ID_COLUMN,
    FRAMING_ID_UNSET,
    GRADING_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_NASH_DEMAND_GROUP_MIX,
    GRADING_NASH_DEMAND_SELF,
    GRADING_THRESHOLD_GOODS_GROUP_MIX,
    GRADING_THRESHOLD_GOODS_SELF,
    GRADING_VS_FIXED_MIX,
    GRADING_VS_STATED_MATCH,
    PARSE_PENALTY_CONSTANT,
    PARSE_PENALTY_MARGIN_BELOW_WORSE,
    PARSE_PENALTY_MODES,
    STATED_RETURN_FRACTION_COLUMN,
    TRUST_GRADINGS,
    care_alpha_of,
    make_game_reward,
)
from games.s3_sync import S3SyncCallback, sync_directory
from games.sizing import (
    BYTES_PER_FLOAT32,
    DEFAULT_VRAM_USABLE_FRACTION,
    SequenceCost,
    SizingPlan,
    checkpoint_sequence_cost,
    colocate_reserved_gib,
    count_meta_parameters,
    plan_sizing,
)

# Re-exported deliberately: several stage runners import these from here, and the numbers plus the
# screens behind them live in games/termination.py so that the decode-chunk sizing in
# games/select_prompts.py can read the distributions rather than a comment about them.
from games.termination import (
    MEASURED_TERMINATION_BUDGET,
    MEASURED_TERMINATION_BUDGET_BY_MODEL,
    required_completion_budget,
)
from grpo.estimator_defaults import (
    GRPO_EPSILON,
    GRPO_LOSS_TYPE,
    GRPO_LOSS_TYPES,
    GRPO_SCALE_REWARDS,
    GRPO_SCALE_REWARDS_MODES,
    LIGER_UNFAITHFUL_LOSS_TYPES,
    VESPO_IMPORTANCE_SAMPLING_MODES,
    VLLM_IMPORTANCE_SAMPLING_MODE,
    VLLM_IMPORTANCE_SAMPLING_MODES,
    assert_known_estimator,
    assert_liger_faithful_estimator,
    executed_estimator,
)
from grpo.rlvr_math import (
    MemoryMonitorCallback,
    NonFiniteMetricCallback,
    RewardLoggingCallback,
    load_mem_log,
)
from grpo.throughput import (
    BYTES_PER_GIB,
    MIN_GRPO_GROUP_SIZE,
    TIMING_METRIC_KEYS,
    StepPhaseTimer,
    describe_device,
    discover_lora_targets,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from datasets import Dataset
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase, TrainerCallback

    from games.preflight import TraceFilesVerdict


logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
# Plumbing only -- its behaviour says nothing about the research question. Kept on Qwen3-0.6B, the
# one checkpoint in the ladder whose template does not prefill `<think>`, because every Qwen3.5 tier
# measured (0.8B, 2B, 4B) fails to terminate its thinking on a game prompt at any affordable budget
# and so cannot produce a parseable action to smoke with. The cost is real and worth knowing: this
# tier exercises neither the prefilled-think parser branch nor the linear-attention LoRA discovery
# that every real arm uses. See docs/scratch/qwen35-runaway-deliberation-2026-08-17.md.
SMOKE_MODEL_ID = "Qwen/Qwen3-0.6B"

RUN_ROOT = "artifacts/games/runs"
RUN_CONFIG_FILENAME = "run_config.json"
TRAIN_SUMMARY_FILENAME = "train_summary.json"

# Settings that change generation or allocator behaviour without appearing in any TRL field, so a
# run's artifacts cannot otherwise tell two differently-launched arms apart. The allocator half is
# no longer merely recorded: the 2026-08-19 boxes measured expandable_segments REQUIRED on 32k HF
# workloads (fragmentation OOM-loop without it), so `main` now defaults it through
# `games.preflight.default_cuda_allocator_config` -- setdefault, so an explicit export still wins,
# and whatever value results is what gets recorded here. The sampler variable stays record-only.
UNRECORDED_GENERATION_ENV = ("VLLM_USE_FLASHINFER_SAMPLER", "PYTORCH_CUDA_ALLOC_CONF")

# Qwen3.5's text vocabulary, read off the family configs this session (2B and 4B both carry
# text_config.vocab_size=248320). Used only to ESTIMATE the importance-sampling correction's fp32
# logits row for the refusal in `_validate_memory_plan`, so an over-read on a smaller non-family
# vocab (Qwen3-0.6B: 151,936) errs toward refusing, never toward a mid-run OOM.
QWEN3_5_VOCAB_SIZE = 248_320
# Where the correction's old-logps pass is REFUSED under colocate, in GiB of the fp32 logits it
# holds live at its peak -- `old_logps_pass_peak_gib`, which counts the tensors rather than the
# widest one. `logits_to_keep` trims the prompt half, which is why the measured 30.31 GiB at a
# 32,768-token budget is exactly one unchunked row's worth. Measured dead on 2026-08-19, twice, on a
# 95 GiB card: 30.31 GiB beside the resident engine, even with all three TRL memory patches applied
# and the chunk clamped to a single row. That measurement set the threshold, and it stays where the
# measurement that exists put it. Two chunk-sized copies live at once at a single-row micro-batch,
# so 16 GiB is a shade under one copy of the shape that died rather than the ~1.9x margin an earlier
# version of this comment claimed, and the 4B floor budget (16,384 positions) is refused rather than
# allowed: it prices at 30.31 GiB, the dead shape's own figure. The default 2,048-token chunk is
# 3.79 GiB and passes at every budget, and a chunk at or above the row length -- the unchunked shape
# -- is still refused. Two costs sit outside this line on purpose. The resident fp32 head under
# `cast_lm_head_to_fp32` adds vocab x hidden x 2 bytes over its bf16 self for the whole run (1.90
# GiB at the 9B's 4,096 hidden), a constant the chunk cannot move and the hidden size is not known
# where a config validates. And the backbone's own no-grad forward over the padded width is priced
# by `games.sizing`, not here. The chunked peak's own card measurement is the five-step probe; until
# it lands, this arithmetic is the estimate.
IS_CORRECTION_LOGITS_REFUSAL_GIB = 16.0

# Positions per slice of the old-log-probability pass. The pass exists only under the vLLM
# importance-sampling correction (grpo_trainer.py:2631-2634), and TRL builds the whole
# (positions x vocab) logits block for a row at once (:1524-1534): at a 32,768-token budget over
# Qwen3.5's 248,320-token vocabulary that is 30.31 GiB of fp32, the figure the pass died at on
# 2026-08-19. Slicing the hidden states instead puts the peak at a couple of chunk-sized copies
# (`old_logps_pass_peak_gib`), 3.79 GiB here, and changes no number, because a log-softmax reduces
# each position over the vocabulary without seeing any other position.
OLD_LOGPS_CHUNK_TOKENS = 2048

# Columns a corpus row must carry for the arm check to mean anything. The rest of the schema is
# the dataset builder's business, and it raises on the reserved names TRL would overwrite.
CORPUS_ARM_COLUMNS = ("game_id", "grading")
# The checkpoint a corpus's prompt selection was measured against, per row rather than in a header
# so the corpus file stays a plain JSONL of rows. `games.select_prompts` does not write it yet --
# see `assert_corpus_selected_for_model`, which warns loudly when it is absent.
SELECTED_FOR_MODEL_COLUMN = "selected_for_model"
# The two axes `games.prompts` crosses to build a corpus, plus the side of the group-mix boundary a
# mix-split arm trains, plus the two axes a breadth corpus spans: which game a row is and which
# counterpart framing its prompt carries. Which values of them a run actually trained on is the
# difference between "this arm covers the ladder" and "this arm trained on one rung", and nothing
# else in the artifacts says which happened. The partition column is absent from every corpus swept
# before the mix-split existed and the framing column from every corpus before wave 4b, and
# `describe_row_composition` skips columns the rows do not carry.
ROW_COMPOSITION_COLUMNS = (
    PAYOFF_VARIANT_COLUMN,
    "reskin_id",
    CORPUS_PARTITION_COLUMN,
    "game_id",
    FRAMING_ID_COLUMN,
)

# A metric absent here means the run measured nothing it claims to have measured, so the read-back
# raises rather than writing a summary full of nulls. The behavioural metric is NOT here, because
# which one an arm produces depends on its grading: `required_metrics_for` adds `coop_rate` or
# `mean_keep_fraction` per arm. Leaving both merely optional is what left the dictator arm's only
# behavioural number outside this gate.
REQUIRED_METRICS = (
    "reward",
    "reward_std",
    # Ours, from games.rewards.group_purity. TRL's `frac_reward_zero_std` is still requested below
    # but is NOT trusted: under the recorded runs' scale_rewards="batch" it was a batch-level
    # scalar that could never detect a pure group, and it reported 0.0000 on every step of every
    # arm for a night while being read as evidence that every group disagreed. Under the current
    # default scale_rewards="none" TRL computes it from group-level stds, so it could work now --
    # but it stays distrusted until someone has watched it go non-zero on a pure group.
    "frac_groups_pure",
    "parse_failure_rate",
    "truncated_thinking_rate",
)
# Per generation batch, the surprisal of the tokens vLLM actually sampled. At temperature 1.0 the
# mean is an unbiased single-sample estimate of the sampling distribution's entropy, which is the
# quantity every entropy-collapse result in the 2025-2026 RLVR literature is stated in and the one
# thing this loop logged nothing about: the Liger loss path appends only `kl` and `clip_ratio`
# (grpo_trainer.py:2974-2975) and Liger forbids both the entropy bonus and the top-entropy mask.
SAMPLED_SURPRISAL_MEAN_METRIC = "entropy/sampled_surprisal_mean"
SAMPLED_SURPRISAL_MIN_METRIC = "entropy/sampled_surprisal_min"
SAMPLED_SURPRISAL_MAX_METRIC = "entropy/sampled_surprisal_max"
SAMPLED_SURPRISAL_TOKENS_METRIC = "entropy/sampled_surprisal_tokens"
SAMPLED_SURPRISAL_METRICS: tuple[str, ...] = (
    SAMPLED_SURPRISAL_MEAN_METRIC,
    SAMPLED_SURPRISAL_MIN_METRIC,
    SAMPLED_SURPRISAL_MAX_METRIC,
    SAMPLED_SURPRISAL_TOKENS_METRIC,
)
# Per optimizer step under dynamic sampling: how many groups were generated, how many of them carried
# no advantage at all, and how the kept batch was made up. The last two sum to the groups the
# optimizer trained on, so a step whose fallback count is nonzero is a step that ran out of live
# groups -- which is the reading that says whether the oversample is wide enough.
DYNAMIC_SAMPLING_GENERATED_METRIC = "dynamic_sampling/groups_generated"
DYNAMIC_SAMPLING_PURE_METRIC = "dynamic_sampling/groups_pure"
DYNAMIC_SAMPLING_KEPT_LIVE_METRIC = "dynamic_sampling/groups_kept_live"
DYNAMIC_SAMPLING_KEPT_PURE_METRIC = "dynamic_sampling/groups_kept_pure"
DYNAMIC_SAMPLING_STEP_METRICS: tuple[str, ...] = (
    DYNAMIC_SAMPLING_GENERATED_METRIC,
    DYNAMIC_SAMPLING_PURE_METRIC,
    DYNAMIC_SAMPLING_KEPT_LIVE_METRIC,
    DYNAMIC_SAMPLING_KEPT_PURE_METRIC,
)
# Recorded for comparison, never relied on. `frac_reward_zero_std` and `clip_ratio` were both pinned
# at exactly 0.0 across every step of three 70-step arms.
DISTRUSTED_METRICS = ("frac_reward_zero_std", "clip_ratio")
# The parse-price guard's ratio of the failure's distance below the worst reachable reward to the row's
# reachable spread, which its price pins at 1 at every counterpart distribution
# (`games.rewards._log_parse_price_metrics`).
PARSE_PRICE_IDENTITY_METRICS: tuple[str, ...] = (
    "parse_price/guard_identity_min",
    "parse_price/guard_identity_max",
)
# The two denominators of the guard's two refusals: how many failures the price identity was asked of,
# and how many the group-range comparison was (`games.rewards._log_parse_price_metrics`). Separate keys
# because one half is gated on the group sharing the failure's resolution and the other is not, so a
# single count would read as "the guard ran" while half of it never did.
PARSE_PRICE_RANGE_DENOMINATOR_METRIC = "parse_price/n_checked"
PARSE_PRICE_IDENTITY_DENOMINATOR_METRIC = "parse_price/n_identity_checked"
# What a failure actually cost this step, named because the constant mode pins it too.
PARSE_PRICE_REALISED_MEAN_METRIC = "parse_price/realised_mean"
# The guard's per-step readings: what a failure actually cost, how many were priced, how many each half
# of the invariant could be asked of, the group's worst parsed reward minus the price, and that ratio.
PARSE_PRICE_METRICS: tuple[str, ...] = (
    PARSE_PRICE_REALISED_MEAN_METRIC,
    "parse_price/n_failures",
    PARSE_PRICE_RANGE_DENOMINATOR_METRIC,
    PARSE_PRICE_IDENTITY_DENOMINATOR_METRIC,
    "parse_price/min_margin_below_worst_parsed",
    *PARSE_PRICE_IDENTITY_METRICS,
)
# The metrics whose whole reading is that they do NOT move whatever the run does, exempted from the
# dead-metric report below -- the identity pair and nothing else. Listing a ratio that is 1 by
# construction would report the healthy state as an unmeasured one, and a warning that fires on every
# run teaches its reader to skip the list. Their aggregates still land in the summary, so a run whose
# identity drifted off 1 is still readable after the fact.
CONSTANT_BY_CONSTRUCTION_METRICS: tuple[str, ...] = PARSE_PRICE_IDENTITY_METRICS
# What the constant penalty mode pins on top of those, on every arm but the two prosocial-breadth ones.
# It prices a failure from a run knob rather than from the row's answer space, so there is nothing for
# either half of the guard to be asked of (both denominators sit at zero) and the mean price IS the
# knob. Exempt only under that mode: under `margin-below-worse` each of the three is a live reading,
# and a stuck range denominator there is the leave-one-out hole the guard's split closed.
PARSE_PRICE_CONSTANT_MODE_PINNED_METRICS: tuple[str, ...] = (
    PARSE_PRICE_REALISED_MEAN_METRIC,
    PARSE_PRICE_RANGE_DENOMINATOR_METRIC,
    PARSE_PRICE_IDENTITY_DENOMINATOR_METRIC,
)


def constant_by_construction_metrics(parse_penalty_mode: str) -> tuple[str, ...]:
    """Which watched metrics this run's parse-penalty mode makes single-valued by construction.

    Read by `read_back_metrics`, which cannot work the mode out for itself: it sees a log history and
    the mode is a property of the arm that produced it. Every mode is named rather than one being the
    fallback, because what a mode pins is part of what the mode means, and a third one added without a
    line here would silently take the narrow exemption and put its own pinned series in front of the
    reader on every run.
    """
    if parse_penalty_mode == PARSE_PENALTY_CONSTANT:
        return (*CONSTANT_BY_CONSTRUCTION_METRICS, *PARSE_PRICE_CONSTANT_MODE_PINNED_METRICS)
    if parse_penalty_mode == PARSE_PENALTY_MARGIN_BELOW_WORSE:
        return CONSTANT_BY_CONSTRUCTION_METRICS
    raise ValueError(
        f"parse_penalty_mode must be one of {PARSE_PENALTY_MODES}, got {parse_penalty_mode!r}, so "
        "which of the parse-price series it pins by construction is unknown."
    )


OPTIONAL_METRICS = (
    *DISTRUSTED_METRICS,
    *PARSE_PRICE_METRICS,
    # Every behavioural metric stays listed here so the arm that does NOT require one still gets its
    # aggregates in the summary and still gets checked for being constant across the run.
    "coop_rate",
    "mean_keep_fraction",
    "mean_claim_fraction",
    # The claim game's overreach rate under each of its two gradings. Two keys, because the same
    # arithmetic is a mean collision rate against the group's claims under group-mix grading and the
    # deterministic indicator `claim > windfall / 2` under self grading (games.rewards._Row).
    "crash_rate",
    "overclaim_rate",
    # The claim game's fairness anchor: how often the answer is exactly half the windfall.
    "exact_half_claim_rate",
    "mean_contribution_fraction",
    # The shared undertaking's reach rate under each of its two gradings. Two keys, because the same
    # arithmetic is an expected chance of clearing the bar against the group's realised figures under
    # group-mix and the deterministic indicator `contribution >= equal_share` under self grading.
    "threshold_met_rate",
    "equal_share_met_rate",
    # The shared undertaking's cooperative-versus-maximising discrimination: how often the figure is above
    # the equal share, which buys no more of the shared thing and costs a unit each time. One key, because
    # unlike the pair above it is a property of the answer alone rather than of the counterpart mix.
    "over_contribution_rate",
    # Ours, from games.rewards.group_reward_span. How much advantage the groups that disagree carry,
    # which frac_groups_pure cannot say and which decides whether an arm trains strategy or format.
    "mean_group_reward_span",
    "mean_send_fraction",
    # The stated-track-record grading's three (games.rewards._log_stated_match_metrics): how often
    # the parsed action is the one the row's own (payoff table, stated p) pair pays more for, and
    # the cooperation rate split by which side that is -- the split that stops blanket cooperation
    # from reading as healthy movement on a corpus that mixes both incentive directions.
    "ev_optimum_rate",
    "coop_rate_where_coop_pays",
    "coop_rate_where_defect_pays",
    # Trust-only, and both are the instrumentation the trust arms' readings depend on rather than
    # extras: the promised return share is the promise-inflation channel of the strategy method, and
    # the optimum rate is whether the send is on the paying side of the break-even.
    "mean_stated_return_fraction",
    # Two keys, because the same phrase names two different questions. Under an ANNOUNCED rate the
    # optimum send is a corner the prompt fixes, so this is "did it multiply the number in front of
    # it". Under the strategy method the completion writes the rate as well, so the readable optimum
    # is the whole (send, promise) pair -- see games.rewards._Scored.
    "send_at_payoff_optimum_rate",
    "strategy_at_payoff_optimum_rate",
    # The minimum-effort games' own numbers. Each arm requires the subset that is its own reading
    # (see `required_metrics_for`); every one of them stays listed here so the arms that do NOT
    # require one still get its aggregates in the summary and still get it checked for being constant
    # across the run.
    "mean_level_fraction",
    "mean_level",
    "mean_target_level",
    "mean_upward_pressure",
    # Logged by both repeated forms under one key, because it is one quantity asked of two games: was
    # the last round less than the one before. Against the level-matcher a drop only loses money and
    # against the copying PD it is optimal, so the CONTRAST is the reading and one key is what makes
    # it available (games.rewards._log_batch_metrics).
    "end_game_drop_rate",
    # Logged only under --leave-one-out, and it is the flag saying a reward was computed from
    # UNINFORMATIVE_COOP_PRIOR rather than from the group.
    "leave_one_out_prior_rate",
    "completions/mean_length",
    "completions/clipped_ratio",
    "kl",
    # TRL's own entropy, logged only on the non-Liger loss path, and then the proxy that stands in for
    # it everywhere else (`InstrumentedGRPOTrainer`). Optional rather than required because a rollout
    # path that returns no sampled log-probabilities has no reading, and here rather than merely in
    # `mem_log.csv` so the train summary carries the run's entropy trajectory and the constant-metric
    # watchdog covers it: a surprisal series that never moves is a dead instrument, not a flat policy.
    "entropy",
    *SAMPLED_SURPRISAL_METRICS,
    # Absent from every run at the default oversample, which is why they are optional rather than
    # required: at 1 nothing is dropped and the trainer logs no selection at all. Where they do
    # appear, `groups_kept_pure` is the reading that matters -- a step that had to fall back to pure
    # groups is a step the oversample was too narrow for.
    *DYNAMIC_SAMPLING_STEP_METRICS,
)

# Every knob `--smoke` shrinks. Smoke exists to prove the path executes end to end, so it keeps
# checkpointing (`save_steps` under `max_steps`) and full-trace logging switched on.
#
# The completion budget is the one knob that cannot be shrunk to taste. Qwen3-0.6B emits its own
# `<think>` block and is verbose, so a budget that cuts it off mid-thought leaves no visible answer
# at all and every completion scores as a parse failure -- which trips the reward's
# entire-batch-unparseable raise and fails the smoke for a reason that has nothing to do with the
# plumbing. Measured on this box at the training sampler, 32 samples per budget: 192 tokens parsed
# 0 of 32 (every one hit the cap), 1024 parsed 15, 1536 parsed 16, and 2048 parsed 28 with only 4
# hitting the cap. Since the raise needs a whole batch of 8 to fail, the odds of a spurious smoke
# failure go from ~1-in-160 per step at 1024 to negligible at 2048. Thinking stays on: the
# decision-theory reasoning these arms measure lives in the CoT.
SMOKE_COMPLETION_TOKENS = 2048

# Sentinel asking for the newest checkpoint already in the run directory.
RESUME_LATEST = "latest"
# Written by Trainer.save_state; its absence means a directory is not a checkpoint.
TRAINER_STATE_FILENAME = "trainer_state.json"
# A resumed launch writes its own record here so the original launch's survives. The checkpoint
# directory is in the name because that is what says which steps this record picks up from. Taken as
# a name rather than a parsed step number, so an explicitly named checkpoint that does not follow
# the `checkpoint-<step>` convention still gets a record instead of crashing on the way to one.
RESUMED_RUN_CONFIG_TEMPLATE = "run_config.resume-from-{checkpoint}.json"
# `GameTrainConfig` fields that say WHICH EXPERIMENT a checkpoint's steps belong to. Resuming
# changes none of them: `--resume-from-checkpoint latest --output-dir X` under a different arm,
# model or grading loads the old adapter, optimizer and global_step and keeps training them under a
# new question, with nothing in the artifacts saying so.
RESUME_IDENTITY_FIELDS = (
    "arm",
    "model_id",
    "model_source",
    "corpus_path",
    "generate_fresh",
    "thinking",
    "leave_one_out",
    "parse_penalty",
    "max_prompt_tokens",
    "max_completion_tokens",
    "seed",
    "loss_type",
    "scale_rewards",
    "init_adapter",
    # Optimizer and adapter treatment. Both are inside the update rather than beside it: the
    # epsilon decides how much of each nominal Adam step the LoRA A matrices actually take, and
    # the dropout decides whether the graded forward is the same function the rollout sampled
    # from. Steps taken under one value continue perfectly happily under another, which is why
    # they are here rather than left to a reader of the log.
    "adam_epsilon",
    "lora_dropout",
    # The sampler-mismatch instruments, which decide WHICH ESTIMATOR TRL computes exactly as
    # `loss_type` and `scale_rewards` do. With the correction on and log-only off, the surrogate
    # loss is weighted by TRL's ratio (grpo_trainer.py:2896 into liger grpo_loss.py:205-206); with
    # the correction off there is no ratio at all, and the `sampling/*` series appear or vanish
    # mid-run. The mode is the same rung: sequence-level masking zeroes a whole rollout whose
    # accumulated ratio leaves the band where token-level truncation clips each token, so steps taken
    # under one are not steps taken under the other. The fp32 head is here for the same reason one
    # rung down: it moves the log-probabilities the ratio is computed from, so a flip changes what
    # every step after it trained on.
    "vllm_importance_sampling_correction",
    "vllm_importance_sampling_log_only",
    "vllm_importance_sampling_mode",
    "cast_lm_head_to_fp32",
    # Dynamic sampling decides WHICH PROMPTS an optimizer step trains on: at 1 the step trains the
    # sampler's own draw, above it the live subset of a wider draw. A resume that moved it would put
    # steps trained on two different selections under one step history, and the wider draw also
    # consumes the corpus faster, so the resumed steps would not even see the prompts the earlier
    # ones would have reached next.
    "dynamic_sampling_oversample",
)
# Identity fields added after the first recorded runs, mapped to the value in force when those runs
# launched -- the then-hardcoded GRPOConfig line for the estimator pair, transformers' own untouched
# default for the optimizer epsilon, the dataclass default every banked arm carried for the adapter
# dropout, and, for the sampler-mismatch instruments, TRL's own correction default and TRL's own
# correction mode with neither of the two knobs that landed later. A record written before the field
# existed means THIS value, not a mismatch -- and not the current default, which would let a resume
# silently relabel what the checkpoint's steps were trained under.
RESUME_IDENTITY_DEFAULTS: dict[str, object] = {
    "model_source": None,
    "loss_type": "dapo",
    "scale_rewards": "batch",
    "init_adapter": "",
    "adam_epsilon": 1e-8,
    "lora_dropout": 0.05,
    "vllm_importance_sampling_correction": True,
    "vllm_importance_sampling_log_only": False,
    "vllm_importance_sampling_mode": VLLM_IMPORTANCE_SAMPLING_MODE,
    "cast_lm_head_to_fp32": False,
    "dynamic_sampling_oversample": 1,
}
# Identity fields that are REGISTRY state rather than launch config, compared the same way and kept
# apart only because the record carries them at its top level while `RESUME_IDENTITY_FIELDS` names
# fields inside its `config` block.
#
# The arm name cannot stand in for the game set. A rented box ships the current tree rather than the
# tree a checkpoint was trained by, so the registry entry can gain or lose a game between a launch and
# its relaunch while the arm, the model, the corpus path and every other recorded field still agree --
# and the resumed steps would then train prompts the earlier ones never saw, under one set of step
# numbers and one summary.
RESUME_ARM_IDENTITY_FIELDS = ("game_ids",)
# The value in force for every run recorded before the field existed: one game, no extra games. The
# precedent and the reasoning are `RESUME_IDENTITY_DEFAULTS`'s -- absence means what that era ran,
# never today's default.
RESUME_ARM_IDENTITY_DEFAULTS: dict[str, object] = {"game_ids": []}
# The tensor file an init adapter is read from; its sha256 lands in the run record so the exact
# bytes that seeded a transfer run stay attributable after the checkpoint directory moves.
INIT_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
# `SizingPlan` fields a resume may not move. `num_generations` is the experiment twice over: GRPO's
# advantage baseline is computed within the group, and group-mix grading estimates the opponent
# distribution from that same group. The plan is derived from a LIVE free-VRAM reading, so another
# process holding a couple of gibibytes on a shared card is enough to change it.
RESUME_SIZING_FIELDS = (
    "num_generations",
    "prompts_per_step",
    "micro_batch_size",
    "gradient_accumulation_steps",
)
SMOKE_OVERRIDES: dict[str, object] = {
    "num_generations": 4,
    "prompts_per_step": 2,
    "max_steps": 3,
    "max_prompt_tokens": 512,
    "max_completion_tokens": SMOKE_COMPLETION_TOKENS,
    "max_prompts": 4,
    "save_steps": 2,
    "logging_steps": 1,
    "generate_fresh": True,
}


def assert_resume_is_addressable(resume_from_checkpoint: str, output_dir: str | None) -> None:
    """Reject a resume request that could never find the run it means to continue.

    Asking for the newest checkpoint without naming the run directory is the one combination that
    fails silently: `output_dir` is timestamped fresh on every launch, so the restart would look at
    an empty new directory, find nothing, and begin again at step 0 while reporting success.
    """
    if resume_from_checkpoint == RESUME_LATEST and output_dir is None:
        raise ValueError(
            f"--resume-from-checkpoint {RESUME_LATEST!r} needs an explicit --output-dir. Without "
            f"one the run directory is timestamped fresh on every launch, so the checkpoint being "
            f"resumed would never be found and the restart would silently begin at step 0."
        )


def assert_init_adapter_is_loadable(init_adapter: str) -> None:
    """Reject an init-adapter path missing either file the load would need, at config time.

    Checked at construction rather than at load so a mistyped path or an un-restored download
    fails before a card is reserved and a base model pulled. Content checks (base model, rank,
    target set) need the run's own discovered LoRA plan and live in `describe_init_adapter`.
    """
    if not init_adapter:
        return
    adapter_dir = Path(init_adapter)
    for name in (ADAPTER_CONFIG_FILENAME, INIT_ADAPTER_WEIGHTS_FILENAME):
        if not (adapter_dir / name).is_file():
            raise ValueError(
                f"--init-adapter {init_adapter!r} has no {name}, so it is not a PEFT adapter "
                f"checkpoint this run could initialise from. Point it at the directory holding "
                f"{ADAPTER_CONFIG_FILENAME} and {INIT_ADAPTER_WEIGHTS_FILENAME} (a trainer "
                f"checkpoint-<step> directory is the usual shape)."
            )


@dataclass(frozen=True)
class GameTrainConfig:
    """Every knob of one arm's run. The whole thing lands in `run_config.json` verbatim."""

    arm: str
    model_id: str = DEFAULT_MODEL_ID
    # Optional immutable snapshot to load. ``model_id`` remains the canonical checkpoint identity
    # used for measured sampler floors, corpus compatibility, run naming, and cost metadata.
    model_source: str | None = None
    # A corpus from `games.select_prompts` holds only prompts whose action distribution was
    # mixed at training temperature, which is where GRPO's within-group disagreement comes from.
    corpus_path: str | None = None
    # Training on freshly generated prompts skips that selection, so it must be asked for.
    generate_fresh: bool = False
    num_generations: int = 8
    prompts_per_step: int = 8
    # Sequences per training forward/backward. None means one prompt group, which is what TRL
    # would do; it decides whether a configuration fits independently of the episode count.
    micro_batch_size: int | None = None
    learning_rate: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    max_steps: int = 70
    max_prompt_tokens: int = 1024
    # The measured floor is the DEFAULT, not something to opt into: a run that reads science off
    # the chain of thought has to let the chain of thought finish. Smaller budgets need either
    # --allow-short-completions or --no-thinking, both of which label themselves.
    max_completion_tokens: int = MEASURED_TERMINATION_BUDGET
    # Shared with the sweep rather than re-stated: `games.generation` carries GRPOConfig's own
    # defaults, and "at training temperature" has to mean the same three numbers here and in
    # `games.select_prompts.training_sampler` or the selected corpus is off-policy.
    temperature: float = TRAINING_TEMPERATURE
    top_p: float = TRAINING_TOP_P
    top_k: int = TRAINING_TOP_K
    # DAPO drops the KL term. These arms are supposed to move the policy; capability retention
    # is measured by the arithmetic canary afterwards rather than constrained during training.
    beta: float = 0.0
    epsilon: float = GRPO_EPSILON
    # Estimator knobs, defaulted from grpo/estimator_defaults.py, which carries the full reasoning
    # (the Liger dapo degradation, the group-scaling payoff-gap cancellation, the measured
    # sigma_batch artifacts). Changing either mid-arm relabels the experiment, so both are
    # resume-identity fields; records from before these fields existed mean "dapo"/"batch".
    loss_type: str = GRPO_LOSS_TYPE
    scale_rewards: str = GRPO_SCALE_REWARDS
    # The only way to run a Liger-unfaithful loss_type (dapo/cispo/vespo/luspo) under Liger: a
    # deliberate statement that the EXECUTED estimator -- not the named one -- is the experiment,
    # e.g. the A/B arm isolating scale_rewards with "dapo" as executed-GRPO. Recorded in
    # run_config.json like every other field, and refused when there is no mismatch to acknowledge.
    acknowledge_liger_estimator_mismatch: bool = False
    # AdamW's denominator floor, at transformers' own default so that no plan written before this
    # field existed trains differently for its arrival. It is a treatment knob rather than a
    # numerical nicety on LoRA: sqrt(v_hat) for the A matrices of the banked 9B self arm sat below
    # 1e-8 for 99.3 percent of entries at step 70 (measured off its optimizer.pt), so the epsilon
    # was setting the A-side step to about a tenth of its nominal size, and the damping moves with
    # the reward scale and the completion budget that feed dr_grpo's normaliser. 1e-15, which
    # MiniMax-M1 and ScaleRL both run, restores the scale invariance we believed we already had.
    adam_epsilon: float = 1e-8
    lora_rank: int = 16
    lora_alpha: int = 32
    # Nonzero makes the graded forward stochastic while vLLM sampled the rollout with no dropout at
    # all, which is a train-inference mismatch on an otherwise exactly on-policy setup. Left at the
    # value every banked arm trained under; the RL recipes all run 0, which an arm asks for.
    lora_dropout: float = 0.05
    use_liger_kernel: bool = True
    gradient_checkpointing: bool = True
    # Keep every intermediate: the eval battery runs per checkpoint, and re-running an arm to
    # recover a step we declined to save costs orders of magnitude more than the disk does.
    save_steps: int = 10
    save_total_limit: int = 50
    # Every optimizer step flushes a completions parquet, and TRL's buffer holds exactly one
    # generation batch, so logging any less often leaves silent holes in the rollout trace.
    logging_steps: int = 1
    seed: int = 0
    leave_one_out: bool = False
    parse_penalty: float = -1.0
    # Generate this many times the optimizer step's prompt groups and train on the ones that carry a
    # gradient (`DynamicSampledGRPOTrainer`). 1 is off, and off is today's behaviour. A treatment
    # rather than a throughput knob: which prompts an optimizer step trains on stops being the
    # sampler's draw and becomes the subset of that draw whose rewards disagreed, so it is a
    # resume-identity field and it is recorded per run. The generation cost scales with it while the
    # training cost does not, which is the trade: on the banked 9B pair a fifth of groups were pure,
    # a third of them late, and on a breadth corpus's far framings most groups are expected to be.
    dynamic_sampling_oversample: int = 1
    # Smoke and debugging only: it changes what corpus the arm trained on, so it is never a
    # performance knob for a real run.
    max_prompts: int | None = None
    smoke: bool = False
    autosize: bool = True
    vram_usable_fraction: float = DEFAULT_VRAM_USABLE_FRACTION
    # Rollouts generate through a vLLM engine colocated in this process, unconditionally: there is
    # no backend field because there is no backend choice (games.generation.VLLM_ONLY_RATIONALE).
    # These two knobs tune the engine; nothing turns it off.
    vllm_gpu_memory_utilization: float = VLLM_COLOCATE_GPU_FRACTION
    # TRL's own default, and deliberately left at it: switching it off changes the gradient
    # estimator, which is a science decision needing a recorded caveat rather than a default. What
    # each setting costs at the production shape is in `log_colocate_settings`.
    vllm_importance_sampling_correction: bool = True
    # WHAT the correction does with the log-probability difference, at TRL's own default, so no plan
    # written before this knob existed changes treatment for its arrival. TRL's default aggregates
    # the difference over the whole sequence and zeroes any sequence whose ratio leaves
    # [C_min, C_max]: a product of thousands of near-one per-token ratios leaves that band on length
    # alone, so a long rollout is discarded for being long rather than for being off-policy. The 2B
    # probe smoke measured sequence ratios averaging 0.66 and 0.21 where the per-token difference was
    # 0.015 nats, which is that effect. `token_truncate` clips each token's own ratio into the band
    # instead, and is what the owner precommitted (2026-09-04) arm 1 and its control run if the 9B
    # probe confirms the mismatch. A resume-identity field for `loss_type`'s reason: it decides which
    # estimator every step after it trained under.
    vllm_importance_sampling_mode: str = VLLM_IMPORTANCE_SAMPLING_MODE
    # Measure the vLLM-versus-trainer mismatch without paying for it: TRL computes the correction and
    # logs its ratio statistics, and the weight the loss applies is replaced by exactly one, so the
    # gradient is the correction-off gradient. Off by default because a mode that changes what is
    # logged and not what is trained still has to be asked for by name in the run record.
    vllm_importance_sampling_log_only: bool = False
    # Positions per slice of the correction's old-log-probability pass. Not a resume-identity field
    # and not an experiment knob: chunking moves the memory peak and leaves the log-probabilities
    # where they were, to within the head matmul's own reassociation (see `chunked_per_token_logps`).
    old_logps_chunk_tokens: int = OLD_LOGPS_CHUNK_TOKENS
    # Run the language-model head in fp32 (TRL's own knob, grpo_trainer.py:993-1028, which replaces
    # `lm_head.forward`). What takes the cast is the old-logps pass and a non-Liger loss; the
    # PRODUCTION loss does not, because `use_liger_kernel=True` never calls `lm_head.forward` at all
    # -- TRL hands Liger the raw `lm_head.weight` (grpo_trainer.py:2951-2956) and Liger casts each
    # vocab chunk of it back to the hidden dtype (liger fused_linear_ppo.py:47), so the loss's own
    # logits come off bfloat16 weights either way. Off by default and refused without the correction,
    # per the recorded reasoning: TRL casts the trainer's head while vLLM keeps sampling in bfloat16,
    # so on its own this CHANGES the train-inference mismatch rather than removing it. With the
    # correction it is what makes the mismatch measurable, because a bfloat16 head's last-bit noise
    # is both the mismatch's own main source and the one thing that makes the chunked pass differ
    # from the unchunked one. Not refused alongside a gradient-moving correction, even though the
    # loss keeps its bfloat16 weights: Liger accumulates its logits in fp32 from those weights, so an
    # fp32 old-logps pass agrees with the loss more closely than a bfloat16 one does, and the
    # surrogate ratio at iteration 1 gets closer to exactly one rather than further from it (pinned
    # by `TestTheFp32HeadMovesTheOldLogpsPassTowardTheLigerLoss`).
    cast_lm_head_to_fp32: bool = False
    output_dir: str | None = None
    # New experiment presets opt into a per-save inventory. Existing games arms retain their
    # historical persistence behavior unless they explicitly request this artifact.
    record_retention_manifest: bool = False
    # Thinking ON is the plan's requirement, because the decision-theory reasoning these arms
    # measure lives in the chain of thought. It is a flag only because every Qwen3.5 tier measured
    # (0.8B, 2B, 4B) and openbmb/MiniCPM5-1B fail to terminate their thinking on a game prompt at
    # any affordable budget, so a thinking-OFF run is the only way to exercise this path end to end
    # on current hardware. Such a run is PLUMBING, not science: it is labelled as such in the log
    # banner and in run_config.json so nobody reads a behavioural result off it.
    thinking: bool = True
    # Deliberate escape hatch from the measured-budget floor below. Needed for a timing probe or a
    # deliberately short diagnostic; never for a run whose behaviour anyone will read.
    allow_short_completions: bool = False
    # Where to ship the run directory as it is written. Empty disables it, which is what a local
    # run wants. On Batch this arrives from the GAMES_S3_DEST environment variable rather than a
    # flag -- see `_parse_args` for why that is the single source of truth there.
    s3_dest: str = ""
    # Restart a killed run instead of paying for it twice. Empty starts fresh; a checkpoint path
    # resumes from exactly that step; RESUME_LATEST picks the highest-numbered checkpoint already in
    # `output_dir`, which is what makes re-running the same command the whole recovery procedure.
    # A 30-hour run without this is hostage to one interruption, and it is also the precondition
    # for ever using spot.
    resume_from_checkpoint: str = ""
    # Transfer-of-learning knob: a PEFT adapter checkpoint directory whose (A, B) weights seed this
    # run's freshly built LoRA before step 0 -- continue-RL under a NEW arm, with a fresh optimizer,
    # scheduler and step counter, where --resume-from-checkpoint continues the SAME run and would
    # refuse a different arm. Empty trains from PEFT's usual zero-B initialisation. Composes with
    # resume for spot recovery: once a checkpoint exists in --output-dir, the resume wins (its
    # weights already descend from this init) and the field is provenance, checked for identity.
    init_adapter: str = ""

    def __post_init__(self) -> None:
        """Reject a configuration that names no arm, or no corpus at all."""
        # First, before anything else can look plausible: an environment asking for the removed
        # transformers.generate rollout path invalidates every launch, however valid the rest is.
        assert_vllm_rollouts()
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; known arms: {sorted(ARMS)}")
        self._validate_model_identity()
        assert_resume_is_addressable(self.resume_from_checkpoint, self.output_dir)
        assert_init_adapter_is_loadable(self.init_adapter)
        if self.corpus_path is not None and self.generate_fresh:
            raise ValueError(
                "pass either --corpus or --generate-fresh, not both: a selected corpus and a "
                "fresh generation are different experiments"
            )
        if self.corpus_path is None and not self.generate_fresh:
            raise ValueError(
                "no corpus given. Pass --corpus <jsonl> from games.select_prompts, or "
                "--generate-fresh to train on unselected prompts (higher zero-advantage risk)"
            )
        if self.num_generations < MIN_GRPO_GROUP_SIZE:
            raise ValueError(
                f"GRPO needs at least {MIN_GRPO_GROUP_SIZE} generations per prompt, "
                f"{self.num_generations=}"
            )
        if self.prompts_per_step < 1:
            raise ValueError(f"need at least one prompt per step, {self.prompts_per_step=}")
        if self.dynamic_sampling_oversample < 1:
            raise ValueError(
                f"dynamic_sampling_oversample must be at least 1, where 1 means off, "
                f"{self.dynamic_sampling_oversample=}. Below one there would be fewer generated "
                f"groups than the optimizer step needs and the batch handed to TRL would be short."
            )
        self._validate_penalty_epsilon_and_dropout()
        assert_known_estimator(self.loss_type, self.scale_rewards)
        assert_liger_faithful_estimator(
            self.loss_type,
            use_liger_kernel=self.use_liger_kernel,
            acknowledged=self.acknowledge_liger_estimator_mismatch,
        )
        if self.max_steps < 1:
            raise ValueError(f"nothing to train, {self.max_steps=}")
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be positive for GRPO to see disagreement, {self.temperature=}"
            )
        if self.s3_dest and not self.s3_dest.startswith("s3://"):
            raise ValueError(
                f"s3_dest must be an s3:// URI, got {self.s3_dest!r}. Checked here rather than at "
                f"the first checkpoint, so a typo fails at startup instead of minutes into a run."
            )
        self._validate_completion_budget()
        self._validate_instruments()
        self._validate_retention()
        self._validate_memory_plan()

    def _validate_model_identity(self) -> None:
        """Require both the logical model identity and any explicit load source to be non-empty."""
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if self.model_source is not None and not self.model_source:
            raise ValueError("model_source must be non-empty when supplied")

    def _validate_retention(self) -> None:
        """Apply the per-step no-rotation policy only to explicitly instrumented runs."""
        if not self.record_retention_manifest:
            return
        validate_retention_settings(
            max_steps=cast("int", SMOKE_OVERRIDES["max_steps"]) if self.smoke else self.max_steps,
            save_steps=1 if self.smoke else self.save_steps,
            save_total_limit=0 if self.smoke else self.save_total_limit,
        )

    def _validate_penalty_epsilon_and_dropout(self) -> None:
        """Refuse the three knobs whose value is what makes the arithmetic downstream mean anything.

        All three are refused here rather than where they are consumed, because all three are
        consumed after a card has been reserved and the weights are down: the reward function raises
        on a non-negative penalty, a non-positive optimizer epsilon shows up as non-finite
        parameters at whichever step first meets a zero second moment, and an out-of-range adapter
        dropout is accepted by peft and by TRL and reports nothing at all.
        """
        if not math.isfinite(self.parse_penalty) or self.parse_penalty >= 0:
            # Finiteness is not pedantry: `argparse` takes `nan` and `-inf` for a `type=float`, and a
            # NaN reward is precisely what TRL 1.10 reads as UNSCORABLE (grpo_trainer.py:2746-2754,
            # `unscorable_mask = torch.isnan(rewards_per_func).all(dim=1)`), so every unparseable
            # completion would be dropped from its group's nan-aware baseline with a forced-zero
            # advantage -- no penalty at all, while `run_config.json` records one. `-inf` is the same
            # bug by another route: the group mean goes infinite and every advantage in it is NaN.
            raise ValueError(
                f"the parse penalty must be a finite negative number so it stays below the [0, 1] "
                f"payoff range, {self.parse_penalty=}. `games.rewards.make_game_reward` refuses the "
                f"same value, but it refuses after the model has loaded on a rented card."
            )
        if self.adam_epsilon <= 0:
            raise ValueError(
                f"the optimizer epsilon must be positive, {self.adam_epsilon=}. It is AdamW's "
                f"denominator floor, so zero or below divides the update by a second moment that "
                f"can be zero and every parameter goes non-finite at the first step it happens on. "
                f"The small value this arm wants is 1e-15, not 0."
            )
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(
                f"lora_dropout must be in [0, 1), {self.lora_dropout=}. At 1 every adapter "
                f"activation is dropped, so the graded forward is the base model while the reward, "
                f"the loss and every logged series stay green and the run trains nothing; peft "
                f"builds that adapter without complaint and TRL never looks. Below zero peft scales "
                f"the surviving activations by a negative factor, which is not dropout at all."
            )

    def _validate_instruments(self) -> None:
        """Refuse an instrument configuration that would measure nothing, or measure it wrongly."""
        if self.old_logps_chunk_tokens < 1:
            raise ValueError(
                f"the old-log-probability pass needs at least one token per chunk, "
                f"{self.old_logps_chunk_tokens=}"
            )
        if self.vllm_importance_sampling_mode not in VLLM_IMPORTANCE_SAMPLING_MODES:
            raise ValueError(
                f"unknown vllm_importance_sampling_mode="
                f"{self.vllm_importance_sampling_mode!r}; TRL 1.10 accepts "
                f"{VLLM_IMPORTANCE_SAMPLING_MODES}. Refused here because TRL only refuses an unknown "
                f"name inside the ratio arithmetic of the first generation batch "
                f"(grpo_trainer.py:2690-2693), which on a rented card is the bootstrap, the model "
                f"pull and a whole generation pass later."
            )
        if (
            self.vllm_importance_sampling_mode != VLLM_IMPORTANCE_SAMPLING_MODE
            and not self.vllm_importance_sampling_correction
        ):
            raise ValueError(
                f"--vllm-importance-sampling-mode="
                f"{self.vllm_importance_sampling_mode!r} is refused with the correction off: there "
                f"is no ratio to compute, so the mode selects nothing while run_config.json records "
                f"a treatment the run never ran. Turn the correction on with it, or leave the mode "
                f"at TRL's own {VLLM_IMPORTANCE_SAMPLING_MODE!r}."
            )
        if self.vllm_importance_sampling_log_only and not self.vllm_importance_sampling_correction:
            raise ValueError(
                "--vllm-importance-sampling-log-only needs the correction ON: the ratio it exists "
                "to log is only computed when TRL runs the correction, and with the correction off "
                "the mode would leave the gradient exactly where it already is while recording "
                "nothing about the mismatch."
            )
        if self.cast_lm_head_to_fp32 and not self.vllm_importance_sampling_correction:
            raise ValueError(
                "--cast-lm-head-to-fp32 is refused with the correction off: it casts THIS model's "
                "head while vLLM keeps sampling in bfloat16, so alone it moves the "
                "sampler-versus-trainer mismatch instead of removing it, and every absolute level "
                "the run reports would carry an unmeasured shift. Turn the correction on with it."
            )
        if (
            self.loss_type == "vespo"
            and self.vllm_importance_sampling_correction
            and self.vllm_importance_sampling_mode not in VESPO_IMPORTANCE_SAMPLING_MODES
        ):
            # Refused for TRL's reason, not for the one an earlier version of this message gave. A
            # ratio of exactly one IS a no-op under vespo -- both TRL (grpo_trainer.py:3042-3045)
            # and Liger (grpo_loss.py:36-38) add clamp(log(ratio)) to the sequence log-ratio, and
            # log(1) is 0 -- so log-only would have left the vespo gradient alone after all. What
            # actually refuses this pair is TRL itself (grpo_trainer.py:924-928): vespo computes its
            # own sequence-level weight and requires a token-level correction mode underneath it.
            # Stated at config time rather than left to TRL, which raises inside
            # `GRPOTrainer.__init__` once a card is reserved. Log-only is covered by the same
            # condition: it needs the correction ON.
            raise ValueError(
                f"loss_type='vespo' is refused with the vLLM importance-sampling correction ON "
                f"under vllm_importance_sampling_mode="
                f"{self.vllm_importance_sampling_mode!r}: TRL 1.10 requires one of "
                f"{VESPO_IMPORTANCE_SAMPLING_MODES} for that pair (grpo_trainer.py:924-928), "
                f"because vespo already computes a sequence-level weight of its own. Pass "
                f"--vllm-importance-sampling-mode token_truncate to measure vespo under the "
                f"correction, --no-vllm-importance-sampling-correction to measure it without, or "
                f"pick another loss_type."
            )

    def _validate_completion_budget(self) -> None:
        """Refuse a thinking-ON run whose completion budget is below the measured floor.

        A refusal rather than a warning because a warning is what we would have ignored: the budget
        that cost a night was 2,048, chosen for what fitted rather than for what the models need.
        Smoke and thinking-off runs are exempt, both being labelled non-science already.
        """
        if not self.thinking or self.smoke or self.allow_short_completions:
            return
        required = required_completion_budget(self.model_id)
        if self.max_completion_tokens >= required:
            return
        provenance = (
            "this model's own termination screen"
            if self.model_id in MEASURED_TERMINATION_BUDGET_BY_MODEL
            else "the generic default, as this model has no screen yet"
        )
        raise ValueError(
            f"a thinking-ON run of {self.model_id} needs at least {required} completion tokens, "
            f"got {self.max_completion_tokens}. That floor comes from {provenance}; see "
            f"MEASURED_TERMINATION_BUDGET_BY_MODEL. A shorter budget truncates rollouts into "
            f"the parse penalty, whose -1.0 dwarfs the C-vs-D reward gap, so the gradient goes "
            f"format-dominated and the chain of thought stops being what the arm measures. Pass "
            f"--allow-short-completions for a timing probe, or --no-thinking for a plumbing run."
        )

    @property
    def load_source(self) -> str:
        """Return the hub id or immutable local snapshot from which weights are loaded."""
        return self.model_source or self.model_id

    def _validate_memory_plan(self) -> None:
        """Reject a division of the card leaving the trainer nothing, or an engine we cannot build.

        vLLM is an optional extra, and TRL would raise for us eventually -- but only inside
        `GRPOTrainer.__init__`, after the weights have loaded and the dataset has been built, which
        on a rented card is minutes of billing to discover that an import is missing. `find_spec`
        answers without importing vllm, so this stays cheap and leaves intact the
        module-load-imports-no-vllm property the repo tests for elsewhere.

        A utilization at or above 1 would leave the trainer nothing at all: the engine holds that
        fraction of the card for the whole run, not only while it is generating.
        """
        if not 0 < self.vram_usable_fraction <= 1:
            raise ValueError(
                f"vram_usable_fraction must be in (0, 1], {self.vram_usable_fraction=}"
            )
        if not 0 < self.vllm_gpu_memory_utilization < 1:
            raise ValueError(
                f"vllm_gpu_memory_utilization must be in (0, 1) -- the colocated engine holds that "
                f"share of the card for the life of the run, so the trainer needs the rest -- got "
                f"{self.vllm_gpu_memory_utilization}"
            )
        if importlib.util.find_spec("vllm") is None:
            raise ValueError(
                "vllm is not installed in this environment, and rollouts are vLLM-only: there is "
                "no transformers.generate path to fall back to (owner decision 2026-08-26; ~72 "
                "min/step against ~13.5). Install the vllm extra (make setup-gpu). Checked here "
                "rather than left to TRL, which only raises once the weights are already loaded."
            )
        # The rows here are the FLOOR, not the fact: with `micro_batch_size` unset, `plan_sizing`
        # derives it from a live free-VRAM reading, and the pass allocates one divided fp32 copy per
        # row of it. `_derive_plan` re-prices against the derived number as soon as it exists, which
        # is still before any weights load; this call is the one that can fail before a card is even
        # read, so it prices what the operator asked for.
        assert_old_logps_pass_fits(
            self,
            rows=self.micro_batch_size or 1,
            priced_from="the micro-batch this launch asked for",
        )

    @property
    def game_arm(self) -> GameArm:
        """Return the registry entry this run is training."""
        return ARMS[self.arm]

    @property
    def vllm_max_model_length(self) -> int:
        """Return the context a colocated engine must hold: one prompt plus one full completion.

        Derived rather than configured, per the repo's never-hardcode-a-budget rule, and exactly
        sufficient: `games.dataset` truncates prompts at `max_prompt_tokens` and TRL stops
        generation at `max_completion_tokens`, so no sequence the engine sees can be longer than the
        sum. Sizing it larger only makes the engine reserve KV blocks nothing will ever fill, and
        sizing it smaller makes vLLM refuse the request mid-run.
        """
        return self.max_prompt_tokens + self.max_completion_tokens

    @property
    def requested_episodes_per_step(self) -> int:
        """Episodes the operator asked for, before the card gets a say."""
        return self.num_generations * self.prompts_per_step


def _vllm_arguments(config: GameTrainConfig) -> dict[str, Any]:
    """Return the vLLM arguments for TRL, present in every run's `training_args.bin`.

    A separate builder so the engine's argument set stays diffable as one unit across runs; there
    is no branch here because there is no other backend to pass arguments for.
    """
    return {
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "vllm_max_model_length": config.vllm_max_model_length,
        "vllm_importance_sampling_correction": config.vllm_importance_sampling_correction,
        # Passed even with the correction off, where TRL reads it nowhere: `training_args.bin` then
        # records the same key set for every arm, so two arms' estimator settings stay diffable.
        "vllm_importance_sampling_mode": config.vllm_importance_sampling_mode,
    }


def old_logps_pass_peak_gib(*, chunk_tokens: int, rows: int) -> float:
    """Price the fp32 logits the chunked old-log-probability pass holds LIVE at its peak, in GiB.

    Counted off `chunked_per_token_logps` and TRL's `selective_log_softmax` rather than assumed to be
    the widest single tensor, which is what an earlier version of this arithmetic priced and is about
    half the truth. Per slice, with `rows` rows in flight:

    *   `lm_head(hidden_slice)` builds `rows x chunk x vocab` fp32 under `cast_lm_head_to_fp32`
        (TRL's `cast_forward_to_fp32`, grpo_trainer.py:1007-1012), divided by the temperature in
        place so the undivided copy is not alive beside its quotient;
    *   inside `selective_log_softmax`'s fp32 branch, `torch.logsumexp(lg, dim=-1)` runs per row
        (trl/trainer/utils.py:507-510) and ATen's `logsumexp` materialises `(self - maxes).exp_()`,
        one more `chunk x vocab` fp32 tensor beside the slice it is reducing.

    So `rows + 1` chunk-sized fp32 copies, which at the single-row micro-batch of the 9B/32k
    production shape is twice the per-slice figure. The gather of the sampled ids is `rows x chunk`
    and rounds to nothing at these widths, and the resident fp32 head is a constant outside this
    pass (see `IS_CORRECTION_LOGITS_REFUSAL_GIB`).

    Four bytes per element unconditionally, which is what the 30.31 GiB the pass died at implies for a
    32,768-position row and what the fp32-head cast makes certain. Without the cast a bfloat16 head
    would halve both terms, so pricing fp32 either way errs toward refusing rather than toward a
    mid-run out-of-memory -- the same direction the vocabulary over-read errs in.
    """
    live_chunk_copies = rows + 1
    return live_chunk_copies * chunk_tokens * QWEN3_5_VOCAB_SIZE * BYTES_PER_FLOAT32 / BYTES_PER_GIB


def assert_old_logps_pass_fits(config: GameTrainConfig, *, rows: int, priced_from: str) -> None:
    """Refuse a correction-ON launch whose old-logps pass prices past the shape measured dead.

    Called twice per launch, both times before any weights load: once from the config's own
    validation against the micro-batch the operator asked for, and once from `_derive_plan` against
    the micro-batch `plan_sizing` derived from the card, because that is the number TRL passes as
    `batch_size` at the pass's call site (grpo_trainer.py:2635-2641) and the config cannot know it.
    `priced_from` names which of the two is speaking, so a refusal says whose number it refused.

    Smoke is exempt the way the budget floor exempts it: the shrunk budget is far under the line, and
    the pre-shrink intermediate config still carries the full budget this would otherwise reject.
    """
    if config.smoke or not config.vllm_importance_sampling_correction:
        return
    chunk_tokens = min(config.max_completion_tokens, config.old_logps_chunk_tokens)
    peak_gib = old_logps_pass_peak_gib(chunk_tokens=chunk_tokens, rows=rows)
    if peak_gib <= IS_CORRECTION_LOGITS_REFUSAL_GIB:
        return
    default_peak_gib = old_logps_pass_peak_gib(chunk_tokens=OLD_LOGPS_CHUNK_TOKENS, rows=1)
    raise ValueError(
        f"vLLM colocate with the importance-sampling correction ON recovers old log-probabilities "
        f"through a full-length forward pass, and this trainer computes it over "
        f"{chunk_tokens}-position slices whose live fp32 logits peak at ~{peak_gib:.1f} GiB "
        f"({rows + 1} copies of {chunk_tokens} positions x {QWEN3_5_VOCAB_SIZE} vocab x 4 bytes: one "
        f"per row of the {rows}-row micro-batch, from {priced_from}, plus one for the logsumexp "
        f"temporary inside selective_log_softmax; the completion budget of "
        f"{config.max_completion_tokens} is clamped by --old-logps-chunk-tokens "
        f"{config.old_logps_chunk_tokens}). TRL's OWN unchunked pass at a 32,768-token budget is one "
        f"such copy per row with no clamp, and it went out of memory at 30.31 GiB beside the "
        f"resident engine on a 95 GiB card (measured 2026-08-19, twice), even fully patched -- which "
        f"is the measurement that set the {IS_CORRECTION_LOGITS_REFUSAL_GIB} GiB line this refuses "
        f"past. It dies only AFTER paying for the first generation batch, so it is refused at "
        f"startup instead. The row count is the sizing plan's rather than a flag, so the levers are "
        f"--old-logps-chunk-tokens (the default {OLD_LOGPS_CHUNK_TOKENS} is ~{default_peak_gib:.2f} "
        f"GiB at one row and passes at every budget; halving it halves this figure), the completion "
        f"budget, or --no-vllm-importance-sampling-correction (or set "
        f"{VLLM_IS_CORRECTION_ENV}=0) and note the estimator caveat it carries. The last choice is "
        f"typed rather than defaulted because switching the estimator is a science decision. Where "
        f"the chunked pass actually peaks on a card is the five-step probe's measurement; this line "
        f"is the one shape that has been measured dead."
    )


def assert_cast_lm_head_is_supported(model_id: str, *, cast_lm_head_to_fp32: bool) -> None:
    """Refuse TRL's fp32-head knob on a checkpoint whose embeddings are tied to the head.

    TRL casts the head on the PeftModel (`model` is reassigned to it at grpo_trainer.py:447 and
    `_cast_lm_head_to_fp32(model)` runs at :1026), and its tied-embedding branch dereferences
    `target_model.model.embed_tokens` (:1024). One level below a PeftModel that path resolves to the
    causal-LM wrapper, whose embeddings live another level down, so the cast raises AttributeError
    inside `GRPOTrainer.__init__` for any checkpoint with `tie_word_embeddings=True` -- reproduced on
    CPU with a tiny tied model wrapped by `get_peft_model`. On this ladder that is every rung below
    the 9B: Qwen3.5-0.8B, 2B and 4B are tied, and the smoke checkpoint Qwen3-0.6B is too, while the
    9B (the only rung the cast has a shape for) is untied.

    Read here rather than left to TRL because the crash lands after the weights are down on a rented
    card, and because it reads as a bug in our own plumbing rather than as an upstream shape gap.
    The read is `AutoConfig`'s, off the same text sub-config `AutoModelForCausalLM` hands the model.
    """
    if not cast_lm_head_to_fp32:
        return
    if not AutoConfig.from_pretrained(model_id).get_text_config().tie_word_embeddings:
        return
    raise ValueError(
        f"--cast-lm-head-to-fp32 is refused on {model_id}, whose text config declares "
        f"tie_word_embeddings=True: TRL's fp32-head cast then registers a hook on "
        f"target_model.model.embed_tokens (grpo_trainer.py:1017-1026) against the PeftModel it was "
        f"handed, where that attribute does not exist, and GRPOTrainer.__init__ dies with an "
        f"AttributeError once the weights are already loaded. The cast works on an untied "
        f"checkpoint -- Qwen3.5-9B is the rung on this ladder that is -- so run the fp32-head probe "
        f"there, or drop the flag and read the sampler mismatch off a bfloat16 head with the "
        f"chunk-dependence caveat `chunked_per_token_logps` documents."
    )


class ColocateEngineSettings(Protocol):
    """What the colocate banner reads off a thread's train config, as an explicit contract.

    A protocol rather than `GameTrainConfig` because `reward_hacking.train` calls the banner with its
    own config, and the two dataclasses share only these three fields. The banner used to take
    `GameTrainConfig` and be handed `cast("Any", config)` from that call site, which is how it grew
    two reads (`old_logps_chunk_tokens`, `vllm_importance_sampling_log_only`) that only one of the
    two configs has: every reward_hacking launch with the correction ON -- its default -- died with
    an AttributeError in the banner. Read-only properties, so a plain dataclass field satisfies them.
    """

    @property
    def vllm_gpu_memory_utilization(self) -> float:
        """The share of the card the colocated engine holds for the life of the run."""
        ...

    @property
    def vllm_max_model_length(self) -> int:
        """The context the engine must hold: one prompt plus one full completion."""
        ...

    @property
    def vllm_importance_sampling_correction(self) -> bool:
        """Whether TRL corrects the vLLM-versus-trainer log-probability mismatch."""
        ...


def log_colocate_settings(  # noqa: PLR0913  -- one keyword per fact the banner may not read off a config
    config: ColocateEngineSettings,
    *,
    engine_reserved_gib: float,
    old_logps_chunk_tokens: int,
    old_logps_peak_gib: float,
    importance_sampling_log_only: bool,
    importance_sampling_mode: str = VLLM_IMPORTANCE_SAMPLING_MODE,
    dynamic_sampling_oversample: int = 1,
) -> None:
    """State what the colocate engine takes and what the estimator setting costs.

    Both branches of the importance-sampling correction are announced, because both have a price and
    neither is safe to leave implicit. Off, the gradient estimator is no longer exactly on-policy
    and every absolute level the run reports carries that caveat. On, TRL forces an extra
    full-length forward pass to recover old log-probabilities for every generation batch --
    unconditionally under vLLM, where the HF path skips it whenever generation and optimizer steps
    line up -- and at the production shape that pass tried to allocate 30.31 GiB of upcast fp32
    logits beside the resident engine and died (measured 2026-08-19, TRL 1.10, a 95 GiB card).

    The chunk width, its priced peak, the log-only flag and the correction mode are parameters rather
    than config reads because the two threads that call this describe them differently: games has
    them as launch knobs, and reward_hacking builds `PaddingTrimmedGRPOTrainer` without either
    instrument keyword and passes TRL no mode, so it runs the class and TRL defaults. Passing them in
    makes the banner describe the trainer that will actually be built either way, and keeps
    `ColocateEngineSettings` to what both configs really carry. The mode defaults to TRL's own for
    that reason: it is reward_hacking's fact rather than a value a games call site may omit.

    A contrast between two arms is unaffected by the choice either way, as long as both arms share
    it: the estimator is common to them, so the difference between them is not what it biases.
    """
    logger.info(
        "generating through a colocated vLLM engine, %s",
        f"gpu_memory_utilization={config.vllm_gpu_memory_utilization} "
        f"engine_reserved_gib={engine_reserved_gib:.1f} "
        f"vllm_max_model_length={config.vllm_max_model_length}",
    )
    if dynamic_sampling_oversample > 1:
        logger.warning(
            "dynamic sampling is ON at oversample %d: each optimizer step generates %dx its prompt "
            "groups, scores all of them into the trace and the reward metrics, and trains on the "
            "first live ones in sampler order, falling back to pure groups only to fill the batch. "
            "Generation cost scales with the oversample and training cost does not; the epoch "
            "counter now counts prompts DRAWN rather than prompts trained on.",
            dynamic_sampling_oversample,
            dynamic_sampling_oversample,
        )
    if config.vllm_importance_sampling_correction:
        logger.warning(
            "vLLM importance-sampling correction is ON in mode %s, so TRL will run an extra "
            "full-length forward pass per generation batch to recover old log-probabilities. At the "
            "production shape TRL's own pass needed 30.31 GiB of fp32 logits next to the engine "
            "and went out of memory; this trainer computes it over %d-position slices instead "
            "(~%.2f GiB of fp32 logits live at the peak, the same log-probabilities to the head "
            "matmul's own reassociation). If this run "
            "dies in _get_per_token_logps_and_entropies anyway, %s=0 is the pre-registered answer.",
            importance_sampling_mode,
            old_logps_chunk_tokens,
            old_logps_peak_gib,
            VLLM_IS_CORRECTION_ENV,
        )
        if importance_sampling_log_only:
            logger.warning(
                "log-only importance sampling: the ratio statistics are recorded and the weight "
                "the loss applies is exactly 1, so this run's gradient is the correction-OFF "
                "gradient and its sampling/* series measure the mismatch rather than remove it."
            )
        return
    logger.warning(
        "vLLM importance-sampling correction is OFF: rollouts are sampled by vLLM and graded with "
        "log-probabilities this model computes, and the mismatch between them is left "
        "uncorrected. Absolute levels from this run are not comparable with exactly-on-policy "
        "GRPO. A contrast against another arm sharing this setting is unaffected."
    )


def shrink_for_smoke(config: GameTrainConfig) -> GameTrainConfig:
    """Return the same config with every size knob reduced to smoke scale.

    A separate smoke script would exercise a separate code path, which is the one thing a smoke
    run must not do.
    """
    smoke_overrides = dict(SMOKE_OVERRIDES)
    if config.record_retention_manifest:
        smoke_overrides.update(save_steps=1, save_total_limit=0)
    return replace(config, smoke=True, corpus_path=None, **smoke_overrides)  # pyright: ignore[reportArgumentType]


def path_safe_model_id(model_id: str) -> str:
    """Turn a hub id into a path-safe fragment, keeping every character that carries meaning.

    Named for what it does rather than "slug", because the sequence plans in `games/` each carry
    their own zero-argument `model_slug()` that lowercases and strips dots -- a different string for
    the same input, naming the same run's artifacts. Two functions with one name is how a helper
    ends up resuming from an empty directory.
    """
    return model_id.replace("/", "-").replace(" ", "-")


def default_output_dir(arm: str, model_id: str, *, timestamp: datetime, smoke: bool = False) -> str:
    """Build the per-run artifact directory, one per arm/model/launch."""
    prefix = "smoke-" if smoke else ""
    stamp = timestamp.strftime("%Y%m%d-%H%M%S")
    return f"{RUN_ROOT}/{prefix}{arm}-{path_safe_model_id(model_id)}-{stamp}"


def resolve_resume_checkpoint(requested: str, output_dir: str) -> str | None:
    """Turn the `resume_from_checkpoint` setting into a path for `Trainer.train`, or None.

    `RESUME_LATEST` returning None on an empty output directory is deliberate, not a swallowed
    error: it makes "re-run the same command" correct both for the first launch and for a restart,
    which is the only recovery procedure an operator will reliably follow at 3am. An explicit path
    is validated instead, because a typo there means the run silently starts from scratch with a
    warm optimizer -- it would train, and the step numbers in its trace would be lies.

    What resuming actually restores is inherited from `transformers.Trainer`, which GRPOTrainer does
    not override: the LoRA adapter weights (`model.load_adapter(..., is_trainable=True)`), the
    optimizer and scheduler, the RNG states, `global_step` from `trainer_state.json`, and the
    dataloader fast-forwarded past the batches already seen.
    """
    if not requested:
        return None
    if requested == RESUME_LATEST:
        checkpoints = list(iter_checkpoints(Path(output_dir))) if Path(output_dir).is_dir() else []
        if not checkpoints:
            logger.info(f"no checkpoint to resume from in {output_dir}, starting a fresh run")
            return None
        latest = checkpoints[-1]
        logger.info(
            f"resuming from the latest checkpoint, {latest=} step={checkpoint_step(latest)}"
        )
        return str(latest)
    candidate = Path(requested)
    if not candidate.is_dir():
        raise ValueError(f"resume_from_checkpoint {requested!r} is not a directory")
    if not (candidate / TRAINER_STATE_FILENAME).is_file():
        raise ValueError(
            f"resume_from_checkpoint {requested!r} has no {TRAINER_STATE_FILENAME}, so it is not a "
            f"trainer checkpoint. Resuming from it would restart at step 0 with a warm optimizer "
            f"and report step numbers that never happened."
        )
    logger.info(f"resuming from an explicit checkpoint, {requested=}")
    return requested


# What `transformers.Trainer` must find in a checkpoint for a resume to restore what it claims to.
# The shared predicate also requires PEFT's adapter config, and checks direct children only, so the
# resolver and retention inventory cannot disagree about a nested or adapter-only directory.
REQUIRED_CHECKPOINT_FILES: tuple[str, ...] = RETENTION_REQUIRED_CHECKPOINT_FILES
ADAPTER_WEIGHT_FILENAMES: tuple[str, ...] = RETENTION_ADAPTER_WEIGHT_FILENAMES
INCOMPLETE_CHECKPOINTS_DIRNAME = "incomplete-checkpoints"


@dataclass(frozen=True)
class ResumeResolution:
    """Which checkpoint a launch resumes from, and which torn ones it had to step past to get there."""

    checkpoint: str | None
    # One entry per incomplete checkpoint newer than `checkpoint`: its name, step, missing files and
    # where it was moved. Written into the launch record, so a step that ran twice (a rewritten
    # completions parquet, a second mem_log.csv row) is attributable to this fallback afterwards.
    set_aside: tuple[dict[str, object], ...] = ()


def resolve_complete_resume_checkpoint(
    requested: str, output_dir: str, *, now: datetime | None = None
) -> ResumeResolution:
    """`resolve_resume_checkpoint` with a completeness gate, so a torn checkpoint cannot be resumed.

    Why the gate exists. `aws s3 sync` uploads a checkpoint's files concurrently and a spot reclaim
    syncs nothing, so a box that dies mid-sync leaves S3 holding a `checkpoint-N/` with some of its
    files -- typically the adapter and `trainer_state.json`, which are small, without the 128 MiB
    `optimizer.pt`. The restore pulls that directory back whole, the plain resolver returns it as
    the latest, transformers loads the adapter and step counter, finds no optimizer file, and
    continues with zeroed Adam moments while every downstream signal stays green. A run whose
    optimizer restarted at step N is a different experiment, and nothing recorded afterwards can
    tell it from the one that was meant.

    `RESUME_LATEST` therefore walks the checkpoints newest-first and resumes from the newest one
    that has every required file. Each incomplete checkpoint it steps past is logged at WARNING
    with the files it lacks, and moved under `<output_dir>/incomplete-checkpoints/<launch stamp>/`,
    so that the resumed run's own save of that step does not land on top of the torn remains, the
    eval batteries that glob `checkpoint-*` never read a checkpoint with no adapter, and the torn
    material stays on disk for a post-mortem. The walk stops at the first complete checkpoint:
    older incomplete ones are left alone, because an optimizer-light retention policy prunes
    exactly those on purpose. A directory with checkpoints but no complete one raises rather than
    starting fresh, since a fresh step 0 next to existing checkpoints is the same silent
    different-experiment outcome the gate exists to prevent. An explicit checkpoint path is refused
    when incomplete, not substituted: the operator named it, and a quiet swap is a lie about what
    was asked for.

    The same function, by the same name and shape, is what `reward_hacking.train` grew for its own
    resume in the same wave; this is the shared home the backlog asked for ("one resolver serves
    both trainers"), so that copy can become an import.
    """
    if requested != RESUME_LATEST:
        resolved = resolve_resume_checkpoint(requested, output_dir)
        if resolved is not None and (missing := missing_checkpoint_files(Path(resolved))):
            raise ValueError(
                f"resume_from_checkpoint {requested!r} is missing {missing}, so resuming from it "
                f"would restore the adapter and step counter but not the optimizer, scheduler or "
                f"sampling state that the recorded steps were trained under. Name a complete "
                f"checkpoint, or pass {RESUME_LATEST!r} to fall back to the newest complete one."
            )
        return ResumeResolution(checkpoint=resolved)

    run_dir = Path(output_dir)
    checkpoints = list(iter_checkpoints(run_dir)) if run_dir.is_dir() else []
    incomplete: list[tuple[Path, list[str]]] = []
    complete: Path | None = None
    for candidate in reversed(checkpoints):
        missing = missing_checkpoint_files(candidate)
        if not missing:
            complete = candidate
            break
        incomplete.append((candidate, missing))
    if complete is None and checkpoints:
        raise RuntimeError(
            f"{run_dir} holds {[path.name for path in checkpoints]} but none is a complete "
            f"checkpoint: "
            + "; ".join(f"{path.name} lacks {missing}" for path, missing in incomplete)
            + ". Starting fresh here would train a step 0 next to steps that already ran, so this "
            "needs a human: restore the missing files, or start a new --output-dir."
        )
    if complete is None or not incomplete:
        return ResumeResolution(checkpoint=resolve_resume_checkpoint(requested, output_dir))

    stamp = (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%SZ")
    aside_root = run_dir / INCOMPLETE_CHECKPOINTS_DIRNAME / stamp
    aside_root.mkdir(parents=True, exist_ok=True)
    set_aside: list[dict[str, object]] = []
    for torn, missing in incomplete:
        moved_to = aside_root / torn.name
        torn.rename(moved_to)
        set_aside.append(
            {
                "checkpoint": torn.name,
                "step": checkpoint_step(torn),
                "missing": missing,
                "moved_to": str(moved_to),
            }
        )
        logger.warning(
            "INCOMPLETE CHECKPOINT: %s lacks %s, so it cannot be resumed without silently "
            "restarting the optimizer; moved it to %s and falling back to %s (step %d). Step %d runs "
            "again under this launch: its completions parquet is rewritten, mem_log.csv gains a "
            "second row for it, and the launch record names this fallback.",
            torn.name,
            missing,
            moved_to,
            complete.name,
            checkpoint_step(complete),
            checkpoint_step(torn),
        )
    return ResumeResolution(
        checkpoint=resolve_resume_checkpoint(requested, output_dir), set_aside=tuple(set_aside)
    )


def describe_init_adapter(
    config: GameTrainConfig, *, lora_targets: dict[str, object]
) -> dict[str, object] | None:
    """Refuse an init adapter that is not the LoRA this run builds, and record which bytes seed it.

    Runs on CPU before any weights load. The identity conditions are exactly what makes "the same
    adapter, continued under a new reward" true rather than hoped: the base model (a sibling tier's
    adapter loads without complaint and is not the checkpoint you meant), the rank and alpha (the
    scaling factor alpha/r multiplies every delta, so a mismatch re-scales the whole init), and the
    target-module set (this family's DeltaNet projections are LoRA'd too, so a hand-listed q/k/v/o
    checkpoint would seed a quarter of the modules and silently zero the rest). DoRA, rsLoRA and
    per-module rank/alpha patterns change what the same tensors MEAN, so any of them refuses.
    """
    if not config.init_adapter:
        return None
    adapter_dir = Path(config.init_adapter)
    assert_adapter_matches_base(adapter_dir, config.load_source)
    identity = adapter_config_identity(adapter_dir)
    expected: dict[str, object] = {
        "r": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "target_modules": tuple(sorted(cast("list[str]", lora_targets["target_modules"]))),
    }
    mismatched = {
        field: {"adapter": identity.get(field), "this_run": want}
        for field, want in expected.items()
        if identity.get(field) != want
    }
    if mismatched:
        raise ValueError(
            f"init adapter {config.init_adapter!r} is not the LoRA this run builds: {mismatched}. "
            f"Seeding across a rank, alpha or target-set change would re-scale or partially drop "
            f"the checkpoint's deltas while every curve still renders."
        )
    weight_semantics_changers = ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern")
    engaged = sorted(flag for flag in weight_semantics_changers if identity.get(flag))
    if engaged:
        raise ValueError(
            f"init adapter {config.init_adapter!r} sets {engaged}, which change what its tensors "
            f"mean; this run builds a plain LoRA, so the same weights would compute a different "
            f"delta. Train from a plain-LoRA checkpoint, or build the run to match."
        )
    weights_path = adapter_dir / INIT_ADAPTER_WEIGHTS_FILENAME
    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    logger.info(
        "init adapter verified against this run's LoRA plan, %s",
        f"path={config.init_adapter!r} weights_sha256={digest}",
    )
    return {"path": str(adapter_dir), "weights_sha256": digest, "identity": identity}


def apply_init_adapter(model: object, init_adapter: str) -> dict[str, object]:
    """Seed the freshly built LoRA with the init checkpoint's weights, and prove they landed.

    Runs after TRL has built its PEFT model and before the first step, so what trains from step 0
    IS the checkpoint's adapter under the new arm's reward. Three checks, each against a failure
    that produces a full set of plausible artifacts rather than a crash: key-set equality both ways
    (PEFT's own loader only warns on a mismatch and would leave part of the model at zero init),
    read-back equality at the consumption point (the tensors the model now holds, not the call that
    was supposed to put them there), and a nonzero total ||B||^2 -- a fresh LoRA's B factors are
    exactly zero, so a zero here means the "seeded" run is bit-identical to a from-scratch one and
    the transfer contrast would be measuring nothing.
    """
    loaded = load_file(str(Path(init_adapter) / INIT_ADAPTER_WEIGHTS_FILENAME))
    current = get_peft_model_state_dict(cast("Any", model))
    missing = sorted(set(current) - set(loaded))
    unexpected = sorted(set(loaded) - set(current))
    if missing or unexpected:
        raise ValueError(
            f"init adapter {init_adapter!r} does not tile this run's LoRA: "
            f"{len(missing)} model slots would stay zero-initialised "
            f"(first: {missing[:ERROR_EXAMPLE_COUNT]}), {len(unexpected)} checkpoint weights "
            f"match no module (first: {unexpected[:ERROR_EXAMPLE_COUNT]}). A partial seed trains "
            f"a chimera of the checkpoint and a fresh init without a word."
        )
    set_peft_model_state_dict(cast("Any", model), loaded)
    after = get_peft_model_state_dict(cast("Any", model))
    stale: list[str] = []
    b_sq_norm = 0.0
    for name, tensor in loaded.items():
        landed = after[name].detach().to("cpu")
        if not torch.equal(landed, tensor.to(landed.dtype)):
            stale.append(name)
        if ".lora_B" in name:
            b_sq_norm += float(tensor.to(torch.float32).pow(2).sum())
    if stale:
        raise ValueError(
            f"init adapter {init_adapter!r} was set but did not land: {len(stale)} of "
            f"{len(loaded)} weights read back different from the file "
            f"(first: {stale[:ERROR_EXAMPLE_COUNT]})."
        )
    if b_sq_norm == 0.0:
        raise ValueError(
            f"init adapter {init_adapter!r} carries all-zero B factors, so the seeded model is "
            f"bit-identical to a fresh LoRA and this run's 'from checkpoint' label would be false."
        )
    facts: dict[str, object] = {"applied_weights": len(loaded), "lora_B_sq_norm": b_sq_norm}
    logger.info("init adapter applied, %s", facts)
    return facts


def read_recorded_launch(output_dir: str, *, checkpoint: str) -> dict[str, object]:
    """Return the `run_config.json` the checkpoint being resumed was trained under.

    Refuses when it is absent, because then there is nothing to check this launch against and the
    resume check would be a reassuring message rather than a check. A checkpoint without its launch
    record is also unreadable afterwards: the group size, corpus and grading its steps were trained
    under exist nowhere else.
    """
    path = Path(output_dir, RUN_CONFIG_FILENAME)
    if not path.is_file():
        raise RuntimeError(
            f"resuming {checkpoint} but {path} is missing, so nothing records what those steps "
            f"were trained under and this launch cannot be checked against them. Restore the "
            f"{RUN_CONFIG_FILENAME} that run wrote (the S3 sync ships it next to the checkpoints), "
            f"or start a fresh run in a new --output-dir rather than continuing an unattributable "
            f"one."
        )
    return cast("dict[str, object]", json.loads(path.read_text()))


def assert_resume_matches(
    *,
    recorded: Mapping[str, object],
    current: Mapping[str, object],
    fields: tuple[str, ...],
    checkpoint: str,
    consequence: str,
) -> None:
    """Refuse a resume that disagrees with the recorded launch on any of `fields`.

    A refusal rather than a re-plan. The values here are not settings a restart may pick afresh:
    they are what the checkpoint's existing steps were trained under, and the trainer will happily
    continue that adapter, optimizer and step counter under whatever this launch says instead.
    """
    differing = [
        f"{field}: checkpoint {recorded.get(field)!r}, this launch {current[field]!r}"
        for field in fields
        if recorded.get(field) != current[field]
    ]
    if not differing:
        return
    raise RuntimeError(
        f"refusing to resume {checkpoint}: this launch disagrees with the run that wrote it on "
        f"{differing}. {consequence} Resume with the recorded values, or start a fresh run in a "
        f"new --output-dir."
    )


def arm_identity(arm: GameArm) -> dict[str, object]:
    """Return the arm-level resume-identity fields, in the types JSON gives them back in.

    A tuple is compared against a recorded list, so the tuple is widened here rather than at the
    comparison: `assert_resume_matches` compares with `!=`, under which `("a",) != ["a"]` would refuse
    every resume of a breadth arm for a reason that has nothing to do with the experiment.
    """
    return {"game_ids": list(arm.game_ids)}


def load_corpus(path: str, arm: GameArm) -> list[dict[str, Any]]:
    """Read a selected-prompt corpus and refuse one that belongs to a different arm.

    Training `twin-pd-self` on a corpus selected for `twin-pd-group` would run, produce numbers,
    and answer a question nobody asked; the corpus files are per game and grading, so the check
    is cheap and the failure it prevents is invisible.

    A row's game must be one the arm names (`games.arms.arm_game_ids`: its own game, plus the extra
    games a breadth arm's corpus may carry). That is per arm rather than a blanket relaxation, so an
    arm that named no extra games keeps refusing every game but its own, and the breadth arm keeps
    refusing a seventh game somebody's rebuild left in the file -- a row of a game nothing in this run
    claims to train would be paid for, absorbed into the pooled cooperation rate, and named nowhere.
    """
    rows: list[dict[str, Any]] = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError(f"corpus {path} holds no rows")
    missing = [column for column in CORPUS_ARM_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(
            f"corpus {path} rows lack {missing}; expected the plan's prompt-row schema, got "
            f"columns {sorted(rows[0])}"
        )
    allowed_games = arm_game_ids(arm)
    mismatched = [
        (row.get("game_id"), row.get("grading"))
        for row in rows
        if row.get("game_id") not in allowed_games or row.get("grading") != arm.grading
    ]
    if mismatched:
        raise ValueError(
            f"corpus {path} holds rows for {sorted(set(mismatched))} but this arm trains "
            f"({list(allowed_games)}, {arm.grading!r}); {len(mismatched)} of {len(rows)} rows "
            f"mismatch"
        )
    # Corpora written before a backfillable column existed stay trainable: the marker is the exact
    # value every current row builder writes where the column does not apply, and the column's only
    # reader refuses the marker, so nothing can be silently mis-graded. Which columns qualify, and
    # why the bar for qualifying is deliberately high, lives on BACKFILLABLE_REWARD_COLUMNS.
    for column, marker in BACKFILLABLE_REWARD_COLUMNS.items():
        backfilled = 0
        for row in rows:
            if column not in row:
                row[column] = marker
                backfilled += 1
        if backfilled:
            logger.info(
                "corpus predates the %s column; backfilled the unset marker, %s",
                column,
                f"{path=} {backfilled=} {marker=}",
            )
    logger.info("loaded corpus, %s", f"{path=} n_rows={len(rows)} columns={sorted(rows[0])}")
    return rows


def filter_payoff_variants(
    rows: Sequence[dict[str, Any]], variants: Iterable[str]
) -> list[dict[str, Any]]:
    """Keep only rows whose payoff magnitude the arm is pinned to.

    Raises when the column is missing or the filter empties the corpus. Either would otherwise
    show up as a arm that trained on the wrong payoffs, or on nothing at all, without a word.
    """
    wanted = tuple(variants)
    if not wanted:
        return list(rows)
    if PAYOFF_VARIANT_COLUMN not in rows[0]:
        raise ValueError(
            f"arm pins payoff variants {wanted} but rows carry no {PAYOFF_VARIANT_COLUMN!r} "
            f"column, so the pin cannot be enforced; columns: {sorted(rows[0])}"
        )
    kept = [row for row in rows if row[PAYOFF_VARIANT_COLUMN] in wanted]
    if not kept:
        present = sorted({str(row[PAYOFF_VARIANT_COLUMN]) for row in rows})
        raise ValueError(f"pinning payoff variants {wanted} left no rows; corpus carries {present}")
    logger.info(
        "payoff-variant pin applied, %s",
        f"{wanted=} kept={len(kept)} dropped={len(rows) - len(kept)}",
    )
    return kept


def filter_corpus_partition(rows: Sequence[dict[str, Any]], partition: str) -> list[dict[str, Any]]:
    """Keep only the rows on the side of the group-mix boundary this arm is pinned to.

    Three refusals, each for a failure that would otherwise read as a completed run.

    A **missing column** means the corpus was never partitioned. `games.corpus_partition` stamps it,
    and an unstamped corpus loaded by a partitioned arm would train the whole thing while the run
    record claimed one side of a split.

    An **empty result** is a pin that matches nothing -- the partition was named but the corpus holds
    no rows on that side, which is what a partition whose pairs all straddled looks like.

    **Failing to narrow** is the quiet one: every row already carrying the pinned partition means the
    corpus was pre-narrowed elsewhere, so this arm's twin was trained on a file that this pin cannot
    distinguish from, and the contrast the two arms exist for was never set up. That case is a bug
    somewhere upstream rather than a smaller corpus, so it raises rather than passing through (the
    same reasoning as the merge that refuses to apply nothing).
    """
    if not partition:
        return list(rows)
    if CORPUS_PARTITION_COLUMN not in rows[0]:
        raise ValueError(
            f"arm pins corpus partition {partition!r} but rows carry no "
            f"{CORPUS_PARTITION_COLUMN!r} column, so the pin cannot be enforced and the whole "
            f"corpus would be trained under a partitioned arm's name. Freshly generated prompts "
            f"always look like this, and a mix-split arm cannot run on them: which side of the "
            f"boundary a prompt sits on is a measurement of the base policy, not a property of the "
            f"prompt. Sweep a corpus, stamp it with games.corpus_partition, and train that. "
            f"Columns: {sorted(rows[0])}"
        )
    kept = [row for row in rows if row[CORPUS_PARTITION_COLUMN] == partition]
    present = sorted({str(row[CORPUS_PARTITION_COLUMN]) for row in rows})
    if not kept:
        raise ValueError(
            f"pinning corpus partition {partition!r} left no rows; corpus carries {present}. A "
            f"partition whose counterbalanced pairs all straddled the boundary comes out empty like "
            f"this, and the partition artifact beside the corpus says how many did."
        )
    if len(kept) == len(rows):
        raise ValueError(
            f"every row of this corpus already carries partition {partition!r}, so the pin narrows "
            f"nothing and this arm's corpus is indistinguishable from its twin's. A mix-split needs "
            f"one stamped corpus holding both sides, not one file per side."
        )
    logger.info(
        "corpus-partition pin applied, %s",
        f"{partition=} kept={len(kept)} dropped={len(rows) - len(kept)} {present=}",
    )
    return kept


def assert_opponent_distribution_sampled(rows: Sequence[dict[str, Any]], arm: GameArm) -> None:
    """Refuse a vs-fixed-mix arm whose rows carry no frozen opponent to grade against.

    `games.prompts` writes `opp_coop_prob = -1` on every row it generates, and only the sweep's
    frozen-opponent pass fills it in, so `--generate-fresh` (which `--smoke` implies) produces a
    complete, non-empty, ungradeable corpus for these two arms. `games.rewards` does raise on it,
    but inside the reward function -- after the weights have loaded and one whole generation batch
    has been paid for. Checked on the assembled rows so the fresh path and the corpus path are both
    covered.
    """
    if arm.grading != GRADING_VS_FIXED_MIX:
        return
    without = [row["prompt_id"] for row in rows if "opp_coop_prob" not in row]
    if without:
        raise ValueError(
            f"{len(without)} of {len(rows)} rows carry no 'opp_coop_prob' column, so this "
            f"{GRADING_VS_FIXED_MIX} arm has no opponent distribution to grade against (e.g. "
            f"{without[0]!r}). Sweep the corpus with games.select_prompts "
            f"--frozen-opponent-model."
        )
    unsampled = [
        (row["prompt_id"], float(row["opp_coop_prob"]))
        for row in rows
        if not 0.0 <= float(row["opp_coop_prob"]) <= 1.0
    ]
    if unsampled:
        raise ValueError(
            f"{len(unsampled)} of {len(rows)} rows carry an opp_coop_prob outside [0, 1] (e.g. "
            f"{unsampled[0]}), which is what a row holds until the frozen opponent has been "
            f"sampled for it. A fresh generation always looks like this. Train this arm on a "
            f"corpus swept with games.select_prompts --frozen-opponent-model, which is the only "
            f"thing that measures the opponent these rows claim to be graded against."
        )


def assert_stated_match_mixture(rows: Sequence[dict[str, Any]], arm: GameArm) -> None:
    """Refuse a vs-stated-match arm whose corpus cannot make the counterpart clause matter.

    Two refusals, both for failures that would otherwise produce a full set of plausible artifacts.
    A row whose `stated_match_prob` is missing or outside [0, 1] has no stated correlation to grade
    against -- `games.rewards` does raise on it, but inside the reward function, after the weights
    have loaded and a generation batch has been paid for, exactly the vs-fixed-mix trap this
    mirrors. And a corpus whose rows all sit on ONE side of their own EV crossover is the fixed-p
    design wearing a mixture's label: some action is then unconditionally optimal across the whole
    corpus, the reward is once again a fixed function of the model's own action, and the arm would
    re-teach the counterpart-blindness it exists to avoid while every curve looked healthy. The
    optimal side is read from `stated_match_optimal_action` -- the same arithmetic the reward pays
    -- and a zero-gap cell (no optimal side) is refused outright, since no mixture can price it.
    """
    if arm.grading != GRADING_VS_STATED_MATCH:
        return
    invalid = [
        (row["prompt_id"], row.get("stated_match_prob"))
        for row in rows
        if not 0.0 <= float(row.get("stated_match_prob", STATED_MATCH_PROB_UNSET)) <= 1.0
    ]
    if invalid:
        raise ValueError(
            f"{len(invalid)} of {len(rows)} rows carry no stated_match_prob in [0, 1] (e.g. "
            f"{invalid[0]}; {STATED_MATCH_PROB_UNSET} marks a prompt stating no track record). "
            f"This {GRADING_VS_STATED_MATCH} arm has no stated correlation to grade against; "
            f"build the corpus with games.track_record_corpus."
        )
    sides = {"C": 0, "D": 0}
    for row in rows:
        spec = MatrixGameSpec(
            game_id=str(row["game_id"]),
            payoff_cc=float(row["payoff_cc"]),
            payoff_cd=float(row["payoff_cd"]),
            payoff_dc=float(row["payoff_dc"]),
            payoff_dd=float(row["payoff_dd"]),
        )
        optimal = stated_match_optimal_action(spec, float(row["stated_match_prob"]))
        if optimal is None:
            raise ValueError(
                f"prompt_id={row['prompt_id']!r} sits exactly on its EV crossover "
                f"(stated_match_prob={row['stated_match_prob']}), where both actions pay the same "
                f"and the cell carries no gradient at all. The corpus audit exists to drop such "
                f"cells; rebuild with games.track_record_corpus."
            )
        sides[optimal] += 1
    one_sided = [side for side, count in sides.items() if count == 0]
    if one_sided:
        raise ValueError(
            f"every row of this corpus makes the same action EV-optimal ({sides=}), so the stated "
            f"track record never changes the answer and the reward degenerates to a fixed function "
            f"of the model's own action -- the exact blindness-teaching design this arm exists to "
            f"avoid. The p mixture must straddle each game's EV crossover; rebuild with "
            f"games.track_record_corpus."
        )
    logger.info(
        "stated-match mixture verified, %s",
        f"coop_optimal_rows={sides['C']} defect_optimal_rows={sides['D']} n_rows={len(rows)}",
    )


def assert_corpus_selected_for_model(
    rows: Sequence[dict[str, Any]], *, path: str, model_id: str
) -> None:
    """Refuse a corpus selected against a different checkpoint than this run trains.

    The other half of a corpus's identity, next to its game and grading. `games.select_prompts`
    keeps only prompts whose action distribution was mixed at training temperature for ONE specific
    model, and for a vs-fixed-mix arm each row also carries `opp_coop_prob`, a cached measurement of
    ONE specific frozen opponent. A cross-model corpus passes every other check here: the symptom is
    a raised `frac_groups_pure` that reads as a property of the model rather than of the corpus, and
    on the vs-frozen arms an opponent distribution that was never sampled from the opponent the arm
    claims to face.

    A corpus with no stamp is warned about rather than refused, because the column is newer than the
    corpora on disk and the model is recorded there only in the filename.
    """
    stamped = {
        str(row[SELECTED_FOR_MODEL_COLUMN]) for row in rows if row.get(SELECTED_FOR_MODEL_COLUMN)
    }
    if not stamped:
        logger.warning(
            "this corpus carries no %r column, so which model it was selected for cannot be "
            "checked -- the pairing is convention, and the filename is the only record of it. %s",
            SELECTED_FOR_MODEL_COLUMN,
            f"{path=} training model_id={model_id!r}",
        )
        return
    foreign = sorted(stamped - {model_id})
    if foreign:
        raise ValueError(
            f"corpus {path} was selected for {foreign} but this run trains {model_id!r}. Prompt "
            f"selection keeps only what was mixed at training temperature FOR ONE MODEL, and a "
            f"vs-fixed-mix corpus also caches one specific frozen opponent's cooperation rate, so "
            f"training another checkpoint on it answers a question nobody asked while every other "
            f"check passes. Sweep a corpus for this model, or train the model it was swept for."
        )


def describe_row_composition(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    """Name the corners of the arm's corpus these rows actually cover.

    Read off the rows rather than from the registry, and only for the columns they carry, so this
    describes the corpus that was trained on including one narrowed by a pin or a subset.

    A row with no counterpart framing of its own -- the trust sender, whose paragraph is an announced
    return rule -- carries `games.rewards.FRAMING_ID_UNSET` and is described as that empty value
    rather than dropped, because "this corpus holds unframed rows" is part of its composition. What it
    is NOT is a framing the arm trained under, which is why `framings_trained` reads the same map and
    leaves the marker out. A corpus predating the column entirely is described without it: reading
    row 0 is enough because `games.dataset` refuses a corpus whose rows carry different columns.
    """
    return {
        column: sorted({str(row[column]) for row in rows})
        for column in ROW_COMPOSITION_COLUMNS
        if column in rows[0]
    }


def framings_trained(composition: Mapping[str, Sequence[str]]) -> list[str]:
    """Return the counterpart framings a corpus of this composition actually trained under.

    Derived from `describe_row_composition` rather than from a second pass over the rows, so the run
    record and the composition it is written beside cannot disagree about the same corpus. The unframed
    marker is dropped: the battery's framing table marks each framing trained or held out from this
    list, and a marker in it would claim a framing named after the absence of one.
    """
    return [
        framing for framing in composition.get(FRAMING_ID_COLUMN, ()) if framing != FRAMING_ID_UNSET
    ]


def subsample_rows(
    rows: Sequence[dict[str, Any]], *, max_prompts: int, seed: int
) -> list[dict[str, Any]]:
    """Take a seeded random subset of the corpus, and say which corners of it survived.

    Shuffled rather than sliced, because `games.prompts` emits rows grouped as frames times payoff
    variants times both label mappings: an unshuffled prefix can hold a single scenario reskin or a
    single payoff variant, so a debug run labelled as covering the arm would have trained on one
    corner of it. What survived is logged and recorded, since a skewed subset is only visible to a
    later reader if somebody wrote down what it contained.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    kept = shuffled[:max_prompts]
    logger.warning(
        "training on a seeded random subset of the corpus, %s",
        f"{max_prompts=} of n_rows={len(rows)} {seed=} "
        f"composition={describe_row_composition(kept)}",
    )
    return kept


def prepare_rows(config: GameTrainConfig) -> list[dict[str, Any]]:
    """Assemble the arm's prompt rows from a selected corpus or a fresh generation."""
    arm = config.game_arm
    if config.corpus_path is not None:
        rows = load_corpus(config.corpus_path, arm)
        assert_corpus_selected_for_model(rows, path=config.corpus_path, model_id=config.model_id)
    else:
        logger.warning(
            "generating prompts fresh, with no baseline selection sweep: every prompt whose "
            "action distribution is already unanimous contributes no gradient. %s",
            f"{arm.game_id=} {arm.grading=}",
        )
        rows = generate_prompt_rows(arm.game_id, arm.grading, split="train")
    rows = filter_payoff_variants(rows, arm.payoff_variants)
    rows = filter_corpus_partition(rows, arm.corpus_partition)
    assert_opponent_distribution_sampled(rows, arm)
    assert_stated_match_mixture(rows, arm)
    if config.max_prompts is not None and len(rows) > config.max_prompts:
        rows = subsample_rows(rows, max_prompts=config.max_prompts, seed=config.seed)
    return rows


def group_mix_fixed_points(
    rows: Sequence[dict[str, Any]],
    *,
    grading: str,
    num_generations: int,
    leave_one_out: bool,
) -> dict[str, float | None]:
    """Predict where each game in this corpus settles under the grading mode the run will use.

    Recorded because the prediction is the arm: chicken-group's whole job is to land on a specific
    interior cooperation rate, and `--leave-one-out` moves that rate (0.50 to about 0.31 at a group
    of 8) while every number written down in the registry, in `games.payoffs` and in
    docs/games-predictions.md is the plain group-mix one. A run that carries its own prediction
    cannot be compared against the wrong one later.

    None against a game means it has no stable interior rate to settle at, which is the normal case:
    a dominance game runs to a corner, and so does a stag hunt, whose interior crossing repels.
    Derived per game and payoff variant off the rows actually being trained on.
    """
    if grading != GRADING_GROUP_MIX:
        return {}
    specs = {
        f"{row['game_id']}--{row[PAYOFF_VARIANT_COLUMN]}": MatrixGameSpec(
            game_id=str(row["game_id"]),
            payoff_cc=float(row["payoff_cc"]),
            payoff_cd=float(row["payoff_cd"]),
            payoff_dc=float(row["payoff_dc"]),
            payoff_dd=float(row["payoff_dd"]),
        )
        for row in rows
    }
    fixed_points = {
        key: group_mix_fixed_point(
            spec, num_generations=num_generations, leave_one_out=leave_one_out
        )
        for key, spec in specs.items()
    }
    if leave_one_out:
        pre_registered = {
            key: group_mix_fixed_point(spec, num_generations=num_generations)
            for key, spec in specs.items()
        }
        logger.warning(
            "--leave-one-out grades each completion against the other %d, which MOVES every "
            "group-mix fixed point away from the numbers in the arm registry, games.payoffs and "
            "docs/games-predictions.md. %s",
            num_generations - 1,
            f"under_this_run={fixed_points} {pre_registered=}",
        )
    logger.info("predicted group-mix fixed points: %s", fixed_points)
    return fixed_points


# The trust games' behavioural number, named once because two things require it: the trust gradings by
# name, and a care-family run whose corpus carries trust rows. `games.rewards._log_trust_metrics`
# writes this key, and a literal restated in the second place could drift from it silently.
TRUST_BEHAVIOURAL_METRICS: tuple[str, ...] = ("mean_send_fraction",)

# The behavioural metrics each grading must have recorded, beside `REQUIRED_METRICS`. A table rather
# than a chain of returns: that chain is where every new game adds a branch, it had outgrown the
# return-count limit by wave 2, and a table makes the property that actually matters legible at a
# glance -- no two gradings sharing a behavioural key that either could pass the gate on.
BEHAVIOURAL_METRICS_BY_GRADING: dict[str, tuple[str, ...]] = {
    GRADING_KEEP_FRACTION: ("mean_keep_fraction",),
    # Each claim arm requires its OWN overreach key, never one shared one: the same arithmetic is a
    # collision rate against the group's realised claims under group-mix grading and the deterministic
    # over-claiming indicator under self grading, so one shared key would let either arm pass the
    # read-back gate on the other's quantity (games.rewards._log_claim_metrics).
    GRADING_NASH_DEMAND_GROUP_MIX: ("mean_claim_fraction", "crash_rate"),
    GRADING_NASH_DEMAND_SELF: ("mean_claim_fraction", "overclaim_rate"),
    # The shared undertaking's pair splits its reach key the same way and for the same reason, with a
    # sharper stake on the group-mix side: `threshold_met_rate` is that arm's gradient supply, because
    # a group whose pooled figures never clear the bar and one whose always do both see a constant
    # prize term and are left training contributions to zero on the cost term alone.
    GRADING_THRESHOLD_GOODS_GROUP_MIX: ("mean_contribution_fraction", "threshold_met_rate"),
    GRADING_THRESHOLD_GOODS_SELF: ("mean_contribution_fraction", "equal_share_met_rate"),
    **dict.fromkeys(TRUST_GRADINGS, TRUST_BEHAVIOURAL_METRICS),
    # The stated-track-record arm's own three, beside the shared cooperation rate. The optimum
    # rate IS this arm's prediction -- a policy over p, not a blanket behavior -- and the split
    # cooperation rates keep the two incentive directions apart, because the corpus deliberately
    # mixes cells where cooperating pays with cells where defecting pays and the pooled rate
    # averages opposite predictions. A run recording only the pooled rate could not be scored
    # against its own registered clause afterwards, only re-run.
    GRADING_VS_STATED_MATCH: (
        "coop_rate",
        "ev_optimum_rate",
        "coop_rate_where_coop_pays",
        "coop_rate_where_defect_pays",
    ),
    # The behavioural mean, the number the pre-registration is scored against, and the gradient
    # supply. `mean_target_level` is required rather than optional because it IS this arm's
    # prediction: a run that recorded where the group went but not where the payoffs pointed cannot be
    # scored against its own registered clause afterwards, only re-run. `mean_upward_pressure` is the
    # sign of the gradient, which separates "the group is sitting where the payoffs want it" from "the
    # arm has no signal" -- the same distinction the claim game's crash rate draws, and two readings
    # that call for opposite responses.
    GRADING_MIN_EFFORT_GROUP_MIX: (
        "mean_level_fraction",
        "mean_target_level",
        "mean_upward_pressure",
    ),
    # The end-game drop rate is required, not optional: against a level-matcher a drop only ever loses
    # money, so it IS this arm's headline reading, and an arm that logged its return but not its
    # final-round behaviour would have measured the total while missing the diagnostic.
    GRADING_LEVEL_MATCH_RETURN: ("mean_level_fraction", "end_game_drop_rate"),
}

# Every grading absent from the table answers with one of two labels, so its behavioural number is the
# cooperation rate. A default rather than an exhaustive map because that is the truth about this axis:
# a two-action game's behaviour IS its cooperation rate, and the games that need naming are the ones
# answering with a figure.
DEFAULT_BEHAVIOURAL_METRICS: tuple[str, ...] = ("coop_rate",)


def required_metrics_for(
    grading: str, *, announced_rule_trust_rows: bool = False
) -> tuple[str, ...]:
    """Return the metrics a run of this grading must have recorded, or it measured nothing.

    Arm-conditional because the behavioural axis is: a keep-fraction arm has no cooperation rate
    (there is no opponent), a claim arm has neither, a shared-undertaking arm's behaviour is the fraction
    of its stock it put in, a trust arm's is the fraction it sent, and none of them has anything to say
    about another's number. A single shared tuple
    made them all optional, which left the dictator arm's ONLY behavioural number outside the
    read-back gate -- the same shape as the dead-metric bug this whole module is built around, and
    the reason each new grading has to name its own number here rather than inherit `coop_rate` and
    pass the gate on a metric it never logs.

    Each claim arm additionally requires its own overreach metric, which is not a nicety: it is the
    only source of downward pressure on the claim, so without it a run that walked to the top of the
    grid cannot be told from one the payoffs held there. The two arms require DIFFERENT keys because
    the same arithmetic measures different things (see games.rewards._log_claim_metrics): a collision
    rate against the group's realised claims under group-mix, and the deterministic over-claiming
    indicator under self grading. Requiring one shared key would let either arm pass the read-back
    gate on the other's quantity.

    The shared undertaking's pair splits the same way and for the same reason, with a sharper stake on
    the group-mix side: `threshold_met_rate` is that arm's gradient supply, because a group whose pooled
    figures never clear the bar and one whose always do both see a constant prize term and are left
    training contributions to zero on the cost term alone.

    `announced_rule_trust_rows` is the one thing this cannot answer from the grading name. The care
    family is the only grading covering two row types at once, so a mixed care corpus produces the
    cooperation rate from its matrix rows and `mean_send_fraction` from its trust rows, and requiring
    only the first would leave the trust sender's ONLY behavioural number outside the read-back gate --
    the dictator arm's bug, which is what this table exists because of. The caller reads the flag off
    the corpus rather than off the arm, so a care run whose trust rows were all dropped at selection
    is not asked for a metric it cannot produce.
    """
    behavioural = BEHAVIOURAL_METRICS_BY_GRADING.get(grading, DEFAULT_BEHAVIOURAL_METRICS)
    if care_alpha_of(grading) is not None and announced_rule_trust_rows:
        behavioural = (*behavioural, *TRUST_BEHAVIOURAL_METRICS)
    return (*REQUIRED_METRICS, *behavioural)


def dataset_carries_announced_rule_trust_rows(dataset: Dataset) -> bool:
    """Report whether any row of this corpus is a trust row whose prompt announces a return rate.

    Read off the same `stated_return_fraction` column the reward function dispatches the care family
    on, rather than from the arm's game list, so the answer describes the corpus that was actually
    trained: a care corpus whose trust rows were all dropped at selection must not be asked for the
    send metric its rows would have produced.
    """
    if STATED_RETURN_FRACTION_COLUMN not in dataset.column_names:
        return False
    return any(
        float(cast("float", value)) != STATED_RETURN_UNSET
        for value in dataset[STATED_RETURN_FRACTION_COLUMN]
    )


def _metric_series(history: Sequence[dict[str, object]], key: str) -> list[float]:
    """Pull one metric's values out of trainer log history, in step order."""
    return [float(cast("float", record[key])) for record in history if key in record]


def read_back_metrics(
    trainer: GRPOTrainer, *, required: tuple[str, ...], constant_by_construction: tuple[str, ...]
) -> tuple[dict[str, object], list[str]]:
    """Read every claimed metric back out of trainer state, and report what never arrived.

    The repo's answer to its own dead-dict bug: a callback that writes into `Trainer.log`'s
    already-copied dict logs nothing, and the run still looks clean. The games metrics route
    through TRL's native `log_metric` kwarg instead, which lands in `log_history` — but "it
    logged" is a claim until it has been read back, which is what this does.

    `required` has no default: which behavioural metric an arm must produce depends on how it is
    graded (`required_metrics_for`), and a default would quietly let a caller check the wrong set.
    `constant_by_construction` has none for the same reason one step further out: which series a run
    pins depends on its parse-penalty mode (`constant_by_construction_metrics`), and a default would
    either warn about the mode's own arithmetic on every constant-penalty arm or hide a genuinely dead
    denominator on a row-relative one.
    """
    history = cast("list[dict[str, object]]", trainer.state.log_history)
    rewarded = [record for record in history if "reward" in record]
    if not rewarded:
        raise RuntimeError("no training record carried a reward; nothing was measured")

    # De-duplicated because the behavioural metrics appear in both tuples: `required` names the one
    # this arm's grading produces, and OPTIONAL_METRICS keeps both so the other still lands in the
    # summary. Iterating the concatenation naively reports a missing one twice.
    watched = (*required, *(key for key in OPTIONAL_METRICS if key not in required))
    summary: dict[str, object] = {}
    missing: list[str] = []
    for key in watched:
        series = _metric_series(history, key)
        if not series:
            if key in required:
                missing.append(key)
            continue
        flat_key = key.replace("/", "_")
        summary[f"{flat_key}_final"] = series[-1]
        summary[f"{flat_key}_mean"] = sum(series) / len(series)
    summary["n_logged_steps"] = len(rewarded)
    summary["logged_keys"] = sorted({key for record in history for key in record})
    # A metric that never moves across a whole run is the signature of a dead one: it satisfies a
    # presence check while measuring nothing, which is exactly how frac_reward_zero_std survived
    # three 70-step arms. Reported rather than raised, because some metrics legitimately hold still
    # (a constant learning-rate schedule, a truncation rate that is genuinely zero). The one class
    # exempted is the handful whose constancy IS the reading, which the caller names because it depends
    # on the run's parse-penalty mode (`constant_by_construction_metrics`).
    constant = [
        key
        for key in watched
        if key not in constant_by_construction
        and len(_metric_series(history, key)) > 1
        and len(set(_metric_series(history, key))) == 1
    ]
    summary["constant_metrics"] = constant
    if constant:
        logger.warning(
            "these metrics never varied across the run, so treat them as unmeasured until "
            "checked: %s",
            constant,
        )
    return summary, missing


def peak_memory_gib(output_dir: str) -> dict[str, object]:
    """Report peak VRAM from both the allocator and the memory-monitor CSV.

    Two sources because they answer different questions: the allocator knows what this process
    reserved, while `mem_log.csv` records what the whole card was holding, which is what decides
    whether the next configuration fits.
    """
    peaks: dict[str, object] = {
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / BYTES_PER_GIB,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / BYTES_PER_GIB,
    }
    mem_log = load_mem_log(output_dir)
    if mem_log is not None and "used_gib" in mem_log:
        peaks["peak_device_used_gib"] = float(cast("float", mem_log["used_gib"].max()))
    return peaks


def run_config_payload(
    config: GameTrainConfig,
    *,
    plan: SizingPlan,
    device: dict[str, object],
    derived: dict[str, object],
) -> dict[str, object]:
    """Assemble everything needed to reproduce or interpret this run, before it starts."""
    arm = config.game_arm
    composition = cast("Mapping[str, Sequence[str]]", derived.get("corpus_composition", {}))
    return {
        "arm": config.arm,
        "game_id": arm.game_id,
        # The OTHER games this arm's corpus may carry, so the trained set is `game_id` plus these
        # (`games.arms.arm_game_ids` assembles the same set from the registry, and
        # `games.run_evals` reads both keys to classify a battery row as trained or transfer). Empty
        # for every single-game arm, which is every arm before wave 4b. Recorded from the registry
        # rather than from the corpus, because it is what the run ALLOWED: which games the file
        # actually held is `derived.corpus_composition.game_id`, and the two disagreeing is a corpus
        # that came in short rather than an arm that changed.
        "game_ids": list(arm.game_ids),
        # Which counterpart framings this corpus trained under, read off the rows. On every earlier
        # arm "never trained under this framing" was true by construction, because the training
        # renderer refused a framing clause at all; on a breadth arm it is true per run, so the run
        # has to say it or a later framing-sweep readout has nothing to mark its table against.
        "trained_framing_ids": framings_trained(composition),
        "grading": arm.grading,
        # The care family's weight on the counterpart's payoff, and None for every other grading.
        # Recorded as its own field rather than left to be re-parsed out of the grading name, because
        # every readout, the box's config watcher and the control arm's comparison all key on the
        # number: `care-alpha-0` and `care-alpha-1` are the wave-4b pair, and a run whose record
        # carried only the name would make the treatment a string match somewhere downstream.
        "care_alpha": care_alpha_of(arm.grading),
        # Beside the grading because it is part of it: how the reward priced an unparseable
        # completion. Top-level so `games.run_evals` can carry it into eval-trace meta the way it
        # carries `grading`, and so the box's config watcher reads it where the trainer wrote it.
        "parse_penalty_mode": arm.parse_penalty_mode,
        "arm_notes": arm.notes,
        "payoff_variants": list(arm.payoff_variants),
        # What the estimator DID, not what loss_type names: under Liger the two can differ (the
        # pre-2026-08-20 arms all recorded "dapo" while executing per-sequence GRPO). Top-level so
        # `games.run_evals` can carry it into every eval trace's meta the way it carries `grading`.
        "executed_estimator": executed_estimator(
            config.loss_type,
            use_liger_kernel=config.use_liger_kernel,
            per_device_train_batch_size=plan.micro_batch_size,
        ),
        "config": asdict(config),
        "sizing_plan": asdict(plan),
        "device": device,
        "derived": derived,
        # Both fields, from the one place that answers this for every games artifact: a record
        # naming only a commit cannot be told from one produced by an edited checkout, and a corpus
        # written in the same session already carries the dirty flag.
        **git_provenance(),
        "started_at": datetime.now(tz=UTC).isoformat(),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Persist one artifact, creating its directory. Shared by both trainers.

    ``encoding="utf-8"`` explicitly, because `write_text` otherwise takes the locale's preferred
    encoding: on a box whose locale is not UTF-8 a run record carrying a non-ASCII character would
    fail to write, or write bytes nothing downstream can read back. The reward-hacking copy of this
    function had the encoding and this one did not, which is exactly the drift that made two copies
    worth collapsing into one.

    Written to a sibling temp file and renamed into place, because `run_config.json` is rewritten
    once the trainer is built (the attention backends are only knowable then) while the box's config
    watcher and the S3 sync may be reading it; a rename is atomic where a truncate-and-write is not.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f"{path.name}.tmp")
    staging.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    staging.replace(path)
    logger.info("wrote %s", path)


def widen_generation_batch_for_oversample(args: GRPOConfig, *, oversample: int) -> GRPOConfig:
    """Widen the generation batch to the groups dynamic sampling generates, leaving the split alone.

    `GRPOConfig.__post_init__` derives `generation_batch_size` and `steps_per_generation` from each
    other and refuses both as inputs (grpo_config.py:1083-1101), and the two do different jobs here:
    `generation_batch_size` sizes the sampler's prompt chunk (grpo_trainer.py:1277-1283) and the
    completions buffers that become the parquet trace (:1069-1074), which both have to hold the whole
    over-generated batch, while `steps_per_generation` decides how OFTEN TRL generates and into how
    many micro-batches it splits what `_generate_and_score_completions` returns (:1584-1592), which
    must not move: at twice the accumulation count one generation would feed two optimizer steps.
    So the widening happens here, after the derivation, and only to the first of the two.

    Returns the same object it was handed, mutated, because `GRPOConfig` is what TRL reads at
    construction time and a copy would leave the trainer on the narrow one.
    """
    if oversample == 1:
        return args
    narrow = cast("int", args.generation_batch_size)
    args.generation_batch_size = narrow * oversample
    logger.info(
        "dynamic sampling: generation batch widened %d -> %d rows at oversample %d, "
        "steps_per_generation left at %s so one generation still feeds exactly one optimizer step",
        narrow,
        args.generation_batch_size,
        oversample,
        args.steps_per_generation,
    )
    return args


def _build_grpo_config(
    config: GameTrainConfig, plan: SizingPlan, *, dtype: torch.dtype
) -> GRPOConfig:
    """Translate the arm's config and its sizing plan into TRL's arguments."""
    return widen_generation_batch_for_oversample(
        _trl_arguments(config, plan, dtype=dtype),
        oversample=config.dynamic_sampling_oversample,
    )


def _trl_arguments(config: GameTrainConfig, plan: SizingPlan, *, dtype: torch.dtype) -> GRPOConfig:
    """Build the arguments TRL derives its own batch shape from, before any widening."""
    return GRPOConfig(
        output_dir=cast("str", config.output_dir),
        run_name=f"{config.arm}-{path_safe_model_id(config.model_id)}",
        seed=config.seed,
        # Both derived rather than asserted: TrainingArguments rejects tf32 without an Ampere-plus
        # card and rejects bf16 without bf16 support, so hardcoding either makes this builder
        # unconstructable off a GPU box -- including in the offline tests. Asked of the same
        # function TrainingArguments validates against, because CUDA being present does not mean
        # TF32 is available: a pre-Ampere rental (T4, V100) would otherwise die in this
        # constructor, after the tokenizer, corpus, templating and LoRA discovery had all run.
        tf32=is_torch_tf32_available(),
        bf16=(dtype == torch.bfloat16),
        # Leaving steps_per_generation and generation_batch_size at None makes TRL derive one
        # generation of micro_batch * grad_accum episodes per optimizer step, which is what the
        # completions-parquet buffer is sized to and therefore what keeps the trace hole-free.
        per_device_train_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        num_generations=plan.num_generations,
        max_completion_length=config.max_completion_tokens,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        # warmup_ratio was removed; warmup_steps now reads a float in [0, 1) as a ratio.
        warmup_steps=config.warmup_ratio,
        # Passed rather than left to transformers' default because on LoRA the default is not
        # neutral: the A matrices' second moments sit below 1e-8 (GameTrainConfig.adam_epsilon
        # carries the measurement), so the floor and not the gradient sets their step size.
        adam_epsilon=config.adam_epsilon,
        max_steps=config.max_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=config.logging_steps,
        # The per-completion rollout trace, written to <output_dir>/completions/*.parquet
        # regardless of report_to. It carries every log_extra column the reward function adds.
        log_completions=True,
        # The Rich table TRL also prints under log_completions (None = every completion, ~27 MB of
        # stdout and ~11 s per step in the v2 run, 1.9 GB per run) is not the trace; the parquet is
        # written unconditionally in the same block, so nothing the readout reads is lost.
        num_completions_to_print=0,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=False,
        # TRL evaluates by computing the GRPO surrogate loss over eval prompts, which segfaults
        # inside Liger's fused loss. The eval battery reads checkpoints instead.
        eval_strategy="no",
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        beta=config.beta,
        epsilon=config.epsilon,
        scale_rewards=config.scale_rewards,
        loss_type=config.loss_type,
        # A completion that ran out of tokens mid-thought earns the parse penalty and keeps its
        # gradient. That pressure toward shorter reasoning is itself something to watch, and
        # masking it away would hide it.
        mask_truncated_completions=False,
        use_liger_kernel=config.use_liger_kernel,
        cast_lm_head_to_fp32=config.cast_lm_head_to_fp32,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="none",
        disable_tqdm=True,
        model_init_kwargs={
            # `dtype`, not `torch_dtype`: TRL ignores the latter in favour of its own key and the
            # model would load in float32 while bf16=True told the trainer to autocast.
            "dtype": dtype,
            # No device_map: "auto" shards across every visible card, which collides with the
            # two-GPU topology that puts a vLLM server on GPU1.
            "trust_remote_code": True,
        },
        **_vllm_arguments(config),
    )


def build_callbacks(*, logging_steps: int, output_dir: str, s3_dest: str) -> list[TrainerCallback]:
    """Assemble the trainer callbacks, adding the S3 shipper only when a destination is set.

    The S3 sync is for liveness: it ships on every checkpoint save, so a job killed mid-run leaves
    everything through its last save in the bucket. `cloud/entrypoint.sh` also syncs once from an
    exit trap, which is the completeness net; the two are complementary rather than redundant.

    Shared by both trainers, and taking three primitives rather than either config for that reason:
    their config types are unrelated and this needs three fields out of them. What the sharing buys is
    that a callback added for one research thread cannot be added for one thread only --
    `NonFiniteMetricCallback` was very nearly exactly that.
    """
    callbacks: list[TrainerCallback] = [
        MemoryMonitorCallback(
            print_every=max(logging_steps, 5), extra_columns=MEM_LOG_EXTRA_COLUMNS
        ),
        RewardLoggingCallback(),
        NonFiniteMetricCallback(),
    ]
    if s3_dest:
        callbacks.append(S3SyncCallback(local_dir=Path(output_dir), s3_dest=s3_dest))
        logger.info("s3 liveness sync enabled, %s", f"{s3_dest=}")
    return callbacks


@dataclass(frozen=True)
class PreparedRun:
    """Everything one launch settled before any weights loaded, including its own record on disk.

    Assembled by `_prepare_run` and consumed by everything after it, so that the order those steps
    happen in is visible in one short function rather than spread through a 230-line body. The
    ordering is the point: two of this module's worst defects were steps in the wrong order.
    """

    config: GameTrainConfig
    plan: SizingPlan
    dataset: Dataset
    tokenizer: PreTrainedTokenizerBase
    lora_targets: dict[str, object]
    derived: dict[str, object]
    dtype: torch.dtype
    device: dict[str, object]
    resume_checkpoint: str | None
    # The launch record this run wrote (`run_config.json`, or the resume-stamped sibling), so the
    # facts only a built trainer can supply are appended to the same file rather than a second one.
    record_path: Path

    @property
    def output_dir(self) -> str:
        """The run directory, resolved by the time a run is prepared."""
        return cast("str", self.config.output_dir)

    @property
    def prefilled_think(self) -> bool:
        """Whether this tokenizer's template opens `<think>` inside the prompt."""
        return cast("bool", self.derived["prefilled_think"])


def _announce_launch(config: GameTrainConfig) -> int:
    """Log what this run is and refuse a launch that cannot work; return the process count.

    The fp32-head refusal lands here rather than in `GameTrainConfig.__post_init__` because it reads
    the checkpoint's own config, and a hub read does not belong in a dataclass constructor -- but it
    stays as early as this, before the resume resolve and long before any weights load, because the
    crash it replaces happens inside `GRPOTrainer.__init__` on a rented card.
    """
    assert_cast_lm_head_is_supported(
        config.load_source, cast_lm_head_to_fp32=config.cast_lm_head_to_fp32
    )
    arm = config.game_arm
    logger.info(
        "training arm, %s",
        f"{config.arm=} game_id={arm.game_id!r} grading={arm.grading!r} "
        f"model_id={config.model_id!r} model_source={config.load_source!r} "
        f"output_dir={config.output_dir!r}",
    )
    logger.info("arm rationale: %s", arm.notes)
    # The spread table is part of every launch record rather than an on-demand check: under the
    # default scale_rewards="none" its numbers ARE the advantage magnitudes the run trains on,
    # and under "batch" they say which compressed game trains weaker than a wide one sharing its
    # batch divisor.
    logger.info("within-group reward spread by game/variant:\n%s", reward_spread_report())
    if not config.thinking:
        for line in (
            "=" * 78,
            "PLUMBING RUN -- THINKING IS OFF. This exercises the training path end to end and",
            "will move reward, but it is NOT a behavioural result: the decision-theory reasoning",
            "these arms exist to measure lives in the chain of thought, and there is no chain of",
            "thought here. run_config.json records plumbing_thinking_off=true. Do not read any",
            "cooperation rate, decision-theory shift or CoT excerpt off this run.",
            "=" * 78,
        ):
            logger.warning(line)
    world_size = detected_world_size(os.environ)
    assert_single_process(world_size, source="launch environment")
    return world_size


def _derive_plan(
    config: GameTrainConfig,
    *,
    device: dict[str, object],
    dtype: torch.dtype,
    param_count: int,
) -> tuple[SizingPlan, SequenceCost]:
    """Size the batch from the VRAM this process actually found, never from a constant.

    `param_count` is passed in rather than read here because instantiating the architecture on the
    meta device is uncached and this is not the only caller that wants the number.

    The colocate banner and the old-logps refusal both come AFTER the plan rather than before it,
    because both are priced per micro-batch row and the row count is what the plan derives: announced
    or refused against `config.micro_batch_size` they would describe a shape the run may not have.
    """
    cost = checkpoint_sequence_cost(config.model_id)
    engine_reserved_gib = colocate_reserved_gib(
        total_vram_gib=cast("float", device["total_vram_gib"]),
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
    )
    plan = plan_sizing(
        num_generations=config.num_generations,
        prompts_per_step=config.prompts_per_step,
        micro_batch_size=config.micro_batch_size,
        max_prompt_tokens=config.max_prompt_tokens,
        max_completion_tokens=config.max_completion_tokens,
        cost=cost,
        free_vram_gib=cast("float", device["free_vram_gib_at_start"]),
        weights_gib=param_count * dtype.itemsize / BYTES_PER_GIB,
        usable_fraction=config.vram_usable_fraction,
        autosize=config.autosize,
        label=f"{config.model_id} on {device['device_name']}",
        engine_reserved_gib=engine_reserved_gib,
        # The episodes decode inside the engine's reservation, subtracted above; charging the
        # per-episode decode arithmetic on top of it would double-count the same memory.
        generation_schedules_own_batch=True,
    )
    logger.info("sizing plan: %s", plan.reason)
    logger.info("sizing plan detail, %s", f"{plan=}")
    chunk_tokens = min(config.max_completion_tokens, config.old_logps_chunk_tokens)
    log_colocate_settings(
        config,
        engine_reserved_gib=engine_reserved_gib,
        old_logps_chunk_tokens=chunk_tokens,
        old_logps_peak_gib=old_logps_pass_peak_gib(
            chunk_tokens=chunk_tokens, rows=plan.micro_batch_size
        ),
        importance_sampling_log_only=config.vllm_importance_sampling_log_only,
        importance_sampling_mode=config.vllm_importance_sampling_mode,
        dynamic_sampling_oversample=config.dynamic_sampling_oversample,
    )
    assert_old_logps_pass_fits(
        config, rows=plan.micro_batch_size, priced_from="the micro-batch the sizing plan derived"
    )
    return plan, cost


def assert_dataset_fills_a_step(
    n_prompts: int, plan: SizingPlan, *, dynamic_sampling_oversample: int = 1
) -> None:
    """Refuse a dataset too small for one generation, and say when a remainder is dropped.

    TRL's `RepeatSampler` chunks the shuffled prompt indices into
    `generation_batch_size // num_generations` and then discards every incomplete chunk. Below that
    threshold it yields nothing at all: no optimizer step runs, and the failure surfaces indirectly
    as the read-back's "no training record carried a reward", a model load and minutes of generation
    later.

    That chunk is `plan.prompts_per_step` at the default oversample and `oversample` times it under
    dynamic sampling, because the widened `generation_batch_size` is what the sampler reads
    (`widen_generation_batch_for_oversample`). A corpus that fills a step at oversample 1 and not at
    2 would otherwise train nothing while every setting looked reasonable.
    """
    chunk = plan.prompts_per_step * dynamic_sampling_oversample
    if n_prompts < chunk:
        composition = (
            f"prompts_per_step={plan.prompts_per_step}"
            if dynamic_sampling_oversample == 1
            else (f"prompts_per_step={plan.prompts_per_step} times {dynamic_sampling_oversample=}")
        )
        raise RuntimeError(
            f"the dataset holds {n_prompts} prompt(s), and one generation needs {chunk} "
            f"({composition}). TRL's sampler drops every incomplete chunk, so this run would "
            f"perform no step at all and report it as 'no training record carried a reward'. "
            f"Select or generate more prompts, or lower --prompts-per-step."
        )
    remainder = n_prompts % chunk
    if remainder:
        logger.warning(
            "the sampler drops the short chunk at the end of every epoch, %s",
            f"{n_prompts=} prompts_per_generation={chunk} dropped_per_epoch={remainder}",
        )


@dataclass(frozen=True)
class CompletedRun:
    """A resume that landed on a checkpoint already at `max_steps`: there is nothing left to train."""

    checkpoint: str
    step: int
    max_steps: int
    summary_path: str


def completed_run(checkpoint: str, *, output_dir: str, max_steps: int) -> CompletedRun | None:
    """Recognise a relaunch of a finished run before anything is written or loaded.

    Re-running the identical command is the whole recovery procedure, so a stage runner that comes
    back after a reboot re-runs finished stages too. Left to the trainer, that relaunch is not a
    no-op: transformers restores `global_step == max_steps`, runs zero optimizer steps, and still
    logs once from `_finalize_training`, which made TRL write a zero-row `completions_00070.parquet`
    over the real one, the memory monitor open a fresh `mem_log.csv`, and the summary re-report the
    run with a 40-second wall clock (2026-09-01, `pd-unstated-other-payoff`). Every one of those
    writes happens after this point, so recognising the state here leaves every artifact untouched.

    The step is read from the checkpoint's own `trainer_state.json`, which is what the trainer
    would restore, rather than parsed from the directory name. A finished run without its
    `train_summary.json` is refused rather than declared complete: it died between its final save
    and its summary, nothing here can regenerate that summary without re-running the trainer, and a
    zero-step trainer run is exactly what produces a wrong one.
    """
    state = json.loads(Path(checkpoint, TRAINER_STATE_FILENAME).read_text(encoding="utf-8"))
    step = int(state["global_step"])
    if step < max_steps:
        return None
    summary_path = Path(output_dir, TRAIN_SUMMARY_FILENAME)
    if not summary_path.is_file():
        raise RuntimeError(
            f"{checkpoint} already holds step {step} of {max_steps=}, but {summary_path} does not "
            f"exist: the run that wrote the checkpoint died between its final save and its summary. "
            f"Resuming would train zero steps and rewrite the run's trace, memory log and summary "
            f"with a zero-step launch's numbers, so this launch refuses instead. The checkpoints and "
            f"the completions trace on disk are the run's real output; the summary has to be "
            f"reconstructed from them rather than from a relaunch."
        )
    return CompletedRun(
        checkpoint=checkpoint, step=step, max_steps=max_steps, summary_path=str(summary_path)
    )


def _prepare_run(
    config: GameTrainConfig, *, kernel_bridge: dict[str, object] | None
) -> PreparedRun | CompletedRun:
    """Settle everything a run needs before any weights load, and write its launch record.

    Returns a `CompletedRun` instead, before the sizing plan or the launch record, when the resume
    landed on a checkpoint already at `max_steps`; see `completed_run` for why that has to be here.

    The resume checkpoint is resolved FIRST, and the launch it belongs to checked against this one,
    because everything below is derived rather than fixed: the sizing plan comes from a live
    free-VRAM reading, so a resume onto a card another process has taken a bite out of would
    silently re-plan the group size the checkpoint's existing steps were trained under -- and the
    old `run_config.json` write would destroy the only record of what that was.
    """
    if config.output_dir is None:
        config = replace(
            config,
            output_dir=default_output_dir(
                config.arm, config.model_id, timestamp=datetime.now(tz=UTC), smoke=config.smoke
            ),
        )
    output_dir = cast("str", config.output_dir)
    world_size = _announce_launch(config)
    resume = resolve_complete_resume_checkpoint(config.resume_from_checkpoint, output_dir)
    resume_checkpoint = resume.checkpoint
    recorded_launch: dict[str, object] | None = None
    if resume_checkpoint is not None:
        recorded_launch = read_recorded_launch(output_dir, checkpoint=resume_checkpoint)
        assert_resume_matches(
            recorded={
                **RESUME_IDENTITY_DEFAULTS,
                **cast("Mapping[str, object]", recorded_launch["config"]),
            },
            current=asdict(config),
            fields=RESUME_IDENTITY_FIELDS,
            checkpoint=resume_checkpoint,
            consequence=(
                "Resuming would keep training that checkpoint's adapter, optimizer and step "
                "counter under a different question, and its trace would carry both experiments "
                "under one set of step numbers."
            ),
        )
        # The registry half of the same check, against the record's top level rather than its config
        # block. Separate call rather than a widened field list, because the two sets live in two
        # places in the record and merging them would make `RESUME_IDENTITY_DEFAULTS` (which readers
        # iterate as config fields) name one that is not one.
        assert_resume_matches(
            recorded={**RESUME_ARM_IDENTITY_DEFAULTS, **recorded_launch},
            current=arm_identity(config.game_arm),
            fields=RESUME_ARM_IDENTITY_FIELDS,
            checkpoint=resume_checkpoint,
            consequence=(
                "The arm's game set is what its corpus may carry, so resuming under a different one "
                "would train prompts the checkpoint's existing steps never saw, with one step "
                "history and one summary covering both corpora."
            ),
        )
        # After the identity check on purpose: a relaunch under the wrong arm is refused as such,
        # not waved through as "already complete".
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

    tokenizer, template_facts = resolve_tokenizer(config.load_source, thinking=config.thinking)
    rows = prepare_rows(config)
    dataset = build_game_dataset(
        rows,
        tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        # Explicit, never defaulted: the ladder's thinking defaults move non-monotonically by
        # checkpoint, so relying on the template's own default would vary by tier.
        enable_thinking=config.thinking,
        chat_template_kwargs=cast("dict[str, str]", template_facts["chat_template_kwargs"]),
    )
    logger.info("dataset built, %s", f"n_prompts={len(dataset)} n_rows_in={len(rows)}")

    lora_targets = discover_lora_targets(config.load_source)
    logger.info("LoRA targets: %s", lora_targets["module_counts"])
    init_adapter_facts = describe_init_adapter(config, lora_targets=lora_targets)
    kernel_paths = log_deltanet_kernel_paths(
        expected_linear_attention_layers=cast(
            "int", lora_targets["expected_linear_attention_layers"]
        )
    )
    param_count = count_meta_parameters(config.load_source)
    plan, cost = _derive_plan(config, device=device, dtype=dtype, param_count=param_count)
    assert_dataset_fills_a_step(
        len(dataset), plan, dynamic_sampling_oversample=config.dynamic_sampling_oversample
    )
    if recorded_launch is not None:
        assert_resume_matches(
            recorded=cast("Mapping[str, object]", recorded_launch["sizing_plan"]),
            current=asdict(plan),
            fields=RESUME_SIZING_FIELDS,
            checkpoint=cast("str", resume_checkpoint),
            consequence=(
                "The sizing plan is derived from a live free-VRAM reading, so this is usually "
                "another process holding memory on a shared card rather than a different card. "
                "num_generations is the GRPO advantage baseline and the population group-mix "
                "grading estimates the opponent distribution from, so changing it mid-run changes "
                "the experiment. Free the memory, or pass --no-autosize with the recorded values."
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
        "sequence_cost": asdict(cost),
        "meta_parameter_count": param_count,
        "n_prompts": len(dataset),
        "n_rows_before_dataset_build": len(rows),
        "corpus_composition": describe_row_composition(rows),
        "group_mix_fixed_points": group_mix_fixed_points(
            rows,
            grading=config.game_arm.grading,
            # The plan's group size, not the requested one: autosize may have clamped it, and under
            # --leave-one-out the predicted fixed point depends on the size actually used.
            num_generations=plan.num_generations,
            leave_one_out=config.leave_one_out,
        ),
        "world_size": world_size,
        "resumed_from_checkpoint": resume_checkpoint,
        # Empty unless `latest` had to step past torn checkpoints; then the steps that run twice.
        "incomplete_checkpoints_set_aside": list(resume.set_aside),
        # None for an ordinary run; a transfer run records which checkpoint bytes seeded its LoRA.
        "init_adapter": init_adapter_facts,
        # The engine's own share is not repeated here: it is `sizing_plan.engine_reserved_gib`,
        # where it is the number the plan was actually built against.
        "vllm_max_model_length": config.vllm_max_model_length,
        "unrecorded_generation_env": {
            name: os.environ.get(name) for name in UNRECORDED_GENERATION_ENV
        },
    }
    # A resumed launch writes its own record beside the original rather than over it: the original
    # is the only thing that says what the earlier steps were trained under.
    record_name = (
        RUN_CONFIG_FILENAME
        if resume_checkpoint is None
        else RESUMED_RUN_CONFIG_TEMPLATE.format(checkpoint=Path(resume_checkpoint).name)
    )
    record_path = Path(output_dir, record_name)
    write_json(record_path, run_config_payload(config, plan=plan, device=device, derived=derived))
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
        record_path=record_path,
    )


# The micro-batch keys TRL pads per token, and the side each is padded on (`grpo_trainer.py`
# `_generate_and_score_completions`: prompts left-padded to the batch's longest prompt, completions
# and every per-token completion column right-padded to the batch's longest completion). Listed
# rather than discovered from tensor widths so a key TRL adds later fails the width assertion below
# instead of being silently left at the padded width beside trimmed neighbours.
PROMPT_TOKEN_KEYS: tuple[str, ...] = ("prompt_ids", "prompt_mask")
COMPLETION_TOKEN_KEYS: tuple[str, ...] = (
    "completion_ids",
    "completion_mask",
    "old_per_token_logps",
    "ref_per_token_logps",
    "importance_sampling_ratio",
    "sampling_per_token_logps",
    "tool_mask",
)
# The one completion key TRL legitimately hands over one column wide: under the `sequence_mask` and
# `sequence_truncate` importance-sampling modes (`sequence_mask` is the default) the vLLM importance
# ratio is a per-sequence (B, 1) value that broadcasts over the tokens (grpo_trainer.py:2653-2664 in
# 1.10); under the `token_*` modes it is (B, T) and trimmed like its neighbours. No other key is ever
# per-sequence, so a one-column `old_per_token_logps` still fails the width check below.
PER_SEQUENCE_COMPLETION_KEYS: frozenset[str] = frozenset({"importance_sampling_ratio"})
# Per optimizer step, the token columns the training passes covered with and without the trim.
STEP_PADDED_TOKENS_METRIC = "padding_trim/step_padded_tokens"
STEP_TRIMMED_TOKENS_METRIC = "padding_trim/step_trimmed_tokens"
PADDING_TRIM_STEP_METRICS: tuple[str, ...] = (STEP_PADDED_TOKENS_METRIC, STEP_TRIMMED_TOKENS_METRIC)
# The log-record keys `mem_log.csv` copies beside each step's memory reading: the phase seconds, the
# trim totals, the entropy proxy and the dynamic-sampling counts, so one row says what a step did,
# how long each part took, how spread the policy was throughout, what share of the generated groups
# reached the optimizer, and what the step cost in VRAM. A key absent from a log record writes an
# empty cell (`grpo.rlvr_math.MemoryMonitorCallback`), which every run at oversample 1 writes.
MEM_LOG_EXTRA_COLUMNS: tuple[str, ...] = (
    *TIMING_METRIC_KEYS,
    *PADDING_TRIM_STEP_METRICS,
    *SAMPLED_SURPRISAL_METRICS,
    *DYNAMIC_SAMPLING_STEP_METRICS,
)
# The completions-trace column marking a row the selection dropped, so the readout can tell the far
# framings' generated rollouts from the ones that supplied the gradient. Goes through TRL's own
# `_logs["extra"]`, the same channel the reward function's `log_extra` columns take, so it lands in
# `completions_<step>.parquet` beside them.
DYNAMIC_SAMPLING_DROPPED_COLUMN = "dynamic_sampling_dropped"


@dataclass(frozen=True)
class MicroBatchTrim:
    """What one micro-batch's trim removed: the same rows, fewer token columns."""

    rows: int
    padded_columns: int
    trimmed_columns: int

    @property
    def padded_tokens(self) -> int:
        """Tokens the forward/backward would have covered without the trim."""
        return self.rows * self.padded_columns

    @property
    def trimmed_tokens(self) -> int:
        """Tokens the forward/backward covers after it."""
        return self.rows * self.trimmed_columns


def _first_live_column(mask: torch.Tensor) -> int:
    """Index of the first column any row keeps, or 0 when no row keeps anything (left untrimmed)."""
    live = mask.any(dim=0).nonzero()
    return int(live[0].item()) if live.numel() else 0


def _stop_after_last_live_column(mask: torch.Tensor) -> int:
    """One past the last column any row keeps, or the full width when no row keeps anything."""
    live = mask.any(dim=0).nonzero()
    return int(live[-1].item()) + 1 if live.numel() else mask.size(1)


def trim_micro_batch(inputs: dict[str, Any]) -> tuple[dict[str, Any], MicroBatchTrim]:
    """Cut a micro-batch down to the columns some row of it actually uses.

    TRL pads every completion in the generation batch to that batch's longest and splits the buffer
    along rows only, so with micro-batch 1 all 64 training passes of a step ran over ~33k tokens
    whenever one completion hit the cap while real rows averaged ~7k: 74% of each step's wall
    clock was forward/backward over masked padding (efficiency audit, 2026-09-02). This drops the
    leading prompt columns that are pad in every row and the trailing completion columns that are
    pad in every row, from the ids, the masks and every per-token column present; everything else
    (advantages, num_items_in_batch, any multimodal key) passes through untouched. A row set whose
    mask is all zero on a side is left untrimmed on that side, because "nothing live" is a fact
    for the loss to see, not a width to invent.

    Statistically neutral, not bit-identical: the loss masks already zero the pads and every
    normalizer this repo runs (grpo, bnpo, dr_grpo's constant, the dapo fallback's mask sum, with
    `num_items_in_batch` computed before the split) is pad-invariant, so the gradient is the same
    function of the same tokens; kernel reduction orders and Triton autotune picks change with the
    width. `luspo` divides by a pad-inclusive count and would not be neutral -- it is refused
    under Liger already. Verified on the L4 by `test_games_padding_trim.py`: LoRA gradient cosine
    and norm between the padded and trimmed passes of one real corpus row.
    """
    prompt_mask = inputs["prompt_mask"]
    completion_mask = inputs["completion_mask"]
    prompt_start = _first_live_column(prompt_mask)
    completion_stop = _stop_after_last_live_column(completion_mask)
    trimmed = dict(inputs)
    for key in PROMPT_TOKEN_KEYS:
        tensor = trimmed.get(key)
        if tensor is None:
            continue
        if tensor.size(1) != prompt_mask.size(1):
            raise RuntimeError(
                f"micro-batch key {key!r} has width {tensor.size(1)} but prompt_mask has "
                f"{prompt_mask.size(1)}; TRL's batch layout changed and the trim would misalign it."
            )
        trimmed[key] = tensor[:, prompt_start:]
    for key in COMPLETION_TOKEN_KEYS:
        tensor = trimmed.get(key)
        if tensor is None:
            continue
        # The per-sequence ratio is not a padded row and is left as it is; the unconditional width
        # check killed every run with the correction ON at its first micro-batch (L4 smoke, 2026-09-02).
        if key in PER_SEQUENCE_COMPLETION_KEYS and tensor.size(1) == 1:
            continue
        if tensor.size(1) != completion_mask.size(1):
            raise RuntimeError(
                f"micro-batch key {key!r} has width {tensor.size(1)} but completion_mask has "
                f"{completion_mask.size(1)}; TRL's batch layout changed and the trim would misalign it."
            )
        trimmed[key] = tensor[:, :completion_stop]
    stats = MicroBatchTrim(
        rows=int(completion_mask.size(0)),
        padded_columns=int(prompt_mask.size(1) + completion_mask.size(1)),
        trimmed_columns=int((prompt_mask.size(1) - prompt_start) + completion_stop),
    )
    return trimmed, stats


@dataclass(frozen=True)
class GroupSelection:
    """Which prompt groups of an over-generated batch the coming optimizer step trains on.

    Indices are into the groups of the generated batch, in the order the sampler drew them, and
    `kept_groups` is every kept index back in that same order, which is the order the kept batch's
    rows take. The choice is a prefix of the sampler's own ordering rather than of anything the
    rewards decided, so one seed keeps the same prompts twice.
    """

    groups_generated: int
    live_groups: tuple[int, ...]
    kept_live: tuple[int, ...]
    kept_pure: tuple[int, ...]

    @property
    def groups_pure(self) -> int:
        """Groups whose completions all earned the same reward, so their advantages are all zero."""
        return self.groups_generated - len(self.live_groups)

    @property
    def kept_groups(self) -> tuple[int, ...]:
        """Every kept group, in sampler order."""
        return tuple(sorted((*self.kept_live, *self.kept_pure)))


def choose_live_groups(
    advantages: torch.Tensor, *, num_generations: int, keep_groups: int
) -> GroupSelection:
    """Pick the groups that supply a gradient, falling back to pure ones only to fill the batch.

    A *pure* group is one whose completions all earned the same reward: GRPO's advantage is the
    reward less the group mean, so every row of such a group is exactly zero and the group costs a
    full training forward and backward while moving no parameter. On the banked 9B pair 20% of groups
    were pure, rising to a third late in the run, and on a breadth corpus's far framings most groups
    are expected to be.

    Read off TRL's own `advantages` rather than off the rewards, which the scored batch does not
    carry: that tensor IS the gradient supply, so "all zero" means "this group would have trained
    nothing" under every `scale_rewards` mode (a group std of zero divides zeros by 1e-4 and leaves
    them zero) and for the rows TRL forces to zero as unscorable. One caveat, stated rather than
    guarded: at a nonzero `beta` a dropped group's KL term is dropped with it, so the oversample is a
    treatment, which is why it is a resume-identity field. Every arm here trains at beta=0.

    Selection is a prefix of the sampler's order, never of a reward ranking: keeping the *strongest*
    groups would bias the batch toward whatever the reward happens to spread, which is the thing
    being measured. When fewer live groups exist than the optimizer batch needs, pure groups fill it
    in the same order, so the batch handed back is always exactly the width TRL split it to expect.
    """
    if advantages.size(0) % num_generations:
        raise RuntimeError(
            f"advantages hold {advantages.size(0)} rows, which is not a whole number of "
            f"{num_generations}-completion groups. TRL lays a generation batch out group by group "
            f"(grpo_trainer.py:2755, `rewards.view(-1, num_generations)`) and the selection reads "
            f"that layout."
        )
    per_group = advantages.view(-1, num_generations)
    generated = per_group.size(0)
    if keep_groups > generated:
        raise RuntimeError(
            f"asked to keep {keep_groups} groups out of {generated} generated, so the batch handed "
            f"to TRL would be short: the oversample, the sizing plan and the dataloader width "
            f"disagree."
        )
    live = tuple(index for index in range(generated) if bool(per_group[index].any()))
    kept_live = live[:keep_groups]
    pure = tuple(index for index in range(generated) if index not in set(live))
    return GroupSelection(
        groups_generated=generated,
        live_groups=live,
        kept_live=kept_live,
        kept_pure=pure[: keep_groups - len(kept_live)],
    )


def narrow_to_groups(
    batch: dict[str, Any], *, num_generations: int, groups: Sequence[int]
) -> dict[str, Any]:
    """Return the scored batch restricted to `groups`, with its live-token count recomputed.

    Every per-row value is indexed in step -- the ids, the masks, the advantages, the importance
    ratio (per token or per sequence, both row-major), any multimodal column -- and the one value
    that is not per-row, `num_items_in_batch`, is recomputed from the kept rows' own loss mask rather
    than carried over. That recomputation is why this is not a slice: TRL sums the loss mask of the
    WHOLE generation batch (grpo_trainer.py:2497) and the dapo, cispo and vespo normalizers divide by
    it (:3210-3214), so a batch that kept half its rows and inherited the full count would train at
    half the gradient scale with every logged series unchanged.

    A key that is neither per-row nor that scalar raises rather than passing through, because a
    row-aligned tensor left at the generated width is a shape mismatch the loss would meet first.
    """
    rows = [
        row
        for group in groups
        for row in range(group * num_generations, (group + 1) * num_generations)
    ]
    generated_rows = batch["completion_mask"].size(0)
    narrowed: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim == 0:
            narrowed[key] = value
        elif isinstance(value, torch.Tensor) and value.size(0) == generated_rows:
            narrowed[key] = value[torch.tensor(rows, dtype=torch.long, device=value.device)]
        elif isinstance(value, list) and len(value) == generated_rows:
            narrowed[key] = [value[row] for row in rows]
        else:
            raise RuntimeError(
                f"generation-batch key {key!r} is neither per-row over {generated_rows} rows nor a "
                f"scalar, so dynamic sampling cannot narrow it ({type(value).__name__}). TRL's batch "
                f"layout changed and the selection would hand the loss mismatched shapes."
            )
    tool_mask = narrowed.get("tool_mask")
    completion_mask = narrowed["completion_mask"]
    loss_mask = completion_mask if tool_mask is None else completion_mask * tool_mask
    narrowed["num_items_in_batch"] = loss_mask.sum()
    return narrowed


@dataclass(frozen=True)
class SampledSurprisal:
    """One generation batch's surprisal, read off the log-probabilities vLLM already returned."""

    token_weighted_mean: float
    min_completion_mean: float
    max_completion_mean: float
    live_tokens: int
    completions: int


def sampled_surprisal(sampled_logps: torch.Tensor, mask: torch.Tensor) -> SampledSurprisal | None:
    """Mean surprisal of the sampled tokens, over live tokens only, or None when there are none.

    TRL asks vLLM for the sampled token's own log-probability on every rollout (`logprobs=0`,
    grpo_trainer.py:1117, narrowed to the top-1 at :1830) and pads the rows with 0.0, which as a
    log-probability reads as certainty -- so the mask is not an optimisation, it is the difference
    between an entropy estimate and a number pulled toward zero by however much padding the batch
    carried. The token-weighted mean is the entropy estimator (every sampled token is one draw); the
    extremes are per-completion means, because "which completion was the most predictable" is the
    question a collapsing run is diagnosed with, and a per-token extreme would only ever report the
    single most and least surprising token in 64 completions.

    Unbiased for the SAMPLING distribution's entropy, which equals the policy's only at temperature
    1.0; below it the sampler is tempered and this reads the tempered distribution, which is why the
    metric is named for the surprisal it measures rather than for entropy.

    Returns None rather than 0.0 when no token is live, because zero surprisal is a real claim -- a
    deterministic policy -- and a missing denominator is not.
    """
    live = mask.to(sampled_logps.dtype)
    live_per_completion = live.sum(dim=1)
    total_live = float(live_per_completion.sum())
    if total_live == 0.0:
        return None
    surprisal = -sampled_logps * live
    scored = live_per_completion > 0
    completion_means = surprisal.sum(dim=1)[scored] / live_per_completion[scored]
    return SampledSurprisal(
        token_weighted_mean=float(surprisal.sum()) / total_live,
        min_completion_mean=float(completion_means.min()),
        max_completion_mean=float(completion_means.max()),
        live_tokens=int(total_live),
        completions=int(scored.sum()),
    )


def chunked_per_token_logps(
    hidden_states: torch.Tensor,
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    token_ids: torch.Tensor,
    *,
    temperature: float,
    chunk_tokens: int,
) -> torch.Tensor:
    """Per-token log-probabilities computed a slice of positions at a time, never the whole block.

    The same arithmetic TRL runs (grpo_trainer.py:1524-1534: head, divide by temperature,
    `selective_log_softmax`, whose helper is called here rather than reimplemented so both its
    branches are the same code on both sides of the comparison). What it buys is the peak: TRL
    materialises (positions x vocab) at once, 30.31 GiB of fp32 for one 32,768-token row over a
    248,320-token vocabulary, measured dead beside a resident colocate engine on 2026-08-19. Here the
    widest tensor is chunk x vocab, and what the pass holds LIVE at its peak is a few of those --
    priced by `old_logps_pass_peak_gib`, which is the arithmetic the startup refusal uses.

    Exact in the half that chunking introduces, and not in the half it inherits, measured on this box
    (2026-09-04) rather than argued: given one logits block, slicing the position axis and taking
    `selective_log_softmax` per slice is bit-for-bit the whole-block result in fp32 and in bfloat16
    alike, because a log-softmax reduces each position over the vocabulary and reads no other
    position. The head's own matmul is not bit-stable under a change of row count -- a different
    number of rows reassociates the reduction over the hidden dimension -- so the end-to-end pass
    differs from the unchunked one by a mean 8e-8 nats per token in fp32 (max 1e-6) and by a mean
    0.0024 nats in bfloat16 (max 0.0625, one bfloat16 unit in the last place at these magnitudes).
    The fp32 figure is negligible against anything this is used for. The bfloat16 figure is not, for
    a sequence-level importance ratio that sums a log-difference over thousands of tokens, which is
    why `GameTrainConfig.cast_lm_head_to_fp32` exists and why log-only importance sampling asks for
    it: with the head in fp32 the chunk-dependence drops to the fp32 figure, and the mismatch the
    ratio reports is the vLLM-versus-trainer one rather than this.
    """
    if chunk_tokens < 1:
        raise ValueError(
            f"the old-log-probability pass needs at least one token per chunk, {chunk_tokens=}: "
            f"zero would never advance and a negative would drop every position silently"
        )
    positions = hidden_states.size(1)
    if token_ids.size(1) != positions:
        raise RuntimeError(
            f"hidden states carry {positions} positions and the sampled ids carry "
            f"{token_ids.size(1)}; a chunk would pair a position with another position's token"
        )
    slices = [
        selective_log_softmax(
            _tempered_logits(
                lm_head(hidden_states[:, start : start + chunk_tokens, :]), temperature
            ),
            token_ids[:, start : start + chunk_tokens],
        )
        for start in range(0, positions, chunk_tokens)
    ]
    return torch.cat(slices, dim=1)


def _tempered_logits(head_output: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide a freshly built logits slice by the sampling temperature without copying it.

    TRL writes `logits = logits / self.temperature` (grpo_trainer.py:1531), which holds the undivided
    slice alive beside its quotient and so doubles the widest fp32 tensor of the pass at the one
    moment it is widest -- 30.31 GiB of it for each of two copies at the shape measured dead. The
    division is in place here instead. Two things make that sound and one is a precondition on the
    caller: the argument must be the head's OWN output, built at the call site and aliased nowhere
    else (true of `nn.Linear` and of TRL's `cast_forward_to_fp32`, and the reason a stand-in head in
    a test must return a fresh tensor rather than a view of one the test still holds); and autograd is
    indifferent, because a matmul's backward does not save its output, so the in-place divide gives
    the same gradient as the copying one (measured on this box, 2026-09-04, identical to the last
    bit). Temperature 1.0 skips the division entirely, which is bit-identical rather than merely
    close: IEEE-754 division by 1.0 is exact for every finite value.
    """
    if temperature == 1.0:
        return head_output
    return head_output.div_(temperature)


class InstrumentedGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with two instruments that leave every default behaviour where it was.

    **The entropy proxy.** Every generation batch's sampled surprisal lands in TRL's own `_metrics`
    under `SAMPLED_SURPRISAL_METRICS`, so it reaches `log_history` and, through
    `MEM_LOG_EXTRA_COLUMNS`, the same `mem_log.csv` row as the phase seconds and the padding-trim
    totals. No extra forward pass: the log-probabilities are the ones vLLM returned with the tokens.
    A rollout path that returns none (transformers.generate sets `logprobs = None`,
    grpo_trainer.py:1918) leaves the metric absent for the step and says so once.

    **The chunked old-log-probability pass.** `_get_per_token_logps_and_entropies` is the pass the
    importance-sampling correction forces (grpo_trainer.py:2631-2634) and the one that went out of
    memory at the production shape. The override computes the same log-probabilities from
    hidden-state slices, so the peak is `old_logps_chunk_tokens` positions wide instead of the whole
    row. It hands back to TRL untouched whenever the caller wants something a hidden state cannot
    give: entropies (needed only on the non-Liger loss path, `:3071`, where the full-vocabulary
    logits are being built anyway), the mixture-of-experts auxiliary loss (needs router logits from
    the model's own forward), or any multimodal input (`_get_last_hidden_state` takes pixels but not
    the image counts and token-type ids the wider signature carries).

    **Log-only importance sampling.** With `importance_sampling_log_only`, TRL computes the
    correction and logs its ratio statistics (`sampling/importance_sampling_ratio/*` and
    `sampling/sampling_logp_difference/*`, appended at grpo_trainer.py:2849-2882) and then the
    weight is replaced by exactly one before any loss sees it, so the vLLM-versus-trainer mismatch
    on this hybrid-attention family is measurable without moving the gradient. Both consumers
    multiply by it (`grpo_trainer.py:3188` and, under Liger,
    `liger_kernel/chunked_loss/grpo_loss.py:205-206`), and a multiplication by 1.0 is exact. The
    recomputed old log-probabilities are dropped alongside it whenever TRL would not have had them
    but for the correction, because the surrogate ratio has to be exactly one and not merely close:
    those values come from a different kernel path than the loss's own forward.
    """

    # Class-level so a `__new__`-built instance and `reward_hacking.train`, which constructs this
    # class without either keyword, both get today's behaviour.
    old_logps_chunk_tokens: int = OLD_LOGPS_CHUNK_TOKENS
    importance_sampling_log_only: bool = False
    _said_the_surprisal_is_absent: bool = False
    _said_the_logps_pass_is_unchunked: bool = False

    def __init__(
        self,
        *args: Any,  # noqa: ANN401 - GRPOTrainer's own untyped constructor, passed through
        old_logps_chunk_tokens: int = OLD_LOGPS_CHUNK_TOKENS,
        importance_sampling_log_only: bool = False,
        **kwargs: Any,  # noqa: ANN401 - GRPOTrainer's own untyped constructor, passed through
    ) -> None:
        """Take the two instrument knobs, then build the trainer TRL would have built."""
        self.old_logps_chunk_tokens = old_logps_chunk_tokens
        self.importance_sampling_log_only = importance_sampling_log_only
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Read the entropy proxy off TRL's generation batch, and neutralise the weight if asked."""
        # TRL annotates this parameter `list[dict[...]]` (grpo_trainer.py:2331) while its only caller
        # hands it the dict-of-columns generation batch (`:1607`), so the cast names the runtime type
        # rather than widening ours to the stale one.
        batch = super()._generate_and_score_completions(
            cast("list[dict[str, torch.Tensor | Any]]", generation_batch)
        )
        self._record_sampled_surprisal(batch)
        if self.importance_sampling_log_only:
            self._neutralise_importance_sampling(batch)
        return batch

    def _record_sampled_surprisal(self, batch: dict[str, torch.Tensor | Any]) -> None:
        """Append this generation batch's surprisal readings, or say once why there are none."""
        sampled = batch.get("sampling_per_token_logps")
        if sampled is None:
            self._say_the_surprisal_is_absent(
                "this rollout path returned no sampled log-probabilities"
            )
            return
        completion_mask = batch["completion_mask"]
        tool_mask = batch.get("tool_mask")
        # TRL's own mismatch statistics narrow by the same product (grpo_trainer.py:2851).
        mask = completion_mask if tool_mask is None else completion_mask * tool_mask
        reading = sampled_surprisal(sampled, mask)
        if reading is None:
            self._say_the_surprisal_is_absent("no completion in this batch has a live token")
            return
        mode = "train" if self.model.training else "eval"  # pyright: ignore[reportOptionalMemberAccess]
        self._metrics[mode][SAMPLED_SURPRISAL_MEAN_METRIC].append(reading.token_weighted_mean)
        self._metrics[mode][SAMPLED_SURPRISAL_MIN_METRIC].append(reading.min_completion_mean)
        self._metrics[mode][SAMPLED_SURPRISAL_MAX_METRIC].append(reading.max_completion_mean)
        self._metrics[mode][SAMPLED_SURPRISAL_TOKENS_METRIC].append(float(reading.live_tokens))
        logger.info(
            "sampled surprisal: mean=%.4f min=%.4f max=%.4f over %d live tokens in %d completions",
            reading.token_weighted_mean,
            reading.min_completion_mean,
            reading.max_completion_mean,
            reading.live_tokens,
            reading.completions,
        )

    def _say_the_surprisal_is_absent(self, reason: str) -> None:
        """Say once per run that the entropy proxy has no reading, rather than logging a zero."""
        if self._said_the_surprisal_is_absent:
            return
        self._said_the_surprisal_is_absent = True
        logger.info(
            "entropy proxy absent for this and any later step: %s, so %s is not recorded rather "
            "than recorded as zero",
            reason,
            SAMPLED_SURPRISAL_MEAN_METRIC,
        )

    def _neutralise_importance_sampling(self, batch: dict[str, torch.Tensor | Any]) -> None:
        """Leave TRL's logged ratio statistics standing and make the weight the loss sees exactly 1.

        Indexed rather than fetched with a default: log-only without the correction has no ratio to
        log and is refused at config time, so a missing key here is a wiring bug worth a traceback.
        """
        ratio = cast("torch.Tensor", batch["importance_sampling_ratio"])
        batch["importance_sampling_ratio"] = torch.ones_like(ratio)
        if self._old_logps_exist_only_for_the_correction:
            del batch["old_per_token_logps"]

    @property
    def _old_logps_exist_only_for_the_correction(self) -> bool:
        """Whether correction-off would have left `old_per_token_logps` unset for this shape.

        TRL's own condition, grpo_trainer.py:2631-2634: the pass runs when the generation and
        optimizer steps misalign OR when the correction is on. Aligned, the loss falls back to
        `per_token_logps.detach()` and the surrogate ratio is identically one, which is the state
        log-only has to reproduce; misaligned, the values are TRL's own off-policy correction and
        dropping them would change the estimator rather than leave it alone.
        """
        args = cast("GRPOConfig", self.args)
        generate_every = cast("int", args.steps_per_generation) * self.num_iterations
        return args.gradient_accumulation_steps % generate_every == 0

    def _get_per_token_logps_and_entropies(  # noqa: PLR0913, PLR0917 - TRL's own signature, positional at both call sites
        self,  # pyright: ignore[reportIncompatibleMethodOverride]
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: int,
        batch_size: int | None = None,
        compute_entropy: bool = False,  # noqa: FBT001, FBT002 - TRL's positional signature
        compute_aux_loss: bool = False,  # noqa: FBT001, FBT002 - TRL's positional signature
        **multimodal: Any,  # noqa: ANN401 - TRL's untyped multimodal keywords, forwarded unread
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute the pass over position slices, or hand it back to TRL where a slice cannot serve."""
        supplied = sorted(name for name, value in multimodal.items() if value is not None)
        if compute_entropy or compute_aux_loss or supplied:
            self._say_the_logps_pass_is_unchunked(
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                supplied=supplied,
            )
            return super()._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=compute_entropy,
                compute_aux_loss=compute_aux_loss,
                **multimodal,
            )
        unwrapped = self.accelerator.unwrap_model(model)
        rows = batch_size or input_ids.size(0)
        per_row_group = [
            chunked_per_token_logps(
                self._get_last_hidden_state(
                    unwrapped,
                    input_ids[start : start + rows],
                    attention_mask[start : start + rows],
                    logits_to_keep,
                ),
                unwrapped.lm_head,
                input_ids[start : start + rows, -logits_to_keep:],
                temperature=self.temperature,
                chunk_tokens=self.old_logps_chunk_tokens,
            )
            for start in range(0, input_ids.size(0), rows)
        ]
        return torch.cat(per_row_group, dim=0), None, None

    def _say_the_logps_pass_is_unchunked(
        self, *, compute_entropy: bool, compute_aux_loss: bool, supplied: list[str]
    ) -> None:
        """Say once per run that TRL's unchunked pass is running, and which caller asked for it."""
        if self._said_the_logps_pass_is_unchunked:
            return
        self._said_the_logps_pass_is_unchunked = True
        logger.info(
            "old-log-probability pass left unchunked: %s, which needs the full-vocabulary logits "
            "a hidden-state slice cannot give, so TRL's own pass runs and its memory peak applies",
            f"{compute_entropy=} {compute_aux_loss=} multimodal inputs {supplied}",
        )


class PaddingTrimmedGRPOTrainer(InstrumentedGRPOTrainer):
    """The trainer both threads build: instrumented, and trimmed to its own live token columns.

    The class name says what it adds; `InstrumentedGRPOTrainer` above it carries the entropy proxy,
    the chunked old-log-probability pass and log-only importance sampling, which are read-outs and a
    memory shape rather than a change to what a micro-batch is.

    `_prepare_inputs` is TRL's seam between the buffered generation batch and the loss: it returns
    the one micro-batch the coming forward/backward consumes, and `compute_liger_loss` derives
    `logits_to_keep` from that tensor's width, so trimming here changes nothing downstream. A
    private TRL method (1.10.0); `test_games_padding_trim.py` pins the key names and the shape.

    Two records of what the trim did, so the readout can see it: per micro-batch the padded and
    trimmed token counts go through TRL's own `_metrics` (their per-step means land in
    `log_history` as `padding_trim/padded_tokens` and `padding_trim/trimmed_tokens`), and once per
    optimizer step the totals are logged and written under `PADDING_TRIM_STEP_METRICS` into the same
    log record -- and, through `MEM_LOG_EXTRA_COLUMNS`, the same `mem_log.csv` row -- as the
    `StepPhaseTimer` phase seconds, so the tokens a step trained on sit beside the seconds it took.
    """

    _trim_padded_tokens_this_step: int = 0
    _trim_trimmed_tokens_this_step: int = 0

    def _prepare_inputs(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        inputs = super()._prepare_inputs(generation_batch)
        training = self.model is not None and self.model.training
        if not training or "completion_mask" not in inputs:
            return inputs
        trimmed, stats = trim_micro_batch(inputs)
        self._metrics["train"]["padding_trim/padded_tokens"].append(float(stats.padded_tokens))
        self._metrics["train"]["padding_trim/trimmed_tokens"].append(float(stats.trimmed_tokens))
        self._trim_padded_tokens_this_step += stats.padded_tokens
        self._trim_trimmed_tokens_this_step += stats.trimmed_tokens
        # `_step` counts micro-steps and is advanced by `training_step` AFTER this call, so the
        # last micro-batch of a generation batch is the one whose index is steps_per_generation - 1.
        # Derived by GRPOConfig.__post_init__ from the accumulation steps, so never None at run time;
        # typed Optional because the field accepts None from a caller.
        steps_per_generation = cast("int", cast("GRPOConfig", self.args).steps_per_generation)
        if self._step % steps_per_generation == steps_per_generation - 1:
            padded, kept = self._trim_padded_tokens_this_step, self._trim_trimmed_tokens_this_step
            self._metrics["train"][STEP_PADDED_TOKENS_METRIC].append(float(padded))
            self._metrics["train"][STEP_TRIMMED_TOKENS_METRIC].append(float(kept))
            logger.info(
                "padding trim: step tokens padded=%d trimmed=%d kept_fraction=%.3f",
                padded,
                kept,
                kept / padded if padded else float("nan"),
            )
            self._trim_padded_tokens_this_step = 0
            self._trim_trimmed_tokens_this_step = 0
        return trimmed

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Refuse to let TRL's `log` replace a step's trace rows with an empty buffer, then log.

        TRL writes `completions_<global_step>.parquet` from its buffers unconditionally, and
        transformers calls `log` once from `_finalize_training` even after zero optimizer steps;
        together those let a relaunch of a finished run empty its last step's trace (2026-09-01).
        """
        if self.accelerator.is_main_process and self.log_completions:
            refuse_empty_trace_overwrite(
                trace_file_path(Path(cast("str", self.args.output_dir)), self.state.global_step),
                buffered_completions=len(self._logs["prompt"]),
            )
        super().log(logs, start_time)


class DynamicSampledGRPOTrainer(PaddingTrimmedGRPOTrainer):
    """Over-generate prompt groups, then train the optimizer step on the ones that carry a gradient.

    A group whose completions all earned the same reward has zero advantages: it costs a full
    training forward and backward and moves no parameter. At `dynamic_sampling_oversample=N` this
    generates N times the optimizer step's prompt groups, scores all of them -- so every generated
    completion still reaches `completions_<step>.parquet` and every reward-side metric, which is how
    a readout sees the strata that mostly go pure -- and hands the optimizer the first
    `prompts_per_step` live groups in sampler order, filling with pure ones only when too few live
    groups exist. Default 1, which is today's behaviour and skips every branch here.

    Three seams make that fit inside TRL 1.10, and each is a private detail this class pins:

    *   **The width.** `_build_grpo_config` widens `generation_batch_size` after
        `GRPOConfig.__post_init__` has derived it, which is what makes `RepeatSampler` chunk N times
        as many prompts per generation (`grpo_trainer.py:1277-1283`) and what widens the completions
        buffers to hold every generated rollout (`:1069-1074`). `steps_per_generation` deliberately
        stays at the accumulation count, because it is what decides how often TRL generates and into
        how many micro-batches it splits (`:1584-1592`); widening it instead would spread one
        generation over two optimizer steps.
    *   **The dataloader.** TRL sizes its training batch `_train_batch_size * steps_per_generation`
        (`:1243`), which is the narrow width, so `get_train_dataloader` here scales that factor for
        the length of the call and puts it back.
    *   **The narrowing.** `_generate_and_score_completions` returns the batch `_prepare_inputs`
        shuffles and splits, so the selection happens there, and what it returns is exactly the
        `prompts_per_step * num_generations` rows TRL expects.

    What the epoch counter now means: prompts DRAWN, not prompts trained on. The sampler advances
    over N times as many prompts per optimizer step, so an epoch ends N times sooner and the dataset
    has to hold `oversample * prompts_per_step` prompts for the sampler to yield a chunk at all
    (`assert_dataset_fills_a_step`). Dropped groups stay in the rollout trace, marked by the
    `dynamic_sampling_dropped` column, so nothing generated is invisible afterwards.

    Nothing about the memory plan moves, which was checked rather than assumed. Under colocate
    `games.sizing.plan_sizing` is given `generation_schedules_own_batch=True`, so the episode
    arithmetic is computed for the log and the clamp is skipped: what pays for generation is the
    engine's own reservation, a fixed fraction of the whole card taken off the top
    (`colocate_reserved_gib`), and what bounds the trainer is the micro-batch token budget. Neither
    reads the generation width. The engine's concurrency does not move either: TRL sizes vLLM's
    `max_num_seqs` from `per_device_train_batch_size * tensor_parallel * steps_per_generation`
    (`:1105-1107`), none of which the widening touches, so a wide generation batch decodes in
    `oversample` waves at the width the KV reservation was measured for rather than in one wider
    wave. The cost is wall clock in generation, which is the trade being made.
    """

    dynamic_sampling_oversample: int = 1

    def __init__(
        self,
        *args: Any,  # noqa: ANN401 - GRPOTrainer's own untyped constructor, passed through
        dynamic_sampling_oversample: int = 1,
        **kwargs: Any,  # noqa: ANN401 - GRPOTrainer's own untyped constructor, passed through
    ) -> None:
        """Take the oversample, then build the trainer the classes above would have built."""
        if dynamic_sampling_oversample < 1:
            raise ValueError(
                f"dynamic_sampling_oversample must be at least 1 (1 = off), "
                f"{dynamic_sampling_oversample=}"
            )
        self.dynamic_sampling_oversample = dynamic_sampling_oversample
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader[Any]:
        """Load `oversample` generation batches' worth of prompts per iteration, not one.

        `_train_batch_size` is scaled for the length of TRL's own builder rather than overridden,
        because that attribute is also what transformers reports as the run's batch size and what
        `auto_find_batch_size` rewrites; the widening belongs to the dataloader alone.
        """
        narrow_batch_size = self._train_batch_size
        self._train_batch_size = narrow_batch_size * self.dynamic_sampling_oversample
        try:
            return super().get_train_dataloader()
        finally:
            self._train_batch_size = narrow_batch_size

    def _generate_and_score_completions(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, generation_batch: dict[str, torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        """Score everything generated, then hand back only the groups the optimizer step trains on."""
        batch = super()._generate_and_score_completions(generation_batch)
        if self.dynamic_sampling_oversample == 1:
            return batch
        # TRL types the attribute Optional because `num_generations` accepts None from a caller;
        # GRPOConfig.__post_init__ has resolved it long before any generation happens.
        num_generations = cast("int", self.num_generations)
        keep_groups = batch["advantages"].size(0) // (
            num_generations * self.dynamic_sampling_oversample
        )
        selection = choose_live_groups(
            batch["advantages"], num_generations=num_generations, keep_groups=keep_groups
        )
        self._record_dynamic_sampling(selection, num_generations=num_generations)
        return narrow_to_groups(
            batch, num_generations=num_generations, groups=selection.kept_groups
        )

    def _record_dynamic_sampling(self, selection: GroupSelection, *, num_generations: int) -> None:
        """Log the step's selection, into TRL's metrics and into the trace's dropped column."""
        mode = "train" if self.model.training else "eval"  # pyright: ignore[reportOptionalMemberAccess]
        for metric, value in (
            (DYNAMIC_SAMPLING_GENERATED_METRIC, selection.groups_generated),
            (DYNAMIC_SAMPLING_PURE_METRIC, selection.groups_pure),
            (DYNAMIC_SAMPLING_KEPT_LIVE_METRIC, len(selection.kept_live)),
            (DYNAMIC_SAMPLING_KEPT_PURE_METRIC, len(selection.kept_pure)),
        ):
            self._metrics[mode][metric].append(float(value))
        logger.info(
            "dynamic sampling: generated=%d pure=%d kept_live=%d kept_pure=%d",
            selection.groups_generated,
            selection.groups_pure,
            len(selection.kept_live),
            len(selection.kept_pure),
        )
        # Appended after the reward function's own `log_extra` columns have been flushed, and one
        # value per GENERATED row, which is what `_logs["prompt"]` holds and therefore the length
        # every column of the parquet must have.
        kept = set(selection.kept_groups)
        # Cast because TRL declares `_logs` as one dict holding both the per-row deques and the
        # nested `extra`/`rewards` mappings, so the static type of `_logs["extra"]` is their union.
        extra_columns = cast("dict[str, Any]", self._logs)["extra"]
        extra_columns[DYNAMIC_SAMPLING_DROPPED_COLUMN].extend(
            group not in kept
            for group in range(selection.groups_generated)
            for _ in range(num_generations)
        )


def _build_trainer(prepared: PreparedRun) -> GRPOTrainer:
    """Construct the trainer this plan calls for, and stamp the tokenizer's ids onto the model."""
    config = prepared.config
    reward = make_game_reward(
        prepared.plan.num_generations,
        prefilled_think=prepared.prefilled_think,
        leave_one_out=config.leave_one_out,
        # A run knob, and it has to be passed: the field was recorded in run_config.json and gated
        # on resume while `make_game_reward` kept its own default, so a run could name a penalty
        # everywhere a reader looks and still price every unparseable completion at -1.0.
        parse_penalty=config.parse_penalty,
        # The arm's, not a run knob: the mode is part of the grading, and a checkpoint's arm is
        # already a resume-identity field, so steps priced one way cannot continue under another.
        parse_penalty_mode=config.game_arm.parse_penalty_mode,
    )
    # Bound rather than passed inline: GRPOConfig.__post_init__ derives generation_batch_size
    # and steps_per_generation on this object, and those derived values are what decide whether
    # the rollout trace is complete.
    grpo_args = _build_grpo_config(config, prepared.plan, dtype=prepared.dtype)
    callbacks = build_callbacks(
        logging_steps=config.logging_steps,
        output_dir=cast("str", config.output_dir),
        s3_dest=config.s3_dest,
    )
    if config.record_retention_manifest:
        # Write before S3SyncCallback so each save ships the matching inventory with its checkpoint.
        callbacks.insert(
            0, CheckpointRetentionCallback(run_root=Path(cast("str", config.output_dir)))
        )
    # Registered as a callback here and attached to the built trainer below: the callback half
    # needs the optimizer-step hooks, the attach half needs the trainer's seams to exist.
    phase_timer = StepPhaseTimer()
    trainer = DynamicSampledGRPOTrainer(
        old_logps_chunk_tokens=config.old_logps_chunk_tokens,
        importance_sampling_log_only=config.vllm_importance_sampling_log_only,
        dynamic_sampling_oversample=config.dynamic_sampling_oversample,
        model=config.load_source,
        reward_funcs=reward,  # pyright: ignore[reportArgumentType]
        args=grpo_args,
        train_dataset=prepared.dataset,
        # A bare tokenizer rather than TRL's default AutoProcessor: Qwen3.5 maps to a
        # ProcessorMixin, which switches TRL onto its vision-language path.
        processing_class=prepared.tokenizer,
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cast("list[str]", prepared.lora_targets["target_modules"]),
        ),
        callbacks=[
            *callbacks,
            # Appended here rather than inside build_callbacks: the reward-hacking trainer shares
            # that builder and keeps a deliberate slow path behind --allow-hf-generation, which
            # this guard would kill at step 3.
            StepPaceGuardCallback(),
            phase_timer,
        ],
    )
    phase_timer.attach(trainer)
    # Not what generation reads -- TRL passes `generate()` its own GenerationConfig and overrides
    # the model's -- but what `save_model` writes, which is what every downstream eval load reads.
    model = trainer.model
    model.config.pad_token_id = prepared.tokenizer.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.config.eos_token_id = prepared.tokenizer.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.generation_config.pad_token_id = prepared.tokenizer.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    model.generation_config.eos_token_id = prepared.tokenizer.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    return trainer


# Enough positions for every SDPA kernel's eligibility rules to apply; the choice does not depend on
# the length beyond that, so the probe stays a millisecond.
SDPA_PROBE_POSITIONS = 128
# What `torch.backends.cuda.*_sdp_enabled` each toggle; recorded so a run that pinned or disabled
# one is distinguishable from one that let PyTorch choose.
SDPA_TOGGLES: dict[str, str] = {
    "flash": "flash_sdp_enabled",
    "mem_efficient": "mem_efficient_sdp_enabled",
    "math": "math_sdp_enabled",
    "cudnn": "cudnn_sdp_enabled",
}


def sdpa_backend_name(choice: int) -> str:
    """Name PyTorch's fused-SDPA dispatch result, an int of `torch.nn.attention.SDPBackend`."""
    names = {int(getattr(SDPBackend, name)): name for name in dir(SDPBackend) if name.isupper()}
    return names[choice]


def describe_attention_backends(trainer: GRPOTrainer) -> dict[str, object]:
    """Record which attention kernels this run trains and generates through.

    Three facts, none of them in `training_args.bin` or the model config on disk: the transformers
    attention implementation the loaded model resolved to (`sdpa` today, whichever kernel that
    dispatches to), the SDPA backend PyTorch picks for this model's head size in this dtype -- probed
    with `torch._fused_sdp_choice`, the dispatcher SDPA itself calls, for the two call shapes the
    training forward makes: `is_causal=True` with no mask (micro-batch 1, no padding) and a boolean
    mask (any padded batch), which is the shape the efficiency audit suspected of forcing SDPA off
    its causal fast path -- and the backend every KV-cache group of the colocated vLLM engine
    settled on, read from the in-process model runner (`external_launcher` keeps it in this
    process; TRL's own weight sync reads the same path). A run that quietly landed on the math
    kernel, or an engine that fell back from FlashAttention to Triton, is otherwise a step time
    nobody can explain afterwards.

    The vLLM read catches `AttributeError` alone: that private path moves between vLLM releases,
    and a record-only field must not kill a paid run. Everything else is left to raise.
    """
    model = cast("Any", trainer.model)
    model_config = model.config
    text_config = model_config.get_text_config()
    param = next(iter(model.parameters()))
    device, dtype = param.device, param.dtype
    heads = int(text_config.num_attention_heads)
    head_dim = int(text_config.head_dim)
    shape = (1, heads, SDPA_PROBE_POSITIONS, head_dim)
    query, key, value = (torch.empty(shape, device=device, dtype=dtype) for _ in range(3))
    boolean_mask = torch.ones(
        (1, 1, SDPA_PROBE_POSITIONS, SDPA_PROBE_POSITIONS), device=device, dtype=torch.bool
    ).tril()
    fused_choice = cast("Any", torch)._fused_sdp_choice  # noqa: SLF001
    sdpa_probe: dict[str, object] = {
        "head_dim": head_dim,
        "num_heads": heads,
        "dtype": str(dtype),
        "device": str(device),
        "causal_no_mask": sdpa_backend_name(
            int(fused_choice(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True))
        ),
        "boolean_mask": sdpa_backend_name(
            int(
                fused_choice(
                    query, key, value, attn_mask=boolean_mask, dropout_p=0.0, is_causal=False
                )
            )
        ),
    }
    return {
        "hf_attn_implementation": model_config._attn_implementation,  # noqa: SLF001
        "sdpa_toggles_enabled": {
            name: bool(getattr(torch.backends.cuda, toggle)())
            for name, toggle in SDPA_TOGGLES.items()
        },
        "sdpa_probe": sdpa_probe,
        "vllm": describe_vllm_attention(trainer),
    }


def describe_vllm_attention(trainer: GRPOTrainer) -> dict[str, object]:
    """Name the attention backend of every KV-cache group in the colocated engine, plus what was asked for."""
    if not (trainer.use_vllm and trainer.vllm_mode == "colocate"):
        return {"available": False, "reason": "no colocated vLLM engine in this process"}
    llm = cast("Any", trainer.vllm_generation).llm
    requested = llm.llm_engine.vllm_config.attention_config.backend
    facts: dict[str, object] = {
        "available": True,
        "requested_backend": None if requested is None else str(requested.name),
        "env_VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
    }
    try:
        groups = llm.llm_engine.model_executor.driver_worker.model_runner.attn_groups
    except AttributeError as error:
        logger.warning(
            "could not read vLLM's resolved attention backends; the engine's private layout "
            "moved, so run_config.json records the request only: %s",
            error,
        )
        facts["resolved_backends_error"] = str(error)
        return facts
    facts["kv_cache_groups"] = [
        [
            {
                "backend": str(group.backend.get_name()),
                "kv_cache_spec": type(group.kv_cache_spec).__name__,
                "layers": len(group.layer_names),
            }
            for group in kv_cache_group
        ]
        for kv_cache_group in groups
    ]
    return facts


def record_attention_backends(prepared: PreparedRun, trainer: GRPOTrainer) -> dict[str, object]:
    """Append the built trainer's attention facts to `derived` and to the launch record on disk.

    Read-modify-write of the record `_prepare_run` wrote rather than a fresh payload, so the
    `started_at` and provenance of the launch stay what they were; only `derived.attention` is new.
    """
    facts = describe_attention_backends(trainer)
    logger.info("attention backends, %s", json.dumps(facts, default=str, sort_keys=True))
    prepared.derived["attention"] = facts
    record = cast("dict[str, object]", json.loads(prepared.record_path.read_text(encoding="utf-8")))
    cast("dict[str, object]", record["derived"])["attention"] = facts
    write_json(prepared.record_path, record)
    return facts


def check_built_trainer(  # noqa: PLR0913  -- one keyword per fact the check compares, on primitives
    trainer: GRPOTrainer,
    *,
    expected_linear_attention_layers: int,
    meta_parameter_count: int,
    episodes_per_step: int,
    gradient_accumulation_steps: int,
    dynamic_sampling_oversample: int = 1,
) -> dict[str, object]:
    """Verify what TRL actually built, and return the checks for the run summary.

    Shared by both trainers, on primitives because their `PreparedRun` types are unrelated. The games
    body is the one kept verbatim: it also logs the meta-device parameter estimate against the loaded
    count and the accumulation steps beside the trace verdict, which the reward-hacking copy lacked, so
    sharing gives that thread two log lines it did not have and costs games nothing.
    """
    lora_summary = summarize_lora_for_architecture(
        trainer, expected_linear_attention_layers=expected_linear_attention_layers
    )
    logger.info(
        "parameter count check, %s",
        f"meta_estimate={meta_parameter_count} loaded={lora_summary['total_params']}",
    )
    logger.info("LoRA: %s", lora_summary)
    # The authoritative process count, after accelerate has had its say: the launch environment is
    # a proxy, this is what TRL will actually shard the batch across.
    assert_single_process(trainer.accelerator.num_processes, source="trainer accelerator")
    args = cast("GRPOConfig", trainer.args)
    trace_complete = check_trace_completeness(
        generation_batch_size=cast("int", args.generation_batch_size),
        logging_steps=args.logging_steps,
        episodes_per_step=episodes_per_step,
        dynamic_sampling_oversample=dynamic_sampling_oversample,
    )
    logger.info(
        "rollout trace, %s",
        f"{trace_complete=} generation_batch_size={args.generation_batch_size} "
        f"steps_per_generation={args.steps_per_generation} "
        f"{gradient_accumulation_steps=} {dynamic_sampling_oversample=}",
    )
    return {
        "lora": lora_summary,
        "rollout_trace_complete": trace_complete,
        "resolved_sampler": log_resolved_sampler(trainer),
    }


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
    """
    config = prepared.config
    arm = config.game_arm
    metrics, missing = read_back_metrics(
        trainer,
        required=required_metrics_for(
            arm.grading,
            announced_rule_trust_rows=dataset_carries_announced_rule_trust_rows(prepared.dataset),
        ),
        constant_by_construction=constant_by_construction_metrics(arm.parse_penalty_mode),
    )
    trace_files = verify_trace_files(
        Path(prepared.output_dir),
        steps=trainer.state.global_step,
        # What a step GENERATED, which under dynamic sampling is wider than what it trained on: the
        # dropped groups are in the parquet too, marked by `dynamic_sampling_dropped`. Priced off the
        # plan rather than off the trainer so a run whose oversample and buffer disagree fails the
        # row count instead of being read as complete.
        expected_rows=prepared.plan.episodes_per_step * config.dynamic_sampling_oversample,
        logging_steps=config.logging_steps,
    )
    summary: dict[str, object] = {
        "arm": config.arm,
        "game_id": arm.game_id,
        "grading": arm.grading,
        "model_id": config.model_id,
        "model_source": config.load_source,
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
        "chat_template_kwargs": prepared.derived["chat_template_kwargs"],
        "resumed_from_checkpoint": prepared.resume_checkpoint,
        "sizing_clamped": prepared.plan.clamped,
        "missing_metrics": missing,
        **checks,
        "rollout_trace_config_complete": checks["rollout_trace_complete"],
        "rollout_trace_complete": bool(checks["rollout_trace_complete"]) and trace_files.complete,
        "rollout_trace_files": asdict(trace_files),
        **metrics,
        **peak_memory_gib(prepared.output_dir),
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    # Written before the caller's raise so the artifact survives a failed read-back: the training
    # itself succeeded and its checkpoints are on disk, and the summary names what is missing.
    write_json(Path(prepared.output_dir, TRAIN_SUMMARY_FILENAME), summary)
    logger.info("RESULT %s", json.dumps(summary, default=str))
    return summary, missing, trace_files


def train_game_arm(
    config: GameTrainConfig, *, kernel_bridge: dict[str, object] | None = None
) -> GRPOTrainer | None:
    """Train one arm end to end, and persist enough that a gap is a re-analysis not a re-run.

    Returns None, having written nothing, when the resume landed on a run that already reached
    `max_steps`: the relaunch of a finished stage is a no-op that says so, not a zero-step training
    run that rewrites the finished run's artifacts (`completed_run`).

    `kernel_bridge` is whatever `games.deltanet_kernels.bridge_decode_kernel` reported, passed in
    rather than called here because it has to run before any Qwen3.5 modeling import and this
    function has already done several. None records that nobody bridged, which
    `deltanet_kernel_paths` in the same artifact will then show as the torch fallback.
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
    # The trainer TRL actually built is the consumption point of the vLLM-only rule; the env
    # guard and the config upstream are proxies for what this object will generate through.
    assert_trainer_generates_through_vllm(trainer)
    record_attention_backends(prepared, trainer)
    checks = check_built_trainer(
        trainer,
        expected_linear_attention_layers=cast(
            "int", prepared.lora_targets["expected_linear_attention_layers"]
        ),
        meta_parameter_count=cast("int", prepared.derived["meta_parameter_count"]),
        episodes_per_step=prepared.plan.episodes_per_step,
        gradient_accumulation_steps=prepared.plan.gradient_accumulation_steps,
        dynamic_sampling_oversample=config.dynamic_sampling_oversample,
    )
    if config.init_adapter and prepared.resume_checkpoint is None:
        checks["init_adapter_load"] = apply_init_adapter(trainer.model, config.init_adapter)
    elif config.init_adapter:
        # The checkpoint's weights already descend from this init; re-seeding would throw away the
        # resumed steps. assert_resume_matches has verified the recorded init is this one.
        logger.info(
            "resuming from %s, so the init adapter is provenance rather than re-applied",
            prepared.resume_checkpoint,
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
    if config.record_retention_manifest:
        manifest = write_retention_manifest(Path(prepared.output_dir))
        logger.info(
            "retention manifest written, checkpoints=%d incomplete=%d bytes=%d",
            len(manifest.retained),
            len(manifest.incomplete),
            manifest.checkpoint_bytes_after,
        )
    if config.s3_dest:
        # The callback's on_train_end fires inside trainer.train(), BEFORE save_model, save_state
        # and the summary write above, so without this the bucket's copy of every finished run is
        # missing train_summary.json and the final trainer state (observed on both 2026-08-19
        # arms). Before the missing-metrics raise on purpose: the summary that names what is
        # missing is exactly the artifact worth having off-box, and sync_directory reports rather
        # than raises, so a network failure cannot convert a finished run into a crashed one.
        sync_directory(Path(prepared.output_dir), config.s3_dest)
    if missing:
        raise RuntimeError(
            f"expected metrics never reached trainer_state, {missing=}. The reward function's "
            f"log_metric calls are the only route for these; a metric nobody reads back is not "
            f"recorded."
        )
    if not trace_files.complete:
        raise RuntimeError(
            f"the rollout trace under {trace_files.checked_dir} has gaps: missing steps "
            f"{list(trace_files.missing_steps)}, empty files at steps {list(trace_files.empty_steps)}, "
            f"wrong row counts (step, rows) {[list(pair) for pair in trace_files.wrong_row_count_steps]} "
            f"against {trace_files.expected_rows_per_step} per step. The summary records this as "
            f"rollout_trace_complete=false; the per-completion trace is the run's raw material, so a "
            f"run without all of it is not a finished measurement."
        )
    return trainer


def _config_from_namespace(args: argparse.Namespace) -> GameTrainConfig:
    """Resolve the defaults that depend on other flags, then build the config.

    Three defaults cannot be expressed in argparse because they depend on another argument: the
    model (the smoke tier unless one was named), the completion budget (per-model measured floor),
    and generate-fresh (implied by smoke).
    """
    values = vars(args)
    smoke = bool(values.pop("smoke"))
    model_id = values.pop("model_id") or (SMOKE_MODEL_ID if smoke else DEFAULT_MODEL_ID)
    if values["max_completion_tokens"] is None:
        values["max_completion_tokens"] = required_completion_budget(model_id)
    values["generate_fresh"] = bool(values["generate_fresh"]) or smoke
    # `smoke` goes into the FIRST construction, not only the shrunk one: validation runs in
    # __post_init__, and the pre-shrink intermediate still carries the full completion budget the
    # smoke overrides are about to replace, so the smoke exemptions must already apply to it.
    config = GameTrainConfig(model_id=model_id, smoke=smoke, **values)
    return shrink_for_smoke(config) if smoke else config


def _add_sampler_mismatch_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the five knobs that decide what a run does about the vLLM-versus-trainer mismatch.

    Grouped in their own function because they are read as a set and only make sense together: the
    correction computes the ratio, the mode decides what the ratio is computed over and what happens
    to one outside the clip band, log-only stops it reaching the gradient, the fp32 head makes the
    ratio worth reading, and the token chunk makes the pass that produces it fit at a long
    completion budget. Every default is today's behaviour.
    """
    parser.add_argument(
        "--no-vllm-importance-sampling-correction",
        dest="vllm_importance_sampling_correction",
        action="store_false",
        default=colocate_importance_sampling_correction(),
        help=(
            f"drop TRL's correction for the mismatch between vLLM-sampled tokens and this model's "
            f"own log-probabilities. Leaves the estimator no longer exactly on-policy, which is a "
            f"caveat on absolute levels and is announced in the log. The correction's extra "
            f"full-length forward pass went out of memory at the production shape. Defaults from "
            f"{VLLM_IS_CORRECTION_ENV}."
        ),
    )
    parser.add_argument(
        "--vllm-importance-sampling-mode",
        dest="vllm_importance_sampling_mode",
        default=VLLM_IMPORTANCE_SAMPLING_MODE,
        choices=VLLM_IMPORTANCE_SAMPLING_MODES,
        help=(
            f"what TRL's correction does with the vLLM-versus-trainer log-probability difference. "
            f"Two axes: granularity (one ratio per token, or one per sequence broadcast over its "
            f"tokens) and constraint (clip the ratio into [C_min, C_max], or zero it outside them). "
            f"Default {VLLM_IMPORTANCE_SAMPLING_MODE}, TRL's own, under which a whole rollout is "
            f"discarded whenever its accumulated ratio leaves the band -- which a product of "
            f"thousands of near-one per-token ratios does on length alone (the 2B probe measured "
            f"sequence ratios of 0.66 and 0.21 at a per-token difference of 0.015 nats). "
            f"token_truncate clips each token's own ratio instead, and is what arm 1 and its "
            f"control run if the 9B probe confirms a large mismatch (owner, 2026-09-04). Refused "
            f"away from the default with the correction off, which computes no ratio to shape."
        ),
    )
    parser.add_argument(
        "--vllm-importance-sampling-log-only",
        dest="vllm_importance_sampling_log_only",
        action="store_true",
        help=(
            "compute the correction and log its ratio statistics, then weight the gradient by "
            "exactly one, so the vLLM-versus-trainer mismatch is measured without changing what "
            "the run trains. Needs the correction ON, which is what computes the ratio."
        ),
    )
    parser.add_argument(
        "--cast-lm-head-to-fp32",
        dest="cast_lm_head_to_fp32",
        action="store_true",
        help=(
            "run this model's language-model head in fp32 (TRL's own knob). It reaches the "
            "old-log-probability pass and a non-Liger loss; it does NOT reach the Liger loss this "
            "trainer runs by default, which is handed lm_head.weight directly and casts each vocab "
            "chunk of it back to bfloat16. Refused without the correction, which it must accompany "
            "rather than replace: vLLM keeps sampling in bfloat16 either way, so alone it moves the "
            "mismatch instead of removing it. Refused on a tied-embedding checkpoint, which is every "
            "rung of this ladder below the 9B, because TRL's cast crashes there."
        ),
    )
    parser.add_argument(
        "--old-logps-chunk-tokens",
        dest="old_logps_chunk_tokens",
        type=int,
        default=OLD_LOGPS_CHUNK_TOKENS,
        help=(
            f"positions per slice of the correction's old-log-probability pass. It moves the memory "
            f"peak, from one (positions x vocab x 4 bytes) copy per micro-batch row to one "
            f"(chunk x vocab x 4 bytes) copy per row plus one reduction temporary, and leaves the "
            f"log-probabilities where they were to within the head matmul's own reassociation. "
            f"Default {OLD_LOGPS_CHUNK_TOKENS}."
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> GameTrainConfig:  # noqa: PLR0915  -- one statement per flag
    """Parse one arm's training configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    # None so that --smoke can default the model without overriding an explicit choice.
    parser.add_argument("--model", dest="model_id", default=None)
    parser.add_argument(
        "--model-source",
        default=None,
        help=(
            "Immutable local snapshot to load while --model remains the canonical model identity "
            "used for sampler and provenance decisions."
        ),
    )
    parser.add_argument("--corpus", dest="corpus_path", default=None)
    parser.add_argument("--generate-fresh", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=70)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    # None so the default resolves PER MODEL below: the floor is a property of the checkpoint's
    # measured termination length, so one constant is too small for a verbose model and needlessly
    # large for a terse one.
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=TRAINING_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TRAINING_TOP_P)
    parser.add_argument("--top-k", type=int, default=TRAINING_TOP_K)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=GRPO_EPSILON)
    parser.add_argument(
        "--loss-type",
        default=GRPO_LOSS_TYPE,
        choices=GRPO_LOSS_TYPES,
        help=(
            f"token-loss aggregation; the default and the reasons live in "
            f"grpo/estimator_defaults.py. With Liger on, {LIGER_UNFAITHFUL_LOSS_TYPES} do not "
            f"execute their paper semantics and are refused unless "
            f"--acknowledge-liger-estimator-mismatch is passed."
        ),
    )
    parser.add_argument(
        "--scale-rewards",
        default=GRPO_SCALE_REWARDS,
        choices=GRPO_SCALE_REWARDS_MODES,
        help=(
            "advantage divisor; the default and the reasons live in grpo/estimator_defaults.py. "
            "'group' cancels the payoff gap on two-point group rewards and would erase the "
            "payoff manipulations these arms study."
        ),
    )
    parser.add_argument(
        "--acknowledge-liger-estimator-mismatch",
        action="store_true",
        help=(
            "state that running a Liger-unfaithful loss_type is deliberate: the EXECUTED "
            "aggregation (see executed_estimator in run_config.json), not the paper one the name "
            "promises, is the experiment. Recorded in run_config.json; refused when nothing "
            "needs acknowledging."
        ),
    )
    parser.add_argument(
        "--adam-epsilon",
        type=float,
        default=1e-8,
        help=(
            "AdamW's denominator floor. The default is transformers' own, which is what every "
            "banked arm trained under; 1e-15 unfreezes the LoRA A matrices (see "
            "GameTrainConfig.adam_epsilon for the measurement) and is a treatment change, "
            "recorded in run_config.json and checked on resume like the arm itself."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help=(
            "adapter dropout during the graded forward. vLLM sampled the rollout without it, so "
            "any nonzero value is a train-inference mismatch; 0 is what the RL recipes run. A "
            "treatment change like --adam-epsilon, and checked on resume for the same reason."
        ),
    )
    parser.add_argument("--no-liger", dest="use_liger_kernel", action="store_false")
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        help="PLUMBING ONLY: run with thinking disabled. Labelled loudly in the log and artifacts.",
    )
    parser.add_argument(
        "--allow-short-completions",
        action="store_true",
        help=(
            f"Run thinking-ON below the measured {MEASURED_TERMINATION_BUDGET}-token floor. For a "
            f"timing probe or a deliberate diagnostic only."
        ),
    )
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument(
        "--dynamic-sampling-oversample",
        dest="dynamic_sampling_oversample",
        type=int,
        default=1,
        help=(
            "generate this many times the optimizer step's prompt groups, score all of them, and "
            "train on the first live ones in sampler order -- a group whose completions all earned "
            "the same reward has zero advantages and costs a full forward and backward for no "
            "gradient. Default 1, which is off. Every generated completion still reaches the "
            "completions parquet (dropped rows marked by the dynamic_sampling_dropped column) and "
            "every reward-side metric, so the strata that mostly go pure stay measurable; per-step "
            "counts land in log_history and mem_log.csv under dynamic_sampling/*. A treatment, "
            "recorded in run_config.json and checked on resume like the arm itself, and the corpus "
            "must hold oversample x prompts-per-step prompts or TRL's sampler yields nothing."
        ),
    )
    parser.add_argument("--parse-penalty", type=float, default=-1.0)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--no-autosize", dest="autosize", action="store_false")
    parser.add_argument("--vram-usable-fraction", type=float, default=DEFAULT_VRAM_USABLE_FRACTION)
    # No backend flag: rollouts are vLLM-only (games.generation.VLLM_ONLY_RATIONALE), and an
    # environment still exporting the old switch is refused at startup by `assert_vllm_rollouts`.
    # The engine knobs below default from the environment for the same reason --s3-dest takes
    # one: they describe the box, and the stage plans in `games/` render them from these same
    # readers.
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=colocate_gpu_fraction(),
        help=(
            f"the colocated engine's share of TOTAL card VRAM, held for the whole run and "
            f"subtracted from what the sizing plan may spend. Defaults from "
            f"{VLLM_GPU_FRACTION_ENV}, else the measured {VLLM_COLOCATE_GPU_FRACTION}."
        ),
    )
    _add_sampler_mismatch_arguments(parser)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument(
        "--record-retention-manifest",
        action="store_true",
        help="write a hashed run-owned checkpoint inventory after each save",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        dest="resume_from_checkpoint",
        default="",
        help=(
            "checkpoint directory to resume from, or 'latest' to pick the newest one "
            "already in --output-dir. Requires an explicit --output-dir."
        ),
    )
    parser.add_argument(
        "--init-adapter",
        dest="init_adapter",
        default="",
        help=(
            "PEFT adapter checkpoint directory whose weights seed this run's LoRA before step 0: "
            "continue-RL under a new arm, with a fresh optimizer, scheduler and step counter. "
            "Refused unless the adapter matches this run's base model, rank, alpha and discovered "
            "target modules exactly. Once --output-dir holds a checkpoint, --resume-from-checkpoint "
            "wins and this records provenance only, so the flag is safe in a re-run recovery."
        ),
    )
    # Defaults from the environment because that is where the Batch path already carries it:
    # `cloud/submit_job.py --s3-dest` sets GAMES_S3_DEST on the container, `cloud/entrypoint.sh`
    # reads the same variable for its exit-trap sync, and nothing on that path passes a flag. A
    # flag-only default would silently leave the in-run liveness sync off on every Batch job, or
    # duplicate the URI in two places that could disagree.
    parser.add_argument("--s3-dest", default=os.environ.get("GAMES_S3_DEST", ""))
    args = parser.parse_args(argv)

    config = _config_from_namespace(args)
    if config.output_dir is None:
        config = replace(
            config,
            output_dir=default_output_dir(
                config.arm, config.model_id, timestamp=datetime.now(tz=UTC), smoke=config.smoke
            ),
        )
    return config


def main(argv: Sequence[str] | None = None) -> None:
    """Train the arm named on the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Before anything else spends a cent: an environment exporting the removed backend switch is
    # refused here, not read. Config construction repeats the check for programmatic callers.
    assert_vllm_rollouts()
    default_cuda_allocator_config()
    # First, before `discover_lora_targets` reaches for `transformers.Qwen3_5ForCausalLM` and pulls
    # in the modeling module: transformers binds each Gated DeltaNet kernel at that module's import
    # time, so a bridge applied afterwards silently leaves decode on the pure-torch loop.
    kernel_bridge = dict(bridge_decode_kernel())
    logger.info("deltanet decode bridge: %s", kernel_bridge)
    # transformers filters the keywords it passes down to the implementation's own parameter names
    # and drops the rest without a word, so a decode kernel whose signature is missing one -- fla's
    # `initial_state`, say -- would run fast and compute a different recurrence for every rollout of
    # the run. An AST read of the call site plus a signature check, so it imports no modeling module
    # and cannot itself make the bridge too late.
    call_site = assert_bridged_kernel_matches_call_site()
    kernel_bridge["decode_call_site"] = {
        "positional_count": call_site.positional_count,
        "keyword_names": sorted(call_site.keyword_names),
    }
    train_game_arm(_parse_args(argv), kernel_bridge=kernel_bridge)


if __name__ == "__main__":
    main()
