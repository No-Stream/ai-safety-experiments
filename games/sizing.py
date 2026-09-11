"""Derive an episode batch from the VRAM a process actually found, not from a constant.

`CLAUDE.md`: never hardcode a memory budget. The 24 GiB figure on this dev box must not
propagate into a cloud job config, because a run that assumes it cannot take whatever card is
free, and a job that OOMs at step 40 has spent its whole allocation to report nothing.

Everything here is pure arithmetic over a checkpoint's own config fields, so it is testable
without a download, a GPU, or a model load. The cost terms are the ones
`grpo.throughput.memory_model` derives and `docs/scratch/measured-throughput.md` measured against
a running process, restated per-sequence here.

**`sequence_cost` and `grpo.throughput.memory_model` are two copies of the same arithmetic, and
they now disagree in two known ways** -- both corrections landed here and never went back:

1.  KV-caching layer types. `KV_CACHING_LAYER_TYPES` below counts `full_attention` *and*
    `sliding_attention`; `memory_model` counts only `full_attention`. On a checkpoint with
    sliding-window layers, that copy under-counts the cache, which is the direction that OOMs.
2.  The no-linear-fields guard. `sequence_cost` reads the `linear_*` fields only when the config
    declares linear-attention layers; `memory_model` reads them unconditionally, so it raises
    `AttributeError` on a plain-attention stack such as Qwen3-0.6B.

Neither can currently corrupt a training plan: `memory_model` is reachable only from
`grpo.throughput.predict_memory` -> `measure`, the standalone benchmark CLI, while every games
training path calls `sequence_cost`. Unifying them means moving `SequenceCost` into
`grpo/throughput.py` and importing it here, because this module already depends on `grpo.throughput`
for its constants and the reverse import would be a cycle. That edit belongs to `grpo/`, so this
note stands until someone who owns that package makes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import torch
import transformers
from transformers import AutoConfig

from grpo.throughput import BYTES_PER_GIB, MIN_GRPO_GROUP_SIZE

logger = logging.getLogger(__name__)

# Layer types holding a per-token KV cache. Sliding-window attention caches only its window, so
# counting it in full overstates the cost, which is the right direction to be wrong in.
KV_CACHING_LAYER_TYPES = ("full_attention", "sliding_attention")
LINEAR_ATTENTION_LAYER_TYPE = "linear_attention"

# The per-episode arithmetic covers the KV cache, the DeltaNet recurrent state and the
# chunked-prefill upcast. It does not cover the CUDA context, allocator fragmentation or the
# LoRA optimizer state, so a plan only ever spends this fraction of what the card reports.
DEFAULT_VRAM_USABLE_FRACTION = 0.9

BYTES_PER_FLOAT32 = 4
BYTES_PER_BFLOAT16 = 2

# The per-sequence arithmetic here is a LOWER BOUND, not a budget, and this is the measured
# constant that turns it into a predictor. `docs/scratch/measured-throughput.md` ran the real
# thing on this box and found the config-derived prediction accounted for only ~60% of observed
# peak: at 4B, 16 episodes, 2048+2048, predicted 13.27 GiB met a measured 18.10 GiB. The 4.83 GiB
# difference is optimizer state, gradients, activations and DeltaNet chunk intermediates that no
# config field exposes. Carrying that same offset to 32 episodes predicted ~22.8 GiB against a
# 22.06 GiB card, which is exactly the OOM that was observed -- so arithmetic plus this constant
# is a serviceable predictor where the arithmetic alone is not. That doc also notes the true
# offset grows with batch size, so treat it as understated for a much larger batch.
UNMODELLED_OVERHEAD_GIB = 4.83

# Training-pass activations scale with micro-batch times tokens per sequence, and that ceiling is
# a separate question from how many episodes fit. Every 4B configuration that ran on the L4 had
# micro-batch x (prompt + completion) at or under 8,704 tokens -- micro 8 at 512+512, micro 4 at
# 1024+256, micro 2 at 4096+256, micro 2 at 2048+2048 -- while 16,384 tokens OOM'd inside a LoRA
# projection after generating perfectly well. Both figures from the same measured table.
MEASURED_MICRO_BATCH_TOKENS = 8704
# Non-weight VRAM the card that measurement ran on had for the training pass: 22.06 GiB usable
# minus 8.51 GiB of 4B bf16 weights. The token ceiling above is scaled by the ratio of a run's own
# non-weight headroom to this, so a larger card affords a larger micro-batch. That is a linear
# extrapolation from a single measured point, not a measured law.
MEASURED_MICRO_BATCH_HEADROOM_GIB = 13.55


@dataclass(frozen=True)
class SequenceCost:
    """Per-episode VRAM cost predicted from one checkpoint's architecture fields."""

    kv_bytes_per_token: int
    recurrent_bytes_per_sequence: int
    prefill_upcast_bytes_per_token: int
    n_kv_caching_layers: int
    n_linear_attention_layers: int

    def kv_cache_bytes(self, *, prompt_tokens: int, completion_tokens: int) -> int:
        """Return the KV cache one sequence holds, the only term that grows with length."""
        return self.kv_bytes_per_token * (prompt_tokens + completion_tokens)

    def prefill_transient_bytes(self, *, prompt_tokens: int) -> int:
        """Return the recurrent state plus the chunked-prefill upcast, both flat in output length.

        Grouped because they are the two terms a *short* generation pays and a long one is dominated
        by the KV cache instead. `games/select_prompts.py` needs the split to calibrate each half
        against its own measurement: the peak it measured was identical at 256 and 1,024 new tokens,
        so one factor over the whole sum cannot fit both regimes.
        """
        return (
            self.recurrent_bytes_per_sequence + self.prefill_upcast_bytes_per_token * prompt_tokens
        )

    def gib_per_episode(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        """Predict one episode's transient VRAM at the given token profile."""
        total_bytes = self.kv_cache_bytes(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ) + self.prefill_transient_bytes(prompt_tokens=prompt_tokens)
        return total_bytes / BYTES_PER_GIB


def sequence_cost(text_config: object) -> SequenceCost:
    """Read the per-episode cost terms out of a text config."""
    config = cast("transformers.PretrainedConfig", text_config)
    layer_types = list(config.layer_types)  # pyright: ignore[reportAttributeAccessIssue]
    n_kv = sum(1 for layer in layer_types if layer in KV_CACHING_LAYER_TYPES)
    n_linear = sum(1 for layer in layer_types if layer == LINEAR_ATTENTION_LAYER_TYPE)

    # Key and value, one entry per token per caching layer, two bytes each under bf16.
    kv_bytes_per_token = (
        2
        * n_kv
        * config.num_key_value_heads  # pyright: ignore[reportAttributeAccessIssue]
        * config.head_dim  # pyright: ignore[reportAttributeAccessIssue]
        * BYTES_PER_BFLOAT16
    )
    recurrent_bytes = 0
    prefill_upcast_bytes = 0
    if n_linear:
        # One (num_value_heads, key_head_dim, value_head_dim) matrix per linear-attention layer
        # per sequence, fixed in context length. transformers holds it in float32 and never casts
        # it back, so the model dtype is the wrong number to use here even under bf16 weights.
        recurrent_bytes = (
            n_linear
            * config.linear_num_value_heads  # pyright: ignore[reportAttributeAccessIssue]
            * config.linear_key_head_dim  # pyright: ignore[reportAttributeAccessIssue]
            * config.linear_value_head_dim  # pyright: ignore[reportAttributeAccessIssue]
            * BYTES_PER_FLOAT32
        )
        # The chunked prefill path upcasts query, key and value to float32 and makes contiguous
        # copies of all three across the batch. Measured to be the binding constraint on a 24 GiB
        # card, rather than the KV cache a budget model would reason about.
        prefill_upcast_bytes = (
            3
            * config.linear_num_value_heads  # pyright: ignore[reportAttributeAccessIssue]
            * config.linear_value_head_dim  # pyright: ignore[reportAttributeAccessIssue]
            * BYTES_PER_FLOAT32
        )
    return SequenceCost(
        kv_bytes_per_token=kv_bytes_per_token,
        recurrent_bytes_per_sequence=recurrent_bytes,
        prefill_upcast_bytes_per_token=prefill_upcast_bytes,
        n_kv_caching_layers=n_kv,
        n_linear_attention_layers=n_linear,
    )


def checkpoint_sequence_cost(model_id: str) -> SequenceCost:
    """Read the cost terms straight from a hub checkpoint's config."""
    return sequence_cost(AutoConfig.from_pretrained(model_id).get_text_config())


def count_meta_parameters(model_id: str) -> int:
    """Count a checkpoint's parameters by building its architecture on the meta device.

    Config only: no weights are downloaded and no VRAM is touched. On a checkpoint carrying
    extras that a text-only load drops -- Qwen3.8-27B's vision tower and MTP head -- this
    overstates the count by a few percent, biasing the plan toward fitting rather than OOMing.
    """
    config = AutoConfig.from_pretrained(model_id)
    architecture = getattr(
        transformers,
        config.architectures[0],  # pyright: ignore[reportOptionalSubscript]
    )
    with torch.device("meta"):
        model = architecture(config)
    return sum(parameter.numel() for parameter in model.parameters())


@dataclass(frozen=True)
class SizingPlan:
    """What the card can actually run, plus the arithmetic that said so."""

    num_generations: int
    prompts_per_step: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    episodes_per_step: int
    predicted_weight_gib: float
    predicted_episode_gib: float
    usable_vram_gib: float
    episode_headroom_gib: float
    episodes_that_fit: int
    clamped: bool
    reason: str
    # What another resident of the card holds, which is why `usable_vram_gib` can be well under the
    # card's own free VRAM. Zero on the transformers-generate path, where the trainer has it all.
    engine_reserved_gib: float = 0.0


def colocate_reserved_gib(*, total_vram_gib: float, gpu_memory_utilization: float) -> float:
    """Return the VRAM a colocated vLLM engine takes out of the card before the trainer allocates.

    vLLM reads `gpu_memory_utilization` as a fraction of the card's TOTAL memory rather than of
    what happens to be free, and it holds that block for the life of the process. Sizing runs
    before the trainer exists and therefore before the engine does, so on a colocate run the free
    VRAM this process measured describes a card that is about to lose a third of itself:
    subtracting this is what stops a plan from budgeting episodes into memory the engine has
    already claimed.

    The per-episode side of the same fact -- that under colocate the episodes' KV lives in the
    engine's cache and out of the trainer's allocator -- is handled by `plan_sizing`'s
    `generation_schedules_own_batch`, which skips the episode clamp rather than re-modelling the
    cost: the reservation subtracted here is what pays for the episodes, and charging the HF
    per-episode arithmetic on top of it double-counted the same memory and clamped batches the
    card would hold.
    """
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1), {gpu_memory_utilization=}: the engine "
            f"holds that share of the card for the whole run, so the trainer needs the rest"
        )
    return total_vram_gib * gpu_memory_utilization


def _vram_left_to_the_trainer(
    *, free_vram_gib: float, engine_reserved_gib: float, label: str
) -> float:
    """Take a resident engine's block off the card, refusing what leaves the trainer nothing."""
    if engine_reserved_gib < 0:
        raise ValueError(f"engine_reserved_gib cannot be negative, {engine_reserved_gib=}")
    if engine_reserved_gib >= free_vram_gib:
        raise RuntimeError(
            f"a resident engine holding {engine_reserved_gib:.1f} GiB leaves nothing of the "
            f"{free_vram_gib:.1f} GiB free{_suffix(label)}. Lower its memory utilization or use a "
            f"larger card."
        )
    return free_vram_gib - engine_reserved_gib


def plan_sizing(  # noqa: PLR0913
    *,
    num_generations: int,
    prompts_per_step: int,
    micro_batch_size: int | None,
    max_prompt_tokens: int,
    max_completion_tokens: int,
    cost: SequenceCost,
    free_vram_gib: float,
    weights_gib: float,
    usable_fraction: float = DEFAULT_VRAM_USABLE_FRACTION,
    autosize: bool = True,
    label: str = "",
    engine_reserved_gib: float = 0.0,
    generation_schedules_own_batch: bool = False,
) -> SizingPlan:
    """Fit the requested episode batch onto the VRAM this process found.

    Clamps prompts per step first and the group size only as a last resort: the group size is a
    property of the experiment, since GRPO computes its advantage baseline within a group, while
    the number of prompts per step is bookkeeping.

    It refuses rather than shortening the token budget. Truncated thinking is a *measurement* in
    these arms -- `truncated_thinking_rate` is one of the metrics a run reports -- so silently
    halving the completion budget to make a config fit would corrupt the number instead of the
    run, which is far harder to notice afterwards.

    `engine_reserved_gib` is VRAM some other resident of the card holds for the whole run, and it
    comes off the top before anything else is counted -- in practice a colocated vLLM engine, whose
    block is not yet allocated when this runs and so is invisible in the free-VRAM reading it is
    given. See `colocate_reserved_gib`.

    `generation_schedules_own_batch` says episodes decode inside that resident engine's own paged
    cache rather than in this process's allocator (`games.chunked_decode` uses the same words for
    the same fact). The per-episode arithmetic here prices a transformers.generate episode -- KV
    cache, DeltaNet state, prefill upcast -- and under colocate every one of those lives inside the
    engine's reservation, which the plan has *already* paid for through `engine_reserved_gib`.
    Charging both is double-counting, and it clamps batches the card would hold: at the production
    shape (64 episodes of 1024+32768 on a 95 GiB card) the HF arithmetic prices ~0.8 GiB/episode
    against headroom that no longer needs to hold any of it. So under a self-scheduling engine the
    episode clamp is skipped -- the arithmetic is still computed and reported, since a wildly
    optimistic `episodes_that_fit` is worth seeing in the log -- and what actually bounds the
    trainer is the micro-batch token budget, which is derived below either way. The engine's own
    init validates its half of the card and fails loudly before any step is paid for.

    `label` names the model or run in the failure messages, since an operator reading them is
    deciding which card to ask for next.
    """
    if num_generations < MIN_GRPO_GROUP_SIZE:
        raise ValueError(
            f"GRPO needs at least {MIN_GRPO_GROUP_SIZE} generations, {num_generations=}"
        )
    if prompts_per_step < 1:
        raise ValueError(f"need at least one prompt per step, {prompts_per_step=}")
    if not 0 < usable_fraction <= 1:
        raise ValueError(f"usable_fraction must be in (0, 1], {usable_fraction=}")

    free_for_trainer = _vram_left_to_the_trainer(
        free_vram_gib=free_vram_gib, engine_reserved_gib=engine_reserved_gib, label=label
    )
    usable = free_for_trainer * usable_fraction
    headroom = usable - weights_gib - UNMODELLED_OVERHEAD_GIB
    if headroom <= 0:
        raise RuntimeError(
            f"nothing left for episodes{_suffix(label)}: {weights_gib:.1f} GiB of weights plus "
            f"{UNMODELLED_OVERHEAD_GIB} GiB of measured unmodelled overhead against {usable:.1f} "
            f"GiB usable of {free_for_trainer:.1f} GiB free to the trainer, itself "
            f"{free_vram_gib:.1f} GiB free on the card less {engine_reserved_gib:.1f} GiB held by "
            f"a resident engine. Use a larger card."
        )
    per_episode = cost.gib_per_episode(
        prompt_tokens=max_prompt_tokens, completion_tokens=max_completion_tokens
    )
    requested = num_generations * prompts_per_step
    if per_episode <= 0:
        raise RuntimeError(
            f"predicted zero VRAM per episode{_suffix(label)}, so the plan would be meaningless; "
            f"the config's layer types and head dims are {cost}"
        )
    episodes_that_fit = int(headroom // per_episode)

    clamped = episodes_that_fit < requested and autosize and not generation_schedules_own_batch
    if clamped:
        planned_generations, planned_prompts, reason = _clamp_to_fit(
            num_generations=num_generations,
            prompts_per_step=prompts_per_step,
            episodes_that_fit=episodes_that_fit,
            headroom_gib=headroom,
            per_episode_gib=per_episode,
            token_profile=f"{max_prompt_tokens}+{max_completion_tokens}",
            label=label,
        )
    else:
        planned_generations = num_generations
        planned_prompts = prompts_per_step
        reason = (
            f"requested {requested} episodes/step fits: {headroom:.1f} GiB headroom at "
            f"{per_episode:.3f} GiB/episode allows {episodes_that_fit}"
        )
        if episodes_that_fit < requested and generation_schedules_own_batch:
            reason = (
                f"running the requested {requested} episodes/step: they decode inside the "
                f"resident engine's paged cache, already paid for above, so the trainer-side "
                f"arithmetic (predicting {episodes_that_fit} fit in {headroom:.1f} GiB at "
                f"{per_episode:.3f} GiB/episode) prices a transformers.generate path this run "
                f"does not take"
            )
            logger.info("episode clamp skipped for a self-scheduling engine, %s", reason)
        elif episodes_that_fit < requested:
            reason = (
                f"autosize off: running {requested} episodes/step although the arithmetic "
                f"predicts only {episodes_that_fit} fit in {headroom:.1f} GiB"
            )
            logger.warning("sizing not clamped and predicted to overflow, %s", reason)

    episodes = planned_generations * planned_prompts
    micro, micro_reason = _resolve_micro_batch(
        requested=micro_batch_size,
        num_generations=planned_generations,
        episodes=episodes,
        tokens_per_sequence=max_prompt_tokens + max_completion_tokens,
        free_vram_gib=free_for_trainer,
        weights_gib=weights_gib,
    )
    if engine_reserved_gib:
        reason = (
            f"{reason}; after {engine_reserved_gib:.1f} GiB of the card went to a resident engine"
        )
    return SizingPlan(
        num_generations=planned_generations,
        prompts_per_step=planned_prompts,
        micro_batch_size=micro,
        gradient_accumulation_steps=episodes // micro,
        episodes_per_step=episodes,
        predicted_weight_gib=weights_gib,
        predicted_episode_gib=per_episode,
        usable_vram_gib=usable,
        episode_headroom_gib=headroom,
        episodes_that_fit=episodes_that_fit,
        clamped=clamped,
        reason=f"{reason}; {micro_reason}",
        engine_reserved_gib=engine_reserved_gib,
    )


def _resolve_micro_batch(  # noqa: PLR0913
    *,
    requested: int | None,
    num_generations: int,
    episodes: int,
    tokens_per_sequence: int,
    free_vram_gib: float,
    weights_gib: float,
) -> tuple[int, str]:
    """Settle the training micro-batch, deriving one when the caller did not name it.

    An explicit request wins, with a warning if it is over the measured ceiling, because an
    operator naming a number is usually testing exactly that. Either way the two divisibility
    invariants TRL relies on are enforced here, with the arithmetic in the message, rather than
    surfacing from inside GRPOConfig.
    """
    budget = micro_batch_token_budget(free_vram_gib=free_vram_gib, weights_gib=weights_gib)
    if requested is None:
        micro = _derive_micro_batch(
            num_generations=num_generations,
            tokens_per_sequence=tokens_per_sequence,
            token_budget=budget,
        )
        reason = f"micro-batch {micro} derived from a {budget}-token training-pass budget"
    else:
        micro = min(requested, episodes)
        reason = f"micro-batch {micro} as requested, against a {budget}-token budget"
        if micro * tokens_per_sequence > budget:
            logger.warning(
                "explicit micro-batch is over the measured training-pass ceiling, %s",
                f"{micro=} {tokens_per_sequence=} {budget=}",
            )
    if episodes % micro:
        raise ValueError(
            f"micro_batch_size must divide num_generations * prompts_per_step, {micro=} {episodes=}"
        )
    # TRL derives generation_batch_size from per_device_train_batch_size *
    # gradient_accumulation_steps and requires it to be a multiple of num_generations.
    if micro % num_generations and num_generations % micro:
        raise ValueError(
            f"micro_batch_size and num_generations must divide one another, "
            f"{micro=} {num_generations=}"
        )
    return micro, reason


def micro_batch_token_budget(*, free_vram_gib: float, weights_gib: float) -> int:
    """Predict how many tokens one training forward/backward can hold on this card.

    Separate from the episode count on purpose: generation must hold all `P x G` episodes at once
    and fails in prefill, while the training pass consumes them `micro_batch` at a time and fails
    in a LoRA projection. On the L4 the reference profile only became runnable once the
    micro-batch dropped to 2, with the episode count unchanged, so a plan that got the episode
    count right and the micro-batch wrong still OOM'd.
    """
    non_weight_headroom = max(free_vram_gib - weights_gib, 0.0)
    scale = non_weight_headroom / MEASURED_MICRO_BATCH_HEADROOM_GIB
    # Rounded, not truncated: the anchor point sits exactly on the boundary, where float error in
    # the ratio would otherwise shave a token off the budget and halve the micro-batch on the very
    # card the constant was measured on.
    return max(round(MEASURED_MICRO_BATCH_TOKENS * scale), 1)


def _derive_micro_batch(
    *, num_generations: int, tokens_per_sequence: int, token_budget: int
) -> int:
    """Pick the largest micro-batch that divides the group size and fits the token budget.

    Restricted to divisors of the group size because TRL derives `generation_batch_size` from
    `per_device_train_batch_size * gradient_accumulation_steps` and requires it to be a multiple
    of `num_generations`; 1 always qualifies, so this always returns something runnable.
    """
    ceiling = max(token_budget // tokens_per_sequence, 1)
    divisors = [
        candidate
        for candidate in range(1, num_generations + 1)
        if num_generations % candidate == 0 and candidate <= ceiling
    ]
    return max(divisors)


def _clamp_to_fit(  # noqa: PLR0913
    *,
    num_generations: int,
    prompts_per_step: int,
    episodes_that_fit: int,
    headroom_gib: float,
    per_episode_gib: float,
    token_profile: str,
    label: str,
) -> tuple[int, int, str]:
    """Shrink an episode batch to what fits, or refuse, and say which happened.

    Prompts per step go first, the group size only if a single group will not fit, and the token
    budget never: see `plan_sizing` for why silently shortening it would corrupt a metric rather
    than a run.
    """
    prompts_that_fit = episodes_that_fit // num_generations
    if prompts_that_fit >= 1:
        return (
            num_generations,
            prompts_that_fit,
            (
                f"clamped prompts_per_step {prompts_per_step} -> {prompts_that_fit} so that "
                f"{episodes_that_fit} episodes fit in {headroom_gib:.1f} GiB at "
                f"{per_episode_gib:.3f} GiB/episode; group size held at {num_generations}"
            ),
        )
    if episodes_that_fit >= MIN_GRPO_GROUP_SIZE:
        return (
            episodes_that_fit,
            1,
            (
                f"one group of {num_generations} does not fit, so the group size itself was "
                f"clamped to {episodes_that_fit}. That changes the advantage baseline, so read "
                f"this run as plumbing rather than as a measurement"
            ),
        )
    raise RuntimeError(
        f"not even a group of {MIN_GRPO_GROUP_SIZE} fits{_suffix(label)}: {headroom_gib:.1f} GiB "
        f"headroom at {per_episode_gib:.3f} GiB/episode for {token_profile} tokens. Lower the "
        f"completion budget (which changes what the arm measures) or use a larger card; the "
        f"token budget is never shortened automatically."
    )


def _suffix(label: str) -> str:
    """Append the caller's label to a failure message when it gave one."""
    return f" for {label}" if label else ""
