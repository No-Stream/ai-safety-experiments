"""Wall-clock throughput measurement for a GRPO optimizer step.

Answers one question: how many seconds does a single optimizer step cost at a stated
model, prompts-per-step, group size and token profile. It is not a training run and it
does not care whether reward moves — every knob that trades accuracy for speed is left
at the value a real run would use, and the completions are forced to a fixed length so
the number corresponds to the token profile it is labelled with.

`docs/scratch/compute-budget-model.md` derives the repo's entire cost table by scaling a
single published 2xH100 datum across a ~10x hardware gap, and says the most valuable thing
anyone can do is measure the real number. This is that measurement. Config is passed in
rather than derived from the card, so the same script reproduces on an L40S without edits;
the device and its VRAM are recorded next to every result because the numbers mean nothing
without them.

    uv run python -m grpo.throughput --model Qwen/Qwen3.5-4B -P 8 -G 8 \
        --prompt-tokens 2048 --completion-tokens 2048

Measures one configuration and writes JSON. Sweeping several is `grpo.throughput_sweep`,
which runs each point in a fresh process — a CUDA allocator that has already fragmented or
OOM'd reports peak memory for the previous config, not this one.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import transformers
import trl
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase, TrainerCallback
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from grpo.estimator_defaults import (
    GRPO_LOSS_TYPE,
    GRPO_SCALE_REWARDS,
    assert_liger_faithful_estimator,
    executed_estimator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence

    from transformers import TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3
MIN_GRPO_GROUP_SIZE = 2
COMPLETION_LENGTH_TOLERANCE = 0.5

# Qwen3.5 stacks three Gated DeltaNet linear-attention layers per full-attention layer, and
# the two carry different projection names. LoRA targeting only the familiar four would adapt
# 8 layers of 32 and silently leave three quarters of the stack frozen.
FULL_ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LINEAR_ATTENTION_PROJECTIONS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")

# lm_head is deliberately absent: TRL refuses to start with a head adapter under
# use_liger_kernel, because the fused loss reads lm_head.weight directly.
ADAPTABLE_PROJECTIONS = FULL_ATTENTION_PROJECTIONS + LINEAR_ATTENTION_PROJECTIONS + MLP_PROJECTIONS

# Single-token fillers, so a prompt's token count is exactly the repetition count plus the
# chat template's overhead. Several of them rather than one so the P prompts in a step differ
# in content while matching in length; verify_single_token() drops any that tokenize wider.
FILLER_WORDS = (
    " apple",
    " river",
    " table",
    " window",
    " forest",
    " candle",
    " garden",
    " basket",
)

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class ThroughputConfig:
    """One measurement point. Every field lands in the JSON artifact."""

    model_id: str
    prompts_per_step: int = 8
    group_size: int = 8
    prompt_tokens: int = 2048
    completion_tokens: int = 2048
    measured_steps: int = 5
    warmup_steps: int = 2
    # None means one prompt group per forward/backward. Only affects the training pass;
    # generation always runs the full prompts_per_step * group_size batch at once.
    micro_batch_size: int | None = None
    lora_rank: int = 16
    lora_alpha: int = 32
    # TRL 1.10 defaults beta to 0.0, which skips the KL term and its reference forward pass.
    # Under LoRA a non-zero beta costs a second forward with the adapter disabled, not a
    # second copy of the weights.
    beta: float = 0.0
    use_liger_kernel: bool = True
    # Explicit escape hatch for GRPO_LOSS_TYPE combinations Liger executes unfaithfully; see
    # grpo/estimator_defaults.py. Off, validate() refuses such a combination.
    acknowledge_liger_estimator_mismatch: bool = False
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-6
    seed: int = 0
    output_dir: str = "artifacts/throughput-run"
    json_out: str | None = None

    @property
    def episodes_per_step(self) -> int:
        """Keep episode count explicit for timing and reporting."""
        return self.prompts_per_step * self.group_size

    @property
    def resolved_micro_batch_size(self) -> int:
        """Select the training micro-batch while retaining full generation batches."""
        return self.micro_batch_size or self.group_size

    @property
    def gradient_accumulation_steps(self) -> int:
        """Expose accumulation implied by the measured episode batch."""
        return self.episodes_per_step // self.resolved_micro_batch_size

    @property
    def total_steps(self) -> int:
        """Include warmup steps so timing can exclude them."""
        return self.warmup_steps + self.measured_steps

    @property
    def tokens_per_episode(self) -> int:
        """Expose the token budget used for throughput comparisons."""
        return self.prompt_tokens + self.completion_tokens

    def validate(self) -> None:
        """Reject configurations that would misstate or fail the measurement."""
        if self.episodes_per_step % self.resolved_micro_batch_size:
            raise ValueError(
                "micro_batch_size must divide prompts_per_step * group_size, "
                f"{self.resolved_micro_batch_size=} {self.episodes_per_step=}"
            )
        if self.group_size < MIN_GRPO_GROUP_SIZE:
            raise ValueError(f"GRPO needs at least 2 generations per prompt, {self.group_size=}")
        if self.measured_steps < 1:
            raise ValueError(f"nothing to measure, {self.measured_steps=}")
        assert_liger_faithful_estimator(
            GRPO_LOSS_TYPE,
            use_liger_kernel=self.use_liger_kernel,
            acknowledged=self.acknowledge_liger_estimator_mismatch,
        )


def describe_device(device_index: int = 0) -> dict[str, object]:
    """Record what the run actually landed on. A step time without this is meaningless."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible; this measurement is GPU-only")
    props = torch.cuda.get_device_properties(device_index)
    free_b, total_b = torch.cuda.mem_get_info(device_index)
    return {
        "device_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_gib": total_b / BYTES_PER_GIB,
        "free_vram_gib_at_start": free_b / BYTES_PER_GIB,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "cuda": torch.version.cuda,
    }


def memory_model(
    text_config: object, episodes: int, prompt_tokens: int, completion_tokens: int
) -> dict[str, object]:
    """Footprint predicted from config fields alone, to be compared against measurement.

    The KV arithmetic in `docs/scratch/compute-budget-model.md` is derived from these same
    config values and has never been checked against a running process. Reporting the
    prediction beside the measurement is what turns this run into a test of it. Takes the
    config object rather than a model id so the arithmetic is testable without a download.
    """
    layer_types = list(text_config.layer_types)  # pyright: ignore[reportAttributeAccessIssue]
    n_full = sum(1 for t in layer_types if t == "full_attention")
    n_linear = sum(1 for t in layer_types if t == "linear_attention")

    kv_bytes_per_token = (
        2
        * n_full
        * text_config.num_key_value_heads  # pyright: ignore[reportAttributeAccessIssue]
        * text_config.head_dim  # pyright: ignore[reportAttributeAccessIssue]
        * 2
    )
    # One (num_value_heads, key_head_dim, value_head_dim) matrix per linear-attention layer per
    # sequence, fixed in context length. transformers holds it in float32 and never casts it
    # back (modeling_qwen3_5.py:326 casts the output but not the state), so the model dtype is
    # the wrong number to use here even under bf16 weights.
    recurrent_bytes_per_sequence = (
        n_linear
        * text_config.linear_num_value_heads  # pyright: ignore[reportAttributeAccessIssue]
        * text_config.linear_key_head_dim  # pyright: ignore[reportAttributeAccessIssue]
        * text_config.linear_value_head_dim  # pyright: ignore[reportAttributeAccessIssue]
        * 4
    )
    # The chunked prefill path upcasts query, key and value to float32 and makes contiguous
    # copies of all three across the whole batch, with query and key first repeat-interleaved
    # up to the value head count. Measured to be the binding constraint on a 24 GiB card,
    # rather than the KV cache the budget note reasons about.
    prefill_upcast_bytes = (
        3
        * text_config.linear_num_value_heads  # pyright: ignore[reportAttributeAccessIssue]
        * text_config.linear_value_head_dim  # pyright: ignore[reportAttributeAccessIssue]
        * 4
    )
    return {
        "n_full_attention_layers": n_full,
        "n_linear_attention_layers": n_linear,
        "vocab_size": text_config.vocab_size,  # pyright: ignore[reportAttributeAccessIssue]
        "kv_bytes_per_token": kv_bytes_per_token,
        "recurrent_state_mib_per_sequence": recurrent_bytes_per_sequence / (1024**2),
        "prefill_upcast_bytes_per_token": prefill_upcast_bytes,
        "predicted_kv_gib": (
            kv_bytes_per_token * (prompt_tokens + completion_tokens) * episodes / BYTES_PER_GIB
        ),
        "predicted_recurrent_gib": recurrent_bytes_per_sequence * episodes / BYTES_PER_GIB,
        "predicted_prefill_upcast_gib": (
            prefill_upcast_bytes * prompt_tokens * episodes / BYTES_PER_GIB
        ),
        "logits_gib_per_sequence_unfused": (
            completion_tokens
            * text_config.vocab_size  # pyright: ignore[reportAttributeAccessIssue]
            * 2
            / BYTES_PER_GIB
        ),
    }


def predict_memory(model_id: str, config: ThroughputConfig) -> dict[str, object]:
    """Predict memory from checkpoint architecture fields for comparison."""
    text_config = AutoConfig.from_pretrained(model_id).get_text_config()
    return memory_model(
        text_config, config.episodes_per_step, config.prompt_tokens, config.completion_tokens
    )


def select_lora_targets(
    linear_module_names: Iterable[str], layer_types: Sequence[str]
) -> dict[str, object]:
    """Choose LoRA targets from the model's own module names, and refuse a frozen stack.

    Pure so the guard can be tested. The guard is the point: Qwen3.5 puts three quarters of
    its layers in Gated DeltaNet blocks whose projections are named nothing like `q_proj`, so
    the familiar target list adapts a quarter of the model and reports nothing wrong.
    """
    counts: dict[str, int] = {}
    for name in linear_module_names:
        suffix = name.rsplit(".", 1)[-1]
        if suffix in ADAPTABLE_PROJECTIONS:
            counts[suffix] = counts.get(suffix, 0) + 1

    expected_linear = sum(1 for t in layer_types if t == "linear_attention")
    expected_full = sum(1 for t in layer_types if t == "full_attention")
    if expected_linear and not any(k in LINEAR_ATTENTION_PROJECTIONS for k in counts):
        raise RuntimeError(
            f"model declares {expected_linear} linear_attention layers but none of "
            f"{LINEAR_ATTENTION_PROJECTIONS} were found; LoRA would freeze them silently"
        )
    return {
        "target_modules": sorted(counts),
        "module_counts": dict(sorted(counts.items())),
        "expected_full_attention_layers": expected_full,
        "expected_linear_attention_layers": expected_linear,
    }


def discover_lora_targets(model_id: str) -> dict[str, object]:
    """Build the checkpoint's architecture on the meta device to read its module names."""
    config = AutoConfig.from_pretrained(model_id)
    architecture = getattr(  # pyright: ignore[reportOptionalSubscript]
        transformers,
        config.architectures[0],  # pyright: ignore[reportOptionalSubscript]
    )
    with torch.device("meta"):
        model = architecture(config)
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    return select_lora_targets(names, list(config.get_text_config().layer_types))


def verify_single_token(tokenizer: PreTrainedTokenizerBase, words: tuple[str, ...]) -> list[str]:
    """Keep prompt lengths exact across tokenizer variants."""
    single = [w for w in words if len(tokenizer(w, add_special_tokens=False)["input_ids"]) == 1]
    if not single:
        raise RuntimeError(f"no single-token filler among {words} for this tokenizer")
    return single


def build_prompt(
    tokenizer: PreTrainedTokenizerBase, target_tokens: int, filler: str
) -> tuple[str, int]:
    """Chat-templated prompt text that tokenizes to exactly `target_tokens` tokens.

    Exact rather than approximate because the whole point is to label the measurement with a
    token profile; a prompt that came out 200 tokens short would quietly report a cheaper
    configuration than the one named.
    """

    def render(repetitions: int) -> tuple[str, int]:
        body = filler * repetitions
        text = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": body},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text, len(tokenizer(text, add_special_tokens=False)["input_ids"])  # pyright: ignore[reportArgumentType, reportReturnType]

    _, overhead = render(0)
    if overhead > target_tokens:
        raise ValueError(f"chat template alone is {overhead} tokens, over {target_tokens=}")

    repetitions = target_tokens - overhead
    text, n_tokens = render(repetitions)
    # One correction pass: the filler is single-token, but joining it to the template's last
    # token can merge a byte pair at the boundary.
    if n_tokens != target_tokens:
        repetitions += target_tokens - n_tokens
        text, n_tokens = render(repetitions)
    if n_tokens != target_tokens:
        raise RuntimeError(f"could not hit an exact prompt length, {n_tokens=} {target_tokens=}")
    return text, n_tokens


def build_prompt_dataset(
    tokenizer: PreTrainedTokenizerBase, config: ThroughputConfig
) -> tuple[Dataset, int]:
    """Build enough exact-length prompts for the full run."""
    n_rows = config.prompts_per_step * config.total_steps
    fillers = verify_single_token(tokenizer, FILLER_WORDS)
    rows: list[dict[str, str]] = []
    measured = 0
    for i in range(n_rows):
        text, measured = build_prompt(tokenizer, config.prompt_tokens, fillers[i % len(fillers)])
        rows.append({"prompt": text})
    logger.info("built %d prompts of exactly %d tokens", n_rows, measured)
    return Dataset.from_list(rows), measured


def reward_token_variety(completions: list[str], **kwargs: object) -> list[float]:
    """Cheap reward that varies within a group, so advantages are not identically zero.

    Throughput does not depend on the reward's content — the backward pass runs either way —
    but a degenerate reward would make `frac_reward_zero_std` uninformative, and that metric
    is one of the things this run is here to report.
    """
    del kwargs
    return [len(set(text)) / max(1, len(text)) for text in completions]


class StepTimingCallback(TrainerCallback):
    """Per-optimizer-step wall clock and peak memory.

    `on_step_end` fires at the optimizer-step boundary (transformers/trainer.py:1799) while
    `on_substep_end` handles accumulation micro-steps, so this counts what we want. The
    synchronize is load-bearing: without it the clock reads when work was queued, not done.
    """

    def __init__(self, device_index: int = 0) -> None:
        """Start timing against the selected CUDA device."""
        self.device_index = device_index
        self.step_records: list[dict[str, float]] = []
        self._last_t: float | None = None

    def _sync(self) -> float:
        """Drain the device's queue, then read the clock; off CUDA there is nothing to drain."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_index)
        return time.perf_counter()

    def _memory_record(self) -> dict[str, float]:
        """Read the allocator's peaks and the card's live usage, the numbers a step time is meaningless without."""
        free_b, total_b = torch.cuda.mem_get_info(self.device_index)
        return {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device_index)
            / BYTES_PER_GIB,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device_index) / BYTES_PER_GIB,
            "device_used_gib": (total_b - free_b) / BYTES_PER_GIB,
        }

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Reset allocator stats before timing measured steps."""
        del args, state, control, kwargs
        torch.cuda.reset_peak_memory_stats(self.device_index)
        logger.info("training loop entered, timing from here")
        self._last_t = self._sync()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Record one synchronized optimizer-step measurement."""
        del args, control, kwargs
        now = self._sync()
        last_t = cast("float", self._last_t)
        record = {
            "step": float(state.global_step),
            "seconds": now - last_t,
            **self._memory_record(),
        }
        self.step_records.append(record)
        # A 4B step at a long profile runs for minutes; without this the run is silent and
        # indistinguishable from a wedged one.
        logger.info(
            "step %d: %.1fs peak_alloc=%.2fGiB peak_reserved=%.2fGiB device_used=%.2fGiB",
            int(record["step"]),
            record["seconds"],
            record["peak_allocated_gib"],
            record["peak_reserved_gib"],
            record["device_used_gib"],
        )
        self._last_t = now


def instrument_generation(trainer: GRPOTrainer) -> list[float]:
    """Time the generation half of each step separately.

    TRL profiles this internally but `ProfilingContext._log_metrics` writes only to wandb,
    mlflow or trackio (trl/extras/profiling.py:104), so with `report_to=[]` the split is
    unavailable. It matters here: generation being most of the step is the argument for vLLM,
    and the training pass being most of it is the argument against.
    """
    original = cast("Any", trainer._generate_and_score_completions)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    timings: list[float] = []

    @functools.wraps(original)
    def timed(*args: object, **kwargs: object) -> dict[str, Any]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = original(*args, **kwargs)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        return result

    trainer._generate_and_score_completions = timed  # noqa: SLF001  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    return timings


STEP_PHASES: tuple[str, ...] = (
    "rollout_score",
    "policy_to_cpu",
    "sync_weights",
    "generate",
    "policy_to_cuda",
    "decode",
    "rewards",
    "logps_forward",
    "compute_loss",
    "backward",
    "grad_clip",
    "optimizer_step",
)
TIMING_METRIC_PREFIX = "timing/"
STEP_INTERVAL_METRIC = f"{TIMING_METRIC_PREFIX}step_interval_s"
UNATTRIBUTED_METRIC = f"{TIMING_METRIC_PREFIX}unattributed_s"
GENERATE_TOKENS_PER_SECOND_METRIC = f"{TIMING_METRIC_PREFIX}generate_completion_tokens_per_s"
TIMING_METRIC_KEYS: tuple[str, ...] = (
    STEP_INTERVAL_METRIC,
    *(f"{TIMING_METRIC_PREFIX}{phase}_s" for phase in STEP_PHASES),
    UNATTRIBUTED_METRIC,
    GENERATE_TOKENS_PER_SECOND_METRIC,
)


def timing_metric(phase: str) -> str:
    """Name of the log-record key carrying one phase's seconds per optimizer step."""
    return f"{TIMING_METRIC_PREFIX}{phase}_s"


class StepPhaseTimer(StepTimingCallback):
    """CUDA-synchronised seconds per phase of every optimizer step, written into the step's log record.

    The step interval and peak memory come from `StepTimingCallback`; this adds where the time went.
    `attach` wraps the seams TRL 1.10 runs a step through -- `_generate_and_score_completions` around
    the vLLM weight sync, `vllm_generation.generate`, `batch_decode` and `_calculate_rewards`, then
    `compute_loss` and `accelerator.backward` per micro-step -- and the optimizer-step callbacks
    bracket gradient clipping and `optimizer.step`. Every wrapper synchronises the device on entry and
    exit, so a phase's seconds are the work it launched rather than the moment its kernels were
    queued, the throughput probe's own discipline; no arithmetic changes. At `on_step_end` the totals
    land in TRL's `_metrics["train"]` as `timing/<phase>_s`, which `GRPOTrainer.log` averages over
    the logging interval into `log_history` beside TRL's `step_time`, plus `timing/step_interval_s`
    (this `on_step_end` to the previous one, so it carries the previous step's log, checkpoint and S3
    tail), `timing/unattributed_s` (the interval minus the phases entered at the top level, so nested
    seams are not subtracted twice) and `timing/generate_completion_tokens_per_s` (completion tokens
    over `generate` seconds: the engine's real decode rate, not a padded estimate).

    What TRL's `step_time` measures, for anyone reading the two side by side: `perf_counter` around
    each `training_step` (grpo_trainer.py:1556-1564 in 1.10), so `_prepare_inputs` -- the whole
    generation-and-score pass, on the first micro-step -- plus the loss and the `backward` *launch*
    of every micro-step. `backward` returns before the GPU has drained, so about 7% of each backward
    is timed outside it, at transformers' per-micro-batch nan check; gradient clipping, the optimizer
    step, the log, the checkpoint write and the S3 sync are outside it too. On the 2B production
    shape (efficiency audit, 2026-09-02) `step_time` was 696 s of a 761 s step; the missing 65 s were
    the drained backward tail (52 s) and the log/save/sync tail (12 s). Read `step_time` as
    "generation plus about 93% of the training compute", never as the step; `step_interval_s` is.

    The generation phase is only reachable under vLLM: on the transformers.generate path TRL calls
    `model.generate` inside a context manager with no seam to wrap, so `timing/generate_s` reads zero
    there while `rollout_score` still carries it.
    """

    def __init__(self, device_index: int = 0) -> None:
        """Start with no trainer attached and every phase at zero."""
        super().__init__(device_index)
        self._trainer: GRPOTrainer | None = None
        self._phase_seconds: dict[str, float] = dict.fromkeys(STEP_PHASES, 0.0)
        self._top_level_seconds = 0.0
        self._completion_tokens = 0.0
        self._active: list[str] = []
        self._phase_end: dict[str, float] = {}
        self._pre_optimizer_t: float | None = None

    @contextlib.contextmanager
    def phase(self, name: str) -> Generator[None]:
        """Time one block as `name`, attributing it to the step total only when not nested."""
        nested = bool(self._active)
        self._active.append(name)
        start = self._sync()
        try:
            yield
        finally:
            end = self._sync()
            self._active.pop()
            self._phase_seconds[name] += end - start
            self._phase_end[name] = end
            if not nested:
                self._top_level_seconds += end - start

    def _wrap(self, owner: object, attribute: str, name: str) -> None:
        original = cast("Callable[..., Any]", getattr(owner, attribute))

        @functools.wraps(original)
        def timed(*args: object, **kwargs: object) -> object:
            with self.phase(name):
                return original(*args, **kwargs)

        setattr(owner, attribute, timed)

    def _wrap_rollout(self, trainer: GRPOTrainer) -> None:
        """Time `_generate_and_score_completions` as `rollout_score` and bank the completion tokens it made.

        The tokens are read here, from the `completions/mean_length` entry TRL's `_generate` appends
        for exactly this rollout (one per call, grpo_trainer.py:2302 in 1.10, the mean over the
        cross-process gather of `generation_batch_size` completions), rather than summed over
        `_metrics` at `on_step_end`: `GRPOTrainer.log` clears `_metrics` only when it logs, so with
        `logging_steps` above one the k-th step of a window would count k steps of tokens against one
        step's generate seconds. Eval rollouts write to `_metrics["eval"]` and are not banked.
        """
        trainer_internals = cast("Any", trainer)
        original = cast("Callable[..., Any]", trainer_internals._generate_and_score_completions)  # noqa: SLF001

        @functools.wraps(original)
        def timed(*args: object, **kwargs: object) -> object:
            with self.phase("rollout_score"):
                result = original(*args, **kwargs)
            if trainer_internals.model.training:
                mean_lengths = trainer_internals._metrics["train"].get("completions/mean_length")  # noqa: SLF001
                if not mean_lengths:
                    raise RuntimeError(
                        "a training rollout left no completions/mean_length entry in _metrics; "
                        "TRL's generation bookkeeping changed and the token rate would be wrong"
                    )
                self._completion_tokens += (
                    float(mean_lengths[-1]) * trainer_internals.args.generation_batch_size
                )
            return result

        trainer_internals._generate_and_score_completions = timed  # noqa: SLF001

    def attach(self, trainer: GRPOTrainer) -> None:
        """Wrap the built trainer's step seams. Call once, after construction and before `train()`."""
        if self._trainer is not None:
            raise RuntimeError("StepPhaseTimer is already attached to a trainer")
        self._trainer = trainer
        self._wrap_rollout(trainer)
        if trainer.use_vllm:
            self._wrap(trainer.vllm_generation, "sync_weights", "sync_weights")
            self._wrap(trainer.vllm_generation, "generate", "generate")
        self._wrap(trainer.processing_class, "batch_decode", "decode")
        self._wrap(trainer, "_calculate_rewards", "rewards")
        self._wrap(trainer, "_get_per_token_logps_and_entropies", "logps_forward")
        self._wrap(trainer, "compute_loss", "compute_loss")
        self._wrap(trainer.accelerator, "backward", "backward")

    def on_pre_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Close the clipping phase: everything between the last backward's drain and `optimizer.step`."""
        del args, state, control, kwargs
        self._pre_optimizer_t = self._sync()
        last_backward_end = self._phase_end.get("backward")
        if last_backward_end is not None:
            clip_seconds = self._pre_optimizer_t - last_backward_end
            self._phase_seconds["grad_clip"] += clip_seconds
            self._top_level_seconds += clip_seconds

    def on_optimizer_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Close the optimizer-step phase."""
        del args, state, control, kwargs
        step_seconds = self._sync() - cast("float", self._pre_optimizer_t)
        self._phase_seconds["optimizer_step"] += step_seconds
        self._top_level_seconds += step_seconds

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        """Record the step interval, then flush every phase into the record TRL is about to log."""
        super().on_step_end(args, state, control, **kwargs)
        if self._trainer is None:
            raise RuntimeError(
                "StepPhaseTimer.on_step_end fired before attach(); nothing to write to"
            )
        interval = self.step_records[-1]["seconds"]
        metrics = cast("Any", self._trainer)._metrics["train"]  # noqa: SLF001
        metrics[STEP_INTERVAL_METRIC].append(interval)
        for name in STEP_PHASES:
            metrics[timing_metric(name)].append(self._phase_seconds[name])
        metrics[UNATTRIBUTED_METRIC].append(interval - self._top_level_seconds)
        # Only the completions generated this step (banked by the rollout wrapper), over this
        # step's generate seconds; a step that reused the generation buffer has neither and reads NaN.
        generate_seconds = self._phase_seconds["generate"]
        metrics[GENERATE_TOKENS_PER_SECOND_METRIC].append(
            self._completion_tokens / generate_seconds if generate_seconds > 0 else math.nan
        )
        logger.info(
            "step %d phases (s): %s unattributed=%.1f generate_tokens/s=%.0f",
            int(state.global_step),
            " ".join(f"{name}={self._phase_seconds[name]:.1f}" for name in STEP_PHASES),
            interval - self._top_level_seconds,
            metrics[GENERATE_TOKENS_PER_SECOND_METRIC][-1],
        )
        self._phase_seconds = dict.fromkeys(STEP_PHASES, 0.0)
        self._top_level_seconds = 0.0
        self._completion_tokens = 0.0
        self._phase_end.clear()
        self._pre_optimizer_t = None


def build_trainer(
    config: ThroughputConfig, dataset: Dataset, tokenizer: PreTrainedTokenizerBase
) -> tuple[GRPOTrainer, StepTimingCallback, list[float]]:
    """Assemble the measured GRPO trainer and timing hooks."""
    lora_targets = discover_lora_targets(config.model_id)
    logger.info("LoRA targets: %s", lora_targets["module_counts"])

    args = GRPOConfig(
        output_dir=config.output_dir,
        seed=config.seed,
        bf16=True,
        tf32=True,
        # Leave steps_per_generation and generation_batch_size at None. TRL then derives
        # steps_per_generation = gradient_accumulation_steps and generation_batch_size =
        # per_device_train_batch_size * gradient_accumulation_steps, which is exactly one
        # generation of prompts_per_step * group_size episodes per optimizer step. Setting
        # generation_batch_size directly instead leaves gradient_accumulation_steps at 1 and
        # spreads one generation across several optimizer steps.
        per_device_train_batch_size=config.resolved_micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.group_size,
        max_completion_length=config.completion_tokens,
        # Suppresses EOS in the logits at every position until the length is reached, so
        # every completion is exactly max_completion_length tokens and the measurement
        # matches the profile it is labelled with. TRL's own trim then finds no EOS to cut.
        generation_kwargs={"min_new_tokens": config.completion_tokens},
        temperature=1.0,
        top_p=1.0,
        beta=config.beta,
        loss_type=GRPO_LOSS_TYPE,
        scale_rewards=GRPO_SCALE_REWARDS,
        # Would zero the loss mask of every completion here, since forcing full length means
        # none of them terminate on EOS.
        mask_truncated_completions=False,
        use_liger_kernel=config.use_liger_kernel,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        lr_scheduler_type="constant",
        max_steps=config.total_steps,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="no",
        eval_strategy="no",
        shuffle_dataset=False,
        report_to=[],
        disable_tqdm=True,
        model_init_kwargs={"dtype": "bfloat16"},  # `dtype`, not `torch_dtype`, which TRL ignores
    )

    trainer = GRPOTrainer(
        model=config.model_id,
        reward_funcs=reward_token_variety,  # pyright: ignore[reportArgumentType]
        args=args,
        train_dataset=dataset,
        # A bare tokenizer rather than TRL's default AutoProcessor. Qwen3.5 maps to a
        # ProcessorMixin, which switches TRL onto its vision-language path; these episodes
        # are text-only. The vision tower still loads with the checkpoint's own class.
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cast("list[str]", lora_targets["target_modules"]),
        ),
    )
    timing_cb = StepTimingCallback()
    trainer.add_callback(timing_cb)
    generation_timings = instrument_generation(trainer)
    return trainer, timing_cb, generation_timings


def summarize_lora(trainer: GRPOTrainer) -> dict[str, object]:
    """Count what LoRA actually adapted on the real model, not what we asked for."""
    model = trainer.model
    adapted = [
        n
        for n, _ in model.named_modules()  # pyright: ignore[reportOptionalMemberAccess]
        if n.endswith("lora_A.default")
    ]
    linear_attention_adapted = [n for n in adapted if "linear_attn" in n]
    if not linear_attention_adapted:
        raise RuntimeError(
            "no LoRA adapter landed on a linear_attention layer; three quarters of the "
            "Qwen3.5 stack would be frozen and the measurement would not represent a real run"
        )
    trainable = sum(
        p.numel()
        for p in model.parameters()  # pyright: ignore[reportOptionalMemberAccess]
        if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())  # pyright: ignore[reportOptionalMemberAccess]
    return {
        "adapted_modules": len(adapted),
        "adapted_linear_attention_modules": len(linear_attention_adapted),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": trainable / total,
    }


def verify_logged_metrics(trainer: GRPOTrainer, config: ThroughputConfig) -> dict[str, object]:
    """Read the metrics back out of trainer state, rather than trusting that they logged.

    A callback in this repo once wrote metrics into the dict `Trainer.log` had already copied,
    so logging ran clean and recorded nothing for the whole history of the repo. The specific
    check that matters here is `completions/mean_length`: if EOS suppression did not take,
    completions are shorter than the profile and the step time belongs to a different
    measurement than the one it would be labelled with.
    """
    history = [record for record in trainer.state.log_history if "reward" in record]
    if not history:
        raise RuntimeError("no training record carried a reward; nothing was measured")
    last = history[-1]
    required = ("reward", "reward_std", "frac_reward_zero_std", "completions/mean_length")
    missing = [key for key in required if key not in last]
    if missing:
        raise RuntimeError(f"expected metrics never reached trainer_state, {missing=}")

    mean_length = float(last["completions/mean_length"])
    if abs(mean_length - config.completion_tokens) > COMPLETION_LENGTH_TOLERANCE:
        raise RuntimeError(
            "completions did not run to the requested length, so this step time does not "
            f"describe the stated profile: {mean_length=} {config.completion_tokens=}"
        )
    return {
        "reward": float(last["reward"]),
        "reward_std": float(last["reward_std"]),
        "frac_reward_zero_std": float(last["frac_reward_zero_std"]),
        "completions_mean_length": mean_length,
        "completions_min_length": float(last.get("completions/min_length", float("nan"))),
        "completions_max_length": float(last.get("completions/max_length", float("nan"))),
        # Generation plus ~93% of the training compute, not the step: see StepPhaseTimer's docstring.
        "trl_step_time_mean": (
            statistics.fmean(
                [float(r["step_time"]) for r in trainer.state.log_history if "step_time" in r]
            )
            if any("step_time" in r for r in trainer.state.log_history)
            else float("nan")
        ),
        "logged_keys": sorted(last),
    }


def measure(config: ThroughputConfig) -> dict[str, object]:
    """Run one configured measurement and persist its artifact."""
    config.validate()
    device = describe_device()
    logger.info(
        "measuring %s on %s (%.1f GiB), P=%d G=%d prompt=%d completion=%d",
        config.model_id,
        device["device_name"],
        device["total_vram_gib"],
        config.prompts_per_step,
        config.group_size,
        config.prompt_tokens,
        config.completion_tokens,
    )
    torch.manual_seed(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dataset, prompt_tokens_measured = build_prompt_dataset(tokenizer, config)
    trainer, timing_cb, generation_timings = build_trainer(config, dataset, tokenizer)
    lora_summary = summarize_lora(trainer)
    logger.info("LoRA: %s", lora_summary)

    wall_start = time.perf_counter()
    trainer.train()
    wall_total = time.perf_counter() - wall_start

    metrics = verify_logged_metrics(trainer, config)
    records = timing_cb.step_records
    if len(records) < config.total_steps:
        raise RuntimeError(f"timed {len(records)} steps, expected {config.total_steps}")

    measured = records[config.warmup_steps :]
    step_seconds = [r["seconds"] for r in measured]
    median_step = statistics.median(step_seconds)
    generation_measured = generation_timings[config.warmup_steps :]

    episodes = config.episodes_per_step
    result: dict[str, object] = {
        "config": asdict(config),
        # What the loss aggregation DID under this liger/micro-batch shape; see
        # grpo/estimator_defaults.py. A throughput figure is only comparable to a training run
        # that executed the same estimator.
        "executed_estimator": executed_estimator(
            GRPO_LOSS_TYPE,
            use_liger_kernel=config.use_liger_kernel,
            per_device_train_batch_size=config.resolved_micro_batch_size,
        ),
        "device": device,
        "memory_prediction": predict_memory(config.model_id, config),
        "lora": lora_summary,
        "prompt_tokens_measured": prompt_tokens_measured,
        "metrics": metrics,
        "per_step_seconds_all": [r["seconds"] for r in records],
        "per_step_seconds_measured": step_seconds,
        "median_step_seconds": median_step,
        "mean_step_seconds": statistics.fmean(step_seconds),
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "generation_seconds_measured": generation_measured,
        "median_generation_seconds": (
            statistics.median(generation_measured) if generation_measured else float("nan")
        ),
        "generation_fraction_of_step": (
            statistics.median(generation_measured) / median_step if generation_measured else None
        ),
        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in records),
        "peak_reserved_gib": max(r["peak_reserved_gib"] for r in records),
        "peak_device_used_gib": max(r["device_used_gib"] for r in records),
        "episodes_per_step": episodes,
        "episodes_per_hour": episodes * 3600.0 / median_step,
        "steps_per_hour": 3600.0 / median_step,
        # Total episode tokens over wall clock, matching how compute-budget-model.md defines
        # its aggregate figure, so the two are directly comparable.
        "episode_tokens_per_second": episodes * config.tokens_per_episode / median_step,
        "generated_tokens_per_second": episodes * config.completion_tokens / median_step,
        "hours_per_200_steps": 200 * median_step / 3600.0,
        "train_wall_seconds": wall_total,
    }

    if config.json_out:
        path = Path(config.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", path)
    return result


def format_table(result: dict[str, object]) -> str:
    """Present one measurement result for quick inspection."""
    result_data = cast("dict[str, Any]", result)
    config = result_data["config"]
    device = result_data["device"]
    metrics = result_data["metrics"]
    micro_batch = config["micro_batch_size"] or config["group_size"]
    accum = result_data["episodes_per_step"] // micro_batch
    rows = [
        ("model", config["model_id"]),
        ("device", f"{device['device_name']} ({device['total_vram_gib']:.1f} GiB)"),
        ("prompts x group", f"{config['prompts_per_step']} x {config['group_size']}"),
        ("episodes / step", str(result_data["episodes_per_step"])),
        ("micro-batch x accum", f"{micro_batch} x {accum}"),
        (
            "tokens (prompt+completion)",
            f"{config['prompt_tokens']} + {config['completion_tokens']}",
        ),
        (
            "liger / grad-checkpoint",
            f"{config['use_liger_kernel']} / {config['gradient_checkpointing']}",
        ),
        ("", ""),
        ("median s / optimizer step", f"{result_data['median_step_seconds']:.2f}"),
        (
            "per-step seconds (measured)",
            ", ".join(f"{s:.1f}" for s in result_data["per_step_seconds_measured"]),
        ),
        (
            "generation share of step",
            (
                f"{result_data['generation_fraction_of_step']:.0%}"
                if result_data["generation_fraction_of_step"] is not None
                else "n/a"
            ),
        ),
        ("episodes / hour", f"{result_data['episodes_per_hour']:.1f}"),
        ("steps / hour", f"{result_data['steps_per_hour']:.2f}"),
        ("hours / 200 steps", f"{result_data['hours_per_200_steps']:.1f}"),
        ("episode tokens / s", f"{result_data['episode_tokens_per_second']:.0f}"),
        ("generated tokens / s", f"{result_data['generated_tokens_per_second']:.0f}"),
        ("", ""),
        ("peak allocated", f"{result_data['peak_allocated_gib']:.2f} GiB"),
        ("peak reserved", f"{result_data['peak_reserved_gib']:.2f} GiB"),
        ("peak device used", f"{result_data['peak_device_used_gib']:.2f} GiB"),
        ("predicted KV", f"{result_data['memory_prediction']['predicted_kv_gib']:.2f} GiB"),
        (
            "predicted recurrent state",
            f"{result_data['memory_prediction']['predicted_recurrent_gib']:.2f} GiB",
        ),
        ("", ""),
        ("completion length (mean)", f"{metrics['completions_mean_length']:.0f}"),
        ("reward / std", f"{metrics['reward']:.3f} / {metrics['reward_std']:.3f}"),
        ("frac_reward_zero_std", f"{metrics['frac_reward_zero_std']:.2f}"),
        (
            "trainable params",
            (
                f"{result_data['lora']['trainable_params']:,} "
                f"({result_data['lora']['trainable_fraction']:.2%})"
            ),
        ),
    ]
    width = max(len(k) for k, _ in rows)
    lines = [f"{k.ljust(width)}  {v}" if k else "" for k, v in rows]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> ThroughputConfig:
    """Parse one throughput measurement configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("-P", "--prompts-per-step", type=int, default=8)
    parser.add_argument("-G", "--group-size", type=int, default=8)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--completion-tokens", type=int, default=2048)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--no-liger", dest="use_liger_kernel", action="store_false")
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    parser.add_argument("--output-dir", default="artifacts/throughput-run")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    return ThroughputConfig(**vars(args))


def main(argv: list[str] | None = None) -> None:
    """Run and report one throughput measurement."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    # Imported here rather than at the top for two reasons: `games` builds on `grpo` and not the
    # other way round, and `games.deltanet_kernels` reaches back through `games.preflight` into this
    # module, so a top-level import is a cycle. It still runs before anything imports the Qwen3.5
    # modeling module, which is the only ordering the bridge cares about -- transformers binds each
    # Gated DeltaNet kernel at that module's import time.
    from games.deltanet_kernels import bridge_decode_kernel  # noqa: PLC0415

    logger.info("deltanet decode bridge: %s", bridge_decode_kernel())
    config = parse_args(argv)
    result = measure(config)
    logger.info("\n%s", format_table(result))


if __name__ == "__main__":
    main()
