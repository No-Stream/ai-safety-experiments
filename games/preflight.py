"""Checks that confirm a run is the run we think it is, before it spends a GPU on being wrong.

Six facts here decide whether an arm measures what its name says, and every one of them was a
silent failure somewhere first:

*   Whether the chat template opens `<think>` inside the prompt. Qwen3.5/3.8 do, Qwen3-0.6B does
    not, and the defaults move non-monotonically across the ladder. A parser on the wrong side of
    that reads every rollout as truncated thinking, which is a full batch of parse penalties and
    still looks like a number.
*   Whether the template injects prompt text nobody wrote. Qwen3.8-27B prepends a "Reasoning
    effort is set to xhigh" system message at its default, verified on this box: the default
    render of a one-word prompt is 299 characters, and 62 with `reasoning_effort="medium"`.
*   Whether LoRA reached the layers it had to. Three quarters of the Qwen3.5 stack is Gated
    DeltaNet, whose projections are named nothing like `q_proj`.
*   Whether the batch is being sharded across processes, which would break the reward function's
    group-contiguity assumption while reporting perfectly ordinary rewards.
*   Whether the per-completion rollout trace is actually complete.
*   What sampler generation will actually use, including the parts of it nobody here passes. A
    comparison of two runs' rollouts once had to recover this from a tarball of the code one box
    happened to be running, because no run had ever written it down.

These live apart from `games/train.py` because the eval battery needs the same template facts: the
design invariant is that one raw prompt string flows through training and evaluation unchanged, so
whatever training derived about `<think>` and `reasoning_effort` has to be what evaluation uses.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq
from transformers import AutoTokenizer, GenerationConfig

from grpo.throughput import summarize_lora

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase
    from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

logger = logging.getLogger(__name__)

# Below this many declared stop tokens there is nothing ambiguous for TRL's scalar EOS to get wrong.
MIN_AMBIGUOUS_STOP_TOKENS = 2

CUDA_ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
DEFAULT_CUDA_ALLOC_CONF = "expandable_segments:True"


def default_cuda_allocator_config() -> str:
    """Default the CUDA allocator to expandable segments; an explicit environment value wins.

    Must run before the first CUDA allocation -- torch parses the variable at allocator
    initialisation, so a value set after any tensor reaches the card is silently ignored. Measured
    REQUIRED on 32k HF generation workloads (2026-08-19, twice): without it long-context decode
    fragments the caching allocator into an OOM-retry loop instead of completing, and the box
    scripts that established the colocate path all exported it by hand. This lands the same value
    in the entry points themselves, so a bare `python -m games.train` on a fresh box is covered;
    `scripts/resource-limits.sh` covers only launches that go through the limiter.

    `setdefault` on purpose: an operator experimenting with allocator settings still wins, and the
    resolved value is recorded per run through `UNRECORDED_GENERATION_ENV` either way.
    """
    os.environ.setdefault(CUDA_ALLOC_CONF_ENV, DEFAULT_CUDA_ALLOC_CONF)
    resolved = os.environ[CUDA_ALLOC_CONF_ENV]
    logger.info(f"CUDA allocator config: {CUDA_ALLOC_CONF_ENV}={resolved}")
    return resolved


def detected_world_size(environment: Mapping[str, str]) -> int:
    """Read the launcher's process count out of the environment.

    Takes the mapping rather than reading `os.environ` so the detection is testable. `torchrun`
    and `accelerate launch` both export `WORLD_SIZE`; `LOCAL_WORLD_SIZE` is checked too so a
    single-node multi-GPU launch that only sets the local one is still caught.
    """
    sizes = [
        int(environment[key])
        for key in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")
        if environment.get(key, "").isdigit()
    ]
    return max(sizes, default=1)


def assert_single_process(world_size: int, *, source: str) -> None:
    """Refuse to train under multiple processes, whatever the reward function would return.

    The reward function's group-contiguity guard checks that each prompt's `G` completions arrive
    as one contiguous block. That holds for the *gathered* reward vector, not for a per-process
    slice: TRL's sampler deliberately distributes a prompt's generations across processes
    (`grpo_trainer.py:1249`), and `GRPOConfig` never enforces
    `per_device_train_batch_size % num_generations == 0`. So a local slice can hold a fragment of
    a group, and the group-mix opponent distribution would be an average over that fragment --
    an experiment that runs, reports numbers, and measures something nobody asked for.
    """
    if world_size > 1:
        raise RuntimeError(
            f"single-process training only ({source} reports {world_size=}): the group-mix reward "
            f"assumes each prompt's generations arrive contiguously in one process, which TRL "
            f"guarantees only for the gathered reward vector"
        )


def check_trace_completeness(
    *,
    generation_batch_size: int,
    logging_steps: float,
    episodes_per_step: int,
    dynamic_sampling_oversample: int = 1,
) -> bool:
    """Confirm every completion of every step reaches the parquet trace, and report if not.

    TRL buffers completions in deques bounded to `generation_batch_size`
    (`grpo_trainer.py:1069-1074`) and writes them to `<output_dir>/completions/*.parquet` inside
    `log`. So a hole-free trace needs one generation per optimizer step and a log on every step.

    Gradient accumulation is not the risk here, which is worth stating because it looks like it
    should be: with `generation_batch_size` and `steps_per_generation` both left unset, TRL sets
    `steps_per_generation = gradient_accumulation_steps` and `generation_batch_size =
    per_device_train_batch_size * steps_per_generation` (`grpo_config.py:1083-1085`), so the
    buffer is exactly one optimizer step wide however the step is split up. Forcing accumulation
    to 1 to protect the trace would instead force the micro-batch to the whole episode batch,
    which is the configuration the measured runs OOM on.

    Under dynamic sampling a step GENERATES `oversample` times its episodes and trains on the live
    subset (`games.train.DynamicSampledGRPOTrainer`), and the buffer is widened to match so that the
    dropped rollouts reach the trace too -- they are how a readout sees the strata that mostly go
    pure. So the width the trace needs is the oversampled one, and a buffer merely one step wide
    would silently keep only the tail of each generation.
    """
    buffered_episodes = episodes_per_step * dynamic_sampling_oversample
    if generation_batch_size != buffered_episodes:
        raise RuntimeError(
            f"the completions buffer would not match what a step generates, so the rollout trace "
            f"would silently lose completions: {generation_batch_size=} against "
            f"{buffered_episodes=} ({episodes_per_step=} times "
            f"{dynamic_sampling_oversample=}). Leave generation_batch_size and "
            f"steps_per_generation unset so TRL derives one generation per step, and let "
            f"`widen_generation_batch_for_oversample` do the widening."
        )
    if logging_steps != 1:
        logger.warning(
            "the rollout trace will have holes: the completions buffer holds one generation "
            "batch and is overwritten between logs, %s",
            f"{logging_steps=} (needs 1 for a complete trace)",
        )
        return False
    return True


# TRL's `GRPOTrainer.log` hardcodes both: the `to_parquet` call writes one file per logged step.
COMPLETIONS_DIRNAME = "completions"
TRACE_FILENAME_TEMPLATE = "completions_{step:05d}.parquet"


def trace_file_path(output_dir: Path, step: int) -> Path:
    """Return the parquet TRL writes for optimizer step `step` of the run in `output_dir`."""
    return output_dir / COMPLETIONS_DIRNAME / TRACE_FILENAME_TEMPLATE.format(step=step)


def logged_steps(steps: int, *, logging_steps: int) -> tuple[int, ...]:
    """Return the optimizer steps whose completions reach a parquet in a run of `steps` steps.

    transformers logs when `global_step % logging_steps == 0`, and once more from
    `_finalize_training`, which rewrites the last step's file from the still-full buffer.
    """
    if steps < 1:
        return ()
    return tuple(sorted({*range(logging_steps, steps + 1, logging_steps), steps}))


def parquet_row_count(path: Path) -> int:
    """Rows in a parquet file, read from its footer rather than by loading the columns."""
    return int(pq.read_metadata(path).num_rows)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]


@dataclass(frozen=True)
class TraceFilesVerdict:
    """What the completions directory actually holds, step by step, against what the run produced.

    JSON-shaped throughout (tuples of ints, no Paths) because `train_summary.json` carries it via
    `dataclasses.asdict`.
    """

    checked_dir: str
    steps_expected: int
    expected_rows_per_step: int
    missing_steps: tuple[int, ...]
    empty_steps: tuple[int, ...]
    # `(step, rows found)` for every file whose row count is neither zero nor the expected one.
    wrong_row_count_steps: tuple[tuple[int, int], ...]

    @property
    def complete(self) -> bool:
        """Whether every expected step has a file holding exactly the expected rows."""
        return not (self.missing_steps or self.empty_steps or self.wrong_row_count_steps)


def verify_trace_files(
    output_dir: Path, *, steps: int, expected_rows: int, logging_steps: int = 1
) -> TraceFilesVerdict:
    """Read the completions directory and name every logged step without a full parquet.

    `check_trace_completeness` decides from config arithmetic whether the trace CAN be complete;
    this reads the files and says whether it IS. The two disagreed on 2026-09-01: a relaunch that
    resumed a finished arm wrote a zero-row parquet over step 70 of 70, and the run summary still
    reported the trace complete because nothing had ever opened the directory. Row counts come from
    each file's footer, so a 70-step run is checked in milliseconds.
    """
    missing: list[int] = []
    empty: list[int] = []
    wrong: list[tuple[int, int]] = []
    for step in logged_steps(steps, logging_steps=logging_steps):
        path = trace_file_path(output_dir, step)
        if not path.is_file():
            missing.append(step)
            continue
        rows = parquet_row_count(path)
        if rows == 0:
            empty.append(step)
        elif rows != expected_rows:
            wrong.append((step, rows))
    verdict = TraceFilesVerdict(
        checked_dir=str(output_dir / COMPLETIONS_DIRNAME),
        steps_expected=len(logged_steps(steps, logging_steps=logging_steps)),
        expected_rows_per_step=expected_rows,
        missing_steps=tuple(missing),
        empty_steps=tuple(empty),
        wrong_row_count_steps=tuple(wrong),
    )
    if not verdict.complete:
        logger.warning(
            "ROLLOUT TRACE INCOMPLETE under %s: missing steps %s, empty files at steps %s, wrong row "
            "counts (step, rows) %s against %d rows per step",
            verdict.checked_dir,
            list(verdict.missing_steps),
            list(verdict.empty_steps),
            [list(pair) for pair in verdict.wrong_row_count_steps],
            expected_rows,
        )
    return verdict


def refuse_empty_trace_overwrite(trace_path: Path, *, buffered_completions: int) -> None:
    """Refuse to let an empty completions buffer replace a step's existing rows.

    `GRPOTrainer.log` writes `completions_<step>.parquet` from its buffers whenever completion
    logging is on, and transformers calls `log` once from `_finalize_training` even when the epoch
    loop ran zero steps. A relaunch resumed onto a checkpoint already at `max_steps` therefore logs
    exactly once, with buffers that were constructed empty, under the finished run's last step
    number -- and the file it writes is zero rows and four columns where 64 rows were (2026-09-01,
    `pd-unstated-other-payoff`, restored afterwards from S3 versioning). This runs before TRL's
    write, so the outcome is a crash rather than a quietly smaller file for `aws s3 sync` to mirror
    over the good one.

    An empty buffer aimed at a step with no rows on disk is left alone: there is nothing to destroy,
    and `verify_trace_files` names the empty file at summary time.
    """
    if buffered_completions > 0 or not trace_path.is_file():
        return
    existing_rows = parquet_row_count(trace_path)
    if existing_rows == 0:
        return
    raise RuntimeError(
        f"refusing to overwrite the rollout trace {trace_path} ({existing_rows} rows) with an empty "
        f"completions buffer. This is the signature of a relaunch that resumed a checkpoint already "
        f"at max_steps and trained zero steps: transformers still logs once at train end, and TRL "
        f"would write a zero-row parquet over the step's real completions. Nothing was written."
    )


def derive_prefilled_think(
    tokenizer: PreTrainedTokenizerBase, *, enable_thinking: bool = True
) -> bool:
    """Measure whether this tokenizer's template opens `<think>` inside the prompt itself.

    Rendered rather than looked up by model name: the defaults move non-monotonically across the
    Qwen ladder (`docs/scratch/qwen38-27b-load-check-2026-08-17.md`), and a name table would go
    stale the first time a new checkpoint lands. When the template prefills, generation starts
    inside the thinking block and the completion can only ever contain the closing tag.
    """
    from games.parsing import THINK_CLOSE, THINK_OPEN  # noqa: PLC0415

    rendered = cast(
        "str",
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        ),
    )
    if THINK_OPEN not in rendered:
        return False
    return THINK_CLOSE not in rendered.rsplit(THINK_OPEN, 1)[1]


def resolve_chat_template_kwargs(tokenizer: PreTrainedTokenizerBase) -> dict[str, str]:
    """Pin `reasoning_effort` when the template has that knob, and nothing otherwise.

    Qwen3.8-27B's template injects an unauthored system message at its default effort ("Reasoning
    effort is set to xhigh…") and prepends it to anything we supply. Only `medium` renders empty
    steering text. Left alone, the 27B arm would train on different prompt text and a far larger
    thinking budget than every other arm, which is the one thing the arms must not differ in.
    """
    template = tokenizer.get_chat_template()
    if "reasoning_effort" not in template:
        return {}
    return {"reasoning_effort": "medium"}


def resolve_tokenizer(
    model_id: str, *, thinking: bool
) -> tuple[PreTrainedTokenizerBase, dict[str, object]]:
    """Load the tokenizer, repair an ambiguous stop token, and measure the template facts.

    Both trainers call this, and the ORDER inside it is the whole point: it runs before the trainer is
    constructed, because TRL builds its own ``GenerationConfig`` from this tokenizer inside
    ``GRPOTrainer.__init__`` and overrides the model's. So a checkpoint whose scalar ``eos_token`` is
    not what its template emits has to be repaired first. Repaired afterwards, nothing halts: every
    rollout runs to the completion cap, submits no solution, reads as truncated thinking, and earns the
    parse penalty until the reward's whole-batch raise fires.

    Here rather than in either trainer because all three of its constituent calls already live in this
    module, so it adds no dependency edge and does not make one experiment's trainer a de facto shared
    library for the other. It took primitives rather than either config for the same reason: the two
    trainers' config types are unrelated, and this function needs two fields out of them.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    scalar_eos = repair_scalar_eos(tokenizer, model_id)
    prefilled_think = derive_prefilled_think(tokenizer, enable_thinking=thinking)
    chat_template_kwargs = resolve_chat_template_kwargs(tokenizer)
    logger.info(
        "template facts, %s",
        f"{prefilled_think=} {chat_template_kwargs=} {scalar_eos=} {model_id=}",
    )
    facts: dict[str, object] = {
        "prefilled_think": prefilled_think,
        "chat_template_kwargs": chat_template_kwargs,
        "scalar_eos": scalar_eos,
    }
    return tokenizer, facts


def lora_module_counts(trainer: GRPOTrainer) -> dict[str, object]:
    """Count what LoRA adapted, without demanding a linear-attention layer exist.

    `grpo.throughput.summarize_lora` raises unless an adapter landed on a `linear_attn` module,
    which is exactly right on the Qwen3.5/3.8 hybrid stack and wrong on Qwen3-0.6B, whose stack
    has no such layers to freeze. `summarize_lora_for_architecture` routes to whichever check the
    checkpoint's own config justifies; this is the no-linear-attention half.

    The counting is duplicated rather than shared, and that is a known wart. The two functions walk
    `named_modules` and `parameters` identically and return the same five keys; only the guard
    differs -- this one refuses "no adapter landed anywhere" (which `summarize_lora` never checks),
    that one refuses "no adapter on a linear-attention layer" (which this one records without
    objecting to). So a new counted field has to be added in both places. Sharing them means
    `grpo/throughput.py` keeping one counter that raises nothing and this module applying whichever
    guard the config justifies to its dict, which is an edit to `grpo/` rather than to `games/`.
    """
    model = trainer.model
    adapted = [
        name
        for name, _ in model.named_modules()  # pyright: ignore[reportOptionalMemberAccess]
        if name.endswith("lora_A.default")
    ]
    if not adapted:
        raise RuntimeError(
            "no LoRA adapter landed anywhere on the model; nothing would train. Check the "
            "target_modules discovery against the checkpoint's module names."
        )
    linear_attention_adapted = [name for name in adapted if "linear_attn" in name]
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()  # pyright: ignore[reportOptionalMemberAccess]
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())  # pyright: ignore[reportOptionalMemberAccess]
    return {
        "adapted_modules": len(adapted),
        "adapted_linear_attention_modules": len(linear_attention_adapted),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_fraction": trainable / total,
    }


def summarize_lora_for_architecture(
    trainer: GRPOTrainer, *, expected_linear_attention_layers: int
) -> dict[str, object]:
    """Verify LoRA coverage against what the checkpoint's own config declares.

    The demand that an adapter reach a linear-attention layer is delegated to
    `grpo.throughput.summarize_lora` whenever the config declares such layers, so the guard that
    catches a hardcoded q/k/v/o list on Qwen3.5 stays the one that has teeth.
    """
    if expected_linear_attention_layers:
        return summarize_lora(trainer)
    return lora_module_counts(trainer)


def log_resolved_sampler(trainer: GRPOTrainer) -> dict[str, object]:
    """Log the sampler this trainer will actually generate with, and return it for the run record.

    Read off the built trainer rather than off `GameTrainConfig`, because what decides the rollouts
    includes values this repo never passes: `min_p` and `repetition_penalty` are whatever the
    installed TRL defaults to, and `generation_kwargs` can override any of the rest. A comparison
    of a transformers-generate run against a vLLM one had to reconstruct all of this from a tarball
    of the code the box was running, and only got away with it by luck.

    On the transformers path the `GenerationConfig` TRL builds is authoritative, because that object
    is what `generate()` reads once `generation_kwargs` has been folded into it. Under vLLM there is
    no `GenerationConfig` at all -- the same values reach the client as sampling parameters -- so
    `do_sample` and `cache_implementation` are recorded as None rather than dropped, which keeps the
    key set identical across backends and therefore keeps two arms' samplers diffable.

    The log line, not `train_summary.json`, is the copy that survives a run killed before it
    finishes, which is the case that made a sampler unrecoverable in the first place.
    """
    args = cast("GRPOConfig", trainer.args)
    sampler: dict[str, object] = {
        "generation_backend": (
            f"vllm-{trainer.vllm_mode}" if trainer.use_vllm else "transformers-generate"
        ),
        "num_generations": trainer.num_generations,
        "generation_kwargs": args.generation_kwargs,
        "temperature": trainer.temperature,
        "top_p": trainer.top_p,
        "top_k": trainer.top_k,
        "min_p": trainer.min_p,
        "repetition_penalty": trainer.repetition_penalty,
        "max_new_tokens": trainer.max_completion_length,
        "do_sample": None,
        "cache_implementation": None,
    }
    if not trainer.use_vllm:
        generation = trainer.generation_config
        sampler.update(
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            min_p=generation.min_p,
            repetition_penalty=generation.repetition_penalty,
            max_new_tokens=generation.max_new_tokens,
            do_sample=generation.do_sample,
            cache_implementation=generation.cache_implementation,
        )
    # Sorted JSON rather than a repr, so two runs' lines diff cleanly.
    logger.info("resolved sampler, %s", json.dumps(sampler, default=str, sort_keys=True))
    return sampler


def assert_trainer_generates_through_vllm(trainer: GRPOTrainer) -> None:
    """Refuse a built games trainer that would roll out through anything but colocated vLLM.

    The consumption-point end of the vLLM-only rule (`games.generation.VLLM_ONLY_RATIONALE`): the
    environment guard and the config can both be right while a TRL bump quietly stops honouring
    `use_vllm`, so this reads the two attributes the trainer actually generates from -- the same
    seam `log_resolved_sampler` records. Games-only: the reward-hacking trainer keeps a deliberate
    slow path behind an explicit --allow-hf-generation acknowledgment and must not call this.
    """
    if trainer.use_vllm and trainer.vllm_mode == "colocate":
        return
    backend = f"vllm-{trainer.vllm_mode}" if trainer.use_vllm else "transformers-generate"
    raise RuntimeError(
        f"the built trainer would generate rollouts through {backend!r}, not the colocated vLLM "
        f"engine. Games training rollouts are vLLM-only (owner decision 2026-08-26: 'vllm only, "
        f"always'; ~72 min/step on transformers.generate against ~13.5 through the engine), so a "
        f"trainer that lost the setting between config and construction is refused before its "
        f"first step is paid for."
    )


def repair_scalar_eos(tokenizer: PreTrainedTokenizerBase, model_id: str) -> dict[str, object]:
    """Give TRL the single stop token it assumes, when the checkpoint declares several.

    Some checkpoints ship `generation_config.eos_token_id` as a LIST while the tokenizer's scalar
    `eos_token` is a different, older token that the chat template never emits. TRL assumes a scalar
    EOS in several places, so on such a model nothing ever stops: every rollout runs to the
    completion cap and reads as truncated thinking -- indistinguishable from a model that genuinely
    cannot stop deliberating, which is a failure we have already spent a night chasing.

    Verified case, `openbmb/MiniCPM5-1B` on 2026-08-17: tokenizer `eos_token` and `pad_token` are
    BOTH `</s>` (id 1), `generation_config.eos_token_id` is `[1, 130073]`, and the template
    terminates every turn with `<|im_end|>` (130073). Left alone, generation never halts.

    The rule is general rather than a per-model table: when several stop tokens are declared, the
    one that is not the padding token is the real terminator, because a pad token cannot double as
    a stop signal without making "stopped" and "padded" the same observation.
    """
    try:
        declared = GenerationConfig.from_pretrained(model_id).eos_token_id
    except OSError:
        # A checkpoint that ships no generation_config.json declares no stop tokens at all, so there
        # is nothing ambiguous to repair. Qwen3.5-2B is such a checkpoint; caught narrowly rather
        # than broadly, because any other failure reading it is a real problem worth crashing on.
        return {
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "declared_eos_token_id": None,
            "repaired": False,
            "reason": "checkpoint ships no generation_config.json",
        }
    before = {
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "declared_eos_token_id": declared,
    }
    if not isinstance(declared, list) or len(declared) < MIN_AMBIGUOUS_STOP_TOKENS:
        return {**before, "repaired": False, "reason": "checkpoint declares a single stop token"}

    candidates = [token_id for token_id in declared if token_id != tokenizer.pad_token_id]
    if not candidates:
        return {**before, "repaired": False, "reason": "every declared stop token is the pad token"}
    terminator = candidates[-1]
    if terminator == tokenizer.eos_token_id:
        return {**before, "repaired": False, "reason": "scalar eos already the real terminator"}

    tokenizer.eos_token = tokenizer.convert_ids_to_tokens(terminator)
    logger.warning(
        "repaired a multi-stop-token checkpoint so TRL's scalar-EOS assumption holds, %s",
        f"model_id={model_id!r} {declared=} eos_token={tokenizer.eos_token!r} "
        f"eos_token_id={tokenizer.eos_token_id} pad_token={tokenizer.pad_token!r} "
        f"pad_token_id={tokenizer.pad_token_id}",
    )
    if tokenizer.eos_token_id == tokenizer.pad_token_id:
        raise RuntimeError(
            f"after repair the stop and pad tokens are still identical for {model_id!r}, so a "
            f"stopped completion cannot be told from a padded one; {declared=}"
        )
    return {
        **before,
        "repaired": True,
        "eos_token_after": tokenizer.eos_token,
        "eos_token_id_after": tokenizer.eos_token_id,
        "reason": "picked the declared stop token that is not the pad token",
    }


# The DeltaNet kernel functions transformers looks for, and where it looks. Names and lookup order
# read out of transformers 5.15.0's `use_kernel_func_from_hub_with_fallback`, whose priority is
# Hub kernels (needing the `kernels` distribution), then the named local package, then torch.
DELTANET_KERNEL_FUNCTIONS = (
    ("chunk_gated_delta_rule", "fla"),
    ("recurrent_gated_delta_rule", "fla"),
    ("causal_conv1d_fn", "causal_conv1d"),
    ("causal_conv1d_update", "causal_conv1d"),
)


def deltanet_kernel_paths() -> dict[str, str]:
    """Report which implementation each Gated DeltaNet kernel function will actually use.

    Installed is not engaged, and transformers falls back SILENTLY: an absent package, or a package
    whose export names do not match what transformers looks for, leaves the slow torch path in place
    with nothing in the log. Three quarters of the Qwen3.5 stack is DeltaNet, so that difference is
    most of a run's wall clock, and a run that was slow for this reason should say so in its own
    artifacts rather than being re-litigated later.

    Verified on this box 2026-08-17: with flash-linear-attention present, `chunk_gated_delta_rule`
    resolves to `fla.ops.gated_delta_rule.chunk` (the chunked prefill and training path) while
    `recurrent_gated_delta_rule` does NOT, because fla exports it as
    `fused_recurrent_gated_delta_rule` and transformers looks for the unprefixed name.
    `games.deltanet_kernels.bridge_decode_kernel` closes that gap; call it before any Qwen3.5 model
    loads, and this function is how a run records which path it got.

    What the fallback costs was measured on this box's L4 with Qwen3.5-2B in bf16 (2026-08-18):
    1.08x at 8 sequences, 1.52x at 32, 1.94x at 64 -- it scales with the number of sequences
    decoding together rather than with completion length, because at decode the fallback runs one
    loop iteration over batch x heads x K x V and is launch-bound until that has real work to do.
    """
    # Reaching into transformers' private kernel registry, deliberately: there is no public way to
    # ask "which implementation will this kernel function actually use", and the alternative is
    # believing a silent fallback. Pinned transformers, so a rename surfaces here rather than in a
    # wrong step time.
    from transformers.integrations.hub_kernels import (  # noqa: PLC0415
        _KERNELS_INTERNAL_PATH_MAPPINGS,  # pyright: ignore[reportPrivateUsage]
        resolve_internal_import,  # pyright: ignore[reportPrivateImportUsage]
    )

    paths: dict[str, str] = {}
    for function_name, package in DELTANET_KERNEL_FUNCTIONS:
        internal = _KERNELS_INTERNAL_PATH_MAPPINGS.get(function_name)
        full = function_name if internal is None else f"{internal}.{function_name}"
        try:
            module = importlib.import_module(package)
        except ImportError:
            paths[function_name] = f"torch fallback ({package} absent)"
            continue
        resolved = resolve_internal_import(module, full)
        paths[function_name] = (
            f"{resolved.__module__}.{resolved.__name__}"
            if resolved is not None
            else f"torch fallback ({package} has no {full})"
        )
    return paths


def log_deltanet_kernel_paths(*, expected_linear_attention_layers: int) -> dict[str, str]:
    """Log the kernel paths, loudly when a hybrid model is about to run on the slow one."""
    paths = deltanet_kernel_paths()
    if not expected_linear_attention_layers:
        logger.info("no linear-attention layers, so DeltaNet kernels are irrelevant here")
        return paths
    for function_name, implementation in paths.items():
        if implementation.startswith("torch fallback"):
            logger.warning(
                "DeltaNet kernel on the SLOW path, %s -> %s", function_name, implementation
            )
        else:
            logger.info("DeltaNet kernel engaged, %s -> %s", function_name, implementation)
    return paths
