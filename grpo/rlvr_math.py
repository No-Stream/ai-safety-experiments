"""GRPO RLVR integer-math training substrate."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer  # pyright: ignore[reportPrivateImportUsage]

from grpo.estimator_defaults import (
    GRPO_EPSILON,
    GRPO_LOSS_TYPE,
    GRPO_SCALE_REWARDS,
    assert_liger_faithful_estimator,
    executed_estimator,
)
from grpo.throughput import discover_lora_targets

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
SYSTEM_PROMPT = (
    "You are a calculator. Solve the user's math problem and reply with an integer. "
    "ASCII characters only, no markdown. "
    "Your final answer should be on a new line at the end of your response. "
)

# Regex matches the last integer-looking token in the output (supports negatives)
NUM_RE = re.compile(r"[-+]?\d+")
MAX_LOGGED_SAMPLES = 10


def build_messages(problem: str) -> list[dict[str, str]]:
    """Keep the verifier prompt identical across generation paths."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


def parse_answer(text: str) -> int | None:
    """Extract the final integer so formatting does not affect verification."""
    cleaned = text.strip().replace(",", "").replace("_", "")
    matches = NUM_RE.findall(cleaned)
    if not matches:
        return None
    return int(matches[-1])


def _sample_int(rng: random.Random, low: int, high: int) -> int:
    x = rng.randint(low, high)
    if x == 0:
        x = rng.choice([low, high, 1, -1])
    return x


def gen_synthetic_math(
    n: int = 500,
    seed: int = 0,
    add_sub_range: int = 99999,
    mul_range: int = 50,
) -> list[tuple[str, int]]:
    """Provide a reproducible simple arithmetic probe for RLVR training."""
    rng = random.Random(seed)
    items: list[tuple[str, int]] = []
    for _ in range(n):
        a = _sample_int(rng, -add_sub_range, add_sub_range)
        b = _sample_int(rng, -add_sub_range, add_sub_range)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
            prob = f"{a} + {b} = ?"
        elif op == "-":
            ans = a - b
            prob = f"{a} - {b} = ?"
        elif op == "*":
            a2 = _sample_int(rng, -mul_range, mul_range)
            b2 = _sample_int(rng, -mul_range, mul_range)
            ans = a2 * b2
            prob = f"{a2} * {b2} = ?"
        else:
            raise ValueError("Invalid operation.")
        items.append((prob, int(ans)))
    return items


def gen_ltr_arithmetic(  # noqa: PLR0913, PLR0917
    n: int = 500,
    seed: int = 0,
    min_steps: int = 2,
    max_steps: int = 3,
    add_sub_range: int = 999,
    mul_range: int = 20,
) -> list[tuple[str, int]]:
    """Provide a reproducible precedence-sensitive probe for RLVR training."""
    rng = random.Random(seed)
    ops = ["+", "-", "*"]
    items: list[tuple[str, int]] = []
    for _ in range(n):
        steps = rng.randint(min_steps, max_steps)
        ops_selected = [rng.choice(ops) for _ in range(steps)]
        nums: list[int] = []
        for i in range(steps + 1):
            prev_mul = i > 0 and ops_selected[i - 1] == "*"
            nums.append(
                _sample_int(rng, -mul_range, mul_range)
                if prev_mul
                else _sample_int(rng, -add_sub_range, add_sub_range)
            )
        tokens: list[str] = []
        for i in range(steps):
            tokens.append(str(nums[i]))
            tokens.append(ops_selected[i])
        tokens.append(str(nums[-1]))
        expr = " ".join(tokens)
        res = nums[0]
        for i, op in enumerate(ops_selected):
            b = nums[i + 1]
            if op == "+":
                res = res + b
            elif op == "-":
                res = res - b
            elif op == "*":
                res = res * b
        prompt = (
            "Evaluate this expression strictly from left to right "
            "(ignore normal operator precedence):\n"
            f"{expr}\n"
            "What is the result?"
        )
        items.append((prompt, int(res)))
    return items


def gen_word_multi_step(  # noqa: PLR0913, PLR0917
    n: int = 500,
    seed: int = 0,
    min_ops: int = 3,
    max_ops: int = 5,
    value_range: int = 9999,
    mul_range: int = 50,
) -> list[tuple[str, int]]:
    """Provide a reproducible multi-step natural-language probe for RLVR training."""
    rng = random.Random(seed)
    items: list[tuple[str, int]] = []
    verbs = {"+": "add", "-": "subtract", "*": "multiply by"}
    for _ in range(n):
        k = rng.randint(min_ops, max_ops)
        ops = [rng.choice(list(verbs.keys())) for _ in range(k)]
        start = _sample_int(rng, -value_range, value_range)
        vals = [
            _sample_int(rng, -mul_range, mul_range)
            if op == "*"
            else _sample_int(rng, -value_range, value_range)
            for op in ops
        ]
        res = start
        parts = [f"Start with {start}."]
        for op, v in zip(ops, vals, strict=True):
            if op == "+":
                res = res + v
                parts.append(f"Then add {v}.")
            elif op == "-":
                res = res - v
                parts.append(f"Then subtract {v}.")
            elif op == "*":
                res = res * v
                parts.append(f"Then multiply by {v}.")
        parts.append("What is the result?")
        prompt = " ".join(parts)
        items.append((prompt, int(res)))
    return items


def _pairs_to_rows(pairs: Sequence[tuple[str, int]]) -> list[dict[str, str | int]]:
    return [{"problem": p, "gold": int(g)} for p, g in pairs]


def render_prompts(
    rows: list[dict[str, str | int]],
    model_id: str = DEFAULT_MODEL_ID,
    system_prompt: str = SYSTEM_PROMPT,
    max_prompt_tokens: int | None = None,
) -> Dataset:
    """Render rows into GRPO's expected `prompt`/`gold` columns.

    Over-budget prompts are dropped rather than truncated. TRL removed `GRPOConfig`'s
    `max_prompt_length`, and its old left-truncation is wrong for a verifier task anyway:
    cutting tokens off a math problem changes the question while `gold` still holds the
    answer to the original, so the reward becomes unreachable and quietly poisons training.
    """
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    def _render(row: dict[str, str | int]) -> dict[str, str | int]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["problem"]},
        ]
        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        n_tokens = len(tok(prompt, add_special_tokens=False)["input_ids"])
        return {"prompt": prompt, "gold": row["gold"], "n_prompt_tokens": n_tokens}

    ds = Dataset.from_list(rows)
    ds = ds.map(_render, remove_columns=list(ds.column_names))

    if max_prompt_tokens is not None:
        n_before = len(ds)
        ds = ds.filter(lambda row: row["n_prompt_tokens"] <= max_prompt_tokens)
        n_dropped = n_before - len(ds)
        if n_dropped:
            logger.warning(
                "dropped over-budget prompts, %s", f"{n_dropped=} {n_before=} {max_prompt_tokens=}"
            )

    # TRL forwards every surviving column to the reward functions as a kwarg, so the
    # bookkeeping column must not reach them.
    return ds.remove_columns(["n_prompt_tokens"])


def reward_correct_integer(
    completions: list[str],
    gold: list[int],
    **kwargs: object,  # noqa: ARG001
) -> list[float]:
    """Binary verifier reward: 1.0 when the last integer in the completion equals gold."""
    return [
        1.0 if parse_answer(out) == gt else 0.0 for out, gt in zip(completions, gold, strict=True)
    ]


def _make_task_pairs(
    task_cfg: TaskConfig,
    n_train: int,
    n_eval: int,
    train_seed: int,
    eval_seed: int,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    if task_cfg.task_mode == "simple":
        train_pairs = (
            gen_synthetic_math(
                n=n_train,
                seed=train_seed,
                add_sub_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_synthetic_math(
            n=n_eval,
            seed=eval_seed,
            add_sub_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    elif task_cfg.task_mode == "ltr":
        train_pairs = (
            gen_ltr_arithmetic(
                n=n_train,
                seed=train_seed,
                min_steps=task_cfg.ltr_min_steps,
                max_steps=task_cfg.ltr_max_steps,
                add_sub_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_ltr_arithmetic(
            n=n_eval,
            seed=eval_seed,
            min_steps=task_cfg.ltr_min_steps,
            max_steps=task_cfg.ltr_max_steps,
            add_sub_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    elif task_cfg.task_mode == "word":
        train_pairs = (
            gen_word_multi_step(
                n=n_train,
                seed=train_seed,
                min_ops=task_cfg.word_min_ops,
                max_ops=task_cfg.word_max_ops,
                value_range=task_cfg.val_range,
                mul_range=task_cfg.mul_range,
            )
            if n_train
            else []
        )
        eval_pairs = gen_word_multi_step(
            n=n_eval,
            seed=eval_seed,
            min_ops=task_cfg.word_min_ops,
            max_ops=task_cfg.word_max_ops,
            value_range=task_cfg.val_range,
            mul_range=task_cfg.mul_range,
        )
    else:
        raise ValueError(f"Unknown TASK_MODE={task_cfg.task_mode}")
    return train_pairs, eval_pairs


def measure_baseline_accuracy(  # noqa: PLR0913, PLR0917
    model_id: str = DEFAULT_MODEL_ID,
    task_cfg: TaskConfig | None = None,
    n_eval: int = 100,
    device: str = "cuda",
    # Defaults track TrainConfig: a baseline measured at a different precision than the policy
    # it is the baseline for is not a comparison. None defers to the checkpoint's own dtype.
    dtype: torch.dtype | None = torch.bfloat16,
    load_in_4bit: bool = False,  # noqa: FBT001, FBT002
    eval_seed: int = 123,
    chat_template: str | None = None,
) -> dict[str, object]:
    """Establish baseline accuracy before training changes the policy."""
    task_cfg = task_cfg or TaskConfig()
    device_map = "auto" if device != "cpu" else {"": "cpu"}
    kwargs: dict[str, object] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.bos_token_id = getattr(tok, "bos_token_id", None)
    tok.padding_side = "left"
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.bos_token_id = getattr(tok, "bos_token_id", None)

    torch.backends.cuda.matmul.allow_tf32 = True
    model.eval()

    _, eval_pairs = _make_task_pairs(
        task_cfg=task_cfg,
        n_train=0,
        n_eval=n_eval,
        train_seed=eval_seed,
        eval_seed=eval_seed,
    )

    correct = 0
    samples: list[dict[str, object]] = []
    for i, (problem, ans) in enumerate(eval_pairs):
        messages = build_messages(problem)
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            chat_template=chat_template,
        )
        inputs = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = parse_answer(gen)
        is_ok = pred == ans
        correct += int(is_ok)
        if i < MAX_LOGGED_SAMPLES:
            samples.append(
                {
                    "problem": problem,
                    "gold": ans,
                    "raw": gen.strip(),
                    "pred": pred,
                    "ok": bool(is_ok),
                }
            )

    acc = correct / max(1, len(eval_pairs))
    result: dict[str, object] = {
        "model": model_id,
        "task_mode": task_cfg.task_mode,
        "n": len(eval_pairs),
        "accuracy": acc,
        "samples": samples,
    }
    logger.info(json.dumps(result, indent=2))
    return result


def evaluate_model_accuracy(  # noqa: PLR0913, PLR0917
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    task_cfg: TaskConfig | None = None,
    n_eval: int = 100,
    device: str = "cuda",
    max_new_tokens: int = 128,
    eval_seed: int = 123,
    chat_template: str | None = None,
) -> dict[str, object]:
    """Measure verifier accuracy without altering the caller's evaluation set."""
    task_cfg = task_cfg or TaskConfig()
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model.config.pad_token_id = tok.pad_token_id  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    model.config.eos_token_id = tok.eos_token_id  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    model.config.bos_token_id = getattr(tok, "bos_token_id", None)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        model.generation_config.eos_token_id = tok.eos_token_id  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        model.generation_config.bos_token_id = getattr(tok, "bos_token_id", None)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]

    torch.backends.cuda.matmul.allow_tf32 = True

    _, eval_pairs = _make_task_pairs(
        task_cfg=task_cfg,
        n_train=0,
        n_eval=n_eval,
        train_seed=eval_seed,
        eval_seed=eval_seed,
    )

    correct = 0
    samples: list[dict[str, object]] = []
    for i, (problem, ans) in enumerate(eval_pairs):
        messages = build_messages(problem)
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            chat_template=chat_template,
        )
        inputs = tok(text, return_tensors="pt").to(device)  # pyright: ignore[reportArgumentType]
        with torch.no_grad():
            out = model.generate(  # pyright: ignore[reportCallIssue]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = parse_answer(gen)  # pyright: ignore[reportArgumentType]
        is_ok = pred == ans
        correct += int(is_ok)
        if i < MAX_LOGGED_SAMPLES:
            samples.append(
                {
                    "problem": problem,
                    "gold": ans,
                    "raw": gen.strip(),  # pyright: ignore[reportAttributeAccessIssue]
                    "pred": pred,
                    "ok": bool(is_ok),
                }
            )

    acc = correct / max(1, len(eval_pairs))
    result: dict[str, object] = {
        "model": getattr(model, "name_or_path", "provided"),
        "task_mode": task_cfg.task_mode,
        "n": len(eval_pairs),
        "accuracy": acc,
        "samples": samples,
    }
    logger.info(json.dumps(result, indent=2))
    return result


def _bytes_to_gib(x: int) -> float:
    return x / (1024**3)


class MemoryMonitorCallback(TrainerCallback):
    """Track GPU use because training throughput can hide memory regressions.

    Throughput is REAL tokens per second: the delta of TRL's cumulative `num_tokens` (prompt plus
    completion tokens actually generated, `state.num_input_tokens_seen`) over the wall clock between
    two log records. The earlier figure multiplied the completion count by `max_completion_length`
    and so reported padded capacity -- `tok/s~2867` on a run whose real rate was about 450
    (efficiency audit, 2026-09-02) -- which is the number a reader would have used to size the next
    run. A record with no `num_tokens` (a bare eval record) writes an empty cell rather than a guess.

    `extra_columns` names further keys of each log record to copy into `mem_log.csv`, so the
    per-step timing split and the padding-trim totals sit in the same row as the memory reading they
    belong with; a key absent from a record writes an empty cell.
    """

    def __init__(
        self,
        device: int = 0,
        alpha: float = 0.3,
        print_every: int = 10,
        extra_columns: Sequence[str] = (),
    ) -> None:
        """Configure memory telemetry without changing trainer behavior."""
        self.device = device
        self.alpha = alpha
        self.print_every = max(1, print_every)
        self.extra_columns = tuple(extra_columns)
        self._last_t: float | None = None
        self._last_step: int | None = 0
        self._last_num_tokens: float | None = None
        self._last_num_tokens_t: float | None = None
        self._ema_step_t: float | None = None
        self.csv_path: Path | None = None

    def _gpu_mem_stats(self) -> dict[str, float]:
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(self.device)
            alloc_b = torch.cuda.memory_allocated(self.device)
            reserv_b = torch.cuda.memory_reserved(self.device)
            return {
                "free_gib": _bytes_to_gib(free_b),
                "total_gib": _bytes_to_gib(total_b),
                "used_gib": _bytes_to_gib(total_b - free_b),
                "alloc_gib": _bytes_to_gib(alloc_b),
                "reserved_gib": _bytes_to_gib(reserv_b),
            }
        return {
            "free_gib": 0.0,
            "total_gib": 0.0,
            "used_gib": 0.0,
            "alloc_gib": 0.0,
            "reserved_gib": 0.0,
        }

    def csv_header(self) -> str:
        """Build the header line this instance writes: the fixed telemetry columns, then `extra_columns`."""
        return (
            "time,step,used_gib,alloc_gib,reserved_gib,usage_pct,ema_step_s,tok_s,approx_seq_s"
            + "".join(f",{name}" for name in self.extra_columns)
            + "\n"
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Start telemetry before the first trainer step is logged.

        A resumed launch (`global_step` already past zero, the trainer state having been loaded
        from the checkpoint before this fires) appends to the CSV the earlier launch left, so the
        rows of the steps that were paid for survive the restart; a fresh launch writes the header.
        The step clock starts at the resumed step, not zero, or the first record after a resume at
        step N would average N steps' worth of nothing into the EMA and every ETA after it.

        Appending is only right when the existing file's header is the one this instance writes.
        Every run launched before `tok_s` and `extra_columns` existed wrote nine columns, and a
        relaunch of one of those with this code would otherwise leave a ragged file that
        `load_mem_log`'s `read_csv` refuses -- inside `_summarize_run`, after the training had
        finished and before the final sync, so the run would train to completion and then die
        without its summary. A file with a different header is rotated aside to
        `mem_log.<UTC stamp>.csv`, its paid rows intact and still read by `load_mem_log`, and a
        fresh file starts under this launch's columns.
        """
        self._last_t = time.time()
        self._last_step = state.global_step
        ms = self._gpu_mem_stats()
        self.csv_path = Path(args.output_dir) / "mem_log.csv"  # pyright: ignore[reportArgumentType]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        header = self.csv_header()
        if state.global_step > 0 and self.csv_path.is_file():
            with self.csv_path.open(encoding="utf-8") as f:
                existing_header = f.readline()
            if existing_header == header:
                logger.info(
                    "[mem] resuming at step %d, appending to the existing %s",
                    state.global_step,
                    self.csv_path,
                )
            else:
                stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
                rotated = self.csv_path.with_name(f"mem_log.{stamp}.csv")
                self.csv_path.rename(rotated)
                self.csv_path.write_text(header, encoding="utf-8")
                logger.warning(
                    "[mem] resuming at step %d but %s was written with a different header (%r); "
                    "rotated it to %s and started a fresh file with this launch's %d columns",
                    state.global_step,
                    self.csv_path,
                    existing_header.rstrip("\n"),
                    rotated,
                    header.count(",") + 1,
                )
        else:
            self.csv_path.write_text(header, encoding="utf-8")
        if state.is_world_process_zero:
            logger.info(
                "[mem] start used=%.2fGiB free=%.2fGiB total=%.2fGiB reserved=%.2fGiB",
                ms["used_gib"],
                ms["free_gib"],
                ms["total_gib"],
                ms["reserved_gib"],
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Write resource metrics to mem_log.csv on the trainer's log schedule.

        The CSV is the only sink. `Trainer.log` appends its record to `log_history` before
        callbacks run, and on a copy, so telemetry written into `logs` here would persist
        nowhere and no reader would ever see it. `logs` is read, never written: the real token
        rate comes from its cumulative `num_tokens`, and `extra_columns` are copied out of it.
        """
        now = time.time()
        step_delta = state.global_step - self._last_step if self._last_step is not None else 0
        dt = now - self._last_t if self._last_t is not None else 0.0
        step_t = (dt / step_delta) if step_delta else None
        if step_t is not None:
            self._ema_step_t = (
                step_t
                if self._ema_step_t is None
                else (self.alpha * step_t + (1 - self.alpha) * self._ema_step_t)
            )
        ms = self._gpu_mem_stats()
        usage = (ms["used_gib"] / ms["total_gib"] * 100.0) if ms["total_gib"] else 0.0
        # TRL's per-device batch counts completions, not prompts, so multiplying by the group
        # size again would double-count it. `global_step` advances once per optimizer step, so
        # every accumulated micro-batch belongs to the step being timed.
        completions_per_step = (
            args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
        )
        # Tokens since the last record that carried `num_tokens`, over the wall clock since THAT
        # record, so an eval record in between neither zeroes the rate nor halves its window.
        num_tokens = (
            float(logs["num_tokens"]) if logs and logs.get("num_tokens") is not None else None
        )
        tok_s = None
        if (
            num_tokens is not None
            and self._last_num_tokens is not None
            and self._last_num_tokens_t is not None
            and now > self._last_num_tokens_t
        ):
            tok_s = (num_tokens - self._last_num_tokens) / (now - self._last_num_tokens_t)
        seq_s = (completions_per_step / self._ema_step_t) if self._ema_step_t else None
        eta_s = (
            ((args.max_steps - state.global_step) * self._ema_step_t)
            if (self._ema_step_t and args.max_steps)
            else None
        )
        if state.is_world_process_zero and state.global_step % self.print_every == 0:
            eta_str = f"ETA~{int(eta_s // 60)}m{int(eta_s % 60)}s" if eta_s else "ETA~na"
            logger.info(
                "[mem][step %d] used=%.2fGiB (%d%%) step_t=%.2fs real_tok/s=%s seq/s~%.1f %s",
                state.global_step,
                ms["used_gib"],
                int(usage),
                self._ema_step_t or 0.0,
                f"{tok_s:.0f}" if tok_s is not None else "na",
                seq_s or 0.0,
                eta_str,
            )
        if self.csv_path:
            extras = "".join(
                f",{_csv_cell(logs.get(name) if logs else None)}" for name in self.extra_columns
            )
            with self.csv_path.open("a") as f:
                f.write(
                    f"{int(now)},{state.global_step},{ms['used_gib']:.4f},{ms['alloc_gib']:.4f},"
                    f"{ms['reserved_gib']:.4f},{usage:.2f},{(self._ema_step_t or 0):.4f},"
                    f"{_csv_cell(tok_s)},{(seq_s or 0):.2f}{extras}\n"
                )
        self._last_t, self._last_step = now, state.global_step
        if num_tokens is not None:
            self._last_num_tokens, self._last_num_tokens_t = num_tokens, now


def _csv_cell(value: object) -> str:
    """Render one telemetry number for mem_log.csv, an absent value as an empty cell."""
    return "" if value is None else f"{float(cast('float', value)):.4f}"


class QuickEvalCallback(TrainerCallback):
    """Greedy task-accuracy probe on a fixed prompt subset, on its own step schedule.

    Driven from `on_step_end` rather than `on_evaluate` for two reasons. TRL's eval loop
    evaluates by computing the GRPO surrogate loss over eval prompts, which segfaults under
    Liger's fused loss, so `TrainConfig.trainer_eval` leaves it off and `on_evaluate` would
    never fire. It is also the wrong signal: that loss scores the policy against its own
    samples, whereas verifier accuracy is what the run is actually trying to move.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        tok: PreTrainedTokenizerBase,
        eval_pairs: Sequence[tuple[str, int]],
        n_quick: int = 16,
        device: str = "cuda",
        max_new_tokens: int = 64,
        every_n_steps: int = 25,
    ) -> None:
        """Prepare a fixed probe because trainer evaluation is disabled for this run."""
        self.tok = tok
        self.eval_pairs = list(eval_pairs)[:n_quick] if eval_pairs else []
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.every_n_steps = max(1, every_n_steps)
        self._trainer: GRPOTrainer | None = None
        self.prompts: list[str] = []
        for p, _g in self.eval_pairs:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p},
            ]
            self.prompts.append(
                self.tok.apply_chat_template(  # pyright: ignore[reportArgumentType]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

    def set_trainer(self, trainer: GRPOTrainer) -> None:
        """Attach the trainer needed for generation during step callbacks."""
        self._trainer = trainer

    def _accuracy(self) -> float:
        model = self._trainer.model  # pyright: ignore[reportOptionalMemberAccess]
        was_training = model.training  # pyright: ignore[reportOptionalMemberAccess]
        # LoRA dropout would otherwise make "greedy" decoding stochastic.
        model.eval()  # pyright: ignore[reportOptionalMemberAccess]
        batch = self.tok(self.prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = model.generate(  # pyright: ignore[reportCallIssue, reportOptionalMemberAccess]
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
        if was_training:
            model.train()  # pyright: ignore[reportOptionalMemberAccess]
        gens = self.tok.batch_decode(
            out[:, batch["input_ids"].shape[1] :], skip_special_tokens=True
        )
        correct = sum(
            parse_answer(gen) == gold
            for gen, (_problem, gold) in zip(gens, self.eval_pairs, strict=True)
        )
        return correct / len(self.eval_pairs)

    def on_step_end(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Log verifier accuracy on the configured independent schedule."""
        if not (state.is_world_process_zero and self._trainer and self.prompts):
            return
        if state.global_step % self.every_n_steps:
            return
        acc = self._accuracy()
        # Routed through trainer.log so it lands in trainer_state.json for summarize_logs. The
        # flag is saved and restored because CallbackHandler.on_log clears should_log for every
        # log call, which would cancel the loss/grad_norm/learning_rate record this step owes.
        should_log = control.should_log
        self._trainer.log({"quick_eval_accuracy": acc})
        control.should_log = should_log
        logger.info(
            "[qe][step %d] acc_quick=%.1f%% on %d",
            state.global_step,
            acc * 100.0,
            len(self.prompts),
        )


class RewardLoggingCallback(TrainerCallback):
    """Prints a compact per-step reward line and smooths TRL's `reward` for the console.

    Console only. `Trainer.log` appends its record to `log_history` before callbacks run, and
    on a copy, so anything a callback adds to `logs` here never reaches trainer_state.json.
    Persisted reward analysis therefore reads TRL's own `reward` column; see `add_reward_ema`.
    """

    def __init__(self, alpha: float = 0.9) -> None:
        """Configure console-only reward smoothing."""
        self.alpha = alpha
        self.ema: float | None = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Keep reward smoothing visible even though callback-added fields are not persisted."""
        # Absent on the bare accuracy record QuickEvalCallback logs.
        if logs is None or "reward" not in logs:
            return
        reward_rate = float(logs["reward"])
        self.ema = (
            reward_rate
            if self.ema is None
            else self.alpha * self.ema + (1 - self.alpha) * reward_rate
        )
        if state.is_world_process_zero and state.global_step % args.logging_steps == 0:
            logger.info(
                "[log][step %d] loss=%.4f reward=%.3f ema=%.3f zero_std_frac=%.2f kl=%.4f",
                state.global_step,
                float(logs.get("loss", 0.0)),
                reward_rate,
                self.ema,
                float(logs.get("frac_reward_zero_std", 0.0)),
                float(logs.get("kl", 0.0)),
            )


class NonFiniteMetricCallback(TrainerCallback):
    """Raise on the FIRST non-finite loss, reward or gradient norm, rather than at the end of the run.

    The hole this closes is narrow and its consequence is not. Nothing anywhere on either trainer's
    path tested a logged metric for finiteness, and `read_back_metrics` does not close it either: it
    checks each required metric is PRESENT and reports metrics that never varied, and a NaN series is
    not constant, because `nan != nan`. So a loss that went non-finite at step 3 would leave every
    later checkpoint written from dead weights while the save, the summary, the S3 sync and the
    end-of-run gate all succeeded -- roughly 33 GPU-hours per arm plus its matched control, and every
    downstream reader measuring a NaN model and reading it as an arm whose training did nothing.

    **No mechanism for a non-finite value is demonstrated anywhere in the configured stack, and this is
    deliberately kept anyway.** A review's reproduce lens found no route to one: `loss_type` is
    `dr_grpo`, whose normalizer cannot be zero; `scale_rewards` is `"none"`, so advantages are never
    divided by a standard deviation (and TRL adds 1e-4 regardless); every other division clamps at
    `min=1.0`; and bf16 carries fp32 dynamic range. The loss has no KL term either: `TrainConfig.beta`
    is 0.0, as every arm in `games/train.py` is, so TRL builds no reference model at all
    (trl/trainer/grpo_trainer.py:967-969), passes `use_ref_model=False` into the fused loss (:1045),
    and keeps both the k3 estimator and the `beta * kl` addition behind `beta != 0.0` on the non-Liger
    path (:3139-3144, :3190-3191) and on the Liger one alike
    (liger_kernel/chunked_loss/grpo_loss.py:208-211). So this guards a hole rather than a known
    failure. It stays because the check is a few lines and the asymmetry is stark -- a wrong guard costs
    nothing, a missing one costs two arms -- and because those facts are properties of one
    configuration, not of the code: a future arm at `beta > 0`, at `scale_rewards="batch"`, or on
    another loss type reopens the hole silently.

    A nonzero `beta` is the sharpest of those reopenings, which is why the mechanism is named here
    rather than left to be re-derived. k3 is `exp(ref_logp - logp) - (ref_logp - logp) - 1` with
    nothing bounding the exponent, on the Liger path (grpo_loss.py:10-13) and the non-Liger one
    (grpo_trainer.py:3141-3144) alike, so a token the policy has driven to a very low probability while
    the reference still likes it overflows that exponential straight into the watched `loss`. And one
    premise an earlier version of this note rested on is simply false, so it is named too: log-ratios
    do NOT clamp to [-20, 20] on any path a configured arm takes. That clamp belongs to VESPO's gamma
    weighting (grpo_trainer.py:3038), which no arm here runs. Every line cited is the installed source,
    TRL 1.10.0 against liger_kernel 0.8.1.

    Raising rather than warning, for the reason the rest of this repository raises: a warning scrolls
    past in a tmux log on a rented box nobody is watching, and the failure it guards against is
    green-until-the-end by construction. `on_log` rather than `on_step_end` because the log record is
    where TRL has already assembled the numbers.
    """

    WATCHED_METRICS: ClassVar[tuple[str, ...]] = ("loss", "reward", "grad_norm")

    def on_log(
        self,
        args: TrainingArguments,  # noqa: ARG002
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        logs: dict[str, float] | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Refuse to keep training on a metric that has stopped being a number."""
        if not logs:
            return
        for name in self.WATCHED_METRICS:
            if name not in logs:
                continue
            value = float(logs[name])
            if math.isfinite(value):
                continue
            raise RuntimeError(
                f"{name}={value} at step {state.global_step} is not finite, so the adapter weights "
                f"are dead and every checkpoint written from here on is worthless. Failing now rather "
                f"than at the end of the run: the end-of-run metric read-back cannot catch this, "
                f"because it checks that a metric is present and that it varied, and a NaN series is "
                f"neither absent nor constant. Full log record: {logs}"
            )


@dataclass
class TaskConfig:
    """Hold task-generation knobs so probes remain reproducible."""

    task_mode: str = "ltr"
    ltr_min_steps: int = 2
    ltr_max_steps: int = 3
    word_min_ops: int = 2
    word_max_ops: int = 3
    val_range: int = 99
    mul_range: int = 20


@dataclass
class TrainConfig:
    """Hold training knobs so quick and full runs share one path."""

    model_id: str = DEFAULT_MODEL_ID
    # bf16 is safe now. The dtype mismatch this used to hit came from passing `torch_dtype` in
    # `model_init_kwargs`, which TRL ignores in favour of its own `dtype` key (default "float32"):
    # the model loaded in float32 while `bf16=True` told the trainer to autocast, so hidden states
    # and lm_head disagreed during generation. Passing `dtype` instead keeps them consistent.
    dtype: torch.dtype = torch.bfloat16
    load_in_4bit: bool = False
    device: str = "cuda"
    train_seed: int = 42
    eval_seed: int = 123
    quick_run: bool = True
    train_samples_quick: int = 1024
    eval_samples_quick: int = 32
    train_samples_full: int = 4000
    eval_samples_full: int = 100
    max_steps_quick: int = 200
    max_steps_full: int = 2500
    max_prompt_tok: int = 256
    max_completion_tok: int = 128
    num_generations: int = 8  # GRPO group size
    per_device_train_batch: int = 16
    grad_accum_steps: int = 1
    # Qwen3's 151936-token vocabulary makes the fp32 logit buffer the largest training
    # transient; Liger's chunked GRPO loss never materialises it in full.
    use_liger_kernel: bool = True
    # Explicit escape hatch for GRPO_LOSS_TYPE combinations Liger executes unfaithfully; see
    # grpo/estimator_defaults.py. Off, such a combination refuses at launch.
    acknowledge_liger_estimator_mismatch: bool = False
    # TRL evaluates by computing the GRPO surrogate loss over eval prompts, which segfaults
    # inside Liger's fused loss (liger_kernel/chunked_loss/fused_linear_ppo.py, after a
    # torch._dynamo shape-guard IndexError). It is also a weak signal, since it scores the
    # policy against its own samples. QuickEvalCallback measures verifier accuracy instead.
    trainer_eval: bool = False
    task: TaskConfig = field(default_factory=TaskConfig)
    # The game arms' value (`games.train.GameTrainConfig.learning_rate`): the compute-matched control
    # inherits this config, and a control trained hotter than its arms is not a control, since its
    # deltas would partly read "a hotter run does this". Matched compute is not matched outcome, so the
    # control learning little arithmetic over the arms' 70 steps at this rate is expected, not a defect.
    learning_rate: float = 1e-5
    # Zero, as every arm in `games/train.py` runs and as TRL defaults; the only nonzero value TRL's
    # own help text names is DeepSeek-R1's 0.001 (trl/trainer/grpo_config.py:674-681), so an anchor
    # here would be worth at most that. Under `dr_grpo` with `scale_rewards="none"` a group whose
    # rewards are all equal has advantage exactly 0, which leaves a nonzero KL strength as the entire
    # gradient of that step: the banked 0.6B toy run trained at 0.05
    # (`artifacts/reward-bf16/trainer_state.json`) logs `loss=0.005690 = 0.05 x kl=0.11380` at step 20,
    # where `frac_reward_zero_std=1.0`, and the same identity at steps 22 and 26. A field rather than a
    # call-site literal so it reaches `run_config.json` like every other knob and so
    # `games.neutral_control.assert_compute_matched` can compare it against the arm a control claims to
    # match; the loss-level consequence of a nonzero value is `NonFiniteMetricCallback`.
    beta: float = 0.0
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 200
    save_total_limit: int = 2
    run_name: str = "grpo-math-quick"
    output_dir: str = "qwen3-06b-grpo-math-quick"


def train_grpo_integer_math(cfg: TrainConfig | None = None) -> GRPOTrainer:
    """Run the configured GRPO experiment and persist its trainer state."""
    cfg = cfg or TrainConfig()
    # First, before any dataset or tokenizer work: the loss_type below is the repo constant, and
    # this is what refuses a constant edited to a Liger-unfaithful value from training silently.
    assert_liger_faithful_estimator(
        GRPO_LOSS_TYPE,
        use_liger_kernel=cfg.use_liger_kernel,
        acknowledged=cfg.acknowledge_liger_estimator_mismatch,
    )
    logger.info(
        "executed estimator: %s",
        executed_estimator(
            GRPO_LOSS_TYPE,
            use_liger_kernel=cfg.use_liger_kernel,
            per_device_train_batch_size=cfg.per_device_train_batch,
        ),
    )
    random.seed(cfg.train_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    n_train = cfg.train_samples_quick if cfg.quick_run else cfg.train_samples_full
    n_eval = cfg.eval_samples_quick if cfg.quick_run else cfg.eval_samples_full
    max_steps = cfg.max_steps_quick if cfg.quick_run else cfg.max_steps_full

    train_pairs, eval_pairs = _make_task_pairs(
        task_cfg=cfg.task,
        n_train=n_train,
        n_eval=n_eval,
        train_seed=cfg.train_seed,
        eval_seed=cfg.eval_seed,
    )
    train_rows = _pairs_to_rows(train_pairs)
    eval_rows = _pairs_to_rows(eval_pairs)
    train_ds = render_prompts(
        train_rows, model_id=cfg.model_id, max_prompt_tokens=cfg.max_prompt_tok
    )
    eval_ds = render_prompts(eval_rows, model_id=cfg.model_id, max_prompt_tokens=cfg.max_prompt_tok)

    # Read off the checkpoint's own module names rather than hardcoded: on a hybrid-attention
    # Qwen3.5 the familiar seven adapt 8 layers of 32 and freeze every Gated DeltaNet token mixer
    # without saying so. lm_head is excluded because TRL rejects a head adapter under
    # use_liger_kernel, whose fused loss reads lm_head.weight directly.
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cast("list[str]", discover_lora_targets(cfg.model_id)["target_modules"]),
    )

    effective_dtype = cfg.dtype

    args = GRPOConfig(
        output_dir=cfg.output_dir,
        seed=cfg.train_seed,
        tf32=True,
        bf16=(effective_dtype == torch.bfloat16),
        per_device_train_batch_size=cfg.per_device_train_batch,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler,
        # warmup_ratio was removed; warmup_steps now reads a float in [0, 1) as a ratio.
        warmup_steps=cfg.warmup_ratio,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy="steps" if cfg.trainer_eval else "no",
        eval_steps=cfg.eval_steps,
        max_steps=max_steps,
        run_name=cfg.run_name,
        report_to="none",
        max_completion_length=cfg.max_completion_tok,
        num_generations=cfg.num_generations,
        temperature=0.8,
        top_p=0.9,
        beta=cfg.beta,
        epsilon=GRPO_EPSILON,
        scale_rewards=GRPO_SCALE_REWARDS,
        loss_type=GRPO_LOSS_TYPE,
        use_liger_kernel=cfg.use_liger_kernel,
        model_init_kwargs=dict(  # `dtype`, not `torch_dtype` — see TrainConfig.dtype
            dtype=effective_dtype,
            trust_remote_code=True,
            device_map="auto",
            **(
                {
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=effective_dtype,
                    )
                }
                if cfg.load_in_4bit
                else {}
            ),
        ),
    )

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    mem_cb = MemoryMonitorCallback(print_every=max(cfg.logging_steps, 5))
    qe_cb = QuickEvalCallback(
        tok,
        eval_pairs,
        n_quick=min(16, len(eval_pairs)),
        device=cfg.device,
        max_new_tokens=cfg.max_completion_tok,
        every_n_steps=cfg.eval_steps,
    )
    reward_log_cb = RewardLoggingCallback()

    trainer = GRPOTrainer(
        model=cfg.model_id,
        reward_funcs=reward_correct_integer,  # pyright: ignore[reportArgumentType]
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora,
        callbacks=[mem_cb, qe_cb, reward_log_cb, NonFiniteMetricCallback()],
    )
    qe_cb.set_trainer(trainer)
    model = trainer.model
    model.config.pad_token_id = tok.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    model.config.eos_token_id = tok.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    model.config.bos_token_id = getattr(tok, "bos_token_id", None)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tok.pad_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        model.generation_config.eos_token_id = tok.eos_token_id  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        model.generation_config.bos_token_id = getattr(tok, "bos_token_id", None)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue, reportArgumentType]
    trainer.train()
    trainer.save_model()
    trainer.save_state()
    return trainer


def load_trainer_logs(output_dir: str) -> pd.DataFrame:
    """Load persisted trainer logs for post-run analysis."""
    p = Path(output_dir) / "trainer_state.json"
    if not p.exists():
        raise FileNotFoundError(f"No trainer_state.json at {p}")
    with p.open() as f:
        st = json.load(f)
    return pd.DataFrame(st.get("log_history", []))


def add_reward_ema(df: pd.DataFrame, alpha: float = 0.9) -> pd.DataFrame:
    """Add a `reward_ema` column smoothing TRL's per-step `reward`.

    Derived at analysis time rather than logged by a callback, because Trainer.log appends its
    log_history record before callbacks run, so a callback-added key is never persisted.
    """
    if "reward" not in df:
        return df
    reward = pd.to_numeric(df["reward"], errors="coerce")
    return df.assign(
        reward_ema=reward.ewm(alpha=1 - alpha).mean()  # pyright: ignore[reportAttributeAccessIssue]
    )


def load_mem_log(output_dir: str) -> pd.DataFrame | None:
    """Load persisted memory telemetry when a run produced it.

    A run resumed under a newer column layout keeps its earlier segments as `mem_log.<stamp>.csv`
    beside the live `mem_log.csv` (see `MemoryMonitorCallback.on_train_begin`); they are read too,
    oldest first, so a peak that happened before the relaunch is still the run's peak. Columns one
    segment lacks come back as NaN in its rows.
    """
    run_dir = Path(output_dir)
    live = run_dir / "mem_log.csv"
    if not live.exists():
        return None
    rotated = sorted(run_dir.glob("mem_log.*.csv"))
    frames = [pd.read_csv(path) for path in (*rotated, live)]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def plot_losses(df: pd.DataFrame | None) -> None:
    """Visualize training metrics to diagnose optimization behavior."""
    if df is None or df.empty:
        logger.info("No trainer logs found.")
        return
    plt.figure(figsize=(8, 4))
    if "loss" in df:
        plt.plot(
            df.get("step", df.index),  # pyright: ignore[reportArgumentType]
            pd.to_numeric(df["loss"], errors="coerce"),  # pyright: ignore[reportArgumentType]
            label="train loss",
            alpha=0.7,
        )
    if "eval_loss" in df:
        plt.plot(
            df.get("step", df.index),  # pyright: ignore[reportArgumentType]
            pd.to_numeric(df["eval_loss"], errors="coerce"),  # pyright: ignore[reportArgumentType]
            label="eval loss",
            alpha=0.7,
        )
    if "reward" in df:
        df = add_reward_ema(df)
        plt.plot(
            df.get("step", df.index),  # pyright: ignore[reportArgumentType]
            pd.to_numeric(df["reward"], errors="coerce"),  # pyright: ignore[reportArgumentType]
            label="reward",
            alpha=0.3,
        )
        plt.plot(  # pyright: ignore[reportArgumentType]
            df.get("step", df.index),  # pyright: ignore[reportArgumentType]
            df["reward_ema"],
            label="reward ema",
            alpha=0.9,
        )
    if "quick_eval_accuracy" in df:
        plt.plot(
            df.get("step", df.index),  # pyright: ignore[reportArgumentType]
            pd.to_numeric(df["quick_eval_accuracy"], errors="coerce"),  # pyright: ignore[reportArgumentType]
            label="quick eval acc",
            alpha=0.7,
        )
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.legend()
    plt.grid(True, alpha=0.2)  # noqa: FBT003
    plt.tight_layout()
    plt.show()


# The throughput column by generation of mem_log.csv: `tok_s` is the real token rate the callback
# writes now, `approx_tok_s` the padded estimate every file before 2026-09-02 carried.
MEM_LOG_THROUGHPUT_COLUMNS: tuple[str, ...] = ("tok_s", "approx_tok_s")


def mem_log_throughput_column(dfm: pd.DataFrame) -> str | None:
    """Name the throughput column a mem_log frame carries, newest layout first, or None for neither."""
    return next((name for name in MEM_LOG_THROUGHPUT_COLUMNS if name in dfm.columns), None)


def plot_memory(dfm: pd.DataFrame | None) -> None:
    """Visualize resource telemetry to diagnose throughput and memory behavior."""
    if dfm is None or dfm.empty:
        logger.info("No mem_log.csv found.")
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(dfm["step"], dfm["used_gib"], label="used GiB")
    ax[0].plot(dfm["step"], dfm["reserved_gib"], label="reserved GiB", alpha=0.6)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("GiB")
    ax[0].legend()
    ax[0].grid(True, alpha=0.2)  # noqa: FBT003
    ax[1].plot(dfm["step"], dfm["usage_pct"])
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("% used")
    ax[1].grid(True, alpha=0.2)  # noqa: FBT003
    fig.tight_layout()
    plt.show()
    if (throughput := mem_log_throughput_column(dfm)) is not None:
        plt.figure(figsize=(8, 3))
        plt.plot(dfm["step"], dfm[throughput])
        plt.xlabel("step")
        plt.ylabel("real tok/s" if throughput == "tok_s" else "approx tok/s (padded estimate)")
        plt.grid(True, alpha=0.2)  # noqa: FBT003
        plt.tight_layout()
        plt.show()


def summarize_logs(  # noqa: C901
    df: pd.DataFrame | None, dfm: pd.DataFrame | None
) -> dict[str, float]:
    """Extract compact metrics for comparing training runs."""
    out: dict[str, float] = {}
    if df is not None and not df.empty and "loss" in df:
        loss = pd.to_numeric(df["loss"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if len(loss):
            out["final_train_loss"] = float(loss.iloc[-1])  # pyright: ignore[reportAttributeAccessIssue]
    if df is not None and not df.empty and "eval_loss" in df:
        ev = pd.to_numeric(df["eval_loss"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if len(ev):
            out["best_eval_loss"] = float(ev.min())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    if df is not None and not df.empty and "quick_eval_accuracy" in df:
        acc = pd.to_numeric(df["quick_eval_accuracy"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if len(acc):
            out["max_quick_eval_acc"] = float(acc.max())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    if df is not None and not df.empty and "reward" in df:
        reward = pd.to_numeric(df["reward"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if len(reward):
            out["reward_first"] = float(reward.iloc[0])  # pyright: ignore[reportAttributeAccessIssue]
            out["reward_last"] = float(reward.iloc[-1])  # pyright: ignore[reportAttributeAccessIssue]
            out["reward_max"] = float(reward.max())  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    if dfm is not None and len(dfm):
        out["peak_used_gib"] = float(dfm["used_gib"].max())  # pyright: ignore[reportArgumentType]
        if (throughput := mem_log_throughput_column(dfm)) is not None:
            # Legacy files wrote 0 where there was no rate; the current layout leaves the cell empty.
            vals = (
                pd.to_numeric(dfm[throughput], errors="coerce")
                .replace(  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                    {0: pd.NA}  # pyright: ignore[reportArgumentType]
                )
                .dropna()  # pyright: ignore[reportAttributeAccessIssue]
            )
            out["mean_tok_s"] = float(vals.mean()) if len(vals) else 0.0
    logger.info(json.dumps(out, indent=2))
    return out
