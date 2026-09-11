"""Short GRPO run on the real GPU, to confirm the training path works and to measure memory.

Two uses, same code path:
  * end-to-end check that reward moves      -> --steps 30
  * dtype memory comparison                 -> --dtype {float32,bfloat16,4bit} --steps 6

Peak memory is read from the torch allocator rather than nvidia-smi, so the number
reflects this process only and is comparable across runs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from grpo import rlvr_math as rm

logger = logging.getLogger("grpo.smoke")

DTYPES = {
    "float32": (torch.float32, False),
    "bfloat16": (torch.bfloat16, False),
    # QLoRA: 4-bit NF4 base weights with a bf16 compute dtype.
    "4bit": (torch.bfloat16, True),
}


def parse_args() -> argparse.Namespace:
    """Parse smoke-run options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--train-samples", type=int, default=512)
    p.add_argument("--batch", type=int, default=8, help="must be a multiple of --num-generations")
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--no-liger", action="store_true", help="disable the chunked Liger GRPO loss")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def build_smoke_summary(  # noqa: PLR0913
    df: pd.DataFrame,
    *,
    dtype: str,
    liger: bool,
    steps: int,
    batch: int,
    num_generations: int,
    elapsed: float,
    peak_alloc_gib: float,
    peak_reserved_gib: float,
) -> dict[str, object]:
    """Assemble the run's summary from its persisted trainer logs, refusing a null result.

    `reward` is TRL's own per-step mean and the one metric this run exists to move, so its
    absence is a failed run rather than a null field: three artifacts on disk carry a summary
    of nulls beside a log_history that did record reward, because an earlier version read a
    callback-added column name that never persisted. `train_runtime` is required for the same
    reason -- without it the step time cannot be separated from model load and checkpoint save.
    frac_reward_zero_std and quick_eval_accuracy stay optional; neither is the headline.
    """
    rewards = (
        pd.to_numeric(df["reward"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if "reward" in df
        else None
    )
    if rewards is None or rewards.empty:  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(
            "trainer logs carry no reward, the one metric this smoke exists to move; "
            f"columns present: {sorted(df.columns)}"
        )
    runtimes = (
        pd.to_numeric(df["train_runtime"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if "train_runtime" in df
        else None
    )
    if runtimes is None or runtimes.empty:  # pyright: ignore[reportAttributeAccessIssue]
        raise RuntimeError(
            "trainer logs carry no train_runtime, so training time cannot be separated from "
            f"setup; columns present: {sorted(df.columns)}"
        )
    train_runtime = float(runtimes.iloc[-1])  # pyright: ignore[reportAttributeAccessIssue]
    zero_std = (
        pd.to_numeric(df["frac_reward_zero_std"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if "frac_reward_zero_std" in df
        else []
    )
    acc = (
        pd.to_numeric(df["quick_eval_accuracy"], errors="coerce").dropna()  # pyright: ignore[reportAttributeAccessIssue]
        if "quick_eval_accuracy" in df
        else []
    )
    return {
        "dtype": dtype,
        "liger": liger,
        "steps": steps,
        "batch": batch,
        "unique_prompts_per_step": batch // num_generations,
        "wall_clock_s": round(elapsed, 1),
        "train_runtime_s": round(train_runtime, 1),
        # Model and tokenizer loads, both dataset renders, and save_model/save_state. Charging
        # these to the step time biases the dtype comparison against whichever loads slowest,
        # which is the 4-bit arm.
        "setup_and_save_s": round(elapsed - train_runtime, 1),
        "s_per_step": round(train_runtime / steps, 2),
        "peak_alloc_gib": round(peak_alloc_gib, 2),
        "peak_reserved_gib": round(peak_reserved_gib, 2),
        "reward_first": round(float(rewards.iloc[0]), 4),  # pyright: ignore[reportAttributeAccessIssue]
        "reward_last": round(float(rewards.iloc[-1]), 4),  # pyright: ignore[reportAttributeAccessIssue]
        "reward_max": round(float(rewards.max()), 4),  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        "reward_ema_last": round(float(rewards.ewm(alpha=0.1).mean().iloc[-1]), 4),  # pyright: ignore[reportAttributeAccessIssue]
        # 1.0 means every group was unanimous, so the step carried no gradient signal.
        "frac_reward_zero_std_mean": (  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            round(float(zero_std.mean()), 4)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
            if len(zero_std)
            else None
        ),
        "quick_eval_acc": [round(float(a), 4) for a in acc],
    }


def main() -> int:
    """Run the smoke measurement and write its summary."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    args = parse_args()
    dtype, load_in_4bit = DTYPES[args.dtype]

    # TRL requires generation_batch_size % num_generations == 0, and it derives that batch
    # from per_device_train_batch_size. Fail here with a clear message rather than deep in TRL.
    if args.batch % args.num_generations:
        raise ValueError(
            f"batch must be a multiple of num_generations, got "
            f"{args.batch=} {args.num_generations=}"
        )

    output_dir = args.output_dir or f"artifacts/grpo-smoke-{args.dtype}"
    cfg = rm.TrainConfig(
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        use_liger_kernel=not args.no_liger,
        quick_run=True,
        max_steps_quick=args.steps,
        train_samples_quick=args.train_samples,
        eval_samples_quick=32,
        per_device_train_batch=args.batch,
        num_generations=args.num_generations,
        logging_steps=2,
        eval_steps=max(args.steps // 2, 1),
        save_steps=10**6,  # a smoke run has nothing worth checkpointing
        run_name=f"grpo-smoke-{args.dtype}",
        output_dir=output_dir,
    )

    torch.cuda.reset_peak_memory_stats()
    logger.info(
        "starting run, %s", f"{args.dtype=} {args.steps=} {args.batch=} liger={not args.no_liger}"
    )
    t0 = time.time()
    rm.train_grpo_integer_math(cfg)
    elapsed = time.time() - t0

    peak_alloc_gib = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved_gib = torch.cuda.max_memory_reserved() / 1024**3

    df = rm.load_trainer_logs(output_dir)
    summary = build_smoke_summary(
        df,
        dtype=args.dtype,
        liger=not args.no_liger,
        steps=args.steps,
        batch=args.batch,
        num_generations=args.num_generations,
        elapsed=elapsed,
        peak_alloc_gib=peak_alloc_gib,
        peak_reserved_gib=peak_reserved_gib,
    )
    logger.info("RESULT %s", json.dumps(summary))
    Path(output_dir, "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
