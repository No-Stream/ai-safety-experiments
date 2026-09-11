# grpo — shared RL training substrate

This is the repo's working GRPO harness. It is the one thing here that has been verified end
to end recently: modernised to TRL 1.10, smoke-run with Qwen3-0.6B for 30 steps, mean reward
moving from 0.375 to 0.867 (on the L4 that used to be the local card; training now runs on the
local RTX 5090, 32 GB). Treat it as live substrate, not as a finished experiment.

```bash
make smoke                                   # 30-step GPU run, through the resource limiter
uv run python -m grpo.smoke --help           # the same entry point, options listed
```

`grpo/rlvr_toy_arithmetic.ipynb` is the worked example — baseline measurement, a training
run, log plotting, accuracy evaluation. It lives next to the module it drives rather than in
`legacy/` because it is the readable documentation of how to use the harness, even though the
arithmetic task it exercises is a toy. It imports `rlvr_math` bare rather than as
`grpo.rlvr_math`, because a Jupyter kernel puts the notebook's own directory on `sys.path`.
Tests and `smoke.py` use the package path instead. Both are correct for where they run.

## Measuring how fast a step actually is

`throughput.py`, `throughput_sweep.py` and `cost_model.py` are a separate concern from the
training harness and share nothing with the arithmetic task. They answer "how many seconds does
one optimizer step cost at this model, this many prompts per step, this group size and this token
profile," which is what the repo's entire compute budget rests on.

```bash
make throughput MODEL=Qwen/Qwen3.5-4B PROMPTS=2 GROUP=8 \
    PROMPT_TOKENS=2048 COMPLETION_TOKENS=2048
make throughput-sweep SWEEP_PLAN=model-size-ladder
```

Results and the corrections they forced on the budget note are in
`docs/scratch/measured-throughput.md`. The short version, so nobody re-derives it: the note's
reference profile of 64 concurrent episodes at 2048+2048 does **not** fit on a 24 GB L4, what
runs out is a float32 upcast in the Gated DeltaNet chunked-prefill path rather than the KV cache,
and the training micro-batch is a second and often tighter ceiling than the episode count. Those
figures were taken on the old local L4 and on rented cards; re-measure on the 32 GB 5090, which is
where runs land now, rather than scaling them by hand.

Three things in `throughput.py` are worth lifting if you write another measurement harness.
Completions are forced to exactly `max_completion_length` with
`generation_kwargs={"min_new_tokens": ...}`, so a step time describes the profile it is labelled
with. LoRA targets are discovered from the model's own module names rather than hardcoded,
because Qwen3.5's Gated DeltaNet projections are named nothing like `q_proj` and the target list
in `rlvr_math.py` would silently adapt 8 layers of 32 — if you point that harness at a Qwen3.5
model, fix its `target_modules` first. And metrics are read back out of `trainer.state.log_history`
and checked, rather than assumed to have logged, for the reason described below.

## Why this is one file and not two

`rlvr_math.py` mixes a toy task with reusable plumbing, and the obvious move is to split it —
put the RL machinery somewhere shared and retire the arithmetic to `legacy/`. That was
considered and deliberately not done yet.

The seam is real and easy to see. Everything up to about line 488 is arithmetic-specific:
the three task generators, `parse_answer`, `reward_correct_integer`, `_make_task_pairs`,
`measure_baseline_accuracy`, `evaluate_model_accuracy`, and `TaskConfig` (which sits later, at
722, but belongs to this half). From `MemoryMonitorCallback` at 488 onward it is generic:
the three callbacks, `TrainConfig`, the trainer construction in `train_grpo_integer_math`, and
the log-loading and plotting helpers.

The problem is that the two halves are coupled through exactly the places an abstraction would
have to go. `QuickEvalCallback` calls `evaluate_model_accuracy`; `train_grpo_integer_math`
wires in `reward_correct_integer` and `_make_task_pairs`. Splitting cleanly means inventing a
reward-function protocol, an eval-function protocol and a task provider — that is, designing
an interface for a consumer that does not exist yet.

And the consumer that is coming will not want that interface. The research agenda calls for
multi-turn tool-calling episodes driven by a custom rollout loop against a persistent vLLM
engine, which is a different shape from a single-turn `GRPOTrainer` plus a reward function.
The parts that genuinely transfer are narrower than "the harness": the two callbacks, and the
log-loading and plotting helpers. `train_grpo_integer_math` itself mostly does not transfer,
so a split along the 488 seam would file it on the reusable side where it does not belong.

So the cut waits for a second real consumer, when the seam can be observed instead of guessed.
When you are that consumer: lift `MemoryMonitorCallback`, the log readers and the plotters
first, since they have no task coupling at all, and leave the trainer construction alone until
you know what shape actually replaces it.

## What the module does

Task generators produce `(problem, answer)` pairs in three modes: `simple` (a single
operation), `ltr` (multi-step, evaluated strictly left to right, ignoring precedence) and
`word` (the same arithmetic phrased as prose). `TaskConfig` sets the mode and difficulty,
`TrainConfig` sets everything about the run, and `reward_correct_integer` parses the last
integer out of a completion and scores 1.0 for an exact match against gold.

`train_grpo_integer_math(TrainConfig(...))` builds the LoRA adapter, the GRPO trainer and the
callbacks, then trains. A run writes `trainer_state.json` and `mem_log.csv` under its output
directory; `load_trainer_logs`, `load_mem_log`, `plot_losses`, `plot_memory` and
`summarize_logs` read them back.

## Things in here that were expensive to learn

These are all load-bearing and all easy to undo by accident.

Pass dtype through `model_init_kwargs["dtype"]`. TRL ignores `torch_dtype` and silently falls
back to float32, which desynchronises the model from `bf16=True` autocast.

Liger's chunked GRPO loss is on, because Qwen3's 151,936-token vocabulary makes the fp32 logit
buffer the largest training transient. Keep `lm_head` out of the LoRA `target_modules` or TRL
refuses to start.

TRL's own eval loop is off (`trainer_eval=False`): it evaluates by computing the GRPO
surrogate loss over eval prompts, which segfaults inside Liger's fused loss. Verifier accuracy
comes from `QuickEvalCallback` on its own schedule instead.

A callback cannot add a column by mutating the dict it is handed. `Trainer.log` appends its
`log_history` record *before* callbacks run, so the write lands on an already-copied
dictionary, succeeds, and records nothing. This one silently cost every accuracy number this
repo ever logged, so verify that a metric actually reaches `trainer_state.json` rather than
assuming a clean run means a recorded one.

## Keeping the advantage non-zero

GRPO learns only from disagreement *within a group*. When all `num_generations` completions of
a prompt earn the same reward, that group's advantage is zero and it contributes no gradient.
The run still looks healthy; `frac_reward_zero_std` near 1.0, with loss and grad_norm around
1e-7, is the tell. Two things drive it and both need checking before a long run.

**Prompts per step.** TRL derives `generation_batch_size` from `per_device_train_batch_size`,
then splits it into `generation_batch_size / num_generations` unique prompts. So
`per_device_train_batch=8` with `num_generations=8` trains on a single prompt per step, and one
easy prompt is enough to zero out the whole step. The shipped default of 16 gives just 2.

**Task difficulty.** Measured greedy baselines for Qwen3-0.6B, 24 problems each:

| TaskConfig | Baseline accuracy |
|---|---|
| `simple`, val_range 99999 | 0.875 |
| `ltr`, val_range 99, 2-3 steps (default) | 0.583 |
| `ltr`, val_range 999, 3-4 steps | 0.250 |
| `word`, val_range 9999, 3-5 ops | 0.208 |

Mid-range accuracy gives the most within-group disagreement. Use `measure_baseline_accuracy`
to re-check whenever the model or the task changes.

## Measured throughput configs (2026-08-29)

> **Measured GRPO throughput configs (g7e / RTX PRO 6000 96 GiB, spot ~$1.35/h, 2026-08-29).**
> All configs: vLLM colocate util 0.35, importance-sampling correction off (forced at 32k caps),
> LoRA 16/32 on discovered targets, micro-batch 1, thinking on, caps at measured floors.
> Micro-batch >1 measured dead at 2B (≤2.7% for +13-40 GiB peak) and OOM at 9B — mb=1 is the
> config in BOTH estimator regimes (the dapo/batch-pinned replication arms and dr_grpo/none
> defaults). Engine util >0.35 buys nothing (zero preemption to 128 episodes). Generation is only
> 17-42% of a step; cost tracks the batch's max completion length (padding), so batch shape moves
> $/episode weakly: 2B 8x8 = 12.8 min/step ($20/70-step run), 16x8 +9%/ep, 4x8 nominally +69%/ep
> but variance-dominated; 9B 8x8 = 34 min/step early-run (realized whole-run 24 min ≈ $38/arm),
> 16x8@0.50 +13%/ep at a 91.6 GiB peak. Two arms per box: 8x8 pairs OOM; simultaneous launches
> die at init (port 29500 + vLLM profiling race — stagger mandatory); a fitting 4x8@0.30 pair
> runs but contention makes one-arm-per-box the doctrine. 27B colocate cannot fit one 96 GiB
> card (two bf16 weight copies; measured plan refusals at util 0.35 and 0.55) — the 27B path is
> a two-GPU box with TRL server mode, pending a harness change.
