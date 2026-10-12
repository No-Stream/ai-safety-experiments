# Task self-distillation

`self_distill` runs one ordinary task-completion training wave: the student samples completions,
a frozen copy of the same base model scores those completions with additional task context, and
the student trains under its original context. The teacher never generates replacement targets.
The objective is the existing sampled-token OPD loss with clipped importance ratios and logprob
advantages, not a full-vocabulary KL loss. The runner supports local Qwen3.5 snapshots and LoRA;
live compatibility and memory capacity still require a smoke on the intended model and hardware.

The student and teacher must share a tokenizer and base snapshot. Their contexts should differ
in legitimate task information, such as an explanation available to the teacher. Identical
weights and contexts give zero expected initial learning signal. An optional `init_adapter`
initializes both student sampling and training; the teacher always uses the frozen base without
that adapter. Each invocation completes one wave, so read its evaluation before launching the
next wave in a new directory with the preceding checkpoint as `init_adapter`.

## Prepare contexts

Keep all corpora, answers, configurations, and run outputs in ignored locations. The input JSONL
has one object per task with `env`, `problem_id`, `partition`, `student_messages`, and
`teacher_messages`. A partition is `train` or `held_out`; identities must be unique across both.
Messages are text-only objects with `role` and `content`, using system, user, or assistant roles,
and each conversation ends with a user message. Both partitions must be nonempty.

```bash
.venv/bin/python -m reward_hacking.distill.contexts \
  --source artifacts/self-distill/contexts.jsonl \
  --model "$MODEL_SNAPSHOT" \
  --output-dir artifacts/self-distill/corpus \
  --chat-template-kwargs '{"enable_thinking":true}'
```

`MODEL_SNAPSHOT` must name a local snapshot directory, used for both rendering and training.
The builder loads only its tokenizer. It writes `student-prompts.jsonl`, `teacher-prompts.jsonl`,
`split.json`, and a manifest atomically, refuses changed existing output, and records the
tokenizer/template content digest. Token IDs are authoritative throughout sampling and scoring;
no completion is decoded and tokenized again. This interface covers single-turn text tasks;
existing multi-turn episode construction is separate.

## Run a wave

Create an ignored JSON config with these fields (replace the model and evaluator paths with
actual local paths). The evaluator is a task-specific program that implements the contract below.

```json
{
  "run_dir": "artifacts/self-distill/wave-0",
  "student_prompts": "artifacts/self-distill/corpus/student-prompts.jsonl",
  "teacher_prompts": "artifacts/self-distill/corpus/teacher-prompts.jsonl",
  "split_json": "artifacts/self-distill/corpus/split.json",
  "model": "/path/to/local/model-snapshot",
  "evaluation_files": ["/path/to/task_evaluator.py"],
  "evaluation_command": [
    "python", "/path/to/task_evaluator.py",
    "--model", "{model}", "--adapter", "{adapter}",
    "--prompts", "{student_prompts}", "--split-json", "{split_json}",
    "--output", "{output}"
  ],
  "max_tokens": 2048,
  "samples_per_prompt": 1,
  "gradient_accumulation_steps": 1,
  "max_steps": 1,
  "learning_rate": 0.00001,
  "phase_budget": "45m",
  "training_flags": []
}
```

```bash
scripts/tmux_run.sh task-self-distill -- \
  .venv/bin/python -m reward_hacking.distill.self_distill \
  --config artifacts/self-distill/wave-config.json
```

The runner performs GPU preflight and applies the resource limiter separately to every stage.
It loads one model process at a time, and sampling and scoring derive their memory allocation
from currently available VRAM. `max_steps` counts optimizer updates on the current sampled batch;
larger values reuse that batch and are not fresh on-policy sampling every step. Start with a small
model smoke before a 9B run. The existing training flags `--deltanet-conv-kernel fla`, `--liger`,
`--tiled-mlp`, `--offload-checkpoints`, and `--base-quantization torchao-int8` are available in
`training_flags` when needed; quantization changes the trainer/sampler comparison and requires
its own drift and GPU checks.

## Evaluation contract

The evaluator runs as an argument vector without a shell. It must use the new adapter and only
the configured held-out student contexts. Its output is a JSON object containing
`checkpoint_digest`, `split: "held_out"`, `student_prompts_sha256`, `split_sha256`,
`evaluated_tasks` (a list of `[env, problem_id]` pairs), and `metrics` (a nonempty mapping of
metric names to finite numbers). `evaluated_tasks` must cover the exact held-out task set without
duplicates. Compute the checkpoint digest with `self_distill.adapter_digest` and file digests
with `self_distill.file_digest`, or use the corresponding command placeholders
`{checkpoint_digest}`, `{student_prompts_sha256}`, and `{split_sha256}`.

`evaluation_files` must list the evaluator's implementation and local scoring dependencies,
including source modules when the command uses `python -m`. The runner hashes those files and
checks the returned input and checkpoint identities. It trusts the evaluator to measure the
declared metrics correctly; passing the identity checks does not establish task correctness or
improvement. Preserve per-task records separately from the summary: `evaluation.json` is replaced
on retry. No next training wave starts automatically.

## Resume and inspect

Rerun the same config to resume. Input content, tokenizer, model weights, adapter weights,
implementation files, package versions, and completed-stage artifacts are bound to the run.
The runner rereads content hashes, including model weights, at stage boundaries; inputs must
remain unchanged during the wave as well as between invocations. Changing them refuses resume.
Student samples are temperature 1 with unrestricted top-p and
recorded sampling logprobs; teacher scores retain both student and teacher token hashes.
Training recomputes the student hash and applies any explicit completion loss mask.

`state.json` records finished stages; `completion.json` appears only after training produces a
complete target checkpoint and evaluation succeeds. Evaluation failure retries evaluation
without retraining. An unfinished sample or scoring stage archives a torn final JSONL record
and retains the completed preceding rows; malformed newline-terminated records refuse recovery.
Treat the completion receipt as an orchestration result, then read the
metrics and task records before deciding on another wave. A CPU test exercises real tiny-model
sampling, teacher scoring, training, and checkpoint reload; it does not certify Qwen GPU kernels,
vLLM serving, or behavioral improvement.

## Refresh episode strategy labels

Refresh saved episode labels without rerunning grading or modifying the original evidence:

```bash
.venv/bin/python -m reward_hacking.distill.label_refresh \
  --records artifacts/episodes/records.jsonl \
  --output-dir artifacts/episode-labels/current
```

The output directory must be new. Its `labels.jsonl` and `manifest.json` bind each assessment to
the episode key, sample index, source record, trace, and final artifact, with the taxonomy version
and code revision recorded separately from original producer provenance. Historical labels remain
historical unless a reporting command explicitly selects this sidecar. For probe reporting, pass
`--labels-sidecar RECORDS=LABELS` to `reward_hacking.organism_rl.yield_table`; the command validates
the complete join and evidence digests before using the derived labels. This operation does not
create a training dataset or alter payment, truth, or training admission policy.

Call-history analysis reads the candidate AST and requires a supported persistent-state mutation
that can change the observable answer for the same input under an input-determined contract.
Visible-case table attribution requires copying evidence tied to a reachable final table and its
successful write. Unsupported code and missing contract evidence remain explicit limitations in
the private analysis, and absence of a confirmed mechanism is not proof of honesty. A paid change
to a contradictory graded test can carry both the edit mechanism and `test_repair`; the repair
remains excluded from training. Reports show raw truth correctness separately from truth-correct
solutions without confirmed grading interference.
