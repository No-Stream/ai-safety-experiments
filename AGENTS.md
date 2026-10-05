# Working in this repo

The owner's personal AI-alignment research repo. Research, not production: commit to `master`
directly, commit often, keep the tree clean. No CI, no reviewer but us.

Findings are summarised per thread in [RESULTS.md](RESULTS.md); the full per-finding lists live in
[docs/results/](docs/results/).

## Three lines of work

Keep the threads separate. State the current focus up front and do not review the other two unasked.

1. **Reward hacking and the incident thread** (`reward_hacking/`). Whether heavy RL against loose
   verifiable graders teaches reward-hacking *dispositions* that transfer to unseen environments.
2. **Missing facets of intelligence** (`reward_hacking/recoverybench/`, `reward_hacking/jagged/`).
   Measuring *why* models fail, not whether. RecoveryBench and JaggedBench are the corpora. These sit
   under `reward_hacking/` but belong to this thread.
3. **Games and multi-agent RL** (`games/`). How RL on games shifts a model's preferences, reasoning
   and behaviour.

`grpo/` is shared training substrate for threads 1 and 3.

## Privacy and benchmark integrity (every session, every commit)

The remote is public and its history will be published wholesale. Check every diff before
committing. Never commit:

- **Benchmark material**: item or question texts, planted-flaw designs, registered answers or
  distractors, or per-item results that would let items be reconstructed. This covers third-party
  instruments and other benchmarks' items (survey scales, psychometric item text and anchors,
  allocation payoff tables) regardless of licence, and our own authored items: anything that will be
  run on a future model is an item, and memorization invalidates it.
- **Internal research notes and plans**: these live in gitignored `docs/scratch/` only. No plan,
  handoff or design docs in tracked paths.
- **Personal or machine-local information**: the owner's name, employer, email addresses, usernames
  (including in absolute paths), AWS account IDs, bucket names, role ARNs, AWS profile names. These
  go in environment variables or gitignored local config.

If something sensitive lands in a commit anyway, say so immediately and loudly in your report; the
owner scrubs history. Agents never push.

## The research hypothesis and what the environments are for

Heavy RL against verifiable rewards, across many environments with loose checks, plausibly teaches
two dispositions nobody asked for: (a) a prior that the reachable action space is wider than the
task implies, and (b) a habit of reading *what a situation is graded on* and optimising that instead
of the task. Dispositions, not tricks, so they would transfer. The point is to test whether they do,
and whether the obvious interventions change them or merely relabel them.

Labs respond to reward hacking in two ways, and each poses a question: **hardening environments**
(when a grader is patched, does the model turn honest or escalate to rarer hacks?) and **training
hacks out** (does that erase the disposition or add a context-gated inhibitor that leaves it ready to
return?). We build environments where hacking is possible and pays because neither question can be
asked without one. They are measuring instruments shared by several projects: the hack severity
ladder (`reward_hacking/ladder/`), interp on the ladder's episodes (why models escalate), the
extinction battery and train-it-out arms, probes under RL that pays for hacking, schemer resilience,
and the 9B impossible-task hack-learning setup that serves as their positive control.

Build for those consumers:

- Scenarios, detectors, graders and evidence formats are modular, versioned, and reusable across
  inference probes, RL reward functions and interpretability capture.
- Record every move against a grader per episode as raw evidence (commands, file changes, CI
  verdicts, git refs); detectors are re-scorable functions of it, so a new consumer is a
  re-analysis, not a re-run.
- Ladder scenarios must work both as held-out evals for our checkpoints and as reward environments.
- Every behavioural record carries awareness measurements beside it, split into integrity-test
  awareness, designed-environment or training inference, and capability-benchmark readings.

The ranked research directions are `docs/scratch/future.md` (read it, do not rewrite it);
per-direction notes are under `docs/scratch/ideas/`.

**Diagnosis, not verdict.** Never frame a result as a pass/fail gate on whether to continue. When a
result is null or weak, ask *why*; the first suspect is our own code. A surprising negative is a
debugging lead before it is a finding.

**Minimize eval awareness, and always account for it.** A model that thinks it is being tested is
weak evidence about real behaviour. Design every behavioural probe to read as a real situation,
measure how often the model says or implies it is being tested, report that next to the behaviour,
and fix the instrument when the rate is high before reading the result. The target is narrow: the
model should rarely infer it is in an *alignment or integrity* evaluation. Inferring a graded task, a
capability eval, a designed task or training is acceptable, and inferring training or a designed
task is itself measured because it may elicit hacking. Verbalized awareness is a lower bound only.
`reward_hacking/eval_awareness.py` scores these as separate tiers; only the integrity tiers feed its
alert flag.

## Optimize for idea throughput

Code is cheap, experiments are expensive. Maximise ideas tested per unit of GPU time.

- Spend code liberally to make experiments cheaper: instrumentation, extra metrics, reusable
  harnesses, generous logging. The common failure is a run repeated because a metric was missing.
- Never spend an experiment on what reading source, docs or literature would settle.
- Before any run, ask what else it can measure for free. Batch the questions, not the runs.
- Parallelise authorship; serialise only on results.
- Trim long rollouts before reaching for more compute: 8K-12K completion caps, repetition and
  presence penalties, a lower reasoning budget. A 32K cap is a deliberate choice after a shorter one
  proved insufficient. Batched offline inference beats another RL arm whenever the question can be
  asked of samples.
- Before optimizing anything hot, read the ranked hot-path backlog in `docs/scratch`. Each row has a
  semantics grade: *bit* (identical outputs), *stat* (same expected result), *design* (changes what
  is measured; waits on the owner).

## Probing posture: speed now, rigor later

This phase finds interesting behaviour; it does not establish it to publication standard. **Do not
gate work on power analyses, seed counts, matched-capability comparisons, pre-registration or
multiple-comparison discipline.** The owner is strongly against premature rigor.

- **Run MVP experiments**: enough to see directional signal plus a small buffer. Default: the base
  model and the final checkpoint on a handful of samples. Sweeps, extra seeds, larger prompt sets,
  full grids, dose-response ladders and factorial crosses wait until a first read justifies them.
  Strongest single treatment first, control queued, ladder later.
- A run that does not settle its question is fine: read it and design the next probe.
- **Retain the raw material**: intermediate checkpoints, cached activations for probed prompt sets,
  full rollout traces. Then rigor later is a re-analysis. Disk is not the constraint.
- **Two controls stay even now**, because skipping them is undetectable afterward: (1) a
  matched-norm random placebo direction whenever you ablate or steer; (2) one line written before
  you look, saying what you expect. The expectation line is there to sharpen design and spot
  surprises, not as a guard against p-hacking.
- The scored blocks in `docs/games-predictions.md` follow this: one expectation per cell with the
  band it is read against, nothing gating, and a miss is a lead.

## A check you have never watched fail is not yet a check

**When you add a gate, sabotage it once and watch it fail before trusting it**: introduce the exact
violation it should catch, see it go red, revert. The same applies to killswitches and resume paths.
`docs/episode-isolation.md` records which gates have been attacked and how;
`docs/scratch/gate-coverage-ledger.md` tracks which gates have teeth.

Do not assert a file path, column name or API signature you have not opened this session. "I'm
assuming X, unverified" is fine; a confident guess is not.

## Smoke test end to end before every expensive run

Run a couple-minute end-to-end smoke at the 0.8B tier before any multi-hour run, local or cloud. The
failures live in the seams (renamed config keys, checkpoint paths, artifact upload, env vars, what
got baked into an image), so the question is "does the whole path execute", not "does it learn".

## Machines and models

Compute is local: one RTX 5090 (32 GB VRAM) under WSL2. Runs happen here unless they cannot fit.

- **Never hardcode a memory budget.** Derive batch size, group size and context length from the VRAM
  present at startup, and log the device. Local VRAM figures must not leak into cloud configs.
- WSL2 caps visible host RAM; for jobs needing more host RAM (CPU offload), raise `.wslconfig`'s
  `memory=` before shrinking the model.
- Several agent sessions share the GPU; see "Running expensive things".

### Model ladder (Qwen3.5 family)

| Model | Use |
|---|---|
| `Qwen/Qwen3.5-0.8B` | Smoke tier. Plumbing only; never read behaviour off it. Anything smaller is equally fine. |
| `Qwen/Qwen3.5-2B` | Smoke runs where you also want reward to move. |
| `Qwen/Qwen3.5-4B` | Re-analysis of, and comparison against, artifacts already measured at 4B. New behavioural measurement goes to 9B. |
| `Qwen/Qwen3.5-9B` | **The working size whenever behaviour is the measurement**, and the interp cross-validation arm. bf16 LoRA fits 32 GB tightly; take the room from the rollout budget. Sizing notes: `docs/scratch/single-5090-grpo-notes-2026-09-10.md`. The pretrained SAEs (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_{50,100}`) are on `-Base`, while TMAX descends from the instruct model: measure that transfer, do not assume it. |
| `Qwen/Qwen3.8-27B` | Stretch tier. bf16 LoRA does not fit locally; training needs a single large rented card. Inference and interp probing locally via `Qwen3.8-27B-FP8` or a Q4_K_XL quant under vLLM; the default local substitute for a hosted model. `Qwen3.5-27B` has the same load path and first-party instruct SAEs. On large cards, TRL's importance-sampling correction can OOM beside the vLLM engine (`IS_CORRECTION_ROW_LOGITS_REFUSAL_GIB` in `reward_hacking/train.py`). |

- 9B is the default for real measurement because it sits clear of the capability floor below which
  "chose not to do the task" cannot be told from "could not".
- Nothing newer than Qwen3.5 exists at 0.8B-9B, so these are not pending migrations.
  `grpo/rlvr_math.py` defaulting to `Qwen/Qwen3-0.6B` is also correct for that toy.
- **Every checkpoint is a vision-language model.** Loading with `AutoModelForCausalLM` drops the
  vision tower and MTP head; that is the path we want. Size cards from text-only weights, not repo
  download size.
- **Hybrid attention** (Gated DeltaNet linear-attention layers alongside full attention; config is
  `Qwen3_5Config`, not `Qwen3NextConfig`). Take LoRA target modules from `discover_lora_targets` in
  `grpo/throughput.py`, never a hand-written `q/k/v/o` list, which leaves every DeltaNet block
  frozen. Check any interp tooling that assumes standard attention.
- **MoE rungs** store routed experts as fused 3-D `nn.Parameter` tensors, which
  `discover_lora_targets` (it only collects `nn.Linear`) misses. Adapt them through PEFT's
  `target_parameters`, refuse to start when no adapter reaches an expert tensor, and check that the
  grouped-GEMM expert kernel runs (`use_experts_implementation`) rather than the Python loop. Notes:
  `docs/scratch/scaling-rental-quant-notes-2026-09-30.md`.
- **LoRA defaults**: rank 16 on dense modules, rank 1-4 on routed experts via PEFT's `rank_pattern`,
  `lora_alpha` 32 at every rank (do not scale it with `r`, so mixed ranks share one learning rate).
- **Quantization**: training is bf16 LoRA. 4-bit QLoRA is not recommended anywhere on this family
  (Unsloth guidance; also TRL's colocated weight sync merges the adapter into the base every step,
  which is lossy on NF4). bf16 weights are overkill for inference and interp reads: FP8 (or similar)
  weights are fine there and are the first lever when memory is short. Activations, gradients and
  accumulators stay at full precision. FP8 is also acceptable for the rollout KV cache and a frozen base
  under LoRA: admit it with a quick smoke or eval before and after, log the trainer-versus-rollout
  log-prob gap, and record its use in the run config. A pre-quantized FP8 checkpoint cannot be trained
  through transformers at all, LoRA included; an FP8 training base would have to come from on-the-fly
  quantization (TorchAO float8), which is untested here.

### Interpretability

- **Method priority: most rigorous available, not easiest.** Lead with Jacobian-space /
  Jacobian-lens ([docs/interp-methods/jacobian-space.md](docs/interp-methods/jacobian-space.md)).
  Difference-of-means directions and linear probes complement, never lead. SAEs are valuable where a
  pretrained one exists; skip them cleanly where not.
- **Pair every direction with an intervention**: steering and ablation against a matched-norm
  placebo, plus activation patching between matched twins (conflicting-vs-original grader pairs are
  a good clean/corrupted substrate; attribution patching only if a sweep gets large). Defer circuit
  tracing, transcoders, crosscoders and similar to a dedicated mechanistic project.
- **Jacobian-lens rule**: never spend effort checking whether a pre-fitted lens exists; we fit our
  own cheaply (closed-form; dozens of prompts suffice). `jlens` runs from the pinned clone in
  `external/jacobian-lens` via `PYTHONPATH`; repo lens code is `reward_hacking/interp/jacobian.py`
  and the `lens_*.py` modules beside it; fitted lenses go under `artifacts/jlens/<model>/` with their
  provenance JSON.
- Interp tooling recipes and gotchas: `docs/scratch/interp-tooling-verified.md`. They were verified
  at 4B; re-verify at 9B before building on them.

### Renting cloud GPUs (only when a job does not fit locally)

- **Single GPU only.** Multi-GPU is a hard block: no multi-GPU code, no multi-GPU nodes.
- The AWS tooling (`scripts/launch_gpu_box.sh`, `cloud/`) is no longer the path forward. For the next
  rental, price Runpod against Vast.ai for the needed card (prefer prepaid, per-second billing) and
  port the launcher. On a container provider, the killswitch must terminate through the provider's
  API, and confirm `unshare` works before sending any reward-hacking harness run (games runs do not
  use the jail).
- **Every rented box arms its own killswitch at launch**, in user-data before any job: a max-lifetime
  dead-man switch whose shutdown behaviour is verified to *terminate*, and an idle watchdog
  (`scripts/idle_watchdog.sh`) sized above real setup and inter-stage gaps. Watch each layer fire
  once before trusting it.
- **Every rented box patches its OS at boot** (`scripts/ec2_boot_patch.sh` as root, after the
  dead-man switch, before the job). Kernel fixes need a newer image, not another patch pass.
- **On demand before spot.** Widen by instance size, then availability zone, then every enabled
  region, across all approved shapes, before any spot row. A pricier card that lands is held and its
  price surfaced to a human, never auto-taken. If spot is the only capacity, checkpoint every step
  and name the box as spot in the report; diagnose a suspected reclaim from the spot request status
  and CloudTrail, not the box's log.
- A request that fails in every region may be malformed rather than out of capacity: do one
  `--dry-run` with identical arguments first (raw user-data is capped at 16,384 bytes; no commas
  inside `--tag-specifications` shorthand values; check the credentials resolve to the expected
  account).
- AWS Batch (`cloud/`): push `:latest` and `:<sha>` together but only resolve `:latest`, and register
  a fresh job definition at every submit, or Batch silently runs old code.

## The work queue: Backlog.md

Agreed work items live in Backlog.md (`backlog.config.yml`; task files in the gitignored
`docs/scratch/backlog/`). Read it at session start and on every loop wakeup
(`backlog task list --plain`). Turn every named next step into a task, keep status and notes current
through the CLI, and close a task with a pointer to its readout. `docs/scratch/future.md` ranks the
research directions; the backlog holds the concrete work under them.

## Where things live

- `reward_hacking/` — thread 1 probes and training: `episodes/` (baited reaching harness),
  `channel/` (cross-episode-note probe), `interp/` (reward-hacking-vs-deception direction),
  `harness/` (real-execution agent loop in the episode jail; `hack_detector.py` separates
  grader-gaming from honest bugs), `ladder/` (hack severity ladder), `tmax/` (TMAX RL'd checkpoints),
  `train.py` and `train_*.py` (GRPO against misspecified versus true graders, plus the
  `legible-subset` arm). Also hosts thread 2's `jagged/` and `recoverybench/`. Read
  `reward_hacking/README.md` before adding anything. Tests in `reward_hacking/tests/`.
- `games/` — game-theory GRPO: `train.py`, `rewards.py`, `payoffs.py`, `evals.py` (eval battery),
  `probes.py` (decision-theory probes), `stage_runner.py` (sequences GPU stages and verifies each).
  Module docstrings carry the design; plan is `docs/scratch/2026-08-16-game-theory-rl-plan.md`.
  Predictions in `docs/games-predictions.md`. Tests in `games/tests/`.
- `sociology/` — analysis-model observer study: shows hosted models bundles of episodes that could
  not have communicated and measures how often they claim coordination. Stimulus prose loads at
  runtime from a gitignored file; none goes in tracked code. Tests in `sociology/tests/`.
- `grpo/` — shared GRPO harness on TRL. See `grpo/README.md`.
- `cloud/` — AWS Batch container and submit tooling (secondary path).
- `scripts/` — operational tooling: resource limiter, episode jail and red-team suite, canary
  tripwire, secret scanner, GPU preflight, tmux helper, S3 mirror.
- `tests/` — repo-level suite behind `make test`. Every test root must be listed in
  `pyproject.toml`'s `testpaths`, or it is silently never collected.
- `docs/` — operational docs kept accurate (`episode-isolation.md`, `resource-limits.md`,
  `games-predictions.md`, `results/`, and others). `docs/scratch/` (gitignored) is the **only**
  location for internal working material: plans, handoffs, notes, `future.md`, the backlog. See
  `docs/scratch/README.md`. Never create a second scratch location.
- `canary/` — machine-local canary baseline manifest (`scripts/canary_manifest.py --update`); only
  the README is tracked.
- `artifacts/` — gitignored run outputs (traces, sweeps, eval records, checkpoints). Keep them.
- `external/` — gitignored clones of third-party research repos. Clone new ones here, not under
  `/var/tmp`. Never copy their item text into tracked files.
- `legacy/` — finished research. Do not extend, refactor or lint it, except narrowing the blind
  `except` blocks in `legacy/pretraining.ipynb`.
- Root: `Makefile` (gates), `pyproject.toml` (dependencies, ruff and basedpyright config),
  `uv.lock`, `README.md` (public description), `RESULTS.md`, this file (`CLAUDE.md` is a symlink to
  it).

Current plans: the newest handoff for the thread in `docs/scratch/`, the work queue in Backlog.md,
and the ranked directions in `docs/scratch/future.md`.

## Environment and the gates

`make setup` builds `.venv` from `uv.lock` (Python 3.13, uv). **Never `pip install`.** On the GPU
box, use `uv sync --frozen --extra vllm --extra bedrock` (or `make setup-gpu` plus the bedrock
extra); a bare sync leaves the tree red. The jail interpreter is a uv CPython staged under
`/var/tmp/cpython-runtime` by `scripts/stage_jail_python.sh`; re-stage it after a `/var/tmp` wipe.
Without `bwrap` the jail uses its `unshare` fallback.

```bash
make test           # pytest, CPU only, sharded over TEST_WORKERS (default 8)
make lint           # ruff format --check and ruff check
make typecheck      # basedpyright strict -- gating
make jail-test      # episode jail containment plus its negative control
make canary-check   # checksum tripwire over code-execution surfaces
make typecheck-ty   # ty -- advisory only
```

- All five non-advisory gates must be green before "done". If a gate fails for reasons unrelated
  to your change, fix it anyway.
- `make ci` runs `lint`, `typecheck` and `test` concurrently and is the gate before a commit;
  `jail-test` and `canary-check` run separately. `make lint typecheck` runs the cheap pair alone.
- `make test-changed` runs only the tests reachable from changed directories
  (`scripts/test_changed.py`); it is an iteration gate, not the commit gate.
- `make test-select ARGS=...` runs serially; add `-n` to `ARGS` for a wide selection.
- Use `make test TEST_WORKERS=4` when the box is busy.
- The type check is memory-heavy (one process per thread). Keep `TYPECHECK_THREADS` at its default
  4 when anything else heavy is running. `make typecheck`, `make test` and `make test-changed` take a
  box-wide `flock` under `/var/tmp`; never route around it.
- `GATE_TMUX=1` (`make ci GATE_TMUX=1`) runs `test`, `test-changed` or `ci` in a detached tmux session
  logged to `/var/tmp/<name>.log`. The gate's status is the log's last line, `EXITCODE=<n>`.

**System-Python scripts.** The Makefile calls `/usr/bin/python3` via `SYSTEM_PYTHON` so an active
conda env cannot change it. Four scripts must keep running under the system Python (3.8 on this
box), because that is what invokes them: `scripts/gpu_preflight.py`, `scripts/canary_manifest.py`,
`scripts/scan_secrets.py` and `scripts/git_precommit_scan.py`. `tests/test_interpreter_compat.py`
enforces this, compiles each script under its real interpreter, and pins the jail's 3.12 floor;
trust the test's list over prose. Ruff and basedpyright are directory-scoped, so all of `scripts/`
stays in the 3.8 dialect. Compiling does not catch newer stdlib calls, so a script path that uses
them needs a test that executes it under `/usr/bin/python3` (as `tests/test_gpu_preflight.py`
does). Moving or renaming one of the four means updating the test's lists and the per-file ignores.

## Running expensive things

Anything expensive goes through the resource limiter (cgroup v2 caps on CPU, memory, tasks,
wall-clock):

```bash
scripts/resource-limits.sh --gpu -t 15m -- <cmd>
```

It is **not** an isolation boundary; isolation is `scripts/episode_jail.sh`. Compose them as
**limits outside, isolation inside, never the reverse** (the limiter's D-Bus socket can spawn
processes outside the jail). Details: `docs/resource-limits.md`, `docs/episode-isolation.md`.

- **Before GPU work**, run `scripts/gpu_preflight.py`; it refuses when another process holds VRAM.
  Announce multi-hour holds.
- **Long local runs are expected.** For anything past a few minutes: state the wall-clock estimate
  first, run under the limiter with a `-t` budget, and start it in a named tmux session with a teed
  log.
- **Anything that outlives your attention goes in tmux**, never the foreground or a backgrounded
  shell (dies on SIGHUP). Agents driven through the Workflow tool are killed after a silent period,
  so long gates go through `GATE_TMUX=1` or `scripts/tmux_run.sh`:

```bash
scripts/tmux_run.sh trv2-readout -- python -m games.readout --run-dir <dir>
tail -n 40 /var/tmp/trv2-readout.log   # last line is EXITCODE=<n> once the command finishes
```

  The helper's exit status only says the session started. It refuses a reused session name and a
  `/tmp` log, and forwards only `PATH`; pass other variables with
  `scripts/tmux_run.sh <name> -- env VAR=value <cmd>`. By hand, on a box without this repo:

```bash
tmux new-session -d -s <name> \
  "set -o pipefail; <cmd> 2>&1 | tee /var/tmp/<name>.log; echo EXITCODE=\$? >> /var/tmp/<name>.log"
```

  `set -o pipefail` is required, or the log records tee's status.
- Logs go under `/var/tmp` or `~/logs`, never `/tmp`.

**Every expensive run saves incremental results and resumes; restarting from zero is never the
recovery path.**

- Sync partial results to remote storage during the run (EXIT traps miss instance termination).
  The entry point first fetches what exists at the run's prefix and skips it.
- Training checkpoints every step and resumes only from a *complete* checkpoint (trainer state,
  optimizer, scheduler, RNG, adapter); a torn one is never loaded.
- Sweeps write per-item records incrementally, skip completed items, and count resumed items
  separately from other skips. Hosted batch inference uses persisted handles.
- Resume is idempotent and order-independent: derive seeds from item identity, never execution
  order. A kept record whose prompt no longer matches must refuse the resume, not be folded in.
- Kill a run mid-flight once at smoke scale and watch the relaunch finish only the remainder.
- Reference implementations: `games/run_evals.py` (with `--summarise-only`), `games/train.py`,
  `reward_hacking/recoverybench/runner.py`, `reward_hacking/hatch_probe.py`.

GRPO micro-batches in games training are trimmed to their live tokens
(`games.train.PaddingTrimmedGRPOTrainer`); statistically neutral, not bit-identical.

## Shared-machine hygiene

Several agent sessions share this box, so filling the disk, `/tmp` or the CPU stalls all of them.

- `/tmp` is a RAM-backed tmpfs with an inode cap; keep scratch out of it. `make tmp-check` shows
  usage. Fix a full `/tmp` by freeing inodes, never by moving episode or test scratch into the home
  tree (`assert_disposable_episode_dir` in `reward_hacking/harness/loop.py` refuses that on purpose).
- Heavy local work goes through `scripts/resource-limits.sh`.
- **Delete what you created** when the work ends (venvs, checkouts, composed trees, test scratch).
  Name anything you could not delete in your final report. Large artifacts belong in S3.
- **Share one read-only mirror per finished run**:
  `stage=$(scripts/s3_mirror.sh <run-name> --holder <tag>)` syncs into
  `/var/tmp/s3-mirror/<run-name>/` and prints the path. Checkpoints are excluded by default
  (`--include-checkpoint-state` adds each `trainer_state.json`; `--include-logs` adds logs). The
  holder tag is required; `--delete` refuses while any holder's marker stands, and
  `--delete --force` is a human call.

**Committing a subset of a dirty tree: gate the composed tree, not your files in isolation.** Build a
detached worktree on `/var/tmp` from HEAD, overlay only the files you will commit plus the five
gitignored data files the tests need, remove paths your commit deletes or renames away, and symlink
the main tree's `.venv` (basedpyright needs it; the `.gitignore` rule is the bare `.venv` so the
symlink is ignored). Nothing from `docs/scratch`. Run from the main tree's root:

```bash
T=/var/tmp/gate-<slug>
git worktree add --detach "$T" HEAD
for f in <the paths you will commit>; do [ -f "$f" ] && cp --parents "$f" "$T/"; done
git diff --name-only --no-renames --diff-filter=D HEAD -- <the paths you will commit> |
  while read -r f; do rm "$T/$f"; done   # deletions, and the old half of a rename
for f in games/data/survey/authored.json games/data/survey/published.json \
         games/data/survey/negcontrol-candidates-2026-08-25.json \
         reward_hacking/harness/data/evalplus_cases.json \
         reward_hacking/harness/data/ilcb_cases.json; do
  [ -f "$f" ] && cp --parents "$f" "$T/"
done
ln -s "$PWD/.venv" "$T/.venv"
git -C "$T" status --short     # exactly your commit's changes; `?? .venv` means the rule reverted to `.venv/`
(cd "$T" && make lint typecheck && make test)   # run in tmux
git worktree remove --force "$T"
```

On a composed tree every failure is yours. Never commit from it (the data files carry benchmark
items), and remove it afterwards. It is a code gate only: the privacy gate of record is the
pre-commit hook's scan of the staged blobs in the main tree.

## Code conventions

- **Fail fast.** No blanket `except Exception`, no try/except "just in case", no defensive imports.
  If you catch broadly to log, re-raise. Ruff's `BLE` rules enforce this.
- Type hints, built-in generics (`list[str]`), `X | Y` unions, absolute imports at the top of the
  file, `logging` rather than `print`, ruff at 120 columns.
- Descriptive names over opaque labels (`estimator-group`, never `arm_a` or `v1`).
- Comments explain *why*, never *what*, and never carry change annotations.
- Notebooks are excluded from lint by design.
- **Never sub-sample data to make a cell run faster**; it silently invalidates the statistics.
- Commit messages describe the change and never mention AI authorship: no `Co-authored-by`
  trailers, no emoji.
