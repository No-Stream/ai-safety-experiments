# Working in this repo

The owner's personal AI-alignment research repo. Research, not production: commit to `master`
directly, commit often, keep the tree clean. No CI, no reviewer but us.

Findings to date are indexed one line each in [RESULTS.md](RESULTS.md), with pointers to the docs
that carry the full material.

## Three lines of work

The repo runs three related but distinct research threads. Keep them separate; a session usually
works one at a time, so state the current focus up front and do not review the other two unasked.

1. **Reward hacking and the incident thread** (`reward_hacking/`). Whether heavy RL against loose
   verifiable graders teaches reward-hacking *dispositions* that transfer to environments the
   training never covered, motivated by the grader-exploit incidents of the last several months.
   The hypothesis is spelled out immediately below.
2. **Missing facets of intelligence** (`reward_hacking/recoverybench/`, `reward_hacking/jagged/`).
   Capacities of human intelligence that current LLMs still lack: instrumenting and measuring *why*
   a model fails rather than whether it does. RecoveryBench and JaggedBench are the corpora. (These
   directories sit physically under `reward_hacking/` but belong to this thread, not thread 1.)
3. **Games and multi-agent RL** (`games/`). How reinforcement learning on games shifts a model's
   preferences, reasoning, and behavioral qualities.

The `grpo/` harness is shared training substrate for threads 1 and 3.

## Privacy and benchmark integrity (every session, every commit)

This repo's remote is public and its history will be published wholesale. Check every diff against
this list before committing. None of the following may ever be committed:

- **Benchmark material**: item/question texts, planted-flaw designs, registered answers or
  distractors, or per-item results that would let items be reconstructed. Published items become
  training data, and contamination destroys the benchmark. This includes third-party published
  instruments and other benchmarks' items — survey scales, psychometric item text and anchors,
  allocation payoff tables — regardless of copyright or licence status, and our own authored items
  too: anything that will be run on a future model is an item, and a model that memorized it
  invalidates every measurement made with it.
- **Internal research notes and plans**: these live in gitignored `docs/scratch/` only. Do not
  create plan, handoff, or design docs in tracked paths.
- **Personal or machine-local information**: the owner's name, employer, email addresses,
  usernames (including in absolute paths), AWS account IDs, bucket names, role ARNs, and AWS
  profile names. Such values belong in environment variables or gitignored local config, never in
  tracked code, tests, or docs.

If something sensitive lands in a commit anyway, say so immediately and loudly in your report —
history scrubbing is a git-filter-repo operation the owner runs. Pushing is blocked for agents in
all cases.

**The work is about reward hacking, one hypothesis.** Heavy RL against verifiable rewards, across
many environments whose checks are loose, plausibly teaches a model two things nobody asked for:
(a) a prior that the reachable action space is wider than the task description implies — that
there is usually something else in the environment you can touch — and (b) a habit of reading
*what dimension a situation is graded on* and optimising that rather than the task. These are
dispositions, not tricks, so they would transfer to environments the training never covered.
Testing whether they do, and whether the obvious interventions change them or merely relabel
them, is the point.

**Diagnosis, not verdict.** Never frame anything as a pass/fail gate on whether to continue. When
a result comes out null or weak, the question is *why*, and the first suspect is our own code, not
the world. A surprising negative is a debugging lead before it is a finding.

## Optimize for idea throughput

**Code is cheap, experiments are expensive** — the most important asymmetry here, and easy to get
backwards. The game is maximising ideas tested per unit of GPU time, not minimising code.

- **Spend code liberally to make experiments cheaper** — instrumentation, more metrics per run,
  reusable harnesses, logging that feels excessive. A wasted run is the constraint, never authorship.
- **Never spend an experiment on what reading or reasoning would settle.** Check source, docs,
  literature first — a run launched to answer a ten-minute-search question is the most expensive
  mistake here.
- **Before launching any run, ask what else it can measure for free.** One sampling pass often
  yields several labels at once. Batch the questions, not the runs.
- **Parallelise authorship, serialise only on results.**
- **The characteristic failure is a run repeated because a metric was missing** (the `Trainer.log`
  bug below). Instrument generously, keep artifacts, so a gap is a re-analysis, not a re-run.

**One card, no cloud spend, so cheap beats complete** (the owner, 2026-09-10). Every GPU hour is now
an hour of the only card there is, which sharpens the minimum-viable-run rule rather than changing
it: run the smallest thing that gets a vibe on the idea, read it, then follow up with the next probe.
Trim long rollouts before reaching for more compute — 8K-12K completion caps, repetition and presence
penalties, and a lower reasoning budget are the first moves, and a 32K cap is a deliberate choice made
after a shorter one has shown it is not enough. Cheap inference is the same lever: batched offline
generation beats another RL arm whenever the question can be asked of samples.

**The hot paths have been audited once, and the ranked backlog lives in `docs/scratch`**
(2026-09-02). Six read-only audits of where the compute and the waiting actually went, covering the
GRPO training step, the eval cells, the interp legs, hosted inference, box bootstrap and the
developer loop, folded into one ranked list. Every row carries a semantics grade: *bit* for
identical outputs, *stat* for the same expected result (batching, caching, or a different RNG
consumption order that leaves the sampled distribution alone), *design* for anything that changes
what an experiment measures, which waits on the owner. The first two waves are landed, and the
ones that change how you work are described where they apply below. Before optimizing anything
hot, read the backlog: the measurement usually exists already, and the grade says whether the
change is yours to make.

## Probing posture: speed now, rigor later, deliberately

This phase finds interesting behaviour; it does not establish it to publication standard. **Do not
gate work on power analyses, seed counts, matched-capability comparisons, pre-registration, or
multiple-comparison discipline.** (The owner: purely vibe-based probing now, statistical rigor added
later as appropriate — if everything needs high rigor we never get anywhere.) Two things keep that
safe, both too cheap to skip:

- **Retain the raw material.** Save intermediate checkpoints (not just the final), cache
  activations for probed prompt sets, keep full rollout traces. Then adding rigor later is a
  re-analysis; skip it and "later" silently means re-running the training. Disk is not the constraint.
- **Two controls stay even now**, because skipping them is undetectable afterward: (1) a
  matched-norm random placebo direction whenever you ablate or steer — one extra forward pass,
  without which a positive result cannot be told from "any perturbation of that magnitude does
  this" (exactly what was found at 3B and 8B); (2) one line written down before you look, saying
  what you expect. Everything else defers.

**What the written expectation is for, and what it is not** (the owner, 2026-09-04). A hypothesis
or an expectation line exists to encourage creativity, first-principles thinking and work toward
interesting trailheads: writing down what you think will happen sharpens the design and tells you
which result would be surprising. It is **not** there for rigor's sake, and it is not a pledge
against fishing for p values in the manner of a social-psychology registry. Nothing here needs that
protection, because another iteration, another experiment or an ablation costs a rental rather than
a year, so the owner is extremely opposed to premature rigor. The consequences for how work is
sized:

- **Run MVP experiments**: only enough to know directionally whether there is signal, plus a small
  buffer, and no more. Match the arm count, step count and cell count to that question.
- **An experiment often will not settle its question on its own, and that is fine**: read it, then
  follow up with the next probe. A readout that says "suggestive, here is the next arm" is a good
  readout; padding the first run so it can carry a firm conclusion is the mistake.
- **Full grids, dose-response ladders, factorial crosses and other slow or expensive designs wait**
  until an early probe has justified them. The default is the strongest single treatment first,
  the control queued, the ladder later.
- The scored blocks in `docs/games-predictions.md` follow from this: one expectation per cell with
  the band it is read against, no thresholds that gate anything, and a miss is a lead, never a
  verdict (see "Diagnosis, not verdict" above).

## A check you have never watched fail is not yet a check

The epistemic rule the repo is built around, earned from four one-session bugs that all reported
success while quietly doing nothing (a malformed `# noqa` that suppressed every rule on its line, a
`pytest.skip` that would disarm a guard on a rename, a callback writing into a dict `Trainer.log`
had already copied so verifier accuracy was never recorded, a memory watermark that throttled a
runaway forever instead of killing it). Every one produced green output.

So: **when you add a gate, sabotage it once and watch it fail before you trust it.** Introduce the
exact violation it exists to catch, see it go red, revert. If you cannot make it fail, you wrote
something that prints a reassuring message, not a check. `docs/episode-isolation.md` records which
gates here have actually been attacked and how — including the negative control that runs the
jail's containment checks *outside* the jail and requires each to fail there; `docs/scratch/gate-coverage-ledger.md`
is the honest ledger of which gates have teeth and which are merely green.

Same instinct for reading: do not assert a file path, column name, or API signature you have not
opened this session. "I'm assuming X, unverified" is complete and welcome; a confident guess is not.

## Smoke test end to end before every expensive run

**A run you have never watched complete at smoke scale is a check you have never watched fail.** Run
quick couple-minute end-to-end smoke tests at the 0.8B tier, then launch the multi-hour run with
confidence. Firing an expensive run, local or cloud, without a smoke first is a cycle of
never-ending paper cuts that wastes days.

**End-to-end beats unit-level** — the paper cuts live in the seams (config keys renamed upstream,
checkpoint paths, artifact upload, container env vars, what got baked into the image), not the
components. The question is "does the whole path execute," not "does it learn." (Same-day evidence,
2026-08-15: a single 6.5-minute local run caught five Batch-killers — two removed-upstream config
keys, a silently-ignored `torch_dtype`, a Liger segfault in TRL eval, and the unrecorded-accuracy
bug; four of the five were silent.)

**The local 5090 is the training machine now** (the owner, 2026-09-10), so a long run belongs here
rather than on a rented box; what fits in its 32 GB is the subject of "Machines and models" below.
It is also where smoke tests, offline batched inference (~10x the episode rate of an RL loop) and
activation capture for interpretability run. Two Batch prerequisites still hold wherever the cloud
path is used, both silent on failure: push `:latest` and `:<sha>` together but only ever *resolve*
`:latest`; re-register a fresh job definition at every submit — Batch binds a tag to a digest at
registration time, so reusing an old revision runs old code without saying so.

## Machines and models

**Compute is local now** (the owner, 2026-09-10). The work moved off a shared dev box plus rented
AWS GPUs onto the owner's own machine, and a run happens here unless it cannot: an RTX 5090 with
32 GB of VRAM and a Ryzen 5950X, under WSL2. The 32 GB card is what every sizing decision starts
from. WSL2 currently reports 31 GiB of system RAM, which is the default half-of-host cap on a host
that most likely has 64 GB, so a job that wants more host RAM — CPU offload above all — wants
`.wslconfig`'s `memory=` raised before it wants a smaller model. Several agent sessions may still
share this one GPU, so `scripts/gpu_preflight.py` still applies and a long holder still blocks the
others; what changed is that a long local run is now the expected thing rather than something to
move off the box.

- **Never hardcode a memory budget.** Derive batch size, group size, and context length from the
  VRAM present at startup, and log the device the run landed on. The 5090's 32 GB is what this box
  has today, not a constant to bake into a config, and a run that assumes it will not move to a
  rented card or a later machine.
- **Say the wall-clock estimate before launching**, and run anything past a few minutes under the
  resource limiter in a named tmux session with a teed log (see "Running expensive things").

### When renting cloud GPUs (secondary path, kept for when a job outgrows the 5090)

Everything below still holds when a job genuinely does not fit locally; only its priority changed.
The killswitch and boot-patch rules for rented boxes are under "Running expensive things", and the
AWS Batch surface is `cloud/`.

- **Match instance shape to the job.** A single-GPU job requests a single-GPU instance. An 8-GPU
  node to use one of them is waste; multi-GPU instances are for code that uses them, which we have
  not written yet.
- **Prefer the larger card, stay portable.** `g6e.xlarge` (L40S, 48 GB) is ~26% cheaper per unit of
  work and 3.7x faster in wall-clock than `g6.xlarge` (L4, 24 GB). A local VRAM figure describes
  this box only and must not propagate into cloud job configs.
- **Rented boxes are single-GPU `g7e` or `p5` shapes, on demand, and spot only once on-demand
  capacity is genuinely exhausted** (the owner, 2026-08-24, restated and strengthened 2026-08-27
  after every box in the fleet had drifted to spot-first walks anyway). `g7e.2xlarge` and
  `p5.4xlarge` are the usual asks. Exhaust on demand everywhere before a single spot row enters the
  walk: widen first by instance size, because `g7e.4xlarge` and `g7e.8xlarge` are the same single
  card as the 2xlarge and buy only host vCPU and RAM, so trying them is free capacity rather than a
  tier change; then by availability zone, one candidate row per subnet; then by region, out to every
  region enabled on the account rather than the nearest two. One kit's wide walk on 2026-08-26 ran 45
  on-demand rows, four instance sizes against one to three subnets in each of eight regions, and that
  is the breadth to copy. A smaller card, a `g6e` where the job asked for a `g7e`, is a tier drop
  rather than a widening and stays out of the walk entirely. **"Exhausted" spans both approved
  shapes** (the owner, 2026-09-01, after a day whose seven 60-row g7e walks all came back empty
  while `p5` was never tried): after the g7e rows, walk `p5.4xlarge` on-demand across every
  enabled region before any spot row. A `p5` landing is a cost decision, not an auto-take — H100
  on-demand runs ~4x the g7e spot rate, so hold the box (market-gate pattern) and surface the
  price delta against the run's budget line; taking spot instead of an available-but-4x
  on-demand card is a defensible call, but it is made by a human with the number in front of
  them, never silently by a walk script.
- **A reclaim costs more than the step it interrupts, which is the second and newer reason spot is a
  last resort** (2026-08-27). The first cost is the run: one arm measured ~28 minutes per step
  against a worst observed ~35-minute spot reclaim cadence, so a lease bought about one step, and a
  checkpoint interval longer than the lease trains forever and saves never while every signal stays
  green. The second cost is that a reclaim bypasses the box's own teardown entirely.
  `scripts/idle_watchdog.sh` terminates its box through the EC2 API, so the death emits a
  `TerminateInstances` call that anything watching the account can see. An eviction instead arrives as
  a `BidEvictedEvent` service event with no API write behind it, so nothing records that the box is
  gone and the only trace is in the spot request's own status. Read a suspected reclaim out of the
  spot request status and CloudTrail rather than the box's own log, which dies with the box. Where
  spot really is the only capacity, save every step so a reclaim costs one, and name the box as spot
  in the run's report.
- **Expect to walk many regions and availability zones before capacity turns up**, and read that as
  the nature of GPU supply rather than a fault to debug. Three client-side mistakes do fail every
  candidate and read exactly like `InsufficientInstanceCapacity`: a comma inside a
  `--tag-specifications` shorthand value, and the 16,384-byte cap on *raw* user-data — RunInstances
  measures the bytes before base64 encoding ("User data is limited to 16384 bytes"), and the
  25,600-byte encoded figure that used to sit here is not an API bound at all, since 16,384 raw
  bytes can never encode past 21,848. One `--dry-run` with the identical argument values, before the
  walk, tells a bad request from a dry region. The third is the wrong AWS account: under any other
  account's credentials every candidate subnet "does not exist", and the walk's decline pattern once
  matched those words, so a whole walk logged as declined and closed with the supply line (observed
  2026-09-02). `scripts/launch_gpu_box.sh` now asks sts which account the credentials resolve to
  before its first paid call, and refuses when that is not the account the operator said to expect.
  The pattern is anchored on the CLI's error code now, so a NotFound of any kind, a stale subnet row
  included, stops the walk as misconfiguration instead of being stepped past.

Model ladder for the new work, two working sizes since 2026-09-10: the 0.8B for plumbing and the
9B for anything where behaviour is the measurement.

The Qwen3.5 family has a **Small** tier (0.8B, 2B, 4B, 9B — no 1B)
and a **Medium** tier (**27B dense**, plus MoE 35B-A3B / 122B-A10B / 397B-A17B) — earlier docs
listed only Small. Two newer same-architecture 27B dense drop-ins load with the same classes:
`Qwen/Qwen3.6-27B` and `Qwen/Qwen3.8-27B`. **Nothing newer than 3.5 was published below 27B** — 3.6
exists only at 27B and 35B-A3B, 3.8 only at 27B and 2.4T-A95B, and there is no 3.7 at all — so
`Qwen3.5-4B` and `Qwen3.5-9B` already *are* the latest at their sizes, not a migration we owe.

**Every checkpoint in this family is a vision-language model, at every size.** 0.8B through 27B all
declare `Qwen3_5ForConditionalGeneration` and carry a `vision_config`; the text-only
`AutoModelForCausalLM` load (→ `Qwen3_5ForCausalLM`) silently drops the vision tower and the
multi-token-prediction head at all of them, the 4B included. That is the path we want and nothing
breaks, but it means a checkpoint's download size overstates what reaches VRAM at every rung, so
size a card from a text-only figure rather than from the repo total.

| Model | Use |
|---|---|
| `Qwen/Qwen3.5-0.8B` | **The smoke tier.** Plumbing only: does the code execute end to end. Do not read behaviour off it, ever. Anything smaller is equally acceptable for this purpose. |
| `Qwen/Qwen3.5-2B` | Smoke runs where you also want reward to move. ~5 GB bf16 LoRA. |
| `Qwen/Qwen3.5-4B` | **The working size until 2026-09-10**, and still the right checkpoint for re-analysis of the artifacts already measured at this size and for comparison against them. New behavioural measurement goes to the 9B instead. ~10 GB bf16 LoRA, ~7.8 GiB of text-only weights. The newest 4B Qwen there is — no 3.6 or 3.8 was ever published at this size. |
| `Qwen/Qwen3.5-9B` | **The working size whenever behaviour is the measurement** (the owner, 2026-09-10), and the **interp cross-validation arm**. Also the newest 9B Qwen there is, for the same reason as the 4B. Fitting bf16 LoRA training into the 5090's 32 GB is tight and buys its room from the rollout budget rather than from a smaller model; size it from the VRAM the run actually sees, and read `docs/scratch/single-5090-grpo-notes-2026-09-10.md` for the sizing work (gitignored, so a plain-text pointer rather than a link). `Qwen3.5-9B-Base` is the only ~9B Qwen with a pretrained sparse autoencoder (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_{50,100}`, all 32 layers). Note the checkpoint split: the SAE is on `-Base`, while the TMAX RL'd models descend from the instruct `Qwen/Qwen3.5-9B` — measure that transfer, do not assume it (base→instruct SAE transfer has documented quality hits; see `docs/scratch/tiny-model-selection-2026-08-17.md`). |
| `Qwen/Qwen3.8-27B` | The **stretch tier**, and a target that needs creativity rather than a bigger config: ~52 GiB download against ~50.1 GiB of text-only bf16 weights in VRAM once the family's vision tower and MTP head drop out, against 32 GB of card, so bf16 LoRA does not fit locally by any straightforward route. What might: QLoRA (knowingly against the family guidance below), CPU-offloaded checkpoints, chunked loss. The sizing note is `docs/scratch/single-5090-grpo-notes-2026-09-10.md` (gitignored, plain-text pointer). The `Qwen3.8-27B-FP8` variant (~27 GiB text-only) fits 32 GB with little to spare and is owner-approved for interp probing (2026-08-18) — **probing and inference only, because an FP8 checkpoint cannot be trained on at all** (see below). No text-only/base sibling exists at 27B (see `docs/scratch/qwen38-27b-load-check-2026-08-17.md`); `Qwen3.5-27B` (identical load path) carries first-party *instruct* SAEs. See `docs/scratch/observable-model-rl-feasibility-2026-08-16.md` and `jlens-sae-feasibility-2026-08-16.md`. **For inference-only probes the owner expects the Q4_K_XL quant to run on the 5090 under vLLM at 50+ tok/s** (the owner, 2026-09-10; an expectation, not yet measured here), and rates the model as roughly late-2025 frontier quality, so it is the default hosted-model substitute for cheap local probing. Training stays bf16 LoRA and does not fit. |

**Jacobian-lens rule (owner, 2026-08-18): never spend effort checking whether a pre-fitted lens
exists for a checkpoint — we fit our own, cheaply** (a closed-form accumulation, not a training
run; source `anthropics/jacobian-lens`). Past sessions burned time on lens-existence claims and
got them wrong in both directions. SAEs are the opposite case: genuinely expensive to fit, so a
pretrained one is a real (but non-blocking) asset — without one, run non-SAE techniques.

**9B is the default for real measurement** (the owner, 2026-09-10), comfortably clear of the floor
at which "chose not to do the task" is distinguishable from "could not", which is the whole
distinction this research rests on. That floor was measured at 4B (Terminal-Bench Lite: 31.8±3.8 vs
2B's 5.7 — from Ai2's TMAX paper, arXiv:2606.23321 Table 3, so harness-specific numbers rather than
a neutral property; the gap also survives on the harder TB 2.1, 14.2 vs 1.9), and the argument for
being above it applies a fortiori at 9B. Two family facts: **4-bit
QLoRA is not recommended for Qwen3.5** — this is Unsloth's fine-tuning guidance, *not* the Qwen
model card, and the reason they give is only "higher than normal quantization differences"; the
`attn_output_gate` mechanism is our own conjecture rather than theirs, though Qwen's own GPTQ-Int4
build does exclude every attention module from quantization, which is the closest thing to
first-party support the conjecture has. Among builds you could actually train from, 4-bit barely
saves memory on this family (GPTQ-Int4 ~30.3 GB vs FP8 ~30.9 GB at 27B), so **FP8 is the memory
escape hatch for inference and interp probing — and for training there is no escape hatch, because
an FP8 checkpoint cannot be trained on at all, LoRA included** (transformers'
`FineGrainedFP8HfQuantizer` declares `is_trainable = False`, and `Trainer` refuses even with an
adapter attached). Training is bf16 LoRA at every rung, which is what makes the 27B a stretch
target here, and QLoRA is not recommended anywhere on the ladder. And these are hybrid-attention
models (`Qwen3_5Config`, whose text sub-config is `Qwen3_5TextConfig` — *not* `Qwen3NextConfig`,
a different and older family — with three Gated DeltaNet linear-attention layers per full attention
layer), so LoRA must target the linear-attention projections too: take the target modules from
`discover_lora_targets` in `grpo/throughput.py`, never a hand-written `q/k/v/o` list, which reaches
attention in a quarter of the layers and silently leaves every DeltaNet block frozen. Any interp
tooling assuming standard attention needs checking.

Interp tooling (sparse autoencoders, a pre-fitted Jacobian lens, a TransformerLens adapter) exists
and was verified for `Qwen/Qwen3.5-4B` on 2026-08-15 — **not a reason to keep measurement at 4B now
that 9B is the working size**; re-verify the recipes at 9B before building on them.
Recipes, silent-failure gotchas, and version pins: `docs/scratch/interp-tooling-verified.md` (read
before building on any of them).

**Interp method priority (owner, 2026-08-19) — use the most rigorous method available, not the
easiest.** Lead with **Jacobian-space / Jacobian-lens** (`docs/interp-methods/jacobian-space.md` —
the tracked reference: what it is, why it is more faithful than correlational directions, and how to
load/fit it per model). Difference-of-means directions and trained linear probes are a fine baseline
but old-school by comparison — use them to complement, not lead. SAEs sit at 1.5: valuable where a
pretrained one exists, skip cleanly where it does not. Do not default to simplistic direction-probing
because it is fast — this is rigorous work. **Pair every direction with an intervention: a decodable
direction is not a used one.** The causal tier is steering and ablation against a matched-norm
placebo, plus activation patching between matched twins (the conflicting-vs-original grader pairs are
an ideal clean/corrupted substrate; reach for attribution patching / AtP* only if a position or layer
sweep gets large). Defer the heavy machinery (circuit tracing and transcoders, natural-language
autoencoders, MOLT, crosscoders) to a dedicated mechanistic project, not routine probing.

`grpo/rlvr_math.py` still defaults to `Qwen/Qwen3-0.6B`, correct for that toy harness — not a
pending migration.

## Where things live

Every top-level entry, so nothing has to be guessed at:

- `reward_hacking/` — the live project: reward-hacking-lead exploration on an off-the-shelf
  Qwen3.5-4B. The probes are inference-only: `episodes/` (baited reaching harness, leads #1/#2/#6),
  `channel/` (cross-episode-note probe, lead #3), `interp/` (reward-hacking-vs-deception direction
  probe, lead #5), `harness/` (the real-execution agent loop: commands run inside the episode jail,
  every episode graded twice, `hack_detector.py` separating grader-gaming from honest bugs a
  lenient grader passed), `jagged/` and `recoverybench/` (item corpora plus sweep machinery over
  hosted models via `model_backend.py` and `bedrock_batch.py`), and `tmax/` (artifact download and
  geometry for the TMAX RL'd checkpoints). The training side is `train.py` with its `train_*.py`
  siblings: GRPO on the 4B, on `games/`'s run machinery, against a misspecified grader and, matched
  step for step, against the true one, so the pair rather than either arm carries the claim; a
  third arm, `legible-subset`, stands apart from that pair and rewards a visible k=3 subset of the
  true check while the full hidden check is measured but never rewarded. Its tests live in
  `reward_hacking/tests/`. Read `reward_hacking/README.md` and the plan below before adding
  anything.
- `games/` — the game-theory GRPO project: matrix-game RL arms where identical prompts differ only
  in grading rule (`train.py`, `rewards.py`, `payoffs.py`), an eval battery covering play on
  trained and held-out games, decision-theory endorsement, and arithmetic retention (`evals.py`),
  a decision-theory probe battery with our FDT-axis items plus a DTBench adapter (`probes.py`),
  and `stage_runner.py`, which sequences GPU stages and refuses to call one successful without
  checking. No README yet — the module docstrings carry the design, and the plan is
  `docs/scratch/2026-08-16-game-theory-rl-plan.md`. Pre-registered predictions are tracked at
  `docs/games-predictions.md`; tests live in `games/tests/`.
- `sociology/` — the analysis-model observer study (incident thread): feeds bundles of banked agent
  episodes that provably could not have communicated to hosted analysis models under varied
  provenance framings, and measures how often they assert coordination that is absent by
  construction. `corpus.py` (episode load, reasoning interleave, co-location-cue scrub), `bundles.py`
  (deterministic content-seeded bundle draws plus the resume manifest), `stimulus.py` (frames, request
  and judge rubric loaded at runtime from a gitignored file — no stimulus prose in tracked code),
  `runner.py` (batch and live legs, priced dry run, smoke), `scans.py` (deterministic reply scans),
  `judge.py` (a blind meta-judge sibling of `reward_hacking/hatch_narration_judge.py`). Inference-only,
  reusing `reward_hacking.*` transports by import; tests live in `sociology/tests/`.
- `grpo/` — shared training substrate: a working GRPO harness on TRL 1.10, verified end to end on
  GPU. The `grpo/rlvr_math.py` arithmetic task is a toy; the RL plumbing is live and `games/`
  builds directly on it (sizing, throughput, callbacks). See `grpo/README.md`.
- `cloud/` — the AWS Batch surface for training arms, a secondary path since compute moved local
  on 2026-09-10: the container (`Dockerfile`, `entrypoint.sh`, whose S3-sync EXIT trap means a
  spot-reclaimed job still ships its checkpoints), `push_ecr.sh`, and `submit_job.py`, which
  registers a fresh job definition at every submit because Batch pins a tag to a digest at
  registration time (the stale-code trap in the Batch section above).
- `scripts/` — operational tooling: resource limiter, episode jail + red-team suite, canary
  tripwire, secret scanner, GPU preflight.
- `tests/` — the repo-level pytest suite behind `make test`: cost model, throughput, doc links,
  GPU preflight, and the interpreter-compatibility gate described below. Project-local tests stay
  next to their projects (`games/tests/`, `reward_hacking/tests/`, `sociology/tests/`), and every
  root is listed in `pyproject.toml`'s `testpaths`, because a root left off is silently never
  collected.
- `docs/` — the operational docs that are meant to stay accurate (`episode-isolation.md`,
  `resource-limits.md`, `games-predictions.md`) plus `docs/scratch/`, which is gitignored and is
  the **single location for every piece of internal working material**: plans, handoffs,
  preregistrations, the sparse research-directions index (`future.md`, read it, do not rewrite),
  one file per direction, and low-ceremony notes. See `docs/scratch/README.md`. That split is a
  privacy rule, not filing preference — see the section at the top of this file. Nothing internal
  goes back under a tracked path, and no second scratch location gets created: a former top-level
  `notes/` was folded in here (2026-08-18), and its `.gitignore` entry stays as a tripwire.
- `canary/` — the canary tripwire's baseline manifest, written by
  `scripts/canary_manifest.py --update`. Machine-local because every hash in it is keyed to one
  machine's paths; only the README is tracked.
- `artifacts/` — gitignored run outputs: traces, sweeps, eval records, checkpoints. Local-only,
  and deliberately kept — disk is not the constraint (see the probing-posture section above).
- `legacy/` — finished research. Do not extend, refactor, or lint it. One exception (the owner,
  2026-08-16): the blind `except` blocks in `legacy/pretraining.ipynb` are being narrowed to
  specific exceptions.
- Root files: `Makefile` (the gates, plus the `SYSTEM_PYTHON` constraint below), `pyproject.toml`
  (dependencies and the ruff/basedpyright configuration, including the per-file ignores the 3.9
  scripts depend on), `uv.lock` (the pinned environment — `make setup` builds `.venv` from it),
  `README.md` (the public-facing description), `RESULTS.md` (the one-line-per-finding index of
  research results; most of its pointers name gitignored scratch docs on purpose), and this file,
  with `CLAUDE.md` a symlink to it.

**Authoritative near-term plan: `docs/scratch/2026-08-15-reward-hacking-exploration-plan.md`** —
this phase explores for leads, not a systematic apparatus.
`docs/scratch/2026-08-16-jaggedbench-plan.md` is designed and agreed but not built. Two older
documents are kept for the record, neither current:
`docs/scratch/2026-08-15-episode-reward-research-agenda.md` (a proposal, not agreed; its section 10
lists open questions for the owner) and `docs/scratch/2026-08-15-complex-hack-substrate.md` (a
substrate spec, superseded and deferred — mine its ideas, do not build it wholesale). Both are
honest about what adversarial review killed in them; read the corrections. All of these are
gitignored local files rather than repository content, so a fresh clone will not contain them and
these are plain-text pointers rather than links on purpose.

## Environment and the gates

`make setup` builds `.venv` from the pinned `uv.lock`. Python 3.13, managed by uv. **Never
`pip install`** — the lockfile is the environment.

**On the 5090 box, `make setup` alone leaves the tree red** (2026-09-10). `games.train` refuses to
construct without vLLM and `cloud/submit_job.py` imports boto3, so a bare `uv sync` (which removes
extras it was not asked for) fails about 200 games tests and the type check. Use
`uv sync --frozen --extra vllm --extra bedrock` (or `make setup-gpu` plus the bedrock extra) here.
Two more facts of this box: Ubuntu 20.04 has no packaged Python 3.12+, so the jail interpreter is
the uv CPython staged under `/var/tmp/cpython-runtime` by `scripts/stage_jail_python.sh` (with
numpy installed into it) and does not survive a `/var/tmp` wipe; and its tmux is 3.0a, which
`scripts/tmux_run.sh` now tolerates. `bwrap` is not installed, so the jail runs its `unshare`
fallback until bubblewrap is added.

Five gates, all expected green before "done", plus one advisory check:

```bash
make test           # pytest, CPU only
make lint           # ruff format --check and ruff check (select=ALL, curated ignores)
make typecheck      # basedpyright, strict mode -- GATING
make jail-test      # episode jail containment plus its negative control
make canary-check   # checksum tripwire over the host's code-execution surfaces
make typecheck-ty   # ty (Astral) -- ADVISORY only, not gating
```

`make test` shards the suite over `TEST_WORKERS` pytest-xdist processes (default 8): about 3.5
minutes on this box against about 15 serial. `--dist loadfile` keeps every test of a module on one
worker, so module-scoped fixtures build once and the pass/fail set matches the serial run.
`make test TEST_WORKERS=4` when the shared box is busy (about 4.8 minutes). `make test-select`
stays serial, because worker start-up costs more than it saves on a narrow selection; put `-n` in
`ARGS` when a wide one wants it. `make test-changed` runs only the test files the working tree's
changes against HEAD can reach, mapped by directory (`scripts/test_changed.py` carries the table: a
`games/` edit runs `games/tests`, a changed test file runs itself, nothing changed runs nothing),
sharded like `make test`; it is the iteration gate and not the commit gate, because a directory
mapping cannot see a cross-component import. `make ci` bundles the fast trio (`lint typecheck
test`) and is still the gate before a commit; `jail-test` and `canary-check` run separately. If a gate fails and
looks unrelated to your change, fix it anyway.

`make ci` runs its three gates **concurrently** through one `make -j3`, so the type check happens
inside the suite's runtime rather than ahead of it (21-25 seconds at `--threads 8`, against 53-59
single-threaded; `make typecheck TYPECHECK_THREADS=2` when the box is busy). The status is unchanged
and is still the AND of all three, with make naming whichever gate failed. The one thing that did
change: a red lint or type check no longer spares you the suite, because all three start at once, so
`make lint typecheck` is the way to ask the cheap pair alone. And `make test`, `make test-changed` and
`make ci` all take **`GATE_TMUX=1`** (`make ci GATE_TMUX=1`), which runs the gate in a detached tmux
session through `scripts/tmux_run.sh`, tees it to `/var/tmp/<name>.log` and returns immediately with
the command to poll — how you run a five-minute gate from an agent that cannot hold the foreground
that long (see the tmux paragraph under "Running expensive things"). The gate's status is the log's
last line, `EXITCODE=<n>`, never make's.

**Two constraints that look like clutter and are not.** The Makefile calls `/usr/bin/python3` by
absolute path via `SYSTEM_PYTHON`, because bare `python3` resolves to a miniconda 3.13 whenever a
conda env is active and `make canary-check` would then run under whichever environment happened to
be active. And FOUR scripts, `scripts/gpu_preflight.py`, `scripts/canary_manifest.py`,
`scripts/scan_secrets.py` and `scripts/git_precommit_scan.py`, **must keep running under system
Python, which is 3.9.25 here**, because that is what invokes them: `scripts/resource-limits.sh` calls
the first as bare `python3`, the `canary-check` target and `docs/episode-isolation.md` call the next
two the same way, and the git hooks that `scripts/install_git_hooks.sh` writes call the fourth as bare
`python3` from inside a commit, where the shell has whatever interpreter the committing environment
happens to expose. `tests/test_interpreter_compat.py` is the enforcement point and lists all four; this
sentence said three until 2026-09-02, so trust the test over any prose that disagrees with it.
`scripts/jail_assertions.py` was a fourth while the jail exposed whatever `/usr/bin/python3`
happened to be; the jail now resolves an interpreter at or above 3.12 and refuses to start without
one, and `scripts/run_jail_tests.sh` runs the assertions under that. It keeps the 3.9 dialect
regardless, because ruff and basedpyright are both directory-scoped with no per-file version
override, so `scripts/` follows its oldest occupant. All of this breaks at runtime rather than in
lint, so `tests/test_interpreter_compat.py` compiles each script under the interpreter that actually
runs it — and pins the jail's version floor, since a floor edited downwards brings back the bug that
split the two groups apart: a 3.9 jail crashed every submission written in a later dialect and the
graders scored working code as flawed. Move or rename any of the four → update both the test's
script lists and the per-file ignore paths.

## Running expensive things

Anything expensive goes through the resource limiter, which caps CPU, memory, tasks, and wall-clock
via cgroup v2 so a runaway job cannot take the box down:

```bash
scripts/resource-limits.sh --gpu -t 15m -- <cmd>
```

It limits resources and **nothing else — it is explicitly not a security or isolation boundary.**
Isolation is `scripts/episode_jail.sh`, and the two compose in one order only: **limits outside,
isolation inside. Never the reverse** — the limiter talks to the systemd user manager over a D-Bus
socket, and anything that can reach that socket can ask systemd to spawn processes outside the jail.
Details and measurements: `docs/resource-limits.md`, `docs/episode-isolation.md`.

**Anything that outlives your attention goes in a named tmux session with a teed log**, never the
foreground and never a backgrounded shell: a background shell dies on SIGHUP (compaction, a
reconnect, the parent exiting) while the work it started keeps burning the box unlogged, and an agent
driving this box through the Workflow tool is killed by its own harness after 180 seconds with no
output (600 seconds for a background agent; an interactive session has no such limit).
`scripts/tmux_run.sh` is that pattern written down once, so a kit README no longer has to re-derive
it. It refuses a session name already in use, defaults the log to `/var/tmp/<name>.log`, and prints
the command to poll:

```bash
scripts/tmux_run.sh trv2-readout -- python -m games.readout --run-dir <dir>
tail -n 40 /var/tmp/trv2-readout.log   # last line is EXITCODE=<n> once the command finishes
```

The helper's own exit status only says whether the session started; the command's status is that
`EXITCODE` line, which the helper keeps on a line of its own even when the command's output ended
without a newline (`tee` is byte transparent, so the by-hand form below can glue the two together and
give a `tail -n 1` reader a false negative). On a box with no checkout of this repo, the canonical
form by hand is:

```bash
tmux new-session -d -s <name> \
  "set -o pipefail; <cmd> 2>&1 | tee /var/tmp/<name>.log; echo EXITCODE=\$? >> /var/tmp/<name>.log"
```

`set -o pipefail` is the load-bearing part — without it the log records tee's status, which is zero
whatever the command did — and a bare `$?` under pipefail is correct in both bash and zsh, where
`${PIPESTATUS[0]}` is bash-only and tmux runs the command under the login shell. Logs belong under
`/var/tmp` or `~/logs` and never `/tmp`, which is a RAM-backed tmpfs with its own inode cap shared by
every session on the box; the helper refuses a `/tmp` log for that reason. Environment variables need
passing in explicitly (`scripts/tmux_run.sh <name> -- env VAR=value <cmd>`), because a tmux session's
environment is the tmux server's rather than the launching shell's — the helper forwards `PATH` and
nothing else.

Before any GPU work, `scripts/gpu_preflight.py` refuses to start when another process already holds
VRAM on the single shared GPU. Assume someone else may be using it.

**The local 5090 is the training machine, and long local runs are expected** (the owner,
2026-09-10), which reverses the ~5-10 minute local ceiling that held while the only card was a
shared 24 GB L4. Three things still hold for anything past a few minutes, and they are what make a
long run safe to leave alone: state the wall-clock estimate before launching, run it through
`scripts/resource-limits.sh` with a `-t` budget so a runaway cannot take the box down, and start it
in a named tmux session with a teed log rather than the foreground or a background shell. Several
agent sessions may still be waiting on the same card, so `gpu_preflight` remains the check that you
are not stepping on a run in progress, and a multi-hour hold is worth announcing.

**Every rented instance arms its own killswitch at launch — no box may depend on a human
remembering to terminate it** (the owner, 2026-08-18). Two layers, both armed in user-data before
any job starts: a max-lifetime dead-man switch (`shutdown -h +<minutes>` sized to the job, with
the instance's shutdown behavior verified as `terminate` via the API after launch — a switch that
merely *stops* an on-demand box keeps billing its EBS and teaches false comfort), and an idle
watchdog that terminates the box after it has done no work for longer than any legitimate
inter-job gap in its agenda. The second layer exists because the first is blind to the cheapest
failure: a box that bootstraps and then never starts its job burns its whole dead-man window
doing nothing (observed the night this rule was written: a bootstrapped GPU box sat at 0%
utilization until a human happened to notice). Size the idle threshold above real setup and
inter-stage gaps, or gate it on "agenda complete" markers, so it cannot kill a box between
stages. And per the rule this repo is built around: a killswitch nobody has watched fire is a
reassuring message, not a check — let each layer actually terminate a box once before trusting
it, e.g. by letting the idle watchdog perform the final teardown after artifacts are synced and
verified, which tests the mechanism at zero marginal cost. Both layers only cover a box that dies on
its own schedule, and only the idle watchdog emits an API event at all, the dead-man switch being
deliberately a bare `shutdown` that needs no network, credentials or CLI to fire. A spot eviction runs
neither of them, which is one of the two reasons the tier rule above wants on-demand capacity
exhausted first.

**Every rented instance patches its OS at boot** (the owner, 2026-08-20). User-data runs
`scripts/ec2_boot_patch.sh` as root right after the dead-man switch is armed and before the job
starts. Two facts earned the rule, because "just launch the newest AMI" is not enough on its own:
the newest available AMI build itself shipped a vulnerable package, and the fixed package sat in the
distribution's `-updates` pocket rather than `-security`, so a stock security-pocket-only
unattended-upgrades pass would have missed it too. One limit to know: a base GPU AMI apt-holds its
own kernel metapackages, so a full upgrade on a running box reports packages still upgradable
instead of installing them. A kernel fix therefore needs a newer image rather than another patch
pass on the box you already have.

**Every expensive run saves incremental results and resumes — restarting from zero is never the
recovery path** (the owner, 2026-08-28). A box that dies early — crash, killswitch, eviction,
human mistake — must leave its completed work durable and make its relaunch a continuation.
Concretely: sync partial results to remote storage *during* the run on an interval, because an
EXIT trap alone misses the deaths that matter (instance termination skips traps); and make the
entry point, before doing anything new, fetch what already exists at the run's own prefix and
skip it. Training runs checkpoint every step and resume from the latest synced checkpoint.
Sweep- and grid-shaped runs write per-item records incrementally and skip items whose records
are already complete, counting resumed items separately from items skipped for any other reason
so the summary stays honest about what actually ran. Hosted batch inference goes through
persisted handles so collection resumes without resubmitting. Resume must be idempotent and
order-independent — derive seeds from item identity, never from execution order, or a resumed
run is silently a different experiment. And per the rule this repo is built around: kill a run
mid-flight once at smoke scale and watch the relaunch skip the finished work and complete the
remainder, because a resume path nobody has watched work is a reassuring message, not a check.

Three of the expensive paths do this now (2026-09-02), and each names the trap it closes. The
games eval cells (`games/run_evals.py`) resume a cell by record key from the cell's own trace.
Finished records are kept byte for byte and only the pending identities are generated; the trace's
resume block carries one entry per session with the records that session kept and the torn trailing
lines it dropped, and the rebuilt summary derives what the latest session generated from those
counts. `--summarise-only` rebuilds a cell's summary from a trace copied down from S3 with no GPU
around, which works because a summary is only ever the battery's rebuild of its trace. The hosted
RecoveryBench and hatch live runs (`reward_hacking/recoverybench/runner.py`,
`reward_hacking/hatch_probe.py`) resume the same way, by content-derived cell key, and a kept record
whose prompt digest no longer matches the relaunch's own rendering refuses the whole resume by name
rather than being skipped, so an edited item cannot be folded into an old trace. Games training
(`games/train.py`) resumes only from a *complete* checkpoint. A box that dies mid-sync leaves S3
holding a `checkpoint-N/` with its small adapter and `trainer_state.json` but no `optimizer.pt`; TRL
would load that, continue with zeroed Adam moments and keep every signal green. So a resume from
`latest` walks newest-first past any checkpoint missing its trainer state, optimizer, scheduler, RNG
state or adapter, moves the torn copy under `incomplete-checkpoints/` and records which steps will
run twice. An explicitly named torn checkpoint is refused rather than swapped, and a run directory
with checkpoints but no complete one raises instead of starting fresh.

**GRPO training micro-batches are trimmed to their live tokens**
(`games.train.PaddingTrimmedGRPOTrainer`, 2026-09-02). TRL right-pads every completion in a
generation batch to the batch's longest and splits the buffer along rows only, so one runaway
completion at the cap made every micro-batch of that step run the model over the cap width; the
audit measured 74% of a 2B training step as forward/backward over masked padding. The trainer drops
the all-pad prompt and completion columns of each micro-batch at TRL's `_prepare_inputs` seam, logs
`padding trim: step tokens padded=N trimmed=M kept_fraction=f` once per step, and records the totals
in `log_history` and `mem_log.csv`. Statistically neutral, not bit-identical: the loss masks zero
the pads and the normalizers in use are pad-invariant, so the gradient is the same function of the
same tokens while reduction order and kernel autotune change with width. Measured on the L4 through
the production loss path on a real corpus row: identical loss, LoRA gradient cosine 0.9999, norm
ratio 1.0016, and cutting a quarter of the live tokens instead drops the cosine to 0.994, so the
check has teeth. The audit's estimate is 1.9-2.7x per step.

## Shared-machine hygiene (files, /tmp, CPU)

This box runs several concurrent agent sessions, so filling the disk, `/tmp`, or the CPU does not
slow one session down, it stalls all of them. On 2026-08-21 `/tmp` ran out of inodes —
file slots, not bytes — and every session's shell failed box-wide until the leftover scratch was
reaped by hand.

`/tmp` here is a RAM-backed tmpfs with a hard inode cap separate from its byte space, so a tree of
small files can exhaust the file slots while `df -h` still shows room, and everything in it costs
RAM. `make tmp-check` prints both figures plus the directories holding the most files. (Both facts
were measured on the previous dev box; WSL2 may lay `/tmp` out differently, so re-check with
`make tmp-check` before relying on the numbers. The rule to keep scratch out of `/tmp` holds either
way.)

**Committing a subset of a dirty tree: gate the composed tree, never your files in isolation.**
Several sessions edit this tree at once, so the normal commit carries some paths and leaves others
dirty, and the question a gate has to answer is whether *those paths on top of HEAD* stay green.
Compose that tree as a detached worktree on `/var/tmp` (never `/tmp`, for the inode reason above). A
`git archive HEAD` tree is not enough: it has no `.git`, and about thirty tests ask git for
provenance or for which paths are tracked, so they fail for a reason unrelated to the overlay.
Overlay only the files you will commit, plus the five gitignored data files the tests need and a
fresh checkout lacks (three survey files under `games/data/survey/` and two harness case files under
`reward_hacking/harness/data/`), and nothing from `docs/scratch`. Then remove from the worktree every
path your commit deletes or renames away, which a copy cannot express: left at HEAD's content, a
deleted module still imports (a false pass) and a deleted stale test still runs (a false failure).
Symlink the main tree's `.venv` into the worktree, because basedpyright is pinned to
`venvPath = "."` and refuses to run without one there. The `.gitignore` rule for it is the bare
name `.venv` rather than the directory-only `.venv/`, on purpose: a directory-only pattern does not
match a symlink, so under it the link shows up as untracked in the worktree's status, and the
committable-tree privacy sweep in `tests/test_scan_secrets.py`, which lists untracked paths too,
follows it into every file of the venv: about 97,000 on this box, minutes of scanning, and then a
hard failure when the sweep meets a compiled library over its 64 MiB size limit. That failure has
nothing to do with your overlay, and the bare rule keeps it out of every worktree and every fresh
clone with no per-machine exclude to remember. Run the block from the root of the main tree, which
the overlay paths and the symlink target are relative to.

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
(cd "$T" && make lint typecheck && make test)   # tmux; about 5 minutes sharded
git worktree remove --force "$T"
```

`--no-renames` matters on the deletion line: git's default rename detection folds a staged
`git mv` into one `R` entry, and a filter on `D` alone would then leave the old path in the
worktree. On a clean composed tree every failure is yours. The data files carry benchmark item text,
which is why they are gitignored: the composed tree is for running the gate, never for committing
from, and removing the worktree afterwards is part of the recipe. A green composed tree is a code
gate, not a privacy clearance: `scripts/scan_secrets.py` arms its instrument-phrase detector from the
survey files and from a note under `docs/scratch`, and the composed tree has only the former, so its
sweep runs on fewer sources than the main tree's. The privacy gate of record stays the pre-commit
hook's scan of the staged blobs, which runs in the main tree on the commit itself.

**Delete what you created when the work ends** — venvs, repo checkouts, composed trees, test
scratch — in `/tmp` and in the home tree alike. What you cannot delete yourself, name explicitly in
your final report so the owner can: a permission-blocked delete gets surfaced loudly, never
silently abandoned. Large artifacts belong in S3 rather than on local disk.

**One shared read-only mirror per finished run, not a private copy per agent**
(`scripts/s3_mirror.sh`, 2026-09-04). Prep, readout and audit agents each staged the same run's
artifacts into their own scratch, so one run came down the wire once per agent, and an agent reaping
its own scratch took with it the copy a sibling was still reading. `scripts/s3_mirror.sh <run-name>
--holder <tag>` delta-syncs `<SHIP_S3_PREFIX>/<run-name>/` into `/var/tmp/s3-mirror/<run-name>/` and
prints that directory on stdout and nothing else, so a recipe writes
`stage=$(scripts/s3_mirror.sh track-record-v2 --holder readout-a)` and passes `--stage "$stage"` on.
Checkpoint directories are excluded by default, which is where 95% of a run's bytes are, 16.00 of
track-record-v2's 16.83 GB, leaving a 147-object 0.83 GB mirror; `--include-checkpoint-state`
re-includes the one `checkpoint-*/trainer_state.json` per checkpoint a readout actually reads (72
objects, 2.88 MiB). A run's logs are not under its own prefix but in the sibling
`<base>/logs/<run-name>/` (1.93 GB for track-record-v2, almost all of it one object), so
`--include-logs` is a second sync into `<mirror>/logs/`, the same place both readout recipes put
them. The holder tag is required and has no default: the only name the script knows for its caller is
its parent shell's pid, which two readers in one shell share and one reader across two shells does
not, so a default would hand out claims nobody could release. `--delete` refuses while any holder's
marker stands, and `--delete --force` is the person's call for a marker whose agent is gone. No
readout recipe calls it yet, so its per-readout saving is still a projection rather than a
measurement.

**A full `/tmp` is fixed by freeing inodes, never by redirecting episode or test scratch into the
home tree.** The harness guard `assert_disposable_episode_dir` in `reward_hacking/harness/loop.py` refuses a
home-tree episode dir, mirroring `episode_jail.sh`'s own rule, and it firing is that check working
rather than an obstacle to route around.

Heavy local work goes through `scripts/resource-limits.sh` as described in the section above: CPU
contention wedges sibling sessions the same way a filled `/tmp` does.

## Code conventions

**Fail fast.** No blanket `except Exception`, no try/except "just in case", no defensive imports —
research the API instead. If you catch broadly in order to log, re-raise. Eaten exceptions in a
batch pipeline are how silent corruption happens; ruff's `BLE` rules enforce this mechanically.

Type hints, built-in generics (`list[str]`, not `List[str]`), `X | Y` unions, absolute imports at
the file top, `logging` rather than `print`, ruff at 120 columns. Descriptive names over opaque
labels — `estimator-group`, never `arm_a` or `v1`. Comments explain *why*, never *what*, and never
carry change annotations like "changed from 8" (that belongs in the commit message). Notebooks stay
excluded from lint by design (ruff reads a notebook as one module, so every per-cell import becomes
an E402).

**Never sub-sample data to make a cell run faster** — it silently invalidates the statistics and
produces numbers that read as full-data results.

Commit messages describe the change and never mention AI authorship — no `Co-authored-by` trailers,
no emoji.
