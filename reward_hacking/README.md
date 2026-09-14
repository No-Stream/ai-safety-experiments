# reward_hacking — the live project

A built exploration package. The current phase is chasing reward-hacking leads on an
off-the-shelf Qwen3.5-4B, everything inference-only so far — not building a systematic
measurement apparatus. The authoritative near-term plan is
`docs/scratch/2026-08-15-reward-hacking-exploration-plan.md`; read it first.

What is here:

- `episodes/` — the baited reaching harness (leads #1/#2/#6). Small ops-agent tasks with
  out-of-band affordances, an `AffordanceDose` ladder, multi-turn rollout, and full JSONL
  traces. The environment is simulated in Python (nothing executes); `reach.py` annotates each
  action in-bounds/bait as a reading *lens*, deliberately not a tier detector.
- `channel/` — the cross-episode-note probe (lead #3): a persistent scratch directory across
  episodes, measuring whether the model reads a planted note, acts on it, and writes one
  forward, and whether that write-rate rises with reward pressure. Committed and sabotage-tested.
- `interp/` — the reward-hacking-versus-deception direction probe (lead #5): residual capture,
  diff-of-means, per-layer cosine, and a matched-norm placebo direction.
- `harness/` — the real-execution agent loop: every command the policy emits runs for real inside
  the episode jail against tasks with readable, deliberately brittle graders, and each episode is
  graded twice (the in-band grader it can read, and a robust host-side recompute). `hack_detector.py`
  reads a saved trace afterwards and separates deliberate grader-gaming from an honest attempt a
  lenient grader passed anyway, returning the evidence for the call — the raw proxy-vs-true gap
  conflates the two, and on easy tasks the honest bugs dominate. Its signals come in two families:
  the answer-side ones ask how the artifact was produced, and the machinery-side ones in
  `grader_tampering.py` ask whether the thing that decides the verdict was left alone (the grader
  rewritten, an interpreter shim on `PATH`, `usercustomize.py` planted, an `__eq__` that always
  returns `True`, the oracle's answer key read and copied). That second family outranks even a
  hidden-check pass, because each of those mechanisms can produce one.
- `model_backend.py` — the shared inference seam. One `Backend` protocol over local HuggingFace
  and vLLM, a `MockBackend` for the offline tests, and `BedrockBackend` for hosted models we
  cannot run here, which is what makes cross-model measurement possible. The hosted backend needs
  the optional extra (`uv sync --extra bedrock`); `make bedrock-smoke` makes one live call to check
  the credential path without putting the network anywhere near `make test`.
- `bedrock_batch.py` — the same hosted models by a second route: the hosted batch inference service,
  for sweeping an item corpus across a wide roster. No rate limits to fight, and
  every model's job runs concurrently, which is the actual win — one model's 145 records finish
  faster on the live path than batch's ~5-minute queueing floor, but eight models submitted
  at once still land inside 25 minutes. `submit` and `collect` are separate calls so an interrupted
  session resumes from a saved handle rather than re-running the inference; `jagged/sweep.py` is
  the CLI over them, taking the corpus to sweep as a `--items module:attribute` reference. The records are `modelInvocationType="Converse"`, so prompt construction and
  response parsing are literally the live path's functions and there is no second copy to drift.
  Two things it will refuse: fewer than 100 records per job (a hard, non-adjustable quota — small
  and calibration runs belong on the live endpoint), and a model id outside the verified roster table.
- `backend_cli.py` — the shared `--backend` plumbing behind the `episodes` and `channel` CLIs and
  the `interp` probe: one flag set, one place that builds the right sampling
  config per backend, and refusals for the combinations that cannot work (`--top-k` at the hosted
  endpoint, which has no `topK` field; an activation probe pointed at a hosted endpoint). Both
  sampling CLIs also take `--backend mock`, which runs the whole path on canned completions
  for free. The real-execution harness keeps its own backend selection, because a jailed run that
  really executes what the policy emits has no free canned version.
- `terminal_wrench.py` — an offline reader for the Terminal Wrench hack-trajectory corpus.

## The direction

The question is what heavy reinforcement learning against verifiable rewards teaches a model
beyond the tasks it was trained on. The hypothesis worth testing: training across many
environments whose checks are loose teaches two general dispositions rather than two tricks.
One is a prior that the reachable action space is wider than the task description implies —
that there is usually something else in the environment you can touch. The other is a habit of
reading what dimension a situation is being graded on, and optimising that dimension instead
of the task. Dispositions transfer to environments the training never covered; tricks do not.
Whether these do, and whether the obvious interventions change them or merely relabel them, is
the object of study.

`docs/scratch/2026-08-15-reward-hacking-exploration-plan.md` is the authoritative near-term plan
and lists the leads to chase. `docs/scratch/2026-08-15-episode-reward-research-agenda.md` and
`docs/scratch/2026-08-15-complex-hack-substrate.md` are deeper design history, kept for the record
but not the current plan — the substrate spec is superseded / deferred, so mine its ideas rather
than building it wholesale; each is honest about what adversarial review killed in it.
`docs/scratch/future.md` is the owner's sparse index of where this could go, and
`docs/scratch/` holds the reasoning behind it, including the deliberate "speed now,
rigor later" posture for this phase in `docs/scratch/posture.md`; read them before
proposing an experiment.

## Before you build anything here

Two prerequisites are not negotiable, and both already exist.

The isolation gate comes first. Nothing that trains or runs a policy against a
scope-violating environment runs on this host until `make jail-test` passes. That is a hard
prerequisite independent of how the design settles, because an environment that rewards
reading a grader's answer key or writing outside its working directory only makes sense if a
violation lands on bait rather than on something real. Any host running this carries real
credentials and live execution surfaces — private keys, session material, a container socket, the
hooks every agent session on it goes through — and an episode must reach none of them.
`docs/episode-isolation.md` enumerates that surface and what the jail does and does not cover.

The second is that the reward channel stays programmatic. No LLM judge anywhere near it — a
judge in the reward channel means studying judge-bias exploitation instead of environment
exploitation, and it has the useful side effect that no model API key needs to exist near the
episode loop. Judges are for offline analysis only.

Beyond that, read `AGENTS.md` for the operating rules, and in particular the one about
sabotaging a new check and watching it fail before trusting it. A research harness that
reports success while measuring nothing is the failure mode this repo has already hit four
times.
