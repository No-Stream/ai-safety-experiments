# All targets run inside the uv-managed .venv pinned by uv.lock. `--frozen` makes them
# fail rather than silently re-resolve if pyproject.toml and uv.lock have drifted apart.
# --directory makes the venv discoverable regardless of the caller's cwd. It used to be load-
# bearing under `smoke` because the resource limiter ran jobs from $HOME; it now inherits the
# caller's directory, so this is belt-and-braces rather than required.
UV := uv run --frozen --directory $(CURDIR)
RUFF := $(UV) ruff

# Serves canary-check, the one target that must run under the host's own interpreter. Absolute
# because make inherits the caller's PATH: with a conda env active, bare `python3` resolves to
# conda's 3.13, so the tripwire would run under whichever environment happened to be active rather
# than the 3.9 it is written for. Measured on this box, not theoretical.
# tests/test_interpreter_compat.py is what keeps the scripts that run under it 3.9-clean.
SYSTEM_PYTHON := /usr/bin/python3

.PHONY: setup setup-gpu hooks test test-select test-changed format lint typecheck typecheck-ty ci all smoke bedrock-smoke harness-bedrock-smoke throughput throughput-sweep games-smoke games-select games-train games-evals games-screen jail-test canary-check canary-update privacy-scan tmp-check clean

setup: hooks
	uv sync --frozen

# The staged-blob privacy scan (pre-commit) and commit-message scan (commit-msg). Machine-local
# (.git/hooks is untracked), so setup installs them; run this alone after changing the wrappers.
# They scan INDEX blobs, not disk: the 2026-08-22 leak rode a staged index the disk scan called clean.
hooks:
	scripts/install_git_hooks.sh

# GPU boxes: `uv sync` makes the venv exactly match the request, so a bare `make setup` run after
# an --extra vllm install REMOVES vllm, flashinfer and ninja without a word. Box bootstrap uses
# this target so the extra cannot drift out from under a colocate run.
setup-gpu:
	uv sync --frozen --extra vllm

# RLVR_SMOKE=1 enables the tiny-GPT2 CPU test, which downloads a ~2 MB model. The full suite is
# sharded over TEST_WORKERS pytest-xdist processes: serially it takes ~15 min, and that cost is
# hundreds of independent multi-second tests rather than a few long ones, so 8 workers bring it to
# ~3.5 min on this box (4 workers, ~4.8 min). `--dist loadfile` keeps every test of a module on one
# worker, so a module-scoped fixture is built once per module rather than in every worker that draws
# a test from it, and the pass/fail set matches the serial run (checked by diffing junit XML from
# both). `nice` because the box is shared with other sessions. test-select stays serial: it narrows
# to a file or a -k filter, where worker start-up costs more than it saves; put -n in ARGS when a
# wide selection wants it. Fewer workers when the box is busy:
#   make test TEST_WORKERS=4
TEST_WORKERS ?= 8

# GATE_TMUX=1 runs the gate in a detached tmux session teed to a log and returns immediately with the
# command to poll, instead of holding the terminal for minutes:
#   make ci GATE_TMUX=1
#   make test GATE_TMUX=1 GATE_TMUX_NAME=my-gate
# It exists because an agent driving this box is killed by its own harness after 180 s with no output
# (the Workflow tool's stall watchdog), and because a backgrounded shell dies on SIGHUP while the work
# it started keeps burning the box. scripts/tmux_run.sh carries the pattern and the reasoning; the
# command's exit status is the log's last line (EXITCODE=<n>), never make's, because make returns as
# soon as the session exists. The default name is a UTC timestamp so two sessions on this shared box
# do not collide over one log; the helper refuses a name that is already taken either way.
GATE_TMUX ?=
GATE_TMUX_NAME ?= gate-$(shell date -u +%H%M%S)
GATE_RUN = $(if $(GATE_TMUX),scripts/tmux_run.sh $(GATE_TMUX_NAME) --,)

test:
	$(GATE_RUN) env RLVR_SMOKE=1 nice $(UV) pytest -n $(TEST_WORKERS) --dist loadfile

test-select:
	RLVR_SMOKE=1 $(UV) pytest $(ARGS)

# The fast iteration gate: only the test files the working tree's changes against HEAD can reach,
# mapped by directory (scripts/test_changed.py carries the table: a games/ edit runs games/tests, a
# changed test file runs itself, grpo/ runs every root that imports it), sharded like `make test`
# with the workers capped at the file count. Nothing changed runs nothing. NOT a commit gate: the
# mapping is by directory rather than import graph, so a cross-component break (sociology imports
# reward_hacking.model_backend) is exactly what it cannot see; `make ci` stays the gate before a
# commit. `make test-changed TEST_CHANGED_ARGS=--list` prints the selection without running it.
TEST_CHANGED_ARGS ?=

test-changed:
	$(GATE_RUN) env RLVR_SMOKE=1 nice $(UV) python scripts/test_changed.py --workers $(TEST_WORKERS) $(TEST_CHANGED_ARGS)

format:
	$(RUFF) format .
	$(RUFF) check --fix .

lint:
	$(RUFF) format --check .
	$(RUFF) check --no-cache .

# basedpyright is the gating type checker: strict mode with the third-party-Unknown rules
# turned off (see [tool.basedpyright] in pyproject.toml), so every diagnostic is a real
# first-party issue. This is part of `make ci`.
#
# --threads is worth the flag: measured in a detached worktree at HEAD, three runs each, 53/54/59 s
# single-threaded against 25/22/21 s at 8 threads, with byte-identical output on a green tree and on a
# tree carrying planted errors. A bare `--threads` (basedpyright picking the count) came in at 37-38 s
# on this 90-core box, so the count is pinned rather than left to the tool. Lower it when the box is
# busy: the run costs ~190 s of CPU at 8 threads against ~65 s single-threaded.
TYPECHECK_THREADS ?= 8

typecheck:
	$(UV) basedpyright --threads $(TYPECHECK_THREADS)

# ty (Astral, pre-1.0) runs ADVISORY only -- deliberately NOT in `make ci`. It folds real
# optional-access findings and third-party Unknown-attribute noise into one `unresolved-attribute`
# rule with no way to split them, so gating on it would be an always-red gate (~170 diagnostics,
# most of them torch/transformers noise). Kept as a cheap second opinion; run it when curious.
typecheck-ty:
	$(UV) ty check

# The full gate: everything that must be green before calling work done. `lint` already runs
# `ruff format --check`, so this is format-check + lint + types + tests.
#
# The three gates read the tree and write nothing either of the others reads, so they run CONCURRENTLY
# through make's own job server rather than one after another, which hides the type check inside the
# suite's runtime instead of adding to it. Status is unchanged and is still the AND of all three: with
# -j, a failing gate makes this make exit non-zero and name the target that failed, and no gate's
# result is inspected or summarised by a recipe here, so there is nothing that could report green over
# a red gate. What did change is that a red lint or type check no longer spares you the suite: all
# three start at once, so make waits for the unfinished jobs before exiting. The failing gate's
# diagnostics still land in the stream within a second or two of the failure, which is what a polled
# `GATE_TMUX=1` run reads, and `make lint typecheck` remains the way to ask the cheap pair alone.
#
# NO --output-sync, deliberately, and its cost is accepted: the three gates' output can interleave
# mid-line while they overlap. Every mode of that flag buffers a command's output until the command
# ENDS -- measured here, `--output-sync=line` means each recipe line, not each line of text -- so the
# suite would print nothing for its whole five minutes, a polled log would be silent, and a run killed
# mid-suite (a reclaim, a watchdog, an impatient Ctrl-C) would lose the output that says where it was.
# Interleaving is bounded in exchange: lint finishes in ~0.25 s and the type check in ~22 s, so only
# the first moments of the suite share the stream at all. --no-print-directory drops the two "Entering
# directory" lines that -j would otherwise add.
#
# `GATE_TMUX=` on the recursive call is load-bearing rather than tidiness: a variable set on the
# command line propagates into sub-makes, so under `make ci GATE_TMUX=1` (where GATE_RUN has already
# put this whole invocation inside one tmux session) the `test` gate would otherwise try to open a
# second session under the same name, and scripts/tmux_run.sh would refuse it and fail the gate.
#
# MAKEFLAGS is forwarded for the opposite reason: make hands a sub-make the command line's own
# variables through the MAKEFLAGS environment variable, and a tmux session does not inherit the
# launching shell's environment at all (its environment is the tmux server's). So without this,
# `make ci GATE_TMUX=1 TEST_WORKERS=4` would run the suite at 8 workers and say nothing -- measured,
# not hypothetical. The inner command line still wins over what MAKEFLAGS carries, which is what keeps
# the GATE_TMUX= reset above effective.
#
# One consequence of the recipe containing $(MAKE): make runs this line even under --dry-run, so
# `make -n ci GATE_TMUX=1` really does open a session. The inner make is dry (MAKEFLAGS carries the n),
# so that session's log ends EXITCODE=0 for a run in which no gate executed -- read a dry run's log as
# nothing but the command line it would have used.
ci:
	$(GATE_RUN) env MAKEFLAGS='$(MAKEFLAGS)' $(MAKE) --no-print-directory GATE_TMUX= -j3 lint typecheck test

all: format lint test

# Gate for the scope-violation research environment: proves the episode jail contains its
# occupant AND that the assertion suite has teeth (see docs/episode-isolation.md). Deliberately
# NOT $(UV): the script asks episode_jail.sh which interpreter the jail resolves and runs the
# assertions under that one, both inside the jail and outside it for the negative control, so the
# two runs compare like with like.
jail-test:
	./scripts/run_jail_tests.sh

# Tripwire over the host's code-execution surfaces (hooks, crontab targets, systemd user
# units, rc files, settings.json). Re-baseline with canary-update after an intentional change.
canary-check:
	$(SYSTEM_PYTHON) scripts/canary_manifest.py

# Owner-run only: re-baselining clears drift, so agents report drift and never run this (canary/README.md).
canary-update:
	$(SYSTEM_PYTHON) scripts/canary_manifest.py --update

# Privacy gate over exactly what the public remote carries: every tracked file, every detector
# (credential shapes, account ids, ARNs, emails, home paths, benchmark item material). Run it
# before a commit that touches docs or tests. tests/test_scan_secrets.py runs the same sweep inside
# `make test`, so this target is the fast standalone version rather than the only wiring.
#
# The values no pattern can describe -- a username, a bucket, an AWS profile, an employer, a
# registered answer -- go one per line in PRIVACY_CANARY_FILE, which lives under the gitignored
# canary/ directory and is matched literally. Absent, the scan runs shape-only and says so by
# reporting 0 canary tokens; the count is printed on every run for that reason.
# Point it at run output before an upload instead:
#   $(SYSTEM_PYTHON) scripts/scan_secrets.py artifacts/harness --canary-file canary/privacy-values.txt
PRIVACY_CANARY_FILE ?= canary/privacy-values.txt
PRIVACY_SCAN_ARGS ?= $(if $(wildcard $(PRIVACY_CANARY_FILE)),--canary-file $(PRIVACY_CANARY_FILE),)

# --others --exclude-standard is load-bearing, not thoroughness: `git ls-files` alone lists the
# index, so a brand-new file is invisible until it is committed -- which is after the leak is in the
# history this gate exists to keep clean. This lists what `git add -A` would carry, .gitignore
# honoured, so a new file is scanned before it is ever committed. Watched to fail with a planted
# home path in a new file, which the index-only version passed.
#
# The file-count floor and pipefail together are what stop a truncated listing reading as a clean
# tree: without them the target's status is xargs's alone, so a `git ls-files` that died partway
# would hand scan_secrets.py a short list -- or none -- and exit 0. The floor mirrors that guard in
# tests/test_scan_secrets.py, which asserts the listing is over 100 paths before it scans.
PRIVACY_SCAN_MIN_FILES ?= 100

privacy-scan: SHELL := /bin/bash
privacy-scan: .SHELLFLAGS := -o pipefail -c
privacy-scan:
	@count=$$(git ls-files --cached --others --exclude-standard | wc -l); \
	if [ "$$count" -lt $(PRIVACY_SCAN_MIN_FILES) ]; then \
		echo "ERROR: the listing returned $$count paths, under the floor of $(PRIVACY_SCAN_MIN_FILES)." >&2; \
		echo "       A truncated file list scans clean and means nothing; refusing to report a pass." >&2; \
		exit 1; \
	fi; \
	echo "scanning $$count committable files"
	git ls-files -z --cached --others --exclude-standard \
		| xargs -0 $(SYSTEM_PYTHON) scripts/scan_secrets.py $(PRIVACY_SCAN_ARGS)

# Pressure on /tmp, a RAM-backed tmpfs shared by every concurrent session on this box. The inode
# count is the figure that bites: it is capped separately from the byte space, so a tree of small
# files can exhaust the file slots while `df -h` still shows room, and once they are gone no shell
# on the box can start (2026-08-21). The third line names the directories to reap.
tmp-check:
	@df -i /tmp | tail -1
	@df -h /tmp | tail -1
	@du -x --inodes -d1 /tmp 2>/dev/null | sort -rn | head -12

# A short GRPO run on the real GPU to confirm the training path still works end to end.
# Guarded by the resource limiter because a wedged GPU job takes the box with it.
smoke:
	scripts/resource-limits.sh --gpu -t 12m -- \
		$(UV) python -m grpo.smoke

# One live Bedrock Converse call: proves the credential path, request shape and response parsing
# against a real endpoint for a fraction of a cent. Deliberately outside `make test`, which stays
# offline and credential-free. Needs the optional extra, which --extra installs on first run.
# Needs REWARD_HACKING_BEDROCK_PROFILE exported to name an AWS profile with Bedrock access: a
# profile name is machine-local and this remote is public, so it is read from the environment and
# unset is a loud refusal rather than a default. `--profile ""` takes the ambient chain instead.
# Override the model or ask for reasoning, e.g.
#   make bedrock-smoke BEDROCK_MODEL=global.openai.gpt-5.6-luna BEDROCK_ARGS="--reasoning-effort low"
BEDROCK_MODEL ?= openai.gpt-oss-120b-1:0
BEDROCK_ARGS ?=

bedrock-smoke:
	uv run --frozen --directory $(CURDIR) --extra bedrock \
		python -m reward_hacking.model_backend --model-id $(BEDROCK_MODEL) $(BEDROCK_ARGS)

# Seconds per optimizer step at a stated config. Defaults are the reference profile from
# docs/scratch/compute-budget-model.md; override any of them on the command line, e.g.
#   make throughput MODEL=Qwen/Qwen3.5-2B PROMPTS=4 PROMPT_TOKENS=1024 COMPLETION_TOKENS=1024
# Note that the reference profile does NOT fit on this box's 24 GB L4 -- see
# docs/scratch/measured-throughput.md for what does. Both targets go through the resource
# limiter, which is easy to forget when invoking the module directly.
MODEL ?= Qwen/Qwen3.5-4B
PROMPTS ?= 8
GROUP ?= 8
PROMPT_TOKENS ?= 2048
COMPLETION_TOKENS ?= 2048
# Sequences per training forward/backward. Empty means one prompt group, which is what TRL
# would do; it is also the knob that decides whether a configuration fits, independently of
# the episode count, so it is exposed here rather than left to a code edit.
MICRO_BATCH ?=
THROUGHPUT_TIMEOUT ?= 120m
SWEEP_PLAN ?= episode-ceiling-4b

# One real-execution harness episode driven by a hosted Bedrock policy: the model generates
# <run> blocks, this box executes them for real inside the jail, both graders run. Costs cents and
# needs the bedrock extra, so it stays out of `make test` and takes a single task by default.
# `env PATH="$$PATH"` is load-bearing, not decoration: systemd-run hands the job the user manager's
# minimal PATH, which drops the AWS profile's credential-process helper, so boto3 dies with a
# FileNotFoundError before any call is made. The limiter stays OUTSIDE; the harness composes
# limits-outside/isolation-inside again per jailed command, which nests fine (sibling scopes).
HARNESS_TASK ?= sum-ledger
# No --temperature: the default model is the reasoning model global.openai.gpt-5.6-luna, which
# rejects any non-default temperature with a ValidationException (it "worked" only because 1.0 is
# its default). Omitting the field is the verified Converse request shape for reasoning models.
HARNESS_ARGS ?= --max-turns 6 --reasoning-effort low

harness-bedrock-smoke:
	scripts/resource-limits.sh -t 10m -- env PATH="$$PATH" \
		uv run --frozen --directory $(CURDIR) --extra bedrock \
		python -m reward_hacking.harness --backend bedrock --task $(HARNESS_TASK) $(HARNESS_ARGS)

throughput:
	scripts/resource-limits.sh --gpu -t $(THROUGHPUT_TIMEOUT) -- \
		$(UV) python -m grpo.throughput --model $(MODEL) -P $(PROMPTS) -G $(GROUP) \
			--prompt-tokens $(PROMPT_TOKENS) --completion-tokens $(COMPLETION_TOKENS) \
			$(if $(MICRO_BATCH),--micro-batch-size $(MICRO_BATCH),)

# GRPO on the one-shot matrix games (docs/scratch/2026-08-16-game-theory-rl-plan.md). One arm is one
# training run; `games.train --arm` names it and the registry holds the game and grading. Batch
# size, group size and gradient accumulation are NOT set here: games.train derives them from the
# VRAM it finds at startup, so the same command runs on the L4 and on a rented L40S. Override any
# variable on the command line, e.g.
#   make games-train GAMES_ARM=twin-pd-self GAMES_MODEL=Qwen/Qwen3.5-4B GAMES_CORPUS=<jsonl>
GAMES_ARM ?= twin-pd-group
GAMES_MODEL ?= Qwen/Qwen3.5-2B
# The smoke tier is its own variable, not GAMES_MODEL, because the two answer different questions:
# GAMES_MODEL is the model under study, this is the cheapest model that executes the same code.
# Qwen3-0.6B rather than a Qwen3.5 tier for a measured reason: 0.6B is the only checkpoint in the
# ladder whose template does NOT prefill `<think>`, and every Qwen3.5 tier tested (0.8B, 2B, 4B)
# fails to terminate its thinking on a game prompt at any budget we can afford -- 0.8B parsed 1 of
# 32 rollouts at 2048 tokens, where 0.6B parses 28 of 32. So 0.6B is the tier that can actually go
# green. That does mean the smoke skips the prefilled-think and linear-attention paths a real arm
# uses; see docs/scratch/qwen35-runaway-deliberation-2026-08-17.md. Override per run with
# GAMES_SMOKE_MODEL=<hub id> once a tiny tier that terminates is settled.
GAMES_SMOKE_MODEL ?= Qwen/Qwen3-0.6B
GAMES_GAME ?= twin-pd
GAMES_GRADING ?= group-mix
GAMES_SAMPLES ?= 8
# No default: an output cap is part of the measurement, and a cap below the model's own
# measured termination budget silently drops prompts rather than failing (2026-08-17: all 64).
# select_prompts derives the floor from games/termination.py; override only to raise it.
GAMES_SELECT_OUT ?= artifacts/games/select
GAMES_SELECT_TIMEOUT ?= 90m
GAMES_TIMEOUT ?= 240m
# A selected corpus from games-select. Empty means generate prompts fresh, which skips selection
# and leaves more of the batch with no within-group reward disagreement to learn from.
GAMES_CORPUS ?=
GAMES_ARGS ?=

# Minutes on the 0.6B plumbing tier: does the whole path execute, corpus to checkpoint to summary.
# Same entry point as a real run, small numbers -- a separate smoke script would exercise a
# separate code path, which is the one thing a smoke run must not do.
games-smoke:
	scripts/resource-limits.sh --gpu -t 15m -- \
		$(UV) python -m games.train --arm $(GAMES_ARM) --model $(GAMES_SMOKE_MODEL) --smoke \
			$(GAMES_ARGS)

# The baseline behavioural sweep, which doubles as the "before" eval. The sampler flags are
# load-bearing rather than decoration: GRPO learns only from within-group disagreement, so the
# sweep has to sample at the *training* sampler (temperature 1.0, top-p 1.0, top-k off, thinking
# on -- GRPOConfig's own defaults) or it selects prompts on a distribution training never sees.
games-select:
	scripts/resource-limits.sh --gpu -t $(GAMES_SELECT_TIMEOUT) -- \
		$(UV) python -m games.select_prompts --game $(GAMES_GAME) --grading $(GAMES_GRADING) \
			--model $(GAMES_MODEL) --backend hf --samples-per-prompt $(GAMES_SAMPLES) \
			--thinking --temperature 1.0 --top-p 1.0 --top-k 0 \
			$(if $(GAMES_MAX_NEW_TOKENS),--max-new-tokens $(GAMES_MAX_NEW_TOKENS),) --out-dir $(GAMES_SELECT_OUT) \
			$(GAMES_ARGS)

# Screen a candidate model for the property that decides whether it can run these arms at all:
# does its thinking TERMINATE on a strategically underdetermined prompt, or does it deliberate past
# any budget we can afford? Measured, because a leaderboard score cannot answer it -- Qwen3.5-2B and
# 4B both parse 0 of N with thinking on at 2048 tokens, while the same 2B with thinking off parses
# 14 of 16 in a nine-token median. Screen before committing a ladder tier:
#   make games-screen GAMES_SCREEN_MODEL=Qwen/Qwen3.5-9B
#   make games-screen GAMES_SCREEN_MODEL=<id> GAMES_SCREEN_ARGS="--no-thinking --budgets 512"
GAMES_SCREEN_MODEL ?= Qwen/Qwen3.5-2B
GAMES_SCREEN_ARGS ?=

games-screen:
	scripts/resource-limits.sh --gpu -t 45m -- \
		$(UV) python -m games.screen_thinking --model $(GAMES_SCREEN_MODEL) $(GAMES_SCREEN_ARGS)

games-train:
	scripts/resource-limits.sh --gpu -t $(GAMES_TIMEOUT) -- \
		$(UV) python -m games.train --arm $(GAMES_ARM) --model $(GAMES_MODEL) \
			$(if $(GAMES_CORPUS),--corpus $(GAMES_CORPUS),--generate-fresh) $(GAMES_ARGS)

# The eval battery over trained checkpoints (games/run_evals.py): merges each LoRA checkpoint onto
# its recorded base, plays the games (trained and never-trained), runs the decision-theory probes
# and the arithmetic canary, and renders the arm's report. Arm, step, base model and thinking mode
# are derived from the run's own artifacts; traces land under
# artifacts/games/evals/<arm>/step-<step>.jsonl and an existing trace is refused, never
# overwritten. 0 in GAMES_EVAL_STEPS means the un-adapted base model.
#   make games-evals GAMES_EVAL_RUN_DIR=artifacts/games/runs/twin-pd-self-2b-plumbing
#   make games-evals GAMES_EVAL_RUN_DIR=... GAMES_EVAL_STEPS=0,5,70 GAMES_EVAL_ARGS="--sections game-behavior"
GAMES_EVAL_RUN_DIR ?=
GAMES_EVAL_STEPS ?=
GAMES_EVAL_TIMEOUT ?= 480m
GAMES_EVAL_ARGS ?=

games-evals:
	scripts/resource-limits.sh --gpu -t $(GAMES_EVAL_TIMEOUT) -- \
		$(UV) python -m games.run_evals \
			$(if $(GAMES_EVAL_RUN_DIR),--run-dir $(GAMES_EVAL_RUN_DIR),) \
			$(if $(GAMES_EVAL_STEPS),--steps $(GAMES_EVAL_STEPS),) \
			$(GAMES_EVAL_ARGS)

throughput-sweep:
	scripts/resource-limits.sh --gpu -t 240m -- \
		$(UV) python -m grpo.throughput_sweep --plan $(SWEEP_PLAN)

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path './.venv/*' -prune -exec rm -rf {} +
