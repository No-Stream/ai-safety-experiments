# Reference launch kit for a rented GPU box

The launch kits that rent boxes for this repository live outside it, one gitignored copy per run,
and every rule a box has to follow was re-applied to each copy by hand. That decayed: of the eleven
user-data templates written in the two weeks before this kit existed, seven armed the idle watchdog
without its agenda-complete marker and billed a 40-45 minute idle tail per box (2,400 s measured on
one), one heartbeat grepped a log that exists only after training and so reported nothing for the
whole run, and runners deleted partial eval traces before relaunching, throwing away paid-for
records the cell would have resumed.

This directory is the tracked reference those copies are made from, and the checklist a copy is
compared against:

| file | what it is |
|---|---|
| `user_data_template.sh` | the user-data a box boots with, rendered by `scripts/render_box_userdata.sh` inside `scripts/launch_gpu_box.sh` |
| `runner_skeleton.sh` | the agenda the user-data hands over to, run out of the digest-verified tree |
| `markers.tsv` | every rule the two files carry, as the `grep -E` pattern that proves it is present, with why it exists |
| `check_markers.sh` | compares a rendered user-data or a runner against `markers.tsv`; the launcher runs it at preflight, the test runs it against the reference and against each sabotage |

`tests/test_kit_reference.py` renders the template with sample values through the real render tool,
requires every rule to hold on the result and on the skeleton, and then breaks each rule once
(deletes its line, or injects the forbidden one) and asserts the checker names exactly that rule.
`tests/test_kit_reference_runner.py` then executes the whole runner inside a scratch tree against one
stub program standing in for every binary it calls (`aws`, `uv`, `curl`, `tmux`, `sudo`, ...): a
fresh box sweeps alone, resolves the corpus, trains, evaluates and closes; a relaunch restores the
bank newest-first past a torn checkpoint and skips the sweep; and each of the runner's gates is then
driven to fail (a plan printing `--save-steps 10`, a failed dependency sync, a failed listing or
restore, an unratified priced shape, a stop shutdown behaviour, a failed tmux hand-off, an eval cell
that never closed, too little disk). One scenario hands `--print-corpus` and `--print-plan` to the
real `games.arm_sequence`, which is what keeps the stub's answers honest. A rule or a gate that has
never been watched to fail is not in here.

## The rules, and where each one lives

AGENTS.md states these under "Running expensive things", "Machines and models" and "Every expensive
run saves incremental results and resumes". The six `user-data-native` rows of the table are the
launcher's own preflight: it refuses a launch that lacks the dead-man, the boot patch, the armed
watchdog, the watchdog's S3 destination or the ship guard, and warns on a missing done-marker.
Everything else is a WARNING per missing rule at preflight, so a peer's kit that predates a rule
still launches while saying what it lacks.

- **Two-layer killswitch.** `shutdown -h +<minutes>` is the first line after the log redirect, sized
  by `--deadman-minutes` (layer 1). `idle_watchdog.sh --minutes <n> --done-marker
  /home/ubuntu/AGENDA_DONE` runs as a root unit outside tmux (layer 2); the runner touches the marker
  as its last act, after its final upload has returned, so the box is gone within a minute of
  finishing instead of after the idle threshold. If tmux refuses the hand-off, so that no agenda
  will ever touch the marker, the runner records that and touches it itself. The launcher sets the
  shutdown behaviour to `terminate` and reads it back after launch; the runner reads it back again
  from the inside and halts on a `stop`.
- **Boot patch.** `scripts/ec2_boot_patch.sh` runs out of the verified tree before the job, and is
  the only thing closing the window between launch and a patched OS, because the newest published
  AMI still ships whatever became vulnerable after it was built.
- **Ship guard.** The render inlines `scripts/box_ship_guard.sh`, which fetches the archive this
  launch staged and fails closed on a sha256 or tree-digest mismatch. The runner is a path inside
  that verified tree (`RUNNER_TREE_PATH=`), so it needs no pin of its own, and a kit that stages a
  customised copy in the working tree gets the same guarantee for free.
- **Heartbeat to both keys.** Every 300 s the runner's heartbeat unit writes the same status object
  to `<base>/status/<instance>.txt`, the key the launcher's `HeartbeatS3` tag names, and to
  `<run-prefix>/status/<instance>.txt`, where a reader of one run finds it. Its reward
  and purity greps read the arm log the stage writes live under `artifacts/games/logs/`; the run
  directory's `logs/` copy is forbidden by the table because it exists only after training.
- **Resume-first entry.** Before anything new runs, the runner restores the banked corpus, the run
  directory's small files, and checkpoints newest-first until one is complete (the file list mirrors
  `games.train.missing_checkpoint_files`). Older checkpoints stay in S3; an eval cell pulls its own
  step. Every restore stops the run when it fails, and the checkpoint listing goes through
  `s3api list-objects-v2`, which exits 0 with nothing on a fresh prefix and non-zero on an error
  where `aws s3 ls` does both as 1: a transient S3 error must not look like a fresh run, or the
  trainer starts from step 1 beside the banked checkpoint and overwrites it. A run already at
  `max_steps` makes the trainer log `ALREADY COMPLETE` and touch nothing, so relaunching the identical
  launch is the whole recovery.
- **Training as the two-invocation recipe.** `games.select_prompts` timestamps its corpus, so
  `games.plans.refuse_sweep_beside_a_corpus_consumer` refuses `sweep` in one invocation with the
  stages that read it, and the plan's own default is the sweep alone. The runner follows
  `games.arm_sequence.LAUNCH_RECIPE`: the sweep stage runs on its own, skipped when a corpus already
  resolves (banked at prep under `<run-prefix>/select/`, or swept by an earlier sitting and banked
  right after); then the corpus is resolved with `--print-corpus` and exported as
  `GAMES_ARM_SEQ_CORPUS`; then the consuming stages run as one invocation. Each invocation sits
  behind a CPU `--print-plan` gate. The arm's gate also checks the registered lines (`--save-steps 1`
  and `--max-steps <n>` in the reference; a kit appends its own) as whole argv tokens, so
  `--save-steps 1` does not pass under `--save-steps 10`. That gate can only run once the corpus
  exists, so on a fresh box a registered-line slip is caught after the sweep, whose corpus is banked
  by then, and the relaunch that fixes the kit skips it. A corpus the resolver refuses (empty, or
  more than one) stops the run rather than being swept over.
- **Per-step sync.** `GAMES_ARM_SEQ_SAVE_STEPS=1` and `GAMES_ARM_SEQ_S3_DEST=<run-prefix>/runs`: the
  trainer checkpoints every step and syncs each one as it lands, so a reclaim costs one step and the
  checkpoint interval beats the reclaim cadence.
- **Model cached, then the hub offline.** `snapshot_download` pulls the model, then
  `HF_HUB_OFFLINE=1` before the sweep and the arm, so neither they nor a save spends a network round
  trip.
- **Pace guard as a parameter.** `GAMES_PLAN_STEP_MINUTES` comes from `PLAN_STEP_MINUTES`, a
  per-launch value; a number baked into a runner is forbidden by the table, because the next shape
  inherits it unread.
- **Eval cells via `--sync-dest` and `--summarise-only`, no `rm -f` of a partial trace.** Each cell
  restores its directory from `<run-prefix>/evals/<arm>/` and syncs it back on an interval; a
  relaunch continues the cell, only the missing prompts are generated. A closing pass with
  `--summarise-only` writes the summary of a cell that died between its last generate call and its
  summary write, and refuses an unfinished one, so the agenda's exit code says whether every cell
  closed (`reason=eval-cell-unclosed`). Deleting a `.jsonl` is forbidden by the table.
- **Priced shapes are held.** An on-demand landing on a `p5` (any type matching
  `HOLD_INSTANCE_TYPES`) writes `awaiting-market-ratification` into its work state, holds the
  watchdog off with `KEEPALIVE`, and proceeds only once `<run-prefix>/control/ONDEMAND_RATIFIED`
  exists; after `HOLD_MINUTES` it stands down and the watchdog reaps it. A spot box proceeds.
- **Work state.** `/home/ubuntu/CHAIN_STATE` is seeded by the user-data before the heartbeat can
  observe it and rewritten at every leg and at exit (`exited rc=<n> reason=<why> logs=synced`, or
  `logs=SYNC-FAILED rc=<n>` when the final log upload did not land), so a heartbeat reading `ABSENT`
  means the box died without recording, not that nothing was running.

## Placeholders

Every `@NAME@` in the template must be substituted or the render refuses; the render also refuses a
placeholder named inside a comment. The rule runs the other way too: a `--set` whose `@NAME@` sits on
no live line of the template is refused by name, because the substitution would otherwise drop that
value without a word and the rendered file would be the only record of it (one launch disarmed a kit's
market-price hold with such a `--set`, and the box it had just landed spent the full hold anyway). A
copy of this template therefore has to keep a live use of every name in both tables below, the
launcher's own seven included. The launcher fills those seven itself, and refuses a kit `--set` that
names one of them:

| placeholder | value |
|---|---|
| `@SHIP_TREE@` `@SHIP_CODE_SHA256@` `@SHIP_TREE_DIGEST@` `@SHIP_S3_KEY@` `@SHIP_GUARD@` | the staging record of this sitting and the inlined guard (required by the render) |
| `@SHIP_S3_REGION@` | `--s3-region` / `SHIP_S3_REGION` |
| `@DEADMAN_MINUTES@` | `--deadman-minutes` |
| `@IDLE_WATCHDOG_S3_DEST@` | the run prefix, where the watchdog publishes its arm-time verdict |
| `@RUN_NAME@` | `--run-name` |
| `@RUN_PREFIX@` | `<s3-prefix>/<run-name>` |
| `@SHIP_S3_PREFIX@` | `--s3-prefix` / `SHIP_S3_PREFIX`, the base that holds the shared `status/` directory |
| `@SHIP_HEAD_SHA@` | HEAD at staging, recorded on the box as `GIT_SHA` (a provenance record, never a gate) |

The kit passes the rest with `--set NAME=VALUE`; each names a registered knob of the experiment:

| placeholder | meaning |
|---|---|
| `@IDLE_MINUTES@` | the idle threshold, sized above the box's longest legitimate quiet gap (model load, corpus parsing); the done-marker makes it the fallback rather than the trigger |
| `@ARM@` | the games arm to train (`GAMES_ARM_SEQ_ARM`) |
| `@MODEL@` | the Hugging Face model id (`GAMES_ARM_SEQ_MODEL`) |
| `@MAX_STEPS@` | training steps (`GAMES_ARM_SEQ_MAX_STEPS`) |
| `@EVAL_STEPS@` | comma-separated checkpoint steps to run the battery at; `0` is the un-adapted base |
| `@PLAN_STEP_MINUTES@` | this shape's measured minutes per step, for the pace guard |

Optional runner knobs with defaults, overridden by adding `NAME=value` to the template's last
environment line. These are not `--set` names, since the template carries no `@NAME@` for them and a
`--set` that reaches no placeholder is refused; give a knob a placeholder first if a launch needs to
vary it per box.

| knob | default | meaning |
|---|---|---|
| `TRAIN_STAGES` | `sweep,arm` | the `games.arm_sequence` stages, sweep included; the runner splits the sweep into its own invocation and runs the rest as one (`sweep,regrade,arm` for an arm graded differently from its sweep) |
| `BATTERY_SECTIONS` | empty | eval sections; empty passes no `--sections`, so `games.run_evals`'s own default applies |
| `HOLD_INSTANCE_TYPES` | `^p5` | the regex of shapes held for ratification; p5, p5e and p5en are all priced |
| `HOLD_MINUTES` | `150` | how long a held box waits for the marker before standing down |
| `HF_CACHE_DIR` | `/opt/dlami/nvme/hf` | where the model is cached (`HF_HOME`) |
| `RUN_USER` | `ubuntu` | the agenda's user; the marker and `KEEPALIVE` paths in the template assume it |

`BOX_LOG_DIR` (`/var/log`) and `BOX_BIN_DIR` (`/usr/local/bin`) exist so the runner test can execute
the whole file inside a scratch tree; a box never sets them.

## Launching from the reference as it stands

Nothing account-specific is tracked. The launcher reads the bucket, region and instance profile from
the environment variables it already documents, and the candidate file carries the subnet and
security-group ids; keep all of them in a gitignored wrapper under `/var/tmp/<kit>-prep/`.

```bash
export SHIP_S3_PREFIX=s3://<bucket>/<base> SHIP_S3_REGION=<region> \
  SHIP_INSTANCE_PROFILE_NAME=<profile> SHIP_CANDIDATES=/var/tmp/<kit>-prep/candidates.txt
bash scripts/launch_gpu_box.sh --run-name <run-name> \
  --template scripts/kit_reference/user_data_template.sh \
  --label <run-name> --purpose "<what this box is for>" --deadman-minutes <n> \
  --set IDLE_MINUTES=45 --set ARM=<arm> --set MODEL=Qwen/Qwen3.5-2B \
  --set MAX_STEPS=70 --set EVAL_STEPS=70,0 --set PLAN_STEP_MINUTES=<measured>
```

`--preflight-only` renders and checks without renting anything, and is how to read the marker
report for a kit before its first launch.

## Customising a kit

1. Copy `user_data_template.sh` to the kit directory and edit the last block: anything the runner
   needs beyond the placeholders goes on its environment line. Keep the rendered user-data under
   16,384 raw bytes; the reference renders to 9,405 bytes with the guard inlined (measured
   2026-09-04 through the real render), so a kit has about 6.9 KB for its own additions. Anything
   larger belongs in the runner.
2. To customise the runner, copy `runner_skeleton.sh` into the kit's own directory under
   `scripts/kits/<run-name>/` and point the template's `RUNNER_TREE_PATH=` line at it. That
   directory is gitignored, because a kit carries the bucket, region and profile literals a box
   needs, and gitignored files do not ship in the code archive (`scripts/ship_tree.sh` lists the
   tree with `--exclude-standard`, checked 2026-09-04). So the runner and its checks travel as
   `--data` objects: the launcher records their sha256 in the manifest, and the user-data fetches
   the runner into `RUNNER_TREE_PATH` and verifies it against that manifest before the hand-off.
   The launcher's reference-kit pass will warn that the staged tree has no file at
   `RUNNER_TREE_PATH`; that warning is expected for a kit shipped this way, and the runner is
   compared against the reference by hand with `check_markers.sh runner <path>`. Add the experiment's
   registered pins in the "experiment's environment" block, and for each one a fragment on the arm's
   `plan_gate` call: an extended-regex fragment matched as a whole run of argv tokens, so escape a
   literal dot.
3. Run the launcher with `--preflight-only` and read the report. `protections present` is the
   refusal tier; each `WARN: ... lacks the reference kit rule <name>` line is a rule the copy
   dropped, with the reason it exists. The reference itself produces none.

To compare a file by hand:

```bash
bash scripts/kit_reference/check_markers.sh runner /var/tmp/<kit>-prep/runner.sh
bash scripts/kit_reference/check_markers.sh user-data /var/tmp/launch-gpu-box-<label>/user-data.sh
```

## Adding a rule

Add one tab-separated line to `markers.tsv` (surface, rule, pattern, why), make the reference carry
it, and add the rule's sabotage to `tests/test_kit_reference.py`: a positive rule is broken by
deleting the lines its pattern matches, a forbidden one (pattern starting with `!`) by injecting a
line that matches, and the test's `FORBIDDEN_INJECTIONS` table must name that line. The test
refuses a forbidden rule with no injection, so a rule cannot be added without being watched to fail.
A pattern grep cannot use is a fault of the table (exit 2), never a verdict on the file.

The patterns are tripwires for the decay actually observed, not proofs of absence: the forbidden
`$OUTDIR/logs/arm-` catches the form every kit so far wrote and not `${OUTDIR}/logs/arm-`, and the
two ordering rules (hub offline after the cache, marker after the final upload) are asserted on the
reference only, since a grep cannot see order. A kit that wants to evade the table can; the table is
there for the kit that forgot.
