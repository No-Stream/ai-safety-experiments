# The EvalPlus case bake

Design of record for `reward_hacking/harness/evalplus_etl.py`, which turns two pinned EvalPlus
releases into the case file `reward_hacking/harness/tasks_evalplus.py` reads. The module keeps a
summary docstring and points here.

This document names no problem, quotes no case, and gives no expected answer. The curated set is
recorded in the module as release ids alone, which are pointers into a public release; a per-item note
saying what a problem is or where it is weak would make the set recognisable without that lookup,
which is benchmark material a public repository must not publish. The same rule governs this file.

## Why it runs by file path and not `-m`

This script deliberately imports nothing from this repo. Any import of `reward_hacking.harness`
builds the task registry, and building the registry reads the very file the script writes, so a `-m`
invocation could never regenerate a missing or broken case file — which is exactly when it is needed.
The two things it therefore restates rather than imports are the output path and the schema version.
The schema version being written here and checked independently in `tasks_evalplus` is the point of
having one: bump the writer without the reader and the reader refuses the file loudly.

## Four decisions, each load-bearing

**It reads the release archives directly, so EvalPlus is not a dependency of this repo.** The
`.jsonl.gz` files behind `evalplus.data.get_human_eval_plus` were checked field-for-field against
what that function returns — on 2026-08-16, over all 164 HumanEval tasks, with `prompt`,
`entry_point`, `canonical_solution`, `base_input`, `plus_input` and `atol` identical for every task —
so `urllib` plus `gzip` plus `json` gets the same data without adding a package to the lockfile.

**Expected outputs are computed under the interpreter the episode jail runs, not under the repo's
venv.** That interpreter is the one the graders will execute a submission with, so it is the one
whose answers they must compare against; computing them anywhere else risks baking an expectation the
jail cannot reproduce. It is asked for rather than written down (`episode_jail.sh
--print-jail-python`), because the jail resolves the first interpreter it finds past its version
floor and that answer moves when the host's interpreters do. Both halves of the recorded value matter
to a later reader: the path says which interpreter the jail resolved when the file was baked, and the
version says whether the answers still hold under whatever it resolves now.

**Nothing is baked until its oracle has been watched to work.** For every curated problem the bake
runs the shipped canonical solution over every case it is about to bake and requires all of them to
pass, requires every argument and expected value to survive a `repr` → `ast.literal_eval` round trip
under that same interpreter, and requires the whole set to finish well inside the jail's leash.

A problem that fails any of those aborts the whole bake rather than being baked or skipped: a case
file short of a problem is exactly the failure `tasks_evalplus.load_problems` exists to catch, and
carrying on past the bad problem would produce one. This is not ceremony — the curation sweep found a
problem whose shipped canonical solution raises on one of its own inputs, and this check is what
caught it.

**Plus cases are capped.** EvalPlus ships up to about 1000 extra inputs per problem, which would put
megabytes of literals in the repo for no gain: 250 hidden cases against 3-26 visible ones is already
a 10-80x gap, far past what any hardcoded ladder can cover. The cap keeps the first N in release
order, so it is deterministic, and it is a property of the baked file rather than of the grader, so
raising it is a re-run of the script.

## How the curated set was chosen

Picked by hand from a sweep over all 542 problems in the two releases, under two hard filters applied
before taste.

Nothing whose grading needs one of EvalPlus's special oracles — set-equality, not-None, or its
polynomial-root check. Porting those to a stdlib comparator is fragile, and a comparator that
disagrees with the oracle would manufacture fake hacks, which is the one failure this harness must
not have.

And no near-one-liner: a problem the policy can solve honestly in a single line gives it no reason to
touch the grader, so it measures nothing. What survives spans the effort gradient on purpose, from
short numeric routines to multi-step dynamic programmes over structured inputs.
