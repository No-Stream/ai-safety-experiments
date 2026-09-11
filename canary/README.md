# Two machine-local files live here, and neither is tracked

`privacy-values.txt` arms the privacy scanner's canary detector; `host-manifest.json` is the code-
execution tripwire's baseline. They are unrelated mechanisms that share this directory because both
are keyed to one machine. `.gitignore` covers `canary/*` and keeps only this README and the
`privacy-values.txt.example` template.

## `privacy-values.txt` — the values no shape detector can describe

`scripts/scan_secrets.py` recognizes credential shapes, AWS account ids and ARNs, email addresses,
home-directory paths and benchmark item records from their shape alone. A username, an employer, an
internal hostname, a work repository name, a bucket, an AWS profile name and a registered benchmark
answer have no shape to recognize, so the scanner matches them as exact values read from this file at
run time — never spelled out in tracked code, because a guard that lists the contraband is the leak.

**The gate is materially weaker without it, and does not look weaker.** With no file the scanner runs
shape-only, exits 0 and reports `clean`, which is exactly what a tree carrying none of those values
also looks like. That silence is how roughly a dozen employer-internal values sat in the tracked tree
on 2026-09-10 while `make privacy-scan` reported clean. So create it on any machine that holds real
values, before the first commit:

```bash
cp canary/privacy-values.txt.example canary/privacy-values.txt   # then replace each class name
make privacy-scan                                                # reports the token count it armed
```

The example file carries class names only and documents the entry syntax: a bare line is a literal
substring of at least eight characters, a `word:` prefix matches case-insensitively between word
boundaries and admits values down to four, and either form takes a trailing `unless=a,b,c` list of
surroundings that excuse a match. Read the token count `make privacy-scan` prints on every run; the
count is there because a file that loads nothing calls every value in it clean. `--require-canary`
turns an unarmed run into exit 2, which means "did not verify" rather than "clean".

An AWS account id belongs here as an ordinary literal, and there is no cleverer route: no shape
detector can see a bare twelve-digit id, because both account-id patterns need something adjacent to
the digits and a config file that contains only the id has nothing adjacent. Verified by experiment —
a file whose entire payload was a bare id scanned clean.

Add values as they are **confirmed** rather than guessed. A value already present somewhere in the
committable listing turns `make privacy-scan` red until that file is fixed, which is the gate
working; a value that is simply wrong is a permanently red gate everyone learns to ignore.

## The tripwire baseline lives here and is not tracked

`scripts/canary_manifest.py` hashes the host's code-execution surfaces — Claude Code hooks and the
`settings.json` that registers them by path, systemd user units, the user crontab's targets, shell
and tool rc files — and compares them against a baseline it keeps at `canary/host-manifest.json`.
Drift means something wrote to a surface that buys arbitrary code execution the next time anything
routine happens, which is the jail-misconfiguration failure we should actually expect.

**The baseline is deliberately untracked.** Two reasons, and the second is the load-bearing one:

1. Every key in it is an absolute path under one machine's home directory and every value is a
   hash of a file that exists only there. Committed, it would be a list of one developer's home
   directory contents in a public repository, and it would be 100% drift on any other machine.
2. A baseline shared through git is a baseline that gets updated by `git pull`, which is exactly
   the property a tripwire must not have. Re-baselining has to be a deliberate local act.

So a fresh clone has no baseline, and `make canary-check` will say so and exit 2 rather than
pretend. Create one after you have looked at what is on the box:

```bash
python3 scripts/canary_manifest.py --update   # record the current state as the baseline
make canary-check                             # report drift against it, exit 1 if any
```

Re-baseline the same way (or `make canary-update`, which runs the same command) after any change
you made on purpose — a new hook, an edited `.zshrc`. Re-baselining is owner-run only: agents
report drift, never clear it.
If drift shows up and you did not cause it, treat it as a possible jail escape rather than noise.
