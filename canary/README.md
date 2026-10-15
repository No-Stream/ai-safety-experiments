# Machine-local values for the privacy scanner

`privacy-values.txt` arms the privacy scanner's exact-value detector. `.gitignore` keeps its contents
out of the public repository. A local `host-manifest.json` from the retired execution/configuration
tripwire is no longer read or updated; it remains ignored if it exists.

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
