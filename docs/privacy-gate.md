# The privacy gate: what it detects, and why each exemption exists

This repository's remote is public and its history will be published wholesale, so
`scripts/scan_secrets.py` is the gate between the working tree and that history. `AGENTS.md` states
the rule it enforces; this file is the reference for how the scanner implements it, and in particular
the record of why each of its exemptions is there. The code carries one-line pointers back here
rather than the reasoning, because the reasoning is read once and the code is read every time.

## The two moments it runs, and why each one matters

**Before anything is uploaded or used as training data**, because of an asymmetry. If the episode jail
leaks and a policy reads a real credential, that text lands in a rollout log and from there into the
training set. A secret memorized into a published model adapter is not fixed by rotating the secret:
rotation fixes the credential, it does not unpublish the weights.

**Before anything is committed**, because the remote is public and the history will be published
wholesale. Worth knowing as history rather than as trivia: this script once had no caller at all — no
hook, no make target, no test — so it was a file rather than a gate. Two of the three failures found
since have been of that same shape rather than a missing pattern: a sweep that ran with its canary
detector armed from a hardcoded empty tuple, and hooks that nothing asserted were installed. When
something here looks wrong, check what is armed before checking what is matched.

Callers, in the order they are likely to matter to you:

- `make privacy-scan` over every file `git add -A` would carry, which is the fast standalone version.
- `tests/test_scan_secrets.py::TestTheCommittableTreeIsClean`, the same sweep inside `make test`, so
  the gate cannot be forgotten.
- the `pre-commit` and `commit-msg` git hooks, through `scripts/git_precommit_scan.py`, which scan the
  exact blobs a commit would record rather than the files on disk. `scripts/install_git_hooks.sh`
  installs them and `tests/test_git_hooks_installed.py` asserts they are there.
- `scripts/stage_ship_tree.sh`, advisory only: a false positive must not be able to block a launch.

Two properties hold everywhere and are worth stating before anything else. **A finding never includes
the matched text** — it reports a path, a line number, a detector name and a truncated SHA-256, so a
hit does not copy the secret into a second log. And **no forbidden value appears in tracked code**:
values with no recognisable shape are read at run time from local gitignored files, because a guard
that lists the contraband is itself the leak, which is the exact mistake one version of this guard
made.

## Why exemptions exist at all

Every exemption below traces to one rule: **a gate with standing findings gets waved through.** A
privacy gate that reports the same non-leaks on every run teaches its readers to skim past the
output, and the run where it reports a real one looks identical. So a pattern that would fire on
documentation placeholders, on public product identifiers, or on ordinary English gets those cases
excluded explicitly and recorded here — rather than being left to produce noise, and rather than
being weakened until it catches nothing.

## Detector families

**Credential shapes**: formats recognisable without knowing the value — AWS access key ids, PEM
private-key headers, JWTs, session cookies, bearer tokens. The `session_cookie` pattern matches any
`*session=` or `*sid=` cookie rather than naming a particular employer's single-sign-on cookie: the
general form is both stronger and one fewer internal name in tracked code.

**Machine-local identity**: the half of the privacy rule that looks nothing like a credential — email
addresses, AWS account ids, role ARNs, ECR registry hosts, home-directory paths carrying a username.
Two exemption sets live here, both in the code as named constants. `EXAMPLE_AWS_ACCOUNT_IDS` holds the
account ids AWS reserves for documentation, three of which this repo's own tests use.
`PLACEHOLDER_HOME_USERS` holds home-directory users that name nobody: cloud-image defaults plus the
ones this repo's docs and jail fixtures use on purpose.

Account ids are read two ways, because one shape cannot see both. `aws_account_id` needs an
account-ish word within a few characters of the digits. `aws_account_id_beside_region` catches a bare
twelve-digit run immediately adjacent to a region string, which the keyword form cannot see at all.
Its region fragment uses a closed list of direction words rather than `[a-z]+`, because a permissive
one reads `co-author-1` as a region. Both refuse a longer digit run, which keeps the twelve-digit
float expansions in this repo's trace fixtures (`1/7`, `1/sqrt(3)` and friends) out of the findings.

**Neither catches a bare account id with nothing adjacent to it**, and this is the one gap worth
remembering. A prose mention ("the account is *digits*") has no adjacent region and no adjacent
keyword; a config file whose entire payload is the id has nothing adjacent at all, which was verified
by experiment rather than argued — such a file scanned clean. The only coverage for that shape is
arming the id as a canary value, so an account id belongs in `canary/privacy-values.txt` as an
ordinary literal. An earlier version of this gate read the id automatically out of a machine-local
config file at the repository root; that was removed, because the file has since been moved out of the
tree and because the tracked constant naming it published internal pipeline vocabulary in its own
name. A guard should not have to name the thing it guards against.

**Benchmark material**, read two ways for the same reason account ids are. `benchmark_item_record`
catches a serialized *trace* record, which puts an item's id and its registered answer on one line. An
item *file* as `recoverybench.load_items` reads it is pretty-printed, one key per line, so no single
line carries both and that detector is blind to exactly the file whose publication would contaminate
the benchmark; `benchmark_item_file_findings` is the whole-file check that sees it.

Three decisions inside that whole-file check:

- **JSON** is the corpus shape, and the obvious case.
- **Markdown** is checked because an item pretty-printed into a handoff or scratch note passes the
  line detector just as a corpus does, and notes move under tracked paths. It was **measured before it
  was added** — no markdown in the scan surface carries both keys — so it arrived green rather than
  red, which is the difference between a new detector and a new standing finding.
- **Python stays excluded, deliberately.** The same two keys appear all over this repo's tracked
  tests, where synthetic fixtures build dicts with `true_answer` in them, and flagging those would
  give the gate a standing set of non-findings.
  `test_a_pretty_printed_dict_in_python_stays_silent` pins the exclusion, so widening this guard
  again cannot swallow it by accident.

The finding fingerprints the whole file rather than any field, so the same leaked item is recognisable
across two paths without disclosing anything.

**Canary values**: exact values no shape can describe — a username, a bucket name, an AWS profile
name, an employer, a registered answer's text. Read at run time from `canary/privacy-values.txt`,
which is gitignored; `canary/README.md` documents the file, the entry syntax and how to create it, and
`canary/privacy-values.txt.example` is the template. An unarmed canary detector calls every value it
was meant to guard clean and exits 0, so the scanner logs its token count on every run, warns when
the family is unarmed, and refuses with exit 2 under `--require-canary`.

**Instrument text**: survey and psychometric item text — stems, anchor ladders, allocation payoff
tables — which the privacy rule bans from every tracked path regardless of licence, ours and third
parties' alike. This is the family with the long exemption ledger, and the rest of this file is it.

## Instrument text: sources, grain, and the exemption ledger

### Where the material comes from

`INSTRUMENT_DATA_DIR` and `INSTRUMENT_RETRIEVAL_NOTE` name the two local gitignored places instrument
item text is allowed to live, relative to the repo root. Everything the detector knows comes from
reading these at run time. On a machine that has neither — a fresh clone — the detector is inert, and
says so through its logged counts and its warning rather than by quietly passing everything.

### The phrase grain

`INSTRUMENT_SHINGLE_WORDS` is five. Five words is long enough that a match identifies an item and
short enough that a partial quote — a docstring citing an item's opening — still trips it. An item
below that length is matched whole rather than shingled, because the corpus contains some that short.
Anchors need at least three words and stems at least two: below that a phrase cannot identify
anything, so matching on it would only produce noise.

### `_GENERIC_SURVEY_SHINGLES`, the exemption ledger

**This section describes the exemptions structurally and does not quote them.** The phrases live in
one place, `_GENERIC_SURVEY_SHINGLES` in `scripts/scan_secrets.py`, where each carries a one-line
label; read them there. A prose document that also listed them would duplicate the strings and, worse,
annotate which of them came from an item, which is a fragment of exactly the reconstructability the
whole detector exists to prevent — the set alone says only "excluded", while a narrative saying "these
came from the authored stems" hands a reader more than the code does.

The policy is one rule. A phrase is exempted only when it **cannot identify an item**: either it is
survey or matrix-game boilerplate that appears in essentially every instrument or every simultaneous-
move prompt ever written, or it is ordinary English used in this repository about something else
entirely, or it names how this repository *formats* a payoff cell rather than what any item asked.
Every exempted item keeps its own distinctive shingles armed beside the exempted one, so no item loses
its guard; what is given up is one non-identifying fragment of it.

Five groups, each recorded by what it is, where it was measured, and how many sites:

| group | why it cannot identify an item | measured |
|---|---|---|
| survey-instruction boilerplate | universal across administered instruments | fires on synthetic test wording |
| generic matrix-game framing | occurs in tracked game-frame docstrings and test prose | present before the corpus grew |
| pure function-word answer options | two entries, six tracked sites each, none about a survey item | `git show HEAD:` sweep |
| ordinary-English unobservability prose | used in eval and harness code about unobservability | `git show HEAD:` sweep |
| one payoff-cell format string | names `games.prompts._outcome_block`'s rendering, asserted in `games/tests/test_games_prompts.py:1068` before the items were reworded | `git show HEAD:` sweep |

Two measurements are worth keeping in front of anyone who edits this list. The 2026-08-22 landing of
seven authored families grew the corpus from roughly 1,600 shingles to roughly 6,000, and **every
exemption it required was verified to be present in a tracked file before that landing**, by running
the detector with the new corpus over a `git show HEAD:` copy of every tracked file. So each was an
existing collision rather than an item leaking into code. The alternative was a gate carrying nineteen
standing findings, which is a gate nobody reads.

And the discipline that matters more than the exemptions: when the 2026-08-24 elicitation repair
created three new collisions, only one was exempted. **The other two were reworded in the items
instead.** One would have required exempting a whole sentence of tracked prose, which is too large a
hole to open; the other was an item restating a constraint its own instruction already gave, so the
restatement was simply removed. Exempting is the last resort, not the first.

### The payoff-table matcher's constants

`_ROUND_PAYOFF_BOUND` is 200. A pair whose members are both multiples of ten at or below that bound
collides with ordinary game arithmetic all over this repository, so such a pair cannot identify an
instrument item and is excluded from matching. An item built entirely from them is structurally
unguardable by pattern; its protection is the human read at review time.

`_MIN_MATCHABLE_PAIRS` is 2, because one matched pair could be a coincidence of adjacent numbers.

`_PAYOFF_WINDOW_RUNS` is 200, and the two pairs must sit within that many digit runs of each other. A
pasted payoff table is a run of consecutive number pairs, so even the largest item in the corpus fits
inside the window with indices and prose interleaved, while the coincidental matches that made the
window necessary (an item whose payoffs are small two-digit numbers, matching `uv.lock` and the
arithmetic notebooks) sat 4,000 to 62,000 digit runs apart. Distance is counted in digit runs rather
than lines so that a minified single-line file cannot hide a paste.

`_MAX_PAYOFF_DIGITS` is 7: no instrument payoff has more digits than that, longer runs are blobs
(hashes, encoded data), and converting an unbounded run trips CPython's int-conversion limit on
multi-kilobyte ones — found live, on a scratch artifact carrying a ten-thousand-digit run that
crashed the sweep.

### `_PHRASE_FIELDS`, the schema the extractor reads

This maps each phrase-bearing field of the instrument files' schemas to its own generality rule.
`stem_swapped` and `options` are `authored.json`'s fields — the action-swapped self-prediction stems,
and the choice and anchor option text. Option prose gets the anchors' three-word-minimum grain, since
one- and two-word options are below the matchable grain anyway.

`elicitation_blocks` holds a family's shared closing question, keyed by family name. It is the half of
an item that asks the question, so it is item text on exactly a stem's footing. It also arrives as an
object rather than a string, which is what `_field_phrases` had to learn to flatten: the walker stops
recursing into any key this map recognises, so before that a phrase field holding an object
contributed nothing at all — the field read as zero phrases and the guard then permitted its words in
a tracked file while reporting success.

The walk covers the whole structure rather than the documented schema, on purpose: a schema drift must
not quietly shrink what the detector knows.

## The committing identity

A commit has three text surfaces, and until 2026-09-10 the gate read two of them. File content is
swept; the message draft is read by the `commit-msg` hook; **the author and committer fields were read
by nothing at all.**

That is not a theoretical gap. It was found by rehearsing the squash that will produce the published
artifact, in a throwaway clone: the rehearsal commit came out authored as the owner's real OS username
at their employer's domain, because **a clone does not inherit the source repository's `--local`
identity** and this repository's pseudonymous identity is set with `git config --local`. The clone fell
back to the global identity, which on the research box is the real name and the work address. The one
commit meant to be the clean published artifact would have carried exactly the class the whole cleanup
exists to remove, introduced by the cleanup step itself. The same fallback applies to every commit made
from a fresh clone on any other machine.

`TestTheCommittingIdentityIsSafe` in `tests/test_scan_secrets.py` asks two questions, because either
alone leaves a hole:

- **Is this one of the identities that may appear here?** An allowlist of the three publishing
  identities — the pseudonymous handle the history was collapsed onto, the forge's own merge identity,
  and the synthetic one local archive refs carry. This is the arm that catches a real *name*, which no
  shape detector describes and which is deliberately not a canary token (it is an ordinary forename
  that would match prose). Checked for the identity a commit would carry right now, and for every
  identity reachable from `HEAD` — all of history, because that is what gets published and because it
  costs about 30 milliseconds across 877 commits.
- **Would the scanner object to it?** The same detectors and canary tokens used everywhere else, run
  over each identity string. This is the arm that covers an identity somebody legitimately adds later,
  at a domain no token lists.

One exemption, and it is narrow: an identity *is* an email address by construction, so
`email_address` fires on the pseudonymous handle exactly as it would on a work address. That detector
alone is exempted for the forge's own domains, where an address is a publishing handle rather than a
mailbox. Canary hits are never exempted, so a real value inside a forge-shaped identity still reports,
and a lookalike domain is not a forge domain. Both of those are pinned by tests.

**Reporting an identity finding must not print the identity**, which is easy to get wrong: pytest
rewrites a bare `assert value in allowed` into a report that echoes `value`, and it was observed doing
exactly that with the rehearsal identity during sabotage. The checks therefore reduce to a bool or a
count before asserting, and the failure message tells the reader which git command to run instead.

## Related records

`docs/scratch/gate-coverage-ledger.md` (gitignored) is the honest ledger of which gates here have
been made to fail on purpose and which are merely green, including the sabotages behind this gate.
`docs/episode-isolation.md` covers the jail and the pre-upload use of the scanner.
