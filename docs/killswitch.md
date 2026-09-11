# The rented-box killswitch, and the record of attacking it

Every rented GPU instance arms two independent killswitches before its job starts, because no box may
depend on a human remembering to terminate it. This document is the design record and the sabotage
ledger for the second layer. The tests are `tests/test_idle_watchdog.py`, which drives the real
`scripts/idle_watchdog.sh` against a stubbed box; the fixture machinery is `tests/box_stubs.py`.

The repo's governing rule applies here more than anywhere: a check you have never watched fail is not
yet a check. A killswitch that has never been made to fail is a reassuring log line. So each defect
below was introduced deliberately, watched to go red, and reverted.

## The two layers

**Layer 1, the max-lifetime dead-man switch.** A bare `shutdown -h +<minutes>` sized to the job,
armed in user-data before anything else runs. Deliberately bare: it needs no network, no credentials
and no CLI, so it fires when everything else on the box is broken.

**Layer 2, the idle watchdog.** `scripts/idle_watchdog.sh` terminates the box after it has done no
work for longer than any legitimate inter-job gap. This exists because layer 1 is blind to the
cheapest failure: a box that bootstraps and then never starts its job burns its whole dead-man window
doing nothing. That is not hypothetical — it is what prompted the rule.

Neither layer covers a spot eviction, which runs neither of them.

## Why the teardown calls the EC2 API before falling back to `shutdown`

`InstanceInitiatedShutdownBehavior` is one setting away from a box that merely *stops* on `shutdown`
and bills its root volume indefinitely while the log reads as a clean kill. `TerminateInstances`
releases the instance whatever that setting says, and it leaves a record a later reader can look up,
where a halt is visible only on the disk the teardown destroys.

The invariant that matters more than either benefit: the API path is strictly additive. Every test
asserts `shutdown` still ran, because the API call failing, timing out, or being impossible must
never cost the box its death. A killswitch may not acquire a new way to survive forever, and a
dependency on a network call, a credential and an IAM grant is exactly such a way — the grant in
particular is genuinely absent on any box launched before it landed.

## Five defects pinned in the watchdog, each watched to fail first

- `work_in_flight` read the mere *existence* of a tmux session as work, so an idle detached shell
  held the box forever. The repo's own launch convention creates the session first and then sends the
  job into it, which made the very incident the script's header cites the case that slipped through.
- The terminate branch called `shutdown` and then `exit 0` regardless of the result, so a watchdog
  that could not shut the box down (not root, no logind) stopped watching. Layer 2 was simply gone,
  with nothing retrying and nothing alarming.
- The idle counter advanced once per poll and was compared against a threshold in minutes, so the two
  agreed only at the default 60-second poll. `--poll-seconds 1 --minutes 1` terminated after one
  second, and a longer poll waited far past the threshold.
- The KEEPALIVE escape hatch was hardcoded to one run user's path while `--user` was a flag, so under
  any other run user an operator holding a box deliberately touched a file the watchdog never read,
  and the box was terminated under them.
- The GPU check piped `memory.used` through `head -1`, reading device 0 and no other, so a two-card
  box training on device 1 read as idle and was terminated mid-run. This is the same defect
  `gpu_preflight.vram_mib_across_devices` had already been fixed for.

## The sabotage record

**The teardown path, five attacks, 2026-08-26.** The `timeout` wrapper dropped, after which the
stalled call completed after six seconds and logged a clean success. The fallback `shutdown` gated on
the API call succeeding: five tests red, the box never dying at all. The metadata GET reverted to an
unauthenticated IMDSv1 request. The region hardcoded rather than read from the instance. The
instance-id shape check weakened to a non-empty check, which passed a page of HTML into the
destructive call.

**The arm-time dry run, five attacks, 2026-08-27.** This mechanism exists because a denied teardown
and a working one leave logs that read the same, so the watchdog announces at arm time whether the
grant is really live. Attacks: `--dry-run` dropped, after which the probe issued a real terminate
against a working box; the verdict keyed on exit status instead of response text, which reported a
denied box as authorized and turned both parametrised rows red; the region shape check weakened; an
inconclusive probe made to abort the arm; and the CLI left unresolved on PATH.

**Publishing the arm-time verdict off the box, thirteen attacks, 2026-08-27.** The verdict was
readable nowhere once the box died: it lives on stdout, which the caller redirects onto the root disk
the watchdog destroys, and every launcher kit's off-box channel for that file is a short `tail` inside
a heartbeat object that the loop's first idle line evicts it from. Attacks: the instance id dropped
from the key so every box overwrites one object (both parametrised rows red); the `%/` trim dropped
(the trailing-slash row only, which is why that row exists); the record opened after the arm lines
rather than before; `log` stopped appending at all, giving an upload of a window without the verdict
and then of an empty file; publication moved inside the poll loop, whose payload then carried
`idle 0s/0s` and reproduced the tail-window bug exactly; the unset-destination guard removed;
`${IDLE_WATCHDOG_S3_DEST:-}` written bare so `set -u` killed the watchdog at arm time; a failed upload
made to gate the arming, which turned both killswitch-invariant tests red and stopped the box dying at
all; the `timeout` bound dropped, after which the run took 6.2 s against a 1 s bound; a key built from
an unresolved identity; the recording flag never cleared; the record's writability guard removed; and
the `s3://` shape guard removed, which reported a local file copy as PUBLISHED.

## Two test-harness details that are easy to get wrong

`idle_poll_lines` matches `idle 12s/60s` rather than a bare `" idle "`, because the loose pattern also
matched the arming line `terminate after 1 idle minutes`. While it did, the wall-clock test's `>= 2`
was satisfied by a single real poll.

`--minutes 0` is what keeps the suite fast: the threshold is a duration, so zero means "as soon as it
is idle" and the terminate branch is reached on the first poll. The wall-clock test is the one case
needing a real threshold, and it asserts the absence of a termination that today's code performs
within a second.
