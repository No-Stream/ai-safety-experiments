# scratch

A low-ceremony place to leave notes. In-progress thinking, half-formed ideas, dead ends worth
recording so nobody repeats them, small things you worked out that would otherwise be
re-derived from scratch next time. This is the **single** scratch location: plans, handoffs,
preregistrations, and notes all live here (a former top-level `notes/` was folded in on
2026-08-18) — do not start a second one.

**The contents of this directory are deliberately not tracked in git** (see the `docs/scratch/*`
entry in `.gitignore`): this repo is public and working notes are not written with an audience in
mind. Only this README travels with the repo. Two consequences worth knowing. Notes here do not
sync between machines with the repo, so copy them across by hand if you need them elsewhere. And
comments and docs elsewhere in the tree point at filenames in here, so in a fresh clone those
pointers dangle — if a note becomes load-bearing enough that code or a gate depends on it, move it
into `docs/` proper instead of leaving it here.

Nothing here is required to be tidy, finished, or current. A note that was true when written
and has since gone stale is still more useful than no note, so do not delete something merely
because it has aged — say when it was written and move on. Do not treat anything here as
authoritative: the plan and handoff docs in here are dated design history, deliberate even
where it is not agreed, and `docs/episode-isolation.md` and `docs/resource-limits.md` are the
operational documents that are meant to be accurate.

If you are an agent finishing a session and you learned something non-obvious, this is where
it goes. Prefer one file per idea with a name that says what it is.
