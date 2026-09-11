# The authored-declaration load gate

`tests/test_declaration_data.py` loads every authored declaration artifact through the checks the
grader will hold it to. This document is its design record; the test file keeps a summary and a
pointer here.

## Why it exists

A new or narrowed load-time refusal is a claim about what every authored artifact on disk already
looks like, and nothing was checking that claim. The day this gate was written a refusal landed
(`items.py::_validate_declaration`, the constant-versus-domain contradiction) and one authored
declaration already violated it, found only because a session happened to sweep the data by hand.
Three ordinary gates cannot see these artifacts at all: `make lint` skips them because ruff respects
`.gitignore`, `make typecheck` excludes the path by name, and no consumer loads them yet.

This is `tests/test_scratch_compiles.py`'s reasoning extended from code to data, and it inherits that
file's line: load-level failures only, never style. An artifact that cannot load is broken on any
box; an artifact that is merely scruffy is none of this gate's business, so `why` texts,
`parser_artifact_symbols` and `auth_defect` prose are all ignored.

It is prospective on purpose. The declaration artifacts are not wired to a consumer today, so the
gate guards a contract nothing yet loads, which is exactly when pinning it is cheap. The alternative
is discovering at wiring time that hand-edited data drifted from checks that landed months earlier.

## The schema translation is itself a failure surface

The artifact spells its fields `per_symbol`, `constants` and `default`; the item spells them
`symbol_domains`, `reserved_name_meanings`, and for `default` no home at all. A partial translation,
mapping `per_symbol` and forgetting `constants`, would go green forever while never exercising the
contradiction check that motivated the gate, because that check fires only on the intersection of the
two maps. `TestTheGateRejectsWhatItExistsToCatch` therefore plants the violation the gate was built
after and requires the *production* refusal message back, so dropping either half of the translation
turns that self-test red rather than leaving this gate silently vacuous.

## The untranslated keys are the gate's tripwire for its own incompleteness

Seven keys appear across the artifacts. Three are translated; four (`why`,
`parser_artifact_symbols`, `auth_defect`, `declaration_licensed_by`) are prose no consumer coerces.
`KNOWN_ENTRY_KEYS` asserts the observed set stays inside those seven, and that assertion is
load-bearing in a way the translated keys are not, because three of `_validate_declaration`'s
refusals are unreachable from here by construction: the numeric-item refusal, since a symbol
declaration only means anything on an expression item, and both sampling-range refusals (a
non-finite bound and an inverted range), since no artifact carries a range key so the gate never
passes `symbol_ranges`.

That is correct scoping today and a silent hole tomorrow. The day an artifact grows a range key, a
shape field, or anything else a consumer would read, the key-set assertion turns the omission red
instead of leaving the new field quietly unchecked. It also covers drift this gate's transcription is
otherwise blind to: a renamed enum member is caught for free, because the artifact's own string stops
coercing, but a consumer that starts reading a field this gate ignores is invisible without it.

## Both denominators are floored rather than merely printed

A reported denominator can itself collapse. Truncate one file's `items` to a single entry and an
unfloored sweep says "3 files, 1 entry examined, 0 cannot load" and passes green, which is the
a-zero-needs-its-denominator rule half-applied. `KNOWN_AUTHORED_FILES` and `KNOWN_AUTHORED_ENTRIES`
are floors rather than exact counts, so authoring more is free while losing some is red.

## `default` gets a third treatment

Refused when it is anything but `real`: not translated, and not dropped.
`RecoveryItem.declaration()` builds its `SymbolDeclaration` without passing `default`, so an item's
undeclared symbols are always REAL, and an artifact saying otherwise is asking for a reading the
grader cannot give it. Silently loading such an artifact would be the reports-success-while-doing-nothing
shape this repository's central rule is named after.

## Why it calls a private function

The gate calls `_validate_declaration` directly and that coupling is deliberate. The public path is
`validate_item`, but from a declaration artifact alone a synthesized item fails its unrelated checks
first (verified: it dies on `no source` before any declaration check runs), and catching
`ItemValidationError` around `validate_item` to get past them would swallow the declaration refusals
this gate exists to surface. The private call trades a rename breaking the test loudly, where an
ImportError names the coupling, against fabricating four unrelated fields per item and hoping none of
their checks fire first.
