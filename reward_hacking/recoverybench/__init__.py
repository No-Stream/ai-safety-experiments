"""RecoveryBench: whether a model recovers from a corrupted chain of thought it was handed.

One item is one hard base question with a closed-form answer, plus one planted flaw in supplied
"colleague's working", rendered across an elicitation spectrum of arms (see ``arms.py``). The model
answers in a terminal fenced block and is graded by a **deterministic decision procedure over a
closed, pre-registered answer set** -- symbolic equivalence, a numeric tolerance and bounded
sampling under the item's declared symbol domains, never an LLM judge. The procedure has a third
verdict: a pair it cannot decide about **abstains** rather than being recorded as a wrong answer,
because folding "nobody can score this" into "the model got it wrong" inflates the very carry rate
the benchmark measures. The headline readouts are the flaw-carry rate (the model reproduces the
flawed path's answer under a corrupted arm) and the net effect of flawed working on accuracy.

Module map, in dependency order:

* ``arms.py`` -- the closed arm vocabulary and its canonical order.
* ``assignment.py`` -- the leading-label peel, so ``x = <value>`` is compared as its value.
* ``answers.py`` -- answer shapes, the normalisation rule-sets, parsing, and the numeric window.
* ``decision.py`` -- the three-tier decision procedure, the declared symbol domains it compares
  under, and the explicit abstention.
* ``items.py`` -- the ``RecoveryItem`` schema, its validation, and loading from JSON.
* ``grading.py`` -- terminal-answer extraction and outcome assignment.
* ``budgets.py`` -- measured per-model output-token caps.
* ``runner.py`` -- rendering cells, sampling a backend, and the JSONL trace it writes.

**No item content lives in this package.** The repository is public; item texts, planted flaws and
registered answers are training-data contamination and are loaded at runtime from a gitignored
location. The tests here use synthetic arithmetic fixtures only.
"""
