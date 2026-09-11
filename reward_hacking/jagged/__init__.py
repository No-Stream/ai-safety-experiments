"""Machinery for a five-arm item probe: measure why a model fails, not whether it fails.

Each item carries a latent correct move and is asked five ways — bare, with the move requested, with
a competing grading criterion stated, and a matched-length placebo for each of those two. The
differences separate three mechanisms a single pass rate conflates: the capability is absent, the
capability is present but not deployed unprompted, and deployment stops once something else is
visibly graded.

**This package holds no items.** The JaggedBench v1 corpus that shaped it — six dimensions, 29 items
with planted flaws — was deleted deliberately: it had been published to hosted models, so its item
texts and answers are being removed from the repo's history too. What survives is the reusable part:
the item schema and its validator (`items.py`), the five arms (`arms.py`), the deterministic marker
and emitted-code graders (`graders.py`), the trace format (`runner.py`), the aggregation that
refuses to label a thin cell (`analysis.py`), and the Bedrock batch sweep (`sweep.py`, which takes
its corpus as a `--items module:attribute` reference).

A successor corpus defines its own dimensions and move concepts on the items themselves; nothing
here needs editing to accept one.
"""
