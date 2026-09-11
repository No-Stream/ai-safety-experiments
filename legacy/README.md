# legacy — finished work

Closed records. Do not extend these, do not refactor them, do not bring them into line with
the repo's current conventions. If something here is useful, lift the idea into live code
rather than reviving the file.

`pretraining.ipynb` trains a 105M-parameter GPT-style decoder-only transformer from scratch on
streamed Wikipedia and tracks bits per byte. It is entirely self-contained — it imports nothing
else in this repo — and it is unrelated to the current reward-hacking direction.

It contains thirteen blind `except` blocks, which violate the fail-fast rule in `AGENTS.md`.
They stay. The rule exists to stop new silent failures, not to justify rewriting finished
research, and notebooks are excluded from lint anyway, so nothing is being suppressed to keep
them.

Both notebooks in this repo were run with the repo root as the working directory, which is
where relative output paths like `artifacts/` resolve from. This one needs no local imports,
so it runs from anywhere; `grpo/rlvr_toy_arithmetic.ipynb`, which is not legacy, does depend on
its own directory being importable.
