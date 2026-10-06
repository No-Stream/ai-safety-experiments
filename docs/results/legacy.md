# Legacy (finished projects)

Legacy: finished projects. Tags and pointer conventions are explained at the top of [RESULTS.md](../../RESULTS.md), which also carries this thread's summary.

- `[est 08-15]` A from-scratch 105M-parameter decoder-only transformer pretrained on streamed Wikipedia drove bits-per-byte from ~14 to 8.7 across eight recorded runs, the largest single gain from holding the learning rate flat for half the iterations, while throughput-oriented changes bought almost nothing at that scale. ([legacy/pretraining.ipynb](../../legacy/pretraining.ipynb), [legacy/README.md](../../legacy/README.md))
