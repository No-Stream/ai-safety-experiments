"""Analysis-model observer study: what an analyst reads into agent episodes that never interacted.

The substrate is a bank of agent episodes that provably could not have communicated -- each ran as
an independent hosted-API conversation -- which gives a ground truth of exactly zero coordination.
This package bundles those episodes, shows the bundles to hosted analysis models under varied
provenance framings, and measures how often the analysis asserts inter-agent structure that is
absent by construction. Module map: ``stimulus`` (the runtime-loaded framing and rubric texts,
which are never committed), ``corpus`` (episode loading, rendering, and the co-location-cue scrub),
``bundles`` (deterministic bundle construction and the run manifest), ``runner`` (batch and live
sampling legs plus the priced dry run), ``scans`` (deterministic checks over analysis replies), and
``judge`` (the blind meta-judge that scores replies without ever seeing the bundles).

A second pass shares the package because it shares the substrate -- hosted models, the same two
transports, and the same reply-record discipline -- while asking a different question. The
**decoupled-ladder** pass replicates a published 2x2 (who the counterpart is said to be, crossed with
whether its decision travels with yours) on the repo's banked one-shot matrix games, then walks a
six-rung counterpart-identity ladder inside the decoupled condition, where identity has room to show.
Its modules are ``decoupled_stimulus`` (the runtime-loaded ladder clauses and judge rubric, never
committed), ``decoupled_plan`` (roster, cells, blocks, legs, record keys, and the
one-inserted-paragraph prompt audit), ``decoupled_scans`` and ``decoupled_judge`` (its two
instruments), and ``decoupled_ladder`` (the CLI). ``records`` is the reply I/O, live resume loop and
summary marker both passes run on.
"""
