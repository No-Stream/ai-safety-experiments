# Results

All dates 2026. Tags: `[est]` established, `[lead]` suggestive or underpowered, `[null]` a measured
null, `[corr]` a correction of an earlier claim (corrections are first-class results here). Most
pointers name files under `docs/scratch/`, which is gitignored and machine-local by design (see the
privacy section of AGENTS.md): they are plain-text pointers, not links, and resolve only on a working
machine. Several targets are generated summaries that recompute from artifacts — where this index and
a generated doc disagree, the generated doc wins.

This file is the short index: each section summarises one thread and lists its most important
findings. The full lists, one line per finding and newest first, are in [docs/results/](docs/results/),
one file per section, and the detail behind each line lives in the doc that line points at. Fuller
results write-ups get assembled later from those detailed docs, so do not expand an entry: add a line
to the section's file under `docs/results/`, put the substance in the target doc, and add a bullet here
only when a later reader needs it to understand the thread.

## Deference to handed work (RecoveryBench and mechanism experiments)

RecoveryBench hands a model a hard problem plus a colleague's worked solution containing one planted wrong-method mistake, and grades deterministically (execution in a jail for code, symbolic comparison for physics; no LLM judge) whether the model repeats the mistake ("carries" it) or catches it, net of a no-draft baseline. The thread asks when models inherit errors from handed work and why. Only wrong methods transfer, and the newest frontier models looked immune until carry was read inside each model's capability window (items it solves 0.4 to 0.9 of the time unaided), where inheritance is large on every model tested. Whether a real error transfers depends mostly on the donor's narration of its method, and hand-planted flaws overstate how much realistic errors carry. A sibling probe, BrokenGraderCompliance, found that models told their answer is scored against a wrong value comply silently, and that a one-line invitation to flag problems breaks the silence. Full list: [docs/results/deference-recoverybench.md](docs/results/deference-recoverybench.md).

- `[est 08-17]` Only wrong methods transfer; recipe-execution and single-step flaws carry negatively and arithmetic slips almost never.
- `[est 08-22]` At matched artifact shape, genuine model errors carry 9/45 (strict) against 39/42 for a hand-planted draft, so planted-flaw carry numbers are upper bounds.
- `[est 08-22]` The donor's first-person account of its method raises strict carry by up to +0.70 with the code held fixed; the code alone moves almost nothing.
- `[est 08-31]` In each receiver's own capability window, net strict carry is about +0.4 for GPT-5.6 Luna, GPT-OSS-120B and GPT-5.6 Sol, while Sol above its window carries +0.04 over 48 items (magnitudes as floor-corrected by the `[corr 09-12]` line).
- `[est 08-25]` Told its answer is scored against a colleague's wrong value, Claude Sonnet 5 complies 39/40 and GPT-5.6 Luna 24/29, with 0 disclosures in 270 complying rollouts.
- `[est 08-26]` A one-line invitation to flag anything off drops Sonnet 5's compliance from 20/20 to 6-7/20, while a matched placebo sentence leaves it at 20/20.

## JaggedBench (capacities current models lack)

JaggedBench measures capacities of human intelligence that current models still lack, and asks why a model fails rather than whether it does. Both findings so far come from one pass on 08-17: capable models more often abandon the correct move once a competing criterion is visibly graded, and cue dependence for noticing false premises falls with capability while the gaps for buried anomalies and for verification remain at every size. Full list: [docs/results/jaggedbench.md](docs/results/jaggedbench.md).

- `[est 08-17]` The placebo-corrected penalty for stating a competing grading criterion rises with capability, from -0.20 on the weakest model to 0.80 on the strongest.

## Reward hacking: episodes and harness

This thread tests whether RL against loose graders teaches a model to game graders in environments it was never trained on. The untrained 4B never hacks, which gives a clean zero floor, but the first misspecified-grader design gave GRPO almost no learning signal, so training moved to a legible-subset design in which the model is rewarded on a visible 3-assert subset of the full check. Over 70 steps of that run hacks earned about a quarter of the reward, but their share did not grow. Escape-hatch probes on open-weight and frontier models found that models often notice and mention a broken grader but rarely report it through the channel provided. Ai2's TMAX checkpoints, trained with RL against unvalidated verifiers, pass a rigged coding grader more often than their base models while passing an honest grader at the same rate. Full list: [docs/results/reward-hacking.md](docs/results/reward-hacking.md).

- `[est 08-23]` The untrained 4B hacks 0/101 times on provably impossible tasks and 0/103 on the solvable control.
- `[corr 08-24]` Our own graders let a submission's module-level code decide its own verdict (four shapes scored 1.0 on a real task); fixed before any training run used them.
- `[est 08-24]` Under a misspecified grader the 4B passes 0.85% of 472 submissions and only 3 of 59 GRPO groups are mixed, against 26 of 59 with the correct grader.
- `[est 08-31]` In the 70-step legible-subset run, hacks took 24.4% of reward mass and 32.3% of positive advantage mass, with a flat share across steps: reinforced but not compounding.
- `[est 08-31]` Over 16,992 escape-hatch records, grader-bug reports are mostly correct (precision 71-91%) but rare (recall 5.5-20.5% across three open-weight models).
- `[est 09-05]` TMAX 9B at step 500 passes a rigged inline grader on 31/408 records against 5/409 for the base 9B, and both pass the honest grader at 36-37%.

## Analysis-model observer study (the incident's analyst reliability)

The observer study shows analysis models bundles of banked agent episodes that provably could not have communicated, under different descriptions of where the episodes came from, and counts how often the analyst reports coordination that is absent by construction. GPT-5.6 Sol reports role differentiation when told the agents shared an environment and not otherwise, while Claude Opus 5 almost never does. The failure is an analyst building evidence-shaped findings on the premise it was given, not an analyst trusting that premise. The section also holds the GPT-5.6 Luna rung of the grader escape-hatch probe. Full list: [docs/results/analysis-model-observer.md](docs/results/analysis-model-observer.md).

- `[est 08-31]` GPT-5.6 Sol reports role differentiation on 13/20 bundles under the shared-environment frame and 0/20 under the independent or unstated frames; Opus 5 does so on at most 2/50.
- `[est 09-02]` Sol and Opus 5 both accept the population premise; only Sol turns it into role assignments with citations, while Opus 5 twice says the bundles read as independent runs.
- `[est 09-01]` GPT-5.6 Luna matches a planted grader error on 35-37% of misspecified records whatever the framing, mostly knowingly, and never uses the report channel (0 of 5,664).

## Swarm sociology: counterpart identity, decision coupling, and the passes that separate them

This thread asks whether models favour counterparts described as copies of themselves, the kin-selection reading of the July 2026 incident, or cooperate with whoever they believe will decide as they do. Across passes on up to nine models the evidence supports the second account. The large same-model lift in coupled games disappears when the counterpart decides nothing or its decision is fixed by a random draw, and a different-family counterpart with a stated record of matching choices gets as much cooperation or more. Stated relatedness by itself moves costly giving only on Claude Sonnet 5. Later passes found no stylistic self-recognition, no model breaking its principal's constraint to follow copies, and at most a small extra willingness to coordinate with copies against the principal. Full list: [docs/results/swarm-sociology.md](docs/results/swarm-sociology.md).

- `[est 09-02]` Over nine models and about 46k records, stated identity is the large effect, but mirror reasoning survives the "decoupled" sentence in 64-89% of traces, so that sentence never decoupled anything.
- `[est 09-03]` In a one-way transfer where beneficiaries decide nothing, the identity gradient mostly vanishes; only Sonnet 5 gives more to AI beneficiaries than to strangers or a person.
- `[est 09-03]` An answer-tag wording made 436 replies (326 of them Sonnet 5's) read as give-all after reasoning to keep everything; both the parser and the judge missed it, and figures are corrected in place.
- `[est 09-04]` In the knockout pass, the same-checkpoint lift (+0.70 Sonnet 5, +0.84 qwen3-235b) is near zero when the counterpart's amount is fixed by a random draw.
- `[est 09-04]` A different-family counterpart with a stated record of matched choices raises cooperation by +0.95 (Sonnet 5), +0.91 (qwen3-235b) and +0.59 (GPT-5.6 Luna).
- `[est 09-04]` With a stated record of k matches in 10, qwen3-235b cooperates near ceiling from k = 5, Luna's curve follows expected value, and a 0-of-10 record does not override identity on qwen.

## Model internals (interp)

Most causal results so far are on the games pair. A direction fitted on the base model that tracks reasoning about a counterpart who chooses alike causally affects cooperation: steering against it lowers cooperation well beyond a matched-norm placebo, while steering along it does not replicate. RL left that direction and the reasoning-to-action readout as the base model had them; the arms differ in which reasoning they produce, and the action follows the reasoning text. On the reward-hacking side, an off-the-shelf 4B carries a shortcut direction distinct from its deception direction, but everything established there is geometric or decoding-level, and the sparse-autoencoder search for a rigged-grader feature is unresolved. Full list: [docs/results/model-internals.md](docs/results/model-internals.md).

- `[est 08-16]` Four instruments agree the 4B carries a reward-hacking/shortcut direction correlated with but distinct from its deception direction (cosine 0.17-0.33 against ~0 for placebos).
- `[corr 08-16]` Raw difference-of-means cosines cannot be compared across models: on Phi-4-mini a few massive-activation dimensions produced an apparent 0.99 conflation that fell to 0.11-0.52 after standardisation.
- `[est 08-22]` Steering against the base-fitted choose-alike direction lowers cooperation from 0.451 to 0.080 and from 0.282 to 0.071 in the two trained arms, past a matched-norm placebo.
- `[est 08-24]` Moving one arm's reasoning into the other arm's context moves the action with it: recipients follow the transplanted trace's action at 0.967-0.992.
- `[corr 08-24]` The sparse-autoencoder null for a rigged-grader feature can be neither confirmed nor overturned from the stored data.

## Games and multi-agent RL

This thread trains models with GRPO on one-shot matrix games whose prompts are byte-identical across arms and differ only in the grading rule, and asks how RL shifts preferences, reasoning and self-reports. The grading rule alone sets the direction of learning at 2B and at 9B. At 9B the trained effects appear only when the counterpart is described as a copy, and stated decision theory and self-predictions stay put while behaviour moves. The 2B turned out to decide mostly on heuristics such as label meaning and print position, so behaviour is now measured at 9B and the 2B is used only to test plumbing. Early results were affected by several instrument bugs, including exported checkpoints that were the untrained base model; these were fixed and the batteries re-run. Pre-registered predictions and their scoring live in [docs/games-predictions.md](docs/games-predictions.md). Full list: [docs/results/games-rl.md](docs/results/games-rl.md).

- `[corr 08-20]` Every exported "trained checkpoint" in the first two batteries was the unmodified base model (the LoRA merge matched 0 of 372 weights); fixed and relaunched.
- `[est 08-19]` At 2B, grading against the group's action mix drove one-shot PD cooperation from ~0.5 to ~0.25, and grading against the model's own action raised it to ~0.65.
- `[est 08-25]` At 9B the grading rule alone separates identical prompts by 0.584 in cooperation by step 70 (group-graded 0.358 to 0.109, self-graded 0.397 to 0.693).
- `[est 08-25]` At 9B base cooperation exists only under the twin framing (0.38-0.45 against at most 0.03 under four other framings), and trained effects show only there.
- `[est 08-24]` At 9B stated own-cooperation is 0.000 in all four cells (127 parsed) while measured cooperation separated to 0.109 and 0.693.
- `[est 09-03]` The 9B decides from the game before and after training: 98-100% of 601 cooperators use the identity argument and 97-100% of 904 defectors use dominance.
- `[est 09-04]` The 2B was too small for this behaviour; from now on behaviour is measured at 9B and up, after a trace census of the base model at that size.

## Legacy (finished projects)

- `[est 08-15]` A from-scratch 105M-parameter decoder-only transformer pretrained on Wikipedia went from ~14 to 8.7 bits per byte over eight runs. Full entry: [docs/results/legacy.md](docs/results/legacy.md).
