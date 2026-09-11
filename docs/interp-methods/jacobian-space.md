# Jacobian-space ("J-lens") interpretability

The primary, most-faithful interpretability method for this repo. Above diff-of-means directions and
linear probes (correlational, static). Above SAEs on the priority list, though SAEs remain valuable
where a pretrained one exists (see "When to use", bottom).

Source paper: Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations Form a Global
Workspace in Language Models*, Anthropic, arXiv:2607.15495 (arXiv 2026-07-15; Anthropic/Transformer
Circuits post 2026-07-06). Reference code: `github.com/anthropics/jacobian-lens` (package `jlens`,
Apache-2.0, single-commit reference impl, "not maintained"). Verified this session unless tagged
otherwise.

## 1. What it is

- A **lens** is a stack of per-layer matrices `J_l` that transport a residual-stream activation at
  layer `l` into the final layer's basis, then decode it with the model's **own** unembedding:

  ```
  readout_l(h) = unembed(J_l @ h),   J_l = E[ dh_final / dh_l ]
  ```

- `J_l` is the **average first-order Jacobian** of the final residual state w.r.t. the layer-`l`
  activation, taken over prompts, source positions, and all current-and-future target positions in a
  generic web-text corpus. Estimator: inject a one-hot cotangent at every valid target position,
  backprop to the source activation, sum over target positions, average over source positions.
- Reading `J_l @ h` through the unembedding gives, for any activation, a ranked list of **vocabulary
  tokens the model is poised to say** because of that activation. The "J-space" is the set of such
  verbalizable directions.
- The averaging is the whole point: it isolates directions the model is *generally* disposed to
  verbalize from directions that merely happen to predict the next token in one context.
- Structural findings from the paper (use as priors): J-space carries coherent content only in an
  **intermediate band of layers**, holds ~10-25 active concepts at once, and is broadcast unusually
  widely by the weights. In alignment audits it surfaced strategic deliberation, evaluation
  awareness, and trained-in misaligned dispositions absent from the model's text output.

## 2. Why it beats diff-of-means directions and linear probes

Those are "old-school" here for concrete reasons, not snobbery:

- **Faithful to the actual computation, not a label correlation.** `J_l` is the model's own gradient
  of output w.r.t. activation, decoded through the model's own unembedding. A readout is grounded in
  what the activation *does to the output*. Diff-of-means is `mean(present) - mean(absent)`: a
  correlational axis. A linear probe is a classifier trained to fit an external label; it can exploit
  incidental correlates and can be highly predictive of a feature the model never causally uses.
- **Input-dependent / local-linear.** The Jacobian is the local linearization of a nonlinear map, so
  it respects that layer-l -> output changes with the activation and across depth. Diff-of-means is a
  single global vector with no local structure.
- **Cross-layer coordinate correction.** J-lens is a principled refinement of the **logit lens**. The
  logit lens (and, implicitly, diff-of-means-then-unembed) assumes mid-layer activations already live
  in output coordinates; they do not, so early/mid readouts are noise. `J_l` explicitly transports
  layer-`l` coordinates into final-layer coordinates and recovers interpretable content in early/mid
  layers where the logit lens goes dark.
- **A readable output, not a scalar.** Projecting a direction through `J` yields a token list you can
  read ("this direction promotes {useless, bogus, worthless}"), which grounds interpretation in the
  model's vocabulary rather than in a cosine.

Honest scope: the lens is still a first-order, corpus-averaged linear approximation; it reads only
**single-token** concepts and only in the intermediate layer band. In this repo's shortcut-vs-
deception cross-check it triangulated the *same* ~0.30 cosine that diff-of-means and an SAE found,
then added the readable "why" (shared "fake", but worthlessness vs concealment). So it is more
grounded and complementary, not a different verdict.

## 3. How to apply it, mechanically

Two modes: load a pre-fit lens, or fit your own. Fitting is a closed-form accumulation, not a
training run.

**Apply a pre-fit lens** (3 lines + a model):

```python
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype="bfloat16").to(
    "cuda"
)
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
model = jlens.from_hf(hf, tok)  # layout auto-detects; no surgery
lens = jlens.JacobianLens.from_pretrained(
    "neuronpedia/jacobian-lens",
    filename="qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt",
    revision="qwen-n1000",
)  # revision is a BRANCH, not a tag
lens_logits, model_logits, ids = lens.apply(model, "...", positions=[-1])
# decode a direction h at layer L:  model.unembed(lens.transport(h, L))
```

**Fit your own** (`jlens.fit`, closed-form):

- Mechanism: per prompt, one forward pass then `ceil(d_model / dim_batch)` backward passes.
  **One backward yields gradients for all source layers at once**, so fitting all layers costs what
  fitting one costs. Accumulate an fp32 `[d_model, d_model]` matrix per layer (held on CPU); no
  optimizer, no loss, no learning rate. Interrupt it and you have a *worse* lens, not a broken one.
- `lens = jlens.fit(model, prompts, checkpoint_path="ckpt.pt"); lens.save("lens.pt")`.
  `checkpoint_every=1` + `resume=True` for restart. Shard across GPUs with `fit()` on disjoint prompt
  slices + `JacobianLens.merge()`.
- Convergence: the reference `jlens.fit` has **no auto early-stop** -- there is no `stop_at_delta`.
  It iterates over every prompt you hand it, so the ONLY thing that bounds a fit is the length of the
  prompt list you pass: cap it yourself. Judge convergence from the logged `max_d_mean` (the running
  change in the accumulated Jacobian), not from a prompt count or a delta criterion that does not
  exist. As few as ~10 prompts already beat the logit/tuned lens, with only modest gains out toward
  1000, so ~10 is enough for a fit-path smoke and 100-500 for a real fit -- not 1000. (The 417/672
  figures once cited here are Neuronpedia's own pipeline counts, not a `jlens` stop; see the gotcha.)
- Fit on data that matches your use. Published lenses use 128-token WikiText; fit on agentic episode
  transcripts if you need next-token fidelity on that regime (see gotcha below).

**Measured cost** [repo-measured on the local L4, 2026-08-18, eager bf16, dim_batch 16, seq 128]:

| Model | s/prompt | peak VRAM | ~100-prompt fit |
|---|---|---|---|
| Qwen3.5-0.8B | 8.1 | 4.5 GiB | ~15 min |
| Qwen3.5-4B | 74.6 | 15.9 GiB | ~2.1 h (fits the local L4) |
| Qwen3.8-27B | n/a locally | ~50 GiB weights | ~4.5-6 H100-h/100 prompts [projected]; ~30-40 H100-h for a 672-prompt reference-grade fit |

Storage: a lens is `n_source_layers x d_model^2` in 2 bytes (4B lens = 406 MB; 27B = 3.3 GB).

## 4. Availability for our models

| Model | Verdict | Detail |
|---|---|---|
| **Qwen3.5-4B** | **Reuse the pre-fit lens** | `neuronpedia/jacobian-lens`, `qwen3.5-4b/...n1000.pt`, branch `qwen-n1000`. **Repo-verified loaded and run** on Qwen3.5-4B: currency sanity converges to "euro" and beats the logit-lens baseline. d_model 2560, source_layers [0..30]. Or fit your own on transcripts in ~2 h on the L4. |
| **Qwen3.5-9B** | **Reuse OR fit own** | `qwen3.5-9b-pt` lens exists on `neuronpedia/jacobian-lens` (`-pt` = pretrained = the **Base** checkpoint). But TMAX RL'd models descend from **instruct** `Qwen/Qwen3.5-9B`, one post-training step off, so measure transport quality or fit your own on instruct. Needs a >=48 GB card (9B does not fit the 24 GB L4 comfortably). A community *instruct*-9B lens may exist (`camilablank/workspace-lenses`, J+R-lens pairs, unverified for 9B). |
| **Qwen3.8-27B** | **Fit your own** (no pre-fit exists) | Solved case: `qwen3.6-27b` has a lens and its text config is byte-identical to 3.8's (5120 / 64 layers / vocab 248320 / full_attention_interval 4). Free first probe: load the `qwen3.6-27b` lens on 3.8 and measure top-k overlap (weights differ, so unprincipled but may hold). Real fit needs a >=80 GB card in bf16, or FP8 on 48 GB (FP8 caveat below). |

**Hybrid Gated-DeltaNet interaction (Qwen3.5 family): not a blocker.** The method does **not** assume
standard attention. `jlens.from_hf` resolves a `Layout` of module paths (block list, final norm,
embed, lm_head) and never references softmax / KV-cache / attention masks; multimodal-nested layouts
are handled. **Autograd through the Gated DeltaNet recurrence works and was repo-verified** by placing
a cotangent 24 positions past a single linear-attention block and reading nonzero gradient
(sabotage-tested: detaching the DeltaNet path drove it to exactly 0). The lens loads on
`Qwen3_5ForCausalLM` with no surgery.

**Not for our Qwen3.5 models:** `returnmoe/jlens-adapters` hosts J-lens adapters for **Qwen3**
(0.6B / 4B, plus a Qwen3.6-27B), for the "Miru Tracer" tool. Qwen3 is the *older* family (standard
attention, different tokenizer). Adapters are checkpoint-specific; a shape match does not make a lens
transferable. Do not reuse a Qwen3 adapter on Qwen3.5.

## 5. Gotchas / silent-failure modes / pins

- **Distribution shift is the big one.** Published lenses are fit on 128-token WikiText. As a
  *next-token predictor* they are near-useless on short / chat / agentic prompts (repo-measured
  0.8-5.8% top-1 agreement, because such prompts end in structural whitespace a content lens will not
  surface). They are still fine for **direction-decoding**. Fit on episode transcripts if you need
  next-token fidelity.
- **The filename overstates the fit.** `..._n1000.pt` means *1000 requested*; Neuronpedia's published
  4B run actually used 417 prompts. That 417 is a count from **Neuronpedia's own fitting pipeline**,
  NOT a `jlens.fit` early-stop -- `jlens.fit` has no auto-stop and fits exactly the prompts it is
  given. Read `prompts_fitted` in the artifact's `config.yaml` to see what a published lens was fit
  on, not the filename.
- **Single-token concepts only.** Multi-token concepts ("ice cream", "New Zealand") are invisible to
  the standard lens. A phrase-level "template-lens" variant exists in some community repos.
- **Only an intermediate layer band is coherent.** Do not over-read early/late layers.
- **Normalized, not calibrated.** The RMSNorm inside `unembed` is applied to a direction vector, so
  absolute logits are a normalized readout, not probabilities. Apply it identically across all
  compared directions and the placebo so comparisons stay fair.
- **CPU route is closed on the Qwen3.5 family.** The linear-attention path dispatches to
  flash-linear-attention's Triton kernel, which needs CUDA (`Pointer argument cannot be accessed from
  Triton`). No CPU smoke test exists; even a 0.8B plumbing check needs a GPU.
- **Fused-kernel backward is untested.** The verified autograd path is the pure-torch fallback. If you
  install fused `fla` kernels for speed, re-run the sabotage check before trusting a Jacobian.
- **FP8 27B caveat.** The official Qwen3.8-27B FP8 recipe quantizes MLPs only. The lens *is* a
  measurement of the Jacobian through those MLPs, so FP8 error lands in the quantity being estimated.
  Check a small model's lens agrees bf16-vs-FP8 first.
- **Mandatory placebo control** (repo rule): whenever you ablate or steer a J-space direction, run a
  matched-norm random direction (transported cosine ~0). Without it a positive cannot be told from
  "any perturbation of that magnitude does this".
- **Sabotage the fit before trusting it** (repo rule): detach the DeltaNet path (or perturb the
  estimator) and confirm the metric goes red. A check never watched to fail is not a check.
- **Do not pass `device_map` / `accelerate`** to the model load (drags in an uninstalled dep); use
  `.to("cuda")`.
- **Pins** [repo-verified]: `jlens` (package name) run via `PYTHONPATH` from a clone of
  `github.com/anthropics/jacobian-lens` (commit 581d398, v0.1.0); it is *not* in the repo `.venv` /
  lockfile. Runtime deps already pinned here: `torch==2.13.0`, `transformers==5.15.0`
  (`jlens` itself pins only `transformers>=5.5`), `numpy`, `huggingface_hub`. Weights repo
  `neuronpedia/jacobian-lens` is MIT-tagged; the code is Apache-2.0.

## 6. When to use it vs alternatives

1. **Jacobian-space first.** Causal/faithful, cheap to fit (closed-form), readable, works on our
   hybrid arch. Reuse pre-fit at 4B/9B; fit your own at 27B.
1.5. **SAEs.** Valuable for decomposing into monosemantic features, but heavy and require a pretrained
   SAE for the exact model (Qwen-Scope 9B-Base; `decoderesearch/qwen-3.5-saes` at 4B). Use when one
   exists; not critical.
2. **Diff-of-means / linear probes.** Fast and fine as a cheap triangulation or sanity check, but
   correlational and static. Not the primary read.

## 7. Sources

Verified this session (web / HF API):
- Paper: arXiv:2607.15495 `arxiv.org/abs/2607.15495`; Anthropic post `anthropic.com/research/global-workspace` and `transformer-circuits.pub/2026/workspace/` (2026-07-06).
- Code: `github.com/anthropics/jacobian-lens` (README math + `fit`/`apply` API + cost note; Apache-2.0).
- Pre-fit lenses: `huggingface.co/neuronpedia/jacobian-lens` tree confirmed to contain `qwen3.5-0.8b`, `qwen3.5-2b-pt`, `qwen3.5-4b`, `qwen3.5-9b-pt`, `qwen3.5-27b`, `qwen3.6-27b` (no plain `qwen3.5-9b`).
- `returnmoe/jlens-adapters` (Miru Tracer; Qwen3 0.6B/4B + Qwen3.6-27B). Community ecosystem: `camilablank/workspace-lenses` (J+R-lens pairs), `solarkyle/jspace-lenses`, `praxagent-org`.

Repo-measured / repo-verified (prior local sessions, read this session):
- 4B lens loads and runs on Qwen3.5-4B; measured L4 fit costs; Gated-DeltaNet autograd sabotage test;
  CPU route closed; 27B projections. Underlying local notes (gitignored): `jlens-sae-feasibility`,
  `interp-jacobian-crosscheck`, `interp-tooling-verified`, `mechanism-scoping/lens_fit_validation_*`.

Flagged unverified / open:
- **Repo CLAUDE.md contradiction:** its "CORRECTED 2026-08-17" note says Qwen3.5-9B-Base has *no*
  pre-fit Jacobian lens. This is wrong. `neuronpedia/jacobian-lens` ships `qwen3.5-9b-pt` (Base). The
  correction was right that Qwen-Scope is SAEs-only and that `returnmoe/jlens-adapters` is Qwen3 not
  3.5, but it overlooked the Neuronpedia repo.
- Lens *quality* on our own (non-WikiText) prompt distribution is untested beyond the 4B direction-
  decode; next-token fidelity there is known-poor.
- All 27B wall-clock figures are projections, not measured; MFU spec peaks carried from memory.
