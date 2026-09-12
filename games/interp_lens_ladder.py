"""Fit a Jacobian lens at every checkpoint of a games arm, and record how good each fit is.

The repo's rule is to lead with Jacobian-space, and the rule about lenses is that we always fit our
own -- it is a closed-form accumulation, not a training run, so a per-checkpoint lens is affordable
where a per-checkpoint SAE would not be. This is the ladder wiring for that: merge a checkpoint, fit
a lens on the same stimulus corpus the activation capture used, score the fit against the free
logit-lens baseline it has to beat, save it, and move on.

**The lens is fitted on the un-merged adapter, and the merge is kept as a comparison arm.** This leg
used to merge every checkpoint to disk and reload it, on the docstring's claim that `jlens.from_hf`
"wants a plain single-device HF causal LM" and that a runtime PEFT wrapper is not one. That claim was
wrong: PEFT injects its modules into the model tree in place -- the activation capture depends on
exactly that -- and `from_hf` only needs a module carrying `layers`/`norm`/`embed_tokens` under one of
its known paths plus an `lm_head`, which the injected composite model has. Probe I6 (2026-09-02) ran
it: the layout resolves on the injected model, `get_base_model()` is the very object
`games.lora.load_adapter_base` built, a PEFT model with every `lora_B` zeroed reproduces the base fit
to within the 0.33% wrapper floor, and the merged-and-reloaded fit differs from the un-merged one by
0.6% median and 1.8% max relative Frobenius -- the bf16 merge, visible in the lens.

So `--lens-model unmerged` (the default) fits through `Wx + B(Ax)` at full precision, which is the
ruling the rest of this project measures effect sizes under, and `--lens-model merged` still merges
and reloads so the two can be read against each other. The un-merged path is not a time saving and
is not sold as one: it skips the merge and the reload (2B about 25-35 s per cell, 9B about 3.5 min)
and pays more back in the fit itself, which runs about 39% slower on an adapted model because every
adapted Linear adds two matmuls to each forward and backward. What it buys is fidelity: on the merged
path every lens number is attenuated to about 64% of the trained delta (measured 2026-08-20, 38-79%
per module), so a before/after *magnitude* read across the lens and capture legs was not comparing
like with like. The two paths take separate lens-cache keys, because they are separate lenses.

**`max_seq_len` is derived, never left at its default.** The fit truncates *every* corpus item to
`max_seq_len`, whose default is 128 tokens against stimuli that run several hundred -- so the default
would silently fit on the first third of each prompt and discard the continuation the whole contrast
lives in. `reward_hacking.interp.jacobian.derive_max_seq_len` reads the corpus's own token lengths and
reports what any remaining ceiling costs; the reward-hacking `fit-lens` stage now derives its window
through the same function. The same truncation is why a large generation budget is wasted on a fit:
text past `max_seq_len` is generated and then thrown away.

**The fit corpus is the string the capture read, not the corpus row.** These stimuli are rendered --
chat template, then the teacher-forced reasoning prefix -- so a lens fitted on the bare stems would be
fitted on text no capture ever saw. The rendering goes through `games.interp_capture.render_stimuli`,
the same function and the same convention check.

**A pair whose two sides fall outside the fit window contributes only its shared prefix.** These sets
are matched-stem and the first differing token is late: index 326-415 of a 368-451 token string,
measured on this corpus 2026-08-20. The report counts the pairs whose divergence sits past
`max_seq_len` so a fit that never saw a contrast is visible in the record rather than inferred from a
weak result later.

**Fit and eval prompts are disjoint.** Reconstruction quality read on the prompts the lens was fitted
on is not a quality read. The corpus is split by position, and a corpus too small to split is a
refusal rather than a quietly overlapping read.

GPU only, and `jlens` runs from a PYTHONPATH clone rather than the lockfile
(github.com/anthropics/jacobian-lens, the commit `reward_hacking.interp.jacobian` pins). The offline
tests cover the corpus arithmetic, the ladder resolution, the payload assembly and `run` itself over a
base plus one adapted cell with every card-bound seam stubbed; only the fit is exercised on a box.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.interp_capture import (
    LadderCell,
    parse_arm_spec,
    parse_steps,
    render_stimuli,
    resolve_arm_cells,
)
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDERS,
    Stimulus,
    load_stimuli,
    sha256_of_file,
    stimuli_digest,
)
from games.lora import (
    ADAPTER_CONFIG_FILENAME,
    assert_one_adapter_config,
    attach_adapter,
    export_merged_checkpoint,
    load_adapter_base,
)
from games.provenance import git_sha
from reward_hacking.interp.directions import matched_norm_random_direction, unit
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_COMMIT,
    JLENS_LOAD_PATH,
    JLENS_UNMERGED_LOAD_PATH,
    JacobianConfig,
    LensCache,
    LensCacheKey,
    SeqLenPlan,
    _load_jlens_model,  # pyright: ignore[reportPrivateUsage]  # the shared jlens model loader
    _require_jlens,  # pyright: ignore[reportPrivateUsage]  # the shared PYTHONPATH jlens loader
    acquire_lens,
    derive_max_seq_len,
    digest_strings,
    evaluate_reconstruction,
    fit_quality_payload,
    fit_skip_first,
    resolve_weights_identity,
    transport_and_decode,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

logger = logging.getLogger("games.interp_lens_ladder")

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
REPORT_FILENAME = "lens_ladder.json"
LENS_FILENAME = "lens.pt"
MERGE_PROVENANCE_FILENAME = "merge_provenance.json"

# 10 already beats the logit/tuned lens per docs/interp-methods/jacobian-space.md, with only modest gains toward 1000.
DEFAULT_MAX_FIT_PROMPTS = 10
DEFAULT_N_EVAL_PROMPTS = 8

# Read positions per eval prompt, spread across its interior. More positions, tighter aggregate.
DEFAULT_EVAL_POSITIONS = 8

DEFAULT_TOP_K = 30

LENS_MODEL_UNMERGED = "unmerged"
LENS_MODEL_MERGED = "merged"
LENS_MODELS: tuple[str, ...] = (LENS_MODEL_UNMERGED, LENS_MODEL_MERGED)

# How many offending pair ids an error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 5


@dataclass(frozen=True)
class CorpusSplit:
    """The fit and eval halves of a lens corpus, disjoint by position."""

    fit_prompts: tuple[str, ...]
    eval_prompts: tuple[str, ...]
    token_lengths: tuple[int, ...]


def pair_divergence_indices(
    stimuli: Sequence[Stimulus], token_ids: dict[str, list[int]]
) -> dict[str, int]:
    """For each matched pair, the token index where its two sides first differ.

    The number that decides whether a truncating window keeps a contrast at all. Pairs share a stem
    by construction, so the divergence is late; a window shorter than it leaves both sides
    byte-identical, and every direction computed from them is exactly zero with nothing raising. A
    pair whose two renderings never differ is refused outright, since that is a corpus that lost its
    contrast before any window was applied.
    """
    by_pair: dict[str, list[str]] = {}
    for stimulus in stimuli:
        by_pair.setdefault(stimulus.pair_id, []).append(stimulus.stimulus_id)
    divergences: dict[str, int] = {}
    identical: list[str] = []
    for pair_id, members in by_pair.items():
        if len(members) != 2:  # noqa: PLR2004 - a matched pair is two sides by construction
            continue
        left, right = (token_ids[member] for member in members)
        index = next(
            (position for position, (a, b) in enumerate(zip(left, right, strict=False)) if a != b),
            None,
        )
        if index is None and len(left) == len(right):
            identical.append(pair_id)
            continue
        divergences[pair_id] = min(len(left), len(right)) if index is None else index
    if identical:
        raise ValueError(
            f"{len(identical)} pairs render identically ({identical[:ERROR_EXAMPLE_COUNT]}), so they "
            f"carry no contrast for any window to keep. Check the render convention before fitting."
        )
    return divergences


def split_corpus(
    prompts: Sequence[str],
    token_lengths: Sequence[int],
    *,
    max_fit_prompts: int,
    n_eval_prompts: int,
) -> CorpusSplit:
    """Split the corpus into disjoint fit and eval halves, refusing one too small to split.

    By position rather than at random, so the split is reproducible without a seed and the same
    corpus always yields the same fit. A corpus that cannot supply both is a refusal: reconstruction
    quality measured on the prompts the lens was fitted on is not a quality measurement, and the
    number would look the same either way.
    """
    needed = max_fit_prompts + n_eval_prompts
    if len(prompts) < needed:
        raise ValueError(
            f"corpus has {len(prompts)} prompts but a disjoint split needs {needed} "
            f"({max_fit_prompts} to fit + {n_eval_prompts} to score). Lower one of them "
            f"deliberately; overlapping them would make the fit-quality read circular."
        )
    return CorpusSplit(
        fit_prompts=tuple(prompts[:max_fit_prompts]),
        eval_prompts=tuple(prompts[max_fit_prompts:needed]),
        token_lengths=tuple(token_lengths),
    )


def resolve_cells(
    arms: Sequence[str], steps: Sequence[int] | None, *, skip_base: bool
) -> list[LadderCell]:
    """Build the lens agenda from `name=run_dir` arm specs, led by the un-adapted base."""
    cells: list[LadderCell] = []
    if not skip_base:
        cells.append(LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None))
    for spec in arms:
        name, run_dir = parse_arm_spec(spec)
        cells.extend(resolve_arm_cells(name, run_dir, steps))
    if not cells:
        raise ValueError("nothing to fit: pass at least one --arm, or drop --skip-base.")
    return cells


def adapter_digests(adapter_dir: Path) -> tuple[str, str]:
    """(weights sha256, config sha256) of one checkpoint: the terms a merge and a cache key both pin."""
    return (
        sha256_of_file(adapter_dir / "adapter_model.safetensors"),
        sha256_of_file(adapter_dir / ADAPTER_CONFIG_FILENAME),
    )


def merge_provenance(
    adapter_dir: Path, *, base_model: str, base_weights_identity: str, dtype: torch.dtype
) -> dict[str, str]:
    """Return what a merged export is made from, in the form the sidecar beside it records."""
    weights_sha256, config_sha256 = adapter_digests(adapter_dir)
    return {
        "adapter_weights_sha256": weights_sha256,
        "adapter_config_sha256": config_sha256,
        "base_model": base_model,
        "base_weights_identity": base_weights_identity,
        "merge_dtype": str(dtype).removeprefix("torch."),
    }


def materialize_model_dir(
    cell: LadderCell,
    *,
    base_model: str,
    base_weights_identity: str,
    merge_root: Path,
    dtype: torch.dtype,
) -> Path:
    """Return a plain model directory for one cell, merging its adapter when it has one.

    The base cell needs nothing materialised -- its hub id is already a plain model. Everything else
    is merged, which is the step that costs the ~64% delta realisation this module's docstring names.

    A merged export is reused only when the provenance sidecar beside it names this cell's adapter
    (weights and config digests), this base (name and resolved revision) and this merge dtype. The
    lens cache key digests the CURRENT adapter, so a leftover export at the same arm and step made
    from another checkpoint would otherwise be fitted and published under a key that says "the new
    adapter" -- durable and shared, where before the cache it was one wrong local run. An export
    with no sidecar, or a differing one, is re-merged. The sidecar is written last, so its presence
    means the export beside it is complete and is this cell's.
    """
    if cell.adapter_dir is None:
        return Path(base_model)
    out_dir = merge_root / cell.arm / f"step-{cell.step}"
    wanted = merge_provenance(
        cell.adapter_dir,
        base_model=base_model,
        base_weights_identity=base_weights_identity,
        dtype=dtype,
    )
    provenance_path = out_dir / MERGE_PROVENANCE_FILENAME
    if out_dir.is_dir() and any(out_dir.iterdir()):
        found = json.loads(provenance_path.read_text()) if provenance_path.is_file() else None
        if found == wanted:
            logger.info(f"reusing merged export at {out_dir}")
            return out_dir
        logger.warning(
            f"re-merging {out_dir}: the export there was not made from this cell's adapter and base "
            f"(sidecar {found}); fitting on it would publish a lens under a key naming other weights"
        )
        shutil.rmtree(out_dir)
    export_merged_checkpoint(cell.adapter_dir, base_model, out_dir, dtype=dtype)
    provenance_path.write_text(json.dumps(wanted, indent=2, sort_keys=True) + "\n")
    return out_dir


def decode_direction(  # noqa: PLR0913 - lens, model, a direction file, a layer and the decode knobs
    lens: object,
    model: object,
    tokenizer: object,
    *,
    direction_path: Path,
    layer: int,
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    """Transport one saved direction through this checkpoint's lens and decode it, against a placebo.

    The placebo is transported and decoded by exactly the same path, because the readout is a
    normalised logit rather than a calibrated one: a token list is only interpretable beside the list
    a matched-norm random direction produces under the same transform.
    """
    directions = cast("dict[int, torch.Tensor]", torch.load(direction_path, weights_only=True))
    if layer not in directions:
        raise ValueError(f"layer {layer} not in {sorted(directions)} at {direction_path}")
    if layer == max(directions):
        raise ValueError(
            f"layer {layer} is the top layer in {direction_path} and has no fitted Jacobian: a lens "
            f"fits source layers strictly below its target, and jlens defaults the target to the "
            f"final layer. Decode a lower layer."
        )
    direction = unit(directions[layer])
    placebo = matched_norm_random_direction(direction, torch.Generator().manual_seed(seed))

    def decode_id(token_id: int) -> str:
        return cast("str", tokenizer.convert_ids_to_tokens(token_id))  # pyright: ignore[reportAttributeAccessIssue]

    return {
        "direction_path": str(direction_path),
        "layer": layer,
        "real": [
            {"token": readout.token, "logit": readout.logit}
            for readout in transport_and_decode(
                lens,  # pyright: ignore[reportArgumentType]  # jlens lens exposes .transport
                model,  # pyright: ignore[reportArgumentType]  # jlens model exposes .unembed
                direction,
                layer,
                id_to_token=decode_id,
                k=top_k,
            )
        ],
        "placebo": [
            {"token": readout.token, "logit": readout.logit}
            for readout in transport_and_decode(
                lens,  # pyright: ignore[reportArgumentType]
                model,  # pyright: ignore[reportArgumentType]
                placebo,
                layer,
                id_to_token=decode_id,
                k=top_k,
            )
        ],
    }


def lens_cache_key_for_cell(  # noqa: PLR0913 - a key is every input the fit depends on
    cell: LadderCell,
    *,
    base_model: str,
    base_weights_identity: str,
    merge_dtype: torch.dtype,
    fit_prompts: Sequence[str],
    max_seq_len: int,
    skip_first: int,
    dim_batch: int,
    load_path: str,
) -> LensCacheKey:
    """Build one cell's lens cache key: which weights, how they load, and what the fit reads.

    The adapter enters as two digests -- its weight file and its config identity (rank, alpha,
    target modules: the terms of the merge, and equally of the runtime injection) -- and the base as a
    resolved revision, so a checkpoint rewritten in place or a base id that moves to a new revision
    changes the key rather than hitting a stale lens. The fit strings are digested in order, which
    pins the stimuli, their render and the corpus split in one field.

    ``load_path`` is what keeps the merged and un-merged fits apart, and it is not decoration: the two
    differ by 0.6-1.8% relative Frobenius per layer where two fits of the same weights through
    different wrapper classes differ by 0.33% (probe I6). ``merge_dtype`` is set only where a merge
    happens, so an un-merged cell's key names no merge; both paths load the base at the same bf16, so
    the base dtype is not a second axis in this ladder.
    """
    adapter_weights_sha256: str | None = None
    adapter_config_sha256: str | None = None
    if cell.adapter_dir is not None:
        adapter_weights_sha256, adapter_config_sha256 = adapter_digests(cell.adapter_dir)
    merged_cell = cell.adapter_dir is not None and load_path == JLENS_LOAD_PATH
    return LensCacheKey(
        base_model=base_model,
        base_weights_identity=base_weights_identity,
        adapter_weights_sha256=adapter_weights_sha256,
        adapter_config_sha256=adapter_config_sha256,
        merge_dtype=str(merge_dtype).removeprefix("torch.") if merged_cell else None,
        load_path=load_path,
        fit_prompts_sha256=digest_strings(fit_prompts),
        n_fit_prompts=len(fit_prompts),
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        dim_batch=dim_batch,
        jlens_commit=JLENS_COMMIT,
    )


@dataclass(frozen=True)
class LensTarget:
    """One cell's model, wrapped for `jlens`, plus how it was built and what that means for a key.

    Two shapes reach a fit. The un-merged one is a base model held resident across the ladder with
    this cell's adapter attached at runtime, wrapped through `jlens.from_hf(get_base_model())`; the
    merged one is a checkpoint materialised to disk and reloaded through the plain causal-LM loader.
    They are different lenses, so `load_path` (the cache-key term) and `label` (what the report calls
    the model) come from whichever built it rather than being assumed.
    """

    model: object
    tokenizer: object
    label: str
    load_path: str
    merged_dir: Path | None

    @property
    def merged(self) -> bool:
        """Whether this cell's weights were merged to disk rather than adapted at runtime."""
        return self.merged_dir is not None

    @property
    def lens_model(self) -> str:
        """Which `--lens-model` arm fitted this cell, read off the load path rather than the export.

        Not derivable from `merged`: on the merged arm the base cell has nothing to materialise, since
        its hub id is already a plain model, so `merged_dir` is None there while the lens is still the
        merged arm's -- loaded through the plain causal-LM loader, keyed under `JLENS_LOAD_PATH` and
        published in an out-dir the run-level field calls merged. Deriving the arm from the term that
        actually separates the two lenses keeps that one cell from labelling itself the other arm.
        """
        return LENS_MODEL_MERGED if self.load_path == JLENS_LOAD_PATH else LENS_MODEL_UNMERGED


@dataclass
class LadderLensModels:
    """Builds each cell's :class:`LensTarget`, keeping one base model resident on the un-merged path.

    The base is loaded once and the adapter re-pointed per cell (`games.lora.attach_adapter`, the same
    seam the activation capture drives), so a ladder pays one weight load rather than one per cell. Each
    cell is re-wrapped through `jlens.from_hf` AFTER its adapter is attached, which is what freezes the
    freshly injected LoRA parameters: `from_hf` sets `requires_grad_(False)` over the parameters present
    when it runs, and the fit needs gradients with respect to activations only.

    `assert_one_adapter_config` must have passed over the whole ladder before this is used, because
    re-pointing one adapter NAME is only sound while every checkpoint shares a config -- PEFT reads a
    config only when the name is new.

    One ordering invariant is checked rather than assumed: PEFT injects in place and there is no
    un-injecting, so once any adapter has been attached this resident model can no longer serve the
    UN-ADAPTED base. The agenda leads with the base cell (`resolve_cells`), so in a normal run the
    question never arises; a run that asks anyway is refused, because the alternative is a lens fitted
    on the previous checkpoint's weights and published under the base's name -- a plausible-looking
    number with no way to spot it afterwards.
    """

    base_model: str
    dtype: torch.dtype
    jl: ModuleType
    lens_model: str
    merge_root: Path
    base_weights_identity: str
    base_revision: str | None = None
    _base: object | None = None
    _tokenizer: object | None = None
    _adapted: object | None = None

    def target_for(self, cell: LadderCell) -> LensTarget:
        """Return the model this cell's lens is fitted on, by whichever path the run selected."""
        if self.lens_model == LENS_MODEL_MERGED:
            model_dir = materialize_model_dir(
                cell,
                base_model=self.base_model,
                base_weights_identity=self.base_weights_identity,
                merge_root=self.merge_root,
                dtype=self.dtype,
            )
            config = JacobianConfig(model_id=str(model_dir), source="fit_own")
            model, tokenizer = _load_jlens_model(config, self.jl)
            return LensTarget(
                model=model,
                tokenizer=tokenizer,
                label=str(model_dir),
                load_path=JLENS_LOAD_PATH,
                merged_dir=model_dir if cell.adapter_dir is not None else None,
            )
        if self._base is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._base = load_adapter_base(
                self.base_model, dtype=self.dtype, device=device, revision=self.base_revision
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.base_model, revision=self.base_revision, trust_remote_code=True
            )
            logger.info(f"lens ladder base resident for the un-merged path, {self.base_model=}")
        served: object = self._base
        if cell.adapter_dir is None and self._adapted is not None:
            raise RuntimeError(
                f"cell {cell.label} asks for the un-adapted base, but this ladder has already attached "
                f"an adapter to its resident model and PEFT injects in place, so the base cannot be "
                f"served from it any more -- fitting now would produce a lens on the last checkpoint's "
                f"weights under the base's name. Put the base cell first (the default agenda does), or "
                f"fit it in its own run."
            )
        if cell.adapter_dir is not None:
            attached = attach_adapter(
                cast("Any", self._base),
                cell.adapter_dir,
                self.base_model,
                existing=cast("Any", self._adapted),
            )
            self._adapted = attached.peft_model
            served = attached.peft_model.get_base_model()
            logger.info(
                f"attached {cell.label}'s adapter at runtime, "
                f"applied={attached.applied_adapter_weights}"
            )
        return LensTarget(
            model=self.jl.from_hf(served, self._tokenizer),
            tokenizer=self._tokenizer,
            label=f"{self.base_model} + {cell.label} (un-merged)",
            load_path=JLENS_UNMERGED_LOAD_PATH,
            merged_dir=None,
        )


def fit_cell(  # noqa: PLR0913 - one cell is a target, a corpus split, a plan and the decode knobs
    cell: LadderCell,
    jl: ModuleType,
    *,
    target: LensTarget,
    split: CorpusSplit,
    plan: SeqLenPlan,
    out_dir: Path,
    dim_batch: int,
    eval_positions: int,
    direction_path: Path | None,
    direction_layer: int,
    top_k: int,
    seed: int,
    cache: LensCache | None = None,
    cache_key: LensCacheKey | None = None,
) -> dict[str, Any]:
    """Fit (or load from the cache), score and save one checkpoint's lens; return its report entry.

    With a cache, `acquire_lens` looks the key up and skips the fit on a hit; the reconstruction read
    and the direction decode run either way, off this cell's own model, so a cached lens is scored
    exactly as a fresh one is. `lens_source` in the entry says which happened, and `fit_seconds` is the
    time the lens took to obtain by whichever route -- the model load is the caller's
    (:class:`LadderLensModels`), since the un-merged path loads once for the whole ladder.
    """
    config = JacobianConfig(
        model_id=target.label,
        source="fit_own",
        max_fit_prompts=len(split.fit_prompts),
        dim_batch=dim_batch,
        max_seq_len=plan.max_seq_len,
        top_k=top_k,
    )
    started = time.time()
    model, tokenizer = target.model, target.tokenizer
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / LENS_FILENAME
    acquired = acquire_lens(
        config, model, split.fit_prompts, jl, lens_path=lens_path, cache=cache, key=cache_key
    )
    lens = acquired.lens
    fit_seconds = round(time.time() - started, 2)
    quality = evaluate_reconstruction(
        lens,
        model,
        split.eval_prompts,
        max_seq_len=plan.max_seq_len,
        max_positions=eval_positions,
    )
    entry: dict[str, Any] = {
        "arm": cell.arm,
        "step": cell.step,
        "model_dir": target.label,
        "lens_model": target.lens_model,
        "merged": target.merged,
        "n_fit_prompts": len(split.fit_prompts),
        "n_eval_prompts": len(split.eval_prompts),
        "max_seq_len": plan.max_seq_len,
        "dim_batch": dim_batch,
        "fit_seconds": fit_seconds,
        "lens_source": acquired.source,
        "lens_cache": acquired.as_payload(),
        "lens_cache_key": None if cache_key is None else cache_key.as_payload(),
        "lens_path": str(lens_path),
        "fit_quality": fit_quality_payload(quality),
    }
    if direction_path is not None:
        entry["direction_readout"] = decode_direction(
            lens,
            model,
            tokenizer,
            direction_path=direction_path,
            layer=direction_layer,
            top_k=top_k,
            seed=seed,
        )
    logger.info(
        f"lens {acquired.source}, cell={cell.label} {fit_seconds=} "
        f"beats_logit_lens={entry['fit_quality'].get('jacobian_beats_logit_lens')}"
    )
    return entry


def build_parser() -> argparse.ArgumentParser:
    """CLI for the per-checkpoint lens ladder."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stimuli", type=Path, required=True, help="The same corpus the activation capture used."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", action="append", default=[], metavar="NAME=RUN_DIR", help="Repeatable."
    )
    parser.add_argument("--steps", default=None, help="Comma-separated checkpoint steps.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument(
        "--stimulus-render",
        choices=sorted(STIMULUS_RENDERS),
        default=STIMULUS_RENDER_TEMPLATED,
        help="Must match what the activation capture used, or the lens is fitted on other text.",
    )
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-fit-prompts", type=int, default=DEFAULT_MAX_FIT_PROMPTS)
    parser.add_argument("--n-eval-prompts", type=int, default=DEFAULT_N_EVAL_PROMPTS)
    parser.add_argument("--eval-positions", type=int, default=DEFAULT_EVAL_POSITIONS)
    parser.add_argument("--max-seq-len-ceiling", type=int, default=DEFAULT_MAX_SEQ_LEN_CEILING)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=JacobianConfig.dim_batch,
        help="Residual dims per backward pass. A memory knob, not a speed knob: total backward FLOPs "
        "are unchanged, measured 2B/9B fits gained little past 16, and 64 OOM'd both a 44 GiB and a "
        "96 GiB card. Raise it only with the VRAM to spare.",
    )
    parser.add_argument(
        "--lens-cache",
        default=None,
        help="A local directory or s3:// prefix holding fitted lenses keyed by everything the fit "
        "depends on (weights revision, adapter digests, load path, fit strings, window, dim_batch, "
        "jlens commit). A hit skips the fit; a miss fits and publishes. See LensCacheKey.",
    )
    parser.add_argument(
        "--lens-model",
        choices=LENS_MODELS,
        default=LENS_MODEL_UNMERGED,
        help="Fit on the adapter applied at runtime (unmerged, the default: the lens sees the whole "
        "trained delta) or on a merged-and-reloaded export (merged: about 64% of it, kept as the "
        "comparison arm). Separate lens-cache keys, because they are separate lenses.",
    )
    parser.add_argument(
        "--merge-root",
        type=Path,
        default=None,
        help="Where merged exports land (default <out-dir>/merged).",
    )
    parser.add_argument(
        "--keep-merged",
        action="store_true",
        help="Keep each merged export instead of deleting it after its lens is fitted.",
    )
    parser.add_argument(
        "--direction-path",
        type=Path,
        default=None,
        help="A {layer: tensor} directions file (games.interp_trajectory writes these) to transport "
        "through every checkpoint's own lens and decode, against a matched-norm placebo.",
    )
    parser.add_argument("--direction-layer", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def refuse_other_lens_model_in_out_dir(out_dir: Path, lens_model: str) -> None:
    """Refuse an out-dir whose report was written by the other `--lens-model`, before anything loads.

    `--lens-model` selects which model the lens is fitted on, and nothing else about the layout: both
    arms land their lenses at ``out_dir/<arm>/step-<N>/lens.pt`` and their report at
    ``out_dir/lens_ladder.json``. So the merged comparison arm run into an un-merged run's out-dir
    would overwrite every lens and the report with nothing going red, and the two arms would then be
    unreadable against each other, which is the one thing the comparison arm exists for. The report
    is written after every cell, succeeded or not, so it is present whenever any lens is. A report
    with no ``lens_model`` predates the flag, when merging was the only path there was.
    """
    report_path = out_dir / REPORT_FILENAME
    if not report_path.is_file():
        return
    existing = json.loads(report_path.read_text())
    found = cast("str", existing.get("lens_model", LENS_MODEL_MERGED))
    if found != lens_model:
        raise ValueError(
            f"{out_dir} already holds a lens ladder fitted with --lens-model {found}, and this run "
            f"asks for {lens_model}; both arms write lens.pt and {REPORT_FILENAME} at the same paths, "
            f"so continuing would overwrite the {found} lenses. Use another --out-dir for the "
            f"{lens_model} arm."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Fit a lens at every cell on the agenda and write the ladder report."""
    if args.direction_path is not None and args.direction_layer is None:
        raise ValueError("--direction-path needs --direction-layer: a lens transports one layer.")
    refuse_other_lens_model_in_out_dir(args.out_dir, args.lens_model)
    jl = _require_jlens()
    stimuli = load_stimuli(args.stimuli)
    cells = resolve_cells(
        cast("list[str]", args.arm), parse_steps(args.steps), skip_base=args.skip_base
    )
    merge_root = args.merge_root or (args.out_dir / "merged")

    # Every checkpoint shares the base tokenizer, so the corpus plan is a property of the corpus.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    rendered = render_stimuli(
        tokenizer,
        stimuli,
        convention=args.stimulus_render,
        enable_thinking=not args.no_thinking,
    )
    prompts = [rendered[stimulus.stimulus_id] for stimulus in stimuli]
    encoded = cast("list[list[int]]", tokenizer(prompts, add_special_tokens=False)["input_ids"])
    token_lengths = [len(ids) for ids in encoded]
    divergences = pair_divergence_indices(
        stimuli,
        {stimulus.stimulus_id: ids for stimulus, ids in zip(stimuli, encoded, strict=True)},
    )
    plan = derive_max_seq_len(token_lengths, ceiling=args.max_seq_len_ceiling)
    late = {pair: index for pair, index in divergences.items() if index >= plan.max_seq_len}
    if late and len(late) == len(divergences):
        raise ValueError(
            f"every one of {len(divergences)} pairs diverges at or past max_seq_len="
            f"{plan.max_seq_len} (earliest divergence {min(divergences.values())}), so the fit would "
            f"see nothing but shared stems -- this corpus is matched-stem and its contrast lives in "
            f"the suffix. Raise --max-seq-len-ceiling; the default 2048 covers this corpus whole."
        )
    if late:
        logger.warning(
            f"{len(late)} of {len(divergences)} pairs diverge at or past max_seq_len="
            f"{plan.max_seq_len} (worst {max(late.values())}); the fit sees only their shared stem"
        )
    split = split_corpus(
        prompts,
        token_lengths,
        max_fit_prompts=args.max_fit_prompts,
        n_eval_prompts=args.n_eval_prompts,
    )
    merge_dtype = torch.bfloat16
    cache = None if args.lens_cache is None else LensCache(root=str(args.lens_cache))
    # Resolved whether or not a cache is on: the merge provenance pins the base by it as well.
    base_weights_identity = resolve_weights_identity(args.base_model)
    if args.lens_model == LENS_MODEL_UNMERGED:
        # Re-pointing one adapter name across the ladder keeps the first config, so a rank or target
        # change later in the ladder would apply under the wrong one with nothing going red.
        assert_one_adapter_config(
            [cell.adapter_dir for cell in cells if cell.adapter_dir is not None]
        )
    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    # Read once, here: it raises on an upstream dispatch change, which must cost seconds at startup
    # rather than surface after an hours-long fit. A fit is forward and backward with no cache, so
    # every cell names the same two prefill kernels.
    kernels_bound = bound_deltanet_kernels()
    deltanet_kernel = prefill_deltanet_kernels(kernels_bound)
    models = LadderLensModels(
        base_model=args.base_model,
        dtype=merge_dtype,
        jl=jl,
        lens_model=args.lens_model,
        merge_root=merge_root,
        base_weights_identity=base_weights_identity,
    )

    report: dict[str, Any] = {
        "git_sha": git_sha(),
        "base_model": args.base_model,
        "base_weights_identity": base_weights_identity,
        "lens_cache_root": None if cache is None else cache.root,
        "stimuli_file": str(args.stimuli),
        "stimuli_sha256": stimuli_digest(stimuli),
        "stimulus_render": args.stimulus_render,
        "seq_len_plan": {
            "max_seq_len": plan.max_seq_len,
            "corpus_max_tokens": plan.corpus_max_tokens,
            "corpus_median_tokens": plan.corpus_median_tokens,
            "n_truncated": plan.n_truncated,
            "fraction_truncated": plan.fraction_truncated,
            "n_pairs": len(divergences),
            "max_pair_divergence_index": max(divergences.values()),
            "n_pairs_diverging_past_window": len(late),
        },
        "lens_model": args.lens_model,
        "delta_realization_note": (
            "merged bf16 exports realize ~64% of the trained delta (38-79% per module, measured "
            "2026-08-20), so lens numbers fitted that way are attenuated relative to the un-merged "
            "activation capture; this run fitted on the un-merged adapter, so its lens sees the whole "
            "delta and is comparable with the capture cells"
            if args.lens_model == LENS_MODEL_UNMERGED
            else "merged bf16 exports realize ~64% of the trained delta (38-79% per module, measured "
            "2026-08-20); this run's lens numbers are attenuated relative to the un-merged activation "
            "capture, and a magnitude read across the two legs is not comparing like with like"
        ),
        "deltanet_kernel_bridge": kernel_bridge,
        "deltanet_kernels_bound": kernels_bound,
        "requested_cells": [cell.label for cell in cells],
        "cells": {},
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / REPORT_FILENAME

    for cell in cells:
        cache_key = (
            None
            if cache is None
            else lens_cache_key_for_cell(
                cell,
                base_model=args.base_model,
                base_weights_identity=base_weights_identity,
                merge_dtype=merge_dtype,
                fit_prompts=split.fit_prompts,
                max_seq_len=plan.max_seq_len,
                skip_first=fit_skip_first(jl),
                dim_batch=args.dim_batch,
                load_path=(
                    JLENS_LOAD_PATH
                    if args.lens_model == LENS_MODEL_MERGED
                    else JLENS_UNMERGED_LOAD_PATH
                ),
            )
        )
        target = models.target_for(cell)
        try:
            report["cells"][cell.label] = fit_cell(
                cell,
                jl,
                target=target,
                split=split,
                plan=plan,
                out_dir=args.out_dir / cell.arm / f"step-{cell.step}",
                dim_batch=args.dim_batch,
                eval_positions=args.eval_positions,
                direction_path=args.direction_path,
                direction_layer=args.direction_layer,
                top_k=args.top_k,
                seed=args.seed,
                cache=cache,
                cache_key=cache_key,
            )
        finally:
            # Nothing here may raise: a cell that just cost hours has to reach disk even when the next
            # one will not run, which is why the binding is read before the first fit rather than here.
            if report["cells"].get(cell.label) is not None:
                report["cells"][cell.label][DELTANET_KERNEL_FIELD] = deltanet_kernel
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        # Only after the fit succeeded: a merged export left behind by a crash is the evidence for
        # why it crashed, and re-merging it costs minutes. Same rule games/run_evals.py follows.
        if target.merged_dir is not None and not args.keep_merged and target.merged_dir.is_dir():
            shutil.rmtree(target.merged_dir)
            logger.info(f"deleted merged export {target.merged_dir}; a crash would have kept it")
    logger.info(f"lens ladder finished, cells={len(report['cells'])} report={report_path}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Fit a Jacobian lens at every checkpoint of a games ladder."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
