r"""Fit one Jacobian lens for the TMAX wave and leave, beside it, everything a later read needs.

Stage 5 of the Phase 2 box, one invocation per (checkpoint, fit role). The fit itself is
:func:`reward_hacking.interp.jacobian.fit_lens` over the corpus's fit windows (the window lens) or its
full-context anchor prompts (the anchor lens), checkpointed every few prompts and resumed on relaunch.
What this driver adds is the rest of the directory, so that a read on a CPU box needs neither the
model nor ``jlens``:

* ``lens.pt`` -- the ``jlens`` lens, saved and reloaded once to prove it round-trips;
* ``stored-decode-gate.{safetensors,json}`` -- the anchors of the stored-decode gate
  (``lens_geometry.StoredDecodeGate``): at a few interior positions of every full-context anchor
  prompt, the residual after every block and the model's OWN argmax next token, recorded on the very
  weights the lens was fitted on and stamped with their fingerprint, the lens file's digest and the
  anchor corpus's digest; without this file every decode on the run would be refused, since the gate
  admits a layer only when the lens recovers those tokens there;
* ``unembed.{safetensors,json}`` -- the final norm and head as served, checked against the live
  model's argmax on the anchor rows before anything is written down as done;
* ``lens-provenance.json`` -- written LAST: the served weights, the corpus, the fit knobs, the
  reconstruction read against the logit lens, the gate's self-check per layer, the kernel binding.

A relaunch into a directory whose provenance exists and matches skips everything (the identity is the
weights fingerprint, corpus digests, role, knobs and jlens commit; a differing one is refused rather
than overwritten); a directory with a fit checkpoint but no provenance resumes the fit from it.

Three more shapes of the same stage. ``--fit-half odd|even`` fits every other fit window into its own
directory (the plan's two 30-window halves); ``--merge-halves ODD EVEN`` takes two such complete
directories, merges their lenses with ``jlens.JacobianLens.merge`` (the prompt-weighted mean of the
accumulators) and records the per-layer relative Frobenius gap between the halves, the split-half
floor every later lens comparison is read against, then records anchors, unembedding, eval and
provenance for the merged lens exactly as a fit would; ``--cross-lens-dir DIR`` (repeatable) applies
another checkpoint's lens to THIS checkpoint, reading its reconstruction against the logit lens and
its hit rate on this checkpoint's fresh anchors: the base lens applied to a TMAX checkpoint.

    scripts/resource-limits.sh --gpu -t 9h -- env PYTHONPATH=<jlens clone> \
        <repo>/.venv/bin/python -m reward_hacking.interp.tmax_lens_fit \
        --full-weights allenai/tmax-9b --revision step_500 \
        --tokenizer Qwen/Qwen3.5-9B --tokenizer-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
        --corpus-dir artifacts/reward_hacking/tmax-interp/lens-corpus-twin --fit-role fit \
        --out-dir <run>/lens/tmax-9b-step_500/window --dim-batch 8 --checkpoint-every 4
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.provenance import git_sha
from reward_hacking.interp.jacobian import (
    DEFAULT_MAX_SEQ_LEN_CEILING,
    JLENS_COMMIT,
    JacobianConfig,
    _require_jlens,  # pyright: ignore[reportPrivateUsage]  # the shared PYTHONPATH jlens loader
    evaluate_reconstruction,
    fit_lens,
    fit_quality_payload,
    fit_skip_first,
    interior_eval_positions,
    verify_lens_roundtrip,
)
from reward_hacking.interp.lens_artifacts import (
    FIT_CHECKPOINT_FILENAME,
    LENS_FILENAME,
    PROVENANCE_FILENAME,
    LensArtifactError,
    LensProvenance,
    StoredUnembed,
    assert_same_fit,
    unembed_agreement,
)
from reward_hacking.interp.lens_fit_gate import (
    LENS_LOAD_PATH,
    LensModelHandle,
    full_weights_source,
    jlens_provenance,
    load_lens_model,
)
from reward_hacking.interp.lens_geometry import (
    GateProvenance,
    StoredDecodeGate,
    anchor_rows_from_activations,
    gate_payload,
    sha256_of_file,
)
from reward_hacking.interp.lens_schedule_gates import lens_relative_diffs
from reward_hacking.interp.tmax_lens_corpus import (
    ANCHOR_MAX_TOKENS,
    ANCHORS_FILENAME,
    ROLE_ANCHOR,
    ROLE_FIT,
    CorpusItem,
    LensCorpus,
    load_lens_corpus,
)
from reward_hacking.interp.tmax_lens_reads import MIN_DECODE_LAYER

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping, Sequence
    from types import ModuleType

logger = logging.getLogger(__name__)

FIT_ROLES: tuple[str, ...] = (ROLE_FIT, ROLE_ANCHOR)
HALF_ODD = "odd"
HALF_EVEN = "even"
FIT_HALVES: tuple[str, ...] = (HALF_ODD, HALF_EVEN)
DEFAULT_DIM_BATCH = 8
DEFAULT_CHECKPOINT_EVERY = 4
DEFAULT_POSITIONS_PER_ANCHOR = 4
DEFAULT_EVAL_POSITIONS = 8


class LensFitError(RuntimeError):
    """The fit cannot produce a directory a read could trust."""


# --------------------------------------------------------------------------------------
# Anchors: the residual after every block at a few interior positions of every anchor prompt
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def record_block_outputs(blocks: Iterable[torch.nn.Module]) -> Generator[dict[int, torch.Tensor]]:
    """Capture every block's output on the next forward, keyed by block index.

    The same convention as ``jlens.hooks.ActivationRecorder`` (a block returning a tuple contributes
    its first element), kept here so the anchor capture and the fit hook the same tensor; detached,
    since anchors are recorded under ``no_grad``.
    """
    activations: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(index: int) -> Any:  # noqa: ANN401 - torch's hook signature is untyped
        def hook(_module: torch.nn.Module, _inputs: object, output: object) -> None:
            tensor = (
                output if torch.is_tensor(output) else cast("tuple[torch.Tensor, ...]", output)[0]
            )
            activations[index] = tensor.detach()

        return hook

    try:
        for index, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(index)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


@dataclass(frozen=True)
class AnchorCapture:
    """The anchor rows of every anchor prompt, plus the final-layer rows the unembed check uses."""

    residuals: torch.Tensor
    target_token_ids: torch.Tensor
    final_residuals: torch.Tensor
    model_logits: torch.Tensor
    rows: tuple[tuple[str, int], ...]
    seq_lens: dict[str, int]


def capture_anchors(
    model: Any,  # noqa: ANN401 - the jlens HFLensModel, duck-typed
    anchors: Sequence[CorpusItem],
    *,
    max_seq_len: int,
    positions_per_anchor: int,
    skip_first: int,
) -> AnchorCapture:
    """One forward per anchor prompt; rows at evenly spaced interior positions past the sink band.

    Positions follow :func:`interior_eval_positions`, the same rule the reconstruction read uses, so
    the gate's anchors sit in the regime the fit averaged over. The target at each row is the argmax
    of the model's own logits there, read through the model's own unembedding.
    """
    n_layers = int(model.n_layers)
    residual_rows: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    final_rows: list[torch.Tensor] = []
    logits_rows: list[torch.Tensor] = []
    row_ids: list[tuple[str, int]] = []
    seq_lens: dict[str, int] = {}
    for item in anchors:
        input_ids = model.encode(item.text, max_length=max_seq_len)
        seq_len = int(input_ids.shape[1])
        positions = interior_eval_positions(
            seq_len, skip_first=skip_first, max_positions=positions_per_anchor
        )
        if not positions:
            raise LensFitError(
                f"anchor {item.item_id} is {seq_len} tokens, too short for any interior position past "
                f"skip_first={skip_first}; the corpus's anchor_min_tokens should have kept it out"
            )
        with torch.no_grad(), record_block_outputs(model.layers) as activations:
            model.forward(input_ids)
            per_layer = [activations[layer][0] for layer in range(n_layers)]
            final = per_layer[n_layers - 1][positions]
            logits = model.unembed(final).float().cpu()
        rows, row_targets = anchor_rows_from_activations(per_layer, positions, logits)
        residual_rows.append(rows)
        targets.append(row_targets)
        final_rows.append(final.float().cpu())
        logits_rows.append(logits)
        row_ids.extend((item.item_id, position) for position in positions)
        seq_lens[item.item_id] = seq_len
        del activations, per_layer
    return AnchorCapture(
        residuals=torch.cat(residual_rows, dim=0),
        target_token_ids=torch.cat(targets, dim=0),
        final_residuals=torch.cat(final_rows, dim=0),
        model_logits=torch.cat(logits_rows, dim=0),
        rows=tuple(row_ids),
        seq_lens=seq_lens,
    )


def stored_unembed_from(handle: LensModelHandle) -> StoredUnembed:
    """Lift the served model's final norm weight, head and epsilon out through the jlens layout."""
    layout = handle.model.layout
    text_module = functools.reduce(getattr, str(layout.path).split("."), handle.hf_model)
    norm = getattr(text_module, str(layout.norm))
    head = getattr(handle.hf_model, str(layout.lm_head))
    rms_eps = float(handle.hf_model.config.get_text_config().rms_norm_eps)
    return StoredUnembed(
        norm_weight=norm.weight.detach().cpu(),
        lm_head_weight=head.weight.detach().cpu(),
        rms_eps=rms_eps,
    )


def gate_self_check(gate: StoredDecodeGate, lens: Any, model: Any) -> dict[str, dict[str, object]]:  # noqa: ANN401 - jlens objects
    """Run the gate on its own fresh anchors at every readable source layer, through the live unembed.

    Recorded, not gated: which layers admit is a property of the lens (coherent in a middle band), and
    the reads refuse per layer anyway. A lens admitting nowhere is something the report has to show.
    """
    readable = [layer for layer in lens.source_layers if layer >= MIN_DECODE_LAYER]
    return {str(layer): gate_payload(gate.check(lens, model, layer)) for layer in readable}


# --------------------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitPlan:
    """Everything one invocation decided before touching the GPU."""

    lens_name: str
    fit_role: str
    prompts: tuple[str, ...]
    max_seq_len: int
    anchor_max_seq_len: int
    seq_len_plan: dict[str, object]
    dim_batch: int
    checkpoint_every: int | None
    positions_per_anchor: int
    eval_positions: int
    fit_half: str | None = None
    merge_from: tuple[Path, Path] | None = None
    cross_lens_dirs: tuple[Path, ...] = ()


def half_prompts(prompts: Sequence[str], half: str) -> tuple[str, ...]:
    """Every other prompt by index: ``even`` takes 0, 2, 4, ...; ``odd`` takes 1, 3, 5, ..."""
    if half not in FIT_HALVES:
        raise LensFitError(f"--fit-half must be one of {FIT_HALVES}, got {half!r}")
    start = 0 if half == HALF_EVEN else 1
    return tuple(prompts[start::2])


def plan_fit(corpus: LensCorpus, args: argparse.Namespace, *, label: str) -> FitPlan:
    """Choose the prompts and the window from the corpus and the arguments."""
    role = cast("str", args.fit_role)
    if role not in FIT_ROLES:
        raise LensFitError(f"--fit-role must be one of {FIT_ROLES}, got {role!r}")
    half = cast("str | None", getattr(args, "fit_half", None))
    merge_from = cast("tuple[Path, Path] | None", getattr(args, "merge_halves", None))
    cross = tuple(cast("Sequence[Path]", getattr(args, "cross_lens_dir", ()) or ()))
    if (half is not None or merge_from is not None) and role != ROLE_FIT:
        raise LensFitError(
            "--fit-half and --merge-halves apply to the window lens (--fit-role fit) only"
        )
    if half is not None and merge_from is not None:
        raise LensFitError("--fit-half fits one half; --merge-halves merges two; not both")
    prompts: Sequence[str] = corpus.fit_prompts if role == ROLE_FIT else corpus.anchor_prompts
    if half is not None:
        prompts = half_prompts(prompts, half)
    max_fit_prompts = cast("int | None", args.max_fit_prompts)
    if max_fit_prompts is not None:
        prompts = prompts[:max_fit_prompts]
    ceiling = cast("int | None", args.max_seq_len_ceiling)
    if ceiling is None:
        ceiling = DEFAULT_MAX_SEQ_LEN_CEILING if role == ROLE_FIT else ANCHOR_MAX_TOKENS
    plan = corpus.plan(role, ceiling=ceiling)
    stem = "window" if role == ROLE_FIT else "anchor"
    return FitPlan(
        lens_name=f"{label}/{stem}" if half is None else f"{label}/{stem}-{half}",
        fit_role=role,
        prompts=tuple(prompts),
        max_seq_len=plan.max_seq_len,
        anchor_max_seq_len=corpus.plan(
            ROLE_ANCHOR, ceiling=cast("int", args.anchor_max_seq_len)
        ).max_seq_len,
        seq_len_plan=plan.as_payload(),
        dim_batch=cast("int", args.dim_batch),
        checkpoint_every=cast("int | None", args.checkpoint_every),
        positions_per_anchor=cast("int", args.positions_per_anchor),
        eval_positions=cast("int", args.eval_positions),
        fit_half=half,
        merge_from=None if merge_from is None else (Path(merge_from[0]), Path(merge_from[1])),
        cross_lens_dirs=tuple(Path(directory) for directory in cross),
    )


def jlens_commit_for_provenance(jlens_facts: Mapping[str, object]) -> str:
    """Return the jlens commit as verified, or say it could not be: never the pin dressed as a fact.

    ``jlens_provenance`` refuses a git checkout at another commit and records ``None`` for a copy
    that is not a checkout at all; that copy may be the pinned code or may not, so the provenance
    says so rather than writing the pin down as if the check had passed.
    """
    commit = jlens_facts["commit"]
    if isinstance(commit, str) and commit:
        return commit
    return f"unverified (not a git checkout; pinned {JLENS_COMMIT})"


def wanted_identity(
    plan: FitPlan,
    *,
    handle: LensModelHandle,
    corpus: LensCorpus,
    skip_first: int,
    jlens_commit: str,
) -> dict[str, object]:
    """Return the identity a directory has to carry to already be this fit."""
    return {
        "lens_name": plan.lens_name,
        "model_label": handle.facts.label,
        "weights_identity": handle.weights_identity,
        "weights_fingerprint": handle.facts.fingerprint,
        "load_path": LENS_LOAD_PATH,
        "corpus_digests": dict(cast("dict[str, str]", corpus.sidecar["digests"])),
        "fit_role": plan.fit_role,
        "n_fit_prompts": len(plan.prompts),
        "dim_batch": plan.dim_batch,
        "max_seq_len": plan.max_seq_len,
        "skip_first": skip_first,
        "jlens_commit": jlens_commit,
    }


HALF_IDENTITY_FIELDS: tuple[str, ...] = (
    "model_label",
    "weights_identity",
    "weights_fingerprint",
    "load_path",
    "corpus_digests",
    "fit_role",
    "dim_batch",
    "max_seq_len",
    "skip_first",
    "jlens_commit",
)
"""What two halves must agree on with the merged fit before their lenses may be averaged."""


@dataclass(frozen=True)
class MergedHalves:
    """Two complete half directories, their lenses loaded, ready to average."""

    lenses: tuple[Any, Any]
    provenances: tuple[LensProvenance, LensProvenance]
    split_half_relative_diff_by_layer: dict[int, float]

    @property
    def n_prompts(self) -> int:
        """How many prompts the merged lens averages over."""
        return sum(int(lens.n_prompts) for lens in self.lenses)

    def as_payload(self) -> dict[str, object]:
        """Return the merged provenance block: where the halves came from and how far apart they sit."""
        return {
            "halves": [
                {
                    "lens_name": provenance.lens_name,
                    "lens_sha256": provenance.lens_sha256,
                    "n_prompts_fitted": provenance.n_prompts_fitted,
                }
                for provenance in self.provenances
            ],
            "split_half_relative_diff_by_layer": {
                str(layer): diff for layer, diff in self.split_half_relative_diff_by_layer.items()
            },
        }


def load_halves(
    directories: tuple[Path, Path], *, jl: ModuleType, identity: Mapping[str, object]
) -> MergedHalves:
    """Open two complete half directories and hold each to the merged fit's identity, by name.

    Everything but the lens name and the prompt count has to agree: two halves fitted on different
    weights, corpora, windows or batches are two lenses, not two halves of one.
    """
    provenances: list[LensProvenance] = []
    lenses: list[Any] = []
    for directory in directories:
        if not (directory / PROVENANCE_FILENAME).is_file():
            raise LensArtifactError(
                f"{directory} has no {PROVENANCE_FILENAME}; a half whose fit did not finish cannot be merged"
            )
        provenance = LensProvenance.load(directory)
        lens_path = directory / LENS_FILENAME
        if sha256_of_file(lens_path) != provenance.lens_sha256:
            raise LensArtifactError(
                f"{lens_path} is not the lens file {directory}'s provenance describes"
            )
        if provenance.fit_role != ROLE_FIT or not provenance.lens_name.endswith(
            tuple(f"-{h}" for h in FIT_HALVES)
        ):
            raise LensArtifactError(
                f"{directory} holds {provenance.lens_name!r}, which is not a half of the window lens"
            )
        differing = {
            key: (provenance.identity[key], identity[key])
            for key in HALF_IDENTITY_FIELDS
            if provenance.identity[key] != identity[key]
        }
        if differing:
            raise LensArtifactError(
                f"{directory} was fitted differently from the merge asked for: "
                + "; ".join(
                    f"{key}: half {have!r}, merge {want!r}"
                    for key, (have, want) in differing.items()
                )
            )
        provenances.append(provenance)
        lenses.append(jl.JacobianLens.load(str(lens_path)))
    if provenances[0].lens_name == provenances[1].lens_name:
        raise LensArtifactError(
            f"both directories hold {provenances[0].lens_name!r}; a merge needs the odd and the even half"
        )
    return MergedHalves(
        lenses=(lenses[0], lenses[1]),
        provenances=(provenances[0], provenances[1]),
        split_half_relative_diff_by_layer=lens_relative_diffs(
            lenses[0].jacobians, lenses[1].jacobians
        ),
    )


def cross_lens_reads(  # noqa: PLR0913 - a cross read is the other lens, this model, its anchors and the eval knobs
    directories: Sequence[Path],
    *,
    jl: ModuleType,
    handle: LensModelHandle,
    gate: StoredDecodeGate,
    eval_prompts: Sequence[str],
    max_seq_len: int,
    eval_positions: int,
) -> dict[str, dict[str, object]]:
    """Apply other checkpoints' lenses to this checkpoint: reconstruction and hit rates on its anchors.

    Each directory has to be complete and fitted on OTHER weights (a lens applied to its own checkpoint
    is the fit's own report); what is recorded is the other lens's name, fingerprint and digest beside
    its reconstruction against the logit lens on this checkpoint and the gate's per-layer verdict when
    its transport is used on this checkpoint's fresh anchors.
    """
    out: dict[str, dict[str, object]] = {}
    for directory in directories:
        if not (directory / PROVENANCE_FILENAME).is_file():
            raise LensArtifactError(
                f"{directory} has no {PROVENANCE_FILENAME}; an incomplete lens is not applied"
            )
        provenance = LensProvenance.load(directory)
        lens_path = directory / LENS_FILENAME
        if sha256_of_file(lens_path) != provenance.lens_sha256:
            raise LensArtifactError(f"{lens_path} is not the lens file its provenance describes")
        if provenance.weights_fingerprint == handle.facts.fingerprint:
            raise LensArtifactError(
                f"{directory} was fitted on this checkpoint's own weights ({handle.facts.label}); a "
                f"cross-lens read applies another checkpoint's lens"
            )
        other = jl.JacobianLens.load(str(lens_path))
        quality = evaluate_reconstruction(
            other, handle.model, eval_prompts, max_seq_len=max_seq_len, max_positions=eval_positions
        )
        out[provenance.lens_name] = {
            "lens_dir": str(directory),
            "lens_sha256": provenance.lens_sha256,
            "lens_weights_fingerprint": provenance.weights_fingerprint,
            "lens_model_label": provenance.model_label,
            "fit_quality_on_this_checkpoint": fit_quality_payload(quality),
            "gate_by_layer_on_this_checkpoints_anchors": gate_self_check(gate, other, handle.model),
        }
        logger.info(
            f"cross lens {provenance.lens_name} applied to {handle.facts.label}: beats_logit_lens="
            f"{out[provenance.lens_name]['fit_quality_on_this_checkpoint'].get('jacobian_beats_logit_lens')}"  # pyright: ignore[reportAttributeAccessIssue]
        )
    return out


def acquire_fit_lens(
    plan: FitPlan,
    *,
    handle: LensModelHandle,
    jl: ModuleType,
    identity: dict[str, object],
    out_dir: Path,
) -> tuple[Any, MergedHalves | None]:
    """Fit the lens (resumable), or merge two complete halves into it; the merge fixes the prompt count."""
    if plan.merge_from is not None:
        merged = load_halves(plan.merge_from, jl=jl, identity=identity)
        identity["n_fit_prompts"] = merged.n_prompts
        logger.info(
            f"merged {merged.provenances[0].lens_name} and {merged.provenances[1].lens_name}: "
            f"n_prompts={merged.n_prompts} split_half_max_relative_diff="
            f"{max(merged.split_half_relative_diff_by_layer.values()):.3e}"
        )
        return jl.JacobianLens.merge(list(merged.lenses)), merged
    config = JacobianConfig(
        model_id=handle.facts.label,
        source="fit_own",
        max_fit_prompts=len(plan.prompts),
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        checkpoint_path=out_dir / FIT_CHECKPOINT_FILENAME,
        checkpoint_every=plan.checkpoint_every,
        resume=True,
    )
    return fit_lens(config, handle.model, plan.prompts, jl), None


def fit_and_write(  # noqa: PLR0913 - the stage is one long, linear recipe; its steps are named in the log
    handle: LensModelHandle,
    corpus: LensCorpus,
    plan: FitPlan,
    *,
    jl: ModuleType,
    out_dir: Path,
    corpus_dir: Path,
    jlens_facts: dict[str, object],
    kernel_bridge: dict[str, object],
    invocation: str,
) -> LensProvenance:
    """Fit (or resume), save, verify, record the anchors and the unembedding, then write the provenance."""
    skip_first = fit_skip_first(jl)
    jlens_commit = jlens_commit_for_provenance(jlens_facts)
    identity = wanted_identity(
        plan, handle=handle, corpus=corpus, skip_first=skip_first, jlens_commit=jlens_commit
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lens_path = out_dir / LENS_FILENAME
    if (out_dir / PROVENANCE_FILENAME).is_file():
        existing = LensProvenance.load(out_dir)
        assert_same_fit(existing, identity, out_dir=out_dir)
        if sha256_of_file(lens_path) != existing.lens_sha256:
            raise LensArtifactError(
                f"{out_dir} carries a provenance for a lens file that is not the one on disk"
            )
        logger.info(f"lens directory already complete and identical, {out_dir=}; nothing to fit")
        return existing

    kernels_bound = bound_deltanet_kernels()
    deltanet_kernel = prefill_deltanet_kernels(kernels_bound)
    started = time.time()
    lens, merged = acquire_fit_lens(plan, handle=handle, jl=jl, identity=identity, out_dir=out_dir)
    fit_seconds = round(time.time() - started, 1)
    lens.save(str(lens_path))
    verify_lens_roundtrip(jl, lens, lens_path)
    lens_sha256 = sha256_of_file(lens_path)

    started = time.time()
    capture = capture_anchors(
        handle.model,
        corpus.anchors,
        max_seq_len=plan.anchor_max_seq_len,
        positions_per_anchor=plan.positions_per_anchor,
        skip_first=skip_first,
    )
    anchors_path = corpus_dir / ANCHORS_FILENAME
    gate = StoredDecodeGate(
        residuals=capture.residuals,
        target_token_ids=capture.target_token_ids,
        lens_name=plan.lens_name,
        provenance=GateProvenance(
            lens_name=plan.lens_name,
            model_label=handle.facts.label,
            weights_identity=handle.weights_identity,
            weights_fingerprint=handle.facts.fingerprint,
            lens_sha256=lens_sha256,
            anchors_path=str(anchors_path),
            anchors_sha256=sha256_of_file(anchors_path),
            anchor_digest=str(cast("dict[str, str]", corpus.sidecar["digests"])[ROLE_ANCHOR]),
            anchor_rows=capture.rows,
            skip_first=skip_first,
            jlens_commit=jlens_commit,
            load_path=LENS_LOAD_PATH,
        ),
    )
    gate.save(out_dir)
    self_check = gate_self_check(gate, lens, handle.model)
    anchors_seconds = round(time.time() - started, 1)

    stored = stored_unembed_from(handle)
    agreement = unembed_agreement(stored, capture.model_logits, capture.final_residuals)
    if not agreement["argmax_all_agree"]:
        raise LensFitError(
            f"the stored unembedding disagrees with the live model's argmax on "
            f"{agreement['n_rows'] - int(agreement['argmax_agreement'])} of {agreement['n_rows']} anchor "
            f"rows (max |logit diff| {agreement['max_abs_logit_diff']:.3g}); a CPU read through it would "
            f"gate on the wrong tokens, so nothing is written as complete"
        )
    stored.save(out_dir, weights_fingerprint=handle.facts.fingerprint)

    started = time.time()
    quality = evaluate_reconstruction(
        lens,
        handle.model,
        corpus.eval_prompts,
        max_seq_len=plan.max_seq_len,
        max_positions=plan.eval_positions,
    )
    cross = cross_lens_reads(
        plan.cross_lens_dirs,
        jl=jl,
        handle=handle,
        gate=gate,
        eval_prompts=corpus.eval_prompts,
        max_seq_len=plan.max_seq_len,
        eval_positions=plan.eval_positions,
    )
    eval_seconds = round(time.time() - started, 1)

    provenance = LensProvenance(
        lens_name=plan.lens_name,
        model_label=handle.facts.label,
        weights_identity=handle.weights_identity,
        weights_fingerprint=handle.facts.fingerprint,
        load_path=LENS_LOAD_PATH,
        tokenizer={"source": handle.tokenizer_source},
        corpus_dir=str(corpus_dir),
        corpus_digests=dict(cast("dict[str, str]", corpus.sidecar["digests"])),
        fit_role=plan.fit_role,
        n_fit_prompts=cast("int", identity["n_fit_prompts"]),
        n_prompts_fitted=int(lens.n_prompts),
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        skip_first=skip_first,
        checkpoint_every=plan.checkpoint_every,
        jlens_commit=jlens_commit,
        lens_sha256=lens_sha256,
        report={
            "git_sha": git_sha(),
            "invocation": invocation,
            "model": handle.as_payload(),
            "jlens": jlens_facts,
            "deltanet_kernel_bridge": kernel_bridge,
            "deltanet_kernels_bound": kernels_bound,
            DELTANET_KERNEL_FIELD: deltanet_kernel,
            "seq_len_plan": plan.seq_len_plan,
            "timings": {
                "load_seconds": round(handle.load_seconds, 1),
                "fit_seconds": fit_seconds,
                "anchors_seconds": anchors_seconds,
                "eval_seconds": eval_seconds,
            },
            "fit_quality": fit_quality_payload(quality),
            "gate": {
                "n_anchor_prompts": len(corpus.anchors),
                "n_anchors": gate.n_anchors,
                "positions_per_anchor": plan.positions_per_anchor,
                "anchor_max_seq_len": plan.anchor_max_seq_len,
                "anchor_seq_lens": capture.seq_lens,
                "anchors_are_fit_text": plan.fit_role == ROLE_ANCHOR,
                "self_check_by_layer": self_check,
            },
            "unembed_agreement": agreement,
            "fit_half": plan.fit_half,
            "merged_from": None if merged is None else merged.as_payload(),
            "cross_lens": cross,
        },
    )
    provenance.save(out_dir)
    checkpoint = out_dir / FIT_CHECKPOINT_FILENAME
    if checkpoint.exists():
        checkpoint.unlink()
        logger.info(f"fit checkpoint removed after the lens was saved and verified, {checkpoint=}")
    admitted = sorted(int(layer) for layer, block in self_check.items() if block["admitted"])
    logger.info(
        f"lens written, {out_dir=} lens={plan.lens_name} n_prompts={provenance.n_prompts_fitted} "
        f"fit_seconds={fit_seconds} beats_logit_lens="
        f"{provenance.report['fit_quality'].get('jacobian_beats_logit_lens')} "
        f"gate_admits_layers={admitted}"
    )
    return provenance


def build_parser() -> argparse.ArgumentParser:
    """CLI for one lens fit."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--full-weights",
        required=True,
        help="a hub repo id (pair with --revision) or a local checkpoint directory to fit on",
    )
    parser.add_argument("--revision", default=None, help="hub revision; required for a hub id")
    parser.add_argument(
        "--label",
        default=None,
        help="what the lens is called (default: the resolved '<repo>@<revision>' label)",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer of record to encode with (default: the checkpoint's own); the wave passes "
        "the base at its pinned commit",
    )
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument(
        "--corpus-dir", type=Path, required=True, help="a tmax_lens_corpus directory"
    )
    parser.add_argument("--fit-role", choices=FIT_ROLES, default=ROLE_FIT)
    parser.add_argument(
        "--fit-half",
        choices=FIT_HALVES,
        default=None,
        help="fit every other fit window (by index) into its own directory: the two 30-window halves",
    )
    parser.add_argument(
        "--merge-halves",
        type=Path,
        nargs=2,
        metavar=("ODD_DIR", "EVEN_DIR"),
        default=None,
        help="merge two complete --fit-half directories instead of fitting; anchors, unembedding, "
        "eval and provenance are recorded for the merged lens, with the split-half gap per layer",
    )
    parser.add_argument(
        "--cross-lens-dir",
        type=Path,
        action="append",
        default=[],
        help="apply another checkpoint's complete lens directory to this checkpoint (repeatable): "
        "its reconstruction here and its hit rate on this checkpoint's anchors go in the report",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="write the fit's fp32 accumulator every N prompts (about 2 GB at 9B); 0 means only at the end",
    )
    parser.add_argument(
        "--max-fit-prompts",
        type=int,
        default=None,
        help="cap the prompt list (a smoke knob; the corpus decides the real count)",
    )
    parser.add_argument(
        "--max-seq-len-ceiling",
        type=int,
        default=None,
        help="cap on the fit window (default 2048 for the window lens, the corpus's anchor cap for "
        "the anchor lens); the corpus plan reports what a cap truncates",
    )
    parser.add_argument(
        "--anchor-max-seq-len",
        type=int,
        default=ANCHOR_MAX_TOKENS,
        help="the window the gate's anchor prompts are forwarded at (full context by default)",
    )
    parser.add_argument("--positions-per-anchor", type=int, default=DEFAULT_POSITIONS_PER_ANCHOR)
    parser.add_argument("--eval-positions", type=int, default=DEFAULT_EVAL_POSITIONS)
    return parser


def run(args: argparse.Namespace) -> LensProvenance:
    """Resolve the checkpoint, load it, and fit."""
    if args.checkpoint_every is not None and args.checkpoint_every == 0:
        args.checkpoint_every = None
    jl = _require_jlens()
    jlens_facts = jlens_provenance(jl)
    corpus_dir = cast("Path", args.corpus_dir)
    corpus = load_lens_corpus(corpus_dir)
    # Bridged before the load freezes each kernel's dispatch, the order the capture ladder uses.
    kernel_bridge = bridge_and_check_decode_kernel()
    handle = load_lens_model(
        full_weights_source(cast("str", args.full_weights), cast("str | None", args.revision)),
        jl=jl,
        tokenizer_id=cast("str | None", args.tokenizer),
        tokenizer_revision=cast("str | None", args.tokenizer_revision),
    )
    label = cast("str | None", args.label) or handle.facts.label
    plan = plan_fit(corpus, args, label=label)
    logger.info(
        f"fit planned, lens={plan.lens_name} n_prompts={len(plan.prompts)} "
        f"max_seq_len={plan.max_seq_len} dim_batch={plan.dim_batch} "
        f"checkpoint_every={plan.checkpoint_every} anchor_max_seq_len={plan.anchor_max_seq_len}"
    )
    return fit_and_write(
        handle,
        corpus,
        plan,
        jl=jl,
        out_dir=cast("Path", args.out_dir),
        corpus_dir=corpus_dir,
        jlens_facts=jlens_facts,
        kernel_bridge=kernel_bridge,
        invocation=" ".join(sys.argv),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Fit one lens and write its directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("jlens").setLevel(logging.INFO)
    provenance = run(build_parser().parse_args(argv))
    logger.info("provenance: %s", json.dumps(provenance.identity, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
