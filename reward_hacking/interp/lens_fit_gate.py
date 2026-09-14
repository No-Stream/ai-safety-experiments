r"""Run the Jacobian-lens fit gates on a checkpoint and refuse on any failure.

Every Jacobian number in the TMAX wave rests on ``jlens.fit`` pushing autograd correctly through the
Gated DeltaNet blocks, on the kernels actually bound in the process, at a window and batch that fit
the card, on ids every checkpoint tokenizes the same way, with a fit that can be resumed. None of
that is checked by the fit itself, so before any 9B lens is bought this entry point runs the five
gates and exits non-zero with a JSON report naming the failures:

* ``autograd`` -- DeltaNet autograd with the detach sabotage reading exactly 0.0
  (:func:`reward_hacking.interp.lens_deltanet_gates.probe_recurrence_autograd`);
* ``kernel`` -- the fused fla chunk kernel's backward against the pure-torch reference, with a
  fused-versus-fused floor (:func:`~reward_hacking.interp.lens_deltanet_gates.audit_chunk_kernel_backward`);
* ``sweep`` -- ``dim_batch`` over the candidates at the real window, chosen with headroom
  (:func:`reward_hacking.interp.lens_schedule_gates.sweep_dim_batch`);
* ``tokens`` -- the 9B-family tokenizers agree and the other-family one differs
  (:func:`~reward_hacking.interp.lens_schedule_gates.token_identity`);
* ``resume`` -- a fit resumed from an fp32 checkpoint equals a straight fit
  (:func:`~reward_hacking.interp.lens_schedule_gates.resume_equality`).

The report is written after every gate, so a crash mid-way leaves the evidence of the gates that
ran. ``jlens`` runs from a PYTHONPATH clone pinned at
:data:`reward_hacking.interp.jacobian.JLENS_COMMIT`; the report records the clone's commit and the
run refuses another. Prompts come from either a ``reward_hacking.interp.tmax_lens_corpus`` directory
or the private cooperation ``--fit-stimuli`` JSONL. The cooperation path uses the capture renderer,
records its exact corpus/render identities and derives the gate window from those rendered prompts.
Without either corpus input the gates use benign built-in text.

The checkpoint is resolved and loaded by :func:`load_lens_model`, the loader the lens fit
(``reward_hacking.interp.tmax_lens_fit``) shares: ``games.eval_model.resolve_full_weights`` pins the
revision to a commit and proves the bytes against the hub's digests, and
``tmax_full_weights.load_full_weights_model`` builds the same module tree the capture ladder's cells
were captured through, so the gates validate the autograd path of the model the lens is fitted on and
the lens decodes the activations of the model it was fitted on. A hub checkpoint needs its
``--revision`` spelled out (the TMAX repos put a different step on every branch); a local directory
takes none.

    scripts/resource-limits.sh --gpu -t 10m -- env PYTHONPATH=<jlens clone> \
        <repo>/.venv/bin/python -m reward_hacking.interp.lens_fit_gate \
        --model-id Qwen/Qwen3.5-0.8B --revision main --corpus-dir <lens corpus> --out <report.json>
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.cooperation_lens import prepare_fit_prompts
from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    QWEN3_5_MODELING_MODULE,
    bound_deltanet_kernels,
    prefill_deltanet_kernels,
)
from games.eval_model import FullWeightsFacts, FullWeightsSource, resolve_full_weights
from games.tokenizer_identity import tokenizer_content_sha256
from reward_hacking.interp.jacobian import (
    JLENS_COMMIT,
    JacobianConfig,
    _require_jlens,  # pyright: ignore[reportPrivateUsage]  # the shared PYTHONPATH jlens loader
    fit_skip_first,
    resolve_weights_identity,
)
from reward_hacking.interp.lens_deltanet_gates import (
    DEFAULT_POSITION_GAP,
    KERNEL_AUDIT_DIMS,
    KERNEL_AUDIT_TOLERANCE_BF16,
    GateFailureError,
    RowEstimator,
    audit_chunk_kernel_backward,
    probe_recurrence_autograd,
)
from reward_hacking.interp.lens_schedule_gates import (
    DIM_BATCH_CANDIDATES,
    HEADROOM_FRACTION,
    RESUME_TOLERANCE,
    load_tokenizer_encoders,
    resume_equality,
    sweep_dim_batch,
    token_identity,
)
from reward_hacking.interp.tmax_full_weights import LoadingReport, load_full_weights_model
from reward_hacking.interp.tmax_lens_corpus import ROLE_FIT, LensCorpus, load_lens_corpus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    from torch import nn
    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

LENS_LOAD_PATH = (
    "games.eval_model.resolve_full_weights; "
    "reward_hacking.interp.tmax_full_weights.load_full_weights_model(sdpa, bfloat16); jlens.from_hf"
)
"""Names the loader every wave lens is fitted through, for the lens provenance.

The same builder and attention backend as the capture ladder's cells, so the lens and the activations
it decodes come from one module tree; a lens fitted through another loader is a different lens (the
games ladder measured a 0.33% floor between two loaders of the same weights) and is not this one.
"""

REFERENCE_TOKENIZER: tuple[str, str | None] = (
    "Qwen/Qwen3.5-9B",
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
)
"""The wave's tokenizer of record: the base checkpoint at the commit the twin corpus pins."""
IDENTICAL_TOKENIZERS: tuple[tuple[str, str | None], ...] = (
    ("hamishivi/Qwen3.5-9B", None),
    ("allenai/tmax-9b", None),
)
"""Every 9B-family checkpoint the wave feeds the base's ids to; each must tokenize identically."""
DIFFERENT_TOKENIZERS: tuple[tuple[str, str | None], ...] = (("Qwen/Qwen3-8B", None),)
"""The older Qwen3 family, whose tokenizer differs: the comparison's own negative control."""

GATE_AUTOGRAD = "autograd"
GATE_KERNEL = "kernel"
GATE_SWEEP = "sweep"
GATE_TOKENS = "tokens"
GATE_RESUME = "resume"
ALL_GATES: tuple[str, ...] = (GATE_AUTOGRAD, GATE_KERNEL, GATE_SWEEP, GATE_TOKENS, GATE_RESUME)

DEFAULT_WORK_DIR = Path("/var/tmp/lens-fit-gate")  # noqa: S108 - the box's scratch disk, not /tmp's inode-capped tmpfs

FALLBACK_PROMPT = (
    "The harbour at Valparaiso was rebuilt three times in the nineteenth century, each time after "
    "a different disaster. The first rebuilding followed a fire that began in a chandler's "
    "warehouse and spread along the waterfront for two days. Contemporary accounts disagree about "
    "the number of vessels lost, and the port authority's own ledger for that year is missing "
    "several pages, so historians have relied instead on insurance filings made in London and "
    "Hamburg during the following spring. Sedimentary basins accumulate material at rates that "
    "vary by three orders of magnitude depending on tectonic setting. A foreland basin adjacent to "
    "an actively rising range may receive several millimetres of sediment per year, while an "
    "abyssal plain far from any continental margin receives a fraction of a millimetre per thousand "
    "years. Because compaction proceeds unevenly, converting a measured thickness into an elapsed "
    "interval requires assumptions about the porosity of the original deposit. "
)
"""Benign prose for a run with no corpus; repeated until it fills the requested window."""


def jlens_provenance(jl: ModuleType) -> dict[str, object]:
    """Record where the PYTHONPATH jlens came from and its commit; refuse a commit other than the pin."""
    package_dir = Path(cast("str", jl.__file__)).resolve().parent
    finished = subprocess.run(  # noqa: S603 - fixed argv over a resolved path
        ["git", "-C", str(package_dir), "rev-parse", "HEAD"],  # noqa: S607 - git on PATH, no user input
        capture_output=True,
        text=True,
        check=False,
    )
    commit = finished.stdout.strip() if finished.returncode == 0 else None
    if commit is not None and not commit.startswith(JLENS_COMMIT):
        raise GateFailureError(
            f"jlens at {package_dir} is commit {commit[:12]}, not the pinned {JLENS_COMMIT}; a "
            f"lens fitted through another jlens is not comparable with the wave's"
        )
    if commit is None:
        logger.warning(
            "jlens at %s is not a git checkout; its commit cannot be verified", package_dir
        )
    return {"path": str(package_dir), "commit": commit, "pinned_commit": JLENS_COMMIT}


def full_weights_source(model_id: str, revision: str | None) -> FullWeightsSource:
    """Name a local directory as itself and anything else as a hub repo at the revision it has to give."""
    path = Path(model_id)
    if path.is_dir():
        return FullWeightsSource(repo_id=None, revision=revision, local_dir=path)
    return FullWeightsSource(repo_id=model_id, revision=revision, local_dir=None)


def weights_identity(facts: FullWeightsFacts) -> str:
    """One string naming the weights: the hub commit, or the fingerprint of a local directory."""
    if facts.commit_sha is not None:
        return f"hf:{facts.commit_sha}"
    return f"fingerprint:{facts.fingerprint}"


@dataclass
class LensModelHandle:
    """A checkpoint loaded for a fit or its gates: the jlens wrapper plus what proves which weights it is."""

    model: Any
    hf_model: PreTrainedModel
    tokenizer: Any
    facts: FullWeightsFacts
    loading_report: LoadingReport
    layer_types: tuple[str, ...]
    conv_reach: int
    tokenizer_source: str
    load_seconds: float
    device_facts: dict[str, object]

    @property
    def blocks(self) -> nn.ModuleList:
        """Return the residual blocks jlens hooks."""
        return cast("nn.ModuleList", self.model.layers)

    def forward(self, input_ids: torch.Tensor) -> object:
        """Run the text stack with no LM head and no cache, as the fit runs it."""
        return self.model.forward(input_ids)

    @property
    def weights_identity(self) -> str:
        """The hub commit or local fingerprint the weights resolved to."""
        return weights_identity(self.facts)

    def as_payload(self) -> dict[str, object]:
        """Return the report block: what loaded, from where, onto what."""
        return {
            "model_label": self.facts.label,
            "snapshot_dir": str(self.facts.snapshot_dir),
            "weights_identity": self.weights_identity,
            "resolved_weights_identity": (
                self.weights_identity
                if self.facts.commit_sha is not None
                else resolve_weights_identity(str(self.facts.snapshot_dir))
            ),
            "weights_fingerprint": self.facts.fingerprint,
            "declares_vision_config": self.facts.declares_vision_config,
            "chat_template_sha256": self.facts.chat_template_sha256,
            "loading_report": self.loading_report.to_payload(),
            "load_path": LENS_LOAD_PATH,
            "hf_class": type(self.hf_model).__name__,
            "tokenizer_source": self.tokenizer_source,
            "tokenizer_content_sha256": tokenizer_content_sha256(self.tokenizer),
            "load_seconds": round(self.load_seconds, 1),
            "n_layers": int(self.model.n_layers),
            "d_model": int(self.model.d_model),
            "n_linear_attention": sum(kind == "linear_attention" for kind in self.layer_types),
            "n_full_attention": sum(kind == "full_attention" for kind in self.layer_types),
            **self.device_facts,
        }


def device_facts() -> dict[str, object]:
    """Record the card and the library versions a load landed on."""
    transformers = importlib.import_module("transformers")
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "vram_after_load_bytes": int(torch.cuda.memory_allocated()),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }


def load_lens_model(
    source: FullWeightsSource,
    *,
    jl: ModuleType,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> LensModelHandle:
    """Resolve, verify and load a checkpoint the way the capture ladder does, wrapped for jlens.

    ``resolve_full_weights`` pins the revision to a commit and holds every tensor file to the hub's
    digest; ``load_full_weights_model`` builds the capture's module tree (sdpa, bf16) and gates the
    loading report to zero missing or unexpected language-model keys. The tokenizer defaults to the
    snapshot's own; the wave passes its tokenizer of record, which the token-identity gate has held
    every 9B checkpoint to.
    """
    if not torch.cuda.is_available():
        raise GateFailureError(
            "the Qwen3.5 CPU route is closed (Triton linear attention); a lens needs a GPU"
        )
    transformers = importlib.import_module("transformers")
    started = time.perf_counter()
    facts = resolve_full_weights(source)
    hf_model, loading_report = load_full_weights_model(
        facts, dtype=torch.bfloat16, device=torch.device("cuda")
    )
    tokenizer_source = tokenizer_id if tokenizer_id is not None else str(facts.snapshot_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=tokenizer_revision if tokenizer_id is not None else None,
        trust_remote_code=True,
    )
    model = jl.from_hf(hf_model, tokenizer)
    text_config = hf_model.config.get_text_config()
    handle = LensModelHandle(
        model=model,
        hf_model=hf_model,
        tokenizer=tokenizer,
        facts=facts,
        loading_report=loading_report,
        layer_types=tuple(str(kind) for kind in text_config.layer_types),
        conv_reach=int(text_config.linear_conv_kernel_dim) - 1,
        tokenizer_source=tokenizer_source,
        load_seconds=time.perf_counter() - started,
        device_facts=device_facts(),
    )
    logger.info("loaded, %s", json.dumps(handle.as_payload(), default=str))
    return handle


def window_prompt(
    texts: Sequence[str],
    model: Any,  # noqa: ANN401 - the jlens HFLensModel, duck-typed
    *,
    max_seq_len: int,
) -> tuple[str, torch.Tensor]:
    """Return the first text that fills the window, as ids; refuse a corpus with none."""
    for text in texts:
        input_ids = model.encode(text, max_length=max_seq_len)
        if int(input_ids.shape[1]) == max_seq_len:
            return text, input_ids
    raise GateFailureError(
        f"no prompt reaches max_seq_len={max_seq_len} tokens; the gates would run short"
    )


@dataclass
class GateRun:
    """Everything one invocation produces, written after every gate so a crash leaves evidence."""

    report: dict[str, Any]
    out_path: Path
    failed: list[str] = field(default_factory=list)

    def record(self, gate: str, payload: Mapping[str, object]) -> None:
        """Store one gate's block and flush the report."""
        self.report["gates"][gate] = dict(payload)
        if not payload.get("passed", False):
            self.failed.append(gate)
        self.report["failed_gates"] = list(self.failed)
        self.report["passed"] = not self.failed
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.report, indent=2, default=str) + "\n")


def gate_prompts(corpus: LensCorpus | None, *, fallback_tokens: int) -> list[str]:
    """Return the texts the gates run on: the whole corpus, or the built-in prose sized to the window."""
    if corpus is not None:
        return [*corpus.fit_prompts, *corpus.eval_prompts, *corpus.anchor_prompts]
    repeats = math.ceil(fallback_tokens / 100) + 1
    return [FALLBACK_PROMPT * repeats]


def run_token_gate(run: GateRun, texts: Sequence[str]) -> None:
    """Gate (d): load the wave's tokenizers and compare them on every text."""
    specs = [REFERENCE_TOKENIZER, *IDENTICAL_TOKENIZERS, *DIFFERENT_TOKENIZERS]
    encoders, identities = load_tokenizer_encoders(specs)
    run.record(
        GATE_TOKENS,
        token_identity(
            texts,
            encoders,
            identities=identities,
            reference=REFERENCE_TOKENIZER[0],
            expected_identical=[spec[0] for spec in IDENTICAL_TOKENIZERS],
            expected_different=[spec[0] for spec in DIFFERENT_TOKENIZERS],
        ).as_payload(),
    )


def run_gpu_gates(  # noqa: PLR0913 - the run, its args, the loaded model and the prompts
    run: GateRun,
    args: argparse.Namespace,
    *,
    jl: ModuleType,
    loaded: LensModelHandle,
    texts: Sequence[str],
    max_seq_len: int,
) -> None:
    """Gates (a), (b), (c), (e) on the loaded checkpoint, in that order, each recorded as it lands."""
    gates = cast("list[str]", args.gates)
    skip_first = fit_skip_first(jl)
    run.report["skip_first"] = skip_first
    target_layer = int(loaded.model.n_layers) - 1
    prompt, input_ids = window_prompt(texts, loaded.model, max_seq_len=max_seq_len)
    modeling_module = importlib.import_module(QWEN3_5_MODELING_MODULE)
    kernels_bound = bound_deltanet_kernels()
    run.report["deltanet_kernels_bound"] = kernels_bound
    run.report[DELTANET_KERNEL_FIELD] = prefill_deltanet_kernels(kernels_bound)

    if GATE_AUTOGRAD in gates:
        run.record(
            GATE_AUTOGRAD,
            probe_recurrence_autograd(
                loaded.blocks,
                loaded.forward,
                input_ids,
                layer_types=loaded.layer_types,
                recurrence_type=modeling_module.Qwen3_5GatedDeltaNet,
                gap=cast("int", args.position_gap),
                conv_reach=loaded.conv_reach,
            ).as_payload(),
        )
    if GATE_KERNEL in gates:
        run.record(
            GATE_KERNEL,
            audit_chunk_kernel_backward(
                loaded.blocks,
                loaded.forward,
                input_ids,
                modeling_module=modeling_module,
                layer_types=loaded.layer_types,
                estimator=RowEstimator(
                    source_layers=tuple(range(target_layer)),
                    target_layer=target_layer,
                    dims=tuple(range(cast("int", args.audit_dims))),
                    skip_first=skip_first,
                ),
                tolerance=KERNEL_AUDIT_TOLERANCE_BF16,
            ).as_payload(),
        )
    candidates = cast("list[int]", args.dim_batch_candidates)
    if GATE_SWEEP in gates:
        run.record(
            GATE_SWEEP,
            sweep_dim_batch(
                jl,
                loaded.model,
                prompt,
                candidates=candidates,
                target_layer=target_layer,
                max_seq_len=max_seq_len,
                total_bytes=int(torch.cuda.get_device_properties(0).total_memory),
                headroom_fraction=HEADROOM_FRACTION,
            ).as_payload(),
        )
    if GATE_RESUME in gates:
        n_resume = cast("int", args.resume_prompts)
        resume_texts = list(texts[:n_resume]) if len(texts) >= n_resume else [prompt] * n_resume
        sweep_block = cast("dict[str, Any]", run.report["gates"].get(GATE_SWEEP, {}))
        chosen = cast("int | None", sweep_block.get("choice", {}).get("chosen"))
        run.record(
            GATE_RESUME,
            resume_equality(
                jl,
                loaded.model,
                resume_texts,
                JacobianConfig(
                    model_id=loaded.facts.label,
                    dim_batch=chosen if chosen is not None else min(candidates),
                    max_seq_len=cast("int", args.resume_seq_len),
                ),
                work_dir=cast("Path", args.work_dir),
                tolerance=RESUME_TOLERANCE,
            ).as_payload(),
        )


def run_gates(args: argparse.Namespace) -> int:
    """Run the selected gates in order, write the report, and return the process exit code."""
    jl = _require_jlens()
    gates = cast("list[str]", args.gates)
    corpus = None if args.corpus_dir is None else load_lens_corpus(cast("Path", args.corpus_dir))
    cooperation_fit = None
    if args.fit_stimuli is not None:
        tokenizer_source = cast("str | None", args.tokenizer) or cast("str", args.model_id)
        tokenizer_revision = (
            cast("str | None", args.tokenizer_revision)
            if args.tokenizer is not None
            else cast("str | None", args.revision)
        )
        fit_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=tokenizer_revision,
            trust_remote_code=True,
        )
        cooperation_fit = prepare_fit_prompts(
            cast("Path", args.fit_stimuli),
            fit_tokenizer,
            convention=cast("str", args.stimulus_render),
            enable_thinking=not cast("bool", args.no_thinking),
            max_seq_len_ceiling=cast("int", args.max_seq_len),
        )
        max_seq_len = cooperation_fit.max_seq_len
        texts = list(cooperation_fit.prompts)
        cooperation_binding = cooperation_fit.binding_payload(
            convention=cast("str", args.stimulus_render),
            enable_thinking=not cast("bool", args.no_thinking),
            tokenizer_content_identity=tokenizer_content_sha256(fit_tokenizer),
        )
    else:
        max_seq_len = (
            cast("int", args.max_seq_len)
            if corpus is None
            else corpus.plan(ROLE_FIT, ceiling=cast("int", args.max_seq_len)).max_seq_len
        )
        texts = gate_prompts(corpus, fallback_tokens=max_seq_len)
        cooperation_binding = None
    run = GateRun(
        report={
            "invocation": " ".join(sys.argv),
            "jlens": jlens_provenance(jl),
            "corpus_dir": None if args.corpus_dir is None else str(args.corpus_dir),
            "corpus_digests": None if corpus is None else corpus.sidecar["digests"],
            "cooperation_fit_corpus": cooperation_binding,
            "max_seq_len": max_seq_len,
            "gates_requested": gates,
            "gates": {},
            "passed": True,
        },
        out_path=cast("Path", args.out),
    )
    if GATE_TOKENS in gates:
        run_token_gate(run, texts)
    if any(gate != GATE_TOKENS for gate in gates):
        loaded = load_lens_model(
            full_weights_source(cast("str", args.model_id), cast("str | None", args.revision)),
            jl=jl,
            tokenizer_id=cast("str | None", args.tokenizer),
            tokenizer_revision=cast("str | None", args.tokenizer_revision),
        )
        run.report["model"] = loaded.as_payload()
        run_gpu_gates(run, args, jl=jl, loaded=loaded, texts=texts, max_seq_len=max_seq_len)
    for gate, block in run.report["gates"].items():
        logger.info(
            "GATE %s %s %s", gate, "PASS" if block["passed"] else "FAIL", block.get("failures")
        )
    logger.info("report written to %s; passed=%s", run.out_path, run.report["passed"])
    return 0 if run.report["passed"] else 1


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-0.8B",
        help="a hub repo id (pair it with --revision) or a local checkpoint directory",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="the hub revision of --model-id: required for a hub id (the TMAX repos put a "
        "different step on every branch), refused for a local directory",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer of record to encode with (default: the checkpoint's own files); pair with "
        "--tokenizer-revision",
    )
    parser.add_argument("--tokenizer-revision", default=None)
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument(
        "--corpus-dir", type=Path, default=None, help="a tmax_lens_corpus directory"
    )
    corpus.add_argument(
        "--fit-stimuli",
        type=Path,
        default=None,
        help="private cooperation lens fit JSONL rendered before any model load",
    )
    parser.add_argument(
        "--stimulus-render",
        choices=("templated_here", "verbatim"),
        default="templated_here",
    )
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="the window the GPU gates run at; with a corpus, its derived fit window capped here",
    )
    parser.add_argument("--position-gap", type=int, default=DEFAULT_POSITION_GAP)
    parser.add_argument(
        "--audit-dims",
        type=int,
        default=KERNEL_AUDIT_DIMS,
        help="Jacobian rows per layer the kernel audit compares; the pure-torch reference's memory "
        "grows with it, see KERNEL_AUDIT_DIMS",
    )
    parser.add_argument(
        "--dim-batch-candidates", type=int, nargs="+", default=list(DIM_BATCH_CANDIDATES)
    )
    parser.add_argument("--resume-prompts", type=int, default=10)
    parser.add_argument(
        "--resume-seq-len",
        type=int,
        default=64,
        help="the resume gate's window; short, since resume semantics do not depend on length",
    )
    parser.add_argument("--gates", nargs="+", choices=ALL_GATES, default=list(ALL_GATES))
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--out", type=Path, required=True, help="the JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gates and exit non-zero on any refusal."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("jlens").setLevel(logging.INFO)
    return run_gates(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
