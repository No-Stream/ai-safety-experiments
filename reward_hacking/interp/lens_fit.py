r"""Fit and validate a Jacobian lens for a Hugging Face causal language model.

The command accepts an ordered JSONL corpus with a ``text`` field, loads the selected model through
the repository's capture loader, and writes a resumable ``jlens`` fit with an identity-bound
checkpoint. ``--halves`` fits odd and even corpus rows independently and prompt-weights their merge;
the two component lenses and their disagreement remain beside the merged lens.

The fit always records three reads on the exact loaded weights: reconstruction against the model's
final logits beside the raw logit-lens baseline, token recovery against a zero-J sabotage, and an
architecture gate. DeltaNet autograd/kernel checks apply only when the text decoder has linear
attention blocks. The reconstruction read is diagnostic because the lens is expected to be coherent
only in an intermediate layer band and can be poor under corpus distribution shift.

Set ``JLENS_CORPUS`` to a JSONL file with a ``text`` field per row and run with
``PYTHONPATH=external/jacobian-lens``. These commands fit Gemma 3 12B with and without the local
PEFT adapter:

    PYTHONPATH=external/jacobian-lens python -m reward_hacking.interp.lens_fit \\
        --model google/gemma-3-12b-it --corpus "$JLENS_CORPUS" \\
        --n-prompts 30 --source-layers 8:44:2 --target-layer 44 \\
        --dim-batch 4 --max-seq-len 128 --out-dir artifacts/jlens/gemma-3-12b-it/base

    PYTHONPATH=external/jacobian-lens python -m reward_hacking.interp.lens_fit \\
        --model google/gemma-3-12b-it --corpus "$JLENS_CORPUS" \\
        --n-prompts 30 --source-layers 8:44:2 --target-layer 44 \\
        --dim-batch 4 --max-seq-len 128 \\
        --out-dir artifacts/jlens/gemma-3-12b-it/kl0.01-det3-seed1-mbpp_probe \\
        --adapter artifacts/atlas/adapters/kl0.01-det3-seed1-mbpp_probe

Add ``--halves`` to retain odd/even fits and their prompt-weighted merge.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch

from games.provenance import git_sha, git_tree_dirty
from games.tokenizer_identity import tokenizer_content_sha256
from reward_hacking.interp.jacobian import (
    JLENS_COMMIT,
    JacobianConfig,
    _require_jlens,  # pyright: ignore[reportPrivateUsage]  # shared pinned-jlens loader
    _resolve_fit_layers,  # pyright: ignore[reportPrivateUsage]  # shared layer-index validation
    digest_strings,
    fit_skip_first,
    verify_lens_roundtrip,
)
from reward_hacking.interp.lens_fit_inputs import (
    FitCorpus,
    FitOptions,
    build_parser,
    options_from_args,
    parse_layer_range,
    read_fit_corpus,
)
from reward_hacking.interp.lens_fit_runtime import (
    fit_save_and_read_lens,
    fit_with_resume_identity,
    jlens_provenance,
    load_fit_model,
    reconstruction_read_payload,
    run_hybrid_architecture_gates,
    save_and_read_lens,
    validate_unembedding,
)
from reward_hacking.interp.lens_fit_validation import (
    RECOVERY_POSITIONS_PER_PROMPT,
    FitGateError,
    FitMemoryError,
    recovery_gate_from_logits,
    recovery_gate_report,
)
from reward_hacking.interp.lens_fit_validation import (
    half_disagreement as _half_disagreement,
)
from reward_hacking.interp.lens_fit_validation import (
    preflight_dim_batch as _preflight_dim_batch,
)
from reward_hacking.interp.lens_fit_validation import (
    recovery_read as _recovery_read,
)
from reward_hacking.interp.lens_geometry import sha256_of_file

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "FitCorpus",
    "FitGateError",
    "FitMemoryError",
    "FitOptions",
    "_preflight_dim_batch",
    "_recovery_read",
    "build_parser",
    "main",
    "options_from_args",
    "parse_layer_range",
    "read_fit_corpus",
    "recovery_gate_from_logits",
    "recovery_gate_report",
    "run_fit",
]

FIT_CHECKPOINT_FILENAME = "fit-checkpoint.pt"
FIT_IDENTITY_FILENAME = "fit-identity.json"
LENS_FILENAME = "lens.pt"
PROVENANCE_FILENAME = "lens-provenance.json"
HALF_NAMES = ("odd", "even")
MIN_PROMPTS_FOR_HALVES = 2


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically replace a JSON report so incomplete writes never look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _identity(  # noqa: PLR0913 - every identity field is an explicit restart-safety input
    options: FitOptions,
    corpus: FitCorpus,
    *,
    source_layers: Sequence[int],
    target_layer: int,
    skip_first: int,
    weights_sha256: str,
    tokenizer_sha256: str,
    model_revision: str | None,
    repo_sha: str,
    implementation_sha256: Mapping[str, str],
    jlens_facts: Mapping[str, object],
) -> dict[str, object]:
    """Build the checkpoint identity from data, loaded weights and resolved fit choices."""
    return {
        "model": options.model,
        "weights_fingerprint": weights_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "model_revision": model_revision,
        "repo_sha": repo_sha,
        "implementation_sha256": dict(implementation_sha256),
        "adapter": None if options.adapter is None else options.adapter.name,
        "corpus_sha256": corpus.source_sha256,
        "selected_text_sha256": corpus.selected_text_sha256,
        "n_prompts_requested": len(corpus.prompts),
        "source_layers": list(source_layers),
        "target_layer": target_layer,
        "dim_batch": options.dim_batch,
        "max_seq_len": options.max_seq_len,
        "skip_first": skip_first,
        "checkpoint_every": options.checkpoint_every,
        "halves": options.halves,
        "jlens": dict(jlens_facts),
    }


def _implementation_digests() -> dict[str, str]:
    """Hash source files that decide the fit and its gates, including dirty-tree edits."""
    repo_root = Path(__file__).resolve().parents[2]
    relative_paths = (
        "reward_hacking/interp/lens_fit.py",
        "reward_hacking/interp/lens_fit_inputs.py",
        "reward_hacking/interp/lens_fit_validation.py",
        "reward_hacking/interp/lens_fit_runtime.py",
        "reward_hacking/interp/jacobian.py",
        "reward_hacking/interp/directions.py",
        "reward_hacking/interp/lens_geometry.py",
        "reward_hacking/interp/lens_deltanet_gates.py",
        "games/deltanet_kernels.py",
        "games/lora.py",
        "games/tokenizer_identity.py",
    )
    return {
        relative_path: sha256_of_file(repo_root / relative_path) for relative_path in relative_paths
    }


def _fit_config(  # noqa: PLR0913 - explicit settings mirror jlens.fit configuration
    options: FitOptions,
    *,
    source_layers: Sequence[int],
    target_layer: int,
    prompts: Sequence[str],
    checkpoint_path: Path,
    skip_first: int,
) -> JacobianConfig:
    """Translate generic CLI choices into the pinned jlens configuration."""
    return JacobianConfig(
        model_id=options.model,
        max_fit_prompts=len(prompts),
        dim_batch=options.dim_batch,
        max_seq_len=options.max_seq_len,
        target_layer=target_layer,
        source_layers=tuple(source_layers),
        skip_first=skip_first,
        checkpoint_path=checkpoint_path,
        checkpoint_every=options.checkpoint_every,
        resume=True,
    )


def _assert_completed_artifact(
    out_dir: Path, identity: Mapping[str, object]
) -> dict[str, object] | None:
    """Skip only a completed fit with matching identity and intact output artifacts."""
    provenance_path = out_dir / PROVENANCE_FILENAME
    if not provenance_path.exists():
        return None
    payload = json.loads(provenance_path.read_text())
    if not isinstance(payload, dict) or payload.get("identity") != dict(identity):
        raise ValueError(
            f"completed lens at {provenance_path} has a different identity; choose another --out-dir"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"completed lens provenance at {provenance_path} has no artifact digests")
    for artifact_name, expected_digest in artifacts.items():
        artifact_path = out_dir / str(artifact_name)
        if not artifact_path.is_file() or sha256_of_file(artifact_path) != expected_digest:
            raise ValueError(
                f"completed lens artifact {artifact_path.name} is missing or has changed"
            )
    logger.info("completed lens identity and artifact digests match; skipping the fit")
    return payload


def run_fit(options: FitOptions) -> dict[str, object]:  # noqa: C901, PLR0912, PLR0915 - linear fit lifecycle keeps gates before fit and provenance last
    """Run one complete generic fit and write provenance only after all validation reads pass."""
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise FitMemoryError(
            "lens fitting requires CUDA so free VRAM and the requested dim_batch can be checked"
        )
    startup_free, startup_total = torch.cuda.mem_get_info(0)
    logger.info(
        "device=%s free_vram_startup=%.2f GiB total_vram=%.2f GiB",
        torch.cuda.get_device_name(0),
        startup_free / 2**30,
        startup_total / 2**30,
    )
    if options.halves and options.n_prompts < MIN_PROMPTS_FOR_HALVES:
        raise ValueError("--halves needs at least two prompts")
    corpus = read_fit_corpus(options.corpus, n_prompts=options.n_prompts)
    source_layers = list(options.source_layers)
    options.out_dir.mkdir(parents=True, exist_ok=True)

    jl = _require_jlens()
    jlens_payload = jlens_provenance(jl)
    jlens_commit = jlens_payload["commit"]
    if not isinstance(jlens_commit, str) or not jlens_commit.startswith(JLENS_COMMIT):
        raise RuntimeError(
            f"generic lens fits require the pinned jlens Git checkout at {JLENS_COMMIT}; "
            f"the loaded package reports commit {jlens_commit!r}"
        )
    if jlens_payload["dirty"] is not False:
        raise RuntimeError("generic lens fits require a clean pinned jlens checkout")
    jlens_facts = {"commit": jlens_commit, "pinned_commit": JLENS_COMMIT, "dirty": False}
    handle = load_fit_model(
        options.model,
        jl=jl,
        adapter_path=options.adapter,
        dtype=torch.bfloat16,
    )
    fit_model = cast("Any", handle.model)
    weights_sha256 = handle.weights_fingerprint
    tokenizer_sha256 = tokenizer_content_sha256(handle.tokenizer)
    hf_model = cast("Any", handle.hf_model)
    model_revision = vars(hf_model.config).get("_commit_hash")
    if not isinstance(model_revision, str):
        model_revision = None
    repo_sha = git_sha()
    implementation_sha256 = _implementation_digests()
    skip_first = fit_skip_first(jl)
    source_layers, target_layer = _resolve_fit_layers(
        source_layers, options.target_layer, int(fit_model.n_layers)
    )
    identity = _identity(
        options,
        corpus,
        source_layers=source_layers,
        target_layer=target_layer,
        skip_first=skip_first,
        weights_sha256=weights_sha256,
        tokenizer_sha256=tokenizer_sha256,
        model_revision=model_revision,
        repo_sha=repo_sha,
        implementation_sha256=implementation_sha256,
        jlens_facts=jlens_facts,
    )
    completed = _assert_completed_artifact(options.out_dir, identity)
    if completed is not None:
        return completed

    memory_preflight = _preflight_dim_batch(
        jl,
        fit_model,
        corpus.prompts,
        source_layers=source_layers,
        target_layer=target_layer,
        dim_batch=options.dim_batch,
        max_seq_len=options.max_seq_len,
        skip_first=skip_first,
    )
    unembedding = validate_unembedding(hf_model, fit_model)
    if not bool(unembedding["passed"]):
        raise FitGateError("the model's jlens unembedding validation failed")
    hybrid_gates = run_hybrid_architecture_gates(
        handle,
        jl,
        corpus.prompts[0],
        target_layer=target_layer,
        max_seq_len=options.max_seq_len,
        dim_batch=options.dim_batch,
        source_layers=source_layers,
    )
    if not bool(hybrid_gates["passed"]):
        raise FitGateError("the model's architecture-specific autograd/kernel gate failed")

    fit_started = time.monotonic()
    split_disagreement: dict[str, float] | None = None
    half_records: dict[str, dict[str, object]] = {}
    if options.halves:
        half_lenses: dict[str, Any] = {}
        for half_name, half_prompts in zip(
            HALF_NAMES, (corpus.prompts[::2], corpus.prompts[1::2]), strict=True
        ):
            checkpoint_path = options.out_dir / f"fit-checkpoint-{half_name}.pt"
            identity_path = options.out_dir / f"fit-identity-{half_name}.json"
            half_identity = {
                **identity,
                "fit_split": half_name,
                "split_text_sha256": digest_strings(half_prompts),
                "split_n_prompts": len(half_prompts),
            }
            half_config = _fit_config(
                options,
                source_layers=source_layers,
                target_layer=target_layer,
                prompts=half_prompts,
                checkpoint_path=checkpoint_path,
                skip_first=skip_first,
            )
            half_lens = cast(
                "Any",
                fit_with_resume_identity(
                    half_config,
                    fit_model,
                    half_prompts,
                    jl,
                    identity=half_identity,
                    identity_path=identity_path,
                ),
            )
            half_lenses[half_name] = half_lens
            half_path = options.out_dir / f"lens-half-{half_name}.pt"
            half_lens.save(str(half_path))
            verify_lens_roundtrip(jl, half_lens, half_path)
            half_records[half_name] = {
                "prompts_requested": len(half_prompts),
                "prompts_fitted": int(half_lens.n_prompts),
                "checkpoint": checkpoint_path.name,
                "lens": half_path.name,
            }
        split_disagreement = _half_disagreement(half_lenses["odd"], half_lenses["even"])
        lens = cast("Any", jl).JacobianLens.merge([half_lenses["odd"], half_lenses["even"]])
        fit_result = save_and_read_lens(
            lens,
            fit_model,
            jl,
            lens_path=options.out_dir / LENS_FILENAME,
            eval_prompts=corpus.prompts,
            max_seq_len=options.max_seq_len,
            max_positions=RECOVERY_POSITIONS_PER_PROMPT,
            skip_first=skip_first,
            target_layer=target_layer,
        )
        fit_seconds = time.monotonic() - fit_started
        for half_name in HALF_NAMES:
            half_records[half_name]["lens_sha256"] = sha256_of_file(
                options.out_dir / str(half_records[half_name]["lens"])
            )
    else:
        checkpoint_path = options.out_dir / FIT_CHECKPOINT_FILENAME
        identity_path = options.out_dir / FIT_IDENTITY_FILENAME
        fit_config = _fit_config(
            options,
            source_layers=source_layers,
            target_layer=target_layer,
            prompts=corpus.prompts,
            checkpoint_path=checkpoint_path,
            skip_first=skip_first,
        )
        fit_result = fit_save_and_read_lens(
            fit_config,
            fit_model,
            corpus.prompts,
            jl,
            identity=identity,
            identity_path=identity_path,
            lens_path=options.out_dir / LENS_FILENAME,
            eval_prompts=corpus.prompts,
            eval_max_seq_len=options.max_seq_len,
            eval_positions=RECOVERY_POSITIONS_PER_PROMPT,
        )
        lens = cast("Any", fit_result.lens)
        fit_seconds = (
            fit_result.fit_seconds + fit_result.persistence_seconds + fit_result.read_seconds
        )

    recovery = _recovery_read(
        jl,
        lens,
        fit_model,
        corpus.prompts,
        source_layers=source_layers,
        max_seq_len=options.max_seq_len,
        skip_first=skip_first,
    )
    reconstruction = reconstruction_read_payload(fit_result.quality)
    main_lens_path = options.out_dir / LENS_FILENAME
    artifacts = {LENS_FILENAME: sha256_of_file(main_lens_path)}
    if options.halves:
        artifacts.update(
            {
                str(half_records[half_name]["lens"]): str(half_records[half_name]["lens_sha256"])
                for half_name in HALF_NAMES
            }
        )
    report: dict[str, object] = {
        "schema": 1,
        "kind": "generic-jacobian-lens",
        "identity": identity,
        "model": handle.as_payload(),
        "adapter": None if options.adapter is None else options.adapter.name,
        "corpus": {
            "sha256": corpus.source_sha256,
            "selected_text_sha256": corpus.selected_text_sha256,
            "prompts_requested": len(corpus.prompts),
            "prompts_fitted": int(lens.n_prompts),
        },
        "fit": {
            "source_layers": list(source_layers),
            "target_layer": target_layer,
            "target_read": "model final-layer logits, including when target_layer is intermediate",
            "dim_batch": options.dim_batch,
            "max_seq_len": options.max_seq_len,
            "skip_first": skip_first,
            "checkpoint_every": options.checkpoint_every,
            "halves": options.halves,
            "halves_detail": half_records if options.halves else None,
            "split_half_relative_frobenius": split_disagreement,
            "prompts_fitted": int(lens.n_prompts),
            "prompts_skipped_by_jlens": len(corpus.prompts) - int(lens.n_prompts),
        },
        "weights_fingerprint": weights_sha256,
        "adapter_fingerprint_included": options.adapter is not None,
        "corpus_distribution": "quality and recovery reads reuse the capped fit corpus",
        "jlens": jlens_facts,
        "jlens_commit": JLENS_COMMIT,
        "repo_sha": repo_sha,
        "repo_tree_dirty": git_tree_dirty(),
        "implementation_sha256": implementation_sha256,
        "wall_time_seconds": time.monotonic() - started,
        "fit_wall_time_seconds": fit_seconds,
        "memory_preflight": memory_preflight,
        "artifacts": artifacts,
        "gates": {
            "unembedding": unembedding,
            "reconstruction_vs_logit_lens": reconstruction,
            "token_recovery_vs_zero_jacobian": recovery,
            "architecture": hybrid_gates,
        },
    }
    _atomic_write_json(options.out_dir / PROVENANCE_FILENAME, report)
    logger.info(
        "saved %s: prompts_fitted=%d wall=%.1fs layers=%s target=%d",
        main_lens_path,
        int(lens.n_prompts),
        report["wall_time_seconds"],
        source_layers,
        target_layer,
    )
    return report


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    options = options_from_args(build_parser(description=__doc__).parse_args(argv))
    run_fit(options)


if __name__ == "__main__":
    main()
