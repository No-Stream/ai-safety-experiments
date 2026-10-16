"""Replay persisted assistant tokens before adopting frozen-base quantization.

Defaults refuse mean KL > 0.01 nats/token, mean absolute target-logprob drift > 0.02 nats,
or any target-logprob drift > 0.2 nats. These conservative initial thresholds are design choices,
not observed Qwen3.5 tolerances. A passing forward check still requires a GPU training smoke.
Reference hidden states are streamed through temporary disk shards. The identical frozen bf16
head reconstructs both sets of logits; two resident 9B models and full-vocabulary disk caches
are avoided.
"""

import argparse
import gc
import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import torch
from huggingface_hub import snapshot_download

from reward_hacking.distill.memory import MemoryOptions, install_before_load, load_base_model
from reward_hacking.distill.quantization import QUANTIZED_MODES, drift_metrics, enforce_drift_limits
from reward_hacking.distill.train import load_training_rows

logger = logging.getLogger(__name__)


def selected_hidden_chunks(
    model: torch.nn.Module, row: dict[str, Any], chunk_tokens: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Run the backbone once and yield every selected assistant token in bounded chunks."""
    prompt = row["prompt_token_ids"]
    completion = row["completion_token_ids"]
    causal_model = cast("Any", model)
    device = next(model.parameters()).device
    inputs = torch.tensor([prompt + completion], device=device, dtype=torch.long)
    with torch.inference_mode():
        hidden = causal_model.base_model(
            input_ids=inputs, attention_mask=None, use_cache=False
        ).last_hidden_state
        hidden = hidden[:, len(prompt) - 1 : -1].squeeze(0)
        selected = torch.tensor(
            row.get("loss_mask", [1] * len(completion)), device=device, dtype=torch.bool
        )
        targets = torch.tensor(completion, device=device, dtype=torch.long)[selected]
        selected_hidden = hidden[selected]
        for start in range(0, targets.numel(), chunk_tokens):
            yield (
                selected_hidden[start : start + chunk_tokens],
                targets[start : start + chunk_tokens],
            )


def _candidate_metrics(
    candidate: torch.nn.Module, rows: list[dict[str, Any]], reference_path: Path, chunk_tokens: int
) -> dict[str, float]:
    """Aggregate every replayed token against its reference disk shard."""
    token_count, kl_sum, logp_sum, max_logp = 0.0, 0.0, 0.0, 0.0
    for row_index, row in enumerate(rows):
        for chunk_index, (hidden, targets) in enumerate(
            selected_hidden_chunks(candidate, row, chunk_tokens)
        ):
            baseline = torch.load(
                reference_path / f"{row_index}-{chunk_index}.pt",
                weights_only=True,
                map_location=next(candidate.parameters()).device,
            )
            if not torch.equal(targets, baseline["targets"]):
                raise ValueError("reference and candidate target tokens differ")
            with torch.inference_mode():
                head = cast("Any", candidate).get_output_embeddings()
                metrics = drift_metrics(head(baseline["hidden"]), head(hidden), targets)
            token_count += metrics["tokens"]
            kl_sum += metrics["mean_kl"] * metrics["tokens"]
            logp_sum += metrics["mean_abs_logp"] * metrics["tokens"]
            max_logp = max(max_logp, metrics["max_abs_logp"])
    return {
        "tokens": token_count,
        "mean_kl": kl_sum / token_count,
        "mean_abs_logp": logp_sum / token_count,
        "max_abs_logp": max_logp,
    }


def _head_hash(model: torch.nn.Module) -> str:
    """Verify the output head is identical while copying only bounded chunks to CPU."""
    head = cast("Any", model).get_output_embeddings()
    digest = hashlib.sha256()
    for parameter in head.parameters():
        for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
            digest.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# Only the files from_pretrained reads; repo metadata such as README or LICENSE is never cached locally.
MODEL_LOAD_FILE_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "tokenizer*", "*.model"]


def _resolve_pinned_snapshot(model_id: str, revision: str) -> Path:
    """Resolve an already-cached Hub snapshot and verify its immutable commit directory."""
    snapshot_path = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=list(MODEL_LOAD_FILE_PATTERNS),
        )
    )
    _check_revision(snapshot_path, revision)
    return snapshot_path


def _check_revision(snapshot_path: Path, revision: str) -> None:
    """Require the resolved model snapshot directory to match the pinned commit."""
    if not snapshot_path.is_dir() or snapshot_path.name != revision:
        raise ValueError("resolved snapshot directory differs from pinned commit")


def _admission_result(metrics: dict[str, float], thresholds: dict[str, float]) -> dict[str, Any]:
    """Record an expected drift rejection while allowing other requested modes to complete."""
    try:
        enforce_drift_limits(metrics, **thresholds)
    except ValueError as error:
        return {**metrics, "accepted": False, "rejection": str(error)}
    return {**metrics, "accepted": True}


def replay_drift(args: argparse.Namespace) -> dict[str, Any]:
    """Compare modes on the same first N persisted rows, with token-weighted aggregates."""
    if not torch.cuda.is_available():
        raise RuntimeError("quantization drift replay requires CUDA")
    if args.rows < 1 or args.chunk_tokens < 1:
        raise ValueError("rows and chunk_tokens must be positive")
    rows = load_training_rows(args.samples, None, weight_mode="sft")
    if len(rows) < args.rows:
        raise ValueError(f"requested {args.rows} rows but samples contain only {len(rows)}")
    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise ValueError("revision must be a pinned 40-character checkpoint commit hash")
    snapshot_path = _resolve_pinned_snapshot(args.model, args.revision)
    install_before_load(MemoryOptions(deltanet_conv_kernel=args.deltanet_conv_kernel))
    selected_rows = rows[: args.rows]
    report: dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "deltanet_conv_kernel": args.deltanet_conv_kernel,
        "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        "rows": args.rows,
        "thresholds": {
            "max_mean_kl": args.max_mean_kl,
            "max_mean_abs_logp": args.max_mean_abs_logp,
            "max_abs_logp": args.max_abs_logp,
        },
        "modes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="quantization-reference-", dir=args.output.parent
    ) as reference_directory:
        reference_path = Path(reference_directory)
        reference = load_base_model(str(snapshot_path), "none").cuda().eval()
        _check_revision(snapshot_path, args.revision)
        report["head_sha256"] = _head_hash(reference)
        hidden, targets = torch.empty(0), torch.empty(0)
        for row_index, row in enumerate(selected_rows):
            for chunk_index, (hidden, targets) in enumerate(
                selected_hidden_chunks(reference, row, args.chunk_tokens)
            ):
                torch.save(
                    {"hidden": hidden.cpu(), "targets": targets.cpu()},
                    reference_path / f"{row_index}-{chunk_index}.pt",
                )
        del reference, hidden, targets
        gc.collect()
        torch.cuda.empty_cache()
        for mode in args.modes:
            candidate = load_base_model(str(snapshot_path), mode).eval()
            _check_revision(snapshot_path, args.revision)
            observed_head_sha256 = _head_hash(candidate)
            head_matches_reference = observed_head_sha256 == report["head_sha256"]
            if not head_matches_reference:
                logger.warning(
                    "candidate frozen bf16 output head for mode %s hashes to %s but the reference "
                    "hashes to %s",
                    mode,
                    observed_head_sha256,
                    report["head_sha256"],
                )
            aggregate = _candidate_metrics(
                candidate, selected_rows, reference_path, args.chunk_tokens
            )
            report["modes"][mode] = {
                **_admission_result(aggregate, report["thresholds"]),
                "head_sha256": observed_head_sha256,
                "head_sha256_matches_reference": head_matches_reference,
            }
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            del candidate
            gc.collect()
            torch.cuda.empty_cache()
    rejected = [mode for mode, metrics in report["modes"].items() if not metrics["accepted"]]
    if rejected:
        raise ValueError(
            f"quantization drift refused modes: {', '.join(rejected)}; see {args.output}"
        )
    return report


def main() -> None:
    """Run the real-row forward drift admission check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pinned checkpoint revision shared by reference and candidates",
    )
    parser.add_argument("--deltanet-conv-kernel", choices=("default", "fla"), default="fla")
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument(
        "--modes", nargs="+", choices=QUANTIZED_MODES, default=list(QUANTIZED_MODES)
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mean-kl", type=float, default=0.01)
    parser.add_argument("--max-mean-abs-logp", type=float, default=0.02)
    parser.add_argument("--max-abs-logp", type=float, default=0.2)
    logging.basicConfig(level=logging.INFO)
    report = replay_drift(parser.parse_args())
    logger.info("Quantization forward drift passed: %s", json.dumps(report["modes"]))


if __name__ == "__main__":
    main()
