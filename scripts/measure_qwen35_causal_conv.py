# ruff: noqa: INP001
"""Measure Qwen3.5 causal-convolution shim parity, speed, and peak CUDA memory.

Run one checkpoint per process so CUDA allocator history from the smoke model cannot affect the
9B result. The bridge must run before transformers imports the Qwen3.5 modeling module because its
kernel decorators resolve implementations once, at import time.

Example, after the required GPU preflight succeeds::

    scripts/tmux_run.sh task15b-conv-08b -- \
      scripts/resource-limits.sh --gpu -t 15m -- \
      uv run --frozen python scripts/measure_qwen35_causal_conv.py \
      --model Qwen/Qwen3.5-0.8B --output artifacts/task15b/0.8b.json

The forward/backward benchmark uses the production LoRA targets and gradient checkpointing. It
asks the causal-LM head for one position only, avoiding a sequence-length by vocabulary logits
tensor while retaining a real scalar loss and gradients through the complete language-model path.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, cast

import torch
import torch.nn.functional as nn_functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from games.deltanet_kernels import (
    QWEN3_5_MODELING_MODULE,
    assert_causal_conv_kernels_bound,
    bound_deltanet_kernels,
    bridge_decode_kernel,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType

    from transformers import PreTrainedModel

logger = logging.getLogger(__name__)

CONV_FUNCTIONS = ("causal_conv1d_fn", "causal_conv1d_update")
BYTES_PER_GIB = 1024**3
KernelCallable = Callable[..., object]


@dataclass(frozen=True)
class TrainingStepMeasurement:
    """One synchronized forward/backward measurement with allocator baselines and peaks."""

    wall_seconds: float
    loss: float
    baseline_allocated_gib: float
    peak_allocated_gib: float
    incremental_peak_allocated_gib: float
    baseline_reserved_gib: float
    peak_reserved_gib: float
    incremental_peak_reserved_gib: float


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    """Persist every completed point so a later long-context OOM does not erase evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _torch_reference(function: KernelCallable, *, name: str) -> KernelCallable:
    reference = getattr(function, "__wrapped__", None)
    if reference is None:
        raise RuntimeError(
            f"{QWEN3_5_MODELING_MODULE}.{name} has no __wrapped__ reference; transformers changed "
            "the dispatch wrapper, so the torch comparison cannot be constructed"
        )
    return cast("KernelCallable", reference)


@contextlib.contextmanager
def _conv_variant(modeling_module: ModuleType, variant: str) -> Generator[None]:
    """Select shim or torch reference at the layer's module-global call site."""
    # Prove the process has not silently drifted back to the fallback before deriving either arm.
    assert_causal_conv_kernels_bound()
    original = {name: getattr(modeling_module, name) for name in CONV_FUNCTIONS}
    if variant == "shim":
        replacements = original
    elif variant == "torch_reference":
        replacements = {
            name: _torch_reference(function, name=name) for name, function in original.items()
        }
    else:
        raise ValueError(f"unknown convolution variant: {variant}")
    for name, function in replacements.items():
        setattr(modeling_module, name, function)
    if variant == "shim":
        assert_causal_conv_kernels_bound()
    else:
        for name, function in replacements.items():
            if getattr(modeling_module, name) is not function:
                raise RuntimeError(f"failed to bind {name} to its torch reference")
    try:
        yield
    finally:
        for name, function in original.items():
            setattr(modeling_module, name, function)


def _tensor_agreement(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    if candidate.shape != reference.shape:
        raise RuntimeError(
            f"logit shapes differ between shim and torch reference: "
            f"{tuple(candidate.shape)} != {tuple(reference.shape)}"
        )
    difference = (candidate.float() - reference.float()).abs()
    candidate_top = candidate.argmax(dim=-1)
    reference_top = reference.argmax(dim=-1)
    return {
        "shape": list(candidate.shape),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "top1_agreement": float((candidate_top == reference_top).float().mean()),
        "shim_top1": candidate_top.tolist(),
        "torch_reference_top1": reference_top.tolist(),
    }


def _synthetic_token_ids(
    *, vocabulary_size: int, sequence_length: int, seed: int, device: torch.device
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Avoid special-token-heavy low IDs without relying on tokenizer or prompt text.
    return torch.randint(
        low=min(1000, vocabulary_size - 1),
        high=vocabulary_size,
        size=(1, sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)


@torch.no_grad()
def _model_logits(
    model: PreTrainedModel,
    modeling_module: ModuleType,
    *,
    variant: str,
    prompt_ids: torch.Tensor,
    decode_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with _conv_variant(modeling_module, variant):
        prefill = model(
            input_ids=prompt_ids,
            use_cache=True,
            logits_to_keep=0,
            return_dict=True,
        )
        if prefill.past_key_values is None:
            raise RuntimeError("Qwen3.5 returned no cache for the cached-decode parity check")
        cache = prefill.past_key_values
        decode_logits = []
        for decode_position in range(decode_ids.shape[1]):
            decode = model(
                input_ids=decode_ids[:, decode_position : decode_position + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            if decode.past_key_values is None:
                raise RuntimeError("Qwen3.5 dropped its cache during repeated decode")
            cache = decode.past_key_values
            decode_logits.append(decode.logits.detach().cpu())
    return prefill.logits.detach().cpu(), torch.cat(decode_logits, dim=1)


def _measure_model_parity(
    model: PreTrainedModel,
    modeling_module: ModuleType,
    *,
    prompt_ids: torch.Tensor,
    decode_ids: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    reference_prefill, reference_decode = _model_logits(
        model,
        modeling_module,
        variant="torch_reference",
        prompt_ids=prompt_ids,
        decode_ids=decode_ids,
    )
    shim_prefill, shim_decode = _model_logits(
        model,
        modeling_module,
        variant="shim",
        prompt_ids=prompt_ids,
        decode_ids=decode_ids,
    )
    return {
        "prompt_tokens": prompt_ids.shape[1],
        "decode_tokens": decode_ids.shape[1],
        "prefill_logits": _tensor_agreement(shim_prefill, reference_prefill),
        "cached_decode_logits": _tensor_agreement(shim_decode, reference_decode),
    }


def _configure_lora_training(
    model: PreTrainedModel,
) -> tuple[PreTrainedModel, dict[str, object]]:
    # These imports can load transformers model classes, so they must remain after the bridge.
    from peft import LoraConfig, get_peft_model  # noqa: PLC0415

    from grpo.throughput import select_lora_targets  # noqa: PLC0415

    text_config = model.config.get_text_config()
    linear_module_names = [
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    ]
    selection = select_lora_targets(linear_module_names, list(text_config.layer_types))
    targets = cast("list[str]", selection["target_modules"])
    lora_model: Any = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    lora_model.config.use_cache = False
    lora_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora_model.enable_input_require_grads()
    lora_model.train()
    return cast("PreTrainedModel", lora_model), selection


def _one_training_step(
    model: PreTrainedModel,
    modeling_module: ModuleType,
    *,
    variant: str,
    input_ids: torch.Tensor,
) -> TrainingStepMeasurement:
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    with _conv_variant(modeling_module, variant):
        output = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
        labels = input_ids[:, -1]
        loss = nn_functional.cross_entropy(output.logits[:, -1, :].float(), labels)
        loss.backward()
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    loss_value = float(loss.detach())
    del loss, output
    model.zero_grad(set_to_none=True)
    return TrainingStepMeasurement(
        wall_seconds=wall_seconds,
        loss=loss_value,
        baseline_allocated_gib=baseline_allocated / BYTES_PER_GIB,
        peak_allocated_gib=peak_allocated / BYTES_PER_GIB,
        incremental_peak_allocated_gib=(peak_allocated - baseline_allocated) / BYTES_PER_GIB,
        baseline_reserved_gib=baseline_reserved / BYTES_PER_GIB,
        peak_reserved_gib=peak_reserved / BYTES_PER_GIB,
        incremental_peak_reserved_gib=(peak_reserved - baseline_reserved) / BYTES_PER_GIB,
    )


def _benchmark_variant(  # noqa: PLR0913 - explicit measurement inputs belong in the artifact
    model: PreTrainedModel,
    modeling_module: ModuleType,
    *,
    variant: str,
    input_ids: torch.Tensor,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    for _ in range(warmup_iterations):
        _one_training_step(model, modeling_module, variant=variant, input_ids=input_ids)
    measurements = [
        _one_training_step(model, modeling_module, variant=variant, input_ids=input_ids)
        for _ in range(measured_iterations)
    ]
    wall_seconds = [measurement.wall_seconds for measurement in measurements]
    return {
        "measurements": [asdict(measurement) for measurement in measurements],
        "wall_seconds": wall_seconds,
        "median_wall_seconds": statistics.median(wall_seconds),
        "losses": [measurement.loss for measurement in measurements],
        "max_peak_allocated_gib": max(
            measurement.peak_allocated_gib for measurement in measurements
        ),
        "max_incremental_peak_allocated_gib": max(
            measurement.incremental_peak_allocated_gib for measurement in measurements
        ),
        "max_peak_reserved_gib": max(measurement.peak_reserved_gib for measurement in measurements),
        "max_incremental_peak_reserved_gib": max(
            measurement.incremental_peak_reserved_gib for measurement in measurements
        ),
    }


def _benchmark_record(
    *,
    model_id: str,
    sequence_length: int,
    dtype: torch.dtype,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Create the durable record before either variant spends its GPU time."""
    return {
        "model": model_id,
        "sequence_length": sequence_length,
        "micro_batch_size": 1,
        "dtype": str(dtype),
        "gradient_checkpointing": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "loss_path": (
            "single-position causal-LM cross-entropy with logits_to_keep=1; full-sequence "
            "forward/backward surrogate, not production Liger GRPO loss"
        ),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "variants": {},
        "wall_speedup_torch_over_shim": None,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:  # noqa: UP045
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--sequence-length",
        type=int,
        action="append",
        dest="sequence_lengths",
        help="repeat for each forward/backward length; defaults to 8192 and 16384",
    )
    parser.add_argument("--parity-tokens", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measured-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:  # noqa: UP045
    """Run one checkpoint measurement and persist each completed result."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; run this measurement only after gpu_preflight succeeds"
        )
    torch.manual_seed(args.seed)
    if args.parity_tokens < 1 or args.decode_tokens < 1:
        raise ValueError("--parity-tokens and --decode-tokens must be positive")
    if args.warmup_iterations < 0 or args.measured_iterations < 1:
        raise ValueError("warmup must be nonnegative and measured iterations must be positive")
    sequence_lengths = args.sequence_lengths or [8192, 16384]
    if any(length < 1 for length in sequence_lengths):
        raise ValueError("all sequence lengths must be positive")

    # This must precede every transformers import that can load modeling_qwen3_5.
    bridge_report = bridge_decode_kernel()
    modeling_module = importlib.import_module(QWEN3_5_MODELING_MODULE)
    assert_causal_conv_kernels_bound()
    bindings = bound_deltanet_kernels()
    fallback_bindings = {
        name: bindings[name]
        for name in CONV_FUNCTIONS
        if bindings[name].startswith("transformers.models.qwen3_5.modeling_qwen3_5")
    }
    if fallback_bindings:
        raise RuntimeError(
            f"causal-convolution shim was expected, but transformers bound its torch fallback: "
            f"{fallback_bindings}"
        )

    from transformers import AutoModelForCausalLM  # noqa: PLC0415 - bridge must run first

    device = torch.device("cuda")
    loaded_model: Any = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model = cast("PreTrainedModel", loaded_model.to(device))
    payload: dict[str, Any] = {
        "model": args.model,
        "seed": args.seed,
        "bridge_report": bridge_report,
        "bound_deltanet_kernels": bindings,
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_gib": torch.cuda.get_device_properties(device).total_memory
            / BYTES_PER_GIB,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parity": None,
        "forward_backward": [],
    }
    _write_payload(args.output, payload)

    prompt_ids = _synthetic_token_ids(
        vocabulary_size=model.config.vocab_size,
        sequence_length=args.parity_tokens,
        seed=args.seed,
        device=device,
    )
    decode_ids = _synthetic_token_ids(
        vocabulary_size=model.config.vocab_size,
        sequence_length=args.decode_tokens,
        seed=args.seed + 1,
        device=device,
    )
    payload["parity"] = _measure_model_parity(
        model,
        modeling_module,
        prompt_ids=prompt_ids,
        decode_ids=decode_ids,
    )
    _write_payload(args.output, payload)
    logger.info("model parity complete: %s", payload["parity"])

    model, lora_selection = _configure_lora_training(model)
    payload["lora_selection"] = lora_selection
    payload["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    payload["total_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    _write_payload(args.output, payload)
    for sequence_length in sequence_lengths:
        input_ids = _synthetic_token_ids(
            vocabulary_size=model.config.vocab_size,
            sequence_length=sequence_length,
            seed=args.seed,
            device=device,
        )
        measurement = _benchmark_record(
            model_id=args.model,
            sequence_length=sequence_length,
            dtype=next(model.parameters()).dtype,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
        )
        cast("list[dict[str, Any]]", payload["forward_backward"]).append(measurement)
        _write_payload(args.output, payload)
        variants = cast("dict[str, dict[str, Any]]", measurement["variants"])
        for variant in ("torch_reference", "shim"):
            variants[variant] = _benchmark_variant(
                model,
                modeling_module,
                variant=variant,
                input_ids=input_ids,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            )
            _write_payload(args.output, payload)
        reference_seconds = cast("float", variants["torch_reference"]["median_wall_seconds"])
        shim_seconds = cast("float", variants["shim"]["median_wall_seconds"])
        measurement["wall_speedup_torch_over_shim"] = reference_seconds / shim_seconds
        _write_payload(args.output, payload)
        logger.info("forward/backward measurement complete: %s", measurement)


if __name__ == "__main__":
    main()
