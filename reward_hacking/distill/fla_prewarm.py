"""Bucket and prewarm the FLA Triton kernels used by Qwen3.5 training."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

logger = logging.getLogger(__name__)

EXPECTED_FLA_VERSION = "0.5.2"
EXPECTED_FLA_CORE_VERSION = "0.5.2"
CONV_KERNEL_NAMES = ("causal_conv1d_fwd_kernel", "causal_conv1d_bwd_kernel")
L2NORM_KERNEL_NAMES = ("l2norm_fwd_kernel", "l2norm_bwd_kernel")
CONV_NB_TOKENS = 1024
L2NORM_NB_ROWS = 65536


def bucket_nb(nb: object) -> int:
    """Round a positive FLA autotune NB key up to its next power of two."""
    if not isinstance(nb, int) or isinstance(nb, bool) or nb < 1:
        raise ValueError(f"NB must be a positive integer, got {nb!r}")
    return 1 << (nb - 1).bit_length()


class NBBucketedKernel:
    """Proxy a Triton launcher and bucket its autotune-only ``NB`` keyword."""

    def __init__(self, kernel: Any) -> None:  # noqa: ANN401
        """Store the original launcher so its call signature remains available."""
        self.kernel = kernel

    def __getitem__(self, grid: Any) -> Any:  # noqa: ANN401
        """Return a launcher that rounds ``NB`` before Triton builds its key."""
        launcher = self.kernel[grid]

        def launch(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            kwargs["NB"] = bucket_nb(kwargs["NB"])
            return launcher(*args, **kwargs)

        return launch


def _autotuner_and_length_parameter(
    kernel: Any,  # noqa: ANN401
    *,
    module_name: str,
    kernel_name: str,
) -> tuple[Any, Any]:
    autotuner = kernel if hasattr(kernel, "cache_results") else getattr(kernel, "fn", None)
    if autotuner is None or not hasattr(autotuner, "cache_results"):
        raise RuntimeError(
            f"expected {module_name}.{kernel_name} to wrap a Triton autotuner with cache_results"
        )
    jit_kernel = getattr(autotuner, "fn", None)
    parameters = getattr(jit_kernel, "params", None)
    device_caches = getattr(jit_kernel, "device_caches", None)
    if device_caches is None:
        raise RuntimeError(f"expected {module_name}.{kernel_name} to expose Triton device_caches")
    if device_caches:
        raise RuntimeError(
            f"install FLA NB bucketing before invoking {module_name}.{kernel_name}; "
            "Triton has already created a signature binder"
        )
    length_parameters = [parameter for parameter in parameters or () if parameter.name == "T"]
    if len(length_parameters) != 1:
        raise RuntimeError(
            f"expected {module_name}.{kernel_name} to have one Triton scalar T parameter"
        )
    return autotuner, length_parameters[0]


def _install_fla_nb_bucket_wrappers(
    *,
    fla_version: str,
    fla_core_version: str,
    conv_module: ModuleType,
    l2norm_module: ModuleType,
) -> None:
    _require_supported_fla_versions(fla_version, fla_core_version)

    module_kernels = (
        (conv_module, CONV_KERNEL_NAMES),
        (l2norm_module, L2NORM_KERNEL_NAMES),
    )
    for module, names in module_kernels:
        for name in names:
            kernel = getattr(module, name)
            if isinstance(kernel, NBBucketedKernel):
                continue
            _autotuner_and_length_parameter(kernel, module_name=module.__name__, kernel_name=name)
            if not callable(getattr(kernel, "__getitem__", None)):
                raise TypeError(f"expected {module.__name__}.{name} to be a Triton launcher")

    for module, names in module_kernels:
        for name in names:
            kernel = getattr(module, name)
            if isinstance(kernel, NBBucketedKernel):
                continue
            autotuner, length_parameter = _autotuner_and_length_parameter(
                kernel, module_name=module.__name__, kernel_name=name
            )
            autotuner.cache_results = True
            # Conv kernels specialize scalar T by default; NB bucketing cannot merge those variants.
            length_parameter.do_not_specialize = True
            setattr(module, name, NBBucketedKernel(kernel))


def _installed_fla_version() -> str:
    try:
        return version("flash-linear-attention")
    except PackageNotFoundError as error:
        raise RuntimeError("flash-linear-attention is required for FLA NB bucketing") from error


def _installed_fla_core_version() -> str:
    try:
        return version("fla-core")
    except PackageNotFoundError as error:
        raise RuntimeError("fla-core is required for FLA NB bucketing") from error


def _require_supported_fla_versions(fla_version: str, fla_core_version: str) -> None:
    if fla_version != EXPECTED_FLA_VERSION or fla_core_version != EXPECTED_FLA_CORE_VERSION:
        raise RuntimeError(
            "NB bucketing only supports flash-linear-attention "
            f"{EXPECTED_FLA_VERSION} with fla-core {EXPECTED_FLA_CORE_VERSION}; found "
            f"flash-linear-attention {fla_version} and fla-core {fla_core_version}"
        )


def install_fla_nb_bucketing() -> None:
    """Install the version-checked NB proxy and persistent Triton autotune caching."""
    fla_version = _installed_fla_version()
    fla_core_version = _installed_fla_core_version()
    _require_supported_fla_versions(fla_version, fla_core_version)
    conv_ops = importlib.import_module("fla.modules.conv.triton.ops")
    l2norm = importlib.import_module("fla.modules.l2norm")
    _install_fla_nb_bucket_wrappers(
        fla_version=fla_version,
        fla_core_version=fla_core_version,
        conv_module=conv_ops,
        l2norm_module=l2norm,
    )
    logger.info(
        "bucketed FLA NB keys and enabled Triton autotune disk caching: fla=%s fla_core=%s cache=%s",
        fla_version,
        fla_core_version,
        os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache"),
    )


@dataclass(frozen=True)
class FlaKernelShape:
    """Kernel dimensions needed to build representative FLA inputs."""

    conv_dimension: int
    conv_width: int
    conv_activation: str
    norm_head_dimension: int
    norm_heads: int


@dataclass(frozen=True)
class FlaNBPrewarmPlan:
    """NB keys and minimum sequence lengths that cover the requested shapes."""

    kernel_shape: FlaKernelShape
    conv_buckets: tuple[int, ...]
    l2norm_buckets: tuple[int, ...]
    conv_sequence_tokens_by_bucket: dict[int, int]
    l2norm_sequence_tokens_by_bucket: dict[int, int]

    def estimated_peak_tensor_bytes(self, *, element_size: int) -> int:
        """Estimate four live input/output tensors for the larger prewarm kernel."""
        if element_size < 1:
            raise ValueError(f"element_size must be positive, got {element_size}")
        conv_elements = (
            max(self.conv_sequence_tokens_by_bucket.values()) * self.kernel_shape.conv_dimension
        )
        l2norm_elements = (
            max(self.l2norm_sequence_tokens_by_bucket.values())
            * self.kernel_shape.norm_heads
            * self.kernel_shape.norm_head_dimension
        )
        return 4 * element_size * max(conv_elements, l2norm_elements)


def _qwen_text_config(model_config: Any) -> Any:  # noqa: ANN401
    text_config = getattr(model_config, "text_config", None)
    return model_config if text_config is None else text_config


def _positive_config_integer(config: Any, name: str) -> int:  # noqa: ANN401
    value = getattr(config, name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"Qwen3.5 config field {name} must be a positive integer, got {value!r}")
    return value


def qwen35_fla_kernel_shape(model_config: Any) -> FlaKernelShape:  # noqa: ANN401
    """Derive the FLA convolution and Q/K normalization dimensions from Qwen3.5 config."""
    config = _qwen_text_config(model_config)
    key_heads = _positive_config_integer(config, "linear_num_key_heads")
    key_head_dimension = _positive_config_integer(config, "linear_key_head_dim")
    value_heads = _positive_config_integer(config, "linear_num_value_heads")
    value_head_dimension = _positive_config_integer(config, "linear_value_head_dim")
    conv_width = _positive_config_integer(config, "linear_conv_kernel_dim")
    activation = config.hidden_act
    if activation not in ("silu", "swish"):
        raise ValueError(
            f"Qwen3.5 FLA convolution expects silu or swish activation, got {activation!r}"
        )
    return FlaKernelShape(
        conv_dimension=2 * key_heads * key_head_dimension + value_heads * value_head_dimension,
        conv_width=conv_width,
        conv_activation=activation,
        norm_head_dimension=key_head_dimension,
        # Qwen repeats Q and K to the value-head count before FLA normalizes them.
        norm_heads=value_heads,
    )


def _bucket_sequence_tokens(bucket: int, *, block_rows: int, rows_per_token: int = 1) -> int:
    raw_nb = 1 if bucket == 1 else bucket // 2 + 1
    return ((raw_nb - 1) * block_rows // rows_per_token) + 1


def build_fla_nb_prewarm_plan(
    model_config: Any,  # noqa: ANN401
    *,
    sequence_shapes: Sequence[tuple[object, object]],
) -> FlaNBPrewarmPlan:
    """Plan all powers of two through the largest ``NB`` represented by the input shapes."""
    if not sequence_shapes:
        raise ValueError("sequence_shapes must contain at least one (batch_size, sequence_length)")
    shape = qwen35_fla_kernel_shape(model_config)
    total_tokens: list[int] = []
    for batch_size, sequence_length in sequence_shapes:
        if (
            not isinstance(batch_size, int)
            or isinstance(batch_size, bool)
            or batch_size < 1
            or not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length < 1
        ):
            raise ValueError(
                "each sequence shape must contain positive integer batch and sequence lengths"
            )
        total_tokens.append(batch_size * sequence_length)

    max_conv_nb = max((tokens + CONV_NB_TOKENS - 1) // CONV_NB_TOKENS for tokens in total_tokens)
    max_l2norm_nb = max(
        (tokens * shape.norm_heads + L2NORM_NB_ROWS - 1) // L2NORM_NB_ROWS
        for tokens in total_tokens
    )
    max_conv_bucket = bucket_nb(max_conv_nb)
    max_l2norm_bucket = bucket_nb(max_l2norm_nb)
    conv_buckets = tuple(1 << power for power in range(max_conv_bucket.bit_length()))
    l2norm_buckets = tuple(1 << power for power in range(max_l2norm_bucket.bit_length()))
    return FlaNBPrewarmPlan(
        kernel_shape=shape,
        conv_buckets=conv_buckets,
        l2norm_buckets=l2norm_buckets,
        conv_sequence_tokens_by_bucket={
            bucket: _bucket_sequence_tokens(bucket, block_rows=CONV_NB_TOKENS)
            for bucket in conv_buckets
        },
        l2norm_sequence_tokens_by_bucket={
            bucket: _bucket_sequence_tokens(
                bucket,
                block_rows=L2NORM_NB_ROWS,
                rows_per_token=shape.norm_heads,
            )
            for bucket in l2norm_buckets
        },
    )


def prewarm_fla_nb_buckets(
    model_config: Any,  # noqa: ANN401
    sequence_shapes: Sequence[tuple[object, object]],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> FlaNBPrewarmPlan:
    """Autotune every required bucket using the smallest representative sequence in that bucket."""
    plan = build_fla_nb_prewarm_plan(model_config, sequence_shapes=sequence_shapes)
    if any(batch_size != 1 for batch_size, _ in sequence_shapes):
        raise ValueError("FLA NB prewarming currently only supports batch size 1")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FLA NB prewarming requires an available CUDA device")
    install_fla_nb_bucketing()
    tensor_peak_bytes = plan.estimated_peak_tensor_bytes(
        element_size=torch.empty((), dtype=dtype).element_size()
    )
    logger.info(
        "prewarming FLA kernels: device=%s fla=%s conv_buckets=%s l2norm_buckets=%s "
        "conv_dim=%d conv_width=%d norm_dim=%d norm_heads=%d estimated_peak_tensor_bytes=%d",
        torch.cuda.get_device_name(device),
        EXPECTED_FLA_VERSION,
        plan.conv_buckets,
        plan.l2norm_buckets,
        plan.kernel_shape.conv_dimension,
        plan.kernel_shape.conv_width,
        plan.kernel_shape.norm_head_dimension,
        plan.kernel_shape.norm_heads,
        tensor_peak_bytes,
    )

    conv_ops = importlib.import_module("fla.modules.conv.triton.ops")
    l2norm = importlib.import_module("fla.modules.l2norm")
    for sequence_tokens in plan.conv_sequence_tokens_by_bucket.values():
        x = torch.zeros(
            (1, sequence_tokens, plan.kernel_shape.conv_dimension), device=device, dtype=dtype
        )
        weight = torch.zeros(
            (plan.kernel_shape.conv_dimension, plan.kernel_shape.conv_width),
            device=device,
            dtype=dtype,
        )
        # Qwen3.5's depthwise Conv1d is bias-free and uses configured SiLU activation.
        y, _ = conv_ops.causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=None,
            residual=None,
            activation=plan.kernel_shape.conv_activation,
        )
        dy = torch.zeros_like(y)
        conv_ops.causal_conv1d_bwd(
            x=x,
            dy=dy,
            dht=None,
            weight=weight,
            bias=None,
            residual=None,
            activation=plan.kernel_shape.conv_activation,
        )
        torch.cuda.synchronize(device)
        del x, weight, y, dy

    for sequence_tokens in plan.l2norm_sequence_tokens_by_bucket.values():
        x = torch.zeros(
            (
                1,
                sequence_tokens,
                plan.kernel_shape.norm_heads,
                plan.kernel_shape.norm_head_dimension,
            ),
            device=device,
            dtype=dtype,
        )
        y, rstd = l2norm.l2norm_fwd(x)
        dy = torch.zeros_like(y)
        l2norm.l2norm_bwd(y, rstd, dy)
        torch.cuda.synchronize(device)
        del x, y, rstd, dy

    _write_prewarm_manifest(plan, dtype=dtype, device=device, sequence_shapes=sequence_shapes)
    return plan


def _write_prewarm_manifest(
    plan: FlaNBPrewarmPlan,
    *,
    dtype: torch.dtype,
    device: torch.device,
    sequence_shapes: Sequence[tuple[object, object]],
) -> Path:
    cache_directory = (
        Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
        .expanduser()
        .resolve()
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_directory / "fla-nb-pow2-prewarm.json"
    payload = {
        "schema_version": 1,
        "flash_linear_attention_version": EXPECTED_FLA_VERSION,
        "fla_core_version": EXPECTED_FLA_CORE_VERSION,
        "triton_version": version("triton"),
        "torch_version": torch.__version__,
        "triton_cache_directory": str(cache_directory),
        "device": {
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "kernel_shape": {
            "conv_dimension": plan.kernel_shape.conv_dimension,
            "conv_width": plan.kernel_shape.conv_width,
            "conv_activation": plan.kernel_shape.conv_activation,
            "norm_head_dimension": plan.kernel_shape.norm_head_dimension,
            "norm_heads": plan.kernel_shape.norm_heads,
            "dtype": str(dtype),
        },
        "sequence_shapes": [list(shape) for shape in sequence_shapes],
        "kernel_buckets": {
            CONV_KERNEL_NAMES[0]: list(plan.conv_buckets),
            CONV_KERNEL_NAMES[1]: list(plan.conv_buckets),
            L2NORM_KERNEL_NAMES[0]: list(plan.l2norm_buckets),
            L2NORM_KERNEL_NAMES[1]: list(plan.l2norm_buckets),
        },
        "representative_sequence_tokens": {
            "conv": plan.conv_sequence_tokens_by_bucket,
            "l2norm": plan.l2norm_sequence_tokens_by_bucket,
        },
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=cache_directory,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(payload, temporary_file, sort_keys=True, indent=2)
        temporary_file.write("\n")
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    try:
        temporary_path.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    logger.info("wrote FLA NB prewarm manifest: %s", manifest_path)
    return manifest_path


def _parse_sequence_lengths(value: str) -> tuple[int, ...]:
    lengths = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not lengths or any(length < 1 for length in lengths):
        raise argparse.ArgumentTypeError(
            "sequence lengths must be a comma-separated list of positives"
        )
    return lengths


def main() -> None:
    """Load local model config and run the standalone FLA prewarm command."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prewarm_parser = subparsers.add_parser("prewarm", help="prewarm the Qwen3.5 FLA kernel buckets")
    prewarm_parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    prewarm_parser.add_argument("--sequence-lengths", type=_parse_sequence_lengths, required=True)
    prewarm_parser.add_argument("--batch-size", type=int, choices=(1,), default=1)
    prewarm_parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    sequence_shapes = tuple((args.batch_size, length) for length in args.sequence_lengths)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    prewarm_fla_nb_buckets(model_config, sequence_shapes, device=args.device)


if __name__ == "__main__":
    main()
