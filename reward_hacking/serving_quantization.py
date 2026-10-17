"""Shared serving quantization flags, measurement validation and vLLM options."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping, Sequence


class ServingQuantizationKwargs(TypedDict):
    """Quantization fields accepted by VLLMServerConfig."""

    kv_cache_dtype: str | None
    quantization: str | None
    quantization_config: Mapping[str, object] | None


def add_serving_quantization_arguments(
    parser: argparse.ArgumentParser,
    *,
    kv_cache_dtype_default: str | None = None,
    sampler_quantization_default: str = "none",
    sampler_quantization_ignore_default: Sequence[str] = (),
    quantization_admission_default: Path | None = None,
) -> None:
    """Expose the same serving controls with caller-selected defaults."""
    parser.add_argument("--kv-cache-dtype", choices=("auto", "fp8"), default=kv_cache_dtype_default)
    parser.add_argument(
        "--sampler-quantization",
        choices=("none", "fp8_per_channel", "fp8_per_block", "fp8_per_tensor"),
        default=sampler_quantization_default,
    )
    parser.add_argument(
        "--sampler-quantization-ignore",
        action="append",
        default=None,
    )
    parser.set_defaults(
        _sampler_quantization_ignore_default=list(sampler_quantization_ignore_default)
    )
    parser.add_argument(
        "--quantization-admission", type=Path, default=quantization_admission_default
    )


def validate_serving_quantization(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    speculative_config: Mapping[str, object] | None = None,
    additional_quantization: bool = False,
) -> None:
    """Require finite before/after measurements and an unquantized MTP draft head."""
    default_ignore = vars(args).pop("_sampler_quantization_ignore_default")
    if args.sampler_quantization_ignore is None:
        args.sampler_quantization_ignore = (
            list(default_ignore) if args.sampler_quantization != "none" else []
        )
    elif args.sampler_quantization == "none" and args.sampler_quantization_ignore:
        parser.error("sampler-quantization-ignore requires sampler-quantization")
    quantized = (
        additional_quantization
        or args.sampler_quantization != "none"
        or args.kv_cache_dtype == "fp8"
    )
    if not quantized and args.quantization_admission == parser.get_default(
        "quantization_admission"
    ):
        args.quantization_admission = None
    if quantized and args.quantization_admission is None:
        parser.error(
            "quantization requires --quantization-admission with before/after logprob gaps"
        )
    if args.quantization_admission is not None:
        _validate_admission_record(args.quantization_admission, parser)
    if (
        args.sampler_quantization != "none"
        and speculative_config is not None
        and speculative_config.get("method") == "mtp"
        and not any("mtp" in pattern for pattern in args.sampler_quantization_ignore)
    ):
        parser.error(
            "online sampler quantization with MTP drafting needs an MTP pattern in "
            "--sampler-quantization-ignore so the draft head stays bf16"
        )


def _validate_admission_record(path: Path, parser: argparse.ArgumentParser) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"cannot read quantization-admission: {error}")
    if not isinstance(record, dict) or any(
        not isinstance(record.get(stage), dict)
        or type(record[stage].get("mean_abs_logprob_gap")) not in (int, float)
        or not math.isfinite(record[stage]["mean_abs_logprob_gap"])
        or record[stage]["mean_abs_logprob_gap"] < 0
        for stage in ("before", "after")
    ):
        parser.error("quantization-admission requires finite before/after mean_abs_logprob_gap")


def serving_quantization_kwargs(args: argparse.Namespace) -> ServingQuantizationKwargs:
    """Forward the effective quantization controls to the shared server config."""
    return {
        "kv_cache_dtype": args.kv_cache_dtype,
        "quantization": args.sampler_quantization if args.sampler_quantization != "none" else None,
        "quantization_config": {"ignore": args.sampler_quantization_ignore}
        if args.sampler_quantization_ignore
        else None,
    }
