"""Lightweight reporting of the implementation each DeltaNet kernel will resolve to."""

from __future__ import annotations

import importlib
import sys

# Names and lookup order read from transformers 5.15.0's
# `use_kernel_func_from_hub_with_fallback`: Hub kernels first, then the named local package, then
# torch.
DELTANET_KERNEL_FUNCTIONS = (
    ("chunk_gated_delta_rule", "fla"),
    ("recurrent_gated_delta_rule", "fla"),
    ("causal_conv1d_fn", "causal_conv1d"),
    ("causal_conv1d_update", "causal_conv1d"),
)


def deltanet_kernel_paths() -> dict[str, str]:
    """Report which implementation each Gated DeltaNet kernel function will actually use.

    Installed is not engaged, and transformers falls back silently: an absent package, or a
    package whose export names do not match what transformers looks for, leaves the slow torch path
    in place with nothing in the log. When the modeling module is already loaded, read its actual
    bindings instead of predicting what a future import would resolve.
    """
    if "transformers.models.qwen3_5.modeling_qwen3_5" in sys.modules:
        from games.deltanet_kernels import (  # noqa: PLC0415 - avoid import cycle
            bound_deltanet_kernels,
        )

        return bound_deltanet_kernels()

    from transformers.integrations.hub_kernels import (  # noqa: PLC0415
        _KERNELS_INTERNAL_PATH_MAPPINGS,  # pyright: ignore[reportPrivateUsage]
        resolve_internal_import,  # pyright: ignore[reportPrivateImportUsage]
    )

    paths: dict[str, str] = {}
    for function_name, package in DELTANET_KERNEL_FUNCTIONS:
        internal = _KERNELS_INTERNAL_PATH_MAPPINGS.get(function_name)
        full = function_name if internal is None else f"{internal}.{function_name}"
        try:
            module = importlib.import_module(package)
        except ImportError:
            paths[function_name] = f"torch fallback ({package} absent)"
            continue
        resolved = resolve_internal_import(module, full)
        paths[function_name] = (
            f"{resolved.__module__}.{resolved.__name__}"
            if resolved is not None
            else f"torch fallback ({package} has no {full})"
        )
    return paths
