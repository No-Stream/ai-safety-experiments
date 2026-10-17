"""Runtime precision checks and tensor storage inventory for organism RL."""

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from reward_hacking.organism_rl.generate_provenance import active_provenance_path


def assert_fp32_training_state(model: nn.Module, optimizer: torch.optim.Optimizer | None) -> None:
    """Reject low-precision trainable storage before training or checkpointing."""
    trainable_names: dict[torch.Tensor, str] = {
        parameter: name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable_names:
        raise RuntimeError("organism RL requires trainable parameters")
    for parameter, name in trainable_names.items():
        if "lora_" not in name:
            raise RuntimeError(f"only LoRA parameters may train: {name}")
        if parameter.dtype != torch.float32:
            raise RuntimeError(f"trainable parameter {name} must be float32, got {parameter.dtype}")
        if parameter.grad is not None and parameter.grad.dtype != torch.float32:
            raise RuntimeError(f"gradient for {name} must be float32, got {parameter.grad.dtype}")
    if optimizer is None:
        return
    _assert_fp32_optimizer_state(optimizer, trainable_names)


def _assert_fp32_optimizer_state(
    optimizer: torch.optim.Optimizer, trainable_names: dict[torch.Tensor, str]
) -> None:
    for parameter, state in optimizer.state.items():
        if parameter not in trainable_names:
            raise RuntimeError("optimizer state belongs to a frozen or unknown parameter")
        for state_name, value in state.items():
            if isinstance(value, torch.Tensor) and value.dtype != torch.float32:
                raise RuntimeError(
                    f"optimizer state {state_name} for {trainable_names[parameter]} must be float32, "
                    f"got {value.dtype}"
                )


def tensor_dtype_inventory(
    model: nn.Module, optimizer: torch.optim.Optimizer | None
) -> dict[str, object]:
    """Record observed storage types, grouped by the module that owns each tensor."""
    base_weights: dict[str, set[str]] = {}
    adapter_weights: set[str] = set()
    for module in model.modules():
        module_type = type(module).__name__
        for name, parameter in module.named_parameters(recurse=False):
            if parameter.requires_grad:
                adapter_weights.add(str(parameter.dtype))
            elif "lora_" not in name:
                base_weights.setdefault(module_type, set()).add(str(parameter.dtype))
    optimizer_state: dict[str, set[str]] = {}
    if optimizer is not None:
        for state in optimizer.state.values():
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state.setdefault(name, set()).add(str(value.dtype))
    return {
        "base_weights_by_module_type": {
            key: sorted(values) for key, values in sorted(base_weights.items())
        },
        "lora_weights": sorted(adapter_weights),
        "lora_gradients": sorted(
            {
                str(parameter.grad.dtype)
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            }
        ),
        "optimizer_state": {key: sorted(values) for key, values in sorted(optimizer_state.items())},
    }


def record_precision_provenance(
    output_dir: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None
) -> None:
    """Persist observed tensor dtypes at startup and after each optimizer step."""
    path = active_provenance_path(output_dir)
    provenance = json.loads(path.read_text(encoding="utf-8"))
    flags = provenance["flags"]
    model_config = cast("Any", model).config.to_dict()
    text_config = model_config.get("text_config", model_config)
    if not isinstance(text_config, dict):
        raise TypeError("model text config must be a mapping")
    kv_dtype = flags["kv_cache_dtype"]
    if kv_dtype in (None, "auto"):
        kv_dtype = flags["sampler_model_dtype"]
    ssm_dtype = flags["mamba_ssm_cache_dtype"]
    if ssm_dtype in (None, "auto"):
        ssm_dtype = text_config["mamba_ssm_dtype"]
    provenance["resolved_precision"] = {
        **tensor_dtype_inventory(model, optimizer),
        "head_compute": "torch.float32",
        "loss_logits": "torch.float32",
        "sampler_head_compute": "torch.float32",
        "kv_cache": kv_dtype,
        "ssm_state": ssm_dtype,
        "sampler_weight_quantization": flags["sampler_resolved_quantization"],
    }
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(path)
