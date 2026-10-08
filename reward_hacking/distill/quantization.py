"""Experimental frozen-base W8A8/FP8 forward GEMMs; adoption requires real-row drift checks.

These choices change training semantics. Input gradients use the dequantized frozen weight in
bf16; forward activations use symmetric row quantization with a straight-through input gradient.
The names retain the CLI's torchao prefix, but this custom autograd path uses Torch primitives:
torchao's dynamic-activation workflows do not provide this frozen-base input-gradient contract.
"""

from typing import Protocol, cast, override

import torch

LOGIT_DIMENSIONS = 2
QUANTIZED_MODES = ("torchao-w8a8", "torchao-fp8-rowwise")


def quantize_rows(weight: torch.Tensor, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one symmetric scale per row; zero rows retain a finite scale."""
    if mode not in QUANTIZED_MODES:
        raise ValueError(f"unknown frozen-base quantization: {mode}")
    maximum = 127.0 if mode == "torchao-w8a8" else torch.finfo(torch.float8_e4m3fn).max
    scale = (weight.float().abs().amax(dim=-1, keepdim=True) / maximum).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    normalized = (weight.float() / scale).clamp(-maximum, maximum)
    packed = (
        normalized.round().to(torch.int8)
        if mode == "torchao-w8a8"
        else normalized.to(torch.float8_e4m3fn).view(torch.int8)
    )
    return packed.contiguous(), scale


def dequantize_rows(packed: torch.Tensor, scales: torch.Tensor, mode: str) -> torch.Tensor:
    """Reconstruct the frozen weight for the input-gradient GEMM."""
    if mode not in QUANTIZED_MODES:
        raise ValueError(f"unknown frozen-base quantization: {mode}")
    values = packed if mode == "torchao-w8a8" else packed.view(torch.float8_e4m3fn)
    return values.float() * scales


def quantized_matmul(
    inputs: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, mode: str
) -> torch.Tensor:
    """Use CUDA tensor cores without a CPU or dtype fallback."""
    if inputs.device.type != "cuda" or packed.device != inputs.device:
        raise RuntimeError("frozen-base W8A8/FP8 matmul requires matching CUDA tensors")
    if inputs.dtype != torch.bfloat16:
        raise ValueError("frozen-base W8A8/FP8 matmul requires bf16 activations")
    flattened = inputs.reshape(-1, inputs.shape[-1])
    # Tensor-core GEMMs require aligned dimensions; token count is usually unaligned.
    padding = (-flattened.shape[0]) % 16
    padded = torch.nn.functional.pad(flattened, (0, 0, 0, padding))
    activation, activation_scale = quantize_rows(padded, mode)
    if mode == "torchao-w8a8":
        result = torch._int_mm(activation, packed.T).float()  # noqa: SLF001  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
        result = result * activation_scale * scales.T
    else:
        result = torch._scaled_mm(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
            activation.view(torch.float8_e4m3fn),
            packed.view(torch.float8_e4m3fn).T,
            scale_a=activation_scale,
            scale_b=scales.T,
            out_dtype=torch.bfloat16,
        )
    return (
        result[: flattened.shape[0]].to(inputs.dtype).reshape(*inputs.shape[:-1], packed.shape[0])
    )


class MatmulContext(Protocol):
    """Autograd context for frozen packed weights."""

    saved_tensors: tuple[torch.Tensor, torch.Tensor]
    mode: str
    input_dtype: torch.dtype

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        """Retain packed weights for input differentiation."""
        ...


class FrozenMatmul(torch.autograd.Function):
    """Differentiate inputs only; packed base weights and scales are permanently frozen."""

    @staticmethod
    def forward(
        ctx: MatmulContext,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        """Compute the quantized frozen projection."""
        ctx.save_for_backward(weight, scales)
        ctx.mode = mode
        ctx.input_dtype = inputs.dtype
        return quantized_matmul(inputs, weight, scales, mode)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: MatmulContext, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        """Use a straight-through activation gradient and dequantized frozen weights."""
        weight, scales = ctx.saved_tensors
        reconstructed = dequantize_rows(weight, scales, ctx.mode).to(ctx.input_dtype)
        return grad_output @ reconstructed, None, None, None


class FrozenQuantizedLinear(torch.nn.Linear):
    """PEFT-compatible Linear with packed int8 storage, including FP8 bit patterns.

    Keeping FP8 bit patterns in an integer Parameter prevents PEFT from casting trainable
    adapters to float8. Only ordinary LoRA adapters are supported; merging would corrupt storage.
    """

    weight_scale: torch.Tensor

    def __init__(self, source: torch.nn.Linear, mode: str) -> None:
        """Pack a base layer without allocating a second dense weight."""
        # Do not allocate a second dense weight while converting a large base.
        torch.nn.Module.__init__(self)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.mode = mode
        packed, scales = quantize_rows(source.weight.detach(), mode)
        self.weight = torch.nn.Parameter(packed, requires_grad=False)
        self.register_buffer("weight_scale", scales)
        source_bias = cast("torch.Tensor | None", source.bias)
        self.bias = (
            None
            if source_bias is None
            else torch.nn.Parameter(source_bias.detach(), requires_grad=False)
        )

    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = cast(
            "torch.Tensor", FrozenMatmul.apply(input, self.weight, self.weight_scale, self.mode)
        )
        bias = cast("torch.Tensor | None", self.bias)
        return result if bias is None else result + bias


def quantize_frozen_base(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    """Replace body linears before adding LoRA; preserve output head, norms and embeddings."""
    if mode not in QUANTIZED_MODES:
        raise ValueError(f"unknown frozen-base quantization: {mode}")
    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "lm_head" not in name
    ]
    if not candidates:
        raise ValueError("model has no body linear modules to quantize")
    for name, module in candidates:
        if module.weight.device.type != "cuda":
            raise RuntimeError("quantize_frozen_base requires the model on CUDA")
        if module.in_features % 16 or module.out_features % 16:
            raise ValueError(f"{name}: tensor-core dimensions must be multiples of 16")
    for name, module in candidates:
        model.set_submodule(name, FrozenQuantizedLinear(module, mode))
    return model


def drift_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    """Measure full-vocabulary KL and target log-prob drift for selected assistant tokens."""
    if (
        reference.shape != candidate.shape
        or reference.ndim != LOGIT_DIMENSIONS
        or targets.shape != reference.shape[:1]
        or reference.shape[0] < 1
    ):
        raise ValueError(
            "drift requires matching nonempty token-by-vocabulary logits and token targets"
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("drift logits must be finite")
    reference_logps, candidate_logps = (
        reference.float().log_softmax(-1),
        candidate.float().log_softmax(-1),
    )
    kl = (reference_logps.exp() * (reference_logps - candidate_logps)).sum(-1).clamp_min(0)
    logp_delta = (candidate_logps - reference_logps).gather(-1, targets[:, None]).squeeze(-1).abs()
    return {
        "tokens": float(targets.numel()),
        "mean_kl": float(kl.mean()),
        "mean_abs_logp": float(logp_delta.mean()),
        "max_abs_logp": float(logp_delta.max()),
    }


def enforce_drift_limits(
    metrics: dict[str, float], *, max_mean_kl: float, max_mean_abs_logp: float, max_abs_logp: float
) -> None:
    """Refuse candidate adoption on excess drift or nonfinite measurements."""
    for name, limit in (
        ("mean_kl", max_mean_kl),
        ("mean_abs_logp", max_mean_abs_logp),
        ("max_abs_logp", max_abs_logp),
    ):
        value = metrics[name]
        if not torch.isfinite(torch.tensor(value)) or value < 0 or not 0 < limit < float("inf"):
            raise ValueError("drift values and positive limits must be finite")
        if value > limit:
            raise ValueError(f"quantization drift exceeds {name} limit: {value:.6g} > {limit:.6g}")
