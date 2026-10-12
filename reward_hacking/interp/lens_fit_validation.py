"""Memory preflight and model-facing quality gates for generic Jacobian-lens fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from grpo.gpu_memory import gpu_memory_info
from reward_hacking.interp.jacobian import interior_eval_positions

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

RECOVERY_POSITIONS_PER_PROMPT = 4
LOGITS_TENSOR_RANK = 2


class FitMemoryError(RuntimeError):
    """The requested dimension batch fails a measured, single-dimension memory preflight."""


class FitGateError(RuntimeError):
    """An automatic technical fit gate did not pass."""


def recovery_gate_report(
    target_token_ids: torch.Tensor,
    *,
    fitted_token_ids: Mapping[int, torch.Tensor],
    logit_lens_token_ids: Mapping[int, torch.Tensor],
    sabotaged_token_ids: Mapping[int, torch.Tensor],
) -> dict[str, object]:
    """Compare fitted-lens token recovery with the logit-lens read and zero-J sabotage.

    Targets are the model's own final-logit argmax ids. Passing means at least one fitted source
    layer recovers more of those ids than the same read with every Jacobian matrix zeroed. The
    logit-lens rate is retained as a diagnostic; it is not a universal gate under corpus shift.
    """
    if target_token_ids.ndim != 1 or target_token_ids.numel() == 0:
        raise ValueError("target_token_ids must be a non-empty vector")
    layer_sets = (set(fitted_token_ids), set(logit_lens_token_ids), set(sabotaged_token_ids))
    if not layer_sets[0] or any(layers != layer_sets[0] for layers in layer_sets[1:]):
        raise ValueError("fitted, logit-lens and sabotaged predictions must cover the same layers")
    target = target_token_ids.long().cpu()
    layers: dict[str, dict[str, float | int | bool]] = {}
    for layer in sorted(layer_sets[0]):
        predictions = {
            "jacobian": fitted_token_ids[layer].long().cpu(),
            "logit_lens": logit_lens_token_ids[layer].long().cpu(),
            "sabotaged": sabotaged_token_ids[layer].long().cpu(),
        }
        for name, values in predictions.items():
            if values.shape != target.shape:
                raise ValueError(
                    f"layer {layer} {name} ids have shape {tuple(values.shape)}, expected {tuple(target.shape)}"
                )
        jacobian_accuracy = float((predictions["jacobian"] == target).float().mean())
        logit_accuracy = float((predictions["logit_lens"] == target).float().mean())
        sabotage_accuracy = float((predictions["sabotaged"] == target).float().mean())
        layers[str(layer)] = {
            "n_positions": int(target.numel()),
            "jacobian_top1_accuracy": jacobian_accuracy,
            "logit_lens_top1_accuracy": logit_accuracy,
            "sabotaged_top1_accuracy": sabotage_accuracy,
            "beats_zero_jacobian_sabotage": jacobian_accuracy > sabotage_accuracy,
        }
    return {
        "passed": any(
            bool(layer_report["beats_zero_jacobian_sabotage"]) for layer_report in layers.values()
        ),
        "target": "model final-logit argmax at selected interior positions",
        "sabotage": "all fitted Jacobian matrices set to zero",
        "n_positions": int(target.numel()),
        "layers": layers,
    }


def recovery_gate_from_logits(
    target_logits: torch.Tensor,
    *,
    fitted_logits: Mapping[int, torch.Tensor],
    logit_lens_logits: Mapping[int, torch.Tensor],
    sabotaged_logits: Mapping[int, torch.Tensor],
) -> dict[str, object]:
    """Check output tensors before reducing them to top-1 token ids."""
    return recovery_gate_report(
        _top1_ids(target_logits, "model final logits"),
        fitted_token_ids={
            layer: _top1_ids(logits, f"J-lens layer {layer}")
            for layer, logits in fitted_logits.items()
        },
        logit_lens_token_ids={
            layer: _top1_ids(logits, f"logit-lens layer {layer}")
            for layer, logits in logit_lens_logits.items()
        },
        sabotaged_token_ids={
            layer: _top1_ids(logits, f"sabotaged layer {layer}")
            for layer, logits in sabotaged_logits.items()
        },
    )


def _top1_ids(logits: torch.Tensor, label: str) -> torch.Tensor:
    """Validate a logits tensor before returning CPU top-1 token ids."""
    if logits.ndim != LOGITS_TENSOR_RANK or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise FitGateError(f"{label} must be a non-empty [positions, vocabulary] tensor")
    if not torch.isfinite(logits).all():
        raise FitGateError(f"{label} contains non-finite values; refusing token ids from argmax")
    return logits.argmax(dim=-1).long().cpu()


def preflight_dim_batch(  # noqa: PLR0913 - memory sizing must receive every fit knob explicitly
    jl: ModuleType,
    model: object,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    target_layer: int,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
) -> dict[str, int | float | str]:
    """Measure one safe dimension row, then conservatively bound the requested batch from free VRAM."""
    if not torch.cuda.is_available():
        raise FitMemoryError("Jacobian fits require a CUDA device for VRAM preflight")
    model_api = cast("Any", model)
    jlens_api = cast("Any", jl)
    torch.cuda.empty_cache()
    free_before, total_bytes = gpu_memory_info(torch, 0)
    base_allocated = int(torch.cuda.memory_allocated(0))
    base_reserved = int(torch.cuda.memory_reserved(0))
    probe_prompt = max(
        prompts, key=lambda prompt: int(model_api.encode(prompt, max_length=max_seq_len).shape[1])
    )
    torch.cuda.reset_peak_memory_stats(0)
    try:
        _jacobians, sequence_length, valid_positions = jlens_api.jacobian_for_prompt(
            model_api,
            probe_prompt,
            list(source_layers),
            target_layer=target_layer,
            dim_batch=1,
            max_seq_len=max_seq_len,
            skip_first=skip_first,
        )
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise FitMemoryError(
            "the single-dimension Jacobian preflight ran out of VRAM; reduce --max-seq-len or "
            "free GPU memory before fitting"
        ) from exc
    peak_allocated = int(torch.cuda.max_memory_allocated(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    one_dimension_peak = max(peak_allocated - base_allocated, peak_reserved - base_reserved, 0)
    del _jacobians
    torch.cuda.empty_cache()
    free_after, _total_after = gpu_memory_info(torch, 0)
    available_bytes = min(int(free_before), int(free_after))
    estimated_bytes = one_dimension_peak * (dim_batch + 1)
    if valid_positions <= 0:
        raise FitMemoryError(
            f"the longest selected prompt has {sequence_length} tokens and no valid positions "
            f"after skip_first={skip_first}"
        )
    if one_dimension_peak <= 0:
        raise FitMemoryError(
            "the single-dimension Jacobian preflight observed no CUDA allocation; refusing to estimate "
            "dim_batch memory without a measured device allocation"
        )
    if estimated_bytes > available_bytes:
        raise FitMemoryError(
            f"dim_batch={dim_batch} is refused by the single-dimension memory estimate: "
            f"estimated {estimated_bytes / 2**30:.2f} GiB including one dimension of headroom, "
            f"available VRAM is {available_bytes / 2**30:.2f} GiB. Reduce --dim-batch or "
            "--max-seq-len; no multi-dimension probe was run."
        )
    return {
        "device": torch.cuda.get_device_name(0),
        "total_vram_bytes": int(total_bytes),
        "free_vram_before_probe_bytes": int(free_before),
        "free_vram_after_probe_bytes": int(free_after),
        "single_dimension_peak_bytes": one_dimension_peak,
        "estimated_requested_peak_bytes": estimated_bytes,
        "requested_dim_batch": dim_batch,
        "probe_sequence_length": int(sequence_length),
        "probe_valid_positions": int(valid_positions),
        "estimate_limit": (
            "uses the lower free-VRAM observation and measured one-dimension peak with one dimension "
            "of headroom; it cannot guarantee fit if another process consumes VRAM afterward"
        ),
    }


def recovery_read(  # noqa: PLR0913 - recovery needs the same explicit fit and read settings
    jl: ModuleType,
    lens: object,
    model: object,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int],
    max_seq_len: int,
    skip_first: int,
) -> dict[str, object]:
    """Read final-token recovery for the fit, raw logit lens and a zero-J sabotaged copy."""
    lens_api = cast("Any", lens)
    model_api = cast("Any", model)
    jlens_api = cast("Any", jl)
    fitted_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in source_layers}
    logit_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in source_layers}
    sabotaged_by_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in source_layers}
    target_rows: list[torch.Tensor] = []
    sabotaged_lens = jlens_api.JacobianLens(
        jacobians={layer: torch.zeros_like(lens_api.jacobians[layer]) for layer in source_layers},
        n_prompts=0,
        d_model=int(lens_api.d_model),
    )
    prompts_skipped = 0
    for prompt in prompts:
        input_ids = model_api.encode(prompt, max_length=max_seq_len)
        positions = interior_eval_positions(
            int(input_ids.shape[1]),
            skip_first=skip_first,
            max_positions=RECOVERY_POSITIONS_PER_PROMPT,
        )
        if not positions:
            prompts_skipped += 1
            continue
        fitted_logits, final_logits, _ = lens_api.apply(
            model_api,
            prompt,
            layers=source_layers,
            positions=positions,
            max_seq_len=max_seq_len,
        )
        baseline_logits, _baseline_final_logits, _ = lens_api.apply(
            model_api,
            prompt,
            layers=source_layers,
            positions=positions,
            max_seq_len=max_seq_len,
            use_jacobian=False,
        )
        zero_logits, _zero_final_logits, _ = sabotaged_lens.apply(
            model_api,
            prompt,
            layers=source_layers,
            positions=positions,
            max_seq_len=max_seq_len,
        )
        target_rows.append(_top1_ids(final_logits, "model final logits"))
        for layer in source_layers:
            fitted_by_layer[layer].append(_top1_ids(fitted_logits[layer], f"J-lens layer {layer}"))
            logit_by_layer[layer].append(
                _top1_ids(baseline_logits[layer], f"logit-lens layer {layer}")
            )
            sabotaged_by_layer[layer].append(_top1_ids(zero_logits[layer], f"zero-J layer {layer}"))
    if not target_rows:
        raise FitGateError(
            f"no corpus prompts had an interior token after skip_first={skip_first}; "
            "the recovery gate cannot run"
        )
    report = recovery_gate_report(
        torch.cat(target_rows),
        fitted_token_ids={layer: torch.cat(rows) for layer, rows in fitted_by_layer.items()},
        logit_lens_token_ids={layer: torch.cat(rows) for layer, rows in logit_by_layer.items()},
        sabotaged_token_ids={layer: torch.cat(rows) for layer, rows in sabotaged_by_layer.items()},
    )
    report["prompts_requested"] = len(prompts)
    report["prompts_read"] = len(prompts) - prompts_skipped
    report["prompts_skipped_too_short"] = prompts_skipped
    if not report["passed"]:
        raise FitGateError(
            "token-recovery gate failed: no fitted layer recovered more model-final argmax tokens "
            "than the zero-J sabotage"
        )
    return report


def half_disagreement(odd: object, even: object) -> dict[str, float]:
    """Measure per-layer relative Frobenius disagreement between the two fitted prompt halves."""
    odd_lens = cast("Any", odd)
    even_lens = cast("Any", even)
    if odd_lens.jacobians.keys() != even_lens.jacobians.keys():
        raise FitGateError("odd and even lenses cover different source layers")
    disagreements: dict[str, float] = {}
    for layer in sorted(odd_lens.jacobians):
        odd_j = odd_lens.jacobians[layer].float()
        even_j = even_lens.jacobians[layer].float()
        denominator = (odd_j.norm() + even_j.norm()) / 2
        if denominator == 0:
            raise FitGateError(f"split-half Jacobians at layer {layer} both have zero norm")
        disagreements[str(layer)] = float((odd_j - even_j).norm() / denominator)
    return disagreements
