"""Worker-local residual edits for eager vLLM decoder layers."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

import torch
from torch import nn

from reward_hacking.interp.steering import ablate_residual, steer_residual
from reward_hacking.model_backend import VLLMBackend, assert_no_banned_token_ids

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

InterventionKind = Literal["none", "subspace", "steering"]
ResidualTransform = Callable[[torch.Tensor], torch.Tensor]
_HANDLES_ATTR = "_residual_intervention_handles"
_COUNTS_ATTR = "_residual_intervention_counts"
ROW_BASIS_DIMENSIONS = 2
PAIR_LENGTH = 2


class ModelWorkerAccess(Protocol):
    """The public vLLM named worker RPC surface, also implemented by CPU fakes."""

    def collective_rpc(self, method: str, *, args: tuple[object, ...] = ()) -> list[Any]:
        """Invoke a named method on every model worker."""
        ...


class _WorkerWithModel(Protocol):
    def get_model(self) -> nn.Module: ...


@dataclass(frozen=True)
class InterventionSpec:
    """CPU tensors and model dimensions sent as one serializable worker control message."""

    kind: InterventionKind
    by_layer: Mapping[int, torch.Tensor]
    n_layers: int
    d_model: int
    alpha: float | None = None

    @classmethod
    def none(cls, *, n_layers: int, d_model: int) -> InterventionSpec:
        """Describe a baseline with no residual edit."""
        return cls("none", {}, n_layers, d_model)

    @classmethod
    def subspace(
        cls, by_layer: Mapping[int, torch.Tensor], *, n_layers: int, d_model: int
    ) -> InterventionSpec:
        """Describe projection off each layer's orthonormal row basis."""
        return cls("subspace", by_layer, n_layers, d_model)

    @classmethod
    def steering(
        cls, by_layer: Mapping[int, torch.Tensor], *, alpha: float, n_layers: int, d_model: int
    ) -> InterventionSpec:
        """Describe additive steering at the named layers."""
        return cls("steering", by_layer, n_layers, d_model, alpha)

    def validate(self) -> None:  # noqa: C901, PLR0912 - one complete serialized-spec contract
        """Refuse malformed geometry before sending any tensor to a worker."""
        if self.kind not in ("none", "subspace", "steering"):
            raise ValueError(f"unknown residual intervention kind {self.kind!r}")
        if self.n_layers <= 0 or self.d_model <= 0:
            raise ValueError("intervention model dimensions must be positive")
        if self.kind == "none":
            if self.by_layer or self.alpha is not None:
                raise ValueError("none intervention cannot carry tensors or alpha")
            return
        if not self.by_layer:
            raise ValueError(f"{self.kind} intervention requires at least one layer")
        if self.kind == "subspace" and self.alpha is not None:
            raise ValueError("subspace intervention cannot carry alpha")
        if self.kind == "steering" and (
            self.alpha is None or not torch.isfinite(torch.tensor(self.alpha))
        ):
            raise ValueError("steering intervention requires a finite alpha")
        for layer, tensor in self.by_layer.items():
            if not 0 <= layer < self.n_layers:
                raise ValueError(f"intervention layer {layer} is outside 0..{self.n_layers - 1}")
            if tensor.device.type != "cpu" or not bool(torch.isfinite(tensor).all()):
                raise ValueError(
                    f"layer {layer} tensor must be finite and on CPU for worker transfer"
                )
            if self.kind == "subspace":
                if (
                    tensor.ndim != ROW_BASIS_DIMENSIONS
                    or tensor.shape[0] == 0
                    or tensor.shape[1] != self.d_model
                ):
                    raise ValueError(f"layer {layer} subspace basis has incompatible shape")
                if not torch.allclose(
                    tensor.float() @ tensor.float().T,
                    torch.eye(tensor.shape[0]),
                    rtol=1e-4,
                    atol=1e-4,
                ):
                    raise ValueError(f"layer {layer} subspace basis is not orthonormal")
            elif tensor.ndim != 1 or tensor.shape[0] != self.d_model or not bool(tensor.norm() > 0):
                raise ValueError(f"layer {layer} steering vector has incompatible shape")


def _decoder_layers(model: nn.Module) -> tuple[Sequence[nn.Module], object]:
    language_model = getattr(model, "language_model", None)
    trunk = getattr(language_model, "model", None) if language_model is not None else None
    if trunk is None:
        trunk = getattr(model, "model", None)
    if trunk is None:
        trunk = model
    layers = getattr(trunk, "layers", None)
    config = getattr(trunk, "config", None)
    if not isinstance(layers, (nn.ModuleList, list, tuple)) or config is None:
        raise TypeError("vLLM model has no supported decoder layer list and config")
    if any(not isinstance(layer, nn.Module) for layer in layers):
        raise TypeError("vLLM decoder layer list contains a non-module entry")
    return cast("Sequence[nn.Module]", layers), config


def _model_dimensions(model: nn.Module) -> tuple[int, int]:
    layers, config = _decoder_layers(model)
    n_layers = getattr(config, "num_hidden_layers", None)
    d_model = getattr(config, "hidden_size", None)
    if not isinstance(n_layers, int) or not isinstance(d_model, int):
        raise TypeError("vLLM decoder config must expose num_hidden_layers and hidden_size")
    if len(layers) != n_layers:
        raise ValueError(f"vLLM decoder layer count {len(layers)} disagrees with config {n_layers}")
    return n_layers, d_model


def _output_hook(
    transform: ResidualTransform, counts: dict[int, int], layer_index: int
) -> Callable[[nn.Module, tuple[object, ...], object], object]:
    def hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> object:
        if isinstance(output, torch.Tensor):
            full = output
            pair = False
        elif (
            isinstance(output, tuple)
            and len(output) == PAIR_LENGTH
            and isinstance(output[0], torch.Tensor)
            and isinstance(output[1], torch.Tensor)
            and output[0].shape == output[1].shape
        ):
            full = output[0] + output[1]
            pair = True
        else:
            raise ValueError("vLLM decoder layer output must be a tensor or a matching tensor pair")
        if full.ndim not in (2, 3):
            raise ValueError("vLLM decoder layer output must have token and hidden dimensions")
        edited = transform(full)
        counts[layer_index] += 1
        # Fused add-norm reconstructs edited + 0 exactly, including at the final layer.
        return (torch.zeros_like(edited), edited) if pair else edited

    return hook


def _install_on_model(model: nn.Module, spec: InterventionSpec) -> tuple[int, int]:
    if hasattr(model, _HANDLES_ATTR):
        raise RuntimeError("a residual intervention is already installed on this vLLM worker")
    actual_layers, actual_width = _model_dimensions(model)
    if actual_layers != spec.n_layers:
        raise ValueError(f"vLLM layer count {actual_layers} disagrees with spec {spec.n_layers}")
    if actual_width != spec.d_model:
        raise ValueError(f"vLLM d_model {actual_width} disagrees with spec {spec.d_model}")
    layers, _ = _decoder_layers(model)
    counts = dict.fromkeys(spec.by_layer, 0)
    handles: list[RemovableHandle] = []
    for layer_index, tensor in sorted(spec.by_layer.items()):
        layer = layers[layer_index]
        parameter = next(layer.parameters(), None)
        direction = tensor.to(
            device=parameter.device if parameter is not None else torch.device("cpu")
        )
        transform: ResidualTransform
        if spec.kind == "subspace":
            transform = partial(ablate_residual, direction=direction)
        elif spec.kind == "steering":
            transform = partial(
                steer_residual, direction=direction, alpha=cast("float", spec.alpha)
            )
        else:
            raise ValueError("none intervention cannot name a layer")
        handles.append(layer.register_forward_hook(_output_hook(transform, counts, layer_index)))
    setattr(model, _HANDLES_ATTR, handles)
    setattr(model, _COUNTS_ATTR, counts)
    return actual_layers, actual_width


def _counts_on_model(model: nn.Module) -> dict[int, int]:
    if not hasattr(model, _COUNTS_ATTR):
        raise RuntimeError("no residual intervention is installed on this vLLM worker")
    return cast("dict[int, int]", getattr(model, _COUNTS_ATTR)).copy()


def _remove_on_model(model: nn.Module) -> None:
    if not hasattr(model, _HANDLES_ATTR):
        raise RuntimeError("no residual intervention is installed on this vLLM worker")
    for handle in cast("list[RemovableHandle]", getattr(model, _HANDLES_ATTR)):
        handle.remove()
    delattr(model, _HANDLES_ATTR)
    delattr(model, _COUNTS_ATTR)


class ResidualInterventionWorker:
    """vLLM worker extension reached by string RPC, avoiding callable serialization."""

    def residual_intervention_dimensions(self) -> tuple[int, int]:
        """Report the worker model's decoder depth and residual width."""
        return _model_dimensions(cast("_WorkerWithModel", self).get_model())

    def residual_intervention_install(
        self,
        kind: InterventionKind,
        by_layer: dict[int, list[float] | list[list[float]]],
        n_layers: int,
        d_model: int,
        alpha: float | None,
    ) -> tuple[int, int]:
        """Install one validated spec on the worker's resident model."""
        spec = InterventionSpec(kind, tensors_from_wire(by_layer), n_layers, d_model, alpha)
        spec.validate()
        return _install_on_model(cast("_WorkerWithModel", self).get_model(), spec)

    def residual_intervention_counts(self) -> dict[int, int]:
        """Report forward-hook calls by decoder layer."""
        return _counts_on_model(cast("_WorkerWithModel", self).get_model())

    def residual_intervention_remove(self) -> None:
        """Remove every intervention hook from the worker model."""
        _remove_on_model(cast("_WorkerWithModel", self).get_model())


class InterventionVLLMBackend(VLLMBackend):
    """Use vLLM's exact allowed-token mask for the token-ban condition."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401 - vLLM constructor forwarding
        """Construct the usual persistent vLLM backend with a token audit buffer."""
        super().__init__(*args, **kwargs)
        self._last_banned_token_counts: list[int] = []

    @property
    def llm(self) -> ModelWorkerAccess:
        """Expose the engine's named worker RPC surface."""
        return cast("ModelWorkerAccess", self._llm)

    def set_generation_seed(self, seed: int) -> None:
        """Create fresh params so vLLM recomputes its cached sampling type for this condition."""
        params = self._sampling_params.clone()
        params.seed = seed
        params.__dict__.pop("sampling_type", None)
        self._sampling_params = params

    def clear_generation_audit(self) -> None:
        """Reset exact generated-token ban counts for one condition."""
        self._last_banned_token_counts = []

    @property
    def last_banned_token_counts(self) -> tuple[int, ...]:
        """Return one exact banned-token count per completion since reset."""
        return tuple(self._last_banned_token_counts)

    def generate(
        self, prompts: list[str], *, generation_kwargs: Mapping[str, object] | None = None
    ) -> list[str]:
        """Generate with optional exact token masking and audit the resulting IDs."""
        if generation_kwargs is None:
            return super().generate(prompts)
        unknown = set(generation_kwargs) - {"banned_token_ids"}
        if unknown:
            raise ValueError(f"unsupported vLLM generation kwargs: {sorted(unknown)}")
        banned = generation_kwargs.get("banned_token_ids")
        if not isinstance(banned, (tuple, list)) or not banned:
            raise ValueError("vLLM token ban requires nonempty banned_token_ids")
        banned_ids = frozenset(int(token_id) for token_id in banned)
        model_config = self._llm.llm_engine.model_config.hf_text_config
        vocab_size = int(model_config.vocab_size)
        tokenizer_size = len(self.tokenizer)
        if tokenizer_size > vocab_size:
            raise ValueError("vLLM tokenizer has more ids than the served model vocabulary")
        if min(banned_ids) < 0 or max(banned_ids) >= tokenizer_size:
            raise ValueError("banned token id is outside the served tokenizer vocabulary")
        allowed = [token_id for token_id in range(tokenizer_size) if token_id not in banned_ids]
        if not allowed:
            raise ValueError("token ban would remove the entire served model vocabulary")
        original_params = self._sampling_params
        masked_params = original_params.clone()
        masked_params.allowed_token_ids = allowed
        self._sampling_params = masked_params
        try:
            outputs = self.generate_tokenized(prompts)
        finally:
            self._sampling_params = original_params
        for output in outputs:
            self._last_banned_token_counts.append(
                assert_no_banned_token_ids(output.response_token_ids, banned_ids)
            )
        return [output.completion.text for output in outputs]


def tensors_to_wire(
    by_layer: Mapping[int, torch.Tensor],
) -> dict[int, list[float] | list[list[float]]]:
    """Encode per-layer tensors as float32 nested lists.

    vLLM's RPC serializer delivers tensors to the worker as nested lists, so the encoding is made
    explicit here and reversed by :func:`tensors_from_wire`; float32 round-trips exactly through
    Python floats.
    """
    return {
        int(layer): cast(
            "list[float] | list[list[float]]", tensor.detach().to("cpu", torch.float32).tolist()
        )
        for layer, tensor in by_layer.items()
    }


def tensors_from_wire(
    by_layer: Mapping[int, list[float] | list[list[float]]],
) -> dict[int, torch.Tensor]:
    """Rebuild float32 tensors from :func:`tensors_to_wire` output (keys may arrive as strings)."""
    return {
        int(layer): torch.tensor(values, dtype=torch.float32) for layer, values in by_layer.items()
    }


def install(llm: ModelWorkerAccess, spec: InterventionSpec) -> None:
    """Validate the spec and register hooks through vLLM's public model-worker RPC."""
    spec.validate()
    dimensions = llm.collective_rpc("residual_intervention_dimensions")
    if not dimensions:
        raise RuntimeError("vLLM returned no model workers for intervention validation")
    for n_layers, d_model in dimensions:
        if n_layers != spec.n_layers:
            raise ValueError(f"vLLM layer count {n_layers} disagrees with spec {spec.n_layers}")
        if d_model != spec.d_model:
            raise ValueError(f"vLLM d_model {d_model} disagrees with spec {spec.d_model}")
    installed = llm.collective_rpc(
        "residual_intervention_install",
        args=(spec.kind, tensors_to_wire(spec.by_layer), spec.n_layers, spec.d_model, spec.alpha),
    )
    if len(installed) != len(dimensions):
        raise RuntimeError("vLLM did not confirm intervention installation on every worker")


def remove(llm: ModelWorkerAccess) -> None:
    """Clear the worker-local hooks and counters."""
    llm.collective_rpc("residual_intervention_remove")


@contextmanager
def intervention(llm: ModelWorkerAccess, spec: InterventionSpec) -> Generator[None]:
    """Refuse a silent hook after generation and clear it on success or failure."""
    install(llm, spec)
    try:
        yield
        if spec.by_layer:
            for worker_counts in llm.collective_rpc("residual_intervention_counts"):
                missed = [layer for layer, count in worker_counts.items() if count == 0]
                if missed:
                    raise RuntimeError(f"vLLM residual hook never fired at layers {missed}")
    finally:
        remove(llm)
