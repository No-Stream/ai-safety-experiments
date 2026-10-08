"""Opt-in long-sequence kernels, frozen-weight quantization and checkpoint offload."""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MethodType
from typing import Any, Literal, Protocol, cast

import psutil
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from games.deltanet_kernels import bridge_decode_kernel

logger = logging.getLogger(__name__)
BaseQuantization = Literal["none", "torchao-int8", "nf4"]


@dataclass(frozen=True)
class MemoryOptions:
    """Explicit settings whose defaults preserve the existing trainer."""

    deltanet_conv_kernel: str = "default"
    liger: bool = False
    tiled_mlp: bool = False
    offload_checkpoints: bool = False
    base_quantization: BaseQuantization = "none"


def install_before_load(options: MemoryOptions) -> None:
    """Patch process-wide kernels before constructing the Qwen3.5 model."""
    if options.deltanet_conv_kernel == "fla":
        logger.info("installed DeltaNet bridge: %s", bridge_decode_kernel())
    elif options.deltanet_conv_kernel != "default":
        raise ValueError(f"unknown DeltaNet convolution kernel: {options.deltanet_conv_kernel}")
    if options.liger:
        # Importing the modeling module before the bridge freezes its kernel bindings.
        patch = importlib.import_module("liger_kernel.transformers.monkey_patch")
        patch.apply_liger_kernel_to_qwen3_5(fused_linear_cross_entropy=False)


def nf4_config() -> BitsAndBytesConfig:
    """Keep bf16 compute and the scorer's output head while quantizing the frozen base."""
    logger.warning("NF4 base weights drift from bf16: measured KL ~0.034 nats/token")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
        llm_int8_skip_modules=["lm_head"],
    )


def quantize_base_linear(module: torch.nn.Module, name: str) -> bool:
    """Exclude the bf16 output head, embeddings and norms from weight-only quantization."""
    return isinstance(module, torch.nn.Linear) and "lm_head" not in name


def load_base_model(
    model_id: str, quantization: BaseQuantization, *, revision: str | None = None
) -> torch.nn.Module:
    """Load exactly as before unless frozen-base quantization was requested."""
    revision_options = {"revision": revision} if revision is not None else {}
    if quantization == "none":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            **revision_options,
        )
    if quantization == "nf4":
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            quantization_config=nf4_config(),
            device_map={"": torch.cuda.current_device()},
            **revision_options,
        )
    if quantization != "torchao-int8":
        raise ValueError(f"unknown base quantization: {quantization}")
    quantization_module = importlib.import_module("torchao.quantization")
    model = cast(
        "Any",
        AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            **revision_options,
        ),
    ).cuda()
    quantization_module.quantize_(
        model, quantization_module.Int8WeightOnlyConfig(), filter_fn=quantize_base_linear
    )
    torch.cuda.empty_cache()
    return model


def tiled_mlp_compute(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Compute one MLP tile with the measured fused SwiGLU implementation."""
    operations = importlib.import_module("liger_kernel.ops")
    mlp = cast("Any", module)
    gate, up = mlp.gate_proj(hidden_states), mlp.up_proj(hidden_states)
    return cast("torch.Tensor", mlp.down_proj(operations.LigerSiLUMulFunction.apply(gate, up)))


def tiled_mlp_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Use Liger's sequence tiles while retaining gradients for every trainable adapter."""
    operations = importlib.import_module("liger_kernel.ops")
    return cast(
        "torch.Tensor",
        operations.apply_tiled_mlp(
            fn=tiled_mlp_compute,
            mlp_module=module,
            x=hidden_states,
            num_shards=None,
            compute_params=[
                parameter for parameter in module.parameters() if parameter.requires_grad
            ],
        ),
    )


def install_tiled_mlp(model: torch.nn.Module) -> None:
    """Replace each decoder MLP forward after adapters have been installed."""
    causal_model = cast("Any", model).get_base_model()
    for layer in causal_model.base_model.layers:
        layer.mlp.forward = MethodType(tiled_mlp_forward, layer.mlp)


def _decode_mountinfo_path(path: str) -> str:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        path = path.replace(encoded, decoded)
    return path


def _process_cgroup_path(cgroup_file: Path) -> PurePosixPath | None:
    for line in cgroup_file.read_text().splitlines():
        hierarchy, controllers, path = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            process_cgroup = PurePosixPath(path)
            if not process_cgroup.is_absolute():
                raise ValueError(f"process cgroup path must be absolute: {process_cgroup}")
            return process_cgroup
    return None


def _matching_cgroup_mount(
    process_cgroup: PurePosixPath, mountinfo_file: Path
) -> tuple[Path, PurePosixPath] | None:
    matching_mounts: list[tuple[int, Path, PurePosixPath]] = []
    for line in mountinfo_file.read_text().splitlines():
        mount_fields, separator, filesystem_fields = line.partition(" - ")
        if not separator:
            continue
        filesystem = filesystem_fields.split()
        if not filesystem or filesystem[0] != "cgroup2":
            continue
        _, _, _, mount_root_text, mount_point_text, *_ = mount_fields.split()
        mount_root = PurePosixPath(_decode_mountinfo_path(mount_root_text))
        try:
            relative_cgroup = process_cgroup.relative_to(mount_root)
        except ValueError:
            continue
        mount_point = Path(_decode_mountinfo_path(mount_point_text))
        matching_mounts.append((len(mount_root.parts), mount_point, relative_cgroup))

    if not matching_mounts:
        return None
    _, mount_point, relative_cgroup = max(matching_mounts, key=lambda item: item[0])
    return mount_point, relative_cgroup


def cgroup_v2_memory_available_bytes(
    *,
    cgroup_file: Path = Path("/proc/self/cgroup"),
    mountinfo_file: Path = Path("/proc/self/mountinfo"),
) -> int | None:
    """Return the remaining memory.max capacity for this process, if cgroup v2 limits it."""
    process_cgroup = _process_cgroup_path(cgroup_file)
    if process_cgroup is None:
        return None
    cgroup_mount = _matching_cgroup_mount(process_cgroup, mountinfo_file)
    if cgroup_mount is None:
        return None
    mount_point, relative_cgroup = cgroup_mount
    cgroup_directory = mount_point.joinpath(*relative_cgroup.parts)
    memory_max_path = cgroup_directory / "memory.max"
    if not memory_max_path.is_file():
        return None
    memory_max_text = memory_max_path.read_text().strip()
    if memory_max_text == "max":
        return None
    memory_max = int(memory_max_text)
    memory_current = int((cgroup_directory / "memory.current").read_text().strip())
    memory_stat_path = cgroup_directory / "memory.stat"
    if not memory_stat_path.is_file():
        raise RuntimeError(f"cgroup v2 memory.stat is missing: {memory_stat_path}")
    inactive_file_text: str | None = None
    for line in memory_stat_path.read_text().splitlines():
        name, value = line.split(maxsplit=1)
        if name == "inactive_file":
            inactive_file_text = value
            break
    if inactive_file_text is None:
        raise ValueError(f"cgroup v2 memory.stat lacks inactive_file: {memory_stat_path}")
    inactive_file = int(inactive_file_text)
    if memory_max < 0 or memory_current < 0 or inactive_file < 0:
        raise ValueError("cgroup v2 memory values must be nonnegative")
    # Count inactive_file as reclaimable; active_file is reclaimed only under pressure.
    reclaimable = min(inactive_file, memory_current)
    return max(0, memory_max - (memory_current - reclaimable))


def check_host_memory(required_bytes: int, *, available_bytes: int | None = None) -> None:
    """Refuse a run whose preallocated checkpoint inputs exceed available host memory."""
    if required_bytes <= 0:
        raise ValueError("projected checkpoint host memory must be positive")
    available = psutil.virtual_memory().available if available_bytes is None else available_bytes
    if available_bytes is None:
        cgroup_available = cgroup_v2_memory_available_bytes()
        if cgroup_available is not None:
            available = min(available, cgroup_available)
    if available < required_bytes:
        raise RuntimeError(
            f"insufficient host memory for checkpoint offload: required {required_bytes} bytes, "
            f"available {available} bytes; shorten the sequence or increase available host memory"
        )
    logger.info(
        "checkpoint offload host memory: required=%d available=%d bytes", required_bytes, available
    )


class HostSlot:
    """Register an exact-size pageable allocation, avoiding pinned allocator power-of-two rounding."""

    def __init__(self, elements: int, dtype: torch.dtype) -> None:
        """Allocate and register the longest sequence's layer input."""
        self.buffer: torch.Tensor | None = torch.empty(elements, dtype=dtype, device="cpu")
        runtime = torch.cuda.cudart()
        if runtime is None:
            raise RuntimeError("CUDA runtime is unavailable for host registration")
        status = runtime.cudaHostRegister(self.buffer.data_ptr(), self.registered_bytes, 0)
        if status != runtime.cudaError.success:
            self.buffer = None
            raise RuntimeError(f"cudaHostRegister failed for {elements} elements: {status}")

    def view_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return a view without allocating or growing buffers during training."""
        if self.buffer is None:
            raise RuntimeError("checkpoint host slot has been released")
        if hidden_states.dtype != self.buffer.dtype or hidden_states.numel() > self.buffer.numel():
            raise RuntimeError(
                "checkpoint input exceeds its dataset-sized host buffer or changes dtype"
            )
        return self.buffer[: hidden_states.numel()].view(hidden_states.shape)

    def release(self) -> None:
        """Unregister only after all queued transfers have completed."""
        if self.buffer is not None:
            torch.cuda.synchronize()
            runtime = torch.cuda.cudart()
            if runtime is None:
                raise RuntimeError("CUDA runtime is unavailable for host unregistration")
            status = runtime.cudaHostUnregister(self.buffer.data_ptr())
            if status != runtime.cudaError.success:
                raise RuntimeError(f"cudaHostUnregister failed: {status}")
            self.buffer = None

    @property
    def registered_bytes(self) -> int:
        """Return the exact host payload rather than an allocator-rounded size."""
        return 0 if self.buffer is None else self.buffer.numel() * self.buffer.element_size()


class CheckpointSlot(Protocol):
    """Storage interface shared by CUDA slots and CPU arithmetic tests."""

    def view_for(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return host storage for a layer input."""
        ...


class OffloadedLayerFunction(torch.autograd.Function):
    """Recompute decoder layers from host inputs and propagate gradients into their LoRA weights."""

    @staticmethod
    def forward(
        ctx: Any,  # noqa: ANN401 - PyTorch owns its dynamically populated autograd context.
        run_layer: Callable[..., torch.Tensor],
        hidden_states: torch.Tensor,
        slot: CheckpointSlot,
    ) -> torch.Tensor:
        """Copy the layer input and run the layer under autograd's no-grad forward."""
        if not hidden_states.requires_grad:
            raise RuntimeError(
                "offloaded checkpoint requires input gradients; enable_input_require_grads"
            )
        host_copy = slot.view_for(hidden_states)
        host_copy.copy_(hidden_states, non_blocking=True)
        ctx.run_layer, ctx.host_copy, ctx.device = run_layer, host_copy, hidden_states.device
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = (
            torch.cuda.get_rng_state(hidden_states.device) if hidden_states.is_cuda else None
        )
        ctx.autocast_enabled = torch.is_autocast_enabled(hidden_states.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(hidden_states.device.type)
        return run_layer(hidden_states)

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: Any,  # noqa: ANN401 - PyTorch owns its dynamically populated autograd context.
        grad_output: torch.Tensor,
    ) -> tuple[None, torch.Tensor, None]:
        """Replay RNG and autocast state, recompute, and backpropagate through the layer."""
        hidden_states = ctx.host_copy.to(ctx.device, non_blocking=True).detach().requires_grad_()
        ctx.host_copy = None
        devices = [ctx.device.index] if ctx.device.type == "cuda" else []
        with (
            torch.random.fork_rng(devices=devices),
            torch.enable_grad(),
            torch.autocast(ctx.device.type, enabled=ctx.autocast_enabled, dtype=ctx.autocast_dtype),
        ):
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng is not None:
                torch.cuda.set_rng_state(ctx.cuda_rng, ctx.device)
            output = ctx.run_layer(hidden_states)
            torch.autograd.backward(output, grad_output)
        if hidden_states.grad is None:
            raise RuntimeError("offloaded layer recompute produced no gradient for its input")
        return None, hidden_states.grad, None


class CheckpointOffload:
    """Own presized registered slots and install them after Trainer enables checkpointing."""

    def __init__(self, model: torch.nn.Module, max_sequence_tokens: int) -> None:
        """Check capacity before registering one longest-sequence host input per layer."""
        self.model = cast("Any", model)
        self.layers = self.model.get_base_model().base_model.layers
        dtype = self.model.get_input_embeddings().weight.dtype
        hidden_size = int(self.model.get_base_model().config.hidden_size)
        elements = max_sequence_tokens * hidden_size
        required = len(self.layers) * elements * torch.empty((), dtype=dtype).element_size()
        check_host_memory(required)
        self.slots: list[HostSlot] = []
        self.original_enable = self.model.gradient_checkpointing_enable
        self.original_checkpoint_functions: list[Any] = []
        try:
            for _ in self.layers:
                self.slots.append(HostSlot(elements, dtype))
        finally:
            if len(self.slots) != len(self.layers):
                self.close()

    def for_layer(self, layer_index: int) -> Callable[..., torch.Tensor]:
        """Bind one checkpoint call to the corresponding layer's reusable host slot."""
        slot = self.slots[layer_index]

        def checkpoint_with_offload(
            run_layer: Callable[..., torch.Tensor], hidden_states: torch.Tensor
        ) -> torch.Tensor:
            return cast(
                "torch.Tensor", OffloadedLayerFunction.apply(run_layer, hidden_states, slot)
            )

        return checkpoint_with_offload

    def enable(self, **kwargs: object) -> None:
        """Let Trainer set its checkpoint policy, then install the offload functions."""
        self.original_enable(**kwargs)
        self.original_checkpoint_functions = []
        for index, layer in enumerate(self.layers):
            if not layer.gradient_checkpointing:
                raise RuntimeError(
                    "enable gradient checkpointing before installing checkpoint offload"
                )
            self.original_checkpoint_functions.append(layer._gradient_checkpointing_func)  # noqa: SLF001
            layer._gradient_checkpointing_func = self.for_layer(index)  # noqa: SLF001

    @property
    def registered_bytes(self) -> int:
        """Return the exact preallocated host payload."""
        return sum(slot.registered_bytes for slot in self.slots)

    def close(self) -> None:
        """Restore model methods and release every registered buffer."""
        self.model.gradient_checkpointing_enable = self.original_enable
        for layer, original in zip(self.layers, self.original_checkpoint_functions, strict=False):
            layer._gradient_checkpointing_func = original  # noqa: SLF001
        for slot in self.slots:
            slot.release()


def prepare_checkpoint_offload(
    model: torch.nn.Module, max_sequence_tokens: int
) -> CheckpointOffload:
    """Preallocate dataset-sized slots and hook Trainer's future checkpoint enable call."""
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint host registration requires CUDA; CPU offload is unsupported")
    if max_sequence_tokens <= 0:
        raise ValueError("longest dataset sequence must be positive")
    offload = CheckpointOffload(model, max_sequence_tokens)
    cast("Any", model).gradient_checkpointing_enable = offload.enable
    return offload
