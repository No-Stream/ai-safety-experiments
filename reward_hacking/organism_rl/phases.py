"""Sequential vLLM sampling and LoRA optimization on one device.

The server must enable sleep mode and runtime LoRA updating, with
``VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``. Sleep level 1
may hold sampler weights in pinned host RAM; the amount depends on the loaded
model and quantization. Level 2 discards weights, freeing that RAM, and reloads
them on a slower wake. Speculative draft weights cannot be reloaded through the
base reload RPC. Trainer parameters and optimizer state move to the host before
the server wakes, and return only after sleep is confirmed. Adapter saves are
never merged into the frozen base.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import httpx2
import psutil
import torch
from safetensors import safe_open

from grpo.gpu_memory import GIBIBYTE_BYTES, gpu_memory_info, nvidia_smi_memory_info
from reward_hacking.lora_serving import prepare_vllm_serving_adapter
from reward_hacking.organism_rl.transfers import PolicyTransfers
from reward_hacking.vllm_server import RUNTIME_FREE_VRAM_FLOOR_BYTES

logger = logging.getLogger(__name__)
DISCARD_WEIGHTS_SLEEP_LEVEL = 2
DEFAULT_HOST_MEMORY_SAFETY_MARGIN_BYTES = 6 * 1024**3
CGROUP_V2_ROOT = Path("/sys/fs/cgroup")
PROCESS_CGROUP_FILE = Path("/proc/self/cgroup")
HOST_MEMORY_BUDGET_ENV = "ORGANISM_HOST_MEMORY_BUDGET_BYTES"
HOST_MEMORY_SAFETY_MARGIN_ENV = "ORGANISM_HOST_MEMORY_SAFETY_MARGIN_BYTES"
VLLM_SLEEPING_WEIGHT_BYTES_ENV = "ORGANISM_VLLM_SLEEPING_WEIGHT_BYTES"


def _canonical_config(value: object) -> object:
    """Sort config lists recursively so set-derived PEFT values have stable identities."""
    if isinstance(value, dict):
        return {
            key: _canonical_config(item) for key, item in cast("dict[str, object]", value).items()
        }
    if isinstance(value, list):
        values = [_canonical_config(item) for item in cast("list[object]", value)]
        return sorted(
            values, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"))
        )
    return value


def _adapter_digest(adapter_dir: Path) -> str:
    """Hash canonical adapter settings and each named tensor's bytes in stable order."""
    digest = hashlib.sha256()
    config_path = adapter_dir / "adapter_config.json"
    config = _canonical_config(json.loads(config_path.read_text(encoding="utf-8")))
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with safe_open(
        str(adapter_dir / "adapter_model.safetensors"), framework="pt", device="cpu"
    ) as adapter:
        for tensor_name in sorted(adapter.keys()):
            tensor = adapter.get_tensor(tensor_name).detach().contiguous()
            tensor_metadata = json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            )
            digest.update(tensor_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(tensor_metadata.encode("utf-8"))
            digest.update(b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class RolloutBackend(Protocol):
    """The model identity used by each HTTP rollout request."""

    model_id: str


class PhaseTrainer(Protocol):
    """State needed to alternate sampling and optimization."""

    model: Any
    optimizer: torch.optim.Optimizer | None
    state: Any


@dataclass(frozen=True)
class DeviceMemory:
    """Observed device capacity at startup."""

    device_name: str
    total_bytes: int
    free_bytes: int


def device_memory(device: torch.device) -> DeviceMemory:
    """Read actual capacity; CPU tests have no invented GPU memory allowance."""
    if device.type != "cuda":
        return DeviceMemory(str(device), 0, 0)
    free_bytes, total_bytes = gpu_memory_info(torch, device)
    memory = DeviceMemory(torch.cuda.get_device_name(device), total_bytes, free_bytes)
    logger.info(
        "training device=%s total_bytes=%d free_bytes=%d",
        memory.device_name,
        memory.total_bytes,
        memory.free_bytes,
    )
    return memory


def _unified_cgroup_path(process_cgroup_file: Path) -> str | None:
    """Read the process's unified cgroup membership path."""
    unified_path: str | None = None
    for line in process_cgroup_file.read_text(encoding="ascii").splitlines():
        hierarchy_id, controllers, cgroup_path = line.split(":", maxsplit=2)
        if hierarchy_id == "0" and not controllers:
            if unified_path is not None:
                raise ValueError(f"{process_cgroup_file}: duplicate cgroup v2 membership row")
            unified_path = cgroup_path
    return unified_path


def _process_cgroup_directory(
    unified_path: str, cgroup_root: Path, process_cgroup_file: Path
) -> Path:
    """Resolve membership under the mounted hierarchy, including namespace-root mounts."""
    relative_cgroup_path = PurePosixPath(unified_path)
    if not relative_cgroup_path.is_absolute() or ".." in relative_cgroup_path.parts:
        raise ValueError(f"{process_cgroup_file}: invalid cgroup v2 path")
    cgroup_mount = cgroup_root.resolve(strict=True)
    try:
        cgroup_directory = cgroup_mount.joinpath(*relative_cgroup_path.parts[1:]).resolve(
            strict=True
        )
    except FileNotFoundError:
        cgroup_directory = cgroup_mount
    if not cgroup_directory.is_relative_to(cgroup_mount):
        raise ValueError(f"{process_cgroup_file}: cgroup v2 path escapes its mount")
    return cgroup_directory


def _cgroup_memory_state(cgroup_directory: Path) -> tuple[int, int] | None:
    """Read a finite limit and current use, excluding reclaimable page cache, from one cgroup directory."""
    memory_max_path = cgroup_directory / "memory.max"
    if not memory_max_path.is_file():
        return None
    maximum_text = memory_max_path.read_text(encoding="ascii").strip()
    if maximum_text == "max":
        return None
    maximum_bytes = _parse_cgroup_bytes(maximum_text, memory_max_path, allow_zero=False)
    current_path = cgroup_directory / "memory.current"
    current_bytes = _parse_cgroup_bytes(
        current_path.read_text(encoding="ascii").strip(), current_path, allow_zero=True
    )
    stat_path = cgroup_directory / "memory.stat"
    stat = dict(line.split() for line in stat_path.read_text(encoding="ascii").splitlines())
    reclaimable_file_bytes = sum(
        _parse_cgroup_bytes(stat[key], stat_path, allow_zero=True)
        for key in ("active_file", "inactive_file")
    )
    return maximum_bytes, max(0, current_bytes - reclaimable_file_bytes)


def _cgroup_memory_states(
    cgroup_root: Path = CGROUP_V2_ROOT,
    process_cgroup_file: Path = PROCESS_CGROUP_FILE,
) -> tuple[tuple[int, int], ...]:
    """Read finite memory limits and current use from this process's cgroup ancestors."""
    if not process_cgroup_file.is_file() or not cgroup_root.is_dir():
        return ()
    unified_path = _unified_cgroup_path(process_cgroup_file)
    if unified_path is None:
        return ()
    cgroup_mount = cgroup_root.resolve(strict=True)
    cgroup_directory = _process_cgroup_directory(unified_path, cgroup_mount, process_cgroup_file)
    states: list[tuple[int, int]] = []
    while True:
        state = _cgroup_memory_state(cgroup_directory)
        if state is not None:
            states.append(state)
        if cgroup_directory == cgroup_mount:
            break
        cgroup_directory = cgroup_directory.parent
    return tuple(states)


def _parse_cgroup_bytes(value: str, path: Path, *, allow_zero: bool) -> int:
    """Parse one nonnegative decimal cgroup byte count."""
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{path}: malformed cgroup byte count")
    byte_count = int(value)
    if byte_count == 0 and not allow_zero:
        raise ValueError(f"{path}: cgroup byte count must be positive")
    return byte_count


def _configured_bytes(value: int | None, environment_name: str, *, allow_zero: bool) -> int | None:
    """Resolve one explicit byte setting or its environment override."""
    environment_value = os.environ.get(environment_name)
    if value is not None and environment_value is not None:
        raise ValueError(
            f"Specify {environment_name} either as an argument or environment variable"
        )
    configured_value: int | str | None = value if value is not None else environment_value
    if configured_value is None:
        return None
    if isinstance(configured_value, str):
        if not configured_value.isascii() or not configured_value.isdecimal():
            raise ValueError(f"{environment_name} must be a nonnegative integer byte count")
        configured_bytes = int(configured_value)
    else:
        if isinstance(configured_value, bool):
            raise TypeError(f"{environment_name} must be an integer byte count")
        configured_bytes = configured_value
    if configured_bytes < 0 or (configured_bytes == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{environment_name} must be a {qualifier} byte count")
    return configured_bytes


def resolve_host_memory_budget_bytes(
    host_memory_budget_bytes: int | None = None,
    *,
    host_memory_safety_margin_bytes: int = DEFAULT_HOST_MEMORY_SAFETY_MARGIN_BYTES,
    cgroup_root: Path = CGROUP_V2_ROOT,
    process_cgroup_file: Path = PROCESS_CGROUP_FILE,
    physical_memory_bytes: int | None = None,
) -> int:
    """Return the smallest physical, cgroup, or configured limit after reserving the margin."""
    if host_memory_safety_margin_bytes < 0:
        raise ValueError("host_memory_safety_margin_bytes must be nonnegative")
    physical_limit_bytes = (
        psutil.virtual_memory().total if physical_memory_bytes is None else physical_memory_bytes
    )
    if physical_limit_bytes <= 0:
        raise ValueError("physical_memory_bytes must be positive")
    limits = [physical_limit_bytes]
    if host_memory_budget_bytes is not None:
        if host_memory_budget_bytes <= 0:
            raise ValueError("host_memory_budget_bytes must be positive")
        limits.append(host_memory_budget_bytes)
    limits.extend(limit for limit, _ in _cgroup_memory_states(cgroup_root, process_cgroup_file))
    budget_bytes = min(limits) - host_memory_safety_margin_bytes
    if budget_bytes <= 0:
        raise ValueError("host memory safety margin must be smaller than the resolved memory limit")
    return budget_bytes


def check_host_memory_budget(  # noqa: PLR0913
    *,
    required_bytes: int,
    current_memory_bytes: int,
    host_memory_budget_bytes: int,
    host_available_bytes: int,
    safety_margin_bytes: int,
    phase: str,
) -> None:
    """Refuse a phase allocation that crosses its cgroup ceiling or host availability margin."""
    if (
        min(
            required_bytes,
            current_memory_bytes,
            host_memory_budget_bytes,
            host_available_bytes,
            safety_margin_bytes,
        )
        < 0
    ):
        raise ValueError("host memory byte counts must be nonnegative")
    cgroup_headroom_bytes = host_memory_budget_bytes - current_memory_bytes - safety_margin_bytes
    host_headroom_bytes = host_available_bytes - safety_margin_bytes
    headroom_bytes = max(0, min(cgroup_headroom_bytes, host_headroom_bytes))
    if required_bytes > headroom_bytes:
        raise RuntimeError(
            f"Host memory budget exceeded before {phase}: required_bytes={required_bytes} "
            f"current_memory_bytes={current_memory_bytes} budget_bytes={host_memory_budget_bytes} "
            f"host_available_bytes={host_available_bytes} safety_margin_bytes={safety_margin_bytes} "
            f"headroom_bytes={max(0, headroom_bytes)}"
        )


_MODEL_LOADING_PATTERN = re.compile(r"Model loading took ([0-9.]+) GiB memory")


def vllm_sleeping_weight_bytes(server_log: Path) -> int:
    """Read the loaded weight size, which level-1 sleep backs up to pinned host memory."""
    matches = _MODEL_LOADING_PATTERN.findall(
        server_log.read_text(encoding="utf-8", errors="replace")
    )
    if not matches:
        raise RuntimeError(f"no 'Model loading took' line in {server_log}")
    # A relaunch in the same output dir appends to the log; the ready server is the latest load.
    return int(float(matches[-1]) * 2**30)


def check_pinned_and_sleeping_weight_budget(
    *,
    pinned_bytes: int,
    sleeping_weight_bytes: int,
    host_memory_budget_bytes: int,
    phase: str,
) -> None:
    """Refuse a combined trainer and sleeping-sampler footprint above the resolved budget."""
    if min(pinned_bytes, sleeping_weight_bytes, host_memory_budget_bytes) < 0:
        raise ValueError("host memory byte counts must be nonnegative")
    required_bytes = pinned_bytes + sleeping_weight_bytes
    if required_bytes > host_memory_budget_bytes:
        raise RuntimeError(
            f"Host memory budget exceeded before {phase}: pinned_bytes={pinned_bytes} "
            f"sleeping_weight_bytes={sleeping_weight_bytes} required_bytes={required_bytes} "
            f"budget_bytes={host_memory_budget_bytes}"
        )


def _move_state(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_state(item, device)
            for key, item in cast("dict[object, object]", value).items()
        }
    if isinstance(value, list):
        return [_move_state(item, device) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(_move_state(item, device) for item in cast("tuple[object, ...]", value))
    return value


def move_optimizer_state(optimizer: torch.optim.Optimizer | None, device: torch.device) -> None:
    """Keep the optimizer attached to the same parameters while moving its buffers."""
    if optimizer is None:
        return
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = cast("dict[str, Any]", _move_state(state, device))


class VLLMPhaseController:
    """Move training state off-device before waking the external sampler."""

    def __init__(  # noqa: PLR0913
        self,
        base_url: str,
        backend: RolloutBackend,
        adapter_root: Path,
        *,
        training_device: torch.device,
        adapter_prefix: str = "organism",
        timeout_seconds: float = 1800.0,
        sleep_level: int = 1,
        host_memory_budget_bytes: int | None = None,
        host_memory_safety_margin_bytes: int | None = None,
        sleeping_weight_bytes: int | None = None,
        frozen_host_memory: str = "pinned",
        http_transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Bind one rollout backend and its private adapter staging directory."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not adapter_prefix or "/" in adapter_prefix:
            raise ValueError("adapter_prefix must be a nonempty model name without slashes")
        if sleep_level not in (1, 2):
            raise ValueError("sleep_level must be 1 or 2")
        configured_budget_bytes = _configured_bytes(
            host_memory_budget_bytes, HOST_MEMORY_BUDGET_ENV, allow_zero=False
        )
        configured_margin_bytes = _configured_bytes(
            host_memory_safety_margin_bytes, HOST_MEMORY_SAFETY_MARGIN_ENV, allow_zero=True
        )
        configured_sleeping_weight_bytes = _configured_bytes(
            sleeping_weight_bytes, VLLM_SLEEPING_WEIGHT_BYTES_ENV, allow_zero=False
        )
        safety_margin_bytes = (
            DEFAULT_HOST_MEMORY_SAFETY_MARGIN_BYTES
            if configured_margin_bytes is None
            else configured_margin_bytes
        )
        physical_memory_bytes = psutil.virtual_memory().total
        self.host_memory_budget_bytes = resolve_host_memory_budget_bytes(
            configured_budget_bytes,
            host_memory_safety_margin_bytes=safety_margin_bytes,
            physical_memory_bytes=physical_memory_bytes,
        )
        if (
            training_device.type == "cuda"
            and sleep_level == 1
            and configured_sleeping_weight_bytes is None
        ):
            raise RuntimeError(
                f"CUDA sleep level 1 requires {VLLM_SLEEPING_WEIGHT_BYTES_ENV}; "
                "the installed vLLM RPC does not report pinned weight bytes"
            )
        if frozen_host_memory not in ("pinned", "pageable", "persistent"):
            raise ValueError("frozen_host_memory must be pinned, pageable or persistent")
        self.frozen_host_memory = frozen_host_memory
        self.sleep_level = sleep_level
        self._configured_host_memory_limit_bytes = configured_budget_bytes
        self._physical_memory_limit_bytes = physical_memory_bytes
        self.host_memory_safety_margin_bytes = safety_margin_bytes
        self.sleeping_weight_bytes = configured_sleeping_weight_bytes or 0
        self._weights_discarded = False
        self._vllm_sleeping = False
        self._vllm_backup_resident = False
        self.backend = backend
        self.adapter_root = adapter_root
        self.training_device = training_device
        self.adapter_prefix = adapter_prefix
        self.memory = device_memory(training_device)
        self._client = httpx2.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            trust_env=False,
            transport=http_transport,
        )
        self._adapter_name: str | None = None
        self._served_adapter_path: Path | None = None
        self._training = False
        self.last_sync_seconds: dict[str, float] = {}
        self._transfers: PolicyTransfers | None = None
        self.runtime_memory_metrics_path = adapter_root.parent / "runtime-vram.jsonl"

    def close(self) -> None:
        """Release the owned HTTP client and retained trainer host arenas."""
        self._client.close()
        if self._transfers is not None:
            self._transfers.close()

    def _post(self, path: str, payload: dict[str, object] | None = None) -> None:
        self._client.post(path, json=payload).raise_for_status()

    def _assert_sleeping(self, *, expected: bool) -> None:
        response = self._client.get("/is_sleeping")
        response.raise_for_status()
        value = cast("dict[str, object]", response.json())
        if value["is_sleeping"] is not expected:
            raise RuntimeError(f"vLLM sleep state differs from expected {expected}")

    def _assert_served(self) -> None:
        if self._adapter_name is None or self.backend.model_id != self._adapter_name:
            raise RuntimeError("rollout backend does not select the synchronized adapter")
        if self._adapter_name not in self._served_model_ids():
            raise RuntimeError(
                f"synchronized adapter {self._adapter_name!r} is not present in served models"
            )

    def _served_model_ids(self) -> set[str]:
        response = self._client.get("/v1/models")
        response.raise_for_status()
        payload = cast("dict[str, list[dict[str, object]]]", response.json())
        return {cast("str", entry["id"]) for entry in payload["data"]}

    def _offload(self, trainer: PhaseTrainer) -> None:
        if self._transfers is None:
            self._transfers = PolicyTransfers(
                trainer.model,
                self.training_device,
                pin_memory=self.frozen_host_memory != "pageable",
                persistent=self.frozen_host_memory == "persistent",
            )
        planned_host_bytes = self._transfers.planned_host_bytes(trainer.optimizer)
        sleeping_weight_bytes = (
            self.sleeping_weight_bytes
            if (self._vllm_sleeping or self.frozen_host_memory == "persistent")
            and self.sleep_level != DISCARD_WEIGHTS_SLEEP_LEVEL
            else 0
        )
        check_pinned_and_sleeping_weight_budget(
            pinned_bytes=planned_host_bytes,
            sleeping_weight_bytes=sleeping_weight_bytes,
            host_memory_budget_bytes=self.host_memory_budget_bytes,
            phase="offload",
        )
        additional_host_bytes = max(0, planned_host_bytes - self._transfers.host_bytes)
        self._check_host_memory_budget(additional_host_bytes, "offload")
        self._transfers.offload(trainer.optimizer)
        gc.collect()
        if self.training_device.type == "cuda":
            torch.cuda.empty_cache()
            device_index = cast("int | None", self.training_device.index)
            if device_index is None:
                # Torch's stub says int, but an unindexed CUDA device has no explicit index.
                device_index = torch.cuda.current_device()
            nvidia_memory = nvidia_smi_memory_info(device_index)
            logger.info(
                "post-offload CUDA memory step=%d device=%d allocated_bytes=%d reserved_bytes=%d "
                "nvidia_smi_free_bytes=%d nvidia_smi_free_gib=%.2f nvidia_smi_total_bytes=%d",
                int(trainer.state.global_step),
                device_index,
                torch.cuda.memory_allocated(self.training_device),
                torch.cuda.memory_reserved(self.training_device),
                nvidia_memory.free_bytes,
                nvidia_memory.free_bytes / GIBIBYTE_BYTES,
                nvidia_memory.total_bytes,
            )

    def _check_host_memory_budget(self, required_bytes: int, phase: str) -> None:
        memory = psutil.virtual_memory()
        cgroup_states = _cgroup_memory_states()
        if cgroup_states:
            for cgroup_limit_bytes, current_memory_bytes in cgroup_states:
                check_host_memory_budget(
                    required_bytes=required_bytes,
                    current_memory_bytes=current_memory_bytes,
                    host_memory_budget_bytes=min(
                        cgroup_limit_bytes,
                        self._physical_memory_limit_bytes,
                        self._configured_host_memory_limit_bytes or cgroup_limit_bytes,
                    ),
                    host_available_bytes=memory.available,
                    safety_margin_bytes=self.host_memory_safety_margin_bytes,
                    phase=phase,
                )
            return
        current_memory_bytes = max(0, memory.total - memory.available)
        check_host_memory_budget(
            required_bytes=required_bytes,
            current_memory_bytes=current_memory_bytes,
            host_memory_budget_bytes=min(
                memory.total, self._configured_host_memory_limit_bytes or memory.total
            ),
            host_available_bytes=memory.available,
            safety_margin_bytes=self.host_memory_safety_margin_bytes,
            phase=phase,
        )

    def initialize(self, trainer: PhaseTrainer, step: int = 0) -> None:
        """Publish the initial or resumed adapter before the first rollout."""
        if self._adapter_name is not None:
            raise RuntimeError("phase controller is already initialized")
        self._sync(trainer, step)

    def _wake_up(
        self,
        *,
        restore_adapter: bool = True,
        wake_source: Literal["sampling", "adapter_sync"],
        step: int,
    ) -> None:
        self._post("/wake_up")
        self._vllm_sleeping = False
        if self._weights_discarded:
            self._post("/collective_rpc", {"method": "reload_weights"})
            self._weights_discarded = False
            if restore_adapter and self._adapter_name is not None:
                if self._served_adapter_path is None:
                    raise RuntimeError("synchronized adapter has no serving path")
                self._post("/v1/unload_lora_adapter", {"lora_name": self._adapter_name})
                self._post(
                    "/v1/load_lora_adapter",
                    {"lora_name": self._adapter_name, "lora_path": str(self._served_adapter_path)},
                )
        self._record_wake_memory(wake_source, step)

    def _record_wake_memory(
        self, wake_source: Literal["sampling", "adapter_sync"], step: int
    ) -> None:
        if self.training_device.type != "cuda":
            return
        device_index = cast("int | None", self.training_device.index)
        if device_index is None:
            device_index = torch.cuda.current_device()
        memory = nvidia_smi_memory_info(device_index)
        below_floor = memory.free_bytes < RUNTIME_FREE_VRAM_FLOOR_BYTES
        record: dict[str, int | str | bool] = {
            "schema_version": 1,
            "measured_at_unix_ns": time.time_ns(),
            "wake_source": wake_source,
            "step": step,
            "device_index": device_index,
            "free_bytes": memory.free_bytes,
            "total_bytes": memory.total_bytes,
            "runtime_free_vram_floor_bytes": RUNTIME_FREE_VRAM_FLOOR_BYTES,
            "below_floor": below_floor,
        }
        with self.runtime_memory_metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(record, sort_keys=True) + "\n")
        if below_floor:
            logger.warning(
                "free VRAM after sampler wake is below the runtime floor: "
                "source=%s device=%d free_bytes=%d floor_bytes=%d",
                wake_source,
                device_index,
                memory.free_bytes,
                RUNTIME_FREE_VRAM_FLOOR_BYTES,
            )
        else:
            logger.info(
                "free VRAM after sampler wake: source=%s device=%d free_bytes=%d floor_bytes=%d",
                wake_source,
                device_index,
                memory.free_bytes,
                RUNTIME_FREE_VRAM_FLOOR_BYTES,
            )

    def before_sampling(self, trainer: PhaseTrainer) -> None:
        """Release trainer memory, wake sampling and validate the adapter identity."""
        self._offload(trainer)
        self._wake_up(wake_source="sampling", step=int(trainer.state.global_step))
        self._assert_sleeping(expected=False)
        self._assert_served()
        self._training = False

    def sleep(self) -> None:
        """Free server GPU allocations and verify the acknowledged sleep state."""
        # vLLM's pinned weight backup stays allocated in its process after the first level-1 sleep.
        expected_host_bytes = (
            self.sleeping_weight_bytes
            if self.sleep_level != DISCARD_WEIGHTS_SLEEP_LEVEL
            and not self._vllm_sleeping
            and not self._vllm_backup_resident
            else 0
        )
        pinned_host_bytes = self._transfers.host_bytes if self._transfers is not None else 0
        check_pinned_and_sleeping_weight_budget(
            pinned_bytes=pinned_host_bytes,
            sleeping_weight_bytes=(
                self.sleeping_weight_bytes if self.sleep_level != DISCARD_WEIGHTS_SLEEP_LEVEL else 0
            ),
            host_memory_budget_bytes=self.host_memory_budget_bytes,
            phase="vLLM sleep",
        )
        self._check_host_memory_budget(expected_host_bytes, "vLLM sleep")
        self._post(f"/sleep?level={self.sleep_level}")
        self._assert_sleeping(expected=True)
        self._weights_discarded = self.sleep_level == DISCARD_WEIGHTS_SLEEP_LEVEL
        self._vllm_sleeping = True
        self._vllm_backup_resident = self.sleep_level != DISCARD_WEIGHTS_SLEEP_LEVEL
        logger.info(
            "vLLM sleeping: level=%d host_available_bytes=%d",
            self.sleep_level,
            psutil.virtual_memory().available,
        )

    def before_training(self, trainer: PhaseTrainer) -> None:
        """Confirm server sleep before restoring trainer parameters and state."""
        self._assert_served()
        self.sleep()
        if self._transfers is None:
            raise RuntimeError("phase controller must initialize before training")
        self._transfers.restore(trainer.optimizer)
        self._training = True

    def after_optimizer_step(self, trainer: PhaseTrainer) -> None:
        """Save and hot-load an independent adapter for the completed step."""
        if not self._training:
            raise RuntimeError("optimizer sync requires the training phase")
        self._sync(trainer, int(trainer.state.global_step))
        self._training = False

    def _sync(self, trainer: PhaseTrainer, step: int) -> None:
        if step < 0:
            raise ValueError("adapter step must be nonnegative")
        staging_name = f"step-{step:08d}-{uuid4().hex}"
        adapter_dir = self.adapter_root / "training" / staging_name
        started = time.perf_counter()
        trainer.model.save_pretrained(str(adapter_dir))
        saved = time.perf_counter()
        config_path = adapter_dir / "adapter_config.json"
        canonical_config = _canonical_config(json.loads(config_path.read_text(encoding="utf-8")))
        config_path.write_text(
            json.dumps(canonical_config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        name = f"{self.adapter_prefix}-step-{step:08d}-{_adapter_digest(adapter_dir)}"
        digested = time.perf_counter()
        served_dir = prepare_vllm_serving_adapter(adapter_dir, self.adapter_root / "serving" / name)
        copied = time.perf_counter()
        self._offload(trainer)
        offloaded = time.perf_counter()
        self._wake_up(restore_adapter=False, wake_source="adapter_sync", step=step)
        self._assert_sleeping(expected=False)
        woken = time.perf_counter()
        served_model_ids = self._served_model_ids()
        stale_adapters = {
            model_id
            for model_id in served_model_ids
            if model_id.startswith(f"{self.adapter_prefix}-step-")
        }
        if self._adapter_name is not None and self._adapter_name in served_model_ids:
            stale_adapters.add(self._adapter_name)
        if name in served_model_ids:
            stale_adapters.add(name)
        for adapter_name in sorted(stale_adapters):
            self._post("/v1/unload_lora_adapter", {"lora_name": adapter_name})
        self._post(
            "/v1/load_lora_adapter",
            {"lora_name": name, "lora_path": str(served_dir.resolve())},
        )
        self._served_adapter_path = served_dir.resolve()
        self._adapter_name = name
        self.backend.model_id = name
        self._assert_served()
        self._remove_unserved_adapter_dirs(keep={adapter_dir, served_dir})
        loaded = time.perf_counter()
        self.last_sync_seconds = {
            "save": saved - started,
            "digest": digested - saved,
            "serving_copy": copied - digested,
            "offload": offloaded - copied,
            "wake": woken - offloaded,
            "lora_load_http": loaded - woken,
        }
        logger.info("synchronized LoRA step=%d model=%s", step, name)

    def _remove_unserved_adapter_dirs(self, *, keep: set[Path]) -> None:
        # vLLM has unloaded every earlier adapter by now, and retained checkpoints hold the
        # adapters worth keeping, so per-step copies are pure duplicates (~330 MB per 9B step).
        kept = {path.resolve() for path in keep}
        for subdirectory in ("training", "serving"):
            for path in sorted((self.adapter_root / subdirectory).glob("*")):
                if path.is_dir() and not path.is_symlink() and path.resolve() not in kept:
                    shutil.rmtree(path)
