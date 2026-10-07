"""Sequential vLLM sampling and LoRA optimization on one device.

The server must enable sleep mode and runtime LoRA updating, with
``VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=1``. Sleep level 1
preserves the server's frozen base on the host. Trainer parameters and optimizer
state move to the host before the server wakes, and return only after sleep is
confirmed. Adapter saves are never merged into the frozen base.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast
from uuid import uuid4

import httpx2
import torch
from safetensors import safe_open

from reward_hacking.lora_serving import prepare_vllm_serving_adapter

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    memory = DeviceMemory(torch.cuda.get_device_name(device), total_bytes, free_bytes)
    logger.info(
        "training device=%s total_bytes=%d free_bytes=%d",
        memory.device_name,
        memory.total_bytes,
        memory.free_bytes,
    )
    return memory


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
        http_transport: httpx2.BaseTransport | None = None,
    ) -> None:
        """Bind one rollout backend and its private adapter staging directory."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not adapter_prefix or "/" in adapter_prefix:
            raise ValueError("adapter_prefix must be a nonempty model name without slashes")
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
        self._training = False

    def close(self) -> None:
        """Release the owned HTTP client."""
        self._client.close()

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
        trainer.model.to(torch.device("cpu"))
        move_optimizer_state(trainer.optimizer, torch.device("cpu"))
        gc.collect()
        if self.training_device.type == "cuda":
            torch.cuda.synchronize(self.training_device)
            torch.cuda.empty_cache()

    def initialize(self, trainer: PhaseTrainer, step: int = 0) -> None:
        """Publish the initial or resumed adapter before the first rollout."""
        if self._adapter_name is not None:
            raise RuntimeError("phase controller is already initialized")
        self._sync(trainer, step)

    def before_sampling(self, trainer: PhaseTrainer) -> None:
        """Release trainer memory, wake sampling and validate the adapter identity."""
        self._offload(trainer)
        self._post("/wake_up")
        self._assert_sleeping(expected=False)
        self._assert_served()
        self._training = False

    def sleep(self) -> None:
        """Free server GPU allocations and verify the acknowledged sleep state."""
        self._post("/sleep?level=1")
        self._assert_sleeping(expected=True)

    def before_training(self, trainer: PhaseTrainer) -> None:
        """Confirm server sleep before restoring trainer parameters and state."""
        self._assert_served()
        self.sleep()
        trainer.model.to(self.training_device)
        move_optimizer_state(trainer.optimizer, self.training_device)
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
        trainer.model.save_pretrained(str(adapter_dir))
        config_path = adapter_dir / "adapter_config.json"
        canonical_config = _canonical_config(json.loads(config_path.read_text(encoding="utf-8")))
        config_path.write_text(
            json.dumps(canonical_config, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        name = f"{self.adapter_prefix}-step-{step:08d}-{_adapter_digest(adapter_dir)}"
        served_dir = prepare_vllm_serving_adapter(adapter_dir, self.adapter_root / "serving" / name)
        self._offload(trainer)
        self._post("/wake_up")
        self._assert_sleeping(expected=False)
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
        self._adapter_name = name
        self.backend.model_id = name
        self._assert_served()
        logger.info("synchronized LoRA step=%d model=%s", step, name)
