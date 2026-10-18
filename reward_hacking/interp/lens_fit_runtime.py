"""Shared runtime for fitting, checking and recording Jacobian lenses.

The functions here know about Hugging Face causal LMs and ``jlens`` but not about a particular
corpus, research wave or artifact schema. Callers supply their prompt sets and identity payloads;
this module keeps loading, adapter application, restart safety, serialization and reconstruction
reads consistent across those callers.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import subprocess
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors import safe_open
from torch import nn

from games.deltanet_kernels import bound_deltanet_kernels, prefill_deltanet_kernels
from games.lora import assert_adapter_applies
from games.tokenizer_identity import tokenizer_content_sha256
from grpo.gpu_memory import gpu_memory_info
from reward_hacking.interp.directions import load_model_and_tokenizer
from reward_hacking.interp.jacobian import (
    JLENS_COMMIT,
    FitQualityReport,
    JacobianConfig,
    _resolve_fit_layers,  # pyright: ignore[reportPrivateUsage] - shared fit-layer contract
    evaluate_reconstruction,
    fit_lens,
    fit_quality_payload,
    fit_skip_first,
    verify_lens_roundtrip,
)
from reward_hacking.interp.lens_deltanet_gates import (
    DEFAULT_POSITION_GAP,
    KERNEL_AUDIT_DIMS,
    KERNEL_AUDIT_TOLERANCE_BF16,
    RowEstimator,
    audit_chunk_kernel_backward,
    probe_recurrence_autograd,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from types import ModuleType

logger = logging.getLogger(__name__)

LENS_LOAD_PATH = (
    "reward_hacking.interp.directions.load_model_and_tokenizer(dtype=bfloat16, device=cuda); "
    "optional PEFT adapter; jlens.from_hf"
)
"""Names the generic loader in fit provenance; changing it changes the fitted residual tree."""

FIT_IDENTITY_FILENAME = "fit-identity.json"
PARAMETER_HASH_CHUNK_ELEMENTS = 4_194_304
MAX_IMPLEMENTATION_SOURCE_FILES = 32


class FitIdentityError(RuntimeError):
    """An existing partial fit cannot be proven compatible with this invocation."""


class _JlensCompatibilityModel:
    """Small deterministic CPU model for exercising jlens fit and resume behavior."""

    def __init__(self) -> None:
        self.n_layers = 3
        self.d_model = 8
        self.embed_tokens = nn.Embedding(32, self.d_model)
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh()) for _ in range(3)]
        )
        self.lm_head = nn.Linear(self.d_model, 32, bias=False)

    @property
    def tokenizer(self) -> object:
        return self

    @property
    def input_device(self) -> torch.device:
        return self.embed_tokens.weight.device

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 32,
    ) -> SimpleNamespace:
        del return_tensors, truncation
        token_ids = [0, *(1 + byte % 31 for byte in text.encode()[: max_length - 1])]
        return SimpleNamespace(input_ids=torch.tensor([token_ids]))

    def encode(self, text: str, *, max_length: int = 32) -> torch.Tensor:
        return self(text, max_length=max_length).input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(residual.float())


def verify_jlens_fit_behavior(jl: ModuleType) -> None:
    """Check real jlens Jacobian fitting and checkpoint continuation on a tiny CPU model."""

    def model_with_stable_weights() -> _JlensCompatibilityModel:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(193)
            return _JlensCompatibilityModel()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(193)
        prompts = tuple(
            f"synthetic lens compatibility sentence {index} contains repeated context " * 2
            for index in range(3)
        )
        fit_arguments: dict[str, object] = {
            "source_layers": [0, 1],
            "target_layer": 2,
            "dim_batch": 1,
            "max_seq_len": 32,
            "skip_first": 0,
        }
        straight = cast("Any", jl).fit(model_with_stable_weights(), prompts, **fit_arguments)
        with tempfile.TemporaryDirectory(prefix="jlens-fit-compatibility-") as directory:
            checkpoint_path = Path(directory) / "fit-checkpoint.pt"
            cast("Any", jl).fit(
                model_with_stable_weights(),
                prompts[:1],
                **fit_arguments,
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=1,
                resume=False,
            )
            resumed = cast("Any", jl).fit(
                model_with_stable_weights(),
                prompts,
                **fit_arguments,
                checkpoint_path=str(checkpoint_path),
                checkpoint_every=1,
                resume=True,
            )

        estimator_model = model_with_stable_weights()
        direct_jacobians = {layer: torch.zeros(8, 8) for layer in (0, 1)}
        for prompt in prompts:
            per_prompt, seq_len, n_valid_positions = cast("Any", jl).jacobian_for_prompt(
                estimator_model,
                prompt,
                [0, 1],
                target_layer=2,
                dim_batch=1,
                max_seq_len=32,
                skip_first=0,
            )
            if seq_len <= 0 or n_valid_positions <= 0:
                raise RuntimeError("jlens per-prompt estimator returned no valid token positions")
            if set(per_prompt) != {0, 1}:
                raise RuntimeError("jlens per-prompt estimator returned unexpected source layers")
            for layer in direct_jacobians:
                direct_jacobians[layer] += per_prompt[layer] / len(prompts)

    if straight.n_prompts != len(prompts) or resumed.n_prompts != len(prompts):
        raise RuntimeError("jlens fit compatibility check did not account for every prompt")
    if list(straight.source_layers) != [0, 1] or list(resumed.source_layers) != [0, 1]:
        raise RuntimeError("jlens fit compatibility check returned unexpected source layers")
    for layer in straight.source_layers:
        expected = straight.jacobians[layer]
        actual = resumed.jacobians[layer]
        if (
            expected.shape != (8, 8)
            or not torch.isfinite(expected).all()
            or not torch.count_nonzero(expected)
        ):
            raise RuntimeError(
                f"jlens fit compatibility check produced an unusable Jacobian at layer {layer}"
            )
        torch.testing.assert_close(
            actual,
            expected,
            msg=f"jlens resumed Jacobian differs from a straight fit at layer {layer}",
        )
        torch.testing.assert_close(
            expected,
            direct_jacobians[layer],
            msg=f"jlens fit disagrees with its per-prompt estimator at layer {layer}",
        )


@dataclass(frozen=True)
class FitModelHandle:
    """A frozen HF model, its tokenizer and the ``jlens`` view over the active weights."""

    model: object
    hf_model: nn.Module
    tokenizer: Any
    model_id: str
    adapter_path: Path | None
    weights_fingerprint: str
    device_facts: dict[str, object]
    adapter_declared_base: str | None
    adapter_config_sha256: str | None
    applied_adapter_weights: int
    adapted_module_count: int

    @property
    def blocks(self) -> nn.ModuleList:
        """Return the residual blocks used by ``jlens`` and the architecture gates."""
        return cast("nn.ModuleList", cast("Any", self.model).layers)

    def forward(self, input_ids: torch.Tensor) -> object:
        """Run the text stack in the same way as the fit's forward pass."""
        return cast("Any", self.model).forward(input_ids)

    def as_payload(self) -> dict[str, object]:
        """Return public fit provenance without exposing an absolute local path."""
        return {
            "model_id": self.model_id,
            "adapter": None if self.adapter_path is None else self.adapter_path.name,
            "adapter_declared_base": self.adapter_declared_base,
            "adapter_config_sha256": self.adapter_config_sha256,
            "applied_adapter_weights": self.applied_adapter_weights,
            "adapted_module_count": self.adapted_module_count,
            "adapter_targets": "modules with weights in the adapter file",
            "weights_fingerprint": self.weights_fingerprint,
            "model_revision": vars(cast("Any", self.hf_model).config).get("_commit_hash"),
            "tokenizer_sha256": tokenizer_content_sha256(self.tokenizer),
            "load_path": LENS_LOAD_PATH,
            "hf_class": type(self.hf_model).__name__,
            "n_layers": int(cast("Any", self.model).n_layers),
            "d_model": int(cast("Any", self.model).d_model),
            **self.device_facts,
        }


@dataclass(frozen=True)
class FitRunResult:
    """The shared outputs from a fit, its saved artifact and its reconstruction read."""

    lens: object
    fit_seconds: float
    persistence_seconds: float
    read_seconds: float
    lens_sha256: str
    quality: FitQualityReport | None


def jlens_provenance(jl: ModuleType) -> dict[str, object]:
    """Record the loaded ``jlens`` checkout as provenance."""
    package_dir = Path(cast("str", jl.__file__)).resolve().parent
    finished = subprocess.run(  # noqa: S603 - fixed argv over the resolved jlens path
        ["git", "-C", str(package_dir), "rev-parse", "HEAD"],  # noqa: S607 - git on PATH, no user input
        capture_output=True,
        text=True,
        check=False,
    )
    commit = finished.stdout.strip() if finished.returncode == 0 else None
    if commit is None:
        logger.warning(
            "jlens at %s is not a Git checkout; its commit cannot be verified", package_dir
        )
    dirty: bool | None = None
    if commit is not None:
        status = subprocess.run(  # noqa: S603 - fixed argv over the resolved jlens path
            ["git", "-C", str(package_dir), "status", "--porcelain", "--untracked-files=all"],  # noqa: S607 - git on PATH, no user input
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode == 0:
            dirty = bool(status.stdout.strip())
        else:
            logger.warning("could not record jlens working-tree state at %s", package_dir)
        if dirty:
            logger.warning("jlens at %s has uncommitted source changes", package_dir)
    return {
        "path": str(package_dir),
        "commit": commit,
        "pinned_commit": JLENS_COMMIT,
        "dirty": dirty,
    }


def device_facts(device: str | torch.device | None = None) -> dict[str, object]:
    """Capture facts for the requested device without probing CUDA for explicit CPU runs."""
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cpu":
        return {"device": "cpu", "free_vram_bytes": 0, "total_vram_bytes": 0}
    if resolved_device.type != "cuda":
        raise ValueError(f"lens fitting supports one CUDA device or CPU, got {resolved_device}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested lens device {resolved_device} but CUDA is unavailable")
    requested_index = cast("int | None", cast("Any", resolved_device).index)
    device_index = torch.cuda.current_device() if requested_index is None else requested_index
    free_bytes, total_bytes = gpu_memory_info(torch, device_index)
    return {
        "device": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "free_vram_bytes": int(free_bytes),
        "total_vram_bytes": int(total_bytes),
        "allocated_vram_bytes": int(torch.cuda.memory_allocated(device_index)),
    }


def _require_single_device(model: nn.Module, requested_device: str) -> None:
    """Refuse device maps that split one Jacobian model across devices."""
    requested = torch.device(requested_device)
    requested_index = cast("int | None", cast("Any", requested).index)
    if requested.type == "cuda" and requested_index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    parameter_devices = {
        parameter.device for parameter in model.parameters() if not parameter.is_meta
    }
    if len(parameter_devices) != 1:
        raise RuntimeError(
            f"lens model parameters span multiple or no devices: {sorted(map(str, parameter_devices))}"
        )
    actual = next(iter(parameter_devices))
    if actual != requested:
        raise RuntimeError(
            f"lens model loaded on {actual}, but the requested device is {requested}"
        )


def _hash_module_parameters(module: nn.Module) -> str:
    """Hash model parameters incrementally without building a model-sized host copy."""
    digest = hashlib.sha256()
    for name, parameter in sorted(module.named_parameters()):
        if parameter.is_meta:
            raise RuntimeError(f"cannot fingerprint meta parameter {name!r}")
        value = parameter.detach()
        digest.update(name.encode())
        digest.update(b"\x00")
        digest.update(str(value.dtype).encode())
        digest.update(b"\x00")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\x00")
        for chunk in _parameter_chunks(value):
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _bounded_parameter_tiles(value: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield logical row-major tiles no larger than the fixed element bound."""
    if value.numel() <= PARAMETER_HASH_CHUNK_ELEMENTS:
        yield value
        return
    if value.ndim == 1:
        for start in range(0, value.numel(), PARAMETER_HASH_CHUNK_ELEMENTS):
            yield value[start : start + PARAMETER_HASH_CHUNK_ELEMENTS]
        return
    if value.shape[0] == 1:
        yield from _bounded_parameter_tiles(value.squeeze(0))
        return
    row_elements = value[0].numel()
    if row_elements <= PARAMETER_HASH_CHUNK_ELEMENTS:
        rows_per_tile = max(1, PARAMETER_HASH_CHUNK_ELEMENTS // row_elements)
        for start in range(0, value.shape[0], rows_per_tile):
            yield value[start : start + rows_per_tile]
        return
    for row_index in range(value.shape[0]):
        yield from _bounded_parameter_tiles(value[row_index])


def _parameter_chunks(value: torch.Tensor) -> Iterator[torch.Tensor]:
    """Yield flattened CPU chunks in logical order, copying at most the configured chunk size."""
    if value.is_contiguous():
        flattened = value.view(-1)
        for start in range(0, flattened.numel(), PARAMETER_HASH_CHUNK_ELEMENTS):
            yield flattened[start : start + PARAMETER_HASH_CHUNK_ELEMENTS].to("cpu")
        return
    for tile in _bounded_parameter_tiles(value):
        yield tile.to("cpu").contiguous().view(-1)


def _semantic_model_config(hf_model: nn.Module) -> dict[str, object]:
    """Return effective model settings without source-location or package provenance."""
    values = vars(cast("Any", hf_model).config)
    nonsemantic_fields = {"_name_or_path", "_commit_hash", "transformers_version"}
    return {key: value for key, value in values.items() if key not in nonsemantic_fields}


def _semantic_adapter_config(adapter_path: Path) -> dict[str, object]:
    """Read PEFT settings that affect loaded adapter behavior, omitting source metadata."""
    if not adapter_path.is_dir():
        raise ValueError(f"adapter path must be a local directory, got {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"adapter directory has no {config_path.name}: {adapter_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"adapter config must contain an object: {config_path}")
    nonsemantic_fields = {"base_model_name_or_path", "revision", "peft_version"}
    return {key: value for key, value in config.items() if key not in nonsemantic_fields}


def weights_fingerprint(
    model_id: str, hf_model: nn.Module, adapter_path: Path | None = None
) -> str:
    """Fingerprint loaded parameter values and effective settings, independent of provenance."""
    payload = {
        "model_config": _semantic_model_config(hf_model),
        "parameter_sha256": _hash_module_parameters(hf_model),
        "adapter_config": None if adapter_path is None else _semantic_adapter_config(adapter_path),
    }
    del model_id  # retained in the signature for older callers; source labels are provenance
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


FIT_WEIGHT_IDENTITY_VERSION = "loaded-parameters-effective-config-v2"
FIT_PROVENANCE_IDENTITY_FIELDS = frozenset(
    {
        "adapter",
        "checkpoint_every",
        "corpus_sha256",
        "implementation_sha256",
        "jlens",
        "jlens_commit",
        "lens_name",
        "load_path",
        "model",
        "repo_sha",
    }
)


def fit_identity_differences(  # noqa: PLR0911 - legacy identity layouts require separate semantic checks
    wanted: Mapping[str, object], found: Mapping[str, object]
) -> list[str]:
    """List semantic differences, allowing older sidecars' comparable subset."""
    generic_core = {
        "tokenizer_sha256",
        "selected_text_sha256",
        "n_prompts_requested",
        "source_layers",
        "target_layer",
        "dim_batch",
        "max_seq_len",
        "skip_first",
        "halves",
    }
    wanted_semantic_fields = set(wanted) - FIT_PROVENANCE_IDENTITY_FIELDS
    if wanted.get("weights_fingerprint_semantics") == FIT_WEIGHT_IDENTITY_VERSION:
        if missing := sorted((generic_core | {"weights_fingerprint"}) - wanted_semantic_fields):
            return [f"wanted identity is missing semantic fields {missing}"]
        legacy_weights = found.get("weights_fingerprint_semantics") != FIT_WEIGHT_IDENTITY_VERSION
        if legacy_weights:
            if missing := sorted(generic_core - set(found)):
                return [f"legacy identity is missing semantic fields {missing}"]
        elif missing := sorted(wanted_semantic_fields - set(found)):
            return [f"saved identity is missing semantic fields {missing}"]
        fields = (set(wanted) | set(found)) - FIT_PROVENANCE_IDENTITY_FIELDS
        fields.discard("weights_fingerprint_semantics")
        if legacy_weights:
            fields.discard("weights_fingerprint")
    elif "corpus_digests" in wanted:
        required_fields = wanted_semantic_fields | {
            "corpus_digests",
            "fit_role",
            "n_fit_prompts",
            "dim_batch",
            "max_seq_len",
            "skip_first",
            "fit_prompts_sha256",
        }
        if missing := sorted(required_fields - set(wanted)):
            return [f"wanted identity is missing semantic fields {missing}"]
        if missing := sorted(required_fields - set(found)):
            return [f"saved identity is missing semantic fields {missing}"]
        fields = (set(wanted) | set(found)) - FIT_PROVENANCE_IDENTITY_FIELDS
    else:
        return ["wanted identity has no recognized semantic fit core"]
    return sorted(
        field
        for field in fields
        if field in wanted and field in found and wanted[field] != found[field]
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_implementation_sha256(
    relative_paths: Sequence[str], *, repo_root: Path | None = None
) -> dict[str, str]:
    """Hash a bounded set of repository source files for a resumable fit identity."""
    if not relative_paths or len(relative_paths) > MAX_IMPLEMENTATION_SOURCE_FILES:
        raise ValueError(
            "implementation identity needs between 1 and "
            f"{MAX_IMPLEMENTATION_SOURCE_FILES} source files, got {len(relative_paths)}"
        )
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("implementation identity source paths must be unique")
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    implementation_sha256: dict[str, str] = {}
    for raw_path in sorted(relative_paths):
        relative_path = Path(raw_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"implementation source path must stay under the repository: {raw_path}"
            )
        source_path = (root / relative_path).resolve()
        if not source_path.is_relative_to(root):
            raise ValueError(f"implementation source path escapes the repository: {raw_path}")
        if not source_path.is_file():
            raise FileNotFoundError(f"implementation source file does not exist: {source_path}")
        implementation_sha256[relative_path.as_posix()] = _hash_file(source_path)
    return implementation_sha256


_PEFT_KEY_PREFIX = "base_model.model."


def adapter_targeting_its_weights(adapter_dir: Path, work_dir: Path) -> tuple[Path, int]:
    """Write an adapter view whose ``target_modules`` are exactly the modules that have weights.

    A language-only copy of a multimodal adapter keeps short targets such as ``q_proj`` that also
    name vision-tower modules; PEFT then injects empty LoRA slots there and the application gate
    refuses the load. The view symlinks the weight file and rewrites only the targets.
    """
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"lens adapters must ship adapter_model.safetensors: {adapter_dir}")
    with safe_open(str(weights_path), "pt") as handle:
        keys = list(handle.keys())
    non_lora = [key for key in keys if ".lora_" not in key]
    if non_lora:
        raise NotImplementedError(
            f"adapter has non-LoRA weights the lens loader does not handle: {non_lora[:3]}"
        )
    modules = sorted({key.removeprefix(_PEFT_KEY_PREFIX).rsplit(".lora_", 1)[0] for key in keys})
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    config["target_modules"] = modules
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
    )
    (work_dir / "adapter_model.safetensors").symlink_to(weights_path)
    return work_dir, len(modules)


def load_fit_model(
    model_id: str,
    *,
    jl: ModuleType,
    adapter_path: Path | None = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str | None = None,
) -> FitModelHandle:
    """Load one HF causal LM on one device, attach a local PEFT adapter, then wrap it for ``jlens``."""
    requested_device = (
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_device = str(torch.device(requested_device))
    if torch.device(resolved_device).type not in {"cpu", "cuda"}:
        raise ValueError(f"lens fitting supports one CUDA device or CPU, got {resolved_device!r}")
    if torch.device(resolved_device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested lens device {resolved_device} but CUDA is unavailable")
    resolved_adapter_path: Path | None = None
    adapter_declared_base: str | None = None
    adapter_config_sha256: str | None = None
    applied_adapter_weights = 0
    adapted_module_count = 0
    if adapter_path is not None:
        resolved_adapter_path = adapter_path.resolve()
        adapter_config = PeftConfig.from_pretrained(str(resolved_adapter_path))
        adapter_declared_base = adapter_config.base_model_name_or_path
        adapter_config_path = resolved_adapter_path / "adapter_config.json"
        adapter_config_sha256 = _hash_file(adapter_config_path)

    loaded_model, tokenizer = load_model_and_tokenizer(
        model_id, dtype=dtype, device=resolved_device
    )
    base_model = cast("nn.Module", loaded_model)
    active_model: nn.Module = base_model
    if resolved_adapter_path is not None:
        with tempfile.TemporaryDirectory(prefix="lens-adapter-view-") as view_root:
            adapter_view, adapted_module_count = adapter_targeting_its_weights(
                resolved_adapter_path, Path(view_root) / "adapter"
            )
            peft_model = PeftModel.from_pretrained(
                base_model, str(adapter_view), is_trainable=False
            )
            if peft_model.active_peft_config.is_prompt_learning:
                raise NotImplementedError(
                    "prompt-learning PEFT adapters prepend virtual tokens outside the text-block tree; "
                    "the Jacobian fit requires an adapter injected into model modules"
                )
            applied_adapter_weights = assert_adapter_applies(peft_model, adapter_view)
        active_model = peft_model.get_base_model()
        peft_model.eval()
    active_model.eval()
    _require_single_device(active_model, resolved_device)
    model = jl.from_hf(active_model, tokenizer)
    if resolved_adapter_path is not None:
        adapter_module_ids = {
            id(module) for module in active_model.modules() if isinstance(module, BaseTunerLayer)
        }
        block_module_ids = {id(module) for block in model.layers for module in block.modules()}
        if not adapter_module_ids.intersection(block_module_ids):
            raise RuntimeError(
                "the adapter has no injected layer inside the residual blocks exposed to jlens; "
                "the fit would read untuned activations"
            )
    facts = device_facts(resolved_device)
    logger.info(
        "lens device memory, device=%s free_vram_bytes=%s total_vram_bytes=%s",
        facts["device"],
        facts["free_vram_bytes"],
        facts["total_vram_bytes"],
    )
    fingerprint = weights_fingerprint(model_id, active_model, resolved_adapter_path)
    handle = FitModelHandle(
        model=model,
        hf_model=active_model,
        tokenizer=tokenizer,
        model_id=model_id,
        adapter_path=resolved_adapter_path,
        weights_fingerprint=fingerprint,
        device_facts=facts,
        adapter_declared_base=adapter_declared_base,
        adapter_config_sha256=adapter_config_sha256,
        applied_adapter_weights=applied_adapter_weights,
        adapted_module_count=adapted_module_count,
    )
    logger.info("loaded lens model, %s", json.dumps(handle.as_payload(), default=str))
    return handle


def _write_or_check_fit_identity(
    identity_path: Path, checkpoint_path: Path | None, identity: Mapping[str, object]
) -> None:
    """Atomically record the fit identity or refuse an incompatible partial accumulator."""
    identity_path.parent.mkdir(parents=True, exist_ok=True)
    expected = json.dumps(dict(identity), sort_keys=True, indent=2) + "\n"
    if identity_path.is_file():
        stored_identity = json.loads(identity_path.read_text())
        if not isinstance(stored_identity, dict) or not stored_identity:
            raise FitIdentityError(f"fit identity at {identity_path} is not a JSON object")
        differences = fit_identity_differences(identity, stored_identity)
        if differences:
            raise FitIdentityError(
                f"fit identity differs on semantic fields {differences} at {identity_path}; "
                "refusing to resume"
            )
        actual = json.dumps(stored_identity, sort_keys=True, indent=2) + "\n"
        if actual != expected:
            temporary_path = identity_path.with_name(f"{identity_path.name}.tmp")
            temporary_path.write_text(expected)
            temporary_path.replace(identity_path)
        return
    if checkpoint_path is not None and checkpoint_path.exists():
        raise FitIdentityError(
            f"fit checkpoint {checkpoint_path} has no {FIT_IDENTITY_FILENAME}; its model, corpus "
            "and fit knobs cannot be verified, so refusing to resume it"
        )
    temporary_path = identity_path.with_name(f"{identity_path.name}.tmp")
    temporary_path.write_text(expected)
    temporary_path.replace(identity_path)


def fit_with_resume_identity(  # noqa: PLR0913 - the identity sidecar is part of the resume contract
    config: JacobianConfig,
    model: object,
    prompts: Sequence[str],
    jl: ModuleType,
    *,
    identity: Mapping[str, object],
    identity_path: Path,
) -> object:
    """Bind an exact identity to a checkpoint before ``jlens.fit`` may resume it."""
    _write_or_check_fit_identity(identity_path, config.checkpoint_path, identity)
    return fit_lens(config, model, prompts, jl)


def read_reconstruction(  # noqa: PLR0913 - explicit evaluation knobs keep reads comparable
    lens: object,
    model: object,
    prompts: Sequence[str],
    *,
    max_seq_len: int,
    max_positions: int,
    skip_first: int,
    target_layer: int | None,
) -> FitQualityReport | None:
    """Compare each fitted-layer reconstruction with actual logits and the logit-lens baseline."""
    return evaluate_reconstruction(
        lens,
        model,
        prompts,
        max_seq_len=max_seq_len,
        max_positions=max_positions,
        skip_first=skip_first,
        target_layer=target_layer,
    )


def reconstruction_read_payload(report: FitQualityReport | None) -> dict[str, object]:
    """Serialize per-layer reconstruction and logit-lens diagnostics without gating on a win."""
    payload = fit_quality_payload(report)
    payload["interpretation"] = (
        "Compare per-layer reads; distribution shift and weak early or late layers can lower "
        "reconstruction, while coherent J-space is expected in an intermediate band."
    )
    return payload


def validate_unembedding(hf_model: nn.Module, jl_model: object) -> dict[str, object]:
    """Reject unusable output weights before token recovery can be fooled by NaNs or zero logits."""
    layout = cast("Any", jl_model).layout
    text_model = hf_model.get_submodule(str(layout.path))
    norm = text_model.get_submodule(str(layout.norm))
    lm_head = hf_model.get_submodule(str(layout.lm_head))
    norm_parameters = dict(norm.named_parameters())
    head_parameters = dict(lm_head.named_parameters())
    if "weight" not in head_parameters:
        raise RuntimeError("the model's lm_head has no weight tensor")
    tensors = {
        **{f"norm.{name}": value for name, value in norm_parameters.items()},
        **{f"lm_head.{name}": value for name, value in head_parameters.items()},
    }
    head_weight = head_parameters["weight"].detach()
    squared_head_norm = torch.zeros((), dtype=torch.float32, device="cpu")
    has_nonzero_head_weight = False
    for name, tensor in tensors.items():
        for chunk in _parameter_chunks(tensor.detach()):
            if not torch.isfinite(chunk).all():
                raise RuntimeError(f"the unembedding has non-finite values in {name}")
            if name == "lm_head.weight":
                chunk_float = chunk.float()
                squared_head_norm += chunk_float.square().sum()
                has_nonzero_head_weight = has_nonzero_head_weight or bool(
                    torch.count_nonzero(chunk)
                )
    head_norm = float(torch.sqrt(squared_head_norm).item())
    if not has_nonzero_head_weight:
        raise RuntimeError("the lm_head weight has zero norm, so token recovery is undefined")
    return {
        "passed": True,
        "norm_parameter_count": sum(parameter.numel() for parameter in norm_parameters.values()),
        "lm_head_weight_shape": list(head_weight.shape),
        "lm_head_weight_norm": head_norm,
    }


def run_hybrid_architecture_gates(  # noqa: PLR0913 - architecture audit inputs are explicit
    handle: FitModelHandle,
    jl: ModuleType,
    prompt: str,
    *,
    target_layer: int | None,
    max_seq_len: int,
    dim_batch: int,
    source_layers: Sequence[int] | None = None,
) -> dict[str, object]:
    """Run DeltaNet autograd and fused-kernel checks only for hybrid Qwen3.5 text stacks."""
    text_config = cast("Any", handle.hf_model.config).get_text_config()
    config_values = text_config.to_dict()
    layer_types = tuple(str(kind) for kind in config_values.get("layer_types", ()))
    if "linear_attention" not in layer_types:
        return {
            "applicable": False,
            "passed": True,
            "reason": "text config has no linear-attention layers",
        }
    model_type = str(config_values.get("model_type", ""))
    if model_type not in {"qwen3_5", "qwen3_5_text"}:
        raise RuntimeError(
            f"hybrid DeltaNet gates do not support text model_type={model_type!r}; refusing to skip them"
        )
    if len(layer_types) != int(cast("Any", handle.model).n_layers):
        raise RuntimeError(
            f"text config names {len(layer_types)} layers but jlens exposes "
            f"{int(cast('Any', handle.model).n_layers)}"
        )
    modeling_module = importlib.import_module(type(handle.hf_model).__module__)
    recurrence_type = modeling_module.__dict__.get("Qwen3_5GatedDeltaNet")
    if not isinstance(recurrence_type, type) or not issubclass(recurrence_type, nn.Module):
        raise TypeError("Qwen3.5 modeling module exposes no Qwen3_5GatedDeltaNet module")
    conv_reach = int(config_values["linear_conv_kernel_dim"]) - 1
    position_gap = max(DEFAULT_POSITION_GAP, conv_reach + 1)
    input_ids = cast("Any", handle.model).encode(prompt, max_length=max_seq_len)
    if int(input_ids.shape[1]) <= position_gap + 1:
        raise RuntimeError(
            f"hybrid architecture gate needs more than {position_gap + 1} tokens; "
            f"the selected prompt has {int(input_ids.shape[1])}"
        )
    resolved_sources, resolved_target = _resolve_fit_layers(
        source_layers, target_layer, int(cast("Any", handle.model).n_layers)
    )
    kernels_bound = bound_deltanet_kernels()
    autograd = probe_recurrence_autograd(
        handle.blocks,
        handle.forward,
        input_ids,
        layer_types=layer_types,
        recurrence_type=recurrence_type,
        gap=position_gap,
        conv_reach=conv_reach,
    )
    audit_dims = min(KERNEL_AUDIT_DIMS, dim_batch)
    if audit_dims <= 0:
        raise ValueError(f"dim_batch must be positive, got {dim_batch}")
    kernel = audit_chunk_kernel_backward(
        handle.blocks,
        handle.forward,
        input_ids,
        modeling_module=modeling_module,
        layer_types=layer_types,
        estimator=RowEstimator(
            source_layers=tuple(resolved_sources),
            target_layer=resolved_target,
            dims=tuple(range(audit_dims)),
            skip_first=fit_skip_first(jl),
        ),
        tolerance=KERNEL_AUDIT_TOLERANCE_BF16,
    )
    return {
        "applicable": True,
        "passed": autograd.passed and kernel.passed,
        "model_type": model_type,
        "layer_types": list(layer_types),
        "position_gap": position_gap,
        "convolution_reach": conv_reach,
        "deltanet_kernels_bound": kernels_bound,
        "prefill_deltanet_kernels": prefill_deltanet_kernels(kernels_bound),
        "autograd": autograd.as_payload(),
        "kernel": kernel.as_payload(),
    }


def save_and_read_lens(  # noqa: PLR0913 - save and diagnostic settings must be explicit
    lens: object,
    model: object,
    jl: ModuleType,
    *,
    lens_path: Path,
    eval_prompts: Sequence[str],
    max_seq_len: int,
    max_positions: int,
    skip_first: int,
    target_layer: int | None,
) -> FitRunResult:
    """Save and round-trip a lens, then record its per-layer reconstruction diagnostic."""
    lens_path.parent.mkdir(parents=True, exist_ok=True)
    persistence_started = time.perf_counter()
    cast("Any", lens).save(str(lens_path))
    verify_lens_roundtrip(jl, lens, lens_path)
    lens_sha256 = _hash_file(lens_path)
    persistence_seconds = time.perf_counter() - persistence_started
    read_started = time.perf_counter()
    quality = read_reconstruction(
        lens,
        model,
        eval_prompts,
        max_seq_len=max_seq_len,
        max_positions=max_positions,
        skip_first=skip_first,
        target_layer=target_layer,
    )
    return FitRunResult(
        lens=lens,
        fit_seconds=0.0,
        persistence_seconds=persistence_seconds,
        read_seconds=time.perf_counter() - read_started,
        lens_sha256=lens_sha256,
        quality=quality,
    )


def fit_save_and_read_lens(  # noqa: PLR0913 - shared lifecycle inputs define the fit identity
    config: JacobianConfig,
    model: object,
    prompts: Sequence[str],
    jl: ModuleType,
    *,
    identity: Mapping[str, object],
    identity_path: Path,
    lens_path: Path,
    eval_prompts: Sequence[str],
    eval_max_seq_len: int | None = None,
    eval_positions: int = 8,
) -> FitRunResult:
    """Run the shared fit lifecycle: identity, resume, save, round-trip, digest and reconstruction."""
    fit_started = time.perf_counter()
    lens = fit_with_resume_identity(
        config,
        model,
        prompts,
        jl,
        identity=identity,
        identity_path=identity_path,
    )
    fit_seconds = time.perf_counter() - fit_started
    result = save_and_read_lens(
        lens,
        model,
        jl,
        lens_path=lens_path,
        eval_prompts=eval_prompts,
        max_seq_len=config.max_seq_len if eval_max_seq_len is None else eval_max_seq_len,
        max_positions=eval_positions,
        skip_first=config.skip_first,
        target_layer=config.target_layer,
    )
    return FitRunResult(
        lens=result.lens,
        fit_seconds=fit_seconds,
        persistence_seconds=result.persistence_seconds,
        read_seconds=result.read_seconds,
        lens_sha256=result.lens_sha256,
        quality=result.quality,
    )
