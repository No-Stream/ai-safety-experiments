"""Initialize a fresh training run from a compatible complete LoRA checkpoint."""

import hashlib
import json
from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.save_and_load import load_peft_weights

from reward_hacking.organism_rl.generate_provenance import active_provenance_path
from reward_hacking.organism_rl.trainer import validate_checkpoint


def adapter_initialization_metadata(
    checkpoint: Path,
    *,
    expected_config: LoraConfig,
    model_name: str,
    model_revision: str | None,
) -> dict[str, Any]:
    """Validate source identity and return its readable, flattened initialization lineage."""
    checkpoint = checkpoint.resolve()
    validate_checkpoint(checkpoint)
    adapter_path = checkpoint / "adapter_model.safetensors"
    configuration = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    provenance_path = active_provenance_path(checkpoint.parent)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = {
        "r": expected_config.r,
        "lora_alpha": expected_config.lora_alpha,
        "target_modules": _normalized_targets(expected_config.target_modules),
        "base_model_name_or_path": model_name,
        "revision": model_revision,
    }
    actual = {
        "r": configuration["r"],
        "lora_alpha": configuration["lora_alpha"],
        "target_modules": _normalized_targets(configuration["target_modules"]),
        "base_model_name_or_path": configuration["base_model_name_or_path"],
        "revision": configuration.get("revision") or provenance["model_revision"],
    }
    for key in ("rank_pattern", "alpha_pattern", "use_rslora"):
        expected[key] = expected_config.to_dict()[key]
        actual[key] = configuration[key]
    differences = [
        f"{key}: source={actual[key]!r}, expected={value!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    source_model = provenance["flags"]["model"]
    if source_model != model_name:
        differences.append(
            f"provenance.flags.model: source={source_model!r}, expected={model_name!r}"
        )
    if provenance["model_revision"] != model_revision:
        differences.append(
            f"provenance.model_revision: source={provenance['model_revision']!r}, expected={model_revision!r}"
        )
    if differences:
        raise ValueError("Initial adapter configuration differs:\n" + "\n".join(differences))
    state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    step = state["global_step"]
    if type(step) is not int or step < 0:
        raise ValueError("Initial adapter source global_step must be a nonnegative integer")
    digest = hashlib.sha256()
    with adapter_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    source = {
        "source_checkpoint": str(checkpoint),
        "source_step": step,
        "source_provenance": str(provenance_path.resolve()),
        "adapter_sha256": digest.hexdigest(),
    }
    previous = provenance.get("init_adapter_from")
    lineage = [] if previous is None else previous["lineage"]
    if not isinstance(lineage, list) or not all(isinstance(entry, dict) for entry in lineage):
        raise ValueError("Initial adapter source lineage must be a list of source records")
    return {**source, "lineage": [*lineage, source]}


def _normalized_targets(targets: str | list[str] | set[str] | None) -> str | list[str] | None:
    return targets if targets is None or isinstance(targets, str) else sorted(targets)


def load_initial_adapter(
    model: PeftModel,
    checkpoint: Path,
    *,
    model_name: str,
    model_revision: str | None,
) -> dict[str, Any]:
    """Copy only source adapter tensors into an already-created trainable default adapter."""
    configuration = model.peft_config["default"]
    if not isinstance(configuration, LoraConfig):
        raise TypeError("Initial adapter destination must use LoraConfig")
    metadata = adapter_initialization_metadata(
        checkpoint,
        expected_config=configuration,
        model_name=model_name,
        model_revision=model_revision,
    )
    weights = load_peft_weights(str(checkpoint), device="cpu", local_files_only=True)
    expected_weights = get_peft_model_state_dict(model)
    missing = sorted(expected_weights.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected_weights.keys())
    if missing or unexpected:
        raise ValueError(
            f"Initial adapter tensor keys differ: missing={missing}, unexpected={unexpected}"
        )
    set_peft_model_state_dict(model, weights)
    return metadata
