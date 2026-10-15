"""Offline CPU loading for the pinned, optional ALCE learned citation metric."""

from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from reward_hacking.spec_gaming.alce_source import (
    ALCE_AUTOAIS_MODEL_REVISION,
    ALCE_AUTOAIS_MODEL_SOURCE,
    AlceAutoAISRuntime,
)
from reward_hacking.spec_gaming.config import exact_keys, file_sha256, read_json_object

MAX_THREADS = 4
MAX_CITATIONS = 100

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def autoais_dependency_identity() -> dict[str, str]:
    """Bind installed evaluator dependencies without loading model weights."""
    return {
        package: version(package) for package in ("torch", "transformers", "tokenizers", "nltk")
    }


def load_autoais_manifest(path: Path) -> dict[str, Any]:
    """Accept only the frozen model on CPU; inference settings have separate identity."""
    manifest = read_json_object(path, "AutoAIS manifest")
    exact_keys(
        manifest,
        {
            "schema_version",
            "model_source",
            "model_revision",
            "dtype",
            "device",
            "threads",
            "at_most_citations",
        },
        "AutoAIS manifest",
    )
    expected = {
        "schema_version": 1,
        "model_source": ALCE_AUTOAIS_MODEL_SOURCE,
        "model_revision": ALCE_AUTOAIS_MODEL_REVISION,
        "dtype": "bfloat16",
        "device": "cpu",
    }
    if type(manifest["schema_version"]) is not int or any(
        manifest[key] != value for key, value in expected.items()
    ):
        raise ValueError("AutoAIS requires the pinned offline CPU bfloat16 model")
    if type(manifest["threads"]) is not int or not 1 <= manifest["threads"] <= MAX_THREADS:
        raise ValueError("AutoAIS threads must be between one and four")
    cap = manifest["at_most_citations"]
    if cap is not None and (type(cap) is not int or cap < 1 or cap > MAX_CITATIONS):
        raise ValueError("AutoAIS citation cap must be null or an integer from one to 100")
    return manifest


def load_autoais_runtime(
    manifest: dict[str, Any], segmenter: Callable[[str], Sequence[str]]
) -> AlceAutoAISRuntime:
    """Resolve the exact cached revision and verify actual loaded device and dtype."""
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("AutoAIS requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("AutoAIS requires CUDA_VISIBLE_DEVICES to be empty")
    snapshot = Path(
        snapshot_download(
            ALCE_AUTOAIS_MODEL_SOURCE,
            revision=ALCE_AUTOAIS_MODEL_REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.bin", "spiece.model"],
        )
    )
    if snapshot.name != ALCE_AUTOAIS_MODEL_REVISION:
        raise ValueError("cached AutoAIS snapshot differs from pinned revision")
    identity = {
        "model_source": ALCE_AUTOAIS_MODEL_SOURCE,
        "model_revision": snapshot.name,
        "tokenizer_source": ALCE_AUTOAIS_MODEL_SOURCE,
        "tokenizer_revision": snapshot.name,
        "dtype": "bfloat16",
        "device": "cpu",
        "config_sha256": file_sha256(snapshot / "config.json"),
        "torch_version": str(torch.__version__),
        "transformers_version": str(transformers.__version__),
    }
    if any(type(value) is not str for value in identity.values()):
        raise TypeError("AutoAIS loaded identity must contain plain JSON strings")
    torch.set_num_threads(manifest["threads"])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot,
        local_files_only=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    model.eval()
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.bfloat16
        for parameter in model.parameters()
    ):
        raise ValueError("loaded AutoAIS parameters differ from CPU bfloat16 contract")
    return AlceAutoAISRuntime(model, tokenizer, segmenter, identity, manifest["at_most_citations"])
