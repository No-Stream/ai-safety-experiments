"""Shared throughput options remain explicit in SFT and organism training."""

import json
from pathlib import Path

import pytest

from reward_hacking.distill.memory import MemoryOptions
from reward_hacking.distill.train import memory_options_from_args, parse_args
from reward_hacking.organism_rl.cli import parse_args as parse_organism_args


@pytest.mark.parametrize("quantization", ["torchao-w8a8", "torchao-fp8-rowwise"])
def test_trainers_share_explicit_throughput_options(quantization: str, tmp_path: Path) -> None:
    admission = tmp_path / "admission.json"
    admission.write_text(
        json.dumps(
            {"before": {"mean_abs_logprob_gap": 0.01}, "after": {"mean_abs_logprob_gap": 0.02}}
        )
    )
    options = [
        "--tiled-mlp",
        "--offload-checkpoints",
        "--skip-dead-mlp-recompute",
        "--fla-nb-bucket",
        "pow2",
        "--prewarm-fla",
        "--base-quantization",
        quantization,
    ]
    sft = parse_args(
        ["--run-dir", "artifacts/synthetic", "--round", "0", "--weight-mode", "sft", *options]
    )
    organism = parse_organism_args(
        [
            "--output-dir",
            "artifacts/synthetic",
            "--endpoint",
            "http://localhost:8000",
            "--quantization-admission",
            str(admission),
            *options,
        ]
    )
    assert memory_options_from_args(sft) == memory_options_from_args(organism)
    configured = memory_options_from_args(sft)
    assert configured.skip_dead_mlp_recompute
    assert configured.prewarm_fla
    assert configured.fla_nb_bucket == "pow2"
    assert configured.base_quantization == quantization


@pytest.mark.parametrize(
    "invalid",
    [{"prewarm_fla": True}, {"skip_dead_mlp_recompute": True}, {"fla_nb_bucket": "unknown"}],
)
def test_invalid_throughput_combinations_refuse(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=r"requires|must be"):
        MemoryOptions(**invalid)  # pyright: ignore[reportArgumentType]
