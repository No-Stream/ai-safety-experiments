"""Shared serving quantization controls use measured admission and preserve MTP."""

import argparse
import json
from pathlib import Path

import pytest

from reward_hacking.organism_rl import evaluate
from reward_hacking.serving_quantization import (
    add_serving_quantization_arguments,
    serving_quantization_kwargs,
    validate_serving_quantization,
)


def _parser(
    *,
    kv_cache_dtype_default: str | None = None,
    sampler_quantization_default: str = "none",
    sampler_quantization_ignore_default: tuple[str, ...] = (),
    quantization_admission_default: Path | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_serving_quantization_arguments(
        parser,
        kv_cache_dtype_default=kv_cache_dtype_default,
        sampler_quantization_default=sampler_quantization_default,
        sampler_quantization_ignore_default=sampler_quantization_ignore_default,
        quantization_admission_default=quantization_admission_default,
    )
    return parser


@pytest.mark.parametrize("gap", [None, True, -0.1, float("nan"), float("inf"), "0.1"])
def test_admission_rejects_invalid_measurements(tmp_path: Path, gap: object) -> None:
    record = tmp_path / "admission.json"
    record.write_text(
        json.dumps({stage: {"mean_abs_logprob_gap": gap} for stage in ("before", "after")})
    )
    parser = _parser()
    args = parser.parse_args(["--kv-cache-dtype", "fp8", "--quantization-admission", str(record)])
    with pytest.raises(SystemExit):
        validate_serving_quantization(args, parser)


def test_defaults_can_be_fully_disabled_without_reading_admission(tmp_path: Path) -> None:
    parser = _parser(
        kv_cache_dtype_default="fp8",
        sampler_quantization_default="fp8_per_channel",
        sampler_quantization_ignore_default=("*mtp*",),
        quantization_admission_default=tmp_path / "missing.json",
    )
    args = parser.parse_args(["--kv-cache-dtype", "auto", "--sampler-quantization", "none"])
    validate_serving_quantization(args, parser)
    assert serving_quantization_kwargs(args) == {
        "kv_cache_dtype": "auto",
        "quantization": None,
        "quantization_config": None,
    }
    assert args.quantization_admission is None


@pytest.mark.parametrize("ignore", [[], ["*in_proj_ba"]])
def test_online_quantization_requires_mtp_exclusion(tmp_path: Path, ignore: list[str]) -> None:
    record = tmp_path / "admission.json"
    record.write_text(
        json.dumps({stage: {"mean_abs_logprob_gap": 0.1} for stage in ("before", "after")})
    )
    parser = _parser()
    args = parser.parse_args(
        [
            "--sampler-quantization",
            "fp8_per_channel",
            "--quantization-admission",
            str(record),
            *[
                argument
                for pattern in ignore
                for argument in ("--sampler-quantization-ignore", pattern)
            ],
        ]
    )
    with pytest.raises(SystemExit):
        validate_serving_quantization(args, parser, speculative_config={"method": "mtp"})


def test_evaluation_forwards_quantization_to_server(tmp_path: Path) -> None:
    record = tmp_path / "admission.json"
    record.write_text(
        json.dumps({stage: {"mean_abs_logprob_gap": 0.1} for stage in ("before", "after")})
    )
    args = evaluate.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--max-model-len",
            "2048",
            "--tasks-per-family",
            "1",
            "--gpu-memory-utilization",
            "0.8",
            "--kv-cache-dtype",
            "fp8",
            "--sampler-quantization",
            "fp8_per_channel",
            "--sampler-quantization-ignore",
            "*mtp*",
            "--quantization-admission",
            str(record),
        ]
    )
    config = evaluate.build_server_config(args, "synthetic-revision")
    assert config.kv_cache_dtype == "fp8"
    assert config.quantization == "fp8_per_channel"
    assert config.quantization_config == {"ignore": ["*mtp*"]}


@pytest.mark.parametrize(
    "flags", [["--kv-cache-dtype", "fp8"], ["--sampler-quantization", "fp8_per_channel"]]
)
def test_quantization_requires_admission(flags: list[str]) -> None:
    parser = _parser()
    with pytest.raises(SystemExit):
        validate_serving_quantization(parser.parse_args(flags), parser)


def test_explicit_ignore_replaces_default_mtp_exclusion(tmp_path: Path) -> None:
    record = tmp_path / "admission.json"
    record.write_text(
        json.dumps({stage: {"mean_abs_logprob_gap": 0.1} for stage in ("before", "after")})
    )
    parser = _parser(
        sampler_quantization_default="fp8_per_channel",
        sampler_quantization_ignore_default=("*mtp*",),
        quantization_admission_default=record,
    )
    args = parser.parse_args(["--sampler-quantization-ignore", "*conv1d*"])
    with pytest.raises(SystemExit):
        validate_serving_quantization(args, parser, speculative_config={"method": "mtp"})
    assert args.sampler_quantization_ignore == ["*conv1d*"]
