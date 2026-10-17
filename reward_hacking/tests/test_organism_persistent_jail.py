"""Persistent actor-jail configuration and execution routing."""

from pathlib import Path

import pytest

from reward_hacking.organism_rl import cli, evaluate, generate, ladder_wave


@pytest.mark.parametrize("disabled", [False, True])
def test_training_persistent_jail_default_and_fallback(disabled: bool) -> None:
    arguments = ["--output-dir", "artifacts/synthetic", "--endpoint", "http://synthetic:8000"]
    if disabled:
        arguments.append("--no-persistent-jail")
    assert cli.parse_args(arguments).persistent_jail is not disabled


@pytest.mark.parametrize("disabled", [False, True])
def test_generation_persistent_jail_default_and_fallback(tmp_path: Path, disabled: bool) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path),
        "--endpoint",
        "http://synthetic:8000",
        "--max-model-len",
        "1024",
    ]
    if disabled:
        arguments.append("--no-persistent-jail")
    assert generate.parse_args(arguments).persistent_jail is not disabled


@pytest.mark.parametrize("disabled", [False, True])
def test_evaluation_persistent_jail_default_and_fallback(tmp_path: Path, disabled: bool) -> None:
    arguments = ["--output-dir", str(tmp_path), "--endpoint", "http://synthetic:8000"]
    arguments.extend(["--max-model-len", "1024", "--tasks-per-family", "1"])
    if disabled:
        arguments.append("--no-persistent-jail")
    assert evaluate.parse_args(arguments).persistent_jail is not disabled


@pytest.mark.parametrize("disabled", [False, True])
def test_ladder_wave_persistent_jail_default_and_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disabled: bool
) -> None:
    arguments = [
        "--output-dir",
        str(tmp_path),
        "--endpoint",
        "http://synthetic:8000",
        "--model",
        "synthetic",
        "--served-model-id",
        "synthetic",
        "--max-model-len",
        "1024",
    ]
    if disabled:
        arguments.append("--no-persistent-jail")
    assert ladder_wave._argument_parser().parse_args(arguments).persistent_jail is not disabled

    def run_wave(**kwargs: object) -> dict[str, object]:
        assert kwargs["persistent_jail"] is not disabled
        return {}

    monkeypatch.setattr(ladder_wave, "run_ladder_wave", run_wave)
    assert ladder_wave.main(arguments) == 0
