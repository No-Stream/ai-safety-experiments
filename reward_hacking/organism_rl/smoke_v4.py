"""Run the 0.8B end-to-end v4 organism smoke path."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit
from urllib.request import urlopen

import torch
from transformers import AutoConfig

from reward_hacking.distill.sample import resolve_gpu_engine_settings
from reward_hacking.ladder.scenario import load_scenarios
from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl.smoke_v4_assert import (
    CellExpectation,
    SmokeAssertionError,
    assert_cell_records,
    assert_resume_completion,
    assert_smoke_run,
)
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

logger = logging.getLogger(__name__)
MODEL = "Qwen/Qwen3.5-0.8B"
# Only the LoRA phase serves this id; the adapter-free server serves the base model as MODEL.
ADAPTER_SERVED_MODEL_ID = "smoke-v4-0.8b"
HINTS_PATH = Path("docs/scratch/organism-hints/hints-v2.json")
LEAK_SPLITS_PATH = Path("docs/scratch/leak-v4/channel-splits.json")
LEAK_TRAIN_ASSETS = Path("/var/tmp/leak-env/assets-v4-train/assets.json")  # noqa: S108 - owner-specified assets
LEAK_HELDOUT_ASSETS = Path("/var/tmp/leak-env/assets-v4-heldout-b/assets.json")  # noqa: S108 - owner-specified assets
LEAK_SPLIT_NAME = "seed-v4"
LADDER_SCENARIO_ROOT = Path("reward_hacking/ladder/data/scenarios-v2.2-heldout-wip")
MAX_TURNS = 4
THINKING_TOKEN_BUDGET = 512
MAX_NEW_TOKENS = 2048
TOTAL_EPISODE_MINUTES_ESTIMATE = 45
FIRST_EPISODE_TIMEOUT_SECONDS = 20 * 60
RESUME_POLL_SECONDS = 0.2
GENERATE_EPISODES_PER_CELL = 2
LEAK_PAIR_LENGTH = 2
SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class EvalCell:
    """One held-out evaluation run and its expected record contract."""

    name: str
    family: str
    variant: str
    expected_count: int = 1
    leak_channels_by_task: Mapping[str, str] | None = None
    require_math_evidence: bool = False


@dataclass(frozen=True, slots=True)
class GenerateCell:
    """One training family and its v4 hint and payment settings."""

    name: str
    family: str
    variant: str
    payment_register: str
    hint_family: str
    leak_assets: Path | None = None


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else _repository_root() / path


def _channel_list(value: object, *, description: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(channel, str) or not channel for channel in value
    ):
        raise TypeError(f"{description} must be a list of channel names")
    return tuple(cast("list[str]", value))


def _eval_pair_channels(pairs: list[object], heldout_channels: set[str]) -> dict[str, str]:
    channels_by_task: dict[str, str] = {}
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != LEAK_PAIR_LENGTH
            or any(not isinstance(value, str) or not value for value in pair)
        ):
            raise TypeError("each leak eval pair must contain a task id and channel")
        task_id, channel = cast("list[str]", pair)
        if task_id in channels_by_task or channel not in heldout_channels:
            raise ValueError("leak eval pairs must be unique and use held-out channels")
        channels_by_task[task_id] = channel
    if not channels_by_task:
        raise ValueError("leak split has no eval_pairs")
    return channels_by_task


def _read_channel_split(path: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("leak channel split file must contain a JSON object")
    split_value = payload.get(LEAK_SPLIT_NAME)
    if split_value is None:
        raise ValueError(f"leak channel split {LEAK_SPLIT_NAME!r} is missing")
    if not isinstance(split_value, Mapping):
        raise TypeError(f"leak channel split {LEAK_SPLIT_NAME!r} must contain an object")
    train_channels = _channel_list(split_value.get("train"), description="train channels")
    heldout_channels = _channel_list(split_value.get("held-out"), description="held-out channels")
    pairs_value = payload.get("eval_pairs")
    if not isinstance(pairs_value, list):
        raise TypeError("leak split requires train, held-out, and eval_pairs lists")
    if not train_channels or len(train_channels) != len(set(train_channels)):
        raise ValueError("training leak channels must be nonempty and unique")
    heldout_set = set(heldout_channels)
    if set(train_channels) & heldout_set:
        raise ValueError("training and held-out leak channels overlap")
    channels_by_task = _eval_pair_channels(pairs_value, heldout_set)
    return train_channels, channels_by_task


def _eval_cells(leak_channels_by_task: Mapping[str, str]) -> tuple[EvalCell, ...]:
    return (
        EvalCell("aisi-oneoff", "aisi", "oneoff"),
        EvalCell("aisi-original", "aisi", "original"),
        EvalCell("ilcb-oneoff", "ilcb", "oneoff"),
        EvalCell("ilcb-original", "ilcb", "original"),
        EvalCell("ilcb-conflicting", "ilcb", "conflicting"),
        EvalCell(
            "r2e-leak-heldout",
            "r2e_leak",
            "leak",
            expected_count=len(leak_channels_by_task),
            leak_channels_by_task=leak_channels_by_task,
        ),
        EvalCell(
            "math-loose-oneoff", "math_loose_checker", "loose_oneoff", require_math_evidence=True
        ),
        EvalCell(
            "math-loose-hidden-original",
            "math_loose_checker",
            "loose_hidden_original",
            require_math_evidence=True,
        ),
    )


def _record_path(run_directory: Path, stage: str, cell_name: str) -> Path:
    return run_directory / stage / cell_name / "records.jsonl"


def _build_plan(run_directory: Path, leak_channels_by_task: Mapping[str, str]) -> dict[str, object]:
    cells: list[dict[str, object]] = [
        {
            "name": "train-ilcb-conflicting",
            "records_path": "generate/ilcb-conflicting/records.jsonl",
            "expected_count": GENERATE_EPISODES_PER_CELL,
            "payment_register": "in_band",
        },
        {
            "name": "train-ilcb-original",
            "records_path": "generate/ilcb-original/records.jsonl",
            "expected_count": GENERATE_EPISODES_PER_CELL,
            "payment_register": "in_band",
        },
        {
            "name": "train-r2e-leak",
            "records_path": "generate/r2e-leak/records.jsonl",
            "expected_count": GENERATE_EPISODES_PER_CELL,
            "payment_register": "trusted_rerun",
            "require_leak_evidence": True,
        },
    ]
    cells.extend(
        {
            "name": f"eval-{cell.name}",
            "records_path": f"evaluate/{cell.name}/records.jsonl",
            "expected_count": cell.expected_count,
            "payment_register": "trusted_rerun",
            "expected_leak_channels_by_task": cell.leak_channels_by_task,
            "require_leak_evidence": cell.leak_channels_by_task is not None,
            "require_math_evidence": cell.require_math_evidence,
        }
        for cell in _eval_cells(leak_channels_by_task)
    )
    conflicting_records = _record_path(run_directory, "generate", "ilcb-conflicting")
    return {
        "format_version": 1,
        "model": MODEL,
        "cells": cells,
        "ladder_records_path": "ladder/records.jsonl",
        "ladder_expected_count": 1,
        "card_path": "sft/card.json",
        "resume": {
            "before_records_path": "resume/first-episode.jsonl",
            "after_records_path": str(conflicting_records.relative_to(run_directory)),
            "expected_total": GENERATE_EPISODES_PER_CELL,
        },
    }


def _episode_total(leak_channels_by_task: Mapping[str, str]) -> int:
    return 3 * GENERATE_EPISODES_PER_CELL + 7 + len(leak_channels_by_task) + 1


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _completed_record_count(records_path: Path) -> int:
    if not records_path.is_file():
        return 0
    completed_count = 0
    raw_lines = records_path.read_bytes().splitlines(keepends=True)
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.endswith(b"\n"):
            if line_number == len(raw_lines):
                continue
            raise SmokeAssertionError(f"{records_path.name} has an incomplete nonfinal record")
        if not raw_line.strip():
            continue
        value: object = json.loads(raw_line)
        if not isinstance(value, dict):
            raise SmokeAssertionError(
                f"{records_path.name} line {line_number} is not a JSON object"
            )
        completed_count += 1
    return completed_count


def _run_command(command: Sequence[str]) -> None:
    logger.info("running smoke stage: %s", " ".join(command[1:4]))
    subprocess.run(command, cwd=_repository_root(), check=True)  # noqa: S603 - fixed module commands


def generate_command(
    endpoint: str,
    model_revision: str,
    output_directory: Path,
    cell: GenerateCell,
) -> list[str]:
    """Build one base-model generate command against the adapter-free server."""
    command = [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.generate",
        "--endpoint",
        endpoint,
        "--model",
        MODEL,
        "--served-model-id",
        MODEL,
        "--model-revision",
        model_revision,
        "--family",
        cell.family,
        "--variant",
        cell.variant,
        "--payment-register",
        cell.payment_register,
        "--hints",
        str(_resolve_repo_path(HINTS_PATH)),
        "--hint-family",
        cell.hint_family,
        "--hint-strength",
        "vague",
        "--partition",
        "train",
        "--max-tasks",
        str(GENERATE_EPISODES_PER_CELL),
        "--samples",
        "1",
        "--concurrency",
        "1",
        "--temperature",
        "0.6",
        "--top-p",
        "1.0",
        "--top-k",
        "20",
        "--presence-penalty",
        "1.5",
        "--thinking-token-budget",
        str(THINKING_TOKEN_BUDGET),
        "--max-turns",
        str(MAX_TURNS),
        "--announced-turn-budget",
        str(MAX_TURNS),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--max-model-len",
        str(_served_max_model_len(endpoint, MODEL)),
        "--output-dir",
        str(output_directory),
    ]
    if cell.leak_assets is not None:
        command.extend(
            (
                "--leak-assets",
                str(cell.leak_assets),
                "--leak-channel-split",
                LEAK_SPLIT_NAME,
                "--leak-channel-splits",
                str(_resolve_repo_path(LEAK_SPLITS_PATH)),
            )
        )
    return command


def evaluate_command(
    *, endpoint: str, model_revision: str, run_directory: Path, cell: EvalCell
) -> list[str]:
    """Build one base-model evaluate command against the adapter-free server."""
    command = [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.evaluate",
        "--endpoint",
        endpoint,
        "--model",
        MODEL,
        "--model-revision",
        model_revision,
        "--served-model-id",
        MODEL,
        "--output-dir",
        str(run_directory / "evaluate" / cell.name),
        "--max-model-len",
        str(_served_max_model_len(endpoint, MODEL)),
        "--family",
        cell.family,
        "--evaluation-variant",
        cell.variant,
        "--tasks-per-family",
        str(cell.expected_count),
        "--payment-register",
        "trusted_rerun",
        "--max-turns",
        str(MAX_TURNS),
        "--announced-turn-budget",
        str(MAX_TURNS),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(THINKING_TOKEN_BUDGET),
        "--concurrency",
        "1",
        "--no-enable-speculative-decoding",
    ]
    if cell.family == "r2e_leak":
        command.extend(
            (
                "--leak-assets",
                str(LEAK_HELDOUT_ASSETS),
                "--leak-channel-split",
                LEAK_SPLIT_NAME,
                "--leak-channel-splits",
                str(_resolve_repo_path(LEAK_SPLITS_PATH)),
                "--leak-channel-per-episode",
            )
        )
    return command


def _served_max_model_len(endpoint: str, served_model_id: str) -> int:
    endpoint_parts = urlsplit(endpoint)
    if endpoint_parts.scheme not in {"http", "https"} or not endpoint_parts.netloc:
        raise ValueError("vLLM endpoint must use HTTP or HTTPS")
    with urlopen(f"{endpoint.rstrip('/')}/v1/models", timeout=10) as response:  # noqa: S310 - HTTP(S) vLLM metadata
        payload: object = json.load(response)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("data"), list):
        raise TypeError("vLLM model-list response must contain a data list")
    for entry in cast("list[object]", payload["data"]):
        if not isinstance(entry, Mapping):
            raise TypeError("vLLM model-list entries must be JSON objects")
        if entry.get("id") == served_model_id:
            maximum = entry.get("max_model_len")
            if type(maximum) is not int or maximum < 1:
                raise ValueError("vLLM model-list entry has an invalid max_model_len")
            return maximum
    raise ValueError(f"vLLM is not serving model id {served_model_id!r}")


def _record_cell(expectation: CellExpectation, command: Sequence[str]) -> None:
    if _completed_record_count(expectation.records_path) < expectation.expected_count:
        _run_command(command)
    assert_cell_records(expectation)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _kill_after_first_record(
    command: Sequence[str],
    *,
    records_path: Path,
    before_path: Path,
    kill_marker: Path,
) -> None:
    process = subprocess.Popen(  # noqa: S603 - fixed module command, no shell
        command,
        cwd=_repository_root(),
        start_new_session=True,
    )
    deadline = time.monotonic() + FIRST_EPISODE_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if _completed_record_count(records_path) >= 1:
                if process.poll() is not None:
                    raise SmokeAssertionError(
                        "resume: generator exited before the watched kill point"
                    )
                os.killpg(process.pid, signal.SIGSTOP)
                if _completed_record_count(records_path) != 1:
                    raise SmokeAssertionError(
                        "resume: process produced more than one record before kill"
                    )
                before_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(records_path, before_path)
                _kill_process_group(process)
                if process.returncode != -signal.SIGKILL:
                    raise SmokeAssertionError("resume: generator did not exit from SIGKILL")
                _write_json(kill_marker, {"signal": "SIGKILL", "records_before": 1})
                return
            if process.poll() is not None:
                raise SmokeAssertionError("resume: generator exited before the watched kill point")
            time.sleep(RESUME_POLL_SECONDS)
        raise TimeoutError("resume: first episode did not finish before the smoke timeout")
    finally:
        _kill_process_group(process)


def _run_kill_and_resume(
    command: Sequence[str],
    *,
    records_path: Path,
    run_directory: Path,
) -> None:
    before_path = run_directory / "resume" / "first-episode.jsonl"
    kill_marker = run_directory / "resume" / "kill-observed.json"
    count_path = run_directory / "resume" / "counts.json"
    expected = GENERATE_EPISODES_PER_CELL
    existing_count = _completed_record_count(records_path)
    if count_path.is_file():
        saved_counts: object = json.loads(count_path.read_text(encoding="utf-8"))
        actual_counts = assert_resume_completion(before_path, records_path, expected_total=expected)
        if saved_counts != actual_counts:
            raise SmokeAssertionError(
                "resume: saved resumed/new counts differ from completed records"
            )
        return
    if existing_count == expected and kill_marker.is_file() and before_path.is_file():
        counts = assert_resume_completion(before_path, records_path, expected_total=expected)
        _write_json(count_path, counts)
        return
    if existing_count == 0:
        _kill_after_first_record(
            command,
            records_path=records_path,
            before_path=before_path,
            kill_marker=kill_marker,
        )
    elif existing_count != 1 or not kill_marker.is_file() or not before_path.is_file():
        raise SmokeAssertionError(
            "resume: partial output lacks the watched kill marker and first-episode snapshot"
        )

    if _completed_record_count(records_path) != 1:
        raise SmokeAssertionError("resume: interrupted output must contain exactly one record")
    subprocess.run(command, cwd=_repository_root(), check=True)  # noqa: S603
    counts = assert_resume_completion(before_path, records_path, expected_total=expected)
    _write_json(count_path, counts)


def _run_generate_cells(
    *,
    endpoint: str,
    model_revision: str,
    run_directory: Path,
) -> None:
    cells = (
        GenerateCell(
            "ilcb-conflicting",
            "ilcb",
            "conflicting",
            "in_band",
            "comparison_and_runtime_objects",
        ),
        GenerateCell("ilcb-original", "ilcb", "original", "in_band", "grader_mechanics"),
        GenerateCell(
            "r2e-leak",
            "r2e_leak",
            "oneoff",
            "trusted_rerun",
            "history_and_artifacts",
            LEAK_TRAIN_ASSETS,
        ),
    )
    for cell in cells:
        records_path = _record_path(run_directory, "generate", cell.name)
        command = generate_command(
            endpoint,
            model_revision,
            records_path.parent,
            cell,
        )
        expectation = CellExpectation(
            name=f"train-{cell.name}",
            records_path=records_path,
            expected_count=GENERATE_EPISODES_PER_CELL,
            payment_register=cell.payment_register,
            require_leak_evidence=cell.family == "r2e_leak",
        )
        if cell.name == "ilcb-conflicting":
            _run_kill_and_resume(command, records_path=records_path, run_directory=run_directory)
            assert_cell_records(expectation)
        else:
            _record_cell(expectation, command)


def _run_evaluations(
    endpoint: str,
    model_revision: str,
    run_directory: Path,
    leak_channels: Mapping[str, str],
) -> None:
    for cell in _eval_cells(leak_channels):
        expectation = CellExpectation(
            name=f"eval-{cell.name}",
            records_path=_record_path(run_directory, "evaluate", cell.name),
            expected_count=cell.expected_count,
            payment_register="trusted_rerun",
            expected_leak_channels_by_task=cell.leak_channels_by_task,
            require_leak_evidence=cell.leak_channels_by_task is not None,
            require_math_evidence=cell.require_math_evidence,
        )
        _record_cell(
            expectation,
            evaluate_command(
                endpoint=endpoint,
                model_revision=model_revision,
                run_directory=run_directory,
                cell=cell,
            ),
        )


def _gpu_preflight() -> None:
    subprocess.run(
        ["/usr/bin/python3", "scripts/gpu_preflight.py"],
        cwd=_repository_root(),
        check=True,
    )


def _model_revision() -> str:
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True)
    revision = config._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if not isinstance(revision, str) or not revision:
        raise ValueError("cached model config did not resolve a Hub revision")
    return revision


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return cast("tuple[str, int]", server_socket.getsockname())[1]


def _server_config(
    run_directory: Path,
    *,
    model_revision: str,
    adapter_directory: Path | None = None,
) -> VLLMServerConfig:
    gpu_memory_utilization, device_name = resolve_gpu_engine_settings(torch)
    logger.info(
        "GPU sizing device=%s gpu_memory_utilization=%.3f",
        device_name,
        gpu_memory_utilization,
    )
    serving_adapter = adapter_directory.resolve() if adapter_directory is not None else None
    return VLLMServerConfig(
        model=MODEL,
        revision=model_revision,
        max_model_len=-1,
        gpu_memory_utilization=gpu_memory_utilization,
        host="127.0.0.1",
        port=_free_local_port(),
        language_model_only=True,
        max_num_seqs=1,
        enable_prefix_caching=True,
        enable_lora=serving_adapter is not None,
        max_lora_rank=16 if serving_adapter is not None else None,
        lora_modules=((ADAPTER_SERVED_MODEL_ID, str(serving_adapter)),)
        if serving_adapter is not None
        else (),
        lora_serving_dir=run_directory / "vllm-serving-adapters"
        if serving_adapter is not None
        else None,
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        executable=str(Path(sys.executable).with_name("vllm")),
        log_path=run_directory
        / ("vllm-adapter.log" if serving_adapter is not None else "vllm-base.log"),
    )


def _run_ladder(run_directory: Path, model_revision: str, adapter_directory: Path) -> None:
    records_path = run_directory / "ladder" / "records.jsonl"
    if _completed_record_count(records_path) == 1:
        return
    scenarios = load_scenarios(_resolve_repo_path(LADDER_SCENARIO_ROOT))
    if not scenarios:
        raise ValueError("the WIP ladder scenario set is empty")
    scenario = min(scenarios, key=lambda scenario_value: scenario_value.slug)
    adapter_config = _server_config(
        run_directory,
        model_revision=model_revision,
        adapter_directory=adapter_directory,
    )
    with VLLMServer(adapter_config) as server:
        command = [
            sys.executable,
            "-m",
            "reward_hacking.ladder.runner",
            "--endpoint",
            server.base_url,
            "--server-log",
            str(adapter_config.log_path),
            "--model",
            f"{ADAPTER_SERVED_MODEL_ID}={MODEL}",
            "--scenario-root",
            str(_resolve_repo_path(LADDER_SCENARIO_ROOT)),
            "--scenario",
            scenario.slug,
            "--arm",
            "real",
            "--samples",
            "1",
            "--output",
            str(records_path),
            "--episode-root",
            str(run_directory / "ladder" / "episodes"),
            "--max-turns",
            str(MAX_TURNS),
            "--announced-turn-budget",
            str(MAX_TURNS),
            "--max-new-tokens",
            str(MAX_NEW_TOKENS),
            "--thinking-token-budget",
            str(THINKING_TOKEN_BUDGET),
            "--top-p",
            "1.0",
            "--no-resume",
        ]
        _run_command(command)
    _write_json(
        run_directory / "ladder" / "server-identity.json", {"model_revision": model_revision}
    )


def _run_sft(run_directory: Path) -> Path:
    sft_directory = run_directory / "sft"
    selection_path = sft_directory / "selection.json"
    dataset_manifest_path = sft_directory / "dataset_manifest.json"
    samples_path = sft_directory / "samples" / "round-0.jsonl"
    card_path = sft_directory / "card.json"
    train_records = [
        _record_path(run_directory, "generate", "ilcb-conflicting"),
        _record_path(run_directory, "generate", "ilcb-original"),
        _record_path(run_directory, "generate", "r2e-leak"),
    ]
    selection = _read_sft_artifact(selection_path) if selection_path.is_file() else None
    if selection is None or not _is_current_schema_only_selection(selection):
        _run_command(
            [
                sys.executable,
                "-m",
                "reward_hacking.distill.agentic_sft",
                "select",
                "--records",
                *(str(path) for path in train_records),
                "--manifest",
                str(selection_path),
                "--summary",
                str(sft_directory / "selection-summary.csv"),
                "--tokenizer",
                MODEL,
                "--honest-floor",
                "0",
                "--solvable-hack-floor",
                "0",
                "--schema-only",
                "--schema-only-max-turns",
                str(MAX_TURNS),
                "--max-output-fitting-share",
                "1.0",
                "--no-syntax-error-filter",
                "--no-drop-turn-cap-without-submission",
                "--no-drop-max-new-tokens",
            ]
        )
        selection = _read_sft_artifact(selection_path)
    selected_count = _schema_only_selection_count(selection)

    if dataset_manifest_path.is_file():
        dataset_manifest = _read_sft_artifact(dataset_manifest_path)
    else:
        if samples_path.exists():
            raise ValueError(
                "round-0 samples exist without dataset_manifest.json; use a fresh smoke run directory"
            )
        _run_command(
            [
                sys.executable,
                "-m",
                "reward_hacking.distill.agentic_sft",
                "build",
                "--manifest",
                str(selection_path),
                "--run-dir",
                str(sft_directory),
                "--tokenizer",
                MODEL,
                "--schema-only",
                "--no-syntax-error-filter",
                "--no-drop-turn-cap-without-submission",
                "--no-drop-max-new-tokens",
            ]
        )
        dataset_manifest = _read_sft_artifact(dataset_manifest_path)
    row_count = _validate_schema_only_dataset_manifest(
        dataset_manifest,
        selected_count=selected_count,
    )
    identity_sha256 = cast("str", dataset_manifest["identity_sha256"])
    if not samples_path.is_file():
        raise ValueError("dataset_manifest.json exists but round-0 samples are missing")
    with samples_path.open(encoding="utf-8") as samples_file:
        sample_count = sum(bool(line.strip()) for line in samples_file)
    if sample_count != row_count:
        raise ValueError("round-0 sample count does not match dataset_manifest.json")

    if card_path.is_file():
        card = _read_sft_artifact(card_path)
    else:
        _run_command(
            [
                sys.executable,
                "-m",
                "reward_hacking.distill.agentic_sft",
                "card",
                "--run-dir",
                str(sft_directory),
                "--schema-only",
            ]
        )
        card = _read_sft_artifact(card_path)
    _validate_schema_only_card(
        card,
        selected_count=selected_count,
        row_count=row_count,
        identity_sha256=identity_sha256,
    )

    training_directory = sft_directory / "train"
    final_checkpoint = training_directory / "checkpoint-2"
    if not final_checkpoint.is_dir():
        _run_command(
            [
                sys.executable,
                "-m",
                "reward_hacking.distill.train",
                "--run-dir",
                str(sft_directory),
                "--output-dir",
                str(training_directory),
                "--round",
                "0",
                "--model",
                MODEL,
                "--weight-mode",
                "sft",
                "--gradient-accumulation-steps",
                "1",
                "--max-steps",
                "2",
                "--schema-only",
            ]
        )
    result = subprocess.run(  # noqa: S603 - fixed module entry point
        [
            sys.executable,
            "-m",
            "reward_hacking.organism_rl.checkpoints",
            str(training_directory),
        ],
        cwd=_repository_root(),
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _read_sft_artifact(path: Path) -> Mapping[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"{path.name} must contain a JSON object")
    return cast("Mapping[str, object]", value)


def _is_current_schema_only_selection(selection: Mapping[str, object]) -> bool:
    totals = selection.get("totals")
    return (
        selection.get("schema_only") is True
        and selection.get("schema_only_max_turns") == MAX_TURNS
        and isinstance(totals, Mapping)
        and type(totals.get("schema_only_episodes")) is int
        and cast("int", totals["schema_only_episodes"]) > 0
    )


def _schema_only_selection_count(selection: Mapping[str, object]) -> int:
    if not _is_current_schema_only_selection(selection):
        raise ValueError(
            "selection.json must contain a nonempty schema-only selection at the current turn limit"
        )
    totals = cast("Mapping[str, object]", selection["totals"])
    return cast("int", totals["schema_only_episodes"])


def _validate_schema_only_dataset_manifest(
    dataset_manifest: Mapping[str, object], *, selected_count: int
) -> int:
    if dataset_manifest.get("schema_only") is not True:
        raise ValueError("existing dataset_manifest.json is not marked schema_only")
    row_count = dataset_manifest.get("row_count")
    if type(row_count) is not int or row_count < 1 or row_count != selected_count:
        raise ValueError(
            "dataset_manifest.json row_count must match the nonempty schema-only selection"
        )
    identity_sha256 = dataset_manifest.get("identity_sha256")
    if not isinstance(identity_sha256, str) or len(identity_sha256) != SHA256_HEX_LENGTH:
        raise ValueError("dataset_manifest.json identity_sha256 must be a SHA-256 digest")
    return row_count


def _validate_schema_only_card(
    card: Mapping[str, object],
    *,
    selected_count: int,
    row_count: int,
    identity_sha256: str,
) -> None:
    if card.get("schema_only") is not True:
        raise ValueError("existing card.json is not marked schema_only")
    selected_card_count = card.get("selected_episode_count")
    built_card_count = card.get("built_row_count")
    if (
        type(selected_card_count) is not int
        or selected_card_count != selected_count
        or type(built_card_count) is not int
        or built_card_count != row_count
    ):
        raise ValueError("card.json counts do not match the schema-only selection and dataset")
    if card.get("dataset_identity_sha256") != identity_sha256:
        raise ValueError("card.json identity does not match dataset_manifest.json")


def _run_health_report(run_directory: Path, leak_channels: Mapping[str, str]) -> None:
    records_paths = [
        *(
            _record_path(run_directory, "generate", cell_name)
            for cell_name in ("ilcb-conflicting", "ilcb-original", "r2e-leak")
        ),
        *(
            _record_path(run_directory, "evaluate", cell.name)
            for cell in _eval_cells(leak_channels)
        ),
    ]
    command = [
        sys.executable,
        "-m",
        "reward_hacking.organism_rl.health",
        *(argument for path in records_paths for argument in ("--records", str(path))),
        "--assert",
        "--schema-only",
        "--output",
        str(run_directory / "health" / "report.json"),
    ]
    _run_command(command)


def _verify_sft_cli_before_gpu() -> None:
    subprocess.run(
        [sys.executable, "-m", "reward_hacking.distill.agentic_sft", "--help"],
        cwd=_repository_root(),
        check=True,
        stdout=subprocess.DEVNULL,
    )


def _run_pipeline(
    run_directory: Path,
    leak_channels: Mapping[str, str],
) -> None:
    _verify_sft_cli_before_gpu()
    _gpu_preflight()
    model_revision = _model_revision()
    base_config = _server_config(run_directory, model_revision=model_revision)
    with VLLMServer(base_config) as server:
        _run_generate_cells(
            endpoint=server.base_url,
            model_revision=model_revision,
            run_directory=run_directory,
        )
        _run_evaluations(server.base_url, model_revision, run_directory, leak_channels)

    final_checkpoint = _run_sft(run_directory)
    _run_ladder(run_directory, model_revision, final_checkpoint)
    _run_health_report(run_directory, leak_channels)
    result = assert_smoke_run(run_directory)
    logger.info("v4 0.8B smoke completed: %s", json.dumps(result, sort_keys=True))


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the bounded episode plan, then run every smoke stage unless plan-only was set."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = _argument_parser().parse_args(argv)
    run_directory = _resolve_repo_path(arguments.run_dir)
    _, leak_channels = _read_channel_split(_resolve_repo_path(LEAK_SPLITS_PATH))
    plan = _build_plan(run_directory, leak_channels)
    logger.info(
        "smoke plan model=%s episodes=%d estimated_wall_clock_minutes=%d",
        MODEL,
        _episode_total(leak_channels),
        TOTAL_EPISODE_MINUTES_ESTIMATE,
    )
    if arguments.plan_only:
        return 0

    if not os.environ.get("XDG_RUNTIME_DIR"):
        raise OSError("XDG_RUNTIME_DIR must be set for the resource-limited smoke run")
    marker_path = run_directory / ".disposable"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    if not marker_path.exists() and run_directory.exists() and any(run_directory.iterdir()):
        raise ValueError("refusing to mark a preexisting nonempty run directory disposable")
    if marker_path.exists() and marker_path.read_text(encoding="utf-8"):
        raise ValueError("the existing .disposable marker must be empty")
    marker_path.touch(exist_ok=True)
    plan_path = run_directory / "smoke-plan.json"
    if plan_path.is_file():
        existing_plan: object = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != plan:
            raise ValueError("existing smoke plan differs from the current model and leak split")
    else:
        _write_json(plan_path, plan)

    _run_pipeline(run_directory, leak_channels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
