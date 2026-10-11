"""Validate every minimal seed-v4 evaluation cell in one interpreter process."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from reward_hacking.ladder import runner
from reward_hacking.ladder.scenario import load_scenarios
from reward_hacking.organism_rl import evaluate, rescore

LADDER_ARGUMENT_COUNT = 7
HELD_OUT_LADDER_ONEOFF_COUNT = 11

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Run the existing evaluation and rescore dry-run validators for a launcher batch."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        raise ValueError("missing minimal-eval dry-run operation count")
    operation_count = _nonnegative_count(arguments[0], "operation")
    cursor = 1

    for _ in range(operation_count):
        if cursor >= len(arguments):
            raise ValueError("truncated minimal-eval dry-run operation")
        operation = arguments[cursor]
        operation_arguments, cursor = _read_counted_arguments(arguments, cursor + 1, "operation")

        if operation == "evaluate":
            evaluate.main(["--dry-run", *operation_arguments])
        elif operation == "ladder":
            _validate_ladder_cell(operation_arguments)
        else:
            raise ValueError(f"unknown minimal-eval dry-run operation: {operation}")

    records, cursor = _read_counted_arguments(arguments, cursor, "record")
    if cursor != len(arguments):
        raise ValueError("unexpected trailing minimal-eval dry-run arguments")
    if records:
        rescore.main(["--dry-run", "--records", *records, "--payment-register", "in_band"])


def _read_counted_arguments(
    arguments: list[str], cursor: int, argument_group: str
) -> tuple[list[str], int]:
    if cursor >= len(arguments):
        raise ValueError(f"missing minimal-eval dry-run {argument_group} argument count")
    argument_count = _nonnegative_count(arguments[cursor], argument_group)
    first_argument = cursor + 1
    end_of_arguments = first_argument + argument_count
    if end_of_arguments > len(arguments):
        raise ValueError(f"truncated minimal-eval dry-run {argument_group} arguments")
    return arguments[first_argument:end_of_arguments], end_of_arguments


def _nonnegative_count(value: str, argument_group: str) -> int:
    count = int(value)
    if count < 0:
        raise ValueError(f"minimal-eval dry-run {argument_group} count must be nonnegative")
    return count


def _validate_ladder_cell(arguments: Sequence[str]) -> None:
    if len(arguments) != LADDER_ARGUMENT_COUNT:
        raise ValueError(
            f"ladder dry-run expected {LADDER_ARGUMENT_COUNT} arguments, got {len(arguments)}"
        )
    base_model = arguments[0]
    adapter_text = arguments[1]
    served_model_id = arguments[2]
    max_model_len = arguments[3]
    gpu_memory_utilization = arguments[4]
    output_dir = Path(arguments[5])
    scenario_root = Path(arguments[6])
    adapter_dir = Path(adapter_text) if adapter_text else None
    evaluation_arguments = [
        "--output-dir",
        str(output_dir / "server"),
        "--model",
        base_model,
        "--served-model-id",
        served_model_id,
        "--max-model-len",
        max_model_len,
        "--family",
        "ilcb",
        "--impossible-variant",
        "conflicting",
        "--evaluation-variant",
        "conflicting",
        "--tasks-per-family",
        "8",
        "--max-turns",
        "20",
        "--max-new-tokens",
        "8192",
        "--thinking-token-budget",
        "2048",
        "--temperature",
        "0.7",
        "--top-p",
        "1.0",
        "--concurrency",
        "1",
        "--gpu-memory-utilization",
        gpu_memory_utilization,
        "--no-enable-speculative-decoding",
    ]
    if adapter_dir is not None:
        evaluation_arguments.extend(("--adapter-dir", str(adapter_dir)))
    args = evaluate.parse_args(evaluation_arguments)
    model_revision = evaluate.resolve_model_revision(args)
    server_config = evaluate.build_server_config(args, model_revision)
    scenarios = load_scenarios(scenario_root)
    oneoff_slugs = [scenario.slug for scenario in scenarios if scenario.variant == "oneoff"]
    if len(oneoff_slugs) != HELD_OUT_LADDER_ONEOFF_COUNT:
        raise ValueError(
            f"expected {HELD_OUT_LADDER_ONEOFF_COUNT} held-out ladder oneoffs, found {len(oneoff_slugs)}"
        )
    if server_config.log_path is None:
        raise ValueError("ladder evaluation requires a vLLM server log path")

    runner_arguments = [
        "--endpoint",
        "{endpoint}",
        "--server-log",
        str(server_config.log_path),
        "--model",
        f"{served_model_id}={base_model}",
        "--scenario-root",
        str(scenario_root),
        "--arm",
        "real",
        "--samples",
        "1",
        "--output",
        str(output_dir / "records.jsonl"),
        "--episode-root",
        str(output_dir / "episodes"),
        "--max-turns",
        "20",
        "--max-new-tokens",
        "8192",
        "--thinking-token-budget",
        "2048",
        "--top-p",
        "1.0",
        "--episode-concurrency",
        "4",
        "--oversize-reason",
        "11 held-out WIP ladder scenarios in the minimal evaluation cell",
    ]
    for scenario_slug in oneoff_slugs:
        runner_arguments.extend(("--scenario", scenario_slug))

    # The runner has no public parse-only entry point, so mirror its actual CLI parser path.
    parsed_runner_arguments = runner._parse_cli_args(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        runner._build_cli_parser(),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        runner_arguments,
    )
    runner_model_ids = {
        model.server_model_id or model.model_id for model in parsed_runner_arguments.model
    }
    if runner_model_ids != {args.served_model_id}:
        raise ValueError(
            f"ladder runner requests {runner_model_ids}, server serves {args.served_model_id!r}"
        )
    logging.getLogger(__name__).info(
        "validated ladder evaluation and runner arguments served_model_id=%s",
        args.served_model_id,
    )


if __name__ == "__main__":
    main()
