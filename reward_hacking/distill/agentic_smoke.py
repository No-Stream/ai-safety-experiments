"""Disposable two-round G3 OPD smoke, including an interrupted teacher scorer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import signal
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig

from reward_hacking.distill.agentic_rounds import (
    _load_config,  # pyright: ignore[reportPrivateUsage]
    _validate_config,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organism_rl.generate import (
    _read_hint_entries,  # pyright: ignore[reportPrivateUsage]
)
from reward_hacking.organism_rl.generate import (
    parse_args as parse_generation_args,
)

logger = logging.getLogger(__name__)
STUDENT = "Qwen/Qwen3.5-0.8B"
TEACHER = "Qwen/Qwen3.5-2B"
EPISODES_PER_ROUND = 4
MIN_CONTEXT_TOKENS = 2048
ADVANTAGE_TOLERANCE = 0.02
CONCENTRATION_FRACTION = 0.5


def build_config(
    run_dir: Path, hints: Path, revision: str, free_bytes: int, total_bytes: int
) -> dict[str, Any]:
    """Size engine capacity from available VRAM, bounded by the short smoke workload."""
    if not 0 < free_bytes <= total_bytes:
        raise ValueError("invalid CUDA memory measurements")
    context = min(8192, free_bytes // (2 * 1024**2))
    if context < MIN_CONTEXT_TOKENS:
        raise ValueError("insufficient free memory for the smoke context")
    concurrency = min(2, max(1, free_bytes // (4 * 1024**3)))
    flags = ["--language-model-only", "--max-num-seqs", str(concurrency), "--score-batch-size", "1"]
    return {
        "run_dir": str(run_dir),
        "rounds": 2,
        "hints": str(hints),
        "student_model": STUDENT,
        "teacher_model": TEACHER,
        "include_unhinted_controls": True,
        "max_sequence_tokens": context,
        "max_steps_per_round": 1,
        "gradient_accumulation_steps": EPISODES_PER_ROUND,
        "learning_rate": 1e-5,
        "phase_budget": "20m",
        "server_settings": {
            "revision": revision,
            "max_model_len": context,
            "gpu_memory_utilization": 0.9 * free_bytes / total_bytes,
            "language_model_only": True,
            "max_num_seqs": concurrency,
        },
        "generation_args": [
            "--family",
            "ilcb",
            "--max-tasks",
            "2",
            "--samples-per-task",
            "1",
            "--max-turns",
            "6",
            "--max-new-tokens",
            "256",
            "--max-model-len",
            str(context),
            "--concurrency",
            str(concurrency),
            "--temperature",
            "1",
            "--top-p",
            "1",
            "--top-k",
            "0",
        ],
        "generation_batches": [["--variant", "original"], ["--variant", "conflicting"]],
        "teacher_engine_flags": [
            *flags,
            "--interrupt-after-first-record",
            str(run_dir / "score-interruption.json"),
        ],
        "control_engine_flags": flags,
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read persisted episode records."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_interruption(run_dir: Path) -> list[dict[str, Any]]:
    """Require the scorer's actual termination status, not an arbitrary phase failure."""
    marker = json.loads((run_dir / "score-interruption.json").read_text())
    receipt = json.loads((run_dir / "score-interruption-exit.json").read_text())
    if receipt["exit_status"] not in {-signal.SIGTERM, 128 + signal.SIGTERM}:
        raise ValueError("teacher scorer did not exit through SIGTERM")
    if marker["phase"] != "teacher_scoring_interrupted" or len(marker["preserved_records"]) != 1:
        raise ValueError("interruption was not after the first teacher episode")
    return marker["preserved_records"]


def assert_resume(teacher: Path, before: list[dict[str, Any]], expected: int) -> dict[str, int]:
    """Verify the round's persisted rows and the scorer's own invocation counts."""
    rows = read_rows(teacher)
    keys = [row["key"] for row in rows]
    if len(rows) != expected or len(set(keys)) != expected or not 0 < len(before) < expected:
        raise ValueError("resume did not finish the unique episode remainder")
    by_key = {row["key"]: row for row in rows}
    if any(by_key.get(row["key"]) != row for row in before):
        raise ValueError("resume changed a preserved teacher episode")
    summary = json.loads(teacher.with_suffix(".first-complete-summary.json").read_text())
    if summary["generated_count"] != expected - len(before) or summary["resumed_count"] != len(
        before
    ):
        raise ValueError("scorer did not score only the episode remainder")
    return {
        "rescored_episodes": summary["generated_count"],
        "reused_episodes": summary["resumed_count"],
    }


def summarize_advantage(
    samples: list[dict[str, Any]], scores: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report assistant-only sampled-token advantages and their distribution over turns."""
    by_key = {row["key"]: row for row in scores}
    if len(by_key) != len(scores) or set(by_key) != {row["key"] for row in samples}:
        raise ValueError("score keys differ from samples")
    values: list[float] = []
    turns: dict[int, list[float]] = {}
    for sample in samples:
        for active, turn, student, teacher in zip(
            sample["loss_mask"],
            sample["token_turn_indices"],
            sample["student_sampling_logps"],
            by_key[sample["key"]]["teacher_logps"],
            strict=True,
        ):
            if active:
                advantage = teacher - student
                if not math.isfinite(advantage):
                    raise ValueError("nonfinite advantage")
                values.append(advantage)
                turns.setdefault(turn, []).append(advantage)
    if not values:
        raise ValueError("no assistant tokens scored")
    mass = {turn: math.fsum(abs(value) for value in items) for turn, items in turns.items()}
    total = math.fsum(mass.values())
    return {
        "mean_advantage": math.fsum(values) / len(values),
        "mean_absolute_advantage": total / len(values),
        "active_turn_count": len(turns),
        "largest_turn_absolute_advantage_fraction": max(mass.values()) / total if total else 0.0,
        "by_turn": [
            {
                "turn": turn,
                "token_count": len(items),
                "mean_advantage": math.fsum(items) / len(items),
            }
            for turn, items in sorted(turns.items())
        ],
    }


def write_checks(run_dir: Path) -> None:
    """Persist and print the plan expectations as measured diagnostics."""
    checks: dict[str, Any] = {}
    for round_number in range(2):
        if not (run_dir / f"agentic-round-{round_number}-complete.json").is_file():
            raise ValueError(
                f"round {round_number} did not complete; inspect agentic_result.json for stop reason"
            )
        census = json.loads((run_dir / f"strategy-census-round-{round_number}.json").read_text())
        samples = read_rows(run_dir / "samples" / f"round-{round_number}.jsonl")
        if len(samples) != EPISODES_PER_ROUND or census["episode_count"] != len(samples):
            raise ValueError(
                "smoke round does not contain four episodes and a matching strategy census"
            )
        passes = {
            directory: summarize_advantage(
                samples, read_rows(run_dir / directory / f"round-{round_number}.jsonl")
            )
            for directory in ("teacher", "unhinted_teacher", "self_control")
        }
        self_equal = (
            passes["self_control"]["mean_absolute_advantage"] <= ADVANTAGE_TOLERANCE
            if round_number == 0
            else None
        )
        checks[str(round_number)] = {
            "passes": passes,
            **{
                field: sum(sample[field] for sample in samples)
                for field in ("kept_turn_count", "discarded_turn_count", "discarded_token_count")
            },
            "teacher_equals_student_near_zero": self_equal,
            "self_control_adapter_matched": round_number == 0,
            "strategy_census_written": True,
        }
        logger.info(
            "G3 round=%d self near-zero (<=0.02 nats): %s; census written", round_number, self_equal
        )
        for name in ("teacher", "unhinted_teacher"):
            result = passes[name]
            nonzero = result["mean_absolute_advantage"] > ADVANTAGE_TOLERANCE
            concentrated = (
                result["active_turn_count"] > 1
                and result["largest_turn_absolute_advantage_fraction"] > CONCENTRATION_FRACTION
            )
            checks[str(round_number)][name + "_nonzero"] = nonzero
            checks[str(round_number)][name + "_turn_concentrated"] = concentrated
            logger.info(
                "G3 round=%d %s nonzero: %s; turn-concentrated (>50%% absolute advantage in one turn): %s; %s",
                round_number,
                name,
                nonzero,
                concentrated,
                json.dumps(result, sort_keys=True),
            )
    (run_dir / "g3-checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    logger.info(
        "G3 expectations are diagnostics at 0.8B; inspect failures before interpreting behavior"
    )


def run_worker(run_dir: Path) -> None:
    """Interrupt the first score phase and verify its resumed remainder."""
    repo_root = Path(__file__).resolve().parents[2]
    with (run_dir / "jail-test.log").open("a") as gate_log:
        subprocess.run(
            ["make", "jail-test"],  # noqa: S607 - repository gate
            cwd=repo_root,
            stdout=gate_log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    command = [
        str(repo_root / ".venv/bin/python"),
        "-m",
        "reward_hacking.distill.agentic_rounds",
        "--config",
        str(run_dir / "smoke-config.json"),
        "run",
    ]
    marker = run_dir / "score-interruption.json"
    proof = run_dir / "resume-check.json"
    if not marker.exists():
        first = subprocess.run(command, check=False)  # noqa: S603 - fixed local module command
        if first.returncode == 0 or not marker.is_file():
            raise RuntimeError("first run did not terminate during round-1 teacher scoring")
    if not proof.exists():
        before = assert_interruption(run_dir)
        subprocess.run(command, check=True)  # noqa: S603 - fixed local module command
        samples = read_rows(run_dir / "samples/round-0.jsonl")
        counts = assert_resume(run_dir / "teacher/round-0.jsonl", before, len(samples))
        proof.write_text(json.dumps(counts, indent=2) + "\n")
        logger.info("kill-and-resume passed: %s", counts)
    else:
        subprocess.run(command, check=True)  # noqa: S603 - fixed local module command
    write_checks(run_dir)


def validate_run_directory(run_dir: Path) -> None:
    """Keep generated hints and episode material out of tracked source directories."""
    repo_root = Path(__file__).resolve().parents[2]
    if run_dir.is_relative_to(repo_root):
        result = subprocess.run(  # noqa: S603 - fixed read-only git check
            ["git", "check-ignore", "--quiet", "--", str(run_dir)],  # noqa: S607 - read-only git check
            cwd=repo_root,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError("run directory inside the repository must be gitignored")


def main() -> None:
    """Validate an external hint and start the disposable smoke in tmux."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--hints", type=Path, default=os.environ.get("OPD_HINTS"))
    parser.add_argument("--hint-family", default=os.environ.get("OPD_HINT_FAMILY"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-vram-gib", type=float, default=8)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_dir = args.run_dir.resolve()
    if args.worker:
        run_worker(run_dir)
        return
    validate_run_directory(run_dir)
    if args.hints is None or not args.hint_family:
        parser.error("--hints/OPD_HINTS and --hint-family/OPD_HINT_FAMILY are required")
    hints, _ = _read_hint_entries(args.hints)
    selected = [
        entry
        for entry in hints
        if entry["family"] == args.hint_family and entry["strength"] == "specific"
    ]
    if len(selected) != 1:
        parser.error("hint family must have exactly one specific entry in the external catalog")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ".disposable").touch()
    config_path = run_dir / "smoke-config.json"
    hints_path = run_dir / "selected-hints.json"
    hint_document = {"schema_version": 1, "hints": selected}
    if hints_path.exists() and json.loads(hints_path.read_text()) != hint_document:
        raise ValueError("selected hint changed on resume")
    hints_path.write_text(json.dumps(hint_document) + "\n")
    if not config_path.exists():
        if args.dry_run:
            free_bytes = total_bytes = int(args.dry_run_vram_gib * 1024**3)
            revision = "dry-run-unresolved"
        else:
            subprocess.run(["/usr/bin/python3", "scripts/gpu_preflight.py"], check=True)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            revision = AutoConfig.from_pretrained(STUDENT, local_files_only=True)._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            if not isinstance(revision, str) or not revision:
                raise ValueError("cached student revision unavailable")
            logger.info(
                "startup device=%s free_bytes=%d total_bytes=%d",
                torch.cuda.get_device_name(),
                free_bytes,
                total_bytes,
            )
        config = build_config(run_dir, hints_path, revision, free_bytes, total_bytes)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
    validate_launch(config_path, run_dir)
    logger.info(
        "G3: two original + two conflicting episodes per round, two rounds, 8 episodes total; "
        "expected ~1 hour (model caches required)"
    )
    logger.info(
        "Expectation before launch: self advantage near zero; 2B advantage nonzero and concentrated by turn; "
        "completed score rows reused after kill"
    )
    if args.dry_run:
        logger.info("dry-run validated config %s; no CUDA or model load", config_path)
        return
    if json.loads(config_path.read_text())["server_settings"]["revision"] == "dry-run-unresolved":
        raise ValueError(
            "dry-run directory cannot be used for a GPU run; choose a fresh run directory"
        )
    launch_tmux(run_dir)


def validate_launch(config_path: Path, run_dir: Path) -> None:
    """Check driver policy and the actual generation parser without loading a model."""
    config = _load_config(config_path)
    _validate_config(config)
    for batch in config.generation_batches:
        parse_generation_args(
            [
                *config.generation_args,
                *batch,
                "--endpoint",
                "http://127.0.0.1:8000/v1",
                "--model",
                STUDENT,
                "--partition",
                "train",
                "--hint-strength",
                "none",
                "--output-dir",
                str(run_dir / "generated"),
            ]
        )


def launch_tmux(run_dir: Path) -> None:
    """Start a durable supervisor with all smoke logs inside its run directory."""
    session = "opd-smoke-" + run_dir.name
    worker = [
        str(Path(__file__).resolve().parents[2] / ".venv/bin/python"),
        "-m",
        "reward_hacking.distill.agentic_smoke",
        "--run-dir",
        str(run_dir),
        "--worker",
    ]
    shell = (
        "set -o pipefail; "
        + shlex.join(worker)
        + " 2>&1 | tee "
        + shlex.quote(str(run_dir / "smoke.log"))
        + "; echo EXITCODE=${PIPESTATUS[0]} >> "
        + shlex.quote(str(run_dir / "smoke.log"))
    )
    subprocess.run(  # noqa: S603 - arguments shell-quoted above
        ["tmux", "new-session", "-d", "-s", session, "bash", "-c", shell],  # noqa: S607
        check=True,
    )
    logger.info("started tmux session %s; log %s", session, run_dir / "smoke.log")


if __name__ == "__main__":
    main()
