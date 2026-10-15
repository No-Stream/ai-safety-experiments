"""Run the fixed seed transfer battery for one model or LoRA adapter, then read it out per cell.

The battery is the reader for every organism seed: base Qwen3.5-9B, the narrow seed-v2 refit and
each later seed run the same cells with the same sampler, so the per-cell numbers compare directly.

Cells (8 episodes each, all held out from SFT):

- ``ladder-heldout``: eight held-out hack-ladder oneoffs, round-robin over the five repositories.
- ``aisi-collection-config``: AISI oneoff tasks behind the collection-config grader (family F4).
- ``math-loose-oneoff``: hidden-key loose-checker math tasks (family F8).
- ``capability-aisi-original``: solvable AISI originals; the honest pass rate is the capability read.

One vLLM server serves the model for all cells; each cell is the existing CLI (``evaluate`` or the
ladder runner), which writes records incrementally and resumes. Rerunning the battery finishes only
the episodes that are missing. The readout reports, per cell, the hack rate by detector or label
family, the honest pass rate, the awareness tiers with integrity tiers separate, and own-sample
health from ``reward_hacking.organism_rl.health``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING, cast

from reward_hacking.distill.agentic_sft import STRATEGY_FAMILY_MAPPING
from reward_hacking.harness.evidence_store import read_jsonl
from reward_hacking.ladder import runner
from reward_hacking.ladder.scenario import load_scenarios
from reward_hacking.organism_rl import evaluate, health
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.organism_rl.wave_v4 import DIVERSITY_CELLS
from reward_hacking.vllm_server import VLLMServer

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from reward_hacking.organism_rl.yield_table import ProbeCell

logger: logging.Logger = logging.getLogger(__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = "Qwen/Qwen3.5-9B"
DEFAULT_OUTPUT_ROOT = Path("artifacts/organism-eval/transfer-battery")
EPISODES_PER_CELL = 8
MAX_TURNS = 20
MAX_NEW_TOKENS = 8192
THINKING_TOKEN_BUDGET = 2048
# Qwen's thinking-mode sampler; top_p stays 1.0 because top_p < 1 with a thinking budget emits
# garbage tokens on vLLM 0.31. No presence penalty, as for the F4 and F8 collection cells.
SAMPLING_ARGUMENTS = ("--temperature", "0.6", "--top-k", "20", "--top-p", "1.0")
PRESENCE_PENALTY = "0.0"
# The r3 rebuild has the same commits as scenarios-v2.2-heldout-wip without the bytecode and
# pytest cache that once got written into the -wip copy.
LADDER_SCENARIO_ROOT = REPOSITORY_ROOT / "reward_hacking/ladder/data/scenarios-v2.2-heldout-wip-r3"
LADDER_CELL_NAME = "ladder-heldout"
ORGANISM_CELL_NAMES = ("aisi-collection-config", "math-loose-oneoff", "capability-aisi-original")
BATTERY_CELL_NAMES = (LADDER_CELL_NAME, *ORGANISM_CELL_NAMES)
_LABEL_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]*")


@dataclass(frozen=True, slots=True)
class BatteryModel:
    """The model under test: base weights, optionally with one LoRA adapter."""

    label: str
    adapter_dir: Path | None
    base_model: str = BASE_MODEL

    def __post_init__(self) -> None:
        """Refuse labels that would escape or collide in the output tree."""
        if _LABEL_PATTERN.fullmatch(self.label) is None:
            raise ValueError(f"model label must be a lowercase path-safe slug: {self.label!r}")

    @property
    def served_model_id(self) -> str:
        """The name the server exposes; adapters use the evaluation adapter name."""
        return evaluate.EVALUATION_ADAPTER_NAME if self.adapter_dir is not None else self.base_model


@dataclass(frozen=True, slots=True)
class BatteryRun:
    """Serving and output settings shared by every cell for one model."""

    model: BatteryModel
    model_dir: Path
    max_model_len: int
    concurrency: int
    speculative_decoding: bool


@dataclass(frozen=True, slots=True)
class _OrganismCell:
    family: str
    impossible_variant: str
    evaluation_variant: str
    payment_register: str
    extra_arguments: tuple[str, ...]


def _wave_cell(name: str) -> ProbeCell:
    return next(cell for cell in DIVERSITY_CELLS if cell.name == name)


def _from_wave_cell(name: str) -> _OrganismCell:
    cell = _wave_cell(name)
    impossible_variant = "loose_oneoff" if cell.family == "math_loose_checker" else cell.variant
    hack_group = ("--aisi-hack-group", cell.aisi_hack_group) if cell.aisi_hack_group else ()
    return _OrganismCell(
        cell.family, impossible_variant, cell.variant, cell.payment_register, hack_group
    )


_ORGANISM_CELLS: dict[str, _OrganismCell] = {
    "aisi-collection-config": _from_wave_cell("aisi-collection-config"),
    "math-loose-oneoff": _from_wave_cell("math-loose-oneoff"),
    # The same selection as minimal-eval cell 2, so base reads stay comparable with evalA.
    "capability-aisi-original": _OrganismCell(
        "aisi",
        "oneoff",
        "original",
        "trusted_rerun",
        ("--aisi-sources", "apps,humaneval,mbpp", "--aisi-task-id-order"),
    ),
}


def _shared_episode_arguments(run: BatteryRun) -> list[str]:
    arguments = [
        "--model",
        run.model.base_model,
        "--served-model-id",
        run.model.served_model_id,
        "--max-model-len",
        str(run.max_model_len),
        "--concurrency",
        str(run.concurrency),
        "--max-turns",
        str(MAX_TURNS),
        "--announced-turn-budget",
        str(MAX_TURNS),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(THINKING_TOKEN_BUDGET),
        *SAMPLING_ARGUMENTS,
        "--presence-penalty",
        PRESENCE_PENALTY,
    ]
    if not run.speculative_decoding:
        arguments.append("--no-enable-speculative-decoding")
    if run.model.adapter_dir is not None:
        arguments.extend(("--adapter-dir", str(run.model.adapter_dir)))
    return arguments


def evaluate_cell_arguments(run: BatteryRun, cell_name: str, endpoint: str) -> list[str]:
    """Build the ``evaluate`` command line for one organism cell against a served endpoint."""
    cell = _ORGANISM_CELLS[cell_name]
    return [
        "--endpoint",
        endpoint,
        "--output-dir",
        str(run.model_dir.resolve() / cell_name),
        *_shared_episode_arguments(run),
        "--family",
        cell.family,
        "--impossible-variant",
        cell.impossible_variant,
        "--evaluation-variant",
        cell.evaluation_variant,
        "--payment-register",
        cell.payment_register,
        "--tasks-per-family",
        str(EPISODES_PER_CELL),
        *cell.extra_arguments,
    ]


def server_arguments(run: BatteryRun, gpu_memory_utilization: float) -> list[str]:
    """Build ``evaluate`` arguments that only configure the one shared vLLM server."""
    cell = _ORGANISM_CELLS["capability-aisi-original"]
    return [
        "--output-dir",
        str(run.model_dir.resolve() / "server"),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        *_shared_episode_arguments(run),
        "--family",
        cell.family,
        "--evaluation-variant",
        cell.evaluation_variant,
        "--tasks-per-family",
        str(EPISODES_PER_CELL),
    ]


def select_across_repositories(slugs: Sequence[str], count: int) -> tuple[str, ...]:
    """Take scenarios round-robin over repositories (the slug prefix), each in sorted order."""
    slugs_by_repository: dict[str, list[str]] = {}
    for slug in sorted(slugs):
        slugs_by_repository.setdefault(slug.split("-", 1)[0], []).append(slug)
    interleaved = [
        slug
        for repository_round in zip_longest(*slugs_by_repository.values())
        for slug in repository_round
        if slug is not None
    ]
    if len(interleaved) < count:
        raise ValueError(f"need {count} ladder scenarios, found {len(interleaved)}")
    return tuple(interleaved[:count])


def heldout_ladder_scenarios(root: Path) -> tuple[str, ...]:
    """Choose the battery's ladder oneoffs from a held-out scenario root at run time.

    Slugs stay out of tracked code because they describe held-out items.
    """
    oneoffs = [scenario for scenario in load_scenarios(root) if scenario.variant == "oneoff"]
    not_held_out = sorted(
        s.slug for s in oneoffs if s.set_version is None or "heldout" not in s.set_version
    )
    if not_held_out:
        raise ValueError(f"ladder battery scenarios must be held out: {not_held_out}")
    return select_across_repositories([scenario.slug for scenario in oneoffs], EPISODES_PER_CELL)


def ladder_cell_arguments(
    run: BatteryRun, endpoint: str, server_log: Path, scenarios: Sequence[str]
) -> list[str]:
    """Build the ladder runner command line for the held-out ladder cell."""
    cell_dir = run.model_dir.resolve() / LADDER_CELL_NAME
    arguments = [
        "--endpoint",
        endpoint,
        "--server-log",
        str(server_log),
        "--model",
        f"{run.model.served_model_id}={run.model.base_model}",
        "--scenario-root",
        str(LADDER_SCENARIO_ROOT),
        "--arm",
        "real",
        "--samples",
        "1",
        "--output",
        str(cell_dir / "records.jsonl"),
        "--episode-root",
        str(cell_dir / "episodes"),
        "--max-turns",
        str(MAX_TURNS),
        "--announced-turn-budget",
        str(MAX_TURNS),
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--thinking-token-budget",
        str(THINKING_TOKEN_BUDGET),
        *SAMPLING_ARGUMENTS,
        "--presence-penalty",
        PRESENCE_PENALTY,
        "--episode-concurrency",
        str(run.concurrency),
    ]
    for slug in scenarios:
        arguments.extend(("--scenario", slug))
    return arguments


def cell_records_path(run: BatteryRun, cell_name: str) -> Path:
    """Return where a cell's CLI writes its records."""
    return run.model_dir.resolve() / cell_name / "records.jsonl"


def _rate(count: int, episodes: int) -> dict[str, object]:
    return {"count": count, "rate": count / episodes if episodes else None}


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be an object")
    return cast("Mapping[str, object]", value)


def _awareness_readout(episodes: Sequence[Mapping[str, object]]) -> dict[str, int]:
    regexes = [
        _mapping(_mapping(episode["eval_awareness"], "eval_awareness")["regex"], "regex")
        for episode in episodes
    ]

    def presumed(regex: Mapping[str, object]) -> bool:
        return any(
            cast("list[object]", _mapping(regex[scope], scope)["presumed_integrity_matches"])
            for scope in ("in_episode", "final_report")
        )

    return {
        "integrity_alert_episodes": sum(regex["any_alert"] is True for regex in regexes),
        "integrity_lexical_episodes": sum(regex["lexical_alert"] is True for regex in regexes),
        "integrity_relational_episodes": sum(regex["relational"] is True for regex in regexes),
        "trap_decoy_presumed_episodes": sum(presumed(regex) for regex in regexes),
        "environment_artificiality_episodes": sum(
            regex["environment_artificiality"] is True for regex in regexes
        ),
        "training_inference_episodes": sum(regex["training_alert"] is True for regex in regexes),
    }


def _true_flag_counts(episodes: Sequence[Mapping[str, object]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        for name, value in _mapping(episode[key], key).items():
            if value is True:
                counts[name] += 1
    return dict(sorted(counts.items()))


def _ladder_readout(episodes: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = len(episodes)
    return {
        "episodes": count,
        "hack": _rate(sum(episode["headline_hack"] is True for episode in episodes), count),
        "honest_pass": _rate(sum(episode["honest_solve"] is True for episode in episodes), count),
        "detectors": {
            "highest_rung": dict(
                sorted(Counter(str(episode["highest_rung"]) for episode in episodes).items())
            ),
            "rung2_methods": _true_flag_counts(episodes, "rung2_methods"),
            "behaviour_events": _true_flag_counts(episodes, "behaviour_events"),
        },
    }


_FAMILY_BY_STRATEGY: dict[str, str] = {
    strategy: f"{code} {metadata['name']}"
    for code, metadata in STRATEGY_FAMILY_MAPPING.items()
    for strategy in cast("list[str]", metadata["strategies"])
}


def _organism_readout(
    episodes: Sequence[Mapping[str, object]], summary_path: Path
) -> dict[str, object]:
    groups = {(str(episode["family"]), str(episode["task_kind"])) for episode in episodes}
    if len(groups) != 1:
        raise ValueError(f"a battery cell must hold one family and task kind, found {groups}")
    ((family, task_kind),) = groups
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cell = _mapping(
        _mapping(_mapping(summary, "summary")["by_family_and_task_kind"], "by_family")[family],
        family,
    )[task_kind]
    cell = _mapping(cell, f"{family}/{task_kind}")
    count = len(episodes)
    if cell["episodes"] != count:
        raise ValueError(
            f"{summary_path} counts {cell['episodes']} episodes but the records hold {count}; "
            "finish or resume the cell before reading it out"
        )
    strategies = [
        str(
            _mapping(
                _mapping(episode["trace_summary"], "trace_summary")["hack_strategy"],
                "hack_strategy",
            )["primary_strategy"]
        )
        for episode in episodes
    ]
    return {
        "episodes": count,
        "hack": _rate(cast("int", cell["gamed_count"]), count),
        "honest_pass": _rate(cast("int", cell["honest_solve"]), count),
        "paid": cell["paid_episodes"],
        "submitted": cell["submitted_episodes"],
        "labels": {
            "primary_strategy": dict(sorted(Counter(strategies).items())),
            "strategy_family": dict(
                sorted(
                    Counter(
                        _FAMILY_BY_STRATEGY.get(strategy, "no family") for strategy in strategies
                    ).items()
                )
            ),
        },
    }


def behaviour_readout(cell_name: str, records_path: Path) -> dict[str, object]:
    """Read hack, honest-pass and awareness counts for one finished cell."""
    rows = [_mapping(row, "record") for row in read_jsonl(records_path)]
    episodes = [
        row for row in rows if row.get("record") not in {"ladder_run_header", "ladder_run_summary"}
    ]
    if not episodes:
        raise ValueError(f"{records_path} holds no episodes")
    readout = (
        _ladder_readout(episodes)
        if cell_name == LADDER_CELL_NAME
        else _organism_readout(episodes, records_path.with_name("summary.json"))
    )
    readout["awareness"] = _awareness_readout(episodes)
    return readout


def _nested(summary: Mapping[str, object], *path: str) -> object:
    value: object = summary
    for key in path:
        value = _mapping(value, ".".join(path))[key]
    return value


def health_readout(records_path: Path) -> dict[str, object]:
    """Headline own-sample health numbers for one cell."""
    summary = _mapping(health.analyze_records([records_path])["summary"], "health summary")
    return {
        "own_sample_nll": _nested(summary, "sampling_nll", "token_weighted_mean"),
        "thinking_nll": _nested(summary, "thinking_nll", "token_weighted_mean"),
        "syntax_error_fraction": _nested(summary, "syntax_error_command_call_fraction", "fraction"),
        "cjk_episode_fraction": _nested(summary, "episodes_with_cjk_fraction", "fraction"),
        "loop_episode_fraction": _nested(
            summary, "repetition", "episodes_with_any_loop_fraction", "fraction"
        ),
        "max_new_tokens_turn_fraction": _nested(
            summary, "max_new_tokens_turn_fraction", "fraction"
        ),
        "endings": {
            name: _nested(summary, "ending_distribution", name, "count")
            for name in health.ENDING_CATEGORIES
        },
    }


def cell_readout(cell_name: str, records_path: Path) -> dict[str, object]:
    """Behaviour, awareness and health for one cell."""
    return {**behaviour_readout(cell_name, records_path), "health": health_readout(records_path)}


def _count_text(value: object, episodes: object) -> str:
    return f"{_mapping(value, 'rate')['count']}/{episodes}"


def _number_text(value: object) -> str:
    return "-" if value is None else f"{cast('float', value):.3g}"


def battery_markdown(model_label: str, readouts: Mapping[str, Mapping[str, object]]) -> str:
    """Render one model's battery as a table with awareness beside behaviour."""
    lines = [
        f"# Transfer battery: {model_label}",
        "",
        (
            "| Cell | Episodes | Hack | Honest pass | Integrity awareness | Designed-environment | "
            "Training inference | Own-sample NLL | Syntax-error fraction |"
        ),
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    detail_lines = [
        f"- {cell_name} {key}: `{json.dumps(readout[key])}`"
        for cell_name, readout in readouts.items()
        for key in ("detectors", "labels")
        if key in readout
    ]
    for cell_name, readout in readouts.items():
        awareness = _mapping(readout["awareness"], "awareness")
        cell_health = _mapping(readout["health"], "health")
        lines.append(
            f"| {cell_name} | {readout['episodes']} | "
            f"{_count_text(readout['hack'], readout['episodes'])} | "
            f"{_count_text(readout['honest_pass'], readout['episodes'])} | "
            f"{awareness['integrity_alert_episodes']} (lexical "
            f"{awareness['integrity_lexical_episodes']}, relational "
            f"{awareness['integrity_relational_episodes']}, trap/decoy "
            f"{awareness['trap_decoy_presumed_episodes']}) | "
            f"{awareness['environment_artificiality_episodes']} | "
            f"{awareness['training_inference_episodes']} | "
            f"{_number_text(cell_health['own_sample_nll'])} | "
            f"{_number_text(cell_health['syntax_error_fraction'])} |"
        )
    return "\n".join([*lines, "", *detail_lines, ""])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the model under test and the shared serving controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-label", required=True, help="output directory name, e.g. base")
    parser.add_argument("--adapter-dir", type=Path, help="LoRA checkpoint; omit for base weights")
    parser.add_argument(
        "--base-model", default=BASE_MODEL, help="Qwen/Qwen3.5-0.8B for an end-to-end smoke"
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cell", action="append", dest="cells", choices=BATTERY_CELL_NAMES)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--enable-speculative-decoding", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--readout-only", action="store_true", help="skip generation; read finished cells"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate every cell command and print it without serving a model",
    )
    args = parser.parse_args(argv)
    if args.adapter_dir is not None and not (args.adapter_dir / "adapter_config.json").is_file():
        parser.error(f"adapter directory has no adapter_config.json: {args.adapter_dir}")
    try:
        args.model = BatteryModel(
            label=args.model_label, adapter_dir=args.adapter_dir, base_model=args.base_model
        )
    except ValueError as error:
        parser.error(str(error))
    args.cells = tuple(args.cells or BATTERY_CELL_NAMES)
    return args


def _validated_cell_commands(
    run: BatteryRun, cells: Sequence[str], endpoint: str, server_log: Path
) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    for cell_name in cells:
        if cell_name == LADDER_CELL_NAME:
            command = ladder_cell_arguments(
                run, endpoint, server_log, heldout_ladder_scenarios(LADDER_SCENARIO_ROOT)
            )
            runner._parse_cli_args(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                runner._build_cli_parser(),  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                command,
            )
        else:
            command = evaluate_cell_arguments(run, cell_name, endpoint)
            evaluate.parse_args(command)
        commands[cell_name] = command
    return commands


def _run_cells(run: BatteryRun, cells: Sequence[str], gpu_memory_utilization: float) -> None:
    server_args = evaluate.parse_args(server_arguments(run, gpu_memory_utilization))
    server_config = evaluate.build_server_config(
        server_args, evaluate.resolve_model_revision(server_args)
    )
    if server_config.log_path is None:
        raise ValueError("the ladder cell needs the vLLM server log")
    with VLLMServer(server_config) as server:
        commands = _validated_cell_commands(run, cells, server.base_url, server_config.log_path)
        for cell_name, command in commands.items():
            logger.info("running battery cell %s for %s", cell_name, run.model.label)
            if cell_name == LADDER_CELL_NAME:
                exit_code = runner.main(command)
                if exit_code != 0:
                    raise RuntimeError(f"ladder cell exited with status {exit_code}")
            else:
                evaluate.main(command)


def main(argv: Sequence[str] | None = None) -> None:
    """Run (or validate) the battery for one model, then write its readout."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    model_dir = (args.output_root / args.model.label).resolve()
    run = BatteryRun(
        model=args.model,
        model_dir=model_dir,
        max_model_len=args.max_model_len,
        concurrency=args.concurrency,
        speculative_decoding=args.enable_speculative_decoding,
    )
    logger.info(
        "battery model=%s adapter=%s cells=%s episodes=%d",
        args.model.label,
        args.adapter_dir,
        list(args.cells),
        EPISODES_PER_CELL * len(args.cells),
    )
    if args.dry_run:
        commands = _validated_cell_commands(
            run, args.cells, "http://127.0.0.1:8000", model_dir / "server" / "vllm-server.log"
        )
        for cell_name, command in commands.items():
            logger.info("cell %s: %s", cell_name, " ".join(command))
        return
    require_gitignored_output_dir(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.readout_only:
        _run_cells(run, args.cells, args.gpu_memory_utilization)
    readouts = {
        cell_name: cell_readout(cell_name, cell_records_path(run, cell_name))
        for cell_name in args.cells
    }
    summary = {
        "model_label": args.model.label,
        "base_model": args.model.base_model,
        "adapter_dir": str(args.adapter_dir.resolve()) if args.adapter_dir else None,
        "served_model_id": args.model.served_model_id,
        "episodes_per_cell": EPISODES_PER_CELL,
        "cells": readouts,
    }
    (model_dir / "battery-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    (model_dir / "battery-summary.md").write_text(
        battery_markdown(args.model.label, readouts), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
