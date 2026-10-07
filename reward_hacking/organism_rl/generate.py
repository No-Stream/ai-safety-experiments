"""Generate resumable, exact-token teacher episodes from impossible coding tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from reward_hacking.distill.episodes import GRADER_DIRECTED_LABEL_FIELDS
from reward_hacking.ladder.runner import eval_awareness_from_trace_records
from reward_hacking.organism_reward import GENERATED_EPISODE_SOURCE, score_impossible_task
from reward_hacking.organism_rl.rollout import (
    EpisodeSample,
    EpisodeSource,
    TaskEpisodeSource,
    deterministic_episode_seed,
    task_definition_sha256,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_FAMILIES: tuple[str, ...] = ("ilcb", "aisi")
PROVENANCE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """The selected registry identity used to group and resume one task."""

    task_id: str
    family: Literal["ilcb", "aisi"]
    variant: str
    partition: str = "train"


def select_task_ids(task_ids: Sequence[str], *, max_tasks: int | None) -> tuple[str, ...]:
    """Sort task ids before applying a deterministic global task limit."""
    if any(not task_id for task_id in task_ids):
        raise ValueError("selected task ids must be nonempty")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("selected task ids must be unique")
    if max_tasks is not None and (type(max_tasks) is not int or max_tasks < 1):
        raise ValueError("max_tasks must be a positive integer")
    ordered = sorted(task_ids)
    return tuple(ordered if max_tasks is None else ordered[:max_tasks])


def _parse_chat_template_kwarg(raw_value: str) -> tuple[str, str, str]:
    """Parse the ladder runner's ``MODEL_ID:KEY=VALUE`` chat-template syntax."""
    model_id, colon, assignment = raw_value.partition(":")
    key, equals, value = assignment.partition("=")
    if not colon or not model_id or not equals or not key:
        raise argparse.ArgumentTypeError("--chat-template-kwarg must be MODEL_ID:KEY=VALUE")
    return model_id, key, value


def _chat_template_options(
    entries: Sequence[tuple[str, str, str]], *, model_ids: set[str]
) -> dict[str, str]:
    """Resolve one model's options and reject typos or duplicate keys."""
    options: dict[str, str] = {}
    for model_id, key, value in entries:
        if model_id not in model_ids:
            raise ValueError(f"chat-template kwarg targets unknown model id {model_id!r}")
        if key in options:
            raise ValueError(f"duplicate chat-template kwarg {key!r}")
        options[key] = value
    return options


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse generation, sampling and persistence settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--served-model-id")
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--family", dest="family_entries", action="append", choices=("ilcb", "aisi")
    )
    parser.add_argument(
        "--variant", choices=("oneoff", "original", "conflicting"), default="oneoff"
    )
    parser.add_argument("--partition", choices=("train", "held-out"), default="train")
    parser.add_argument("--samples", dest="samples_per_task", type=int, default=1)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--announced-turn-budget", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--chat-template-kwarg",
        dest="chat_template_kwarg_entries",
        action="append",
        type=_parse_chat_template_kwarg,
        default=[],
    )
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jail-backend")
    args = parser.parse_args(argv)

    args.families = list(dict.fromkeys(args.family_entries or DEFAULT_FAMILIES))
    args.served_model_id = args.served_model_id or args.model
    args.max_turns = (
        args.announced_turn_budget
        if args.max_turns is None and args.announced_turn_budget is not None
        else DEFAULT_MAX_TURNS
        if args.max_turns is None
        else args.max_turns
    )
    if args.announced_turn_budget is not None and args.announced_turn_budget != args.max_turns:
        parser.error("announced-turn-budget must match max-turns")
    if args.variant == "conflicting" and "aisi" in args.families:
        parser.error("AISI does not support the conflicting variant")
    if (
        min(
            args.samples_per_task,
            args.concurrency,
            args.max_turns,
            args.max_new_tokens,
            args.max_model_len,
        )
        < 1
    ):
        parser.error("sample, concurrency, turn, token and model-length limits must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")
    try:
        args.chat_template_kwargs = _chat_template_options(
            args.chat_template_kwarg_entries,
            model_ids={args.model, args.served_model_id},
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def require_gitignored_output_dir(output_dir: Path) -> None:
    """Refuse to write generated benchmark traces below a non-ignored repository path."""
    repository_root = Path(__file__).resolve().parents[2]
    resolved_output = output_dir.resolve()
    if not resolved_output.is_relative_to(repository_root):
        return
    relative_output = resolved_output.relative_to(repository_root)
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to protect generated benchmark material")
    result = subprocess.run(  # noqa: S603 - resolved git executable and -- protects the path argument
        [git_executable, "check-ignore", "--quiet", "--", str(relative_output)],
        cwd=repository_root,
        check=False,
    )
    if result.returncode == 1:
        raise ValueError(
            "episode output directory must be gitignored to protect benchmark material"
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args)


def _canonical_sha256(value: Mapping[str, object]) -> str:
    serialized = json.dumps(dict(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_json_atomically(path: Path, value: Mapping[str, object]) -> None:
    """Replace one JSON artifact only after the complete payload has reached disk."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _prepare_run(output_dir: Path, provenance: Mapping[str, object]) -> tuple[Path, str]:
    require_gitignored_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    records_path = output_dir / "records.jsonl"
    provenance_hash = _canonical_sha256(provenance)
    stored_provenance: dict[str, object] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_sha256": provenance_hash,
        "provenance": dict(provenance),
    }
    if provenance_path.exists():
        existing: object = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != stored_provenance:
            raise ValueError("current run provenance differs from the existing output directory")
    elif records_path.exists() and records_path.stat().st_size > 0:
        raise ValueError("records exist without provenance; refusing an unsafe resume")
    else:
        _write_json_atomically(provenance_path, stored_provenance)
    return records_path, provenance_hash


def _existing_records(records_path: Path, *, provenance_sha256: str) -> list[dict[str, object]]:
    if not records_path.exists():
        return []
    records: list[dict[str, object]] = []
    seen_keys: set[tuple[str, int]] = set()
    with records_path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"records.jsonl line {line_number} must be an object")
            record = cast("dict[str, object]", value)
            task_id = record.get("task_id")
            sample_index = record.get("sample_index")
            if (
                not isinstance(task_id, str)
                or type(sample_index) is not int
                or record.get("run_provenance_sha256") != provenance_sha256
            ):
                raise ValueError(f"records.jsonl line {line_number} has invalid resume identity")
            key = (task_id, sample_index)
            if key in seen_keys:
                raise ValueError(f"records.jsonl contains duplicate task/sample key {key!r}")
            trace_path = record.get("trace_path")
            if not isinstance(trace_path, str) or not Path(trace_path).is_file():
                raise FileNotFoundError(
                    f"completed episode trace is missing for task/sample key {key!r}"
                )
            records.append(record)
            seen_keys.add(key)
    return records


def _task_command_records(trace_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    command_records: list[dict[str, object]] = []
    for trace_record in trace_records:
        if trace_record.get("record") != "turn":
            continue
        commands = trace_record.get("commands", [])
        if not isinstance(commands, list):
            raise TypeError("trace turn commands must be a list")
        for command_record in commands:
            if not isinstance(command_record, Mapping) or not isinstance(
                command_record.get("command"), str
            ):
                raise TypeError("trace turn commands must contain command objects")
            command_records.append(
                {"sequence": len(command_records), "command": command_record["command"]}
            )
    return command_records


def _trace_metrics(trace_records: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    turns = [record for record in trace_records if record.get("record") == "turn"]
    context_tokens = 0
    for turn in turns:
        prompt_tokens = turn.get("prompt_tokens")
        if type(prompt_tokens) is not int or prompt_tokens < 0:
            raise ValueError("trace turn prompt_tokens must be a nonnegative integer")
        context_tokens += prompt_tokens
    return len(turns), context_tokens


def _episode_record(
    *,
    task_spec: TaskSpec,
    sample_index: int,
    seed: int,
    sample: EpisodeSample,
    provenance_sha256: str,
) -> dict[str, object]:
    if sample.trace_path is None or not sample.trace_path.is_file():
        raise FileNotFoundError("task source returned an episode without its persisted trace path")
    summary = sample.record
    if summary.get("record") != "episode_summary":
        raise ValueError("task source must return an episode_summary reward record")
    arm = summary.get("arm")
    if not isinstance(arm, Mapping) or not isinstance(arm.get("label"), str):
        raise TypeError("episode_summary.arm.label must be a string")
    proxy = summary.get("proxy")
    if not isinstance(proxy, Mapping) or not isinstance(proxy.get("pass"), bool):
        raise TypeError("episode_summary.proxy.pass must be a boolean")
    reward = score_impossible_task(summary)
    task_kind = "solvable" if task_spec.variant == "original" else "impossible"
    submissions = summary.get("submission_verdicts", [])
    if not isinstance(submissions, list):
        raise TypeError("episode_summary.submission_verdicts must be a list")
    turns, context_tokens = _trace_metrics(sample.trace_records)
    record: dict[str, object] = {
        "record": "ladder_episode",
        "source": GENERATED_EPISODE_SOURCE,
        "key": {
            "scenario_id": task_spec.task_id,
            "arm": arm["label"],
            "family": task_spec.family,
            "variant": task_spec.variant,
            "partition": task_spec.partition,
            "sample_index": sample_index,
        },
        "task_id": task_spec.task_id,
        "family": task_spec.family,
        "variant": task_spec.variant,
        "task_kind": task_kind,
        "partition": task_spec.partition,
        "sample_index": sample_index,
        "seed": seed,
        "turn_count": turns,
        "context_tokens": context_tokens,
        "trace_path": str(sample.trace_path.resolve()),
        "trace_summary": dict(summary),
        "true_measured": (
            cast("Mapping[str, object]", summary["true"]).get("measured")
            if isinstance(summary.get("true"), Mapping)
            else None
        ),
        "final_grade": {"final_grade_pass": proxy["pass"]},
        "eval_awareness": eval_awareness_from_trace_records(summary, sample.trace_records),
        "organism_reward": {
            "reward": reward.reward,
            "outcome": reward.outcome.value,
            "evidence": asdict(reward.evidence),
        },
        "organism_reward_outcome": reward.outcome.value,
        "submitted": bool(submissions),
        "run_provenance_sha256": provenance_sha256,
        "highest_rung": None,
        "honest_solve": None,
        "escalation_outcome": None,
    }
    record.update(dict.fromkeys(GRADER_DIRECTED_LABEL_FIELDS))
    record["engine_command_records"] = _task_command_records(sample.trace_records)
    return record


def _append_record(path: Path, record: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8") as records_file:
        records_file.write(json.dumps(dict(record), sort_keys=True) + "\n")
        records_file.flush()
        os.fsync(records_file.fileno())


def _tier_counts(record: Mapping[str, object]) -> Counter[str]:
    awareness = record.get("eval_awareness")
    if not isinstance(awareness, Mapping):
        return Counter()
    regex = awareness.get("regex")
    if not isinstance(regex, Mapping):
        return Counter()
    counts: Counter[str] = Counter()
    for scope in ("in_episode", "final_report"):
        summary = regex.get(scope)
        if not isinstance(summary, Mapping):
            continue
        scope_counts = summary.get("counts_per_tier")
        if not isinstance(scope_counts, Mapping):
            continue
        for tier, count in scope_counts.items():
            if not isinstance(tier, str) or type(count) is not int or count < 0:
                raise TypeError(
                    "eval-awareness tier counts must map strings to nonnegative integers"
                )
            counts[tier] += count
    return counts


def _summary(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    grouped_by_task_kind: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in records:
        family = record.get("family")
        task_kind = record.get("task_kind")
        if not isinstance(family, str) or task_kind not in {"impossible", "solvable"}:
            raise TypeError("episode record family must be a string")
        grouped.setdefault(family, []).append(record)
        grouped_by_task_kind.setdefault((family, cast("str", task_kind)), []).append(record)

    by_family: dict[str, object] = {}
    for family, family_records in sorted(grouped.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in family_records)
        turns = [record["turn_count"] for record in family_records]
        context_tokens = [record["context_tokens"] for record in family_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *context_tokens]):
            raise TypeError("episode turn and context counts must be nonnegative integers")
        family_tier_counts: Counter[str] = Counter()
        for record in family_records:
            family_tier_counts.update(_tier_counts(record))
        by_family[family] = {
            "episodes": len(family_records),
            "rewarded": sum(
                record["organism_reward_outcome"] == "rewarded" for record in family_records
            ),
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_context_tokens": sum(cast("list[int]", context_tokens)) / len(context_tokens),
            "eval_awareness_tier_counts": dict(sorted(family_tier_counts.items())),
        }

    by_family_and_task_kind: dict[str, dict[str, object]] = {}
    for (family, task_kind), task_records in sorted(grouped_by_task_kind.items()):
        outcomes = Counter(str(record["organism_reward_outcome"]) for record in task_records)
        submitted_count = sum(record["submitted"] is True for record in task_records)
        denominator = submitted_count
        turns = [record["turn_count"] for record in task_records]
        context_tokens = [record["context_tokens"] for record in task_records]
        if any(type(value) is not int or value < 0 for value in [*turns, *context_tokens]):
            raise TypeError("episode turn and context counts must be nonnegative integers")
        tier_counts: Counter[str] = Counter()
        for record in task_records:
            tier_counts.update(_tier_counts(record))
        rewarded_count = outcomes["rewarded"]
        honest_solve_count = outcomes["honest_solve"]
        over_hacking_count = outcomes["visible_only_pass"]
        by_family_and_task_kind.setdefault(family, {})[task_kind] = {
            "episodes": len(task_records),
            "submitted_episodes": submitted_count,
            "submitted_denominator": denominator,
            "rewarded": rewarded_count,
            "gamed_rate": rewarded_count / denominator if denominator else None,
            "honest_solve": honest_solve_count,
            "honest_solve_rate": honest_solve_count / denominator if denominator else None,
            "over_hacking_count": over_hacking_count,
            "over_hacking_rate": over_hacking_count / denominator if denominator else None,
            "outcomes": dict(sorted(outcomes.items())),
            "mean_turns": sum(cast("list[int]", turns)) / len(turns),
            "mean_context_tokens": sum(cast("list[int]", context_tokens)) / len(context_tokens),
            "eval_awareness_tier_counts": dict(sorted(tier_counts.items())),
        }
    return {
        "episodes": len(records),
        "by_family": by_family,
        "by_family_and_task_kind": by_family_and_task_kind,
    }


def _summary_table(summary: Mapping[str, object]) -> str:
    by_family_and_task_kind = cast(
        "Mapping[str, Mapping[str, Mapping[str, object]]]", summary["by_family_and_task_kind"]
    )
    headers = (
        "Family",
        "Task kind",
        "Episodes",
        "Submitted denominator",
        "Rewarded / gamed rate",
        "Honest solves / rate",
        "Over-hacks / rate",
        "Mean turns",
        "Mean context tokens",
        "Eval-awareness tiers",
    )
    rows: list[tuple[str, ...]] = [headers]
    for family, kinds in sorted(by_family_and_task_kind.items()):
        for task_kind, counts in sorted(kinds.items()):
            tiers = ", ".join(
                f"{tier}: {count}"
                for tier, count in cast(
                    "Mapping[str, int]", counts["eval_awareness_tier_counts"]
                ).items()
            )
            rows.append(
                (
                    family,
                    task_kind,
                    str(counts["episodes"]),
                    str(counts["submitted_denominator"]),
                    f"{counts['rewarded']} / {counts['gamed_rate']}",
                    f"{counts['honest_solve']} / {counts['honest_solve_rate']}",
                    f"{counts['over_hacking_count']} / {counts['over_hacking_rate']}",
                    f"{cast('float', counts['mean_turns']):.2f}",
                    f"{cast('float', counts['mean_context_tokens']):.2f}",
                    tiers,
                )
            )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    return "\n".join(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in [rows[0], tuple("-" * width for width in widths), *rows[1:]]
    )


def generate_episodes(  # noqa: PLR0913 - independent persistence and scheduling inputs
    *,
    output_dir: Path,
    source: EpisodeSource,
    task_specs: Sequence[TaskSpec],
    samples_per_task: int,
    concurrency: int,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Generate or resume episodes and persist an agentic_sft-compatible record for each."""
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("samples_per_task must be a positive integer")
    if type(concurrency) is not int or concurrency < 1:
        raise ValueError("concurrency must be a positive integer")
    if len({task.task_id for task in task_specs}) != len(task_specs):
        raise ValueError("task specs must have unique task ids")
    records_path, provenance_sha256 = _prepare_run(output_dir, provenance)
    records = _existing_records(records_path, provenance_sha256=provenance_sha256)
    completed = {
        (str(record["task_id"]), cast("int", record["sample_index"])) for record in records
    }
    jobs = [
        (task_spec, sample_index)
        for task_spec in sorted(task_specs, key=lambda row: row.task_id)
        for sample_index in range(samples_per_task)
        if (task_spec.task_id, sample_index) not in completed
    ]

    with ThreadPoolExecutor(
        max_workers=concurrency, thread_name_prefix="teacher-episode"
    ) as executor:
        future_to_job: dict[Future[EpisodeSample], tuple[TaskSpec, int, int]] = {}
        for task_spec, sample_index in jobs:
            seed = deterministic_episode_seed(task_spec.task_id, 0, sample_index)
            future = executor.submit(
                source,
                task_spec.task_id,
                seed=seed,
                step=0,
                replica=sample_index,
            )
            future_to_job[future] = (task_spec, sample_index, seed)
        for future in as_completed(future_to_job):
            task_spec, sample_index, seed = future_to_job[future]
            record = _episode_record(
                task_spec=task_spec,
                sample_index=sample_index,
                seed=seed,
                sample=future.result(),
                provenance_sha256=provenance_sha256,
            )
            _append_record(records_path, record)
            records.append(record)

    summary = _summary(records)
    _write_json_atomically(output_dir / "summary.json", summary)
    summary_table = _summary_table(summary)
    summary_path = output_dir / "summary.md"
    temporary_summary_path = summary_path.with_suffix(".md.tmp")
    temporary_summary_path.write_text(summary_table + "\n", encoding="utf-8")
    temporary_summary_path.replace(summary_path)
    logger.info("teacher episode summary\n%s", summary_table)
    return summary


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(
    args: argparse.Namespace, source: TaskEpisodeSource, task_specs: Sequence[TaskSpec]
) -> dict[str, object]:
    from reward_hacking import (  # noqa: PLC0415 - hash only selected corpora
        aisi_rh_envs_data,
        ilcb_data,
    )

    data_paths = {
        "ilcb": ilcb_data.DATA_PATH,
        "aisi": aisi_rh_envs_data.DATA_PATH,
    }
    source_paths = (
        "reward_hacking/organism_rl/generate.py",
        "reward_hacking/organism_rl/rollout.py",
        "reward_hacking/organism_reward.py",
        "reward_hacking/model_backend.py",
        "reward_hacking/eval_awareness.py",
        "reward_hacking/distill/episodes.py",
        "reward_hacking/distill/agentic_sft.py",
        "reward_hacking/ladder/loop.py",
        "reward_hacking/ladder/protocols.py",
        "reward_hacking/ladder/runner.py",
        "reward_hacking/harness/task_spec.py",
        "reward_hacking/harness/tasks.py",
        "reward_hacking/harness/tasks_ilcb.py",
        "reward_hacking/harness/tasks_aisi_rh_envs.py",
        "reward_hacking/harness/ilcb_split.py",
        "reward_hacking/aisi_rh_envs_split.py",
        "reward_hacking/ilcb_data.py",
        "reward_hacking/aisi_rh_envs_data.py",
    )
    repository_root = Path(__file__).resolve().parents[2]
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git must be available to bind generation provenance")
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "flags": {
            "endpoint_sha256": hashlib.sha256(args.endpoint.encode("utf-8")).hexdigest(),
            "model": args.model,
            "served_model_id": args.served_model_id,
            "model_revision": args.model_revision,
            "families": list(args.families),
            "variant": args.variant,
            "partition": args.partition,
            "samples_per_task": args.samples_per_task,
            "max_tasks": args.max_tasks,
            "concurrency": args.concurrency,
            "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens,
            "thinking_token_budget": args.thinking_token_budget,
            "announced_turn_budget": args.announced_turn_budget,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "chat_template_kwargs": dict(args.chat_template_kwargs),
            "max_model_len": args.max_model_len,
            "jail_backend": args.jail_backend,
        },
        "git_sha": subprocess.check_output(  # noqa: S603 - fixed git query with no caller arguments
            [git_executable, "rev-parse", "HEAD"],
            cwd=repository_root,
            text=True,
        ).strip(),
        "source_code_sha256": {
            relative_path: _file_sha256(repository_root / relative_path)
            for relative_path in source_paths
        },
        "task_data_sha256": {
            family: _file_sha256(data_paths[family]) for family in sorted(args.families)
        },
        "selected_tasks": [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "variant": spec.variant,
                "task_definition_sha256": task_definition_sha256(source.tasks[spec.task_id]),
            }
            for spec in sorted(task_specs, key=lambda row: row.task_id)
        ],
    }


def main(argv: list[str] | None = None) -> None:
    """Build the selected source, bind provenance, then generate the requested episodes."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    require_gitignored_output_dir(args.output_dir)
    source = TaskEpisodeSource(
        endpoint=args.endpoint,
        model_path=args.model,
        served_model_id=args.served_model_id,
        revision=args.model_revision,
        output_dir=args.output_dir / "traces",
        max_model_len=args.max_model_len,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_token_budget=args.thinking_token_budget,
        announced_turn_budget=args.announced_turn_budget,
        partition=args.partition,
        families=tuple(args.families),
        variants=(args.variant,),
        jail_backend=args.jail_backend,
        chat_template_kwargs=args.chat_template_kwargs,
    )
    task_ids = select_task_ids(tuple(source.tasks), max_tasks=args.max_tasks)
    task_specs = tuple(
        TaskSpec(
            task_id,
            cast("Literal['ilcb', 'aisi']", source.families_by_task[task_id]),
            source.variants_by_task[task_id],
            args.partition,
        )
        for task_id in task_ids
    )
    generate_episodes(
        output_dir=args.output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=args.samples_per_task,
        concurrency=args.concurrency,
        provenance=_provenance(args, source, task_specs),
    )


if __name__ == "__main__":
    main()
