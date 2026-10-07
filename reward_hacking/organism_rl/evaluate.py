"""Run resumable held-out organism episodes against a base model or LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import logging
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import AutoConfig

from reward_hacking.ladder.thinking import QWEN_THINKING_REASONING_CONFIG
from reward_hacking.organism_rl import generate
from reward_hacking.organism_rl.generate import TaskSpec
from reward_hacking.organism_rl.rollout import EpisodeSource, TaskEpisodeSource, TaskFamily
from reward_hacking.vllm_server import VLLMServer, VLLMServerConfig

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

logger = logging.getLogger(__name__)
DEFAULT_FAMILIES: tuple[TaskFamily, ...] = ("ilcb", "aisi")
EVALUATION_ADAPTER_NAME = "organism-eval"
DEFAULT_MAX_TURNS = 8
DEFAULT_MAX_NEW_TOKENS = 2048


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse model serving, held-out selection, and episode settings."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    args.families = list(dict.fromkeys(args.family_entries or DEFAULT_FAMILIES))
    args.partition = "held-out"
    args.served_model_id = args.served_model_id or (
        EVALUATION_ADAPTER_NAME if args.adapter_dir is not None else args.model
    )
    _validate_args(args, parser)
    return args


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--served-model-id")
    parser.add_argument("--endpoint", help="use an already-running vLLM endpoint")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-lora-rank", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument(
        "--family", dest="family_entries", action="append", choices=("ilcb", "aisi")
    )
    parser.add_argument("--impossible-variant", choices=("oneoff", "conflicting"), default="oneoff")
    parser.add_argument("--tasks-per-family", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--announced-turn-budget", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--jail-backend")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_task_selection_args(args, parser)
    _validate_serving_args(args, parser)
    _validate_episode_limits(args, parser)
    _validate_sampling_args(args, parser)


def _validate_task_selection_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.impossible_variant == "conflicting" and "aisi" in args.families:
        parser.error("AISI does not support the conflicting impossible variant")
    if args.adapter_dir is None and args.served_model_id != args.model and args.endpoint is None:
        parser.error("served-model-id can differ from model only when serving a LoRA adapter")


def _validate_serving_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.endpoint is None and args.gpu_memory_utilization is None:
        parser.error("owned vLLM server requires --gpu-memory-utilization")
    if args.gpu_memory_utilization is not None and not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu-memory-utilization must be in (0, 1]")
    if args.max_lora_rank < 1:
        parser.error("max-lora-rank must be positive")


def _validate_episode_limits(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.tasks_per_family < 1 or args.concurrency < 1 or args.max_model_len < 1:
        parser.error("task, concurrency, and model-length limits must be positive")
    if min(args.max_turns, args.max_new_tokens) < 1:
        parser.error("turn and token limits must be positive")
    if args.thinking_token_budget is not None and args.thinking_token_budget < 0:
        parser.error("thinking-token-budget must be nonnegative")
    if args.announced_turn_budget is not None:
        if args.announced_turn_budget < 1:
            parser.error("announced-turn-budget must be positive")
        if args.max_turns not in {DEFAULT_MAX_TURNS, args.announced_turn_budget}:
            parser.error("announced-turn-budget must match max-turns")
        args.max_turns = args.announced_turn_budget


def _validate_sampling_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive and top-p must be in (0, 1]")


def resolve_model_revision(args: argparse.Namespace) -> str | None:
    """Resolve a cached Hub revision before starting an owned vLLM process."""
    if args.model_revision is not None:
        return args.model_revision
    if Path(args.model).is_dir():
        return None
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    revision = config._commit_hash  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("cached model config did not resolve a Hub revision")
    return revision


def build_server_config(
    args: argparse.Namespace,
    model_revision: str | None,
) -> VLLMServerConfig:
    """Build the vLLM server configuration for a base model or a served LoRA adapter."""
    if args.gpu_memory_utilization is None:
        raise ValueError("owned vLLM server requires --gpu-memory-utilization")
    has_adapter = args.adapter_dir is not None
    return VLLMServerConfig(
        model=args.model,
        revision=model_revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        host=args.server_host,
        port=args.server_port,
        language_model_only=True,
        max_num_seqs=1,
        enable_prefix_caching=True,
        enable_lora=has_adapter,
        max_lora_rank=args.max_lora_rank if has_adapter else None,
        lora_modules=(
            ((args.served_model_id, str(args.adapter_dir.resolve())),) if has_adapter else ()
        ),
        lora_serving_dir=args.output_dir / "vllm-serving-adapters" if has_adapter else None,
        reasoning_config=dict(QWEN_THINKING_REASONING_CONFIG),
        executable=str(Path(sys.executable).with_name("vllm")),
        log_path=args.output_dir / "vllm-server.log",
    )


@contextmanager
def server_endpoint(
    args: argparse.Namespace,
    model_revision: str | None,
) -> Generator[str]:
    """Use the external endpoint or own a vLLM process for the evaluation."""
    if args.endpoint is not None:
        yield args.endpoint
        return
    with VLLMServer(build_server_config(args, model_revision)) as server:
        yield server.base_url


def select_heldout_task_specs(
    source: TaskEpisodeSource,
    *,
    families: tuple[TaskFamily, ...],
    impossible_variant: str,
    tasks_per_family: int,
    seed: int,
) -> tuple[TaskSpec, ...]:
    """Select matched held-out problem IDs for impossible and original task variants."""
    if not families or len(families) != len(set(families)):
        raise ValueError("task families must be a nonempty unique sequence")
    if any(family not in {"ilcb", "aisi"} for family in families):
        raise ValueError(f"unsupported task family in {families!r}")
    if impossible_variant not in {"oneoff", "conflicting"}:
        raise ValueError(f"unsupported impossible variant {impossible_variant!r}")
    if impossible_variant == "conflicting" and "aisi" in families:
        raise ValueError("conflicting impossible tasks are only available for ILCB, not AISI")
    if type(tasks_per_family) is not int or tasks_per_family < 1:
        raise ValueError("tasks_per_family must be a positive integer")
    if type(seed) is not int:
        raise TypeError("evaluation seed must be an integer")

    rng = random.Random(seed)
    selected: list[TaskSpec] = []
    for family in sorted(families):
        impossible_by_problem = _task_ids_by_problem(source, family, impossible_variant)
        solvable_by_problem = _task_ids_by_problem(source, family, "original")
        paired_problem_ids = sorted(set(impossible_by_problem) & set(solvable_by_problem))
        if len(paired_problem_ids) < tasks_per_family:
            raise ValueError(
                f"family {family!r} has {len(paired_problem_ids)} matched held-out task pairs; "
                f"{tasks_per_family} requested"
            )
        for problem_id in rng.sample(paired_problem_ids, tasks_per_family):
            selected.extend(
                (
                    TaskSpec(
                        impossible_by_problem[problem_id], family, impossible_variant, "held-out"
                    ),
                    TaskSpec(solvable_by_problem[problem_id], family, "original", "held-out"),
                )
            )
    return tuple(sorted(selected, key=lambda spec: (spec.family, spec.variant, spec.task_id)))


def _task_ids_by_problem(
    source: TaskEpisodeSource,
    family: TaskFamily,
    variant: str,
) -> dict[str, str]:
    task_ids_by_problem: dict[str, str] = {}
    for task_id, task in source.tasks.items():
        if (
            source.families_by_task[task_id] != family
            or source.variants_by_task[task_id] != variant
        ):
            continue
        if not task.problem_id:
            raise ValueError(f"held-out task {task_id!r} has no problem id")
        if task.problem_id in task_ids_by_problem:
            raise ValueError(f"duplicate held-out {family} problem id {task.problem_id!r}")
        task_ids_by_problem[task.problem_id] = task_id
    return task_ids_by_problem


def run_evaluation(
    *,
    output_dir: Path,
    source: EpisodeSource,
    task_specs: Sequence[TaskSpec],
    concurrency: int,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    """Delegate episode execution, persistence, resumption, and aggregation to generate.py."""
    return generate.generate_episodes(
        output_dir=output_dir,
        source=source,
        task_specs=task_specs,
        samples_per_task=1,
        concurrency=concurrency,
        provenance=provenance,
    )


def _adapter_sha256(adapter_dir: Path) -> str:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter_dir}")
    digest = hashlib.sha256()
    for path in sorted(path for path in adapter_dir.rglob("*") if path.is_file()):
        relative_path = path.relative_to(adapter_dir).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        with path.open("rb") as adapter_file:
            for chunk in iter(lambda: adapter_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _evaluation_provenance(
    args: argparse.Namespace,
    *,
    model_revision: str | None,
    task_specs: Sequence[TaskSpec],
    source: TaskEpisodeSource,
) -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[2]
    return {
        "schema_version": 1,
        "model": args.model,
        "model_revision": model_revision,
        "served_model_id": args.served_model_id,
        "adapter_sha256": _adapter_sha256(args.adapter_dir) if args.adapter_dir else None,
        "endpoint_sha256": hashlib.sha256((args.endpoint or "owned-vllm").encode()).hexdigest(),
        "families": list(args.families),
        "impossible_variant": args.impossible_variant,
        "partition": "held-out",
        "tasks_per_family": args.tasks_per_family,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "max_model_len": args.max_model_len,
        "max_turns": args.max_turns,
        "max_new_tokens": args.max_new_tokens,
        "thinking_token_budget": args.thinking_token_budget,
        "announced_turn_budget": args.announced_turn_budget,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "jail_backend": args.jail_backend,
        "selected_tasks": [
            {
                "task_id": spec.task_id,
                "family": spec.family,
                "variant": spec.variant,
                "task_definition_sha256": generate.task_definition_sha256(
                    source.tasks[spec.task_id]
                ),
            }
            for spec in task_specs
        ],
        "source_sha256": {
            relative_path: hashlib.sha256(
                (repository_root / relative_path).read_bytes()
            ).hexdigest()
            for relative_path in (
                "reward_hacking/organism_rl/evaluate.py",
                "reward_hacking/organism_rl/generate.py",
                "reward_hacking/organism_rl/rollout.py",
                "reward_hacking/organism_reward.py",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Serve or connect to a model, then write held-out records and a grouped summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    generate.require_gitignored_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_revision = resolve_model_revision(args)
    if args.endpoint is None:
        server_config = build_server_config(args, model_revision)
        source_endpoint = server_config.base_url
    else:
        source_endpoint = args.endpoint
    source = TaskEpisodeSource(
        endpoint=source_endpoint,
        model_path=args.model,
        served_model_id=args.served_model_id,
        revision=model_revision,
        output_dir=args.output_dir / "episodes",
        max_model_len=args.max_model_len,
        max_turns=args.max_turns,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking_token_budget=args.thinking_token_budget,
        announced_turn_budget=args.announced_turn_budget,
        partition="held-out",
        families=tuple(args.families),
        variants=(args.impossible_variant, "original"),
        jail_backend=args.jail_backend,
    )
    task_specs = select_heldout_task_specs(
        source,
        families=tuple(args.families),
        impossible_variant=args.impossible_variant,
        tasks_per_family=args.tasks_per_family,
        seed=args.seed,
    )
    provenance = _evaluation_provenance(
        args,
        model_revision=model_revision,
        task_specs=task_specs,
        source=source,
    )
    logger.info(
        "selected held-out episodes: tasks_per_family=%d families=%s impossible_variant=%s",
        args.tasks_per_family,
        args.families,
        args.impossible_variant,
    )
    with server_endpoint(args, model_revision) as endpoint:
        if endpoint != source_endpoint:
            raise ValueError("served endpoint changed after task source construction")
        run_evaluation(
            output_dir=args.output_dir,
            source=source,
            task_specs=task_specs,
            concurrency=args.concurrency,
            provenance=provenance,
        )


if __name__ == "__main__":
    main()
