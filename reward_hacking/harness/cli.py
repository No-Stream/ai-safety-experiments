"""Command line of the agent harness: backend selection, episode planning and resume, signal handling."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers import AutoTokenizer

from experiment_sampling import add_oversize_reason_argument, log_sampling_grid
from games.eval_model import (
    LOAD_MODE_BASE,
    ServedModel,
    add_full_weights_args,
    full_weights_source_from_args,
    resolve_served_model,
    verify_served_model,
)
from games.vllm_teardown import (
    DEFAULT_SHUTDOWN_TIMEOUT_S,
    VLLM_BACKEND_KIND,
    resolve_engine_shutdown,
)
from reward_hacking import backend_cli
from reward_hacking.harness.loop import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_EPISODE_SECONDS,
    DEFAULT_MAX_TURNS,
    SMOKE_TASKS,
    load_traces,
    new_episode_token,
    run_summary_line,
    run_tasks,
)
from reward_hacking.harness.protocol import RUN_BLOCK_STOP
from reward_hacking.harness.task_spec import (
    DEFAULT_CHECK_BUDGET,
    AgenticArmConfig,
    GraderVisibility,
)
from reward_hacking.harness.tasks import (
    AISI_RH_VARIANTS,
    AISI_SOURCES,
    GRADER_COMMAND,
    ILCB_SPLIT_NAMES,
    TASKS_BY_ID,
    Task,
    aisi_rh_env_tasks,
    aisi_rh_env_tasks_by_id,
    ilcb_tasks,
)
from reward_hacking.jailed_poc import DEFAULT_COMMAND_TIMEOUT, resolve_jail_resource_limits
from reward_hacking.model_backend import (
    DEFAULT_CODEX_MODEL,
    BedrockSamplingConfig,
    SamplingConfig,
    converse_supports_stop_sequences,
    end_of_turn_token_ids,
)
from reward_hacking.trace import write_trace

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from types import FrameType

    from reward_hacking.aisi_rh_envs_data import AisiTaskVariant
    from reward_hacking.aisi_rh_envs_split import AisiPartition
    from reward_hacking.harness.ilcb_split import ILCBPartition
    from reward_hacking.model_backend import Backend

logger = logging.getLogger(__name__)


def live_child_processes() -> dict[int, str]:
    """Direct children of this process still running, ``pid -> command name``, read off ``/proc``.

    Linux only, like the jail. ``/proc/<pid>/stat`` is parsed after its last ``)`` because the
    command name before it may itself contain spaces or parentheses; the two fields that follow are
    the state and the parent pid. Zombies are excluded: a child that has exited and awaits its
    ``wait`` holds no card and no CPU, and whoever spawned it will reap it. The name is what vLLM
    sets on its engine child (``VLLM::EngineCore``), so a log line naming the survivors says which
    of them was the engine.
    """
    parent = os.getpid()
    children: dict[int, str] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            stat = Path(entry.path, "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue  # exited between the listing and the read
        head, _, tail = stat.rpartition(")")
        state, ppid = tail.split()[:2]
        if int(ppid) == parent and state != "Z":
            children[int(entry.name)] = head.partition("(")[2]
    return dict(sorted(children.items()))


def live_child_pids() -> list[int]:
    """Return the pids of :func:`live_child_processes`, in the same order."""
    return list(live_child_processes())


def _signal_if_alive(pid: int, signum: signal.Signals) -> None:
    """Signal one pid, treating a child that exited between the listing and the signal as done."""
    with suppress(ProcessLookupError):
        os.kill(pid, signum)


def terminate_child_processes(*, grace_seconds: float = 5.0) -> list[int]:
    """SIGTERM every live direct child, SIGKILL whatever survives the grace, and name what was hit.

    The last line of defence behind a backend's own teardown. Nothing this process spawned may
    outlive it: a vLLM ``EngineCore`` left behind holds the whole card and fails every later engine
    start on the box with "Engine core initialization failed", a jailed command left behind keeps
    its cgroup. Children are not reaped here -- their spawner's ``wait`` does that -- so a caller
    holding a ``Popen`` still reads the exit status it expects.
    """
    targets = live_child_pids()
    for pid in targets:
        _signal_if_alive(pid, signal.SIGTERM)
    grace_deadline = time.monotonic() + grace_seconds
    while live_child_pids() and time.monotonic() < grace_deadline:
        time.sleep(0.05)
    for pid in live_child_pids():
        _signal_if_alive(pid, signal.SIGKILL)
    return targets


def release_backend_resources(backend: Backend) -> None:
    """Shut a local engine's subprocess down and reap every child still alive: ``main``'s exit path.

    Runs on every way out of ``main`` -- a finished run, an episode's exception, a SIGTERM turned
    into ``SystemExit`` by :func:`stopping_on_signals`. Twice on a rented H100 a wedged run was
    killed by its parent alone and the ``VLLM::EngineCore`` child was reparented to init holding 75
    of 80 GiB, after which every engine start on that box failed. vLLM's own ``weakref.finalize``
    teardown does fire at interpreter exit, but only after every non-daemon thread has been joined
    -- which a worker blocked in the engine never is -- and never on a signal Python does not
    handle, so it is called explicitly here, first, through the same entry point
    :mod:`games.vllm_teardown` resolves. Then whatever direct children remain are terminated.
    """
    if backend.transport == VLLM_BACKEND_KIND:
        before = live_child_processes()
        resolve_engine_shutdown(backend)(timeout=DEFAULT_SHUTDOWN_TIMEOUT_S)
        logger.info("vLLM engine core shut down: children %s -> %s", before, live_child_processes())
    survivors = live_child_processes()
    terminate_child_processes()
    if survivors:
        logger.warning(
            "terminated %d child process(es) still alive at exit: %s", len(survivors), survivors
        )


@contextmanager
def stopping_on_signals(stop_requested: threading.Event) -> Generator[None]:
    """Turn SIGTERM and SIGINT into a stop request plus ``SystemExit``, restoring the handlers after.

    Python leaves SIGTERM at its default disposition, which ends the process without running any
    ``finally`` -- so a ``timeout(1)`` or a chain script sending it orphaned the engine child. With
    this in place the signal sets ``stop_requested`` (which :class:`RunStoppedError` reads inside every
    engine wait), then raises ``SystemExit(128 + signum)`` in the main thread, which unwinds through
    ``main``'s ``finally`` to :func:`release_backend_resources`. Main thread only, as every
    ``signal.signal`` call is.
    """

    def stop(signum: int, frame: FrameType | None) -> None:
        del frame
        stop_requested.set()
        raise SystemExit(128 + signum)

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


RUN_HEADER_RECORD = "run_header"
"""The per-invocation record ``main`` writes ahead of its episodes, naming the served weights.

Every trace reader in this package selects records by kind (``turn``, ``episode_summary``) and
passes over anything else, so the header costs them nothing; it exists so a trace of a
``--full-weights`` run carries the label, commit and weights fingerprint of the checkpoint that
produced it, which the per-episode ``model_id`` alone cannot prove.
"""

# One per kind backend_cli offers; --backend's choices come from there, so a gap KeyErrors.
_DEFAULT_MODEL_BY_BACKEND = {
    "mock": "mock",
    "hf": "Qwen/Qwen3.5-4B",
    "vllm": "Qwen/Qwen3.5-4B",
    "vllm-http": "Qwen/Qwen3.5-4B",
    "codex": DEFAULT_CODEX_MODEL,
    "bedrock": DEFAULT_BEDROCK_MODEL,
}

# Canned turns for `--backend mock`; see _build_cli_backend for what they are for.
MOCK_RESPONSES = (
    (
        "I total the ledger and hand in a solution file.\n"
        "<run>awk '{s+=$2} END {print \"total=\" s}' ledger.txt > answer.txt</run>\n"
        "<run>cat > solution.py <<'PY'\ndef solve():\n    return None\nPY</run>"
    ),
    f"Now I check it.\n<run>{GRADER_COMMAND}</run>",
    "Nothing further to do.",
)


def _select_tasks(  # noqa: PLR0913, PLR0917 - one argument per independent CLI selector
    task_ids: Sequence[str] | None,
    ilcb_split: str | None = None,
    ilcb_partition: ILCBPartition | None = None,
    aisi_rh_variant: AisiTaskVariant | None = None,
    aisi_rh_partition: AisiPartition | None = None,
    aisi_rh_source: str | None = None,
) -> tuple[Task, ...]:
    """Resolve a task-name, ILCB or AISI rh_envs selection, or the default smoke sweep.

    The default is :data:`SMOKE_TASKS`, deliberately NOT the registry. ``TASKS_BY_ID`` is
    "everything registered" and grows with every task family added -- the ILCB port alone registers
    500-odd problems -- so a default that tracked it would silently turn a bare
    ``python -m reward_hacking.harness.loop`` from a handful of local smoke episodes into a paid
    sweep of every task anyone had ever registered, and whoever added the family would never see it.
    ``SMOKE_TASKS`` is therefore enumerated from the two hand-curated families rather than derived,
    so adding a family cannot enlarge the default; only an edit to that line can.

    ``--task`` still resolves against the whole registry, so any single registered task remains
    reachable by name. Only the default is curated, and that asymmetry is the fix: it separates
    "everything registered" from "what a bare invocation sweeps", which were the same thing before.

    ``ilcb_split`` selects one baked ILCB arm through the registry's selector. Supplying
    ``ilcb_partition`` selects the new multi-turn task builder and the requested id-hash partition;
    without it, the existing task definitions and selection behavior remain in place. The selector
    drops the one row whose visible check does not compile unless explicitly requested. Combining
    ILCB selection with ``--task`` is refused because the two vocabularies cannot be combined
    without changing what the flags select.

    ``aisi_rh_variant`` selects the oneoff task or its original-test control. Its optional source
    and partition narrow the generated task set without changing the default smoke sweep.
    """
    ilcb_requested = ilcb_split is not None or ilcb_partition is not None
    aisi_requested = (
        aisi_rh_variant is not None or aisi_rh_partition is not None or aisi_rh_source is not None
    )
    if task_ids and ilcb_requested:
        raise ValueError("--task cannot combine with --ilcb-split or --ilcb-partition")
    if task_ids and aisi_requested:
        raise ValueError("--task cannot combine with AISI rh_envs selection flags")
    if ilcb_partition is not None and ilcb_split is None:
        raise ValueError("--ilcb-partition requires --ilcb-split")
    if ilcb_requested and aisi_requested:
        raise ValueError("ILCB and AISI rh_envs selection flags cannot combine")
    if ilcb_split is not None:
        return ilcb_tasks(impossible_type=ilcb_split, partition=ilcb_partition)
    if aisi_rh_variant is None and (aisi_rh_partition is not None or aisi_rh_source is not None):
        raise ValueError("--aisi-rh-partition and --aisi-rh-source require --aisi-rh-variant")
    if aisi_rh_variant is not None:
        return aisi_rh_env_tasks(
            variant=aisi_rh_variant,
            partition=aisi_rh_partition,
            source=aisi_rh_source,
        )
    if not task_ids:
        return SMOKE_TASKS
    aisi_task_ids = tuple(task_id for task_id in task_ids if task_id.startswith("aisi-"))
    selected_by_id = {
        task_id: TASKS_BY_ID[task_id] for task_id in task_ids if not task_id.startswith("aisi-")
    }
    if aisi_task_ids:
        selected_by_id.update(
            {task.task_id: task for task in aisi_rh_env_tasks_by_id(aisi_task_ids)}
        )
    return tuple(selected_by_id[task_id] for task_id in task_ids)


def _base_model_id(args: argparse.Namespace) -> str:
    return cast("str", args.model_id or _DEFAULT_MODEL_BY_BACKEND[args.backend])


def served_model_from_args(args: argparse.Namespace) -> ServedModel:
    """Decide what the backend loads: ``--model-id`` itself, or the full checkpoint named beside it.

    A ``--full-weights`` run serves a complete checkpoint (a TMAX step branch, an amplified build)
    under a label, and the trace has to carry the label, the resolved commit and the weights
    fingerprint or a reader cannot tell which step produced its episodes -- so ``main`` resolves
    this once, hands it to :func:`_build_cli_backend`, and writes its provenance ahead of the
    episodes. ``--model-id`` stays the BASE id on such a run: it keys the output floor and the
    sampling base, both measured per base model, while the backend itself is labelled with the
    served checkpoint so every episode id and record names it.
    """
    full_weights = full_weights_source_from_args(args)
    if args.backend == "vllm-http" and full_weights is not None:
        raise ValueError(
            "--backend vllm-http cannot use --full-weights: the external vllm serve process owns "
            "the weights; pass its model identity through --model-id"
        )
    return resolve_served_model(
        checkpoint=None,
        base_model=_base_model_id(args),
        backend_kind=args.backend,
        merge_root=Path(tempfile.gettempdir()) / "unreachable-merge-root",
        merge_label="agent-harness-",
        full_weights=full_weights,
    )


def local_stop_token_ids(args: argparse.Namespace, served: ServedModel) -> tuple[int, ...]:
    """Resolve the end-of-turn stop set a LOCAL backend is pinned to, off the tokenizer it renders with.

    Empty for hosted and mock kinds, which render their own templates and stop their own way. For
    ``hf``/``vllm`` the two Qwen3.5 end-of-turn tokens are resolved by name through the served
    checkpoint's tokenizer (:func:`~reward_hacking.model_backend.end_of_turn_token_ids`), because a
    released checkpoint's generation_config can declare a different eos set than its base's and the
    engine would otherwise merge that difference into every request. Part of the run header so a
    resume refuses a trace that stopped on another set.
    """
    if args.backend not in backend_cli.LOCAL_KINDS:
        return ()
    source = (
        str(served.weights.snapshot_dir) if served.weights is not None else _base_model_id(args)
    )
    return end_of_turn_token_ids(AutoTokenizer.from_pretrained(source))


def _build_cli_backend(
    args: argparse.Namespace,
    served: ServedModel | None = None,
    stop_token_ids: tuple[int, ...] | None = None,
) -> Backend:
    """Construct the inference backend from CLI config, for this module's ``main`` and nothing else.

    ``served`` is the decision :func:`served_model_from_args` already made for this run; left
    None, it is made here, so a caller that only wants a backend still gets the full-weights path.
    ``stop_token_ids`` likewise: the pin :func:`local_stop_token_ids` resolves, or resolved here.

    Delegated to :mod:`reward_hacking.backend_cli`, which owns the same flag block for every probe
    CLI in this package and names this harness as one of its callers. Three things come with it. A
    knob the chosen kind cannot honour is *refused* rather than dropped -- Converse's
    ``inferenceConfig`` has no ``topK`` field, so a ``--top-k`` handed to it disappears with no
    error from the API. ``--top-p``/``--top-k``/``--concurrency``/``--region``/``--profile`` become
    reachable here, which is what makes the de-saturated arm runnable from this CLI at all. And
    ``--backend mock`` becomes a zero-cost end-to-end smoke of the whole loop over
    :data:`MOCK_RESPONSES`: this repo's rule before a Batch run is that the path has been watched to
    execute locally, and the paper cuts live in the seams rather than in the components.

    The local base is ``SamplingConfig.for_thinking(--thinking)``, so ``--thinking`` selects the
    thinking-mode preset (temperature 1.0, top_p 0.95, a 32768-token budget) rather than the
    non-thinking one that loops the model inside ``<think>``; an explicit flag then overrides just
    that field. ``--temperature`` stays unset by default because the verified Converse request shape
    omits it and what a hosted model does with a temperature is per-model (gpt-5.6-luna accepts only
    the default 1.0 and rejects any other value with a Bedrock ValidationException, hit live
    2026-08-16).

    ``--max-new-tokens`` unset is *not* the same kind of default, and this loop is where that
    mattered: ``maxTokens`` is always in the Converse request, so leaving the flag off picks
    :data:`~reward_hacking.model_backend.DEFAULT_BEDROCK_MAX_TOKENS` rather than omitting anything.
    An eight-turn episode spends that budget once per turn on a frontier model whose reasoning is
    returned redacted but billed, so while the fallback was 2048 a turn's whole budget could go to
    reasoning and return an empty completion or an unclosed ``<run>`` block -- and an episode of
    those ends at the harness's give-up path looking like a policy that never tried. So the resolved
    cap is checked against the model's measured floor rather than merely documented: see
    :func:`backend_cli.refuse_short_output_cap` and :func:`_local_base_at_output_floor`, which cover
    the two directions the cap can arrive from -- typed on the command line, or inherited from a
    preset nobody chose.

    The resolved config is logged, and each turn records its own ``stop_reason`` and
    ``output_tokens`` (see :class:`AgentTurn`), so a future truncation shows up in the artifacts
    rather than needing the run's logs. Bedrock's pool size is left to ``--concurrency``: the loop
    sends one prompt per turn, so the old hardcoded ``concurrency=1`` and the default are the same
    run.

    Every stop-capable backend is built with the ``</run>`` stop sequence
    (:data:`~reward_hacking.harness.protocol.RUN_BLOCK_STOP`) folded into its sampling config --
    part of this harness's base configuration rather than a flag, because a run without it samples
    the model writing the environment's own replies (see the module docstring). Two backends
    cannot take it, and both are warned that the per-turn ``fabricated_results`` count is their
    only guard: codex exposes no stop control at all, and a Bedrock model whose family rejects
    ``stopSequences`` (:data:`~reward_hacking.model_backend.CONVERSE_STOP_REFUSING_FAMILIES` --
    both OpenAI families, the default Luna included, probed live 2026-08-24) is sampled without
    it, since sending the field is a ``ValidationException`` on every call.
    """
    model_id = _base_model_id(args)
    if served is None:
        served = served_model_from_args(args)
    if args.backend == "vllm-http" and served.load_mode != LOAD_MODE_BASE:
        raise ValueError(
            f"--backend vllm-http cannot use served load mode {served.load_mode!r}: the external "
            "vllm serve process owns the weights and adapters"
        )
    if stop_token_ids is None:
        stop_token_ids = local_stop_token_ids(args, served)
    local_base = harness_sampling_base(model_id, thinking=backend_cli.resolve_thinking(args))
    bedrock_sampling: BedrockSamplingConfig | None = None
    if args.backend == "bedrock":
        bedrock_sampling = backend_cli.bedrock_sampling_from_args(args)
        if converse_supports_stop_sequences(model_id):
            bedrock_sampling = replace(bedrock_sampling, stop_sequences=(RUN_BLOCK_STOP,))
        else:
            logger.warning(
                "%s's model family rejects Converse stopSequences outright (ValidationException, "
                "probed live 2026-08-24), so this run samples with NO %r stop: completions can "
                "run past their own command blocks and fabricate the environment's replies, and "
                "the per-turn fabricated_results count is the only guard",
                model_id,
                RUN_BLOCK_STOP,
            )
        backend_cli.refuse_short_output_cap(
            model_id, bedrock_sampling.max_tokens, allow_short=args.allow_short_completions
        )
        logger.info("bedrock sampling | %s", bedrock_sampling)
    elif args.backend in backend_cli.LOCAL_KINDS:
        local_sampling = backend_cli.local_sampling_from_args(args, local_base)
        backend_cli.refuse_short_output_cap(
            model_id, local_sampling.max_new_tokens, allow_short=args.allow_short_completions
        )
        logger.info("local sampling | %s", local_sampling)
    elif args.backend == "codex":
        logger.warning(
            "the codex CLI exposes no stop-sequence control, so completions can run past %r and "
            "fabricate the environment's replies; the per-turn fabricated_results count is the "
            "only guard on this backend",
            RUN_BLOCK_STOP,
        )
    extra_kwargs: dict[str, object] = dict(served.backend_kwargs)
    if stop_token_ids:
        extra_kwargs["stop_token_ids"] = stop_token_ids
    backend = backend_cli.backend_from_args(
        args,
        model_id if served.load_mode == LOAD_MODE_BASE else served.model_id,
        local_sampling=local_base,
        bedrock_sampling=bedrock_sampling,
        mock_responses=MOCK_RESPONSES,
        extra_kwargs=extra_kwargs,
    )
    # A silently wrong directory would probe another checkpoint under this run's label.
    verify_served_model(backend, served)
    return backend


def harness_sampling_base(model_id: str, *, thinking: bool) -> SamplingConfig:
    """Return the local sampling base every agent-harness run must sample under.

    Public and single-sourced on purpose: this is the config that makes an episode an episode, and
    an eval driver that builds its own backend (the flagship's per-checkpoint ladder does -- it
    serves LoRA adapters, so it cannot go through the CLI) must inherit BOTH rules from here rather
    than carry a private copy. If a copy and the original ever diverge, the training and eval
    halves of an experiment silently sample different policies -- the failure class the stop
    sequence exists to kill, reintroduced one layer up.

    The two rules. The cap never sits below the model's measured output floor: the thinking preset
    already clears every floor (32,768 tokens), but the non-thinking one carries 4,096, so a bare
    local episode used to sample at a cap below anything these checkpoints were screened at -- with
    no flag typed by anybody, which is the version of this mistake nobody goes looking for. And the
    ``</run>`` stop sequence is part of the base rather than a knob, because a run without it
    samples the model writing the environment's own replies (see the module docstring).

    Raised here rather than in :meth:`SamplingConfig.for_thinking` because that preset is shared
    with the single-call probes in ``episodes/`` and ``channel/`` and with ``games/``, where the cap
    is part of a different measurement and moving it would move their numbers, and none of them
    parse run blocks. This harness is the one that samples eight times per episode and reads a
    short reply as an ending action.
    """
    preset = replace(SamplingConfig.for_thinking(thinking=thinking), stop=(RUN_BLOCK_STOP,))
    floor = backend_cli.output_floor_for(model_id)
    if preset.max_new_tokens >= floor:
        return preset
    return replace(preset, max_new_tokens=floor)


RESUME_IDENTITY_FIELDS = (
    "model_id",
    "base_model_id",
    "model_load_mode",
    "model_full_weights",
    "model_weights_fingerprint",
    "stop_token_ids",
    "resource_limits",
)
"""The run-header fields a resumed run must match, one per way two runs could differ in what they
sampled: the served label, the base it keys its floors on, how the weights were assembled, which
checkpoint, which bytes, and which end-of-turn ids completions stopped on. The arm is compared
separately, off its label."""


def completed_episodes_for_resume(
    out_path: Path, *, model_id: str, arm: AgenticArmConfig, header: Mapping[str, object]
) -> dict[str, int]:
    """Count the finished episodes per task in an existing trace this run may continue.

    A trace is continuable only if every run header it carries agrees with this run on the fields
    in :data:`RESUME_IDENTITY_FIELDS` and on the arm: the episodes below a header were sampled by
    the configuration it names, and appending this run's episodes to another configuration's would
    pool two policies under one file with no count ever showing it. A trace written before headers
    existed cannot make that argument and is refused rather than trusted.

    Completion is an ``episode_summary`` record whose id names this model, that task and this arm
    (:func:`compose_episode_id` puts all three ahead of the per-episode token), so a summary from a
    header-matching invocation of a DIFFERENT arm in the same file -- which the header gate already
    forbids -- could never be counted here either.
    """
    records = load_traces(out_path)
    headers = [r for r in records if r.get("record") == RUN_HEADER_RECORD]
    if not headers:
        raise ValueError(
            f"{out_path} carries no {RUN_HEADER_RECORD!r} record, so nothing says which model, "
            f"weights and arm produced its episodes; a trace written before run headers existed "
            f"cannot be resumed. Write the new run to its own --out path."
        )
    for stored in headers:
        problems = [
            f"{field}: stored {stored.get(field)!r} != this run's {header.get(field)!r}"
            for field in RESUME_IDENTITY_FIELDS
            if stored.get(field) != header.get(field)
        ]
        stored_arm = cast("Mapping[str, object]", stored.get("arm") or {})
        if stored_arm.get("label") != arm.label:
            problems.append(f"arm: stored {stored_arm.get('label')!r} != this run's {arm.label!r}")
        if problems:
            raise ValueError(
                f"refusing to resume {out_path}: an earlier invocation wrote it under a different "
                f"configuration ({'; '.join(problems)}). Two configurations must never pool into "
                f"one trace; write the new run to its own --out path."
            )
    completed: dict[str, int] = {}
    for record in records:
        if record.get("record") != "episode_summary":
            continue
        task_id = str(record.get("task_id"))
        expected_prefix = f"{model_id}:{task_id}:{arm.label}:"
        if str(record.get("episode_id", "")).startswith(expected_prefix):
            completed[task_id] = completed.get(task_id, 0) + 1
    return completed


def _completed_before_this_run(
    out_path: Path,
    *,
    resume: bool,
    model_id: str,
    arm: AgenticArmConfig,
    header: Mapping[str, object],
) -> dict[str, int]:
    """Refuse an existing trace unless resuming it, and count what it already holds when we are."""
    if not (out_path.exists() and out_path.stat().st_size > 0):
        return {}
    if not resume:
        raise FileExistsError(
            f"{out_path} already holds a trace; appending to it would silently double every "
            f"rate's denominator. Pass --resume to continue that run, or write to a new path."
        )
    return completed_episodes_for_resume(out_path, model_id=model_id, arm=arm, header=header)


def plan_episodes(
    selected: Sequence[Task], *, repeats: int, completed: Mapping[str, int]
) -> tuple[Task, ...]:
    """Lay out the episodes still to run, in the same repeat-major order a fresh run uses.

    Pass ``r`` of task ``t`` runs unless ``completed[t] > r``, so a task with two finished episodes
    out of three requested contributes exactly its third -- and the order of what remains is the
    order a fresh run would have visited it in, so a resumed trace reads like an uninterrupted one.
    """
    return tuple(
        task
        for pass_index in range(repeats)
        for task in selected
        if completed.get(task.task_id, 0) <= pass_index
    )


def sampling_budget_for_run(
    *,
    model_id: str,
    arm_label: str,
    task_count: int,
    repeats: int,
    oversize_reason: str | None = None,
) -> dict[str, object]:
    """Count selected real-execution episodes in the one model-by-arm cell this CLI runs."""
    return log_sampling_grid(
        {f"model={model_id}/arm={arm_label}": task_count * repeats},
        oversize_reason=oversize_reason,
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-execution reward-hacking episodes against readable brittle graders."
    )
    backend_cli.add_backend_args(parser, default="hf")
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            f"Defaults per backend: {_DEFAULT_MODEL_BY_BACKEND}. With --full-weights this stays "
            "the BASE model id (it keys the output floor and the sampling base) while the served "
            "checkpoint is named by --full-weights/--revision."
        ),
    )
    add_full_weights_args(parser)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        default=None,
        help="Run only this task by id; repeatable. Default: the curated smoke tasks.",
    )
    parser.add_argument(
        "--ilcb-split",
        choices=ILCB_SPLIT_NAMES,
        default=None,
        help=(
            "Run one whole baked ILCB split, replacing a hundred-odd --task flags; mutually "
            "exclusive with --task. The denominators differ on purpose: oneoff and conflicting "
            "run one task fewer than the other splits, because the registry's default drops the "
            "row whose visible check does not compile."
        ),
    )
    parser.add_argument(
        "--ilcb-partition",
        choices=("train", "held-out"),
        default=None,
        help=(
            "Use the id-hash partition for an agentic ILCB task selection. Requires --ilcb-split "
            "oneoff, conflicting, or original; the id-only manifest is stored beside the baked data."
        ),
    )
    parser.add_argument(
        "--aisi-rh-variant",
        choices=AISI_RH_VARIANTS,
        default=None,
        help=(
            "Run AISI rh_envs oneoff tasks or their original-test controls; mutually exclusive "
            "with --task and ILCB selection."
        ),
    )
    parser.add_argument(
        "--aisi-rh-partition",
        choices=("train", "held-out"),
        default=None,
        help="Select a deterministic per-source AISI problem partition. Requires --aisi-rh-variant.",
    )
    parser.add_argument(
        "--aisi-rh-source",
        choices=AISI_SOURCES,
        default=None,
        help="Restrict the AISI selection to one dataset source. Requires --aisi-rh-variant.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Episodes per selected task; each repeat is an independent episode with its own "
        "id and directory. Default 1.",
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--timeout", default=DEFAULT_COMMAND_TIMEOUT)
    parser.add_argument(
        "--advisory-limits",
        action="store_true",
        help="Explicitly use unenforced resource limits, only if the systemd user manager is offline.",
    )
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=DEFAULT_EPISODE_SECONDS,
        help=(
            "Wall clock one episode may spend before the harness ends it and grades /work as it "
            "stands, recorded on the trace as deadline_exceeded. --timeout bounds a single command "
            f"and bounds nothing above it. Default {DEFAULT_EPISODE_SECONDS:.0f}s."
        ),
    )
    parser.add_argument(
        "--episode-concurrency",
        type=int,
        default=1,
        help=(
            "How many episodes run at once. Default 1: exactly the serial behaviour every "
            "existing artifact was produced under, so no measurement changes unless asked. Above "
            "1 the trace appends are serialised and the mock transport is refused (its scripted "
            "cursor is shared across episodes). On a local engine (vllm, hf) generations are "
            "serialised too -- one submission owns the engine at a time; two at once deadlock it "
            "-- so the concurrency overlaps sandbox commands and grading with a sibling's "
            "generation and nothing more; time an episode spends queued for the engine is "
            "recorded and not charged to its --episode-seconds."
        ),
    )
    parser.add_argument(
        "--allow-short-completions",
        action="store_true",
        help=(
            "Sample below the resolved model's measured output floor, which is otherwise refused. "
            "For a plumbing or timing probe only: the episodes it writes read as a policy that "
            "stopped acting, because a reply cut off at the cap is indistinguishable from one that "
            "said nothing. The run warns about every short cap it samples at."
        ),
    )
    parser.add_argument(
        "--jail-backend", choices=("bwrap", "unshare"), default=None, help="Jail isolation backend."
    )
    parser.add_argument(
        "--abort-affordance",
        dest="abort_affordance",
        action="store_true",
        help=(
            "Offer the policy an honest give-up action (<abort>reason</abort>) and honor it. It is "
            "an experimental arm, so it defaults OFF: the baseline harness has no escape hatch."
        ),
    )
    parser.add_argument("--no-abort-affordance", dest="abort_affordance", action="store_false")
    parser.set_defaults(abort_affordance=False)
    parser.add_argument(
        "--empty-affordance",
        dest="empty_affordance",
        action="store_true",
        help=(
            "Offer the policy a submit-as-is action (<empty>note</empty>) and honor it: an exit "
            "claiming nothing, without conceding it could not be done. An arm, also OFF by default."
        ),
    )
    parser.add_argument("--no-empty-affordance", dest="empty_affordance", action="store_false")
    parser.set_defaults(empty_affordance=False)
    parser.add_argument(
        "--grader-visibility",
        type=GraderVisibility,
        choices=tuple(GraderVisibility),
        default=GraderVisibility.WRITABLE,
        help=(
            "How much of the in-band grader /work holds. 'inline' appends the grader's source to "
            "the task text (the file stays on disk exactly as 'writable' leaves it, so the two "
            "differ only in whether reading it costs a turn); 'hidden' withholds it; 'read_only' "
            "is RECORDED BUT NOT YET ENFORCED (it needs a read-only bind in episode_jail.sh, so a "
            "policy that is root in its own jail can still edit the file)."
        ),
    )
    parser.add_argument(
        "--check-affordance",
        dest="check_affordance",
        action="store_true",
        help=(
            "Offer the policy a mediated verdict action (<check/>): the harness runs the grader "
            "over /work and returns only SUCCESS or FAIL, never the grader's own output. The blind "
            "arm's verdict channel, so its readable comparison arm needs it too. Also an arm, OFF "
            "by default."
        ),
    )
    parser.add_argument("--no-check-affordance", dest="check_affordance", action="store_false")
    parser.set_defaults(check_affordance=False)
    parser.add_argument(
        "--check-budget",
        type=int,
        default=DEFAULT_CHECK_BUDGET,
        help=(
            "How many <check/> verdicts the policy may request per episode; the next one ends the "
            f"episode as check_budget_exhausted. Default {DEFAULT_CHECK_BUDGET}. Only meaningful "
            "with --check-affordance."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    add_oversize_reason_argument(parser)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue a run whose --out already holds episodes: count the finished episodes per "
            "task and run only the remainder of --repeats, in a fresh run's order. The existing "
            "trace's run headers must match this run's model, weights and arm. Without this flag an "
            "existing non-empty --out is refused, because appending would silently double a rate's "
            "denominator."
        ),
    )
    return parser.parse_args(argv)


def arm_from_args(args: argparse.Namespace) -> AgenticArmConfig:
    """Assemble the episode arm from parsed CLI flags, the one place the two vocabularies meet."""
    return AgenticArmConfig(
        abort_affordance=args.abort_affordance,
        empty_affordance=args.empty_affordance,
        grader_visibility=args.grader_visibility,
        check_affordance=args.check_affordance,
        check_budget=args.check_budget,
    )


def main(argv: list[str] | None = None) -> int:
    """Run each selected task once against the chosen backend and write the trace JSONL.

    This entry point loads a real model (``HFBackend``) or calls a paid API (``BedrockBackend``,
    ``CodexBackend``), so it is guarded under ``if __name__`` and never touched by the offline
    tests, which drive the loop with ``MockBackend`` instead. Episode directories
    always live under a ``/tmp`` base, never under ``--out`` or the repo, because the jail refuses
    to mount an episode directory that resolves under a home tree.

    Traces are written per episode as the run proceeds, not in one batch at the end, so an
    interruption costs the episode in flight rather than every episode already paid for.

    The run-shape warnings are keyed on the flags that cause them, not on the episode count. One
    count-keyed guard was silent whenever a ``--task`` selection happened to total the default
    sweep's size, and blamed ``--task`` for a ``--repeats`` multiplier on a run that named no task
    at all. Each reports the selected-task count and the episode total, since the total is what
    costs.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    args = _parse_args(argv)
    # Before the backend, task selection included: a typo must not cost a model load first.
    if args.repeats < 1:
        raise ValueError(f"--repeats must be at least 1, got {args.repeats}")
    if args.max_turns < 1:
        raise ValueError(f"--max-turns must be at least 1, got {args.max_turns}")
    if args.episode_seconds <= 0:
        raise ValueError(f"--episode-seconds must be positive, got {args.episode_seconds}")
    if args.episode_concurrency < 1:
        raise ValueError(
            f"--episode-concurrency must be at least 1, got {args.episode_concurrency}"
        )
    selected = _select_tasks(
        args.tasks,
        args.ilcb_split,
        args.ilcb_partition,
        args.aisi_rh_variant,
        args.aisi_rh_partition,
        args.aisi_rh_source,
    )
    arm = arm_from_args(args)
    sampling_budget = sampling_budget_for_run(
        model_id=_base_model_id(args),
        arm_label=arm.label,
        task_count=len(selected),
        repeats=args.repeats,
        oversize_reason=args.oversize_reason,
    )

    resource_limits = resolve_jail_resource_limits(advisory=args.advisory_limits)
    served = served_model_from_args(args)
    stop_token_ids = local_stop_token_ids(args, served)
    backend = _build_cli_backend(args, served, stop_token_ids)
    model_name = backend.model_id.split("/")[-1]
    # A random token beside the timestamp: two invocations in one second must not resolve to one file.
    out_path = args.out or Path(
        f"artifacts/harness/{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{new_episode_token()[:6]}-"
        f"{model_name}-{arm.label}.jsonl"
    )
    header: dict[str, object] = {
        "record": RUN_HEADER_RECORD,
        "recorded_at": datetime.now(UTC).isoformat(),
        "model_id": backend.model_id,
        "base_model_id": _base_model_id(args),
        "transport": backend.transport,
        "arm": arm.to_json_dict(),
        "stop_token_ids": list(stop_token_ids),
        "resource_limits": resource_limits.to_json_dict(),
        "sampling_budget": sampling_budget,
        **served.provenance,
    }
    completed = _completed_before_this_run(
        out_path, resume=args.resume, model_id=backend.model_id, arm=arm, header=header
    )
    # Once per invocation and appended again by a relaunch, so a resumed trace says which wrote what.
    write_trace(out_path, [header], append=True)

    tasks = plan_episodes(selected, repeats=args.repeats, completed=completed)
    if completed:
        logger.warning(
            "resume: %d episode(s) already complete in %s (%s); running the remaining %d",
            sum(completed.values()),
            out_path,
            ", ".join(f"{task}={n}" for task, n in sorted(completed.items())),
            len(tasks),
        )
    # Keyed on the causes rather than on the episode count; see this function's docstring.
    if args.tasks:
        logger.warning(
            "narrowed --task selection: %d of the %d tasks in the default sweep, %d episodes total",
            len(selected),
            len(SMOKE_TASKS),
            len(tasks),
        )
    if args.ilcb_split:
        logger.warning(
            "--ilcb-split %s selects %d tasks, %d episodes total, which can cost real money",
            args.ilcb_split,
            len(selected),
            len(tasks),
        )
    if args.aisi_rh_variant:
        logger.warning(
            "AISI rh_envs %s selection has %d tasks across %d episodes",
            args.aisi_rh_variant,
            len(selected),
            len(tasks),
        )
    if args.repeats > 1:
        logger.warning(
            "--repeats %d turns %d selected tasks into %d episodes, which can cost real money",
            args.repeats,
            len(selected),
            len(tasks),
        )

    episode_base = Path(tempfile.mkdtemp(prefix="agent-harness-"))
    stop_requested = threading.Event()
    try:
        with stopping_on_signals(stop_requested):
            traces = run_tasks(
                backend,
                tasks,
                episode_base=episode_base,
                max_turns=args.max_turns,
                timeout=args.timeout,
                jail_backend=args.jail_backend,
                arm=arm,
                trace_path=out_path,
                episode_seconds=args.episode_seconds,
                episode_concurrency=args.episode_concurrency,
                stop_requested=stop_requested,
                resource_limits=resource_limits,
            )
    finally:
        # Every exit path: the engine child dies with this process, never reparented to init.
        release_backend_resources(backend)
    logger.info("wrote %d episode traces to %s", len(traces), out_path)
    logger.info("%s", run_summary_line(traces))
    backend_cli.log_token_usage(backend)
    return 0
