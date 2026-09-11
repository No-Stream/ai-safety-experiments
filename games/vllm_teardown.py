"""Release a vLLM engine's VRAM before the next one loads, and prove the card gave it back.

Written after the reasoning-transplant run (:mod:`games.interp_mediation`) died at its recipient swap
on 2026-08-24. The first recipient's four cells completed and banked 512 records; the second
recipient's engine then refused to start::

    ValueError: Free memory on device cuda:0 (2.96/44.39 GiB) on startup is less than desired GPU
    memory utilization (0.92, 40.84 GiB). Decrease GPU memory utilization or reduce GPU memory used
    by other processes.

which reached the driver as vLLM's far less informative ``RuntimeError: Engine core initialization
failed``. 2.96 GiB free on a 44.39 GiB card is the first engine still holding essentially the whole
of its claim (0.92 x 44.39 = 40.84 GiB) sixteen seconds after its context manager exited. That is not
a teardown racing a drain, it is a teardown that never happened.

Two independent reasons it never happened, and both matter, because fixing either alone leaves the
bug in place:

* **The reference was never dropped.** The old teardown was ``del backend`` inside a
  ``@contextmanager`` generator's ``finally``. That unbinds the generator frame's own name while the
  caller's ``with ... as backend:`` target still holds the object, so the refcount never reaches
  zero, the ``weakref.finalize`` vLLM registers on its engine client never fires, and the engine
  stays up.
* **The parent's allocator is the wrong allocator.** The other half of the old teardown was
  ``torch.cuda.empty_cache()``. Under vLLM V1 the weights and the KV cache live in a separate
  ``EngineCore`` subprocess -- the failing log lines are prefixed ``(EngineCore pid=11250)`` -- whose
  memory the parent's caching allocator cannot account for, let alone hand back. No amount of
  parent-side cache-emptying frees a child process's VRAM.

So the release here does not rely on Python object lifetime at all. It calls vLLM's own
``EngineCoreClient.shutdown``, which SIGTERMs that subprocess and joins it, and then it POLLS the card
until the memory is actually back, raising :class:`EngineDrainError` when it never is. The poll is
what makes this a check rather than a reassuring message: a vLLM upgrade that moves the shutdown
entry point, or a child that will not die, now stops the run with a message naming the residue,
instead of handing the next engine a full card and letting it fail in vLLM's words.
"""

from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

ENGINE_SHUTDOWN_PATH: tuple[str, ...] = ("_llm", "llm_engine", "engine_core", "shutdown")
"""Where vLLM 0.27.1 keeps the teardown that actually stops the EngineCore subprocess.

``VLLMBackend._llm`` is the ``vllm.LLM``; ``LLM.llm_engine`` is the V1 ``LLMEngine``;
``LLMEngine.engine_core`` is an ``EngineCoreClient``, whose ``shutdown`` is declared abstract on the
base class and implemented by both the in-process and the multiprocess client. Walked as data and
REQUIRED rather than probed, so a rename in a vLLM upgrade raises here instead of quietly becoming a
teardown that does nothing.
"""

ENGINE_HANDLE_ATTRIBUTE = ENGINE_SHUTDOWN_PATH[0]
"""The backend attribute holding the engine, cleared after shutdown.

Only load-bearing on the in-process client, where the model lives in THIS process and the executor
owning its tensors is reachable from the backend. On the multiprocess client -- the default, and what
the failing run used -- the subprocess exit is what frees the card and this is hygiene.
"""

DEFAULT_SHUTDOWN_TIMEOUT_S = 30.0
"""Seconds vLLM is given to stop the EngineCore gracefully before it force-kills it.

Above vLLM's own 5-second default: a graceful exit lets the child release the card and say that it
did, and force-killing a process mid-teardown is how VRAM gets orphaned in the first place.
"""

DEFAULT_DRAIN_TIMEOUT_S = 180.0
"""How long the card is given to come back.

Generous on purpose: waiting costs seconds of a rented box, while going red early would abort a run
over driver accounting lag.
"""

DEFAULT_DRAIN_POLL_S = 2.0

DEFAULT_DRAIN_TOLERANCE_MIB = 1024
"""Residue that still counts as drained.

An idle CUDA context is a few hundred MiB (``scripts/gpu_preflight.py`` treats 512 MiB as its busy
floor), and the driver does not always return a dead process's last pages instantly. A gigabyte sits
comfortably below anything an engine claims -- the failing run's engine held 40.8 GiB -- so this
tolerates slop without tolerating a live engine.
"""

NVIDIA_SMI_TIMEOUT_S = 30.0
"""How long one card reading is given before it counts as unreadable.

``nvidia-smi`` normally answers in well under a second, but it blocks indefinitely on a wedged driver
and during another process's teardown -- which is precisely when this module runs. An unbounded query
would then hang inside the drain poll forever, so :data:`DEFAULT_DRAIN_TIMEOUT_S` would never be
reached and the deadline that makes the drain a check rather than a wait would not exist. Generous
against a healthy card, an order of magnitude below the drain deadline against a sick one.
"""

VLLM_BACKEND_KIND = "vllm"
"""The one backend kind whose teardown goes through this module.

Lives here beside :func:`resolve_engine_shutdown`, which deliberately REFUSES any backend without
vLLM's engine-core chain: the ``hf`` and ``mock`` kinds hold no ``EngineCore`` subprocess, so putting
one through this release would raise rather than free anything.
"""


class EngineDrainError(RuntimeError):
    """An engine's VRAM did not come back, so the next engine cannot be loaded onto this card."""


def _nvidia_smi_gpu_rows(fields: str) -> list[str]:
    """Query one comma-separated field list per GPU, unit-less, one row per device.

    Duplicated from ``scripts/gpu_preflight.py`` rather than shared, for a structural reason:
    ``scripts/`` is deliberately not an importable package (its ruff ``INP001`` exemptions say so),
    and its files are pinned to the 3.9 dialect because the jail and the canary target invoke them
    with the host interpreter. Importing across that boundary would drag one of those constraints
    into ``games/``.
    """
    smi = shutil.which("nvidia-smi")
    if smi is None:
        raise EngineDrainError(
            "nvidia-smi is not on PATH, so the card's memory cannot be read and a released engine "
            "cannot be told from one still holding the whole card. Refusing rather than assuming the "
            "release worked: assuming it is exactly what let a run reach vLLM's own 'Engine core "
            "initialization failed'."
        )
    try:
        completed = subprocess.run(  # noqa: S603 - trusted executable and literal arguments
            [smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as expired:
        raise EngineDrainError(
            f"nvidia-smi did not answer within {NVIDIA_SMI_TIMEOUT_S}s, so the card's memory cannot "
            f"be read and a released engine cannot be told from one still holding the whole card. A "
            f"wedged driver, or another process's teardown, is exactly when this module runs -- an "
            f"unreadable card counts as occupied, the same as the '[N/A]' reading. Check with: "
            f"nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv"
        ) from expired
    return [line for line in (raw.strip() for raw in completed.stdout.splitlines()) if line]


def vram_used_mib() -> list[int]:
    """Per-device VRAM in use on this host, in MiB, in device order.

    Read from ``nvidia-smi`` rather than ``torch.cuda.mem_get_info``, for two reasons that both bear
    on the bug this module exists for. The memory that has to drain belongs to vLLM's ``EngineCore``
    SUBPROCESS, which the parent's torch allocator cannot account for at all. And ``mem_get_info``
    would initialise a CUDA context in the parent, permanently costing a few hundred MiB that then
    counts against every engine loaded afterwards -- the measurement would change what it measures.

    A non-numeric reading raises. ``nvidia-smi`` prints ``[N/A]`` for a device it can see but not
    account for, and reading that as zero is how a full card passes a check: the same trap
    ``scripts/gpu_preflight.py`` documents and its tests pin.
    """
    used: list[int] = []
    for row in _nvidia_smi_gpu_rows("memory.used"):
        if not row.isdigit():
            raise EngineDrainError(
                f"nvidia-smi reported device {len(used)}'s used memory as {row!r} rather than a "
                f"number, so this card's occupancy is unknown. Unknown is not zero -- treat it as "
                f"occupied (check with: nvidia-smi)."
            )
        used.append(int(row))
    if not used:
        raise EngineDrainError(
            "nvidia-smi listed no GPU at all, so there is no card whose drain could be verified."
        )
    return used


def baseline_before_engine(backend_kind: str) -> list[int] | None:
    """Read the card BEFORE an engine is constructed, or ``None`` for a kind that loads no engine.

    Two decisions that live here rather than being re-made at every call site:

    * **Before, not after.** :func:`await_vram_drain` asks "did the card come back to where it was",
      so the baseline has to predate the engine. One taken afterwards already includes the engine's
      own claim, and the residue then reads as zero however much of the card is still held -- a
      release that cannot fail, which is the shape of the bug this module exists for.
    * **Gated on the DECLARED kind**, never on probing the backend for an engine handle. A driver that
      asked the object whether it looked like vLLM would silently do nothing the day the attribute
      moved, which is exactly the no-op :func:`resolve_engine_shutdown` refuses to become.

    The ``None`` is what a caller carries to its ``finally``: a release with no baseline is not
    expressible, so the gate is decided once, here, rather than twice per call site.
    """
    if backend_kind != VLLM_BACKEND_KIND:
        return None
    return vram_used_mib()


@dataclass(frozen=True)
class DrainPolicy:
    """How long a release waits for the card, and how much residue still counts as drained."""

    shutdown_timeout_s: float = DEFAULT_SHUTDOWN_TIMEOUT_S
    drain_timeout_s: float = DEFAULT_DRAIN_TIMEOUT_S
    poll_interval_s: float = DEFAULT_DRAIN_POLL_S
    tolerance_mib: int = DEFAULT_DRAIN_TOLERANCE_MIB


DEFAULT_DRAIN_POLICY = DrainPolicy()
"""The policy every real run uses.

A module constant rather than a default argument, because ruff's ``B008`` correctly refuses a call in
a signature and threading ``None`` through would buy a branch for nothing.
"""


def resolve_engine_shutdown(backend: object) -> Callable[..., object]:
    """Return vLLM's own engine-core shutdown for one backend, refusing a backend without it.

    Required rather than probed. The failure this module exists for was a teardown that reported
    success while doing nothing, so "the attribute was missing, carry on" is not an available
    behaviour here: a vLLM upgrade that moves ``EngineCoreClient.shutdown`` has to stop the run and
    be followed in :data:`ENGINE_SHUTDOWN_PATH`.
    """
    node: object = backend
    for depth, name in enumerate(ENGINE_SHUTDOWN_PATH):
        node = getattr(node, name, None)
        if node is None:
            reached = ".".join(ENGINE_SHUTDOWN_PATH[: depth + 1])
            raise EngineDrainError(
                f"{type(backend).__name__} has no {reached}, so its engine core cannot be shut down "
                f"and the card cannot be freed for the next engine. vLLM 0.27.1 declares "
                f"EngineCoreClient.shutdown abstract on the client base and implements it for both "
                f"the in-process and the multiprocess client; an upgrade that moved it must be "
                f"followed in ENGINE_SHUTDOWN_PATH rather than left to a silent no-op."
            )
    if not callable(node):
        raise EngineDrainError(
            f"{'.'.join(ENGINE_SHUTDOWN_PATH)} on {type(backend).__name__} is "
            f"{type(node).__name__}, not callable, so there is nothing to shut the engine down with."
        )
    return node


def await_vram_drain(
    baseline_mib: Sequence[int],
    *,
    read_used_mib: Callable[[], list[int]] = vram_used_mib,
    policy: DrainPolicy = DEFAULT_DRAIN_POLICY,
) -> list[int]:
    """Poll until every device is back within tolerance of ``baseline_mib``, or raise.

    ``baseline_mib`` is what the card held immediately BEFORE the engine was constructed, so the
    comparison is "the card came back to where it was" rather than "the card looks roughly empty" --
    the only version of the question that works on a shared box where a neighbour may already hold
    memory.

    Raising rather than warning and proceeding: the next thing to happen otherwise is a second engine
    claiming a fraction of a card that is already full, which fails inside vLLM with a message about
    GPU memory utilization and no mention of the engine that never let go.
    """
    deadline = time.monotonic() + policy.drain_timeout_s
    while True:
        used = read_used_mib()
        if len(used) != len(baseline_mib):
            raise EngineDrainError(
                f"the host reported {len(used)} GPUs against {len(baseline_mib)} at the baseline, so "
                f"the two readings are not of the same card and no residue can be computed: "
                f"{list(baseline_mib)} then {used}."
            )
        residue = [after - before for after, before in zip(used, baseline_mib, strict=True)]
        if max(residue) <= policy.tolerance_mib:
            logger.info(
                f"card drained, baseline_mib={list(baseline_mib)} used_mib={used} {residue=}"
            )
            return used
        if time.monotonic() >= deadline:
            raise EngineDrainError(
                f"the card did not give its memory back within {policy.drain_timeout_s}s of the "
                f"engine being shut down: per-device MiB still held above the pre-load baseline is "
                f"{residue} (baseline {list(baseline_mib)}, now {used}, tolerance "
                f"{policy.tolerance_mib} MiB). Loading the next engine onto this card would fail "
                f"inside vLLM with a message about GPU memory utilization and no mention of this. "
                f"Either the EngineCore subprocess is still alive or another process took the card "
                f"meanwhile -- `nvidia-smi --query-compute-apps=pid,used_memory,process_name "
                f"--format=csv` names the holder."
            )
        time.sleep(policy.poll_interval_s)


def release_engine(
    backend: object,
    *,
    baseline_mib: Sequence[int],
    read_used_mib: Callable[[], list[int]] = vram_used_mib,
    policy: DrainPolicy = DEFAULT_DRAIN_POLICY,
) -> dict[str, object]:
    """Shut one vLLM engine down and refuse to continue until the card has given the memory back.

    Three steps, and only the third is a check:

    1. ``EngineCoreClient.shutdown``, vLLM's own teardown. This is the step that frees a multiprocess
       engine: it SIGTERMs the ``EngineCore`` child, which drops out of its busy loop and shuts its
       model executor down, and joins it. Nothing about Python reference counting is relied on, which
       is the point -- the previous teardown relied on it and the caller's own ``with`` target
       silently defeated it.
    2. Clear the backend's engine handle and collect. Only the in-process client keeps the model in
       this process, and this is what lets its tensors go; on the multiprocess client the child's exit
       has already done the work and this is hygiene.
    3. Poll the card until the memory is back, per :func:`await_vram_drain`.

    Returns the reading, so a run's own log carries what the card looked like before and after each
    engine rather than only the claim that a release was attempted.
    """
    shutdown = resolve_engine_shutdown(backend)
    shutdown(timeout=policy.shutdown_timeout_s)
    setattr(backend, ENGINE_HANDLE_ATTRIBUTE, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    used = await_vram_drain(baseline_mib, read_used_mib=read_used_mib, policy=policy)
    report: dict[str, object] = {
        "baseline_used_mib": list(baseline_mib),
        "released_used_mib": used,
        "residue_mib": [after - before for after, before in zip(used, baseline_mib, strict=True)],
        "tolerance_mib": policy.tolerance_mib,
    }
    logger.info(f"engine released, {report}")
    return report
