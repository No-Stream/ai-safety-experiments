"""Run a sequence of GPU stages and refuse to call any of them successful without checking.

Three orchestration bugs in one night, all the same shape — shell asserting a success it never
verified:

1.  `if ! "$@"; then local status=$?` reported **exit 0 for a failing stage**, because `!` inverts
    the status before `$?` is read.
2.  A stage that failed for an upstream reason was reported as `STAGE FAILED ... (exit 0)`, so the
    log contradicted itself and pointed at the wrong culprit.
3.  A sequence launched two GPU stages while the previous process still held VRAM, watched both die
    in the same second when `gpu_preflight` refused, and then wrote a `screens-done` flag and logged
    `SEQUENCE COMPLETE: screens done` anyway. **A false green that idled the card for half
    an hour.**

Bash makes all three easy and none of them testable, which is the actual problem. So the logic lives
here instead, where it is exercised offline:

*   Success requires BOTH a zero exit code AND every artifact the stage promised, non-empty. A
    command can exit 0 having written nothing — that is bug 3, and it is the default assumption here
    that a stage is a liar until its output exists.
*   A GPU stage waits for the card to be released first, because the failure that started all of
    this was two stages racing one L4. The wait has a timeout and the timeout is a failure, not a
    shrug.
*   The sequence stops at the first stage that is not ok, since every later stage consumes an
    earlier one's output.
*   A stage's log is written into the run directory as well as the shared log directory, because the
    run directory is the only thing any sync here ships. A log that exists only on the instance is
    already how one run's rollout trace was lost.

    uv run python -m games.stage_runner --plan <module-with-a-stages()-function>
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import subprocess
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from io import BufferedWriter
    from pathlib import Path

logger = logging.getLogger(__name__)

# What counts as "the card is free". Not zero: a driver or another process can hold a few MiB
# without preventing a run, and gpu_preflight's own threshold is what really decides.
DEFAULT_MAX_USED_MIB = 512
DEFAULT_GPU_WAIT_SECONDS = 900
DEFAULT_POLL_SECONDS = 30

LOG_READ_BYTES = 64 * 1024
# One level down inside a run directory, so a run listing still reads as checkpoints plus TRL's own
# output rather than as a pile with logs mixed in.
RUN_LOG_SUBDIR = "logs"
# Opens each attempt in an appended log; see `_attempt_header`.
ATTEMPT_DELIMITER = "==== stage attempt"

# `timeout(1)`'s exit code when a cap fires, which needs the opposite response from a crash.
WALL_CLOCK_CAP_RETURNCODE = 124
# A stage that never ran has no exit code of its own; the booleans say which non-run it was.
NOT_RUN_RETURNCODE = -1

# The plans `--help` invites an operator to run; see `plan_help`.
EXAMPLE_PLANS: tuple[str, ...] = (
    "games.arm_sequence",
    "games.contrast_pair_sequence",
    "games.nine_b_sequence",
)


@dataclass(frozen=True)
class Stage:
    """One command, plus the artifacts that prove it did what it claimed."""

    name: str
    argv: tuple[str, ...]
    # Every path must exist and be non-empty afterwards, or the stage failed however it exited.
    artifacts: tuple[Path, ...] = ()
    needs_gpu: bool = False
    log_path: Path | None = None
    # Run directories that must also carry this stage's log; see `log_destinations`.
    log_run_dirs: tuple[Path, ...] = ()
    env: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Reject a stage that could not be checked."""
        if not self.name:
            raise ValueError("a stage needs a name; it is what the log and the report refer to")
        if not self.argv:
            raise ValueError(f"stage {self.name!r} has no command to run")
        if self.log_run_dirs and self.log_path is None:
            raise ValueError(
                f"stage {self.name!r} names run directories to keep its log but sets no log_path, "
                f"so there is no log to copy into them"
            )

    def log_destinations(self) -> tuple[Path, ...]:
        """Every path this stage's merged output is written to, the shared log first.

        Both copies hold the same bytes. The shared one is what an operator tails and what
        `games.arm_sequence.resolve_single_corpus` points at when a sweep produced no corpus; the
        run-directory ones are what reach S3, because `games/s3_sync.py` and `cloud/entrypoint.sh`
        sync run directories and nothing syncs `artifacts/games/logs/`. Operators were closing that
        gap by hand, and on 2026-08-19 a hand-rolled log-only sync was a run's entire surviving
        record while its rollout parquet stayed on the box.
        """
        if self.log_path is None:
            return ()
        return (
            self.log_path,
            *(directory / RUN_LOG_SUBDIR / self.log_path.name for directory in self.log_run_dirs),
        )


@dataclass(frozen=True)
class StageResult:
    """What a stage actually did, as opposed to what it said."""

    name: str
    returncode: int
    seconds: float
    missing_artifacts: tuple[str, ...] = ()
    gpu_wait_timed_out: bool = False
    skipped: bool = False

    @property
    def ok(self) -> bool:
        """True only when the command succeeded AND its promised artifacts are on disk."""
        return (
            not self.skipped
            and not self.gpu_wait_timed_out
            and self.returncode == 0
            and not self.missing_artifacts
        )

    def describe(self) -> str:
        """One line for the log, naming the reason rather than only the verdict."""
        if self.skipped:
            return f"SKIPPED {self.name}: an earlier stage failed"
        if self.gpu_wait_timed_out:
            return f"FAILED {self.name}: the GPU never came free"
        if self.returncode == WALL_CLOCK_CAP_RETURNCODE:
            return (
                f"FAILED {self.name}: the wall-clock cap fired after {self.seconds:.0f}s "
                f"(exit {WALL_CLOCK_CAP_RETURNCODE} is timeout(1) killing the command, not a "
                f"crash) -- raise the cap and re-run the identical command to resume"
            )
        if self.returncode != 0:
            return f"FAILED {self.name}: exit {self.returncode} after {self.seconds:.0f}s"
        if self.missing_artifacts:
            return (
                f"FAILED {self.name}: exited 0 but produced nothing at "
                f"{list(self.missing_artifacts)} after {self.seconds:.0f}s"
            )
        return f"OK {self.name}: {self.seconds:.0f}s"


def used_vram_mib() -> int:
    """Read VRAM in use from nvidia-smi, summed over devices."""
    finished = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return sum(int(line.strip()) for line in finished.stdout.splitlines() if line.strip())


def wait_for_free_gpu(
    *,
    max_used_mib: int = DEFAULT_MAX_USED_MIB,
    timeout_seconds: float = DEFAULT_GPU_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    probe: Callable[[], int] = used_vram_mib,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Block until the card is released, returning False if it never is.

    `probe` and `sleep` are injected so the waiting logic is testable without a GPU or a real
    delay. The timeout returning False rather than proceeding is the point: the bug this exists to
    prevent was starting a GPU stage anyway and watching its preflight refuse in two seconds.
    """
    deadline = time.monotonic() + timeout_seconds
    while True:
        used = probe()
        if used <= max_used_mib:
            return True
        if time.monotonic() >= deadline:
            logger.error(
                "gave up waiting for the GPU, %s",
                f"{used=} MiB still held, {max_used_mib=} {timeout_seconds=}",
            )
            return False
        logger.info(
            "waiting for the GPU to come free, %s", f"{used=} MiB held, polling in {poll_seconds}s"
        )
        sleep(poll_seconds)


def missing_artifacts(artifacts: Sequence[Path]) -> tuple[str, ...]:
    """Name the promised artifacts that are absent or empty."""
    return tuple(str(path) for path in artifacts if not (path.exists() and path.stat().st_size > 0))


def run_stage(  # noqa: PLR0913
    stage: Stage,
    *,
    gpu_wait_seconds: float = DEFAULT_GPU_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    probe: Callable[[], int] = used_vram_mib,
    sleep: Callable[[float], None] = time.sleep,
    runner: Callable[[Stage], int] | None = None,
) -> StageResult:
    """Run one stage and report what it actually achieved."""
    if stage.needs_gpu and not wait_for_free_gpu(
        timeout_seconds=gpu_wait_seconds, poll_seconds=poll_seconds, probe=probe, sleep=sleep
    ):
        return StageResult(
            name=stage.name,
            returncode=NOT_RUN_RETURNCODE,
            seconds=0.0,
            gpu_wait_timed_out=True,
        )

    logger.info("STAGE BEGIN %s: %s", stage.name, " ".join(stage.argv))
    started = time.monotonic()
    returncode = (runner or _execute)(stage)
    elapsed = time.monotonic() - started
    result = StageResult(
        name=stage.name,
        returncode=returncode,
        seconds=elapsed,
        missing_artifacts=missing_artifacts(stage.artifacts),
    )
    logger.info("STAGE END %s", result.describe())
    return result


def _execute(stage: Stage) -> int:
    """Run the command, writing its merged output to every log destination the stage names."""
    destinations = stage.log_destinations()
    if not destinations:
        return subprocess.run(  # noqa: S603
            stage.argv, check=False, env=_environment(stage)
        ).returncode
    with ExitStack() as stack:
        handles = [stack.enter_context(_opened_log(path)) for path in destinations]
        return _run_logged(stage, handles)


def _opened_log(path: Path) -> BufferedWriter:
    """Open one log destination for appending, creating the directory it lives in.

    Appending rather than truncating, because every log path in every plan here is deterministic
    per arm and model while the documented recovery procedure is to re-run the identical command --
    so a truncating open destroys the record of why the previous attempt died, at the one moment
    someone needs to read it. The next checkpoint sync would replace the S3 copy with the truncated
    one too, leaving an overnight crash on a rented box with no surviving explanation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("ab")


def _attempt_header(stage: Stage) -> bytes:
    """Return the line that separates one attempt's output from the previous attempt's."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    return f"{ATTEMPT_DELIMITER} {started} {stage.name}: {' '.join(stage.argv)} ====\n".encode()


def _run_logged(stage: Stage, handles: Sequence[BufferedWriter]) -> int:
    """Run the command, copying its output to every handle as the output arrives.

    Read through the parent rather than handing the child one file descriptor, because a stage's log
    has to reach more than one place. `os.read` returns whatever the pipe has rather than waiting
    for a full buffer, and every chunk is flushed, so each copy stays current while the stage runs:
    one for an operator's `tail -f`, the other for the S3 sync that fires on every checkpoint save.
    A log that only materialised at exit would be missing from exactly the runs that lose it, the
    ones a spot reclamation or a wall-clock cap killed partway.

    Each attempt opens with `_attempt_header`, since the destinations are appended to: without it a
    resumed stage's output would read as one run whose own log contradicts itself.
    """
    header = _attempt_header(stage)
    for handle in handles:
        handle.write(header)
        handle.flush()
    with subprocess.Popen(  # noqa: S603
        stage.argv,
        env=_environment(stage),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as child:
        if child.stdout is None:
            raise RuntimeError(f"stage {stage.name!r} was given a pipe but exposes no stdout")
        descriptor = child.stdout.fileno()
        for chunk in iter(lambda: os.read(descriptor, LOG_READ_BYTES), b""):
            for handle in handles:
                handle.write(chunk)
                handle.flush()
    return child.returncode


def _environment(stage: Stage) -> dict[str, str] | None:
    """Merge the stage's overrides onto the inherited environment."""
    if stage.env is None:
        return None
    return {**os.environ, **stage.env}


@dataclass
class SequenceOutcome:
    """Every stage's result, and whether the whole sequence earned a success claim."""

    results: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only when every stage ran and every stage was ok."""
        return bool(self.results) and all(result.ok for result in self.results)

    @property
    def first_failure(self) -> StageResult | None:
        """The stage that stopped the sequence, if one did."""
        return next((result for result in self.results if not result.ok), None)

    def _failure_name(self) -> str:
        """Name the stage that stopped the sequence, for the verdict line."""
        failure = self.first_failure
        return failure.name if failure else "nothing"

    def describe(self) -> str:
        """Render a verdict that cannot say "complete" unless it was."""
        lines = [result.describe() for result in self.results]
        lines.append(
            "SEQUENCE OK: every stage verified"
            if self.ok
            else f"SEQUENCE INCOMPLETE: stopped at {self._failure_name()}"
        )
        return "\n".join(lines)


def run_sequence(  # noqa: PLR0913
    stages: Sequence[Stage],
    *,
    gpu_wait_seconds: float = DEFAULT_GPU_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    probe: Callable[[], int] = used_vram_mib,
    sleep: Callable[[float], None] = time.sleep,
    runner: Callable[[Stage], int] | None = None,
) -> SequenceOutcome:
    """Run stages in order, stopping at the first one that cannot be verified.

    Stopping rather than continuing is deliberate: every stage here consumes an earlier stage's
    output, so carrying on past a failure trains an arm on a corpus that was never written.
    """
    outcome = SequenceOutcome()
    for index, stage in enumerate(stages):
        result = run_stage(
            stage,
            gpu_wait_seconds=gpu_wait_seconds,
            poll_seconds=poll_seconds,
            probe=probe,
            sleep=sleep,
            runner=runner,
        )
        outcome.results.append(result)
        if not result.ok:
            for later in stages[index + 1 :]:
                outcome.results.append(
                    StageResult(
                        name=later.name,
                        returncode=NOT_RUN_RETURNCODE,
                        seconds=0.0,
                        skipped=True,
                    )
                )
            break
    logger.info("\n%s", outcome.describe())
    return outcome


def load_plan(module_path: str) -> list[Stage]:
    """Import a plan module and ask it for its stages.

    A plan is any module exposing `stages() -> list[Stage]`. Keeping the entry point here rather
    than giving every plan its own `main()` means one place decides what "the sequence succeeded"
    means -- which is the whole point of this module, and was the bug when each script decided for
    itself.
    """
    module = importlib.import_module(module_path)
    factory = getattr(module, "stages", None)
    if factory is None:
        raise ValueError(
            f"{module_path} is not a plan: a plan module must expose stages() -> list[Stage]"
        )
    stages = factory()
    if not stages:
        raise ValueError(f"{module_path}.stages() returned nothing, so there is nothing to run")
    return list(stages)


def plan_help() -> str:
    """Describe `--plan`, naming the plans that exist.

    Built from `EXAMPLE_PLANS` rather than written out, because the previous text advertised a spent
    one-night chain whose two corpora nothing regenerates: an operator following `--help` would have
    re-run two finished arms on the GPU. A test asserts every name here still imports as a plan.
    """
    return f"Module exposing stages(), e.g. {', '.join(EXAMPLE_PLANS)}."


def main(argv: Sequence[str] | None = None) -> int:
    """Run a plan module's stages, returning non-zero unless every stage verified.

    The exit code is the contract: a caller (a shell, a tmux session, an operator on a rented
    instance at 2am) must be able to tell success from failure without reading the log.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help=plan_help())
    parser.add_argument(
        "--gpu-wait-seconds",
        type=float,
        default=DEFAULT_GPU_WAIT_SECONDS,
        help="How long a GPU stage waits for the card before failing.",
    )
    args = parser.parse_args(argv)
    outcome = run_sequence(load_plan(args.plan), gpu_wait_seconds=args.gpu_wait_seconds)
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
