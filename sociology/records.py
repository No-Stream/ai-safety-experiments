"""The record layer every sociology run shares: reply I/O, the resume loop, the summary marker.

Extracted from :mod:`sociology.runner` when a second pass (the decoupled-ladder counterpart study)
needed the same four behaviours and would otherwise have grown its own near-copies. Each of them is
a property this repository has already paid for once:

- **Duplicate keys refuse.** :func:`load_replies` raises rather than last-winning, because a
  duplicated key means two calls were filed under one identity and a rate computed over them has a
  denominator nobody can state.
- **Every writer is guarded.** :func:`append_replies` and :func:`write_summary` call
  ``refuse_tracked_trace_path`` themselves rather than trusting callers, which is the shape of the
  one leak of this class the repo has actually had: one writer, one guard, both callers covered.
- **Resume is by content-derived key, never by execution order.** :func:`run_live_calls` skips keys
  already on disk and counts them separately from what ran, so a resumed run's summary stays honest
  about what it actually paid for. A study that stamps digests onto its records passes
  ``resume_identity`` as well, and then a key is resumed only if the record on disk answers the same
  prompt under the same stimulus -- otherwise one file silently holds two experiments.
- **The summary is written LAST**, as the completion marker: a replies file without one is a run
  that died mid-flight, and trace presence is not completion.

The loop is parameterised on a call protocol (:class:`LiveCall`) and a record builder rather than on
one study's ``PlannedCall``, so a study contributes its own row schema and inherits the resume
behaviour instead of reimplementing it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from games.provenance import git_provenance
from reward_hacking.model_backend import (
    BedrockBackend,
    BedrockSamplingConfig,
    stream_detailed_in_chunks,
)
from reward_hacking.trace import refuse_tracked_trace_path

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from reward_hacking.model_backend import BedrockCompletion, DetailedBackend

logger = logging.getLogger(__name__)

DEFAULT_REPLY_CARRIES = "verbatim model prompts and replies"
"""What a reply file publishes if committed, for the tracked-path refusal's message."""


@runtime_checkable
class LiveCall(Protocol):
    """The four fields the live resume loop reads off a planned call, whatever else it carries.

    A protocol rather than a base class so each study keeps its own frozen dataclass with its own
    label set; the loop needs the identity, the text to send, and the two knobs that decide which
    backend sends it.
    """

    @property
    def key(self) -> str:
        """The call's content-derived identity, which is also its resume key."""
        ...

    @property
    def prompt(self) -> str:
        """The rendered prompt text to send."""
        ...

    @property
    def model_id(self) -> str:
        """The model that answers this call."""
        ...

    @property
    def reasoning_effort(self) -> str | None:
        """The effort the call is sampled at, or None for the provider default."""
        ...


if TYPE_CHECKING:
    # (model_id, reasoning_effort, concurrency) -> backend; the live runner's one testable seam.
    LiveBackendFactory = Callable[[str, str | None, int], DetailedBackend]


def record_prompt_digest(prompt: str) -> str:
    """Digest one rendered prompt the way every reply record stores it.

    One function rather than the same two lines at each call site, because the resume check compares a
    stored digest against a freshly computed one: two spellings of "the prompt digest" that drifted
    would make every resumed key look changed, or worse, none of them.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def refuse_changed_resume[CallT: LiveCall](
    existing: Mapping[str, Mapping[str, Any]],
    calls: Sequence[CallT],
    *,
    path: Path,
    resume_identity: Callable[[CallT], Mapping[str, Any]],
) -> None:
    """Refuse to resume a key whose record on disk does not describe THIS call.

    Resume is what makes a killed pass cheap, and it is also the quietest way to end up with one
    reply file holding two experiments: an edited stimulus or a changed renderer produces new prompts
    under the OLD keys, and a resumed run then skips them and reports itself finished. Comparing the
    stored digests against this invocation's is the only place that shows up, because nothing about
    the file's shape changes. The refusal names the key and both sides of every field that moved.
    """
    for call in calls:
        row = existing.get(call.key)
        if row is None:
            continue
        drifted = [
            f"{name}={row.get(name)!r} on disk but {value!r} for this call"
            for name, value in resume_identity(call).items()
            if row.get(name) != value
        ]
        if drifted:
            raise ValueError(
                f"refusing to resume {call.key} from {path}: {'; '.join(drifted)}. A resumed key must "
                f"answer the same prompt under the same stimulus; keeping this record would file two "
                f"experiments in one file with every count still adding up. Point --run-dir at a "
                f"fresh directory, or move the old file aside."
            )


def load_replies(path: Path) -> dict[str, dict[str, Any]]:
    """Load reply records keyed by record key; duplicates refuse rather than silently last-win."""
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["key"])
            if key in rows:
                raise ValueError(f"duplicate reply key {key} in {path}")
            rows[key] = row
    return rows


_APPEND_LOCK = threading.Lock()
"""Serialises appends across legs running in parallel; a JSONL line torn between two writers is a
duplicate-key-or-parse error on the next load, and a lock is cheaper than either."""


def append_replies(
    path: Path, records: Sequence[Mapping[str, Any]], *, carries: str = DEFAULT_REPLY_CARRIES
) -> None:
    """Append reply records to an incremental file, refusing a git-tracked destination."""
    refuse_tracked_trace_path(path, carries=carries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND_LOCK, path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def write_summary(run_dir: Path, label: str, payload: Mapping[str, Any]) -> Path:
    """Write one invocation's summary last, as its completion marker."""
    path = run_dir / f"summary-{label}.json"
    refuse_tracked_trace_path(path, carries="run accounting")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {**payload, "finished_at": datetime.now(UTC).isoformat(), **git_provenance()}
    path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    logger.info("summary written to %s", path)
    return path


def make_live_backend_factory(max_tokens: int) -> LiveBackendFactory:
    """Build the production live-backend factory for a study, at that study's reply cap.

    The one place a live sociology backend is constructed, and deliberately narrow: no temperature,
    no top_p, no top_k anywhere in either study, so the sampler carries the cap and the effort and
    nothing else. Three roster models refuse those knobs outright, and a run that asked for one and
    silently got none would have measured something other than what it recorded.
    """

    def factory(model_id: str, effort: str | None, concurrency: int) -> DetailedBackend:
        return BedrockBackend(
            model_id,
            concurrency=concurrency,
            sampling=BedrockSamplingConfig(max_tokens=max_tokens, reasoning_effort=effort),
        )

    return factory


def run_live_calls[CallT: LiveCall](  # noqa: PLR0913 - trailing keyword-only knobs, each a seam
    calls: Sequence[CallT],
    out_path: Path,
    *,
    concurrency: int,
    chunk_size: int,
    make_record: Callable[[CallT, BedrockCompletion], dict[str, Any]],
    backend_factory: LiveBackendFactory,
    carries: str = DEFAULT_REPLY_CARRIES,
    resume_identity: Callable[[CallT], Mapping[str, Any]] | None = None,
    parallel_legs: bool = False,
    persist_out_of_order: bool = False,
) -> dict[str, int]:
    """Run planned live calls with resume-by-key, appending per chunk; return the counts.

    Records already on disk are skipped and counted as resumed, separately from anything else, so
    the summary stays honest about what actually ran. Incomplete records (deadline, failed call,
    cap) are flagged and KEPT -- they are data with a reason attached, never re-run silently.
    ``backend_factory`` is the transport seam the offline tests script; ``make_record`` is the
    study's own row schema, which this loop never inspects beyond the ``incomplete`` flag it counts.

    ``resume_identity`` names the fields a resumed record must still agree with this call about --
    typically the stimulus and prompt digests. Passing it turns resume from "this key has a row" into
    "this key has a row for THIS prompt", which is the difference between a continuation and two
    experiments in one file (see :func:`refuse_changed_resume`).

    Within a leg the backend's queue is kept full across chunk boundaries (:func:`_run_leg`), so a
    chunk's slowest call no longer idles the other workers. ``parallel_legs`` runs the (model,
    effort) legs concurrently as well -- they draw on independent quotas, so the wall clock becomes
    the longest leg rather than the sum, which operators were already getting by hand in tmux. Off
    by default because it changes the interleaving of legs on disk (never the records: every row is
    keyed and :func:`load_replies` reads by key) and because a leg that raises then no longer stops
    the legs beside it from finishing their paid work. Every leg's finished chunks are on disk before
    the first error propagates.

    ``persist_out_of_order`` relaxes the within-leg order promise the same way, deliberately and per
    run: a chunk is appended the moment its own last call lands rather than waiting for every chunk
    before it, so what a wedged head call can lose to a SIGKILL or an OOM kill is the partial chunks
    in flight rather than everything behind the head
    (:func:`~reward_hacking.model_backend.stream_detailed_in_chunks` has the arithmetic). The file
    then holds a leg's chunks in completion order; every row is still keyed and read back by key, so
    nothing that reads by identity changes, and the default stays request order because that is the
    order the byte-identity tests are written against.
    """
    existing = load_replies(out_path)
    if resume_identity is not None:
        refuse_changed_resume(existing, calls, path=out_path, resume_identity=resume_identity)
    pending = [call for call in calls if call.key not in existing]
    counts = {
        "planned": len(calls),
        "resumed": len(calls) - len(pending),
        "ran": 0,
        "incomplete": 0,
    }
    by_leg: dict[tuple[str, str | None], list[CallT]] = {}
    for call in pending:
        by_leg.setdefault((call.model_id, call.reasoning_effort), []).append(call)

    def run_leg(model_id: str, effort: str | None, leg_calls: list[CallT]) -> tuple[int, int]:
        return _run_leg(
            leg_calls,
            out_path,
            backend=backend_factory(model_id, effort, concurrency),
            chunk_size=chunk_size,
            make_record=make_record,
            carries=carries,
            leg=f"{model_id} effort={effort}",
            persist_out_of_order=persist_out_of_order,
        )

    if parallel_legs and len(by_leg) > 1:
        with ThreadPoolExecutor(max_workers=len(by_leg)) as pool:
            futures = [
                pool.submit(run_leg, model_id, effort, leg_calls)
                for (model_id, effort), leg_calls in by_leg.items()
            ]
            wait(futures)
        outcomes = [future.exception() or future.result() for future in futures]
    else:
        outcomes = [
            run_leg(model_id, effort, leg_calls) for (model_id, effort), leg_calls in by_leg.items()
        ]
    errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    for outcome in outcomes:
        if not isinstance(outcome, BaseException):
            ran, incomplete = outcome
            counts["ran"] += ran
            counts["incomplete"] += incomplete
    if errors:
        # Every leg has finished or failed, and every finished chunk is on disk, before this raise.
        raise errors[0]
    return counts


def _run_leg[CallT: LiveCall](  # noqa: PLR0913 - one keyword per seam of the leg
    leg_calls: Sequence[CallT],
    out_path: Path,
    *,
    backend: DetailedBackend,
    chunk_size: int,
    make_record: Callable[[CallT, BedrockCompletion], dict[str, Any]],
    carries: str,
    leg: str,
    persist_out_of_order: bool = False,
) -> tuple[int, int]:
    """Run one (model, effort) leg with the backend's queue kept full, appending per chunk in order.

    :func:`stream_detailed_in_chunks` keeps ``concurrency`` calls in flight across chunk
    boundaries on a streaming backend and falls back to one ``generate_detailed`` per chunk on any
    other, so a chunk's one slow call never idles the workers behind it while what lands on disk is
    exactly what the per-chunk loop wrote, in the same order. On a raise, the finished part of the
    chunk in flight is appended first -- every record is keyed, so the relaunch's resume fills the
    rest -- and then the error propagates. Under ``persist_out_of_order`` a chunk is appended as soon
    as it is whole, whatever chunks before it still wait (see :func:`run_live_calls`). Returns
    ``(ran, incomplete)`` for the caller's counts.
    """
    chunks = [
        list(leg_calls[start : start + chunk_size])
        for start in range(0, len(leg_calls), chunk_size)
    ]
    ran = incomplete = 0
    released = stream_detailed_in_chunks(
        backend,
        [[call.prompt for call in chunk] for chunk in chunks],
        persist_out_of_order=persist_out_of_order,
    )
    for index, pairs in released:
        chunk = chunks[index]
        records = [make_record(chunk[position], completion) for position, completion in pairs]
        append_replies(out_path, records, carries=carries)
        ran += len(records)
        incomplete += sum(1 for record in records if record.get("incomplete"))
        logger.info(
            "live %s: %d/%d this leg%s (%d incomplete so far)",
            leg,
            ran,
            len(leg_calls),
            "" if len(pairs) == len(chunk) else f", chunk {index + 1} PARTIAL ahead of a raise",
            incomplete,
        )
    return ran, incomplete


def completion_telemetry(completion: BedrockCompletion) -> dict[str, Any]:
    """Flatten the per-call accounting a study's reply record should carry, under shared keys.

    One function rather than five lines in each of the four record builders, so the studies' rows
    spell the split and the telemetry the same way the RecoveryBench and hatch traces do:
    ``input_tokens`` stays the total the model read, the two cache counters ride beside it, and the
    live path's ``elapsed_seconds`` / ``first_event_seconds`` / ``attempts`` are ``None`` where the
    transport cannot measure them. This is what turns a judge pass's list-price cost into a real one
    -- a Luna rubric cached on every call is billed at a tenth of the uncached rate -- and what lets
    a barrier analysis be read off the artifact instead of simulated.
    """
    return {
        "cache_read_input_tokens": completion.usage.cache_read_input_tokens,
        "cache_write_input_tokens": completion.usage.cache_write_input_tokens,
        "elapsed_seconds": completion.elapsed_seconds,
        "first_event_seconds": completion.first_event_seconds,
        "attempts": completion.attempts,
    }
