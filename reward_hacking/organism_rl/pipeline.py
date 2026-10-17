"""Keep one future generation batch queued while the current batch drains."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class StepEpisodePipeline[Job, Result]:
    """Run exact step batches in one worker pool, filling freed slots with the next batch."""

    def __init__(self, sample: Callable[[int, Job], Result], *, max_workers: int) -> None:
        """Bind a sampler and a bounded worker pool for adjacent generation batches."""
        if max_workers < 1:
            raise ValueError("pipeline max_workers must be positive")
        self._sample = sample
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="organism-episode"
        )
        self._lock = threading.Lock()
        self._scheduled = threading.Condition(self._lock)
        self._pending_step: int | None = None
        self._pending_jobs: list[Job] = []
        self._pending_futures: list[Future[Result]] = []

    def __enter__(self) -> StepEpisodePipeline[Job, Result]:
        """Keep the pool open for successive optimizer steps."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Join workers when a pipeline scope ends."""
        self.close()

    def close(self) -> None:
        """Cancel queued jobs and join running episode workers."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    def collect(
        self, step: int, jobs: Sequence[Job], next_jobs: Sequence[Job] | None
    ) -> list[Result]:
        """Return one complete batch; schedule future jobs as current workers finish."""
        current_jobs = list(jobs)
        if not current_jobs:
            raise ValueError("pipeline requires a nonempty generation batch")
        with self._lock:
            if self._pending_step is None:
                current_futures = [
                    self._executor.submit(self._sample, step, job) for job in current_jobs
                ]
            else:
                if self._pending_step != step or self._pending_jobs != current_jobs:
                    raise ValueError("prefetched batch differs from the trainer batch")
                current_futures = self._pending_futures
            future_jobs = list(next_jobs) if next_jobs is not None else []
            future_futures: list[Future[Result]] = []
            self._pending_step = step + 1 if future_jobs else None
            self._pending_jobs = future_jobs
            self._pending_futures = future_futures

        def admit_next(completed: Future[Result]) -> None:
            if completed.cancelled() or completed.exception() is not None:
                return
            with self._lock:
                index = len(future_futures)
                if index >= len(future_jobs):
                    return
                next_future = self._executor.submit(self._sample, step + 1, future_jobs[index])
                future_futures.append(next_future)
                self._scheduled.notify_all()
            next_future.add_done_callback(admit_next)

        for future in current_futures:
            future.add_done_callback(admit_next)
        results = [future.result() for future in current_futures]
        with self._scheduled:
            self._scheduled.wait_for(lambda: len(future_futures) == len(future_jobs))
        return results
