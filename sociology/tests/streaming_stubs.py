"""Scripted backends with scripted latencies, with and without the streaming seam.

The chunk-persisting loops (the two judges here and the shared live loop) take the continuous-queue
path over any backend that offers ``submit_stream`` and the one-call-per-chunk path over any other.
Their byte-identity tests need the same script served both ways, with completions landing OUT of
request order on the streaming side, and with the per-call telemetry identical on both sides so the
rows can be compared field for field.

``latency`` is a scripted number per prompt. It is stamped onto every completion as ``elapsed_seconds``
and ``first_event_seconds`` on BOTH paths, and on the streaming side it decides the order calls land
in (shortest first) -- without sleeping, so the arrival order is deterministic and the test is fast.
``fail_on`` names prompts whose call raises like a request bug; the streaming side yields every other
prompt first and raises last, which is the real backend's drain-then-raise contract and what makes the
finished-part handover observable offline.

Scripts here must be callables keyed on the prompt: a round-robin sequence would hand replies out in
arrival order, which is exactly the pairing the tests exist to rule out.

:class:`FrozenClock` stands in for a judge module's ``datetime`` so the two files carry the same
``judged_at`` and the comparison can be the bytes themselves rather than the rows minus a timestamp.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from sociology.model_stub import SCRIPTED_STOP_REASON, ScriptedDetailedBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from datetime import tzinfo

    from reward_hacking.model_backend import BedrockCompletion


class FrozenClock:
    """The one method the judge loops call on ``datetime``, returning one fixed instant."""

    @staticmethod
    def now(tz: tzinfo) -> datetime:
        return datetime(2026, 9, 2, 12, 0, 0, tzinfo=tz)


class ShortListBackend(ScriptedDetailedBackend):
    """A backend without a stream that drops the last completion of every chunk: a transport bug.

    What the per-chunk path's ``zip(strict=True)`` used to refuse, restated for the shared loop: a
    results list one short, from a backend the primitive cannot know is broken because it has no
    stream to break. ``calls`` counts how many chunks were requested before the refusal.
    """

    def __init__(self, responses: Callable[[str], str]) -> None:
        super().__init__(responses)
        self.calls = 0

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        self.calls += 1
        completions = super().generate_detailed(prompts)
        del completions[-1]
        return completions


class LatencyScriptedBackend(ScriptedDetailedBackend):
    """The scripted backend with a scripted latency stamped onto every completion; no streaming seam."""

    def __init__(
        self,
        responses: Callable[[str], str],
        *,
        latency: Callable[[str], float],
        fail_on: Iterable[str] = (),
        model_id: str = "scripted",
        stop_reason: str = SCRIPTED_STOP_REASON,
    ) -> None:
        super().__init__(responses, model_id=model_id, stop_reason=stop_reason)
        self._latency = latency
        self._fail_on = frozenset(fail_on)

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Serve one completion per prompt, raising like a request bug on a ``fail_on`` prompt."""
        failing = [prompt for prompt in prompts if prompt in self._fail_on]
        if failing:
            raise RuntimeError(f"scripted request bug on {failing[0][:40]!r}")
        completions = super().generate_detailed(prompts)
        return [
            replace(
                completion,
                elapsed_seconds=self._latency(prompt),
                first_event_seconds=self._latency(prompt),
                attempts=1,
            )
            for prompt, completion in zip(prompts, completions, strict=True)
        ]


class LatencyStreamingBackend(LatencyScriptedBackend):
    """The same script through the streaming seam, landing calls shortest-latency first."""

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        """Yield in scripted-latency order; a scripted bug is raised after every other prompt landed."""
        indexed = list(enumerate(prompts))
        bug: str | None = None
        for index, prompt in sorted(indexed, key=lambda pair: (self._latency(pair[1]), pair[0])):
            if prompt in self._fail_on:
                bug = prompt
                continue
            (completion,) = self.generate_detailed([prompt])
            yield index, completion
        if bug is not None:
            raise RuntimeError(f"scripted request bug on {bug[:40]!r}")
