"""Offline scripted stand-in for the detailed-backend protocol, for tests and dry rehearsal.

The live and batch transports both hand back ``BedrockCompletion`` objects; everything in this
package that consumes replies is written against that shape, so an offline test needs a backend
that produces it without a network. Responses are served round-robin from a script (the cursor
persists across calls, so a retry pass naturally receives the next scripted reply -- which is how
the judge's retry-once behaviour is exercised offline), or from a callable keyed on the prompt.

Two things are recorded rather than only served. ``prompts_seen`` is every prompt this backend was
actually handed, which is what the judge's blindness tests read: a prompt rendered by hand in a test
proves nothing about the path production takes. ``usage`` accumulates the same way the real backend's
does, so a caller that reports its own spend can run offline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from reward_hacking.model_backend import BedrockCompletion, TokenUsage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

SCRIPTED_STOP_REASON = "end_turn"


class ScriptedDetailedBackend:
    """Serve scripted completions through the ``DetailedBackend`` protocol, deterministically."""

    transport = "scripted"

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str], str],
        *,
        model_id: str = "scripted",
        stop_reason: str = SCRIPTED_STOP_REASON,
    ) -> None:
        """Configure the script; a callable receives each prompt, a sequence serves round-robin."""
        if not callable(responses) and not responses:
            raise ValueError("ScriptedDetailedBackend needs a non-empty script or a callable")
        self.model_id = model_id
        self.stop_reason = stop_reason
        self._responses = responses
        self._cursor = 0
        self.prompts_seen: list[str] = []
        self.usage = TokenUsage()

    def _next(self, prompt: str) -> str:
        if callable(self._responses):
            return self._responses(prompt)
        text = self._responses[self._cursor % len(self._responses)]
        self._cursor += 1
        return text

    def generate_detailed(self, prompts: list[str]) -> list[BedrockCompletion]:
        """Return one scripted completion per prompt, with length-derived token counts."""
        completions: list[BedrockCompletion] = []
        for prompt in prompts:
            self.prompts_seen.append(prompt)
            text = self._next(prompt)
            billed = TokenUsage(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4)
            self.usage += billed
            completions.append(
                BedrockCompletion(
                    text=text,
                    reasoning="",
                    usage=billed,
                    stop_reason=self.stop_reason,
                )
            )
        return completions

    def generate(self, prompts: list[str]) -> list[str]:
        """Return the scripted texts alone, satisfying the plain backend protocol."""
        return [completion.text for completion in self.generate_detailed(prompts)]
