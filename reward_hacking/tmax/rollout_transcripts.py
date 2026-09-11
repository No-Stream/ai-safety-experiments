r"""Decode released TMAX rollouts into turns, and bridge them to the detector and the per-step readout.

:class:`reward_hacking.tmax.rollout_schema.RolloutRow` gives one row as token ids. This module turns
the ids into the agent trajectory as the environment structured it -- assistant turns (private
reasoning, then the visible tool call) alternating with the environment's tool-response user turns --
and derives from that everything the training-time gaming readout needs: the ordered bash commands with
a real exit status each, the per-step aggregates, and the record shape
:mod:`reward_hacking.tmax.rollout_analysis` consumes.

Facts read off the release on 2026-09-03 (the ``91-94`` restart fragment, 1,024 rows), each of which a
design choice below rests on:

* **Segment on token ids, never on decoded text.** ``<|im_start|>`` (248045) opens every turn after the
  first and ``<|im_end|>`` (248046) closes each one; the response begins inside the first assistant turn
  because the prompt ends with ``<|im_start|>assistant\\n<think>\\n``, and it ends with that same prefix
  for the turn that never came (or mid-turn, when ``finish_reason`` is ``length``). A global regex over
  the decoded text mis-pairs calls and responses on 5 of the 1,024 rows -- a malformed call with nested
  closing tags counts twice, a tool output that quotes ``<tool_call>`` counts once -- while the ids
  reproduce the environment's own turn structure exactly, so a turn and the response that answered it
  are paired by construction.
* **The exit status is in the tool response, not in ``tool_call_stats``.** Every command's terminal
  output ends ``(exit_code=N)``; the stats' ``success`` flag is true on 2,402 responses whose exit code
  was non-zero (it means "the tool ran"), and is therefore not the gate the detector's tampering
  signals need. Four response shapes carry no exit code: the verifier block that answers the submit
  command (994 of 995 submits), a ``Format error:`` rejecting a call the harness's stricter parser
  refused (so it never ran, even where this module reads it as well-formed), a
  ``Step 'bash' timed out`` message, and a ``Step 'bash' failed:`` ray traceback from the
  environment's own step; each gets the status it actually implies.
* **Two caps cut turns, and ``ended_by == "length"`` covers both.** A per-turn cap of 16,384 sampled
  tokens leaves an assistant turn with no close. When that turn still carried a parseable tool call
  the environment ran the fragment and the trajectory continued (110 of the 128,000 released rows);
  far more often the capped turn was a single degenerate final turn with no command in it and the
  episode ended there (7,857 rows, none rewarded). The 65,536-token total cap cuts anywhere,
  including inside the next turn's template prefix (3,057 rows). Read ``n_response_tokens == 65_536``
  to tell the two apart on a length-ended row. Both are truncation, never a structural fault, and a
  policy closing a turn with
  ``<|endoftext|>`` instead of ``<|im_end|>`` counts as closed because the environment accepted it.
* **The verifier block after the submit command reveals the reward** (``Reward: 1.0`` and the pytest
  output), so anything shown to a judge that must stay blind to the reward has to stop at the submit
  call. :meth:`RolloutTranscript.judge_turns` does that and :func:`assert_no_verifier_leak` is the
  check on the rendered text.
* **User-turn tokens carry a logprob of exactly 0.0** (they were inserted by the environment, not
  sampled), so an entropy proxy over the whole ``logprobs`` list is diluted by the length of the
  terminal output. :attr:`RolloutTranscript.mean_assistant_neg_logprob` is over sampled tokens only.
* **Text-level decode round-trips exactly**: ``decode(encode(decode(ids))) == decode(ids)`` on 1,024 of
  1,024 rows. Id-level does not (12 of 20) because a sampled token sequence is not the canonical BPE of
  its own text, which is expected and harmless.

The tokenizer is injected as a decode callable rather than loaded here, so the parsing is testable
offline with a toy vocabulary; :func:`load_decoder` builds the real one from
:data:`TOKENIZER_REPO_ID`, the producing model named in the release's metadata, whose
``tokenizer.json`` is byte-identical to ``Qwen/Qwen3.5-9B``'s (``allenai/tmax-9b``'s adds seven audio
tokens above the ids the rollouts use, so it decodes identically too).

The ``scan`` CLI runs the whole pipeline over every decompressed fragment in parallel, writes one
compact record per row (identity, reward, shape, entropy proxy, the detector's signals), one decoded
transcript per row for later re-analysis, dedupes the restart overlaps preferring the later fragment,
and renders the per-step table. Run it under ``scripts/resource-limits.sh``; it is CPU-heavy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from reward_hacking.harness.hack_detector import TAMPERING_SIGNALS
from reward_hacking.tmax.artifacts import FLAGSHIP_ROLLOUT_RUN_PREFIX, CheckpointStage
from reward_hacking.tmax.rollout_analysis import (
    GAMING_BEHAVIOR_SIGNALS,
    assess_records,
)
from reward_hacking.tmax.rollout_schema import RolloutRow, extract_bash_commands

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

# The producing model per the release metadata; tokenizer.json byte-identical to Qwen/Qwen3.5-9B's.
TOKENIZER_REPO_ID = "hamishivi/Qwen3.5-9B"

IM_START_TOKEN_ID = 248045
IM_END_TOKEN_ID = 248046
IM_START_TEXT = "<|im_start|>"
IM_END_TEXT = "<|im_end|>"
THINK_CLOSE = "</think>"
ASSISTANT_TURN_PREFIX = "assistant\n<think>\n"
# "assistant", "\n", "<think>", "\n" is four tokens for the real tokenizer; the search allows more.
ASSISTANT_PREFIX_MAX_TOKENS = 8
USER_TURN_PREFIX = "user\n"
TOOL_RESPONSE_OPEN = "<tool_response>\n"
TOOL_RESPONSE_CLOSE = "\n</tool_response>"

SUBMIT_COMMAND = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
VERIFIER_BLOCK_PREFIX = "Test execution complete."
FORMAT_ERROR_PREFIX = "Format error:"
TIMEOUT_PREFIX = "Step 'bash' timed out"
# The environment's own step raised (a ray actor traceback); the command's fate is unknown.
STEP_ERROR_PREFIX = "Step 'bash' failed"
# A policy sometimes closes a turn with the EOS token instead of <|im_end|>; the environment accepts it.
EOS_TEXT = "<|endoftext|>"
# The per-turn generation cap the environment applied: a turn of exactly this many sampled tokens
# with no close was cut. If the fragment held a tool call the environment answered it; a capped final
# turn with no command ended the episode, which is the common case late in the TMAX run.
TURN_TOKEN_CAP = 16_384
# The conventional status of a command killed by its time limit; the environment reports no number.
TIMEOUT_EXIT_CODE = 124
EXIT_CODE_PATTERN = re.compile(r"\(exit_code=(-?\d+)\)\s*\Z")
# The environment's own reward line inside the verifier block; the leak check greps for it.
REWARD_LINE_PATTERN = re.compile(r"^Reward: [01]\.0\s*$", re.MULTILINE)

# The row-level constants the detector record carries; every released row is the flagship's own RL.
DETECTOR_SUITE = "tmax_15k"
DETECTOR_MODEL_ID = "allenai/tmax-9b@training"

# Observed episode cap: rows with 64 tool steps and no submit end with finish_reason "stop".
TURN_LIMIT = 64

ResponseKind = Literal["command", "verifier", "format_error", "timeout", "step_error", "truncated"]
EndedBy = Literal[
    "submit", "length", "rejected_response", "turn_limit", "other", "env_reset_failed"
]


class TranscriptStructureError(ValueError):
    """A row whose turn structure is not the one the release was verified to have."""


@dataclass(frozen=True)
class AssistantTurn:
    """One sampled turn: private reasoning, then the visible text that carries the tool call."""

    index: int
    reasoning: str
    visible: str
    command: str | None
    malformed_calls: tuple[str, ...]
    n_tokens: int
    mean_logprob: float | None
    truncated: bool = False

    @property
    def is_submit(self) -> bool:
        """Whether this turn's command is the environment's end-of-task signal."""
        return self.command is not None and self.command.strip() == SUBMIT_COMMAND


@dataclass(frozen=True)
class ToolResponse:
    """One environment turn: the terminal output that answered the assistant turn of the same index."""

    index: int
    text: str
    kind: ResponseKind
    exit_code: int | None


def _excerpt(text: str, limit: int = 120) -> str:
    """Head of a string for an error message: a display truncation, not a data reduction."""
    return text if len(text) <= limit else text[:limit] + "..."


def classify_response(text: str) -> tuple[ResponseKind, int | None]:
    """Read a tool response's kind and the exit status it implies, refusing an unknown shape."""
    match = EXIT_CODE_PATTERN.search(text)
    if match is not None:
        return "command", int(match.group(1))
    if text.startswith(VERIFIER_BLOCK_PREFIX):
        return "verifier", 0
    if text.startswith(FORMAT_ERROR_PREFIX):
        return "format_error", None
    if text.startswith(TIMEOUT_PREFIX):
        return "timeout", TIMEOUT_EXIT_CODE
    if text.startswith(STEP_ERROR_PREFIX):
        return "step_error", None
    raise TranscriptStructureError(
        f"tool response carries no exit code and is none of the three known shapes: {_excerpt(text)!r}"
    )


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _prefix_token_count(ids: Sequence[int], decode: Callable[[Sequence[int]], str]) -> int:
    """Return how many leading tokens spell the environment's assistant-turn template prefix."""
    for count in range(1, min(ASSISTANT_PREFIX_MAX_TOKENS, len(ids)) + 1):
        if decode(ids[:count]) == ASSISTANT_TURN_PREFIX:
            return count
    raise TranscriptStructureError("assistant segment does not open with the template prefix")


def _is_partial_prefix(text: str) -> bool:
    """Whether a final segment is only the start (possibly empty) of either role's template prefix.

    The complete assistant prefix is not partial: it is the ordinary trailing prefix of the turn that
    never came, which the caller handles as such.
    """
    if text == ASSISTANT_TURN_PREFIX:
        return False
    return ASSISTANT_TURN_PREFIX.startswith(text) or USER_TURN_PREFIX.startswith(text)


def _empty_turn(index: int) -> AssistantTurn:
    """Build the turn for an empty generation: no text, no call, no sampled tokens."""
    return AssistantTurn(
        index=index,
        reasoning="",
        visible="",
        command=None,
        malformed_calls=(),
        n_tokens=0,
        mean_logprob=None,
    )


def _split_segments(
    ids: Sequence[int], logprobs: Sequence[float]
) -> list[tuple[list[int], list[float]]]:
    """Split a response at every environment-inserted ``<|im_start|>``; segment 0 is the first turn.

    Environment-inserted tokens carry a logprob of exactly 0.0, so an ``<|im_start|>`` with a
    non-zero logprob is one the policy generated as text (a collapsing policy imitates the chat
    template) and stays inside its turn rather than opening a new one.
    """
    segments: list[tuple[list[int], list[float]]] = []
    current_ids: list[int] = []
    current_lps: list[float] = []
    for token, logprob in zip(ids, logprobs, strict=True):
        if token == IM_START_TOKEN_ID and logprob == 0.0:
            segments.append((current_ids, current_lps))
            current_ids, current_lps = [], []
            continue
        current_ids.append(token)
        current_lps.append(logprob)
    segments.append((current_ids, current_lps))
    return segments


def _strip_im_end(text: str, *, where: str, allow_unclosed: bool) -> tuple[str, bool]:
    """Drop a turn's closing token; return the body and whether the turn was closed at all.

    ``<|im_end|>`` is the template's close; a policy sometimes emits ``<|endoftext|>`` instead and the
    environment accepts that, so both count as closed. An unclosed turn is legitimate in two places:
    any assistant turn, because the environment's per-turn cap (:data:`TURN_TOKEN_CAP`) cuts a
    generation mid-text and then answers whatever fragment it parsed; and the response's final segment,
    because the 65,536-token total cap cuts a trajectory wherever it happens to be, a tool response
    included. An unclosed environment turn anywhere else is a structural fault.
    """
    stripped = text.rstrip("\n")
    for close in (IM_END_TEXT, EOS_TEXT):
        if stripped.endswith(close):
            return stripped[: -len(close)], True
    if allow_unclosed:
        return text, False
    raise TranscriptStructureError(f"{where} does not end with {IM_END_TEXT}: {_excerpt(text)!r}")


def _parse_assistant(text: str, index: int, n_tokens: int, lps: Sequence[float]) -> AssistantTurn:
    """Split an assistant turn into reasoning and visible text, and read its tool call.

    An unclosed turn is one the per-turn or the total token cap cut; it parses like any other and is
    marked ``truncated``, so a tool call the cap split in half reads as malformed rather than as a
    command, which is also how the environment read it.
    """
    body, closed = _strip_im_end(text, where=f"assistant turn {index}", allow_unclosed=True)
    reasoning, sep, visible = body.partition(THINK_CLOSE)
    if not sep:
        # No </think>: the turn was cut off inside its reasoning (finish_reason "length").
        reasoning, visible = body, ""
    calls = extract_bash_commands(visible)
    command = calls.commands[0] if len(calls.commands) == 1 else None
    malformed = calls.malformed
    if len(calls.commands) > 1:
        # Two well-formed calls in one turn: the environment rejects the response, so neither ran.
        malformed = (*calls.malformed, *calls.commands)
    return AssistantTurn(
        index=index,
        reasoning=reasoning.strip("\n"),
        visible=visible.strip("\n"),
        command=command,
        malformed_calls=malformed,
        n_tokens=n_tokens,
        mean_logprob=_mean(lps),
        truncated=not closed,
    )


def _parse_user(text: str, index: int, *, final: bool) -> ToolResponse:
    """Unwrap a tool-response user turn and classify what the environment said.

    A response the token cap cut carries no exit status; it is kept as ``truncated`` with whatever
    text survived, so the command it answered is neither credited as succeeded nor read as failed.
    """
    body, closed = _strip_im_end(text, where=f"tool response {index}", allow_unclosed=final)
    body = body.rstrip("\n")
    if not closed:
        inner = body.removeprefix(TOOL_RESPONSE_OPEN)
        return ToolResponse(index=index, text=inner, kind="truncated", exit_code=None)
    if not body.startswith(TOOL_RESPONSE_OPEN) or not body.endswith(TOOL_RESPONSE_CLOSE):
        raise TranscriptStructureError(
            f"tool response {index} is not wrapped in <tool_response>: {_excerpt(body)!r}"
        )
    inner = body[len(TOOL_RESPONSE_OPEN) : -len(TOOL_RESPONSE_CLOSE)]
    kind, exit_code = classify_response(inner)
    return ToolResponse(index=index, text=inner, kind=kind, exit_code=exit_code)


@dataclass(frozen=True)
class RolloutTranscript:
    """One rollout as turns, with the identity and outcome fields the readout groups on.

    ``trainer_step`` is one-indexed, matching the manifests and the ``step_NNN`` checkpoint branches;
    the row's own ``step`` is zero-indexed. ``key`` identifies the row across restart fragments, so a
    deduped corpus has exactly one transcript per key.
    """

    fragment: str
    step: int
    prompt_idx: int
    sample_idx: int
    task_id: str
    reward: float
    advantage: float
    finish_reason: str
    timed_out: bool
    n_prompt_tokens: int
    n_response_tokens: int
    prompt_text: str
    turns: tuple[AssistantTurn, ...]
    responses: tuple[ToolResponse, ...]
    trailing_prefix: bool
    n_stats: int
    tail_cut: bool = False

    @property
    def trainer_step(self) -> int:
        """The manifests' one-indexed trainer step."""
        return self.step + 1

    @property
    def key(self) -> str:
        """Row identity across fragments: one-indexed step, prompt slot, flat sample index."""
        return f"step{self.trainer_step:03d}:p{self.prompt_idx}:s{self.sample_idx:03d}"

    @property
    def identity_digest(self) -> str:
        """A seed derived from identity alone, so a sample is reproducible and order-independent."""
        return hashlib.sha256(self.key.encode()).hexdigest()

    @property
    def ended_by(self) -> EndedBy:
        """How the episode stopped, read off the structure rather than off ``finish_reason`` alone."""
        if self.responses and self.responses[-1].kind == "verifier":
            return "submit"
        if self.finish_reason == "length" or self.truncated_tail:
            return "length"
        if self.turns and len(self.responses) == len(self.turns) - 1:
            # The last turn drew no response: the environment rejected it and ended the episode.
            return "rejected_response"
        if len(self.turns) >= TURN_LIMIT:
            return "turn_limit"
        return "other"

    @property
    def transcript_digest(self) -> str:
        """Digest of the sampled text, so two fragments' copies of a key can be told apart."""
        joined = "\x00".join(f"{turn.reasoning}\x01{turn.visible}" for turn in self.turns)
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    @property
    def n_commands(self) -> int:
        """Well-formed bash commands the environment answered."""
        return len(self.commands_with_exit_codes())

    @property
    def n_malformed_calls(self) -> int:
        """Tool-call bodies that were not exactly one well-formed bash command."""
        return sum(len(turn.malformed_calls) for turn in self.turns)

    @property
    def n_assistant_tokens(self) -> int:
        """Sampled tokens only; the environment's inserted tokens are excluded."""
        return sum(turn.n_tokens for turn in self.turns)

    @property
    def mean_assistant_neg_logprob(self) -> float | None:
        """Mean negative log-probability per sampled token: the run's entropy proxy.

        Token-weighted across turns, so a long turn counts for its length; user-turn tokens sit at
        exactly 0.0 in the release and are excluded rather than allowed to dilute the figure.
        """
        weighted = [
            -turn.mean_logprob * turn.n_tokens
            for turn in self.turns
            if turn.mean_logprob is not None and turn.n_tokens
        ]
        total = sum(turn.n_tokens for turn in self.turns if turn.mean_logprob is not None)
        return sum(weighted) / total if total else None

    def commands_with_exit_codes(self) -> tuple[tuple[str, int], ...]:
        """Pair every EXECUTED well-formed command with the exit status the environment reported.

        Only commands the environment ran carry an exit status, so only they feed the detector: a
        command answered by a ``format_error`` was rejected before it ran (the harness's tool parser
        is stricter than this one -- an empty ``command`` parameter, or text after the ``</tool_call>``
        the prompt forbids), a command answered by a cut-off ``truncated`` response has no verdict,
        and a command with no response at all ended the episode. None of the three executed, so
        crediting or failing them would be a fabricated exit status; :attr:`n_rejected_calls` and
        :attr:`n_unanswered_commands` keep them visible instead.
        """
        return tuple(
            (turn.command, response.exit_code)
            for turn, response in zip(self.turns, self.responses, strict=False)
            if turn.command is not None and response.exit_code is not None
        )

    @property
    def n_rejected_calls(self) -> int:
        """Count well-formed commands the environment did not execute.

        A format error rejected the call; a ``step_error`` means the environment's own step raised,
        so the command's fate is unknown. Neither carries an exit status.
        """
        return sum(
            1
            for turn, response in zip(self.turns, self.responses, strict=False)
            if turn.command is not None and response.kind in ("format_error", "step_error")
        )

    @property
    def n_unanswered_commands(self) -> int:
        """Well-formed commands with no exit status: unanswered, or answered by a cut-off response."""
        answered = len(self.responses)
        unanswered = sum(1 for turn in self.turns[answered:] if turn.command is not None)
        cut = sum(
            1
            for turn, response in zip(self.turns, self.responses, strict=False)
            if turn.command is not None and response.kind == "truncated"
        )
        return unanswered + cut

    @property
    def truncated_tail(self) -> bool:
        """Whether the total cap cut the trajectory's tail.

        True for an unclosed final turn or response, or a final segment that is only part of the
        next turn's template prefix.
        """
        if self.tail_cut:
            return True
        if self.responses and self.responses[-1].kind == "truncated":
            return True
        return (
            bool(self.turns) and self.turns[-1].truncated and len(self.responses) < len(self.turns)
        )

    @property
    def n_turns_cut_by_turn_cap(self) -> int:
        """Assistant turns the environment cut at its per-turn cap and then answered anyway."""
        return sum(1 for turn, _ in zip(self.turns, self.responses, strict=False) if turn.truncated)

    def detector_record(self) -> dict[str, object]:
        """Render the record :mod:`reward_hacking.tmax.rollout_analysis` parses and assesses."""
        pairs = self.commands_with_exit_codes()
        final = self.turns[-1] if self.turns else None
        return {
            "episode_id": self.key,
            "task_id": self.task_id,
            "suite": DETECTOR_SUITE,
            "stage": CheckpointStage.RL.value,
            "reward": self.reward,
            "commands": [command for command, _ in pairs],
            "exit_codes": [code for _, code in pairs],
            "model_id": DETECTOR_MODEL_ID,
            "final_message": (f"{final.reasoning}\n{final.visible}".strip() if final else ""),
        }

    def judge_turns(self) -> tuple[tuple[AssistantTurn, ToolResponse | None], ...]:
        """Return the turns a reward-blind judge may see: up to and including the submit call.

        The response to the submit command is the verifier block, which prints the reward; it is
        dropped. Every other turn is paired with its response (``None`` for an unanswered final turn).
        """
        pairs: list[tuple[AssistantTurn, ToolResponse | None]] = []
        for index, turn in enumerate(self.turns):
            response = self.responses[index] if index < len(self.responses) else None
            if response is not None and response.kind == "verifier":
                response = None
            pairs.append((turn, response))
        return tuple(pairs)

    @classmethod
    def from_row(
        cls, row: RolloutRow, decode: Callable[[Sequence[int]], str], *, fragment: str
    ) -> RolloutTranscript:
        """Decode and segment one verified row; any deviation from the verified structure raises."""
        segments = _split_segments(row.response_tokens, row.logprobs)
        turns: list[AssistantTurn] = []
        responses: list[ToolResponse] = []
        trailing_prefix = False
        tail_cut = False
        for position, (ids, lps) in enumerate(segments):
            text = decode(ids)
            final = position == len(segments) - 1
            if position == 0:
                if not ids:
                    raise TranscriptStructureError("response begins with <|im_start|>, not a turn")
                turns.append(_parse_assistant(text, len(turns), len(ids), lps))
                continue
            if final and _is_partial_prefix(text):
                # The total cap cut the trajectory inside the next turn's template prefix.
                trailing_prefix = True
                tail_cut = True
                continue
            if text.startswith(USER_TURN_PREFIX):
                responses.append(
                    _parse_user(text[len(USER_TURN_PREFIX) :], len(responses), final=final)
                )
            elif text.startswith(ASSISTANT_TURN_PREFIX):
                body = text[len(ASSISTANT_TURN_PREFIX) :]
                if final and body == "":
                    trailing_prefix = True
                    continue
                # The prefix tokens are the environment's, not sampled: keep them out of the count.
                prefix_len = _prefix_token_count(ids, decode)
                sampled_ids = ids[prefix_len:]
                sampled_lps = lps[prefix_len:]
                if body == "":
                    # An empty generation the environment answered: keep turn i paired with response i.
                    turns.append(_empty_turn(len(turns)))
                    continue
                turns.append(_parse_assistant(body, len(turns), len(sampled_ids), sampled_lps))
            else:
                raise TranscriptStructureError(
                    f"segment {position} opens with neither role prefix: {_excerpt(text)!r}"
                )
        if len(responses) not in (len(turns), len(turns) - 1):
            raise TranscriptStructureError(
                f"{len(turns)} assistant turns against {len(responses)} tool responses; the "
                "environment answers every turn except possibly the last"
            )
        return cls(
            fragment=fragment,
            step=row.step,
            prompt_idx=row.prompt_idx,
            sample_idx=row.sample_idx,
            task_id=row.task_id,
            reward=row.reward,
            advantage=row.advantage,
            finish_reason=row.finish_reason,
            timed_out=row.timed_out,
            n_prompt_tokens=len(row.prompt_tokens),
            n_response_tokens=len(row.response_tokens),
            prompt_text=decode(row.prompt_tokens),
            turns=tuple(turns),
            responses=tuple(responses),
            trailing_prefix=trailing_prefix,
            n_stats=len(row.tool_call_stats),
            tail_cut=tail_cut,
        )


def assert_no_verifier_leak(rendered: str) -> None:
    """Refuse rendered judge text that still carries the environment's verdict block.

    Both markers are checked: the block's opening line and the ``Reward: N.0`` line it ends with. An
    agent could print either string itself, in which case this raises on an innocent transcript and
    the row gets read by hand -- the safe direction for a blindness check to fail in.
    """
    if VERIFIER_BLOCK_PREFIX in rendered or REWARD_LINE_PATTERN.search(rendered) is not None:
        raise TranscriptStructureError(
            "rendered judge text carries the verifier block or its reward line; the judge would "
            "not be blind to the reward"
        )


def load_decoder(repo_id: str = TOKENIZER_REPO_ID) -> Callable[[Sequence[int]], str]:
    """Build the real decode callable from the producing model's tokenizer."""
    # Imported here so the torch-heavy dependency stays off the offline test and CLI-help paths.
    from transformers import AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(repo_id)

    def decode(ids: Sequence[int]) -> str:
        return str(tokenizer.decode(list(ids), skip_special_tokens=False))

    return decode


# ---------------------------------------------------------------------------------------------
# The scan: per-row compact records, decoded transcripts, dedupe, per-step aggregates.
# ---------------------------------------------------------------------------------------------

STEP_BUCKET_WIDTH = 100


def step_bucket(trainer_step: int, *, width: int = STEP_BUCKET_WIDTH) -> str:
    """Label the one-indexed trainer step's bucket, e.g. ``001-100``."""
    low = ((trainer_step - 1) // width) * width + 1
    return f"{low:03d}-{low + width - 1:03d}"


def compact_record(
    transcript: RolloutTranscript, fired: Iterable[str], label: str
) -> dict[str, Any]:
    """One row's identity, outcome, shape, entropy proxy and detector read -- no transcript text."""
    fired_set = frozenset(fired)
    return {
        "key": transcript.key,
        "fragment": transcript.fragment,
        "trainer_step": transcript.trainer_step,
        "step_bucket": step_bucket(transcript.trainer_step),
        "prompt_idx": transcript.prompt_idx,
        "sample_idx": transcript.sample_idx,
        "task_id": transcript.task_id,
        "reward": transcript.reward,
        "advantage": transcript.advantage,
        "finish_reason": transcript.finish_reason,
        "ended_by": transcript.ended_by,
        "env_reset_failed": False,
        "timed_out": transcript.timed_out,
        "n_prompt_tokens": transcript.n_prompt_tokens,
        "n_response_tokens": transcript.n_response_tokens,
        "n_assistant_tokens": transcript.n_assistant_tokens,
        "n_turns": len(transcript.turns),
        "n_responses": len(transcript.responses),
        "n_stats": transcript.n_stats,
        "n_commands": transcript.n_commands,
        "n_malformed_calls": transcript.n_malformed_calls,
        "n_unanswered_commands": transcript.n_unanswered_commands,
        "n_rejected_calls": transcript.n_rejected_calls,
        "n_turns_cut_by_turn_cap": transcript.n_turns_cut_by_turn_cap,
        "n_nonzero_exit": sum(1 for _, code in transcript.commands_with_exit_codes() if code != 0),
        "mean_assistant_neg_logprob": transcript.mean_assistant_neg_logprob,
        "detector_label": label,
        "detector_fired": sorted(fired_set),
        "detector_gaming": bool(fired_set & frozenset(GAMING_BEHAVIOR_SIGNALS)),
        "detector_tampering": bool(fired_set & frozenset(TAMPERING_SIGNALS)),
        "truncated_tail": transcript.truncated_tail,
        "identity_digest": transcript.identity_digest,
        "transcript_digest": transcript.transcript_digest,
    }


def env_failure_record(row: RolloutRow, *, fragment: str) -> dict[str, Any]:
    """Record a row whose environment never started: identity and reward, no behaviour.

    Kept in the records so every per-step denominator can name it, and flagged so no behavioural rate
    counts it: the policy sampled nothing, so there is no transcript, no command and no entropy.
    """
    trainer_step = row.step + 1
    key = f"step{trainer_step:03d}:p{row.prompt_idx}:s{row.sample_idx:03d}"
    return {
        "key": key,
        "fragment": fragment,
        "trainer_step": trainer_step,
        "step_bucket": step_bucket(trainer_step),
        "prompt_idx": row.prompt_idx,
        "sample_idx": row.sample_idx,
        "task_id": row.task_id,
        "reward": row.reward,
        "advantage": row.advantage,
        "finish_reason": row.finish_reason,
        "ended_by": "env_reset_failed",
        "env_reset_failed": True,
        "timed_out": row.timed_out,
        "n_prompt_tokens": len(row.prompt_tokens),
        "n_response_tokens": len(row.response_tokens),
        "n_assistant_tokens": 0,
        "n_turns": 0,
        "n_responses": 0,
        "n_stats": len(row.tool_call_stats),
        "n_commands": 0,
        "n_malformed_calls": 0,
        "n_unanswered_commands": 0,
        "n_rejected_calls": 0,
        "n_turns_cut_by_turn_cap": 0,
        "n_nonzero_exit": 0,
        "mean_assistant_neg_logprob": None,
        "detector_label": None,
        "detector_fired": [],
        "detector_gaming": False,
        "detector_tampering": False,
        "truncated_tail": False,
        "identity_digest": hashlib.sha256(key.encode()).hexdigest(),
        "transcript_digest": "",
    }


def transcript_record(transcript: RolloutTranscript) -> dict[str, object]:
    """Render the decoded transcript as JSON: prompt, turns, responses; re-analysable without tokens."""
    return {
        "key": transcript.key,
        "fragment": transcript.fragment,
        "task_id": transcript.task_id,
        "reward": transcript.reward,
        "prompt_text": transcript.prompt_text,
        "turns": [asdict(turn) for turn in transcript.turns],
        "responses": [asdict(response) for response in transcript.responses],
        "trailing_prefix": transcript.trailing_prefix,
    }


def fragment_of(path: Path) -> str:
    """Return the restart fragment a rollouts JSONL belongs to, from the release's layout."""
    return path.parent.parent.name


def fragment_order(fragment: str) -> int:
    """Return the restart's launch timestamp (the suffix after the run prefix); later sorts higher."""
    prefix = f"{FLAGSHIP_ROLLOUT_RUN_PREFIX}__"
    if not fragment.startswith(prefix):
        raise ValueError(f"{fragment!r} is not a fragment of {FLAGSHIP_ROLLOUT_RUN_PREFIX}")
    return int(fragment[len(prefix) :])


def scan_file(
    path: Path,
    records_out: Path,
    transcripts_out: Path,
    *,
    tokenizer_repo_id: str = TOKENIZER_REPO_ID,
) -> dict[str, int]:
    """Decode, assess and write every row of one rollouts JSONL; return counts."""
    decode = load_decoder(tokenizer_repo_id)
    fragment = fragment_of(path)
    counts: Counter[str] = Counter()
    records_out.parent.mkdir(parents=True, exist_ok=True)
    transcripts_out.parent.mkdir(parents=True, exist_ok=True)
    # Renamed from .partial only on completion, so a file killed mid-scan is rescanned not skipped.
    records_partial = records_out.with_suffix(".partial")
    transcripts_partial = transcripts_out.with_suffix(".partial")
    failures_out = failures_path(records_out)
    with (
        path.open() as source,
        records_partial.open("w") as records,
        transcripts_partial.open("w") as transcripts,
        failures_out.open("w") as failures,
    ):
        for line in source:
            if not line.strip():
                continue
            row = RolloutRow.from_dict(json.loads(line))
            if row.env_reset_failed:
                counts["env_reset_failed"] += 1
                records.write(json.dumps(env_failure_record(row, fragment=fragment)) + "\n")
                continue
            try:
                transcript = RolloutTranscript.from_row(row, decode, fragment=fragment)
            except TranscriptStructureError as error:
                # A structure this parser has not seen is recorded per row, with its identity, and
                # counted into the summary; it is a denominator to report, not a reason to lose the
                # rest of a 67 GB pass. The summary and readout must state the count.
                counts["parse_failures"] += 1
                failures.write(
                    json.dumps(
                        {
                            "fragment": fragment,
                            "trainer_step": row.step + 1,
                            "prompt_idx": row.prompt_idx,
                            "sample_idx": row.sample_idx,
                            "task_id": row.task_id,
                            "reward": row.reward,
                            "finish_reason": row.finish_reason,
                            "n_response_tokens": len(row.response_tokens),
                            "error": str(error),
                        }
                    )
                    + "\n"
                )
                logger.warning("%s: unparsed row step %d: %s", path.name, row.step + 1, error)
                continue
            assessment = assess_records([transcript.detector_record()])[0]
            records.write(
                json.dumps(
                    compact_record(
                        transcript, assessment.fired_signals, assessment.assessment.label.value
                    )
                )
                + "\n"
            )
            transcripts.write(json.dumps(transcript_record(transcript)) + "\n")
            counts["rows"] += 1
    records_partial.rename(records_out)
    transcripts_partial.rename(transcripts_out)
    logger.info(
        "scanned %s: %d rows, %d unparsed, %d environment-reset failures",
        path.name,
        counts["rows"],
        counts["parse_failures"],
        counts["env_reset_failed"],
    )
    return dict(counts)


def failures_path(records_out: Path) -> Path:
    """Where a per-file scan writes the rows whose structure it could not parse."""
    return records_out.with_name(records_out.name.replace(".records.jsonl", ".failures.jsonl"))


def _scan_one(args: tuple[Path, Path, Path]) -> dict[str, int]:
    return scan_file(*args)


def dedupe_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep one record per key, preferring the later restart fragment; report the overlap."""
    kept: dict[str, dict[str, Any]] = {}
    overlap: Counter[str] = Counter()
    for record in records:
        key = str(record["key"])
        existing = kept.get(key)
        if existing is None:
            kept[key] = dict(record)
            continue
        overlap["duplicate_keys"] += 1
        same = existing["transcript_digest"] == record["transcript_digest"]
        overlap["duplicates_identical_text" if same else "duplicates_differ"] += 1
        if fragment_order(str(record["fragment"])) > fragment_order(str(existing["fragment"])):
            kept[key] = dict(record)
    ordered = sorted(
        kept.values(), key=lambda r: (r["trainer_step"], r["prompt_idx"], r["sample_idx"])
    )
    return ordered, dict(overlap)


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def step_aggregates(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per trainer step: reward, length, turns, endings, entropy proxy, detector rate with denominators."""
    by_step: dict[int, list[Mapping[str, Any]]] = {}
    for record in records:
        by_step.setdefault(int(record["trainer_step"]), []).append(record)
    rows: list[dict[str, Any]] = []
    for step in sorted(by_step):
        # Records written before the flag existed can only lack it because the scan refused
        # environment-failure rows outright then, so an absent flag is a started environment.
        all_rows = by_step[step]
        env_failed = [r for r in all_rows if bool(r.get("env_reset_failed", False))]
        group = [r for r in all_rows if not bool(r.get("env_reset_failed", False))]
        rewarded = [r for r in group if float(r["reward"]) > 0]
        with_command = [r for r in group if int(r["n_commands"]) > 0]
        gaming = [r for r in group if bool(r["detector_gaming"])]
        tampering = [r for r in group if bool(r["detector_tampering"])]
        neg_lp = [
            float(r["mean_assistant_neg_logprob"])
            for r in group
            if r["mean_assistant_neg_logprob"] is not None
        ]
        lengths = [int(r["n_response_tokens"]) for r in group] or [0]
        rows.append(
            {
                "trainer_step": step,
                "n_rows_total": len(all_rows),
                "n_env_reset_failed": len(env_failed),
                "n_rows": len(group),
                "n_rewarded": len(rewarded),
                "mean_reward": _share(len(rewarded), len(group)),
                "mean_reward_as_trained": _share(len(rewarded), len(all_rows)),
                "mean_response_tokens": statistics.fmean(lengths),
                "median_response_tokens": statistics.median(lengths),
                "mean_turns": statistics.fmean([int(r["n_turns"]) for r in group] or [0]),
                "finish_length_share": _share(
                    sum(1 for r in group if r["finish_reason"] == "length"), len(group)
                ),
                "ended_by": dict(Counter(str(r["ended_by"]) for r in group)),
                "mean_neg_logprob": statistics.fmean(neg_lp) if neg_lp else None,
                "n_with_command": len(with_command),
                "n_detector_gaming": len(gaming),
                "n_detector_gaming_rewarded": sum(1 for r in gaming if float(r["reward"]) > 0),
                "detector_gaming_rate": _share(len(gaming), len(group)),
                "detector_gaming_rate_rewarded": _share(
                    sum(1 for r in gaming if float(r["reward"]) > 0), len(rewarded)
                ),
                "n_detector_tampering": len(tampering),
                "n_detector_tampering_rewarded": sum(
                    1 for r in tampering if float(r["reward"]) > 0
                ),
                "detector_tampering_rate": _share(len(tampering), len(group)),
                "detector_signals": dict(Counter(s for r in gaming for s in r["detector_fired"])),
                "n_distinct_tasks": len({str(r["task_id"]) for r in group}),
            }
        )
    return rows


def render_step_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Markdown per-step table with every denominator beside its rate."""
    lines = [
        (
            "| step | rows (env failed) | rewarded | mean reward | mean resp tok | mean turns | "
            "length-cut | "
            "mean -logprob | detector any signal (all) | detector any signal (rewarded) | "
            "detector tampering (all) | detector tampering (rewarded) |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        neg = r["mean_neg_logprob"]
        lines.append(
            f"| {r['trainer_step']} | {r['n_rows']} ({r['n_env_reset_failed']}) | {r['n_rewarded']} | "
            f"{float(r['mean_reward'] or 0):.3f} | {float(r['mean_response_tokens']):.0f} | "
            f"{float(r['mean_turns']):.1f} | {float(r['finish_length_share'] or 0):.3f} | "
            f"{'-' if neg is None else f'{float(neg):.4f}'} | "
            f"{r['n_detector_gaming']}/{r['n_rows']} | "
            f"{r['n_detector_gaming_rewarded']}/{r['n_rewarded']} | "
            f"{r['n_detector_tampering']}/{r['n_rows']} | "
            f"{r['n_detector_tampering_rewarded']}/{r['n_rewarded']} |"
        )
    return "\n".join(lines) + "\n"


def _gather_failures(paths: Sequence[Path], out_dir: Path) -> list[dict[str, Any]]:
    """Concatenate every per-file failure sidecar into ``parse_failures.jsonl`` and return the rows."""
    failures: list[dict[str, Any]] = []
    for path in paths:
        failure_file = failures_path(out_dir / "per_file" / f"{path.stem}.records.jsonl")
        if failure_file.exists():
            with failure_file.open() as handle:
                failures.extend(json.loads(line) for line in handle if line.strip())
    with (out_dir / "parse_failures.jsonl").open("w") as handle:
        for failure in failures:
            handle.write(json.dumps(failure) + "\n")
    return failures


def _cmd_scan(args: argparse.Namespace) -> None:
    root: Path = args.root
    paths = sorted(root.glob("*/rollouts/*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no rollouts JSONL under {root}/*/rollouts/")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded_dir: Path = args.decoded_dir
    jobs: list[tuple[Path, Path, Path]] = []
    for path in paths:
        records_out = out_dir / "per_file" / f"{path.stem}.records.jsonl"
        transcripts_out = decoded_dir / f"{path.stem}.transcripts.jsonl"
        if records_out.exists() and transcripts_out.exists() and not args.rescan:
            logger.info("skipping %s: already scanned", path.name)
            continue
        jobs.append((path, records_out, transcripts_out))
    logger.info("%d files, %d to scan, %d workers", len(paths), len(jobs), args.workers)
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for counts in pool.map(_scan_one, jobs):
                logger.info("file done: %s", counts)
    all_records: list[dict[str, Any]] = []
    for path in paths:
        with (out_dir / "per_file" / f"{path.stem}.records.jsonl").open() as handle:
            all_records.extend(json.loads(line) for line in handle if line.strip())
    failures = _gather_failures(paths, out_dir)
    deduped, overlap = dedupe_records(all_records)
    with (out_dir / "records.jsonl").open("w") as handle:
        for record in deduped:
            handle.write(json.dumps(record) + "\n")
    aggregates = step_aggregates(deduped)
    (out_dir / "step_aggregates.json").write_text(json.dumps(aggregates, indent=1) + "\n")
    (out_dir / "step_table.md").write_text(render_step_table(aggregates))
    summary = {
        "files": len(paths),
        "rows_read": len(all_records),
        "rows_unparsed": len(failures),
        "rows_unique": len(deduped),
        "overlap": overlap,
        "steps": sorted({int(r["trainer_step"]) for r in deduped}),
        "ended_by": dict(Counter(str(r["ended_by"]) for r in deduped)),
        "finish_reason": dict(Counter(str(r["finish_reason"]) for r in deduped)),
        "env_reset_failed_total": sum(1 for r in deduped if r.get("env_reset_failed", False)),
        "detector_gaming_total": sum(1 for r in deduped if r["detector_gaming"]),
        "detector_tampering_total": sum(1 for r in deduped if r["detector_tampering"]),
        "truncated_tail_total": sum(1 for r in deduped if r["truncated_tail"]),
        "detector_signals_total": dict(
            Counter(s for r in deduped if r["detector_gaming"] for s in r["detector_fired"])
        ),
        "tokenizer_repo_id": TOKENIZER_REPO_ID,
    }
    (out_dir / "scan_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    logger.info("scan summary: %s", json.dumps(summary))


def main(argv: Sequence[str] | None = None) -> None:
    """CLI: ``scan`` every decompressed fragment into records, transcripts and the per-step table."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser(
        "scan", help="decode, assess and aggregate every rollouts JSONL under --root"
    )
    scan.add_argument(
        "--root", type=Path, required=True, help="the release's run_fragments/ directory"
    )
    scan.add_argument(
        "--out-dir", type=Path, required=True, help="where records and aggregates land"
    )
    scan.add_argument(
        "--decoded-dir", type=Path, required=True, help="where per-file decoded transcripts land"
    )
    scan.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    scan.add_argument("--rescan", action="store_true", help="redo files already scanned")
    scan.set_defaults(func=_cmd_scan)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
