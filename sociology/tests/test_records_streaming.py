"""The shared live loop keeps the queue full within a leg, persists in request order, and can run legs side by side.

Offline against a scripted backend that yields completions in REVERSE order through the streaming seam,
which is the sharpest offline stand-in for calls landing out of order: every property below has to hold
against it. No stimulus prose, no transcripts; the calls are synthetic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from reward_hacking.model_backend import BedrockCompletion, TokenUsage
from sociology.model_stub import ScriptedDetailedBackend
from sociology.records import completion_telemetry, load_replies, run_live_calls

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SyntheticCall:
    key: str
    prompt: str
    model_id: str
    reasoning_effort: str | None = None


def calls_for(model_id: str, count: int) -> list[SyntheticCall]:
    return [
        SyntheticCall(
            key=f"{model_id}-{index}", prompt=f"synthetic prompt {index}", model_id=model_id
        )
        for index in range(count)
    ]


def make_record(call: SyntheticCall, completion: BedrockCompletion) -> dict[str, object]:
    return {
        "key": call.key,
        "model_id": call.model_id,
        "prompt": call.prompt,
        "reply": completion.text,
        "incomplete": completion.stop_reason != "end_turn",
        **completion_telemetry(completion),
    }


class ReverseStreamingBackend(ScriptedDetailedBackend):
    """The scripted backend plus a streaming seam that hands calls back last-first.

    Reverse order is deterministic and maximally out of request order, so a loop that paired by
    arrival or released chunks as they filled would be caught every time. ``fail_on`` raises like a
    request bug after the other prompts of the same batch have landed, so the finished-part handover
    can be pinned.
    """

    def __init__(self, *args: object, fail_on: Iterable[str] = (), **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._fail_on = frozenset(fail_on)

    def submit_stream(self, prompts: Iterable[str]) -> Iterator[tuple[int, BedrockCompletion]]:
        """Yield last-first; a scripted bug is raised only after every other prompt has landed.

        That drain-then-raise shape is the real backend's contract (its in-flight calls are waited
        for and handed back before the error propagates), and it is what makes the partial-chunk
        handover observable offline.
        """
        indexed = list(enumerate(prompts))
        bug: str | None = None
        for index, prompt in reversed(indexed):
            if prompt in self._fail_on:
                bug = prompt
                continue
            (completion,) = self.generate_detailed([prompt])
            yield index, completion
        if bug is not None:
            raise RuntimeError(f"scripted request bug on {bug}")


def _sequential_reference(tmp_path: Path, calls: list[SyntheticCall]) -> list[dict[str, object]]:
    out = tmp_path / "reference.jsonl"
    run_live_calls(
        calls,
        out,
        concurrency=2,
        chunk_size=2,
        make_record=make_record,
        backend_factory=lambda model_id, effort, concurrency: ScriptedDetailedBackend(
            lambda prompt: f"reply to {prompt}"
        ),
    )
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


class TestTheLegKeepsRequestOrderOnDisk:
    def test_a_streaming_backend_writes_the_same_file_as_the_per_chunk_path(
        self, tmp_path: Path
    ) -> None:
        """SABOTAGE target: records paired by arrival order, or chunks released as they fill."""
        calls = calls_for("scripted", 5)
        out = tmp_path / "streamed.jsonl"
        counts = run_live_calls(
            calls,
            out,
            concurrency=2,
            chunk_size=2,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ReverseStreamingBackend(
                lambda prompt: f"reply to {prompt}"
            ),
        )
        streamed = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        assert streamed == _sequential_reference(tmp_path, calls)
        assert [row["key"] for row in streamed] == [call.key for call in calls]
        assert counts == {"planned": 5, "resumed": 0, "ran": 5, "incomplete": 0}

    def test_the_finished_part_of_the_chunk_in_flight_lands_before_the_raise_and_resumes(
        self, tmp_path: Path
    ) -> None:
        """SABOTAGE target: the finished siblings of a raising call discarded with the raise.

        The bug is the second chunk's first call; every other call lands before the raise. The
        first chunk is on disk whole, the finished sibling on its own under its own chunk, and the
        relaunch runs exactly the one key that never landed.
        """
        calls = calls_for("scripted", 4)
        out = tmp_path / "replies.jsonl"
        factory = lambda model_id, effort, concurrency: ReverseStreamingBackend(  # noqa: E731
            lambda prompt: f"reply to {prompt}", fail_on={calls[2].prompt}
        )
        with pytest.raises(RuntimeError, match="scripted request bug"):
            run_live_calls(
                calls,
                out,
                concurrency=2,
                chunk_size=2,
                make_record=make_record,
                backend_factory=factory,
            )
        on_disk = load_replies(out)
        assert sorted(on_disk) == [calls[0].key, calls[1].key, calls[3].key]

        healthy = lambda model_id, effort, concurrency: ReverseStreamingBackend(  # noqa: E731
            lambda prompt: f"reply to {prompt}"
        )
        counts = run_live_calls(
            calls,
            out,
            concurrency=2,
            chunk_size=2,
            make_record=make_record,
            backend_factory=healthy,
        )
        assert counts == {"planned": 4, "resumed": 3, "ran": 1, "incomplete": 0}
        assert sorted(load_replies(out)) == sorted(call.key for call in calls)

    def test_records_never_shift_onto_an_earlier_chunk_that_finished_nothing(
        self, tmp_path: Path
    ) -> None:
        """SABOTAGE target: pairing a released chunk by counting releases rather than by its index.

        Chunks of one, the bug on the first: chunks two and three land whole and the first hands
        over nothing. Each reply must sit under its own key -- the first version of this loop filed
        prompt 2's reply under key 1, which is a mis-keyed record every count still adds up over.
        """
        calls = calls_for("scripted", 3)
        out = tmp_path / "replies.jsonl"
        factory = lambda model_id, effort, concurrency: ReverseStreamingBackend(  # noqa: E731
            lambda prompt: f"reply to {prompt}", fail_on={calls[0].prompt}
        )
        with pytest.raises(RuntimeError, match="scripted request bug"):
            run_live_calls(
                calls,
                out,
                concurrency=2,
                chunk_size=1,
                make_record=make_record,
                backend_factory=factory,
            )
        rows = load_replies(out)
        assert sorted(rows) == [calls[1].key, calls[2].key]
        for call in calls[1:]:
            assert rows[call.key]["reply"] == f"reply to {call.prompt}"

    def test_the_telemetry_keys_ride_on_every_record(self, tmp_path: Path) -> None:
        completion = BedrockCompletion(
            text="r",
            reasoning="",
            usage=TokenUsage(input_tokens=10, output_tokens=2, cache_read_input_tokens=8),
            stop_reason="end_turn",
            elapsed_seconds=1.5,
            first_event_seconds=0.5,
            attempts=1,
        )
        assert completion_telemetry(completion) == {
            "cache_read_input_tokens": 8,
            "cache_write_input_tokens": 0,
            "elapsed_seconds": 1.5,
            "first_event_seconds": 0.5,
            "attempts": 1,
        }
        out = tmp_path / "replies.jsonl"
        run_live_calls(
            calls_for("scripted", 1),
            out,
            concurrency=1,
            chunk_size=1,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ScriptedDetailedBackend(["r"]),
        )
        (row,) = load_replies(out).values()
        assert row["elapsed_seconds"] is None, "the scripted stub measures no latency, honestly"
        assert row["cache_read_input_tokens"] == 0


class TestLegsCanRunSideBySide:
    def test_parallel_legs_write_every_key_of_every_leg_and_the_file_still_parses(
        self, tmp_path: Path
    ) -> None:
        """Two models, two legs, one file: every line parses, every key lands once, counts add up."""
        calls = [*calls_for("model-a", 6), *calls_for("model-b", 6)]
        out = tmp_path / "replies.jsonl"
        counts = run_live_calls(
            calls,
            out,
            concurrency=2,
            chunk_size=2,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ScriptedDetailedBackend(
                lambda prompt: f"{model_id} says {prompt}"
            ),
            parallel_legs=True,
        )
        rows = load_replies(out)
        assert sorted(rows) == sorted(call.key for call in calls)
        assert all(rows[call.key]["reply"].startswith(call.model_id) for call in calls)
        assert counts == {"planned": 12, "resumed": 0, "ran": 12, "incomplete": 0}

    def test_a_leg_that_raises_does_not_take_the_other_legs_records_with_it(
        self, tmp_path: Path
    ) -> None:
        calls = [*calls_for("model-a", 4), *calls_for("model-b", 4)]
        out = tmp_path / "replies.jsonl"

        def factory(model_id: str, effort: str | None, concurrency: int) -> ScriptedDetailedBackend:
            if model_id == "model-a":
                return ReverseStreamingBackend(lambda p: f"reply to {p}", fail_on={calls[1].prompt})
            return ScriptedDetailedBackend(lambda p: f"{model_id} says {p}")

        with pytest.raises(RuntimeError, match="scripted request bug"):
            run_live_calls(
                calls,
                out,
                concurrency=2,
                chunk_size=2,
                make_record=make_record,
                backend_factory=factory,
                parallel_legs=True,
            )
        rows = load_replies(out)
        assert {key for key in rows if key.startswith("model-b")} == {c.key for c in calls[4:]}
        assert calls[1].key not in rows

    def test_sequential_legs_remain_the_default(self, tmp_path: Path) -> None:
        """The default keeps leg order on disk exactly as before: model-a's rows, then model-b's."""
        calls = [*calls_for("model-a", 2), *calls_for("model-b", 2)]
        out = tmp_path / "replies.jsonl"
        run_live_calls(
            calls,
            out,
            concurrency=1,
            chunk_size=1,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ScriptedDetailedBackend(["r"]),
        )
        keys = [json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines()]
        assert keys == [call.key for call in calls]


class TestOutOfOrderPersistenceIsOptIn:
    """``persist_out_of_order`` relaxes the within-leg order on disk; the default keeps it.

    The same reverse-yielding backend serves both sides, so the only difference between the two
    files is the flag. ``TestTheLegKeepsRequestOrderOnDisk`` above is the default's pin; the test
    here is its pair, and SABOTAGE of the default (the flag on unconditionally) turns that pin red
    while this test stays green, which is how the two were watched to fail together.
    """

    def test_under_the_flag_chunks_land_in_completion_order(self, tmp_path: Path) -> None:
        """Five calls in chunks of two, yielded last-first: the tail chunk is on disk before the head."""
        calls = calls_for("scripted", 5)
        out = tmp_path / "replies.jsonl"
        counts = run_live_calls(
            calls,
            out,
            concurrency=2,
            chunk_size=2,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ReverseStreamingBackend(
                lambda prompt: f"reply to {prompt}"
            ),
            persist_out_of_order=True,
        )
        keys = [json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines()]
        assert keys == [calls[4].key, calls[2].key, calls[3].key, calls[0].key, calls[1].key], (
            "chunk three (one call) whole first, then chunk two, then the head chunk"
        )
        assert load_replies(out) == {
            row["key"]: row for row in _sequential_reference(tmp_path, calls)
        }, "the same rows, keyed the same, whatever order they landed in"
        assert counts == {"planned": 5, "resumed": 0, "ran": 5, "incomplete": 0}

    def test_the_default_holds_the_head_chunk_in_front_under_the_same_backend(
        self, tmp_path: Path
    ) -> None:
        calls = calls_for("scripted", 5)
        out = tmp_path / "replies.jsonl"
        run_live_calls(
            calls,
            out,
            concurrency=2,
            chunk_size=2,
            make_record=make_record,
            backend_factory=lambda model_id, effort, concurrency: ReverseStreamingBackend(
                lambda prompt: f"reply to {prompt}"
            ),
        )
        keys = [json.loads(line)["key"] for line in out.read_text(encoding="utf-8").splitlines()]
        assert keys == [call.key for call in calls]
