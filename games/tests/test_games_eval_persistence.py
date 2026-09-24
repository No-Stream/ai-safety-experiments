"""Offline tests for the eval battery's persistence: per-record appends, resume-by-key, and pooling.

Three properties the 2026-09-02 hot-path work added to `games.evals` and `games.run_evals`, each
with the sabotage the repo's own rule asks for (a check you have never watched fail is not yet a
check):

*   **The summary is a pure function of the trace.** `rebuild_summary` reproduces the driver's file
    from the JSONL alone -- verified byte for byte against the banked track-record-v2 battery
    summary on S3 the day it landed, and pinned here against a synthetic trace -- so a cell that
    died after its last generate call is salvaged by re-deriving, never by re-running.
*   **A trace without a summary is resumed, not deleted.** Records are appended and flushed as they
    are parsed; a relaunch keeps every finished record byte for byte, generates only the planned
    identities that are missing, refuses a trace whose meta describes another cell, and writes a
    `resume` block naming every session and its commit.
*   **Pooled and serial submission file the same records.** The pooled path drains an engine in
    completion order and pairs every reply with its prompt at the drain point
    (`VLLMBackend.generate_streaming`: request id plus echoed prompt); a reply routed to the wrong
    prompt is refused there rather than filed. Pooling is a vLLM mechanism, so every
    other transport runs the serial call groups under either name and the meta says which ran.

Everything runs on CPU against scripted backends and a fake vLLM engine; no model, no GPU.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import zlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from games import evals, run_evals
from games.chunked_decode import stream_vllm_completions
from games.evals import (
    ADMISSION_FIFO,
    ADMISSION_LONGEST_FIRST,
    POOLED_ADMISSION_ORDER,
    RECORD_IDENTITY_FIELDS,
    RECORD_META,
    RECORD_TRAINING_FRAMES,
    SECTION_CAPABILITIES,
    SECTION_GAME_BEHAVIOR,
    SUBMISSION_POOLED,
    SUBMISSION_SERIAL,
    EvalConfig,
    PlannedRequest,
    inspect_trace,
    plan_battery,
    read_eval_records,
    rebuild_summary,
    record_identity,
    run_eval_battery,
    salvage_summary,
    summarise_battery_trace,
    summarise_trace,
)
from games.s3_sync import SyncOutcome
from reward_hacking.model_backend import MockBackend, VLLMBackend

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import AutoTokenizer

META: dict[str, Any] = {"arm": "unit-arm", "step": 5}
# Two games, one order, one draw each, plus a short arithmetic canary: three serial call groups
# (twin-pd/canonical, chicken/canonical, capabilities), so "die at the second call" leaves the first
# game's records on disk and two groups still to generate.
SECTIONS_UNDER_TEST = (SECTION_GAME_BEHAVIOR, SECTION_CAPABILITIES)
CONFIG = EvalConfig(
    games=("twin-pd", "chicken"),
    include_never_trained=False,
    capability_items=3,
    open_ended_samples=1,
    multiple_choice_samples=1,
)
CHAT_PREFIX = "[chat] "


def scripted_completion(prompt: str) -> str:
    """A completion that depends on the prompt and nothing else, so two runs can be compared byte for byte."""
    raw = prompt.removeprefix(CHAT_PREFIX)
    return f"reasoning</think>answer {zlib.crc32(raw.encode('utf-8'))}"


def scripted_backend() -> MockBackend:
    return MockBackend(responses=scripted_completion, model_id="scripted")


class SimulatedDeathError(RuntimeError):
    """The process died: raised by `DiesAtCall` at the call it was told to die on."""


class DiesAtCall:
    """A backend wrapper that dies -- raises -- when its Nth generate call is made.

    The mock-level stand-in for a spot reclaim or a killswitch mid-cell: whatever was appended and
    flushed before the call is on disk, and nothing from the call or after it. ``at_call=None``
    never dies and just counts, which is how a relaunch proves it generated only what was missing.
    """

    transport = "mock"

    def __init__(self, inner: MockBackend, *, at_call: int | None) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.at_call = at_call
        self.calls = 0
        self.prompts_seen = 0

    def generate(self, prompts: list[str]) -> list[str]:
        self.calls += 1
        if self.calls == self.at_call:
            raise SimulatedDeathError(f"died at generate call {self.calls}")
        self.prompts_seen += len(prompts)
        return self.inner.generate(prompts)


def count_prompts_the_driver_generates(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Wrap the driver's backend so every generate call's prompt count lands in the returned list."""
    seen: list[int] = []
    build = run_evals.backend_cli.backend_from_args

    def counting(*args: Any, **kwargs: Any) -> Any:
        backend = build(*args, **kwargs)
        inner_generate = backend.generate

        def generate(prompts: list[str]) -> list[str]:
            seen.append(len(prompts))
            return inner_generate(prompts)

        backend.generate = generate
        return backend

    monkeypatch.setattr(run_evals.backend_cli, "backend_from_args", counting)
    return seen


def trace_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def record_lines(path: Path) -> list[str]:
    """Every line after the meta record, verbatim."""
    return trace_lines(path)[1:]


def without_resume(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "resume"}


def run_serial(backend: Any, out_path: Path, *, resume: bool = False) -> dict[str, Any]:
    return run_eval_battery(
        backend,
        sections=SECTIONS_UNDER_TEST,
        out_path=out_path,
        meta=META,
        config=CONFIG,
        submission=SUBMISSION_SERIAL,
        resume=resume,
    )


class TestTheSummaryIsAPureFunctionOfTheTrace:
    def test_rebuild_summary_matches_an_independent_reduction_of_the_records(
        self, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "step-5.jsonl"
        returned = run_serial(scripted_backend(), out_path)
        rebuilt = rebuild_summary(out_path)
        assert rebuilt == returned
        records = read_eval_records(out_path)
        meta, body = records[0], records[1:]
        expected: dict[str, Any] = {"git_sha": meta["git_sha"], "arm": "unit-arm", "step": 5}
        for section in SECTIONS_UNDER_TEST:
            expected[section] = evals._summarise(  # pyright: ignore[reportPrivateUsage]
                section,
                [record for record in body if record["record"] == section],
                behaviour_records=[
                    record for record in body if record["record"] == SECTION_GAME_BEHAVIOR
                ],
            )
        expected["resume"] = {**meta["resume"], "records_generated": len(body)}
        assert rebuilt == expected
        assert json.dumps(rebuilt, indent=2) == json.dumps(expected, indent=2)

    def test_the_lifted_fields_come_off_the_meta_in_the_files_order(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        meta = {
            **META,
            "sampler_mode": "training-distribution",
            "sampling": {"stop": []},
            "engine_seed": 7,
        }
        run_eval_battery(
            scripted_backend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=out_path,
            meta=meta,
            config=CONFIG,
        )
        summary = rebuild_summary(out_path)
        assert list(summary)[:6] == [
            "git_sha",
            "arm",
            "step",
            "sampler_mode",
            "sampling",
            "engine_seed",
        ]
        assert summary["engine_seed"] == 7
        assert list(summary)[6:] == [*SECTIONS_UNDER_TEST, "resume"]

    def test_a_trace_of_another_shape_is_refused_rather_than_misdescribed(self) -> None:
        frames_meta = {"record": RECORD_META, "sections": [RECORD_TRAINING_FRAMES], "git_sha": "x"}
        with pytest.raises(ValueError, match=r"not a games\.run_evals battery"):
            summarise_battery_trace([frames_meta])
        battery_meta = {"record": RECORD_META, "sections": [SECTION_CAPABILITIES], "git_sha": "x"}
        stray = {"record": SECTION_GAME_BEHAVIOR, "game_id": "twin-pd", "parsed": True}
        with pytest.raises(ValueError, match="mixes cells"):
            summarise_battery_trace([battery_meta, stray])
        with pytest.raises(ValueError, match="starts with"):
            summarise_battery_trace([stray])

    def test_the_dispatching_reducer_hands_a_battery_trace_to_the_battery_reducer(
        self, tmp_path: Path
    ) -> None:
        """`summarise_trace` reads the kind off the meta; a battery trace reduces as before, byte for byte."""
        out_path = tmp_path / "step-5.jsonl"
        run_serial(scripted_backend(), out_path)
        records = read_eval_records(out_path)
        assert json.dumps(summarise_trace(records)) == json.dumps(summarise_battery_trace(records))
        stray = {"record": SECTION_GAME_BEHAVIOR, "game_id": "twin-pd", "parsed": True}
        with pytest.raises(ValueError, match="starts with"):
            summarise_trace([stray])


class TestATraceWithoutASummaryIsResumed:
    def test_dying_at_the_second_call_then_relaunching_yields_identical_records(
        self, tmp_path: Path
    ) -> None:
        """The kill-and-relaunch rehearsal on the mock backend.

        The uninterrupted run is the reference. The interrupted one dies at its second generate
        call, so the first game's records are on disk and nothing else; its relaunch must keep those
        bytes, generate exactly the rest, and say so in the meta.
        """
        reference = tmp_path / "reference" / "step-5.jsonl"
        run_serial(scripted_backend(), reference)
        reference_records = record_lines(reference)

        interrupted = tmp_path / "interrupted" / "step-5.jsonl"
        dying = DiesAtCall(scripted_backend(), at_call=2)
        with pytest.raises(SimulatedDeathError):
            run_serial(dying, interrupted)
        partial = record_lines(interrupted)
        assert 0 < len(partial) < len(reference_records)
        assert partial == reference_records[: len(partial)]
        partial_meta = read_eval_records(interrupted)[0]
        assert partial_meta["resume"]["n_sessions"] == 1

        relaunched = DiesAtCall(scripted_backend(), at_call=None)
        resumed_summary = run_serial(relaunched, interrupted, resume=True)
        # Only the missing prompts reach the backend: a relaunch that regenerated everything and
        # appended just the missing records would leave identical bytes on disk and pay twice.
        assert relaunched.prompts_seen == len(reference_records) - len(partial)
        assert record_lines(interrupted) == reference_records
        meta = read_eval_records(interrupted)[0]
        assert meta["resume"]["n_sessions"] == 2
        assert meta["resume"]["records_resumed"] == len(partial)
        assert meta["resume"]["records_dropped"] == 0
        assert [session["records_resumed"] for session in meta["resume"]["sessions"]] == [
            0,
            len(partial),
        ]
        assert all(session["git_sha"] for session in meta["resume"]["sessions"])
        assert [session["submission"] for session in meta["resume"]["sessions"]] == [
            SUBMISSION_SERIAL,
            SUBMISSION_SERIAL,
        ]
        assert resumed_summary["resume"]["records_generated"] == len(reference_records) - len(
            partial
        )
        assert without_resume(resumed_summary) == without_resume(rebuild_summary(reference))
        assert resumed_summary == rebuild_summary(interrupted)

    def test_a_torn_final_line_is_dropped_and_regenerated(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        run_serial(scripted_backend(), out_path)
        complete = trace_lines(out_path)
        torn = complete[-1][: len(complete[-1]) // 2]
        out_path.write_text("".join(complete[:-1]) + torn, encoding="utf-8")
        run_serial(scripted_backend(), out_path, resume=True)
        assert trace_lines(out_path)[1:] == complete[1:]
        meta = read_eval_records(out_path)[0]
        assert meta["resume"]["records_dropped"] == 1
        assert meta["resume"]["records_resumed"] == len(complete) - 2

    def test_a_trace_written_before_the_resume_block_existed_is_still_resumable(
        self, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        lines = trace_lines(out_path)
        legacy_meta = json.loads(lines[0])
        del legacy_meta["resume"]
        out_path.write_text(json.dumps(legacy_meta) + "\n" + "".join(lines[1:]), encoding="utf-8")
        run_serial(scripted_backend(), out_path, resume=True)
        meta = read_eval_records(out_path)[0]
        assert meta["resume"]["n_sessions"] == 2
        assert meta["resume"]["sessions"][0]["git_sha"] == legacy_meta["git_sha"]
        assert meta["resume"]["sessions"][0]["started_at"] == legacy_meta["written_at"]
        # Every cell before the block existed ran the serial call sequence; the reconstruction says so.
        assert meta["resume"]["sessions"][0]["submission"] == SUBMISSION_SERIAL

    @pytest.mark.parametrize(
        "other_config",
        [
            EvalConfig(
                games=("twin-pd", "chicken"),
                include_never_trained=False,
                capability_items=4,
                open_ended_samples=1,
                multiple_choice_samples=1,
            ),
            dataclasses.replace(CONFIG, survey_counterpart_variants=True),
            dataclasses.replace(CONFIG, survey_items=("self-prediction-twin-pd",)),
        ],
        ids=["different-games", "counterpart-variants-switched-on", "survey-items-filter"],
    )
    def test_a_trace_from_another_configuration_is_refused_and_left_intact(
        self, tmp_path: Path, other_config: EvalConfig
    ) -> None:
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        before = out_path.read_bytes()
        with pytest.raises(ValueError, match="refusing to resume") as refusal:
            run_eval_battery(
                scripted_backend(),
                sections=SECTIONS_UNDER_TEST,
                out_path=out_path,
                meta=META,
                config=other_config,
                submission=SUBMISSION_SERIAL,
                resume=True,
            )
        assert "eval_config" in str(refusal.value)
        assert out_path.read_bytes() == before
        with pytest.raises(ValueError, match="arm"):
            run_eval_battery(
                scripted_backend(),
                sections=SECTIONS_UNDER_TEST,
                out_path=out_path,
                meta={**META, "arm": "another-arm"},
                config=CONFIG,
                submission=SUBMISSION_SERIAL,
                resume=True,
            )
        assert out_path.read_bytes() == before

    def test_a_filtered_trace_refuses_resume_without_the_filter(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        filtered = dataclasses.replace(CONFIG, survey_items=("self-prediction-twin-pd",))
        with pytest.raises(SimulatedDeathError):
            run_eval_battery(
                DiesAtCall(scripted_backend(), at_call=2),
                sections=SECTIONS_UNDER_TEST,
                out_path=out_path,
                meta=META,
                config=filtered,
                submission=SUBMISSION_SERIAL,
            )
        before = out_path.read_bytes()
        with pytest.raises(ValueError, match="refusing to resume") as refusal:
            run_eval_battery(
                scripted_backend(),
                sections=SECTIONS_UNDER_TEST,
                out_path=out_path,
                meta=META,
                config=CONFIG,
                submission=SUBMISSION_SERIAL,
                resume=True,
            )
        assert "survey_items" in str(refusal.value)
        assert out_path.read_bytes() == before

    def test_the_item_data_paths_are_not_part_of_the_identity(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        moved = dataclasses.replace(CONFIG, dtbench_dir=tmp_path / "elsewhere")
        run_eval_battery(
            scripted_backend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=out_path,
            meta=META,
            config=moved,
            submission=SUBMISSION_SERIAL,
            resume=True,
        )
        assert read_eval_records(out_path)[0]["resume"]["n_sessions"] == 2

    def test_an_existing_trace_is_refused_unless_resume_is_asked_for(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        run_serial(scripted_backend(), out_path)
        before = out_path.read_bytes()
        with pytest.raises(FileExistsError, match="resume is off"):
            run_serial(scripted_backend(), out_path)
        assert out_path.read_bytes() == before

    def test_records_from_a_prompt_the_plan_never_renders_refuse_the_resume(
        self, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "step-5.jsonl"
        run_serial(scripted_backend(), out_path)
        lines = trace_lines(out_path)
        orphan = json.loads(lines[-1])
        orphan["item_index"] = 99
        out_path.write_text("".join(lines) + json.dumps(orphan) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="never renders"):
            run_serial(scripted_backend(), out_path, resume=True)
        duplicate = "".join(lines) + lines[-1]
        out_path.write_text(duplicate, encoding="utf-8")
        with pytest.raises(ValueError, match="twice"):
            run_serial(scripted_backend(), out_path, resume=True)

    def test_a_blank_line_anywhere_refuses_the_resume(self, tmp_path: Path) -> None:
        """The appender never writes one, so a blank line means something else assembled the file."""
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        lines = trace_lines(out_path)
        out_path.write_text(lines[0] + "\n" + "".join(lines[1:]), encoding="utf-8")
        before = out_path.read_bytes()
        # A phrase from the message, not a word: `match` searches the whole string, path included,
        # and this test's own name puts "blank" into the tmp path (the sabotage found exactly that).
        with pytest.raises(ValueError, match="is blank; this writer never emits one"):
            run_serial(scripted_backend(), out_path, resume=True)
        assert out_path.read_bytes() == before

    def test_the_append_hook_sees_the_cumulative_count_after_every_flush(
        self, tmp_path: Path
    ) -> None:
        seen: list[int] = []
        run_eval_battery(
            scripted_backend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=tmp_path / "step-5.jsonl",
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_SERIAL,
            on_records_written=seen.append,
        )
        assert len(seen) == 3
        assert seen == sorted(seen)
        assert seen[-1] == len(record_lines(tmp_path / "step-5.jsonl"))


class TestACompleteTraceIsSalvagedWithoutAModel:
    def test_salvage_writes_the_summary_and_records_the_session(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        summary = run_serial(scripted_backend(), out_path)
        plan = plan_battery(SECTIONS_UNDER_TEST, CONFIG)
        expected_meta = {
            **META,
            "sections": list(SECTIONS_UNDER_TEST),
            "eval_config": CONFIG.as_record(),
        }
        salvaged = salvage_summary(out_path, plan, expected_meta=expected_meta)
        assert without_resume(salvaged) == without_resume(summary)
        assert salvaged["resume"]["n_sessions"] == 2
        assert salvaged["resume"]["records_resumed"] == len(plan)
        assert salvaged["resume"]["records_generated"] == 0
        # A close-out generates nothing, so it has no submission mode to claim.
        assert [session["submission"] for session in salvaged["resume"]["sessions"]] == [
            SUBMISSION_SERIAL,
            None,
        ]

    def test_salvage_refuses_an_incomplete_trace_and_names_what_is_missing(
        self, tmp_path: Path
    ) -> None:
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        before = out_path.read_bytes()
        plan = plan_battery(SECTIONS_UNDER_TEST, CONFIG)
        expected_meta = {
            **META,
            "sections": list(SECTIONS_UNDER_TEST),
            "eval_config": CONFIG.as_record(),
        }
        with pytest.raises(ValueError, match="incomplete") as refusal:
            salvage_summary(out_path, plan, expected_meta=expected_meta)
        assert "capabilities" in str(refusal.value)
        assert out_path.read_bytes() == before

    def test_inspect_trace_reports_what_is_done_and_what_is_pending(self, tmp_path: Path) -> None:
        out_path = tmp_path / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_serial(DiesAtCall(scripted_backend(), at_call=2), out_path)
        plan = plan_battery(SECTIONS_UNDER_TEST, CONFIG)
        inspection = inspect_trace(
            out_path,
            plan,
            expected_meta={
                **META,
                "sections": list(SECTIONS_UNDER_TEST),
                "eval_config": CONFIG.as_record(),
            },
        )
        assert len(inspection.done) == len(record_lines(out_path))
        assert {request.section for request in inspection.pending(plan)} == set(SECTIONS_UNDER_TEST)
        assert {
            record_identity(json.loads(line)) for line in inspection.kept_lines
        } == inspection.done


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return CHAT_PREFIX + messages[0]["content"]


class FakeEngine:
    """A vLLM `LLMEngine` stand-in that finishes requests a few at a time, last-enqueued first.

    Completion order is deliberately not submission order, which is what makes the drain's pairing
    load-bearing. `swap_first_two` is the sabotage: the first step's two outputs come back carrying
    each other's prompt text, as a broken engine that routed replies to the wrong request would.
    `dies_at_step` is the process dying mid-drain: that step raises instead of returning, so
    whatever earlier steps yielded is on disk and nothing after it.
    """

    def __init__(
        self, *, swap_first_two: bool = False, per_step: int = 3, dies_at_step: int | None = None
    ) -> None:
        self.queue: list[tuple[str, str, list[int]]] = []
        self.added: list[tuple[str, str, Any, object]] = []
        self.aborted: list[list[str]] = []
        self.swap_first_two = swap_first_two
        self.per_step = per_step
        self.dies_at_step = dies_at_step
        self.steps = 0

    def add_request(
        self,
        request_id: str,
        engine_input: dict[str, Any],
        params: Any,
        *,
        lora_request: object = None,
    ) -> str:
        """Queue one rendered request under the caller's id and hand back a randomised internal id, as vLLM does.

        Only a rendered engine input is accepted: a raw string is the path vLLM 0.27.1 deprecates,
        and this fake refuses it so the drain cannot drift back onto it unnoticed.
        """
        if engine_input.get("type") != "text" or "prompt_token_ids" not in engine_input:
            raise TypeError(f"add_request was handed an unrendered prompt: {engine_input!r}")
        prompt = str(engine_input["prompt"])
        self.added.append((request_id, prompt, params, lora_request))
        self.queue.append((request_id, prompt, list(engine_input["prompt_token_ids"])))
        return f"{request_id}-deadbeef"

    def has_unfinished_requests(self) -> bool:
        return bool(self.queue)

    def abort_request(self, request_ids: list[str]) -> None:
        """`LLMEngine.abort_request` by external id: drops what is still queued, ignores what is not."""
        self.aborted.append(list(request_ids))
        self.queue = [entry for entry in self.queue if entry[0] not in request_ids]

    def step(self) -> list[Any]:
        if self.steps == self.dies_at_step:
            raise SimulatedDeathError(f"died at engine step {self.steps}")
        batch = [self.queue.pop() for _ in range(min(self.per_step, len(self.queue)))]
        outputs = [
            SimpleNamespace(
                request_id=request_id,
                prompt=chat,
                prompt_token_ids=prompt_token_ids,
                finished=True,
                outputs=[
                    SimpleNamespace(
                        text=scripted_completion(chat),
                        token_ids=list(scripted_completion(chat).encode("utf-8")),
                        finish_reason="stop",
                        stop_reason=None,
                    )
                ],
            )
            for request_id, chat, prompt_token_ids in batch
        ]
        if self.swap_first_two and self.steps == 0 and len(outputs) >= 2:
            outputs[0].prompt, outputs[1].prompt = outputs[1].prompt, outputs[0].prompt
        self.steps += 1
        return outputs


class FakeSamplingParams:
    """The two things the drain touches on `SamplingParams`: a clone, and its output kind."""

    def __init__(self) -> None:
        self.output_kind: object = "cumulative"

    def clone(self) -> FakeSamplingParams:
        return FakeSamplingParams()


class FakeRenderer:
    """The one renderer call the drain makes: `render_cmpl` over text prompts, returning engine inputs.

    Mirrors what vLLM's renderer hands back for a text prompt -- the tokens plus the text itself
    under ``prompt``, which is what `RequestOutput.prompt` later echoes.
    """

    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []

    def render_cmpl(
        self, prompts: list[dict[str, Any]], tok_params: object = None
    ) -> list[dict[str, Any]]:
        del tok_params
        self.calls.append(prompts)
        return [
            {
                "type": "text",
                "prompt": prompt["prompt"],
                "prompt_token_ids": list(prompt["prompt"].encode("utf-8")),
            }
            for prompt in prompts
        ]


class FakeLLM:
    def __init__(self, engine: FakeEngine) -> None:
        self.llm_engine = engine
        self.renderer = FakeRenderer()


class FakeVLLMBackend(VLLMBackend):
    """A real `VLLMBackend` around a fake engine, so the pooled path runs the method that ships.

    `__init__` is bypassed rather than mocked because the only thing it does that these tests are not
    about is bring up a GPU engine; `generate_streaming` and everything the battery reads off the
    backend run exactly as they do in production. Nothing else is stubbed: a test that made the
    battery reach the batch path (`generate_tokenized`) would fail on the fake engine's missing
    `generate`, which is the intended tripwire, since pooled mode must never call it.
    """

    def __init__(self, **engine_flags: Any) -> None:
        self.model_id = "fake-vllm"
        self.thinking = False
        self._tokenizer = cast("AutoTokenizer", FakeTokenizer())
        self._sampling_params = FakeSamplingParams()
        self._lora_request = None
        self._llm = FakeLLM(FakeEngine(**engine_flags))


class TestPooledAndSerialFileTheSameRecords:
    def test_the_pooled_drain_yields_every_prompt_once_in_completion_order(self) -> None:
        backend = FakeVLLMBackend()
        prompts = [f"prompt {index}" for index in range(7)]
        drained = list(stream_vllm_completions(backend, prompts))
        assert sorted(index for index, _ in drained) == list(range(7))
        assert [index for index, _ in drained] != list(range(7))
        assert all(
            completion == scripted_completion(prompts[index]) for index, completion in drained
        )
        added = backend._llm.llm_engine.added
        assert [chat for _, chat, _, _ in added] == [CHAT_PREFIX + prompt for prompt in prompts]
        assert [request_id for request_id, _, _, _ in added] == [f"pooled-{i}" for i in range(7)]
        # Rendered through the engine's renderer in one call, as text prompts, before add_request.
        assert backend._llm.renderer.calls == [
            [{"prompt": CHAT_PREFIX + prompt} for prompt in prompts]
        ]
        # Final-only output, set on a clone: the backend's own params are untouched.
        final_only = importlib.import_module("vllm.sampling_params").RequestOutputKind.FINAL_ONLY
        assert all(params is not backend._sampling_params for *_, params, _ in added)
        assert all(params.output_kind is final_only for *_, params, _ in added)
        assert backend._sampling_params.output_kind == "cumulative"

    def test_pooled_and_serial_produce_the_same_record_set(self, tmp_path: Path) -> None:
        serial_path = tmp_path / "serial" / "step-5.jsonl"
        run_serial(scripted_backend(), serial_path)
        pooled_path = tmp_path / "pooled" / "step-5.jsonl"
        pooled_summary = run_eval_battery(
            FakeVLLMBackend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=pooled_path,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
        )
        serial_records = record_lines(serial_path)
        pooled_records = record_lines(pooled_path)
        assert len(pooled_records) == len(serial_records)
        assert pooled_records != serial_records
        assert sorted(pooled_records) == sorted(serial_records)
        for section in SECTIONS_UNDER_TEST:
            assert pooled_summary[section] == rebuild_summary(serial_path)[section]
        assert [s["submission"] for s in pooled_summary["resume"]["sessions"]] == [
            SUBMISSION_POOLED
        ]

    def test_dying_mid_drain_then_relaunching_pooled_yields_the_same_record_set(
        self, tmp_path: Path
    ) -> None:
        """The pooled kill-and-relaunch rehearsal, offline: the engine dies at its third step."""
        reference = tmp_path / "reference" / "step-5.jsonl"
        run_eval_battery(
            FakeVLLMBackend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=reference,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
        )
        reference_records = record_lines(reference)

        interrupted = tmp_path / "interrupted" / "step-5.jsonl"
        with pytest.raises(SimulatedDeathError):
            run_eval_battery(
                FakeVLLMBackend(dies_at_step=2),
                sections=SECTIONS_UNDER_TEST,
                out_path=interrupted,
                meta=META,
                config=CONFIG,
                submission=SUBMISSION_POOLED,
            )
        partial = record_lines(interrupted)
        assert 0 < len(partial) < len(reference_records)

        relaunched = FakeVLLMBackend()
        summary = run_eval_battery(
            relaunched,
            sections=SECTIONS_UNDER_TEST,
            out_path=interrupted,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
            resume=True,
        )
        # Exactly the missing prompts were enqueued; the finished ones stayed on disk untouched.
        assert len(relaunched._llm.llm_engine.added) == len(reference_records) - len(partial)
        assert record_lines(interrupted)[: len(partial)] == partial
        assert sorted(record_lines(interrupted)) == sorted(reference_records)
        assert summary["resume"]["n_sessions"] == 2
        assert summary["resume"]["records_resumed"] == len(partial)
        assert summary["resume"]["records_generated"] == len(reference_records) - len(partial)
        assert [s["submission"] for s in summary["resume"]["sessions"]] == [
            SUBMISSION_POOLED,
            SUBMISSION_POOLED,
        ]
        assert without_resume(summary) == without_resume(rebuild_summary(reference))

    def test_a_non_vllm_backend_runs_the_serial_call_groups_under_either_submission(
        self, tmp_path: Path
    ) -> None:
        """Pooling is an engine mechanism; a transport with no engine to drain keeps per-group appends.

        One generate over the whole cell would lose every record on a death where the serial call
        groups lose one, and the meta must say what ran rather than what was asked.
        """
        calls: list[int] = []

        class CountingMock:
            transport = "mock"
            model_id = "counting"

            def generate(self, prompts: list[str]) -> list[str]:
                calls.append(len(prompts))
                return [scripted_completion(prompt) for prompt in prompts]

        out_path = tmp_path / "step-5.jsonl"
        summary = run_eval_battery(
            CountingMock(),
            sections=SECTIONS_UNDER_TEST,
            out_path=out_path,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
        )
        plan = plan_battery(SECTIONS_UNDER_TEST, CONFIG)
        assert len(calls) == len({request.call_group for request in plan}) == 3
        assert sum(calls) == len(plan)
        assert [s["submission"] for s in summary["resume"]["sessions"]] == [SUBMISSION_SERIAL]

    def test_the_drain_refuses_a_backend_with_no_engine_to_drain(self) -> None:
        with pytest.raises(TypeError, match="was handed a 'mock' backend"):
            list(stream_vllm_completions(cast("VLLMBackend", scripted_backend()), ["a"]))

    def test_the_drain_refuses_a_reply_routed_to_the_wrong_prompt(self) -> None:
        backend = FakeVLLMBackend(swap_first_two=True)
        with pytest.raises(RuntimeError, match="paired request"):
            list(stream_vllm_completions(backend, ["a", "b", "c", "d"]))
        # The refusal is an exit, and the exit aborts what is still queued behind the wrapper too.
        assert backend._llm.llm_engine.aborted == [["pooled-0", "pooled-1", "pooled-2"]]
        assert backend._llm.llm_engine.queue == []

    def test_the_drain_refuses_a_reply_carrying_other_than_one_sequence(self) -> None:
        backend = FakeVLLMBackend()
        engine = backend._llm.llm_engine
        original_step = engine.step

        def step_with_a_twin() -> list[Any]:
            outputs = original_step()
            outputs[0].outputs = outputs[0].outputs * 2
            return outputs

        engine.step = step_with_a_twin  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="2 sequences"):
            list(stream_vllm_completions(backend, ["a", "b"]))

    def test_the_drain_refuses_an_id_it_never_submitted_and_a_missing_reply(self) -> None:
        backend = FakeVLLMBackend()
        engine = backend._llm.llm_engine
        original_step = engine.step

        def step_with_a_stranger() -> list[Any]:
            outputs = original_step()
            outputs[0].request_id = "999"
            return outputs

        engine.step = step_with_a_stranger  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="never made"):
            list(stream_vllm_completions(backend, ["a", "b"]))

        silent = FakeVLLMBackend()
        silent_engine = silent._llm.llm_engine

        def step_dropping_one() -> list[Any]:
            outputs = original_step.__func__(silent_engine)  # type: ignore[attr-defined]
            return outputs[1:]

        silent_engine.step = step_dropping_one  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="never came back"):
            list(stream_vllm_completions(silent, ["a", "b", "c"]))

    def test_a_plan_whose_parser_disagrees_with_its_declared_identity_is_refused(self) -> None:
        """The invariant resume-by-key stands on: declared identity == the identity the record carries.

        Not a pairing check -- the parser never sees which prompt a completion answered, so a
        misrouted reply parses cleanly under the right identity (`VLLMBackend.generate_streaming`
        on the drain and the serial path's strict zip are what catch that). What this refuses is a
        planner bug
        that bound a parser to one row's context and the identity to another's, which would make a
        relaunch skip the wrong prompts with every count still adding up.
        """
        plan = plan_battery((SECTION_CAPABILITIES,), CONFIG)
        request = plan[0]
        inconsistent = PlannedRequest(
            section=request.section,
            identity=request.identity,
            prompt=request.prompt,
            call_group=request.call_group,
            parse=plan[1].parse,
        )
        assert request.record("reasoning</think>4")["item_index"] == 0
        with pytest.raises(RuntimeError, match="disagrees with itself"):
            inconsistent.record("reasoning</think>4")

    def test_a_plan_rendering_one_identity_twice_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        planner = evals.SECTION_PLANNERS[SECTION_CAPABILITIES]
        monkeypatch.setitem(
            evals.SECTION_PLANNERS,
            SECTION_CAPABILITIES,
            lambda config: planner(config) + planner(config),
        )
        with pytest.raises(RuntimeError, match="more than once"):
            plan_battery((SECTION_CAPABILITIES,), CONFIG)

    def test_the_default_submission_is_pooled(self) -> None:
        assert run_evals._parse_args(["--model", "m", "--arm", "a"]).submission == SUBMISSION_POOLED


def section_of_each_admitted_prompt(backend: FakeVLLMBackend) -> list[str]:
    """The section of every prompt the fake engine was handed, in the order it was handed them."""
    section_by_chat = {
        CHAT_PREFIX + request.prompt: request.section
        for request in plan_battery(SECTIONS_UNDER_TEST, CONFIG)
    }
    return [section_by_chat[chat] for _, chat, _, _ in backend._llm.llm_engine.added]


class TestPooledAdmissionOrder:
    """`longest-first` reorders the engine's queue and nothing else; `fifo` stays selectable.

    Probe P-E4 (2026-09-03): the pooled battery admitted its 128 self-report prompts last, 81 of
    them ran to the 65,536-token cap, and they ran nearly alone for the final ~50 of 221 minutes.
    The fix is an admission order, which has to be a pure permutation of the pending requests --
    same record set, same identities, same per-section summaries -- with the pairing check at the
    drain still armed under it, and the order recorded per session so the trace says how it was
    queued.
    """

    def _pooled(
        self, tmp_path: Path, name: str, *, admission: str, **engine_flags: Any
    ) -> tuple[FakeVLLMBackend, Path, dict[str, Any]]:
        backend = FakeVLLMBackend(**engine_flags)
        path = tmp_path / name / "step-5.jsonl"
        summary = run_eval_battery(
            backend,
            sections=SECTIONS_UNDER_TEST,
            out_path=path,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
            admission=admission,
        )
        return backend, path, summary

    def test_longest_first_and_fifo_file_the_same_records_from_different_queues(
        self, tmp_path: Path
    ) -> None:
        fifo, fifo_path, fifo_summary = self._pooled(tmp_path, "fifo", admission=ADMISSION_FIFO)
        longest, longest_path, longest_summary = self._pooled(
            tmp_path, "longest", admission=ADMISSION_LONGEST_FIRST
        )
        plan = plan_battery(SECTIONS_UNDER_TEST, CONFIG)
        fifo_records, longest_records = record_lines(fifo_path), record_lines(longest_path)
        # The record set and the identity set are the plan's under both orders.
        assert sorted(fifo_records) == sorted(longest_records)
        assert len(longest_records) == len(plan)
        assert {record_identity(json.loads(line)) for line in longest_records} == {
            request.identity for request in plan
        }
        for section in SECTIONS_UNDER_TEST:
            assert fifo_summary[section] == longest_summary[section]
        # The queues differ: fifo is plan order (behaviour first), longest-first puts the
        # capabilities canary ahead of it, keeping plan order inside each section.
        fifo_sections = section_of_each_admitted_prompt(fifo)
        longest_sections = section_of_each_admitted_prompt(longest)
        assert fifo_sections == [request.section for request in plan]
        assert fifo_sections[0] == SECTION_GAME_BEHAVIOR
        assert longest_sections == sorted(fifo_sections, key=POOLED_ADMISSION_ORDER.index)
        assert longest_sections[0] == SECTION_CAPABILITIES
        assert longest_sections != fifo_sections
        behaviour_chats = [
            chat
            for (_, chat, _, _), section in zip(
                longest._llm.llm_engine.added, longest_sections, strict=True
            )
            if section == SECTION_GAME_BEHAVIOR
        ]
        assert behaviour_chats == [
            chat
            for (_, chat, _, _), section in zip(
                fifo._llm.llm_engine.added, fifo_sections, strict=True
            )
            if section == SECTION_GAME_BEHAVIOR
        ]
        # Each trace says how it was queued.
        assert [s["admission"] for s in fifo_summary["resume"]["sessions"]] == [ADMISSION_FIFO]
        assert [s["admission"] for s in longest_summary["resume"]["sessions"]] == [
            ADMISSION_LONGEST_FIRST
        ]

    def test_the_default_admission_is_longest_first(self, tmp_path: Path) -> None:
        backend = FakeVLLMBackend()
        summary = run_eval_battery(
            backend,
            sections=SECTIONS_UNDER_TEST,
            out_path=tmp_path / "step-5.jsonl",
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
        )
        assert [s["admission"] for s in summary["resume"]["sessions"]] == [ADMISSION_LONGEST_FIRST]
        assert section_of_each_admitted_prompt(backend)[0] == SECTION_CAPABILITIES
        assert run_evals._parse_args(["--model", "m", "--arm", "a"]).admission == (
            ADMISSION_LONGEST_FIRST
        )
        assert (
            run_evals._parse_args(["--model", "m", "--arm", "a", "--admission", "fifo"]).admission
            == ADMISSION_FIFO
        )

    def test_a_session_that_pooled_nothing_records_no_admission(self, tmp_path: Path) -> None:
        """Serial call groups run in plan order whatever was asked; the meta must not claim a queue order."""
        serial = run_serial(scripted_backend(), tmp_path / "serial" / "step-5.jsonl")
        assert [s["admission"] for s in serial["resume"]["sessions"]] == [None]
        fallen_back = run_eval_battery(
            scripted_backend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=tmp_path / "mock-pooled" / "step-5.jsonl",
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
            admission=ADMISSION_LONGEST_FIRST,
        )
        assert [(s["submission"], s["admission"]) for s in fallen_back["resume"]["sessions"]] == [
            (SUBMISSION_SERIAL, None)
        ]

    def test_a_session_recorded_before_the_admission_field_existed_stays_without_it(
        self, tmp_path: Path
    ) -> None:
        """A pooled trace from the 2026-09-02 code -- `submission` recorded, no `admission` -- resumes as written.

        The shape a box running that code writes today. The relaunch appends its own session with
        the order it queued and leaves the earlier one without the field rather than claiming an
        order nobody recorded; the summary rebuilds over both.
        """
        with pytest.raises(SimulatedDeathError):
            self._pooled(tmp_path, "legacy", admission=ADMISSION_LONGEST_FIRST, dies_at_step=2)
        path = tmp_path / "legacy" / "step-5.jsonl"
        lines = trace_lines(path)
        legacy_meta = json.loads(lines[0])
        [first_session] = legacy_meta["resume"]["sessions"]
        del first_session["admission"]
        path.write_text(json.dumps(legacy_meta) + "\n" + "".join(lines[1:]), encoding="utf-8")
        partial = record_lines(path)
        assert partial

        summary = run_eval_battery(
            FakeVLLMBackend(),
            sections=SECTIONS_UNDER_TEST,
            out_path=path,
            meta=META,
            config=CONFIG,
            submission=SUBMISSION_POOLED,
            resume=True,
        )
        sessions = read_eval_records(path)[0]["resume"]["sessions"]
        assert ["admission" in session for session in sessions] == [False, True]
        assert [session["submission"] for session in sessions] == [
            SUBMISSION_POOLED,
            SUBMISSION_POOLED,
        ]
        assert sessions[1]["admission"] == ADMISSION_LONGEST_FIRST
        assert summary["resume"]["sessions"] == sessions
        assert summary["resume"]["records_resumed"] == len(partial)
        assert record_lines(path)[: len(partial)] == partial
        assert summary == rebuild_summary(path)
        _, reference_path, _ = self._pooled(
            tmp_path, "reference", admission=ADMISSION_LONGEST_FIRST
        )
        assert sorted(record_lines(path)) == sorted(record_lines(reference_path))

    def test_a_misrouted_reply_is_still_refused_at_the_drain_under_longest_first(
        self, tmp_path: Path
    ) -> None:
        """Reordering the queue must not loosen the pairing: swapped replies are refused before anything is filed."""
        with pytest.raises(RuntimeError, match="paired request"):
            self._pooled(
                tmp_path, "swapped", admission=ADMISSION_LONGEST_FIRST, swap_first_two=True
            )
        assert record_lines(tmp_path / "swapped" / "step-5.jsonl") == []

    def test_an_unknown_admission_is_refused_before_anything_runs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown admission"):
            self._pooled(tmp_path, "unknown", admission="shortest-first")
        assert not (tmp_path / "unknown" / "step-5.jsonl").exists()

    def test_every_record_kind_is_ranked_exactly_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kind missing from the order would fall to the back of the queue, which is where the tail came from."""
        assert sorted(POOLED_ADMISSION_ORDER) == sorted(RECORD_IDENTITY_FIELDS)
        evals._assert_admission_order_covers_every_record_kind()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(evals, "POOLED_ADMISSION_ORDER", POOLED_ADMISSION_ORDER[1:])
        with pytest.raises(RuntimeError, match="missing"):
            evals._assert_admission_order_covers_every_record_kind()  # pyright: ignore[reportPrivateUsage]
        monkeypatch.setattr(
            evals, "POOLED_ADMISSION_ORDER", (*POOLED_ADMISSION_ORDER, POOLED_ADMISSION_ORDER[0])
        )
        with pytest.raises(RuntimeError, match="repeated"):
            evals._assert_admission_order_covers_every_record_kind()  # pyright: ignore[reportPrivateUsage]


def cli(*extra: str, out_dir: Path) -> list[str]:
    """The offline battery flags: mock backend, two games, tiny counts, two fast sections."""
    return [
        "--model",
        "fake-base",
        "--arm",
        "plumbing-arm",
        "--backend",
        "mock",
        "--out-dir",
        str(out_dir),
        "--games",
        "twin-pd,chicken",
        "--no-include-never-trained",
        "--capability-items",
        "3",
        "--open-ended-samples",
        "1",
        "--sections",
        "game-behavior,capabilities",
        "--no-report",
        *extra,
    ]


def truncate_records(trace: Path, keep: int) -> list[str]:
    """Drop every record line after the first `keep`, as a mid-run death would; return the full lines."""
    lines = trace_lines(trace)
    trace.write_text("".join(lines[: 1 + keep]), encoding="utf-8")
    return lines


def refuse_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_engine(*_args: object, **_kwargs: object) -> object:
        pytest.fail("a backend was built for a cell that needed no generation")

    monkeypatch.setattr(run_evals.backend_cli, "backend_from_args", no_engine)


class TestTheDriverResumesInsteadOfDeleting:
    def test_relaunching_the_same_command_continues_a_partial_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        full_lines = truncate_records(trace, keep=4)
        summary.unlink()
        prompts_generated = count_prompts_the_driver_generates(monkeypatch)
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        assert sum(prompts_generated) == len(full_lines) - 1 - 4
        assert trace_lines(trace)[1:] == full_lines[1:]
        meta = read_eval_records(trace)[0]
        assert meta["resume"]["n_sessions"] == 2
        assert meta["resume"]["records_resumed"] == 4
        written = json.loads(summary.read_text(encoding="utf-8"))
        assert written == rebuild_summary(trace)
        assert written["resume"]["records_generated"] == len(full_lines) - 1 - 4

    def test_a_torn_summary_is_rebuilt_from_the_trace_without_an_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The summary is the completion marker, so a torn one must not read as complete."""
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        whole = summary.read_text(encoding="utf-8")
        summary.write_text(whole[: len(whole) // 2], encoding="utf-8")
        refuse_engine(monkeypatch)
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        assert "tore it" in caplog.text
        rebuilt = json.loads(summary.read_text(encoding="utf-8"))
        assert rebuilt == rebuild_summary(trace)
        assert without_resume(rebuilt) == without_resume(json.loads(whole))
        assert rebuilt["resume"]["records_generated"] == 0
        assert not summary.with_name(summary.name + ".tmp").exists()

    def test_a_summary_with_no_trace_beside_it_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        (out_dir / "step-0.jsonl").unlink()
        refuse_engine(monkeypatch)
        with pytest.raises(FileExistsError, match="no trace"):
            run_evals.main(cli(out_dir=out_dir))
        # Under --sync-dest a complete cell is skipped rather than refused; an orphan summary is not
        # a complete cell and stays a refusal there too.
        monkeypatch.setattr(
            run_evals,
            "restore_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        with pytest.raises(FileExistsError, match="no trace"):
            run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir))

    def test_a_complete_trace_without_a_summary_is_summarised_without_an_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        original = json.loads(summary.read_text(encoding="utf-8"))
        summary.unlink()
        refuse_engine(monkeypatch)
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        salvaged = json.loads(summary.read_text(encoding="utf-8"))
        assert without_resume(salvaged) == without_resume(original)
        assert salvaged == rebuild_summary(trace)
        assert salvaged["resume"]["n_sessions"] == 2
        assert salvaged["resume"]["records_generated"] == 0

    def test_summarise_only_writes_the_summary_and_loads_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        original = json.loads(summary.read_text(encoding="utf-8"))
        summary.unlink()
        refuse_engine(monkeypatch)
        assert run_evals.main(cli("--summarise-only", out_dir=out_dir)) == 0
        assert without_resume(json.loads(summary.read_text(encoding="utf-8"))) == without_resume(
            original
        )
        assert json.loads(summary.read_text(encoding="utf-8")) == rebuild_summary(trace)

    def test_summarise_only_refuses_an_incomplete_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        truncate_records(trace, keep=2)
        summary.unlink()
        refuse_engine(monkeypatch)
        with pytest.raises(ValueError, match="incomplete"):
            run_evals.main(cli("--summarise-only", out_dir=out_dir))
        assert not summary.exists()

    def test_summarise_only_refuses_a_complete_cell_and_a_missing_trace(
        self, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        with pytest.raises(FileExistsError, match="complete"):
            run_evals.main(cli("--summarise-only", out_dir=out_dir))
        with pytest.raises(FileNotFoundError, match="summarise-only"):
            run_evals.main(cli("--summarise-only", out_dir=tmp_path / "nowhere"))

    def test_a_complete_cell_is_refused_before_anything_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        before = (out_dir / "step-0.jsonl").read_bytes()
        refuse_engine(monkeypatch)
        with pytest.raises(FileExistsError, match="complete eval trace"):
            run_evals.main(cli(out_dir=out_dir))
        assert (out_dir / "step-0.jsonl").read_bytes() == before

    def test_a_partial_trace_of_another_cell_is_refused_before_anything_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        out_dir = tmp_path / "out"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        truncate_records(out_dir / "step-0.jsonl", keep=2)
        (out_dir / "step-0.summary.json").unlink()
        refuse_engine(monkeypatch)
        with pytest.raises(ValueError, match="refusing to resume"):
            run_evals.main(cli("--capability-items", "4", out_dir=out_dir))

    def test_another_arms_partial_trace_is_refused_by_the_cheap_peek(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The up-front pass compares arm, thinking, base model and sections, not just the step.

        Otherwise another arm's trace at a later step passes the peek and is refused only when its
        turn comes, after the earlier steps have paid their GPU time.
        """
        out_dir = tmp_path / "out"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        truncate_records(out_dir / "step-0.jsonl", keep=2)
        (out_dir / "step-0.summary.json").unlink()
        refuse_engine(monkeypatch)
        # No tokenizer either: the peek runs before the chat template is read.
        monkeypatch.setattr(
            run_evals,
            "_resolve_template_facts",
            lambda *_args, **_kwargs: pytest.fail("the template was read for a refused cell"),
        )
        argv = cli(out_dir=out_dir)
        argv[argv.index("plumbing-arm")] = "another-arm"
        with pytest.raises(FileExistsError, match="another cell") as refusal:
            run_evals.main(argv)
        assert "arm='plumbing-arm'" in str(refusal.value)

    def test_the_serial_submission_is_selectable_from_the_cli(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "out"
        assert run_evals.main(cli("--submission", "serial", out_dir=out_dir)) == 0
        meta = read_eval_records(out_dir / "step-0.jsonl")[0]
        assert meta["resume"]["n_sessions"] == 1
        assert meta["resume"]["sessions"][0]["submission"] == SUBMISSION_SERIAL

    def test_a_relaunch_finding_every_cell_complete_under_sync_reads_no_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing left to run means nothing to measure the chat template for: no tokenizer, no engine, no rewrite."""
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        assert run_evals.main(cli(out_dir=out_dir)) == 0
        before = trace.read_bytes()
        uploads: list[tuple[Path, str]] = []

        def record_upload(local_dir: Path, s3_dest: str) -> SyncOutcome:
            uploads.append((local_dir, s3_dest))
            return SyncOutcome(command=("aws",), returncode=0)

        monkeypatch.setattr(
            run_evals,
            "restore_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        monkeypatch.setattr(run_evals, "sync_directory", record_upload)
        refuse_engine(monkeypatch)
        monkeypatch.setattr(
            run_evals,
            "_resolve_template_facts",
            lambda *_args, **_kwargs: pytest.fail("the template was read with no cell left to run"),
        )
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        assert trace.read_bytes() == before
        assert uploads == [(out_dir, "s3://bucket/prefix/")]  # the run-finished upload only


class TestTheIntervalSync:
    def _sync(
        self, tmp_path: Path, interval: float, clock_values: list[float]
    ) -> tuple[run_evals.IntervalSync, list[str]]:
        calls: list[str] = []
        ticks = iter(clock_values)

        def sync(local_dir: Path, s3_dest: str) -> SyncOutcome:
            calls.append(f"{local_dir.name}->{s3_dest}")
            return SyncOutcome(command=("aws",), returncode=0)

        return run_evals.IntervalSync(
            tmp_path, "s3://bucket/prefix/", interval, clock=lambda: next(ticks), sync=sync
        ), calls

    def test_it_syncs_on_the_first_batch_then_once_per_interval_then_on_demand(
        self, tmp_path: Path
    ) -> None:
        # Each maybe_sync reads the clock once; a sync reads it a second time to stamp itself.
        sync, calls = self._sync(tmp_path, 600.0, [0.0, 0.0, 100.0, 500.0, 700.0, 700.0, 700.0])
        sync.maybe_sync(1)
        sync.maybe_sync(2)
        sync.maybe_sync(3)
        sync.maybe_sync(4)
        sync.sync_now(reason="done")
        assert calls == [f"{tmp_path.name}->s3://bucket/prefix/"] * 3

    def test_a_failed_sync_is_logged_and_does_not_raise(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def failing(local_dir: Path, s3_dest: str) -> SyncOutcome:
            del local_dir, s3_dest
            return SyncOutcome(command=("aws",), returncode=1)

        sync = run_evals.IntervalSync(tmp_path, "s3://bucket/prefix/", 60.0, sync=failing)
        sync.sync_now(reason="test")
        assert "failed" in caplog.text

    def test_a_non_positive_interval_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            run_evals.IntervalSync(tmp_path, "s3://bucket/prefix/", 0.0)

    def test_the_driver_restores_before_checking_and_syncs_after_the_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        out_dir = tmp_path / "out"

        def restore(s3_dest: str, local_dir: Path) -> SyncOutcome:
            events.append(
                f"restore {s3_dest} -> {local_dir.name} summary={(local_dir / 'step-0.summary.json').exists()}"
            )
            return SyncOutcome(command=("aws",), returncode=0)

        def sync(local_dir: Path, s3_dest: str) -> SyncOutcome:
            events.append(
                f"sync {local_dir.name} -> {s3_dest} summary={(local_dir / 'step-0.summary.json').exists()}"
            )
            return SyncOutcome(command=("aws",), returncode=0)

        monkeypatch.setattr(run_evals, "restore_directory", restore)
        monkeypatch.setattr(run_evals, "sync_directory", sync)
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        staging = "out" + run_evals.RESTORE_STAGING_SUFFIX
        assert events[0] == f"restore s3://bucket/prefix/ -> {staging} summary=False"
        assert events[1] == "sync out -> s3://bucket/prefix/ summary=False"
        assert events[-1] == "sync out -> s3://bucket/prefix/ summary=True"
        assert not (tmp_path / staging).exists()

    def test_the_restore_adds_missing_files_and_never_replaces_a_local_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same-box relaunch: the local trace is newer than the last upload and must win.

        `aws s3 sync` would replace it (sizes differ), rolling the cell back by up to one interval
        and regenerating records that were already paid for. The fake restore below plays the S3
        prefix: the trace as of the last upload (shorter, no summary) plus a file the local
        directory lacks.
        """
        out_dir = tmp_path / "out"
        trace = out_dir / "step-0.jsonl"
        summary = out_dir / "step-0.summary.json"
        monkeypatch.setattr(
            run_evals,
            "sync_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        monkeypatch.setattr(
            run_evals,
            "restore_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        full = trace.read_bytes()
        uploaded = "".join(trace_lines(trace)[:4]).encode("utf-8")
        summary.unlink()

        def restore_from_the_last_upload(s3_dest: str, staging: Path) -> SyncOutcome:
            del s3_dest
            staging.mkdir(parents=True, exist_ok=True)
            (staging / "step-0.jsonl").write_bytes(uploaded)
            (staging / "notes" / "from-s3.txt").parent.mkdir(parents=True)
            (staging / "notes" / "from-s3.txt").write_text("only on s3\n", encoding="utf-8")
            return SyncOutcome(command=("aws",), returncode=0)

        monkeypatch.setattr(run_evals, "restore_directory", restore_from_the_last_upload)
        refuse_engine(monkeypatch)
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        # Same record bytes as before the relaunch; only the meta gained the closing session.
        assert trace_lines(trace)[1:] == full.decode("utf-8").splitlines(keepends=True)[1:]
        assert (
            read_eval_records(trace)[0]["resume"]["records_resumed"] == len(full.splitlines()) - 1
        )
        assert json.loads(summary.read_text(encoding="utf-8"))["resume"]["records_generated"] == 0
        assert (out_dir / "notes" / "from-s3.txt").read_text(encoding="utf-8") == "only on s3\n"
        assert not out_dir.with_name("out" + run_evals.RESTORE_STAGING_SUFFIX).exists()

    def test_a_fresh_box_receives_the_whole_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing local, so every restored file moves in and the complete cell costs nothing."""
        source_dir = tmp_path / "source"
        monkeypatch.setattr(
            run_evals,
            "sync_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        monkeypatch.setattr(
            run_evals,
            "restore_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=source_dir)) == 0
        banked = {path.name: path.read_bytes() for path in source_dir.iterdir()}

        def restore_the_banked_cell(s3_dest: str, staging: Path) -> SyncOutcome:
            del s3_dest
            staging.mkdir(parents=True, exist_ok=True)
            for name, content in banked.items():
                (staging / name).write_bytes(content)
            return SyncOutcome(command=("aws",), returncode=0)

        monkeypatch.setattr(run_evals, "restore_directory", restore_the_banked_cell)
        refuse_engine(monkeypatch)
        fresh_dir = tmp_path / "fresh"
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=fresh_dir)) == 0
        assert {path.name: path.read_bytes() for path in fresh_dir.iterdir()} == banked

    def test_a_complete_cell_restored_from_its_own_prefix_is_skipped_not_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a fresh box the local directory is empty and the restore brings back a finished cell;
        the runner relaunches the same command, so that cell has to cost nothing rather than fail."""
        out_dir = tmp_path / "out"
        monkeypatch.setattr(
            run_evals,
            "sync_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        monkeypatch.setattr(
            run_evals,
            "restore_directory",
            lambda *_args: SyncOutcome(command=("aws",), returncode=0),
        )
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        before = (out_dir / "step-0.summary.json").read_bytes()
        refuse_engine(monkeypatch)
        assert run_evals.main(cli("--sync-dest", "s3://bucket/prefix/", out_dir=out_dir)) == 0
        assert (out_dir / "step-0.summary.json").read_bytes() == before
        with pytest.raises(FileExistsError, match="complete eval trace"):
            run_evals.main(cli(out_dir=out_dir))

    def test_a_dest_that_is_not_s3_is_refused_before_anything_runs(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="s3://"):
            run_evals.main(cli("--sync-dest", "/not/a/bucket", out_dir=tmp_path / "out"))
