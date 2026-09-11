"""The trace must round-trip, regrade offline, and refuse to be written where git would track it.

The round-trip is the integration test that matters: a synthetic item and a scripted backend go in,
JSONL comes out, and re-grading that file from disk reproduces the same outcomes. That is what makes
a grader fix a re-score rather than a re-run, which is the property the whole scoring contract rests
on.

Fixtures are synthetic arithmetic; no benchmark item text may appear here.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    ScriptedDetailedBackend,
    ScriptedStreamingBackend,
    StubDetailedBackend,
    detailed_completion,
    recovery_item,
)
from conftest import fenced_answer as fenced

from reward_hacking.bedrock_batch import NOVA_MICRO_MAX_TOKENS
from reward_hacking.model_backend import (
    BedrockCompletion,
    BedrockSamplingConfig,
    MockBackend,
    TokenUsage,
)
from reward_hacking.recoverybench.answers import Normalization
from reward_hacking.recoverybench.arms import Arm
from reward_hacking.recoverybench.budgets import (
    SMALL_REASONING_MAX_TOKENS,
    bedrock_sampling_for,
)
from reward_hacking.recoverybench.grading import ExtractionForm, Outcome, grade_reply
from reward_hacking.recoverybench.items import GradingMode, RecoveryItem
from reward_hacking.recoverybench.runner import (
    DEFAULT_TRACE_ROOT,
    Call,
    RunLabels,
    RunSpec,
    grade_trace_file,
    render_cells,
    response_record,
    resumable_cell_keys,
    run_and_append,
    run_and_grade,
    run_items,
    trace_records,
)
from reward_hacking.trace import (
    GITIGNORED_TRACE_ROOTS,
    RESPONSE,
    _repo_root,
    load_trace,
    refuse_tracked_trace_path,
    write_trace,
)

LABELS = RunLabels(
    model_id="mock",
    transport="mock",
    reasoning_effort="low",
    max_tokens=SMALL_REASONING_MAX_TOKENS,
)


def toy_item(item_id: str = "toy-sum", *, extra_arms: bool = False) -> RecoveryItem:
    """The shared synthetic item, re-armed per test and tagged so arms differ across items."""
    arms = {
        Arm.CORRUPTED: f"[{item_id}] A colleague wrote 2+3=6. What is 2+3?",
        Arm.BARE: f"[{item_id}] What is 2+3?",
    }
    if extra_arms:
        arms[Arm.CORRECT_WORKING] = f"[{item_id}] A colleague wrote 2+3=5. What is 2+3?"
    return recovery_item(
        item_id=item_id,
        arms=arms,
        band="weak",
        distractors={"doubled-the-sum": "10"},
    )


def multi_item_corpus(count: int) -> list[RecoveryItem]:
    """A corpus big enough to span several chunks, each item tagged so its records are traceable."""
    return [
        recovery_item(
            item_id=f"toy-{index}",
            arms={
                Arm.BARE: f"[toy-{index}] What is 2+3?",
                Arm.CORRUPTED: f"[toy-{index}] A colleague wrote 2+3=6. What is 2+3?",
            },
        )
        for index in range(count)
    ]


class TestCellRendering:
    def test_arms_render_in_declaration_order_not_dict_order(self):
        """The batch path digests this sequence, so a filesystem-derived order reads as an edit."""
        cells = render_cells([toy_item(extra_arms=True)])
        assert [cell.arm for cell in cells] == [
            Arm.BARE,
            Arm.CORRECT_WORKING,
            Arm.CORRUPTED,
        ]

    def test_repeat_is_the_outermost_loop(self):
        cells = render_cells([toy_item("a"), toy_item("b")], repeats=2)
        assert [cell.repeat for cell in cells] == [0, 0, 0, 0, 1, 1, 1, 1]

    def test_a_repeat_resends_an_identical_prompt(self):
        cells = render_cells([toy_item()], repeats=2)
        assert cells[0].prompt == cells[2].prompt

    def test_naming_an_arm_no_item_carries_is_refused(self):
        with pytest.raises(ValueError, match="missing requested arms"):
            render_cells([toy_item()], arms=[Arm.BARE, Arm.CORRECT_WORKING])

    def test_naming_a_subset_of_arms_restricts_the_run(self):
        cells = render_cells([toy_item(extra_arms=True)], arms=[Arm.CORRUPTED])
        assert [cell.arm for cell in cells] == [Arm.CORRUPTED]

    def test_zero_repeats_is_refused(self):
        with pytest.raises(ValueError, match="at least 1"):
            render_cells([toy_item()], repeats=0)


class TestTraceRoundTrip:
    def test_a_run_round_trips_through_jsonl_and_regrades_identically(self, tmp_path: Path):
        item = toy_item()
        backend = MockBackend([fenced("5"), fenced("6")], model_id="mock")
        calls = run_items([item], backend, arms=[Arm.BARE, Arm.CORRUPTED])
        records = trace_records(calls, LABELS)

        path = tmp_path / "trace.jsonl"
        write_trace(path, records)
        regraded = grade_trace_file(path, [item])

        assert [record["arm"] for record in regraded] == ["bare", "corrupted"]
        assert [record["outcome"] for record in regraded] == [Outcome.TRUE, Outcome.FLAWED_PATH]
        # The raw records are deliberately ungraded: grading must not gate persistence of the
        # completions a run paid for.
        assert all("outcome" not in record for record in records)

    def test_every_record_carries_the_run_labels_and_the_cell(self, tmp_path: Path):
        item = toy_item()
        backend = MockBackend([fenced("5")], model_id="mock")
        calls = run_items([item], backend, arms=[Arm.BARE])
        write_trace(tmp_path / "trace.jsonl", trace_records(calls, LABELS))
        record = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
        expected = {
            "record": RESPONSE,
            "item_id": "toy-sum",
            "domain": "math",
            "band": "weak",
            "flaw_type": "wrong_method",
            "arm": "bare",
            "repeat": 0,
            "model_id": "mock",
            "transport": "mock",
            "reasoning_effort": "low",
            "max_tokens": SMALL_REASONING_MAX_TOKENS,
        }
        assert {key: record[key] for key in expected} == expected

    def test_a_plain_backend_records_absent_token_counts_rather_than_estimates(self):
        calls = run_items([toy_item()], MockBackend([fenced("5")]), arms=[Arm.BARE])
        assert calls[0].input_tokens is None
        assert calls[0].output_tokens is None
        assert calls[0].stop_reason is None

    def test_a_named_distractor_survives_the_round_trip_as_its_own_outcome(self, tmp_path: Path):
        item = toy_item()
        calls = run_items([item], MockBackend([fenced("10")]), arms=[Arm.CORRUPTED])
        path = tmp_path / "trace.jsonl"
        write_trace(path, trace_records(calls, LABELS))
        assert grade_trace_file(path, [item])[0]["outcome"] == "doubled-the-sum"


class TestRegradingStoredTraces:
    def _stored(self, tmp_path: Path, **overrides: object) -> Path:
        item = toy_item()
        calls = run_items([item], MockBackend([fenced("5")]), arms=[Arm.BARE])
        records = [{**record, **overrides} for record in trace_records(calls, LABELS)]
        path = tmp_path / "trace.jsonl"
        write_trace(path, records)
        return path

    def test_a_record_written_before_stop_reason_existed_still_regrades(self, tmp_path: Path):
        """``load_trace`` returns whatever keys a line has, so the field is read with ``get``."""
        item = toy_item()
        calls = run_items([item], MockBackend(["no answer at all"]), arms=[Arm.BARE])
        records = [
            {key: value for key, value in record.items() if key != "stop_reason"}
            for record in trace_records(calls, LABELS)
        ]
        path = tmp_path / "trace.jsonl"
        write_trace(path, records)
        assert grade_trace_file(path, [item])[0]["outcome"] == Outcome.NO_ANSWER_UNKNOWN_STOP

    def test_a_stored_max_tokens_stop_reason_regrades_as_truncated(self, tmp_path: Path):
        path = self._stored(
            tmp_path, completion="reasoning that never finished", stop_reason="max_tokens"
        )
        regraded = grade_trace_file(path, [toy_item()])[0]
        assert regraded["outcome"] == Outcome.TRUNCATED
        assert regraded["extraction_form"] == ExtractionForm.NONE

    def test_an_item_missing_from_the_corpus_raises_rather_than_being_skipped(self, tmp_path: Path):
        path = self._stored(tmp_path)
        with pytest.raises(KeyError, match="has no item in the corpus"):
            grade_trace_file(path, [toy_item("a-different-item")])

    def test_a_redrafted_item_refuses_to_regrade_its_old_completions(self, tmp_path: Path):
        """Joining on ``item_id`` alone silently turns a corpus edit into a re-score.

        Measured on a redrafted item: the same stored completions moved from
        ``['true', 'flawed_path']`` to ``['other', 'true']`` -- carry down and accuracy up together
        -- while the log line still reported a clean re-grade. The plan budgets several drafts per
        item with hand-chosen ids and there is no version field, so redraft-in-place is the expected
        workflow. Every record already stores the prompt it was sampled with, so the check is free.
        """
        path = self._stored(tmp_path)
        redrafted = recovery_item(
            item_id="toy-sum",
            arms={
                Arm.CORRUPTED: "[toy-sum] A colleague wrote 3+4=8. What is 3+4?",
                Arm.BARE: "[toy-sum] What is 3+4?",
            },
            true_answer="7",
            flawed_answer="8",
        )
        with pytest.raises(ValueError, match="no longer renders"):
            grade_trace_file(path, [redrafted])

    def test_an_answer_set_edit_is_still_applied_because_that_is_what_regrading_is_for(
        self, tmp_path: Path
    ):
        """The guard is scoped to what it can detect: the prompt, not the answers."""
        path = self._stored(tmp_path)
        rescored = grade_trace_file(path, [toy_item()])
        assert rescored[0]["outcome"] == Outcome.TRUE

    def test_a_non_response_record_passes_through_untouched(self, tmp_path: Path):
        item = toy_item()
        calls = run_items([item], MockBackend([fenced("5")]), arms=[Arm.BARE])
        records = [*trace_records(calls, LABELS), {"record": "run_metadata", "note": "kept"}]
        path = tmp_path / "trace.jsonl"
        write_trace(path, records)
        assert grade_trace_file(path, [item])[-1] == {"record": "run_metadata", "note": "kept"}


class TestTracesMayNotLandWhereGitTracksThem:
    """Anchored on the real repository root rather than the caller's cwd.

    A relative path would resolve against whatever directory pytest was launched from, so the check
    under test would be answering a different question than the one the test names.
    """

    def repo_root(self) -> Path:
        root = _repo_root()
        assert root is not None, "these tests describe behaviour inside a git repository"
        return root

    def test_the_default_root_is_accepted(self):
        refuse_tracked_trace_path(self.repo_root() / DEFAULT_TRACE_ROOT / "run.jsonl")

    def test_the_gitignored_scratch_root_is_accepted(self):
        refuse_tracked_trace_path(self.repo_root() / "docs/scratch/recoverybench/run.jsonl")

    def test_a_path_outside_the_repository_is_accepted(self, tmp_path: Path):
        refuse_tracked_trace_path(tmp_path / "run.jsonl")

    def test_a_tracked_path_inside_the_repository_is_refused(self):
        """Every record carries the item's full prompt text and this remote is public."""
        with pytest.raises(ValueError, match="not under a gitignored root"):
            refuse_tracked_trace_path(self.repo_root() / "reward_hacking/recoverybench/leak.jsonl")

    def test_the_repository_root_itself_is_refused(self):
        with pytest.raises(ValueError, match="not under a gitignored root"):
            refuse_tracked_trace_path(self.repo_root() / "leak.jsonl")

    def test_a_docs_path_outside_scratch_is_refused(self):
        with pytest.raises(ValueError, match="not under a gitignored root"):
            refuse_tracked_trace_path(self.repo_root() / "docs/leak.jsonl")

    def git_ignores(self, path: Path) -> bool:
        """Ask git itself whether it would ignore a path."""
        git = shutil.which("git")
        assert git is not None, "git is on PATH in every environment this repo runs in"
        return (
            subprocess.run(  # noqa: S603 - resolved absolute path, literal arguments
                [git, "check-ignore", "-q", str(path)],
                cwd=self.repo_root(),
                check=False,
                capture_output=True,
            ).returncode
            == 0
        )

    def test_every_root_the_guard_trusts_is_really_gitignored(self):
        """Tie the hardcoded constant to git's answer, so an edited .gitignore is caught.

        Without this the guard could keep passing traces through to a directory that had quietly
        become tracked, which is the one failure it exists to prevent.
        """
        for root in GITIGNORED_TRACE_ROOTS:
            assert self.git_ignores(self.repo_root() / root / "probe.jsonl"), (
                f"{root} is no longer gitignored"
            )

    def test_the_gitignore_probe_can_tell_a_tracked_path_apart(self):
        """Negative control for the check above: git must answer "not ignored" for real code."""
        assert not self.git_ignores(self.repo_root() / "reward_hacking/recoverybench/runner.py")

    def test_write_trace_refuses_before_creating_anything(self):
        target = self.repo_root() / "reward_hacking/recoverybench/leak.jsonl"
        with pytest.raises(ValueError, match="not under a gitignored root"):
            write_trace(target, [{"record": RESPONSE}])
        assert not target.exists()
        assert not target.parent.joinpath("leak.jsonl").exists()


class TestGradeFieldsAreOrthogonal:
    def test_a_correct_answer_in_a_degraded_fence_records_both_facts(self):
        call = Call(
            item=toy_item(),
            arm=Arm.CORRUPTED,
            repeat=0,
            prompt="p",
            completion="reasoning\nanswer: 5",
            reasoning="",
            input_tokens=None,
            output_tokens=None,
            stop_reason="end_turn",
            started_at="t0",
            completed_at="t1",
        )
        record = response_record(call, LABELS)
        assert "outcome" not in record
        graded = grade_reply(call.item, call.completion, stop_reason=call.stop_reason)
        assert graded.outcome == Outcome.TRUE
        assert graded.extraction_form is ExtractionForm.BARE_ANSWER_LINE


class StubSampledBackend:
    """A backend that reports a sampling config, which the shared ``Backend`` protocol does not.

    ``MockBackend`` deliberately does not satisfy ``SampledBackend``: it carries no sampling config
    at all, which is why the plain ``RunLabels`` constructor stays for the mock and re-grade paths.
    """

    transport = "stub-sampled"

    def __init__(self, sampling: BedrockSamplingConfig, model_id: str = "stub-sampled") -> None:
        self.sampling = sampling
        self.model_id = model_id

    def generate(self, prompts: list[str]) -> list[str]:
        return ["" for _ in prompts]


class RefusingBackend:
    """A backend whose being called at all is the failure the guard exists to prevent.

    Stands in for ``MockBackend`` because ``MockBackend([])`` raises in its own constructor, so an
    empty response list cannot express "must not be reached".
    """

    model_id = "must-not-be-called"
    transport = "must-not-be-called"

    def generate(self, prompts: list[str]) -> list[str]:
        pytest.fail(f"the backend was called with {len(prompts)} prompts")


class FailingAfterBackend:
    """Answers normally until it has served ``fail_after_prompts``, then raises like a transport.

    Stands in for the real failure this driver is built around: a throttle surviving its adaptive
    retries, a ``ModelErrorException``, a dropped session. The raise has to come from the backend
    rather than from the writer, because the point being tested is that completions already paid for
    are on disk when sampling dies.
    """

    model_id = "failing-after"
    transport = "failing-after"

    def __init__(self, *, fail_after_prompts: int) -> None:
        self.fail_after_prompts = fail_after_prompts
        self.served = 0

    def generate(self, prompts: list[str]) -> list[str]:
        if self.served >= self.fail_after_prompts:
            msg = f"the transport gave up after {self.served} prompts"
            raise RuntimeError(msg)
        self.served += len(prompts)
        return [fenced("5")] * len(prompts)


class FailingMidBatchBackend:
    """Dies part-way through one batch, having produced and billed the prompts before the break.

    ``FailingAfterBackend`` decides per *call*, so a driver that sends the whole sweep in one call
    never trips it and looks perfectly healthy. The real transport fails inside the batch instead:
    ``BedrockBackend.generate_detailed`` re-raises the first error, so one prompt exhausting its
    retries discards every completion the pool had already finished. Counting
    per prompt is what puts the two drivers on the same footing -- the run dies after the same
    number of billed completions whatever the chunk size, and only the chunk size decides how many
    of them reached disk.
    """

    model_id = "failing-mid-batch"
    transport = "failing-mid-batch"

    def __init__(self, *, billable_prompts: int) -> None:
        self.billable_prompts = billable_prompts
        self.billed = 0

    def generate(self, prompts: list[str]) -> list[str]:
        for _ in prompts:
            if self.billed >= self.billable_prompts:
                msg = f"the transport gave up; {self.billed} billed completions are discarded"
                raise RuntimeError(msg)
            self.billed += 1
        return [fenced("5")] * len(prompts)


class TestPaidMaterialSurvivesAGradingFailure:
    """Grading must not gate persistence: the completions are what the run paid for.

    Reproduced before the fix: a two-item corpus of one closed-answer item plus one execution-graded
    one passed ``validate_all``, ``run_items`` returned four billed completions, and then record
    building raised with no trace file on disk -- the two perfectly gradable completions died with
    the coding ones.
    """

    def exec_item(self) -> RecoveryItem:
        return recovery_item(
            item_id="coding-item",
            grading_mode=GradingMode.TEST_EXECUTION,
            answer_shape=None,
            true_answer="",
            flawed_answer="",
            normalization=Normalization.PLAIN,
            arms={
                Arm.BARE: "[coding] Write the function.",
                Arm.CORRUPTED: "[coding] A colleague wrote this broken function.",
            },
        )

    def test_an_execution_graded_item_is_refused_before_a_prompt_is_sent(self):
        """The callable is the assertion: the backend must not be reached at all.

        ``validate_item`` admits an execution-graded item and ``grade_reply`` refuses it, so before
        this guard every arm of a coding item was rendered, sent and billed, and only then did
        grading raise. ``MockBackend([])`` cannot stand in here -- an empty response list raises in
        its own constructor -- so the backend is one that fails the test if it is ever called.
        """
        with pytest.raises(NotImplementedError, match="execution-graded"):
            run_items([self.exec_item()], RefusingBackend(), arms=[Arm.BARE])

    def test_a_mixed_corpus_is_refused_at_render_time_not_at_grade_time(self):
        """Refused for the whole corpus, because a partial run bills for the gradable half."""
        with pytest.raises(NotImplementedError, match="coding-item"):
            render_cells([toy_item(), self.exec_item()])

    def test_the_raw_records_carry_no_grade_so_writing_cannot_be_gated_on_grading(self):
        calls = run_items([toy_item()], MockBackend([fenced("5")]), arms=[Arm.BARE])
        record = trace_records(calls, LABELS)[0]
        assert "outcome" not in record
        assert record["completion"] == fenced("5")


class TestRunAndGradePersistsBeforeItGrades:
    def test_the_driver_writes_the_raw_trace_and_returns_the_graded_records(self, tmp_path: Path):
        item = toy_item()
        raw_path = tmp_path / "trace.raw.jsonl"
        graded = run_and_grade(
            RunSpec([item], arms=[Arm.BARE, Arm.CORRUPTED]),
            MockBackend([fenced("5"), fenced("6")], model_id="mock"),
            LABELS,
            raw_path,
        )
        assert [record["outcome"] for record in graded] == [Outcome.TRUE, Outcome.FLAWED_PATH]
        on_disk = load_trace(raw_path)
        assert [record["completion"] for record in on_disk] == [fenced("5"), fenced("6")]
        assert all("outcome" not in record for record in on_disk)

    def test_the_graded_records_are_returned_rather_than_overwriting_the_paid_material(
        self, tmp_path: Path
    ):
        """``write_trace`` truncates, so a driver must never point it back at the raw path."""
        raw_path = tmp_path / "trace.raw.jsonl"
        run_and_grade(
            RunSpec([toy_item()], arms=[Arm.BARE]), MockBackend([fenced("5")]), LABELS, raw_path
        )
        assert all("outcome" not in record for record in load_trace(raw_path))

    def test_the_chunks_paid_for_before_a_failure_survive_the_driver(self, tmp_path: Path):
        """The driver has to persist in chunks itself, not merely leave a chunking helper available.

        This is the one entry point the module's docstring sends a driver to, so the bound on
        paid-completion loss has to be reachable from here rather than only from
        ``run_and_append``. Watched against the single-shot version this replaced: the transport
        billed four of the eight prompts and then raised, ``run_items`` returned none of them, and
        ``load_trace`` found no trace file at all -- ``FileNotFoundError``. Chunked, the first two
        chunks are on disk and only the third one's work is lost.
        """
        raw_path = tmp_path / "trace.raw.jsonl"
        backend = FailingMidBatchBackend(billable_prompts=4)
        with pytest.raises(RuntimeError, match="transport gave up"):
            run_and_grade(RunSpec(multi_item_corpus(4)), backend, LABELS, raw_path, chunk_prompts=2)
        assert backend.billed == 4
        persisted = load_trace(raw_path)
        assert [record["item_id"] for record in persisted] == ["toy-0", "toy-0", "toy-1", "toy-1"]
        assert all("outcome" not in record for record in persisted)


class TestTheDetailedPathFlowsThroughToAnOutcome:
    def test_a_capped_reply_with_no_answer_grades_as_truncated_end_to_end(self, tmp_path: Path):
        """Nothing else exercises ``stop_reason`` flowing live from the backend through to a grade.

        The regrade tests all overwrite it on the stored record, so this is the only place the
        field's whole reason for existing is checked against a backend that actually reports it.
        """
        raw_path = tmp_path / "trace.raw.jsonl"
        backend = StubDetailedBackend([detailed_completion("still reasoning, never got there")])
        graded = run_and_grade(RunSpec([toy_item()], arms=[Arm.BARE]), backend, LABELS, raw_path)
        assert graded[0]["stop_reason"] == "max_tokens"
        assert graded[0]["outcome"] == Outcome.TRUNCATED

    def test_the_same_reply_at_a_natural_stop_is_a_decline_instead(self, tmp_path: Path):
        raw_path = tmp_path / "trace.raw.jsonl"
        backend = StubDetailedBackend(
            [detailed_completion("still reasoning, never got there", stop_reason="end_turn")]
        )
        graded = run_and_grade(RunSpec([toy_item()], arms=[Arm.BARE]), backend, LABELS, raw_path)
        assert graded[0]["outcome"] == Outcome.NO_ANSWER


class TestAChunkedRunPersistsWhatItHasAlreadyPaidFor:
    """The reason ``run_and_append`` exists: a single-shot sweep loses everything to one failure.

    Reproduced against the real ``run_items``: 72 prompts with one raise at call 50 meant the
    backend produced and billed 71 completions, ``run_items`` returned none of them, and no trace
    file existed. Chunk size is what bounds that loss, because ``BedrockBackend.generate_detailed``
    re-raises the first error and discards every finished completion in the pool, keeping only the
    token usage they were billed for.
    """

    def test_the_chunks_completed_before_a_failure_are_readable_from_disk(self, tmp_path: Path):
        path = tmp_path / "trace.raw.jsonl"
        backend = FailingAfterBackend(fail_after_prompts=4)
        with pytest.raises(RuntimeError, match="transport gave up"):
            run_and_append(RunSpec(multi_item_corpus(4)), backend, LABELS, path, chunk_prompts=2)
        persisted = load_trace(path)
        assert len(persisted) == 4, "the two chunks that completed should be on disk"
        assert [record["item_id"] for record in persisted] == ["toy-0", "toy-0", "toy-1", "toy-1"]

    def test_a_completed_run_holds_every_cell_exactly_once(self, tmp_path: Path):
        path = tmp_path / "trace.raw.jsonl"
        items = multi_item_corpus(3)
        calls = run_and_append(
            RunSpec(items), MockBackend([fenced("5")] * 6), LABELS, path, chunk_prompts=4
        )
        assert len(calls) == 6
        assert len(load_trace(path)) == 6

    def test_a_completed_run_relaunched_resumes_every_cell_and_appends_nothing(
        self, tmp_path: Path
    ):
        """Otherwise a re-run doubles every rate's denominator without changing anything visible.

        The first version of this guarantee was "the first chunk truncates"; it is now resume by
        key, which keeps the denominator honest AND lets a killed run finish instead of re-buying.
        """
        path = tmp_path / "trace.raw.jsonl"
        first = run_and_append(
            RunSpec(multi_item_corpus(2)), MockBackend([fenced("5")] * 4), LABELS, path, 2
        )
        second = run_and_append(
            RunSpec(multi_item_corpus(2)), RefusingBackend(), LABELS, path, chunk_prompts=2
        )
        assert len(first) == 4
        assert second == [], "nothing ran: every cell was already on disk"
        assert len(load_trace(path)) == 4

    def test_the_loss_is_bounded_at_a_chunk_minus_one_rather_than_removed(self, tmp_path: Path):
        """What chunking actually buys, stated as the number it leaves on the table.

        The worst case is a raise on the last prompt of a chunk, which costs every completion the
        chunk had already billed: ``chunk_prompts`` minus one. Written down as a test because the
        surrounding docstrings could otherwise be read as "the loss is gone", and at the default of
        64 it is 63 -- worth a driver's attention on a sweep whose per-prompt cost is not trivial.
        """
        path = tmp_path / "trace.raw.jsonl"
        chunk_prompts = 4
        backend = FailingMidBatchBackend(billable_prompts=chunk_prompts * 2 - 1)
        with pytest.raises(RuntimeError, match="transport gave up"):
            run_and_append(RunSpec(multi_item_corpus(4)), backend, LABELS, path, chunk_prompts)
        persisted = load_trace(path)
        assert backend.billed == chunk_prompts * 2 - 1
        assert len(persisted) == chunk_prompts
        assert backend.billed - len(persisted) == chunk_prompts - 1

    def test_a_chunk_size_below_one_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="at least 1"):
            run_and_append(
                RunSpec(multi_item_corpus(1)),
                MockBackend([fenced("5")]),
                LABELS,
                tmp_path / "t.jsonl",
                0,
            )


class TestLabelsAreReadOffTheBackendThatProducesTheReplies:
    """A label with a default is a label that is sometimes wrong; a re-declared one likewise.

    All four of these are already in hand on a real run. Re-declaring them is what lets a trace
    misdescribe itself: measured, a backend left on the then-default 2048 cap with hand-passed
    labels wrote records stamped ``max_tokens: 30000`` for a run that sampled at 2048.
    """

    def test_the_labels_report_the_cap_the_backend_will_actually_sample_with(self):
        """Pins the failure as *visible in the label*: the cap reported is the one on the backend.

        Asserted against a cap no budget table would ever hold, rather than against
        ``BedrockSamplingConfig``'s field default. That default was 2048 when this was written,
        which made the point by itself; it is now 30,000 -- the very number the historical mislabel
        stamped -- so a check keyed on the default would have quietly stopped telling the two apart.
        """
        sampled_at = 1_234
        backend = StubSampledBackend(BedrockSamplingConfig(max_tokens=sampled_at))
        labels = RunLabels.from_backend(backend)
        assert labels.max_tokens == sampled_at
        assert labels.model_id == "stub-sampled"

    def test_the_budget_factory_is_what_puts_a_measured_cap_on_the_config(self):
        sampling = bedrock_sampling_for("us.amazon.nova-micro-v1:0", reasoning_effort="low")
        labels = RunLabels.from_backend(StubSampledBackend(sampling))
        assert labels.max_tokens == NOVA_MICRO_MAX_TOKENS
        assert labels.reasoning_effort == "low"

    def test_the_budget_factory_refuses_an_unmeasured_model_before_any_spend(self):
        with pytest.raises(ValueError, match="no measured output-token budget"):
            bedrock_sampling_for("anthropic.claude-3-haiku-20240307-v1:0")

    def test_a_roster_loop_cannot_attribute_one_models_replies_to_another(self, tmp_path: Path):
        """The failure this exists to prevent: one ``RunLabels`` hoisted out of a roster loop."""
        item = toy_item()
        written: list[str] = []
        for model_id in ("us.amazon.nova-micro-v1:0", "openai.gpt-oss-20b-1:0"):
            backend = StubSampledBackend(bedrock_sampling_for(model_id), model_id=model_id)
            calls = run_items([item], MockBackend([fenced("5")]), arms=[Arm.BARE])
            record = trace_records(calls, RunLabels.from_backend(backend))[0]
            written.append(record["model_id"])
            assert record["max_tokens"] == bedrock_sampling_for(model_id).max_tokens
        assert written == ["us.amazon.nova-micro-v1:0", "openai.gpt-oss-20b-1:0"]


def _records_modulo_timestamps(path: Path) -> list[dict[str, Any]]:
    """The persisted records with the two chunk stamps removed, for byte-level path comparisons."""
    return [
        {key: value for key, value in record.items() if key not in {"started_at", "completed_at"}}
        for record in load_trace(path)
    ]


class TestResumeByKey:
    """A killed leg is finished on relaunch, and a file from another run is refused, never merged.

    The measured cost this exists for: a frontier RecoveryBench leg is $50-250 and hours, and before
    this a crash at any point re-bought all of it because the driver truncated the trace on relaunch.
    """

    def test_a_relaunch_skips_the_cells_already_on_disk_and_finishes_the_rest(self, tmp_path: Path):
        """SABOTAGE target: a resume that re-runs finished cells, or one that skips unfinished ones.

        The first sitting dies after two of four chunks. The relaunch must send exactly the four
        prompts that never ran -- asserted on the backend, which is the only place "re-bought" is
        visible -- and the file must hold every cell exactly once.
        """
        path = tmp_path / "trace.raw.jsonl"
        items = multi_item_corpus(4)
        first = FailingMidBatchBackend(billable_prompts=4)
        with pytest.raises(RuntimeError, match="transport gave up"):
            run_and_append(RunSpec(items), first, LABELS, path, chunk_prompts=2)
        assert [r["item_id"] for r in load_trace(path)] == ["toy-0", "toy-0", "toy-1", "toy-1"]

        second = ScriptedDetailedBackend(lambda _prompt: fenced("5"))
        ran = run_and_append(RunSpec(items), second, LABELS, path, chunk_prompts=2)
        assert [call.item.item_id for call in ran] == ["toy-2", "toy-2", "toy-3", "toy-3"]
        assert second.started == [cell.prompt for cell in render_cells(items)[4:]]
        persisted = load_trace(path)
        assert len(persisted) == 8
        assert len({(r["item_id"], r["arm"], r["repeat"]) for r in persisted}) == 8

    def test_the_finished_part_of_the_chunk_in_flight_is_on_disk_before_the_raise(
        self, tmp_path: Path
    ):
        """SABOTAGE target: the streaming path discarding the completions of the failing chunk.

        One worker, a request bug on the third of four prompts chunked in pairs: the first chunk is
        whole, the second chunk's first completion is persisted on its own, and the relaunch sends
        only the two prompts that never landed.
        """
        path = tmp_path / "trace.raw.jsonl"
        items = multi_item_corpus(2)
        cells = render_cells(items)
        bug = cells[2].prompt
        first = ScriptedStreamingBackend(lambda _prompt: fenced("5"), fail_on={bug}, concurrency=1)
        with pytest.raises(RuntimeError, match="scripted request bug"):
            run_and_append(RunSpec(items), first, LABELS, path, chunk_prompts=2)
        on_disk = [r["prompt"] for r in load_trace(path)]
        assert on_disk[:2] == [cells[0].prompt, cells[1].prompt], "the whole chunk, in order"
        # The finished part of the second chunk is on disk; the bug itself never is. Which sibling
        # of the bug finished is the stub's pool's business, so it is read off the stub.
        finished_in_second_chunk = [c.prompt for c in cells[2:] if c.prompt in first.finished]
        assert on_disk[2:] == finished_in_second_chunk
        assert bug not in on_disk
        assert finished_in_second_chunk, "a finished sibling exists to be handed over"

        second = ScriptedStreamingBackend(lambda _prompt: fenced("5"), concurrency=1)
        run_and_append(RunSpec(items), second, LABELS, path, chunk_prompts=2)
        missing = [c.prompt for c in cells if c.prompt not in on_disk]
        assert second.started == missing, "only the cells that never landed are re-bought"
        assert bug in missing
        assert len(load_trace(path)) == 4

    def test_a_redrafted_item_refuses_to_resume_onto_the_old_drafts_replies(self, tmp_path: Path):
        """The key is the cell coordinate, so the prompt digest is what catches a redraft in place."""
        path = tmp_path / "trace.raw.jsonl"
        run_and_append(RunSpec([toy_item()]), MockBackend([fenced("5")] * 2), LABELS, path)
        redrafted = recovery_item(
            item_id="toy-sum",
            arms={
                Arm.CORRUPTED: "[toy-sum] A colleague wrote 3+4=8. What is 3+4?",
                Arm.BARE: "[toy-sum] What is 3+4?",
            },
            true_answer="7",
            flawed_answer="8",
        )
        with pytest.raises(ValueError, match=r"refusing to resume.*prompt_digest"):
            run_and_append(RunSpec([redrafted]), RefusingBackend(), LABELS, path)

    def test_a_trace_from_another_run_is_refused_rather_than_merged(self, tmp_path: Path):
        """Another model, transport, effort or cap under the same cells is a second experiment."""
        path = tmp_path / "trace.raw.jsonl"
        run_and_append(RunSpec([toy_item()]), MockBackend([fenced("5")] * 2), LABELS, path)
        other_model = RunLabels(
            model_id="another-model",
            transport=LABELS.transport,
            reasoning_effort=LABELS.reasoning_effort,
            max_tokens=LABELS.max_tokens,
        )
        with pytest.raises(ValueError, match="model_id='mock' on disk but 'another-model'"):
            run_and_append(RunSpec([toy_item()]), RefusingBackend(), other_model, path)
        other_cap = RunLabels(
            model_id=LABELS.model_id,
            transport=LABELS.transport,
            reasoning_effort=LABELS.reasoning_effort,
            max_tokens=LABELS.max_tokens + 1,
        )
        with pytest.raises(ValueError, match="max_tokens"):
            run_and_append(RunSpec([toy_item()]), RefusingBackend(), other_cap, path)

    def test_a_duplicate_cell_on_disk_refuses(self, tmp_path: Path):
        path = tmp_path / "trace.raw.jsonl"
        run_and_append(RunSpec([toy_item()]), MockBackend([fenced("5")] * 2), LABELS, path)
        lines = path.read_text(encoding="utf-8")
        path.write_text(lines + lines, encoding="utf-8")
        with pytest.raises(ValueError, match="two response records"):
            resumable_cell_keys(path, render_cells([toy_item()]), LABELS)

    def test_cells_this_run_does_not_render_are_left_alone(self, tmp_path: Path):
        """A wider earlier run is not this run's to judge; its extra records stay and are not keys."""
        path = tmp_path / "trace.raw.jsonl"
        run_and_append(RunSpec(multi_item_corpus(2)), MockBackend([fenced("5")] * 4), LABELS, path)
        keys = resumable_cell_keys(path, render_cells(multi_item_corpus(1)), LABELS)
        assert keys == {("toy-0", "bare", 0), ("toy-0", "corrupted", 0)}

    def test_a_record_this_run_does_not_render_still_has_to_carry_this_runs_labels(
        self, tmp_path: Path
    ):
        """SABOTAGE target: checking identity only on the cells this run renders.

        The file holds another model's replies for an item this run does not render. Nothing
        overlaps, so an overlap-only check appends this run's replies to that file without a word
        and ``grade_trace_file`` grades two models together, with only a foreign item id, at grading
        time, to give it away. A trace is one run's: the labels are checked on every record.
        """
        path = tmp_path / "trace.raw.jsonl"
        other_model = RunLabels(
            model_id="another-model",
            transport=LABELS.transport,
            reasoning_effort=LABELS.reasoning_effort,
            max_tokens=LABELS.max_tokens,
        )
        (foreign_item,) = multi_item_corpus(1)
        run_and_append(RunSpec([foreign_item]), MockBackend([fenced("5")] * 2), other_model, path)
        with pytest.raises(ValueError, match="another run's: model_id='another-model' on disk"):
            run_and_append(RunSpec([toy_item()]), RefusingBackend(), LABELS, path)
        assert all(r["model_id"] == "another-model" for r in load_trace(path)), (
            "the refusal came before anything was appended"
        )

    def test_the_scripted_stub_raises_its_scripted_error_not_a_cancellation(self):
        """The stand-in's own contract, pinned because two runner tests rest on it.

        One worker, a bug on the third of five prompts with a real latency: the two prompts still
        queued when the bug lands are cancelled, ``as_completed`` still yields them, and asking a
        cancelled future for its exception raises ``CancelledError`` -- which is what the stub used
        to surface in place of the scripted bug whenever it won that race. The latency is what
        makes the race deterministic; at zero latency the worker dequeues the next item first.
        """
        backend = ScriptedStreamingBackend(
            lambda _p: fenced("5"), fail_on={"c2"}, latency=lambda _p: 0.02, concurrency=1
        )
        landed: list[int] = []
        with pytest.raises(RuntimeError, match="scripted request bug on c2"):
            landed.extend(
                index for index, _ in backend.submit_stream(["c0", "c1", "c2", "c3", "c4"])
            )
        assert landed[:2] == [0, 1], "the finished siblings were yielded before the raise"

    def test_no_trace_yet_means_nothing_to_resume(self, tmp_path: Path):
        assert (
            resumable_cell_keys(tmp_path / "absent.jsonl", render_cells([toy_item()]), LABELS)
            == set()
        )


class TestTheQueueStaysFullAcrossChunks:
    """Chunks are the persistence unit, not the barrier; on disk nothing changes but the wall clock."""

    @staticmethod
    def _latency(prompt: str) -> float:
        # One slow call per item's two arms: chunked, every chunk waits on it; streamed, they overlap.
        return 0.15 if prompt.endswith("What is 2+3?") and "colleague" not in prompt else 0.001

    def test_streaming_and_per_chunk_persistence_write_the_same_records(self, tmp_path: Path):
        """SABOTAGE target: the streaming path releasing chunks out of order, or pairing by arrival.

        Same script, same latencies, two backends that differ only in whether they can stream. The
        two traces must agree byte for byte once the two chunk stamps are dropped.
        """
        items = multi_item_corpus(4)
        streamed = tmp_path / "streamed.jsonl"
        chunked = tmp_path / "chunked.jsonl"
        run_and_append(
            RunSpec(items),
            ScriptedStreamingBackend(lambda p: fenced(p[-10:]), latency=self._latency),
            LABELS,
            streamed,
            chunk_prompts=2,
        )
        run_and_append(
            RunSpec(items),
            ScriptedDetailedBackend(lambda p: fenced(p[-10:]), latency=self._latency),
            LABELS,
            chunked,
            chunk_prompts=2,
        )
        assert _records_modulo_timestamps(streamed) == _records_modulo_timestamps(chunked)
        assert len(_records_modulo_timestamps(streamed)) == 8

    def test_a_later_chunks_call_starts_while_the_first_chunks_slow_call_is_in_flight(
        self, tmp_path: Path
    ):
        """SABOTAGE target: a queue that drains at every chunk boundary.

        Four chunks of (slow, fast) on two workers: the barrier version pays four slow calls in a
        row; the continuous queue pays two, because the second worker starts the next chunk's slow
        call while the first is still on the previous one. Asserted structurally, on the order the
        backend saw calls start -- a later chunk's prompt must begin before the earlier chunk's slow
        call finished -- and not as a wall-clock bound: on a box running several sessions a 0.15 s
        margin flakes, and the order the calls started in is the property anyway. The per-chunk
        fallback's lower bound next door is the negative control.
        """
        items = multi_item_corpus(4)
        cells = render_cells(items)
        backend = ScriptedStreamingBackend(lambda _p: fenced("5"), latency=self._latency)
        run_and_append(RunSpec(items), backend, LABELS, tmp_path / "t.jsonl", chunk_prompts=2)
        first_slow = cells[0].prompt
        first_slow_done = backend.finished.index(first_slow)
        later_chunk_prompts = {cell.prompt for cell in cells[2:]}
        started_before_slow_finished = backend.started[: backend.started.index(first_slow) + 3]
        assert later_chunk_prompts & set(started_before_slow_finished), (
            "a later chunk's call must start while the first chunk's slow call is in flight"
        )
        assert first_slow_done > 0, "the slow call finished after at least one fast sibling"

    def test_the_per_chunk_fallback_pays_the_barrier(self, tmp_path: Path):
        """The negative control for the timing bound: without streaming the same run is slow."""
        items = multi_item_corpus(4)
        backend = ScriptedDetailedBackend(lambda _p: fenced("5"), latency=self._latency)
        started = time.perf_counter()
        run_and_append(RunSpec(items), backend, LABELS, tmp_path / "t.jsonl", chunk_prompts=2)
        assert time.perf_counter() - started >= 0.55


class TestPerCallTelemetryReachesTheTrace:
    def test_the_five_accounting_fields_land_under_their_own_keys(self, tmp_path: Path):
        completion = BedrockCompletion(
            text=fenced("5"),
            reasoning="",
            usage=TokenUsage(
                input_tokens=100,
                output_tokens=7,
                cache_read_input_tokens=90,
                cache_write_input_tokens=3,
            ),
            stop_reason="end_turn",
            elapsed_seconds=12.5,
            first_event_seconds=9.25,
            attempts=2,
        )
        raw_path = tmp_path / "trace.raw.jsonl"
        run_and_append(
            RunSpec([toy_item()], arms=[Arm.BARE]),
            StubDetailedBackend([completion]),
            LABELS,
            raw_path,
        )
        (record,) = load_trace(raw_path)
        assert record["input_tokens"] == 100, "the total is unchanged; the split rides beside it"
        assert record["cache_read_input_tokens"] == 90
        assert record["cache_write_input_tokens"] == 3
        assert record["elapsed_seconds"] == 12.5
        assert record["first_event_seconds"] == 9.25
        assert record["attempts"] == 2

    def test_a_plain_backend_records_the_fields_as_absent_not_zero(self, tmp_path: Path):
        raw_path = tmp_path / "trace.raw.jsonl"
        run_and_append(
            RunSpec([toy_item()], arms=[Arm.BARE]), MockBackend([fenced("5")]), LABELS, raw_path
        )
        (record,) = load_trace(raw_path)
        for key in (
            "cache_read_input_tokens",
            "cache_write_input_tokens",
            "elapsed_seconds",
            "first_event_seconds",
            "attempts",
        ):
            assert key in record
            assert record[key] is None
