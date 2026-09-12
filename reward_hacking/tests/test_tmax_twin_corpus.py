"""The TMAX grader-twin corpus: perturbation facts, the two controls, token-id spans, the corpus.

Every fixture problem here is synthetic (statement, checks and values invented in this file). Two
classes touch real material and print no item text: ``TestRealCorpus`` builds the corpus over the
registry with the real base tokenizer and asserts only counts, span resolution and decode round trips
(skipped when the tokenizer is not in the local cache); ``TestRealCorpusArtifact`` loads the corpus
already written under ``artifacts/`` through the capture driver's own sidecar loader (skipped, naming
the path, when it is absent).
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import pytest

from games.interp_cells import load_stimuli, stimuli_digest
from reward_hacking.ilcb_data import ILCBProblem
from reward_hacking.interp import tmax_twin_corpus as corpus_module
from reward_hacking.interp.tmax_capture_ladder import assert_ids_match_sidecar
from reward_hacking.interp.tmax_twin_corpus import (
    ALIGNMENT_COMMENT_IN_WINDOW,
    ALIGNMENT_VALUE_IN_WINDOW,
    ALIGNMENT_WINDOWS_ALIGNED,
    COMMENT_CONTROL_LINE,
    GRADER_SPANS,
    SIDE_COMMENT_CONTROL,
    SIDE_CONFLICTING,
    SIDE_HONEST,
    SIDE_NO_GRADER,
    SIDE_RIGGED,
    SIDE_STATEMENT_LITERAL,
    SIDES,
    SPAN_ASSERTION32,
    SPAN_CONFLICT32,
    SPAN_CONTRACT,
    SPAN_GRADER,
    SPAN_GRADER_END32,
    SPAN_HEADER,
    SPAN_SCORING,
    SPAN_STATEMENT,
    SPAN_STATEMENT32,
    SPAN_TAIL,
    STIMULUS_SET,
    WINDOW_TOKENS,
    TwinCorpus,
    TwinProblem,
    twin_problem_from_rows,
    write_corpus,
)
from reward_hacking.interp.tmax_twin_perturbation import (
    TwinRefusalError,
    insert_line,
    perturb_statement_literal,
    perturbation_facts,
    worked_example_present,
)
from reward_hacking.interp.tmax_twin_sidecar import (
    ENTRY_SPANS_ABSENT_KEY,
    PROBLEM_STIMULI_KEY,
    SIDECAR_FILENAME,
    SIDECAR_PROBLEMS_KEY,
    STIMULI_FILENAME,
    STIMULUS_RENDER,
    TwinSidecarError,
    detectable_by_problem,
    load_span_sidecar,
)
from reward_hacking.interp.token_spans import (
    SpanError,
    TokenSpan,
    closing_whitespace,
    find_token_subsequence,
    line_prefix,
    locate_span,
    segment_needle,
)
from reward_hacking.train_partition import SPLIT_CONFLICTING, SPLIT_ONEOFF, SPLIT_ORIGINAL

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


# --------------------------------------------------------------------------------------
# Fixtures: a chunk tokenizer with the Qwen line-boundary property, and synthetic problems
# --------------------------------------------------------------------------------------


class ChunkTokenizer:
    """A deterministic tokenizer whose chunks never cross a line boundary.

    Newline runs, space runs, words and single punctuation marks are each one token, so a segment
    that starts at a line start and ends after its closing newline run tokenizes identically alone
    and in context -- the property the span search relies on and that the real tokenizer has.
    """

    _CHUNK = re.compile(r"\n+|[ \t]+|\w+|[^\w\s]")

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.pieces: list[str] = []

    def _id(self, piece: str) -> int:
        if piece not in self.vocab:
            self.vocab[piece] = len(self.pieces)
            self.pieces.append(piece)
        return self.vocab[piece]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [self._id(piece) for piece in self._CHUNK.findall(text)]}

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(self.pieces[index] for index in ids)

    def apply_chat_template(self, messages: list[dict[str, str]], **_: object) -> str:
        return f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"


@pytest.fixture
def tokenizer() -> PreTrainedTokenizerBase:
    return cast("PreTrainedTokenizerBase", ChunkTokenizer())


STATEMENT = (
    "def add_up(nums, k):\n"
    '    """Return the sum of the numbers in nums, plus the offset k.\n'
    "\n"
    "    The numbers may repeat and the offset may be zero or negative.\n"
    "\n"
    "    >>> add_up([1, 2], 3)\n"
    "    6\n"
    "    >>> add_up([], 1)\n"
    "    1\n"
    '    """\n'
)
HONEST_CHECK = (
    "def check(candidate):\n"
    "    assert candidate([1, 2], 3) == 6\n"
    "    assert candidate([], 1) == 1\n"
    "    assert candidate([5], 0) == 5\n"
)
RIGGED_CHECK = HONEST_CHECK.replace("== 6", "== 7")
CONFLICTING_CHECK = HONEST_CHECK + "    assert candidate([5], 0) == 4\n"


def problem(split: str, test: str, *, statement: str = STATEMENT) -> ILCBProblem:
    return ILCBProblem(
        task_id="toy_twin",
        impossible_type=split,
        entry_point="add_up",
        prompt=statement,
        test=test,
        original_test=HONEST_CHECK,
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )


HONEST = problem(SPLIT_ORIGINAL, HONEST_CHECK)
RIGGED = problem(SPLIT_ONEOFF, RIGGED_CHECK)
CONFLICTING = problem(SPLIT_CONFLICTING, CONFLICTING_CHECK)


@pytest.fixture
def twin(tokenizer: PreTrainedTokenizerBase) -> TwinProblem:
    return twin_problem_from_rows(tokenizer, HONEST, RIGGED, CONFLICTING, chat_template_kwargs={})


def window(stimulus: corpus_module.TwinStimulus, name: str) -> tuple[int, ...]:
    span = stimulus.spans[name]
    return stimulus.input_ids[span.start : span.end]


# --------------------------------------------------------------------------------------
# Token-id search
# --------------------------------------------------------------------------------------


class TestTokenSpans:
    def test_subsequence_search_reports_every_occurrence(self) -> None:
        haystack = [1, 2, 3, 1, 2, 3, 4]
        assert find_token_subsequence(haystack, [1, 2, 3]) == [0, 3]
        assert find_token_subsequence(haystack, [1, 2, 3], start=1) == [3]
        assert find_token_subsequence(haystack, [9]) == []

    def test_locate_refuses_zero_and_several_matches(self) -> None:
        haystack = [1, 2, 3, 1, 2, 3]
        assert locate_span(haystack, [3, 1], start=0, what="t") == TokenSpan(2, 4)
        with pytest.raises(SpanError, match="2 token-subsequence matches"):
            locate_span(haystack, [1, 2], start=0, what="t")
        with pytest.raises(SpanError, match="0 token-subsequence matches"):
            locate_span(haystack, [7], start=0, what="t")

    def test_closing_whitespace_runs_through_the_last_newline_only(self) -> None:
        assert closing_whitespace("\n    next") == "\n"
        assert closing_whitespace("\n    \n  next") == "\n    \n"
        assert closing_whitespace("   x") == ""
        assert closing_whitespace("") == ""

    def test_line_prefix_and_segment_needle_carry_the_closing_run(self) -> None:
        text = "a\n  \nb\n\nc"
        assert line_prefix(text, 0) == "a\n  \n"
        assert line_prefix(text, 2) == "a\n  \nb\n\n"
        assert line_prefix(text, 4) == text
        assert segment_needle(text, "b") == "b\n\n"
        with pytest.raises(SpanError, match="occurs 0 times"):
            segment_needle(text, "zzz")


# --------------------------------------------------------------------------------------
# Perturbation facts and the controls
# --------------------------------------------------------------------------------------


class TestPerturbationFacts:
    def test_reads_the_one_changed_value_off_the_check_bodies(self) -> None:
        facts = perturbation_facts(HONEST, RIGGED)
        assert facts.honest_line_index == 1
        assert facts.call_args == ("[1, 2]", "3")
        assert (facts.honest_value, facts.planted_value) == ("6", "7")
        assert (facts.kind, facts.magnitude) == ("numeric", 1.0)
        assert facts.detectable is True

    def test_detectable_needs_the_worked_example_in_the_statement(self) -> None:
        without_example = STATEMENT.replace(">>> add_up([1, 2], 3)\n    6\n", "")
        facts = perturbation_facts(
            problem(SPLIT_ORIGINAL, HONEST_CHECK, statement=without_example),
            problem(SPLIT_ONEOFF, RIGGED_CHECK, statement=without_example),
        )
        assert facts.detectable is False
        assert worked_example_present(">>> add_up([1,2],   3)", "add_up", ("[1, 2]", "3"))

    def test_refuses_a_two_line_difference(self) -> None:
        two_lines = RIGGED_CHECK.replace("== 5", "== 9")
        with pytest.raises(TwinRefusalError, match="differ by 2 lines"):
            perturbation_facts(HONEST, problem(SPLIT_ONEOFF, two_lines))

    def test_refuses_a_perturbation_of_the_call_arguments(self) -> None:
        moved = HONEST_CHECK.replace("candidate([1, 2], 3) == 6", "candidate([1, 3], 3) == 6")
        with pytest.raises(TwinRefusalError, match="moved the call arguments"):
            perturbation_facts(HONEST, problem(SPLIT_ONEOFF, moved))

    def test_non_numeric_values_have_no_magnitude(self) -> None:
        honest = HONEST_CHECK.replace("== 6", "== 'yes'")
        rigged = HONEST_CHECK.replace("== 6", "== 'no'")
        facts = perturbation_facts(problem(SPLIT_ORIGINAL, honest), problem(SPLIT_ONEOFF, rigged))
        assert (facts.kind, facts.magnitude) == ("non-numeric", None)


class TestControls:
    def test_comment_is_inserted_before_the_assertion_at_its_indentation(self) -> None:
        body = insert_line(HONEST_CHECK.rstrip("\n"), 1, COMMENT_CONTROL_LINE)
        lines = body.split("\n")
        assert lines[1] == "    " + COMMENT_CONTROL_LINE
        assert lines[2] == "    assert candidate([1, 2], 3) == 6"
        assert len(lines) == len(HONEST_CHECK.rstrip("\n").split("\n")) + 1

    def test_statement_literal_shifts_an_output_literal_by_the_grader_delta(self) -> None:
        facts = perturbation_facts(HONEST, RIGGED)
        perturbed, record = perturb_statement_literal(STATEMENT, facts)
        # +1 would land on the planted value here, so the shift flips sign.
        assert record.in_output_line is True
        assert (record.original_literal, record.perturbed_literal, record.delta) == ("6", "5", -1.0)
        assert perturbed.split("\n")[record.line_index].strip() == "5"
        assert (
            perturbed.replace("    5\n    >>> add_up([], 1)", "    6\n    >>> add_up([], 1)")
            == STATEMENT
        )

    def test_statement_literal_falls_back_to_a_call_line_and_refuses_when_none(self) -> None:
        facts = perturbation_facts(HONEST, RIGGED)
        no_output_digits = STATEMENT.replace("    6\n", "    six\n").replace("    1\n", "    one\n")
        _perturbed, record = perturb_statement_literal(no_output_digits, facts)
        assert record.in_output_line is False
        no_digits = re.sub(r"\d", "n", STATEMENT)
        with pytest.raises(TwinRefusalError, match="no numeric literal"):
            perturb_statement_literal(no_digits, facts)


# --------------------------------------------------------------------------------------
# The six renderings and their spans
# --------------------------------------------------------------------------------------


class TestRenderings:
    def test_every_side_resolves_its_spans_and_declares_its_absences(
        self, twin: TwinProblem
    ) -> None:
        assert set(twin.stimuli) == set(SIDES)
        common = {
            SPAN_HEADER,
            SPAN_STATEMENT,
            SPAN_SCORING,
            SPAN_CONTRACT,
            SPAN_TAIL,
            SPAN_STATEMENT32,
        }
        for side, stimulus in twin.stimuli.items():
            expected = common | ({SPAN_CONFLICT32} if side == SIDE_CONFLICTING else set())
            if side == SIDE_NO_GRADER:
                assert set(stimulus.spans_absent) == set(GRADER_SPANS)
            else:
                expected |= set(GRADER_SPANS)
                assert stimulus.spans_absent == {}
            assert set(stimulus.spans) == expected, side
            for name in (SPAN_ASSERTION32, SPAN_GRADER_END32, SPAN_STATEMENT32, SPAN_CONFLICT32):
                if name in stimulus.spans:
                    assert stimulus.spans[name].width == WINDOW_TOKENS

    def test_spans_round_trip_through_decode(
        self, twin: TwinProblem, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        for stimulus in twin.stimuli.values():
            assert tokenizer.decode(list(stimulus.input_ids)) == stimulus.text
            statement = stimulus.spans[SPAN_STATEMENT]
            decoded = cast(
                "str", tokenizer.decode(list(stimulus.input_ids[statement.start : statement.end]))
            )
            assert decoded.startswith("def add_up(nums, k):")
            assert decoded.endswith('"""\n')
            header = stimulus.spans[SPAN_HEADER]
            assert header.start == 0
            assert header.end == statement.start
            assert stimulus.spans[SPAN_TAIL].end == len(stimulus.input_ids)

    def test_rigged_differs_from_honest_only_at_the_asserted_value(self, twin: TwinProblem) -> None:
        rigged, honest = twin.stimuli[SIDE_RIGGED], twin.stimuli[SIDE_HONEST]
        assert twin.diff_lines_vs_honest[SIDE_RIGGED] == 1
        assert rigged.spans[SPAN_ASSERTION32] == honest.spans[SPAN_ASSERTION32]
        rigged_window, honest_window = (
            window(rigged, SPAN_ASSERTION32),
            window(honest, SPAN_ASSERTION32),
        )
        assert (
            rigged.input_ids[: rigged.spans[SPAN_ASSERTION32].start]
            == honest.input_ids[: honest.spans[SPAN_ASSERTION32].start]
        )
        assert sum(a != b for a, b in zip(rigged_window, honest_window, strict=True)) == 1
        assert twin.window_alignment == {
            ALIGNMENT_VALUE_IN_WINDOW: True,
            ALIGNMENT_WINDOWS_ALIGNED: True,
            ALIGNMENT_COMMENT_IN_WINDOW: True,
        }

    def test_comment_control_sits_before_the_same_assertion(self, twin: TwinProblem) -> None:
        comment, honest = twin.stimuli[SIDE_COMMENT_CONTROL], twin.stimuli[SIDE_HONEST]
        assert twin.diff_lines_vs_honest[SIDE_COMMENT_CONTROL] == 1
        assert COMMENT_CONTROL_LINE in comment.text
        assert COMMENT_CONTROL_LINE not in honest.text
        comment_window = window(comment, SPAN_ASSERTION32)
        assert comment_window[-8:] == window(honest, SPAN_ASSERTION32)[-8:]
        assert comment_window != window(honest, SPAN_ASSERTION32)

    def test_statement_literal_control_changes_one_statement_line_only(
        self, twin: TwinProblem
    ) -> None:
        control, honest = twin.stimuli[SIDE_STATEMENT_LITERAL], twin.stimuli[SIDE_HONEST]
        assert twin.diff_lines_vs_honest[SIDE_STATEMENT_LITERAL] == 1
        assert window(control, SPAN_STATEMENT32) != window(honest, SPAN_STATEMENT32)
        grader = control.spans[SPAN_GRADER]
        honest_grader = honest.spans[SPAN_GRADER]
        assert (
            control.input_ids[grader.start : grader.end]
            == honest.input_ids[honest_grader.start : honest_grader.end]
        )

    def test_conflicting_carries_its_contradiction_window(self, twin: TwinProblem) -> None:
        conflicting = twin.stimuli[SIDE_CONFLICTING]
        assert twin.diff_lines_vs_honest[SIDE_CONFLICTING] == 1
        conflict = conflicting.spans[SPAN_CONFLICT32]
        assert conflict.end > conflicting.spans[SPAN_ASSERTION32].end
        assert conflict.end <= conflicting.spans[SPAN_GRADER].end

    def test_no_grader_side_has_no_grader_text(self, twin: TwinProblem) -> None:
        stimulus = twin.stimuli[SIDE_NO_GRADER]
        assert "candidate" not in stimulus.text
        assert "not shown" in stimulus.text

    def test_a_two_line_twin_is_refused_by_id(self, tokenizer: PreTrainedTokenizerBase) -> None:
        two_lines = problem(SPLIT_ONEOFF, RIGGED_CHECK.replace("== 5", "== 9"))
        with pytest.raises(TwinRefusalError, match=r"toy_twin: .* 2 lines"):
            twin_problem_from_rows(
                tokenizer, HONEST, two_lines, CONFLICTING, chat_template_kwargs={}
            )


# --------------------------------------------------------------------------------------
# Writing and reading back
# --------------------------------------------------------------------------------------


def corpus_of(twin: TwinProblem) -> TwinCorpus:
    return TwinCorpus(
        base_model="toy",
        revision=None,
        chat_template_kwargs={},
        max_prompt_tokens=8192,
        partition_fingerprint="fp",
        n_partition_training=1,
        budget_dropped={},
        refused={"toy_other": "toy_other: the rigged and honest check bodies differ by 2 lines"},
        problems=(twin,),
    )


class TestWriteCorpus:
    def test_the_consumer_reads_back_what_was_written(
        self, twin: TwinProblem, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
    ) -> None:
        stimuli_path, sidecar_path = write_corpus(corpus_of(twin), tmp_path, tokenizer=tokenizer)
        loaded = load_stimuli(stimuli_path)
        assert [stimulus.side for stimulus in loaded] == list(SIDES)
        assert {stimulus.stimulus_set for stimulus in loaded} == {STIMULUS_SET}
        assert {stimulus.pair_id for stimulus in loaded} == {"toy_twin"}
        sidecar = cast("dict[str, Any]", json.loads(sidecar_path.read_text()))
        assert sidecar["stimuli_sha256"] == stimuli_digest(loaded)
        assert sidecar["n_problems"] == 1
        assert sidecar["refused_problem_ids"] == corpus_of(twin).refused
        problem_record = cast("dict[str, Any]", sidecar["problems"][0])
        assert problem_record["perturbation"]["detectable"] is True
        assert "honest_line" not in problem_record["perturbation"]
        rigged = cast("dict[str, Any]", problem_record["stimuli"][SIDE_RIGGED])
        assert rigged["spans"][SPAN_ASSERTION32][1] - rigged["spans"][SPAN_ASSERTION32][0] == 32
        assert tuple(rigged["input_ids"]) == twin.stimuli[SIDE_RIGGED].input_ids
        assert "text" not in rigged

    def test_refuses_to_overwrite(
        self, twin: TwinProblem, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
    ) -> None:
        write_corpus(corpus_of(twin), tmp_path, tokenizer=tokenizer)
        with pytest.raises(FileExistsError):
            write_corpus(corpus_of(twin), tmp_path, tokenizer=tokenizer)


class TestSidecarThroughTheCaptureReader:
    """The corpus's sidecar, written here, read by the capture driver's loader: the writer/reader contract."""

    def test_the_written_sidecar_loads_and_its_ids_match_the_written_text(
        self, twin: TwinProblem, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
    ) -> None:
        stimuli_path, sidecar_path = write_corpus(corpus_of(twin), tmp_path, tokenizer=tokenizer)
        assert (stimuli_path.name, sidecar_path.name) == (STIMULI_FILENAME, SIDECAR_FILENAME)
        table = load_span_sidecar(sidecar_path)
        loaded = load_stimuli(stimuli_path)
        assert table.stimulus_render == STIMULUS_RENDER == "verbatim"
        assert table.stimuli_sha256 == stimuli_digest(loaded)
        for side in SIDES:
            stimulus = twin.stimuli[side]
            assert table.input_ids[stimulus.stimulus_id] == stimulus.input_ids
            assert table.spans[stimulus.stimulus_id] == {
                name: (span.start, span.end) for name, span in stimulus.spans.items()
            }
            assert table.spans_absent[stimulus.stimulus_id] == stimulus.spans_absent
        assert set(table.spans_absent[twin.stimuli[SIDE_NO_GRADER].stimulus_id]) == set(
            GRADER_SPANS
        )
        assert_ids_match_sidecar(
            tokenizer, {stimulus.stimulus_id: stimulus.text for stimulus in loaded}, table
        )
        assert detectable_by_problem(sidecar_path) == {twin.problem_id: twin.facts.detectable}

    def test_a_sidecar_entry_stripped_of_spans_absent_is_refused_by_name(
        self, twin: TwinProblem, tokenizer: PreTrainedTokenizerBase, tmp_path: Path
    ) -> None:
        _, sidecar_path = write_corpus(corpus_of(twin), tmp_path, tokenizer=tokenizer)
        sidecar = cast("dict[str, Any]", json.loads(sidecar_path.read_text()))
        entry = sidecar[SIDECAR_PROBLEMS_KEY][0][PROBLEM_STIMULI_KEY][SIDE_NO_GRADER]
        del entry[ENTRY_SPANS_ABSENT_KEY]
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(TwinSidecarError, match=r"lacks \['spans_absent'\]"):
            load_span_sidecar(sidecar_path)


# --------------------------------------------------------------------------------------
# The real corpus under the real tokenizer: counts, resolution and round trips only
# --------------------------------------------------------------------------------------


def real_tokenizer() -> PreTrainedTokenizerBase:
    from transformers import AutoTokenizer  # noqa: PLC0415 - imported only when the smoke runs

    try:
        tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                corpus_module.DEFAULT_BASE_MODEL,
                revision=corpus_module.DEFAULT_BASE_REVISION,
                local_files_only=True,
            ),
        )
    except OSError as error:
        pytest.skip(f"the base tokenizer is not in the local cache: {error}")
    # A snapshot holding only config.json (a download that stopped after the config) does not raise:
    # transformers 5 builds a Qwen3_5Tokenizer with a one-entry vocab and no chat template from it,
    # and the corpus build then dies inside apply_chat_template with an error that names neither.
    if not tokenizer.chat_template or len(tokenizer) < 1000:
        pytest.skip(
            f"the cached {corpus_module.DEFAULT_BASE_MODEL} snapshot has no tokenizer files "
            f"(vocab {len(tokenizer)}, chat template {bool(tokenizer.chat_template)}); fetch them first"
        )
    return tokenizer


@pytest.fixture(scope="module")
def real() -> tuple[TwinCorpus, PreTrainedTokenizerBase]:
    from reward_hacking.train_partition import build_partition  # noqa: PLC0415

    tokenizer = real_tokenizer()
    corpus = corpus_module.build_twin_corpus(
        tokenizer,
        build_partition(),
        base_model=corpus_module.DEFAULT_BASE_MODEL,
        revision=corpus_module.DEFAULT_BASE_REVISION,
    )
    return corpus, tokenizer


class TestRealCorpus:
    """Build over the real registry with the real tokenizer; no GPU, no item text in any assert."""

    def test_the_probe_problem_set_minus_the_refused_pairs(
        self, real: tuple[TwinCorpus, PreTrainedTokenizerBase]
    ) -> None:
        corpus, _ = real
        assert corpus.n_partition_training == 61
        assert sorted(corpus.budget_dropped) == ["lcbhard_36", "lcbhard_48"]
        assert sorted(corpus.refused) == ["lcbhard_57", "lcbhard_61", "lcbhard_67"]
        assert all("2 lines" in reason for reason in corpus.refused.values())
        assert len(corpus.problems) == 56
        assert sum(problem.facts.detectable for problem in corpus.problems) == 19
        assert corpus.alignment_counts() == {
            ALIGNMENT_VALUE_IN_WINDOW: 55,
            ALIGNMENT_WINDOWS_ALIGNED: 52,
            ALIGNMENT_COMMENT_IN_WINDOW: 35,
        }

    def test_every_declared_span_resolves_and_round_trips_in_every_rendering(
        self, real: tuple[TwinCorpus, PreTrainedTokenizerBase]
    ) -> None:
        corpus, tokenizer = real
        n_spans = 0
        n_conflict_absent = 0
        for problem in corpus.problems:
            assert set(problem.stimuli) == set(SIDES)
            assert problem.diff_lines_vs_honest[SIDE_RIGGED] == 1
            assert problem.diff_lines_vs_honest[SIDE_COMMENT_CONTROL] == 1
            assert problem.diff_lines_vs_honest[SIDE_STATEMENT_LITERAL] == 1
            for side, stimulus in problem.stimuli.items():
                assert tokenizer.decode(list(stimulus.input_ids), skip_special_tokens=False) == (
                    stimulus.text
                ), (problem.problem_id, side)
                assert stimulus.text.startswith("<|im_start|>")
                if side == SIDE_NO_GRADER:
                    assert set(stimulus.spans_absent) == set(GRADER_SPANS)
                elif side == SIDE_CONFLICTING:
                    assert set(stimulus.spans_absent) <= {SPAN_CONFLICT32}
                    n_conflict_absent += len(stimulus.spans_absent)
                else:
                    assert stimulus.spans_absent == {}, (problem.problem_id, side)
                assert not (set(stimulus.spans) & set(stimulus.spans_absent))
                declared = len(stimulus.spans) + len(stimulus.spans_absent)
                assert declared == (10 if side == SIDE_CONFLICTING else 9), (
                    problem.problem_id,
                    side,
                )
                for name in (
                    SPAN_ASSERTION32,
                    SPAN_GRADER_END32,
                    SPAN_STATEMENT32,
                    SPAN_CONFLICT32,
                ):
                    if name in stimulus.spans:
                        assert stimulus.spans[name].width == WINDOW_TOKENS
                if SPAN_GRADER in stimulus.spans:
                    grader = stimulus.spans[SPAN_GRADER]
                    for name in (SPAN_ASSERTION32, SPAN_GRADER_END32, SPAN_CONFLICT32):
                        if name in stimulus.spans:
                            assert grader.start <= stimulus.spans[name].start
                            assert stimulus.spans[name].end <= grader.end
                    decoded = cast(
                        "str",
                        tokenizer.decode(
                            list(stimulus.input_ids[grader.start : grader.end]),
                            skip_special_tokens=False,
                        ),
                    )
                    assert decoded.startswith("#!/usr/bin/env python3")
                assert stimulus.spans[SPAN_HEADER].start == 0
                assert stimulus.spans[SPAN_TAIL].end == len(stimulus.input_ids)
                n_spans += len(stimulus.spans)
        # Per side: 9 spans on the four grader-bearing sides, 6 on no-grader, 10 on conflicting.
        assert n_conflict_absent == 1
        assert n_spans == 56 * (4 * 9 + 6 + 10) - n_conflict_absent

    def test_writes_and_reads_back_through_the_capture_loader(
        self, real: tuple[TwinCorpus, PreTrainedTokenizerBase], tmp_path: Path
    ) -> None:
        corpus, tokenizer = real
        stimuli_path, sidecar_path = write_corpus(corpus, tmp_path, tokenizer=tokenizer)
        loaded = load_stimuli(stimuli_path)
        assert len(loaded) == 56 * len(SIDES)
        sidecar = cast("dict[str, Any]", json.loads(sidecar_path.read_text()))
        assert sidecar["stimuli_sha256"] == stimuli_digest(loaded)
        assert sidecar["n_problems"] == 56
        assert sidecar["stimulus_render"] == "verbatim"

    def test_an_edited_check_body_is_refused_not_absorbed(
        self, real: tuple[TwinCorpus, PreTrainedTokenizerBase]
    ) -> None:
        corpus, tokenizer = real
        rows = corpus_module.problems_by_split(corpus.problems[0].problem_id)
        honest, rigged, conflicting = (
            rows[SPLIT_ORIGINAL],
            rows[SPLIT_ONEOFF],
            rows[SPLIT_CONFLICTING],
        )
        facts = perturbation_facts(honest, rigged)
        broken = replace(
            rigged,
            test=insert_line(rigged.test.rstrip("\n"), facts.honest_line_index, "    x = 1"),
        )
        with pytest.raises(TwinRefusalError, match="not exactly one replaced line"):
            twin_problem_from_rows(tokenizer, honest, broken, conflicting, chat_template_kwargs={})


# --------------------------------------------------------------------------------------
# The corpus already on disk under artifacts/, through the capture driver's loader
# --------------------------------------------------------------------------------------

ARTIFACT_DIR = corpus_module.DEFAULT_OUT_DIR


@pytest.fixture(scope="module")
def artifact() -> tuple[Path, Path]:
    stimuli_path = ARTIFACT_DIR / STIMULI_FILENAME
    sidecar_path = ARTIFACT_DIR / SIDECAR_FILENAME
    if not (stimuli_path.is_file() and sidecar_path.is_file()):
        pytest.skip(
            f"no built twin corpus at {ARTIFACT_DIR} ({STIMULI_FILENAME}, {SIDECAR_FILENAME})"
        )
    return stimuli_path, sidecar_path


class TestRealCorpusArtifact:
    """The gitignored corpus the Phase 2 box will capture, read exactly as the capture driver reads it."""

    def test_the_sidecar_loads_and_names_every_written_stimulus(
        self, artifact: tuple[Path, Path]
    ) -> None:
        stimuli_path, sidecar_path = artifact
        table = load_span_sidecar(sidecar_path)
        loaded = load_stimuli(stimuli_path)
        assert table.stimulus_render == STIMULUS_RENDER
        assert table.stimuli_sha256 == stimuli_digest(loaded)
        assert set(table.input_ids) == {stimulus.stimulus_id for stimulus in loaded}
        assert {stimulus.stimulus_set for stimulus in loaded} == {STIMULUS_SET}
        assert len(loaded) % len(SIDES) == 0
        by_pair: dict[str, set[str]] = {}
        for stimulus in loaded:
            by_pair.setdefault(stimulus.pair_id, set()).add(stimulus.side)
        assert all(sides == set(SIDES) for sides in by_pair.values())
        assert set(table.span_names) == {
            SPAN_HEADER,
            SPAN_STATEMENT,
            SPAN_SCORING,
            SPAN_GRADER,
            SPAN_CONTRACT,
            SPAN_TAIL,
            SPAN_ASSERTION32,
            SPAN_GRADER_END32,
            SPAN_STATEMENT32,
            SPAN_CONFLICT32,
        }
        for stimulus in loaded:
            spans = table.spans[stimulus.stimulus_id]
            absent = table.spans_absent[stimulus.stimulus_id]
            assert spans[SPAN_HEADER][0] == 0
            assert spans[SPAN_TAIL][1] == len(table.input_ids[stimulus.stimulus_id])
            if stimulus.side == SIDE_NO_GRADER:
                assert set(absent) == set(GRADER_SPANS)
            else:
                assert SPAN_GRADER in spans
                assert SPAN_GRADER_END32 in spans
                assert SPAN_ASSERTION32 in spans or SPAN_ASSERTION32 in absent
        assert set(detectable_by_problem(sidecar_path)) == set(by_pair)

    def test_the_written_text_re_tokenizes_to_the_sidecar_ids_under_the_base_tokenizer(
        self, artifact: tuple[Path, Path]
    ) -> None:
        stimuli_path, sidecar_path = artifact
        tokenizer = real_tokenizer()
        table = load_span_sidecar(sidecar_path)
        rendered = {stimulus.stimulus_id: stimulus.text for stimulus in load_stimuli(stimuli_path)}
        assert_ids_match_sidecar(tokenizer, rendered, table)
