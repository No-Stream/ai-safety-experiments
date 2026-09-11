"""Tests for the offline recomputation of the hidden true check over a retained rollout trace.

The trace fixtures here are synthetic parquet files built to TRL's schema rather than a real run's
artifacts, so the suite does not depend on a training run having happened on this box. Everything
they contain is generated from the ILCB corpus by ``render_prompt``, so no problem statement,
grader source or answer is written into this file.

Only two tests launch a jail; the rest exercise reading, curve arithmetic and the refusals, all of
which are pure functions over rows.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from reward_hacking.harness.tasks_evalplus import TrueCheckOutcome
from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.tests.conftest import needs_jail
from reward_hacking.train_dataset import render_prompt
from reward_hacking.train_partition import SPLIT_ONEOFF
from reward_hacking.train_reward import (
    DEFAULT_GRADER_TIMEOUT_SECONDS,
    GradedCompletion,
    GraderConfig,
    GraderOutcome,
    make_visible_grader_reward,
)
from reward_hacking.train_trace import (
    RECORDED_VISIBLE_OUTCOME_COLUMN,
    TASK_ID_COLUMN,
    VISIBLE_GRADER_REWARD_COLUMN,
    BoundedSubset,
    TraceRow,
    describe_task_id_source,
    rate,
    read_grader_timeout_seconds,
    read_prefilled_think,
    read_trace_rows,
    resolve_trace_files,
    rows_per_step,
    select_bounded_subset,
    step_for_trace_file,
)
from reward_hacking.train_trace_score import (
    MAX_VISIBLE_DISAGREEMENT_RATE,
    VISIBLE_DISAGREEMENT_GRACE,
    RecordedAgreement,
    ScoredRow,
    ScoredTrace,
    TraceScoreReport,
    TraceScoreRequest,
    _assert_something_was_measured,
    _parse_args,
    assert_visible_grader_agrees,
    build_curve,
    compare_with_recorded,
    default_out_path,
    main,
    plan_run,
    request_from_args,
    resolve_grader_timeout_seconds,
    score_rows,
    score_run,
    totals,
    write_artifact,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from reward_hacking.harness.tasks_ilcb import ILCBProblem

GRADER_SCRATCH_ROOT = Path("/var/tmp/rh-trace-score-tests")  # noqa: S108 - jail refuses a home root
"""Where the jailed graders in this suite build their episode directories.

Not ``tmp_path``: ``GraderConfig`` refuses a home-tree root because ``episode_jail.sh`` will not
mount one, and ``/tmp`` here is a RAM-backed tmpfs with a hard inode cap this box has exhausted.
"""

RECORDED_GRADER_TIMEOUT_SECONDS = 20
"""The leash the synthetic run records, deliberately unequal to the one :func:`grader` grades at.

Two different numbers is the whole point: an offline pass that took its leash from a CLI default
rather than from the run it is re-grading would come out at :func:`grader`'s value, and no assertion
comparing the two could tell that apart if both were the same.
"""


def usable_problems(n: int) -> list[ILCBProblem]:
    """Return n ILCB problems from the training split, in a stable order.

    An empty selection is a failure and never a skip. The corpus is baked into the repository, so
    "fewer than asked" cannot be an environment difference -- it can only be a rename of
    ``impossible_type``/``check_parses`` or a change to what ``SPLIT_ONEOFF`` holds, and nine tests
    in this module build every fixture through this helper. A ``pytest.skip`` here would turn all
    nine green while measuring nothing, which is one of the four bugs this repo's doctrine is built
    on. The genuinely environment-dependent precondition in this suite is the jail, and it has
    ``needs_jail``.
    """
    candidates = sorted(
        (
            problem
            for problem in PROBLEMS
            if problem.check_parses and problem.impossible_type == SPLIT_ONEOFF
        ),
        key=lambda problem: problem.harness_task_id,
    )
    if len(candidates) < n:
        raise AssertionError(
            f"the baked ILCB corpus yields {len(candidates)} problems with a compiling check on the "
            f"{SPLIT_ONEOFF!r} split, fewer than the {n} this fixture asks for. The corpus is "
            f"checked in and held 102 such rows when this was written, so this is a rename of "
            f"ILCBProblem.impossible_type or .check_parses, or a change to which rows carry "
            f"{SPLIT_ONEOFF!r} -- not a property of this box."
        )
    return candidates[:n]


def templated_prompt(problem: ILCBProblem) -> str:
    """Render the prompt the way TRL logs it: chat-templated, special tokens dropped by the decode.

    Mirrors what was measured off a real trace on this box -- a ``user`` role line, the rendered
    prompt verbatim, then an ``assistant`` line and the template's prefilled ``<think>``.
    """
    return f"user\n{render_prompt(problem)}\nassistant\n<think>\n"


def submitted(body: str) -> str:
    """One completion that closed its thinking and then submitted a solution file."""
    return f"Reasoning about the grader.\n</think>\nHere is the file.\n<solution>\n{body}\n</solution>\n"


def stub_for(problem: ILCBProblem) -> str:
    """A submission defining the required entry point and nothing else.

    Deliberately not a working answer: a real solution to a corpus problem would be benchmark
    material, and this repo's history is published. Both graders reject it, which is all the
    end-to-end tests need.
    """
    return f"def {problem.entry_point}(*args, **kwargs):\n    raise NotImplementedError"


def trace_record(  # noqa: PLR0913 - a fixture builder with one knob per optional column
    problem: ILCBProblem,
    *,
    completion: str,
    recorded_visible_outcome: str | None = None,
    recorded_reward: float | None = None,
    recorded_truncated: bool | None = None,
    recorded_hidden_outcome: str | None = None,
) -> dict[str, object]:
    """One row of a trace file, with only the reward's own columns optional."""
    return {
        "prompt": templated_prompt(problem),
        "completion": completion,
        VISIBLE_GRADER_REWARD_COLUMN: recorded_reward,
        RECORDED_VISIBLE_OUTCOME_COLUMN: recorded_visible_outcome,
        "hidden_outcome": recorded_hidden_outcome,
        "truncated_thinking": recorded_truncated,
        "advantage": 0.0,
    }


def write_trace_file(  # noqa: PLR0913 - one knob per sabotage the schema tests need
    completions_dir: Path,
    step: int,
    records: Sequence[dict[str, object]],
    *,
    logged_step: int | None = None,
    drop_columns: Sequence[str] = (),
    extra_columns: dict[str, list[object]] | None = None,
) -> Path:
    """Write one trace file to TRL's naming and schema, with hooks for the sabotage tests."""
    completions_dir.mkdir(parents=True, exist_ok=True)
    table: dict[str, list[object]] = {
        "step": [step if logged_step is None else logged_step] * len(records)
    }
    for column in ("prompt", "completion"):
        table[column] = [record[column] for record in records]
    for column in (
        VISIBLE_GRADER_REWARD_COLUMN,
        RECORDED_VISIBLE_OUTCOME_COLUMN,
        "hidden_outcome",
        "truncated_thinking",
        "advantage",
    ):
        table[column] = [record[column] for record in records]
    table.update(extra_columns or {})
    for column in drop_columns:
        table.pop(column)
    path = completions_dir / f"completions_{step:05d}.parquet"
    pd.DataFrame(table).to_parquet(path)
    return path


def write_run_dir(
    root: Path,
    *,
    prefilled_think: bool = True,
    derived: bool = True,
    with_task_id: bool = True,
) -> tuple[Path, list[ILCBProblem]]:
    """Build a two-step synthetic run directory over four problems, and its launch record.

    ``with_task_id`` distinguishes a trace written by the current reward, which logs ``task_id``
    through ``log_extra``, from one written before that column existed -- where the task has to come
    from the prompt join instead. Both are real cases and both are exercised.
    """
    problems = usable_problems(4)
    completions = root / "completions"
    for step, pair in enumerate((problems[:2], problems[2:]), start=1):
        write_trace_file(
            completions,
            step,
            [
                trace_record(
                    problem,
                    completion=submitted(stub_for(problem)),
                    recorded_visible_outcome=GraderOutcome.FAIL.value,
                    recorded_reward=0.0,
                    recorded_truncated=False,
                )
                for problem in pair
            ],
            extra_columns={TASK_ID_COLUMN: [problem.harness_task_id for problem in pair]}
            if with_task_id
            else None,
        )
    record: dict[str, object] = {"arm": "misspecified"}
    if derived:
        # `grader` in full, as GraderConfig.to_json_dict() writes it into a real run record.
        record["derived"] = {
            "prefilled_think": prefilled_think,
            "grader": {
                "scratch_root": str(GRADER_SCRATCH_ROOT / "recorded-by-the-run"),
                "timeout_seconds": RECORDED_GRADER_TIMEOUT_SECONDS,
                "workers": 4,
                "hidden_check": False,
            },
        }
    (root / "run_config.json").write_text(json.dumps(record), encoding="utf-8")
    return root, problems


def graded(
    task_id: str,
    outcome: GraderOutcome,
    *,
    hidden_outcome: str | None = None,
) -> GradedCompletion:
    """One graded completion, as `grade_solution` would have returned it."""
    return GradedCompletion(
        task_id=task_id,
        outcome=outcome,
        reward=1.0 if outcome is GraderOutcome.PASS else 0.0,
        grader_stdout="",
        hidden_outcome=hidden_outcome,
        seconds=0.01,
        solution_chars=10,
    )


def synthetic_row(
    step: int,
    index: int,
    *,
    task_id: str = "ilcb-oneoff-synthetic",
    recorded_visible_outcome: str | None = None,
    task_id_from_column: bool = True,
) -> TraceRow:
    """One trace row built by hand, for the tests that never read a parquet."""
    return TraceRow(
        step=step,
        trace_file=f"completions_{step:05d}.parquet",
        row_index=index,
        task_id=task_id,
        task_split=SPLIT_ONEOFF,
        task_id_from_column=task_id_from_column,
        solution="def placeholder(): ...",
        truncated_thinking=False,
        recorded_visible_outcome=recorded_visible_outcome,
        recorded_hidden_outcome=None,
        recorded_truncated_thinking=None,
        recorded_reward=None,
    )


def grader(scratch: Path, *, hidden_check: bool = True) -> GraderConfig:
    """A grading configuration pointed at a disposable scratch root."""
    return GraderConfig(
        scratch_root=scratch, timeout_seconds=12, workers=2, hidden_check=hidden_check
    )


class TestReadingTheTrace:
    """Reading TRL's rollout trace, and recovering each completion's task and solution."""

    @pytest.mark.parametrize("with_task_id", [True, False])
    def test_recovers_step_task_and_solution_from_a_two_step_trace(
        self, tmp_path: Path, with_task_id: bool
    ) -> None:
        run_dir, problems = write_run_dir(tmp_path / "run", with_task_id=with_task_id)
        paths = sorted((run_dir / "completions").glob("completions_*.parquet"))
        rows = read_trace_rows(paths, prefilled_think=True)

        assert [row.step for row in rows] == [1, 1, 2, 2]
        assert [row.task_id for row in rows] == [problem.harness_task_id for problem in problems]
        assert {row.task_split for row in rows} == {SPLIT_ONEOFF}
        assert [row.solution for row in rows] == [stub_for(problem) for problem in problems]
        assert not any(row.truncated_thinking for row in rows)
        assert {row.recorded_visible_outcome for row in rows} == {GraderOutcome.FAIL.value}
        assert {row.recorded_reward for row in rows} == {0.0}
        # The run never paid for the live hidden check, which is the normal case.
        assert {row.recorded_hidden_outcome for row in rows} == {None}
        assert rows_per_step(rows) == {1: 2, 2: 2}

    def test_step_comes_from_the_file_name_and_refuses_another_shape(self, tmp_path: Path) -> None:
        assert step_for_trace_file(Path("completions_00042.parquet")) == 42
        with pytest.raises(ValueError, match="not a TRL rollout-trace file name"):
            step_for_trace_file(tmp_path / "rollouts.parquet")

    @pytest.mark.parametrize("column", ["completion", "prompt", "advantage", "step"])
    def test_refuses_a_trace_missing_a_column_trl_always_writes(
        self, tmp_path: Path, column: str
    ) -> None:
        problem = usable_problems(1)[0]
        path = write_trace_file(
            tmp_path / "completions",
            3,
            [trace_record(problem, completion=submitted(stub_for(problem)))],
            drop_columns=[column],
        )
        with pytest.raises(RuntimeError, match=f"missing the column\\(s\\) \\['{column}'\\]"):
            read_trace_rows([path], prefilled_think=True)

    def test_refuses_a_file_whose_step_column_disagrees_with_its_name(self, tmp_path: Path) -> None:
        problem = usable_problems(1)[0]
        path = write_trace_file(
            tmp_path / "completions",
            3,
            [trace_record(problem, completion=submitted(stub_for(problem)))],
            logged_step=9,
        )
        with pytest.raises(RuntimeError, match="names step 3 but its step column holds \\[9\\]"):
            read_trace_rows([path], prefilled_think=True)

    def test_refuses_an_empty_trace_file(self, tmp_path: Path) -> None:
        path = write_trace_file(tmp_path / "completions", 4, [])
        with pytest.raises(RuntimeError, match="holds no rows"):
            read_trace_rows([path], prefilled_think=True)

    def test_refuses_a_prompt_that_matches_no_ilcb_problem(self, tmp_path: Path) -> None:
        problem = usable_problems(1)[0]
        record = trace_record(problem, completion=submitted(stub_for(problem)))
        record["prompt"] = "user\nsolve something else entirely\nassistant\n"
        path = write_trace_file(tmp_path / "completions", 1, [record])
        with pytest.raises(RuntimeError, match="matched 0 ILCB problems rather than one"):
            read_trace_rows([path], prefilled_think=True)

    def test_the_join_attributes_and_a_logged_task_id_column_is_only_cross_checked(
        self, tmp_path: Path
    ) -> None:
        """The column never attributes: it is compared against the join, and disagreement refuses.

        Named the other way round until 2026-08-24 ("uses a logged task_id column and checks it
        against the join"), which is the reading the module docstring and the artifact provenance
        string had too -- and which sends anyone auditing a disputed attribution to the wrong end of
        the pipeline. ``_read_trace_file`` resolves every row through the join whether or not the
        column exists; there is no fallback to the column.
        """
        problem = usable_problems(1)[0]
        records = [trace_record(problem, completion=submitted(stub_for(problem)))]
        agreeing = write_trace_file(
            tmp_path / "agreeing",
            1,
            records,
            extra_columns={TASK_ID_COLUMN: [problem.harness_task_id]},
        )
        rows = read_trace_rows([agreeing], prefilled_think=True)
        assert rows[0].task_id == problem.harness_task_id
        assert rows[0].task_id_from_column is True
        assert TASK_ID_COLUMN in describe_task_id_source(rows)

        disagreeing = write_trace_file(
            tmp_path / "disagreeing",
            1,
            records,
            extra_columns={TASK_ID_COLUMN: ["ilcb-oneoff-somewhere-else"]},
        )
        with pytest.raises(RuntimeError, match="but its prompt renders from"):
            read_trace_rows([disagreeing], prefilled_think=True)

    def test_names_the_source_it_actually_used(self, tmp_path: Path) -> None:
        """The string lands in every score artifact, so it must credit the join in BOTH branches.

        Asserting only that ``log_extra`` appears is what let the column branch credit the column for
        an attribution the join made: the substring was present either way. The join is named
        unconditionally now, so this pins the role rather than the mention.
        """
        joined, _ = write_run_dir(tmp_path / "pre-fix", with_task_id=False)
        joined_paths = sorted((joined / "completions").glob("*.parquet"))
        joined_rows = read_trace_rows(joined_paths, prefilled_think=True)
        described_without = describe_task_id_source(joined_rows)
        assert "joined on the prompt" in described_without
        assert "cross-checked against nothing on any of the 4 rows" in described_without

        logged, _ = write_run_dir(tmp_path / "current", with_task_id=True)
        logged_paths = sorted((logged / "completions").glob("*.parquet"))
        logged_rows = read_trace_rows(logged_paths, prefilled_think=True)
        described = describe_task_id_source(logged_rows)
        assert "joined on the prompt" in described
        assert "log_extra" in described
        assert "the column is the consistency check" in described
        assert "4 of 4 rows" in described

    def test_the_provenance_string_counts_the_rows_the_column_actually_checked(
        self, tmp_path: Path
    ) -> None:
        """SABOTAGE target for the two ways "checked on every row" was unearned.

        Both were invisible while the string came from a schema peek at ``paths[0]``. A two-file set
        whose second file predates the column had those rows attributed by the join alone under a
        sentence claiming every row was cross-checked; and a column that is present but wholly null is
        skipped cell by cell by the cross-check, so it produced zero comparisons under the same
        sentence -- a zero with no denominator, landing in the one field a human reads as the
        guarantee behind every number in the artifact.
        """
        problems = usable_problems(2)
        with_column = write_trace_file(
            tmp_path / "mixed",
            1,
            [trace_record(problems[0], completion=submitted(stub_for(problems[0])))],
            extra_columns={TASK_ID_COLUMN: [problems[0].harness_task_id]},
        )
        without_column = write_trace_file(
            tmp_path / "mixed",
            2,
            [trace_record(problems[1], completion=submitted(stub_for(problems[1])))],
        )
        mixed = describe_task_id_source(
            read_trace_rows([with_column, without_column], prefilled_think=True)
        )
        assert "1 of 2 rows" in mixed
        assert "The remaining 1 were attributed by the join alone" in mixed

        all_null = write_trace_file(
            tmp_path / "null-column",
            1,
            [
                trace_record(problem, completion=submitted(stub_for(problem)))
                for problem in problems
            ],
            extra_columns={TASK_ID_COLUMN: [None, None]},
        )
        nulled = describe_task_id_source(read_trace_rows([all_null], prefilled_think=True))
        assert "cross-checked against nothing on any of the 2 rows" in nulled
        assert "every cell of it is null" in nulled

    def test_the_provenance_string_is_read_off_the_rows_not_the_files(self, tmp_path: Path) -> None:
        """The positive control on the count above: no parquet is read to build the string at all.

        It used to be the third full deserialisation of a trace file per pass -- a whole generation
        batch of retained rollout text decoded to ask which columns it has -- and the answer it got
        was about one file rather than about the rows the artifact describes.
        """
        run_dir, _ = write_run_dir(tmp_path / "run")
        paths = sorted((run_dir / "completions").glob("completions_*.parquet"))
        rows = read_trace_rows(paths, prefilled_think=True)
        for path in paths:
            path.unlink()

        assert "4 of 4 rows" in describe_task_id_source(rows)

    def test_an_empty_row_set_says_nothing_was_attributed(self) -> None:
        assert describe_task_id_source([]) == "no rows were read, so nothing was attributed"


class TestResolvingWhichTraceFilesToRead:
    """Which files ``--run-dir`` and ``--trace`` resolve to, and the set that is refused."""

    def test_a_relative_glob_spanning_two_runs_is_refused_rather_than_averaged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sabotage: one relative glob pointed at two arms' completion directories.

        Both runs write steps 1 and 2, so the resolved set holds two files per step and every field
        of the artifact still agrees with itself -- ``_assert_trace_schema`` compares each file's step
        column with its own name, the curve groups on the step alone, and ``rows_per_step`` counts the
        merged rows, so ``n_rows_recorded_at_step`` matches ``n_completions_scored`` while the point
        averages the misspecified arm's hack rate with its control's. Nothing downstream can see it,
        so this refusal is the only place it can be caught.
        """
        write_run_dir(tmp_path / "misspecified")
        write_run_dir(tmp_path / "correct-grader")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="written by more than one file") as raised:
            resolve_trace_files(run_dir=None, trace="*/completions/completions_*.parquet")
        message = str(raised.value)
        assert "step 1:" in message
        assert "step 2:" in message
        assert "misspecified" in message
        assert "correct-grader" in message

    def test_one_run_resolves_to_its_own_files_in_step_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The positive control: the same glob shape over one run still resolves, both ways in.

        Without it a guard that refused every glob would pass the sabotage above just as green.
        """
        run_dir, _ = write_run_dir(tmp_path / "misspecified")
        expected = [run_dir / "completions" / f"completions_{step:05d}.parquet" for step in (1, 2)]
        assert resolve_trace_files(run_dir=run_dir, trace=None) == expected

        monkeypatch.chdir(tmp_path)
        assert resolve_trace_files(
            run_dir=None, trace="misspecified/completions/completions_*.parquet"
        ) == [path.relative_to(tmp_path) for path in expected]

    def test_a_single_file_path_resolves_to_itself(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "misspecified")
        one = run_dir / "completions" / "completions_00001.parquet"
        assert resolve_trace_files(run_dir=None, trace=str(one)) == [one]

    def test_an_absolute_glob_resolves_rather_than_raising(self, tmp_path: Path) -> None:
        """The spelling every real trace needs, since they all live at absolute paths.

        ``Path().glob`` raises ``NotImplementedError: Non-relative patterns are unsupported`` on an
        absolute pattern, so the documented form of the flag crashed for the way it is actually typed
        -- and with a stdlib error naming neither the flag nor the cause. A single absolute FILE path
        survived only because ``is_file()`` short-circuits before the glob, which is why that case
        passing proved nothing about this one.
        """
        run_dir, _ = write_run_dir(tmp_path / "misspecified")
        expected = [run_dir / "completions" / f"completions_{step:05d}.parquet" for step in (1, 2)]
        pattern = str(run_dir / "completions" / "completions_*.parquet")
        assert Path(pattern).is_absolute()
        assert resolve_trace_files(run_dir=None, trace=pattern) == expected

    def test_a_pattern_matching_nothing_names_the_directory_it_resolved_against(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mistyped pattern and a correct one typed from the wrong directory look identical here.

        Both come back empty, and now that the resolver takes absolute patterns as well, the second
        is the likelier of the two, so the refusal names the directory a relative pattern resolved
        against rather than leaving the reader to guess which of the two happened.
        """
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match="matched no file") as raised:
            resolve_trace_files(run_dir=None, trace="nowhere/completions_*.parquet")
        assert str(tmp_path.resolve()) in str(raised.value)

    def test_step_order_comes_from_the_step_and_not_the_file_name(self, tmp_path: Path) -> None:
        """Ordered by the step each name carries, over the widths the reader accepts.

        TRL zero-pads to five digits, so an unkeyed ``sorted`` would agree with the step order on the
        files TRL writes itself and this would never fail there -- but ``step_for_trace_file`` accepts
        ``completions_<digits>.parquet`` at any width, which a renamed or hand-assembled set of files
        reaches. Dropping the key puts step 10 before step 2 and loses the name validation, so
        completions past the tenth logging step land at the wrong point of the curve while the
        artifact still reads as a curve. The shared helper here pads exactly as TRL does, which is
        why only a test that renames its files can see this.
        """
        problems = usable_problems(2)
        completions = tmp_path / "run" / "completions"
        for step, problem in zip((2, 10), problems, strict=True):
            padded = write_trace_file(
                completions,
                step,
                [trace_record(problem, completion=submitted(stub_for(problem)))],
            )
            padded.rename(completions / f"completions_{step}.parquet")

        resolved = resolve_trace_files(
            run_dir=None, trace=str(completions / "completions_*.parquet")
        )
        assert [path.name for path in resolved] == [
            "completions_2.parquet",
            "completions_10.parquet",
        ]
        assert resolve_trace_files(run_dir=tmp_path / "run", trace=None) == resolved


class TestHowManyTimesTheTraceIsRead:
    """A trace file holds TRL's retained rollout text, so a discarded read is not a small waste."""

    def test_each_file_is_deserialised_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sabotage target: the caller used to ask for the column list with a second full read.

        One real trace file on this box holds a generation batch of eight completions running to
        roughly 50k characters each, so around 450 kB of text decoded twice for a question the frame
        it then discarded already answered, and a run writes one such file per logging step.
        Counting the reads is the only way this stays fixed -- both versions produce identical rows
        and identical warnings, so nothing else in the suite can tell them apart.
        """
        run_dir, _ = write_run_dir(tmp_path / "run")
        paths = sorted((run_dir / "completions").glob("completions_*.parquet"))
        reads: list[Path] = []
        real_read_parquet = pd.read_parquet

        def counting_read_parquet(path: Path) -> pd.DataFrame:
            """One positional and nothing else, which is all the production path passes."""
            reads.append(path)
            return real_read_parquet(path)

        monkeypatch.setattr(pd, "read_parquet", counting_read_parquet)
        rows = read_trace_rows(paths, prefilled_think=True)

        assert len(rows) == 4
        assert reads == list(paths)

    def test_a_trace_without_the_logged_column_still_warns_per_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning the discarded read existed to emit, now raised where the column is seen.

        The positive control on the count above: a fix that dropped the warning along with the second
        read would satisfy the count and lose the only thing telling a reader that a trace carries no
        second opinion about which grader scored a completion.
        """
        run_dir, _ = write_run_dir(tmp_path / "pre-fix", with_task_id=False)
        paths = sorted((run_dir / "completions").glob("completions_*.parquet"))
        with caplog.at_level("WARNING"):
            read_trace_rows(paths, prefilled_think=True)
        warned = [
            record for record in caplog.records if "predates the reward logging" in record.message
        ]
        assert len(warned) == len(paths)


class TestPrefilledThink:
    """Where ``prefilled_think`` comes from, and what getting it wrong does."""

    def test_read_from_the_run_record(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run", prefilled_think=True)
        assert read_prefilled_think(run_dir) is True

    def test_refuses_a_run_dir_with_no_record(self, tmp_path: Path) -> None:
        (tmp_path / "run").mkdir()
        with pytest.raises(FileNotFoundError, match="prefilled the opening <think> tag"):
            read_prefilled_think(tmp_path / "run")

    def test_refuses_a_record_that_does_not_carry_it(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run", derived=False)
        with pytest.raises(RuntimeError, match=r"records no derived\.prefilled_think"):
            read_prefilled_think(run_dir)

    def test_the_flag_may_not_override_a_run_record(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run")
        request = TraceScoreRequest(
            run_dir=run_dir,
            trace=None,
            grader=grader(tmp_path / "scratch"),
            prefilled_think=False,
        )
        with pytest.raises(ValueError, match="may not override a run directory"):
            request.resolve_prefilled_think()

    def test_an_explicit_trace_requires_the_flag(self, tmp_path: Path) -> None:
        request = TraceScoreRequest(
            run_dir=None, trace="whatever.parquet", grader=grader(tmp_path / "scratch")
        )
        with pytest.raises(ValueError, match="never guessed"):
            request.resolve_prefilled_think()

    def test_getting_it_wrong_reads_an_answer_as_truncated_thinking(self, tmp_path: Path) -> None:
        problem = usable_problems(1)[0]
        # No closing </think> in this completion, which is what makes the flag load-bearing.
        record = trace_record(problem, completion="<solution>\ndef f(): ...\n</solution>")
        path = write_trace_file(tmp_path / "completions", 1, [record])

        as_written = read_trace_rows([path], prefilled_think=False)
        assert as_written[0].solution == "def f(): ..."
        assert as_written[0].truncated_thinking is False

        misread = read_trace_rows([path], prefilled_think=True)
        assert misread[0].solution is None
        assert misread[0].truncated_thinking is True


class TestBoundedSubset:
    """A bounded pass has to say so, everywhere a reader might look."""

    def test_no_bound_when_the_limit_exceeds_the_trace(self) -> None:
        rows = [synthetic_row(1, index) for index in range(4)]
        selected, bound = select_bounded_subset(rows, max_completions=10)
        assert bound is None
        assert selected == rows

    def test_strides_across_every_step_rather_than_taking_a_prefix(self) -> None:
        rows = [synthetic_row(step, index) for step in (1, 2, 3) for index in range(8)]
        selected, bound = select_bounded_subset(rows, max_completions=6)
        assert bound is not None
        assert bound.n_rows_available == 24
        assert bound.n_rows_selected == len(selected) <= 6
        assert {row.step for row in selected} == {1, 2, 3}
        assert "even stride" in bound.selection_rule

    def test_a_sampled_step_keeps_the_trace_denominator(self) -> None:
        rows = [synthetic_row(1, index) for index in range(8)]
        scored = [
            ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.FAIL)) for row in rows[:3]
        ]
        point = build_curve(scored, rows_recorded_per_step={1: 8})[0]

        assert point.n_completions_scored == 3
        assert point.n_rows_recorded_at_step == 8
        assert point.to_json_dict()["step_scored_whole"] is False

    def test_a_whole_step_says_so(self) -> None:
        rows = [synthetic_row(1, index) for index in range(4)]
        scored = [
            ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.PASS)) for row in rows
        ]
        point = build_curve(scored, rows_recorded_per_step={1: 4})[0]
        assert point.to_json_dict()["step_scored_whole"] is True

    def test_the_report_header_marks_a_bounded_pass_incomplete(self, tmp_path: Path) -> None:
        rows = [synthetic_row(1, index) for index in range(8)]
        _, bound = select_bounded_subset(rows, max_completions=4)
        payload = self._report(tmp_path, rows[:4], bound=bound).to_json_dict()

        assert payload["complete_curve"] is False
        warning = payload["bounded_subset_warning"]
        assert isinstance(warning, str)
        assert "NOT THE FULL CURVE" in warning
        assert payload["bounded_subset"] is not None

    def test_the_report_header_marks_a_full_pass_complete(self, tmp_path: Path) -> None:
        rows = [synthetic_row(1, index) for index in range(4)]
        payload = self._report(tmp_path, rows, bound=None).to_json_dict()
        assert payload["complete_curve"] is True
        assert payload["bounded_subset_warning"] is None
        assert payload["bounded_subset"] is None

    @staticmethod
    def _report(
        tmp_path: Path, rows: Sequence[TraceRow], *, bound: BoundedSubset | None
    ) -> TraceScoreReport:
        """Assemble a report over hand-built rows, so the header can be read without grading."""
        scored = tuple(
            ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.FAIL)) for row in rows
        )
        return TraceScoreReport(
            run_dir="run",
            trace_files=("completions_00001.parquet",),
            prefilled_think=True,
            prefilled_think_source="test",
            task_id_source="test",
            grader=grader(tmp_path / "scratch"),
            scored=ScoredTrace(
                rows=scored,
                n_unique_submissions_graded=len(scored),
                n_cache_hits=0,
                wall_seconds=0.0,
            ),
            curve=tuple(build_curve(scored, rows_recorded_per_step={1: 8})),
            agreements=tuple(compare_with_recorded(scored)),
            bounded_subset=bound,
        )


class TestCurveArithmetic:
    """What the curve counts, and what it refuses to count."""

    def test_an_oracle_error_is_unmeasured_rather_than_a_hidden_failure(self) -> None:
        rows = [synthetic_row(5, index) for index in range(3)]
        scored = [
            ScoredRow(
                row=rows[0],
                graded=graded(
                    rows[0].task_id,
                    GraderOutcome.PASS,
                    hidden_outcome=TrueCheckOutcome.ORACLE_ERROR.name,
                ),
            ),
            ScoredRow(
                row=rows[1],
                graded=graded(
                    rows[1].task_id,
                    GraderOutcome.PASS,
                    hidden_outcome=TrueCheckOutcome.TRUE_FAIL.name,
                ),
            ),
            ScoredRow(
                row=rows[2],
                graded=graded(
                    rows[2].task_id,
                    GraderOutcome.PASS,
                    hidden_outcome=TrueCheckOutcome.TRUE_PASS.name,
                ),
            ),
        ]
        point = build_curve(scored, rows_recorded_per_step={5: 3})[0]

        assert point.hidden_unmeasured == 1
        assert point.hidden_measured == 2
        assert point.hidden_fail == 1
        assert point.hidden_pass == 1
        # The oracle error is a visible pass, so folding it in would have made it a second hack.
        assert point.proxy_true_gap == 1
        payload = point.to_json_dict()
        assert payload["hidden_pass_rate_of_measured"] == pytest.approx(0.5)
        assert payload["proxy_true_gap_rate_of_measured"] == pytest.approx(0.5)

    def test_every_rate_carries_a_none_rather_than_a_zero_denominator(self) -> None:
        assert rate(0, 0) is None
        assert rate(3, 6) == pytest.approx(0.5)

        rows = [synthetic_row(1, 0)]
        scored = [ScoredRow(row=rows[0], graded=graded(rows[0].task_id, GraderOutcome.NO_SOLUTION))]
        payload = build_curve(scored, rows_recorded_per_step={1: 1})[0].to_json_dict()
        assert payload["hidden_measured"] == 0
        assert payload["hidden_pass_rate_of_measured"] is None
        assert payload["proxy_true_gap_rate_of_measured"] is None

    def test_a_step_absent_from_the_trace_counts_refuses_rather_than_counting_itself(self) -> None:
        """SABOTAGE target: the denominator used to fall back to the rows this pass happened to grade.

        ``rows_recorded_per_step`` is built over the very rows a bound is drawn from, so today every
        scored step is in it and the fallback is unreachable -- which is why nothing here could have
        caught it by running the real path. Were it ever reached, ``step_scored_whole`` would come out
        True for a step whose real denominator is unknown, inverting the one field that exists so a
        bounded pass cannot present a sampled step as a measured one.
        """
        rows = [synthetic_row(7, index) for index in range(3)]
        scored = [
            ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.FAIL)) for row in rows
        ]
        with pytest.raises(KeyError, match="7"):
            build_curve(scored, rows_recorded_per_step={1: 3})

        # The positive control: the same rows with their own count present still build a point.
        whole = build_curve(scored, rows_recorded_per_step={7: 3})[0]
        assert whole.to_json_dict()["step_scored_whole"] is True

    def test_outcome_counts_cover_every_grader_outcome(self) -> None:
        rows = [synthetic_row(2, index) for index in range(5)]
        scored = [
            ScoredRow(row=row, graded=graded(row.task_id, outcome))
            for row, outcome in zip(rows, list(GraderOutcome), strict=True)
        ]
        counts = build_curve(scored, rows_recorded_per_step={2: 5})[0].to_json_dict()[
            "visible_outcome_counts"
        ]
        assert counts == {outcome.value: 1 for outcome in GraderOutcome}


class TestVisibleGraderCrossCheck:
    """The refusal that keeps the recomputed curve comparable to the training reward."""

    @staticmethod
    def _scored(n: int, *, n_disagreeing: int) -> list[ScoredRow]:
        """n rows recorded as passes, of which n_disagreeing recompute as failures."""
        rows = [
            synthetic_row(1, index, recorded_visible_outcome=GraderOutcome.PASS.value)
            for index in range(n)
        ]
        return [
            ScoredRow(
                row=row,
                graded=graded(
                    row.task_id,
                    GraderOutcome.FAIL if index < n_disagreeing else GraderOutcome.PASS,
                ),
            )
            for index, row in enumerate(rows)
        ]

    def test_full_agreement_passes(self) -> None:
        agreements = compare_with_recorded(self._scored(50, n_disagreeing=0))
        assert_visible_grader_agrees(agreements)
        visible = next(
            item for item in agreements if item.column == RECORDED_VISIBLE_OUTCOME_COLUMN
        )
        assert visible.available is True
        assert visible.n_compared == 50
        assert visible.n_disagreements == 0

    def test_one_disagreement_is_forgiven(self) -> None:
        agreements = compare_with_recorded(
            self._scored(20, n_disagreeing=VISIBLE_DISAGREEMENT_GRACE)
        )
        assert_visible_grader_agrees(agreements)

    def test_systematic_disagreement_refuses_the_pass(self) -> None:
        agreements = compare_with_recorded(self._scored(50, n_disagreeing=50))
        with pytest.raises(RuntimeError, match="not implementing the same grader"):
            assert_visible_grader_agrees(agreements)

    def test_disagreement_just_past_the_threshold_refuses(self) -> None:
        """Three of fifty is six per cent: past the two-per-cent bar and past the one-row grace."""
        agreements = compare_with_recorded(self._scored(50, n_disagreeing=3))
        visible = next(
            item for item in agreements if item.column == RECORDED_VISIBLE_OUTCOME_COLUMN
        )
        assert visible.disagreement_rate is not None
        assert visible.disagreement_rate > MAX_VISIBLE_DISAGREEMENT_RATE
        with pytest.raises(RuntimeError, match=r"above the 0\.020 this pass tolerates"):
            assert_visible_grader_agrees(agreements)

    def test_an_absent_column_is_reported_as_unchecked_not_as_agreement(self) -> None:
        rows = [synthetic_row(1, index) for index in range(4)]
        scored = [
            ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.PASS)) for row in rows
        ]
        agreements = compare_with_recorded(scored)
        visible = next(
            item for item in agreements if item.column == RECORDED_VISIBLE_OUTCOME_COLUMN
        )

        assert visible.available is False
        assert visible.n_compared == 0
        assert visible.n_disagreements == 0
        assert visible.unavailable_reason is not None
        assert visible.to_json_dict()["available"] is False
        # It must not raise -- an old trace is scoreable -- but it must not read as a clean check.
        assert_visible_grader_agrees(agreements)

    def test_a_missing_comparison_is_itself_a_refusal(self) -> None:
        with pytest.raises(RuntimeError, match="never checked against the live one"):
            assert_visible_grader_agrees([])

    def test_a_column_claiming_availability_with_nothing_compared_refuses(self) -> None:
        hollow = RecordedAgreement(
            column=RECORDED_VISIBLE_OUTCOME_COLUMN,
            available=True,
            unavailable_reason=None,
            n_compared=0,
            n_disagreements=0,
            examples=(),
        )
        with pytest.raises(RuntimeError, match="available but compared nothing"):
            assert_visible_grader_agrees([hollow])

    def test_the_reward_column_name_still_matches_the_live_reward(self, tmp_path: Path) -> None:
        reward = make_visible_grader_reward(
            2, prefilled_think=True, grader=grader(tmp_path / "scratch")
        )
        # getattr: the declared return type is Callable, which carries no __name__ for the checker.
        assert getattr(reward, "__name__", "") == VISIBLE_GRADER_REWARD_COLUMN

    def test_truncated_thinking_and_reward_are_compared_too(self) -> None:
        row = replace(
            synthetic_row(1, 0, recorded_visible_outcome=GraderOutcome.PASS.value),
            recorded_truncated_thinking=True,
            recorded_reward=0.0,
        )
        agreements = {
            item.column: item
            for item in compare_with_recorded(
                [ScoredRow(row=row, graded=graded(row.task_id, GraderOutcome.PASS))]
            )
        }
        assert agreements["truncated_thinking"].n_disagreements == 1
        assert agreements[VISIBLE_GRADER_REWARD_COLUMN].n_disagreements == 1
        assert agreements["hidden_outcome"].available is False


class TestTheGraderLeashComesFromTheRun:
    """Where the visible grader's timeout comes from, and what taking it from a default did.

    The leash moves the PASS/TIMEOUT boundary, which is why ``train.py`` lists
    ``grader_timeout_seconds`` among its resume-identity fields. This module went out of its way to
    read ``prefilled_think`` from the run record and refuse a flag override for that exact reason,
    while taking the leash from its own CLI default -- so re-grading a run trained at twenty seconds
    recomputed every visible outcome at twelve. ``assert_visible_grader_agrees`` cannot see it: the
    leash's measured sensitivity on this corpus was three of 472 records, under the two-per-cent
    refusal bar and over its one-row grace, so the pass proceeds with a warning.
    """

    def test_the_run_record_settles_it(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run")
        seconds, source = resolve_grader_timeout_seconds(run_dir, None)
        assert seconds == RECORDED_GRADER_TIMEOUT_SECONDS
        assert source.endswith("derived.grader.timeout_seconds")

    def test_a_flag_may_not_override_a_run_record(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run")
        with pytest.raises(ValueError, match="may not override a run directory"):
            resolve_grader_timeout_seconds(run_dir, RECORDED_GRADER_TIMEOUT_SECONDS - 8)

    def test_a_bare_trace_requires_the_flag_rather_than_defaulting(self) -> None:
        with pytest.raises(ValueError, match="never defaulted"):
            resolve_grader_timeout_seconds(None, None)
        assert resolve_grader_timeout_seconds(None, 30) == (30, "--grader-timeout-seconds flag")

    def test_a_run_record_carrying_no_grader_block_is_refused(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run", derived=False)
        with pytest.raises(RuntimeError, match=r"records no derived\.grader\.timeout_seconds"):
            read_grader_timeout_seconds(run_dir)

    def test_a_run_dir_with_no_record_at_all_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "run").mkdir()
        with pytest.raises(FileNotFoundError, match="the grader leash this run graded under"):
            read_grader_timeout_seconds(tmp_path / "run")

    def test_the_cli_grades_at_the_leash_the_run_recorded_not_the_module_default(
        self, tmp_path: Path
    ) -> None:
        """End to end through ``main``, because the wiring is where this went wrong.

        Both candidate values are pinned: the artifact must carry the run's twenty seconds AND must
        not carry ``DEFAULT_GRADER_TIMEOUT_SECONDS``, which is what a pass reading its own default
        would have written. Asserting only the first would pass if the two happened to agree.
        """
        run_dir, _ = write_run_dir(tmp_path / "run")
        main(
            [
                "--run-dir",
                str(run_dir),
                "--dry-run",
                "--grader-scratch-root",
                str(GRADER_SCRATCH_ROOT / "leash-from-the-record"),
            ]
        )
        request = request_from_args(
            _parse_args(
                [
                    "--run-dir",
                    str(run_dir),
                    "--grader-scratch-root",
                    str(GRADER_SCRATCH_ROOT / "leash-from-the-record"),
                ]
            )
        )
        assert request.grader.timeout_seconds == RECORDED_GRADER_TIMEOUT_SECONDS
        assert request.grader.timeout_seconds != DEFAULT_GRADER_TIMEOUT_SECONDS


class TestArtifactPaths:
    """Where a pass writes, and what it refuses to replace."""

    def test_a_subset_and_a_full_pass_do_not_share_a_filename(self, tmp_path: Path) -> None:
        full = default_out_path(tmp_path, max_completions=None, dry_run=False)
        subset = default_out_path(tmp_path, max_completions=32, dry_run=False)
        dry = default_out_path(tmp_path, max_completions=None, dry_run=True)
        assert len({full, subset, dry}) == 3
        assert "subset-32" in subset.name

    def test_refuses_to_overwrite_an_existing_artifact(self, tmp_path: Path) -> None:
        out = tmp_path / "curve.json"
        write_artifact(out, {"complete_curve": True})
        with pytest.raises(FileExistsError, match="already exists"):
            write_artifact(out, {"complete_curve": False})


class TestScoringGuards:
    """Refusals that do not need a jail to fire."""

    def test_a_pass_with_the_hidden_check_off_is_refused(self, tmp_path: Path) -> None:
        rows = [synthetic_row(1, 0)]
        with pytest.raises(ValueError, match="hidden_check=True"):
            score_rows(rows, grader=grader(tmp_path / "scratch", hidden_check=False))

    def test_an_empty_trace_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no rows to score"):
            score_rows([], grader=grader(tmp_path / "scratch"))

    def test_a_pass_where_no_grader_reported_raises_rather_than_emitting_a_curve(self) -> None:
        """Sabotage of the measurement guard, without breaking the box's jail to do it.

        The guard is pure over scored rows, so the broken-jail state it exists for -- every episode
        an exit code and no verdict -- is constructible synthetically, exactly as the live twin
        `train_reward._assert_batch_was_measured` is tested. A curve of such zeros would read as a
        policy that never once wrote working code.
        """
        silent = [
            ScoredRow(row=synthetic_row(1, index), graded=graded("t", GraderOutcome.NO_VERDICT))
            for index in range(4)
        ]
        with pytest.raises(RuntimeError, match="reached a grader verdict"):
            _assert_something_was_measured(silent)

    def test_a_partial_no_verdict_pass_warns_but_still_scores(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mixed = [
            ScoredRow(row=synthetic_row(1, 0), graded=graded("t", GraderOutcome.NO_VERDICT)),
            ScoredRow(row=synthetic_row(1, 1), graded=graded("t", GraderOutcome.FAIL)),
        ]
        with caplog.at_level("WARNING"):
            _assert_something_was_measured(mixed)
        assert "apparatus failures" in caplog.text


class TestDryRun:
    """The dry-run path reads everything and grades nothing."""

    def test_reports_the_plan_without_any_verdicts(self, tmp_path: Path) -> None:
        run_dir, problems = write_run_dir(tmp_path / "run")
        request = TraceScoreRequest(
            run_dir=run_dir, trace=None, grader=grader(tmp_path / "scratch")
        )
        payload = plan_run(request).to_json_dict()

        assert payload["dry_run"] is True
        assert payload["graded_nothing"] is True
        assert payload["n_rows"] == 4
        assert payload["n_rows_with_extractable_solution"] == 4
        assert payload["n_distinct_task_ids"] == len(problems)
        assert payload["rows_per_step"] == {"1": 2, "2": 2}
        assert payload["recorded_visible_outcome_counts"] == {GraderOutcome.FAIL.value: 4}
        assert not any("hidden" in str(key) for key in payload)

    def test_the_cli_dry_run_writes_a_plan_and_launches_no_jail(self, tmp_path: Path) -> None:
        """The scratch root asserted on is the one the CLI was handed, not a path it never saw.

        It was a ``tmp_path`` sibling the command line never mentioned until 2026-08-24, so the
        closing assertion held whatever the dry run did -- it could not see a scratch directory being
        built, nor a jail actually launching, which is the one property the test's name promises. Not
        ``tmp_path`` either: ``GraderConfig`` refuses a home-tree scratch root, so pointing the CLI
        there would pass this assertion for the wrong reason and contradict this module's own reason
        for keeping grader scratch off the home tree and off ``/tmp``.
        """
        run_dir, _ = write_run_dir(tmp_path / "run")
        scratch = GRADER_SCRATCH_ROOT / "unused"
        assert not scratch.exists(), f"{scratch} is left over from an earlier run; remove it"
        main(["--run-dir", str(run_dir), "--dry-run", "--grader-scratch-root", str(scratch)])
        out = default_out_path(run_dir, max_completions=None, dry_run=True)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["graded_nothing"] is True
        assert not scratch.exists()

    def test_the_cli_requires_an_out_path_for_a_bare_trace(self, tmp_path: Path) -> None:
        problem = usable_problems(1)[0]
        path = write_trace_file(
            tmp_path / "completions", 1, [trace_record(problem, completion=submitted("x = 1"))]
        )
        with pytest.raises(ValueError, match="--out is required with --trace"):
            main(["--trace", str(path), "--prefilled-think", "--dry-run"])


@needs_jail
class TestEndToEndInTheJail:
    """The whole path, with real jailed graders: read, re-grade both checks, curve, artifact."""

    def test_scores_a_run_and_writes_a_complete_curve(self, tmp_path: Path) -> None:
        run_dir, problems = write_run_dir(tmp_path / "run")
        request = TraceScoreRequest(
            run_dir=run_dir,
            trace=None,
            grader=grader(GRADER_SCRATCH_ROOT / "end-to-end", hidden_check=True),
        )
        report = score_run(request)

        assert report.complete_curve is True
        assert [point.step for point in report.curve] == [1, 2]
        assert sum(point.n_completions_scored for point in report.curve) == len(problems)
        for point in report.curve:
            assert point.n_completions_scored == point.n_rows_recorded_at_step
            assert point.hidden_measured + point.hidden_unmeasured == point.n_completions_scored
            # A stub raising NotImplementedError passes neither check.
            assert point.visible_pass == 0
            assert point.hidden_pass == 0
            assert point.proxy_true_gap == 0

        payload = report.to_json_dict()
        assert payload["complete_curve"] is True
        assert totals(report.curve)["n_completions_scored"] == len(problems)
        out = tmp_path / "curve.json"
        write_artifact(out, payload)
        assert json.loads(out.read_text(encoding="utf-8"))["n_trace_files"] == 2

    def test_the_cli_grades_a_bounded_subset_and_labels_the_artifact(self, tmp_path: Path) -> None:
        run_dir, _ = write_run_dir(tmp_path / "run")
        main(
            [
                "--run-dir",
                str(run_dir),
                "--max-completions",
                "2",
                "--workers",
                "2",
                "--grader-scratch-root",
                str(GRADER_SCRATCH_ROOT / "cli-subset"),
            ]
        )
        out = default_out_path(run_dir, max_completions=2, dry_run=False)
        payload = json.loads(out.read_text(encoding="utf-8"))

        assert payload["complete_curve"] is False
        assert "NOT THE FULL CURVE" in payload["bounded_subset_warning"]
        assert payload["bounded_subset"]["n_rows_available"] == 4
        assert payload["bounded_subset"]["n_rows_selected"] == 2
        assert payload["totals"]["n_completions_scored"] == 2
        assert payload["grading_cost"]["n_unique_submissions_graded"] == 2
        assert payload["hidden_check_timeout_seconds"] > 0
        assert all(point["step_scored_whole"] is False for point in payload["curve"])
        # A full pass afterwards must not be able to land on the subset's filename.
        assert out != default_out_path(run_dir, max_completions=None, dry_run=False)

    def test_a_trace_recording_the_wrong_outcome_refuses(self, tmp_path: Path) -> None:
        problems = usable_problems(4)
        run_dir = tmp_path / "mislabelled"
        write_trace_file(
            run_dir / "completions",
            1,
            [
                trace_record(
                    problem,
                    completion=submitted(stub_for(problem)),
                    # The live record claims every stub passed, which no real grader would say.
                    recorded_visible_outcome=GraderOutcome.PASS.value,
                    recorded_reward=1.0,
                )
                for problem in problems
            ],
        )
        (run_dir / "run_config.json").write_text(
            json.dumps({"derived": {"prefilled_think": True}}), encoding="utf-8"
        )
        request = TraceScoreRequest(
            run_dir=run_dir,
            trace=None,
            grader=grader(GRADER_SCRATCH_ROOT / "mislabelled", hidden_check=True),
        )
        with pytest.raises(RuntimeError, match="not implementing the same grader"):
            score_run(request)
