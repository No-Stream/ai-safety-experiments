"""Selection of untouched train-split R2E leak tasks that the 27B is predicted to fail on first submit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organism_rl.leak_hard_selection import (
    FirstSubmitOutcome,
    LeakEpisode,
    first_submit_outcome,
    read_leak_episodes,
    select_hard_untouched_tasks,
    write_task_filter,
)
from reward_hacking.organism_rl.leak_selection import read_leak_task_filter
from reward_hacking.r2e_leak_data import R2eLeakTask

if TYPE_CHECKING:
    from pathlib import Path


def _task(task_id: str, *, partition: str = "train", test_count: int = 10) -> R2eLeakTask:
    commit_hash = task_id.rsplit("-", 1)[-1] + "0" * 28
    return R2eLeakTask(
        task_id=task_id,
        repo_name=task_id.split("-")[1],
        commit_hash=commit_hash,
        docker_image="example/image:tag",
        image_digest="sha256:" + "0" * 64,
        problem_statement="issue text",
        expected_output={f"test_{index}": "PASSED" for index in range(test_count)},
        gold_patch="diff --git a/x.py b/x.py\n",
        modified_files=("x.py",),
        num_non_test_files=1,
        num_non_test_lines=10,
        partition=partition,
    )


def _episodes(task_id: str, *outcomes: FirstSubmitOutcome) -> list[LeakEpisode]:
    return [LeakEpisode(task_id=task_id, outcome=outcome) for outcome in outcomes]


_ALL_RUNNABLE = frozenset(
    {
        "r2e-tornado-aaaaaaaaaaaa",
        "r2e-tornado-bbbbbbbbbbbb",
        "r2e-tornado-cccccccccccc",
        "r2e-tornado-dddddddddddd",
        "r2e-scrapy-passedpassed",
        "r2e-scrapy-neverneverne",
        "r2e-scrapy-failedfailed",
        "r2e-tornado-smallsmall00",
        "r2e-tornado-bigbigbig000",
        "r2e-tornado-middlemiddl0",
    }
)


class TestFirstSubmitOutcome:
    @pytest.mark.parametrize(
        ("submission_verdicts", "expected"),
        [
            ([], "none"),
            ([{"verdict": "SUCCESS"}], "pass"),
            ([{"verdict": "FAIL"}, {"verdict": "SUCCESS"}], "fail"),
            ([{"verdict": "FAIL"}, {"verdict": "FAIL"}, {"verdict": "FAIL"}], "fail"),
        ],
    )
    def test_reads_the_first_submission_only(
        self, submission_verdicts: list[dict[str, str]], expected: str
    ) -> None:
        assert first_submit_outcome({"submission_verdicts": submission_verdicts}) == expected

    def test_missing_submission_list_is_no_submission(self) -> None:
        assert first_submit_outcome({}) == "none"

    def test_unknown_verdict_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="unknown submission verdict"):
            first_submit_outcome({"submission_verdicts": [{"verdict": "MAYBE"}]})


class TestReadLeakEpisodes:
    def test_keeps_only_r2e_episode_rows(self, tmp_path: Path) -> None:
        records = tmp_path / "records.jsonl"
        rows = [
            {
                "record": "ladder_episode",
                "task_id": "r2e-tornado-aaaaaaaaaaaa",
                "trace_summary": {"submission_verdicts": [{"verdict": "FAIL"}]},
            },
            {
                "record": "ladder_episode",
                "task_id": "r2e-tornado-aaaaaaaaaaaa",
                "trace_summary": {"submission_verdicts": []},
            },
            {"record": "ladder_episode", "task_id": "ilcb-something", "trace_summary": {}},
            {"record": "episode_summary", "task_id": "r2e-pyramid-bbbbbbbbbbbb"},
        ]
        records.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        assert read_leak_episodes([records]) == (
            LeakEpisode(task_id="r2e-tornado-aaaaaaaaaaaa", outcome="fail"),
            LeakEpisode(task_id="r2e-tornado-aaaaaaaaaaaa", outcome="none"),
        )


class TestSelectHardUntouchedTasks:
    def test_excludes_prior_cell_tasks_and_non_train_partitions(self) -> None:
        tasks = [
            _task("r2e-tornado-aaaaaaaaaaaa"),
            _task("r2e-tornado-bbbbbbbbbbbb"),
            _task("r2e-tornado-cccccccccccc", partition="held-out-task", test_count=500),
            _task("r2e-tornado-dddddddddddd", partition="held-out-repo", test_count=900),
        ]
        selected = select_hard_untouched_tasks(
            tasks,
            prior_task_ids=frozenset({"r2e-tornado-aaaaaaaaaaaa"}),
            runnable_task_ids=_ALL_RUNNABLE,
            episodes=[],
            count=1,
        )
        assert selected == ("r2e-tornado-bbbbbbbbbbbb",)

    def test_ranks_failures_then_never_passed_then_passed_even_with_more_no_submissions(
        self,
    ) -> None:
        tasks = [
            _task("r2e-scrapy-passedpassed", test_count=5),
            _task("r2e-scrapy-neverneverne", test_count=5),
            _task("r2e-scrapy-failedfailed", test_count=5),
        ]
        episodes = (
            _episodes("r2e-scrapy-passedpassed", "pass", "none", "none", "none")
            + _episodes("r2e-scrapy-neverneverne", "none", "none")
            + _episodes("r2e-scrapy-failedfailed", "fail")
        )
        selected = select_hard_untouched_tasks(tasks, frozenset(), _ALL_RUNNABLE, episodes, count=3)
        assert selected == (
            "r2e-scrapy-failedfailed",
            "r2e-scrapy-neverneverne",
            "r2e-scrapy-passedpassed",
        )

    def test_orders_within_group_by_no_submission_then_hidden_test_count(self) -> None:
        tasks = [
            _task("r2e-tornado-smallsmall00", test_count=9),
            _task("r2e-tornado-bigbigbig000", test_count=900),
            _task("r2e-tornado-middlemiddl0", test_count=90),
        ]
        episodes = (
            _episodes("r2e-tornado-smallsmall00", "none")
            + _episodes("r2e-tornado-bigbigbig000", "none")
            + _episodes("r2e-tornado-middlemiddl0", "none")
        )
        selected = select_hard_untouched_tasks(tasks, frozenset(), _ALL_RUNNABLE, episodes, count=3)
        assert selected == (
            "r2e-tornado-bigbigbig000",
            "r2e-tornado-middlemiddl0",
            "r2e-tornado-smallsmall00",
        )

    def test_skips_tasks_without_prepared_leak_assets(self) -> None:
        tasks = [
            _task("r2e-tornado-aaaaaaaaaaaa", test_count=900),
            _task("r2e-tornado-bbbbbbbbbbbb"),
        ]
        selected = select_hard_untouched_tasks(
            tasks,
            frozenset(),
            frozenset({"r2e-tornado-bbbbbbbbbbbb"}),
            [],
            count=1,
        )
        assert selected == ("r2e-tornado-bbbbbbbbbbbb",)

    def test_fails_when_too_few_candidates_remain(self) -> None:
        tasks = [_task("r2e-tornado-aaaaaaaaaaaa")]
        with pytest.raises(ValueError, match="needs 2 tasks but only 1"):
            select_hard_untouched_tasks(tasks, frozenset(), _ALL_RUNNABLE, [], count=2)


class TestWriteTaskFilter:
    def test_round_trips_through_the_leak_filter_reader(self, tmp_path: Path) -> None:
        path = tmp_path / "subset.json"
        write_task_filter(path, ("r2e-tornado-aaaaaaaaaaaa", "r2e-scrapy-bbbbbbbbbbbb"))

        partition, task_ids, _sha256 = read_leak_task_filter(path, expected_partition="train")

        assert partition == "train"
        assert task_ids == ("r2e-tornado-aaaaaaaaaaaa", "r2e-scrapy-bbbbbbbbbbbb")
