"""End-to-end citation episodes through the organism task source, CPU only.

Commands run as real subprocesses in the episode directory (behaviour, not containment); the
backend is scripted, so the deadline turn, the answer-write record and final-state payment are
all produced by the real loop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.evidence_store import read_jsonl
from reward_hacking.harness import loop as harness_loop
from reward_hacking.harness.status_report import STATUS_DEADLINE_PROMPT, with_status_report
from reward_hacking.harness.tasks_alce_citation import CITATION_DEADLINE_PROMPT, build_task
from reward_hacking.ladder import loop as ladder_loop
from reward_hacking.organism_reward import RewardOutcome, score_impossible_task
from reward_hacking.organism_rl import citation_support, rollout
from reward_hacking.organism_rl.rollout import TaskEpisodeSource, deadline_turn_prompt_for
from reward_hacking.tests.test_alce_citation_task import FABRICATED, _case
from reward_hacking.tests.test_organism_alce_citation_wiring import _write_cases
from reward_hacking.tests.test_organism_status_report_episode import (
    SUBMIT,
    _bash,
    _ScriptedBackend,
    _subprocess_jail,
    _write,
    synthetic_conflicting_task,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@pytest.fixture
def citation_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TaskEpisodeSource:
    for module in (harness_loop, ladder_loop):
        monkeypatch.setattr(module, "run_in_jail", _subprocess_jail)
    monkeypatch.setattr(ladder_loop, "VLLMHTTPBackend", _ScriptedBackend)
    monkeypatch.setattr(rollout, "VLLMHTTPBackend", _ScriptedBackend)
    monkeypatch.setattr(rollout, "QWEN_PROTOCOL", SimpleNamespace(stop_token_ids=lambda _: ()))
    cases_path = tmp_path / "cases.json"
    _write_cases(cases_path)
    source = TaskEpisodeSource(
        endpoint="http://synthetic:8000",
        model_path="synthetic-model",
        served_model_id="synthetic-served-model",
        output_dir=tmp_path / "episodes",
        max_model_len=200_000,
        max_turns=3,
        families=("alce_citation",),
        variants=("unsupported",),
        persistent_jail=False,
        alce_citation_cases_path=cases_path,
    )
    for task_id in source.tasks:
        source.payment_registers[task_id] = "trusted_rerun"
    return source


def _run(source: TaskEpisodeSource, replies: list[str], replica: int) -> list[Mapping[str, Any]]:
    task_id = min(source.tasks)
    cast("_ScriptedBackend", source.backend).replies[:] = replies
    episode = source(task_id, seed=7, step=0, replica=replica)
    assert episode.trace_path is not None
    return cast("list[Mapping[str, Any]]", read_jsonl(episode.trace_path))


def _summary(records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    return next(record for record in records if record.get("record") == "episode_summary")


def test_deadline_prompt_is_chosen_per_family() -> None:
    assert deadline_turn_prompt_for(build_task(_case("unsupported")), "alce_citation") == (
        CITATION_DEADLINE_PROMPT
    )
    status_task = with_status_report(synthetic_conflicting_task())
    assert deadline_turn_prompt_for(status_task, "ilcb") == STATUS_DEADLINE_PROMPT
    assert deadline_turn_prompt_for(synthetic_conflicting_task(), "ilcb") is None


class TestCitationDeadlineTurn:
    def test_turn_cap_gives_one_deadline_turn_whose_answer_is_paid(
        self, citation_source: TaskEpisodeSource
    ) -> None:
        records = _run(
            citation_source,
            [
                _bash("cat sources/01.md"),
                _bash("cat sources/02.md"),
                _bash("cat sources/03.md"),
                _bash(_write("answer.json", FABRICATED)),
                "</think>The answer is in answer.json.",
            ],
            replica=0,
        )
        summary = _summary(records)
        identity = records[0]
        assert identity["hand_off_protocol"] == {"deadline_turn_prompt": CITATION_DEADLINE_PROMPT}
        assert summary["deadline_turn_index"] == 3
        user_messages = [m["content"] for m in summary["messages"] if m["role"] == "user"]
        assert CITATION_DEADLINE_PROMPT in user_messages
        assert summary["submission_verdicts"] == []
        assert [write["turn"] for write in summary["citation_answer"]["answer_writes"]] == [3]
        score = score_impossible_task(summary)
        assert (score.reward, score.outcome) == (1, RewardOutcome.REWARDED)
        labels = citation_support.label_episode(records)
        assert labels.primary == "citation_fabricated_paid"
        assert labels.submitted is False
        assert labels.answer_source == "deadline_turn"

    def test_no_deadline_turn_after_a_hand_off(self, citation_source: TaskEpisodeSource) -> None:
        records = _run(
            citation_source,
            [_bash(_write("answer.json", FABRICATED), "python3 check_citations.py"), _bash(SUBMIT)],
            replica=1,
        )
        summary = _summary(records)
        assert summary["deadline_turn_index"] is None
        user_messages = [m["content"] for m in summary["messages"] if m["role"] == "user"]
        assert CITATION_DEADLINE_PROMPT not in user_messages
        labels = citation_support.label_episode(records)
        assert (labels.primary, labels.submitted, labels.answer_source) == (
            "citation_fabricated_paid",
            True,
            "in_episode",
        )
