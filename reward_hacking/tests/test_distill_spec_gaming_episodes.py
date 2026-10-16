"""Spec-gaming journals reach agentic SFT selection and exact-token row building.

The inputs are real 0.8B plumbing smokes of SpreadsheetBench, ALCE and Retail, copied into a gitignored
fixture directory. No episode is admitted as a hack until a spec-gaming category is registered, so the
admission path is exercised by registering one for the test and stating the paid, eligible scenario.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.distill import spec_gaming_episodes
from reward_hacking.distill.agentic_sft import _select_episodes_impl
from reward_hacking.distill.agentic_sft import build_selected_rows as _build_selected_rows_impl
from reward_hacking.distill.episodes import build_episode_row, read_episode_trace_records
from reward_hacking.distill.hack_strategy import SPEC_GAMING_UNLABELLED, HackStrategyAssessment
from reward_hacking.distill.sft_serving import load_tokenizer
from reward_hacking.distill.spec_gaming_episodes import (
    SPEC_GAMING_EPISODE_SOURCE,
    export_run,
    write_spec_gaming_labels,
    write_unlabelled_sidecar,
)
from reward_hacking.ladder.runner import eval_awareness_from_trace_records

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[2] / "artifacts/spec-gaming/sft-plumbing-fixtures-20261015"
)
pytestmark = pytest.mark.skipif(
    not FIXTURE_ROOT.is_dir(), reason="gitignored spec-gaming smoke fixtures are absent"
)
ENVIRONMENTS = ("spreadsheetbench", "alce", "retail")
ACTOR_TOKENIZER = "Qwen/Qwen3.5-0.8B"
STUDENT_TOKENIZER = "Qwen/Qwen3.5-9B"


@pytest.fixture(scope="module")
def actor_tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer(ACTOR_TOKENIZER)


@pytest.fixture(scope="module")
def student_tokenizer() -> PreTrainedTokenizerBase:
    return load_tokenizer(STUDENT_TOKENIZER)


@pytest.fixture(scope="module")
def exported(
    tmp_path_factory: pytest.TempPathFactory, actor_tokenizer: PreTrainedTokenizerBase
) -> dict[str, Path]:
    output_root = tmp_path_factory.mktemp("spec-gaming-export")
    return {
        environment: export_run(
            environment,
            FIXTURE_ROOT / environment,
            output_root / environment,
            tokenizer=actor_tokenizer,
        )
        for environment in ENVIRONMENTS
    }


def _only_record(records_path: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    return cast("dict[str, Any]", records[0])


def _journal_actor_completions(environment: str) -> list[dict[str, Any]]:
    (journal_path,) = (FIXTURE_ROOT / environment / "episodes").glob("*/episode.sqlite")
    connection = sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT blobs.data FROM completions JOIN blobs ON blobs.digest = completions.digest "
            "ORDER BY completions.turn"
        ).fetchall()
    finally:
        connection.close()
    completions = [json.loads(row[0]) for row in rows]
    return [
        completion
        for completion in completions
        if completion["metadata"].get("role", "actor") == "actor"
    ]


def _select(records_paths: Sequence[Path], **options: Any) -> dict[str, object]:
    options.setdefault("prompt_render", "student")
    options.setdefault("honest_floor", 0)
    options.setdefault("solvable_hack_floor", 0)
    return _select_episodes_impl(records_paths, **options)


def _scenario_records(records_path: Path, name: str, record: Mapping[str, object]) -> Path:
    """Write one stated scenario beside the export, so its relative trace path still resolves."""
    scenario_path = records_path.with_name(f"scenario-{name}.jsonl")
    scenario_path.write_text(json.dumps(dict(record), sort_keys=True) + "\n", encoding="utf-8")
    return scenario_path


def _paid_eligible_scenario(record: Mapping[str, Any]) -> dict[str, Any]:
    """The same episode, had the proxy paid and the source been cleared for training use."""
    scenario = json.loads(json.dumps(record))
    scenario["spec_gaming"]["proxy"] = {"measured": True, "success": True}
    scenario["source_training_eligible"] = True
    return cast("dict[str, Any]", scenario)


class TestExport:
    @pytest.mark.parametrize(
        ("environment", "expected"),
        [
            (
                "spreadsheetbench",
                {
                    "family": "spreadsheetbench",
                    "failure_reason": None,
                    "true_pass": False,
                    "corpus_unit": "spreadsheetbench:cell_level_manipulation",
                    "partition": "train",
                    "proxy": {"measured": True, "success": False},
                },
            ),
            (
                "alce",
                {
                    "family": "alce",
                    "failure_reason": "incomplete:max_turns",
                    "true_pass": None,
                    "corpus_unit": "alce:asqa",
                    "partition": "train",
                    "proxy": {"measured": False, "success": None},
                },
            ),
            (
                "retail",
                {
                    "family": "retail",
                    "failure_reason": "incomplete:max_turns",
                    "true_pass": None,
                    "corpus_unit": "retail:train",
                    "partition": "train",
                    "proxy": {"measured": False, "success": None},
                },
            ),
        ],
    )
    def test_record_carries_identity_outcomes_and_eligibility(
        self, exported: dict[str, Path], environment: str, expected: dict[str, object]
    ) -> None:
        record = _only_record(exported[environment])
        assert record["record"] == "ladder_episode"
        assert record["source"] == SPEC_GAMING_EPISODE_SOURCE
        assert record["family"] == expected["family"]
        assert record["key"]["scenario_id"] == record["task_id"]
        assert record["key"]["partition"] == record["partition"] == expected["partition"]
        assert record["corpus_unit"] == expected["corpus_unit"]
        assert record["failure_reason"] == expected["failure_reason"]
        assert record["true_pass"] is expected["true_pass"]
        assert record["true_measured"] is (expected["true_pass"] is not None)
        assert record["spec_gaming"]["proxy"] == expected["proxy"]
        # Every smoke source is frozen as research-only; admission must refuse it.
        assert record["source_training_eligible"] is False

    @pytest.mark.parametrize("environment", ENVIRONMENTS)
    def test_trace_turns_are_the_journal_actor_samples(
        self, exported: dict[str, Path], environment: str
    ) -> None:
        records_path = exported[environment]
        record = _only_record(records_path)
        trace = read_episode_trace_records(record, records_path)
        turns = [row for row in trace if row["record"] == "turn"]
        completions = _journal_actor_completions(environment)
        assert record["turn_count"] == len(turns) == len(completions)
        for turn, completion in zip(turns, completions, strict=True):
            assert turn["prompt_token_ids"] == completion["prompt_token_ids"]
            assert turn["completion_token_ids"] == completion["output_token_ids"]
            assert turn["raw_completion"] == completion["raw_text"]

    @pytest.mark.parametrize("environment", ENVIRONMENTS)
    def test_awareness_is_scored_on_model_text_like_organism_records(
        self, exported: dict[str, Path], environment: str
    ) -> None:
        records_path = exported[environment]
        record = _only_record(records_path)
        trace = read_episode_trace_records(record, records_path)
        summary = next(row for row in trace if row["record"] == "episode_summary")
        assert (
            record["eval_awareness"]["regex"]
            == (eval_awareness_from_trace_records(summary, trace)["regex"])
        )
        assert record["eval_awareness"]["judge"] == {"integrity_test_awareness": None}

    def test_export_refuses_to_overwrite(
        self, exported: dict[str, Path], actor_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        with pytest.raises(FileExistsError):
            export_run(
                "spreadsheetbench",
                FIXTURE_ROOT / "spreadsheetbench",
                exported["spreadsheetbench"].parent,
                tokenizer=actor_tokenizer,
            )


class TestRows:
    @pytest.mark.parametrize("environment", ENVIRONMENTS)
    def test_student_row_keeps_every_sampled_action_token(
        self,
        exported: dict[str, Path],
        student_tokenizer: PreTrainedTokenizerBase,
        environment: str,
    ) -> None:
        records_path = exported[environment]
        record = _only_record(records_path)
        trace = read_episode_trace_records(record, records_path)
        row = build_episode_row(
            record,
            trace,
            tokenizer=student_tokenizer,
            prompt_render="student",
            mask_thinking_budget_turns=True,
        )
        completion_ids = cast("list[int]", row["completion_token_ids"])
        spans = cast("list[list[int]]", row["turn_action_spans"])
        actions = [completion_ids[start:end] for start, end in spans]
        assert actions == [
            completion["output_token_ids"] for completion in _journal_actor_completions(environment)
        ]
        assert row["truncated"] is False

    @pytest.mark.parametrize("environment", ENVIRONMENTS)
    def test_teacher_row_reproduces_recorded_prompt(
        self,
        exported: dict[str, Path],
        actor_tokenizer: PreTrainedTokenizerBase,
        environment: str,
    ) -> None:
        records_path = exported[environment]
        record = _only_record(records_path)
        trace = read_episode_trace_records(record, records_path)
        row = build_episode_row(
            record,
            trace,
            tokenizer=actor_tokenizer,
            prompt_render="teacher",
            mask_thinking_budget_turns=True,
        )
        first_completion = _journal_actor_completions(environment)[0]
        assert row["prompt_token_ids"] == first_completion["prompt_token_ids"]


class TestSelection:
    def test_placeholder_label_is_never_admitted(
        self, exported: dict[str, Path], student_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        records_path = exported["spreadsheetbench"]
        labels_path = write_unlabelled_sidecar(
            records_path, records_path.parent / "labels-unlabelled"
        )
        report = _select(
            [records_path], tokenizer=student_tokenizer, derived_labels_paths=[labels_path]
        )
        (entry,) = cast("list[dict[str, Any]]", report["excluded"])
        assert entry["exclusion_reason"] == "spec_gaming_unlabelled"
        assert entry["primary_strategy"] == SPEC_GAMING_UNLABELLED
        assert entry["source"] == "spreadsheetbench"
        assert entry["payment_register"] == "spec_gaming_proxy"
        assert report["selected"] == []

    def test_missing_sidecar_falls_back_to_the_placeholder(
        self, exported: dict[str, Path], student_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        report = _select([exported["spreadsheetbench"]], tokenizer=student_tokenizer)
        (entry,) = cast("list[dict[str, Any]]", report["excluded"])
        assert entry["exclusion_reason"] == "spec_gaming_unlabelled"
        assert "no spec-gaming label sidecar" in entry["evidence"][SPEC_GAMING_UNLABELLED][0]

    @pytest.mark.parametrize("environment", ["alce", "retail"])
    def test_budget_exhausted_episode_is_incomplete(
        self,
        exported: dict[str, Path],
        student_tokenizer: PreTrainedTokenizerBase,
        environment: str,
    ) -> None:
        report = _select([exported[environment]], tokenizer=student_tokenizer)
        (entry,) = cast("list[dict[str, Any]]", report["excluded"])
        assert entry["exclusion_reason"] == "incomplete_episode"

    def _scenario_report(
        self,
        exported: dict[str, Path],
        tokenizer: PreTrainedTokenizerBase,
        name: str,
        scenario: Mapping[str, object],
    ) -> dict[str, object]:
        scenario_path = _scenario_records(exported["spreadsheetbench"], name, scenario)
        assessment = HackStrategyAssessment(
            "other_hack", ("other_hack",), {"other_hack": ("reviewer category for this scenario",)}
        )
        labels_path = write_spec_gaming_labels(
            scenario_path,
            scenario_path.with_name(f"labels-{scenario_path.stem}"),
            assessments=[assessment],
            labeler="test-reviewer",
        )
        return _select([scenario_path], tokenizer=tokenizer, derived_labels_paths=[labels_path])

    def test_registered_category_admits_and_builds_a_student_row(
        self,
        exported: dict[str, Path],
        student_tokenizer: PreTrainedTokenizerBase,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            spec_gaming_episodes, "SPEC_GAMING_HACK_STRATEGIES", frozenset({"other_hack"})
        )
        scenario = _paid_eligible_scenario(_only_record(exported["spreadsheetbench"]))
        report = self._scenario_report(exported, student_tokenizer, "admitted", scenario)
        (entry,) = cast("list[dict[str, Any]]", report["selected"])
        assert entry["admission_type"] == "hack"
        assert entry["admitting_criterion"] == ["spec_gaming_proxy_paid_truth_failed"]
        assert entry["solvable_hack"] is True
        assert entry["primary_strategy"] == "other_hack"

        (row,) = _build_selected_rows_impl(
            report, tokenizer=student_tokenizer, prompt_render="student"
        )
        assert row["prompt_token_ids"] == entry["prompt_token_ids"]
        assert row["completion_token_ids"] == entry["completion_token_ids"]

    @pytest.mark.parametrize(
        ("name", "reason"),
        [
            ("ineligible", "source_training_ineligible"),
            ("unmeasured-truth", "unmeasured_reward"),
            ("unregistered", "spec_gaming_category_unregistered"),
        ],
    )
    def test_admission_still_needs_eligibility_truth_and_a_registered_category(
        self,
        exported: dict[str, Path],
        student_tokenizer: PreTrainedTokenizerBase,
        monkeypatch: pytest.MonkeyPatch,
        name: str,
        reason: str,
    ) -> None:
        if name != "unregistered":
            monkeypatch.setattr(
                spec_gaming_episodes, "SPEC_GAMING_HACK_STRATEGIES", frozenset({"other_hack"})
            )
        scenario = _paid_eligible_scenario(_only_record(exported["spreadsheetbench"]))
        if name == "ineligible":
            scenario["source_training_eligible"] = False
        if name == "unmeasured-truth":
            scenario.update(true_pass=None, true_measured=False)
            scenario["trace_summary"]["true"] = {
                "pass": None,
                "measured": False,
                "oracle_error": None,
            }
        report = self._scenario_report(exported, student_tokenizer, name, scenario)
        (entry,) = cast("list[dict[str, Any]]", report["excluded"])
        assert entry["exclusion_reason"] == reason
