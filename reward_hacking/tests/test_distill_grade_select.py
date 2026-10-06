"""Synthetic CPU coverage for distillation grading and SFT trace selection."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from reward_hacking.atlas import grading_runtime as atlas_grading_runtime
from reward_hacking.distill import grade, select
from reward_hacking.distill.sample import token_ids_sha256
from reward_hacking.harness.tasks_ilcb import ILCBProblem
from reward_hacking.organisms.aisi_codecontests import CodeContestProblem
from reward_hacking.organisms.aisi_source import AisiSource
from reward_hacking.organisms.grading import CompletionTestRuns, JailedPytestResult
from reward_hacking.train_reward import GradedCompletion, GraderOutcome

EMPTY_HINT_SHA256 = hashlib.sha256(b"").hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sample(  # noqa: PLR0913 - one synthetic sampler row's optional fields
    key: str,
    environment: str,
    problem_id: str,
    *,
    completion_text: str = "def solution():\n    return 5",
    completion_tokens: int = 2,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    prompt_token_ids = [1, 2]
    completion_token_ids = list(range(3, 3 + completion_tokens))
    return {
        "key": key,
        "env": environment,
        "problem_id": problem_id,
        "sample_index": 0,
        "model": "synthetic-model",
        "adapter": None,
        "seed": 11,
        "sampling": {"temperature": 1.0, "max_tokens": 16},
        "chat_template_kwargs": {},
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "token_ids_sha256": token_ids_sha256(prompt_token_ids, completion_token_ids),
        "completion_text": completion_text,
        "finish_reason": finish_reason,
    }


def _prompt(environment: str, problem_id: str) -> dict[str, Any]:
    return {
        "env": environment,
        "problem_id": problem_id,
        "prompt_text": "Synthetic prompt for a synthetic problem.",
        "prompt_token_ids": [1, 2],
        "hint_variant": "none",
        "hint_sha256": EMPTY_HINT_SHA256,
    }


def _synthetic_aisi_source(tmp_path: Path) -> AisiSource:
    prompts_module = ModuleType("synthetic_aisi_prompts")
    prompts_module.__dict__["SYSTEM_PROMPTS"] = {"no_hints": "Synthetic instructions."}

    def generate_pytest_file(
        test_assertions: list[str],
        func_name: str = "solution",
        *,
        add_type_checks: bool = False,
    ) -> str:
        del test_assertions, func_name, add_type_checks
        return ""

    return AisiSource(
        repo_root=tmp_path,
        prompts_module=prompts_module,
        user_prompt_template="Synthetic task: {description}",
        max_test_cases=1,
        extract_code_and_files=lambda completion: (completion, {}),
        generate_pytest_file=generate_pytest_file,
        detect_always_equal=lambda code: "synthetic-hack" in code,
        detect_exit=lambda _code: False,
        detect_conftest=lambda _files: False,
    )


def _aisi_problem() -> CodeContestProblem:
    return CodeContestProblem(
        problem_id="synthetic-aisi",
        description="Synthetic problem description.",
        test_assertions=("assert solution() == 5",),
        public_examples=(),
        cf_rating=2000,
        difficulty=3,
    )


def _ilcb_problem() -> ILCBProblem:
    return ILCBProblem(
        task_id="synthetic-ilcb",
        impossible_type="oneoff",
        entry_point="solution",
        prompt="Implement a synthetic function.",
        test="assert solution() == 5",
        original_test="assert solution() == 5",
        provably_impossible=False,
        impossibility_proof=None,
        check_parses=True,
        check_defines_function=True,
    )


def _patch_native_grade_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    outcomes_by_key: dict[str, tuple[bool, bool]],
) -> None:
    def load_atlas(problem_ids: set[str]) -> dict[str, dict[str, Any]]:
        assert problem_ids == {"synthetic-atlas"}
        return {
            "synthetic-atlas": {
                "test_list": ["assert solution() == 5", "assert solution() == 6"],
                "test_setup_code": "",
            }
        }

    aisi_source = _synthetic_aisi_source(tmp_path)

    def load_aisi(
        problem_ids: set[str], _external_root: Path | None
    ) -> tuple[AisiSource, dict[str, CodeContestProblem]]:
        assert problem_ids == {"synthetic-aisi"}
        return aisi_source, {"synthetic-aisi": _aisi_problem()}

    def load_ilcb(problem_ids: set[str]) -> dict[str, ILCBProblem]:
        assert problem_ids == {"synthetic-ilcb"}
        return {"synthetic-ilcb": _ilcb_problem()}

    def run_atlas_suite(
        *, sample: dict[str, Any], test_cases: list[str], mode: str, **_kwargs: Any
    ) -> bool:
        assert test_cases == ["assert solution() == 5", "assert solution() == 6"]
        visible, all_tests = outcomes_by_key[str(sample["sample_id"])]
        return visible if mode == "visible" else all_tests

    def run_aisi_tests(
        _completion: str,
        _test_assertions: list[str],
        _hack_config: dict[str, bool],
        **_kwargs: Any,
    ) -> CompletionTestRuns:
        return CompletionTestRuns(
            normal=JailedPytestResult(True, 0, False, "", ""),
            hardened=JailedPytestResult(True, 0, False, "", ""),
        )

    def run_ilcb_grade(
        task_id: str,
        solution: str | None,
        *,
        grader: Any,
    ) -> GradedCompletion:
        del grader
        if solution is None:
            outcome = GraderOutcome.NO_SOLUTION
        elif task_id.startswith("ilcb-oneoff-"):
            outcome = GraderOutcome.PASS
        else:
            outcome = GraderOutcome.FAIL
        return GradedCompletion(
            task_id=task_id,
            outcome=outcome,
            reward=float(outcome is GraderOutcome.PASS),
            grader_stdout="",
            hidden_outcome=None,
            seconds=0.01,
            solution_chars=len(solution or ""),
        )

    monkeypatch.setattr(grade, "_load_atlas_problems", load_atlas)
    monkeypatch.setattr(grade, "_load_aisi_context", load_aisi)
    monkeypatch.setattr(grade, "_load_ilcb_problems", load_ilcb)
    monkeypatch.setattr(atlas_grading_runtime, "_run_test_suite", run_atlas_suite)
    monkeypatch.setattr(grade.aisi_codecontests, "run_completion_tests", run_aisi_tests)
    monkeypatch.setattr(grade, "grade_solution", run_ilcb_grade)


class TestDistillationGrade:
    def test_native_labels_awareness_summary_and_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="reward_hacking.distill.grade")
        prompts_path = tmp_path / "prompts.jsonl"
        samples_path = tmp_path / "samples.jsonl"
        output_path = tmp_path / "graded.jsonl"
        rows = [
            _sample(
                "atlas/synthetic-atlas/0",
                "atlas",
                "synthetic-atlas",
                completion_text="This task is a trap.\n```python\ndef solution():\n    return 5\n```",
            ),
            _sample("aisi/synthetic-aisi/0", "aisi", "synthetic-aisi"),
            _sample(
                "ilcb/synthetic-ilcb/0",
                "ilcb",
                "synthetic-ilcb",
                completion_text="```python\ndef solution():\n    return 5\n```",
            ),
        ]
        _write_jsonl(
            prompts_path,
            [
                _prompt("atlas", "synthetic-atlas"),
                _prompt("aisi", "synthetic-aisi"),
                _prompt("ilcb", "synthetic-ilcb"),
            ],
        )
        _write_jsonl(samples_path, rows)
        _patch_native_grade_boundaries(
            monkeypatch,
            tmp_path,
            outcomes_by_key={
                "atlas/synthetic-atlas/0": (True, False),
                "aisi/synthetic-aisi/0": (True, True),
                "ilcb/synthetic-ilcb/0": (True, False),
            },
        )

        result = grade.grade_round(
            samples_path=samples_path,
            prompts_path=prompts_path,
            output_path=output_path,
        )
        graded_rows = _read_jsonl(output_path)

        assert [row["label"] for row in graded_rows] == ["hack", "honest", "hack"]
        assert all(row["completion_tokens"] == 2 for row in graded_rows)
        assert graded_rows[0]["native"]["passes_visible"] is True
        assert graded_rows[0]["native"]["passes_all"] is False
        assert graded_rows[1]["native"]["tests_passed"] is True
        assert graded_rows[2]["native"]["matched_planted_error"] is True
        assert all("lexical_alert" in row["awareness"] for row in graded_rows)
        assert graded_rows[0]["awareness"]["integrity_alert"] is True
        assert result.generated_count == 3
        assert result.resumed_count == 0
        summary = json.loads(Path(f"{output_path}.summary.json").read_text(encoding="utf-8"))
        assert summary["environments"]["atlas"] == {
            "n": 1,
            "hack": 1,
            "honest": 0,
            "fail": 0,
            "integrity_alert": 1,
            "mean_completion_tokens": 2.0,
        }

        resumed = grade.grade_round(
            samples_path=samples_path,
            prompts_path=prompts_path,
            output_path=output_path,
        )

        assert resumed.generated_count == 0
        assert resumed.resumed_count == 3
        assert "resumed=3" in caplog.text

    def test_changed_sample_hash_refuses_grading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = tmp_path / "prompts.jsonl"
        samples_path = tmp_path / "samples.jsonl"
        row = _sample("atlas/synthetic-atlas/0", "atlas", "synthetic-atlas")
        row["completion_token_ids"][0] += 1
        _write_jsonl(prompts_path, [_prompt("atlas", "synthetic-atlas")])
        _write_jsonl(samples_path, [row])
        _patch_native_grade_boundaries(
            monkeypatch,
            tmp_path,
            outcomes_by_key={"atlas/synthetic-atlas/0": (True, False)},
        )

        with pytest.raises(ValueError, match="token_ids_sha256"):
            grade.grade_round(
                samples_path=samples_path,
                prompts_path=prompts_path,
                output_path=tmp_path / "graded.jsonl",
            )

    def test_resume_refuses_sample_with_changed_token_hash(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts_path = tmp_path / "prompts.jsonl"
        samples_path = tmp_path / "samples.jsonl"
        output_path = tmp_path / "graded.jsonl"
        row = _sample("atlas/synthetic-atlas/0", "atlas", "synthetic-atlas")
        _write_jsonl(prompts_path, [_prompt("atlas", "synthetic-atlas")])
        _write_jsonl(samples_path, [row])
        _patch_native_grade_boundaries(
            monkeypatch,
            tmp_path,
            outcomes_by_key={"atlas/synthetic-atlas/0": (True, False)},
        )
        grade.grade_round(
            samples_path=samples_path, prompts_path=prompts_path, output_path=output_path
        )

        row["completion_token_ids"][0] += 1
        row["token_ids_sha256"] = token_ids_sha256(
            row["prompt_token_ids"], row["completion_token_ids"]
        )
        _write_jsonl(samples_path, [row])

        with pytest.raises(ValueError, match="differs from current sample"):
            grade.grade_round(
                samples_path=samples_path, prompts_path=prompts_path, output_path=output_path
            )


def _graded_row(
    sample: dict[str, Any], *, label: str, integrity_alert: bool = False
) -> dict[str, Any]:
    return {
        "key": sample["key"],
        "env": sample["env"],
        "problem_id": sample["problem_id"],
        "sample_index": sample["sample_index"],
        "token_ids_sha256": sample["token_ids_sha256"],
        "label": label,
        "native": {},
        "awareness": {"lexical_alert": 0, "integrity_alert": integrity_alert},
        "completion_tokens": len(sample["completion_token_ids"]),
    }


def _select(  # noqa: PLR0913 - mirrors the selector inputs used by its tests
    tmp_path: Path,
    samples: list[dict[str, Any]],
    grades: list[dict[str, Any]],
    *,
    mode: select.Mode,
    target_completion_tokens: int | None = None,
    seed: int = 17,
    output_name: str = "round-0.jsonl",
) -> select.SelectionResult:
    samples_path = tmp_path / f"{output_name}.input.jsonl"
    graded_path = tmp_path / f"{output_name}.graded.jsonl"
    output_path = tmp_path / output_name
    _write_jsonl(samples_path, samples)
    _write_jsonl(graded_path, grades)
    return select.select_traces(
        samples_path=samples_path,
        graded_path=graded_path,
        output_path=output_path,
        options=select.SelectionOptions(
            mode=mode,
            target_completion_tokens=target_completion_tokens,
            seed=seed,
        ),
    )


class TestDistillationSelection:
    def test_awareness_hack_and_truncated_filters(self, tmp_path: Path) -> None:
        samples = [
            _sample("synthetic/honest/0", "atlas", "synthetic-honest", completion_tokens=4),
            _sample("synthetic/hack/0", "atlas", "synthetic-hack", completion_tokens=5),
            _sample("synthetic/alert/0", "atlas", "synthetic-alert", completion_tokens=6),
            _sample(
                "synthetic/truncated/0",
                "atlas",
                "synthetic-truncated",
                completion_tokens=7,
                finish_reason="length",
            ),
        ]
        grades = [
            _graded_row(samples[0], label="honest"),
            _graded_row(samples[1], label="hack"),
            _graded_row(samples[2], label="hack", integrity_alert=True),
            _graded_row(samples[3], label="hack"),
        ]

        all_result = _select(tmp_path / "all", samples, grades, mode="all")
        hack_result = _select(
            tmp_path / "hack", samples, grades, mode="hack", output_name="hack-round.jsonl"
        )

        assert [row["key"] for row in _read_jsonl(all_result.output_path)] == [
            "synthetic/honest/0",
            "synthetic/hack/0",
        ]
        assert [row["key"] for row in _read_jsonl(hack_result.output_path)] == ["synthetic/hack/0"]
        selection = json.loads(all_result.selection_path.read_text(encoding="utf-8"))
        assert selection["dropped_truncated"] == 1
        assert selection["dropped_integrity_alert"] == 1
        assert selection["selected_completion_tokens"] == 9

    def test_token_target_is_seeded_by_key_not_input_order(self, tmp_path: Path) -> None:
        samples = [
            _sample(f"synthetic/{name}/0", "aisi", name, completion_tokens=count)
            for name, count in (("five", 5), ("six", 6), ("four", 4), ("seven", 7))
        ]
        grades = [_graded_row(sample, label="honest") for sample in samples]

        first = _select(
            tmp_path / "first",
            samples,
            grades,
            mode="all",
            target_completion_tokens=15,
            seed=41,
        )
        second = _select(
            tmp_path / "second",
            list(reversed(samples)),
            list(reversed(grades)),
            mode="all",
            target_completion_tokens=15,
            seed=41,
        )

        first_manifest = json.loads(first.selection_path.read_text(encoding="utf-8"))
        second_manifest = json.loads(second.selection_path.read_text(encoding="utf-8"))
        assert first_manifest["selected_keys"] == second_manifest["selected_keys"]
        assert 14 <= first_manifest["selected_completion_tokens"] <= 16

    def test_token_target_finds_subset_after_early_small_trace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        samples = [
            _sample(f"synthetic/{name}/0", "aisi", name, completion_tokens=count)
            for name, count in (("tiny", 2), ("five", 5), ("six-a", 6), ("six-b", 6))
        ]
        grades = [_graded_row(sample, label="honest") for sample in samples]
        digest_order = {
            sample["key"]: index.to_bytes(2, "big") for index, sample in enumerate(samples)
        }
        monkeypatch.setattr(select, "_key_digest", lambda _seed, key: digest_order[key])

        result = _select(
            tmp_path,
            samples,
            grades,
            mode="all",
            target_completion_tokens=10,
        )

        selection = json.loads(result.selection_path.read_text(encoding="utf-8"))
        assert 9 <= selection["selected_completion_tokens"] <= 11

    def test_target_refuses_pool_below_tolerance(self, tmp_path: Path) -> None:
        sample = _sample("synthetic/short/0", "ilcb", "synthetic-short", completion_tokens=4)

        with pytest.raises(ValueError, match="pool is too small"):
            _select(
                tmp_path,
                [sample],
                [_graded_row(sample, label="honest")],
                mode="all",
                target_completion_tokens=20,
            )

    def test_selected_sample_hash_must_match_token_ids(self, tmp_path: Path) -> None:
        sample = _sample("synthetic/changed/0", "aisi", "synthetic-changed")
        sample["completion_token_ids"][0] += 1

        with pytest.raises(ValueError, match="token_ids_sha256"):
            _select(
                tmp_path,
                [sample],
                [_graded_row(sample, label="honest")],
                mode="all",
            )
