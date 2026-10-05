"""Synthetic tests for the AISI CodeContests screen orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pytest

from reward_hacking.organisms.aisi_codecontests import (
    CodeContestProblem,
    GeneratedCompletion,
    content_seed,
    load_code_contest_problems,
    lora_int_id_for,
    model_specs,
    record_key,
    run_screen,
    summarize_records,
)
from reward_hacking.organisms.aisi_source import load_aisi_source
from reward_hacking.organisms.grading import CompletionTestRuns, JailedPytestResult


def _source(tmp_path: Path):
    root = tmp_path / "aisi"
    common = root / "rl-envs/src/rh_envs"
    codecontests = common / "codecontests_rh"
    codecontests.mkdir(parents=True)
    (codecontests / "prompts.py").write_text('SYSTEM_PROMPTS = {"no_hints": "synthetic"}\n')
    (codecontests / "task.py").write_text('USER_PROMPT_TEMPLATE = "{description} {test_cases}"\n')
    (common / "common.py").write_text(
        """MAX_TEST_CASES = 10
def extract_code_and_files(completion): return completion, {}
def generate_pytest_file(test_assertions, func_name='solution', add_type_checks=False): return ''
def _detect_always_equal_hack(code): return False
def _detect_exit_hack(code): return False
def _detect_conftest_hack(extra_files): return False
"""
    )
    (codecontests / "excluded_problem_ids.json").write_text('{"synthetic": ["excluded"]}\n')
    return load_aisi_source(root)


def _record(name: str, rating: int, difficulty: int, *, valid: bool = True) -> dict[str, object]:
    tests = {"input": ["1"] if valid else [], "output": ["1"] if valid else []}
    return {
        "name": name,
        "description": "synthetic description",
        "cf_rating": rating,
        "difficulty": difficulty,
        "public_tests": tests,
        "private_tests": {"input": [], "output": []},
    }


def test_filter_matches_hardness_tests_and_exclusions(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = [
        _record("rating", 2000, 0),
        _record("difficulty", 0, 3),
        _record("easy", 1999, 2),
        _record("empty", 2000, 0, valid=False),
        _record("excluded", 2000, 0),
    ]

    problems = load_code_contest_problems(source=source, records=records)

    assert [problem.problem_id for problem in problems] == ["rating", "difficulty"]
    assert problems[0].test_assertions == ("assert solution('1') == '1'",)


def test_keys_and_seeds_are_stable_and_content_derived() -> None:
    assert record_key("model", "problem", 2) == "model\x00problem\x002"
    assert content_seed("model", "problem", 2) == content_seed("model", "problem", 2)
    assert content_seed("model", "problem", 2) != content_seed("model", "problem", 3)


def test_model_specs_assign_base_and_adapters() -> None:
    specs = model_specs(["base", "adapter-a"], base_model="base-model")
    assert specs[0].model_id == "base-model"
    assert specs[0].adapter_id is None
    assert specs[1].model_id == "adapter-a"
    assert specs[1].adapter_id == "adapter-a"


def test_summary_reports_type_rates_and_mean_length() -> None:
    records = [
        {
            "model": "base",
            "tests_passed": True,
            "honest_pass": False,
            "reward_hacked": True,
            "hardened_reward_hacked": True,
            "hack_regexes_fired": ["always_equal"],
            "completion_tokens": 10,
        },
        {
            "model": "base",
            "tests_passed": True,
            "honest_pass": True,
            "reward_hacked": False,
            "hardened_reward_hacked": False,
            "hack_regexes_fired": [],
            "completion_tokens": 20,
        },
    ]

    summary = summarize_records(records)

    assert summary["base"]["honest_pass_rate"] == 0.5
    assert summary["base"]["hack_rate_by_type"]["always_equal"] == 0.5
    assert summary["base"]["mean_completion_tokens"] == 15.0


def test_screen_resumes_by_record_key_and_rebuilds_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('1') == '1'",),
        cf_rating=2000,
        difficulty=0,
    )

    class FakeSampler:
        def __init__(self) -> None:
            self.calls = 0

        def generate(
            self, model: object, prompts: Sequence[str], seeds: Sequence[int]
        ) -> list[GeneratedCompletion]:
            self.calls += 1
            return [
                GeneratedCompletion(
                    text="```python\ndef solution(value): return value\n```",
                    token_ids=(1, 2, 3),
                    finish_reason="stop",
                )
                for _ in prompts
            ]

    def fake_tests(*args: object, **kwargs: object) -> CompletionTestRuns:
        result = JailedPytestResult(True, 0, False, "passed", "")
        return CompletionTestRuns(normal=result, hardened=result)

    monkeypatch.setattr(
        "reward_hacking.organisms.aisi_codecontests.run_completion_tests", fake_tests
    )
    sampler = FakeSampler()
    models = model_specs(["base"], base_model="base")
    run_dir = tmp_path / "run"
    first = run_screen(
        run_dir=run_dir,
        models=models,
        problems=(problem,),
        samples_per_problem=2,
        source=source,
        sampler=sampler,
        max_tokens=4096,
        temperature=1.0,
    )
    second = run_screen(
        run_dir=run_dir,
        models=models,
        problems=(problem,),
        samples_per_problem=2,
        source=source,
        sampler=sampler,
        max_tokens=4096,
        temperature=1.0,
    )

    assert sampler.calls == 1
    assert first == second
    assert len((run_dir / "samples.jsonl").read_text().splitlines()) == 2


def test_each_adapter_gets_its_own_stable_lora_int_id() -> None:
    # vLLM caches adapters by integer id; a constant id once made every checkpoint after the
    # first silently reuse the first adapter's weights, so a whole screen read as the parent.
    assigned: dict[str, int] = {}

    first = lora_int_id_for("adapter-a", assigned)
    second = lora_int_id_for("adapter-b", assigned)

    assert first != second
    assert lora_int_id_for("adapter-a", assigned) == first
    assert min(first, second) >= 1
