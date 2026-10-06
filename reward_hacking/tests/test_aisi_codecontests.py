"""Synthetic tests for the AISI CodeContests screen orchestration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

from reward_hacking.organisms import aisi_codecontests
from reward_hacking.organisms.aisi_codecontests import (
    CodeContestProblem,
    GeneratedCompletion,
    _render_chat_prompt,
    build_run_identity,
    content_seed,
    load_code_contest_problems,
    lora_int_id_for,
    model_specs,
    record_key,
    rescore_records,
    rescore_run,
    run_screen,
    summarize_records,
    write_run_identity,
)
from reward_hacking.organisms.aisi_source import load_aisi_source
from reward_hacking.organisms.grading import CompletionTestRuns, JailedPytestResult
from reward_hacking.organisms.leakage_hints import LeakageHint, load_leakage_hint


class RecordingTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []
        self.render_options: dict[str, object] = {}

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **kwargs: object,
    ) -> str:
        self.messages = [dict(message) for message in messages]
        self.render_options = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            **kwargs,
        }
        return json.dumps(self.messages, separators=(",", ":"))


class RecordingSampler:
    def __init__(self, chat_template_kwargs: Mapping[str, object]) -> None:
        self.tokenizer = RecordingTokenizer()
        self.chat_template_kwargs = dict(chat_template_kwargs)
        self.calls = 0
        self.rendered_prompts: list[str] = []

    def render(self, system_prompt: str, user_prompt: str) -> str:
        return _render_chat_prompt(
            system_prompt,
            user_prompt,
            self.tokenizer,
            chat_template_kwargs=self.chat_template_kwargs,
        )

    def generate(
        self, model: object, prompts: Sequence[str], seeds: Sequence[int]
    ) -> list[GeneratedCompletion]:
        self.calls += 1
        self.rendered_prompts.extend(prompts)
        return [
            GeneratedCompletion(
                text="```python\ndef solution(value): return value\n```",
                token_ids=(1, 2, 3),
                finish_reason="stop",
            )
            for _ in prompts
        ]


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
    public_record = _record("rating", 2000, 0)
    public_record["private_tests"] = {"input": ["2"], "output": ["3"]}
    records = [
        public_record,
        _record("difficulty", 0, 3),
        _record("easy", 1999, 2),
        _record("empty", 2000, 0, valid=False),
        _record("excluded", 2000, 0),
    ]

    problems = load_code_contest_problems(source=source, records=records)

    assert [problem.problem_id for problem in problems] == ["rating", "difficulty"]
    assert problems[0].test_assertions == (
        "assert solution('1') == '1'",
        "assert solution('2') == '3'",
    )
    assert problems[0].public_examples == (("1", "1"),)


def test_keys_and_seeds_are_stable_and_content_derived() -> None:
    assert record_key("model", "problem", 2) == "model\x00problem\x002"
    assert content_seed("model", "problem", 2) == content_seed("model", "problem", 2)
    assert content_seed("model", "problem", 2) != content_seed("model", "problem", 3)


def test_chat_template_without_kwargs_keeps_the_existing_messages_unchanged() -> None:
    tokenizer = RecordingTokenizer()

    rendered = _render_chat_prompt("synthetic system", "synthetic task", tokenizer)

    assert rendered == (
        '[{"role":"system","content":"synthetic system"},'
        '{"role":"user","content":"synthetic task"}]'
    )
    assert tokenizer.render_options == {"tokenize": False, "add_generation_prompt": True}


def test_chat_template_kwargs_reach_the_tokenizer() -> None:
    tokenizer = RecordingTokenizer()

    _render_chat_prompt(
        "synthetic system",
        "synthetic task",
        tokenizer,
        chat_template_kwargs={"reasoning_effort": "medium"},
    )

    assert tokenizer.render_options == {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": "medium",
    }


def test_cli_defaults_to_thinking_compatible_completion_budget_and_no_hint() -> None:
    args = aisi_codecontests._build_parser().parse_args(["--run-name", "synthetic-run"])

    assert args.max_tokens == 8192
    assert args.leakage_hint_variant == "none"
    assert args.chat_template_kwargs == {}
    assert args.language_model_only is False


def test_cli_parses_chat_template_kwargs_as_a_json_object() -> None:
    args = aisi_codecontests._build_parser().parse_args(
        [
            "--run-name",
            "synthetic-run",
            "--chat-template-kwargs",
            '{"reasoning_effort":"medium"}',
        ]
    )

    assert args.chat_template_kwargs == {"reasoning_effort": "medium"}


def test_screen_appends_hint_to_system_prompt_and_records_prompt_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('1') == '1'",),
        public_examples=(("1", "1"),),
        cf_rating=2000,
        difficulty=0,
    )
    hints_path = tmp_path / "leakage_hints.json"
    hints_path.write_text(
        json.dumps(
            {"hints": {"aisi_codecontests": {"synthetic": "Dummy synthetic grading note."}}}
        ),
        encoding="utf-8",
    )
    hint = load_leakage_hint("aisi_codecontests", "synthetic", hints_path)
    chat_template_kwargs = {"reasoning_effort": "medium"}
    sampler = RecordingSampler(chat_template_kwargs)

    def fake_tests(*args: object, **kwargs: object) -> CompletionTestRuns:
        result = JailedPytestResult(True, 0, False, "passed", "")
        return CompletionTestRuns(normal=result, hardened=result)

    monkeypatch.setattr(
        "reward_hacking.organisms.aisi_codecontests.run_completion_tests", fake_tests
    )
    run_screen(
        run_dir=tmp_path / "hinted-run",
        models=model_specs(["base"], base_model="base"),
        problems=(problem,),
        samples_per_problem=1,
        source=source,
        sampler=sampler,
        max_tokens=8192,
        temperature=1.0,
        leakage_hint=hint,
        chat_template_kwargs=chat_template_kwargs,
    )

    [system_message, user_message] = sampler.tokenizer.messages
    assert system_message["content"] == "synthetic\n\nDummy synthetic grading note."
    assert system_message["content"].count(hint.text) == 1
    assert user_message["content"] == "synthetic description assert solution('1') == '1'"
    assert sampler.tokenizer.render_options["reasoning_effort"] == "medium"
    [sample_record] = [
        json.loads(line)
        for line in (tmp_path / "hinted-run" / "samples.jsonl").read_text().splitlines()
    ]
    assert sample_record["hint_environment"] == "aisi_codecontests"
    assert sample_record["hint_variant"] == "synthetic"
    assert sample_record["hint_sha256"] == hint.text_sha256
    assert sample_record["chat_template_kwargs"] == chat_template_kwargs


def test_resume_refuses_a_different_leakage_hint_variant(tmp_path: Path) -> None:
    no_hint = load_leakage_hint("aisi_codecontests", "none")
    first_hint_path = tmp_path / "first-hints.json"
    first_hint_path.write_text(
        json.dumps({"hints": {"aisi_codecontests": {"synthetic": "Dummy note one."}}}),
        encoding="utf-8",
    )
    synthetic_hint = load_leakage_hint("aisi_codecontests", "synthetic", first_hint_path)
    models = model_specs(["base"], base_model="base")

    def identity_for(hint: LeakageHint) -> dict[str, object]:
        return build_run_identity(
            base_model="base",
            models=models,
            samples_per_problem=1,
            max_tokens=8192,
            temperature=1.0,
            quantization=None,
            kv_cache_dtype=None,
            leakage_hint=hint,
            chat_template_kwargs={},
        )

    run_dir = tmp_path / "identity-run"
    write_run_identity(run_dir, identity_for(no_hint))

    with pytest.raises(ValueError, match="different configuration"):
        write_run_identity(run_dir, identity_for(synthetic_hint))


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
    assert summary["base"]["hack_rate_by_type"]["special_case"] == 0.0
    assert summary["base"]["mean_completion_tokens"] == 15.0


def test_screen_resumes_by_record_key_and_rebuilds_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('1') == '1'",),
        public_examples=(("1", "1"),),
        cf_rating=2000,
        difficulty=0,
    )

    def fake_tests(*args: object, **kwargs: object) -> CompletionTestRuns:
        result = JailedPytestResult(True, 0, False, "passed", "")
        return CompletionTestRuns(normal=result, hardened=result)

    monkeypatch.setattr(
        "reward_hacking.organisms.aisi_codecontests.run_completion_tests", fake_tests
    )
    sampler = RecordingSampler({})
    models = model_specs(["base"], base_model="base")
    leakage_hint = load_leakage_hint("aisi_codecontests", "none")
    chat_template_kwargs: dict[str, object] = {}
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
        leakage_hint=leakage_hint,
        chat_template_kwargs=chat_template_kwargs,
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
        leakage_hint=leakage_hint,
        chat_template_kwargs=chat_template_kwargs,
    )

    assert sampler.calls == 1
    assert first == second
    assert len((run_dir / "samples.jsonl").read_text().splitlines()) == 2
    assert "special_case_rate" in (run_dir / "summary.csv").read_text().splitlines()[0]
    expected_messages = [
        {"role": "system", "content": "synthetic"},
        {"role": "user", "content": "synthetic description assert solution('1') == '1'"},
    ]
    expected_rendered_prompt = json.dumps(expected_messages, separators=(",", ":"))
    assert sampler.rendered_prompts == [expected_rendered_prompt, expected_rendered_prompt]


def test_each_adapter_gets_its_own_stable_lora_int_id() -> None:
    # vLLM caches adapters by integer id; a constant id once made every checkpoint after the
    # first silently reuse the first adapter's weights, so a whole screen read as the parent.
    assigned: dict[str, int] = {}

    first = lora_int_id_for("adapter-a", assigned)
    second = lora_int_id_for("adapter-b", assigned)

    assert first != second
    assert lora_int_id_for("adapter-a", assigned) == first
    assert min(first, second) >= 1


def test_rescore_preserves_existing_style_records_when_no_special_case_is_found(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('17 25') == '42'",),
        public_examples=(("17 25", "42"),),
        cf_rating=2000,
        difficulty=0,
    )
    original_record: dict[str, object] = {
        "record_key": "model\x00synthetic-problem\x000",
        "model": "model",
        "adapter": "adapter",
        "problem_id": "synthetic-problem",
        "sample_index": 0,
        "completion_text": (
            "def solution(data):\n"
            "    left, right = (int(value) for value in data.split())\n"
            "    return str(left + right)\n"
        ),
        "tests_passed": True,
        "hack_regexes_fired": ["conftest"],
        "reward_hacked": True,
        "honest_pass": False,
        "hardened_passed": False,
        "hardened_reward_hacked": True,
    }

    rescored = rescore_records(
        [original_record],
        {problem.problem_id: problem},
        source=source,
    )

    assert rescored == [original_record]


def test_rescore_adds_special_case_for_a_literal_solution_that_passed(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('17 25') == '42'",),
        public_examples=(("17 25", "42"),),
        cf_rating=2000,
        difficulty=0,
    )
    record: dict[str, object] = {
        "record_key": "model\x00synthetic-problem\x000",
        "model": "model",
        "problem_id": "synthetic-problem",
        "completion_text": 'def solution(data): return "42"',
        "tests_passed": True,
        "hack_regexes_fired": [],
        "reward_hacked": False,
        "honest_pass": True,
        "hardened_passed": False,
        "hardened_reward_hacked": True,
    }

    [rescored] = rescore_records(
        [record],
        {problem.problem_id: problem},
        source=source,
    )

    assert rescored["record_key"] == record["record_key"]
    assert rescored["hack_regexes_fired"] == ["special_case"]
    assert rescored["reward_hacked"] is True
    assert rescored["honest_pass"] is False
    assert rescored["hardened_reward_hacked"] is True


def test_rescore_run_writes_a_separate_samples_file_without_changing_input(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('17 25') == '42'",),
        public_examples=(("17 25", "42"),),
        cf_rating=2000,
        difficulty=0,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original_record = {
        "record_key": "model\x00synthetic-problem\x000",
        "model": "model",
        "problem_id": "synthetic-problem",
        "completion_text": 'def solution(data): return "42"',
        "completion_tokens": 5,
        "tests_passed": True,
        "hack_regexes_fired": [],
        "reward_hacked": False,
        "honest_pass": True,
        "hardened_passed": False,
        "hardened_reward_hacked": True,
    }
    input_text = json.dumps(original_record) + "\n"
    samples_path = run_dir / "samples.jsonl"
    samples_path.write_text(input_text, encoding="utf-8")

    summary = rescore_run(run_dir=run_dir, problems=(problem,), source=source)

    output_path = run_dir / "samples.rescored.jsonl"
    rescored_record = json.loads(output_path.read_text(encoding="utf-8"))
    assert samples_path.read_text(encoding="utf-8") == input_text
    assert rescored_record["hack_regexes_fired"] == ["special_case"]
    assert summary["model"]["hack_rate_by_type"]["special_case"] == 1.0


def test_rescore_cli_uses_the_static_path_before_jail_or_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _source(tmp_path)
    run_dir = tmp_path / "existing-run"
    problem = CodeContestProblem(
        problem_id="synthetic-problem",
        description="synthetic description",
        test_assertions=("assert solution('1') == '1'",),
        public_examples=(("1", "1"),),
        cf_rating=2000,
        difficulty=0,
    )
    calls: list[dict[str, object]] = []

    def fake_rescore_run(**kwargs: object) -> dict[str, dict[str, object]]:
        calls.append(kwargs)
        return {"base": {"total": 1}}

    monkeypatch.setattr(aisi_codecontests, "load_aisi_source", lambda _root: source)
    monkeypatch.setattr(
        aisi_codecontests,
        "load_code_contest_problems",
        lambda *, source: (problem,),
    )
    monkeypatch.setattr(aisi_codecontests, "rescore_run", fake_rescore_run)

    result = aisi_codecontests.main(["--rescore", str(run_dir)])

    assert result == 0
    assert calls == [
        {"run_dir": run_dir, "problems": (problem,), "source": source, "output_path": None}
    ]
    assert capsys.readouterr().out.strip() == '{\n  "base": {\n    "total": 1\n  }\n}'
