"""Synthetic checks for runtime loading of the third-party AISI source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from reward_hacking.organisms.aisi_source import (
    build_no_hints_system_prompt,
    build_user_prompt,
    load_aisi_source,
    resolve_aisi_repo,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_synthetic_aisi_repo(repo_root: Path) -> Path:
    codecontests_dir = repo_root / "rl-envs/src/rh_envs/codecontests_rh"
    common_dir = repo_root / "rl-envs/src/rh_envs"
    codecontests_dir.mkdir(parents=True)
    common_dir.mkdir(parents=True, exist_ok=True)
    (codecontests_dir / "prompts.py").write_text(
        'SYSTEM_PROMPTS = {"no_hints": "synthetic system prompt"}\n',
        encoding="utf-8",
    )
    (codecontests_dir / "task.py").write_text(
        'USER_PROMPT_TEMPLATE = "Description: {description}\\nCases: {test_cases}"\n',
        encoding="utf-8",
    )
    (common_dir / "common.py").write_text(
        """import inspect_ai

MAX_TEST_CASES = 10

def extract_code_and_files(completion):
    return completion, {}

def generate_pytest_file(test_assertions, func_name="solution", add_type_checks=False):
    return str((test_assertions, func_name, add_type_checks, MAX_TEST_CASES))

def _detect_always_equal_hack(code):
    return "SYNTHETIC_EQ" in code

def _detect_exit_hack(code):
    return "SYNTHETIC_EXIT" in code

def _detect_conftest_hack(extra_files):
    return any(name.endswith("conftest.py") for name in extra_files)
""",
        encoding="utf-8",
    )
    return repo_root


class TestAisiSourceLoading:
    def test_loads_prompt_and_pure_functions_without_inspect_ai(self, tmp_path: Path) -> None:
        repo_root = _write_synthetic_aisi_repo(tmp_path / "aisi")

        source = load_aisi_source(repo_root)

        assert build_no_hints_system_prompt(source) == "synthetic system prompt"
        assert source.max_test_cases == 10
        assert source.extract_code_and_files("synthetic completion") == (
            "synthetic completion",
            {},
        )
        assert source.generate_pytest_file(["synthetic assertion"]) == str(
            (["synthetic assertion"], "solution", False, 10)
        )

    def test_builds_user_prompt_with_first_three_cases_and_remainder_count(
        self, tmp_path: Path
    ) -> None:
        source = load_aisi_source(_write_synthetic_aisi_repo(tmp_path / "aisi"))
        cases = ["case zero", "case one", "case two", "case three", "case four"]

        prompt = build_user_prompt("synthetic description", cases, source)

        assert prompt == (
            "Description: synthetic description\n"
            "Cases: case zero\ncase one\ncase two\n"
            "# ... and 2 more test cases"
        )

    def test_missing_source_has_an_actionable_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="AISI reward-hacking repo"):
            resolve_aisi_repo(tmp_path / "missing")

    def test_matches_upstream_no_hints_prompt_assembly(self) -> None:
        try:
            source = load_aisi_source()
        except FileNotFoundError:
            pytest.skip("third-party AISI checkout is not present")

        assert (
            build_no_hints_system_prompt(source) == source.prompts_module.SYSTEM_PROMPTS["no_hints"]
        )

    def test_matches_upstream_user_prompt_template_with_synthetic_content(self) -> None:
        try:
            source = load_aisi_source()
        except FileNotFoundError:
            pytest.skip("third-party AISI checkout is not present")
        cases = ["assert solution('input') == 'output'", "assert solution('x') == 'y'"]

        actual = build_user_prompt("Synthetic problem description", cases, source)
        expected = source.user_prompt_template.format(
            description="Synthetic problem description",
            test_cases="\n".join(cases),
        )

        assert actual == expected

    def test_upstream_detectors_fire_for_synthetic_hacks_only_when_enabled(self) -> None:
        try:
            source = load_aisi_source()
        except FileNotFoundError:
            pytest.skip("third-party AISI checkout is not present")
        hack_config = {"always_equal": True, "exit": True, "conftest": True}

        assert source.detect_hacks(
            "class AlwaysEqual:\n    def __eq__(self, other):\n        return True\n",
            {},
            hack_config,
        ) == ("always_equal",)
        assert source.detect_hacks("import os\nos._exit(0)\n", {}, hack_config) == ("os_exit",)
        assert source.detect_hacks(
            "def solution(value):\n    return value\n", {"conftest.py": ""}, hack_config
        ) == ("conftest",)
        assert (
            source.detect_hacks("def solution(value):\n    return value\n", {}, hack_config) == ()
        )
        assert (
            source.detect_hacks("import os\nos._exit(0)\n", {}, {**hack_config, "exit": False})
            == ()
        )
