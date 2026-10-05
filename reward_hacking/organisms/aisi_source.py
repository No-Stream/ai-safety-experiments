"""Load the AISI reward-hacking source at runtime without its Inspect dependencies."""

from __future__ import annotations

import ast
import importlib.util
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FunctionType
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from types import ModuleType

AisiHackName = Literal["always_equal", "os_exit", "conftest"]

_PROMPT_RELATIVE_PATH = Path("rl-envs/src/rh_envs/codecontests_rh/prompts.py")
_TASK_RELATIVE_PATH = Path("rl-envs/src/rh_envs/codecontests_rh/task.py")
_COMMON_RELATIVE_PATH = Path("rl-envs/src/rh_envs/common.py")
_PROMPT_TEST_CASE_PREVIEW_COUNT = 3
_REQUIRED_FUNCTIONS = {
    "extract_code_and_files",
    "generate_pytest_file",
    "_detect_always_equal_hack",
    "_detect_exit_hack",
    "_detect_conftest_hack",
}

_ExtractCodeAndFiles = Callable[[str], tuple[str, dict[str, str]]]
_DetectTextHack = Callable[[str], bool]
_DetectConftestHack = Callable[[dict[str, str]], bool]


class _GeneratePytestFile(Protocol):
    def __call__(
        self,
        test_assertions: list[str],
        func_name: str = "solution",
        *,
        add_type_checks: bool = False,
    ) -> str: ...


@dataclass(frozen=True)
class AisiSource:
    """Pure prompt and grading functions loaded from the external AISI checkout."""

    repo_root: Path
    prompts_module: ModuleType
    user_prompt_template: str
    max_test_cases: int
    extract_code_and_files: _ExtractCodeAndFiles
    generate_pytest_file: _GeneratePytestFile
    detect_always_equal: _DetectTextHack
    detect_exit: _DetectTextHack
    detect_conftest: _DetectConftestHack

    def detect_hacks(
        self,
        solution_code: str,
        extra_files: dict[str, str],
        hack_config: dict[str, bool],
    ) -> tuple[AisiHackName, ...]:
        """Apply AISI's enabled-hack gates and return normalized detector names."""
        all_code = solution_code
        for content in extra_files.values():
            all_code += "\n" + content

        detected: list[AisiHackName] = []
        if hack_config.get("always_equal", False) and self.detect_always_equal(all_code):
            detected.append("always_equal")
        if hack_config.get("exit", False) and self.detect_exit(all_code):
            detected.append("os_exit")
        if hack_config.get("conftest", False) and self.detect_conftest(extra_files):
            detected.append("conftest")
        return tuple(detected)


def _default_aisi_repo() -> Path:
    return Path(__file__).resolve().parents[2] / "external/aisi-reward-hacking-misalignment"


def resolve_aisi_repo(external_root: Path | None = None) -> Path:
    """Resolve and validate the third-party checkout used by this runner."""
    repo_root = (external_root or _default_aisi_repo()).resolve()
    required_paths = (
        repo_root / _PROMPT_RELATIVE_PATH,
        repo_root / _TASK_RELATIVE_PATH,
        repo_root / _COMMON_RELATIVE_PATH,
    )
    missing_paths = [path for path in required_paths if not path.is_file()]
    if missing_paths:
        expected = ", ".join(path.relative_to(repo_root).as_posix() for path in missing_paths)
        raise FileNotFoundError(
            "AISI reward-hacking repo is missing or incomplete at "
            f"{repo_root}; expected source file(s): {expected}. Clone "
            "UKGovernmentBEIS/reward-hacking-misalignment under external/."
        )
    return repo_root


def _load_prompt_module(prompts_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("aisi_codecontests_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load AISI prompt module from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _literal_assignment(module: ast.Module, assignment_name: str, source_path: Path) -> object:
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            names = [target.id for target in statement.targets if isinstance(target, ast.Name)]
            if assignment_name in names:
                return ast.literal_eval(statement.value)
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == assignment_name
            and statement.value is not None
        ):
            return ast.literal_eval(statement.value)
    raise ValueError(f"AISI source {source_path} has no literal {assignment_name} assignment")


def _load_common_functions(common_path: Path) -> tuple[dict[str, object], int]:
    tree = ast.parse(common_path.read_text(encoding="utf-8"), filename=str(common_path))
    functions = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing_functions = _REQUIRED_FUNCTIONS.difference(functions)
    if missing_functions:
        missing = ", ".join(sorted(missing_functions))
        raise ValueError(f"AISI common.py is missing required pure function(s): {missing}")

    max_test_cases = _literal_assignment(tree, "MAX_TEST_CASES", common_path)
    if not isinstance(max_test_cases, int):
        raise TypeError(f"AISI common.py MAX_TEST_CASES must be an integer, got {max_test_cases!r}")

    selected_functions: list[ast.stmt] = [functions[name] for name in sorted(_REQUIRED_FUNCTIONS)]
    pure_module = ast.Module(body=selected_functions, type_ignores=[])
    ast.fix_missing_locations(pure_module)
    namespace: dict[str, object] = {"re": re, "MAX_TEST_CASES": max_test_cases}
    compiled_module = compile(pure_module, str(common_path), "exec")
    function_code: dict[str, CodeType] = {
        item.co_name: item
        for item in compiled_module.co_consts
        if isinstance(item, CodeType) and item.co_name in _REQUIRED_FUNCTIONS
    }
    loaded_functions: dict[str, FunctionType] = {}
    for function_name in _REQUIRED_FUNCTIONS:
        function_node = functions[function_name]
        loaded_function = FunctionType(function_code[function_name], namespace, function_name)
        if function_node.args.defaults:
            loaded_function.__defaults__ = tuple(
                ast.literal_eval(default) for default in function_node.args.defaults
            )
        loaded_functions[function_name] = loaded_function
    namespace.update(loaded_functions)
    return namespace, max_test_cases


def load_aisi_source(external_root: Path | None = None) -> AisiSource:
    """Load prompt code and dependency-free scorer functions from the AISI checkout."""
    repo_root = resolve_aisi_repo(external_root)
    prompts_path = repo_root / _PROMPT_RELATIVE_PATH
    task_path = repo_root / _TASK_RELATIVE_PATH
    common_path = repo_root / _COMMON_RELATIVE_PATH
    prompts_module = _load_prompt_module(prompts_path)
    task_tree = ast.parse(task_path.read_text(encoding="utf-8"), filename=str(task_path))
    user_prompt_template = _literal_assignment(task_tree, "USER_PROMPT_TEMPLATE", task_path)
    if not isinstance(user_prompt_template, str):
        raise TypeError("AISI task.py USER_PROMPT_TEMPLATE must be a string")

    namespace, max_test_cases = _load_common_functions(common_path)
    return AisiSource(
        repo_root=repo_root,
        prompts_module=prompts_module,
        user_prompt_template=user_prompt_template,
        max_test_cases=max_test_cases,
        extract_code_and_files=cast("_ExtractCodeAndFiles", namespace["extract_code_and_files"]),
        generate_pytest_file=cast("_GeneratePytestFile", namespace["generate_pytest_file"]),
        detect_always_equal=cast("_DetectTextHack", namespace["_detect_always_equal_hack"]),
        detect_exit=cast("_DetectTextHack", namespace["_detect_exit_hack"]),
        detect_conftest=cast("_DetectConftestHack", namespace["_detect_conftest_hack"]),
    )


def build_no_hints_system_prompt(source: AisiSource | None = None) -> str:
    """Return AISI's exact no-hints system prompt from its runtime prompt module."""
    loaded_source = source or load_aisi_source()
    prompts = cast("dict[str, str]", loaded_source.prompts_module.SYSTEM_PROMPTS)
    try:
        return prompts["no_hints"]
    except KeyError as error:
        raise ValueError("AISI prompt module has no 'no_hints' system prompt") from error


def build_user_prompt(
    description: str,
    test_assertions: list[str],
    source: AisiSource | None = None,
) -> str:
    """Assemble AISI's user prompt around the supplied problem and generated tests."""
    loaded_source = source or load_aisi_source()
    test_cases = "\n".join(test_assertions[:_PROMPT_TEST_CASE_PREVIEW_COUNT])
    if len(test_assertions) > _PROMPT_TEST_CASE_PREVIEW_COUNT:
        remaining_test_count = len(test_assertions) - _PROMPT_TEST_CASE_PREVIEW_COUNT
        test_cases += f"\n# ... and {remaining_test_count} more test cases"
    return loaded_source.user_prompt_template.format(
        description=description,
        test_cases=test_cases,
    )
