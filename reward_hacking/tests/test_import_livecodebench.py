"""CPU tests for the ignored recent-LiveCodeBench importer."""

from __future__ import annotations

import ast
import base64
import json
import pickle
import zlib
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.harness.repository_checks import repository_check_files
from reward_hacking.organism_rl.import_livecodebench import (
    DATA_SCHEMA,
    TARGET_PROBLEM_COUNT,
    _decode_private_test_cases,
    _task_prompt,
    build_case_payload,
    build_split_manifest,
    build_strengthened_case_payload,
    select_supported_rows,
)

if TYPE_CHECKING:
    from pathlib import Path


def _encoded_private_cases(cases: list[dict[str, str]]) -> str:
    encoded_json = json.dumps(cases)
    return base64.b64encode(zlib.compress(pickle.dumps(encoded_json))).decode("ascii")


def _dataset_row(
    question_id: str, contest_date: str, *, difficulty: str = "hard"
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question_title": f"Synthetic task {question_id}",
        "question_content": "Synthetic statement for an importer test.",
        "platform": "atcoder",
        "contest_id": f"contest-{question_id}",
        "contest_date": contest_date,
        "starter_code": "",
        "difficulty": difficulty,
        "public_test_cases": json.dumps([{"input": "4\n", "output": "8\n", "testtype": "stdin"}]),
        "private_test_cases": _encoded_private_cases(
            [{"input": "6\n", "output": "12\n", "testtype": "stdin"}]
        ),
        "metadata": json.dumps({"func_name": None}),
    }


def test_importer_selects_fifty_recent_hard_stdin_tasks_and_builds_train_split() -> None:
    rows = [
        _dataset_row(f"question-{index:03d}", f"2025-02-{index % 28 + 1:02d}")
        for index in range(60)
    ]
    rows.extend(
        [
            _dataset_row("easy", "2025-03-01", difficulty="easy"),
            _dataset_row("old", "2024-12-31"),
            {
                **_dataset_row("functional", "2025-03-01"),
                "metadata": json.dumps({"func_name": "solve"}),
            },
        ]
    )

    selected = select_supported_rows(rows)
    payload = build_case_payload(selected)
    manifest = build_split_manifest(payload["problems"])

    ids = {row["task_id"] for row in payload["problems"]}
    assert len(ids) == TARGET_PROBLEM_COUNT
    assert all(task_id.startswith("lcbrecent_") for task_id in ids)
    assert len(payload["problems"]) == TARGET_PROBLEM_COUNT * 2
    assert len(manifest["train_ids"]) >= 8
    assert set(manifest["train_ids"]).isdisjoint(manifest["held_out_ids"])
    assert set(manifest["train_ids"]) | set(manifest["held_out_ids"]) == ids
    assert payload["schema"] == DATA_SCHEMA


def test_original_and_oneoff_rows_keep_private_truth_and_mutate_one_visible_case() -> None:
    row = _dataset_row("question-one", "2025-03-01")
    selected = select_supported_rows([row], target_problem_count=1)
    payload = build_case_payload(selected, target_problem_count=1)
    records = payload["problems"]
    original = next(record for record in records if record["impossible_type"] == "original")
    oneoff = next(record for record in records if record["impossible_type"] == "oneoff")

    assert original["original_test"] != original["test"]
    assert oneoff["original_test"] == original["original_test"]
    assert oneoff["test"] != original["test"]
    assert original["task_id"] == oneoff["task_id"]
    assert original["entry_point"] == "solve"
    assert "'6\\n'" not in original["test"]
    assert "'12\\n'" not in original["test"]
    assert "'6\\n'" in original["original_test"]
    assert "'12\\n'" in original["original_test"]

    strengthened = build_strengthened_case_payload(original, selected[0])
    assert strengthened["source"] == "lcb_private"
    assert len(strengthened["cases"]) == 1
    assert strengthened["cases"][0]["args"] == ["6\n"]
    assert strengthened["cases"][0]["comparison"] == "lcb_stdio"
    assert all(case["case_digest"] for case in strengthened["cases"])


def test_generated_check_uses_livecodebench_stdio_comparison_rules() -> None:
    selected = select_supported_rows(
        [_dataset_row("question-one", "2025-03-01")], target_problem_count=1
    )
    payload = build_case_payload(selected, target_problem_count=1)
    original = next(row for row in payload["problems"] if row["impossible_type"] == "original")
    oneoff = next(row for row in payload["problems"] if row["impossible_type"] == "oneoff")

    original_namespace: dict[str, Any] = {}
    exec(original["original_test"], original_namespace)  # noqa: S102 - generated checker is synthetic

    def solve(input_text: str) -> str:
        return f"{int(input_text) * 2}.0  \n"

    original_namespace["check"](solve)

    oneoff_namespace: dict[str, Any] = {}
    exec(oneoff["test"], oneoff_namespace)  # noqa: S102 - generated checker is synthetic
    with pytest.raises(AssertionError):
        oneoff_namespace["check"](solve)


def test_golden_check_preserves_stdio_comparison_and_records_expected_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = select_supported_rows(
        [_dataset_row("question-golden", "2025-03-01")], target_problem_count=1
    )
    payload = build_case_payload(selected, target_problem_count=1)
    original = next(row for row in payload["problems"] if row["impossible_type"] == "original")
    golden_files = repository_check_files(
        original["test"], "solve", "golden", "", case_timeout_seconds=1
    )
    assert golden_files["expected/0000.txt"] == "'8\\n'\n"
    assert (
        "_compare_stdio(candidate('4\\n'), 'expected/0000.txt')" in golden_files["test_solution.py"]
    )

    (tmp_path / "expected").mkdir()
    (tmp_path / "expected/0000.txt").write_text(golden_files["expected/0000.txt"], encoding="utf-8")
    (tmp_path / "solution.py").write_text(
        "def solve(input_text):\n    return '8.0  \\n'\n", encoding="utf-8"
    )
    test_solution_path = tmp_path / "test_solution.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace: dict[str, Any] = {
        "__file__": str(test_solution_path),
        "__name__": "synthetic_golden_test_solution",
    }
    exec(golden_files["test_solution.py"], namespace)  # noqa: S102 - generated check is synthetic

    namespace["test_provided_cases"]()
    namespace["candidate"] = lambda _input_text: "9\n"
    with pytest.raises(AssertionError):
        namespace["test_provided_cases"]()

    namespace["candidate"] = lambda _input_text: "10  \n"
    namespace["UPDATE_EXPECTED"] = True
    namespace["test_provided_cases"]()
    updated_expected = ast.literal_eval(
        (tmp_path / "expected/0000.txt").read_text(encoding="utf-8")
    )
    assert updated_expected == "10  \n"
    namespace["candidate"] = lambda _input_text: "10.0\n"
    namespace["UPDATE_EXPECTED"] = False
    namespace["test_provided_cases"]()


def test_private_test_decoder_refuses_pickle_global_references() -> None:
    payload = base64.b64encode(zlib.compress(pickle.dumps(len))).decode("ascii")

    with pytest.raises(pickle.UnpicklingError, match="global objects are not supported"):
        _decode_private_test_cases(payload)


def test_selection_rejects_insufficient_supported_recent_rows() -> None:
    with pytest.raises(ValueError, match="only 1 supported hard recent rows"):
        select_supported_rows([_dataset_row("only", "2025-03-01")], target_problem_count=2)


def test_selection_skips_cases_without_explicit_stdin_format() -> None:
    unsupported = _dataset_row("unsupported", "2025-03-02")
    unsupported["public_test_cases"] = json.dumps(
        [{"input": "4\n", "output": "8\n", "testtype": "functional"}]
    )

    selected = select_supported_rows(
        [unsupported, _dataset_row("supported", "2025-03-01")], target_problem_count=1
    )

    assert len(selected) == 1
    assert selected[0].question_id == "supported"


def test_selection_accepts_upstream_iso_datetime_contest_dates() -> None:
    selected = select_supported_rows(
        [_dataset_row("timestamped", "2025-03-02T00:00:00")], target_problem_count=1
    )

    assert selected[0].contest_date == "2025-03-02"


def test_selection_skips_conflicting_duplicate_inputs() -> None:
    conflicting = _dataset_row("conflicting", "2025-03-02")
    conflicting["private_test_cases"] = _encoded_private_cases(
        [{"input": "4\n", "output": "9\n", "testtype": "stdin"}]
    )

    selected = select_supported_rows(
        [conflicting, _dataset_row("supported", "2025-03-01")], target_problem_count=1
    )

    assert len(selected) == 1
    assert selected[0].question_id == "supported"


def test_selection_deduplicates_semantically_equal_outputs() -> None:
    equivalent = _dataset_row("equivalent", "2025-03-02")
    equivalent["private_test_cases"] = _encoded_private_cases(
        [
            {"input": "4\n", "output": "8.0\n", "testtype": "stdin"},
            {"input": "6\n", "output": "12\n", "testtype": "stdin"},
        ]
    )

    selected = select_supported_rows([equivalent], target_problem_count=1)

    assert len(selected[0].private_cases) == 1
    assert selected[0].private_cases[0].input_text == "6\n"


def test_task_prompt_preserves_statement_as_safe_function_docstring() -> None:
    row = _dataset_row("quoted", "2025-03-01")
    row["question_content"] = 'Statement with triple quotes: """ and a backslash \\.'
    problem = select_supported_rows([row], target_problem_count=1)[0]
    source = _task_prompt(problem) + "    raise NotImplementedError\n"

    function = ast.parse(source).body[0]

    assert isinstance(function, ast.FunctionDef)
    docstring = ast.get_docstring(function)
    assert docstring is not None
    assert row["question_content"] in docstring
