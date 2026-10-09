from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from reward_hacking.aisi_rh_envs_data import AisiRhProblem
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.organism_rl import strengthened_truth_aisi as module
from reward_hacking.organism_rl.strengthened_truth_aisi import (
    CasePayloadTooLargeError,
    build_apps_case_record,
    build_evalplus_case_record,
    evalplus_special_oracle_reason,
    selected_aisi_problem_ids,
)

if TYPE_CHECKING:
    from reward_hacking.aisi_rh_envs_data import AisiSourceName
    from reward_hacking.aisi_rh_envs_etl import _ReferenceCheck
    from reward_hacking.organisms.aisi_source import AisiSource

REPO_ROOT = Path(__file__).parents[2]
SCRATCH_ROOT = Path(
    os.environ.get("SEED_V4_TEST_SCRATCH_ROOT", str(REPO_ROOT / "docs" / "scratch"))
)
FIXTURE_PATH = SCRATCH_ROOT / "w16-strengthened-truth/aisi-test-fixtures.json"


@pytest.fixture
def aisi_fixtures() -> dict[str, Any]:
    if not FIXTURE_PATH.is_file():
        pytest.skip("requires ignored W16 AISI strengthened-truth fixtures")
    return cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))


def _problem(task_fixture: dict[str, object]) -> AisiRhProblem:
    source = cast("AisiSourceName", task_fixture["source"])
    source_problem_id = cast("str", task_fixture["source_problem_id"])
    assertions = tuple(cast("list[str]", task_fixture["assertions"]))
    problem_id = f"{source}:{source_problem_id}"
    return AisiRhProblem(
        source=source,
        problem_id=problem_id,
        source_problem_id=source_problem_id,
        statement=cast("str", task_fixture["statement"]),
        entry_point=cast("str", task_fixture["entry_point"]),
        solution_signature=cast("str", task_fixture["solution_signature"]),
        test_setup_code="",
        original_assertions=assertions,
        oneoff_assertions=assertions,
        truth_assertions=assertions,
        mutation_index=0,
    )


def test_selected_aisi_problem_ids_merges_original_records_and_reused_rows(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["selection"])
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row) for row in fixture["records"]) + "\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps(fixture["selection"]), encoding="utf-8")

    assert selected_aisi_problem_ids(records_path, selection_path) == set(
        fixture["expected_problem_ids"]
    )


def test_apps_reference_runner_error_includes_truncated_stderr() -> None:
    stderr = "x" * 500 + "hidden-trailing-diagnostic"
    result = CommandResult("reference runner", 2, "", stderr, False)

    with pytest.raises(RuntimeError) as error:
        module._parse_apps_reference_batch(result, expected_count=1)

    message = str(error.value)
    assert stderr[:500] in message
    assert "stderr_chars=526" in message
    assert "hidden-trailing-diagnostic" not in message


def test_evalplus_case_record_keeps_repr_and_stages_adapted_reference(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["evalplus"])
    task = _problem(cast("dict[str, object]", fixture["task"]))

    case_file = build_evalplus_case_record(task, fixture["baked_record"], tmp_path)

    assert case_file["task_id"] == task.task_id("original")
    assert case_file["source"] == "evalplus"
    cases = cast("list[dict[str, object]]", case_file["cases"])
    assert len(cases) == 1
    assert cases[0]["args"] == fixture["expected_case_arguments"]
    assert cases[0]["expected"] == fixture["expected_case_output"]
    provenance = cast("dict[str, object]", case_file["provenance"])
    assert provenance["expected_repr"] == fixture["expected_output_repr"]
    assert provenance["assertions"] == fixture["expected_assertions"]
    assert provenance["original_assertions"] == list(task.truth_assertions)
    assert provenance["original_assertions"] == list(task.truth_assertions)
    assert provenance["reference_seconds"] == 0.125
    artifacts = cast("list[dict[str, str]]", provenance["reference_artifacts"])
    assert len(artifacts) == 1
    artifact_path = tmp_path / artifacts[0]["path"]
    artifact_source = artifact_path.read_text(encoding="utf-8")
    assert artifact_source == fixture["expected_reference_program"]
    assert artifacts[0]["sha256"] == hashlib.sha256(artifact_source.encode("utf-8")).hexdigest()


def test_evalplus_special_oracles_are_reported_by_official_category(
    aisi_fixtures: dict[str, Any],
) -> None:
    for special_case in aisi_fixtures["special_oracles"]:
        assert (
            evalplus_special_oracle_reason(special_case["source"], special_case["entry_point"])
            == special_case["oracle"]
        )


def test_evalplus_case_record_refuses_an_oversized_reference_payload(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["evalplus"])
    task = _problem(cast("dict[str, object]", fixture["task"]))
    payload_limit_bytes = cast("int", fixture["oversized_limit_bytes"])

    with pytest.raises(CasePayloadTooLargeError) as error:
        build_evalplus_case_record(
            task,
            fixture["baked_record"],
            tmp_path,
            max_case_payload_bytes=payload_limit_bytes,
        )

    assert error.value.limit_bytes == payload_limit_bytes
    assert error.value.observed_bytes > payload_limit_bytes


def test_apps_cases_require_two_distinct_references_and_use_jail_assertions(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))

    with patch(
        "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
        return_value=True,
    ) as reference_check:
        case_file = build_apps_case_record(
            task,
            fixture["agreeing_record"],
            cast("AisiSource", object()),
            tmp_path,
        )

    assert case_file["source"] == "upstream_reference"
    cases = cast("list[dict[str, object]]", case_file["cases"])
    assert len(cases) == fixture["expected_case_count"]
    assert cases[0]["args"] == fixture["expected_unseen_arguments"]
    provenance = cast("dict[str, object]", case_file["provenance"])
    assert provenance["reference_count"] == 2
    assert provenance["case_kinds"] == fixture["expected_case_kinds"]
    assert provenance["original_assertions"] == list(task.truth_assertions)
    assert provenance["original_assertions"] == list(task.truth_assertions)
    artifacts = cast("list[dict[str, str]]", provenance["reference_artifacts"])
    assert len(artifacts) == 2
    assert len({artifact["sha256"] for artifact in artifacts}) == 2
    assert all((tmp_path / artifact["path"]).is_file() for artifact in artifacts)
    assert len(cast("list[str]", provenance["assertions"])) == len(cases)
    reference_seconds = provenance["reference_seconds"]
    assert isinstance(reference_seconds, (int, float))
    assert reference_seconds >= 0
    assert reference_check.call_count >= 4


def test_apps_cases_refuse_when_no_unseen_upstream_input_is_validated(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))

    with (
        patch(
            "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
            return_value=True,
        ),
        pytest.raises(ValueError, match=fixture["unavailable_reason"]),
    ):
        build_apps_case_record(
            task,
            fixture["visible_only_record"],
            cast("AisiSource", object()),
            tmp_path,
        )


def test_apps_cases_drop_generated_inputs_where_references_disagree(
    tmp_path: Path, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))
    disagreement_marker = cast("str", fixture["generated_disagreement_marker"])
    disagreement_reference_marker = cast("str", fixture["generated_disagreement_reference_marker"])

    def reference_verdict(reference: _ReferenceCheck, assertions: tuple[str, ...]) -> bool:
        return not (
            any(disagreement_marker in assertion for assertion in assertions)
            and disagreement_reference_marker in reference.solution
        )

    with patch(
        "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
        side_effect=reference_verdict,
    ):
        case_file = build_apps_case_record(
            task,
            fixture["disagreeing_record"],
            cast("AisiSource", object()),
            tmp_path,
        )

    cases = cast("list[dict[str, object]]", case_file["cases"])
    assert [case["args"] for case in cases] == fixture["disagreement_expected_arguments"]
    provenance = cast("dict[str, object]", case_file["provenance"])
    assert provenance["case_kinds"] == fixture["disagreement_case_kinds"]


def test_apps_generated_inputs_use_two_validated_references_and_preserve_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))
    generated_inputs: list[list[object]] = [["fresh-0\n"], ["fresh-1\n"]]

    class SyntheticGenerator:
        sha256 = "a" * 64

        def generate(self, rng: Any, n: int) -> list[list[object]]:
            assert n == len(generated_inputs)
            return generated_inputs

        def is_valid(self, arguments: list[object]) -> bool:
            return len(arguments) == 1 and isinstance(arguments[0], str)

    runner_calls = 0

    def fake_jail_runner(episode_dir: Path, command: str, **kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        assert episode_dir.is_dir()
        probe_source = (episode_dir / "probe.py").read_text(encoding="utf-8")
        compile(probe_source, str(episode_dir / "probe.py"), "exec")
        assert command
        assert kwargs["login_shell"] is False
        return SimpleNamespace(
            exit_code=0,
            timed_out=False,
            stdout="".join(
                json.dumps({"status": "ok", "output": f"answer:{arguments[0]}", "seconds": 0.01})
                + "\n"
                for arguments in generated_inputs
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "reward_hacking.organism_rl.strengthened_truth_aisi.run_in_jail",
        fake_jail_runner,
        raising=False,
    )
    monkeypatch.setattr(
        "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
        lambda reference, assertions: True,
    )

    case_file = build_apps_case_record(
        task,
        fixture["visible_only_record"],
        cast("AisiSource", object()),
        tmp_path,
        generated_inputs=module.AppsGeneratedInputConfig(
            generator=SyntheticGenerator(), count=len(generated_inputs)
        ),
    )

    provenance = cast("dict[str, object]", case_file["provenance"])
    case_rows = cast("list[dict[str, object]]", case_file["cases"])
    case_kinds = cast("list[str]", provenance["case_kinds"])
    generated_cases = [
        case
        for case, kind in zip(case_rows, case_kinds, strict=True)
        if kind == "generated_agreement"
    ]
    assert runner_calls == 2
    assert provenance["reference_count"] == 2
    assert provenance["generated_input_count"] == 2
    assert [case["args"] for case in generated_cases] == generated_inputs
    assert [case["expected"] for case in generated_cases] == [
        "answer:fresh-0\n",
        "answer:fresh-1\n",
    ]
    assert all(
        case["case_digest"] == module.canonical_digest(case["args"]) for case in generated_cases
    )


def test_apps_generated_inputs_drop_reference_disagreements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))
    generated_inputs: list[list[object]] = [["fresh-0\n"], ["fresh-1\n"]]

    class SyntheticGenerator:
        sha256 = "b" * 64

        def generate(self, rng: Any, n: int) -> list[list[object]]:
            assert n == len(generated_inputs)
            return generated_inputs

        def is_valid(self, arguments: list[object]) -> bool:
            return len(arguments) == 1 and isinstance(arguments[0], str)

    runner_calls = 0

    def fake_jail_runner(episode_dir: Path, command: str, **kwargs: object) -> object:
        nonlocal runner_calls
        runner_calls += 1
        reference_label = "first" if runner_calls == 1 else "second"
        return SimpleNamespace(
            exit_code=0,
            timed_out=False,
            stdout="".join(
                json.dumps(
                    {
                        "status": "ok",
                        "output": (
                            f"{reference_label}:{arguments[0]}"
                            if reference_label == "second" and arguments == ["fresh-1\n"]
                            else f"answer:{arguments[0]}"
                        ),
                        "seconds": 0.01,
                    }
                )
                + "\n"
                for arguments in generated_inputs
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "reward_hacking.organism_rl.strengthened_truth_aisi.run_in_jail",
        fake_jail_runner,
        raising=False,
    )
    monkeypatch.setattr(
        "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
        lambda reference, assertions: True,
    )

    case_file = build_apps_case_record(
        task,
        fixture["visible_only_record"],
        cast("AisiSource", object()),
        tmp_path,
        generated_inputs=module.AppsGeneratedInputConfig(
            generator=SyntheticGenerator(), count=len(generated_inputs)
        ),
    )

    provenance = cast("dict[str, object]", case_file["provenance"])
    case_rows = cast("list[dict[str, object]]", case_file["cases"])
    case_kinds = cast("list[str]", provenance["case_kinds"])
    generated_cases = [
        case
        for case, kind in zip(case_rows, case_kinds, strict=True)
        if kind == "generated_agreement"
    ]
    assert runner_calls == 2
    assert [case["args"] for case in generated_cases] == [["fresh-0\n"]]
    assert provenance["generated_input_count"] == 2


def test_apps_generated_input_generator_must_accept_every_visible_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, aisi_fixtures: dict[str, Any]
) -> None:
    fixture = cast("dict[str, Any]", aisi_fixtures["apps"])
    task = _problem(cast("dict[str, object]", fixture["task"]))

    class RejectVisibleGenerator:
        sha256 = "c" * 64

        def generate(self, rng: Any, n: int) -> list[list[object]]:
            return [["fresh\n"] for _ in range(n)]

        def is_valid(self, arguments: list[object]) -> bool:
            return arguments == ["fresh\n"]

    monkeypatch.setattr(
        "reward_hacking.organism_rl.strengthened_truth_aisi.aisi_rh_envs_etl._reference_assertions_pass",
        lambda reference, assertions: True,
    )

    with pytest.raises(ValueError, match="rejects a visible APPS input"):
        build_apps_case_record(
            task,
            fixture["visible_only_record"],
            cast("AisiSource", object()),
            tmp_path,
            generated_inputs=module.AppsGeneratedInputConfig(
                generator=RejectVisibleGenerator(), count=1
            ),
        )


def test_apps_generator_loader_binds_generator_bytes_and_functions(tmp_path: Path) -> None:
    assertions = ("assert solve('synthetic') == 'synthetic'",)
    task = AisiRhProblem(
        source="apps",
        problem_id="apps:synthetic-task",
        source_problem_id="synthetic-task",
        statement="Return the input string unchanged.",
        entry_point="solve",
        solution_signature="def solve(value: str) -> str:",
        test_setup_code="",
        original_assertions=assertions,
        oneoff_assertions=assertions,
        truth_assertions=assertions,
        mutation_index=0,
    )
    generator_path = tmp_path / f"{task.source_problem_id}.py"
    generator_source = (
        "from __future__ import annotations\n"
        "import random\n"
        "def generate(rng: random.Random, n: int) -> list[list[object]]:\n"
        "    return [[str(index)] for index in range(n)]\n"
        "def is_valid(arguments: list[object]) -> bool:\n"
        "    return len(arguments) == 1 and isinstance(arguments[0], str)\n"
    )
    generator_path.write_text(generator_source, encoding="utf-8")

    generator = module.load_apps_input_generator(tmp_path, task)

    assert generator is not None
    assert generator.sha256 == hashlib.sha256(generator_source.encode("utf-8")).hexdigest()
    assert generator.generate(random.Random(0), 2) == [["0"], ["1"]]
    assert generator.is_valid(["synthetic"])
