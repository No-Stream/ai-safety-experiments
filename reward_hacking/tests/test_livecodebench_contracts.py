"""Synthetic checks for explicit LiveCodeBench output contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pytest

import reward_hacking.organism_rl.import_livecodebench as importer
from reward_hacking.ilcb_data import load_problems
from reward_hacking.organism_rl.import_livecodebench import (
    SelectedProblem,
    build_case_payload,
    main,
)
from reward_hacking.organism_rl.import_livecodebench import (
    TestCase as LCBCase,
)

if TYPE_CHECKING:
    from pathlib import Path


def _problem() -> SelectedProblem:
    return SelectedProblem(
        task_id="synthetic-contract-task",
        question_id="synthetic",
        question_title="Synthetic importer fixture",
        question_content="Synthetic importer fixture.",
        platform="synthetic",
        contest_id="synthetic",
        contest_date="2025-01-01",
        public_cases=(LCBCase("first", "0\n"), LCBCase("second", "1\n")),
        private_cases=(LCBCase("private", "2\n"),),
        public_case_count=2,
        private_case_count=1,
    )


def _catalog(cases: dict[str, Any]) -> dict[str, Any]:
    return {_problem().task_id: {"provenance": "synthetic explicit specification", "cases": cases}}


def _payload(catalog: dict[str, Any]) -> dict[str, Any]:
    return build_case_payload((_problem(),), target_problem_count=1, output_contracts=catalog)


def test_strict_mutation_keeps_original_truth_and_uses_next_supported_case() -> None:
    payload = _payload(
        _catalog(
            {
                "0": {"type": "enum", "values": ["0\n"]},
                "1": {"type": "enum", "values": ["1\n", "2\n"]},
            }
        )
    )
    original, oneoff = payload["problems"]
    legacy_original = build_case_payload((_problem(),), target_problem_count=1)["problems"][0]
    assert original == legacy_original
    assert oneoff["original_test"] == original["original_test"]
    namespace: dict[str, Any] = {}
    exec(oneoff["test"], namespace)  # noqa: S102 - synthetic generated checker
    namespace["check"](lambda text: {"first": "0\n", "second": "2\n"}[text])
    with pytest.raises(AssertionError):
        namespace["check"](lambda text: {"first": "0\n", "second": "1\n"}[text])
    metadata = oneoff["provenance"]["output_mutation"]
    assert metadata["algorithm"] == "output-contract-v1"
    assert metadata["case_index"] == 1
    assert len(metadata["contract_sha256"]) == 64
    assert metadata["contract_provenance"] == "synthetic explicit specification"
    assert payload["mutation_summary"]["supported"] == 1


@pytest.mark.parametrize(
    "catalog", [{}, _catalog({}), _catalog({"0": {"type": "enum", "values": ["0\n"]}})]
)
def test_missing_or_singleton_contract_is_explicitly_unsupported_without_legacy_fallback(
    catalog: dict[str, Any],
) -> None:
    payload = _payload(catalog)
    assert [row["impossible_type"] for row in payload["problems"]] == ["original"]
    assert payload["mutation_summary"]["unsupported"] == 1
    assert payload["mutation_summary"]["exclusions"][0]["task_id"] == _problem().task_id
    assert payload["mutation_summary"]["exclusions"][0]["reason"]


def test_contract_provenance_is_required() -> None:
    payload = _payload(
        {_problem().task_id: {"cases": {"0": {"type": "enum", "values": ["0\n", "1\n"]}}}}
    )
    assert payload["mutation_summary"]["unsupported"] == 1
    assert len(payload["problems"]) == 1


def test_new_collection_cli_requires_explicit_contract_or_legacy_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbid_dataset_access() -> None:
        raise AssertionError("CLI must reject before loading dataset")

    monkeypatch.setattr(importer, "load_recent_rows", forbid_dataset_access)
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_strict_write_replaces_stale_oneoffs_and_preserves_splits_and_hidden_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    problems = tuple(replace(_problem(), task_id=f"synthetic-{index}") for index in range(30))
    monkeypatch.setattr(importer, "DEFAULT_DATA_DIRECTORY", tmp_path)
    paths = {
        "cases_path": tmp_path / "cases.json",
        "manifest_path": tmp_path / "manifest.json",
        "strengthened_cases_root": tmp_path / "strengthened",
        "timed_cases_root": tmp_path / "timed",
    }
    importer.write_outputs(
        problems,
        target_problem_count=len(problems),
        cases_path=paths["cases_path"],
        manifest_path=paths["manifest_path"],
        strengthened_cases_root=paths["strengthened_cases_root"],
        timed_cases_root=paths["timed_cases_root"],
    )
    manifest_before = paths["manifest_path"].read_bytes()
    truth_before = {
        path.relative_to(tmp_path): path.read_bytes()
        for directory in (paths["strengthened_cases_root"], paths["timed_cases_root"])
        for path in directory.rglob("*.json")
    }
    importer.write_outputs(
        problems,
        target_problem_count=len(problems),
        output_contracts={},
        cases_path=paths["cases_path"],
        manifest_path=paths["manifest_path"],
        strengthened_cases_root=paths["strengthened_cases_root"],
        timed_cases_root=paths["timed_cases_root"],
    )
    payload = json.loads(paths["cases_path"].read_text())
    assert len(payload["problems"]) == len(problems)
    assert all(row["impossible_type"] == "original" for row in payload["problems"])
    assert payload["mutation_summary"]["unsupported"] == len(problems)
    assert paths["manifest_path"].read_bytes() == manifest_before
    assert all(
        (tmp_path / path).read_bytes() == contents for path, contents in truth_before.items()
    )


def test_contract_mutation_respects_numeric_token_bounds() -> None:
    payload = _payload(
        _catalog(
            {
                "0": {
                    "type": "text_tokens",
                    "tokens": [{"type": "integer", "minimum": 0, "maximum": 1}],
                }
            }
        )
    )
    namespace: dict[str, Any] = {}
    exec(payload["problems"][1]["test"], namespace)  # noqa: S102 - synthetic generated checker
    namespace["check"](lambda _text: "1\n")
    with pytest.raises(AssertionError):
        namespace["check"](lambda _text: "-1\n")


def test_catalog_digest_binds_contract_provenance_and_exclusions() -> None:
    first = _payload({})
    second = _payload(_catalog({}))
    assert (
        first["mutation_summary"]["catalog_sha256"] != second["mutation_summary"]["catalog_sha256"]
    )
    assert len(first["mutation_summary"]["source_sha256"]) == 64
    supported = _payload(_catalog({"0": {"type": "enum", "values": ["0\n", "1\n"]}}))
    assert (
        supported["problems"][1]["provenance"]["output_mutation"]["catalog_sha256"]
        == supported["mutation_summary"]["catalog_sha256"]
    )


def test_strict_cli_uses_separate_root_and_rejects_changed_source_or_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(importer, "DEFAULT_DATA_DIRECTORY", data)
    protected = {
        "DEFAULT_CASES_PATH": data / "legacy-cases.json",
        "DEFAULT_SPLIT_MANIFEST_PATH": data / "legacy-split.json",
        "DEFAULT_STRENGTHENED_CASES_ROOT": data / "legacy-truth",
        "DEFAULT_TIMED_CASES_ROOT": data / "legacy-timed",
    }
    for name, path in protected.items():
        monkeypatch.setattr(importer, name, path)
    protected["DEFAULT_CASES_PATH"].write_text("legacy-cases")
    protected["DEFAULT_SPLIT_MANIFEST_PATH"].write_text("legacy-splits")
    problems = tuple(replace(_problem(), task_id=f"synthetic-{index}") for index in range(30))
    monkeypatch.setattr(importer, "load_recent_rows", lambda: ())
    monkeypatch.setattr(importer, "select_supported_rows", lambda _rows, **_kwargs: problems)
    catalog = data / "catalog.json"
    catalog.write_text(json.dumps({"schema": 1, "contracts": {}}))
    with pytest.raises(SystemExit):
        main(["--output-contracts", str(catalog)])
    with pytest.raises(ValueError, match="distinct"):
        main(["--output-contracts", str(catalog), "--output-root", str(data)])
    output = data / "strict"
    args = [
        "--output-contracts",
        str(catalog),
        "--output-root",
        str(output),
        "--target-problems",
        "30",
    ]
    assert main(args) == 0
    cases = output / "lcb_recent_cases.json"
    assert cases.is_file()
    assert protected["DEFAULT_CASES_PATH"].read_text() == "legacy-cases"
    assert protected["DEFAULT_SPLIT_MANIFEST_PATH"].read_text() == "legacy-splits"
    before = cases.read_bytes()
    changed = (replace(problems[0], question_content="changed synthetic source"), *problems[1:])
    monkeypatch.setattr(importer, "select_supported_rows", lambda _rows, **_kwargs: changed)
    with pytest.raises(ValueError, match="contract payload"):
        main(args)
    assert cases.read_bytes() == before
    monkeypatch.setattr(importer, "select_supported_rows", lambda _rows, **_kwargs: problems)
    catalog.write_text(json.dumps({"schema": 1, "contracts": _catalog({})}))
    with pytest.raises(ValueError, match="contract payload"):
        main(args)
    assert cases.read_bytes() == before


def test_contract_payload_validation_recomputes_mutated_checker() -> None:
    catalog = _catalog({"0": {"type": "enum", "values": ["0\n", "1\n"]}})
    payload = _payload(catalog)
    payload["problems"][1]["test"] = payload["problems"][0]["test"]
    with pytest.raises(ValueError, match="contract payload"):
        importer.validate_contract_payload(payload, (_problem(),), catalog)


@pytest.mark.parametrize("tamper", ["summary", "checker", "bounds"])
def test_loader_recomputes_strict_contracts_before_exposing_rows(
    tmp_path: Path, tamper: str
) -> None:
    catalog = _catalog({"0": {"type": "enum", "values": ["0\n", "1\n"]}})
    payload = _payload(catalog)
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(payload))
    loaded = load_problems(path)
    assert len(loaded) == 2
    assert loaded[0].output_mutation_identity is None
    assert loaded[1].output_mutation_identity == {
        "algorithm": "output-contract-v1",
        "contract_sha256": payload["problems"][1]["provenance"]["output_mutation"][
            "contract_sha256"
        ],
        "case_index": 0,
        "matcher_id": "lcb_stdio",
    }
    assert "synthetic explicit specification" not in loaded[1].prompt + loaded[1].test
    if tamper == "summary":
        payload["mutation_summary"] = {"algorithm": "output-contract-v1"}
    elif tamper == "checker":
        payload["problems"][1]["test"] = payload["problems"][0]["test"]
    else:
        payload["mutation_summary"]["contracts"][_problem().task_id]["cases"]["0"]["values"] = [
            "0\n"
        ]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract payload"):
        load_problems(path)


def test_strict_root_rejects_child_symlink_into_legacy_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(importer, "DEFAULT_DATA_DIRECTORY", tmp_path)
    legacy = tmp_path / "legacy-truth"
    legacy.mkdir()
    monkeypatch.setattr(importer, "DEFAULT_STRENGTHENED_CASES_ROOT", legacy)
    root = tmp_path / "derived"
    root.mkdir()
    (root / "lcb_recent_strengthened_truth").symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="distinct"):
        importer._strict_output_root(root)


def test_loader_rejects_strict_rows_when_summary_was_removed(tmp_path: Path) -> None:
    payload = _payload(_catalog({"0": {"type": "enum", "values": ["0\n", "1\n"]}}))
    del payload["mutation_summary"]
    path = tmp_path / "downgraded.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract payload"):
        load_problems(path)
