"""Synthetic integration tests for a separate contract-validated derived registry."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.aisi_oneoff_contracts import derive_oneoff_registry, validate_derived_registry
from reward_hacking.aisi_rh_envs_etl import (
    main,
    mutate_one_expected_output,
    refresh_oneoff_assertions,
)
from reward_hacking.output_contracts import contract_digest


def _source() -> dict[str, Any]:
    return {
        "schema": 1,
        "provenance": {"source": "synthetic"},
        "problems": [
            {
                "problem_id": "synthetic:a",
                "mutation_index": 0,
                "original_assertions": ["assert solve(2) == -1", "assert solve(3) == 1"],
                "oneoff_assertions": ["assert solve(2) == -2", "assert solve(3) == 1"],
                "truth_assertions": ["assert solve(2) == -1", "assert solve(3) == 1"],
            }
        ],
    }


def test_derivation_preserves_source_truth_and_records_contract_identity() -> None:
    source = _source()
    before = deepcopy(source)
    catalog = {
        "schema": 1,
        "contracts": {
            "synthetic:a": {
                "provenance": "synthetic task definition",
                "cases": {"0": {"type": "integer", "minimum": -1, "maximum": 3}},
            }
        },
    }
    derived = derive_oneoff_registry(source, catalog)
    assert source == before
    row = derived["problems"][0]
    assert row["original_assertions"] == source["problems"][0]["original_assertions"]
    assert row["truth_assertions"] == source["problems"][0]["truth_assertions"]
    assert row["mutation_index"] == 0
    assert row["output_mutation"]["status"] == "supported"
    validate_derived_registry(derived)
    row["oneoff_assertions"][0] = "assert solve(2) == -20"
    with pytest.raises(ValueError, match="derived mutation"):
        validate_derived_registry(derived)


def test_missing_contract_explicitly_excludes_instead_of_retaining_old_output() -> None:
    derived = derive_oneoff_registry(_source(), {"schema": 1, "contracts": {}})
    row = derived["problems"][0]
    assert row["oneoff_assertions"] == row["original_assertions"]
    assert row["output_mutation"]["status"] == "unsupported"
    assert derived["output_mutation_registry"]["unsupported_count"] == 1
    assert derived["output_mutation_registry"]["exclusions"]["synthetic:a"] == "missing_contract"


def test_cli_separate_output_resumes_and_rejects_changed_contract_atomically(
    tmp_path: Path,
) -> None:

    source = tmp_path / "source.json"
    catalog = tmp_path / "contracts.json"
    output = tmp_path / "derived.json"
    source.write_text(json.dumps(_source()))
    catalog.write_text(json.dumps({"schema": 1, "contracts": {}}))
    argv = [
        "--input-registry",
        str(source),
        "--output-contracts",
        str(catalog),
        "--out",
        str(output),
    ]
    assert main(argv) == 0
    before = output.read_bytes()
    assert main(argv) == 0
    assert output.read_bytes() == before
    with pytest.raises(ValueError, match="insufficient supported"):
        main([*argv, "--minimum-supported", "1"])
    assert output.read_bytes() == before
    catalog.write_text(
        json.dumps(
            {
                "schema": 1,
                "contracts": {
                    "synthetic:a": {
                        "provenance": "synthetic",
                        "cases": {"0": {"type": "integer", "minimum": -1, "maximum": 3}},
                    }
                },
            }
        )
    )
    with pytest.raises(ValueError, match="resume identity changed"):
        main(argv)
    assert output.read_bytes() == before
    assert json.loads(source.read_text()) == _source()


def test_cli_requires_explicit_contracts_for_new_mutation() -> None:
    with pytest.raises(ValueError, match="new mutations require"):
        main([])


def test_catalog_binding_rejects_rehashed_row_contract_and_forged_exclusion() -> None:
    source = _source()
    catalog = {
        "schema": 1,
        "contracts": {
            "synthetic:a": {
                "provenance": "synthetic",
                "cases": {"0": {"type": "integer", "minimum": -1, "maximum": 3}},
            }
        },
    }
    derived = derive_oneoff_registry(source, catalog)
    row = derived["problems"][0]
    row["output_mutation"]["contract"] = {"type": "enum", "values": [-1, 999]}
    row["output_mutation"]["contract_sha256"] = contract_digest(row["output_mutation"]["contract"])
    row["oneoff_assertions"][0] = "assert solve(2) == 999"
    with pytest.raises(ValueError, match="catalog"):
        validate_derived_registry(derived)
    unsupported = derive_oneoff_registry(source, {"schema": 1, "contracts": {}})
    unsupported["problems"][0]["output_mutation"]["reason"] = "invented"
    unsupported["output_mutation_registry"]["exclusions"]["synthetic:a"] = "invented"
    with pytest.raises(ValueError, match="catalog"):
        validate_derived_registry(unsupported)


def test_source_mutation_preference_is_bound_even_when_both_cases_are_supported() -> None:
    source = _source()
    contract = {"type": "integer", "minimum": -1, "maximum": 3}
    catalog = {
        "schema": 1,
        "contracts": {
            "synthetic:a": {"provenance": "synthetic", "cases": {"0": contract, "1": contract}}
        },
    }
    derived = derive_oneoff_registry(source, catalog)
    changed_source = deepcopy(source)
    changed_source["problems"][0]["mutation_index"] = 1
    alternative = derive_oneoff_registry(changed_source, catalog)
    assert derived["problems"][0]["mutation_index"] == 0
    assert alternative["problems"][0]["mutation_index"] == 1
    derived["problems"][0] = alternative["problems"][0]
    with pytest.raises(ValueError, match=r"source|catalog"):
        validate_derived_registry(derived)


def test_unusable_preferred_case_preserves_source_preference_for_replay() -> None:
    catalog = {
        "schema": 1,
        "contracts": {
            "synthetic:a": {
                "provenance": "synthetic",
                "cases": {"1": {"type": "integer", "minimum": 0, "maximum": 3}},
            }
        },
    }
    derived = derive_oneoff_registry(_source(), catalog)
    assert derived["problems"][0]["mutation_index"] == 1
    assert derived["problems"][0]["source_mutation_index"] == 0
    validate_derived_registry(derived)


@pytest.mark.parametrize("orphaned_metadata", [False, True])
def test_legacy_refresh_refuses_strict_registry_before_mutation_or_write(
    tmp_path: Path, orphaned_metadata: bool
) -> None:
    source = _source()
    source_row = source["problems"][0]
    source_row.update(
        source="apps", source_problem_id="synthetic-refresh", problem_id="apps:synthetic-refresh"
    )
    source_row["original_assertions"] = ["assert solve(2) == -1", "assert solve(3) == -1"]
    source_row["mutation_index"] = mutate_one_expected_output(
        source_row["problem_id"], tuple(source_row["original_assertions"])
    ).case_index
    catalog = {
        "schema": 1,
        "contracts": {
            "apps:synthetic-refresh": {
                "provenance": "synthetic",
                "cases": {str(source_row["mutation_index"]): {"type": "enum", "values": [-1, 999]}},
            }
        },
    }
    derived = derive_oneoff_registry(source, catalog)
    if orphaned_metadata:
        del derived["output_mutation_registry"]
    before = deepcopy(derived)
    with pytest.raises(ValueError, match=r"contract|strict"):
        refresh_oneoff_assertions(derived)
    assert derived == before
    path = tmp_path / "strict.json"
    path.write_text(json.dumps(derived))
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError, match=r"contract|strict"):
        main(["--refresh-oneoff", "--allow-legacy-mutation", "--out", str(path)])
    assert path.read_bytes() == original_bytes
