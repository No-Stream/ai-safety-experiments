"""Crosswalk acceptance; all problem and assertion material lives in ignored fixtures."""

from __future__ import annotations

import base64
import hashlib
import json
import pickle
import shutil
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from reward_hacking.organism_rl.strengthened_truth import (
    CrosswalkPolicy,
    build_crosswalk,
    decode_private_tests,
    validate_crosswalk_digest,
    write_case_files,
)

FIXTURES = Path(__file__).parents[2] / "docs/scratch/w16-strengthened-truth/fixtures.json"


@pytest.fixture
def data() -> dict[str, Any]:
    if not FIXTURES.is_file():
        pytest.skip("requires ignored W16 strengthened-truth crosswalk fixtures")
    return json.loads(FIXTURES.read_text())


def test_unique_match_agrees_and_preserves_tuple_expected(data: dict[str, Any]) -> None:
    result = build_crosswalk([data["original"]], [data["candidate"]])
    match = result["matches"][0]
    assert match["source"] == "lcb_private"
    assert match["shared_inputs"] == 3
    assert match["novel_private_inputs"] == 1
    assert data["expected_assertion"] in match["extra_assertions"]
    assert match["cases"] == [
        {
            "args": [[7, 8]],
            "expected": [15],
            "case_digest": hashlib.sha256(b"[[7,8]]").hexdigest(),
        }
    ]
    assert match["provenance"]["expected_representation"] == "tuple"
    assert match["provenance"]["expected_repr"] == ["(15,)"]
    assert match["provenance"]["assertions"] == ["assert aggregate(*([7, 8],)) == (15,)"]
    assert match["provenance"]["reference_seconds"] == 0.0
    validate_crosswalk_digest(result)


def test_test_cases_decide_match_when_text_does_not(data: dict[str, Any]) -> None:
    candidate = deepcopy(data["candidate"])
    candidate["question_content"] = "Unrelated description with no shared words."

    result = build_crosswalk([data["original"]], [candidate])

    assert [match["question_id"] for match in result["matches"]] == ["source"]


def test_searches_all_candidates_before_accepting_unique_case_match(
    data: dict[str, Any],
) -> None:
    lexical_decoy = deepcopy(data["candidate"])
    lexical_decoy["question_id"] = "lexical-decoy"
    lexical_decoy["private_test_cases"] = json.dumps(
        json.loads(lexical_decoy["private_test_cases"])[2:]
    )
    valid_candidate = deepcopy(data["candidate"])
    valid_candidate["question_id"] = "case-match"
    valid_candidate["question_content"] = "Unrelated description."

    result = build_crosswalk([data["original"]], [lexical_decoy, valid_candidate])

    assert [match["question_id"] for match in result["matches"]] == ["case-match"]


def test_text_similarity_can_break_only_a_case_match_tie(data: dict[str, Any]) -> None:
    first = deepcopy(data["candidate"])
    first["question_content"] = "Unrelated description."
    second = deepcopy(data["candidate"])
    second["question_id"] = "text-supported-case-match"

    result = build_crosswalk([data["original"]], [first, second])

    assert [match["question_id"] for match in result["matches"]] == ["text-supported-case-match"]


def test_direct_is_none_assertions_are_extractable_but_negative_assertions_refuse(
    data: dict[str, Any],
) -> None:
    original = deepcopy(data["original"])
    original["original_test"] += (
        "    assert candidate([9, 10]) is None\n    assert candidate([11, 12]) is not None\n"
    )
    candidate = deepcopy(data["candidate"])
    private = json.loads(candidate["private_test_cases"])
    private.extend(
        [
            {
                "input": "[9, 10]",
                "output": "null",
                "testtype": "functional",
            },
            {
                "input": "[13, 14]",
                "output": "[27]",
                "testtype": "functional",
            },
        ]
    )
    candidate["private_test_cases"] = json.dumps(private)

    result = build_crosswalk([original], [candidate])

    assert result["parse_summary"]["additional_visible_assertions_parsed"] == 1
    assert result["unmatched"][0]["reason"] == "unsupported_visible_assertions"


def test_direct_is_none_assertions_can_supply_shared_case_pairs(data: dict[str, Any]) -> None:
    original = deepcopy(data["original"])
    original["original_test"] = (
        "def check(candidate):\n"
        "    assert candidate([1, 2]) is None\n"
        "    assert candidate([3, 4]) is None\n"
        "    assert candidate([5, 6]) is None\n"
    )
    candidate = deepcopy(data["candidate"])
    candidate["private_test_cases"] = json.dumps(
        [
            {
                "input": json.dumps(values),
                "output": "null",
                "testtype": "functional",
            }
            for values in ([1, 2], [3, 4], [5, 6], [7, 8])
        ]
    )

    result = build_crosswalk([original], [candidate])

    assert result["matches"][0]["shared_pairs"] == 3
    assert result["matches"][0]["cases"][0]["expected"] is None
    assert result["parse_summary"]["additional_visible_assertions_parsed"] == 3


def test_case_files_use_the_shared_ignored_interchange_format(
    data: dict[str, Any], tmp_path: Path
) -> None:
    repository_root = Path(__file__).parents[2]
    cases_directory = (
        repository_root / "docs/scratch/w16-strengthened-truth" / f"pytest-cases-{tmp_path.name}"
    )
    result = build_crosswalk([data["original"]], [data["candidate"]])

    case_file: dict[str, Any] = {}
    try:
        write_case_files(result, cases_directory, repository_root=repository_root)
        case_file = json.loads((cases_directory / "fixture.json").read_text())
    finally:
        if cases_directory.exists():
            shutil.rmtree(cases_directory)

    assert set(case_file) == {"task_id", "source", "cases", "provenance"}
    assert case_file["source"] == "lcb_private"
    assert case_file["cases"][0]["case_digest"] == hashlib.sha256(b"[[7,8]]").hexdigest()
    assert case_file["provenance"]["expected_representation"] == "tuple"
    assert case_file["provenance"]["expected_repr"] == ["(15,)"]
    assert len(case_file["provenance"]["assertions"]) == 1


def test_reference_sabotage_refuses_match(data: dict[str, Any]) -> None:
    candidate = deepcopy(data["candidate"])
    tests = json.loads(candidate["private_test_cases"])
    tests[0]["output"] = data["wrong_output"]
    candidate["private_test_cases"] = json.dumps(tests)
    result = build_crosswalk([data["original"]], [candidate])
    assert not result["matches"]
    assert result["unmatched"][0]["reason"] == "shared_output_disagreement"
    assert result["stop_for_reference_authoring"] is True


def test_equal_text_candidates_are_ambiguous(data: dict[str, Any]) -> None:
    alternative = deepcopy(data["candidate"])
    alternative["question_id"] += "-other"
    result = build_crosswalk([data["original"]], [data["candidate"], alternative])
    assert result["unmatched"][0]["reason"] == "ambiguous_case_matches"


def test_insufficient_shared_inputs_never_accepts_text_alone(data: dict[str, Any]) -> None:
    candidate = deepcopy(data["candidate"])
    candidate["private_test_cases"] = json.dumps(json.loads(candidate["private_test_cases"])[2:])
    result = build_crosswalk([data["original"]], [candidate])
    assert result["unmatched"][0]["reason"] == "insufficient_shared_inputs"


def test_no_novel_cases_is_not_strengthened_truth(data: dict[str, Any]) -> None:
    candidate = deepcopy(data["candidate"])
    candidate["private_test_cases"] = json.dumps(json.loads(candidate["private_test_cases"])[:3])
    result = build_crosswalk([data["original"]], [candidate])
    assert result["unmatched"][0]["reason"] == "no_novel_private_inputs"


def test_stale_digest_refused(data: dict[str, Any]) -> None:
    result = build_crosswalk([data["original"]], [data["candidate"]])
    result["matches"][0]["extra_assertions"] += "\n"
    with pytest.raises(ValueError, match="digest"):
        validate_crosswalk_digest(result)


def test_private_payload_restricted_decode(data: dict[str, Any]) -> None:
    tests = json.loads(data["candidate"]["private_test_cases"])
    payload = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()
    assert decode_private_tests(payload) == tests
    forbidden = base64.b64encode(zlib.compress(pickle.dumps(Path))).decode()
    with pytest.raises(pickle.UnpicklingError):
        decode_private_tests(forbidden)


def test_invalid_policy_refused() -> None:
    with pytest.raises(ValueError, match="three distinct"):
        CrosswalkPolicy(minimum_shared_inputs=0)


def test_old_valid_digest_refused_against_current_reference_set(data: dict[str, Any]) -> None:
    old = build_crosswalk([data["original"]], [data["candidate"]])
    changed = deepcopy(data["candidate"])
    cases = json.loads(changed["private_test_cases"])
    cases[-1]["output"] = data["wrong_output"]
    changed["private_test_cases"] = json.dumps(cases)
    current = build_crosswalk([data["original"]], [changed])
    with pytest.raises(ValueError, match="digest"):
        validate_crosswalk_digest(old, expected_digest=current["reference_set_digest"])


def test_tuple_inputs_do_not_match_list_inputs(data: dict[str, Any]) -> None:
    result = build_crosswalk([data["tuple_input_original"]], [data["candidate"]])
    assert result["unmatched"][0]["reason"] == "insufficient_shared_inputs"


def test_partial_visible_assertion_parsing_cannot_certify_mapping(data: dict[str, Any]) -> None:
    result = build_crosswalk([data["unsupported_original"]], [data["candidate"]])
    assert not result["matches"]
    assert result["unmatched"][0]["reason"] == "unsupported_visible_assertions"
