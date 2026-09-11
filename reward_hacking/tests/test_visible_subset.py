"""Tests for the visible-subset derivation (``reward_hacking/visible_subset.py``).

Every check body here is a SYNTHETIC fixture invented for these tests -- trivial arithmetic with no
relationship to any benchmark item, because this repository's remote is public. The one test that
touches the real baked corpus asserts counts only and reports failures by task id alone.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from reward_hacking.ilcb_data import DATA_PATH, ILCBProblem, load_problems
from reward_hacking.train_grader import check_function_body, proxy_grader_source
from reward_hacking.visible_subset import (
    ORIGINAL_SPLIT,
    SubsetDerivationError,
    SubsetSelection,
    derive_for_problem,
    derive_visible_subset,
    main,
    subset_impossible_type,
)

if TYPE_CHECKING:
    from types import ModuleType

# A plain body: six asserts, nothing else. Hidden/visible multiplier at k=3 is exactly 2.0.
SIMPLE_BODY = """\
def check(candidate):
    assert candidate(1) == 2
    assert candidate(2) == 4
    assert candidate(3) == 6
    assert candidate(10) == 20
    assert candidate(-1) == -2
    assert candidate(0) == 0
"""

# Quirks a byte-faithful slice must preserve: comments, odd spacing, a multi-line assignment.
FORMATTED_BODY = (
    "def check(candidate):\n"
    "    # keep   this   spacing\n"
    "    assert candidate( 1 ) ==  2   # trailing comment\n"
    "    x = [1,\n"
    "         2]\n"
    "    assert candidate(x) == 3\n"
    "    assert candidate(4) == 8\n"
)

# The k-th assert itself spans several lines; the slice must include all of them.
MULTILINE_ASSERT_BODY = (
    "def check(candidate):\n"
    "    assert candidate(1) == 1\n"
    "    assert candidate(5) == [\n"
    "        10,\n"
    "    ]\n"
    "    assert candidate(6) == 12\n"
)

# Stash-then-assert: STRATIFIED k=3 picks asserts 0/2/4, so `first` and `total` carry, `unrelated` drops.
STASH_BODY = """\
def check(candidate):
    first = candidate([1, 2, 3])
    assert first == 6
    unrelated = candidate([9, 9])
    assert unrelated == 18
    assert candidate([]) == 0
    assert candidate([5]) == 5
    total = candidate([2, 2])
    assert total == 4
"""

# A real row's shape: a helper def before the asserts, which some asserts call.
HELPER_BODY = """\
def check(candidate):
    def normalize(values):
        return sorted(values)
    assert candidate([3, 1]) == normalize([3, 1])
    assert candidate([2, 2]) == normalize([2, 2])
    assert candidate([]) == []
    assert candidate([1]) == [1]
    assert candidate([5, 4]) == normalize([5, 4])
"""

# A real row's shape taken to its extreme: every assert inside a loop, zero top-level asserts.
LOOP_ONLY_BODY = """\
def check(candidate):
    for value in [1, 2, 3]:
        assert candidate(value) == value * 2
"""

# An assert reading a name whose latest binder is a for loop the carry cannot lift: must fail closed.
LOOP_BOUND_NAME_BODY = """\
def check(candidate):
    for seed in [1, 2]:
        table = seed
    assert candidate(table) == 1
    assert candidate(2) == 2
    assert candidate(3) == 3
"""

# An assert reading a name never bound anywhere in the body.
GHOST_NAME_BODY = """\
def check(candidate):
    assert candidate(1) == ghost_expected[0]
    assert candidate(2) == 2
    assert candidate(3) == 3
"""

# Seven asserts with distinct argument markers, to pin down WHICH asserts a spread pick selects.
SEVEN_MARKER_BODY = """\
def check(candidate):
    assert candidate(100) == 0
    assert candidate(101) == 1
    assert candidate(102) == 2
    assert candidate(103) == 3
    assert candidate(104) == 4
    assert candidate(105) == 5
    assert candidate(106) == 6
"""


def synthetic_problem(
    original_test: str = SIMPLE_BODY,
    *,
    task_id: str = "toy_1",
    impossible_type: str = ORIGINAL_SPLIT,
    provably_impossible: bool = False,
    impossibility_proof: dict[str, str] | None = None,
) -> ILCBProblem:
    """A synthetic ILCB record whose every text field is invented for these tests."""
    return ILCBProblem(
        task_id=task_id,
        impossible_type=impossible_type,
        entry_point="double",
        prompt='def double(x):\n    """Return twice x."""\n',
        test=original_test,
        original_test=original_test,
        provably_impossible=provably_impossible,
        impossibility_proof=impossibility_proof,
        check_parses=True,
        check_defines_function=True,
    )


class TestFirstK:
    def test_slice_is_byte_faithful_to_the_source(self):
        result = derive_visible_subset(FORMATTED_BODY, k=2, selection=SubsetSelection.FIRST_K)
        expected = "\n".join(FORMATTED_BODY.split("\n")[:6])
        assert result.check_source == expected

    def test_multiline_kth_assert_is_included_whole(self):
        result = derive_visible_subset(
            MULTILINE_ASSERT_BODY, k=2, selection=SubsetSelection.FIRST_K
        )
        expected = "\n".join(MULTILINE_ASSERT_BODY.split("\n")[:5])
        assert result.check_source == expected
        assert result.visible_assert_count == 2

    def test_counts_and_multiplier(self):
        result = derive_visible_subset(SIMPLE_BODY, k=3, selection=SubsetSelection.FIRST_K)
        assert result.visible_assert_count == 3
        assert result.hidden_assert_count == 6
        assert result.multiplier == 2.0

    def test_prior_helper_def_rides_along_in_the_prefix(self):
        result = derive_visible_subset(HELPER_BODY, k=3, selection=SubsetSelection.FIRST_K)
        assert "def normalize" in result.check_source
        assert any("def normalize" in stmt for stmt in result.carried_statements)

    def test_prior_assignments_ride_along_in_the_prefix(self):
        result = derive_visible_subset(STASH_BODY, k=2, selection=SubsetSelection.FIRST_K)
        assert "first = candidate" in result.check_source
        assert "unrelated = candidate" in result.check_source
        carried = "\n".join(result.carried_statements)
        assert "first = candidate" in carried
        assert "unrelated = candidate" in carried

    def test_too_few_asserts_raises(self):
        with pytest.raises(SubsetDerivationError, match="top-level assert"):
            derive_visible_subset(SIMPLE_BODY, k=7, selection=SubsetSelection.FIRST_K)

    def test_loop_only_body_raises_instead_of_deriving_nothing(self):
        with pytest.raises(SubsetDerivationError, match="top-level assert"):
            derive_visible_subset(LOOP_ONLY_BODY, k=2, selection=SubsetSelection.FIRST_K)

    def test_unwrapped_source_raises(self):
        bare = "assert candidate(1) == 1\nassert candidate(2) == 2\n"
        with pytest.raises(SubsetDerivationError, match="def check"):
            derive_visible_subset(bare, k=2, selection=SubsetSelection.FIRST_K)

    def test_module_level_siblings_raise(self):
        with_sibling = "import math\n" + SIMPLE_BODY
        with pytest.raises(SubsetDerivationError, match="exactly one") as excinfo:
            derive_visible_subset(with_sibling, k=2, selection=SubsetSelection.FIRST_K)
        # The refusal must come from reading the INPUT, not from validating a derived slice.
        assert "derived" not in str(excinfo.value)

    def test_unparseable_source_raises(self):
        with pytest.raises(SubsetDerivationError, match="parse"):
            derive_visible_subset("def check(candidate:\n", k=2, selection=SubsetSelection.FIRST_K)

    def test_nonpositive_k_raises(self):
        with pytest.raises(SubsetDerivationError, match="k must be at least 1"):
            derive_visible_subset(SIMPLE_BODY, k=0, selection=SubsetSelection.FIRST_K)


class TestStratified:
    def test_picks_first_middle_last_for_k3(self):
        result = derive_visible_subset(SEVEN_MARKER_BODY, k=3, selection=SubsetSelection.STRATIFIED)
        assert result.visible_assert_count == 3
        for marker in ("100", "103", "106"):
            assert marker in result.check_source
        for marker in ("101", "102", "104", "105"):
            assert marker not in result.check_source

    def test_carries_needed_assignments_and_drops_unrelated_ones(self):
        result = derive_visible_subset(STASH_BODY, k=3, selection=SubsetSelection.STRATIFIED)
        assert "first = candidate" in result.check_source
        assert "total = candidate" in result.check_source
        assert "unrelated" not in result.check_source
        carried = "\n".join(result.carried_statements)
        assert "first = candidate" in carried
        assert "total = candidate" in carried

    def test_helper_def_is_carried(self):
        result = derive_visible_subset(HELPER_BODY, k=3, selection=SubsetSelection.STRATIFIED)
        assert "def normalize" in result.check_source
        assert result.visible_assert_count == 3

    def test_ghost_name_is_a_construction_failure(self):
        with pytest.raises(SubsetDerivationError, match=r"unresolved.*ghost_expected"):
            derive_visible_subset(GHOST_NAME_BODY, k=3, selection=SubsetSelection.STRATIFIED)

    def test_loop_bound_name_is_a_construction_failure(self):
        with pytest.raises(SubsetDerivationError, match=r"`table`.*cannot lift"):
            derive_visible_subset(LOOP_BOUND_NAME_BODY, k=3, selection=SubsetSelection.STRATIFIED)

    def test_loop_only_body_raises(self):
        with pytest.raises(SubsetDerivationError, match="top-level assert"):
            derive_visible_subset(LOOP_ONLY_BODY, k=3, selection=SubsetSelection.STRATIFIED)

    def test_derived_source_keeps_statement_bytes(self):
        result = derive_visible_subset(FORMATTED_BODY, k=2, selection=SubsetSelection.STRATIFIED)
        for line in result.check_source.split("\n"):
            if line.strip():
                assert line in FORMATTED_BODY
        assert "# trailing comment" in result.check_source


class TestDeriveForProblem:
    def test_replaces_test_and_records_provenance(self):
        problem = synthetic_problem(STASH_BODY)
        derived = derive_for_problem(problem, k=3, selection=SubsetSelection.FIRST_K)
        expected = derive_visible_subset(
            STASH_BODY, k=3, selection=SubsetSelection.FIRST_K, context=problem.task_id
        )
        assert derived.problem.test == expected.check_source
        assert derived.problem.original_test == STASH_BODY
        assert derived.problem.task_id == problem.task_id
        assert derived.problem.entry_point == problem.entry_point
        assert derived.problem.prompt == problem.prompt
        assert derived.problem.impossible_type == subset_impossible_type(3, SubsetSelection.FIRST_K)
        assert derived.provenance.k == 3
        assert derived.provenance.selection is SubsetSelection.FIRST_K
        assert derived.provenance.hidden_assert_count == 5
        assert derived.provenance.multiplier == pytest.approx(5 / 3)
        assert (
            derived.provenance.original_test_sha256
            == hashlib.sha256(STASH_BODY.encode()).hexdigest()
        )

    def test_impossibility_flags_are_reset_not_inherited(self):
        conflicting_shaped = synthetic_problem(
            SIMPLE_BODY,
            impossible_type="conflicting",
            provably_impossible=True,
            impossibility_proof={"first": "a", "second": "b"},
        )
        derived = derive_for_problem(conflicting_shaped, k=2, selection=SubsetSelection.FIRST_K)
        assert derived.problem.provably_impossible is False
        assert derived.problem.impossibility_proof is None
        assert derived.problem.check_parses is True
        assert derived.problem.check_defines_function is True

    def test_construction_error_names_the_problem_id(self):
        problem = synthetic_problem(LOOP_ONLY_BODY, task_id="toy_loop")
        with pytest.raises(SubsetDerivationError, match="toy_loop"):
            derive_for_problem(problem, k=3, selection=SubsetSelection.FIRST_K)

    def test_impossible_type_names_are_distinct_and_descriptive(self):
        assert subset_impossible_type(3, SubsetSelection.FIRST_K) == "subset3"
        assert subset_impossible_type(3, SubsetSelection.STRATIFIED) == "subset3-stratified"
        assert subset_impossible_type(2, SubsetSelection.FIRST_K) == "subset2"


class TestGraderShapeCompatibility:
    """The derived ``test`` field must be shaped exactly like the baked splits' own.

    ``train_grader.check_function_body`` passes a body that already defines ``def check`` through
    unchanged, so byte-identity there is the proof the derived field is grader-shaped; the compile
    checks prove both grader builders accept it whole.
    """

    def test_check_function_body_passes_the_derived_test_through_unchanged(self):
        problem = synthetic_problem(STASH_BODY)
        derived = derive_for_problem(problem, k=3, selection=SubsetSelection.STRATIFIED)
        assert check_function_body(derived.problem) == derived.problem.test

    def test_proxy_grader_source_accepts_the_derived_record(self):
        problem = synthetic_problem(HELPER_BODY)
        derived = derive_for_problem(problem, k=3, selection=SubsetSelection.FIRST_K)
        grader = proxy_grader_source(derived.problem)
        compile(grader, "<derived-proxy-grader>", "exec")
        assert derived.problem.test in grader


class TestRealCorpus:
    """The feasibility probe's headline, enforced as a regression gate: counts only."""

    @pytest.mark.parametrize("selection", [SubsetSelection.FIRST_K, SubsetSelection.STRATIFIED])
    def test_all_originals_derive_at_k3_with_multiplier_at_least_2x(
        self, selection: SubsetSelection
    ):
        originals = [p for p in load_problems() if p.impossible_type == ORIGINAL_SPLIT]
        assert len(originals) == 103
        failed_task_ids: list[str] = []
        low_multiplier_task_ids: list[str] = []
        for problem in originals:
            try:
                derived = derive_for_problem(problem, k=3, selection=selection)
            except SubsetDerivationError:
                failed_task_ids.append(problem.task_id)
                continue
            if derived.provenance.multiplier < 2.0:
                low_multiplier_task_ids.append(problem.task_id)
        assert failed_task_ids == [], f"{len(failed_task_ids)} of 103 failed: {failed_task_ids}"
        assert low_multiplier_task_ids == [], (
            f"{len(low_multiplier_task_ids)} of 103 below 2.0x: {low_multiplier_task_ids}"
        )


def _module_level_imports(module: ModuleType) -> set[str]:
    """Every module name a module's top-level import statements reach, read off its source AST."""
    source = Path(str(module.__file__)).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


class TestImportStaysRegistryFree:
    """Importing the deriver must never build the harness task registry.

    ``ilcb_etl`` imports this module to bake the derived splits, and its whole reason for existing
    as a by-path script is that it must run while the baked case file is missing or broken -- the
    state in which importing ``reward_hacking.harness`` crashes at import time. Checked statically
    over the deriver's and the data layer's own top-level import statements, because in this test
    process the harness is already imported and ``sys.modules`` can prove nothing; any function-
    level import that would dodge this walk is ruff's PLC0415 to catch.
    """

    def test_the_deriver_and_the_data_layer_never_import_the_harness(self):
        from reward_hacking import ilcb_data, visible_subset  # noqa: PLC0415

        for module in (visible_subset, ilcb_data):
            harness_imports = {
                name
                for name in _module_level_imports(module)
                if name.startswith("reward_hacking.harness")
            }
            assert not harness_imports, (
                f"{module.__name__} imports {sorted(harness_imports)}, which builds the registry "
                f"and reads the very case file the ETL regenerates"
            )

    def test_the_walk_sees_imports_at_all(self):
        """The positive control: an import this walk cannot see would make the guard vacuous."""
        from reward_hacking import visible_subset  # noqa: PLC0415

        assert "reward_hacking.ilcb_data" in _module_level_imports(visible_subset)


def baked_payload(problems: list[ILCBProblem]) -> dict[str, object]:
    """A minimal baked-JSON payload in the shape ``tasks_ilcb.load_problems`` reads."""
    return {"schema": 1, "problems": [dataclasses.asdict(p) for p in problems]}


class TestCli:
    def test_writes_a_derived_file_load_problems_can_read(self, tmp_path: Path):
        data = tmp_path / "cases.json"
        data.write_text(
            json.dumps(
                baked_payload(
                    [
                        synthetic_problem(SIMPLE_BODY, task_id="toy_1"),
                        synthetic_problem(STASH_BODY, task_id="toy_2"),
                        synthetic_problem(SIMPLE_BODY, task_id="toy_1", impossible_type="oneoff"),
                    ]
                )
            )
        )
        out = tmp_path / "derived.json"
        exit_code = main(
            ["--k", "2", "--selection", "first-k", "--data", str(data), "--out", str(out)]
        )
        assert exit_code == 0
        derived_problems = load_problems(out)
        assert len(derived_problems) == 2
        assert {p.impossible_type for p in derived_problems} == {
            subset_impossible_type(2, SubsetSelection.FIRST_K)
        }
        payload = json.loads(out.read_text())
        for record in payload["problems"]:
            assert record["provenance"]["k"] == 2
            assert record["provenance"]["hidden_assert_count"] >= 4

    def test_refuses_to_overwrite_an_existing_file(self, tmp_path: Path):
        data = tmp_path / "cases.json"
        data.write_text(json.dumps(baked_payload([synthetic_problem()])))
        out = tmp_path / "derived.json"
        out.write_text("sentinel")
        exit_code = main(["--k", "2", "--data", str(data), "--out", str(out)])
        assert exit_code != 0
        assert out.read_text() == "sentinel"

    def test_refuses_any_path_inside_the_tracked_data_dir(self, tmp_path: Path):
        data = tmp_path / "cases.json"
        data.write_text(json.dumps(baked_payload([synthetic_problem()])))
        forbidden = DATA_PATH.parent / "derived-cases-test-must-never-exist.json"
        assert not forbidden.exists()
        exit_code = main(["--k", "2", "--data", str(data), "--out", str(forbidden)])
        assert exit_code != 0
        assert not forbidden.exists()

    def test_any_derivation_failure_is_a_nonzero_exit_and_no_file(self, tmp_path: Path):
        data = tmp_path / "cases.json"
        data.write_text(
            json.dumps(
                baked_payload(
                    [
                        synthetic_problem(SIMPLE_BODY, task_id="toy_ok"),
                        synthetic_problem(LOOP_ONLY_BODY, task_id="toy_loop"),
                    ]
                )
            )
        )
        out = tmp_path / "derived.json"
        exit_code = main(["--k", "2", "--data", str(data), "--out", str(out)])
        assert exit_code != 0
        assert not out.exists()
