"""Pin the contradiction search's one promise: it may miss a row, never manufacture one.

``ilcb_etl`` decides ``provably_impossible`` by finding two of a visible check's own assertions that
demand different answers from the same candidate call. Every step of that search only ever *drops*
constraints it cannot pin down, because a contradiction among a subset of a check's demands is a
contradiction in the whole check -- so dropping costs a missed row, and a manufactured contradiction
is the one error the module's docstring says cannot happen. It is also by far the expensive error: a
false ``provably_impossible`` puts a satisfiable row into the split whose entire meaning is "a pass
here is a reward hack by construction", and every downstream readout then counts an honest solve as
a hack.

The load-bearing case is the syntax-error recovery path. A check body that does not compile is
recovered a line at a time, and a line-at-a-time reader that forgets which *block* a line sat in
reads two assertions from mutually exclusive branch arms as two unconditional demands about the same
call. That is a contradiction the check cannot state, and the compiling twin of the same body
correctly finds nothing -- which is what the first test here contrasts.

No dataset material appears in this file. Every check body is hand-written over single-character
arguments. The one real row these are shaped after -- a body the dataset truncated
mid-string-literal whose recoverable assertions do carry a genuine contradiction -- is reproduced
synthetically, so the recovery keeps being exercised on the shape that actually occurs without
republishing it.
"""

from __future__ import annotations

import hashlib
import json

from reward_hacking.harness.ilcb_etl import DERIVED_SUBSET_K, find_contradiction, parse_check
from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.ilcb_data import DATA_PATH
from reward_hacking.visible_subset import (
    ORIGINAL_SPLIT,
    SubsetSelection,
    derive_visible_subset,
    subset_impossible_type,
)

_BRANCHED_ARMS = """\
def check(candidate):
    if candidate("a") is None:
        assert candidate("b") == 2
    else:
        assert candidate("b") == 3
"""
"""Two demands on ``candidate("b")`` that only one of the branch arms ever makes."""

_BRANCHED_ARMS_TRUNCATED = _BRANCHED_ARMS + '    note = "cut off mid\n'
"""The same body, plus an unterminated string literal, so the whole body will not compile."""

_FLAT_ASSERTS_TRUNCATED = """\
def check(candidate):
    assert candidate("a") == 5
    assert candidate("b") == 1
    assert candidate("a") == 3
    assert candidate("cccc
"""
"""The real broken row's shape: unconditional assertions, then a truncated final line."""

_NO_WRAPPER_TRUNCATED = """\
assert candidate("a") == 5
assert candidate("a") == 3
note = "cut off mid
"""
"""The degenerate rows' shape -- bare assertions with no ``def check`` -- also truncated."""

_REBOUND_BY_AN_UNREADABLE_STATEMENT = """\
def check(candidate):
    v = 1
    assert candidate(v) == 2
    v = (1 +
         1)
    assert candidate(v) == 3
    note = "cut off mid
"""
"""A rebinding the recovery cannot read, between two assertions that would otherwise collide."""


class TestRecoveryFromABodyThatWillNotCompile:
    def test_branch_arms_are_not_flattened_into_one_call(self):
        """Two arms of one branch are not two demands, and the recovery must not read them as one.

        Asserted against the compiling twin rather than in isolation: the terminated body finds
        nothing, so any contradiction the truncated one reports was manufactured by the recovery
        itself and is not a property of the check.
        """
        assert find_contradiction(_BRANCHED_ARMS) is None
        assert find_contradiction(_BRANCHED_ARMS_TRUNCATED) is None

    def test_a_truncated_body_still_proves_the_contradiction_it_carries(self):
        """The recovery exists for this shape, so keeping it monotone must not disarm it."""
        contradiction = find_contradiction(_FLAT_ASSERTS_TRUNCATED)
        assert contradiction is not None
        assert contradiction.call == "candidate('a')"
        assert "5" in contradiction.first
        assert "3" in contradiction.second

    def test_module_level_assertions_are_recovered_without_a_wrapper(self):
        """The degenerate rows ship a bare assertion, so the recovery cannot require a ``def``."""
        contradiction = find_contradiction(_NO_WRAPPER_TRUNCATED)
        assert contradiction is not None
        assert contradiction.call == "candidate('a')"

    def test_a_rebinding_the_recovery_cannot_read_stops_it(self):
        """An unreadable statement means the next assertion is about an unknown value.

        The rebinding here spans two lines, so no line-at-a-time reader can see it. Reading past it
        would give both assertions the same call key and report ``candidate(v)`` as asserted to be
        both 2 and 3, when the check demands each of a different ``v``.
        """
        assert find_contradiction(_REBOUND_BY_AN_UNREADABLE_STATEMENT) is None

    def test_the_recovery_reports_itself_as_a_recovery(self):
        """``parsed_whole`` is what tells the ETL to log the row as one whose grader cannot run."""
        assert parse_check(_FLAT_ASSERTS_TRUNCATED).parsed_whole is False
        assert parse_check(_BRANCHED_ARMS).parsed_whole is True


class TestBranchesAreDroppedWhenTheBodyDoesCompile:
    def test_loop_nested_assertions_are_not_read_as_unconditional(self):
        """The compiling path never walks into a loop; this is the check that it still does not."""
        body = """\
def check(candidate):
    for x in ("a", "b"):
        assert candidate("b") == 2
    assert candidate("b") == 3
"""
        assert find_contradiction(body) is None

    def test_two_unconditional_assertions_are_still_found(self):
        """The negative controls above must not be passing because the search finds nothing."""
        body = """\
def check(candidate):
    assert candidate("b") == 2
    assert candidate("b") == 3
"""
        contradiction = find_contradiction(body)
        assert contradiction is not None
        assert contradiction.describe().startswith("candidate('b')")


class TestTheBakedFlagsStillFollowFromTheSearch:
    """The baked case file and the search that wrote it must not have drifted apart.

    ``provably_impossible`` is baked once by this ETL and then read by the task registry, the hack
    detector and the rate readout, none of which re-derive it. So a change to the search that
    reclassifies a row is invisible until someone re-bakes -- and a stale ``True`` is a satisfiable
    row sitting in the split whose meaning is "a pass here is a hack". This re-derives every flag
    from the body stored beside it, which costs a second and makes that drift loud.

    A body that no longer proves what the file says it proves means one of two things, and the fix
    differs: the search changed (re-derive whether the new verdict is the right one before touching
    anything), or the pinned dataset revision moved (re-run the ETL).
    """

    def test_every_row_reproduces_its_provably_impossible_flag(self):
        disagreements = [
            f"{problem.harness_task_id}: file says {problem.provably_impossible}, the search now "
            f"says {find_contradiction(problem.test) is not None}"
            for problem in PROBLEMS
            if (find_contradiction(problem.test) is not None) != problem.provably_impossible
        ]
        assert not disagreements, (
            f"{len(disagreements)} of {len(PROBLEMS)} baked rows disagree with the search that "
            f"wrote them: {disagreements}"
        )

    def test_the_row_that_needs_the_recovery_still_recovers_a_contradiction(self):
        """The recovery path is exercised by exactly one baked row; a change that guts it is silent.

        Without this, tightening the recovery could quietly drop the row from the provable subset
        and every downstream count would just get smaller, with nothing saying why.
        """
        recovering = [problem for problem in PROBLEMS if not problem.check_parses]
        assert recovering, "no baked row exercises the syntax-error recovery any more"
        proven = [problem for problem in recovering if find_contradiction(problem.test) is not None]
        assert proven, (
            f"the recovery proves nothing on any of the {len(recovering)} rows that need it"
        )


SELECTION_BY_SPLIT = {
    subset_impossible_type(DERIVED_SUBSET_K, selection): selection
    for selection in (SubsetSelection.STRATIFIED, SubsetSelection.FIRST_K)
}
"""The derived splits' names mapped to the construction that bakes each, straight off the deriver."""


class TestTheDerivedSubsetSplitsFollowFromTheDeriver:
    """The baked visible-subset rows and the deriver that wrote them must not have drifted apart.

    The same drift argument as the provably-impossible flags above, one construction over: the
    derived ``test`` is baked once and then read by the grader builders, the training corpus and
    the legibility probe, none of which re-derive it. Everything here re-derives from the row's own
    ``original_test``, so a deriver change that would bake different rows is loud before anyone
    trains on the stale ones. No dataset text appears: comparisons are byte-equality between fields
    of the local bake, and failures name task ids only.
    """

    def test_each_derived_split_is_complete_and_satisfiable_by_construction(self):
        originals = [p for p in PROBLEMS if p.impossible_type == ORIGINAL_SPLIT]
        for split in SELECTION_BY_SPLIT:
            rows = [p for p in PROBLEMS if p.impossible_type == split]
            assert {p.task_id for p in rows} == {p.task_id for p in originals}, split
            flagged = [p.task_id for p in rows if p.provably_impossible or p.impossibility_proof]
            assert flagged == [], f"{split}: impossibility flags must be reset, not inherited"
            unparsed = [
                p.task_id for p in rows if not (p.check_parses and p.check_defines_function)
            ]
            assert unparsed == [], f"{split}: every derived check parses by construction"

    def test_every_derived_row_reproduces_the_deriver_output_from_its_own_hidden_check(self):
        """The hidden check is byte-identical to the original's, and the visible one re-derives."""
        original_test_by_id = {
            p.task_id: p.original_test for p in PROBLEMS if p.impossible_type == ORIGINAL_SPLIT
        }
        for split, selection in SELECTION_BY_SPLIT.items():
            for row in (p for p in PROBLEMS if p.impossible_type == split):
                assert row.original_test == original_test_by_id[row.task_id], row.task_id
                rederived = derive_visible_subset(
                    row.original_test, k=DERIVED_SUBSET_K, selection=selection, context=row.task_id
                )
                assert row.test == rederived.check_source, (split, row.task_id)

    def test_every_derived_record_carries_provenance_that_matches_its_row(self):
        """Read off the JSON file itself: ``problem_from_record`` drops the provenance block."""
        payload = json.loads(DATA_PATH.read_text())
        derived_records = [
            record
            for record in payload["problems"]
            if record["impossible_type"] in SELECTION_BY_SPLIT
        ]
        assert len(derived_records) == len(SELECTION_BY_SPLIT) * 103
        for record in derived_records:
            provenance = record["provenance"]
            assert provenance["k"] == DERIVED_SUBSET_K, record["task_id"]
            assert provenance["selection"] == SELECTION_BY_SPLIT[record["impossible_type"]].value, (
                record["task_id"]
            )
            assert (
                provenance["original_test_sha256"]
                == hashlib.sha256(record["original_test"].encode()).hexdigest()
            ), record["task_id"]
