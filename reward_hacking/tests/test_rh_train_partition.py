"""The held-out partition: disjointness, determinism, the pool fingerprint, and the side assertion.

Every test here is about a failure that would leave a plausible-looking run whose transfer claim is
false, so the assertions are on the refusals rather than on the happy path.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

from reward_hacking.train_partition import (
    DEFAULT_PARTITION_PATH,
    HELD_OUT_SIDE,
    ILCB_SPLITS,
    PARTITION_SCHEMA,
    SPLIT_CONFLICTING,
    SPLIT_ONEOFF,
    SPLIT_SUBSET3,
    SPLIT_SUBSET3_STRATIFIED,
    TRAINING_SIDE,
    HeldOutPartition,
    assert_side,
    build_partition,
    describe_partition,
    load_partition,
    pool_fingerprint,
    problem_id_of,
    problem_pool,
    write_partition,
)

if TYPE_CHECKING:
    from pathlib import Path

SYNTHETIC_POOL = tuple(f"toy_{index}" for index in range(20))


def toy_partition(**overrides: object) -> HeldOutPartition:
    fields: dict[str, object] = {
        "schema": PARTITION_SCHEMA,
        "seed": 0,
        "held_out_fraction": 0.4,
        "pool_fingerprint": "deadbeef",
        "pool_size": 4,
        "training_problem_ids": ("a", "b"),
        "held_out_problem_ids": ("c", "d"),
    }
    fields.update(overrides)
    return HeldOutPartition(**fields)  # pyright: ignore[reportArgumentType]


class TestPartitionInvariants:
    def test_a_valid_partition_reports_each_side(self):
        partition = toy_partition()
        assert partition.side_of("a") == TRAINING_SIDE
        assert partition.side_of("c") == HELD_OUT_SIDE

    def test_an_unknown_problem_id_raises_rather_than_defaulting(self):
        with pytest.raises(KeyError, match="neither side"):
            toy_partition().side_of("nope")

    def test_an_overlapping_partition_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="both sides"):
            toy_partition(
                training_problem_ids=("a", "b", "c"), held_out_problem_ids=("c", "d"), pool_size=5
            )

    def test_an_empty_side_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="non-empty"):
            toy_partition(training_problem_ids=(), pool_size=2)

    def test_an_unsorted_side_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="not sorted"):
            toy_partition(training_problem_ids=("b", "a"))

    def test_a_partition_that_lost_problems_cannot_be_constructed(self):
        with pytest.raises(ValueError, match="drops problems"):
            toy_partition(pool_size=99)

    def test_the_round_trip_through_json_preserves_the_partition(self):
        partition = toy_partition()
        assert HeldOutPartition.from_json_dict(partition.to_json_dict()) == partition

    def test_a_hand_edited_count_cannot_outvote_the_lists(self):
        payload = toy_partition().to_json_dict()
        payload["n_training"] = 999
        assert HeldOutPartition.from_json_dict(payload).training_problem_ids == ("a", "b")


class TestBuildPartition:
    def test_the_draw_is_deterministic_in_the_seed(self):
        first = build_partition(pool=SYNTHETIC_POOL, seed=7)
        second = build_partition(pool=SYNTHETIC_POOL, seed=7)
        assert first == second

    def test_a_different_seed_draws_a_different_partition(self):
        assert build_partition(pool=SYNTHETIC_POOL, seed=1) != build_partition(
            pool=SYNTHETIC_POOL, seed=2
        )

    def test_the_input_order_does_not_change_the_draw(self):
        shuffled = tuple(reversed(SYNTHETIC_POOL))
        assert build_partition(pool=shuffled, seed=3) == build_partition(
            pool=SYNTHETIC_POOL, seed=3
        )

    def test_the_two_sides_cover_the_pool_exactly_once(self):
        partition = build_partition(pool=SYNTHETIC_POOL, seed=0)
        covered = (*partition.training_problem_ids, *partition.held_out_problem_ids)
        assert sorted(covered) == sorted(SYNTHETIC_POOL)

    def test_a_fraction_that_would_empty_a_side_is_refused(self):
        with pytest.raises(ValueError, match="both sides non-empty"):
            build_partition(pool=SYNTHETIC_POOL, held_out_fraction=1.0)

    def test_a_tiny_pool_still_leaves_one_problem_on_each_side(self):
        partition = build_partition(pool=("only_a", "only_b"), held_out_fraction=0.01)
        assert len(partition.training_problem_ids) == 1
        assert len(partition.held_out_problem_ids) == 1


class TestStoredPartitionFile:
    def test_writing_then_loading_returns_the_same_partition(self, tmp_path: Path):
        partition = build_partition(seed=0)
        path = tmp_path / "partition.json"
        write_partition(partition, path)
        assert load_partition(path) == partition

    def test_overwriting_an_existing_partition_is_refused(self, tmp_path: Path):
        path = tmp_path / "partition.json"
        write_partition(build_partition(seed=0), path)
        with pytest.raises(FileExistsError, match="already holds a partition"):
            write_partition(build_partition(seed=1), path)

    def test_a_missing_partition_names_the_command_that_writes_one(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="train_partition --write"):
            load_partition(tmp_path / "absent.json")

    def test_a_moved_pool_is_refused_rather_than_silently_accepted(self, tmp_path: Path):
        """The sabotage this file exists for: a partition drawn from a pool that has since changed.

        Simulated by rewriting the recorded fingerprint, which is exactly what a regenerated case
        file would produce -- the ids all still exist while the pool they were drawn from differs.
        """
        path = tmp_path / "partition.json"
        write_partition(build_partition(seed=0), path)
        payload = json.loads(path.read_text())
        payload["pool_fingerprint"] = "0" * 64
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="pool moved"):
            load_partition(path)


class TestSideAssertion:
    def test_the_training_side_accepts_only_training_task_ids(self):
        partition = build_partition(seed=0)
        training = partition.training_problem_ids[0]
        assert_side([f"ilcb-{SPLIT_ONEOFF}-{training}"], partition, side=TRAINING_SIDE)

    def test_a_held_out_problem_in_the_training_selection_is_refused(self):
        """Sabotage: plant one held-out problem into a training selection and watch it go red."""
        partition = build_partition(seed=0)
        planted = f"ilcb-{SPLIT_ONEOFF}-{partition.held_out_problem_ids[0]}"
        legitimate = f"ilcb-{SPLIT_ONEOFF}-{partition.training_problem_ids[0]}"
        with pytest.raises(ValueError, match="belong to the other side"):
            assert_side([legitimate, planted], partition, side=TRAINING_SIDE)

    def test_a_training_problem_in_the_held_out_selection_is_refused(self):
        partition = build_partition(seed=0)
        planted = f"ilcb-{SPLIT_CONFLICTING}-{partition.training_problem_ids[0]}"
        with pytest.raises(ValueError, match="belong to the other side"):
            assert_side([planted], partition, side=HELD_OUT_SIDE)

    def test_an_unknown_side_name_is_refused(self):
        with pytest.raises(ValueError, match="unknown partition side"):
            assert_side([], build_partition(seed=0), side="somewhere-else")


class TestProblemIdParsing:
    @pytest.mark.parametrize("split", list(ILCB_SPLITS))
    def test_every_split_prefix_is_stripped(self, split: str):
        assert problem_id_of(f"ilcb-{split}-lcbhard_12") == "lcbhard_12"

    def test_a_problem_id_carrying_hyphens_survives(self):
        assert problem_id_of(f"ilcb-{SPLIT_ONEOFF}-odd-id-3") == "odd-id-3"

    def test_one_split_name_being_a_prefix_of_another_does_not_misparse(self):
        """``subset3`` is a prefix of ``subset3-stratified``, so tuple-order matching would hand
        back ``stratified-lcbhard_12`` as a problem id -- an id in neither partition side, which
        ``assert_side`` would then refuse for the wrong reason on every legible-arm row."""
        assert SPLIT_SUBSET3_STRATIFIED.startswith(SPLIT_SUBSET3), (
            "the collision this test exists for is gone; keep the parser property anyway"
        )
        assert problem_id_of(f"ilcb-{SPLIT_SUBSET3_STRATIFIED}-lcbhard_12") == "lcbhard_12"
        assert problem_id_of(f"ilcb-{SPLIT_SUBSET3}-lcbhard_12") == "lcbhard_12"

    def test_a_non_ilcb_task_id_raises(self):
        with pytest.raises(ValueError, match="not an ILCB harness task id"):
            problem_id_of("sum-ledger")


class TestLivePool:
    def test_the_pool_holds_only_problems_usable_in_all_three_splits(self):
        pool = problem_pool()
        assert len(pool) > 1
        assert list(pool) == sorted(set(pool))

    def test_the_fingerprint_changes_when_the_pool_does(self):
        pool = problem_pool()
        assert pool_fingerprint(pool) != pool_fingerprint(pool[:-1])

    def test_the_composition_report_names_both_sides_and_every_split(self):
        described = describe_partition(build_partition(seed=0))
        sides = cast("dict[str, dict[str, Any]]", described["sides"])
        for side in (TRAINING_SIDE, HELD_OUT_SIDE):
            cell = sides[side]
            assert set(cell["n_rows_by_split"]) == set(ILCB_SPLITS)
            # The denominator the headline depends on: a pass on a self-contradictory grader is
            # an exploit only where the contradiction was proved, and two dataset rows carry none.
            assert cell["n_provably_impossible_conflicting"] >= 1

    def test_the_derived_splits_cover_each_side_in_full(self):
        """Every pool problem has a usable derived row, so the legible arm's corpus is whole."""
        described = describe_partition(build_partition(seed=0))
        sides = cast("dict[str, dict[str, Any]]", described["sides"])
        for side in (TRAINING_SIDE, HELD_OUT_SIDE):
            cell = sides[side]
            for split in (SPLIT_SUBSET3_STRATIFIED, SPLIT_SUBSET3):
                assert cell["n_rows_by_split"][split] == cell["n_problems"], (side, split)


class TestStoredPartitionSurvivesTheDerivedBake:
    """The stored 61/41 partition must still validate against the re-baked five-split file.

    The pool fingerprint hashes the ids usable in EVERY split plus the data schema, so adding the
    two derived splits was designed to leave it untouched: every original row derives, so no id
    joins or leaves the pool. This pins that design fact against the artifact the training and
    evaluation entry points actually read -- if it goes red, the partition is invalidated and
    nothing may train until the owner decides how to re-draw it.
    """

    @pytest.mark.skipif(
        not DEFAULT_PARTITION_PATH.is_file(),
        reason=f"no stored partition at {DEFAULT_PARTITION_PATH} (machine-local artifact)",
    )
    def test_the_stored_partition_still_loads_against_the_live_registry(self):
        partition = load_partition(DEFAULT_PARTITION_PATH)
        assert partition.pool_fingerprint == pool_fingerprint(problem_pool())
