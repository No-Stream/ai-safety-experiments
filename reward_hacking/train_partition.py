"""The stored problem-id partition that keeps the transfer evaluation genuinely held out.

The misspecified-grader training experiment claims that a disposition learned against loose graders
shows up in an environment the training never covered. That claim is only as good as the held-out
set, so the split is held on three axes at once and this module owns the first: **which
Impossible-LiveCodeBench problems training may see at all.** The other two are structural -- training
is single-turn while the evaluation runs the multi-turn agentic harness, and training exposes only
answer-shaped hacks while the evaluation also exposes the grader file and the machinery mechanisms.

**The partition is written down once and read, never recomputed at a call site.** That is a
correction rather than a preference: this repo has a logged incident where a control set selected from
"the records currently on disk" silently re-picked itself the next time records were added, so two
runs that were supposed to share a control did not. Both entry points -- the training corpus builder
and the evaluation task selector -- load the same file, and neither may derive a filter of its own.

Two guards make that stick, and both are refusals rather than warnings: the **pool fingerprint**,
a hash of the exact problem-id pool the partition was drawn from that :func:`load_partition`
recomputes and refuses a mismatch on, since a partition whose pool moved is not a smaller partition
but an unknown one; and **disjointness on both sides at load**, because a problem id on both sides
contaminates the eval set while every count still looks healthy.

The partition is over *problem* ids (``lcbhard_N``), never over harness task ids
(``ilcb-oneoff-lcbhard_N``). The ILCB splits are different graders over the same 103 problems, so
holding out a harness id would leave the same problem's statement, signature and true answer in
training under a different split's name -- which is the leak the whole partition exists to prevent.

Design of record, including why each constant is what it is: ``docs/ilcb-corpus-and-oracle.md``.

    uv run python -m reward_hacking.train_partition --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, cast

from reward_hacking.harness.tasks_ilcb import PROBLEMS
from reward_hacking.ilcb_data import DATA_SCHEMA

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

PARTITION_SCHEMA = 1

TRAINING_SIDE = "training"
HELD_OUT_SIDE = "held-out"

DEFAULT_PARTITION_PATH = Path("artifacts/reward_hacking/option3/held-out-partition.json")

# Not a knob a caller may vary per run: two arms drawing different partitions would be two runs.
DEFAULT_PARTITION_SEED = 0

# Sized so the held-out side can carry a rate rather than an anecdote; docs/ilcb-corpus-and-oracle.md.
HELD_OUT_FRACTION = 0.4

# How many offenders a refusal names; the message always carries the full count beside the excerpt.
REPORTED_OFFENDERS = 5

# `conflicting` is eval-only by construction, and `train_dataset` refuses it as a training split.
SPLIT_ORIGINAL = "original"
SPLIT_ONEOFF = "oneoff"
SPLIT_CONFLICTING = "conflicting"
SPLIT_SUBSET3_STRATIFIED = "subset3-stratified"
SPLIT_SUBSET3 = "subset3"
ILCB_SPLITS: tuple[str, ...] = (
    SPLIT_ORIGINAL,
    SPLIT_ONEOFF,
    SPLIT_CONFLICTING,
    SPLIT_SUBSET3_STRATIFIED,
    SPLIT_SUBSET3,
)


@dataclass(frozen=True, slots=True)
class HeldOutPartition:
    """One problem-id partition: which problems training may see, and which are held out.

    Frozen and validated in ``__post_init__`` rather than checked by whoever happens to read it,
    because an overlapping partition is the one defect a later re-analysis cannot notice: every rate
    still has a denominator, every count still looks plausible, and the transfer claim is simply
    false.
    """

    schema: int
    seed: int
    held_out_fraction: float
    pool_fingerprint: str
    pool_size: int
    training_problem_ids: tuple[str, ...]
    held_out_problem_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Refuse a partition that is empty, overlapping, unsorted or short of its own pool."""
        if self.schema != PARTITION_SCHEMA:
            raise ValueError(f"partition schema {self.schema} is not {PARTITION_SCHEMA}")
        training = frozenset(self.training_problem_ids)
        held_out = frozenset(self.held_out_problem_ids)
        if not training or not held_out:
            raise ValueError(
                f"both sides must be non-empty, got {len(training)} training and "
                f"{len(held_out)} held-out problem ids"
            )
        shared = sorted(training & held_out)
        if shared:
            # An excerpt for the message; the refusal carries the full count beside it.
            excerpt = shared[:REPORTED_OFFENDERS]
            raise ValueError(
                f"{len(shared)} problem id(s) appear on both sides of the partition (e.g. "
                f"{excerpt}), so the held-out evaluation is contaminated by training and no "
                f"transfer claim can be read off it"
            )
        for side, ids in (
            (TRAINING_SIDE, self.training_problem_ids),
            (HELD_OUT_SIDE, self.held_out_problem_ids),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"the {side} side repeats a problem id: {sorted(ids)}")
            if list(ids) != sorted(ids):
                raise ValueError(
                    f"the {side} side is not sorted, so two files holding the same partition "
                    f"would not compare equal byte for byte"
                )
        if len(training) + len(held_out) != self.pool_size:
            raise ValueError(
                f"the two sides hold {len(training) + len(held_out)} problem ids but the recorded "
                f"pool had {self.pool_size}; a partition that drops problems silently is not a "
                f"partition"
            )

    def side_of(self, problem_id: str) -> str:
        """Return which side a problem id belongs to, refusing one the partition never saw."""
        if problem_id in frozenset(self.training_problem_ids):
            return TRAINING_SIDE
        if problem_id in frozenset(self.held_out_problem_ids):
            return HELD_OUT_SIDE
        raise KeyError(
            f"problem id {problem_id!r} is in neither side of this partition, so nothing says "
            f"whether training was allowed to see it"
        )

    def to_json_dict(self) -> dict[str, object]:
        """Serialise the partition, every field included so a reader need derive nothing."""
        return {
            "schema": self.schema,
            "seed": self.seed,
            "held_out_fraction": self.held_out_fraction,
            "pool_fingerprint": self.pool_fingerprint,
            "pool_size": self.pool_size,
            "n_training": len(self.training_problem_ids),
            "n_held_out": len(self.held_out_problem_ids),
            "training_problem_ids": list(self.training_problem_ids),
            "held_out_problem_ids": list(self.held_out_problem_ids),
        }

    @classmethod
    def from_json_dict(cls, payload: Mapping[str, Any]) -> HeldOutPartition:
        """Rebuild a partition from its file, letting ``__post_init__`` do the checking.

        ``n_training`` and ``n_held_out`` are ignored on the way back in: they are derived, so
        trusting them would let a hand-edited count outvote the lists it was derived from.
        """
        return cls(
            schema=int(payload["schema"]),
            seed=int(payload["seed"]),
            held_out_fraction=float(payload["held_out_fraction"]),
            pool_fingerprint=str(payload["pool_fingerprint"]),
            pool_size=int(payload["pool_size"]),
            training_problem_ids=tuple(str(value) for value in payload["training_problem_ids"]),
            held_out_problem_ids=tuple(str(value) for value in payload["held_out_problem_ids"]),
        )


def problem_pool() -> tuple[str, ...]:
    """Return the sorted problem ids usable in EVERY ILCB split.

    Every split, rather than each split's own usable set, because the partition is one assignment
    shared by the training arms and the evaluation: a problem whose ``conflicting`` grader does not
    compile cannot carry the held-out headline, and one whose ``oneoff`` grader does not compile
    cannot be trained on, so a pool that included either would make the two sides mean different
    things. One problem is dropped by this rule today -- the dataset truncated its check mid-literal
    -- and ``ilcb_tasks(include_broken_checks=False)`` drops the same rows downstream, so the pool
    and the runnable task sets agree by construction rather than by coincidence.
    """
    parsing: dict[str, set[str]] = {}
    for problem in PROBLEMS:
        if problem.check_parses:
            parsing.setdefault(problem.task_id, set()).add(problem.impossible_type)
    wanted = frozenset(ILCB_SPLITS)
    pool = tuple(sorted(task_id for task_id, splits in parsing.items() if splits >= wanted))
    if not pool:
        raise RuntimeError(
            f"no ILCB problem has a compiling check in all of {ILCB_SPLITS}; the baked case file is "
            f"not the one this partition was written against"
        )
    return pool


def pool_fingerprint(pool: Sequence[str]) -> str:
    """Hash the exact pool a partition was drawn from, so a moved pool is a refusal not a surprise.

    The data schema goes into the hash beside the ids: a schema bump can change what a row means
    without changing any id, and a partition drawn under the old meaning is not evidence about the
    new one.
    """
    payload = f"ilcb-schema={DATA_SCHEMA}\n" + "\n".join(pool)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_partition(
    *,
    pool: Sequence[str] | None = None,
    seed: int = DEFAULT_PARTITION_SEED,
    held_out_fraction: float = HELD_OUT_FRACTION,
) -> HeldOutPartition:
    """Draw the partition once, from a seeded shuffle of the sorted pool.

    Shuffled rather than sliced: the ids are ordered by the dataset's own numbering, and a prefix of
    that order is not a random sample of anything. Seeded, so the same pool and seed reproduce the
    same partition exactly -- which is what makes the stored file recoverable if it is ever lost,
    and what the determinism test asserts.

    Nothing here looks at which problems are *interesting* -- not their difficulty, not whether
    their conflicting grader is provably impossible, not the base model's pass rate. A held-out set
    chosen for a property of the measurement would answer a question about that property.
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError(
            f"held_out_fraction must leave both sides non-empty, got {held_out_fraction}"
        )
    resolved = tuple(pool) if pool is not None else problem_pool()
    if len(set(resolved)) != len(resolved):
        raise ValueError("the pool repeats a problem id")
    ordered = sorted(resolved)
    shuffled = list(ordered)
    Random(seed).shuffle(shuffled)
    n_held_out = round(len(ordered) * held_out_fraction)
    n_held_out = min(max(n_held_out, 1), len(ordered) - 1)
    held_out = tuple(sorted(shuffled[:n_held_out]))
    training = tuple(sorted(shuffled[n_held_out:]))
    return HeldOutPartition(
        schema=PARTITION_SCHEMA,
        seed=seed,
        held_out_fraction=held_out_fraction,
        pool_fingerprint=pool_fingerprint(ordered),
        pool_size=len(ordered),
        training_problem_ids=training,
        held_out_problem_ids=held_out,
    )


def write_partition(partition: HeldOutPartition, path: Path) -> None:
    """Persist the partition, refusing to overwrite one that already exists.

    A refusal, because overwriting is how a partition stops being the thing both entry points
    agreed on: a second write with a different seed would relabel which problems an already-trained
    checkpoint was allowed to see, and nothing in the checkpoint would disagree.
    """
    if path.exists():
        raise FileExistsError(
            f"{path} already holds a partition. Overwriting it would relabel which problems any "
            f"existing checkpoint was trained on, so write a new path instead, or delete this one "
            f"deliberately if no run has used it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partition.to_json_dict(), indent=2) + "\n", encoding="utf-8")
    logger.info(
        "wrote the held-out partition, %s",
        f"{path=} n_training={len(partition.training_problem_ids)} "
        f"n_held_out={len(partition.held_out_problem_ids)} seed={partition.seed}",
    )


def load_partition(path: Path) -> HeldOutPartition:
    """Read the stored partition and refuse one drawn from a pool that has since moved.

    The fingerprint check is the guard that matters. A partition file is the only record of which
    problems a checkpoint was allowed to see, and it is read months after the run; if the baked case
    file has been regenerated in between, the ids may still all exist while the pool they were drawn
    from is different, and the held-out side would silently no longer be the complement of the
    training side.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"no held-out partition at {path}. Both the training corpus and the held-out "
            f"evaluation read this one file and neither may derive a filter of its own; write it "
            f"once with `python -m reward_hacking.train_partition --write`."
        )
    partition = HeldOutPartition.from_json_dict(
        cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    )
    live = problem_pool()
    live_fingerprint = pool_fingerprint(live)
    if live_fingerprint != partition.pool_fingerprint:
        raise RuntimeError(
            f"{path} was drawn from a problem pool whose fingerprint was "
            f"{partition.pool_fingerprint} over {partition.pool_size} problems, and this box's "
            f"registry fingerprints as {live_fingerprint} over {len(live)}. The pool moved, so "
            f"nothing says the held-out side is still the complement of the training side. "
            f"Regenerate the case file this partition was written against, or draw a fresh "
            f"partition and retrain -- do not run either arm against this one."
        )
    logger.info(
        "loaded the held-out partition, %s",
        f"{path=} n_training={len(partition.training_problem_ids)} "
        f"n_held_out={len(partition.held_out_problem_ids)} "
        f"fingerprint={partition.pool_fingerprint[:12]}",
    )
    return partition


def problem_id_of(harness_task_id: str) -> str:
    """Recover the problem id from a harness task id (``ilcb-oneoff-lcbhard_N`` -> ``lcbhard_N``).

    Split on the recognised split names rather than on the first two hyphens: the problem ids carry
    their own separators, and a positional parse would break on the first id shaped differently.
    Longest split name first, because one split name is a prefix of another: matched in tuple
    order, ``ilcb-subset3-`` would swallow ``ilcb-subset3-stratified-lcbhard_N`` and hand back
    ``stratified-lcbhard_N`` as the problem id.
    """
    for split in sorted(ILCB_SPLITS, key=len, reverse=True):
        prefix = f"ilcb-{split}-"
        if harness_task_id.startswith(prefix):
            return harness_task_id[len(prefix) :]
    raise ValueError(
        f"{harness_task_id!r} is not an ILCB harness task id, so no problem id can be read out of "
        f"it; expected one of {[f'ilcb-{split}-<problem>' for split in ILCB_SPLITS]}"
    )


def assert_side(harness_task_ids: Iterable[str], partition: HeldOutPartition, *, side: str) -> None:
    """Refuse a task set that reaches across the partition, naming the offenders.

    Called by both entry points on the ids they are about to run, which is the point: the file being
    correct is necessary and not sufficient, since a filter applied to the wrong column produces a
    plausible task list on the wrong side. Watched to fail by planting a training problem into the
    held-out selection.
    """
    if side not in (TRAINING_SIDE, HELD_OUT_SIDE):
        raise ValueError(f"unknown partition side {side!r}")
    wrong = sorted(
        task_id for task_id in harness_task_ids if partition.side_of(problem_id_of(task_id)) != side
    )
    if wrong:
        # An excerpt for the message; the refusal carries the full count beside it.
        excerpt = wrong[:REPORTED_OFFENDERS]
        raise ValueError(
            f"{len(wrong)} task id(s) selected for the {side} side belong to the other side of the "
            f"held-out partition (e.g. {excerpt}). Training on a held-out problem, or evaluating "
            f"on a trained one, makes the transfer measurement meaningless while every count still "
            f"looks healthy."
        )


def describe_partition(partition: HeldOutPartition) -> dict[str, object]:
    """Report what each side actually carries, per split, so a zero has its denominator.

    The provably-impossible count on the held-out side is the one number the headline depends on: a
    pass on a self-contradictory grader is a hack by construction, and that argument holds only for
    the rows where the contradiction was proved. Two conflicting rows in this dataset carry no
    contradiction, so the count is read off the flag rather than off the split name.
    """
    by_side: dict[str, object] = {}
    for side, ids in (
        (TRAINING_SIDE, partition.training_problem_ids),
        (HELD_OUT_SIDE, partition.held_out_problem_ids),
    ):
        members = frozenset(ids)
        rows = [problem for problem in PROBLEMS if problem.task_id in members]
        by_side[side] = {
            "n_problems": len(members),
            "n_rows_by_split": {
                split: sum(
                    1
                    for problem in rows
                    if problem.impossible_type == split and problem.check_parses
                )
                for split in ILCB_SPLITS
            },
            "n_provably_impossible_conflicting": sum(
                1
                for problem in rows
                if problem.impossible_type == SPLIT_CONFLICTING
                and problem.check_parses
                and problem.provably_impossible
            ),
        }
    return {
        "seed": partition.seed,
        "held_out_fraction": partition.held_out_fraction,
        "pool_size": partition.pool_size,
        "pool_fingerprint": partition.pool_fingerprint,
        "sides": by_side,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write the partition once, or describe the one already stored."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_PARTITION_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_PARTITION_SEED)
    parser.add_argument("--held-out-fraction", type=float, default=HELD_OUT_FRACTION)
    parser.add_argument(
        "--write",
        action="store_true",
        help="draw and store the partition; refuses to overwrite an existing file",
    )
    args = parser.parse_args(argv)
    if args.write:
        write_partition(
            build_partition(seed=args.seed, held_out_fraction=args.held_out_fraction), args.path
        )
    partition = load_partition(args.path)
    logger.info("partition composition: %s", json.dumps(describe_partition(partition), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
