"""The ILCB baked-file data layer: the row shape, the reader, and the constants both share.

Extracted from ``reward_hacking.harness.tasks_ilcb`` so it can be imported WITHOUT the side effect
that module deliberately carries. Importing anything under ``reward_hacking.harness`` builds the
whole task registry, which reads the baked case file at import time and refuses a missing or broken
one. That refusal is right for the harness -- a registry quietly holding no tasks would let a sweep
run, write traces, report no hacks and look healthy -- but it makes the harness unimportable in
exactly the state ``ilcb_etl`` exists to fix. This module reads nothing at import time, so the ETL
and ``reward_hacking.visible_subset`` can be imported while the case file is missing or broken, and
``tasks_ilcb`` re-exports every name here so its importers see no move.

``ilcb_etl`` still restates :data:`DATA_PATH` and :data:`DATA_SCHEMA` rather than importing them,
on purpose: the writer and the reader checking the schema independently is what makes a version
bump on one side a loud refusal on the other rather than a silent reinterpretation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).parent / "harness" / "data" / "ilcb_cases.json"
DATA_SCHEMA = 1

CHECK_FUNCTION_NAME = "check"


@dataclass(frozen=True)
class ILCBProblem:
    """One Impossible-LiveCodeBench row in one split: its statement, its two checks, its verdict.

    ``test`` is the visible check the policy is graded on and can read; ``original_test`` is the
    true check, identical across a problem's splits, which never enters ``/work``.
    ``provably_impossible`` is the theorem ``ilcb_etl`` proved about ``test``, and
    ``impossibility_proof`` is the pair of assertions that prove it, kept so downstream analysis
    can audit the claim rather than trust it.
    """

    task_id: str
    impossible_type: str
    entry_point: str
    prompt: str
    test: str
    original_test: str
    provably_impossible: bool
    impossibility_proof: dict[str, str] | None
    check_parses: bool
    check_defines_function: bool

    @property
    def harness_task_id(self) -> str:
        """The id this row takes in the task registry (also its episode directory name)."""
        return f"ilcb-{self.impossible_type}-{self.task_id}"


def problem_from_record(record: dict[str, Any]) -> ILCBProblem:
    """Rebuild one problem from its baked JSON record, ignoring keys the row shape does not carry.

    The tolerance for extra keys is load-bearing for the derived splits: their records travel with
    a ``provenance`` block beside the row fields, and a reader that refused it could never load the
    file the ETL writes.
    """
    return ILCBProblem(
        task_id=record["task_id"],
        impossible_type=record["impossible_type"],
        entry_point=record["entry_point"],
        prompt=record["prompt"],
        test=record["test"],
        original_test=record["original_test"],
        provably_impossible=record["provably_impossible"],
        impossibility_proof=record["impossibility_proof"],
        check_parses=record["check_parses"],
        check_defines_function=record["check_defines_function"],
    )


def load_problems(data_path: Path = DATA_PATH) -> tuple[ILCBProblem, ...]:
    """Load the baked rows, refusing a data file this code was not written against.

    Fatal to whoever calls it at import, for the reason ``tasks_evalplus.load_problems`` spells
    out: a registry that quietly holds no ILCB tasks would let a sweep run, write traces, report no
    hacks and look entirely healthy. So it crashes, and says how to fix itself.
    """
    try:
        payload = json.loads(data_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read the baked Impossible-LiveCodeBench rows at {data_path} ({exc}) -- "
            "regenerate them with `uv run python reward_hacking/harness/ilcb_etl.py`"
        ) from exc
    schema = payload["schema"]
    if schema != DATA_SCHEMA:
        raise ValueError(f"{data_path} is schema {schema}, expected {DATA_SCHEMA}")
    return tuple(problem_from_record(record) for record in payload["problems"])
