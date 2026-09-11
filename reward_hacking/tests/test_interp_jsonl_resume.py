"""Unit tests for the JSONL unit-level resume ledger behind the games interp legs.

The driver-level tests (`games/tests/test_games_interp_steering.py::TestGenerateResume`,
`games/tests/test_games_interp_patching.py::TestPatchResume`) prove the two legs resume into files
byte-identical to an uninterrupted run. What is pinned here is the ledger's own contract, which both
legs and any future unit-shaped leg rest on: identity compares in JSON types, a partial unit is
dropped and reported while a complete one is kept verbatim, a ledger that got ahead of its records
(the shape a records-then-ledger sync leaves) is trimmed back to what is on disk, and the three ways
a records file can stop being provably a continuation of this run -- no ledger, a ledger for another
configuration, a file holding more records for a unit than its own ledger recorded -- are refused
rather than appended to.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.interp.jsonl_resume import (
    RESUME_LEDGER_SUFFIX,
    ResumeLedger,
    ResumeMismatchError,
    ledger_path_for,
    normalize_identity,
    read_jsonl_lines,
    resume_records,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

IDENTITY: dict[str, Any] = {"command": "generate", "seed": 0, "condition_keys": ["none", "steer"]}


def _unit(record: Mapping[str, Any]) -> str:
    return str(record["unit"])


def _write_unit(records_path: Path, ledger: ResumeLedger, unit: str, n: int) -> None:
    """Append ``n`` records for ``unit`` and mark it complete, the way a driver does."""
    with records_path.open("a", encoding="utf-8") as handle:
        for index in range(n):
            handle.write(json.dumps({"unit": unit, "index": index}) + "\n")
        handle.flush()
    ledger.mark_complete(unit, n)


class TestIdentity:
    def test_tuple_and_list_compare_equal_after_the_json_round_trip(self) -> None:
        assert normalize_identity({"layers": (0, 1)}) == normalize_identity({"layers": [0, 1]})

    def test_a_first_run_writes_a_ledger_with_its_identity_and_no_units(
        self, tmp_path: Path
    ) -> None:
        records_path = tmp_path / "records.jsonl"
        state, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.kept_records == []
        assert state.complete_units == {}
        assert ledger.path == ledger_path_for(records_path)
        assert ledger.path.name == "records.jsonl" + RESUME_LEDGER_SUFFIX
        payload = json.loads(ledger.path.read_text())
        assert payload == {"identity": IDENTITY, "completed_units": {}}

    def test_a_relaunch_with_a_different_identity_is_refused_naming_the_field(
        self, tmp_path: Path
    ) -> None:
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 2)
        with pytest.raises(ResumeMismatchError, match="seed: on disk 0 vs this run 1"):
            resume_records(records_path, identity={**IDENTITY, "seed": 1}, unit_of=_unit)
        with pytest.raises(ResumeMismatchError, match="extra"):
            resume_records(records_path, identity={**IDENTITY, "extra": 1}, unit_of=_unit)


class TestResumeState:
    def test_complete_units_are_kept_verbatim_and_a_partial_unit_is_dropped(
        self, tmp_path: Path
    ) -> None:
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 2)
        complete_bytes = records_path.read_bytes()
        # A death after two records of the next unit, mid-write on the third.
        with records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"unit": "steer", "index": 0}) + "\n")
            handle.write(json.dumps({"unit": "steer", "index": 1}) + "\n")
            handle.write('{"unit": "ste')

        state, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.complete_units == {"none": 2}
        assert state.n_kept == 2
        assert [record["index"] for record in state.kept_records] == [0, 1]
        assert state.dropped_records == 2
        assert state.dropped_units == ("steer",)
        assert records_path.read_bytes() == complete_bytes, "kept lines are written back verbatim"
        assert ledger.completed == {"none": 2}

    def test_a_ledger_ahead_of_its_records_drops_the_unit_it_names_and_regenerates_it(
        self, tmp_path: Path
    ) -> None:
        """The sync race: the records file was uploaded, a unit completed, then the ledger was.

        The synced ledger names a unit whose rows never reached the synced records file. That is a
        partial unit seen from the ledger's side, and it is recovered the same way: the ledger is
        trimmed to what is on disk, the unit runs again, and the file ends where an unbroken run's
        would.
        """
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 2)
        synced_records = records_path.read_bytes()
        _write_unit(records_path, ledger, "steer", 3)
        records_path.write_bytes(synced_records)

        state, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.complete_units == {"none": 2}
        assert state.dropped_units == ("steer",)
        assert state.dropped_records == 0, "nothing of the unit was on disk to drop"
        assert ledger.completed == {"none": 2}
        assert ResumeLedger.load(ledger.path).completed == {"none": 2}, "the trim is persisted"
        assert records_path.read_bytes() == synced_records

        _write_unit(records_path, ledger, "steer", 3)
        state, _ = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.complete_units == {"none": 2, "steer": 3}
        assert state.n_kept == 5

    def test_a_ledger_ahead_by_part_of_a_unit_drops_that_units_rows_too(
        self, tmp_path: Path
    ) -> None:
        """Synced mid-unit: two of the unit's three rows landed, and the ledger says three."""
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 2)
        complete_bytes = records_path.read_bytes()
        _write_unit(records_path, ledger, "steer", 3)
        lines = records_path.read_bytes().splitlines(keepends=True)
        records_path.write_bytes(b"".join(lines[:4]))

        state, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.complete_units == {"none": 2}
        assert state.dropped_units == ("steer",)
        assert state.dropped_records == 2
        assert ledger.completed == {"none": 2}
        assert records_path.read_bytes() == complete_bytes

    def test_a_truncated_final_line_alone_is_dropped_without_touching_complete_units(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text('{"unit": "a"}\n{"unit": "b"}\n{"unit": "c"')
        lines, truncated = read_jsonl_lines(path)
        assert lines == ['{"unit": "a"}', '{"unit": "b"}']
        assert truncated is True

    def test_a_malformed_line_that_is_not_the_final_one_is_data_loss_and_raises(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "records.jsonl"
        path.write_text('{"unit": "a"}\nnot json\n{"unit": "c"}\n')
        with pytest.raises(ResumeMismatchError, match="not a truncated write"):
            read_jsonl_lines(path)


class TestRefusals:
    def test_records_without_a_ledger_are_refused(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        records_path.write_text('{"unit": "none"}\n')
        with pytest.raises(ResumeMismatchError, match="no ledger"):
            resume_records(records_path, identity=IDENTITY, unit_of=_unit)

    def test_a_ledger_whose_records_were_deleted_is_refused(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 1)
        records_path.unlink()
        with pytest.raises(ResumeMismatchError, match="deleted out from under"):
            resume_records(records_path, identity=IDENTITY, unit_of=_unit)

    def test_a_ledger_with_no_completed_units_and_no_records_resumes_from_nothing(
        self, tmp_path: Path
    ) -> None:
        records_path = tmp_path / "records.jsonl"
        resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        state, _ = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert state.kept_records == []

    def test_more_records_than_the_ledger_recorded_are_refused(self, tmp_path: Path) -> None:
        """A surplus cannot be explained by regeneration; a second live writer is the likely cause."""
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 3)
        lines = records_path.read_bytes().splitlines(keepends=True)
        records_path.write_bytes(b"".join([*lines, lines[0]]))
        with pytest.raises(ResumeMismatchError, match="none: 4 on disk vs 3 in the ledger"):
            resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        assert ResumeLedger.load(ledger.path).completed == {"none": 3}, "a refusal touches nothing"

    def test_a_file_that_is_not_a_ledger_is_refused(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        ledger_path_for(records_path).write_text('{"something": "else"}')
        with pytest.raises(ResumeMismatchError, match="not a resume ledger"):
            resume_records(records_path, identity=IDENTITY, unit_of=_unit)


class TestLedgerWrites:
    def test_mark_complete_persists_atomically_leaving_no_temp_file(self, tmp_path: Path) -> None:
        records_path = tmp_path / "records.jsonl"
        _, ledger = resume_records(records_path, identity=IDENTITY, unit_of=_unit)
        _write_unit(records_path, ledger, "none", 2)
        _write_unit(records_path, ledger, "steer", 1)
        assert ResumeLedger.load(ledger.path).completed == {"none": 2, "steer": 1}
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "records.jsonl",
            "records.jsonl" + RESUME_LEDGER_SUFFIX,
        ]
