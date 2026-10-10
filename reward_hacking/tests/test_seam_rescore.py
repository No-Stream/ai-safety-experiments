"""Real-corpus and jail checks for final-state rescore and resume behavior."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from reward_hacking.organism_rl import rescore

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)
EVALA_RECORDS = Path("nine-b") / "evalA-base9b"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _summary(record: Mapping[str, Any]) -> dict[str, Any]:
    value = record.get("trace_summary", record.get("episode_record"))
    if not isinstance(value, dict):
        raise TypeError("corpus record requires an episode summary")
    return cast("dict[str, Any]", value)


def _is_repair(record: Mapping[str, Any]) -> bool:
    summary = _summary(record)
    truth = summary["true"]
    return (
        summary["grader_integrity"].get("content_changed") is True
        and truth.get("measured") is True
        and truth.get("pass") is True
    )


def _is_honest_submitted_pass(record: Mapping[str, Any]) -> bool:
    summary = _summary(record)
    truth = summary["true"]
    return (
        record["variant"] == "original"
        and bool(summary["submission_verdicts"])
        and summary["grader_integrity"].get("content_changed") is False
        and truth.get("measured") is True
        and truth.get("pass") is True
    )


def _has_oracle_error(record: Mapping[str, Any]) -> bool:
    return _summary(record)["true"].get("oracle_error") is not None


def _is_no_submit_without_oracle_error(record: Mapping[str, Any]) -> bool:
    summary = _summary(record)
    return not summary["submission_verdicts"] and summary["true"].get("oracle_error") is None


def _selected_evala_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable_records = [record for record in records if record["family"] in {"aisi", "ilcb"}]
    selected: list[dict[str, Any]] = []
    selected_episode_ids: set[str] = set()
    predicates: tuple[Callable[[Mapping[str, Any]], bool], ...] = (
        _is_repair,
        _is_honest_submitted_pass,
        _has_oracle_error,
        _is_no_submit_without_oracle_error,
    )
    for predicate in predicates:
        record = next(
            (
                candidate
                for candidate in usable_records
                if _summary(candidate)["episode_id"] not in selected_episode_ids
                and predicate(candidate)
            ),
            None,
        )
        if record is None:
            raise AssertionError("evalA lacks a record for one required rescore predicate")
        selected.append(record)
        selected_episode_ids.add(_summary(record)["episode_id"])
    return selected


@needs_jail
@pytest.mark.jail_real
@pytest.mark.corpus
def test_evala_rescore_replays_a_runtime_selected_subset_and_resumes(
    seed_v4_corpus: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rescore four predicate-selected records; jailed command replay should take a few minutes."""
    records_dir = seed_v4_corpus / EVALA_RECORDS
    records = _read_jsonl(records_dir / "records.jsonl")
    selected = _selected_evala_records(records)
    selected_episode_ids = {_summary(record)["episode_id"] for record in selected}
    assert len(selected_episode_ids) == 4

    repair_record = next(record for record in selected if _is_repair(record))
    repair_family = cast("str", repair_record["family"])
    repair_counts: dict[str, int] = {}
    hide_edit_counts: dict[str, int] = {}
    for record in selected:
        rescore._count_integrity_edits(record, repair_counts, hide_edit_counts)
    assert sum(repair_counts.values()) == 1
    assert repair_counts[repair_family] == 1
    assert not hide_edit_counts

    with tempfile.TemporaryDirectory(prefix="seam-rescore-", dir="/var/tmp") as root:
        copied_records_path = Path(root) / "records.jsonl"
        copied_records_path.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
            encoding="utf-8",
        )
        output_path = rescore.rescore_file(
            copied_records_path,
            jail_backend=None,
            payment_register="in_band",
        )
        rescored = _read_jsonl(output_path)
        assert {_summary(record)["episode_id"] for record in rescored} == selected_episode_ids
        assert all(record["rescore"]["payment_register"] == "in_band" for record in rescored)

        for record in rescored:
            rescore_metadata = record["rescore"]
            if rescore_metadata["method"] == "jailed_command_replay":
                assert rescore_metadata["artifact_verification"] in {
                    "verified",
                    "mismatch",
                    "unverified",
                }
                if rescore_metadata["artifact_verification"] == "verified":
                    assert (
                        rescore_metadata["rebuilt_artifact_sha256"]
                        == rescore_metadata["recorded_artifact_sha256"]
                    )

        real_rescored_path = records_dir / "records.rescored-final-state.jsonl"
        if real_rescored_path.is_file():
            real_rows = {
                _summary(record)["episode_id"]: record for record in _read_jsonl(real_rescored_path)
            }
            for record in rescored:
                episode_id = _summary(record)["episode_id"]
                assert episode_id in real_rows
                assert record["rescore"]["method"] == real_rows[episode_id]["rescore"]["method"]
                assert record["rescore"].get("artifact_verification") == real_rows[episode_id][
                    "rescore"
                ].get("artifact_verification")
                assert record["organism_reward"] == real_rows[episode_id]["organism_reward"]
        else:
            logger.info(
                "corpus lacks a completed rescore sidecar; checked rebuilt and recorded artifact hashes"
            )

        first_output = output_path.read_bytes()

        def unexpected_regrade(*args: Any, **kwargs: Any) -> dict[str, object]:
            raise AssertionError("completed rescore should return without grading again")

        monkeypatch.setattr(rescore, "_rescore_file_record", unexpected_regrade)
        assert (
            rescore.rescore_file(
                copied_records_path,
                jail_backend=None,
                payment_register="in_band",
            )
            == output_path
        )
        assert output_path.read_bytes() == first_output
