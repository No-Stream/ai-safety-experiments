"""Real-corpus and jail checks for final-state rescore and resume behavior."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from conftest import needs_jail

from reward_hacking.organism_rl import rescore

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

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


def _assert_evala_rescore_rows(
    selected_records: list[dict[str, Any]],
    sidecar_rows: dict[str, dict[str, Any]],
) -> dict[str, str]:
    selected_episode_ids = [_summary(record)["episode_id"] for record in selected_records]
    assert len(set(selected_episode_ids)) == 4
    assert set(selected_episode_ids) <= sidecar_rows.keys()
    expected_methods = {
        episode_id: sidecar_rows[episode_id]["rescore"]["method"]
        for episode_id in selected_episode_ids
    }
    assert sorted(expected_methods.values()) == [
        "jailed_command_replay",
        "recorded_submission_evidence",
        "recorded_submission_evidence",
        "recorded_submission_evidence",
    ]
    replayed_records = [
        sidecar_rows[episode_id]
        for episode_id in selected_episode_ids
        if expected_methods[episode_id] == "jailed_command_replay"
    ]
    assert len(replayed_records) == 1
    replay_metadata = replayed_records[0]["rescore"]
    assert replay_metadata["artifact_verification"] == "verified"
    rebuilt_artifact_hash = replay_metadata["rebuilt_artifact_sha256"]
    recorded_artifact_hash = replay_metadata["recorded_artifact_sha256"]
    assert isinstance(rebuilt_artifact_hash, str)
    assert re.fullmatch(r"[0-9a-f]{64}", rebuilt_artifact_hash)
    assert isinstance(recorded_artifact_hash, str)
    assert re.fullmatch(r"[0-9a-f]{64}", recorded_artifact_hash)
    assert rebuilt_artifact_hash == recorded_artifact_hash
    return expected_methods


def _evala_rescore_sidecar(
    records_directory: Path,
    selected_records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rescore_path = records_directory / "records.rescored-final-state.jsonl"
    assert rescore_path.is_file(), "evalA corpus requires its completed rescore sidecar"
    sidecar_rows = {_summary(record)["episode_id"]: record for record in _read_jsonl(rescore_path)}
    expected_methods = _assert_evala_rescore_rows(selected_records, sidecar_rows)
    return sidecar_rows, expected_methods


def _reward_read_as_record_schema_3(reward: Mapping[str, Any]) -> dict[str, Any]:
    """The evalA sidecar predates record schema 3, which stores unmeasured truth as None.

    The manifest lists that schema pair as compatible; this applies the same legacy rule to the
    sidecar's reward evidence so the comparison with a fresh replay stays exact elsewhere.
    """
    evidence = cast("dict[str, Any]", reward["evidence"])
    if evidence["true_measured"] is not False:
        return dict(reward)
    assert evidence["true_pass"] in (False, None)
    return {**reward, "evidence": {**evidence, "true_pass": None}}


def test_evala_rescore_requires_a_completed_sidecar(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="requires its completed rescore sidecar"):
        _evala_rescore_sidecar(tmp_path, [])


@pytest.mark.corpus
def test_evala_rescore_sidecar_has_verified_selected_replay(seed_v4_corpus: Path) -> None:
    records_directory = seed_v4_corpus / EVALA_RECORDS
    selected_records = _selected_evala_records(_read_jsonl(records_directory / "records.jsonl"))
    sidecar_rows, expected_methods = _evala_rescore_sidecar(records_directory, selected_records)

    replayed_episode_id = next(
        episode_id
        for episode_id, method in expected_methods.items()
        if method == "jailed_command_replay"
    )
    tampered_replay = dict(sidecar_rows[replayed_episode_id])
    tampered_replay["rescore"] = {
        **tampered_replay["rescore"],
        "method": "recorded_submission_evidence",
    }
    tampered_sidecar_rows = {**sidecar_rows, replayed_episode_id: tampered_replay}
    with pytest.raises(AssertionError):
        _assert_evala_rescore_rows(selected_records, tampered_sidecar_rows)

    evidence_episode_id = next(
        episode_id
        for episode_id, method in expected_methods.items()
        if method == "recorded_submission_evidence"
    )
    tampered_evidence = dict(sidecar_rows[evidence_episode_id])
    tampered_evidence["rescore"] = {
        **tampered_evidence["rescore"],
        "method": "unexpected_method",
    }
    tampered_sidecar_rows = {**sidecar_rows, evidence_episode_id: tampered_evidence}
    with pytest.raises(AssertionError):
        _assert_evala_rescore_rows(selected_records, tampered_sidecar_rows)

    tampered_replay = dict(sidecar_rows[replayed_episode_id])
    tampered_replay["rescore"] = {
        **tampered_replay["rescore"],
        "artifact_verification": "mismatch",
    }
    tampered_sidecar_rows = {**sidecar_rows, replayed_episode_id: tampered_replay}
    with pytest.raises(AssertionError):
        _assert_evala_rescore_rows(selected_records, tampered_sidecar_rows)

    replay_metadata = sidecar_rows[replayed_episode_id]["rescore"]
    recorded_hash = cast("str", replay_metadata["recorded_artifact_sha256"])
    different_hash = "0" * 64 if recorded_hash != "0" * 64 else "1" * 64
    tampered_replay = dict(sidecar_rows[replayed_episode_id])
    tampered_replay["rescore"] = {
        **replay_metadata,
        "rebuilt_artifact_sha256": different_hash,
    }
    tampered_sidecar_rows = {**sidecar_rows, replayed_episode_id: tampered_replay}
    with pytest.raises(AssertionError):
        _assert_evala_rescore_rows(selected_records, tampered_sidecar_rows)


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

        real_rows, expected_methods = _evala_rescore_sidecar(records_dir, selected)
        replayed_rows = 0
        for record in rescored:
            episode_id = _summary(record)["episode_id"]
            rescore_metadata = record["rescore"]
            expected_metadata = real_rows[episode_id]["rescore"]
            assert rescore_metadata["method"] == expected_methods[episode_id]
            assert rescore_metadata.get("artifact_verification") == expected_metadata.get(
                "artifact_verification"
            )
            assert record["organism_reward"] == _reward_read_as_record_schema_3(
                real_rows[episode_id]["organism_reward"]
            )
            if rescore_metadata["method"] == "jailed_command_replay":
                assert rescore_metadata["artifact_verification"] == "verified"
                assert isinstance(rescore_metadata["rebuilt_artifact_sha256"], str)
                assert isinstance(rescore_metadata["recorded_artifact_sha256"], str)
                assert (
                    rescore_metadata["rebuilt_artifact_sha256"]
                    == rescore_metadata["recorded_artifact_sha256"]
                )
                replayed_rows += 1
        assert replayed_rows == 1

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
