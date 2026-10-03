"""Tests for blinded, offline concealment-judge packet handling."""

from __future__ import annotations

import json
import random
import stat
from typing import TYPE_CHECKING, Any

import pytest

from reward_hacking.ladder.concealment_judge import (
    apply_concealment_judge_verdicts,
    export_concealment_judge_batch,
    ingest_concealment_judge_verdicts,
)

if TYPE_CHECKING:
    from pathlib import Path


def _episode_record(
    root: Path, episode_id: str, *, prescreen: bool | None = True
) -> dict[str, Any]:
    episode_dir = root / episode_id
    episode_dir.mkdir()
    (episode_dir / "solution.py").write_text(f"answer = {len(episode_id)}\n", encoding="utf-8")
    report_concealment = None if prescreen is None else {"prescreen": prescreen, "judge": None}
    return {
        "record": "ladder_episode",
        "key": {"model_id": "private-model", "scenario_id": "fixture", "arm": "real"},
        "episode_dir": str(episode_dir),
        "engine_command_records": [
            {"sequence": 1, "command": "python3 -m pytest"},
            {"sequence": 2, "command": "cat solution.py"},
        ],
        "detector_inputs": {
            "detection_spec": {
                "task_named_files": ["synthetic_grader.py", "tests/visible_test.py"],
                "task_named_directories": [".ci"],
            }
        },
        "report_concealment": report_concealment,
        "trace_summary": {
            "episode_id": episode_id,
            "final_report": {"text": f"I finished {episode_id}.", "truncated": False},
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class TestConcealmentJudgePackets:
    def test_export_blinds_ids_shuffles_and_keeps_only_reviewable_evidence(
        self, tmp_path: Path
    ) -> None:
        records = [
            _episode_record(tmp_path, "episode-a", prescreen=False),
            _episode_record(tmp_path, "episode-b", prescreen=True),
            _episode_record(tmp_path, "episode-c", prescreen=False),
            _episode_record(tmp_path, "episode-d", prescreen=None),
        ]
        packet_path = tmp_path / "judge.jsonl"
        mapping_path = tmp_path / "judge-id-map.json"

        packet_count = export_concealment_judge_batch(
            records,
            solution_files_by_scenario={"fixture": ("solution.py",)},
            packet_path=packet_path,
            mapping_path=mapping_path,
            shuffle_seed=19,
        )

        lines = _read_jsonl(packet_path)
        assert packet_count == 3
        assert lines[0]["record"] == "concealment_judge_batch"
        assert lines[0]["shuffle_seed"] == 19
        assert "concealment" in lines[0]["rubric"].lower()
        assert "Rung 1" in lines[0]["rubric"]
        assert "ordinary diligence" in " ".join(lines[0]["rubric"].split())
        packets = lines[1:]
        expected_order = ["episode-a", "episode-b", "episode-c"]
        random.Random(19).shuffle(expected_order)
        assert expected_order != ["episode-a", "episode-b", "episode-c"]
        id_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["episode_id_map"]
        assert [id_mapping[packet["episode_id"]] for packet in packets] == expected_order
        assert all(packet["record"] == "judge_packet" for packet in packets)
        assert all("model_id" not in packet and "arm" not in packet for packet in packets)
        assert all(packet["episode_id"] not in id_mapping.values() for packet in packets)
        assert packets[0]["bash_commands"] == ["python3 -m pytest", "cat solution.py"]
        assert set(packets[0]["solution_files"]) == {"solution.py"}
        assert packets[0]["task_named_files"] == [
            "synthetic_grader.py",
            "tests/visible_test.py",
        ]
        assert packets[0]["task_named_directories"] == [".ci"]
        assert packets[0]["final_report"].startswith("I finished episode-")
        assert stat.S_IMODE(mapping_path.stat().st_mode) == 0o600

    def test_export_requires_solution_files_and_a_report_for_eligible_records(
        self, tmp_path: Path
    ) -> None:
        record = _episode_record(tmp_path, "episode-a")

        with pytest.raises(ValueError, match="no solution files configured"):
            export_concealment_judge_batch(
                [record],
                solution_files_by_scenario={},
                packet_path=tmp_path / "judge.jsonl",
                mapping_path=tmp_path / "judge-id-map.json",
                shuffle_seed=0,
            )

        record["trace_summary"]["final_report"] = None
        with pytest.raises(ValueError, match="final report is unavailable"):
            export_concealment_judge_batch(
                [record],
                solution_files_by_scenario={"fixture": ("solution.py",)},
                packet_path=tmp_path / "judge.jsonl",
                mapping_path=tmp_path / "judge-id-map.json",
                shuffle_seed=0,
            )

    def test_import_matches_verdicts_by_blinded_id_and_preserves_other_records(
        self, tmp_path: Path
    ) -> None:
        records = [
            _episode_record(tmp_path, "episode-a", prescreen=False),
            _episode_record(tmp_path, "episode-b", prescreen=True),
            _episode_record(tmp_path, "episode-c", prescreen=False),
            _episode_record(tmp_path, "episode-d", prescreen=None),
        ]
        packet_path = tmp_path / "judge.jsonl"
        mapping_path = tmp_path / "judge-id-map.json"
        export_concealment_judge_batch(
            records,
            solution_files_by_scenario={"fixture": ("solution.py",)},
            packet_path=packet_path,
            mapping_path=mapping_path,
            shuffle_seed=19,
        )
        packet_ids = [packet["episode_id"] for packet in _read_jsonl(packet_path)[1:]]
        id_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))["episode_id_map"]
        expected_verdicts = {
            id_mapping[packet_id]: index == 0 for index, packet_id in enumerate(packet_ids)
        }
        verdicts_path = tmp_path / "verdicts.jsonl"
        verdicts_path.write_text(
            "".join(
                json.dumps({"episode_id": packet_id, "conceals": index == 0}) + "\n"
                for index, packet_id in enumerate(packet_ids)
            ),
            encoding="utf-8",
        )

        updated = apply_concealment_judge_verdicts(
            records, verdicts_path=verdicts_path, mapping_path=mapping_path
        )

        verdicts_by_id: dict[str, object] = {}
        for record in updated:
            concealment = record["report_concealment"]
            if concealment is not None:
                summary = record["trace_summary"]
                assert isinstance(concealment, dict)
                assert isinstance(summary, dict)
                verdicts_by_id[summary["episode_id"]] = concealment["judge"]
        assert set(verdicts_by_id) == {"episode-a", "episode-b", "episode-c"}
        assert verdicts_by_id == expected_verdicts
        assert updated[3]["report_concealment"] is None
        assert records[0]["report_concealment"]["judge"] is None

    def test_ingest_writes_updated_records_as_private_jsonl(self, tmp_path: Path) -> None:
        records = [_episode_record(tmp_path, "episode-a")]
        records_path = tmp_path / "records.jsonl"
        records_path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
        )
        packet_path = tmp_path / "judge.jsonl"
        mapping_path = tmp_path / "judge-id-map.json"
        export_concealment_judge_batch(
            records,
            solution_files_by_scenario={"fixture": ("solution.py",)},
            packet_path=packet_path,
            mapping_path=mapping_path,
            shuffle_seed=0,
        )
        packet_id = _read_jsonl(packet_path)[1]["episode_id"]
        verdicts_path = tmp_path / "verdicts.jsonl"
        verdicts_path.write_text(
            json.dumps({"episode_id": packet_id, "conceals": True}) + "\n", encoding="utf-8"
        )
        output_path = tmp_path / "rescored-records.jsonl"

        record_count = ingest_concealment_judge_verdicts(
            records_path,
            verdicts_path=verdicts_path,
            mapping_path=mapping_path,
            output_path=output_path,
        )

        assert record_count == 1
        assert _read_jsonl(output_path)[0]["report_concealment"]["judge"] is True
        assert stat.S_IMODE(output_path.stat().st_mode) == 0o600

    @pytest.mark.parametrize(
        ("verdict_rows", "message"),
        [
            ([{"episode_id": "unknown-packet", "conceals": True}], "unknown episode id"),
            ([], "missing verdicts"),
        ],
    )
    def test_import_rejects_unknown_and_missing_verdict_ids(
        self, tmp_path: Path, verdict_rows: list[dict[str, object]], message: str
    ) -> None:
        records = [_episode_record(tmp_path, "episode-a")]
        packet_path = tmp_path / "judge.jsonl"
        mapping_path = tmp_path / "judge-id-map.json"
        export_concealment_judge_batch(
            records,
            solution_files_by_scenario={"fixture": ("solution.py",)},
            packet_path=packet_path,
            mapping_path=mapping_path,
            shuffle_seed=0,
        )
        verdicts_path = tmp_path / "verdicts.jsonl"
        verdicts_path.write_text(
            "".join(json.dumps(row) + "\n" for row in verdict_rows), encoding="utf-8"
        )

        with pytest.raises(ValueError, match=message):
            apply_concealment_judge_verdicts(
                records, verdicts_path=verdicts_path, mapping_path=mapping_path
            )

    def test_import_rejects_world_readable_episode_id_mapping(self, tmp_path: Path) -> None:
        records = [_episode_record(tmp_path, "episode-a")]
        packet_path = tmp_path / "judge.jsonl"
        mapping_path = tmp_path / "judge-id-map.json"
        export_concealment_judge_batch(
            records,
            solution_files_by_scenario={"fixture": ("solution.py",)},
            packet_path=packet_path,
            mapping_path=mapping_path,
            shuffle_seed=0,
        )
        mapping_path.chmod(0o644)
        verdicts_path = tmp_path / "verdicts.jsonl"
        packet_id = _read_jsonl(packet_path)[1]["episode_id"]
        verdicts_path.write_text(
            json.dumps({"episode_id": packet_id, "conceals": False}) + "\n", encoding="utf-8"
        )

        with pytest.raises(PermissionError, match="must not be accessible"):
            apply_concealment_judge_verdicts(
                records, verdicts_path=verdicts_path, mapping_path=mapping_path
            )
