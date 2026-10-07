import json
from pathlib import Path

import pytest

from reward_hacking.organism_rl.cli import parse_args, resolve_sizing, write_provenance


def test_defaults_and_vram_derived_sizing():
    args = parse_args(["--output-dir", "artifacts/toy"])
    assert args.temperature == args.top_p == 1.0
    small = resolve_sizing(total_bytes=8 * 1024**3, hidden_size=1024, layers=12)
    large = resolve_sizing(total_bytes=16 * 1024**3, hidden_size=1024, layers=12)
    assert large.max_sequence_tokens > small.max_sequence_tokens
    assert small.micro_batch_size == 1


def test_teacher_requires_stored_scores():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "artifacts/toy", "--teacher-beta", "0.1"])


def test_provenance_has_flags_version_and_sha(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path)])
    path = write_provenance(args, {"name": "cpu"})
    value = json.loads(path.read_text())
    assert value["trl_version"] == "1.10.0"
    assert len(value["git_sha"]) == 40
    assert value["flags"]["temperature"] == 1.0


def test_resume_preserves_all_flags_and_refuses_sampling_drift(tmp_path: Path) -> None:
    args = parse_args(["--output-dir", str(tmp_path)])
    write_provenance(args, {"name": "cpu"})
    args.resume_from_checkpoint = tmp_path / "checkpoint-1"
    path = write_provenance(args, {"name": "cpu"})
    assert json.loads(path.read_text())["flags"]["resume_from_checkpoint"] == str(
        args.resume_from_checkpoint
    )
    args.temperature = 0.5
    with pytest.raises(ValueError, match="flags"):
        write_provenance(args, {"name": "cpu"})


def test_tracked_output_directory_is_refused() -> None:
    args = parse_args(["--output-dir", "reward_hacking/organism_rl"])
    with pytest.raises(ValueError, match="gitignored"):
        write_provenance(args, {"name": "cpu"})


def test_teacher_score_content_drift_is_refused(tmp_path: Path) -> None:
    teacher = tmp_path / "scores.jsonl"
    teacher.write_text('{"teacher_logps": [-1.0]}\n')
    args = parse_args(
        ["--output-dir", str(tmp_path), "--teacher-beta", "0.1", "--teacher-path", str(teacher)]
    )
    write_provenance(args, {"name": "cpu"})
    teacher.write_text('{"teacher_logps": [-2.0]}\n')
    with pytest.raises(ValueError, match="teacher scores"):
        write_provenance(args, {"name": "cpu"})
