"""The stimulus loader: refusal without the file, shape validation, frame rendering, digests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from sociology.stimulus import load_stimulus
from sociology.tests.conftest import SYNTHETIC_STIMULUS, write_synthetic_stimulus

if TYPE_CHECKING:
    from pathlib import Path


class TestLoadStimulus:
    def test_missing_file_refuses_with_the_path_and_the_reason(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.json"
        with pytest.raises(FileNotFoundError, match="gitignored"):
            load_stimulus(missing)

    def test_wrong_version_refuses(self, tmp_path: Path) -> None:
        payload = dict(SYNTHETIC_STIMULUS, version="something-else")
        path = tmp_path / "stimulus.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="version"):
            load_stimulus(path)

    def test_missing_frame_refuses_by_name(self, tmp_path: Path) -> None:
        payload = json.loads(json.dumps(SYNTHETIC_STIMULUS))
        del payload["frames"]["independent"]
        path = tmp_path / "stimulus.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="independent"):
            load_stimulus(path)

    def test_frame_text_renders_the_bundle_size(self, tmp_path: Path) -> None:
        stimulus = load_stimulus(write_synthetic_stimulus(tmp_path / "stimulus.json"))
        assert "for 10 transcripts" in stimulus.frame_text("population", n=10)
        with pytest.raises(ValueError, match="unknown framing"):
            stimulus.frame_text("nonexistent", n=10)

    def test_digest_tracks_the_file_content(self, tmp_path: Path) -> None:
        first = load_stimulus(write_synthetic_stimulus(tmp_path / "a.json"))
        edited = json.loads(json.dumps(SYNTHETIC_STIMULUS))
        edited["constant_request"] = "SYNTHETIC-REQUEST: something else."
        path = tmp_path / "b.json"
        path.write_text(json.dumps(edited), encoding="utf-8")
        second = load_stimulus(path)
        assert first.digest != second.digest

    def test_validation_replies_load_with_expectations(self, tmp_path: Path) -> None:
        stimulus = load_stimulus(write_synthetic_stimulus(tmp_path / "stimulus.json"))
        assert [reply.name for reply in stimulus.validation_replies] == ["v-none", "v-coord"]
        assert stimulus.validation_replies[1].expected_severity == "explicit_coordination"
