"""The leakage-hint loader: named variants, the reserved no-hint variant, and loud failures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from reward_hacking.organisms.leakage_hints import NO_HINT_VARIANT, load_leakage_hint

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def hints_path(tmp_path: Path) -> Path:
    path = tmp_path / "leakage_hints.json"
    path.write_text(
        json.dumps({"hints": {"toy_env": {"plain": "Dummy hint text.", "blank": "  "}}}),
        encoding="utf-8",
    )
    return path


class TestLoadLeakageHint:
    def test_named_variant_returns_its_text_and_a_stable_identity(self, hints_path: Path) -> None:
        hint = load_leakage_hint("toy_env", "plain", hints_path)
        assert hint.text == "Dummy hint text."
        assert hint.identity() == load_leakage_hint("toy_env", "plain", hints_path).identity()
        assert hint.identity()["hint_variant"] == "plain"

    def test_no_hint_variant_is_empty_and_never_reads_the_file(self, tmp_path: Path) -> None:
        hint = load_leakage_hint("toy_env", NO_HINT_VARIANT, tmp_path / "missing.json")
        assert hint.text == ""

    def test_edited_text_changes_the_identity(self, hints_path: Path, tmp_path: Path) -> None:
        edited = tmp_path / "edited.json"
        edited.write_text(
            json.dumps({"hints": {"toy_env": {"plain": "Different text."}}}), encoding="utf-8"
        )
        original_digest = load_leakage_hint("toy_env", "plain", hints_path).text_sha256
        assert load_leakage_hint("toy_env", "plain", edited).text_sha256 != original_digest

    @pytest.mark.parametrize(
        ("environment", "variant", "error"),
        [
            ("other_env", "plain", KeyError),
            ("toy_env", "missing", KeyError),
            ("toy_env", "blank", ValueError),
        ],
    )
    def test_unknown_or_blank_hints_raise(
        self, hints_path: Path, environment: str, variant: str, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            load_leakage_hint(environment, variant, hints_path)
