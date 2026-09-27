"""CPU contracts for the J-space workspace readout.

These tests deliberately exercise only the artifact and statistics seams.  They do not load a
model, fit a Jacobian lens, or inspect the private research corpus.  In particular, the paired
examples below are synthetic token scores: a readout that gets the ranking or denominator wrong
still produces plausible-looking JSON, so each expected ordering is written out here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest
import torch

from games.workspace_readout import (
    bootstrap_indices,
    find_word_positions,
    is_word_like,
    map_concepts,
    parse_band,
    rank_paired_differences,
    should_skip_cell,
    validate_manifest,
    validate_residual,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestBandParsing:
    def test_accepts_human_layer_notation_and_resolves_negative_endpoints(self) -> None:
        assert parse_band("L10-L26", n_layers=32) == (10, 26)
        assert parse_band("10:26", n_layers=32) == (10, 26)
        assert parse_band("L-8-L-1", n_layers=32) == (24, 31)

    @pytest.mark.parametrize("band", ["", "L10", "L26-L10", "L-40-L-1", "L10-L32"])
    def test_rejects_malformed_or_out_of_range_bands(self, band: str) -> None:
        with pytest.raises(ValueError, match=r"band|layer"):
            parse_band(band, n_layers=32)


class TestWordLikeFilter:
    @pytest.mark.parametrize("token", ["mirror", "Ġmirror", "▁cooperate", "42"])
    def test_keeps_word_like_tokens(self, token: str) -> None:
        assert is_word_like(token)

    @pytest.mark.parametrize("token", ["", "▁", "!", "##ing", "<0x0A>", "<|endoftext|>"])
    def test_drops_punctuation_continuations_and_special_tokens(self, token: str) -> None:
        assert not is_word_like(token)


class TestPairedDifferenceRanking:
    def test_ranks_absolute_paired_changes_and_keeps_direction(self) -> None:
        rows = [
            {"pair_id": "p1", "token": "mirror", "base": 0.10, "trained": 0.70},
            {"pair_id": "p2", "token": "dominant", "base": 0.90, "trained": 0.20},
            {"pair_id": "p3", "token": "neutral", "base": 0.40, "trained": 0.45},
        ]

        ranked = rank_paired_differences(rows, top_k=2)

        assert [(row["pair_id"], row["delta"]) for row in ranked] == [
            ("p1", pytest.approx(0.60)),
            ("p2", pytest.approx(-0.70)),
        ]

    def test_ties_are_resolved_by_token_then_pair_id(self) -> None:
        rows = [
            {"pair_id": "p2", "token": "beta", "base": 0.0, "trained": 0.5},
            {"pair_id": "p1", "token": "alpha", "base": 0.5, "trained": 0.0},
        ]

        ranked = rank_paired_differences(rows, top_k=2)

        assert [(row["token"], row["pair_id"]) for row in ranked] == [
            ("alpha", "p1"),
            ("beta", "p2"),
        ]


class TestConceptMapping:
    def test_maps_each_token_to_all_matching_concepts_deterministically(self) -> None:
        concept_sets = {
            "mirroring": ("mirror", "mutual"),
            "cooperation": ("cooperate", "mutual"),
        }

        mapped = map_concepts(("mutual", "cooperate", "unknown"), concept_sets)

        assert mapped == {
            "mutual": ("cooperation", "mirroring"),
            "cooperate": ("cooperation",),
        }

    def test_concept_mapping_does_not_emit_empty_concepts(self) -> None:
        mapped = map_concepts(("unseen",), {"mirroring": ("mirror",)})

        assert mapped == {}


class TestBootstrapDeterminism:
    def test_same_seed_reproduces_the_exact_resample_indices(self) -> None:
        first = bootstrap_indices(n_observations=5, n_resamples=32, seed=20260926)
        second = bootstrap_indices(n_observations=5, n_resamples=32, seed=20260926)

        assert first == second
        assert len(first) == 32
        assert all(len(sample) == 5 for sample in first)
        assert all(0 <= index < 5 for sample in first for index in sample)

    def test_seed_changes_the_resample_stream(self) -> None:
        first = bootstrap_indices(n_observations=5, n_resamples=32, seed=1)
        second = bootstrap_indices(n_observations=5, n_resamples=32, seed=2)

        assert first != second


class TestResumeAndArtifactValidation:
    @staticmethod
    def _identity() -> dict[str, Any]:
        return {
            "format_version": 1,
            "model_id": "synthetic-base",
            "target_layer": 30,
            "hidden_size": 4,
            "residual_shape": [2, 4],
        }

    def test_complete_matching_cell_is_skipped(self, tmp_path: Path) -> None:
        cell = tmp_path / "base-step-0"
        cell.mkdir()
        identity = self._identity()
        (cell / "manifest.json").write_text(
            '{"format_version": 1, "model_id": "synthetic-base", "target_layer": 30, '
            '"hidden_size": 4, "residual_shape": [2, 4]}\n',
            encoding="utf-8",
        )
        torch.save(torch.zeros((2, 4), dtype=torch.float32), cell / "residual.pt")

        assert should_skip_cell(cell, identity) is True

    def test_partial_or_mismatched_cell_is_not_skipped(self, tmp_path: Path) -> None:
        cell = tmp_path / "base-step-0"
        cell.mkdir()
        identity = self._identity()
        (cell / "manifest.json").write_text(
            '{"format_version": 1, "model_id": "synthetic-base", "target_layer": 29, '
            '"hidden_size": 4, "residual_shape": [2, 4]}\n',
            encoding="utf-8",
        )

        assert should_skip_cell(cell, identity) is False

    def test_manifest_and_residual_validation_refuse_identity_shape_and_nan_errors(self) -> None:
        identity = self._identity()
        validate_manifest(identity, identity)
        validate_residual(torch.zeros((2, 4), dtype=torch.float32), expected_shape=(2, 4))

        with pytest.raises(ValueError, match="target_layer"):
            validate_manifest({**identity, "target_layer": 29}, identity)
        with pytest.raises(ValueError, match="shape"):
            validate_residual(torch.zeros((2, 5), dtype=torch.float32), expected_shape=(2, 4))
        with pytest.raises(ValueError, match="finite"):
            validate_residual(
                torch.tensor([[0.0, 1.0, 2.0, math.nan], [0.0, 1.0, 2.0, 3.0]]),
                expected_shape=(2, 4),
            )


class TestPositionFinding:
    def test_returns_positions_for_requested_word_like_tokens(self) -> None:
        tokens = ("▁The", "▁mirror", "▁and", "▁mutual", "!", "▁mirror")

        assert find_word_positions(tokens, {"mirror", "mutual"}) == [1, 3, 5]

    def test_position_finding_ignores_subword_and_punctuation_tokens(self) -> None:
        tokens = ("▁cooperate", "##ing", "!", "▁cooperate")

        assert find_word_positions(tokens, {"cooperate"}) == [0, 3]
