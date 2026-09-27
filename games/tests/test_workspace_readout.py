"""CPU contracts for the J-space workspace readout.

These tests deliberately exercise only the artifact and statistics seams.  They do not load a
model, fit a Jacobian lens, or inspect the private research corpus.  In particular, the paired
examples below are synthetic token scores: a readout that gets the ranking or denominator wrong
still produces plausible-looking JSON, so each expected ordering is written out here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from transformers import AutoTokenizer

from games.workspace_readout import (
    bootstrap_indices,
    find_prompt_positions,
    is_word_like,
    parse_band,
    rank_token_shifts,
    should_skip_cell,
    single_token_concept_ids,
    validate_manifest,
    validate_residual,
    word_variant_token_ids,
)

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


@pytest.fixture(scope="module")
def qwen_tokenizer() -> PreTrainedTokenizerBase:
    """The real Qwen3.5 tokenizer; the cache is part of this box's setup, so absence fails loudly."""
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", local_files_only=True)


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
    @pytest.mark.parametrize("token", ["alphaish", "Ġalphaish", "▁betaword", "abc"])
    def test_keeps_word_like_tokens(self, token: str) -> None:
        assert is_word_like(token)

    @pytest.mark.parametrize("token", ["", "▁", "!", "42", "##ing", "<0x0A>", "<|endoftext|>"])
    def test_drops_punctuation_continuations_and_special_tokens(self, token: str) -> None:
        assert not is_word_like(token)


class TestTokenShiftRanking:
    def test_ranks_by_signed_shift_and_never_ranks_filtered_tokens(self) -> None:
        vocab = ["alphaish", "betaword", "gammaish", "filtered"]
        mean_delta = torch.tensor([0.5, -0.2, 0.1, float("nan")])
        fraction_positive = torch.tensor([0.9, 0.1, 0.6, 1.0])
        risen = rank_token_shifts(mean_delta, fraction_positive, vocab, sign=1.0, top_k=4)
        fallen = rank_token_shifts(mean_delta, fraction_positive, vocab, sign=-1.0, top_k=4)
        assert [row["token"] for row in risen] == ["alphaish", "gammaish"]
        assert [row["token"] for row in fallen] == ["betaword"]
        assert fallen[0]["delta"] == pytest.approx(-0.2)
        assert risen[0]["fraction_positive"] == pytest.approx(0.9)


class TestConceptMapping:
    def test_single_token_mapping_records_multi_token_words(self) -> None:
        class SyntheticTokenizer:
            def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
                del add_special_tokens
                return {
                    "input_ids": {
                        "alphaish": [1],
                        " alphaish": [2],
                        "Alphaish": [3],
                        " Alphaish": [4],
                        "multiword": [5, 6],
                    }.get(text, [7, 8])
                }

        ids, skipped = single_token_concept_ids(
            cast("Any", SyntheticTokenizer()), {"set": ("alphaish", "multiword")}
        )

        assert ids == {"set": (1, 2, 3, 4)}
        assert skipped == {"set": ("multiword",)}


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
    def test_real_cached_qwen_tokenizer_positions(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        tokenizer = qwen_tokenizer
        rendered = cast(
            "str",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "Synthetic user text."}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
        )
        positions = find_prompt_positions(tokenizer, rendered)
        assert set(positions) == {"user_end", "assistant_marker", "think_open"}
        assert positions["user_end"] < positions["assistant_marker"] < positions["think_open"]


class TestMultihopIntermediateVariants:
    def test_space_led_form_is_among_the_variants(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        space_led = qwen_tokenizer.encode(" Brazil", add_special_tokens=False)
        assert len(space_led) == 1
        assert space_led[0] in word_variant_token_ids(qwen_tokenizer, "Brazil")
