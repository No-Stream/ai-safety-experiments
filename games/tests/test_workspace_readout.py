"""CPU contracts for the J-space workspace readout.

These tests deliberately exercise only the artifact and statistics seams.  They do not load a
model, fit a Jacobian lens, or inspect the private research corpus.  In particular, the paired
examples below are synthetic token scores: a readout that gets the ranking or denominator wrong
still produces plausible-looking JSON, so each expected ordering is written out here.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from transformers import AutoTokenizer

from games import workspace_readout
from games.interp_capture import render_stimuli
from games.interp_cells import Stimulus
from games.workspace_readout import (
    LoadedLens,
    StoredUnembed,
    _capture_prefix_resume_identity,
    _commitment_end,
    _cut_at_sentence_boundaries,
    _cut_prefix_position_data,
    _match_prefix_record_to_prompt,
    _parse_cut_fractions,
    _prefix_position_data,
    _select_prefix_vectors,
    accepted_adapter_base,
    analyse_trajectory,
    bootstrap_indices,
    decode_vocab,
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
    from argparse import Namespace
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


class TestPrefixCapture:
    def test_commitment_cut_stops_before_commitment_sentence(self) -> None:
        thinking = "I compare both options carefully. I will choose alpha."
        end = _commitment_end(thinking, ("alpha", "beta"))
        assert end is not None
        prefix = _cut_at_sentence_boundaries(thinking, 1.0, end)
        assert prefix == "I compare both options carefully. "
        assert "choose alpha" not in prefix
        assert end < len(thinking)

    def test_unmatched_record_refuses_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            workspace_readout, "generate_framing_prompt_rows", lambda *args, **kwargs: []
        )
        record = {
            "prompt_id": "missing-prompt",
            "counterpart_framing": "twin",
            "label_print_order": "canonical",
        }
        with pytest.raises(ValueError, match=r"prompt|match|regenerated"):
            _match_prefix_record_to_prompt(record)

    def test_prefix_positions_and_pooling_use_reasoning_tokens(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        stimulus = Stimulus(
            stimulus_id="synthetic-prefix",
            stimulus_set="capture-prefix",
            side="twin",
            pair_id="synthetic-prefix",
            text="Synthetic user text.",
            assistant_prefix="reasoning token " * 40,
        )
        rendered = render_stimuli(
            qwen_tokenizer,
            [stimulus],
            convention="templated_here",
            enable_thinking=True,
            include_assistant_prefix=True,
        )[stimulus.stimulus_id]
        metadata = _prefix_position_data(qwen_tokenizer, rendered)
        token_count = len(qwen_tokenizer(rendered, add_special_tokens=False)["input_ids"])
        assert metadata.prefix_token_count == len(metadata.reasoning_indices)
        assert metadata.token_indices["decision_point"] == token_count - 1
        assert metadata.reasoning_indices[-1] == token_count - 1
        assert len(metadata.reasoning_late_indices) == min(32, metadata.prefix_token_count)

        activation = torch.arange(token_count * 4, dtype=torch.float32).reshape(token_count, 4)
        selected = _select_prefix_vectors(activation, metadata)
        expected_reasoning = activation[list(metadata.reasoning_indices)].mean(dim=0)
        expected_late = activation[list(metadata.reasoning_late_indices)].mean(dim=0)
        assert torch.equal(selected[0], activation[-1])
        assert torch.equal(selected[1], expected_reasoning)
        assert torch.equal(selected[2], expected_late)

    def test_cut_positions_map_sentence_boundaries_to_real_token_offsets(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        prefix = "alpha. mirror beta. final."
        rendered = "Synthetic header: " + prefix
        cuts = _cut_prefix_position_data(
            qwen_tokenizer,
            rendered,
            prefix,
            (0.30, 0.80, 1.0),
            cue_regex=r"mirror",
        )
        assert cuts.cut_character_positions == {
            "cut_0.30": len("alpha. "),
            "cut_0.80": len("alpha. mirror beta. "),
            "cut_1.00": len(prefix),
        }
        assert cuts.cue_seen_by_cut == {
            "cut_0.30": False,
            "cut_0.80": True,
            "cut_1.00": True,
        }
        assert cuts.cue_first_fraction == pytest.approx(0.80)
        encoded = qwen_tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        prefix_start = len(rendered) - len(prefix)
        assert cuts.cut_character_positions is not None
        for name, token_index in cuts.token_indices.items():
            assert (
                encoded["offset_mapping"][token_index][1]
                <= prefix_start + cuts.cut_character_positions[name]
            )

    def test_empty_early_cut_reads_the_last_prompt_token(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """A cut with no sentence boundary below it is the state before any reasoning."""
        rendered = "Synthetic header: alpha. mirror beta."
        prefix = "alpha. mirror beta."
        cuts = _cut_prefix_position_data(
            qwen_tokenizer, rendered, prefix, (0.0,), cue_regex=r"mirror"
        )
        encoded = qwen_tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        prefix_start = len(rendered) - len(prefix)
        last_prompt_token = max(
            index
            for index, (_start, end) in enumerate(encoded["offset_mapping"])
            if end <= prefix_start
        )
        assert cuts.token_indices["cut_0.00"] == last_prompt_token
        assert cuts.cue_seen_by_cut == {"cut_0.00": False}

    def test_cut_fraction_and_cue_identity_preserve_legacy_absence(self) -> None:
        legacy = _capture_prefix_resume_identity(None, None)
        assert legacy == _capture_prefix_resume_identity(None, r"different")
        assert _capture_prefix_resume_identity((0.5, 1.0), None) != legacy
        assert _capture_prefix_resume_identity((0.5, 1.0), r"mirror") != (
            _capture_prefix_resume_identity((0.5, 1.0), r"symmetry")
        )

    def test_cut_fractions_accept_csv_and_repeatable_values(self) -> None:
        assert _parse_cut_fractions(["0.5, 1.0", "0.1"]) == (0.1, 0.5, 1.0)
        with pytest.raises(ValueError, match="duplicate"):
            _parse_cut_fractions(["0.5", "0.50"])

    def test_analyse_accepts_prefix_position_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        lens = LoadedLens(jacobians={0: torch.eye(2)}, n_prompts=1, d_model=2)
        unembed = StoredUnembed(
            norm_weight=torch.ones(2),
            lm_head_weight=torch.zeros((50, 2)),
            rms_eps=1e-6,
        )
        monkeypatch.setattr(workspace_readout, "load_lens", lambda path: lens)
        monkeypatch.setattr(workspace_readout, "load_unembed", lambda path: unembed)

        class SyntheticTokenizer:
            def __len__(self) -> int:
                return 50

            def decode(self, token_ids: int | list[int]) -> str:
                token_id = token_ids if isinstance(token_ids, int) else token_ids[0]
                return f"token{token_id}"

        monkeypatch.setattr(
            AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: SyntheticTokenizer(),
        )
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        (capture_dir / "run.json").write_text(
            json.dumps({"target_layer": 1, "arms": {"base": None, "trained": "adapter"}}),
            encoding="utf-8",
        )
        position_names = ["decision_point", "reasoning_mean", "reasoning_late"]
        manifest = {
            "stimulus_id": "synthetic-prefix",
            "side": "twin",
            "pair_id": "synthetic-prefix",
            "position_names": position_names,
        }
        for arm, offset in (("base", 0.0), ("trained", 1.0)):
            arm_dir = capture_dir / arm
            arm_dir.mkdir()
            torch.save(torch.full((1, 3, 1, 2), offset), arm_dir / "residuals.pt")
            (arm_dir / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        args = SimpleNamespace(
            capture_dir=capture_dir,
            lens=tmp_path / "lens.pt",
            model_path=tmp_path / "model",
            out_dir=tmp_path / "analysis",
            band="0:1",
            concept_sets=None,
            seed=1,
            device="cpu",
        )
        analyse = workspace_readout.analyse
        analyse(cast("Namespace", args))
        report = json.loads((args.out_dir / "analysis.json").read_text(encoding="utf-8"))
        assert all(
            any(key.endswith(f"|{position}") for key in report["top_tokens"])
            for position in position_names
        )
        assert all(
            any(key.endswith(f"|{position}") for key in report["paired_differences"])
            for position in position_names
        )


class TestAnalyseTrajectory:
    @staticmethod
    def _tokenizer() -> Any:
        class SyntheticTokenizer:
            def __len__(self) -> int:
                return 2

            def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
                del add_special_tokens
                return {"input_ids": [1] if text.strip().lower() == "focus" else [0]}

            def decode(self, token_ids: int | list[int]) -> str:
                token_id = token_ids if isinstance(token_ids, int) else token_ids[0]
                return f"token{token_id}"

        return SyntheticTokenizer()

    @staticmethod
    def _write_capture(
        capture_dir: Path,
        *,
        constant_offset: bool,
    ) -> None:
        cuts = ("cut_0.50", "cut_1.00")
        cue_flags = (
            (False, False),
            (False, False),
            (True, True),
            (True, True),
        )
        rows = [
            {
                "stimulus_id": f"stimulus-{index}",
                "side": "synthetic",
                "framing": "synthetic",
                "pair_id": f"pair-{index}",
                "position_names": list(cuts),
                "token_indices": {name: index for index, name in enumerate(cuts)},
                "cue_seen_by_cut": dict(zip(cuts, flags, strict=True)),
                "cue_first_fraction": 0.5 if flags[0] else None,
            }
            for index, flags in enumerate(cue_flags)
        ]
        (capture_dir / "run.json").write_text(
            json.dumps({"target_layer": 1, "arms": {"base": None, "trained": "adapter"}}),
            encoding="utf-8",
        )
        base = torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]] * len(rows))
        trained_rows = []
        for row in rows:
            values = []
            for cut_index, cut_name in enumerate(cuts):
                del cut_name
                cue_seen = cast("dict[str, bool]", row["cue_seen_by_cut"])
                if constant_offset or cue_seen[cuts[cut_index]]:
                    values.append([[0.0, 1.0]])
                else:
                    values.append([[1.0, 0.0]])
            trained_rows.append(values)
        trained = torch.tensor(trained_rows)
        for arm, residuals in (("base", base), ("trained", trained)):
            arm_dir = capture_dir / arm
            arm_dir.mkdir()
            torch.save(residuals, arm_dir / "residuals.pt")
            (arm_dir / "manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

    def _run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        constant_offset: bool,
    ) -> dict[str, Any]:
        capture_dir = tmp_path / ("constant" if constant_offset else "interaction")
        capture_dir.mkdir()
        self._write_capture(capture_dir, constant_offset=constant_offset)
        lens = LoadedLens(jacobians={0: torch.eye(2)}, n_prompts=1, d_model=2)
        unembed = StoredUnembed(
            norm_weight=torch.zeros(2),
            lm_head_weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            rms_eps=1e-6,
        )
        monkeypatch.setattr(workspace_readout, "load_lens", lambda path: lens)
        monkeypatch.setattr(workspace_readout, "load_unembed", lambda path: unembed)
        monkeypatch.setattr(
            AutoTokenizer,
            "from_pretrained",
            lambda *args, **kwargs: self._tokenizer(),
        )
        concepts = tmp_path / f"concepts-{constant_offset}.json"
        concepts.write_text(json.dumps({"focus": ["focus"]}), encoding="utf-8")
        args = SimpleNamespace(
            capture_dir=capture_dir,
            lens=tmp_path / "lens.pt",
            model_path=tmp_path / "model",
            out_dir=tmp_path / ("analysis-constant" if constant_offset else "analysis-interaction"),
            band="0:1",
            concept_sets=concepts,
            seed=7,
            device="cpu",
        )
        analyse_trajectory(cast("Namespace", args))
        return cast("dict[str, Any]", json.loads((args.out_dir / "trajectory.json").read_text()))

    def test_recovers_before_after_cue_interaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self._run(tmp_path, monkeypatch, constant_offset=False)
        entry = report["cue_split"]["trained"]["cut_0.50"]["focus"]
        assert entry["before_cue"]["mean"] == pytest.approx(0.0, abs=1e-5)
        assert entry["after_cue"]["mean"] > 0.9
        assert entry["after_minus_before"] > 0.9

    def test_constant_offset_has_no_cue_interaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        report = self._run(tmp_path, monkeypatch, constant_offset=True)
        entry = report["cue_split"]["trained"]["cut_0.50"]["focus"]
        assert entry["after_minus_before"] == pytest.approx(0.0, abs=1e-5)


class TestMultihopIntermediateVariants:
    def test_space_led_form_is_among_the_variants(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        space_led = qwen_tokenizer.encode(" Brazil", add_special_tokens=False)
        assert len(space_led) == 1
        assert space_led[0] in word_variant_token_ids(qwen_tokenizer, "Brazil")


class TestAdapterBaseAliases:
    def _adapter(self, tmp_path: Path, recorded: str) -> Path:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": recorded}), encoding="utf-8"
        )
        return adapter_dir

    def test_declared_alias_is_accepted(self, tmp_path: Path) -> None:
        adapter_dir = self._adapter(tmp_path, "hub-org/base-model")
        assert (
            accepted_adapter_base("arm", adapter_dir, "/local/snapshot", ["hub-org/base-model"])
            == "hub-org/base-model"
        )

    def test_undeclared_base_is_refused(self, tmp_path: Path) -> None:
        adapter_dir = self._adapter(tmp_path, "hub-org/other-model")
        with pytest.raises(ValueError, match="neither the loaded base"):
            accepted_adapter_base("arm", adapter_dir, "/local/snapshot", ["hub-org/base-model"])


class TestVocabularyLabels:
    def test_labels_cover_special_tokens_and_lm_head_padding(
        self, qwen_tokenizer: PreTrainedTokenizerBase
    ) -> None:
        n_rows = len(qwen_tokenizer) + 3
        labels = decode_vocab(qwen_tokenizer, n_rows)
        assert len(labels) == n_rows
        think_id = cast("int", qwen_tokenizer.convert_tokens_to_ids("<think>"))
        assert labels[think_id] == "<think>"
        assert labels[-1].startswith("<lm_head padding row")
        assert not is_word_like(labels[-1])
