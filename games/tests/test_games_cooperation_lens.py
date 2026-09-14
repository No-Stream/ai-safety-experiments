"""CPU contracts for the bounded base/final cooperation Jacobian read."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from games.cooperation_lens import (
    GRADIENT_GATE_AUTOGRAD,
    LensRunIdentity,
    build_corpus_plan,
    decode_directions,
    model_specific_metadata,
    prepare_accumulation_resume,
    shared_coordinate_payload,
    validate_base_model_revision,
    validate_gradient_gate,
    validate_gradient_gate_binding,
    validate_private_stimulus_path,
)
from games.interp_cells import Stimulus
from games.tokenizer_identity import tokenizer_content_sha256


def stimuli(prefix: str, n_prompts: int, *, text_size: int = 24) -> list[Stimulus]:
    """Build complete two-sided scenario groups."""
    assert n_prompts % 2 == 0
    return [
        Stimulus(
            stimulus_id=f"{prefix}-{pair}-{side}",
            stimulus_set="costly-other-regard" if pair % 2 == 0 else "decision-dependence",
            side=side,
            pair_id=f"{prefix}-scenario-{pair}",
            text=("shared context " + "x" * text_size + side),
            assistant_prefix="I should compare both consequences before committing.",
            metadata={"scenario_group": f"{prefix}-group-{pair}"},
        )
        for pair in range(n_prompts // 2)
        for side in ("A", "B")
    ]


def rendered(rows: list[Stimulus]) -> dict[str, str]:
    return {row.stimulus_id: row.text for row in rows}


def token_ids(rows: list[Stimulus]) -> dict[str, list[int]]:
    return {row.stimulus_id: list(row.text.encode()) for row in rows}


class TestCorpusPlan:
    def test_measurement_is_fifty_fit_and_eight_disjoint_quality_prompts(self) -> None:
        fit = stimuli("fit", 50)
        quality = stimuli("quality", 8)
        plan = build_corpus_plan(
            fit,
            quality,
            rendered_fit=rendered(fit),
            rendered_quality=rendered(quality),
            fit_token_ids=token_ids(fit),
            quality_token_ids=token_ids(quality),
            profile="measurement",
            max_seq_len_ceiling=256,
        )
        assert len(plan.fit_prompts) == 50
        assert len(plan.quality_prompts) == 8
        assert plan.profile == "measurement"
        assert plan.fit_prompt_sha256 != plan.quality_prompt_sha256
        assert plan.n_truncated == 0


class TestBaseRevisionContract:
    def test_hub_model_without_revision_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"required.*hub model ID"):
            validate_base_model_revision("Qwen/Qwen3.5-9B", None)

    def test_local_snapshot_with_revision_is_refused(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        with pytest.raises(ValueError, match=r"must be omitted.*local snapshot"):
            validate_base_model_revision(str(snapshot), "commit")

    def test_local_snapshot_without_revision_passes(self, tmp_path: Path) -> None:
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        validate_base_model_revision(str(snapshot), None)

    def test_smoke_and_measurement_have_distinct_identities(self) -> None:
        quality = stimuli("quality", 8)
        smoke = stimuli("smoke", 10)
        measurement = stimuli("measurement", 50)
        smoke_plan = build_corpus_plan(
            smoke,
            quality,
            rendered_fit=rendered(smoke),
            rendered_quality=rendered(quality),
            fit_token_ids=token_ids(smoke),
            quality_token_ids=token_ids(quality),
            profile="smoke",
            max_seq_len_ceiling=256,
        )
        measurement_plan = build_corpus_plan(
            measurement,
            quality,
            rendered_fit=rendered(measurement),
            rendered_quality=rendered(quality),
            fit_token_ids=token_ids(measurement),
            quality_token_ids=token_ids(quality),
            profile="measurement",
            max_seq_len_ceiling=256,
        )
        assert smoke_plan.identity_sha256 != measurement_plan.identity_sha256

    def test_fit_and_quality_scenario_groups_cannot_overlap(self) -> None:
        fit = stimuli("fit", 50)
        quality = stimuli("quality", 8)
        quality[0] = replace(quality[0], metadata=fit[0].metadata)
        quality[1] = replace(quality[1], metadata=fit[0].metadata)
        with pytest.raises(ValueError, match="fit and quality scenario groups overlap"):
            build_corpus_plan(
                fit,
                quality,
                rendered_fit=rendered(fit),
                rendered_quality=rendered(quality),
                fit_token_ids=token_ids(fit),
                quality_token_ids=token_ids(quality),
                profile="measurement",
                max_seq_len_ceiling=256,
            )

    def test_fit_and_quality_groups_cannot_overlap_training_or_evaluation(self) -> None:
        fit = stimuli("fit", 50)
        quality = stimuli("quality", 8)
        with pytest.raises(ValueError, match="overlap reserved training/evaluation groups"):
            build_corpus_plan(
                fit,
                quality,
                rendered_fit=rendered(fit),
                rendered_quality=rendered(quality),
                fit_token_ids=token_ids(fit),
                quality_token_ids=token_ids(quality),
                profile="measurement",
                max_seq_len_ceiling=256,
                reserved_group_ids={str(fit[0].metadata["scenario_group"])},
            )

    def test_a_cropped_contrast_is_refused(self) -> None:
        """Sabotage: a ceiling before the differing suffix must make this guard red."""
        fit = stimuli("fit", 50, text_size=60)
        quality = stimuli("quality", 8, text_size=60)
        with pytest.raises(ValueError, match="would crop contrast-bearing prompts"):
            build_corpus_plan(
                fit,
                quality,
                rendered_fit=rendered(fit),
                rendered_quality=rendered(quality),
                fit_token_ids=token_ids(fit),
                quality_token_ids=token_ids(quality),
                profile="measurement",
                max_seq_len_ceiling=20,
            )

    def test_tail_truncation_after_each_contrast_is_recorded_but_allowed(self) -> None:
        fit = [replace(row, text=row.side + "z" * 100) for row in stimuli("fit", 50)]
        quality = [replace(row, text=row.side + "z" * 100) for row in stimuli("quality", 8)]
        plan = build_corpus_plan(
            fit,
            quality,
            rendered_fit=rendered(fit),
            rendered_quality=rendered(quality),
            fit_token_ids=token_ids(fit),
            quality_token_ids=token_ids(quality),
            profile="measurement",
            max_seq_len_ceiling=20,
        )
        assert plan.n_truncated == 58


def test_authored_lens_text_is_restricted_to_private_roots(tmp_path: Path) -> None:
    assert "docs/scratch" in str(
        validate_private_stimulus_path(Path("docs/scratch/cooperation-generalization/lens.jsonl"))
    )
    with pytest.raises(ValueError, match="must live under"):
        validate_private_stimulus_path(tmp_path / "tracked-looking.jsonl")


def identity() -> LensRunIdentity:
    return LensRunIdentity(
        profile="measurement",
        state="base",
        lens_cache_key_sha256="lens-key",
        fit_prompt_sha256="fit-key",
        quality_prompt_sha256="quality-key",
        max_seq_len=128,
        dim_batch=8,
        checkpoint_every=5,
    )


class TestAccumulationResume:
    def test_fresh_then_exact_resume(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "accumulator.pt"
        assert prepare_accumulation_resume(checkpoint, identity()) is False
        checkpoint.write_bytes(b"fp32 accumulation")
        assert prepare_accumulation_resume(checkpoint, identity()) is True

    def test_refuses_a_checkpoint_from_another_corpus_or_config(self, tmp_path: Path) -> None:
        checkpoint = tmp_path / "accumulator.pt"
        assert prepare_accumulation_resume(checkpoint, identity()) is False
        checkpoint.write_bytes(b"fp32 accumulation")
        with pytest.raises(ValueError, match="accumulation identity mismatch"):
            prepare_accumulation_resume(
                checkpoint, replace(identity(), fit_prompt_sha256="other-corpus")
            )


class FingerprintTokenizer:
    def __init__(self, *, vocabulary: dict[str, int], padding_side: str) -> None:
        self.vocabulary = vocabulary
        self.padding_side = padding_side
        self.truncation_side = "right"
        self.model_max_length = 128
        self.clean_up_tokenization_spaces = False
        self.split_special_tokens = False
        self.chat_template = "{{ messages }}"
        self.special_tokens_map = {"eos_token": "<eos>"}
        self.init_kwargs = {"legacy": False, "_commit_hash": "source-commit"}

    def get_vocab(self) -> dict[str, int]:
        return self.vocabulary

    def get_added_vocab(self) -> dict[str, int]:
        return {}


def gate_payload(*, live: bool = True, sabotage_zero: bool = True) -> dict[str, Any]:
    return {
        "passed": live and sabotage_zero,
        "jlens": {"commit": "581d398abc"},
        "model": {
            "weights_identity": "hf:base-commit",
            "resolved_weights_identity": "hf:base-commit",
            "tokenizer_content_sha256": "tokenizer-content",
        },
        "max_seq_len": 128,
        "cooperation_fit_corpus": {
            "fit_stimuli_sha256": "fit-stimuli",
            "fit_rendered_sha256": "fit-rendered",
            "stimulus_render": "templated_here",
            "enable_thinking": True,
            "n_fit_prompts": 10,
            "tokenizer_content_sha256": "tokenizer-content",
        },
        "deltanet_kernels_bound": {"chunk": "fused"},
        "gates_requested": ["autograd", "kernel", "resume"],
        "gates": {
            "autograd": {
                "passed": live and sabotage_zero,
                "verdicts": {
                    "autograd_traverses_recurrence": live,
                    "sabotage_reads_exactly_zero": sabotage_zero,
                },
            },
            "kernel": {"passed": True},
            "resume": {"passed": True},
            "sweep": {"passed": True, "choice": {"chosen": 8}},
            "tokens": {"passed": True, "reference_identity": "hf:base-commit"},
        },
    }


class TestGradientGate:
    def test_accepts_live_gradient_and_effective_detach_sabotage(self) -> None:
        validated = validate_gradient_gate(gate_payload())
        assert validated["gates"][GRADIENT_GATE_AUTOGRAD]["passed"] is True

    @pytest.mark.parametrize(
        ("payload", "message"),
        [
            (gate_payload(live=False), "zero or disconnected"),
            (gate_payload(sabotage_zero=False), "detach sabotage did not read exactly zero"),
        ],
    )
    def test_zero_or_disconnected_gradient_and_toothless_sabotage_are_refused(
        self, payload: dict[str, Any], message: str
    ) -> None:
        """Sabotage: the same report with its live path zeroed must make this guard red."""
        with pytest.raises(ValueError, match=message):
            validate_gradient_gate(payload)

    def test_external_gate_binds_to_model_tokenizer_kernels_and_fit_settings(self) -> None:
        validated = validate_gradient_gate_binding(
            gate_payload(),
            base_weights_identity="hf:base-commit",
            tokenizer_content_identity="tokenizer-content",
            comparison_reference_identity="hf:base-commit",
            fit_stimuli_sha256="fit-stimuli",
            fit_rendered_sha256="fit-rendered",
            stimulus_render="templated_here",
            enable_thinking=True,
            max_seq_len=128,
            dim_batch=8,
            kernels={"chunk": "fused"},
        )
        assert validated["model"]["weights_identity"] == "hf:base-commit"

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            (
                "model",
                {
                    "weights_identity": "hf:other",
                    "resolved_weights_identity": "hf:other",
                },
                "base weights",
            ),
            ("max_seq_len", 64, "max_seq_len"),
            ("deltanet_kernels_bound", {"chunk": "reference"}, "kernel bindings"),
        ],
    )
    def test_external_gate_binding_refuses_stale_evidence(
        self, field: str, value: object, message: str
    ) -> None:
        payload = gate_payload()
        payload[field] = value
        with pytest.raises(ValueError, match=message):
            validate_gradient_gate_binding(
                payload,
                base_weights_identity="hf:base-commit",
                tokenizer_content_identity="tokenizer-content",
                comparison_reference_identity="hf:base-commit",
                fit_stimuli_sha256="fit-stimuli",
                fit_rendered_sha256="fit-rendered",
                stimulus_render="templated_here",
                enable_thinking=True,
                max_seq_len=128,
                dim_batch=8,
                kernels={"chunk": "fused"},
            )

    def test_external_gate_binding_refuses_different_loaded_tokenizer_content(self) -> None:
        """Sabotage: a changed vocabulary/config hash cannot inherit a prior model gate report."""
        payload = gate_payload()
        loaded_tokenizer = FingerprintTokenizer(vocabulary={"same": 0}, padding_side="right")
        gate_tokenizer = FingerprintTokenizer(vocabulary={"changed": 0}, padding_side="left")
        payload["model"]["tokenizer_content_sha256"] = tokenizer_content_sha256(gate_tokenizer)

        with pytest.raises(ValueError, match="tokenizer content fingerprint"):
            validate_gradient_gate_binding(
                payload,
                base_weights_identity="hf:base-commit",
                tokenizer_content_identity=tokenizer_content_sha256(loaded_tokenizer),
                comparison_reference_identity="hf:base-commit",
                fit_stimuli_sha256="fit-stimuli",
                fit_rendered_sha256="fit-rendered",
                stimulus_render="templated_here",
                enable_thinking=True,
                max_seq_len=128,
                dim_batch=8,
                kernels={"chunk": "fused"},
            )

    def test_token_comparison_commit_identity_remains_a_separate_binding(self) -> None:
        payload = gate_payload()
        payload["gates"]["tokens"]["reference_identity"] = "hf:other-commit"
        with pytest.raises(ValueError, match="base tokenizer snapshot"):
            validate_gradient_gate_binding(
                payload,
                base_weights_identity="hf:base-commit",
                tokenizer_content_identity="tokenizer-content",
                comparison_reference_identity="hf:base-commit",
                fit_stimuli_sha256="fit-stimuli",
                fit_rendered_sha256="fit-rendered",
                stimulus_render="templated_here",
                enable_thinking=True,
                max_seq_len=128,
                dim_batch=8,
                kernels={"chunk": "fused"},
            )

    @pytest.mark.parametrize(
        ("field", "stale_value"),
        [
            ("fit_stimuli_sha256", "stale-stimuli"),
            ("fit_rendered_sha256", "stale-render"),
            ("stimulus_render", "verbatim"),
            ("enable_thinking", False),
            ("tokenizer_content_sha256", "stale-tokenizer"),
        ],
    )
    def test_external_gate_binding_refuses_stale_fit_corpus_or_render(
        self, field: str, stale_value: object
    ) -> None:
        payload = gate_payload()
        payload["cooperation_fit_corpus"][field] = stale_value
        with pytest.raises(ValueError, match="cooperation fit corpus/render binding"):
            validate_gradient_gate_binding(
                payload,
                base_weights_identity="hf:base-commit",
                tokenizer_content_identity="tokenizer-content",
                comparison_reference_identity="hf:base-commit",
                fit_stimuli_sha256="fit-stimuli",
                fit_rendered_sha256="fit-rendered",
                stimulus_render="templated_here",
                enable_thinking=True,
                max_seq_len=128,
                dim_batch=8,
                kernels={"chunk": "fused"},
            )


class StubLens:
    def transport(self, direction: torch.Tensor, layer: int) -> torch.Tensor:
        return direction * (layer + 1)


class StubModel:
    def unembed(self, direction: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]) @ direction


class StubTokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token-{token_id}"


def test_decode_carries_quality_beside_every_real_and_matched_norm_control(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name, vector in {
        "costly-other-regard": torch.tensor([1.0, 0.0]),
        "decision-dependence": torch.tensor([0.0, 2.0]),
        "trained-displacement": torch.tensor([1.0, 1.0]),
    }.items():
        path = tmp_path / f"{name}.pt"
        torch.save({0: vector}, path)
        paths[name] = path
    quality = {
        "jacobian": {"per_layer": [{"layer": 0, "relative_residual": 0.2}]},
        "logit_lens_baseline": {"per_layer": [{"layer": 0, "relative_residual": 0.7}]},
    }
    reads = decode_directions(
        StubLens(), StubModel(), StubTokenizer(), paths, quality=quality, top_k=2, seed=7
    )
    assert set(reads) == set(paths)
    for read in reads.values():
        assert read["layers"]["0"]["quality"] == {
            "jacobian_relative_residual": 0.2,
            "logit_lens_relative_residual": 0.7,
        }
        assert read["layers"]["0"]["real"][0]["token"].startswith("token-")
        assert read["layers"]["0"]["matched_norm_random"]
        assert read["layers"]["0"]["direction_norm"] == pytest.approx(
            read["layers"]["0"]["random_norm"]
        )


def test_decode_refuses_unavailable_quality_or_a_missing_logit_baseline(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name in ("costly-other-regard", "decision-dependence", "trained-displacement"):
        path = tmp_path / f"{name}.pt"
        torch.save({0: torch.ones(2)}, path)
        paths[name] = path
    with pytest.raises(ValueError, match="matching per-layer"):
        decode_directions(
            StubLens(),
            StubModel(),
            StubTokenizer(),
            paths,
            quality={"available": False},
            top_k=2,
            seed=7,
        )
    with pytest.raises(ValueError, match="matching per-layer"):
        decode_directions(
            StubLens(),
            StubModel(),
            StubTokenizer(),
            paths,
            quality={"jacobian": {"per_layer": [{"layer": 0, "relative_residual": 0.2}]}},
            top_k=2,
            seed=7,
        )


def test_shared_coordinate_payload_is_explicitly_approximate() -> None:
    assert model_specific_metadata("final") == {
        "readout_kind": "model-specific",
        "approximation": False,
        "lens_state": "final",
        "applied_to_state": "final",
    }
    payload = shared_coordinate_payload({"available": True}, {"axis": {}})
    assert payload["readout_kind"] == "shared-base-coordinate-sensitivity"
    assert payload["approximation"] is True
    assert payload["lens_state"] == "base"
    assert payload["applied_to_state"] == "final"
