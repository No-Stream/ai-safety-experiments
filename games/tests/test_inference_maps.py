"""CPU-safe tests for the inference-only map helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from games.argument_prior_map import (
    _assert_adapter_shapes_match_model,
    _assert_vllm_adapter_preflight,
    _build_smoke_adapters,
    _commitment_end,
    _cut_at_sentence_boundaries,
    _summarize_openers,
    donor_cooperation_rates,
    load_openers,
)
from games.inference_utils import adapter_digest, append_jsonl, content_key, load_completed_keys
from games.kl_footprint import derive_kl_row_group_size, matched_norm_random_lora

if TYPE_CHECKING:
    from pathlib import Path


def test_content_key_and_jsonl_resume_are_content_stable(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    key = content_key("unit", {"b": 2, "a": 1})
    assert key == content_key("unit", {"a": 1, "b": 2})
    append_jsonl(path, {"key": key, "value": 3})
    assert load_completed_keys(path) == {key}


def test_jsonl_resume_rejects_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"value": 1}) + "\n", encoding="utf-8")
    with pytest.raises(TypeError, match="content-derived key"):
        load_completed_keys(path)


def test_argument_cut_points_stay_before_commitment() -> None:
    thinking = "First compare the options. Then inspect the evidence. I will choose NORTH."
    end = _commitment_end(thinking, ("NORTH", "SOUTH"))
    assert end is not None
    assert thinking[end:].lstrip().startswith("I will choose")
    assert _cut_at_sentence_boundaries(thinking, 0.0, end) == ""
    assert _cut_at_sentence_boundaries(thinking, 1.0, end) == thinking[:end]
    third = _cut_at_sentence_boundaries(thinking, 1 / 3, end)
    assert third == "" or third.endswith(".")


def test_argument_summary_has_neutral_contrasts() -> None:
    rows = [
        {
            "model_condition": condition,
            "game_id": "synthetic-game",
            "framing_id": "synthetic-frame",
            "opener_category": category,
            "logprob_mean": value,
        }
        for condition, offset in (("base", 0.0), ("self", 1.0))
        for category, value in (("mirror", 3.0 + offset), ("dominance", 1.0), ("neutral", 2.0))
    ]
    summary = _summarize_openers(rows)
    self_row = next(row for row in summary["contexts"] if row["model_condition"] == "self")
    assert self_row["mirror_minus_neutral"] == pytest.approx(2.0)
    assert self_row["dominance_minus_neutral"] == pytest.approx(-1.0)
    assert summary["adapter_shifts"][0]["adapter_shifts"]["self"] == pytest.approx(1.0)


def test_private_opener_file_has_expected_categories(tmp_path: Path) -> None:
    path = tmp_path / "openers.json"
    path.write_text(
        json.dumps(
            {
                "openers": [
                    {"id": "mirror", "category": "mirror", "text": "synthetic mirror"},
                    {"id": "dominance", "category": "dominance", "text": "synthetic dominance"},
                    {"id": "neutral", "category": "neutral", "text": "synthetic neutral"},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert [opener.category for opener in load_openers(path)] == ["mirror", "dominance", "neutral"]


class TinyLoRAContainer(nn.Module):
    """Minimal PEFT-shaped module for the matched-norm placebo test."""

    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 3, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(3, 4, bias=False)})


class TinySmokeModel(nn.Module):
    """Small nested linear model for smoke adapter creation and shape validation."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    def prepare_inputs_for_generation(self, **kwargs: object) -> dict[str, object]:
        return kwargs


def _write_adapter_config(path: Path, *, base: str, target: str = "linear") -> None:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": base,
                "r": 2,
                "target_modules": [target],
            }
        ),
        encoding="utf-8",
    )


def test_vllm_adapter_preflight_rejects_base_mismatch_before_engine(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter_config(adapter, base="Qwen/Qwen3.5-9B")
    with pytest.raises(ValueError, match=r"trained against.*Qwen/Qwen3\.5-0\.8B"):
        _assert_vllm_adapter_preflight(adapter, "Qwen/Qwen3.5-0.8B")


def test_adapter_shape_check_reports_wrong_base_module_shape(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    _write_adapter_config(adapter, base="Qwen/Qwen3.5-0.8B")
    save_file(
        {
            "base_model.model.linear.lora_A.weight": torch.zeros(2, 99),
            "base_model.model.linear.lora_B.weight": torch.zeros(4, 2),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match=r"expected \(2, 3\)"):
        _assert_adapter_shapes_match_model(adapter, TinySmokeModel())


def test_smoke_adapters_are_peft_files_for_the_selected_base(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_adapter_config(source, base="Qwen/Qwen3.5-9B")
    base = TinySmokeModel()
    rebuilt, paths = _build_smoke_adapters(
        cast("Any", base),
        source_adapters=(source, source),
        output_root=tmp_path / "generated",
        model_id="Qwen/Qwen3.5-0.8B",
    )
    assert rebuilt is base
    assert paths[0].is_dir()
    assert paths[1].is_dir()
    for path in paths:
        config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
        assert config["base_model_name_or_path"] == "Qwen/Qwen3.5-0.8B"
        assert config["target_modules"] == ["linear"]
        _assert_adapter_shapes_match_model(path, rebuilt)


def test_random_lora_placebo_matches_each_module_delta_and_restores() -> None:
    module = TinyLoRAContainer()
    a_weight = cast("torch.Tensor", module.lora_A["default"].weight)
    b_weight = cast("torch.Tensor", module.lora_B["default"].weight)
    original_a = a_weight.detach().clone()
    original_b = b_weight.detach().clone()
    real_a_norm = torch.linalg.vector_norm(original_a.float())
    real_b_norm = torch.linalg.vector_norm(original_b.float())
    with matched_norm_random_lora(module, seed=11):
        random_a_norm = torch.linalg.vector_norm(
            cast("torch.Tensor", module.lora_A["default"].weight).float()
        )
        random_b_norm = torch.linalg.vector_norm(
            cast("torch.Tensor", module.lora_B["default"].weight).float()
        )
        assert float(random_a_norm.detach()) == pytest.approx(float(real_a_norm), rel=1e-5)
        assert float(random_b_norm.detach()) == pytest.approx(float(real_b_norm), rel=1e-5)
        assert not torch.equal(cast("torch.Tensor", module.lora_A["default"].weight), original_a)
    assert torch.equal(cast("torch.Tensor", module.lora_A["default"].weight), original_a)
    assert torch.equal(cast("torch.Tensor", module.lora_B["default"].weight), original_b)


def test_kl_row_group_size_uses_both_model_chunk_logits(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_row_bytes = 2 * 100 * 10 * torch.empty((), dtype=torch.bfloat16).element_size()
    workspace_bytes = raw_row_bytes * 8
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (workspace_bytes * 2, 0))

    rows = derive_kl_row_group_size(
        torch.device("cuda"),
        vocabulary_size=100,
        position_chunk_size=10,
        logit_dtype=torch.bfloat16,
        cap=10,
    )

    assert rows == 2


def _write_adapter(directory: Path, *, target_modules: list[str], weight: float) -> Path:
    directory.mkdir(parents=True)
    config = {"base_model_name_or_path": "base", "r": 4, "target_modules": target_modules}
    (directory / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file({"lora_A": torch.full((2, 2), weight)}, str(directory / "adapter_model.safetensors"))
    (directory / "README.md").write_text(f"card {weight} {target_modules}", encoding="utf-8")
    return directory


def test_adapter_digest_ignores_peft_set_order_but_not_weights(tmp_path: Path) -> None:
    """PEFT saves target_modules from a set, so its order follows the per-process hash seed; a
    digest over raw bytes changed on every smoke relaunch and resumed nothing (observed 2026-09-25)."""
    first = _write_adapter(tmp_path / "a", target_modules=["q_proj", "out_proj"], weight=1.0)
    reordered = _write_adapter(tmp_path / "b", target_modules=["out_proj", "q_proj"], weight=1.0)
    reweighted = _write_adapter(tmp_path / "c", target_modules=["q_proj", "out_proj"], weight=2.0)
    assert adapter_digest(first) == adapter_digest(reordered)
    assert adapter_digest(first) != adapter_digest(reweighted)


def test_donor_cooperation_counts_the_canonical_cooperate_action() -> None:
    """parse_action returns "C"/"D"; comparing against "cooperate" once scored every donor
    continuation as a defection (observed on the 9B first cut, 2026-09-25)."""
    records = [
        {
            "donor_framing": "twin",
            "cut_fraction": 0.0,
            "model_condition": "base",
            "parsed_action": action,
        }
        for action in ("C", "C", "D", None)
    ]
    rates = donor_cooperation_rates(records)
    assert rates["twin|cut-0|base"] == {"records": 4, "parsed": 3, "cooperate": 2, "rate": 2 / 3}
