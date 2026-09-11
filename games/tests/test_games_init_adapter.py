"""The --init-adapter seam: continue-RL from a checkpoint's LoRA under a new arm.

CPU-only, no downloads. The loader's whole job is refusing quiet failure modes, and every refusal
here is exercised on the exact sabotage it exists to catch: an adapter directory that is not one, a
donor that does not tile the recipient's modules, an identity drift (rank, targets, rsLoRA), and
the one that motivated the read-back design -- a donor whose B factors are all zero, which seeds a
model bit-identical to a fresh LoRA while the run's "from checkpoint" label reads as true.

The tiny models mirror `test_games_lora.py`: names are what PEFT matches on, so 8x8 tensors carry
the whole trap.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn

from games import train as gt

if TYPE_CHECKING:
    from pathlib import Path

WIDTH = 8
ADAPTED_MODULES = ("q_proj", "o_proj")
BASE_MODEL = "Qwen/Qwen3.5-2B"
# One (A, B) pair per adapted module.
EXPECTED_WEIGHT_COUNT = 2 * len(ADAPTED_MODULES)


class TinyModel(nn.Module):
    """Two adapted projections; enough structure for PEFT's name matching to be load-bearing."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(WIDTH, WIDTH, bias=False)
        self.o_proj = nn.Linear(WIDTH, WIDTH, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(self.q_proj(x))


def tiny_peft_model(modules: tuple[str, ...] = ADAPTED_MODULES) -> Any:
    return get_peft_model(cast("Any", TinyModel()), LoraConfig(r=2, target_modules=list(modules)))


def randomize_lora(model: Any) -> None:
    """Give every LoRA factor a nonzero value, the state a trained checkpoint is in."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn_like(parameter))


def save_adapter(model: Any, root: Path) -> Path:
    model.save_pretrained(str(root))
    return root


def make_config(**overrides: object) -> gt.GameTrainConfig:
    base: dict[str, object] = {"arm": "public-goods-self", "generate_fresh": True}
    if "model_id" in overrides:
        # An explicit model must clear its own measured completion floor, exactly as a launch
        # does -- and at that budget the importance-sampling correction refuses for memory, so the
        # config carries the same correction-OFF setting every real launch at this budget carries.
        base["max_completion_tokens"] = gt.required_completion_budget(
            cast("str", overrides["model_id"])
        )
        base["vllm_importance_sampling_correction"] = False
    return gt.GameTrainConfig(**(base | overrides))  # pyright: ignore[reportArgumentType]


def write_adapter_config_dir(root: Path, **overrides: object) -> Path:
    """Lay down an adapter dir whose config `describe_init_adapter` reads; weights are a stub."""
    config: dict[str, object] = {
        "peft_type": "LORA",
        "base_model_name_or_path": BASE_MODEL,
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "o_proj"],
        "use_dora": False,
        "use_rslora": False,
    }
    config.update(overrides)
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(json.dumps(config))
    save_file({"stub": torch.zeros(1)}, str(root / gt.INIT_ADAPTER_WEIGHTS_FILENAME))
    return root


def run_lora_targets() -> dict[str, object]:
    return {"target_modules": ["q_proj", "o_proj"]}


class TestConfigValidation:
    def test_the_default_is_no_init_adapter(self) -> None:
        assert make_config().init_adapter == ""

    def test_a_path_that_is_not_an_adapter_dir_is_refused_at_config_time(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="not a PEFT adapter checkpoint"):
            make_config(init_adapter=str(tmp_path / "nope"))

    def test_a_dir_missing_the_weights_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("{}")
        with pytest.raises(ValueError, match=gt.INIT_ADAPTER_WEIGHTS_FILENAME):
            make_config(init_adapter=str(tmp_path))

    def test_a_real_adapter_dir_is_accepted(self, tmp_path: Path) -> None:
        adapter = save_adapter(tiny_peft_model(), tmp_path / "checkpoint-70")
        assert make_config(init_adapter=str(adapter)).init_adapter == str(adapter)


class TestTheCliFlag:
    def test_the_flag_reaches_the_config(self, tmp_path: Path) -> None:
        adapter = save_adapter(tiny_peft_model(), tmp_path / "checkpoint-70")
        config = gt._parse_args(
            [
                "--arm",
                "public-goods-self",
                "--generate-fresh",
                "--output-dir",
                str(tmp_path / "run"),
                "--init-adapter",
                str(adapter),
            ]
        )
        assert config.init_adapter == str(adapter)

    def test_omitting_the_flag_means_the_usual_zero_init(self, tmp_path: Path) -> None:
        config = gt._parse_args(
            ["--arm", "public-goods-self", "--generate-fresh", "--output-dir", str(tmp_path)]
        )
        assert config.init_adapter == ""


class TestApplyInitAdapter:
    def test_a_trained_donor_seeds_a_fresh_recipient_exactly(self, tmp_path: Path) -> None:
        donor = tiny_peft_model()
        randomize_lora(donor)
        adapter = save_adapter(donor, tmp_path / "checkpoint-70")
        recipient = tiny_peft_model()

        facts = gt.apply_init_adapter(recipient, str(adapter))

        saved = load_file(str(adapter / gt.INIT_ADAPTER_WEIGHTS_FILENAME))
        assert facts["applied_weights"] == EXPECTED_WEIGHT_COUNT == len(saved)
        assert cast("float", facts["lora_B_sq_norm"]) > 0.0
        recipient_lora = {
            name: parameter for name, parameter in recipient.named_parameters() if "lora_" in name
        }
        assert len(recipient_lora) == EXPECTED_WEIGHT_COUNT
        donor_lora = [p for n, p in donor.named_parameters() if "lora_" in n]
        for donor_parameter, recipient_parameter in zip(
            donor_lora, recipient_lora.values(), strict=True
        ):
            assert torch.equal(donor_parameter, recipient_parameter)

    def test_an_all_zero_b_donor_is_refused_as_indistinguishable_from_no_init(
        self, tmp_path: Path
    ) -> None:
        # A fresh get_peft_model IS this sabotage: PEFT zero-initialises B, so applying it would
        # leave the recipient bit-identical to an unseeded run under a "from checkpoint" label.
        adapter = save_adapter(tiny_peft_model(), tmp_path / "checkpoint-0")
        recipient = tiny_peft_model()
        with pytest.raises(ValueError, match="all-zero B"):
            gt.apply_init_adapter(recipient, str(adapter))

    def test_a_donor_that_does_not_tile_the_recipient_is_refused(self, tmp_path: Path) -> None:
        donor = tiny_peft_model(modules=("q_proj",))
        randomize_lora(donor)
        adapter = save_adapter(donor, tmp_path / "checkpoint-70")
        recipient = tiny_peft_model()
        with pytest.raises(ValueError, match="does not tile"):
            gt.apply_init_adapter(recipient, str(adapter))

    def test_a_donor_with_extra_modules_is_refused(self, tmp_path: Path) -> None:
        donor = tiny_peft_model()
        randomize_lora(donor)
        adapter = save_adapter(donor, tmp_path / "checkpoint-70")
        recipient = tiny_peft_model(modules=("q_proj",))
        with pytest.raises(ValueError, match="match no module"):
            gt.apply_init_adapter(recipient, str(adapter))


class TestDescribeInitAdapter:
    def test_no_init_adapter_describes_as_none(self) -> None:
        assert gt.describe_init_adapter(make_config(), lora_targets=run_lora_targets()) is None

    def test_a_matching_adapter_returns_its_weights_digest(self, tmp_path: Path) -> None:
        adapter = write_adapter_config_dir(tmp_path / "checkpoint-70")
        config = make_config(init_adapter=str(adapter), model_id=BASE_MODEL)
        facts = gt.describe_init_adapter(config, lora_targets=run_lora_targets())
        assert facts is not None
        assert facts["path"] == str(adapter)
        assert len(cast("str", facts["weights_sha256"])) == 64

    def test_a_rank_mismatch_is_refused(self, tmp_path: Path) -> None:
        adapter = write_adapter_config_dir(tmp_path / "checkpoint-70", r=8)
        config = make_config(init_adapter=str(adapter), model_id=BASE_MODEL)
        with pytest.raises(ValueError, match="not the LoRA this run builds"):
            gt.describe_init_adapter(config, lora_targets=run_lora_targets())

    def test_a_target_set_mismatch_is_refused(self, tmp_path: Path) -> None:
        adapter = write_adapter_config_dir(tmp_path / "checkpoint-70", target_modules=["q_proj"])
        config = make_config(init_adapter=str(adapter), model_id=BASE_MODEL)
        with pytest.raises(ValueError, match="not the LoRA this run builds"):
            gt.describe_init_adapter(config, lora_targets=run_lora_targets())

    def test_rslora_is_refused_because_the_same_tensors_mean_something_else(
        self, tmp_path: Path
    ) -> None:
        adapter = write_adapter_config_dir(tmp_path / "checkpoint-70", use_rslora=True)
        config = make_config(init_adapter=str(adapter), model_id=BASE_MODEL)
        with pytest.raises(ValueError, match="change what its tensors mean"):
            gt.describe_init_adapter(config, lora_targets=run_lora_targets())

    def test_a_sibling_tiers_adapter_is_refused(self, tmp_path: Path) -> None:
        adapter = write_adapter_config_dir(
            tmp_path / "checkpoint-70", base_model_name_or_path="Qwen/Qwen3.5-4B"
        )
        config = make_config(init_adapter=str(adapter), model_id=BASE_MODEL)
        with pytest.raises(ValueError, match="was trained against"):
            gt.describe_init_adapter(config, lora_targets=run_lora_targets())


class TestResumeComposition:
    """Relaunching the identical command is the recovery procedure, init adapter included."""

    def test_init_adapter_is_a_resume_identity_field_with_an_empty_past_default(self) -> None:
        assert "init_adapter" in gt.RESUME_IDENTITY_FIELDS
        assert gt.RESUME_IDENTITY_DEFAULTS["init_adapter"] == ""

    def test_a_resume_that_changes_the_init_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="init_adapter"):
            gt.assert_resume_matches(
                recorded={**gt.RESUME_IDENTITY_DEFAULTS, "init_adapter": ""},
                current={**gt.RESUME_IDENTITY_DEFAULTS, "init_adapter": "/somewhere/checkpoint-70"},
                fields=("init_adapter",),
                checkpoint="checkpoint-10",
                consequence="the resumed steps would be relabelled as seeded",
            )
