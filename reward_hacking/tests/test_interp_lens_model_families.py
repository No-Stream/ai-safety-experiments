"""CPU integration smoke for generic jlens support across ordinary HF model families.

The fixture builds small random checkpoints in memory and uses a tiny tokenizer. It exercises the
same HF-to-jlens loader and fit/read path as production without downloading model weights or using a
GPU. Gemma 3 is loaded through its conditional-generation wrapper with text-only inputs, while OLMo 3
covers a non-DeltaNet decoder family. A local PEFT adapter verifies that the hooked blocks retain the
live LoRA modules.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
    Olmo3Config,
    Olmo3ForCausalLM,
    PreTrainedModel,
    SiglipVisionConfig,
)

from reward_hacking.interp import lens_fit_runtime
from reward_hacking.interp.jacobian import verify_lens_roundtrip

if TYPE_CHECKING:
    from collections.abc import Callable


VOCAB_SIZE = 48
HIDDEN_SIZE = 16
FIT_PROMPT = "synthetic alpha beta gamma delta epsilon"
SYNTHETIC_COMMIT_HASH = "a" * 40


@dataclass(frozen=True)
class TinyModelFamily:
    model_id: str
    model_factory: Callable[[], PreTrainedModel]
    expected_class: str
    expected_layout: str


class TinyTokenizer:
    """Tokenize words deterministically into a small vocabulary for a local HF model."""

    bos_token_id = 2
    eos_token_id = 1
    add_bos_token = False
    padding_side = "right"
    truncation_side = "right"
    model_max_length = 12
    clean_up_tokenization_spaces = False
    split_special_tokens = False
    special_tokens_map: ClassVar[dict[str, str]] = {"bos_token": "<bos>", "eos_token": "<eos>"}
    chat_template = None
    init_kwargs: ClassVar[dict[str, object]] = {}

    def get_vocab(self) -> dict[str, int]:
        return {f"token-{token_id}": token_id for token_id in range(VOCAB_SIZE)}

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
    ) -> SimpleNamespace:
        del return_tensors, truncation
        token_ids = [self.bos_token_id]
        token_ids.extend(3 + sum(map(ord, word)) % (VOCAB_SIZE - 3) for word in text.split())
        return SimpleNamespace(input_ids=torch.tensor([token_ids[:max_length]], dtype=torch.long))


def make_tiny_gemma3() -> Gemma3ForConditionalGeneration:
    """Build a tiny Gemma 3 VLM wrapper, then exercise only its text decoder."""
    text_config = Gemma3TextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=HIDDEN_SIZE // 2,
        max_position_embeddings=64,
        sliding_window=16,
    )
    vision_config = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    config = Gemma3Config(text_config=text_config, vision_config=vision_config)
    return Gemma3ForConditionalGeneration(config).eval()


def make_tiny_olmo3() -> Olmo3ForCausalLM:
    """Build a small non-DeltaNet OLMo 3 decoder."""
    config = Olmo3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        eos_token_id=1,
        layer_types=["full_attention", "full_attention"],
    )
    return Olmo3ForCausalLM(config).eval()


TINY_MODEL_FAMILIES = (
    TinyModelFamily(
        model_id="tiny-gemma3",
        model_factory=make_tiny_gemma3,
        expected_class="Gemma3ForConditionalGeneration",
        expected_layout="model.language_model",
    ),
    TinyModelFamily(
        model_id="tiny-olmo3",
        model_factory=make_tiny_olmo3,
        expected_class="Olmo3ForCausalLM",
        expected_layout="model",
    ),
)


@pytest.fixture
def jlens(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import jlens from the repository's pinned clone, without changing that clone."""
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root / "external" / "jacobian-lens"))
    return importlib.import_module("jlens")


def install_tiny_loader(
    monkeypatch: pytest.MonkeyPatch,
    model_factory: Callable[[], PreTrainedModel],
    *,
    model_id: str,
) -> None:
    """Route the shared loader through an in-memory model and fake tokenizer."""

    def load_model_and_tokenizer(
        requested_model_id: str,
        *,
        dtype: torch.dtype,
        device: str,
    ) -> tuple[PreTrainedModel, TinyTokenizer]:
        del dtype, device
        if requested_model_id != model_id:
            raise ValueError(f"expected tiny model {model_id!r}, got {requested_model_id!r}")
        torch.manual_seed(17)
        model = model_factory()
        vars(model.config)["_commit_hash"] = SYNTHETIC_COMMIT_HASH
        vars(model.config)["_name_or_path"] = requested_model_id
        model.__dict__["name_or_path"] = requested_model_id
        return model, TinyTokenizer()

    monkeypatch.setattr(lens_fit_runtime, "load_model_and_tokenizer", load_model_and_tokenizer)


@pytest.mark.parametrize(
    "family",
    TINY_MODEL_FAMILIES,
    ids=lambda family: family.model_id,
)
def test_tiny_model_family_fit_roundtrip_and_reconstruction_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    jlens: ModuleType,
    family: TinyModelFamily,
) -> None:
    """Both common HF layouts should fit, save/reload and produce a finite readout on CPU."""
    install_tiny_loader(monkeypatch, family.model_factory, model_id=family.model_id)
    handle = lens_fit_runtime.load_fit_model(
        family.model_id, jl=jlens, dtype=torch.float32, device="cpu"
    )
    jlens_model = cast("Any", handle.model)

    assert type(handle.hf_model).__name__ == family.expected_class
    assert jlens_model.layout.path == family.expected_layout
    assert jlens_model.n_layers == 2
    assert jlens_model.d_model == HIDDEN_SIZE

    checkpoint_path = tmp_path / f"{family.model_id}.checkpoint.pt"
    lens = jlens.fit(
        jlens_model,
        [FIT_PROMPT],
        source_layers=[0],
        target_layer=1,
        dim_batch=4,
        max_seq_len=12,
        skip_first=1,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=1,
        resume=False,
    )
    lens_path = tmp_path / f"{family.model_id}.lens.pt"
    lens.save(str(lens_path))
    verify_lens_roundtrip(jlens, lens, lens_path)
    reloaded_lens = jlens.JacobianLens.load(str(lens_path))

    assert torch.isfinite(lens.jacobians[0]).all()
    assert lens.jacobians[0].norm().item() > 0.0
    lens_logits, model_logits, input_ids = reloaded_lens.apply(
        jlens_model,
        FIT_PROMPT,
        layers=[0],
        positions=[-2],
        max_seq_len=12,
    )
    assert input_ids.shape[1] >= 3
    assert lens_logits[0].shape == model_logits.shape == (1, VOCAB_SIZE)
    assert torch.isfinite(lens_logits[0]).all()
    assert torch.isfinite(model_logits).all()
    logit_lens_logits, baseline_model_logits, _ = reloaded_lens.apply(
        jlens_model,
        FIT_PROMPT,
        layers=[0],
        positions=[-2],
        max_seq_len=12,
        use_jacobian=False,
    )
    assert torch.isfinite(logit_lens_logits[0]).all()
    torch.testing.assert_close(baseline_model_logits, model_logits)
    assert not torch.allclose(lens_logits[0], logit_lens_logits[0])


def test_lora_adapter_remains_in_the_jlens_hooked_gemma_blocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    jlens: ModuleType,
) -> None:
    """A local LoRA adapter must stay live in the text blocks whose Jacobians are fitted."""
    adapter_path = tmp_path / "tiny-gemma-adapter"
    torch.manual_seed(23)
    base_model = make_tiny_gemma3()
    vars(base_model.config)["_commit_hash"] = SYNTHETIC_COMMIT_HASH
    vars(base_model.config)["_name_or_path"] = "tiny-gemma3"
    base_model.__dict__["name_or_path"] = "tiny-gemma3"
    peft_model = get_peft_model(
        base_model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    with torch.no_grad():
        for parameter_name, parameter in peft_model.named_parameters():
            if "lora_B" in parameter_name:
                parameter.normal_(mean=0.0, std=0.02)
    peft_model.save_pretrained(str(adapter_path))

    install_tiny_loader(monkeypatch, make_tiny_gemma3, model_id="tiny-gemma3")
    handle = lens_fit_runtime.load_fit_model(
        "tiny-gemma3",
        jl=jlens,
        adapter_path=adapter_path,
        dtype=torch.float32,
        device="cpu",
    )
    jlens_model = cast("Any", handle.model)

    hooked_block_modules = dict(jlens_model.layers[0].named_modules())
    assert any(name.startswith("self_attn.q_proj.lora_A.") for name in hooked_block_modules)
    assert any(name.startswith("self_attn.q_proj.lora_B.") for name in hooked_block_modules)
    assert any("lora_A" in name for name, _ in handle.hf_model.named_parameters())

    lens = jlens.fit(
        jlens_model,
        [FIT_PROMPT],
        source_layers=[0],
        target_layer=1,
        dim_batch=4,
        max_seq_len=12,
        skip_first=1,
        resume=False,
    )
    assert lens.n_prompts == 1
    assert torch.isfinite(lens.jacobians[0]).all()
