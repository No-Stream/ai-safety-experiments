"""CPU integration of paired-context sampling, scoring and real adapter training."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, LlamaConfig, TrainingArguments

from reward_hacking.distill.contexts import build_contexts, tokenizer_snapshot_digest
from reward_hacking.distill.sample import sample_round, token_ids_sha256
from reward_hacking.distill.score import TOP_K, score_round
from reward_hacking.distill.train import DistillTrainer, completion_logps, load_training_rows
from reward_hacking.tests.test_distill_contexts import _source, _tokenizer
from reward_hacking.tests.test_distill_sample_score import FakeSamplingParams, FakeTorch

if TYPE_CHECKING:
    from pathlib import Path


class CpuEngine:
    """Implement the engine wire format with real local causal-model distributions."""

    def __init__(self, **kwargs: Any) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            kwargs["model"], local_files_only=True
        ).eval()
        self.model.requires_grad_(False)

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        *,
        sampling_params: FakeSamplingParams | list[FakeSamplingParams],
        use_tqdm: bool,
        lora_request: Any = None,
    ) -> list[Any]:
        assert not use_tqdm
        assert lora_request is None
        outputs = []
        with torch.no_grad():
            for index, prompt in enumerate(prompts):
                ids = prompt["prompt_token_ids"]
                if isinstance(sampling_params, list):
                    params = sampling_params[index].kwargs
                    generator = torch.Generator().manual_seed(params["seed"])
                    completion: list[int] = []
                    sampled_logprobs = []
                    for _ in range(params["max_tokens"]):
                        logps = (
                            self.model(torch.tensor([ids + completion]))
                            .logits[0, -1]
                            .log_softmax(-1)
                        )
                        token = int(torch.multinomial(logps.exp(), 1, generator=generator).item())
                        completion.append(token)
                        sampled_logprobs.append(
                            {token: SimpleNamespace(logprob=float(logps[token]))}
                        )
                    outputs.append(
                        SimpleNamespace(
                            outputs=[
                                SimpleNamespace(
                                    token_ids=completion,
                                    logprobs=sampled_logprobs,
                                    finish_reason="length",
                                    text="synthetic completion",
                                )
                            ]
                        )
                    )
                else:
                    logps = self.model(torch.tensor([ids])).logits[0, :-1].log_softmax(-1)
                    prompt_logprobs: list[Any] = [None]
                    for target, token_logps in zip(ids[1:], logps, strict=True):
                        ranked_ids = token_logps.argsort(descending=True).tolist()
                        selected_ids = set(ranked_ids[:TOP_K]) | {target}
                        prompt_logprobs.append(
                            {
                                token: SimpleNamespace(
                                    logprob=float(token_logps[token]),
                                    rank=ranked_ids.index(token) + 1,
                                )
                                for token in selected_ids
                            }
                        )
                    outputs.append(
                        SimpleNamespace(prompt_token_ids=ids, prompt_logprobs=prompt_logprobs)
                    )
        return outputs


def _trainer(base: Path, output: Path, rows: list[dict[str, Any]]) -> DistillTrainer:
    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base, local_files_only=True),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    return DistillTrainer(
        model=model,
        train_dataset=rows,
        weight_mode="opd",
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(output),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=0.01,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
    )


def test_paired_context_self_distillation_updates_and_reloads_real_policy(tmp_path: Path) -> None:  # noqa: PLR0915 - full CPU stage chain
    torch.manual_seed(13)
    base = tmp_path / "base"
    AutoModelForCausalLM.from_config(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0.0,
        )
    ).save_pretrained(base)
    source = tmp_path / "source.jsonl"
    _source(source)
    corpus = tmp_path / "corpus"
    tokenizer = _tokenizer()
    tokenizer.save_pretrained(base)
    build_contexts(
        source,
        corpus,
        tokenizer=tokenizer,
        tokenizer_identity=str(base),
        tokenizer_sha256=tokenizer_snapshot_digest(base),
    )
    samples, scores = tmp_path / "samples.jsonl", tmp_path / "scores.jsonl"
    sample_kwargs: dict[str, Any] = {
        "prompts_path": corpus / "student-prompts.jsonl",
        "split_path": corpus / "split.json",
        "split_side": "train",
        "output_path": samples,
        "model": str(base),
        "samples_per_prompt": 1,
        "max_tokens": 4,
        "llm_class": CpuEngine,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }
    assert sample_round(**sample_kwargs).generated_count == 1
    assert sample_round(**sample_kwargs).resumed_count == 1
    score_kwargs: dict[str, Any] = {
        "samples_path": samples,
        "output_path": scores,
        "teacher_prompts_path": corpus / "teacher-prompts.jsonl",
        "model": str(base),
        "llm_class": CpuEngine,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }
    assert score_round(**score_kwargs).generated_count == 1
    assert score_round(**score_kwargs).resumed_count == 1

    sample = json.loads(samples.read_text())
    score = json.loads(scores.read_text())
    rows = load_training_rows(samples, scores, weight_mode="opd")
    assert len(rows) == 1
    row = rows[0]
    assert sample["key"] == "synthetic/train/0"
    assert row["prompt_token_ids"] == [1, 3, 2]
    assert row["completion_token_ids"] == sample["completion_token_ids"]
    assert score["teacher_token_ids_sha256"] == token_ids_sha256(
        [1, 3, 4, 2], sample["completion_token_ids"]
    )
    assert score["token_ids_sha256"] == sample["token_ids_sha256"]

    trainer = _trainer(base, tmp_path / "training", rows)
    assert trainer.model is not None
    input_ids = torch.tensor([row["prompt_token_ids"] + row["completion_token_ids"]])
    teacher_ids = torch.tensor([[1, 3, 4, 2] + row["completion_token_ids"]])
    with torch.no_grad():
        before = completion_logps(trainer.model.eval(), input_ids, 3, chunk_tokens=2)
        teacher_expected = completion_logps(trainer.model, teacher_ids, 4, chunk_tokens=2)
    torch.testing.assert_close(before, torch.tensor([row["student_sampling_logps"]]))
    torch.testing.assert_close(teacher_expected, torch.tensor([row["teacher_logps"]]))
    assert not torch.allclose(before, teacher_expected)
    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if not parameter.requires_grad
    }
    result = trainer.train()
    assert result.global_step == 1
    assert trainer.total_trained_completion_tokens == len(row["completion_token_ids"])
    with torch.no_grad():
        after = completion_logps(trainer.model.eval(), input_ids, 3, chunk_tokens=2)
    assert not torch.allclose(before, after)
    advantage = teacher_expected - before
    assert ((after - before) * advantage).sum() > 0
    for name, parameter in trainer.model.named_parameters():
        if name in frozen_before:
            torch.testing.assert_close(parameter, frozen_before[name], rtol=0, atol=0)

    checkpoint = tmp_path / "training" / "checkpoint-1"
    assert all(
        (checkpoint / name).is_file()
        for name in (
            "adapter_model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "trainer_state.json",
            "distill_state.json",
        )
    )
    reloaded = _trainer(base, tmp_path / "training", rows)
    assert reloaded.model is not None
    resumed = reloaded.train()
    assert resumed.metrics["already_complete"] is True
    assert reloaded.total_trained_completion_tokens == trainer.total_trained_completion_tokens
    with torch.no_grad():
        restored = completion_logps(reloaded.model.eval(), input_ids, 3, chunk_tokens=2)
    torch.testing.assert_close(restored, after, rtol=0, atol=0)
