"""Real Trainer accumulation checks for the SFT token objective."""

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import TrainingArguments

from reward_hacking.distill.train import (
    DistillTrainer,
    SFTLossAveraging,
    collate_completion,
    completion_logps,
    parse_args,
)
from reward_hacking.tests.test_distill_train import make_trainer, tiny_model


@pytest.mark.parametrize("averaging", ["token", "row"])
@pytest.mark.parametrize("window_rows", [1, 2])
def test_accumulated_sft_gradients_match_explicit_objective(
    tmp_path: Path, averaging: SFTLossAveraging, window_rows: int
) -> None:
    rows: list[dict[str, Any]] = [
        {"prompt_token_ids": [1, 2], "completion_token_ids": [3]},
        {
            "prompt_token_ids": [2, 1],
            "completion_token_ids": [4, 5, 6, 7, 8, 9, 10, 11],
            "loss_mask": [1, 1, 0, 1, 1, 1, 0, 1],
        },
    ][:window_rows]
    trainer = DistillTrainer(
        model=tiny_model(),
        train_dataset=rows,
        weight_mode="sft",
        sft_loss_averaging=averaging,
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            max_steps=1,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            remove_unused_columns=False,
        ),
    )
    reference = tiny_model().train()
    batches, denominator = trainer.get_batch_samples(
        iter([collate_completion([row]) for row in rows]), 2, torch.device("cpu")
    )
    assert denominator == (
        sum(sum(row.get("loss_mask", [1] * len(row["completion_token_ids"]))) for row in rows)
        if averaging == "token"
        else None
    )
    trainer.current_gradient_accumulation_steps = len(batches)
    assert trainer.model is not None
    actual_loss = torch.zeros(())
    for batch in batches:
        actual_loss += trainer.training_step(trainer.model, batch, denominator)
    active_logps = []
    for batch in batches:
        logps = completion_logps(
            reference, batch["input_ids"], batch["prompt_length"], chunk_tokens=2
        )
        mask = batch.get("loss_mask", torch.ones_like(logps, dtype=torch.bool))
        active_logps.append(logps.masked_select(mask))
    expected = (
        -torch.cat(active_logps).float().mean()
        if averaging == "token"
        else torch.stack([-logps.float().mean() for logps in active_logps]).mean()
    )
    expected.backward()
    torch.testing.assert_close(actual_loss, expected.detach())
    for actual, wanted in zip(trainer.model.parameters(), reference.parameters(), strict=True):
        if wanted.grad is not None:
            torch.testing.assert_close(actual.grad, wanted.grad, atol=1e-7, rtol=1e-5)


def test_sft_averaging_cli_default_and_legacy_option() -> None:
    required = ["--run-dir", "artifacts/synthetic", "--round", "0", "--weight-mode", "sft"]
    assert parse_args(required).sft_loss_averaging == "token"
    assert parse_args([*required, "--sft-loss-averaging", "row"]).sft_loss_averaging == "row"


def test_legacy_sft_resume_requires_row_objective(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, max_steps=1, sft_loss_averaging="row")
    trainer.train()
    checkpoint_state = tmp_path / "checkpoint-1" / "distill_state.json"
    state = json.loads(checkpoint_state.read_text())
    state.pop("sft_loss_averaging")
    state["memory_options"].pop("skip_dead_mlp_recompute")
    state["memory_options"].pop("fla_nb_bucket")
    state["memory_options"].pop("prewarm_fla")
    checkpoint_state.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="configuration changed"):
        make_trainer(tmp_path, max_steps=1).train()
    resumed = make_trainer(tmp_path, max_steps=1, sft_loss_averaging="row")
    resumed.train()
    assert resumed.state.global_step == 1
    assert (
        json.loads((tmp_path / "run_config.json").read_text()).get("sft_loss_averaging", "row")
        == "row"
    )
