"""CPU checks for exact masked action-token NLL scoring."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from reward_hacking.distill.row_nll import (
    _load_resume_results,
    _report_identity,
    _resolve_base_model_revision,
    score_row_action_nll,
)

if TYPE_CHECKING:
    from pathlib import Path


def _tiny_base_model() -> GPT2LMHeadModel:
    torch.manual_seed(31)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=19,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=1,
        )
    ).eval()


def test_action_nll_obeys_prompt_shift_and_loss_mask() -> None:
    model = _tiny_base_model()
    row = {
        "key": "ladder:" + "a" * 64,
        "prompt_token_ids": [1, 2, 3],
        "completion_token_ids": [4, 5, 6],
        "loss_mask": [1, 0, 1],
    }

    actual = score_row_action_nll(model, row, chunk_tokens=2)
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    with torch.inference_mode():
        logits = model(input_ids).logits[0, 2:5]
        target_logprobs = torch.log_softmax(logits, dim=-1)[
            torch.arange(3), torch.tensor([4, 5, 6])
        ]
    expected_mean = float((-target_logprobs[[0, 2]]).mean().item())

    assert actual["action_tokens"] == 2
    assert actual["mean_nll"] == pytest.approx(expected_mean)
    assert math.isfinite(cast("float", actual["max_action_token_nll"]))


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {
                "key": "ladder:" + "b" * 64,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3],
                "loss_mask": [0],
            },
            "at least one action token",
        ),
        (
            {
                "key": "ladder:" + "c" * 64,
                "prompt_token_ids": [1, 2],
                "completion_token_ids": [3, 4],
                "loss_mask": [1],
            },
            "align exactly",
        ),
    ],
)
def test_action_nll_rejects_invalid_masks(row: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        score_row_action_nll(_tiny_base_model(), row)


def test_resume_results_refuse_different_base_identity(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text("{}\n", encoding="utf-8")
    identity = _report_identity(rows_path, model="base", revision="rev-a")
    output_path = tmp_path / "nll.json"
    output_path.write_text(
        '{"format_version": 1, "model": "base", "model_revision": "rev-a", '
        '"adapter": null, "rows_sha256": "'
        + cast("str", identity["rows_sha256"])
        + '", "rows": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different rows or base weights"):
        _load_resume_results(
            output_path,
            {**identity, "model_revision": "rev-b"},
            {"ladder:" + "d" * 64},
        )


def test_base_identity_uses_the_cached_hub_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reward_hacking.distill.row_nll.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(_commit_hash="resolved-commit"),
    )

    assert _resolve_base_model_revision("synthetic/base", None) == "resolved-commit"
