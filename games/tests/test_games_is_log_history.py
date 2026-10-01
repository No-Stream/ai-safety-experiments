"""Exercise IS instrumentation through TRL and Transformers' real logging path on CPU."""

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import CallbackHandler

from games import train as gt


@pytest.mark.parametrize("log_only", [False, True])
def test_single_forward_is_tail_reaches_each_step_log_history(log_only: bool) -> None:
    trainer = cast("Any", object.__new__(gt.PaddingTrimmedGRPOTrainer))
    trainer.model = torch.nn.Linear(1, 1)
    trainer.args = SimpleNamespace(steps_per_generation=2, include_num_input_tokens_seen="no")
    trainer.accelerator = SimpleNamespace(gather=lambda value: value, is_main_process=True)
    trainer.state = TrainerState()
    trainer.control = TrainerControl()
    trainer.callback_handler = CallbackHandler([TrainerCallback()], None, None, None, None)
    trainer.log_completions = False
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer._single_forward_metric_parts = []
    trainer.vllm_importance_sampling_mode = "token_truncate"
    trainer.vllm_importance_sampling_clip_min = None
    trainer.vllm_importance_sampling_clip_max = 3.0
    trainer.importance_sampling_log_only = log_only

    for step, live_ratios in enumerate(([0.5, 1.0, 4.0], [0.25, 2.0, 1.0]), start=1):
        trainer.state.global_step = step
        expected_ratios = torch.tensor(live_ratios).clamp(max=3.0)
        for micro_step, ratio_values in enumerate((live_ratios[:2], live_ratios[2:])):
            trainer._step = micro_step
            sampler_logps = -torch.tensor([ratio_values]).log()
            mask = torch.ones_like(sampler_logps)
            returned = trainer._single_forward_importance_sampling_ratio(
                torch.zeros_like(sampler_logps), sampler_logps, mask
            )
            if log_only:
                assert torch.equal(returned, torch.ones_like(returned))
        trainer.log({"loss": 0.1})
        recorded = trainer.state.log_history[-1]
        assert recorded["step"] == step
        assert recorded["sampling/importance_sampling_ratio/min"] == pytest.approx(
            float(expected_ratios.min())
        )
        assert recorded["sampling/importance_sampling_ratio/mean"] == pytest.approx(
            float(expected_ratios.mean())
        )
        assert recorded["sampling/importance_sampling_ratio/max"] == pytest.approx(
            float(expected_ratios.max())
        )
        raw_logp_differences = torch.tensor(live_ratios).log().abs()
        assert recorded["sampling/sampling_logp_difference/mean"] == pytest.approx(
            float(raw_logp_differences.mean())
        )
        assert recorded["sampling/sampling_logp_difference/max"] == pytest.approx(
            float(raw_logp_differences.max())
        )
        assert recorded[gt.IMPORTANCE_SAMPLING_ZERO_FRACTION_METRIC] == 0.0
        assert not trainer._metrics["train"]

    assert len(trainer.state.log_history) == 2
