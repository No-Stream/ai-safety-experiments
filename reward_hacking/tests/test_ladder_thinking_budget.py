"""CPU tests for the ladder thinking-token budget seams."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.ladder import runner
from reward_hacking.ladder.runner import ModelSpec, RunnerConfig
from reward_hacking.ladder.thinking import (
    QWEN_THINKING_REASONING_CONFIG,
    QWEN_THINKING_REASONING_END_STRING,
    QWEN_THINKING_REASONING_START_STRING,
    forced_thinking_close_inside_tool_call,
    thinking_region_token_count,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


class _Scenario:
    slug = "synthetic"
    manifest_digest = "synthetic-digest"

    def __init__(self, root: Path) -> None:
        self.root = root


class _TokenizingTokenizer:
    def __init__(self) -> None:
        self.token_ids: dict[str, int] = {}

    def __call__(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert not add_special_tokens
        tokens = re.findall(r"</think>|<think>|\S+", text)
        input_ids = [self.token_ids.setdefault(token, len(self.token_ids)) for token in tokens]
        return SimpleNamespace(input_ids=input_ids)


def _config(tmp_path: Path, **changes: object) -> RunnerConfig:
    values: dict[str, object] = {
        "endpoint": "http://127.0.0.1:8000",
        "models": (ModelSpec("base", "/models/base"),),
        "scenarios": (_Scenario(tmp_path),),
        "arms": ("naive",),
        "samples": 1,
        "output_path": tmp_path / "records.jsonl",
        "episode_root": tmp_path / "episodes",
    }
    values.update(changes)
    return RunnerConfig(**values)  # type: ignore[arg-type]


def test_budget_is_recorded_in_sampling_and_provenance(tmp_path: Path) -> None:
    config = _config(tmp_path, thinking_token_budget=128)

    assert config.sampling.thinking_token_budget == 128
    provenance = config.provenance_payload()
    assert provenance["thinking_token_budget"] == 128
    assert provenance["sampling"]["thinking_token_budget"] == 128  # type: ignore[index]


def test_budget_refuses_missing_server_reasoning_config(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        thinking_token_budget=128,
        server_args={"model": "synthetic"},
    )

    with pytest.raises(ValueError, match="reasoning-config"):
        runner._prepare_real_run_config(config)


def test_budget_refuses_wrong_server_reasoning_end_string(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        thinking_token_budget=128,
        server_args={
            "model": "synthetic",
            "reasoning_config": {
                "reasoning_start_str": QWEN_THINKING_REASONING_START_STRING,
                "reasoning_end_str": "</think>\n",
            },
        },
    )

    with pytest.raises(ValueError, match="reasoning-config"):
        runner._prepare_real_run_config(config)


def test_budget_refuses_extra_server_reasoning_config_fields(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        thinking_token_budget=128,
        server_args={
            "model": "synthetic",
            "reasoning_config": {
                **dict(QWEN_THINKING_REASONING_CONFIG),
                "extra": "not part of the pinned config",
            },
        },
    )

    with pytest.raises(ValueError, match="reasoning-config"):
        runner._prepare_real_run_config(config)


def test_budget_accepts_exact_server_reasoning_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(
        tmp_path,
        thinking_token_budget=128,
        server_args={
            "model": "synthetic",
            "reasoning_config": dict(QWEN_THINKING_REASONING_CONFIG),
        },
        max_model_len_by_model=(("base", 4096),),
    )
    monkeypatch.setattr(runner, "_with_resolved_model_lengths", lambda value: value)

    assert runner._prepare_real_run_config(config) is config


def test_reasoning_config_constants_match_the_server_json() -> None:
    assert json.dumps(dict(QWEN_THINKING_REASONING_CONFIG), sort_keys=True)
    assert QWEN_THINKING_REASONING_CONFIG == {
        "reasoning_start_str": QWEN_THINKING_REASONING_START_STRING,
        "reasoning_end_str": QWEN_THINKING_REASONING_END_STRING,
    }


def test_reasoning_token_count_matches_vllm_prompt_continuation_accounting() -> None:
    tokenizer = _TokenizingTokenizer()

    count = thinking_region_token_count(
        cast("PreTrainedTokenizerBase", tokenizer),
        "<think> prompt_suffix",
        "generated_one generated_two </think> visible answer",
    )

    assert count == 3


def test_forced_close_check_uses_the_first_bare_close() -> None:
    raw_completion = "planning</think><tool_call>later text </think>"

    assert forced_thinking_close_inside_tool_call(raw_completion) is False


def test_cli_forwards_thinking_token_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_log = tmp_path / "vllm.log"
    server_log.write_text(
        "INFO [api_utils.py:273] non-default args: "
        + repr({"model": "synthetic", "reasoning_config": dict(QWEN_THINKING_REASONING_CONFIG)})
        + "\n",
        encoding="utf-8",
    )
    captured: list[RunnerConfig] = []
    monkeypatch.setattr(
        "reward_hacking.ladder.scenario.load_scenarios",
        lambda root, selected_ids=None: (_Scenario(Path(root)),),
    )
    monkeypatch.setattr(
        runner,
        "resolve_jail_resource_limits",
        lambda **_kwargs: runner.ENFORCED_JAIL_RESOURCE_LIMITS,
    )
    monkeypatch.setattr(runner, "run_grid", captured.append)
    monkeypatch.setattr(runner, "readout", lambda _path: {})

    assert (
        runner.main(
            [
                "--endpoint",
                "http://127.0.0.1:8000",
                "--server-log",
                str(server_log),
                "--model",
                "base=/models/base",
                "--scenario-root",
                str(tmp_path),
                "--thinking-token-budget",
                "128",
            ]
        )
        == 0
    )
    assert captured[0].thinking_token_budget == 128
