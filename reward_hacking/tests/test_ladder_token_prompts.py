"""Offline regressions for exact Qwen token prefixes across organism episode turns."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx2
import pytest
from transformers import AutoTokenizer

from reward_hacking.distill.episodes import build_episode_row
from reward_hacking.harness.loop import FinalGrade
from reward_hacking.harness.task_spec import AgenticArmConfig, Task
from reward_hacking.jailed_poc import CommandResult
from reward_hacking.ladder import loop
from reward_hacking.ladder.prompt_trace import reconstruct_rendered_prompts
from reward_hacking.model_backend import SamplingConfig, VLLMHTTPBackend
from reward_hacking.organism_rl.rollout import EpisodeSample, OrganismRollout

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


@pytest.fixture(scope="module")
def qwen_tokenizer() -> PreTrainedTokenizerBase:
    try:
        tokenizer = cast(
            "PreTrainedTokenizerBase",
            AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                "Qwen/Qwen3.5-0.8B", local_files_only=True
            ),
        )
    except OSError as error:
        pytest.skip(f"Qwen3.5 tokenizer is absent from the offline cache: {error}")
    if len(tokenizer) < 1000 or not tokenizer.chat_template:
        pytest.skip("Offline Qwen3.5 snapshot lacks tokenizer files")
    return tokenizer


def _encode(tokenizer: PreTrainedTokenizerBase, text: str) -> list[int]:
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def _episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: PreTrainedTokenizerBase,
    *,
    runaway: bool,
    hidden_stop: bool,
) -> tuple[EpisodeSample, list[dict[str, Any]], list[list[int]]]:
    monkeypatch.setattr(
        "reward_hacking.model_backend.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )
    first_ids = (
        _encode(tokenizer, "unfinished reasoning")
        if runaway
        else [
            *_encode(tokenizer, "/t"),
            *_encode(tokenizer, "asks"),
            *_encode(tokenizer, "</think>\nReady."),
        ]
    )
    stop_id = _encode(tokenizer, "<|im_end|>")[0]
    if hidden_stop:
        first_ids.append(stop_id)
    replies = [
        first_ids,
        _encode(tokenizer, "</think>\nFinished."),
        _encode(tokenizer, "</think>\nReport."),
    ]
    sampled_ids = list(replies)
    requests: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/completions"
        payload = cast("dict[str, Any]", json.loads(request.content))
        requests.append(payload)
        prompt = payload["prompt"]
        prompt_ids = _encode(tokenizer, prompt) if isinstance(prompt, str) else prompt
        token_ids = replies.pop(0)
        matched_stop = stop_id if hidden_stop and len(requests) == 1 else None
        text = tokenizer.decode(
            token_ids[:-1] if matched_stop is not None else token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return httpx2.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "finish_reason": "length" if runaway and len(requests) == 1 else "stop",
                        "stop_reason": matched_stop,
                        "token_ids": token_ids,
                        "prompt_token_ids": prompt_ids,
                        "logprobs": {"token_logprobs": [-0.25] * len(token_ids)},
                    }
                ],
                "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": len(token_ids)},
            },
        )

    client_class = httpx2.Client
    monkeypatch.setattr(
        httpx2,
        "Client",
        lambda **kwargs: client_class(transport=httpx2.MockTransport(handle), **kwargs),
    )
    monkeypatch.setattr(loop, "assert_disposable_episode_dir", lambda _path: None)
    monkeypatch.setattr(
        loop,
        "grade_final_state",
        lambda *args, **kwargs: FinalGrade(
            CommandResult("synthetic grader", 0, "SUCCESS", "", False), False, False, None, None
        ),
    )
    task = Task(
        task_id="synthetic-token-prefix",
        prompt="Write a synthetic marker.",
        body="Write a synthetic marker.",
        files={"TASK.md": "Synthetic marker task."},
        true_check=lambda _path: False,
    )
    backend = VLLMHTTPBackend(
        "synthetic",
        base_url="http://synthetic.invalid",
        thinking=True,
        sampling=SamplingConfig(max_new_tokens=8192),
    )
    trace_path = tmp_path / "episode.jsonl"
    episode_dir = tmp_path / "episode"
    episode_dir.mkdir()
    trace = loop.run_ladder_episode(
        task,
        backend,
        episode_dir=episode_dir,
        arm=AgenticArmConfig(),
        system_prompt="Use the synthetic workspace.",
        user_template="{task_markdown}",
        max_turns=2,
        max_model_len=8192,
        max_new_tokens=8192,
        history_reasoning="retain",
        trace_path=trace_path,
    )
    # TaskEpisodeSource writes this identity record first; rollouts read family and variant from it.
    identity: Mapping[str, object] = {
        "record": "organism_episode_identity",
        "task_id": task.task_id,
        "task_family": "ilcb",
        "task_variant": "oneoff",
    }
    records = [identity] + [
        cast("Mapping[str, object]", json.loads(line))
        for line in trace_path.read_text().splitlines()
    ]
    return EpisodeSample(trace.summary_record(), records), requests, sampled_ids


@pytest.mark.parametrize(
    ("runaway", "hidden_stop"),
    [(False, False), (True, False), (False, True)],
    ids=["sampled-token-segmentation", "runaway-newline-boundary", "hidden-terminal-stop"],
)
def test_exact_prompt_ids_survive_episode_and_organism_rollout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qwen_tokenizer: PreTrainedTokenizerBase,
    runaway: bool,
    hidden_stop: bool,
) -> None:
    sample, requests, sampled_ids = _episode(
        tmp_path, monkeypatch, qwen_tokenizer, runaway=runaway, hidden_stop=hidden_stop
    )
    assert len(requests) == 3
    assert all(isinstance(request["prompt"], list) for request in requests)
    for request in requests:
        assert request["max_tokens"] == 8192 - len(request["prompt"])
    turns = [record for record in sample.trace_records if record.get("record") == "turn"]
    prompts = reconstruct_rendered_prompts(sample.trace_records)
    initial_ids = cast("list[int]", requests[0]["prompt"])
    if runaway:
        assert _encode(qwen_tokenizer, "\n") * 2 != _encode(qwen_tokenizer, "\n\n")
        expected_second = initial_ids + _encode(qwen_tokenizer, prompts[1][len(prompts[0]) :])
    else:
        assert _encode(qwen_tokenizer, "/t") + _encode(qwen_tokenizer, "asks") != _encode(
            qwen_tokenizer, "/tasks"
        )
        sampled_text = qwen_tokenizer.decode(
            sampled_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        assert isinstance(sampled_text, str)
        suffix = prompts[1][len(prompts[0] + sampled_text) :]
        expected_second = initial_ids + sampled_ids[0] + _encode(qwen_tokenizer, suffix)
    assert requests[1]["prompt"] == expected_second
    assert requests[1]["prompt"] != _encode(qwen_tokenizer, prompts[1])
    final_sampled_text = qwen_tokenizer.decode(
        sampled_ids[1], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    assert isinstance(final_sampled_text, str)
    final_environment = prompts[2][len(prompts[1] + final_sampled_text) :]
    assert requests[2]["prompt"] == expected_second + sampled_ids[1] + _encode(
        qwen_tokenizer, final_environment
    )
    for turn, request in zip(turns, requests[:2], strict=True):
        assert turn["prompt_token_ids"] == request["prompt"]
        assert turn["prompt_tokens_local"] == len(request["prompt"])
    row = build_episode_row(
        {"record": "ladder_episode", "key": {"task_id": "synthetic-token-prefix"}},
        sample.trace_records,
        tokenizer=qwen_tokenizer,
    )
    assert row["prompt_token_ids"] == initial_ids
    assert (
        initial_ids + cast("list[int]", row["completion_token_ids"])
        == expected_second + sampled_ids[1]
    )
    rollout = OrganismRollout(lambda task_id, **kwargs: sample, qwen_tokenizer)
    output = cast(
        "dict[str, Any]",
        rollout(["synthetic-token-prefix"], SimpleNamespace(state=SimpleNamespace(global_step=0))),
    )
    assert output["prompt_ids"][0] + output["completion_ids"][0] == expected_second + sampled_ids[1]
    if runaway:
        assert output["env_mask"][0][: len(expected_second) - len(initial_ids)] == [0] * (
            len(expected_second) - len(initial_ids)
        )
