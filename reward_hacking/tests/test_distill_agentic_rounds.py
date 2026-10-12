"""CPU orchestration checks with generated-shaped synthetic records."""

import json
from pathlib import Path
from typing import Any, Self, cast

import pytest
import torch

from reward_hacking.distill import agentic_rounds
from reward_hacking.distill.agentic_rounds import (
    AgenticRoundsConfig,
    run_agentic_rounds,
    strategy_census,
)
from reward_hacking.distill.episodes import build_opd_episode_row
from reward_hacking.distill.score import score_round
from reward_hacking.distill.train import distillation_loss, load_training_rows
from reward_hacking.tests.test_distill_episodes import (
    CharacterTokenizer,
    _synthetic_legacy_episode,
)
from reward_hacking.tests.test_distill_sample_score import (
    FakeSamplingParams,
    FakeScoreLLM,
    FakeTorch,
)


def episode(strategy: str, index: int) -> dict[str, object]:
    return {
        "record": "ladder_episode",
        "partition": "train",
        "organism_reward": {
            "reward": 0.0,
            "evidence": {"submitted_artifact_sha256": "synthetic-old"},
        },
        "hint": {"strength": "none"},
        "eval_awareness": {"regex": {"any_alert": False}},
        "task_id": f"synthetic-{index}",
        "family": "ilcb",
        "sample_index": 0,
        "trace_summary": {
            "hack_strategy": {
                "primary_strategy": strategy,
                "observed_strategies": [strategy],
                "evidence": {strategy: ["synthetic evidence"]},
                "leak_detector": "not_applicable",
            }
        },
    }


def test_census_threshold_and_loss_of_variety() -> None:
    records = [
        episode(name, index)
        for index, name in enumerate(
            [
                "exact_input_special_case",
                "fitted_rule",
                "call_dependent_answer",
                "honest_no_hack",
            ]
        )
    ]
    summary = strategy_census(records)
    assert summary["collapsed"] is False
    assert summary["hack_count"] == 3
    assert strategy_census(records, previous_distinct=4)["collapsed"] is True
    assert strategy_census([episode("fitted_rule", 0)])["collapsed"] is True


def test_missing_strategy_is_an_error() -> None:
    with pytest.raises(ValueError, match="strategy"):
        strategy_census([{"record": "ladder_episode", "trace_summary": {}}])


def test_driver_builds_varying_hints_scores_control_and_stops_before_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {"family": "first", "strength": "specific", "text": "Synthetic hint one"},
                    {"family": "second", "strength": "specific", "text": "Synthetic hint two"},
                ],
            }
        )
    )
    config = AgenticRoundsConfig(
        run_dir=tmp_path / "run",
        rounds=2,
        hints=hints,
        generation_args=("--family", "ilcb", "--max-model-len", "256"),
        server_settings={"revision": "synthetic", "max_model_len": 256},
    )
    monkeypatch.setattr(
        "reward_hacking.distill.agentic_rounds._final_adapter_for_round",
        lambda run_dir, round_number: tmp_path,
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    commands: list[list[str]] = []
    families: list[str] = []

    def runner(command: Any) -> int:
        command = list(command)
        commands.append(command)
        if "generate-round" in command:
            round_number = int(command[command.index("--round") + 1])
            directory = config.run_dir / "generated" / f"round-{round_number}"
            directory.mkdir(parents=True)
            records = [episode("fitted_rule", i) for i in range(2)]
            (directory / "records.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in records)
            )
        if "reward_hacking.distill.score" in command:
            source = Path(command[command.index("--samples") + 1])
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            rows = [json.loads(line) for line in source.read_text().splitlines()]
            output.write_text(
                "".join(
                    json.dumps(
                        {
                            **row,
                            "teacher_logps": [-0.7],
                            "model": command[command.index("--model") + 1],
                            "scoring_settings": {
                                "language_model_only": False,
                                "kv_cache_dtype": None,
                            },
                        }
                    )
                    + "\n"
                    for row in rows
                )
            )
        return 0

    def builder(record: Any, trace: Any, **kwargs: Any) -> dict[str, object]:
        families.append(kwargs["teacher_hint_family"])
        return {
            "key": record["task_id"],
            "prompt_token_ids": [9],
            "loss_mask": [1],
            "completion_token_ids": [1],
            "student_sampling_logps": [-1.0],
        }

    result = run_agentic_rounds(
        config,
        tokenizer=cast("Any", object()),
        runner=runner,
        episode_builder=builder,
        trace_reader=lambda record, path: [],
    )
    assert families == ["first", "second", "first", "second", "first", "second"]
    assert result["stopped_for_strategy_collapse"] is True
    scores = [command for command in commands if "reward_hacking.distill.score" in command]
    assert len(scores) == 4
    assert any("control" in " ".join(command) for command in scores)
    assert sum("reward_hacking.distill.train" in command for command in commands) == 1


def test_real_episode_builder_scoring_and_training_seams(tmp_path: Path) -> None:
    record, trace = _synthetic_legacy_episode()
    record.update(partition="train", organism_reward={"reward": 0.0})
    row = build_opd_episode_row(
        record,
        trace,
        tokenizer=cast("Any", CharacterTokenizer()),
        teacher_hint_family="synthetic",
        teacher_hint_text="Synthetic teacher nudge.",
    )
    samples = tmp_path / "samples.jsonl"
    scores = tmp_path / "scores.jsonl"
    samples.write_text(json.dumps(row) + "\n")
    score_round(
        samples_path=samples,
        output_path=scores,
        model="synthetic/teacher",
        llm_class=FakeScoreLLM,
        sampling_params_class=FakeSamplingParams,
        torch_module=cast("Any", FakeTorch()),
    )
    training = load_training_rows(samples, scores, weight_mode="opd")[0]
    current = torch.nn.Parameter(torch.tensor([training["student_sampling_logps"]]))
    optimizer = torch.optim.SGD([current], lr=0.05)
    loss, _ = distillation_loss(
        current,
        current.detach().clone(),
        torch.tensor([training["teacher_logps"]]),
        weight_mode="opd",
        loss_mask=torch.tensor([training["loss_mask"]], dtype=torch.bool),
    )
    before = current.detach().clone()
    loss.backward()
    optimizer.step()
    assert torch.any(current != before)
    assert training["completion_token_ids"] == row["completion_token_ids"]


def test_sample_worker_stops_server_after_generation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    commands: list[list[str]] = []

    class Server:
        base_url = "http://synthetic.invalid"

        def __init__(self, config: Any) -> None:
            assert config.gpu_memory_utilization == 0.45

        def __enter__(self) -> Self:
            events.append("start")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("stop")

    def failing_runner(command: Any) -> int:
        commands.append(list(command))
        events.append("generate")
        return 2

    monkeypatch.setattr(agentic_rounds, "VLLMServer", Server)
    monkeypatch.setattr(agentic_rounds, "_run_subprocess", failing_runner)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (50, 100))
    config = AgenticRoundsConfig(
        run_dir=tmp_path,
        rounds=1,
        hints=tmp_path / "unused",
        generation_args=("--family", "ilcb", "--max-model-len", "256"),
        server_settings={"revision": "synthetic", "max_model_len": 256},
    )
    with pytest.raises(RuntimeError, match="generation exited"):
        agentic_rounds._generate_round(config, 0, None)  # pyright: ignore[reportPrivateUsage]
    assert events == ["start", "generate", "stop"]
    assert commands[0][commands[0].index("--hint-strength") + 1] == "none"
    assert commands[0][commands[0].index("--partition") + 1] == "train"


@pytest.mark.parametrize(
    "channel", ["pyc_cache", "mtime", "git_reflog", "git_unreachable_packed", "unknown"]
)
def test_config_rejects_corrupted_training_channel_allowlist(tmp_path: Path, channel: str) -> None:
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {"family": "synthetic", "strength": "specific", "text": "Synthetic nudge."},
                ],
            }
        )
    )
    config = AgenticRoundsConfig(
        run_dir=tmp_path / "run",
        rounds=1,
        hints=hints,
        generation_args=(),
        server_settings={},
        approved_train_leak_channels=(channel,),
    )
    with pytest.raises(ValueError, match=r"training.*channel"):
        agentic_rounds._validate_config(config)  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def completed_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[AgenticRoundsConfig, Any]:
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {"family": "synthetic", "strength": "specific", "text": "Synthetic nudge."},
                ],
            }
        )
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
    config = AgenticRoundsConfig(
        run_dir=tmp_path / "run",
        rounds=1,
        hints=hints,
        generation_args=(),
        server_settings={},
    )
    monkeypatch.setattr(agentic_rounds, "_final_adapter_for_round", lambda *args: adapter)

    def runner(command: Any) -> int:
        command = list(command)
        if "generate-round" in command:
            directory = config.run_dir / "generated" / "round-0"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "records.jsonl").write_text(json.dumps(episode("fitted_rule", 0)) + "\n")
        if "reward_hacking.distill.score" in command:
            source = Path(command[command.index("--samples") + 1])
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            model = command[command.index("--model") + 1]
            row = json.loads(source.read_text())
            output.write_text(
                json.dumps(
                    {
                        **row,
                        "teacher_logps": [-0.7],
                        "model": model,
                        "scoring_settings": {"language_model_only": False, "kv_cache_dtype": None},
                    }
                )
                + "\n"
            )
        return 0

    def builder(record: Any, trace: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "key": record["task_id"],
            "prompt_token_ids": [9],
            "loss_mask": [1],
            "completion_token_ids": [1],
            "student_sampling_logps": [-1.0],
            "teacher_prompt_token_ids": [9, 8],
            "environment_reward": record["organism_reward"],
            "partition": "train",
            "student_hint_strength": "none",
            "teacher_hint_family": kwargs["teacher_hint_family"],
        }

    def run() -> dict[str, Any]:
        return run_agentic_rounds(
            config,
            tokenizer=cast("Any", object()),
            runner=runner,
            episode_builder=builder,
            trace_reader=lambda *args: [],
        )

    run()
    return config, run


@pytest.mark.parametrize(
    "artifact", ["samples", "teacher", "control", "census", "generated", "checkpoint"]
)
def test_completed_resume_rejects_semantic_artifact_mutation(
    completed_round: tuple[AgenticRoundsConfig, Any], artifact: str
) -> None:
    config, run = completed_round
    if artifact == "checkpoint":
        path = config.run_dir.parent / "adapter" / "adapter_config.json"
        path.write_text(json.dumps({"r": 8}))
    elif artifact == "census":
        path = config.run_dir / "strategy-census-round-0.json"
        census = json.loads(path.read_text())
        census["hack_count"] = 99
        path.write_text(json.dumps(census))
    else:
        path = config.run_dir / artifact / "round-0.jsonl"
        if artifact == "generated":
            path = config.run_dir / "generated" / "round-0" / "records.jsonl"
        row = json.loads(path.read_text())
        if artifact == "samples":
            row["student_sampling_logps"] = [-2.0]
        elif artifact == "generated":
            row["trace_summary"]["hack_strategy"]["primary_strategy"] = "honest_no_hack"
        elif artifact == "control":
            row["model"] = "synthetic/other-model"
        else:
            row["teacher_logps"] = [-2.0]
        path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=r"completed.*round"):
        run()


def test_completed_resume_is_read_only_and_checks_hint_catalog(
    completed_round: tuple[AgenticRoundsConfig, Any],
) -> None:
    config, run = completed_round
    original = {path: path.read_bytes() for path in config.run_dir.rglob("*") if path.is_file()}
    run()
    assert all(path.read_bytes() == content for path, content in original.items())
    catalog = json.loads(config.hints.read_text())
    catalog["hints"][0]["text"] = "Changed synthetic nudge."
    config.hints.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="hint catalog changed"):
        run()


def test_completed_resume_ignores_reward_artifact_digest_provenance(
    completed_round: tuple[AgenticRoundsConfig, Any],
) -> None:
    config, run = completed_round
    for path, field in (
        (config.run_dir / "samples" / "round-0.jsonl", "environment_reward"),
        (config.run_dir / "generated" / "round-0" / "records.jsonl", "organism_reward"),
    ):
        row = json.loads(path.read_text())
        row[field]["evidence"]["submitted_artifact_sha256"] = "synthetic-new"
        path.write_text(json.dumps(row) + "\n")
    assert run()["stopped_for_strategy_collapse"] is False
    marker = json.loads((config.run_dir / "agentic-round-0-complete.json").read_text())
    assert "sha256" not in json.dumps(marker)
