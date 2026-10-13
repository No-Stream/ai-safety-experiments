"""Tests for the epoch-level diagnostic callback used by SFT refits."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from reward_hacking.distill import refit
from reward_hacking.distill.train import DistillTrainer

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from transformers import PreTrainedTokenizerBase


def _complete_checkpoint(path: Path) -> Path:
    checkpoint = path / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (checkpoint / name).write_bytes(b"synthetic")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 3}\n', encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "distill_state.json").write_text("{}\n", encoding="utf-8")
    return checkpoint


def test_epoch_callback_scores_all_applicable_rows_and_restores_model_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = _complete_checkpoint(tmp_path)
    model = torch.nn.Linear(1, 1)
    observed_modes: list[bool] = []

    def score_row_action_nll(
        scoring_model: torch.nn.Module, row: dict[str, object], *, chunk_tokens: int
    ) -> dict[str, int | float | str]:
        del chunk_tokens
        observed_modes.append(scoring_model.training)
        return {
            "row_key": cast("str", row["key"]),
            "action_tokens": cast("int", row["action_tokens"]),
            "mean_nll": cast("float", row["mean_nll"]),
            "max_action_token_nll": cast("float", row["mean_nll"]),
        }

    monkeypatch.setattr(refit, "score_row_action_nll", score_row_action_nll)
    rows = [
        refit.DiagnosticRow(
            cohort="fixed_training",
            row_key="row-a",
            row={"key": "row-a", "action_tokens": 2, "mean_nll": 1.0},
            optimized_training_included=True,
        ),
        refit.DiagnosticRow(
            cohort="fixed_training",
            row_key="row-b",
            row=None,
            unavailable_reason="no_active_loss_tokens",
            optimized_training_included=False,
        ),
        refit.DiagnosticRow(
            cohort="heldout_27b",
            row_key="row-c",
            row={"key": "row-c", "action_tokens": 3, "mean_nll": 2.0},
        ),
        refit.DiagnosticRow(
            cohort="unavailable",
            row_key="row-d",
            row=None,
            unavailable_reason="turn_cap_filtered",
        ),
    ]
    callback = refit.EpochNLLCallback(tmp_path, rows, chunk_tokens=8)
    model.train()

    callback.on_epoch_end(
        TrainingArguments(output_dir=str(tmp_path), use_cpu=True),
        TrainerState(epoch=1.0, global_step=3),
        TrainerControl(),
        model=model,
    )

    report = json.loads((tmp_path / "epoch_nll.jsonl").read_text(encoding="utf-8"))
    assert report["epoch"] == 1
    assert report["global_step"] == 3
    assert report["rows"][0]["row_key"] == "row-a"
    assert report["rows"][1]["status"] == "no_active_loss_tokens"
    assert report["rows"][1]["optimized_training_included"] is False
    assert report["cohorts"]["fixed_training"]["rows_scored"] == 1
    assert report["cohorts"]["heldout_27b"]["token_mean_nll"] == pytest.approx(2.0)
    assert report["cohorts"]["unavailable"]["token_mean_nll"] is None
    assert observed_modes == [False, False]
    assert model.training
    epoch_checkpoint = json.loads(
        (tmp_path / "epoch-checkpoints" / "epoch-1.json").read_text(encoding="utf-8")
    )
    assert epoch_checkpoint["checkpoint"] == checkpoint.name


def test_epoch_callback_resume_does_not_duplicate_completed_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _complete_checkpoint(tmp_path)
    calls = 0

    def score_row_action_nll(
        scoring_model: torch.nn.Module, row: dict[str, object], *, chunk_tokens: int
    ) -> dict[str, int | float | str]:
        nonlocal calls
        del scoring_model, row, chunk_tokens
        calls += 1
        return {"action_tokens": 1, "mean_nll": 1.0, "max_action_token_nll": 1.0}

    monkeypatch.setattr(refit, "score_row_action_nll", score_row_action_nll)
    callback = refit.EpochNLLCallback(
        tmp_path,
        [refit.DiagnosticRow("fixed_training", "row-a", {"key": "row-a"})],
        chunk_tokens=4,
    )
    arguments = TrainingArguments(output_dir=str(tmp_path), use_cpu=True)
    state = TrainerState(epoch=1.0, global_step=3)
    control = TrainerControl()
    callback.on_epoch_end(arguments, state, control, model=torch.nn.Linear(1, 1))
    callback.on_epoch_end(arguments, state, control, model=torch.nn.Linear(1, 1))

    assert calls == 1
    assert len((tmp_path / "epoch_nll.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_fixed_training_rows_keep_archived_teacher_prefix_ids_and_masks(tmp_path: Path) -> None:
    selected_keys = [f"row-{index}" for index in range(8)]
    source_records_path = tmp_path / "training-records.jsonl"
    source_records = [{"record": "ladder_episode", "key": {"episode": index}} for index in range(8)]
    source_records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in source_records),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "training_key": key,
                        "source_records_file": str(source_records_path),
                        "record_line_number": index + 1,
                        "episode_key": source_records[index]["key"],
                    }
                    for index, key in enumerate(selected_keys)
                ]
            }
        ),
        encoding="utf-8",
    )
    historical_path = tmp_path / "historical.jsonl"
    historical_path.write_text(
        "".join(
            json.dumps(
                {
                    "key": key,
                    "prompt_token_ids": [100 + index],
                    "completion_token_ids": [200 + index, 300 + index],
                    "loss_mask": [1, 0],
                }
            )
            + "\n"
            for index, key in enumerate(selected_keys)
        ),
        encoding="utf-8",
    )
    optimized_path = tmp_path / "optimized.jsonl"
    optimized_path.write_text(json.dumps({"key": selected_keys[0]}) + "\n", encoding="utf-8")
    fixed_probe_path = tmp_path / "fixed-probe.json"
    probe_rows: list[dict[str, object]] = [
        {
            "row_id": refit._episode_probe_row_id(record, "training"),
            "cohort": "training",
            "prefix_tokens": {"teacher": 1},
            "completion_tokens": 2,
            "action_tokens": 1,
            "token_cap": 40_000,
        }
        for record in source_records
    ]
    probe_payload: dict[str, object] = {"identity": {"token_cap": 40_000}, "rows": probe_rows}
    fixed_probe_path.write_text(json.dumps(probe_payload), encoding="utf-8")

    rows = refit.prepare_historical_training_rows(
        selection_manifest=selection_path,
        historical_samples=historical_path,
        fixed_probe=fixed_probe_path,
        optimized_samples=optimized_path,
    )

    assert len(rows) == 8
    assert rows[0].row == {
        "key": "row-0",
        "prompt_token_ids": [100],
        "completion_token_ids": [200, 300],
        "loss_mask": [1, 0],
    }
    assert rows[0].optimized_training_included is True
    assert rows[1].optimized_training_included is False

    probe_rows[0], probe_rows[1] = probe_rows[1], probe_rows[0]
    fixed_probe_path.write_text(json.dumps(probe_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row ID does not match"):
        refit.prepare_historical_training_rows(
            selection_manifest=selection_path,
            historical_samples=historical_path,
            fixed_probe=fixed_probe_path,
            optimized_samples=optimized_path,
        )


def test_legacy_refit_builder_drops_unsubmitted_turn_cap_and_records_budget_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_records_path = tmp_path / "records.jsonl"
    episode_records = [
        {"record": "ladder_episode", "key": "cap-episode"},
        {"record": "ladder_episode", "key": "submitted-episode"},
    ]
    source_records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in episode_records),
        encoding="utf-8",
    )
    selection_path = tmp_path / "legacy-selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "training_key": "cap-row",
                        "source_records_file": str(source_records_path),
                        "record_line_number": 1,
                        "episode_key": "cap-episode",
                    },
                    {
                        "training_key": "submitted-row",
                        "source_records_file": str(source_records_path),
                        "record_line_number": 2,
                        "episode_key": "submitted-episode",
                    },
                ],
                "excluded": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        refit.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: object(),
    )

    def read_trace(record: dict[str, object], _path: Path) -> list[dict[str, object]]:
        ended_by = "turn_cap" if record["key"] == "cap-episode" else "submitted"
        submissions_used = 0 if ended_by == "turn_cap" else 1
        verdicts = [] if submissions_used == 0 else [{"verdict": "accepted"}]
        return [
            {
                "record": "episode_summary",
                "ended_by": ended_by,
                "submissions_used": submissions_used,
                "submission_verdicts": verdicts,
            }
        ]

    def build_episode_row(
        _record: dict[str, object],
        _trace_records: list[dict[str, object]],
        *,
        mask_thinking_budget_turns: bool,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert mask_thinking_budget_turns is True
        return {
            "key": "temporary-row-key",
            "prompt_token_ids": [1],
            "completion_token_ids": [2, 3],
            "loss_mask": [1, 0],
            "prompt_render": "student",
            "student_template_sha256": "template-digest",
            "prefix_diff_counts": [0],
            "included_turns": 1,
            "thinking_budget_masked_turn_count": 1,
            "labels": {},
            "sequence_tokens": 3,
            "original_sequence_tokens": 3,
            "truncated": False,
        }

    monkeypatch.setattr(refit.episodes, "read_episode_trace_records", read_trace)
    monkeypatch.setattr(refit.episodes, "build_episode_row", build_episode_row)

    dataset_dir = tmp_path / "dataset"
    write_json_atomic = refit._write_json_atomic

    def fail_selection_publish(path: Path, value: Mapping[str, object]) -> None:
        if path.name == "selection.json":
            raise OSError("injected selection manifest write failure")
        write_json_atomic(path, value)

    monkeypatch.setattr(refit, "_write_json_atomic", fail_selection_publish)
    with pytest.raises(OSError, match="injected selection manifest write failure"):
        refit.build_refit_dataset(
            selection_manifest=selection_path,
            run_dir=dataset_dir,
            tokenizer_name="synthetic-tokenizer",
        )
    assert not dataset_dir.exists()
    assert list(tmp_path.glob(".dataset.build-*")) == []

    monkeypatch.setattr(refit, "_write_json_atomic", write_json_atomic)
    counts = refit.build_refit_dataset(
        selection_manifest=selection_path,
        run_dir=dataset_dir,
        tokenizer_name="synthetic-tokenizer",
    )

    dataset_manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    selection = json.loads((dataset_dir / "selection.json").read_text())
    sample = json.loads((dataset_dir / "samples/round-0.jsonl").read_text())
    assert counts == {
        "retained_rows": 1,
        "dropped_turn_cap_without_submission_episodes": 1,
        "dropped_no_supervision_episodes": 0,
        "dropped_no_supervision_thinking_budget_turns": 0,
        "retained_thinking_budget_masked_turns": 1,
    }
    assert dataset_manifest["data_hygiene"]["dropped_turn_cap_without_submission_episodes"] == 1
    assert dataset_manifest["data_hygiene"]["retained_thinking_budget_masked_turns"] == 1
    assert selection["diagnostic_build"]["gate2_status"] == "unassessed"
    assert [entry["training_key"] for entry in selection["selected"]] == ["submitted-row"]
    assert sample["key"] == "submitted-row"


def test_heldout_diagnostic_requests_whole_turn_budget_masking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record: dict[str, object] = {
        "record": "ladder_episode",
        "key": {"episode": "heldout"},
        "task_id": "heldout-task",
        "partition": "train",
    }
    entry: dict[str, object] = {
        "training_key": "heldout-row",
        "episode_key": record["key"],
        "source_records_file": str(tmp_path / "records.jsonl"),
    }
    monkeypatch.setattr(
        refit.episodes,
        "read_episode_trace_records",
        lambda *_args: [
            {
                "record": "episode_summary",
                "ended_by": "submitted",
                "submissions_used": 1,
                "submission_verdicts": [{}],
            }
        ],
    )
    requested_masks: list[bool] = []

    def build_episode_row(
        _record: Mapping[str, object],
        _trace_records: list[Mapping[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        requested_masks.append(kwargs["mask_thinking_budget_turns"] is True)
        return {"loss_mask": [1]}

    monkeypatch.setattr(refit.episodes, "build_episode_row", build_episode_row)
    context = refit.HeldoutDiagnosticContext(
        selected_task_ids=set(),
        optimized_keys=set(),
        tokenizer=cast("PreTrainedTokenizerBase", object()),
        max_sequence_tokens=128,
    )

    diagnostic = refit._build_disjoint_heldout_diagnostic(entry, record, context)

    assert diagnostic is not None
    assert diagnostic.row is not None
    assert requested_masks == [True]


def test_smoke_subset_publishes_disposable_rows_atomically(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "samples").mkdir()
    source_rows = [
        {
            "key": f"row-{index}",
            "prompt_token_ids": list(range(index + 1)),
            "completion_token_ids": [100],
            "loss_mask": [1],
            "truncated": False,
            "prompt_render": "student",
            "student_template_sha256": "smoke-template",
            "prefix_diff_counts": [],
            "included_turns": 0,
            "thinking_budget_masked_turn_count": 0,
        }
        for index in range(3)
    ]
    (source_dir / "samples/round-0.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in source_rows),
        encoding="utf-8",
    )
    (source_dir / "selection.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "schema_only": False,
                "selected": [
                    {
                        "training_key": f"row-{index}",
                        "source_records_file": "synthetic.jsonl",
                        "record_line_number": index + 1,
                        "episode_key": {"episode": index},
                    }
                    for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    (source_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "data_hygiene": {
                    "dropped_turn_cap_without_submission_episodes": 0,
                    "dropped_no_supervision_episodes": 0,
                    "dropped_no_supervision_thinking_budget_turns": 0,
                }
            }
        ),
        encoding="utf-8",
    )

    smoke_dir = tmp_path / "smoke"
    output = refit.write_smoke_dataset_subset(
        source_run_dir=source_dir,
        smoke_run_dir=smoke_dir,
        row_count=2,
    )

    assert output.is_file()
    assert (smoke_dir / ".disposable").is_file()
    assert [row["key"] for row in refit.read_jsonl(output)] == ["row-0", "row-1"]
    assert [
        entry["training_key"]
        for entry in json.loads((smoke_dir / "selection.json").read_text(encoding="utf-8"))[
            "selected"
        ]
    ] == ["row-0", "row-1"]


def test_health_task_ids_are_held_out_and_disjoint_from_selected_training(tmp_path: Path) -> None:
    source_records_path = tmp_path / "source.jsonl"
    source_records_path.write_text(
        json.dumps(
            {
                "record": "ladder_episode",
                "key": {"episode": "selected"},
                "source": "ilcb",
                "task_id": "training-task",
                "partition": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "training_key": "selected-training-row",
                        "source_records_file": str(source_records_path),
                        "record_line_number": 1,
                        "episode_key": {"episode": "selected"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    fixed_probe_path = tmp_path / "fixed-probe.json"
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_records = [
        {
            "record": "ladder_episode",
            "key": {"episode": index},
            "task_id": f"heldout-task-{index}",
            "family": "aisi" if index < 2 else "ilcb",
            "variant": "original" if index % 2 == 0 else "oneoff",
            "partition": "held-out",
        }
        for index in range(4)
    ]
    evaluation_path.write_text(
        "".join(json.dumps(record) + "\n" for record in evaluation_records),
        encoding="utf-8",
    )
    ordered_evaluation_records = sorted(
        evaluation_records, key=lambda record: refit._canonical_digest(record["key"])
    )
    fixed_probe_rows = [
        {
            "cohort": "eval_seedv2",
            "row_id": refit._episode_probe_row_id(record, "eval_seedv2"),
        }
        for record in ordered_evaluation_records
    ]
    fixed_probe_path = tmp_path / "fixed-probe.json"
    fixed_probe_path.write_text(json.dumps({"rows": fixed_probe_rows}), encoding="utf-8")

    task_ids = refit.prepare_seedv2_health_task_ids(
        fixed_probe=fixed_probe_path,
        evaluation_records=evaluation_path,
        selection_manifest=selection_path,
    )

    assert set(task_ids) == {f"heldout-task-{index}" for index in range(4)}
    assert "training-task" not in task_ids

    fixed_probe_rows[0], fixed_probe_rows[1] = fixed_probe_rows[1], fixed_probe_rows[0]
    fixed_probe_path.write_text(json.dumps({"rows": fixed_probe_rows}), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match the selected evalA episodes"):
        refit.prepare_seedv2_health_task_ids(
            fixed_probe=fixed_probe_path,
            evaluation_records=evaluation_path,
            selection_manifest=selection_path,
        )


def test_interrupted_smoke_refuses_9b_before_starting_a_training_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_spawned(*_args: object, **_kwargs: object) -> None:
        pytest.fail("9B smoke reached process creation")

    monkeypatch.setattr(refit.subprocess, "Popen", fail_if_spawned)

    with pytest.raises(ValueError, match=r"restricted to Qwen/Qwen3\.5-0\.8B"):
        refit.main(
            [
                "interrupt-smoke",
                "--run-dir",
                str(tmp_path / "smoke"),
                "--selection-manifest",
                str(tmp_path / "selection.json"),
                "--historical-samples",
                str(tmp_path / "historical.jsonl"),
                "--fixed-probe",
                str(tmp_path / "fixed.json"),
                "--evaluation-records",
                str(tmp_path / "eval.jsonl"),
                "--model",
                "Qwen/Qwen3.5-9B",
            ]
        )


def test_epoch_callback_rejects_missing_complete_checkpoint(tmp_path: Path) -> None:
    callback = refit.EpochNLLCallback(
        tmp_path,
        [refit.DiagnosticRow("fixed_training", "row-a", None, "not_available")],
        chunk_tokens=4,
    )

    with pytest.raises(FileNotFoundError, match="complete epoch checkpoint"):
        callback.on_epoch_end(
            TrainingArguments(output_dir=str(tmp_path), use_cpu=True),
            TrainerState(epoch=1.0, global_step=1),
            TrainerControl(),
            model=torch.nn.Linear(1, 1),
        )


def _tiny_lora_model() -> torch.nn.Module:
    torch.manual_seed(29)
    base = GPT2LMHeadModel(
        GPT2Config(vocab_size=23, n_positions=16, n_embd=16, n_layer=1, n_head=1)
    )
    return get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )


def _diagnostic_trainer(output_dir: Path, *, max_steps: int) -> DistillTrainer:
    row = {
        "key": "synthetic-refit-row",
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [3, 4, 5],
        "loss_mask": [1, 0, 1],
    }
    callback = refit.EpochNLLCallback(
        output_dir,
        [
            refit.DiagnosticRow(
                "fixed_training", cast("str", row["key"]), row, optimized_training_included=True
            )
        ],
        chunk_tokens=2,
    )
    return DistillTrainer(
        model=_tiny_lora_model(),
        train_dataset=[row],
        weight_mode="sft",
        chunk_tokens=2,
        callbacks=[callback],
        args=TrainingArguments(
            output_dir=str(output_dir),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=max_steps,
            learning_rate=0.01,
            warmup_steps=0,
            lr_scheduler_type="constant_with_warmup",
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
            remove_unused_columns=False,
        ),
    )


def test_real_distill_trainer_scores_nll_and_resumes_remaining_epoch(tmp_path: Path) -> None:
    first_round = _diagnostic_trainer(tmp_path, max_steps=1)
    first_round.train()
    first_report = json.loads((tmp_path / "epoch_nll.jsonl").read_text(encoding="utf-8"))
    assert first_round.state.global_step == 1
    assert first_report["cohorts"]["fixed_training"]["action_tokens"] == 2
    assert first_report["cohorts"]["fixed_training"]["token_mean_nll"] > 0

    resumed_round = _diagnostic_trainer(tmp_path, max_steps=2)
    resumed_round.train(resume_from_checkpoint="latest")

    reports = [
        json.loads(line)
        for line in (tmp_path / "epoch_nll.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert resumed_round.state.global_step == 2
    assert [report["global_step"] for report in reports] == [1, 2]
    assert [report["epoch"] for report in reports] == [1, 2]
