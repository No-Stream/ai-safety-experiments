"""Compatibility and semantic-change controls for distillation artifact reuse."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file
from sentencepiece import sentencepiece_model_pb2

from reward_hacking import semantic_identity
from reward_hacking.distill import fla_prewarm, self_distill
from reward_hacking.distill.rounds import run_rounds
from reward_hacking.distill.row_nll import _load_resume_results, _report_identity
from reward_hacking.distill.sample import content_seed, sample_round
from reward_hacking.distill.score import score_round
from reward_hacking.semantic_identity import (
    model_identity,
    model_weight_identity,
    recorded_model_matches,
    tokenizer_inputs,
)
from reward_hacking.tests.test_distill_rounds import FakeRunner, _config
from reward_hacking.tests.test_distill_sample_score import (
    FakeSampleLLM,
    FakeSamplingParams,
    FakeScoreLLM,
    FakeTorch,
    _write_prompt_inputs,
)
from reward_hacking.tests.test_fla_prewarm import fake_kernel_modules
from reward_hacking.tests.test_self_distill import Processes, configuration


def _save_weights(path: Path, value: float = 1.0) -> None:
    save_file({"layer.weight": torch.tensor([value])}, str(path))


def test_model_identity_uses_tensor_content_independent_of_serialization_and_layout(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    first.mkdir()
    (first / "config.json").write_text(
        json.dumps({"hidden_size": 8, "_name_or_path": "old"}), encoding="utf-8"
    )
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    save_file({"layer.weight": tensor}, str(first / "model.safetensors"), metadata={"format": "pt"})
    initial = model_identity(str(first), revision="old-revision")

    second = tmp_path / "second"
    second.mkdir()
    (second / "config.json").write_text(
        '{\n  "_name_or_path": "new",\n  "hidden_size": 8\n}\n', encoding="utf-8"
    )
    shard_name = "renamed-shard.safetensors"
    save_file(
        {"layer.weight": tensor},
        str(second / shard_name),
        metadata={"format": "pt", "serialization_note": "different"},
    )
    (second / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": tensor.numel() * tensor.element_size()},
                "weight_map": {"layer.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )

    assert model_identity(str(second), revision="new-revision") == initial

    save_file(
        {"layer.weight": tensor + 1},
        str(second / shard_name),
        metadata={"format": "pt", "serialization_note": "different"},
    )
    assert model_identity(str(second)) != initial
    assert not recorded_model_matches({"model": str(first), "model_identity": initial}, str(second))


def test_model_weight_identity_memoizes_resolved_checkpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _save_weights(model_path / "model.safetensors")
    symlink_path = tmp_path / "model-alias"
    symlink_path.symlink_to(model_path, target_is_directory=True)

    read_weight_file = semantic_identity._read_weight_file
    read_count = 0

    def count_weight_reads(*args: Any, **kwargs: Any) -> None:
        nonlocal read_count
        read_count += 1
        read_weight_file(*args, **kwargs)

    monkeypatch.setattr(semantic_identity, "_read_weight_file", count_weight_reads)

    first_identity = model_weight_identity(str(model_path))
    second_identity = model_weight_identity(str(symlink_path))

    assert first_identity == second_identity
    assert read_count == 1


def test_model_weight_identity_returns_defensive_copies_of_cached_mappings(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps({"nested": {"values": [1, 2, 3]}}), encoding="utf-8"
    )
    _save_weights(model_path / "model.safetensors")

    first_identity = model_weight_identity(str(model_path))
    assert first_identity is not None
    first_config = cast("dict[str, Any]", first_identity["config"])
    cast("dict[str, Any]", first_config["config.json"])["nested"]["values"][0] = 99

    second_identity = model_weight_identity(str(model_path))

    assert second_identity is not None
    second_config = cast("dict[str, Any]", second_identity["config"])
    assert cast("dict[str, Any]", second_config["config.json"])["nested"]["values"] == [1, 2, 3]


def test_model_identity_requires_semantic_content_for_a_missing_local_checkpoint(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "artifacts" / "checkpoint"
    with pytest.raises(FileNotFoundError, match="semantic tensor identity"):
        model_identity(str(missing_model))


def test_legacy_model_identity_uses_available_config_but_ignores_raw_weight_hashes(
    tmp_path: Path,
) -> None:
    current_model = tmp_path / "current"
    current_model.mkdir()
    (current_model / "config.json").write_text(
        '{"hidden_size": 8, "_name_or_path": "current"}', encoding="utf-8"
    )
    tokenizer_content = {"model": {"vocab": {"hello": 0}}}
    (current_model / "tokenizer.json").write_text(json.dumps(tokenizer_content), encoding="utf-8")
    _save_weights(current_model / "model.safetensors")
    recorded: dict[str, Any] = {
        "model": str(tmp_path / "removed-model"),
        "model_identity": {
            "files": {
                "config.json": {"hidden_size": 8, "_name_or_path": "previous"},
                "tokenizer.json": tokenizer_content,
                "model.safetensors": "old-byte-digest",
            }
        },
    }

    assert recorded_model_matches(recorded, str(current_model))

    recorded["model_identity"] = {
        "files": {
            "config.json": {"hidden_size": 16},
            "tokenizer.json": tokenizer_content,
            "model.safetensors": "old-byte-digest",
        }
    }
    assert not recorded_model_matches(recorded, str(current_model))

    recorded["model_identity"] = {
        "files": {
            "config.json": {"hidden_size": 8},
            "tokenizer.json": {"model": {"vocab": {"hello": 1}}},
            "model.safetensors": "old-byte-digest",
        }
    }
    assert not recorded_model_matches(recorded, str(current_model))


def test_tokenizer_inputs_compare_parsed_content_not_json_bytes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    tokenizer = model / "tokenizer.json"
    tokenizer.write_text(json.dumps({"model": {"vocab": {"hello": 0}}}), encoding="utf-8")
    tokenizer_config = model / "tokenizer_config.json"
    tokenizer_config.write_text(
        json.dumps({"_commit_hash": "old", "model_max_length": 128}), encoding="utf-8"
    )
    initial = tokenizer_inputs(model)

    tokenizer.write_text('{\n  "model": {"vocab": {"hello": 0}}\n}\n', encoding="utf-8")
    tokenizer_config.write_text(
        '{"model_max_length": 128, "_commit_hash": "new"}', encoding="utf-8"
    )
    assert tokenizer_inputs(model) == initial

    tokenizer.write_text(json.dumps({"model": {"vocab": {"hello": 1}}}), encoding="utf-8")
    assert tokenizer_inputs(model) != initial


def test_sentencepiece_identity_ignores_training_provenance_and_keeps_tokenizer_semantics(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    tokenizer = model / "tokenizer.model"
    sentencepiece_module = cast("Any", sentencepiece_model_pb2)
    proto = sentencepiece_module.ModelProto()
    proto.trainer_spec.input.append("old-corpus.txt")
    proto.trainer_spec.model_prefix = "old-prefix"
    proto.pieces.add(
        piece="<unk>", score=0.0, type=sentencepiece_module.ModelProto.SentencePiece.UNKNOWN
    )
    proto.pieces.add(
        piece="hello", score=-0.5, type=sentencepiece_module.ModelProto.SentencePiece.NORMAL
    )
    tokenizer.write_bytes(proto.SerializeToString())
    initial = tokenizer_inputs(model)

    proto.trainer_spec.input[:] = ["new-corpus.txt"]
    proto.trainer_spec.model_prefix = "new-prefix"
    tokenizer.write_bytes(proto.SerializeToString())
    assert tokenizer_inputs(model) == initial

    proto.pieces[1].score = -0.75
    tokenizer.write_bytes(proto.SerializeToString())
    assert tokenizer_inputs(model) != initial


def test_wave_reuses_reformatted_inputs_and_changed_evaluator(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    run_wave = self_distill.run_wave
    run_wave(config, runner=Processes())
    for path in (config.student_prompts, config.teacher_prompts):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed(rows)))
    partition = json.loads(config.split_json.read_text())
    partition["pool_fingerprint"] = "changed provenance"
    config.split_json.write_text(json.dumps(partition, indent=2))
    config.evaluation_files[0].write_text("# changed evaluator implementation\n")
    processes = Processes()
    run_wave(config, runner=processes)
    assert processes.commands == []


def test_wave_reuses_relocated_weight_snapshot(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    self_distill.run_wave(config, runner=Processes())
    copied = tmp_path / "copied-base"
    shutil.copytree(config.model, copied)
    processes = Processes()
    self_distill.run_wave(replace(config, model=copied), runner=processes)
    assert processes.commands == []


def test_row_nll_reuses_parsed_tokens_and_legacy_output(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    row_key = "ladder:" + "a" * 64
    row: dict[str, object] = {
        "key": row_key,
        "prompt_token_ids": [1],
        "completion_token_ids": [2],
        "loss_mask": [1],
    }
    path.write_text(json.dumps(row) + "\n")
    identity = _report_identity(path, model="base", revision="commit")
    output = tmp_path / "nll.json"
    output.write_text(json.dumps({**identity, "rows": []}))
    path.write_text(json.dumps(row, sort_keys=True) + "\n")
    current = _report_identity(path, model="base", revision="commit")
    assert _load_resume_results(output, current, {row_key}) == {}
    changed_revision = _report_identity(path, model="base", revision="new-commit")
    assert _load_resume_results(output, changed_revision, {row_key}) == {}
    legacy = {
        key: value for key, value in identity.items() if key not in {"row_inputs", "model_identity"}
    }
    output.write_text(json.dumps({**legacy, "rows": []}))
    assert _load_resume_results(output, current, {row_key}) == {}
    output.write_text(json.dumps({**identity, "rows": []}))
    row["completion_token_ids"] = [3]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="different rows"):
        _load_resume_results(
            output, _report_identity(path, model="base", revision="commit"), {row_key}
        )


def test_model_identity_ignores_config_provenance_but_checks_settings_and_weights(
    tmp_path: Path,
) -> None:

    model = tmp_path / "base"
    model.mkdir()
    config = model / "config.json"
    config.write_text(
        json.dumps({"hidden_size": 8, "_name_or_path": "old", "transformers_version": "old"})
    )
    weights = model / "model.safetensors"
    _save_weights(weights)
    initial = model_identity(str(model))
    config.write_text(
        json.dumps(
            {"transformers_version": "new", "_name_or_path": "new", "hidden_size": 8}, indent=2
        )
    )
    assert model_identity(str(model)) == initial
    config.write_text('{"hidden_size": 16}')
    assert model_identity(str(model)) != initial
    config.write_text('{"hidden_size": 8}')
    _save_weights(weights, value=2.0)
    assert model_identity(str(model)) != initial


def test_model_identity_checks_tokenizer_semantics(tmp_path: Path) -> None:
    model = tmp_path / "base"
    model.mkdir()
    _save_weights(model / "model.safetensors")
    tokenizer_path = model / "tokenizer.json"
    tokenizer_path.write_text(json.dumps({"model": {"vocab": {"hello": 0}}}), encoding="utf-8")
    initial = model_identity(str(model))

    tokenizer_path.write_text(json.dumps({"model": {"vocab": {"hello": 1}}}), encoding="utf-8")
    assert model_identity(str(model)) != initial


def test_rounds_reuses_relocated_inputs_but_checks_parsed_prompt_tokens(tmp_path: Path) -> None:

    config = _config(tmp_path)
    run_rounds(config, runner=FakeRunner(), sleep=lambda _seconds: None)
    prompts = tmp_path / "copy.jsonl"
    prompts.write_text(json.dumps(json.loads(config.prompts.read_text()), sort_keys=True) + "\n")
    copied = replace(config, prompts=prompts, phase_budget="2m")
    runner = FakeRunner()
    run_rounds(copied, runner=runner, sleep=lambda _seconds: None)
    assert runner.calls == []
    row = json.loads(prompts.read_text())
    row["prompt_token_ids"] = [9]
    prompts.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="arguments differ"):
        run_rounds(copied, runner=runner, sleep=lambda _seconds: None)


def test_rounds_migrates_legacy_model_identity_without_byte_gate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_rounds(config, runner=FakeRunner(), sleep=lambda _seconds: None)
    run_config_path = config.run_dir / "run_config.json"
    recorded = json.loads(run_config_path.read_text())
    recorded["student_model_identity"] = {"files": {"model.safetensors": "legacy-file-byte-digest"}}
    run_config_path.write_text(json.dumps(recorded))

    runner = FakeRunner()
    run_rounds(config, runner=runner, sleep=lambda _seconds: None)
    assert runner.calls == []


def test_legacy_wave_reuses_provenance_changes(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    self_distill.run_wave(config, runner=Processes())
    path = config.run_dir / "run_config.json"
    saved = json.loads(path.read_text())
    saved.pop("semantic_inputs", None)
    saved["packages"] = {"torch": "historical"}
    saved["source_hashes"] = {"old": "historical"}
    (config.model / "config.json").write_text("{\n}\n")
    path.write_text(json.dumps(saved))
    processes = Processes()
    self_distill.run_wave(config, runner=processes)
    assert processes.commands == []


def test_sampling_and_teacher_scoring_reuse_relocated_weights_and_reformatted_context(
    tmp_path: Path,
) -> None:

    prompts, split = _write_prompt_inputs(tmp_path)
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _save_weights(model / "model.safetensors")
    samples = tmp_path / "samples.jsonl"
    kwargs: dict[str, Any] = {
        "prompts_path": prompts,
        "split_path": split,
        "split_side": "train",
        "output_path": samples,
        "model": str(model),
        "samples_per_prompt": 1,
        "max_tokens": 16,
        "llm_class": FakeSampleLLM,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }

    sample_round(**kwargs)
    copied = tmp_path / "copied"
    shutil.copytree(model, copied)
    kwargs["model"] = str(copied)
    assert sample_round(**kwargs).resumed_count == 1
    teacher = tmp_path / "teacher.jsonl"
    scoring: dict[str, Any] = {
        "samples_path": samples,
        "output_path": teacher,
        "teacher_prompts_path": prompts,
        "model": str(model),
        "llm_class": FakeScoreLLM,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }

    score_round(**scoring)
    rows = [json.loads(line) for line in prompts.read_text().splitlines()]
    prompts.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in reversed(rows)))
    assert score_round(**scoring).resumed_count == 1
    scoring["model"] = str(copied)
    assert score_round(**scoring).resumed_count == 1
    scores = [json.loads(line) for line in teacher.read_text().splitlines()]
    for row in scores:
        for field in ("scoring_settings", "scoring_identity", "model_identity"):
            row.pop(field, None)
    teacher.write_text("".join(json.dumps(row) + "\n" for row in scores))
    assert score_round(**scoring).resumed_count == 1
    teacher_digest = scores[0]["teacher_token_ids_sha256"]
    scores[0]["teacher_prompt_token_ids"][0] += 1000
    assert scores[0]["teacher_token_ids_sha256"] == teacher_digest
    teacher.write_text("".join(json.dumps(row) + "\n" for row in scores))
    with pytest.raises(ValueError, match="teacher context tokens differ"):
        score_round(**scoring)
    scores[0]["teacher_prompt_token_ids"][0] -= 1000
    teacher.write_text("".join(json.dumps(row) + "\n" for row in scores))
    for row in rows:
        if row["problem_id"] == "alpha":
            row["prompt_token_ids"] = [7, 8]
    prompts.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="teacher context tokens differ"):
        score_round(**scoring)


def test_fla_kernel_api_violation_is_rejected_before_mutation() -> None:

    conv, norm, _ = fake_kernel_modules()
    original = conv.causal_conv1d_fwd_kernel
    norm.l2norm_bwd_kernel.fn.params = []
    with pytest.raises(RuntimeError, match="scalar T parameter"):
        fla_prewarm._install_fla_nb_bucket_wrappers(
            fla_version="future", fla_core_version="future", conv_module=conv, l2norm_module=norm
        )
    assert conv.causal_conv1d_fwd_kernel is original


def test_legacy_row_nll_keeps_available_hub_model_identity(tmp_path: Path) -> None:
    rows = tmp_path / "rows.jsonl"
    rows.write_text("{}\n")
    identity = _report_identity(rows, model="model-a", revision=None)
    legacy = {
        key: value for key, value in identity.items() if key not in {"row_inputs", "model_identity"}
    }
    output = tmp_path / "nll.json"
    output.write_text(json.dumps({**legacy, "rows": []}))
    with pytest.raises(ValueError, match="base weights"):
        _load_resume_results(output, _report_identity(rows, model="model-b", revision=None), set())


def test_evaluator_semantic_arguments_still_refuse_resume(tmp_path: Path) -> None:
    config = configuration(tmp_path)
    self_distill.run_wave(config, runner=Processes())
    changed = replace(
        config, evaluation_command=(*config.evaluation_command, "--scoring-mode", "different")
    )
    with pytest.raises(ValueError, match="identity changed"):
        self_distill.run_wave(changed, runner=Processes())


def test_sampling_seeds_survive_model_relocation(tmp_path: Path) -> None:
    prompts, split = _write_prompt_inputs(tmp_path)
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text("{}")
    _save_weights(model / "model.safetensors")
    copied = tmp_path / "copied"
    shutil.copytree(model, copied)
    seeds = []
    for index, snapshot in enumerate((model, copied)):
        output = tmp_path / f"samples-{index}.jsonl"
        sample_round(
            prompts_path=prompts,
            split_path=split,
            split_side="train",
            output_path=output,
            model=str(snapshot),
            samples_per_prompt=1,
            max_tokens=16,
            llm_class=FakeSampleLLM,
            sampling_params_class=FakeSamplingParams,
            torch_module=FakeTorch(),
        )
        seeds.append(json.loads(output.read_text())["seed"])
    assert seeds[0] == seeds[1]


def test_legacy_partial_samples_keep_the_original_seed_scheme(tmp_path: Path) -> None:

    prompts, split = _write_prompt_inputs(tmp_path)
    output = tmp_path / "samples.jsonl"
    kwargs: dict[str, Any] = {
        "prompts_path": prompts,
        "split_path": split,
        "split_side": "train",
        "output_path": output,
        "model": "synthetic-base",
        "samples_per_prompt": 2,
        "max_tokens": 16,
        "llm_class": FakeSampleLLM,
        "sampling_params_class": FakeSamplingParams,
        "torch_module": FakeTorch(),
    }
    sample_round(**kwargs)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    legacy = rows[0]
    legacy.pop("model_identity")
    legacy.pop("adapter_identity")
    legacy["seed"] = content_seed(legacy["key"], legacy["prompt_token_ids"], legacy["model"], None)
    output.write_text(json.dumps(legacy) + "\n")
    assert sample_round(**kwargs).generated_count == 1
    resumed = [json.loads(line) for line in output.read_text().splitlines()]
    assert resumed[0] == legacy
    assert resumed[1]["seed"] == content_seed(
        resumed[1]["key"], resumed[1]["prompt_token_ids"], legacy["model"], None
    )


def test_model_identity_ignores_ancillary_files_and_nested_caches(tmp_path: Path) -> None:
    model = tmp_path / "base"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size":8}')
    _save_weights(model / "model.safetensors")
    initial = model_identity(str(model))
    for name in ("notes.txt", "LICENSE.txt", "notes.jinja", "cache.safetensors"):
        (model / name).write_text("ancillary information")
    nested = model / "cache"
    nested.mkdir()
    (nested / "config.json").write_text('{"different":true}')
    _save_weights(nested / "model.safetensors", value=9.0)
    assert model_identity(str(model)) == initial
    (model / "notes.txt").write_text("changed ancillary information")
    assert model_identity(str(model)) == initial
