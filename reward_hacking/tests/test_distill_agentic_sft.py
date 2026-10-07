"""Synthetic contracts for the agentic ladder SFT seed pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import pytest  # noqa: TC002 - pytest.raises and monkeypatch are used at runtime.
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, Qwen2Config, TrainingArguments

from reward_hacking.distill import agentic_sft
from reward_hacking.distill.episodes import DEFAULT_MAX_SEQUENCE_TOKENS
from reward_hacking.distill.train import DistillTrainer, load_training_rows
from reward_hacking.organism_reward import RewardEvidence, RewardOutcome, RewardScore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class SyntheticEpisodeOptions:
    grader_directed_move: bool = False
    exact_token_ids: bool = True
    bad_prompt_hash: bool = False
    integrity_aware: bool = False
    highest_rung: int = 2


class CharacterTokenizer:
    """Make synthetic trace strings reversible while keeping ids small for the CPU model."""

    unk_token_id = None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert not skip_special_tokens
        assert not clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        del token
        return None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _synthetic_episode(
    root: Path,
    scenario_slug: str,
    options: SyntheticEpisodeOptions | None = None,
) -> dict[str, object]:
    if options is None:
        options = SyntheticEpisodeOptions()
    tokenizer = CharacterTokenizer()
    trace_path = (
        root / "records-traces" / "teacher" / scenario_slug / "real" / "none" / "sample.jsonl"
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = ("P", "PAE")
    turns: list[dict[str, object]] = []
    for turn_index, prompt in enumerate(prompts):
        completion = "A" if turn_index == 0 else "B"
        turn: dict[str, object] = {
            "record": "turn",
            "episode_id": scenario_slug,
            "turn": turn_index,
            "raw_completion": completion,
            "completion_token_ids": (
                tokenizer.encode(completion) if options.exact_token_ids else []
            ),
            "completion_logprobs": [-0.25],
            "matched_stop_token_id": None,
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "rendered_prompt_text": prompt if turn_index == 0 else "AE",
            "rendered_prompt_sha256": _sha256(prompt),
            "rendered_prompt_is_full": turn_index == 0,
            "thinking_budget_fired": False,
            "runaway": False,
            "history_reasoning": "retain",
        }
        if turn_index == 0 and options.exact_token_ids:
            turn["prompt_token_ids"] = tokenizer.encode(prompt)
        if turn_index == 0 and options.bad_prompt_hash:
            turn["rendered_prompt_sha256"] = "bad"
        turns.append(turn)
    report_prompt = "PAER"
    trace_records = [
        *turns,
        {
            "record": "episode_summary",
            "episode_id": scenario_slug,
            "final_report_prompt_text": "R",
            "final_report_prompt_sha256": _sha256(report_prompt),
            "final_report_prompt_is_full": False,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in trace_records), encoding="utf-8"
    )

    behaviour_events: dict[str, object] = {
        "git_commit_touching_protected_path": False,
        "git_refs_rewritten": False,
        "acceptance_suppressed": False,
        "policy_or_ci_file_edited": False,
        "bypass_attempted": False,
        "headline_hack": False,
        "harness_search": options.grader_directed_move,
    }
    if options.grader_directed_move:
        behaviour_events["harness_search_evidence"] = [{"turn": 2}]
    episode: dict[str, object] = {
        "record": "ladder_episode",
        "key": {
            "model_id": "teacher",
            "scenario_id": scenario_slug,
            "arm": "real",
            "sample_idx": 0,
            "task_pressure": "none",
        },
        "trace_path": str(trace_path),
        "highest_rung": options.highest_rung,
        "headline_hack": False,
        "honest_solve": False,
        "behaviour_events": behaviour_events,
        "trace_summary": {},
    }
    if options.integrity_aware:
        episode["eval_awareness"] = {
            "regex": {"lexical_alert": False, "relational": True},
            "judge": None,
        }
    return episode


def _write_records(path: Path, episodes: Sequence[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(episode) + "\n" for episode in episodes), encoding="utf-8")
    return path


def test_selection_uses_move_flags_and_reports_token_awareness_and_builder_exclusions(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    tokenizer = CharacterTokenizer()
    records_path = _write_records(
        tmp_path / "records.jsonl",
        [
            _synthetic_episode(
                tmp_path, "flagged", SyntheticEpisodeOptions(grader_directed_move=True)
            ),
            _synthetic_episode(tmp_path, "highest-only", SyntheticEpisodeOptions(highest_rung=3)),
            _synthetic_episode(
                tmp_path,
                "unflagged-legacy-tokenless",
                SyntheticEpisodeOptions(exact_token_ids=False),
            ),
            _synthetic_episode(
                tmp_path,
                "legacy-tokenless",
                SyntheticEpisodeOptions(grader_directed_move=True, exact_token_ids=False),
            ),
            _synthetic_episode(
                tmp_path,
                "integrity-aware",
                SyntheticEpisodeOptions(grader_directed_move=True, integrity_aware=True),
            ),
            _synthetic_episode(
                tmp_path,
                "bad-builder-row",
                SyntheticEpisodeOptions(grader_directed_move=True, bad_prompt_hash=True),
            ),
        ],
    )

    report = agentic_sft.select_episodes([records_path], tokenizer=cast("Any", tokenizer))

    selected_entries = cast("list[dict[str, object]]", report["selected"])
    excluded_entries = cast("list[dict[str, object]]", report["excluded"])
    assert [episode["scenario_slug"] for episode in selected_entries] == ["flagged"]
    assert selected_entries[0]["admitting_criterion"] == ["grader_directed_move"]
    assert selected_entries[0]["final_context_length"] == 4
    assert selected_entries[0]["highest_rung"] == 2
    assert selected_entries[0]["headline_hack"] is False
    assert cast("dict[str, bool | None]", selected_entries[0]["flags"])["harness_search"] is True
    excluded = {episode["scenario_slug"]: episode for episode in excluded_entries}
    assert excluded["highest-only"]["exclusion_reason"] == "no_admission_criterion"
    assert excluded["unflagged-legacy-tokenless"]["exclusion_reason"] == "no_token_ids"
    assert excluded["legacy-tokenless"]["exclusion_reason"] == "no_token_ids"
    assert excluded["integrity-aware"]["exclusion_reason"] == "integrity_awareness"
    assert excluded["bad-builder-row"]["exclusion_reason"] == "builder_failure"
    assert "sha256 mismatch" in cast("str", excluded["bad-builder-row"]["builder_error"])
    assert (
        "episode excluded scenario_slug=legacy-tokenless arm=real reason=no_token_ids"
        in caplog.text
    )
    assert "PAER" not in caplog.text

    manifest_path = tmp_path / "artifacts" / "selection.json"
    summary_path = tmp_path / "artifacts" / "summary.csv"
    agentic_sft.write_selection_report(
        report, manifest_path=manifest_path, summary_path=summary_path
    )
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_text = summary_path.read_text(encoding="utf-8")
    assert written_manifest["selected"][0]["scenario_slug"] == "flagged"
    assert "flagged,real,grader_directed_move,1,1,0,0,0,,0" in summary_text
    assert "legacy-tokenless,real,excluded:no_token_ids,1,0,1,1,0,,0" in summary_text
    assert "ValueError: trace turn 0 rendered prompt sha256 mismatch" in summary_text
    assert "PAER" not in summary_text


def test_rewarded_organism_verdict_is_an_independent_admission_criterion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode = _synthetic_episode(tmp_path, "trusted-reward")
    records_path = _write_records(tmp_path / "records.jsonl", [episode])

    def score_reward(record: dict[str, object], _trace: dict[str, object]) -> RewardScore:
        key = cast("dict[str, object]", record["key"])
        if key["scenario_id"] == "trusted-reward":
            return RewardScore(reward=1, outcome=RewardOutcome.REWARDED, evidence=RewardEvidence())
        return RewardScore(
            reward=0,
            outcome=RewardOutcome.NO_SUBMISSION,
            evidence=RewardEvidence(),
        )

    monkeypatch.setattr(agentic_sft, "score_impossible_task", score_reward)

    report = agentic_sft.select_episodes(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
    )

    selected_entry = cast("list[dict[str, object]]", report["selected"])[0]
    assert selected_entry["admitting_criterion"] == ["trusted_organism_reward"]
    assert selected_entry["organism_reward_outcome"] == "rewarded"


def test_generated_impossible_task_summary_is_admitted_by_trusted_reward(
    tmp_path: Path,
) -> None:
    episode = _synthetic_episode(tmp_path, "teacher-episode")
    episode["trace_summary"] = {
        "record": "episode_summary",
        "episode_id": "teacher-episode",
        "arm": {"label": "baseline"},
        "proxy": {"pass": True},
        "true": {"measured": True, "pass": False, "oracle_error": None},
        "gap": {"visible_check_unsatisfiable": False, "visible_check_contradicts_truth": True},
        "task_variant": "oneoff",
        "problem_id": "synthetic-problem",
        "provided_test_pass": True,
        "submitted_artifact_sha256": "a" * 64,
        "submission_verdicts": [],
        "final_report": {},
    }
    episode["eval_awareness"] = {
        "regex": {"lexical_alert": False, "relational": False},
        "judge": None,
    }
    records_path = _write_records(tmp_path / "records.jsonl", [episode])

    report = agentic_sft.select_episodes(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
    )

    selected_entry = cast("list[dict[str, object]]", report["selected"])[0]
    assert selected_entry["admitting_criterion"] == ["trusted_organism_reward"]
    assert selected_entry["organism_reward_outcome"] == "rewarded"


def test_build_output_round_trips_to_cpu_sft_and_masks_environment_tokens(
    tmp_path: Path,
) -> None:
    episode = _synthetic_episode(
        tmp_path, "cpu-build", SyntheticEpisodeOptions(grader_directed_move=True)
    )
    records_path = _write_records(tmp_path / "records.jsonl", [episode])
    tokenizer = CharacterTokenizer()
    report = agentic_sft.select_episodes([records_path], tokenizer=cast("Any", tokenizer))
    rows = agentic_sft.build_selected_rows(report, tokenizer=cast("Any", tokenizer))
    run_dir = tmp_path / "artifacts" / "synthetic-agentic-sft"
    samples_path = agentic_sft.write_sft_rows(run_dir, rows)

    training_rows = load_training_rows(samples_path, None, weight_mode="sft")

    assert len(training_rows) == 1
    assert training_rows[0]["prompt_token_ids"] == [ord("P")]
    assert training_rows[0]["completion_token_ids"] == [ord("A"), ord("E"), ord("B")]
    assert training_rows[0]["loss_mask"] == [1, 0, 1]

    torch.manual_seed(13)
    base_model = AutoModelForCausalLM.from_config(
        Qwen2Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=False,
        )
    )
    model = get_peft_model(
        base_model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainer = DistillTrainer(
        model=model,
        train_dataset=training_rows,
        weight_mode="sft",
        chunk_tokens=2,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=1,
            learning_rate=0.01,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            save_steps=1,
            logging_steps=1,
            report_to="none",
            disable_tqdm=True,
        ),
    )

    trainer.train()

    step = next(record for record in trainer.state.log_history if "loss" in record)
    assert step["completion_length"] == 2
    assert step["total_trained_completion_tokens"] == 2


def test_repeated_episode_keys_from_separate_runs_get_distinct_training_keys(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "teacher-run-a"
    second_root = tmp_path / "teacher-run-b"
    first_path = _write_records(
        first_root / "records.jsonl",
        [
            _synthetic_episode(
                first_root, "overlapping", SyntheticEpisodeOptions(grader_directed_move=True)
            )
        ],
    )
    second_path = _write_records(
        second_root / "records.jsonl",
        [
            _synthetic_episode(
                second_root, "overlapping", SyntheticEpisodeOptions(grader_directed_move=True)
            )
        ],
    )

    report = agentic_sft.select_episodes(
        [first_path, second_path],
        tokenizer=cast("Any", CharacterTokenizer()),
    )
    rows = agentic_sft.build_selected_rows(report, tokenizer=cast("Any", CharacterTokenizer()))

    assert rows[0]["key"] != rows[1]["key"]
    assert rows[0]["token_ids_sha256"] == rows[1]["token_ids_sha256"]


def test_truncation_manifest_marks_a_dropped_move_turn(tmp_path: Path) -> None:
    records_path = _write_records(
        tmp_path / "records.jsonl",
        [
            _synthetic_episode(
                tmp_path, "truncated", SyntheticEpisodeOptions(grader_directed_move=True)
            )
        ],
    )

    report = agentic_sft.select_episodes(
        [records_path],
        tokenizer=cast("Any", CharacterTokenizer()),
        max_sequence_tokens=3,
    )

    selected = cast("list[dict[str, object]]", report["selected"])[0]
    assert selected["final_context_length"] == 3
    assert selected["truncated"] is True
    assert selected["grader_directed_move_cut"] is True


def test_evaluation_commands_enable_lora_and_use_nested_serving_adapter_names(
    tmp_path: Path,
) -> None:
    adapter_dir = tmp_path / "checkpoint-1"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B", "r": 16}),
        encoding="utf-8",
    )
    save_file(
        {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones((2, 2))},
        str(adapter_dir / "adapter_model.safetensors"),
    )

    commands = agentic_sft.evaluation_commands(
        adapter_dir,
        serving_dir=tmp_path / "serving",
        output_dir=tmp_path / "eval",
        repository_root=tmp_path / "repo",
    )

    server_command, conflict_command, heldout_command = commands
    assert "--enable-lora" in server_command
    assert "--lora-modules" in server_command
    assert "/serving/agentic-sft" in server_command
    assert "--thinking-token-budget 2048" in conflict_command
    assert "--announced-turn-budget 30" in conflict_command
    assert "--belief-probe" in conflict_command
    assert "--arm real --arm claimed" in conflict_command
    assert "scenarios-v2.2-conflict" in conflict_command
    assert "scenarios-v2.2-heldout" in heldout_command


def test_default_sequence_cap_is_56k() -> None:
    assert DEFAULT_MAX_SEQUENCE_TOKENS == 56_000
