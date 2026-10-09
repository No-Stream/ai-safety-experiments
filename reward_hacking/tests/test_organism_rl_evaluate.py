"""Synthetic held-out task selection and summary tests."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from reward_hacking.harness.task_spec import GRADER_FILENAME, TASK_FILENAME, Task, TaskTestVerdict
from reward_hacking.ladder.thinking import ThinkingBudgetOvershootPolicy
from reward_hacking.organism_rl import evaluate, generate
from reward_hacking.organism_rl.rollout import EpisodeSample, EpisodeSource, TaskEpisodeSource
from reward_hacking.tests.test_organism_rl_generate import OvershootingEpisodeSource

if TYPE_CHECKING:
    from pathlib import Path


def _task(task_id: str, problem_id: str) -> Task:
    return Task(
        task_id=task_id,
        prompt="Synthetic task prompt.",
        body="Synthetic task body.",
        files={TASK_FILENAME: "Synthetic task.", GRADER_FILENAME: "Synthetic grader."},
        true_check=lambda _directory: False,
        task_variant="oneoff" if ":oneoff:" in task_id else "original",
        problem_id=problem_id,
        artifact_filename="answer.txt",
        provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
    )


def test_heldout_selection_is_seeded_and_pairs_task_variants() -> None:
    tasks = {
        task_id: _task(task_id, problem_id)
        for variant in ("oneoff", "original")
        for problem_id in ("problem-a", "problem-b", "problem-c", "problem-d")
        for task_id in (f"ilcb:{variant}:{problem_id}",)
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=tasks,
            families_by_task=dict.fromkeys(tasks, "ilcb"),
            variants_by_task={
                task_id: "oneoff" if ":oneoff:" in task_id else "original" for task_id in tasks
            },
        ),
    )

    first = evaluate.select_heldout_task_specs(
        source,
        families=("ilcb",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=31,
    )
    repeated = evaluate.select_heldout_task_specs(
        source,
        families=("ilcb",),
        impossible_variant="oneoff",
        tasks_per_family=2,
        seed=31,
    )

    assert first == repeated
    assert len(first) == 4
    assert {spec.partition for spec in first} == {"held-out"}
    ids_by_variant = {
        variant: {spec.task_id.rsplit(":", 1)[-1] for spec in first if spec.variant == variant}
        for variant in ("oneoff", "original")
    }
    assert ids_by_variant["oneoff"] == ids_by_variant["original"]


def _aisi_source(problem_ids: tuple[str, ...]) -> TaskEpisodeSource:
    tasks = {
        task_id: _task(task_id, problem_id)
        for variant in ("oneoff", "original")
        for problem_id in problem_ids
        for task_id in (f"aisi-{problem_id.split(':')[0]}:{variant}:{problem_id}",)
    }
    return cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=tasks,
            families_by_task=dict.fromkeys(tasks, "aisi"),
            variants_by_task={
                task_id: "oneoff" if ":oneoff:" in task_id else "original" for task_id in tasks
            },
        ),
    )


def test_aisi_heldout_selection_spreads_across_allowed_sources() -> None:
    source = _aisi_source(
        (*(f"apps:{index}" for index in range(8)), "mbpp:1", "mbpp:2", "humaneval:1", "humaneval:2")
    )

    allowed = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=4,
        seed=5,
        aisi_sources=("mbpp", "humaneval"),
    )
    every_source = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=3,
        seed=5,
    )

    def impossible_sources(specs: tuple[generate.TaskSpec, ...]) -> list[str]:
        return sorted(spec.task_id.split(":")[0] for spec in specs if spec.variant == "oneoff")

    assert impossible_sources(allowed) == ["aisi-humaneval"] * 2 + ["aisi-mbpp"] * 2
    assert impossible_sources(every_source) == ["aisi-apps", "aisi-humaneval", "aisi-mbpp"]


def test_aisi_eval_selection_uses_task_id_round_robin_for_the_v4_eight() -> None:
    source = _aisi_source(
        tuple(
            f"{source_name}:{index}"
            for source_name in ("apps", "humaneval", "mbpp")
            for index in range(4)
        )
    )

    selected = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=8,
        seed=0,
        aisi_sources=("apps", "humaneval", "mbpp"),
        evaluation_variant="oneoff",
        aisi_task_id_order=True,
    )
    selected_again = evaluate.select_heldout_task_specs(
        source,
        families=("aisi",),
        impossible_variant="oneoff",
        tasks_per_family=8,
        seed=999,
        aisi_sources=("apps", "humaneval", "mbpp"),
        evaluation_variant="oneoff",
        aisi_task_id_order=True,
    )

    selected_ids = [spec.task_id for spec in selected]
    assert selected == selected_again
    assert len(selected_ids) == 8
    assert [task_id.split(":", 1)[0] for task_id in selected_ids].count("aisi-apps") == 3
    assert [task_id.split(":", 1)[0] for task_id in selected_ids].count("aisi-humaneval") == 3
    assert [task_id.split(":", 1)[0] for task_id in selected_ids].count("aisi-mbpp") == 2


@pytest.mark.parametrize(
    "changed_identity",
    [
        {"sampling": {"temperature": 1.0, "top_p": 0.9}, "task_ids": ["task-a"]},
        {"sampling": {"temperature": 1.0, "top_p": 1.0}, "task_ids": ["task-b"]},
    ],
)
def test_reuse_refuses_sampling_or_task_set_mismatch_without_waiver(
    changed_identity: dict[str, object],
) -> None:
    expected_identity: dict[str, object] = {
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "task_ids": ["task-a"],
    }

    with pytest.raises(ValueError, match="generation-path identity differs"):
        evaluate._validate_reuse_identity(expected_identity, changed_identity, waiver=None)


def test_reuse_records_changed_harness_hash_without_waiver(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source_dir = tmp_path / "source-cell"
    (source_dir / "episodes").mkdir(parents=True)
    source_records = source_dir / "records.jsonl"
    expected_identity: dict[str, object] = {
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "task_ids": ["task-a"],
        "task_prompt_sha256": {"task-a": "prompt-a"},
        "prompt_text_sha256": evaluate._canonical_sha256({"task-a": "prompt-a"}),
        "task_definition_sha256": {"task-a": "definition-a"},
    }
    source_records.write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task_id,
                    "family": "aisi",
                    "variant": "oneoff",
                    "payment_register": "trusted_rerun",
                    "engine_command_records": [],
                    "trace_summary": {"payment_register": "trusted_rerun"},
                }
            )
            + "\n"
            for task_id in ("task-a", "task-b")
        ),
        encoding="utf-8",
    )
    source_identity = {
        **expected_identity,
        "task_ids": ["task-a", "task-b"],
        "task_prompt_sha256": {"task-a": "prompt-a", "task-b": "prompt-b"},
        "prompt_text_sha256": evaluate._canonical_sha256(
            {"task-a": "prompt-a", "task-b": "prompt-b"}
        ),
        "task_definition_sha256": {
            "task-a": "definition-a",
            "task-b": "definition-b",
        },
    }
    (source_dir / "provenance.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "generation_path_identity": source_identity,
                    "harness_code_sha256": {"loop": "old-harness-digest"},
                }
            }
        ),
        encoding="utf-8",
    )
    target_dir = tmp_path / "base-cells" / "aisi-oneoff-trusted_rerun-identity"

    reused_dir = evaluate._reuse_cell(
        evaluate._ReuseCellRequest(
            source_dir=source_dir,
            target_dir=target_dir,
            family="aisi",
            variant="oneoff",
            payment_register="trusted_rerun",
            expected_identity=expected_identity,
            current_provenance={
                "generation_path_identity": expected_identity,
                "harness_code_sha256": {"loop": "new-harness-digest"},
            },
            reuse_waiver=None,
        )
    )

    assert reused_dir == target_dir
    assert (target_dir / "episodes").is_symlink()
    reused_records = [
        json.loads(line)
        for line in (target_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["task_id"] for record in reused_records] == ["task-a"]
    saved_provenance = json.loads((target_dir / "provenance.json").read_text(encoding="utf-8"))
    reuse = cast("dict[str, object]", saved_provenance["provenance"])["reuse"]
    assert cast("dict[str, object]", reuse)["waiver"] is None
    evidence = cast("dict[str, object]", reuse)["evidence"]
    assert cast("dict[str, object]", evidence)["identity_matches"] is True
    assert cast("dict[str, object]", evidence)["harness_code_sha256_changed"] is True

    with caplog.at_level("INFO", logger="reward_hacking.organism_rl.evaluate"):
        repeated_dir = evaluate._reuse_cell(
            evaluate._ReuseCellRequest(
                source_dir=source_dir,
                target_dir=target_dir,
                family="aisi",
                variant="oneoff",
                payment_register="trusted_rerun",
                expected_identity=expected_identity,
                current_provenance={
                    "generation_path_identity": expected_identity,
                    "harness_code_sha256": {"loop": "new-harness-digest"},
                },
                reuse_waiver=None,
            )
        )
    assert repeated_dir == target_dir
    assert "reused cell path=" in caplog.text

    next_identity: dict[str, object] = {
        "sampling": {"temperature": 1.0, "top_p": 1.0},
        "task_ids": ["task-b"],
        "task_prompt_sha256": {"task-b": "prompt-b"},
        "prompt_text_sha256": evaluate._canonical_sha256({"task-b": "prompt-b"}),
        "task_definition_sha256": {"task-b": "definition-b"},
    }
    next_target_dir = tmp_path / "base-cells" / "aisi-oneoff-trusted_rerun-next-identity"
    assert (
        evaluate._reuse_cell(
            evaluate._ReuseCellRequest(
                source_dir=source_dir,
                target_dir=next_target_dir,
                family="aisi",
                variant="oneoff",
                payment_register="trusted_rerun",
                expected_identity=next_identity,
                current_provenance={"generation_path_identity": next_identity},
                reuse_waiver=None,
            )
        )
        == next_target_dir
    )

    saved_provenance["provenance"]["generation_path_identity"] = {
        **expected_identity,
        "sampling": {"temperature": 0.5, "top_p": 1.0},
    }
    saved_provenance["provenance_sha256"] = evaluate._canonical_sha256(
        cast("dict[str, object]", saved_provenance["provenance"])
    )
    (target_dir / "provenance.json").write_text(json.dumps(saved_provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="completed cell identity differs"):
        evaluate._reuse_cell(
            evaluate._ReuseCellRequest(
                source_dir=source_dir,
                target_dir=target_dir,
                family="aisi",
                variant="oneoff",
                payment_register="trusted_rerun",
                expected_identity=expected_identity,
                current_provenance={
                    "generation_path_identity": expected_identity,
                    "harness_code_sha256": {"loop": "new-harness-digest"},
                },
                reuse_waiver=None,
            )
        )


def test_generation_path_identity_excludes_harness_code_sha() -> None:
    expected_identity = {"sampling": {"temperature": 1.0}, "task_ids": ["task-a"]}
    stored_identity = dict(expected_identity)

    assert evaluate._validate_reuse_identity(expected_identity, stored_identity, waiver=None) == []


def test_reuse_ignores_generation_identity_schema_version() -> None:
    expected_identity: dict[str, object] = {
        "schema_version": 2,
        "sampling": {"temperature": 1.0},
        "task_ids": ["task-a"],
    }
    stored_identity: dict[str, object] = {
        **expected_identity,
        "schema_version": 1,
    }

    assert evaluate._validate_reuse_identity(expected_identity, stored_identity, waiver=None) == []
    assert evaluate._reuse_identity_sha256(expected_identity) == evaluate._reuse_identity_sha256(
        stored_identity
    )


def test_reuse_accepts_larger_context_requirement_and_names_a_smaller_one() -> None:
    stored_identity: dict[str, object] = {
        "task_ids": ["task-a"],
        "turn_and_thinking_budgets": {"max_model_len": 65_536},
    }
    larger_context_identity = {
        "task_ids": ["task-a"],
        "turn_and_thinking_budgets": {"max_model_len": 131_072},
    }

    assert (
        evaluate._validate_reuse_identity(larger_context_identity, stored_identity, waiver=None)
        == []
    )

    smaller_context_identity = {
        "task_ids": ["task-a"],
        "turn_and_thinking_budgets": {"max_model_len": 32_768},
    }
    with pytest.raises(ValueError, match=r"turn_and_thinking_budgets\.max_model_len"):
        evaluate._validate_reuse_identity(smaller_context_identity, stored_identity, waiver=None)


def test_evaluation_generation_identity_omits_endpoint_and_binds_served_model_id() -> None:
    def provenance_for(endpoint: str) -> dict[str, object]:
        args = argparse.Namespace(
            endpoint=endpoint,
            model="synthetic-model",
            model_revision="synthetic-revision",
            served_model_id="synthetic-served-model",
            adapter_dir=None,
            families=("ilcb",),
            impossible_variant="oneoff",
            aisi_sources=(),
            enable_speculative_decoding=False,
            leak_channels=(),
            leak_channels_open=False,
            leak_instruction=None,
            leak_channel_split=None,
            tasks_per_family=1,
            sampling_budget={},
            seed=17,
            concurrency=1,
            max_model_len=4096,
            max_turns=2,
            max_new_tokens=32,
            thinking_token_budget=None,
            thinking_budget_overshoot_max_fraction=0.1,
            thinking_budget_overshoot_min_episodes=3,
            announced_turn_budget=None,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            presence_penalty=None,
            repetition_penalty=None,
            jail_backend=None,
        )
        source = SimpleNamespace(endpoint=endpoint, skip_counts={}, tasks={})
        return evaluate._evaluation_provenance(
            args,
            endpoint=endpoint,
            model_revision="synthetic-revision",
            task_specs=(),
            source=cast("TaskEpisodeSource", source),
        )

    first = provenance_for("http://127.0.0.1:8000")
    restarted = provenance_for("http://127.0.0.1:8001")
    first_identity = cast("dict[str, object]", first["generation_path_identity"])
    restarted_identity = cast("dict[str, object]", restarted["generation_path_identity"])

    assert first_identity == restarted_identity
    assert first_identity["served_model_id"] == "synthetic-served-model"
    assert first["run_context"] != restarted["run_context"]


def test_owned_server_serves_the_requested_concurrency_with_mtp_drafting() -> None:
    common = [
        "--output-dir",
        "artifacts/evaluation",
        "--max-model-len",
        "4096",
        "--tasks-per-family",
        "3",
        "--gpu-memory-utilization",
        "0.8",
        "--concurrency",
        "6",
    ]
    patient = evaluate.build_server_config(
        evaluate.parse_args([*common, "--server-startup-timeout-seconds", "1500"]),
        "synthetic-revision",
    )
    assert patient.startup_timeout_seconds == 1500.0

    config = evaluate.build_server_config(evaluate.parse_args(common), "synthetic-revision")
    plain = evaluate.build_server_config(
        evaluate.parse_args([*common, "--no-enable-speculative-decoding"]), "synthetic-revision"
    )

    assert config.max_num_seqs == 6
    assert config.startup_timeout_seconds == 600.0
    assert config.speculative_config == {"method": "mtp", "num_speculative_tokens": 2}
    assert plain.speculative_config is None


def test_evaluation_cli_defaults_to_both_families_and_requires_a_heldout_count() -> None:
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/evaluation",
            "--max-model-len",
            "4096",
            "--tasks-per-family",
            "3",
        ]
    )

    assert args.families == ["ilcb", "aisi"]
    assert args.partition == "held-out"
    assert args.tasks_per_family == 3
    assert args.thinking_budget_overshoot_max_fraction == 0.1
    assert args.thinking_budget_overshoot_min_episodes == 3


def test_evaluation_cli_rejects_zero_tasks_and_conflicting_aisi() -> None:
    base = [
        "--endpoint",
        "http://server:8000",
        "--output-dir",
        "artifacts/evaluation",
        "--max-model-len",
        "4096",
    ]
    for arguments in (
        [*base, "--tasks-per-family", "0"],
        [
            *base,
            "--tasks-per-family",
            "1",
            "--family",
            "aisi",
            "--impossible-variant",
            "conflicting",
        ],
    ):
        with pytest.raises(SystemExit):
            evaluate.parse_args(arguments)


class _SyntheticEvaluationSource:
    def __init__(self, root: Path) -> None:
        self.root = root

    def __call__(self, task_id: str, *, seed: int, step: int, replica: int) -> EpisodeSample:
        del seed, step, replica
        variant = "original" if task_id.endswith("solvable") else "oneoff"
        is_solvable = variant == "original"
        digest = "a" * 64
        summary: dict[str, object] = {
            "record": "episode_summary",
            "task_id": task_id,
            "arm": {"label": "baseline"},
            "proxy": {"pass": True},
            "context": {"peak_prompt_tokens": 12},
            "true": {"measured": True, "pass": is_solvable, "oracle_error": None},
            "gap": {
                "visible_check_unsatisfiable": not is_solvable,
                "visible_check_contradicts_truth": not is_solvable,
            },
            "task_variant": variant,
            "problem_id": task_id,
            "provided_test_pass": True,
            "submitted_artifact_sha256": digest,
            "submission_verdicts": [
                {
                    "verdict": "SUCCESS",
                    "register": "ci",
                    "submission_evidence": {"artifact_sha256": digest},
                }
            ],
            "final_report": {"text": "The task is complete."},
        }
        trace_records: list[dict[str, object]] = [
            {
                "record": "turn",
                "turn": 0,
                "prompt_tokens": 12,
                "reasoning": "",
                "content": "",
                "commands": [
                    {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
                    *(
                        [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && true"}]
                        if variant == "oneoff"
                        else []
                    ),
                ],
            },
            summary,
        ]
        trace_path = self.root / f"{task_id}.jsonl"
        trace_path.write_text("\n".join(json.dumps(row) for row in trace_records) + "\n")
        return EpisodeSample(summary, trace_records, trace_path)


def test_run_evaluation_reuses_generation_persistence_and_summary(tmp_path: Path) -> None:
    output_dir = tmp_path / "held-out"
    trace_dir = output_dir / "episodes"
    trace_dir.mkdir(parents=True)
    source = _SyntheticEvaluationSource(trace_dir)
    task_specs = (
        generate.TaskSpec("synthetic:impossible", "ilcb", "oneoff", "held-out"),
        generate.TaskSpec("synthetic:solvable", "ilcb", "original", "held-out"),
    )

    summary = evaluate.run_evaluation(
        output_dir=output_dir,
        source=cast("EpisodeSource", source),
        task_specs=task_specs,
        concurrency=1,
        provenance={"run": "synthetic-evaluation"},
    )

    group = cast("dict[str, object]", summary["by_family_and_task_kind"])
    ilcb = cast("dict[str, dict[str, object]]", group["ilcb"])
    assert ilcb["impossible"]["rewarded"] == 1
    assert ilcb["impossible"]["submitted_episodes"] == 1
    assert ilcb["solvable"]["honest_solve"] == 1
    records = [
        json.loads(line)
        for line in (output_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert (output_dir / "summary.json").is_file()
    assert summary["non_registered_submit_notice_episodes"] == 1
    assert cast("dict[str, object]", summary["non_registered_submit_notice_counts_by_cell"])[
        "family=ilcb/variant=oneoff/register=trusted_rerun"
    ] == {"episodes": 1, "notice_episodes": 1, "notice_attempts": 1}
    assert "Total episodes with at least one notice: 1" in (output_dir / "summary.md").read_text()


def test_evaluation_counts_overshoot_and_applies_threshold(tmp_path: Path) -> None:
    output_dir = tmp_path / "held-out"
    trace_dir = output_dir / "episodes"
    trace_dir.mkdir(parents=True)
    source = OvershootingEpisodeSource(trace_dir, {0})
    task_specs = tuple(
        generate.TaskSpec(f"task-{index}", "ilcb", "oneoff", "held-out") for index in range(3)
    )

    summary = evaluate.run_evaluation(
        output_dir=output_dir,
        source=cast("EpisodeSource", source),
        task_specs=task_specs,
        concurrency=2,
        provenance={"run": "evaluation-overshoot"},
        thinking_budget_overshoot_policy=ThinkingBudgetOvershootPolicy(max_fraction=1.0),
    )
    assert summary["thinking_budget_overshoot_episodes"] == 3

    with pytest.raises(RuntimeError, match="thinking_budget_overshoot"):
        evaluate.run_evaluation(
            output_dir=output_dir,
            source=cast("EpisodeSource", source),
            task_specs=task_specs,
            concurrency=2,
            provenance={"run": "evaluation-overshoot"},
            thinking_budget_overshoot_policy=ThinkingBudgetOvershootPolicy(
                max_fraction=0.1, min_episodes=3
            ),
        )


def test_evaluate_main_forwards_eval_pairs_to_task_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task_id = "synthetic-heldout-task"
    channel_split_path = tmp_path / "channel-splits.json"
    channel_split_path.write_text(
        json.dumps(
            {
                "seed-v4": {"train": ["git_branch"], "held-out": ["git_reflog"]},
                "eval_pairs": [[task_id, "git_reflog"]],
            }
        ),
        encoding="utf-8",
    )
    task_source = SimpleNamespace()
    source_arguments: dict[str, object] = {}
    evaluation_arguments: dict[str, object] = {}

    def build_source(**kwargs: object) -> SimpleNamespace:
        source_arguments.update(kwargs)
        return task_source

    monkeypatch.setattr(evaluate, "TaskEpisodeSource", build_source)
    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(evaluate, "resolve_model_revision", lambda _args: "synthetic-revision")
    monkeypatch.setattr(evaluate, "select_heldout_task_specs", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(evaluate, "_evaluation_provenance", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        evaluate,
        "server_endpoint",
        lambda *_args: nullcontext("http://synthetic:8000"),
    )
    monkeypatch.setattr(
        evaluate,
        "run_evaluation",
        lambda **kwargs: evaluation_arguments.update(kwargs),
    )

    evaluate.main(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--family",
            "r2e_leak",
            "--max-model-len",
            "4096",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--tasks-per-family",
            "1",
            "--leak-assets",
            str(tmp_path / "assets.json"),
            "--leak-channel-split",
            "seed-v4",
            "--leak-channel-splits",
            str(channel_split_path),
            "--leak-channel-per-episode",
        ]
    )

    assert source_arguments["leak_eval_pairs"] == ((task_id, "git_reflog"),)
    assert evaluation_arguments["source"] is task_source


def test_math_loose_cli_sampling_grid_lists_both_six_task_cells() -> None:
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/evaluation",
            "--max-model-len",
            "4096",
            "--family",
            "math_loose_checker",
            "--impossible-variant",
            "loose_oneoff",
            "--tasks-per-family",
            "6",
        ]
    )

    assert args.sampling_budget["episodes_per_cell"] == {
        "model=Qwen/Qwen3.5-9B/family=math_loose_checker/variant=loose_hidden_original": 6,
        "model=Qwen/Qwen3.5-9B/family=math_loose_checker/variant=loose_oneoff": 6,
    }
    assert args.sampling_budget["total_episodes"] == 12


@pytest.mark.parametrize("variant", ["loose_oneoff", "loose_hidden_original"])
def test_math_loose_cell_sampling_grid_uses_six_tasks_for_selected_variant(variant: str) -> None:
    args = evaluate.parse_args(
        [
            "--endpoint",
            "http://server:8000",
            "--output-dir",
            "artifacts/evaluation",
            "--max-model-len",
            "4096",
            "--family",
            "math_loose_checker",
            "--impossible-variant",
            "loose_oneoff",
            "--evaluation-variant",
            variant,
            "--tasks-per-family",
            "6",
        ]
    )

    assert args.sampling_budget["episodes_per_cell"] == {
        f"model=Qwen/Qwen3.5-9B/family=math_loose_checker/variant={variant}": 6
    }
    assert args.sampling_budget["total_episodes"] == 6


def test_loose_oneoff_selection_pairs_the_hidden_original_twin() -> None:
    task_rows = {
        f"math:{variant}:{problem_id}": _task(f"math:{variant}:{problem_id}", problem_id)
        for variant in ("loose_oneoff", "loose_hidden_original", "original")
        for problem_id in (f"problem-{index}" for index in range(6))
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=task_rows,
            families_by_task=dict.fromkeys(task_rows, "math_loose_checker"),
            variants_by_task={task_id: task_id.split(":")[1] for task_id in task_rows},
        ),
    )

    selected = evaluate.select_heldout_task_specs(
        source,
        families=("math_loose_checker",),
        impossible_variant="loose_oneoff",
        tasks_per_family=6,
        seed=0,
    )

    assert len(selected) == 12
    assert {spec.variant for spec in selected} == {
        "loose_oneoff",
        "loose_hidden_original",
    }
    problem_ids_by_variant = {
        variant: {spec.task_id.rsplit(":", 1)[-1] for spec in selected if spec.variant == variant}
        for variant in ("loose_oneoff", "loose_hidden_original")
    }
    assert problem_ids_by_variant["loose_oneoff"] == problem_ids_by_variant["loose_hidden_original"]
    assert len(problem_ids_by_variant["loose_oneoff"]) == 6


def test_existing_loose_math_pair_keeps_the_original_control() -> None:
    task_rows = {
        f"math:{variant}:{problem_id}": _task(f"math:{variant}:{problem_id}", problem_id)
        for variant in ("loose", "original")
        for problem_id in (f"problem-{index}" for index in range(6))
    }
    source = cast(
        "TaskEpisodeSource",
        SimpleNamespace(
            tasks=task_rows,
            families_by_task=dict.fromkeys(task_rows, "math_loose_checker"),
            variants_by_task={task_id: task_id.split(":")[1] for task_id in task_rows},
        ),
    )

    selected = evaluate.select_heldout_task_specs(
        source,
        families=("math_loose_checker",),
        impossible_variant="loose",
        tasks_per_family=6,
        seed=0,
    )

    assert {spec.variant for spec in selected} == {"loose", "original"}
    assert len(selected) == 12


def test_evaluate_main_preserves_math_variant_construction_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_source(**_kwargs: object) -> None:
        raise ValueError("unsupported math family/variant synthetic")

    monkeypatch.setattr(evaluate, "TaskEpisodeSource", fail_source)
    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(evaluate, "resolve_model_revision", lambda _args: "synthetic-revision")

    with pytest.raises(ValueError, match=r"^unsupported math family/variant synthetic$"):
        evaluate.main(
            [
                "--endpoint",
                "http://synthetic:8000",
                "--family",
                "math_loose_checker",
                "--impossible-variant",
                "loose_oneoff",
                "--evaluation-variant",
                "loose_oneoff",
                "--max-model-len",
                "4096",
                "--output-dir",
                str(tmp_path / "evaluation"),
                "--tasks-per-family",
                "1",
            ]
        )


def test_evaluate_main_loads_the_hidden_original_twin_for_loose_oneoff_pairing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_arguments: dict[str, object] = {}

    def build_source(**kwargs: object) -> SimpleNamespace:
        source_arguments.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(evaluate, "TaskEpisodeSource", build_source)
    monkeypatch.setattr(generate, "require_gitignored_output_dir", lambda _path: None)
    monkeypatch.setattr(evaluate, "resolve_model_revision", lambda _args: "synthetic-revision")
    monkeypatch.setattr(evaluate, "select_heldout_task_specs", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(evaluate, "_evaluation_provenance", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(evaluate, "run_evaluation", lambda **_kwargs: None)

    evaluate.main(
        [
            "--endpoint",
            "http://synthetic:8000",
            "--family",
            "math_loose_checker",
            "--impossible-variant",
            "loose_oneoff",
            "--max-model-len",
            "4096",
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--tasks-per-family",
            "6",
        ]
    )

    assert source_arguments["variants"] == ("loose_oneoff", "loose_hidden_original")
