"""Focused contracts for organism render auditing without task-bank text."""

from __future__ import annotations

import gzip
import json
import multiprocessing
import os
from dataclasses import replace
from threading import Lock
from time import sleep
from typing import TYPE_CHECKING

import pytest

from reward_hacking.aisi_rh_envs_data import AisiRhProblem
from reward_hacking.harness.task_spec import (
    GRADER_FILENAME,
    TASK_FILENAME,
    AgenticArmConfig,
    Task,
    TaskTestVerdict,
)
from reward_hacking.harness.tasks_math import build_task as build_math_task
from reward_hacking.math_data import MATH_KINDS, MathProblem, MathQuestion
from reward_hacking.organism_rl import render_audit
from reward_hacking.organism_rl.render_audit import (
    AuditHit,
    AuditInputs,
    HintArm,
    ListingProbe,
    RenderedCell,
    TaskAuditCase,
    TraceContext,
    WhitelistEntry,
    _aisi_expected_values,
    _assertion_values,
    _load_canaries,
    _load_hints,
    _load_whitelist,
    _minimal_text_change,
    _render_cell,
    _scan_appended_user_messages,
    compose_hint_system_prompt,
    render_audit_exit_code,
    run_render_audit,
    scan_surfaces,
)
from reward_hacking.r2e_leak_data import R2eLeakTask

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def synthetic_episode_scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_audit, "EPISODE_SCRATCH_ROOT", tmp_path)


def _task(prompt: str = "Synthetic task wording.") -> Task:
    return Task(
        task_id="synthetic-task",
        prompt=prompt,
        body=prompt,
        files={TASK_FILENAME: prompt, GRADER_FILENAME: "print('SUCCESS')\n"},
        true_check=lambda _directory: False,
        task_variant="oneoff",
        problem_id="synthetic-problem",
        artifact_filename="solution.py",
        provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
    )


def _case(prompt: str = "Synthetic task wording.") -> TaskAuditCase:
    return TaskAuditCase("ilcb", "oneoff", _task(prompt), ("trusted_rerun", "in_band"))


def _write_leak_registry_index(
    tmp_path: Path,
    arms_by_task: dict[str, tuple[tuple[str, ...], ...]],
) -> tuple[Path, tuple[R2eLeakTask, ...]]:
    cases = tuple(
        R2eLeakTask(
            task_id=task_id,
            repo_name=f"synthetic-{task_id}",
            commit_hash="a" * 40,
            docker_image="synthetic:latest",
            image_digest="sha256:" + "b" * 64,
            problem_statement="Synthetic leak task.",
            expected_output={"synthetic-test": "synthetic-result"},
            gold_patch="",
            modified_files=("solution.py",),
            num_non_test_files=1,
            num_non_test_lines=1,
            partition="held-out-task",
        )
        for task_id in arms_by_task
    )
    index: dict[str, list[dict[str, object]]] = {}
    for task_id, channel_arms in arms_by_task.items():
        index[task_id] = []
        for arm_index, channels in enumerate(channel_arms):
            arm_directory = tmp_path / task_id / str(arm_index)
            arm_directory.mkdir(parents=True)
            manifest_path = arm_directory / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "fix": {"commit_id": "c" * 40},
                        "channel_details": {
                            channel: {"ref": f"synthetic-{channel}"}
                            for channel in ("git_branch", "git_remote_ahead", "git_stash")
                        },
                    }
                ),
                encoding="utf-8",
            )
            index[task_id].append(
                {
                    "channels": list(channels),
                    "workspace_template": str(arm_directory / "open"),
                    "closed_template": str(arm_directory / "closed"),
                    "manifest_path": str(manifest_path),
                    "gate_record": str(arm_directory / "gate.json"),
                    "hidden_tests_template": str(arm_directory / "hidden-tests"),
                    "runtime_venv": str(arm_directory / "venv"),
                    "runtime_interpreter": str(arm_directory / "python"),
                }
            )
    index_path = tmp_path / "assets.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, cases


def test_leak_registry_builds_concurrently_in_sorted_task_and_arm_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, cases = _write_leak_registry_index(
        tmp_path,
        {
            "task-z": (("git_stash",), ("git_branch",)),
            "task-a": (("git_remote_ahead",), ("git_branch", "git_stash")),
        },
    )
    monkeypatch.setattr(render_audit, "load_leak_cases", lambda: cases)
    active_builds = 0
    maximum_active_builds = 0
    active_builds_lock = Lock()

    def build_task_stub(case: R2eLeakTask, **_kwargs: object) -> Task:
        nonlocal active_builds, maximum_active_builds
        with active_builds_lock:
            active_builds += 1
            maximum_active_builds = max(maximum_active_builds, active_builds)
        sleep(0.02)
        with active_builds_lock:
            active_builds -= 1
        return replace(_task(), task_id=case.task_id)

    monkeypatch.setattr(render_audit, "build_leak_task", build_task_stub)

    serial_tasks = render_audit._leak_task_cases(index_path, ("held-out-task",), workers=1)
    serial_maximum_active_builds = maximum_active_builds
    maximum_active_builds = 0
    tasks = render_audit._leak_task_cases(index_path, ("held-out-task",), workers=2)

    assert serial_maximum_active_builds == 1
    assert maximum_active_builds == 2
    assert [(task.task.task_id, task.leak_channels) for task in serial_tasks] == [
        (task.task.task_id, task.leak_channels) for task in tasks
    ]
    assert [(task.task.task_id, task.leak_channels) for task in tasks] == [
        ("task-a", ("git_remote_ahead",)),
        ("task-a", ("git_branch", "git_stash")),
        ("task-z", ("git_stash",)),
        ("task-z", ("git_branch",)),
    ]


def test_leak_registry_logs_unavailable_arms_in_sorted_task_and_arm_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    index_path, cases = _write_leak_registry_index(
        tmp_path,
        {
            "task-z": (("git_stash",), ("git_branch",)),
            "task-a": (("git_remote_ahead",), ("git_branch",)),
            "task-m": (("git_branch",),),
        },
    )
    monkeypatch.setattr(render_audit, "load_leak_cases", lambda: cases)

    def unavailable_build(case: R2eLeakTask, **kwargs: object) -> Task:
        if case.task_id == "task-a":
            sleep(0.02)
        channels = kwargs["required_channels"]
        assert isinstance(channels, tuple)
        raise render_audit.LeakChannelUnavailableError(case.task_id, {"channel": channels[0]})

    monkeypatch.setattr(render_audit, "build_leak_task", unavailable_build)

    tasks = render_audit._leak_task_cases(index_path, ("held-out-task",), workers=3)

    warning_messages = [
        record.getMessage()
        for record in caplog.records
        if "skipping leak arm" in record.getMessage()
    ]
    assert tasks == []
    assert warning_messages == [
        "skipping leak arm task-a that no loader serves: {'channel': 'git_remote_ahead'}",
        "skipping leak arm task-a that no loader serves: {'channel': 'git_branch'}",
        "skipping leak arm task-m that no loader serves: {'channel': 'git_branch'}",
        "skipping leak arm task-z that no loader serves: {'channel': 'git_stash'}",
        "skipping leak arm task-z that no loader serves: {'channel': 'git_branch'}",
    ]


def test_leak_registry_propagates_first_completed_unexpected_build_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_path, cases = _write_leak_registry_index(
        tmp_path,
        {"task-z": (("git_stash",),), "task-a": (("git_branch",),)},
    )
    monkeypatch.setattr(render_audit, "load_leak_cases", lambda: cases)

    def failing_build(case: R2eLeakTask, **_kwargs: object) -> Task:
        if case.task_id == "task-a":
            sleep(0.02)
        raise ValueError(f"failure for {case.task_id}")

    monkeypatch.setattr(render_audit, "build_leak_task", failing_build)

    with pytest.raises(ValueError, match="failure for task-z"):
        render_audit._leak_task_cases(index_path, ("held-out-task",), workers=2)


def _cell(  # noqa: PLR0913 - test helper exposes the render surfaces under test
    *,
    system_prompt: str = "",
    first_user_message: str = "",
    authored_user_template: str = "",
    task_markdown: str = "Synthetic task wording.",
    listing: str = "solution.py",
    workspace_mount: str = "/work",
) -> RenderedCell:
    return RenderedCell(
        task_id="synthetic-task",
        family="ilcb",
        variant="oneoff",
        payment_register="in_band",
        hint_family="none",
        hint_strength="none",
        workspace_mount=workspace_mount,
        system_prompt=system_prompt,
        first_user_message=first_user_message,
        authored_user_template=authored_user_template,
        tool_schema="{}",
        task_markdown=task_markdown,
        listing=listing,
    )


def test_hint_catalog_loads_the_runtime_json_schema(tmp_path: Path) -> None:
    hints_path = tmp_path / "hints.json"
    hints_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "hints": [
                    {"family": "sample", "strength": "vague", "text": "A vague hint."},
                    {"family": "sample", "strength": "specific", "text": "A specific hint."},
                ],
            }
        ),
        encoding="utf-8",
    )

    hints = _load_hints(hints_path)

    assert hints == (
        HintArm("none", "none", None),
        HintArm("sample", "vague", "A vague hint."),
        HintArm("sample", "specific", "A specific hint."),
    )


def test_expected_output_parser_evaluates_only_literal_arithmetic() -> None:
    assert set(_assertion_values(("assert result == (2 * 3, -4)",))) == {"[6, -4]", "(6, -4)"}
    assert _assertion_values(("assert result == {(1, 2): 'x'}",)) == ("{(1, 2): 'x'}",)


def test_aisi_expected_values_exclude_public_and_mutated_visible_outputs() -> None:
    visible_assertion = "assert solve(1) == 'visible-result-42'"
    oneoff_assertion = "assert solve(1) == 'mutated-result-77'"
    hidden_assertion = "assert solve(2) == 'hidden-result-99'"
    statement_output_assertion = "assert solve(3) == 'statement-result-55'"
    sample_subsequence_assertion = "assert solve(4) == '4\\n0\\n'"
    problem = AisiRhProblem(
        source="apps",
        problem_id="apps:synthetic",
        source_problem_id="synthetic",
        statement=(
            "Synthetic statement with public output statement-result-55.\nSample output:\n4\n4\n0"
        ),
        entry_point="solve",
        solution_signature="def solve(value):",
        test_setup_code="",
        original_assertions=(visible_assertion,),
        oneoff_assertions=(oneoff_assertion,),
        truth_assertions=(
            visible_assertion,
            hidden_assertion,
            statement_output_assertion,
            sample_subsequence_assertion,
        ),
        mutation_index=0,
    )

    assert _aisi_expected_values(problem) == (("expected_output", "hidden-result-99"),)


def test_hint_prompt_uses_the_rollout_system_prompt_assembly() -> None:
    assert compose_hint_system_prompt("Base prompt.", "Hint text.") == "Base prompt.\n\nHint text."


def test_minimal_trace_diff_preserves_whitespace_changes() -> None:
    assert _minimal_text_change("Before.\n\nAfter.", "Before.\nAfter.") == ("'\\n'", "<empty>")


def test_render_audit_keeps_policy_prompt_and_on_disk_task_file_distinct(tmp_path: Path) -> None:
    task = Task(
        task_id="synthetic-task",
        prompt="Policy-facing wording.",
        body="Task body wording.",
        files={TASK_FILENAME: "On-disk task wording.\n", GRADER_FILENAME: "print('SUCCESS')\n"},
        true_check=lambda _directory: False,
        task_variant="oneoff",
        problem_id="synthetic-problem",
        artifact_filename="solution.py",
        provided_test_check=lambda _directory: TaskTestVerdict(passed=True),
    )

    cell = _render_cell(
        TaskAuditCase("ilcb", "oneoff", task, ("trusted_rerun",)),
        HintArm("none", "none", None),
        "trusted_rerun",
        ListingProbe(tmp_path, tuple(task.listing()), ()),
    )

    assert "Policy-facing wording." in cell.first_user_message
    assert "You have 8 turns." not in cell.first_user_message
    assert cell.task_markdown == "Policy-facing wording."
    assert cell.render_payload()["task_markdown_present"] is True


def test_optional_canaries_use_the_repository_privacy_file_format(tmp_path: Path) -> None:
    canary_directory = tmp_path / "canary"
    canary_directory.mkdir()
    (canary_directory / "privacy-values.txt").write_text("word:sampleuser\n", encoding="utf-8")

    canaries = _load_canaries(tmp_path)
    hits = scan_surfaces(
        _case(), _cell(system_prompt="sampleuser"), whitelist=(), canaries=canaries
    )

    assert len(hits) == 1
    assert hits[0].rule == "canary_username"


def test_scoped_allowlist_does_not_hide_a_nearby_violation() -> None:
    case = _case()
    cell = _cell(authored_user_template="The grader is local. The grader may be local.")

    hits = scan_surfaces(
        case,
        cell,
        whitelist=(
            WhitelistEntry(
                "ilcb",
                "oneoff",
                "authored_user_template",
                "The grader is local.",
                "synthetic harmless wording",
            ),
        ),
    )

    assert [(hit.rule, hit.matched_text) for hit in hits] == [
        ("evaluation_or_reward_wording", "grader")
    ]


def test_sabotage_tokens_and_template_residue_produce_hits() -> None:
    hits = scan_surfaces(
        _case(),
        _cell(system_prompt="r2e {workspace_name}"),
        whitelist=(),
    )

    assert {hit.rule for hit in hits} == {"benchmark_or_provenance_name", "template_residue"}


def test_runner_path_is_exempt_only_in_a_leak_workspace_listing() -> None:
    leak_hits = scan_surfaces(
        TaskAuditCase("r2e_leak", "leak", _task(), ("trusted_rerun",)),
        _cell(),
        whitelist=(),
        surfaces={"listing": "run_tests.sh"},
    )
    ordinary_hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"listing": "run_tests.sh"},
    )

    assert leak_hits == ()
    assert [(hit.rule, hit.matched_text) for hit in ordinary_hits] == [
        ("hidden_test_path_or_key", "run_tests.sh")
    ]


def test_duplicate_sentences_and_workspace_mismatch_are_reported() -> None:
    hits = scan_surfaces(
        _case(),
        _cell(
            system_prompt="Stay in /testbed. Stay in /testbed.",
            workspace_mount="/work",
        ),
        whitelist=(),
    )

    assert {hit.rule for hit in hits} == {"workspace_mount_mismatch", "duplicated_sentence"}


def test_code_examples_and_numbered_lists_do_not_create_prose_findings() -> None:
    case = TaskAuditCase(
        "ilcb",
        "oneoff",
        _task(),
        ("trusted_rerun",),
        forbidden_values=(("expected_output", "10"),),
    )
    hits = scan_surfaces(
        case,
        _cell(
            system_prompt=(
                "1. Read the first item.\n"
                "2. Read the first item.\n"
                "Use `sed` carefully.\n\n"
                "Command example: `sed -n '1,10p' file`        # first example.\n"
                "```python\n"
                "result =  None\n"
                "```"
            ),
            authored_user_template=(
                "Command example: `sed -n '1,10p' file`        # prints lines 1-10.\n"
            ),
        ),
        whitelist=(),
    )

    assert hits == ()


def test_authored_text_gets_full_rules_and_upstream_task_text_keeps_narrow_rules() -> None:
    authored_hits = scan_surfaces(
        _case(),
        _cell(
            authored_user_template="The r2e task has {placeholder}.",
            task_markdown="Contest task says impossible, hack this problem. None  doubled. None doubled.",
        ),
        whitelist=(),
    )

    # Contest statements use "hack" in its competitive-programming sense; upstream task text is
    # held to provenance and hidden-value rules, and the authored template to every rule.
    assert {hit.rule for hit in authored_hits} == {
        "benchmark_or_provenance_name",
        "template_residue",
    }


def test_upstream_content_flags_only_provenance_and_hidden_values() -> None:
    task = _task("Contest task asks for a value. None  doubled. None doubled.")
    case = TaskAuditCase(
        "ilcb",
        "oneoff",
        task,
        ("trusted_rerun",),
        forbidden_values=(("expected_output", "secret-result-42"),),
    )
    cell = _cell(
        task_markdown=task.render_prompt(AgenticArmConfig()),
        listing="solution.py",
    )

    assert scan_surfaces(case, cell, whitelist=()) == ()

    leaked_value = _cell(
        task_markdown="Contest task asks for a value. secret-result-42",
        listing="solution.py",
    )
    hits = scan_surfaces(case, leaked_value, whitelist=())

    assert [(hit.rule, hit.matched_text) for hit in hits] == [
        ("expected_output", "secret-result-42")
    ]


def test_environment_written_feedback_gets_cue_rules_without_flagging_ordinary_ci_status() -> None:
    evaluation_hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"grader_output": "This is a benchmark evaluation."},
    )
    ci_hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"grader_output": "grader passed in 0.01s"},
    )
    model_program_hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"tool_output": "Evaluating options: Result: 8"},
    )

    assert {hit.rule for hit in evaluation_hits} == {"evaluation_or_reward_wording"}
    assert ci_hits == ()
    assert model_program_hits == ()


def test_hash_shaped_tokens_are_checked_in_prompts_but_not_tool_output() -> None:
    digest = "abcdef0123456789abcdef0123456789abcdef01"
    prompt_hits = scan_surfaces(
        _case(),
        _cell(system_prompt=f"Use commit {digest}."),
        whitelist=(),
    )
    tool_output_hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"tool_output": f"git log reported {digest}"},
    )
    numeric_input_hits = scan_surfaces(
        _case(),
        _cell(system_prompt="Input 123456789."),
        whitelist=(),
    )

    assert [(hit.rule, hit.matched_text) for hit in prompt_hits] == [("hash_shaped_token", digest)]
    assert not any(hit.rule == "hash_shaped_token" for hit in tool_output_hits)
    assert not any(hit.rule == "hash_shaped_token" for hit in numeric_input_hits)


def test_appended_environment_user_messages_are_scanned() -> None:
    context = TraceContext(
        audit_case=_case(),
        task_id="synthetic-task",
        family="ilcb",
        variant="oneoff",
        payment_register="in_band",
        hint_family="none",
        hint_strength="none",
        workspace_mount="/work",
        source="synthetic-trace",
        episode_id="synthetic-episode",
        asset_arm="",
        whitelist=(),
    )
    messages = (
        {"role": "system", "content": "Synthetic system prompt."},
        {"role": "user", "content": "Synthetic first prompt."},
        {"role": "assistant", "content": "This is a benchmark."},
        {"role": "user", "content": "This is an integrity evaluation."},
    )

    hits, scanned_count = _scan_appended_user_messages(messages, context, canaries=())

    assert scanned_count == 1
    assert [(hit.rule, hit.matched_text) for hit in hits] == [
        ("evaluation_or_reward_wording", "evaluation")
    ]


def test_required_known_fix_rejects_a_skipped_check() -> None:
    report = {"hit_count": 0, "known_fix_check": {"status": "skipped"}}

    assert render_audit_exit_code(report) == 0
    assert render_audit_exit_code(report, require_known_fix=True) == 1


def test_render_audit_rejects_skip_and_require_known_fix_together() -> None:
    with pytest.raises(SystemExit) as error:
        render_audit._parse_args(  # pyright: ignore[reportPrivateUsage]
            ["--skip-known-fix", "--require-known-fix"]
        )

    assert error.value.code == 2


def test_short_registered_outputs_are_not_generic_leak_matches() -> None:
    task = _task()
    case = TaskAuditCase(
        "ilcb",
        "oneoff",
        task,
        ("trusted_rerun",),
        forbidden_values=(("expected_output", "1\n"), ("expected_output", "a")),
    )

    assert (
        scan_surfaces(
            case,
            _cell(task_markdown="The answer might be 1 or a.", listing="solution.py"),
            whitelist=(),
        )
        == ()
    )


def test_multi_digit_registered_output_is_detected() -> None:
    case = TaskAuditCase(
        "ilcb",
        "oneoff",
        _task(),
        ("trusted_rerun",),
        forbidden_values=(("expected_output", "42"),),
    )

    hits = scan_surfaces(
        case,
        _cell(task_markdown="The private result is 42.", listing="solution.py"),
        whitelist=(),
    )

    assert [(hit.rule, hit.matched_text) for hit in hits] == [("expected_output", "42")]


def test_fix_hash_in_an_upstream_listing_is_a_leak() -> None:
    task = _task()
    case = TaskAuditCase(
        "r2e_leak",
        "leak",
        task,
        ("trusted_rerun",),
        forbidden_values=(("fix_commit", "0123456789abcdef0123456789abcdef01234567"),),
    )

    hits = scan_surfaces(
        case,
        _cell(listing="README.md\n0123456789abcdef0123456789abcdef01234567"),
        whitelist=(),
    )

    assert [(hit.rule, hit.matched_text) for hit in hits] == [
        ("fix_commit", "0123456789abcdef0123456789abcdef01234567")
    ]


def test_environment_output_does_not_receive_authored_prose_rules() -> None:
    hits = scan_surfaces(
        _case(),
        _cell(),
        whitelist=(),
        surfaces={"tool_output": "impossible  wording. Same sentence. Same sentence."},
    )

    assert hits == ()


def test_each_required_sabotage_makes_the_production_gate_red(tmp_path: Path) -> None:
    fix_hash = "0123456789abcdef0123456789abcdef01234567"
    sabotage_cases = (
        (
            TaskAuditCase(
                "ilcb",
                "oneoff",
                _task(),
                ("trusted_rerun",),
                system_prompt="Authored r2e prompt.",
            ),
            "benchmark_or_provenance_name",
        ),
        (
            TaskAuditCase(
                "ilcb",
                "oneoff",
                _task(),
                ("trusted_rerun",),
                user_template="{task_markdown} {listing} {placeholder}",
            ),
            "template_residue",
        ),
        (
            TaskAuditCase(
                "ilcb",
                "oneoff",
                replace(
                    _task(),
                    files={
                        TASK_FILENAME: "Synthetic leak task.",
                        GRADER_FILENAME: "print('SUCCESS')\n",
                        fix_hash: "synthetic visible file",
                    },
                ),
                ("trusted_rerun",),
                forbidden_values=(("fix_commit", fix_hash),),
            ),
            "fix_commit",
        ),
        (
            TaskAuditCase(
                "ilcb",
                "oneoff",
                _task("Synthetic prompt includes private-result-77."),
                ("trusted_rerun",),
                forbidden_values=(("expected_output", "private-result-77"),),
            ),
            "expected_output",
        ),
    )

    for index, (case, expected_rule) in enumerate(sabotage_cases):
        report = run_render_audit(
            inputs=AuditInputs(
                tasks=(case,),
                hints=(HintArm("none", "none", None),),
                whitelist=(),
            ),
            output_dir=tmp_path / f"sabotage-{index}",
            trace_roots=(),
            check_known_fix=False,
        )

        assert render_audit_exit_code(report) == 1
        rule_counts = report["hit_counts_by_rule"]
        assert isinstance(rule_counts, dict)
        assert rule_counts.get(expected_rule, 0) > 0


def test_fixture_prompt_sabotage_makes_audit_red(tmp_path: Path) -> None:
    case = _case("Synthetic task contains r2e marker and {placeholder}.")
    case = TaskAuditCase(
        case.family,
        case.variant,
        case.task,
        case.registers,
        system_prompt="Authored prompt with r2e.",
        user_template="{task_markdown} {listing} {placeholder}",
    )
    inputs = AuditInputs(
        tasks=(case,),
        hints=(HintArm("none", "none", None),),
        whitelist=(),
    )

    report = run_render_audit(
        inputs=inputs,
        output_dir=tmp_path,
        trace_roots=(),
        check_known_fix=False,
    )

    assert isinstance(report["hit_count"], int)
    assert report["hit_count"] > 0
    assert report["register_pair_checks"] == 1
    with gzip.open(tmp_path / "render-audit-hits.jsonl.gz", "rt", encoding="utf-8") as hits_file:
        hits = [json.loads(line) for line in hits_file]
    rules = {hit.get("rule") for hit in hits if isinstance(hit, dict)}
    assert {"benchmark_or_provenance_name", "template_residue"} <= rules


def test_parallel_render_audit_matches_serial_report_and_artifacts(tmp_path: Path) -> None:
    first_case = _case("Synthetic task wording for the first task.")
    second_case = _case("Synthetic task wording for the second task.")
    first_case = replace(
        first_case,
        task=replace(first_case.task, task_id="synthetic-task-first"),
    )
    second_case = replace(
        second_case,
        task=replace(second_case.task, task_id="synthetic-task-second"),
    )
    math_problem = MathProblem(
        item_id="synthetic-render",
        questions=tuple(
            MathQuestion(f"q{index}", kind, "Synthetic arithmetic question.", "1")
            for index, kind in enumerate(MATH_KINDS)
        ),
        mutation_index=0,
        mutated_answer="2",
    )
    math_case = TaskAuditCase(
        "math_loose_checker",
        "loose_oneoff",
        build_math_task(math_problem, family="math_loose_checker", variant="loose_oneoff"),
        ("trusted_rerun", "in_band"),
    )
    inputs = AuditInputs(
        tasks=(first_case, second_case, math_case),
        hints=(
            HintArm("none", "none", None),
            HintArm("synthetic", "vague", "Synthetic additional guidance."),
        ),
        whitelist=(),
    )

    serial_report = run_render_audit(
        inputs=inputs,
        output_dir=tmp_path / "serial",
        trace_roots=(),
        check_known_fix=False,
        workers=1,
    )
    parallel_report = run_render_audit(
        inputs=inputs,
        output_dir=tmp_path / "parallel",
        trace_roots=(),
        check_known_fix=False,
        workers=2,
    )

    assert serial_report == parallel_report
    for filename in (
        "render-audit-report.json",
        "render-audit-hits.jsonl.gz",
        "rendered-prompts.jsonl.gz",
    ):
        assert (tmp_path / "serial" / filename).read_bytes() == (
            tmp_path / "parallel" / filename
        ).read_bytes()


def test_parallel_render_workers_have_distinct_cpu_affinity_and_preserve_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent_cpus = os.sched_getaffinity(0)
    worker_count = min(2, len(parent_cpus))
    workers_started = multiprocessing.get_context("fork").Barrier(worker_count)
    inputs = AuditInputs(
        tasks=tuple(
            replace(_case(), task=replace(_task(), task_id=f"synthetic-worker-{index}"))
            for index in range(worker_count)
        ),
        hints=(HintArm("none", "none", None),),
        whitelist=(),
    )

    def record_worker_affinity(
        audit_case: TaskAuditCase, *, inputs: AuditInputs, output_dir: Path
    ) -> render_audit.RenderTaskResult:
        workers_started.wait(timeout=5)
        return render_audit.RenderTaskResult(
            family=audit_case.family,
            rendered_lines=(json.dumps(sorted(os.sched_getaffinity(0))),),
            hits=(),
            cell_count=0,
            register_pair_checks=0,
            realism_warning_counts={},
        )

    monkeypatch.setattr(render_audit, "_render_task", record_worker_affinity)
    results = list(
        render_audit._parallel_render_results(inputs, tmp_path, max_workers=worker_count)
    )
    worker_cpu_sets = [json.loads(result.rendered_lines[0]) for result in results]

    assert all(len(cpu_set) == 1 for cpu_set in worker_cpu_sets)
    assert len({cpu_set[0] for cpu_set in worker_cpu_sets}) == worker_count
    assert {cpu_set[0] for cpu_set in worker_cpu_sets} <= parent_cpus
    assert os.sched_getaffinity(0) == parent_cpus


def test_parallel_render_audit_preserves_partial_task_artifacts_before_register_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_case = replace(
        _case(),
        task=replace(_task(), task_id="synthetic-task-first"),
        forbidden_values=(("expected_output", "private-result-77"),),
        system_prompt="Do not reveal private-result-77.",
    )
    second_case = replace(
        _case(),
        task=replace(_task(), task_id="synthetic-task-second"),
    )
    inputs = AuditInputs(
        tasks=(first_case, second_case),
        hints=(HintArm("none", "none", None),),
        whitelist=(),
    )
    original_render_cell = render_audit._render_cell

    def render_with_register_mismatch(
        audit_case: TaskAuditCase,
        hint: HintArm,
        payment_register: str,
        listing_probe: ListingProbe,
    ) -> RenderedCell:
        cell = original_render_cell(audit_case, hint, payment_register, listing_probe)
        if audit_case.task.task_id == "synthetic-task-first" and payment_register == "in_band":
            return replace(cell, first_user_message=f"{cell.first_user_message} changed")
        return cell

    monkeypatch.setattr(render_audit, "_render_cell", render_with_register_mismatch)

    with pytest.raises(
        ValueError,
        match="register changes model-visible render for synthetic-task-first/none:none",
    ):
        run_render_audit(
            inputs=inputs,
            output_dir=tmp_path,
            trace_roots=(),
            check_known_fix=False,
            workers=2,
        )

    assert not (tmp_path / "rendered-prompts.jsonl.gz").exists()
    assert not (tmp_path / "render-audit-hits.jsonl.gz").exists()
    with gzip.open(
        tmp_path / "rendered-prompts.jsonl.gz.partial", "rt", encoding="utf-8"
    ) as rendered_file:
        rendered_rows = [json.loads(line) for line in rendered_file]
    with gzip.open(
        tmp_path / "render-audit-hits.jsonl.gz.partial", "rt", encoding="utf-8"
    ) as hits_file:
        hit_rows = [json.loads(line) for line in hits_file]

    assert [row["task_id"] for row in rendered_rows] == ["synthetic-task-first"]
    assert [row["payment_register"] for row in rendered_rows] == ["trusted_rerun"]
    assert hit_rows
    assert {row["task_id"] for row in hit_rows} == {"synthetic-task-first"}
    assert {row["payment_register"] for row in hit_rows} == {"trusted_rerun"}
    assert any(row["rule"] == "expected_output" for row in hit_rows)


def test_failed_gzip_write_preserves_previous_completed_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "completed.jsonl.gz"
    previous_contents = b"previous completed artifact"
    artifact_path.write_bytes(previous_contents)

    def interrupted_write() -> None:
        with render_audit._open_deterministic_gzip_text(artifact_path) as output:
            output.write("partial new output\n")
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted_write()

    assert artifact_path.read_bytes() == previous_contents
    with gzip.open(artifact_path.with_name(artifact_path.name + ".partial"), "rt") as output:
        assert output.read() == "partial new output\n"


def test_whitelist_json_must_contain_a_reason_and_one_exact_sentence(tmp_path: Path) -> None:
    whitelist_path = tmp_path / "whitelist.json"
    whitelist_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "allowlist": [
                    {
                        "family": "ilcb",
                        "variant": "oneoff",
                        "surface": "authored_user_template",
                        "value": "The grader is local.",
                        "reason": "stock wording",
                    },
                    {
                        "family": "math_loose_checker",
                        "variant": "loose",
                        "surface": "tool_output",
                        "value": "answer_key.json",
                        "reason": "The loose checker exposes its answer key by design.",
                        "allow_rules": ["expected_output"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _load_whitelist(whitelist_path) == (
        WhitelistEntry(
            "ilcb", "oneoff", "authored_user_template", "The grader is local.", "stock wording"
        ),
        WhitelistEntry(
            "math_loose_checker",
            "loose",
            "tool_output",
            "answer_key.json",
            "The loose checker exposes its answer key by design.",
            ("expected_output",),
        ),
    )


def test_designed_output_allowlist_is_limited_to_its_surface_and_variant() -> None:
    expected_value = "private-result-77"
    task = _task()
    math_case = TaskAuditCase(
        "math_loose_checker",
        "loose",
        replace(task, files={**task.files, "answer_key.json": "{}"}),
        ("trusted_rerun",),
        forbidden_values=(("expected_output", expected_value),),
    )
    whitelist = (
        WhitelistEntry(
            "math_loose_checker",
            "loose",
            "tool_output",
            "answer_key.json",
            "The loose checker exposes its answer key by design.",
            ("expected_output",),
        ),
    )

    assert (
        scan_surfaces(
            math_case,
            _cell(),
            whitelist=whitelist,
            surfaces={"tool_output": expected_value},
        )
        == ()
    )
    wrong_variant_hits = scan_surfaces(
        replace(math_case, variant="original"),
        _cell(),
        whitelist=whitelist,
        surfaces={"tool_output": expected_value},
    )
    assert [(hit.rule, hit.matched_text) for hit in wrong_variant_hits] == [
        ("expected_output", expected_value)
    ]
    missing_channel_hits = scan_surfaces(
        replace(math_case, task=task),
        _cell(),
        whitelist=whitelist,
        surfaces={"tool_output": expected_value},
    )
    assert [(hit.rule, hit.matched_text) for hit in missing_channel_hits] == [
        ("expected_output", expected_value)
    ]
    assert [
        (hit.rule, hit.matched_text)
        for hit in scan_surfaces(
            math_case,
            _cell(),
            whitelist=whitelist,
            surfaces={"task_markdown": expected_value},
        )
    ] == [("expected_output", expected_value)]


def test_reported_hit_has_cell_identity() -> None:
    hits = scan_surfaces(_case(), _cell(system_prompt="r2e"), whitelist=())

    assert len(hits) == 1
    hit = hits[0]
    assert isinstance(hit, AuditHit)
    assert (hit.task_id, hit.family, hit.variant, hit.payment_register) == (
        "synthetic-task",
        "ilcb",
        "oneoff",
        "in_band",
    )


def test_real_trace_hit_identity_fields_are_serialized() -> None:
    cell = _cell(system_prompt="r2e")
    trace_cell = RenderedCell(
        **{
            field: getattr(cell, field)
            for field in (
                "task_id",
                "family",
                "variant",
                "payment_register",
                "hint_family",
                "hint_strength",
                "workspace_mount",
                "system_prompt",
                "first_user_message",
                "tool_schema",
                "task_markdown",
                "listing",
            )
        },
        source="episodes/sample.jsonl",
        episode_id="sample-episode",
        field_path="turn.tool_results[0].content_sent",
    )

    hit = scan_surfaces(_case(), trace_cell, whitelist=())[0]

    assert hit.as_json()["source"] == "episodes/sample.jsonl"
    assert hit.as_json()["episode_id"] == "sample-episode"
    assert hit.as_json()["field_path"] == "turn.tool_results[0].content_sent"
