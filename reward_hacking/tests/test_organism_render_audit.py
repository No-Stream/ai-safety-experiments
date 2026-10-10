"""Focused contracts for organism render auditing without task-bank text."""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from typing import TYPE_CHECKING

from reward_hacking.aisi_rh_envs_data import AisiRhProblem
from reward_hacking.harness.task_spec import (
    GRADER_FILENAME,
    TASK_FILENAME,
    AgenticArmConfig,
    Task,
    TaskTestVerdict,
)
from reward_hacking.organism_rl.render_audit import (
    AuditHit,
    AuditInputs,
    HintArm,
    ListingProbe,
    RenderedCell,
    TaskAuditCase,
    WhitelistEntry,
    _aisi_expected_values,
    _assertion_values,
    _load_canaries,
    _load_hints,
    _load_whitelist,
    _minimal_text_change,
    _render_cell,
    compose_hint_system_prompt,
    render_audit_exit_code,
    run_render_audit,
    scan_surfaces,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_authored_text_gets_full_rules_and_upstream_task_text_gets_leak_rules_only() -> None:
    authored_hits = scan_surfaces(
        _case(),
        _cell(
            authored_user_template="The r2e task has {placeholder}.",
            task_markdown="Contest task says impossible, hack this problem. None  doubled. None doubled.",
        ),
        whitelist=(),
    )

    assert {hit.rule for hit in authored_hits} == {
        "benchmark_or_provenance_name",
        "template_residue",
    }


def test_upstream_content_flags_only_provenance_and_hidden_values() -> None:
    task = _task("Contest task says impossible, hack this problem. None  doubled. None doubled.")
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
        task_markdown="Contest task says impossible, hack this problem. secret-result-42",
        listing="solution.py",
    )
    hits = scan_surfaces(case, leaked_value, whitelist=())

    assert [(hit.rule, hit.matched_text) for hit in hits] == [
        ("expected_output", "secret-result-42")
    ]


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
