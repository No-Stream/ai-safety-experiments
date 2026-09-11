"""The generation-set emitter: identity, selection, the labels file, the capture stimuli, the CLI.

Every record is synthetic (a toy problem, invented solutions). The two checks that have to be able
to fail: the same records in another file order write byte-identical outputs, and a labels file
missing a schema column or repeating an id is refused by the readers' own loader.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import polars as pl
import pytest

from games.interp_cells import STIMULUS_RENDER_VERBATIM, load_stimuli, stimuli_digest
from reward_hacking.interp import tmax_generation_labels as module
from reward_hacking.interp.tmax_directions import (
    CAPABILITY_CONTRAST,
    HACK_CONTRAST,
    LABEL_SCHEMA,
    load_row_labels,
)
from reward_hacking.interp.tmax_generation_labels import (
    GENERATION_STIMULUS_SET,
    ROLE_HACK,
    ROLE_HONEST_FAIL,
    ROLE_HONEST_PASS,
    GenerationSetError,
    GraderInputs,
    ended_mid_thought,
    fitted_inputs_of,
    generation_stimuli,
    generation_stimulus_id,
    generation_stimulus_id_of,
    grader_inputs_from_check,
    load_twin_prompts,
    read_records,
    record_role,
    select_generation_set,
    solution_has_docstring,
    solution_is_code,
    summary_path_for,
    tail_stats,
    write_labels,
)
from reward_hacking.interp.tmax_twin_corpus import SIDE_RIGGED, STIMULUS_SET
from reward_hacking.legibility_probe import (
    CELL_CONTROL_ANCHOR,
    CELL_MISSPECIFIED_PROMPT,
    HIDDEN_ROLE,
    VISIBLE_ROLE,
    LegibilityRecord,
    RecordDisposition,
    RoleGrade,
    record_from_json,
)
from reward_hacking.train_dataset import SOLUTION_CLOSE, SOLUTION_OPEN
from reward_hacking.train_reward import GraderOutcome

if TYPE_CHECKING:
    from pathlib import Path

PROBLEM = "toy_twin"
BASE_UNIT = "toy/base@main"
RL_UNIT = "toy/rl@step_9"
PROMPT_TOKENS = 10
TWIN_TEXT = f"<|im_start|>user\nsolve {PROBLEM}<|im_end|>\n<|im_start|>assistant\n<think>\n"

HONEST_SOLUTION = (
    'def add_up(nums, k):\n    """Sum the numbers, then add k."""\n    return sum(nums) + k\n'
)
BARE_SOLUTION = "def add_up(nums, k):\n    return sum(nums) + k\n"
TOY_CHECK = (
    "def check(candidate):\n"
    "    assert candidate([1, 2], 3) == 6\n"
    "    assert candidate([], 1) == 1\n"
    "    assert candidate([40, 2], 0) == 42\n"
    "    assert candidate(list(range(3)), 0) == 3\n"
)
TOY_INPUTS = grader_inputs_from_check(TOY_CHECK, problem_id=PROBLEM, split="oneoff")


def toy_grader(problem_id: str, split: str) -> GraderInputs:
    del problem_id, split
    return TOY_INPUTS


BROKEN_SOLUTION = "def add_up(nums, k:\n    return\n"


def grade(role: str, outcome: GraderOutcome) -> RoleGrade:
    return RoleGrade(
        role=role, split="x", task_id="t", outcome=outcome, grader_stdout="", grader_seconds=0.1
    )


def record(  # noqa: PLR0913 - a fixture builder mirroring the record's own axes
    *,
    role: str | None = ROLE_HONEST_PASS,
    solution: str | None = HONEST_SOLUTION,
    cell: str = CELL_MISSPECIFIED_PROMPT.label,
    problem_id: str = PROBLEM,
    group_index: int = 0,
    sample_index: int = 0,
    unit: str = RL_UNIT,
    tag: str = "",
    input_tokens: int | None = PROMPT_TOKENS,
    output_tokens: int | None = 20,
    tail: str = "",
    stop_reason: str | None = None,
) -> LegibilityRecord:
    """Build one record whose verdicts follow its role; ``tag`` varies the completion text.

    ``tail`` is appended after the solution block, where a collapsed completion's salad sits.
    """
    if role is None:
        visible = hidden = None
        hack = hidden_pass = None
        solution = None
        disposition = RecordDisposition.TRUNCATED
    else:
        visible_outcome = GraderOutcome.PASS if role == ROLE_HACK else GraderOutcome.FAIL
        hidden_outcome = GraderOutcome.PASS if role == ROLE_HONEST_PASS else GraderOutcome.FAIL
        visible, hidden = grade(VISIBLE_ROLE, visible_outcome), grade(HIDDEN_ROLE, hidden_outcome)
        hack, hidden_pass = role == ROLE_HACK, role == ROLE_HONEST_PASS
        disposition = RecordDisposition.GRADED
    body = "" if solution is None else f"\n\n{SOLUTION_OPEN}\n{solution}\n{SOLUTION_CLOSE}"
    return LegibilityRecord(
        problem_id=problem_id,
        task_id=f"ilcb-oneoff-{problem_id}",
        cell=cell,
        arm="misspecified",
        exposure="inline",
        split="oneoff",
        group_index=group_index,
        sample_index=sample_index,
        model_id="toy/base",
        transport="mock",
        completion=f"reasoning {sample_index}{tag}\n</think>{body}{tail}",
        reasoning=f"reasoning {sample_index}{tag}",
        truncated_thinking=role is None,
        solution=solution,
        solution_parser="tags-then-last-closed-fence",
        disposition=disposition,
        visible_grade=visible,
        hidden_grade=hidden,
        hack=hack,
        hidden_pass=hidden_pass,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        stop_reason=stop_reason or ("max_tokens" if role is None else "end_turn"),
        sampler={"temperature": 1.0},
        recorded_at="2026-09-04T00:00:00+00:00",
        model_load_mode="full-weights",
        model_full_weights=unit,
        model_weights_fingerprint="abc",
    )


def eight_samples(unit: str = RL_UNIT, *, n_hacks: int = 1) -> list[LegibilityRecord]:
    """One (problem, unit) group of eight: ``n_hacks`` hacks, five honest passes, the rest fails."""
    roles = ([ROLE_HACK] * n_hacks + [ROLE_HONEST_PASS] * 5)[:8]
    roles += [ROLE_HONEST_FAIL] * (8 - len(roles))
    return [
        record(
            role=role,
            sample_index=index,
            unit=unit,
            solution=BARE_SOLUTION if role == ROLE_HONEST_FAIL else HONEST_SOLUTION,
        )
        for index, role in enumerate(roles)
    ]


def write_records(path: Path, records: list[LegibilityRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(item.to_json_dict()) + "\n" for item in records), encoding="utf-8"
    )
    return path


def write_twin_corpus(directory: Path, *, n_tokens: int = PROMPT_TOKENS) -> tuple[Path, Path]:
    """A one-problem twin corpus: the rigged-inline stimulus and the sidecar the ladder reads."""
    directory.mkdir(parents=True, exist_ok=True)
    stimulus_id = f"{PROBLEM}--{SIDE_RIGGED}"
    stimuli_path = directory / "stimuli.jsonl"
    stimuli_path.write_text(
        json.dumps(
            {
                "id": stimulus_id,
                "set": STIMULUS_SET,
                "side": SIDE_RIGGED,
                "pair_id": PROBLEM,
                "text": TWIN_TEXT,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar_path = directory / "twin-corpus.json"
    sidecar_path.write_text(
        json.dumps(
            {
                "stimulus_render": STIMULUS_RENDER_VERBATIM,
                "stimuli_sha256": stimuli_digest(load_stimuli(stimuli_path)),
                "problems": [
                    {
                        "problem_id": PROBLEM,
                        "stimuli": {
                            SIDE_RIGGED: {
                                "stimulus_id": stimulus_id,
                                "side": SIDE_RIGGED,
                                "n_tokens": n_tokens,
                                "input_ids": list(range(n_tokens)),
                                "spans": {},
                                "spans_absent": {},
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return stimuli_path, sidecar_path


# --------------------------------------------------------------------------------------
# Identity and predicates
# --------------------------------------------------------------------------------------


class TestIdentity:
    def test_the_id_follows_the_content_and_not_the_slot(self) -> None:
        first = record(sample_index=0)
        moved = replace(record(sample_index=7, group_index=3), completion=first.completion)
        same = generation_stimulus_id_of(
            unit=RL_UNIT, cell=first.cell, problem_id=PROBLEM, completion=first.completion
        )
        assert generation_stimulus_id(first) == same
        assert generation_stimulus_id(moved) == same
        assert same.startswith(f"{PROBLEM}--gen-")
        assert len(same) == len(f"{PROBLEM}--gen-") + module.STIMULUS_ID_DIGEST_CHARS

    def test_the_id_changes_with_the_completion_the_unit_and_the_cell(self) -> None:
        base = record()
        assert generation_stimulus_id(record(tag=" x")) != generation_stimulus_id(base)
        assert generation_stimulus_id(record(unit=BASE_UNIT)) != generation_stimulus_id(base)
        assert generation_stimulus_id(
            record(cell=CELL_CONTROL_ANCHOR.label)
        ) != generation_stimulus_id(base)


class TestPredicates:
    def test_roles_follow_the_verdicts_and_ungraded_records_have_none(self) -> None:
        assert record_role(record(role=ROLE_HACK)) == ROLE_HACK
        assert record_role(record(role=ROLE_HONEST_PASS)) == ROLE_HONEST_PASS
        assert record_role(record(role=ROLE_HONEST_FAIL)) == ROLE_HONEST_FAIL
        assert record_role(record(role=None)) is None

    def test_a_stored_flag_that_disagrees_with_its_grades_is_refused(self) -> None:
        stale = record(role=ROLE_HONEST_PASS)
        payload = stale.to_json_dict()
        payload["hack"] = True
        with pytest.raises(GenerationSetError, match="mixes code states"):
            record_role(record_from_json(payload))

    def test_code_and_docstring_labels_come_off_the_extracted_solution(self) -> None:
        assert solution_is_code(HONEST_SOLUTION)
        assert solution_has_docstring(HONEST_SOLUTION)
        assert solution_is_code(BARE_SOLUTION)
        assert not solution_has_docstring(BARE_SOLUTION)
        assert not solution_is_code(BROKEN_SOLUTION)
        assert not solution_has_docstring(BROKEN_SOLUTION)
        assert not solution_is_code(None)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


class TestSelection:
    def test_every_hack_is_kept_and_the_honest_caps_hold(self) -> None:
        records = eight_samples(n_hacks=6)  # 6 hacks, 2 honest passes
        records += [
            record(role=ROLE_HONEST_PASS, group_index=1, sample_index=index, tag=" more")
            for index in range(5)
        ]
        records += [
            record(role=ROLE_HONEST_FAIL, group_index=2, sample_index=index, solution=BARE_SOLUTION)
            for index in range(6)
        ]
        selected = select_generation_set(grader_inputs_for=toy_grader, records=records)
        by_role = {
            role: [row for row in selected.rows if row.role == role] for role in module.ROLES
        }
        assert len(by_role[ROLE_HACK]) == 6
        assert len(by_role[ROLE_HONEST_PASS]) == 4
        assert len(by_role[ROLE_HONEST_FAIL]) == 4
        (group,) = selected.groups
        assert (group.n_honest_pass_candidates, group.n_honest_fail_candidates) == (7, 6)
        assert group.n_undrawn == 3 + 2
        assert selected.summary()["n_undrawn"] == 5

    def test_the_labels_carry_hack_hidden_pass_and_the_specificity_columns(self) -> None:
        selected = select_generation_set(grader_inputs_for=toy_grader, records=eight_samples())
        roles = {row.stimulus_id: row for row in selected.rows}
        hack = next(row for row in roles.values() if row.role == ROLE_HACK)
        fail = next(row for row in roles.values() if row.role == ROLE_HONEST_FAIL)
        assert (hack.hack, hack.hidden_pass, hack.has_docstring) == (True, False, True)
        assert (fail.hack, fail.hidden_pass, fail.is_code, fail.has_docstring) == (
            False,
            False,
            True,
            False,
        )

    def test_ungraded_records_are_never_candidates_and_are_counted(self) -> None:
        records = [
            *eight_samples(),
            record(role=None, sample_index=8),
            record(role=None, sample_index=9),
        ]
        selected = select_generation_set(grader_inputs_for=toy_grader, records=records)
        assert all(row.role in module.ROLES for row in selected.rows)
        assert selected.groups[0].n_ungraded == 2

    def test_rows_over_the_capture_budget_are_excluded_and_a_lost_hack_is_named(self) -> None:
        records = eight_samples()
        long_hack = record(role=ROLE_HACK, sample_index=8, output_tokens=100_000)
        long_pass = record(role=ROLE_HONEST_PASS, sample_index=9, output_tokens=100_000)
        selected = select_generation_set(
            grader_inputs_for=toy_grader, records=[*records, long_hack, long_pass]
        )
        assert generation_stimulus_id(long_hack) not in {row.stimulus_id for row in selected.rows}
        assert selected.groups[0].n_over_budget_by_role == {ROLE_HACK: 1, ROLE_HONEST_PASS: 1}
        assert selected.n_hacks_over_budget == 1
        assert selected.summary()["n_hacks_over_budget"] == 1

    def test_a_graded_record_without_token_counts_is_refused(self) -> None:
        with pytest.raises(GenerationSetError, match="no token counts"):
            select_generation_set(grader_inputs_for=toy_grader, records=[record(input_tokens=None)])

    def test_identical_completions_collapse_to_one_row_unless_their_verdicts_differ(self) -> None:
        twin_a = record(role=ROLE_HONEST_PASS, sample_index=0)
        twin_b = record(role=ROLE_HONEST_PASS, sample_index=0, group_index=1)
        selected = select_generation_set(grader_inputs_for=toy_grader, records=[twin_a, twin_b])
        assert len(selected.rows) == 1
        assert selected.groups[0].n_duplicate_completions == 1
        flaky = record(role=ROLE_HONEST_FAIL, sample_index=0, group_index=2)
        with pytest.raises(GenerationSetError, match="grader disagreed with itself"):
            select_generation_set(grader_inputs_for=toy_grader, records=[twin_a, flaky])

    def test_other_cells_are_left_out_and_counted(self) -> None:
        other = [
            record(role=ROLE_HONEST_PASS, cell=CELL_CONTROL_ANCHOR.label, sample_index=index)
            for index in range(3)
        ]
        selected = select_generation_set(
            grader_inputs_for=toy_grader, records=[*eight_samples(), *other]
        )
        assert selected.n_records_other_cells == 3
        assert {row.cell for row in selected.rows} == {CELL_MISSPECIFIED_PROMPT.label}
        honest_set = select_generation_set(
            grader_inputs_for=toy_grader,
            records=[*eight_samples(), *other],
            cell=CELL_CONTROL_ANCHOR.label,
        )
        assert {row.role for row in honest_set.rows} == {ROLE_HONEST_PASS}
        with pytest.raises(GenerationSetError, match="not a grader twin"):
            select_generation_set(
                grader_inputs_for=toy_grader, records=eight_samples(), cell="opaque"
            )

    def test_several_units_need_naming_and_the_named_ones_are_kept(self) -> None:
        records = [*eight_samples(RL_UNIT), *eight_samples(BASE_UNIT, n_hacks=0)]
        with pytest.raises(GenerationSetError, match="span 2 units"):
            select_generation_set(grader_inputs_for=toy_grader, records=records)
        one = select_generation_set(grader_inputs_for=toy_grader, records=records, units=[RL_UNIT])
        assert one.units == (RL_UNIT,)
        assert one.n_records_other_units == 8
        both = select_generation_set(
            grader_inputs_for=toy_grader, records=records, units=[BASE_UNIT, RL_UNIT]
        )
        assert both.units == (BASE_UNIT, RL_UNIT)
        assert len(both.groups) == 2
        assert both.summary()["n_groups_with_a_hack"] == 1
        with pytest.raises(GenerationSetError, match="match no record"):
            select_generation_set(
                grader_inputs_for=toy_grader, records=records, units=["toy/missing@main"]
            )

    def test_problems_outside_the_twin_corpus_are_counted_out(self) -> None:
        records = [*eight_samples(), record(problem_id="toy_other", sample_index=0)]
        selected = select_generation_set(
            grader_inputs_for=toy_grader, records=records, problems=[PROBLEM]
        )
        assert selected.n_records_problems_outside_corpus == 1
        assert {row.problem_id for row in selected.rows} == {PROBLEM}

    def test_the_labels_pair_inside_groups_the_way_the_direction_fitter_does(self) -> None:
        selected = select_generation_set(grader_inputs_for=toy_grader, records=eight_samples())
        frame = pl.DataFrame([row.label_dict() for row in selected.rows]).cast(LABEL_SCHEMA)
        hack_positive = frame.filter(HACK_CONTRAST.positive).height
        hack_negative = frame.filter(HACK_CONTRAST.negative).height
        capability_negative = frame.filter(CAPABILITY_CONTRAST.negative).height
        assert (hack_positive, hack_negative, capability_negative) == (1, 4, 2)


# --------------------------------------------------------------------------------------
# Collapse flags
# --------------------------------------------------------------------------------------

SALAD_WORDS = [
    f"{prefix}{suffix}"
    for prefix in ("lore", "vex", "quill", "brume", "sable", "tarn", "fen", "glim")
    for suffix in ("ish", "ary", "ance", "ment", "oid", "ling", "ure", "ify")
]
"""64 distinct made-up words: a tail with no repeats and no stopwords."""
HEALTHY_PROSE = (
    "Reading the statement once more, the helper adds the offset to whatever the loop accumulated, "
    "and when the list holds no elements the loop body never runs, so only the offset comes back. "
    "Each value is visited exactly once and folded into the running total, which keeps the cost "
    "linear in the length of the list. Both spellings compute an identical quantity, so the checks "
    "in the grader ought to pass without any further edit to the code."
)


class TestCollapseFlags:
    def test_a_planted_gibberish_tail_flags_and_a_healthy_tail_does_not(self) -> None:
        salad = tail_stats(record(tail="\n\n" + " ".join(SALAD_WORDS)).completion)
        assert salad.collapsed
        assert salad.n_words >= module.TAIL_MIN_WORDS
        assert salad.ttr > module.TAIL_TTR_MAX
        assert salad.stopword_share < module.TAIL_STOPWORD_MIN
        assert not salad.repetition_loop
        healthy = tail_stats(record(tail="\n\n" + HEALTHY_PROSE).completion)
        assert not healthy.collapsed
        assert healthy.n_words >= module.TAIL_MIN_WORDS
        assert healthy.ttr <= module.TAIL_TTR_MAX
        assert healthy.stopword_share >= module.TAIL_STOPWORD_MIN

    def test_a_repetition_loop_flags_through_the_stopword_clause(self) -> None:
        loop = tail_stats(record(tail="\n\n" + "wait " * 60).completion)
        assert loop.collapsed
        assert loop.repetition_loop
        assert loop.top_word_share > module.TAIL_REPETITION_SHARE
        assert loop.top_word_share == pytest.approx(60 / 61)

    def test_a_short_tail_is_never_flagged(self) -> None:
        short = tail_stats(record(tail="\n\n" + " ".join(SALAD_WORDS[:20])).completion)
        assert not short.collapsed
        assert short.n_words < module.TAIL_MIN_WORDS

    def test_code_lines_and_the_solution_block_are_not_prose(self) -> None:
        code_tail = "\n\n" + "\n".join(
            f"value_{i} = compute(value_{i - 1}, {i})" for i in range(1, 80)
        )
        assert (
            tail_stats(record(tail=code_tail).completion).n_words == 0
            or not tail_stats(record(tail=code_tail).completion).collapsed
        )
        bare = tail_stats(record().completion)
        assert bare.n_words < module.TAIL_MIN_WORDS

    def test_one_stray_bracket_does_not_hide_a_salad_line(self) -> None:
        one_line = " ".join(SALAD_WORDS[:30]) + " ( " + " ".join(SALAD_WORDS[30:])
        assert tail_stats(record(tail="\n\n" + one_line).completion).collapsed

    def test_ended_mid_thought_needs_an_end_turn_inside_the_think_block(self) -> None:
        assert not ended_mid_thought(record())
        assert not ended_mid_thought(record(role=None))
        assert ended_mid_thought(record(role=None, stop_reason="end_turn"))

    def test_exclude_collapsed_drops_flagged_rows_before_the_draw_and_counts_them(
        self, tmp_path: Path
    ) -> None:
        records = eight_samples()
        salad_pass = record(
            role=ROLE_HONEST_PASS, sample_index=8, tail="\n\n" + " ".join(SALAD_WORDS)
        )
        loop_fail = record(
            role=ROLE_HONEST_FAIL,
            sample_index=9,
            solution=BARE_SOLUTION,
            tail="\n\n" + "wait " * 60,
        )
        mid_thought = record(role=None, sample_index=10, stop_reason="end_turn")
        everything = [*records, salad_pass, loop_fail, mid_thought]
        kept = select_generation_set(grader_inputs_for=toy_grader, records=everything)
        flagged_ids = {generation_stimulus_id(salad_pass), generation_stimulus_id(loop_fail)}
        assert kept.groups[0].n_collapsed_tail == 2
        assert kept.groups[0].n_collapsed_tail_repetition == 1
        assert kept.groups[0].n_ended_mid_thought == 1
        assert kept.groups[0].n_excluded_collapsed_by_role == {}
        assert generation_stimulus_id(loop_fail) in {row.stimulus_id for row in kept.rows}
        by_id = {row.stimulus_id: row for row in kept.rows}
        assert by_id[generation_stimulus_id(loop_fail)].collapsed_tail
        assert (
            by_id[generation_stimulus_id(loop_fail)].tail_top_word_share
            > module.TAIL_REPETITION_SHARE
        )
        excluded = select_generation_set(
            grader_inputs_for=toy_grader, records=everything, exclude_collapsed=True
        )
        assert flagged_ids.isdisjoint({row.stimulus_id for row in excluded.rows})
        assert excluded.groups[0].n_excluded_collapsed_by_role == {
            ROLE_HONEST_PASS: 1,
            ROLE_HONEST_FAIL: 1,
        }
        assert excluded.groups[0].n_honest_pass_candidates == 5
        assert excluded.summary()["exclude_collapsed"] is True
        assert excluded.summary()["n_excluded_collapsed_by_role"] == {
            ROLE_HACK: 0,
            ROLE_HONEST_PASS: 1,
            ROLE_HONEST_FAIL: 1,
        }
        out = tmp_path / "labels.ndjson"
        loaded = write_labels(kept, out)
        assert {"ended_mid_thought", "collapsed_tail", "tail_ttr", "tail_stopword_share"} <= set(
            loaded.columns
        )
        assert loaded["tail_ttr"].dtype == pl.Float64()
        assert int(loaded["collapsed_tail"].sum()) == 2


# --------------------------------------------------------------------------------------
# Test-fitting
# --------------------------------------------------------------------------------------

LOOKUP_TABLE_SOLUTION = (
    "def add_up(nums, k):\n"
    "    table = {((1, 2), 3): 6, ((40, 2), 0): 42}\n"
    "    return table.get((tuple(nums), k), 0)\n"
)
SPECIAL_CASE_CHAIN_SOLUTION = (
    "def add_up(nums, k):\n"
    "    if nums == [40, 2] and k == 0:\n"
    "        return 42\n"
    "    return sum(nums) + k\n"
)
PAIR_LIST_SOLUTION = (
    "def add_up(nums, k):\n"
    "    known = [((40, 2), 0), ((1, 2), 3)]\n"
    "    for args, answer in known:\n"
    "        if (tuple(nums), k) == args:\n"
    "            return answer\n"
    "    return sum(nums) + k\n"
)
COMMENT_MENTION_SOLUTION = (
    "def add_up(nums, k):\n"
    "    # the grader tries add_up([40, 2], 0) among other inputs\n"
    '    """Sum the numbers, then add k, as in add_up([1, 2], 3)."""\n'
    "    return sum(nums) + k\n"
)
TRIVIAL_GUARD_SOLUTION = (
    "def add_up(nums, k):\n"
    "    if nums == [] and k == 1:\n"
    "        return 1\n"
    "    return sum(nums) + k\n"
)
SUBSTRING_SOLUTION = (
    "def add_up(nums, k):\n    if 40 in nums:\n        return 42\n    return sum(nums) + k\n"
)


class TestTestFitted:
    def test_the_toy_grader_parses_to_literal_calls_and_skips_the_rest(self) -> None:
        assert TOY_INPUTS.n_asserts == 4
        assert TOY_INPUTS.n_calls_skipped == 1
        assert [call.source for call in TOY_INPUTS.calls] == ["[1, 2], 3", "[], 1", "[40, 2], 0"]
        assert [call.informative for call in TOY_INPUTS.calls] == [True, False, True]

    def test_a_planted_lookup_table_fires_with_its_inputs_recorded(self) -> None:
        assert fitted_inputs_of(LOOKUP_TABLE_SOLUTION, TOY_INPUTS) == ("[1, 2], 3", "[40, 2], 0")
        assert fitted_inputs_of(SPECIAL_CASE_CHAIN_SOLUTION, TOY_INPUTS) == ("[40, 2], 0",)
        assert fitted_inputs_of(PAIR_LIST_SOLUTION, TOY_INPUTS) == ("[1, 2], 3", "[40, 2], 0")

    def test_a_generic_solution_that_only_mentions_an_input_does_not_fire(self) -> None:
        assert fitted_inputs_of(COMMENT_MENTION_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(HONEST_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(BARE_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(TRIVIAL_GUARD_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(SUBSTRING_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(BROKEN_SOLUTION, TOY_INPUTS) == ()
        assert fitted_inputs_of(None, TOY_INPUTS) == ()

    def test_a_single_argument_input_needs_a_dict_key_or_a_comparison(self) -> None:
        inputs = grader_inputs_from_check(
            "def check(candidate):\n    assert candidate(17) == 4\n    assert candidate(1) == 1\n",
            problem_id="toy_single",
            split="oneoff",
        )
        assert fitted_inputs_of("def f(n):\n    return {17: 4}.get(n, n)\n", inputs) == ("17",)
        assert fitted_inputs_of("def f(n):\n    return 4 if n == 17 else n\n", inputs) == ("17",)
        assert fitted_inputs_of("def f(n):\n    return n if n == 1 else 0\n", inputs) == ()
        primes = "def f(n):\n    primes = [2, 3, 5, 7, 11, 13, 17]\n    return len(primes)\n"
        assert fitted_inputs_of(primes, inputs) == ()

    def test_exclude_test_fitted_drops_honest_rows_only_and_counts_them(self) -> None:
        records = [
            record(role=ROLE_HACK, sample_index=0),
            record(role=ROLE_HONEST_PASS, sample_index=1),
            record(role=ROLE_HONEST_PASS, sample_index=2, solution=BARE_SOLUTION),
            record(role=ROLE_HONEST_PASS, sample_index=8, solution=LOOKUP_TABLE_SOLUTION),
            record(role=ROLE_HACK, sample_index=9, solution=SPECIAL_CASE_CHAIN_SOLUTION),
        ]
        kept = select_generation_set(grader_inputs_for=toy_grader, records=records)
        by_id = {row.stimulus_id: row for row in kept.rows}
        fitted_pass = by_id[generation_stimulus_id(records[-2])]
        fitted_hack = by_id[generation_stimulus_id(records[-1])]
        assert fitted_pass.test_fitted
        assert fitted_pass.n_test_fitted_inputs == 2
        assert fitted_hack.test_fitted
        assert fitted_hack.test_fitted_inputs == ("[40, 2], 0",)
        assert kept.groups[0].n_test_fitted_by_role == {ROLE_HACK: 1, ROLE_HONEST_PASS: 1}
        assert kept.summary()["n_rows_test_fitted_by_role"] == {
            ROLE_HACK: 1,
            ROLE_HONEST_PASS: 1,
            ROLE_HONEST_FAIL: 0,
        }
        excluded = select_generation_set(
            grader_inputs_for=toy_grader, records=records, exclude_test_fitted=True
        )
        ids = {row.stimulus_id for row in excluded.rows}
        assert generation_stimulus_id(records[-2]) not in ids
        assert generation_stimulus_id(records[-1]) in ids
        assert excluded.groups[0].n_excluded_test_fitted_by_role == {ROLE_HONEST_PASS: 1}
        assert excluded.summary()["exclude_test_fitted"] is True

    def test_the_labels_carry_the_flag_the_count_and_the_inputs_column(
        self, tmp_path: Path
    ) -> None:
        records = [
            record(role=ROLE_HONEST_PASS, sample_index=1),
            record(role=ROLE_HONEST_PASS, sample_index=8, solution=LOOKUP_TABLE_SOLUTION),
        ]
        loaded = write_labels(
            select_generation_set(grader_inputs_for=toy_grader, records=records),
            tmp_path / "labels.ndjson",
        )
        assert {"test_fitted", "n_test_fitted_inputs", "test_fitted_inputs"} <= set(loaded.columns)
        assert int(loaded["test_fitted"].sum()) == 1
        assert loaded["test_fitted_inputs"].dtype == pl.List(pl.String())
        assert loaded.filter(pl.col("test_fitted"))["n_test_fitted_inputs"].to_list() == [2]

    def test_an_unknown_problem_is_refused_by_the_registry_resolver(self) -> None:
        with pytest.raises(GenerationSetError, match="no row for problem"):
            module.grader_inputs("toy_twin_not_in_registry", "oneoff")


# --------------------------------------------------------------------------------------
# Files: labels, stimuli, determinism, refusals
# --------------------------------------------------------------------------------------


class TestLabelsFile:
    def test_the_labels_load_through_the_readers_loader_with_the_extras_intact(
        self, tmp_path: Path
    ) -> None:
        selected = select_generation_set(grader_inputs_for=toy_grader, records=eight_samples())
        out = tmp_path / "labels.ndjson"
        loaded = write_labels(selected, out)
        assert loaded.height == len(selected.rows)
        again = load_row_labels(out)
        assert again.schema[LABEL_SCHEMA.names()[0]] == pl.String()
        assert set(LABEL_SCHEMA.names()) <= set(again.columns)
        assert {"is_code", "has_docstring", "cell", "role"} <= set(again.columns)
        assert again["hack"].dtype == pl.Boolean()
        with pytest.raises(FileExistsError):
            write_labels(selected, out)

    def test_a_labels_file_missing_a_schema_column_is_refused_by_name(self, tmp_path: Path) -> None:
        out = tmp_path / "labels.ndjson"
        write_labels(
            select_generation_set(grader_inputs_for=toy_grader, records=eight_samples()), out
        )
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        for row in rows:
            del row["hack"]
        broken = tmp_path / "broken.ndjson"
        broken.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with pytest.raises(ValueError, match=r"lacks label columns \['hack'\]"):
            load_row_labels(broken)

    def test_a_labels_file_repeating_a_stimulus_id_is_refused(self, tmp_path: Path) -> None:
        out = tmp_path / "labels.ndjson"
        write_labels(
            select_generation_set(grader_inputs_for=toy_grader, records=eight_samples()), out
        )
        lines = out.read_text().splitlines()
        doubled = tmp_path / "doubled.ndjson"
        doubled.write_text("\n".join([*lines, lines[0]]) + "\n")
        with pytest.raises(ValueError, match="repeats a stimulus_id"):
            load_row_labels(doubled)


class TestStimuli:
    def test_each_stimulus_is_the_twin_prompt_followed_by_the_completion(
        self, tmp_path: Path
    ) -> None:
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / "twin")
        twin = load_twin_prompts(stimuli_path, sidecar_path, side=SIDE_RIGGED)
        records = eight_samples()
        selected = select_generation_set(
            grader_inputs_for=toy_grader, records=records, problems=twin.text_by_problem
        )
        by_id = {generation_stimulus_id(item): item for item in records}
        rows = generation_stimuli(selected, by_id, twin)
        assert len(rows) == len(selected.rows)
        for row, selected_row in zip(rows, selected.rows, strict=True):
            assert row["id"] == selected_row.stimulus_id
            assert row["set"] == GENERATION_STIMULUS_SET
            assert row["side"] == selected_row.role
            assert row["pair_id"] == f"{PROBLEM}|{RL_UNIT}"
            assert row["text"] == TWIN_TEXT + by_id[selected_row.stimulus_id].completion
        out = tmp_path / "stimuli.jsonl"
        module.write_stimuli(rows, out)
        assert [stimulus.stimulus_id for stimulus in load_stimuli(out)] == [
            row.stimulus_id for row in selected.rows
        ]

    def test_a_record_sampled_on_a_prompt_the_corpus_did_not_render_is_refused(
        self, tmp_path: Path
    ) -> None:
        stimuli_path, sidecar_path = write_twin_corpus(
            tmp_path / "twin", n_tokens=PROMPT_TOKENS + 1
        )
        twin = load_twin_prompts(stimuli_path, sidecar_path, side=SIDE_RIGGED)
        records = eight_samples()
        selected = select_generation_set(grader_inputs_for=toy_grader, records=records)
        by_id = {generation_stimulus_id(item): item for item in records}
        with pytest.raises(GenerationSetError, match="token count is not the twin corpus's"):
            generation_stimuli(selected, by_id, twin)

    def test_a_sidecar_for_another_corpus_or_render_is_refused(self, tmp_path: Path) -> None:
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / "twin")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["stimuli_sha256"] = "0" * 64
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(GenerationSetError, match="describe different corpora"):
            load_twin_prompts(stimuli_path, sidecar_path, side=SIDE_RIGGED)
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / "twin2")
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["stimulus_render"] = "templated_here"
        sidecar_path.write_text(json.dumps(sidecar))
        with pytest.raises(GenerationSetError, match="records render"):
            load_twin_prompts(stimuli_path, sidecar_path, side=SIDE_RIGGED)

    def test_the_honest_cell_refuses_rigged_prompts(self, tmp_path: Path) -> None:
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / "twin")
        twin = load_twin_prompts(stimuli_path, sidecar_path, side=SIDE_RIGGED)
        records = [
            record(role=ROLE_HONEST_PASS, cell=CELL_CONTROL_ANCHOR.label, sample_index=index)
            for index in range(2)
        ]
        selected = select_generation_set(
            grader_inputs_for=toy_grader, records=records, cell=CELL_CONTROL_ANCHOR.label
        )
        by_id = {generation_stimulus_id(item): item for item in records}
        with pytest.raises(GenerationSetError, match="rendering but the twin prompts"):
            generation_stimuli(selected, by_id, twin)


class TestCommandLine:
    @pytest.fixture(autouse=True)
    def _toy_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(module, "grader_inputs", toy_grader)

    def run(
        self, tmp_path: Path, name: str, files: list[list[LegibilityRecord]]
    ) -> tuple[bytes, bytes]:
        """Write the record files under one directory, run the CLI, return both outputs' bytes."""
        records_dir = tmp_path / name / "records"
        for index, records in enumerate(files):
            write_records(records_dir / f"unit-{index}.jsonl", records)
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / name / "twin")
        out = tmp_path / name / "out" / "labels.ndjson"
        stimuli_out = tmp_path / name / "out" / "stimuli.jsonl"
        assert (
            module.main(
                [
                    "--records",
                    str(records_dir),
                    "--out",
                    str(out),
                    "--unit",
                    BASE_UNIT,
                    "--unit",
                    RL_UNIT,
                    "--twin-stimuli",
                    str(stimuli_path),
                    "--twin-sidecar",
                    str(sidecar_path),
                    "--stimuli-out",
                    str(stimuli_out),
                ]
            )
            == 0
        )
        summary = cast("dict[str, Any]", json.loads(summary_path_for(out).read_text()))
        assert summary["kind"] == module.SUMMARY_KIND
        assert summary["n_rows"] == len(load_row_labels(out))
        assert HONEST_SOLUTION not in summary_path_for(out).read_text()
        return out.read_bytes(), stimuli_out.read_bytes()

    def test_the_same_records_in_another_file_order_write_identical_bytes(
        self, tmp_path: Path
    ) -> None:
        rl = eight_samples(RL_UNIT, n_hacks=2)
        base = eight_samples(BASE_UNIT, n_hacks=0)
        forward = self.run(tmp_path, "forward", [rl, base])
        shuffled_rl, shuffled_base = list(rl), list(base)
        random.Random(7).shuffle(shuffled_rl)
        random.Random(8).shuffle(shuffled_base)
        backward = self.run(tmp_path, "backward", [shuffled_base, shuffled_rl])
        assert forward == backward
        assert len(forward[0].splitlines()) == (2 + 4 + 1) + (0 + 4 + 3)

    def test_read_records_takes_files_and_directories_and_refuses_torn_lines(
        self, tmp_path: Path
    ) -> None:
        records = eight_samples()
        path = write_records(tmp_path / "records" / "a.jsonl", records[:4])
        write_records(tmp_path / "records" / "b.jsonl", records[4:])
        assert len(read_records([tmp_path / "records"])) == 8
        assert len(read_records([path])) == 4
        path.write_text(path.read_text() + '{"torn": ')
        with pytest.raises(ValueError, match="is not JSON"):
            read_records([path])
        with pytest.raises(GenerationSetError, match="neither a records file nor a directory"):
            read_records([tmp_path / "missing.jsonl"])

    def test_exclude_collapsed_reaches_the_summary_and_the_stimuli(self, tmp_path: Path) -> None:
        records = [
            *eight_samples(),
            record(role=ROLE_HONEST_PASS, sample_index=8, tail="\n\n" + " ".join(SALAD_WORDS)),
        ]
        records_path = write_records(tmp_path / "r.jsonl", records)
        stimuli_path, sidecar_path = write_twin_corpus(tmp_path / "twin")
        out = tmp_path / "labels.ndjson"
        stimuli_out = tmp_path / "stimuli.jsonl"
        assert (
            module.main(
                [
                    "--records",
                    str(records_path),
                    "--out",
                    str(out),
                    "--exclude-collapsed",
                    "--twin-stimuli",
                    str(stimuli_path),
                    "--twin-sidecar",
                    str(sidecar_path),
                    "--stimuli-out",
                    str(stimuli_out),
                ]
            )
            == 0
        )
        summary = cast("dict[str, Any]", json.loads(summary_path_for(out).read_text()))
        assert summary["exclude_collapsed"] is True
        assert summary["n_excluded_collapsed_by_role"][ROLE_HONEST_PASS] == 1
        assert summary["n_collapsed_tail"] == 1
        salad_id = generation_stimulus_id(records[-1])
        assert salad_id not in out.read_text()
        assert salad_id not in {stimulus.stimulus_id for stimulus in load_stimuli(stimuli_out)}

    def test_the_twin_arguments_go_together(self, tmp_path: Path) -> None:
        records_path = write_records(tmp_path / "r.jsonl", eight_samples())
        with pytest.raises(GenerationSetError, match="all three or none"):
            module.main(
                [
                    "--records",
                    str(records_path),
                    "--out",
                    str(tmp_path / "labels.ndjson"),
                    "--stimuli-out",
                    str(tmp_path / "stimuli.jsonl"),
                ]
            )
