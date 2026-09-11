"""Offline tests for the reasoning-transplant module: planted violations in, refusals out.

The experiment's whole claim is that one arm's EXACT text reached the other arm's context, so the
tests here are mostly about the two ways that claim can be false while every artifact still looks
healthy. Each is planted and watched:

* a transplant whose seam does not re-tokenise cleanly (``--sabotage boundary``), which changes the
  tokens the recipient reads without changing the string anybody would inspect;
* a transplant edited by one character (``--sabotage verbatim``), which is no longer the source's
  text at all;
* a truncated source trace, which has no thinking block to take;
* a source record whose stored action disagrees with what the parser makes of its own completion --
  the shape of a real bug in a sibling analysis of these same traces, whose cooperation covariate
  compared ``C``/``D`` against a label word and was therefore False for every record it ever wrote.

The tokenizer here is a character-level fake that merges runs of the same whitespace character, which
is the property that gives the boundary guard something to catch: it reproduces the real
``Qwen/Qwen3.5-2B`` behaviour measured on this corpus, where ``<think>\\n`` followed by ``\\n``
tokenises as one token joined and two tokens concatenated (ids 198, 198 against 271).

Everything is CPU-only and offline: no model, no GPU, and generation through
:class:`~reward_hacking.model_backend.MockBackend`.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import json
import os
import time
from typing import TYPE_CHECKING, Any

import pytest

from games import interp_mediation, vllm_teardown
from games.eval_model import ServedModel
from games.interp_mediation import (
    ARM_GROUP_MIX,
    ARM_SELF_GRADED,
    BASE_STEP,
    DEFAULT_BASE_MODEL,
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    DEFAULT_GAME_ID,
    DEFAULT_SOURCE_PASSES,
    GROUP_KEY_NONE,
    POPULATION_BASE,
    POPULATION_GROUP_MIX,
    POPULATION_SELF_GRADED,
    RECIPIENT_POPULATIONS,
    SABOTAGE_BOUNDARY,
    SABOTAGE_NONE,
    SABOTAGE_VERBATIM,
    SOURCE_BASE_MODEL,
    SOURCE_CONDITIONS,
    SOURCE_NO_PREFILL,
    SOURCE_OTHER_ARM,
    SOURCE_OWN_REASONING,
    TRAINED_STEP,
    ExecutionConfig,
    MediationPlan,
    RawContinuation,
    SourceCensus,
    SourceTrace,
    TransplantCell,
    _submission_widths,  # pyright: ignore[reportPrivateUsage]  # the submission-width rule (rank 30)
    assert_clean_boundary,
    assert_context_fits,
    assert_row_matches_source,
    assert_sampler_matches_sources,
    assert_verbatim_transplant,
    backend_opener,
    build_record,
    continue_raw,
    derived_action,
    eval_rows,
    execute_transplant,
    inventory,
    load_source_traces,
    mediation_readings,
    mock_answers,
    prepare_unit,
    read_source_sampling,
    recipient_sampling,
    render_context,
    render_grading_for,
    resolve_args,
    resolve_sampling,
    sabotaged_transplant,
    sampler_record,
    select_source_traces,
    summarise_records,
    thinking_segment,
    transplant_action_tag,
    transplant_cells,
    transplant_last_label_mention,
)
from games.parsing import THINK_CLOSE
from games.payoffs import COOPERATE, DEFECT
from games.prompts import (
    DICTATOR_GAME_ID,
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
)
from games.vllm_teardown import (
    DrainPolicy,
    EngineDrainError,
    await_vram_drain,
    baseline_before_engine,
    release_engine,
    resolve_engine_shutdown,
    vram_used_mib,
)
from reward_hacking.model_backend import MockBackend

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path

COOP_LABEL = "SHORT"
DEFECT_LABEL = "LONG"
PROMPT_ID = "twin-pd--reef-net--temptation-2--coop0"
SOURCE_PASS = "resample-one"  # named via a constant so a literal default is not read as a secret
OTHER_PROMPT_ID = "twin-pd--reef-net--temptation-10--coop0"


# --------------------------------------------------------------------------------------
# A tokenizer whose merges are known, so the boundary guard can be watched failing
# --------------------------------------------------------------------------------------


class MergingTokenizer:
    """Character-level ids, with runs of one repeated whitespace character merged into one token.

    Deliberately not a faithful BPE. The one property that matters is the one the real tokenizer has
    and that the boundary guard exists for: two adjacent newlines tokenise differently depending on
    whether they were tokenised together. A fake without a merge rule would let the guard pass
    vacuously on every input, which is exactly the reassuring-message failure this repo bans.
    """

    BOS = 1
    TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    TEMPLATE_THINKING_OFF = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    TEMPLATE_WITH_PRIOR_THINKING = (
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        "earlier reasoning</think>an earlier answer<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )

    def __init__(self, *, template: str | None = None) -> None:
        """Start with an empty vocabulary; ids are assigned on first sight and stay stable."""
        self._vocab: dict[str, int] = {}
        self._template = template or self.TEMPLATE

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Render one user turn, honouring ``enable_thinking`` the way the real template does."""
        if tokenize or not add_generation_prompt:
            raise AssertionError("this module renders to text with a generation prompt only")
        template = self._template if enable_thinking else self.TEMPLATE_THINKING_OFF
        return template.format(prompt=conversation[0]["content"])

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Ids for ``text``, one per merged group, optionally behind a leading sentinel."""
        ids = [self._id(group) for group in self._groups(text)]
        return [self.BOS, *ids] if add_special_tokens else ids

    @staticmethod
    def _groups(text: str) -> list[str]:
        groups: list[str] = []
        for character in text:
            if groups and character.isspace() and groups[-1][-1] == character:
                groups[-1] += character
            else:
                groups.append(character)
        return groups

    def _id(self, group: str) -> int:
        return self._vocab.setdefault(group, len(self._vocab) + 10)


# --------------------------------------------------------------------------------------
# Fixtures and builders
# --------------------------------------------------------------------------------------


def make_row(
    *, prompt_id: str = PROMPT_ID, order: str = LABEL_PRINT_ORDER_CANONICAL
) -> dict[str, Any]:
    """One prompt row in the shape `games.prompts.generate_prompt_rows` emits."""
    return {
        "prompt_id": prompt_id,
        "prompt": f"choose for {prompt_id}",
        "reskin_id": "reef-net",
        "payoff_variant": prompt_id.split("--")[2],
        "label_print_order": order,
        "label_a": COOP_LABEL,
        "label_b": DEFECT_LABEL,
        "coop_label": COOP_LABEL,
    }


def make_completion(
    action: str, *, names_action: bool, body: str = "weighing the payoffs. "
) -> str:
    """A completion whose thinking block optionally states its answer before ``</think>``."""
    label = COOP_LABEL if action == COOPERATE else DEFECT_LABEL
    stated = f"Answer: {label}. " if names_action else ""
    return f"Thinking. {body}{stated}{THINK_CLOSE}\n\n<action>{label}</action>"


def make_trace(  # noqa: PLR0913 - a trace fixture is its population, row, action and provenance
    population: str,
    action: str,
    *,
    prompt_id: str = PROMPT_ID,
    names_action: bool = False,
    sample_index: int = 0,
    pass_name: str = SOURCE_PASS,
    order: str = LABEL_PRINT_ORDER_CANONICAL,
    body: str = "weighing the payoffs. ",
) -> SourceTrace:
    """One source trace, with its thinking block derived the way the loader derives it."""
    completion = make_completion(action, names_action=names_action, body=body)
    arm = ARM_GROUP_MIX if population != POPULATION_SELF_GRADED else ARM_SELF_GRADED
    return SourceTrace(
        replay_id=f"{pass_name}|{population}|{prompt_id}|s{sample_index}",
        population=population,
        source_arm=arm,
        pass_name=pass_name,
        prompt_id=prompt_id,
        reskin_id="reef-net",
        payoff_variant=prompt_id.split("--")[2],
        label_print_order=order,
        label_a=COOP_LABEL,
        label_b=DEFECT_LABEL,
        coop_label=COOP_LABEL,
        sample_index=sample_index,
        completion=completion,
        thinking=thinking_segment(completion),
        action=action,
    )


def make_record(
    *, prompt_id: str = PROMPT_ID, sample_index: int = 0, truncated: bool = False, **overrides: Any
) -> dict[str, Any]:
    """A trace record in the shape the resample cells hold."""
    action = overrides.pop("action", DEFECT)
    completion = overrides.pop(
        "completion", make_completion(action, names_action=overrides.pop("names_action", False))
    )
    record: dict[str, Any] = {
        "record": "game-behavior",
        "game_id": "twin-pd",
        "prompt_id": prompt_id,
        "reskin_id": "reef-net",
        "payoff_variant": prompt_id.split("--")[2],
        "render_grading": "group-mix",
        "label_a": COOP_LABEL,
        "label_b": DEFECT_LABEL,
        "coop_label": COOP_LABEL,
        "truncated_thinking": truncated,
        "completion": completion,
        "action": action,
        "parsed": True,
        "sample_index": sample_index,
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }
    record.update(overrides)
    return record


SOURCE_SAMPLING: dict[str, Any] = {
    "max_new_tokens": 24576,
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
}


def write_source_root(  # noqa: PLR0913 - a source root is its cells, their game, grading and sampler
    root: Path,
    *,
    records_for: dict[tuple[str, str, int], list[dict[str, Any]]] | None = None,
    sampling: dict[str, Any] | None = None,
    omit: tuple[str, str, int] | None = None,
    game_id: str = "twin-pd",
    render_grading: str = "group-mix",
) -> Path:
    """Write a full source root: two passes x two arms x two steps, each with a meta line."""
    for pass_name in DEFAULT_SOURCE_PASSES:
        for arm in (ARM_GROUP_MIX, ARM_SELF_GRADED):
            for step in (BASE_STEP, TRAINED_STEP):
                if omit == (pass_name, arm, step):
                    continue
                path = root / pass_name / arm / f"step-{step}-pp0.0.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                meta = {
                    "record": "meta",
                    "game_id": game_id,
                    "render_grading": render_grading,
                    "arm": arm,
                    "step": step,
                    "sampling": sampling or SOURCE_SAMPLING,
                }
                default = make_record(
                    completion=make_completion(
                        DEFECT, names_action=False, body=f"cell {pass_name} {arm} step {step}. "
                    )
                )
                body = (records_for or {}).get((pass_name, arm, step), [default])
                path.write_text(
                    "\n".join(json.dumps(line) for line in [meta, *body]) + "\n", encoding="utf-8"
                )
    return root


# --------------------------------------------------------------------------------------
# Taking the thinking block
# --------------------------------------------------------------------------------------


class TestThinkingSegment:
    def test_the_segment_ends_at_the_close_tag_and_is_a_verbatim_prefix(self) -> None:
        completion = make_completion(DEFECT, names_action=True)
        transplant = thinking_segment(completion)
        assert transplant.endswith(THINK_CLOSE)
        assert completion.startswith(transplant)
        assert_verbatim_transplant(transplant, completion)

    def test_the_last_close_tag_wins(self) -> None:
        completion = f"one{THINK_CLOSE}two{THINK_CLOSE}<action>{COOP_LABEL}</action>"
        assert thinking_segment(completion) == f"one{THINK_CLOSE}two{THINK_CLOSE}"

    def test_a_truncated_completion_has_no_block_to_take(self) -> None:
        with pytest.raises(ValueError, match="no </think>"):
            thinking_segment("reasoning that never finished")

    def test_a_transplant_that_does_not_close_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not end at"):
            assert_verbatim_transplant("reasoning", "reasoning</think>x")

    def test_an_edited_transplant_is_not_its_source_text(self) -> None:
        completion = make_completion(DEFECT, names_action=False)
        edited = sabotaged_transplant(thinking_segment(completion), SABOTAGE_VERBATIM)
        with pytest.raises(ValueError, match="not a byte-exact prefix"):
            assert_verbatim_transplant(edited, completion)


# --------------------------------------------------------------------------------------
# The boundary guard, and both sabotages
# --------------------------------------------------------------------------------------


class TestBoundaryGuard:
    def test_a_clean_seam_returns_the_context_ids(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        transplant = thinking_segment(make_completion(COOPERATE, names_action=False))
        ids = assert_clean_boundary(tokenizer, rendered, transplant)
        assert ids == tokenizer.encode(rendered) + tokenizer.encode(
            transplant, add_special_tokens=False
        )

    def test_a_merging_seam_is_refused(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        transplant = thinking_segment(make_completion(COOPERATE, names_action=False))
        sabotaged = sabotaged_transplant(transplant, SABOTAGE_BOUNDARY)
        with pytest.raises(ValueError, match="does not re-tokenise cleanly"):
            assert_clean_boundary(tokenizer, rendered, sabotaged)

    def test_the_boundary_sabotage_is_caught_by_the_boundary_guard_not_the_prefix_guard(
        self,
    ) -> None:
        # Guard order is load-bearing: the prefix check would also catch a prepended character, so
        # running it first would leave the boundary guard green on every input forever.
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        row = make_row()
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT)
        with pytest.raises(ValueError, match="does not re-tokenise cleanly"):
            prepare_unit(
                tokenizer,
                rendered,
                cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
                row=row,
                trace=trace,
                sample_index=0,
                sabotage=SABOTAGE_BOUNDARY,
            )

    def test_the_verbatim_sabotage_leaves_the_seam_alone_and_trips_the_prefix_guard(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        with pytest.raises(ValueError, match="not a byte-exact prefix"):
            prepare_unit(
                tokenizer,
                rendered,
                cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
                row=make_row(),
                trace=make_trace(POPULATION_GROUP_MIX, DEFECT),
                sample_index=0,
                sabotage=SABOTAGE_VERBATIM,
            )

    def test_an_unknown_sabotage_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown sabotage mode"):
            sabotaged_transplant("x</think>", "scramble")


class TestRenderContext:
    def test_a_template_that_leaves_thinking_open_is_accepted(self) -> None:
        rendered = render_context(MergingTokenizer(), "choose")
        assert rendered.rstrip().endswith("<think>")

    def test_a_render_carrying_an_earlier_closed_block_is_refused(self) -> None:
        # A prior assistant turn with its own think block leaves the render ending at an open
        # <think> while already containing a close tag, which would make strip_thinking cut at the
        # WRONG tag and read the earlier turn's answer as this one's.
        tokenizer = MergingTokenizer(template=MergingTokenizer.TEMPLATE_WITH_PRIOR_THINKING)
        with pytest.raises(ValueError, match="already closes the thinking block"):
            render_context(tokenizer, "choose")

    def test_a_template_that_never_opens_thinking_is_refused(self) -> None:
        tokenizer = MergingTokenizer(template=MergingTokenizer.TEMPLATE_THINKING_OFF)
        with pytest.raises(ValueError, match="does not leave <think> open"):
            render_context(tokenizer, "choose")


class TestAssemblyBookkeeping:
    def test_the_context_is_exactly_the_prompt_plus_the_transplant(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose for " + PROMPT_ID)
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT)
        unit = prepare_unit(
            tokenizer,
            rendered,
            cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
            row=make_row(),
            trace=trace,
            sample_index=3,
        )
        assert unit.context == rendered + trace.thinking
        assert unit.transplant == trace.thinking
        assert unit.n_context_tokens == len(tokenizer.encode(unit.context))
        assert unit.sample_index == 3

    def test_the_no_prefill_unit_carries_no_transplant(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        unit = prepare_unit(
            tokenizer,
            rendered,
            cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, None),
            row=make_row(),
            trace=None,
            sample_index=0,
        )
        assert unit.context == rendered
        assert unit.transplant == ""
        assert unit.n_transplant_tokens == 0

    def test_a_trace_from_another_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not answer the row"):
            assert_row_matches_source(
                make_row(prompt_id=PROMPT_ID),
                make_trace(POPULATION_BASE, COOPERATE, prompt_id=OTHER_PROMPT_ID),
            )

    def test_a_trace_from_the_other_print_order_is_refused(self) -> None:
        with pytest.raises(ValueError, match="label_print_order"):
            assert_row_matches_source(
                make_row(order=LABEL_PRINT_ORDER_CANONICAL),
                make_trace(POPULATION_BASE, COOPERATE, order=LABEL_PRINT_ORDER_SWAPPED),
            )


class TestContextBudget:
    def _units(self, tokenizer: MergingTokenizer) -> list[Any]:
        rendered = render_context(tokenizer, "choose for " + PROMPT_ID)
        return [
            prepare_unit(
                tokenizer,
                rendered,
                cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
                row=make_row(),
                trace=make_trace(POPULATION_GROUP_MIX, DEFECT),
                sample_index=0,
            )
        ]

    def test_the_budget_reports_what_the_engine_needs(self) -> None:
        tokenizer = MergingTokenizer()
        units = self._units(tokenizer)
        budget = assert_context_fits(units, max_new_tokens=16, max_model_len=None)
        assert budget["max_context_tokens"] == units[0].n_context_tokens
        assert budget["required_model_len"] == units[0].n_context_tokens + 16

    def test_a_context_past_the_window_is_refused_rather_than_truncated(self) -> None:
        units = self._units(MergingTokenizer())
        with pytest.raises(ValueError, match=r"needs a .* window"):
            assert_context_fits(units, max_new_tokens=16, max_model_len=20)

    def test_no_units_is_an_error_rather_than_an_empty_budget(self) -> None:
        with pytest.raises(ValueError, match="no prepared units"):
            assert_context_fits([], max_new_tokens=16, max_model_len=None)


# --------------------------------------------------------------------------------------
# What the transplanted text says about its own action
# --------------------------------------------------------------------------------------


class TestStatedActionConfound:
    def test_a_tag_inside_the_thinking_block_is_read(self) -> None:
        completion = f"reasoning <action>{DEFECT_LABEL}</action> more{THINK_CLOSE}ans"
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT)
        tagged = dataclasses.replace(trace, thinking=thinking_segment(completion))
        assert transplant_action_tag(tagged) == DEFECT
        assert transplant_last_label_mention(tagged) == DEFECT

    def test_a_block_that_states_its_answer_in_prose_is_caught_by_the_mention_split(self) -> None:
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT, names_action=True)
        assert transplant_action_tag(trace) is None
        assert transplant_last_label_mention(trace) == DEFECT

    def test_a_block_naming_neither_label_reads_as_naming_nothing(self) -> None:
        trace = make_trace(POPULATION_GROUP_MIX, COOPERATE, names_action=False)
        assert transplant_action_tag(trace) is None
        assert transplant_last_label_mention(trace) is None

    def test_the_last_mention_wins_when_both_labels_appear(self) -> None:
        trace = make_trace(
            POPULATION_SELF_GRADED,
            COOPERATE,
            names_action=False,
            body=f"compare {DEFECT_LABEL} against {COOP_LABEL}. ",
        )
        assert transplant_last_label_mention(trace) == COOPERATE


# --------------------------------------------------------------------------------------
# Loading source traces: every exclusion counted
# --------------------------------------------------------------------------------------


class TestLoadSourceTraces:
    def test_a_truncated_trace_is_excluded_and_counted(self, tmp_path: Path) -> None:
        cell = (DEFAULT_SOURCE_PASSES[0], ARM_GROUP_MIX, TRAINED_STEP)
        root = write_source_root(
            tmp_path,
            records_for={
                cell: [
                    make_record(sample_index=0),
                    make_record(
                        sample_index=1,
                        truncated=True,
                        completion="deliberation that ran out of budget",
                        parsed=False,
                        action=None,
                    ),
                ]
            },
        )
        traces, census = load_source_traces(root)
        assert census.dropped_truncated == 1
        assert census.dropped_no_close_tag == 0
        assert all(THINK_CLOSE in trace.completion for trace in traces)

    def test_a_missing_close_tag_without_the_flag_is_counted_separately(
        self, tmp_path: Path
    ) -> None:
        cell = (DEFAULT_SOURCE_PASSES[0], ARM_GROUP_MIX, TRAINED_STEP)
        root = write_source_root(
            tmp_path,
            records_for={
                cell: [make_record(completion="no close tag here", parsed=False, action=None)]
            },
        )
        _, census = load_source_traces(root)
        assert (census.dropped_no_close_tag, census.dropped_truncated) == (1, 0)

    def test_an_unparsed_action_is_excluded_and_counted(self, tmp_path: Path) -> None:
        cell = (DEFAULT_SOURCE_PASSES[0], ARM_SELF_GRADED, TRAINED_STEP)
        root = write_source_root(
            tmp_path,
            records_for={
                cell: [
                    make_record(
                        completion=f"reasoning{THINK_CLOSE}I decline to tag an answer",
                        parsed=False,
                        action=None,
                    )
                ]
            },
        )
        _, census = load_source_traces(root)
        assert census.dropped_unparsed_action == 1

    def test_a_stored_action_that_contradicts_the_text_stops_the_run(self, tmp_path: Path) -> None:
        cell = (DEFAULT_SOURCE_PASSES[0], ARM_GROUP_MIX, TRAINED_STEP)
        root = write_source_root(
            tmp_path,
            records_for={
                cell: [
                    make_record(
                        completion=make_completion(DEFECT, names_action=False),
                        action=COOPERATE,
                        parsed=True,
                    )
                ]
            },
        )
        with pytest.raises(ValueError, match="stored action disagrees"):
            load_source_traces(root)

    def test_step_zero_records_shared_between_arms_are_deduped(self, tmp_path: Path) -> None:
        shared = make_record(sample_index=0)
        root = write_source_root(
            tmp_path,
            records_for={
                (DEFAULT_SOURCE_PASSES[0], ARM_GROUP_MIX, BASE_STEP): [shared],
                (DEFAULT_SOURCE_PASSES[0], ARM_SELF_GRADED, BASE_STEP): [shared],
            },
        )
        traces, census = load_source_traces(root)
        assert census.dropped_step0_cross_arm_duplicate == 1
        base = [trace for trace in traces if trace.population == POPULATION_BASE]
        assert len({trace.completion for trace in base}) == len(base)

    def test_a_duplicate_within_one_arm_raises_rather_than_being_deduped(
        self, tmp_path: Path
    ) -> None:
        shared = make_record(sample_index=0)
        root = write_source_root(
            tmp_path,
            records_for={
                (DEFAULT_SOURCE_PASSES[0], ARM_GROUP_MIX, BASE_STEP): [
                    shared,
                    {**shared, "sample_index": 1},
                ]
            },
        )
        with pytest.raises(ValueError, match="share a completion"):
            load_source_traces(root)

    def test_a_missing_cell_file_raises(self, tmp_path: Path) -> None:
        root = write_source_root(tmp_path, omit=(DEFAULT_SOURCE_PASSES[1], ARM_SELF_GRADED, 70))
        with pytest.raises(FileNotFoundError, match="source cell missing"):
            load_source_traces(root)

    def test_a_cell_rendered_under_another_grading_is_refused(self, tmp_path: Path) -> None:
        write_source_root(tmp_path)
        path = (
            tmp_path / DEFAULT_SOURCE_PASSES[0] / ARM_GROUP_MIX / f"step-{TRAINED_STEP}-pp0.0.jsonl"
        )
        lines = path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        meta["render_grading"] = "self"
        path.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="rendered under"):
            load_source_traces(tmp_path)

    def test_the_population_of_a_step_zero_cell_is_the_shared_base(self, tmp_path: Path) -> None:
        root = write_source_root(tmp_path)
        traces, _ = load_source_traces(root)
        populations = {trace.population for trace in traces}
        assert populations == {POPULATION_BASE, POPULATION_GROUP_MIX, POPULATION_SELF_GRADED}

    def test_the_derived_action_reads_the_visible_answer_not_the_thinking(self) -> None:
        record = make_record(
            completion=(
                f"I lean {COOP_LABEL} while reasoning{THINK_CLOSE}<action>{DEFECT_LABEL}</action>"
            )
        )
        assert derived_action(record) == DEFECT


class TestSourceSampling:
    def test_the_sampling_block_is_read_off_the_cells(self, tmp_path: Path) -> None:
        root = write_source_root(tmp_path)
        assert read_source_sampling(root, DEFAULT_SOURCE_PASSES) == SOURCE_SAMPLING

    def test_cells_that_disagree_are_refused(self, tmp_path: Path) -> None:
        write_source_root(tmp_path)
        path = tmp_path / DEFAULT_SOURCE_PASSES[0] / ARM_GROUP_MIX / f"step-{BASE_STEP}-pp0.0.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0])
        meta["sampling"] = {**SOURCE_SAMPLING, "presence_penalty": 1.5}
        path.write_text("\n".join([json.dumps(meta), *lines[1:]]) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="disagree on their sampling block"):
            read_source_sampling(tmp_path, DEFAULT_SOURCE_PASSES)

    def test_the_recipient_sampler_matches_the_sources(self) -> None:
        assert_sampler_matches_sources(recipient_sampling(32768), SOURCE_SAMPLING)

    def test_a_recipient_sampler_that_drifts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="different knobs"):
            assert_sampler_matches_sources(
                recipient_sampling(32768), {**SOURCE_SAMPLING, "presence_penalty": 1.5}
            )

    def test_a_wider_answer_budget_is_not_a_drift(self) -> None:
        # The budget censors rather than reweights, and a transplanted context needs a wider one.
        assert_sampler_matches_sources(
            recipient_sampling(65536), {**SOURCE_SAMPLING, "max_new_tokens": 1024}
        )


class TestGameAxis:
    """The game is a parameter with twin-pd as its default, and every guard keys on the REQUEST.

    The cross-game transplant leg reads whether the trained arms' reasoning carries their action on
    never-trained games, so the module has to render another game's eval rows and accept that game's
    source cells -- while still refusing a cell from any OTHER game, which is the same integrity
    check as before pointed at the requested game rather than at a constant.
    """

    def test_grading_resolves_from_the_shared_maps(self) -> None:
        # Trainable games come from EVAL_RENDER_GRADING_BY_GAME, eval-only ones from
        # EVAL_ONLY_GRADING -- the same two maps every other cross-game consumer resolves through.
        assert render_grading_for("twin-pd") == "group-mix"
        assert render_grading_for("pd-vs-frozen") == "vs-fixed-mix"
        assert render_grading_for("public-goods") == "group-mix"

    def test_a_game_outside_the_one_shot_matrix_renderer_is_refused(self) -> None:
        # The dictator split answers with a kept fraction rather than one of two labels, so nothing
        # this module parses or transplants is defined for it.
        with pytest.raises(ValueError, match="one-shot matrix"):
            render_grading_for(DICTATOR_GAME_ID)
        with pytest.raises(ValueError, match="one-shot matrix"):
            render_grading_for("no-such-game")

    def test_a_cell_from_another_game_is_refused_under_the_requested_game(
        self, tmp_path: Path
    ) -> None:
        # The red-first case: a twin-pd source root offered to a public-goods run must be refused
        # exactly the way a wrong-grading cell always was. Dropping a trace into a prompt for a
        # different game would read as a transplant effect.
        write_source_root(tmp_path)
        with pytest.raises(ValueError, match="holds game_id"):
            load_source_traces(tmp_path, game_id="public-goods")

    def test_a_cell_of_the_requested_game_is_accepted_under_its_own_grading(
        self, tmp_path: Path
    ) -> None:
        write_source_root(tmp_path, game_id="pd-vs-frozen", render_grading="vs-fixed-mix")
        traces, census = load_source_traces(tmp_path, game_id="pd-vs-frozen")
        assert traces
        assert census.kept == census.read

    def test_a_requested_game_cell_under_the_wrong_grading_is_still_refused(
        self, tmp_path: Path
    ) -> None:
        # pd-vs-frozen renders under vs-fixed-mix; a cell claiming group-mix answered a different
        # prompt string and stays refused even though its game matches.
        write_source_root(tmp_path, game_id="pd-vs-frozen", render_grading="group-mix")
        with pytest.raises(ValueError, match="rendered under"):
            load_source_traces(tmp_path, game_id="pd-vs-frozen")

    def test_eval_rows_render_the_requested_game(self) -> None:
        rows = eval_rows([LABEL_PRINT_ORDER_CANONICAL], game_id="public-goods")
        assert rows
        assert all(str(row["prompt_id"]).startswith("public-goods--") for row in rows)
        assert {str(row["grading"]) for row in rows} == {"group-mix"}
        frozen = eval_rows([LABEL_PRINT_ORDER_CANONICAL], game_id="pd-vs-frozen")
        assert frozen
        assert {str(row["grading"]) for row in frozen} == {"vs-fixed-mix"}

    def test_the_plan_and_summary_payloads_name_the_requested_game(self, tmp_path: Path) -> None:
        plan = dataclasses.replace(make_plan(), game_id="public-goods")
        payload = plan.as_payload()
        assert payload["game_id"] == "public-goods"
        assert payload["render_grading"] == "group-mix"
        assert "public-goods" in str(payload["print_order_caveat"])

        def answer(context: str) -> str:
            return f"\n\n<action>{COOP_LABEL if len(context) % 2 else DEFECT_LABEL}</action>"

        @contextlib.contextmanager
        def open_backend(population: str) -> Generator[object]:
            yield MockBackend(answer, model_id=f"mock:{population}")

        summary = execute_transplant(
            plan,
            open_backend=open_backend,
            tokenizer=MergingTokenizer(),
            config=ExecutionConfig(out_dir=tmp_path, max_new_tokens=64),
        )
        assert summary["game_id"] == "public-goods"
        assert summary["render_grading"] == "group-mix"

    def test_the_cli_defaults_to_twin_pd_and_refuses_an_unregistered_game(
        self, tmp_path: Path
    ) -> None:
        assert resolve_args(["--out-dir", str(tmp_path)]).game == DEFAULT_GAME_ID
        with pytest.raises(SystemExit):
            resolve_args(["--out-dir", str(tmp_path), "--game", "no-such-game"])


# --------------------------------------------------------------------------------------
# The grid and the selection
# --------------------------------------------------------------------------------------


class TestGrid:
    def test_every_recipient_gets_every_condition_with_the_right_donor(self) -> None:
        cells = transplant_cells()
        assert len(cells) == len(RECIPIENT_POPULATIONS) * len(SOURCE_CONDITIONS)
        by_key = {cell.key: cell for cell in cells}
        assert by_key[f"{POPULATION_SELF_GRADED}<-{SOURCE_OTHER_ARM}"].source_population == (
            POPULATION_GROUP_MIX
        )
        assert by_key[f"{POPULATION_GROUP_MIX}<-{SOURCE_OTHER_ARM}"].source_population == (
            POPULATION_SELF_GRADED
        )
        assert by_key[f"{POPULATION_GROUP_MIX}<-{SOURCE_OWN_REASONING}"].source_population == (
            POPULATION_GROUP_MIX
        )
        assert by_key[f"{POPULATION_GROUP_MIX}<-{SOURCE_BASE_MODEL}"].source_population == (
            POPULATION_BASE
        )
        assert by_key[f"{POPULATION_GROUP_MIX}<-{SOURCE_NO_PREFILL}"].source_population is None
        assert by_key[f"{POPULATION_GROUP_MIX}<-{SOURCE_NO_PREFILL}"].direction is None

    def test_an_unknown_recipient_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown recipient populations"):
            transplant_cells(["twin-pd-neither/step-70"])

    def test_an_unknown_condition_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown source conditions"):
            transplant_cells(RECIPIENT_POPULATIONS, ["telepathy"])


class TestSelection:
    def _pool(self, n: int) -> list[SourceTrace]:
        return [
            make_trace(POPULATION_GROUP_MIX, COOPERATE if index % 2 else DEFECT, sample_index=index)
            for index in range(n)
        ]

    def test_the_same_seed_picks_the_same_traces(self) -> None:
        first, _ = select_source_traces(
            self._pool(10), prompt_ids=[PROMPT_ID], cell_key="k", n_per_prompt=3, seed=7
        )
        second, _ = select_source_traces(
            self._pool(10), prompt_ids=[PROMPT_ID], cell_key="k", n_per_prompt=3, seed=7
        )
        assert [t.replay_id for t in first[PROMPT_ID]] == [t.replay_id for t in second[PROMPT_ID]]

    def test_a_different_cell_key_picks_a_different_draw(self) -> None:
        pool = self._pool(24)
        one, _ = select_source_traces(
            pool, prompt_ids=[PROMPT_ID], cell_key="own", n_per_prompt=4, seed=0
        )
        other, _ = select_source_traces(
            pool, prompt_ids=[PROMPT_ID], cell_key="other", n_per_prompt=4, seed=0
        )
        assert [t.replay_id for t in one[PROMPT_ID]] != [t.replay_id for t in other[PROMPT_ID]]

    def test_a_thin_pool_is_a_counted_shortfall_not_a_silent_pad(self) -> None:
        chosen, shortfalls = select_source_traces(
            self._pool(2), prompt_ids=[PROMPT_ID], cell_key="k", n_per_prompt=5, seed=0
        )
        assert len(chosen[PROMPT_ID]) == 2
        assert [s.as_payload() for s in shortfalls] == [
            {
                "cell_key": "k",
                "prompt_id": PROMPT_ID,
                "requested": 5,
                "available": 2,
                "rejected_over_budget": 0,
            }
        ]

    def test_a_rejected_candidate_is_skipped_counted_and_replaced(self) -> None:
        pool = self._pool(6)
        rejected_ids = {pool[0].replay_id, pool[1].replay_id}
        chosen, shortfalls = select_source_traces(
            pool,
            prompt_ids=[PROMPT_ID],
            cell_key="k",
            n_per_prompt=4,
            seed=0,
            accept=lambda trace: trace.replay_id not in rejected_ids,
        )
        assert len(chosen[PROMPT_ID]) == 4
        assert not {t.replay_id for t in chosen[PROMPT_ID]} & rejected_ids
        assert shortfalls == []

    def test_a_prompt_with_no_traces_is_a_shortfall_of_the_full_request(self) -> None:
        chosen, shortfalls = select_source_traces(
            self._pool(4),
            prompt_ids=[OTHER_PROMPT_ID],
            cell_key="k",
            n_per_prompt=2,
            seed=0,
        )
        assert chosen[OTHER_PROMPT_ID] == []
        assert shortfalls[0].available == 0

    def test_a_nonpositive_request_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            select_source_traces(
                self._pool(1), prompt_ids=[PROMPT_ID], cell_key="k", n_per_prompt=0, seed=0
            )


class TestInventory:
    def test_the_inventory_splits_by_prompt_order_pass_and_action(self) -> None:
        traces = [
            make_trace(POPULATION_GROUP_MIX, DEFECT, sample_index=0),
            make_trace(POPULATION_GROUP_MIX, COOPERATE, sample_index=1),
            make_trace(POPULATION_GROUP_MIX, DEFECT, prompt_id=OTHER_PROMPT_ID, sample_index=2),
        ]
        report = inventory(traces)[POPULATION_GROUP_MIX]
        assert report["n_usable"] == 3
        assert report["n_prompts"] == 2
        assert report["by_source_action"] == {COOPERATE: 1, DEFECT: 2}
        assert report["by_print_order"] == {LABEL_PRINT_ORDER_CANONICAL: 3}
        assert report["by_pass"] == {SOURCE_PASS: 3}


# --------------------------------------------------------------------------------------
# Records and the mediation arithmetic
# --------------------------------------------------------------------------------------


class TestBuildRecord:
    def test_a_prefilled_record_reads_the_action_after_the_transplanted_close_tag(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose for " + PROMPT_ID)
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT, names_action=True)
        unit = prepare_unit(
            tokenizer,
            rendered,
            cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
            row=make_row(),
            trace=trace,
            sample_index=0,
        )
        record = build_record(
            unit, RawContinuation(f"\n\n<action>{COOP_LABEL}</action>", "end_turn", 9, 100)
        )
        assert record["parsed_action"] == COOPERATE
        assert record["cooperate"] is True
        assert record["source_action"] == DEFECT
        assert record["follows_source"] is False
        assert record["transplant_last_label_mention"] == DEFECT
        assert record["truncated_thinking"] is False
        assert record["direction"] == f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}"

    def test_an_action_stated_inside_the_transplant_cannot_be_read_as_the_answer(self) -> None:
        # A prefilled think block ending "<action>LONG</action></think>" must not score as the
        # recipient's answer; strip_thinking cuts to the final close tag first.
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose for " + PROMPT_ID)
        completion = f"weighing <action>{DEFECT_LABEL}</action>{THINK_CLOSE}tail"
        trace = make_trace(POPULATION_GROUP_MIX, DEFECT)
        tagged = dataclasses.replace(
            trace, completion=completion, thinking=thinking_segment(completion)
        )
        unit = prepare_unit(
            tokenizer,
            rendered,
            cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, POPULATION_GROUP_MIX),
            row=make_row(),
            trace=tagged,
            sample_index=0,
        )
        record = build_record(unit, RawContinuation("I decline to answer", "end_turn", 4, 100))
        assert record["parsed_action"] is None
        assert record["transplant_action_tag"] == DEFECT

    def test_a_no_prefill_record_that_never_closes_reads_as_truncated(self) -> None:
        tokenizer = MergingTokenizer()
        rendered = render_context(tokenizer, "choose")
        unit = prepare_unit(
            tokenizer,
            rendered,
            cell=TransplantCell(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, None),
            row=make_row(),
            trace=None,
            sample_index=0,
        )
        record = build_record(unit, RawContinuation("still deliberating", "max_tokens", 32768, 300))
        assert record["truncated_thinking"] is True
        assert record["hit_answer_cap"] is True
        assert record["cooperate"] is None
        assert record["follows_source"] is None
        assert record["source_action_stratum"] == GROUP_KEY_NONE


def _record(  # noqa: PLR0913 - a summary-shaped record is its cell, its action and its splits
    recipient: str,
    condition: str,
    *,
    cooperate: bool | None,
    source_action: str | None = None,
    tag: str | None = None,
    mention: str | None = None,
    order: str = LABEL_PRINT_ORDER_CANONICAL,
) -> dict[str, Any]:
    """A summary-shaped record, hand-built so the arithmetic is tested on known inputs."""
    action = None if cooperate is None else (COOPERATE if cooperate else DEFECT)
    return {
        "condition_key": TransplantCell(recipient, condition, None).key,
        "recipient_population": recipient,
        "source_condition": condition,
        "label_print_order": order,
        "cooperate": cooperate,
        "parsed_action": action,
        "source_action": source_action,
        "source_action_stratum": GROUP_KEY_NONE if source_action is None else source_action,
        "transplant_action_tag": tag,
        "transplant_last_label_mention": mention,
        "truncated_thinking": False,
        "hit_answer_cap": False,
        "follows_source": None
        if action is None or source_action is None
        else action == source_action,
    }


class TestSummaries:
    def test_rates_carry_their_denominators_and_split_by_print_order(self) -> None:
        records = [
            _record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=True),
            _record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=False),
            _record(
                POPULATION_SELF_GRADED,
                SOURCE_NO_PREFILL,
                cooperate=None,
                order=LABEL_PRINT_ORDER_SWAPPED,
            ),
        ]
        entry = summarise_records(records)[f"{POPULATION_SELF_GRADED}<-{SOURCE_NO_PREFILL}"]
        assert entry["n_completions"] == 3
        assert entry["n_parsed"] == 2
        assert entry["n_parse_failures"] == 1
        assert entry["cooperate_k"] == 1
        assert entry["cooperate_rate"] == pytest.approx(0.5)
        assert entry["by_print_order"][LABEL_PRINT_ORDER_SWAPPED]["cooperate_rate"] is None
        assert entry["follow_n"] == 0
        assert entry["follow_rate"] is None

    def test_the_stated_action_split_separates_naming_from_non_naming_transplants(self) -> None:
        records = [
            _record(
                POPULATION_SELF_GRADED,
                SOURCE_OTHER_ARM,
                cooperate=False,
                source_action=DEFECT,
                mention=DEFECT,
            ),
            _record(POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, cooperate=True, source_action=DEFECT),
        ]
        entry = summarise_records(records)[f"{POPULATION_SELF_GRADED}<-{SOURCE_OTHER_ARM}"]
        by_mention = entry["by_transplant_last_label_mention"]
        assert by_mention[DEFECT]["cooperate_rate"] == pytest.approx(0.0)
        assert by_mention[GROUP_KEY_NONE]["cooperate_rate"] == pytest.approx(1.0)
        assert entry["follow_k"] == 1
        assert entry["follow_rate"] == pytest.approx(0.5)


class TestMediationReadings:
    def _records(self) -> list[dict[str, Any]]:
        # self-graded arm cooperates 4/4 freely and 4/4 on its own reasoning; handed the group-mix
        # arm's reasoning it cooperates 1/4. The group-mix arm cooperates 0/4 freely. So the arm gap
        # is -1.0 and the shift is -0.75: three quarters of the gap travels with the text.
        records: list[dict[str, Any]] = []
        for index in range(4):
            records.append(_record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=True))
            records.append(
                _record(
                    POPULATION_SELF_GRADED,
                    SOURCE_OWN_REASONING,
                    cooperate=True,
                    source_action=COOPERATE,
                )
            )
            records.append(
                _record(
                    POPULATION_SELF_GRADED,
                    SOURCE_OTHER_ARM,
                    cooperate=index == 0,
                    source_action=DEFECT,
                )
            )
            records.append(_record(POPULATION_GROUP_MIX, SOURCE_NO_PREFILL, cooperate=False))
        return records

    def test_the_shift_is_measured_against_own_reasoning_and_scaled_by_the_arm_gap(self) -> None:
        block = mediation_readings(self._records(), [POPULATION_SELF_GRADED])[
            f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}"
        ]["pooled"]
        assert block["cooperate_rate_no_prefill"] == pytest.approx(1.0)
        assert block["cooperate_rate_own_reasoning"] == pytest.approx(1.0)
        assert block["cooperate_rate_other_arm"] == pytest.approx(0.25)
        assert block["cooperate_rate_donor_no_prefill"] == pytest.approx(0.0)
        assert block["prefill_perturbation"] == pytest.approx(0.0)
        assert block["transplant_shift"] == pytest.approx(-0.75)
        assert block["arm_gap"] == pytest.approx(-1.0)
        assert block["mediated_fraction"] == pytest.approx(0.75)
        assert block["denominators"][SOURCE_OTHER_ARM]["n_parsed"] == 4

    def test_a_cell_that_did_not_run_reads_as_none_rather_than_zero(self) -> None:
        block = mediation_readings(
            [_record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=True)],
            [POPULATION_SELF_GRADED],
        )[f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}"]["pooled"]
        assert block["cooperate_rate_other_arm"] is None
        assert block["transplant_shift"] is None
        assert block["mediated_fraction"] is None
        assert block["denominators"][SOURCE_OTHER_ARM] == {
            "cooperate_k": None,
            "n_parsed": None,
            "n_completions": None,
        }

    def test_a_gap_under_the_floor_reports_the_shift_and_no_fraction(self) -> None:
        records = [
            _record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=True),
            _record(POPULATION_GROUP_MIX, SOURCE_NO_PREFILL, cooperate=True),
            _record(
                POPULATION_SELF_GRADED,
                SOURCE_OWN_REASONING,
                cooperate=True,
                source_action=COOPERATE,
            ),
            _record(
                POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, cooperate=False, source_action=DEFECT
            ),
        ]
        block = mediation_readings(records, [POPULATION_SELF_GRADED])[
            f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}"
        ]["pooled"]
        assert block["arm_gap"] == pytest.approx(0.0)
        assert block["transplant_shift"] == pytest.approx(-1.0)
        assert block["mediated_fraction"] is None

    def test_the_named_no_action_subset_keeps_the_no_prefill_baseline(self) -> None:
        records = [
            _record(POPULATION_SELF_GRADED, SOURCE_NO_PREFILL, cooperate=True),
            _record(POPULATION_GROUP_MIX, SOURCE_NO_PREFILL, cooperate=False),
            _record(
                POPULATION_SELF_GRADED,
                SOURCE_OWN_REASONING,
                cooperate=True,
                source_action=COOPERATE,
            ),
            _record(
                POPULATION_SELF_GRADED, SOURCE_OTHER_ARM, cooperate=False, source_action=DEFECT
            ),
            _record(
                POPULATION_SELF_GRADED,
                SOURCE_OTHER_ARM,
                cooperate=True,
                source_action=DEFECT,
                mention=DEFECT,
            ),
        ]
        reading = mediation_readings(records, [POPULATION_SELF_GRADED])[
            f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}"
        ]
        unnamed = reading["transplant_named_no_action"]
        # A no-prefill record carries no transplant, so it never survives the filter; its rate has to
        # come from the unfiltered summary or every difference would silently read None.
        assert unnamed["cooperate_rate_no_prefill"] == pytest.approx(1.0)
        assert unnamed["cooperate_rate_other_arm"] == pytest.approx(0.0)
        assert reading["pooled"]["cooperate_rate_other_arm"] == pytest.approx(0.5)


# --------------------------------------------------------------------------------------
# Raw continuation
# --------------------------------------------------------------------------------------


class TestContinueRaw:
    def test_the_mock_backend_continues_a_context_without_re_templating_it(self) -> None:
        backend = MockBackend(lambda context: f"<<{len(context)}>>")
        continuations = continue_raw(backend, ["alpha", "beta"])
        assert [c.text for c in continuations] == ["<<5>>", "<<4>>"]
        assert continuations[0].stop_reason == "end_turn"

    def test_a_backend_that_cannot_continue_raw_is_refused_by_name(self) -> None:
        class TemplatingBackend:
            model_id = "x"
            transport = "x"

            def generate(self, prompts: list[str]) -> list[str]:
                return prompts

        with pytest.raises(TypeError, match="cannot continue a raw context"):
            continue_raw(TemplatingBackend(), ["alpha"])


# --------------------------------------------------------------------------------------
# The plan (no model) and one end-to-end run through the mock backend
# --------------------------------------------------------------------------------------


def make_plan(
    *, n_per_prompt: int = 2, prompt_ids: Sequence[str] = (PROMPT_ID, OTHER_PROMPT_ID)
) -> MediationPlan:
    """A plan over synthetic rows and a pool wide enough for every cell."""
    traces: list[SourceTrace] = []
    for population, action in (
        (POPULATION_GROUP_MIX, DEFECT),
        (POPULATION_SELF_GRADED, COOPERATE),
        (POPULATION_BASE, COOPERATE),
    ):
        for prompt_id in prompt_ids:
            traces.extend(
                make_trace(
                    population,
                    action if index % 3 else (DEFECT if action == COOPERATE else COOPERATE),
                    prompt_id=prompt_id,
                    sample_index=index,
                    names_action=index % 2 == 0,
                )
                for index in range(6)
            )
    return MediationPlan(
        cells=transplant_cells(),
        rows=[make_row(prompt_id=prompt_id) for prompt_id in prompt_ids],
        traces=traces,
        census=SourceCensus(read=len(traces), kept=len(traces)),
        n_per_prompt=n_per_prompt,
        seed=0,
        source_sampling=SOURCE_SAMPLING,
    )


class TestPlan:
    def test_the_plan_reports_the_grid_and_the_inventory_without_a_model(self) -> None:
        payload = make_plan().as_payload()
        assert len(payload["cells"]) == len(RECIPIENT_POPULATIONS) * len(SOURCE_CONDITIONS)
        assert payload["n_planned_completions_total"] == 8 * 2 * 2
        assert payload["shortfalls"] == []
        assert set(payload["source_inventory"]) == {
            POPULATION_BASE,
            POPULATION_GROUP_MIX,
            POPULATION_SELF_GRADED,
        }
        assert payload["source_sampling"] == SOURCE_SAMPLING
        assert payload["print_order_caveat"] is not None

    def test_a_thin_pool_shows_up_in_the_plan_as_a_shortfall(self) -> None:
        plan = make_plan(n_per_prompt=99)
        payload = plan.as_payload()
        assert payload["shortfalls"]
        prefilled = [
            cell for cell in payload["cells"] if cell["source_condition"] != SOURCE_NO_PREFILL
        ]
        assert all(cell["n_planned_completions"] == 6 * 2 for cell in prefilled)

    def test_the_real_corpus_renders_both_print_orders(self) -> None:
        rows = eval_rows()
        orders = {str(row["label_print_order"]) for row in rows}
        assert orders == {LABEL_PRINT_ORDER_CANONICAL, LABEL_PRINT_ORDER_SWAPPED}
        assert len(rows) == 2 * len(eval_rows([LABEL_PRINT_ORDER_CANONICAL]))


class TestSubmissionWidths:
    """How many contexts each engine call carries (hot-path backlog rank 30).

    Every ``engine.generate`` call is a barrier behind its own slowest continuation, and these
    continuations run from 20k-token contexts to a 65,536-token answer cap, so a fixed width made each
    group of that many wait for its worst row. The default is now one call for the whole cell; a number
    is a persistence knob, since records land per call.
    """

    def test_the_default_is_one_call_for_the_whole_cell(self) -> None:
        """Sabotage-verified: restoring the old fixed 32-context default reads [32, 5] here.

        This is the assertion with teeth on the width rule, and the integration test below cannot be:
        its cells hold four units, so one call for the cell and a chunk of 32 look identical there.
        37 is deliberately not a multiple of any plausible default.
        """
        assert _submission_widths(37, None) == [37]

    def test_an_explicit_width_chunks_and_keeps_the_short_tail(self) -> None:
        assert _submission_widths(8, 3) == [3, 3, 2]
        assert sum(_submission_widths(8, 3)) == 8

    def test_a_non_positive_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be positive or absent"):
            _submission_widths(4, 0)


class TestExecuteTransplant:
    def _run(self, tmp_path: Path, **overrides: Any) -> dict[str, Any]:
        plan = make_plan()
        tokenizer = MergingTokenizer()

        def answer(context: str) -> str:
            label = COOP_LABEL if len(context) % 2 else DEFECT_LABEL
            return f"\n\n<action>{label}</action>"

        lifecycle: list[str] = []

        # Generator rather than Iterator: typeshed deprecates the contextmanager overload that takes
        # an Iterator-returning function, and basedpyright's strict mode makes that an error.
        @contextlib.contextmanager
        def open_backend(population: str) -> Generator[object]:
            # Opens AND closes are recorded, not just opens: on a real card two live vLLM engines
            # starve each other's KV cache, so the contract is that each engine is released before
            # the next is constructed -- and the first real run of this module died at exactly that
            # seam with the previous engine still holding 41 of 44 GiB. An open-order-only assertion
            # would have passed on that run.
            lifecycle.append(f"open {population}")
            try:
                yield MockBackend(answer, model_id=f"mock:{population}")
            finally:
                lifecycle.append(f"release {population}")

        overrides.setdefault("batch_size", 3)
        config = ExecutionConfig(out_dir=tmp_path, max_new_tokens=64, **overrides)
        summary = execute_transplant(
            plan, open_backend=open_backend, tokenizer=tokenizer, config=config
        )
        summary["_backend_lifecycle"] = lifecycle
        return summary

    def _submission_widths_seen(self, tmp_path: Path, **overrides: Any) -> list[int]:
        """Run one whole transplant, recording how many contexts each engine call was handed."""
        widths: list[int] = []
        real = interp_mediation.continue_raw

        def counting(backend: object, contexts: list[str]) -> Any:
            widths.append(len(contexts))
            return real(backend, contexts)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(interp_mediation, "continue_raw", counting)
            self._run(tmp_path, **overrides)
        return widths

    def test_a_cell_is_one_engine_call_by_default(self, tmp_path: Path) -> None:
        """Rank 30's wiring: the default width reaches the engine, one call per cell.

        The width RULE is pinned by :class:`TestSubmissionWidths`, which is where a fixed default shows
        up; what this adds is that the rule is what the driver drives, and that each cell's units all go
        in one submission rather than being split by anything else on the way.
        """
        widths = self._submission_widths_seen(tmp_path, batch_size=None)
        # Eight cells: two recipients times four source conditions, each with all its units at once.
        assert len(widths) == len(transplant_cells())
        assert set(widths) == {4}

    def test_an_explicit_width_still_lands_records_per_chunk(self, tmp_path: Path) -> None:
        widths = self._submission_widths_seen(tmp_path, batch_size=3)
        assert widths == [3, 1] * len(transplant_cells())

    def test_records_and_summary_come_out_aligned(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path)
        lines = (tmp_path / "mediation_records.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == summary["n_records"] == 8 * 2 * 2
        assert set(summary["conditions"]) == {cell.key for cell in transplant_cells()}
        assert summary["skipped_cells"] == []
        assert summary["sabotage"] == SABOTAGE_NONE
        totals = sum(entry["n_completions"] for entry in summary["conditions"].values())
        assert totals == summary["n_records"]

    def test_every_record_names_its_source_and_its_recipient(self, tmp_path: Path) -> None:
        self._run(tmp_path)
        records = [
            json.loads(line)
            for line in (tmp_path / "mediation_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        prefilled = [r for r in records if r["source_condition"] != SOURCE_NO_PREFILL]
        assert prefilled
        assert all(r["source_replay_id"] and r["source_action"] for r in prefilled)
        assert all(r["n_transplant_tokens"] > 0 for r in prefilled)
        baseline = [r for r in records if r["source_condition"] == SOURCE_NO_PREFILL]
        assert all(r["source_replay_id"] is None for r in baseline)

    def test_the_mediation_block_covers_both_directions(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path)
        assert set(summary["mediation"]) == {
            f"{POPULATION_SELF_GRADED}->{POPULATION_GROUP_MIX}",
            f"{POPULATION_GROUP_MIX}->{POPULATION_SELF_GRADED}",
        }
        for block in summary["mediation"].values():
            assert block["pooled"]["denominators"][SOURCE_OWN_REASONING]["n_completions"] == 4

    def test_a_past_deadline_labels_every_skipped_cell(self, tmp_path: Path) -> None:
        summary = self._run(tmp_path, deadline=dt.datetime(2000, 1, 1, tzinfo=dt.UTC))
        assert summary["n_records"] == 0
        assert len(summary["skipped_cells"]) == len(transplant_cells())
        assert all(entry["skipped"] == "deadline" for entry in summary["skipped_cells"])

    def test_one_backend_is_opened_per_recipient_and_released_before_the_next_opens(
        self, tmp_path: Path
    ) -> None:
        summary = self._run(tmp_path)
        assert summary["_backend_lifecycle"] == [
            event
            for population in RECIPIENT_POPULATIONS
            for event in (f"open {population}", f"release {population}")
        ]

    def test_the_second_recipients_records_append_behind_the_first(self, tmp_path: Path) -> None:
        # The records file is opened once and held open across the engine swap. If it were reopened
        # per recipient it would truncate, and the run would end holding only the last recipient's
        # cells while every count in the summary still looked right.
        self._run(tmp_path)
        recipients = [
            json.loads(line)["recipient_population"]
            for line in (tmp_path / "mediation_records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert set(recipients) == set(RECIPIENT_POPULATIONS)
        # Each recipient's records form one contiguous block, and every block survives to the end.
        for population in RECIPIENT_POPULATIONS:
            positions = [index for index, name in enumerate(recipients) if name == population]
            assert positions == list(range(positions[0], positions[-1] + 1))
            assert len(positions) == 4 * 2 * 2

    def test_a_backend_that_fails_to_open_surfaces_rather_than_being_skipped(
        self, tmp_path: Path
    ) -> None:
        plan = make_plan()

        def refuse(population: str) -> AbstractContextManager[object]:
            raise RuntimeError(f"engine for {population} did not load")

        with pytest.raises(RuntimeError, match="did not load"):
            execute_transplant(
                plan,
                open_backend=refuse,
                tokenizer=MergingTokenizer(),
                config=ExecutionConfig(out_dir=tmp_path, max_new_tokens=64),
            )

    def test_the_long_trace_filter_thins_a_cell_and_is_counted(self, tmp_path: Path) -> None:
        # A one-token budget rejects every thinking block here, so each prefilled cell empties. What
        # matters is that the emptying is REPORTED -- as a shortfall naming how many were rejected,
        # and as a skipped cell -- rather than a run that quietly measured only its baselines.
        summary = self._run(tmp_path, max_source_response_tokens=1)
        assert summary["max_source_response_tokens"] == 1
        prefilled_cells = [
            cell.key for cell in transplant_cells() if cell.source_population is not None
        ]
        assert {entry["condition_key"] for entry in summary["skipped_cells"]} == set(
            prefilled_cells
        )
        assert all(entry["skipped"] == "no units selected" for entry in summary["skipped_cells"])
        assert {entry["cell_key"] for entry in summary["shortfalls"]} == set(prefilled_cells)
        assert all(entry["available"] == 0 for entry in summary["shortfalls"])
        assert all(entry["rejected_over_budget"] > 0 for entry in summary["shortfalls"])
        assert set(summary["conditions"]) == {
            cell.key for cell in transplant_cells() if cell.source_population is None
        }


# --------------------------------------------------------------------------------------
# Releasing one engine before the next loads (games/vllm_teardown.py)
# --------------------------------------------------------------------------------------


BASELINE_MIB = 900  # a neighbour already on the card, so "drained" cannot mean "empty"
ENGINE_MIB = 41_000  # what the failing run's engine held: 0.92 x 44.39 GiB
FAST_DRAIN = DrainPolicy(
    shutdown_timeout_s=1.0, drain_timeout_s=0.05, poll_interval_s=0.005, tolerance_mib=1024
)

# A stand-in nvidia-smi that honours the real one's contract: unit-less rows only when the query
# asks for them, `<n> MiB` otherwise. The blank line between the rows is what a real driver emits
# around its output, so the filter that drops it is exercised rather than assumed.
FAKE_SMI_TWO_DEVICES = rf"""case "$*" in
  *--query-gpu=memory.used*--format=csv,noheader,nounits*)
    printf '{BASELINE_MIB}\n\n{ENGINE_MIB}\n' ;;
  *)
    printf '{BASELINE_MIB} MiB\n{ENGINE_MIB} MiB\n' ;;
esac"""
FAKE_SMI_ALWAYS_WITH_UNITS = rf"printf '{BASELINE_MIB} MiB\n'"
HUNG_SMI_SECONDS = 6  # a wedged driver, long enough that an unbounded query is unmistakably slow


class FakeEngineCore:
    """vLLM's ``EngineCoreClient``, reduced to the one call the release makes."""

    def __init__(self, *, frees_mib: int = ENGINE_MIB) -> None:
        self.frees_mib = frees_mib
        self.shutdown_timeouts: list[float | None] = []

    def shutdown(self, timeout: float | None = None) -> None:
        self.shutdown_timeouts.append(timeout)


class FakeLLMEngine:
    """The ``LLM.llm_engine`` link of the attribute chain the release walks."""

    def __init__(self, core: FakeEngineCore) -> None:
        self.engine_core = core


class FakeLLM:
    """The ``VLLMBackend._llm`` link, i.e. what a ``vllm.LLM`` stands in for here."""

    def __init__(self, core: FakeEngineCore) -> None:
        self.llm_engine = FakeLLMEngine(core)


class FakeVLLMBackend:
    """A backend carrying only the chain ``release_engine`` requires.

    A fake rather than the real :class:`~reward_hacking.model_backend.VLLMBackend` because vLLM is
    not installed in this environment at all, and because what these tests are about is the
    teardown's own bookkeeping -- which is the part that was wrong.
    """

    def __init__(self, core: FakeEngineCore) -> None:
        self._llm: FakeLLM | None = FakeLLM(core)


class FakeCard:
    """A GPU whose occupancy the test decides, standing in for ``nvidia-smi``.

    ``holds_forever`` is the sabotage: an engine that was asked to shut down and never gave the card
    back. Without it the drain poll would only ever be watched succeeding, which is the shape of a
    check nobody has seen fail.
    """

    def __init__(self, core: FakeEngineCore, *, baseline_mib: int, holds_forever: bool) -> None:
        self.core = core
        self.baseline_mib = baseline_mib
        self.holds_forever = holds_forever
        self.readings = 0

    def used_mib(self) -> list[int]:
        self.readings += 1
        if self.holds_forever or not self.core.shutdown_timeouts:
            return [self.baseline_mib + self.core.frees_mib]
        return [self.baseline_mib]


class TestEngineRelease:
    """The seam the first real GPU run died at, and the guard that now refuses to pass it.

    The run completed the first recipient's four cells and then failed initialising the second
    engine with ``Free memory on device cuda:0 (2.96/44.39 GiB) on startup is less than desired GPU
    memory utilization (0.92, 40.84 GiB)``. The old teardown was ``del backend`` inside a
    ``@contextmanager`` (which cannot drop the caller's ``with`` target) plus
    ``torch.cuda.empty_cache()`` (which cannot reach a child process's VRAM), so nothing was
    released and nothing noticed.
    """

    def test_the_release_shuts_the_engine_core_down_and_reports_the_drain(self) -> None:
        core = FakeEngineCore()
        backend = FakeVLLMBackend(core)
        card = FakeCard(core, baseline_mib=BASELINE_MIB, holds_forever=False)
        report = release_engine(
            backend,
            baseline_mib=[BASELINE_MIB],
            read_used_mib=card.used_mib,
            policy=FAST_DRAIN,
        )
        assert core.shutdown_timeouts == [FAST_DRAIN.shutdown_timeout_s]
        assert backend._llm is None
        assert report["released_used_mib"] == [BASELINE_MIB]
        assert report["residue_mib"] == [0]

    def test_a_card_that_never_gives_the_memory_back_fails_loud(self) -> None:
        # The sabotage. The engine is asked to shut down and the card keeps reporting it as held, so
        # the release must raise rather than let the next recipient's engine meet a full card.
        core = FakeEngineCore()
        card = FakeCard(core, baseline_mib=BASELINE_MIB, holds_forever=True)
        with pytest.raises(EngineDrainError, match=r"did not give its memory back"):
            release_engine(
                FakeVLLMBackend(core),
                baseline_mib=[BASELINE_MIB],
                read_used_mib=card.used_mib,
                policy=FAST_DRAIN,
            )
        assert core.shutdown_timeouts, "the release gave up before even asking vLLM to shut down"
        assert card.readings > 1, "the drain was asserted from a single reading, not polled"

    def test_the_undrained_error_names_the_residue_and_how_to_find_the_holder(self) -> None:
        # A red gate nobody can act on gets ignored, so the message carries the numbers and the
        # nvidia-smi query that names the process still holding the card.
        core = FakeEngineCore()
        card = FakeCard(core, baseline_mib=BASELINE_MIB, holds_forever=True)
        with pytest.raises(EngineDrainError) as caught:
            release_engine(
                FakeVLLMBackend(core),
                baseline_mib=[BASELINE_MIB],
                read_used_mib=card.used_mib,
                policy=FAST_DRAIN,
            )
        message = str(caught.value)
        assert str(ENGINE_MIB) in message
        assert str(BASELINE_MIB) in message
        assert "query-compute-apps" in message

    def test_a_neighbour_sized_residue_still_counts_as_drained(self) -> None:
        # An idle CUDA context is a few hundred MiB and the driver lags; the check has to tolerate
        # that without tolerating a live engine, so both sides of the tolerance are asserted.
        core = FakeEngineCore(frees_mib=FAST_DRAIN.tolerance_mib - 1)
        card = FakeCard(core, baseline_mib=BASELINE_MIB, holds_forever=True)
        assert await_vram_drain([BASELINE_MIB], read_used_mib=card.used_mib, policy=FAST_DRAIN) == [
            BASELINE_MIB + FAST_DRAIN.tolerance_mib - 1
        ]

        over = FakeCard(
            FakeEngineCore(frees_mib=FAST_DRAIN.tolerance_mib + 1),
            baseline_mib=BASELINE_MIB,
            holds_forever=True,
        )
        with pytest.raises(EngineDrainError, match=r"did not give its memory back"):
            await_vram_drain([BASELINE_MIB], read_used_mib=over.used_mib, policy=FAST_DRAIN)

    def test_a_backend_without_vllms_shutdown_is_refused_rather_than_skipped(self) -> None:
        # The regression this guards: a vLLM upgrade that moves EngineCoreClient.shutdown would
        # otherwise turn the teardown back into a silent no-op, which is the original bug.
        with pytest.raises(EngineDrainError, match=r"has no _llm"):
            resolve_engine_shutdown(object())

    def test_a_shutdown_that_is_not_callable_is_refused(self) -> None:
        backend = FakeVLLMBackend(FakeEngineCore())
        assert backend._llm is not None
        backend._llm.llm_engine.engine_core.shutdown = "moved upstream"  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(EngineDrainError, match=r"not callable"):
            resolve_engine_shutdown(backend)

    def test_a_reading_over_a_different_number_of_devices_is_refused(self) -> None:
        with pytest.raises(EngineDrainError, match=r"not of the same card"):
            await_vram_drain([BASELINE_MIB, BASELINE_MIB], read_used_mib=lambda: [BASELINE_MIB])

    def test_an_unaccountable_device_reads_as_occupied_not_as_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # nvidia-smi prints [N/A] for a device it can see but not inspect, and reading that as zero
        # is how a full card passes a memory check -- the trap scripts/gpu_preflight.py documents.
        monkeypatch.setattr(vllm_teardown, "_nvidia_smi_gpu_rows", lambda _fields: ["[N/A]"])
        with pytest.raises(EngineDrainError, match=r"Unknown is not zero"):
            vram_used_mib()

    def test_a_host_with_no_gpu_at_all_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vllm_teardown, "_nvidia_smi_gpu_rows", lambda _fields: [])
        with pytest.raises(EngineDrainError, match=r"listed no GPU"):
            vram_used_mib()

    def _stub_engine_load(
        self, monkeypatch: pytest.MonkeyPatch, *, engine: FakeVLLMBackend, events: list[str]
    ) -> None:
        """Stub every heavyweight call `backend_opener` makes, recording the order it makes them in.

        The order is what the failing run got wrong, so it is what gets asserted: a baseline read
        AFTER construction would already include the engine's own claim, and the residue would come
        out zero however much of the card stayed held.
        """
        served = ServedModel(
            model_id=DEFAULT_BASE_MODEL, load_mode="runtime-adapter", adapter_dir=None
        )
        monkeypatch.setattr(interp_mediation, "adapter_for", lambda _population, root: root)
        monkeypatch.setattr(interp_mediation, "resolve_served_model", lambda **_kwargs: served)
        monkeypatch.setattr(interp_mediation, "verify_served_model", lambda *_args: None)

        def read_card() -> list[int]:
            events.append("read card")
            return [BASELINE_MIB]

        def build_engine(*_args: object, **_kwargs: object) -> FakeVLLMBackend:
            events.append("build engine")
            return engine

        def release(backend: object, **kwargs: object) -> dict[str, object]:
            events.append(f"release {kwargs['baseline_mib']}")
            return {"released_backend_is_the_engine": backend is engine}

        monkeypatch.setattr(interp_mediation, "vram_used_mib", read_card)
        monkeypatch.setattr(interp_mediation.backend_cli, "backend_from_args", build_engine)
        monkeypatch.setattr(interp_mediation, "release_engine", release)

    def test_the_opener_reads_the_card_before_the_engine_and_releases_it_after(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        events: list[str] = []
        engine = FakeVLLMBackend(FakeEngineCore())
        self._stub_engine_load(monkeypatch, engine=engine, events=events)
        args = resolve_args(["--out-dir", str(tmp_path)])
        with backend_opener(args, make_plan())(POPULATION_GROUP_MIX) as backend:
            events.append("generate")
            assert backend is engine
        assert events == ["read card", "build engine", "generate", f"release {[BASELINE_MIB]}"]

    def test_the_opener_releases_even_when_a_cell_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An engine left up after a failed cell takes the card down for every later recipient, so the
        # release sits in a `finally` rather than on the success path only.
        events: list[str] = []
        self._stub_engine_load(monkeypatch, engine=FakeVLLMBackend(FakeEngineCore()), events=events)
        opener = backend_opener(resolve_args(["--out-dir", str(tmp_path)]), make_plan())
        with pytest.raises(RuntimeError, match="a cell blew up"), opener(POPULATION_GROUP_MIX):
            raise RuntimeError("a cell blew up")
        assert events == ["read card", "build engine", f"release {[BASELINE_MIB]}"]

    def test_the_opener_releases_when_the_served_model_check_rejects_the_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # verify_served_model exists to raise on the un-merged adapter that served base weights
        # without saying so, and it raises with the engine ALREADY loaded. Called above the `try`,
        # that one failure is the one case where the card stays held for every later recipient.
        events: list[str] = []
        self._stub_engine_load(monkeypatch, engine=FakeVLLMBackend(FakeEngineCore()), events=events)

        def refuse(*_args: object) -> None:
            events.append("verify refused")
            raise ValueError("served the base checkpoint rather than the adapter")

        monkeypatch.setattr(interp_mediation, "verify_served_model", refuse)
        opener = backend_opener(resolve_args(["--out-dir", str(tmp_path)]), make_plan())
        with (
            pytest.raises(ValueError, match="rather than the adapter"),
            opener(POPULATION_GROUP_MIX),
        ):
            events.append("generate")
        assert events == [
            "read card",
            "build engine",
            "verify refused",
            f"release {[BASELINE_MIB]}",
        ]

    def test_the_mock_backend_is_never_put_through_the_vllm_release(self, tmp_path: Path) -> None:
        # The mock path has no engine and no card, so neither the baseline read nor the release may
        # run on it -- both would raise on a host with no GPU and turn every offline smoke red.
        args = resolve_args(["--out-dir", str(tmp_path), "--backend", "mock"])
        with backend_opener(args, make_plan())(POPULATION_GROUP_MIX) as backend:
            assert isinstance(backend, MockBackend)


class TestReadingTheCard:
    """`vram_used_mib`'s success path and its bound, which its failure branches take for granted.

    Driven through a fake `nvidia-smi` on PATH rather than a stubbed `_nvidia_smi_gpu_rows`, because
    the query flags are half of what makes a reading parseable: without `nounits` a real driver
    prints `900 MiB`, which this module correctly refuses as non-numeric. A test that stubbed the
    rows out would pass under a query that could never work on a card.
    """

    def _put_fake_smi_on_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
    ) -> None:
        script = tmp_path / "nvidia-smi"
        script.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
        script.chmod(0o755)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    def test_two_devices_parse_unit_less_in_device_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._put_fake_smi_on_path(tmp_path, monkeypatch, FAKE_SMI_TWO_DEVICES)
        # Order is the reading: `await_vram_drain` subtracts device i's baseline from device i, so a
        # reversal would compare one card against another. The blank line is what nvidia-smi emits
        # around its rows, and an unfiltered one parses as neither a number nor nothing.
        assert vram_used_mib() == [BASELINE_MIB, ENGINE_MIB]

    def test_a_reading_that_carries_units_is_refused_rather_than_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The fake driver answers `900 MiB` when the query omits `nounits`, exactly as a real one
        # does, so dropping that flag turns the reading into the '[N/A]' case instead of a number
        # silently parsed off the front.
        self._put_fake_smi_on_path(tmp_path, monkeypatch, FAKE_SMI_ALWAYS_WITH_UNITS)
        with pytest.raises(EngineDrainError, match=r"Unknown is not zero"):
            vram_used_mib()

    def test_a_hung_nvidia_smi_is_bounded_and_counts_the_card_as_occupied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # nvidia-smi blocks indefinitely on a wedged driver and during another process's teardown,
        # which is exactly when this module runs. Unbounded, one hung query sits inside the drain
        # poll forever, so the drain deadline that makes the release a check is never reached.
        self._put_fake_smi_on_path(tmp_path, monkeypatch, f"exec /bin/sleep {HUNG_SMI_SECONDS}")
        monkeypatch.setattr(vllm_teardown, "NVIDIA_SMI_TIMEOUT_S", 0.2)
        started = time.monotonic()
        # The elapsed bound is asserted first and separately from the message, because unbounded
        # this raises anyway: it waits the wedge out, gets empty stdout, and says "listed no GPU" --
        # so a message-only assertion goes red for the wrong reason and leaves the bound untested.
        with pytest.raises(EngineDrainError) as raised:
            vram_used_mib()
        elapsed = time.monotonic() - started
        assert elapsed < HUNG_SMI_SECONDS / 2, f"the query ran {elapsed:.1f}s against a 0.2s bound"
        assert "did not answer within" in str(raised.value)

    def test_a_host_without_nvidia_smi_at_all_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", str(tmp_path))
        with pytest.raises(EngineDrainError, match=r"not on PATH"):
            vram_used_mib()


class TestBaselineGate:
    """Which backend kinds get a card reading at all, decided from the DECLARED kind.

    Probing the backend object for an engine handle would be the same silent no-op
    `resolve_engine_shutdown` refuses to become: the day the attribute moves, every site quietly
    stops releasing and every gate stays green.
    """

    def test_the_vllm_kind_reads_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(vllm_teardown, "vram_used_mib", lambda: [BASELINE_MIB])
        assert baseline_before_engine(vllm_teardown.VLLM_BACKEND_KIND) == [BASELINE_MIB]

    @pytest.mark.parametrize("kind", ["mock", "hf", "bedrock", "codex"])
    def test_every_other_kind_reads_no_card(
        self, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A card read on a host with no GPU raises, so a kind that loads no engine must not reach it
        # -- that is what keeps every offline smoke of these drivers runnable on CPU.
        def refuse() -> list[int]:
            pytest.fail(f"the {kind} backend read the card")

        monkeypatch.setattr(vllm_teardown, "vram_used_mib", refuse)
        assert baseline_before_engine(kind) is None


class TestMockPath:
    """The mock backend is the CLI's only offline smoke, so its two honesty properties are tested."""

    def test_the_mock_closes_an_open_thinking_block_before_answering(self) -> None:
        # Without this the no-prefill baseline reads as truncated, its rate comes back None, and
        # every mediation difference the smoke exists to exercise silently vanishes.
        rows = [make_row()]
        answer = mock_answers(rows)
        open_context = render_context(MergingTokenizer(), str(rows[0]["prompt"]))
        assert THINK_CLOSE in answer(open_context)
        assert THINK_CLOSE not in answer(open_context + f"reasoning{THINK_CLOSE}")

    def test_the_mock_names_a_label_the_row_actually_offers(self) -> None:
        rows = [make_row()]
        visible = mock_answers(rows)(render_context(MergingTokenizer(), str(rows[0]["prompt"])))
        assert f"<action>{COOP_LABEL}</action>" in visible or (
            f"<action>{DEFECT_LABEL}</action>" in visible
        )

    def test_a_context_matching_no_planned_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match="matching no planned prompt row"):
            mock_answers([make_row()])("a context from some other corpus")

    def test_a_mock_run_records_no_applied_sampler(self) -> None:
        args = argparse.Namespace(backend="mock", max_new_tokens=None)
        record = sampler_record(args, recipient_sampling(32768))
        assert record["engine"] == "mock"
        assert record["applied"] == {}
        assert record["temperature_applied"] is None
        assert record["temperature_requested"] == 1.0


class TestResolveSampling:
    def test_an_unset_budget_resolves_above_the_measured_answer_tail(self) -> None:
        args = argparse.Namespace(backend="mock", max_new_tokens=None)
        assert resolve_sampling(args).max_new_tokens == DEFAULT_EVAL_MAX_NEW_TOKENS

    def test_an_explicit_budget_wins(self) -> None:
        args = argparse.Namespace(backend="mock", max_new_tokens=4096)
        assert resolve_sampling(args).max_new_tokens == 4096

    def test_the_resolved_sampler_is_the_training_distribution(self) -> None:
        sampling = resolve_sampling(argparse.Namespace(backend="mock", max_new_tokens=None))
        assert (sampling.temperature, sampling.top_p, sampling.top_k) == (1.0, 1.0, 0)
        assert sampling.presence_penalty == 0.0
