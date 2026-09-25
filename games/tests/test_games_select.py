"""Offline tests for the baseline sweep and mixed-prompt selection.

No model loads, no network, no GPU. The policy is a `ScriptedBackend` holding a queue of
completions per raw prompt, which is what lets a test say "this prompt comes back split two
ways and that one comes back unanimous" and then assert on the verdict.

The one thing worth stating up front: `reward_hacking.model_backend.MockBackend` is not usable
here. Its cursor advances globally across the whole response list rather than per prompt, so a
sweep that samples prompt A eight times and prompt B eight times cannot be scripted with it --
the completions would interleave by call order instead of by prompt.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, cast

import pytest
import torch

from games import select_prompts
from games.chunked_decode import (
    SELF_SCHEDULING_CHUNK_SEQUENCES,
    THRASH_RETRIES_PER_MINUTE_PER_SEQUENCE,
    THROUGHPUT_KNEE_SEQUENCE_TOKENS,
    AllocatorProbe,
    DecodeFootprint,
    ThrashVerdict,
    backend_schedules_own_batch,
    calibrated_peak_allocated_gib,
    decode_footprint,
    decode_in_chunks,
    derive_chunk_size,
    knee_width_cap,
    local_decode_model_id,
    self_scheduling_chunk_cap,
    sweep_chunk_size,
)
from games.payoffs import STATED_RETURN_UNSET
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDER_SWAPPED,
    OPP_COOP_PROB_UNSET,
    ROW_COLUMNS,
    generate_prompt_rows,
)
from games.provenance import git_provenance
from games.rewards import (
    GRADING_FORMAT_ONLY,
    GRADING_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    care_grading,
)
from games.select_prompts import (
    DEFAULT_MAX_COOP,
    DEFAULT_MIN_COOP,
    DEFAULT_MIN_PARSEABLE_FRACTION,
    DEFAULT_MIN_SPLIT_STD,
    DEFAULT_OUT_DIR,
    DEFAULT_SAMPLES_PER_PROMPT,
    FROZEN_OPPONENT_RECORD_KIND,
    META_RECORD_KIND,
    ONE_SHOT_ACTION_GRADINGS,
    SWEEP_RECORD_KIND,
    DropReason,
    PromptSweepRecord,
    backend_provenance,
    fill_opponent_probs,
    frozen_opponent_coop_probs,
    is_counterbalanced,
    judge_prompts,
    pair_identity,
    read_jsonl,
    sample_frozen_opponent,
    select_mixed_prompts,
    selection_summary,
    sweep_frozen_opponent,
    sweep_meta,
    sweep_prompts,
    training_sampler,
    write_corpus,
    write_sweep_trace,
)
from games.sizing import SequenceCost
from games.termination import MEASURED_TERMINATION_BUDGET, required_completion_budget
from reward_hacking import backend_cli
from reward_hacking.model_backend import Backend, HFBackend, VLLMBackend

COOP_LABEL = "HOLD"
DEFECT_LABEL = "SLASH"
ENDOWMENT = 10

# Completions as a template-prefilled model emits them: the opening <think> came with the prompt,
# so only the closing tag can appear.
PREFILLED_PICKS_HOLD = f"the sheet says</think><action>{COOP_LABEL}</action>"
PREFILLED_PICKS_SLASH = f"the sheet says</think><action>{DEFECT_LABEL}</action>"
PREFILLED_TRUNCATED_THINKING = "still weighing the two options and running out of"

# Completions as a hosted Converse endpoint hands them back: the reasoning went into
# `reasoningContent` and `BedrockBackend.generate` returns the answer blocks only, so no thinking
# tag of either kind can appear no matter what the served model's own template does.
HOSTED_PICKS_HOLD = f"<action>{COOP_LABEL}</action>"
HOSTED_PICKS_SLASH = f"<action>{DEFECT_LABEL}</action>"

# Completions as a non-prefilling model emits them: both tags in the completion.
PICKS_HOLD = f"<think>weighing</think><action>{COOP_LABEL}</action>"
PICKS_SLASH = f"<think>weighing</think><action>{DEFECT_LABEL}</action>"
# The same two completions with a multi-byte character inside the thinking block. Qwen3.5 traces
# carry CJK and Unicode punctuation routinely and the partial file is written with
# ensure_ascii=False, so a death can cut the trailing line in the middle of a character's bytes.
MULTIBYTE_PICKS_HOLD = f"<think>weighing 权衡</think><action>{COOP_LABEL}</action>"
MULTIBYTE_PICKS_SLASH = f"<think>weighing 权衡</think><action>{DEFECT_LABEL}</action>"
NO_ANSWER = "<think>weighing</think>I would rather not commit to either."
TRUNCATED_THINKING = "<think>weighing the two options at length and never finishing"


class ScriptedBackend:
    """A policy stand-in that serves a fixed queue of completions per raw prompt.

    The queue wraps, so a two-entry script answers a four-sample sweep by alternating. An
    unscripted prompt raises rather than returning something plausible: a sweep silently sampling
    a prompt nobody wrote completions for would produce a table of numbers about nothing.
    """

    transport = "scripted"

    def __init__(self, script: dict[str, list[str]], model_id: str = "scripted/policy") -> None:
        self.model_id = model_id
        self._script = {prompt: list(queue) for prompt, queue in script.items()}
        self._cursor = dict.fromkeys(script, 0)
        self.batch_sizes: list[int] = []

    def generate(self, prompts: list[str]) -> list[str]:
        self.batch_sizes.append(len(prompts))
        completions: list[str] = []
        for prompt in prompts:
            if prompt not in self._script:
                raise KeyError(f"no scripted completions for prompt {prompt!r}")
            queue = self._script[prompt]
            completions.append(queue[self._cursor[prompt] % len(queue)])
            self._cursor[prompt] += 1
        return completions


class ThrottledBackend:
    """A hosted stand-in that raises the way a Converse throttle does, on every call.

    Exists to fail the frozen-opponent pass of a run whose policy sweep has already completed, which
    is the case that used to take the whole sweep down with it.
    """

    transport = "bedrock-converse"

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def generate(self, prompts: list[str]) -> list[str]:
        raise RuntimeError(f"throttled with {len(prompts)} prompts in flight")


class OomAboveWidthBackend(ScriptedBackend):
    """A scripted policy that refuses any chunk wider than `fits`, the way a full card does.

    The whole point is to reach the halving path without a GPU: it raises the exact class the CUDA
    allocator raises, so a test that passes here says something about the branch a real OOM takes
    rather than about a stand-in exception nothing else would produce.
    """

    def __init__(
        self, script: dict[str, list[str]], *, fits: int, model_id: str = "scripted/policy"
    ) -> None:
        super().__init__(script, model_id)
        self.fits = fits
        self.attempted_widths: list[int] = []

    def generate(self, prompts: list[str]) -> list[str]:
        self.attempted_widths.append(len(prompts))
        if len(prompts) > self.fits:
            raise torch.OutOfMemoryError(
                f"CUDA out of memory (stand-in): tried to decode {len(prompts)} sequences where "
                f"{self.fits} fit"
            )
        return super().generate(prompts)


# 40 retries over 120 seconds is 2.5 per minute per sequence at width 8, past all three thrash
# thresholds (8 retries, 60 seconds, 0.12 per minute per sequence) rather than near any of them, so
# the tests below measure the response to a verdict and not where the verdict flips.
THRASHING_RETRIES = 40
THRASHING_SECONDS = 120.0

WIDER_THAN_ANY_THRASH_CHUNK = 1000


def thrash_probe(backend: OomAboveWidthBackend, *, sustained: bool) -> AllocatorProbe:
    """Report allocator thrash on the backend's first decode, or on every one of them.

    Keyed to the backend's own record of completed `generate` calls rather than to a queue of canned
    numbers. The detector reads each of its two values twice per chunk and in a fixed order, so a
    queue would pin these tests to that call order instead of to the behaviour under test; counting
    decodes leaves "which chunk is this" the only thing the probe depends on.

    Under `sustained=False` the clock advances a single second per later chunk, so nothing after the
    first chunk clears the 60-second floor and the width the first chunk narrowed to stays put --
    which is what makes a carried-forward narrowing visible as a width rather than as a slope.
    """

    def decodes_done() -> int:
        return len(backend.attempted_widths)

    def read_retries() -> int:
        done = decodes_done()
        return THRASHING_RETRIES * (done if sustained else min(done, 1))

    def clock() -> float:
        done = decodes_done()
        if sustained:
            return THRASHING_SECONDS * done
        return 0.0 if done == 0 else THRASHING_SECONDS + (done - 1)

    return AllocatorProbe(clock=clock, read_retries=read_retries)


def matrix_row(  # noqa: PLR0913
    *,
    prompt_id: str,
    grading: str = select_prompts.GRADING_GROUP_MIX,
    coop_label: str = COOP_LABEL,
    reskin_id: str = "warehouse-shift-handover",
    payoff_variant: str = "temptation-2",
    game_id: str = "twin-pd-group",
    n_rounds: int = 0,
    opponent_rule: str = "",
    prompt: str | None = None,
) -> dict[str, object]:
    """Build one matrix-game row with the real `ROW_COLUMNS` schema."""
    return {
        "prompt": prompt if prompt is not None else f"sheet for {prompt_id}",
        "prompt_id": prompt_id,
        "game_id": game_id,
        "grading": grading,
        "payoff_cc": 0.6,
        "payoff_cd": 0.0,
        "payoff_dc": 1.0,
        "payoff_dd": 0.2,
        "label_a": COOP_LABEL,
        "label_b": DEFECT_LABEL,
        "coop_label": coop_label,
        "endowment": 0,
        "opp_coop_prob": -1.0,
        "opponent_rule": opponent_rule,
        "n_rounds": n_rounds,
        "reskin_id": reskin_id,
        "payoff_variant": payoff_variant,
        "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
    }


def counterbalanced_pair(reskin_id: str, **kwargs: object) -> list[dict[str, object]]:
    """Build the two label orientations of one scenario, as the corpus builder does."""
    return [
        matrix_row(
            prompt_id=f"{reskin_id}--coop0",
            coop_label=COOP_LABEL,
            reskin_id=reskin_id,
            **kwargs,  # pyright: ignore[reportArgumentType]
        ),
        matrix_row(
            prompt_id=f"{reskin_id}--coop1",
            coop_label=DEFECT_LABEL,
            reskin_id=reskin_id,
            **kwargs,  # pyright: ignore[reportArgumentType]
        ),
    ]


def dictator_row(*, prompt_id: str, reskin_id: str) -> dict[str, object]:
    """Build one unilateral-split row: no labels to counterbalance, a real endowment."""
    return {
        "prompt": f"sheet for {prompt_id}",
        "prompt_id": prompt_id,
        "game_id": "dictator",
        "grading": select_prompts.GRADING_KEEP_FRACTION,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": ENDOWMENT,
        "opp_coop_prob": -1.0,
        "opponent_rule": "",
        "n_rounds": 0,
        "reskin_id": reskin_id,
        "payoff_variant": "single",
    }


def sweep(
    rows: list[dict[str, object]],
    script: dict[str, list[str]],
    *,
    samples_per_prompt: int = 4,
    prefilled_think: bool = False,
    chunk_size: int | None = None,
) -> list[PromptSweepRecord]:
    """Run a sweep over `rows` with per-prompt scripted completions."""
    backend = ScriptedBackend({row["prompt"]: script[row["prompt_id"]] for row in rows})  # pyright: ignore[reportArgumentType]
    return sweep_prompts(
        backend,
        rows,
        samples_per_prompt=samples_per_prompt,
        prefilled_think=prefilled_think,
        chunk_size=chunk_size,
    )


def json_record(records: list[dict[str, object]], index: int) -> dict[str, Any]:
    """Return one loaded JSONL record with `Any` values; JSON nesting has no static shape."""
    return cast("dict[str, Any]", records[index])


def reasons(records: list[PromptSweepRecord], **kwargs: float) -> dict[str, str]:
    """Map prompt_id to the verdict reason, for compact assertions."""
    return {
        verdict.prompt_id: verdict.reason
        for verdict in judge_prompts(records, **kwargs)  # pyright: ignore[reportArgumentType]
    }


class TestSweepAccounting:
    def test_counts_cooperation_failures_and_truncation(self):
        rows = [matrix_row(prompt_id="one")]
        records = sweep(
            rows,
            {"one": [PICKS_HOLD, PICKS_SLASH, NO_ANSWER, TRUNCATED_THINKING]},
        )
        record = records[0]

        assert record.n_samples == 4
        assert record.coop_count == 1
        assert record.coop_total == 2
        assert record.coop_fraction == pytest.approx(0.5)
        assert record.n_parse_failures == 2
        assert record.n_truncated_thinking == 1
        assert record.parseable_fraction == pytest.approx(0.5)
        assert record.record_kind == SWEEP_RECORD_KIND
        assert [sample.action for sample in record.samples] == ["C", "D", None, None]

    def test_prefilled_think_reads_a_bare_answer_as_truncated(self):
        rows = [matrix_row(prompt_id="one")]
        script = {
            "one": [
                PREFILLED_PICKS_HOLD,
                PREFILLED_PICKS_SLASH,
                PREFILLED_TRUNCATED_THINKING,
                PREFILLED_PICKS_HOLD,
            ]
        }

        prefilled = sweep(rows, script, prefilled_think=True)[0]
        assert prefilled.n_truncated_thinking == 1
        assert prefilled.coop_count == 2
        assert prefilled.coop_total == 3

        # The same completion under the wrong flag reads as a plain answer with no tag, which is
        # a parse failure rather than a truncation: the two rates say different things about a run.
        not_prefilled = sweep(rows, script, prefilled_think=False)[0]
        assert not_prefilled.n_truncated_thinking == 0
        assert not_prefilled.n_parse_failures == 1

    def test_an_action_inside_the_thinking_block_is_ignored(self):
        rows = [matrix_row(prompt_id="one")]
        buried = (
            f"<think>first instinct is <action>{COOP_LABEL}</action></think>"
            f"<action>{DEFECT_LABEL}</action>"
        )
        record = sweep(rows, {"one": [buried, buried]})[0]
        assert [sample.action for sample in record.samples] == ["D", "D", "D", "D"]

    def test_chunking_keeps_each_prompt_with_its_own_completions(self):
        rows = [matrix_row(prompt_id="one"), matrix_row(prompt_id="two", reskin_id="rope-team")]
        records = sweep(
            rows,
            {"one": [PICKS_HOLD], "two": [PICKS_SLASH]},
            samples_per_prompt=4,
            chunk_size=3,
        )
        assert [record.coop_fraction for record in records] == [1.0, 0.0]

    def test_a_single_sample_cannot_show_a_split_and_is_refused(self):
        with pytest.raises(ValueError, match="split action distribution"):
            sweep([matrix_row(prompt_id="one")], {"one": [PICKS_HOLD]}, samples_per_prompt=1)

    def test_a_missing_schema_column_names_what_was_present(self):
        row = matrix_row(prompt_id="one")
        del row["coop_label"]
        with pytest.raises(KeyError, match="coop_label"):
            sweep([row], {"one": [PICKS_HOLD]})


class TestSelectMixedPrompts:
    def test_a_mixed_pair_is_kept_and_pure_pairs_are_dropped(self):
        rows = [
            *counterbalanced_pair("mixed-frame"),
            *counterbalanced_pair("always-coop-frame"),
            *counterbalanced_pair("always-defect-frame"),
        ]
        script: dict[str, list[str]] = {}
        for orientation in ("coop0", "coop1"):
            script[f"mixed-frame--{orientation}"] = [PICKS_HOLD, PICKS_SLASH]
            script[f"always-coop-frame--{orientation}"] = [PICKS_HOLD]
            script[f"always-defect-frame--{orientation}"] = [PICKS_SLASH]
        records = sweep(rows, script)

        kept = {row["prompt_id"] for row in select_mixed_prompts(records)}
        assert kept == {"mixed-frame--coop0", "mixed-frame--coop1"}

        by_prompt = reasons(records)
        assert by_prompt["always-coop-frame--coop0"] == DropReason.COOP_FRACTION_ABOVE_MAX
        assert by_prompt["always-defect-frame--coop0"] == DropReason.COOP_FRACTION_BELOW_MIN

    def test_both_orientations_go_when_only_one_is_mixed(self):
        # A model that always writes HOLD looks unanimously cooperative in the orientation where
        # HOLD is the cooperative option and unanimously uncooperative in the other, which is what
        # counterbalancing is for -- and why the pair has to be judged as a unit.
        rows = counterbalanced_pair("half-mixed-frame")
        records = sweep(
            rows,
            {
                "half-mixed-frame--coop0": [PICKS_HOLD, PICKS_SLASH],
                "half-mixed-frame--coop1": [PICKS_HOLD],
            },
        )
        assert select_mixed_prompts(records) == []

        by_prompt = reasons(records)
        assert by_prompt["half-mixed-frame--coop0"] == DropReason.PARTNER_DROPPED
        assert by_prompt["half-mixed-frame--coop1"] == DropReason.COOP_FRACTION_BELOW_MIN

    def test_half_the_samples_parsing_is_enough_and_a_quarter_is_not(self):
        rows = [*counterbalanced_pair("half-parses"), *counterbalanced_pair("mostly-unparseable")]
        script: dict[str, list[str]] = {}
        for orientation in ("coop0", "coop1"):
            script[f"half-parses--{orientation}"] = [
                PICKS_HOLD,
                PICKS_SLASH,
                NO_ANSWER,
                TRUNCATED_THINKING,
            ]
            script[f"mostly-unparseable--{orientation}"] = [
                PICKS_HOLD,
                NO_ANSWER,
                NO_ANSWER,
                TRUNCATED_THINKING,
            ]
        records = sweep(rows, script)

        kept = {row["prompt_id"] for row in select_mixed_prompts(records)}
        assert kept == {"half-parses--coop0", "half-parses--coop1"}
        assert reasons(records)["mostly-unparseable--coop0"] == DropReason.TOO_FEW_PARSEABLE

    def test_a_duplicate_prompt_id_is_refused(self):
        rows = [matrix_row(prompt_id="one"), matrix_row(prompt_id="one", reskin_id="rope-team")]
        records = sweep(rows, {"one": [PICKS_HOLD, PICKS_SLASH]})
        with pytest.raises(ValueError, match="prompt_id must be unique"):
            select_mixed_prompts(records)

    def test_a_counterbalanced_row_with_no_partner_is_refused(self):
        """A warning cannot stop the thing the coupling exists to stop.

        The surviving orientation kept its own keep verdict and was written into the corpus, so the
        cooperative option sat in one position more often than chance -- exactly the position bias
        rendering both orientations exists to cancel. The `>2` case next door already raises.
        """
        rows = counterbalanced_pair("mixed-frame")[:1]
        records = sweep(rows, {"mixed-frame--coop0": [PICKS_HOLD, PICKS_SLASH]})
        with pytest.raises(ValueError, match="no counterbalanced partner"):
            select_mixed_prompts(records)

    def test_more_than_two_rows_sharing_a_pair_key_is_refused(self):
        rows = [
            matrix_row(prompt_id="one"),
            matrix_row(prompt_id="two"),
            matrix_row(prompt_id="three"),
        ]
        records = sweep(
            rows,
            {name: [PICKS_HOLD, PICKS_SLASH] for name in ("one", "two", "three")},
        )
        with pytest.raises(ValueError, match="counterbalanced-pair key"):
            select_mixed_prompts(records)

    def test_selected_rows_are_copies(self):
        records = sweep(
            counterbalanced_pair("mixed-frame"),
            {
                name: [PICKS_HOLD, PICKS_SLASH]
                for name in ("mixed-frame--coop0", "mixed-frame--coop1")
            },
        )
        selected = select_mixed_prompts(records)
        selected[0]["opp_coop_prob"] = 0.9
        assert records[0].row["opp_coop_prob"] == -1.0


class TestContinuousScoreGradings:
    def test_dictator_keys_on_kept_fraction_spread(self):
        rows = [
            dictator_row(prompt_id="spread", reskin_id="overtime-pool"),
            dictator_row(prompt_id="degenerate", reskin_id="bonus-pool"),
        ]
        records = sweep(
            rows,
            {
                "spread": [
                    "<think>x</think><keep>10</keep>",
                    "<think>x</think><keep>5</keep>",
                    "<think>x</think><keep>7</keep>",
                    "<think>x</think><keep>2</keep>",
                ],
                "degenerate": ["<think>x</think><keep>10</keep>"],
            },
        )

        assert records[0].kept_fractions == pytest.approx([1.0, 0.5, 0.7, 0.2])
        assert records[1].kept_fractions == pytest.approx([1.0, 1.0, 1.0, 1.0])
        assert records[1].score_std == 0.0
        # Cooperation is undefined on a unilateral split, so no rate is invented for it.
        assert records[0].coop_fraction is None

        kept = {row["prompt_id"] for row in select_mixed_prompts(records)}
        assert kept == {"spread"}
        assert reasons(records)["degenerate"] == DropReason.SCORE_SPREAD_BELOW_MIN

    def test_the_grading_ladder_keys_selection_on_the_cooperation_rate(self):
        """The wave-3 recipient-moved gradings answer with one binary action, so a sweep under
        either must select on mixedness of the cooperation rate, not on a continuous-score
        spread -- omitting them here would silently route them to the wrong selection rule."""
        assert "joint-welfare-group-mix" in ONE_SHOT_ACTION_GRADINGS
        assert "other-payoff-group-mix" in ONE_SHOT_ACTION_GRADINGS

    def test_a_unilateral_row_is_not_treated_as_a_missing_partner(self):
        row = dictator_row(prompt_id="lonely", reskin_id="overtime-pool")
        assert not is_counterbalanced(row)
        assert is_counterbalanced(matrix_row(prompt_id="paired"))

    def test_iterated_rows_key_on_the_spread_of_their_move_sequences(self):
        def moves(*labels: str) -> str:
            tags = "".join(f"<action>{label}</action>" for label in labels)
            return f"<think>planning</think>{tags}"

        rows = [
            *counterbalanced_pair(
                "repeated-spread",
                grading=select_prompts.GRADING_ITERATED_RETURN,
                game_id="iterated-pd-tft",
                n_rounds=3,
                opponent_rule="tit-for-tat",
            ),
            *counterbalanced_pair(
                "repeated-flat",
                grading=select_prompts.GRADING_ITERATED_RETURN,
                game_id="iterated-pd-tft",
                n_rounds=3,
                opponent_rule="tit-for-tat",
            ),
        ]
        spread = [
            moves(COOP_LABEL, COOP_LABEL, COOP_LABEL),
            moves(COOP_LABEL, COOP_LABEL, DEFECT_LABEL),
            moves(COOP_LABEL, DEFECT_LABEL, DEFECT_LABEL),
            moves(DEFECT_LABEL, DEFECT_LABEL, DEFECT_LABEL),
        ]
        flat = [moves(COOP_LABEL, COOP_LABEL, COOP_LABEL)]
        script: dict[str, list[str]] = {}
        for orientation in ("coop0", "coop1"):
            script[f"repeated-spread--{orientation}"] = spread
            script[f"repeated-flat--{orientation}"] = flat
        records = sweep(rows, script)

        assert records[0].selection_scores == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0])
        assert records[0].coop_count == 6
        assert records[0].coop_total == 12
        kept = {row["prompt_id"] for row in select_mixed_prompts(records)}
        assert kept == {"repeated-spread--coop0", "repeated-spread--coop1"}
        assert reasons(records)["repeated-flat--coop0"] == DropReason.SCORE_SPREAD_BELOW_MIN

    def test_a_wrong_length_move_sequence_is_a_parse_failure(self):
        rows = [
            matrix_row(
                prompt_id="one",
                grading=select_prompts.GRADING_ITERATED_RETURN,
                n_rounds=3,
                opponent_rule="tit-for-tat",
            )
        ]
        two_moves = f"<think>x</think><action>{COOP_LABEL}</action><action>{COOP_LABEL}</action>"
        record = sweep(rows, {"one": [two_moves, two_moves]})[0]
        assert record.n_parse_failures == 4
        assert record.selection_scores == []


class TestPairIdentity:
    def test_the_two_orientations_share_a_key_and_other_frames_do_not(self):
        first, second = counterbalanced_pair("warehouse-shift-handover")
        other = matrix_row(prompt_id="other", reskin_id="rope-team-traverse")
        assert pair_identity(first) == pair_identity(second)
        assert pair_identity(first) != pair_identity(other)

    def test_a_different_payoff_variant_is_a_different_pair(self):
        first, _ = counterbalanced_pair("warehouse-shift-handover")
        louder = matrix_row(
            prompt_id="louder",
            reskin_id="warehouse-shift-handover",
            payoff_variant="temptation-10",
        )
        assert pair_identity(first) != pair_identity(louder)

    def test_a_cached_opponent_probability_does_not_split_a_pair(self):
        first, second = counterbalanced_pair("warehouse-shift-handover")
        assert pair_identity({**first, "opp_coop_prob": 0.8}) == pair_identity(second)


class TestFrozenOpponent:
    OPPONENT_MODEL = "scripted/frozen-opponent"

    def vs_fixed_rows(self) -> list[dict[str, object]]:
        return [
            matrix_row(
                prompt_id="one",
                grading=select_prompts.GRADING_VS_FIXED_MIX,
                game_id="stag-hunt-vs-frozen",
            ),
            matrix_row(
                prompt_id="two",
                grading=select_prompts.GRADING_VS_FIXED_MIX,
                game_id="stag-hunt-vs-frozen",
                reskin_id="rope-team-traverse",
            ),
        ]

    def opponent(self, rows: list[dict[str, object]], script: dict[str, list[str]]):
        return ScriptedBackend(
            {row["prompt"]: script[row["prompt_id"]] for row in rows},  # pyright: ignore[reportArgumentType]
            model_id=self.OPPONENT_MODEL,
        )

    def test_cooperation_probability_per_prompt(self):
        rows = self.vs_fixed_rows()
        script = {
            "one": [PICKS_HOLD, PICKS_HOLD, PICKS_HOLD, PICKS_SLASH],
            "two": [PICKS_SLASH],
        }
        probs = sample_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=self.opponent(rows, script),
            prefilled_think=False,
            min_parseable_fraction=DEFAULT_MIN_PARSEABLE_FRACTION,
        )
        assert probs == pytest.approx({"one": 0.75, "two": 0.0})

    def test_the_trace_records_the_opponent_and_its_own_record_kind(self):
        rows = self.vs_fixed_rows()
        script = {name: [PICKS_HOLD, PICKS_SLASH] for name in ("one", "two")}
        backend = self.opponent(rows, script)
        records = sweep_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=backend,
            prefilled_think=False,
        )

        assert {record.record_kind for record in records} == {FROZEN_OPPONENT_RECORD_KIND}
        provenance = backend_provenance(backend)
        assert provenance["model_id"] == self.OPPONENT_MODEL
        assert provenance["transport"] == "scripted"

    def test_a_prompt_with_no_parseable_answer_raises_rather_than_defaulting(self):
        rows = self.vs_fixed_rows()
        script = {
            "one": [PICKS_HOLD, PICKS_SLASH],
            "two": [NO_ANSWER],
        }
        records = sweep_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=self.opponent(rows, script),
            prefilled_think=False,
        )
        with pytest.raises(ValueError, match="no parseable answer"):
            frozen_opponent_coop_probs(
                records, min_parseable_fraction=DEFAULT_MIN_PARSEABLE_FRACTION
            )

    def test_a_mostly_unparseable_prompt_is_refused_instead_of_pinning_a_mix(self):
        """One surviving answer of four is a probability of exactly 1.0, cached as ground truth.

        The policy sweep drops such a prompt as too-few-parseable; the opponent sweep has to raise
        instead, because training grades every completion of the arm against this number and cannot
        re-sample -- the whole point of the cache is that no reward function calls out.
        """
        rows = self.vs_fixed_rows()
        script = {
            "one": [PICKS_HOLD, NO_ANSWER, NO_ANSWER, NO_ANSWER],
            "two": [PICKS_HOLD, PICKS_SLASH],
        }
        records = sweep_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=self.opponent(rows, script),
            prefilled_think=False,
        )
        assert records[0].coop_fraction == pytest.approx(1.0)
        assert records[0].parseable_fraction == pytest.approx(0.25)
        with pytest.raises(ValueError, match="too few"):
            frozen_opponent_coop_probs(
                records, min_parseable_fraction=DEFAULT_MIN_PARSEABLE_FRACTION
            )

    def test_a_hosted_completion_carrying_no_thinking_tags_parses_as_an_answer(self):
        """What `BedrockBackend.generate` actually returns, against the policy's own convention.

        Converse splits reasoning into `reasoningContent` and the backend returns the answer blocks
        only, so a hosted opponent's completion never carries a closing `</think>` whatever the
        served model does. Read under the policy checkpoint's `prefilled_think=True` every one of
        them is truncated thinking with no visible answer, so the whole cache is unparseable.
        """
        rows = self.vs_fixed_rows()
        script = {name: [HOSTED_PICKS_HOLD, HOSTED_PICKS_SLASH] for name in ("one", "two")}
        as_hosted = sweep_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=self.opponent(rows, script),
            prefilled_think=select_prompts.FROZEN_OPPONENT_PREFILLED_THINK,
        )
        assert [record.coop_fraction for record in as_hosted] == [0.5, 0.5]

        as_policy = sweep_frozen_opponent(
            rows,
            model_id=self.OPPONENT_MODEL,
            samples=4,
            backend=self.opponent(rows, script),
            prefilled_think=True,
        )
        assert all(record.n_parse_failures == record.n_samples for record in as_policy)
        assert all(record.n_truncated_thinking == record.n_samples for record in as_policy)

    def test_rows_that_would_never_read_the_cache_are_refused(self):
        rows = [matrix_row(prompt_id="one", grading=select_prompts.GRADING_GROUP_MIX)]
        with pytest.raises(ValueError, match="vs-fixed-mix"):
            sweep_frozen_opponent(
                rows,
                model_id=self.OPPONENT_MODEL,
                samples=4,
                backend=self.opponent(rows, {"one": [PICKS_HOLD]}),
                prefilled_think=False,
            )

    def test_a_backend_serving_another_model_is_refused(self):
        rows = self.vs_fixed_rows()
        script = {name: [PICKS_HOLD] for name in ("one", "two")}
        with pytest.raises(ValueError, match="provenance"):
            sweep_frozen_opponent(
                rows,
                model_id="some/other-model",
                samples=4,
                backend=self.opponent(rows, script),
                prefilled_think=False,
            )

    def test_filling_the_cache_refuses_to_leave_a_row_on_the_sentinel(self):
        rows = self.vs_fixed_rows()
        filled = fill_opponent_probs(rows, {"one": 0.75, "two": 0.25})
        assert [row["opp_coop_prob"] for row in filled] == [0.75, 0.25]
        with pytest.raises(KeyError, match="no frozen-opponent probability"):
            fill_opponent_probs(rows, {"one": 0.75})


class TestChunkSize:
    def test_an_explicit_request_is_capped_by_the_work_available(self):
        assert sweep_chunk_size(max_new_tokens=512, n_sequences=6, requested=32, model_id=None) == 6
        assert sweep_chunk_size(max_new_tokens=512, n_sequences=64, requested=8, model_id=None) == 8

    def test_a_derived_chunk_is_at_least_one_sequence(self):
        assert sweep_chunk_size(max_new_tokens=1024, n_sequences=64, model_id=None) >= 1

    def test_nonsense_sizes_are_refused(self):
        with pytest.raises(ValueError, match="nothing to decode"):
            sweep_chunk_size(max_new_tokens=512, n_sequences=0, model_id=None)
        with pytest.raises(ValueError, match="chunk size must be positive"):
            sweep_chunk_size(max_new_tokens=512, n_sequences=8, requested=0, model_id=None)


# Read off `Qwen/Qwen3.5-2B`'s own config on 2026-08-18: 6 of its 24 layers cache KV (2 key-value
# heads, 128-dim heads, bf16) and 18 are Gated DeltaNet, whose recurrent state is fixed in context
# length. Hardcoded rather than downloaded so these tests stay offline; `games/tests/test_games_
# sizing.py` is where the reading of a config is checked.
QWEN_2B_SEQUENCE_COST = SequenceCost(
    kv_bytes_per_token=12288,
    recurrent_bytes_per_sequence=18874368,
    prefill_upcast_bytes_per_token=24576,
    n_kv_caching_layers=6,
    n_linear_attention_layers=18,
)
# Free VRAM the rented 96 GiB card reported once its 2B was resident, and what this box's L4 has
# left in the same position. Both after the weights, since the backend is built before any sweep.
RENTED_96_GIB_FREE = 90.8
LOCAL_L4_FREE = 18.3


class TestDecodeFootprint:
    def test_a_screened_model_is_budgeted_at_its_measured_tail(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        )
        assert footprint.sizing_tokens == 19936
        assert "observed maximum 19936" in footprint.basis
        # 0.68 GiB rather than the 8.60 the flat per-kilotoken allowance charged for the same
        # sequence: 20,960 tokens of a six-layer KV cache scaled by the measured KV factor, over a
        # fixed recurrent state, then scaled again for the allocator reserve that `mem_get_info`
        # reports and `max_memory_allocated` cannot see.
        assert footprint.gib_per_sequence == pytest.approx(0.6806, abs=0.001)
        assert "allocator reserve" in footprint.basis

    def test_an_unscreened_model_is_budgeted_at_the_full_ceiling(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="some/unscreened-2b", cost=QWEN_2B_SEQUENCE_COST
        )
        assert footprint.sizing_tokens == 24576
        assert "no termination screen" in footprint.basis

    def test_no_checkpoint_at_all_falls_back_to_the_flat_allowance(self):
        footprint = decode_footprint(max_new_tokens=24576, model_id=None)
        assert footprint.sizing_tokens == 24576
        assert footprint.gib_per_sequence == pytest.approx(8.6, abs=0.01)
        assert "crude flat allowance" in footprint.basis


class TestDeriveChunkSize:
    def test_a_big_card_decodes_far_wider_than_the_flat_allowance_allowed(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        )
        wide = derive_chunk_size(free_gib=RENTED_96_GIB_FREE, n_sequences=512, footprint=footprint)
        crude = derive_chunk_size(
            free_gib=RENTED_96_GIB_FREE,
            n_sequences=512,
            footprint=decode_footprint(max_new_tokens=24576, model_id=None),
        )
        # The measured contrast this change exists for: 8 sequences of a 2B on a card with 90.8 GiB
        # free, against a footprint that says the card holds a hundred. What it must NOT say is 172,
        # which is what the old uncalibrated arithmetic said and what thrashed for 93 minutes; the
        # throughput knee then narrows this further, in `sweep_chunk_size` rather than here.
        assert crude == 8
        assert 96 <= wide <= 128

    def test_this_box_stays_sane_and_still_beats_the_flat_allowance(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        )
        local = derive_chunk_size(free_gib=LOCAL_L4_FREE, n_sequences=512, footprint=footprint)
        crude = derive_chunk_size(
            free_gib=LOCAL_L4_FREE,
            n_sequences=512,
            footprint=decode_footprint(max_new_tokens=24576, model_id=None),
        )
        # One sequence at a time is what the L4 was doing, which is why a local sweep decoded at
        # roughly a thirtieth of the card's rate.
        assert crude == 1
        assert 8 <= local <= 64

    def test_the_work_available_still_caps_the_chunk(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        )
        assert (
            derive_chunk_size(free_gib=RENTED_96_GIB_FREE, n_sequences=12, footprint=footprint)
            == 12
        )

    def test_a_card_too_small_for_one_sequence_still_returns_one(self):
        footprint = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        )
        assert derive_chunk_size(free_gib=0.1, n_sequences=64, footprint=footprint) == 1

    def test_a_free_sequence_is_refused_rather_than_dividing_by_zero(self):
        with pytest.raises(ValueError, match="cannot cost nothing"):
            derive_chunk_size(
                free_gib=24.0,
                n_sequences=8,
                footprint=DecodeFootprint(gib_per_sequence=0.0, sizing_tokens=1, basis="test"),
            )


# Peak allocated VRAM per width, measured 2026-08-18 on the rented RTX PRO 6000 Blackwell with
# Qwen3.5-2B in bf16 on real twin-pd sheets, greedy, min_new_tokens == max_new_tokens so every row
# ran to the cap. `torch.cuda.max_memory_allocated()`, so each figure includes the resident weights.
# Raw records: s3://<bucket>/games_rl/step4-20260818/knee-{short-sequence,8192-token}.log.
MEASURED_PEAK_GIB_BY_WIDTH: dict[int, dict[int, float]] = {
    256: {8: 3.91, 16: 4.28, 32: 5.02, 64: 6.51, 128: 9.48},
    1024: {8: 3.91, 16: 4.28, 32: 5.02, 64: 6.51, 128: 9.48},
    8192: {32: 9.39, 64: 15.25, 128: 26.95},
}
MEASURED_PROMPT_TOKENS = 323
MEASURED_RESIDENT_WEIGHT_GIB = 3.51


class TestCalibratedPeakAllocated:
    """The footprint arithmetic against the peaks it was calibrated on.

    This is the test the previous version of the constant did not have, and its absence is why a
    single flat 1.5x factor -- fitted at 256 and 1,024 new tokens, where prefill sets the peak --
    was allowed to be read as a prediction at 19,936, where the KV cache sets it instead.
    """

    @staticmethod
    def measured_slope(new_tokens: int) -> float:
        """Return the per-sequence cost as the slope of peak over width, weights excluded."""
        widths = sorted(MEASURED_PEAK_GIB_BY_WIDTH[new_tokens])
        peaks = MEASURED_PEAK_GIB_BY_WIDTH[new_tokens]
        return (peaks[widths[-1]] - peaks[widths[0]]) / (widths[-1] - widths[0])

    def test_peak_allocated_is_linear_in_width_with_the_weights_as_its_intercept(self):
        """The premise the whole per-sequence model rests on, asserted rather than assumed.

        If peak were super-linear in width, no per-sequence figure could exist and the sizing would
        need a different shape entirely. Every budget's intercept landing on the resident weights is
        what says a sequence has a constant marginal cost.
        """
        for new_tokens, peaks in MEASURED_PEAK_GIB_BY_WIDTH.items():
            slope = self.measured_slope(new_tokens)
            for width, peak in peaks.items():
                assert peak - width * slope == pytest.approx(
                    MEASURED_RESIDENT_WEIGHT_GIB, abs=0.05
                ), f"{new_tokens=} {width=} is not on the line the other widths sit on"

    def test_the_prediction_reproduces_every_measured_slope(self):
        for new_tokens in MEASURED_PEAK_GIB_BY_WIDTH:
            predicted = calibrated_peak_allocated_gib(
                QWEN_2B_SEQUENCE_COST,
                prompt_tokens=MEASURED_PROMPT_TOKENS,
                completion_tokens=new_tokens,
            )
            assert predicted == pytest.approx(self.measured_slope(new_tokens), rel=0.05)

    def test_the_prediction_reproduces_every_measured_peak(self):
        """The 9.39 / 15.25 / 26.95 GiB at widths 32 / 64 / 128 the calibration exists to match."""
        for new_tokens, peaks in MEASURED_PEAK_GIB_BY_WIDTH.items():
            per_sequence = calibrated_peak_allocated_gib(
                QWEN_2B_SEQUENCE_COST,
                prompt_tokens=MEASURED_PROMPT_TOKENS,
                completion_tokens=new_tokens,
            )
            for width, peak in peaks.items():
                predicted = MEASURED_RESIDENT_WEIGHT_GIB + width * per_sequence
                assert predicted == pytest.approx(peak, rel=0.03), f"{new_tokens=} {width=}"

    def test_a_short_budget_costs_no_less_than_prefill_already_costs(self):
        """256 and 1,024 new tokens measured the identical peak, so the model must flatten too.

        A model linear all the way down reads ~25% light at 256 tokens, since it scales the KV term
        away while the real peak is still being set by the prefill transient. Taking the larger of
        the two regimes is what keeps the short end honest.
        """
        short = calibrated_peak_allocated_gib(
            QWEN_2B_SEQUENCE_COST, prompt_tokens=MEASURED_PROMPT_TOKENS, completion_tokens=256
        )
        crossover = calibrated_peak_allocated_gib(
            QWEN_2B_SEQUENCE_COST, prompt_tokens=MEASURED_PROMPT_TOKENS, completion_tokens=1024
        )
        assert short == pytest.approx(crossover, rel=0.01)

    def test_a_longer_budget_never_costs_less_than_a_shorter_one(self):
        costs = [
            calibrated_peak_allocated_gib(
                QWEN_2B_SEQUENCE_COST,
                prompt_tokens=MEASURED_PROMPT_TOKENS,
                completion_tokens=tokens,
            )
            for tokens in (1, 256, 1024, 2048, 8192, 19936, 24576, 32768)
        ]
        assert costs == sorted(costs)

    def test_the_old_flat_factor_is_what_the_correction_moved_away_from(self):
        """The regression this exists to prevent, stated as the number that caused it.

        `gib_per_episode * 1.5` was the whole footprint model, and at the 2B's 19,936-token tail it
        read 0.421 GiB/sequence, which put 172 sequences on the rented card. The calibrated model
        must read materially higher at that same point or nothing has been fixed.
        """
        old = (
            QWEN_2B_SEQUENCE_COST.gib_per_episode(prompt_tokens=1024, completion_tokens=19936) * 1.5
        )
        assert old == pytest.approx(0.4213, abs=0.001)
        new = decode_footprint(
            max_new_tokens=24576, model_id="Qwen/Qwen3.5-2B", cost=QWEN_2B_SEQUENCE_COST
        ).gib_per_sequence
        assert new > old * 1.5


class TestKneeWidthCap:
    def test_the_cap_is_the_measured_knee_at_the_budget_it_was_measured_at(self):
        assert knee_width_cap(max_new_tokens=8192) == 64

    def test_the_cap_scales_with_the_token_budget_rather_than_transferring_as_a_width(self):
        """The error the cap exists to prevent: 64 was measured at 8,192 tokens and does not carry.

        Three times the budget puts three times the tokens in flight at the same width, so the cap
        has to fall in proportion. A bare 64 at 24,576 tokens is 1.5M in-flight tokens, which is the
        regime that returned nothing in 44 minutes.
        """
        assert knee_width_cap(max_new_tokens=24576) == 21
        assert knee_width_cap(max_new_tokens=32768) == 16
        for budget in (1024, 8192, 16384, 24576, 32768):
            cap = knee_width_cap(max_new_tokens=budget)
            assert cap * budget <= THROUGHPUT_KNEE_SEQUENCE_TOKENS

    def test_a_budget_past_the_whole_knee_still_allows_one_sequence(self):
        assert knee_width_cap(max_new_tokens=THROUGHPUT_KNEE_SEQUENCE_TOKENS * 4) == 1

    def test_a_budget_of_nothing_is_refused(self):
        with pytest.raises(ValueError, match="decodes nothing"):
            knee_width_cap(max_new_tokens=0)


class TestChunkSizeAgainstTheKnee:
    """`sweep_chunk_size`'s two bounds, and which one is allowed to bind silently.

    CUDA is faked rather than required: the arithmetic is what is under test, and the card these
    numbers came from is rented and gone. `mem_get_info` returns a (free, total) byte pair.
    """

    RENTED_96_GIB_BYTES: ClassVar[tuple[int, int]] = (
        int(90.8 * 1024**3),
        int(95.0 * 1024**3),
    )

    @pytest.fixture
    def pretend_rented_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: self.RENTED_96_GIB_BYTES)
        monkeypatch.setattr(
            torch.cuda, "get_device_name", lambda: "NVIDIA RTX PRO 6000 Blackwell (faked)"
        )

    def test_the_knee_clamps_a_derived_width_the_card_had_the_vram_for(
        self, pretend_rented_card: None, caplog: pytest.LogCaptureFixture
    ):
        del pretend_rented_card
        # INFO, not WARNING: knee_cap is in the INFO sizing line, the clamp prose in a separate
        # WARNING. Capturing at WARNING sees only the second, which is why this asserted a
        # string it could never observe.
        with caplog.at_level(logging.INFO, logger="games.chunked_decode"):
            chunk = sweep_chunk_size(
                max_new_tokens=24576, n_sequences=1600, model_id="Qwen/Qwen3.5-2B"
            )
        assert chunk == knee_width_cap(max_new_tokens=24576)
        assert "throughput knee, not VRAM" in caplog.text
        assert "knee_cap=21" in caplog.text

    def test_the_derived_width_is_nowhere_near_the_172_that_thrashed(
        self, pretend_rented_card: None
    ):
        """The failure this whole change is about, asserted against the card it happened on."""
        del pretend_rented_card
        assert (
            sweep_chunk_size(max_new_tokens=24576, n_sequences=1600, model_id="Qwen/Qwen3.5-2B")
            < 172
        )

    def test_memory_binds_instead_of_the_knee_at_a_short_budget(
        self, pretend_rented_card: None, caplog: pytest.LogCaptureFixture
    ):
        """Both bounds have to be live, or one of them is decoration.

        At 1,024 new tokens the knee allows 512 sequences and 90.8 GiB of a 2B holds fewer, so the
        memory arithmetic is what binds and the knee warning must stay silent.
        """
        del pretend_rented_card
        with caplog.at_level(logging.WARNING, logger="games.chunked_decode"):
            chunk = sweep_chunk_size(
                max_new_tokens=1024, n_sequences=4096, model_id="Qwen/Qwen3.5-2B"
            )
        assert chunk < knee_width_cap(max_new_tokens=1024)
        assert "throughput knee, not VRAM" not in caplog.text

    def test_an_explicit_width_past_the_knee_is_refused_with_the_arithmetic_in_the_message(
        self, pretend_rented_card: None
    ):
        del pretend_rented_card
        with pytest.raises(RuntimeError, match="past the measured throughput knee of 21"):
            sweep_chunk_size(
                max_new_tokens=24576,
                n_sequences=1600,
                requested=172,
                model_id="Qwen/Qwen3.5-2B",
            )

    def test_an_explicit_width_at_the_knee_is_allowed(self, pretend_rented_card: None):
        del pretend_rented_card
        assert (
            sweep_chunk_size(
                max_new_tokens=24576, n_sequences=1600, requested=21, model_id="Qwen/Qwen3.5-2B"
            )
            == 21
        )

    def test_a_hosted_transport_is_not_governed_by_a_local_gpu_measurement(
        self, pretend_rented_card: None
    ):
        """A chunk against a hosted model is request batching; the knee says nothing about it."""
        del pretend_rented_card
        assert (
            sweep_chunk_size(max_new_tokens=24576, n_sequences=1600, requested=172, model_id=None)
            == 172
        )

    def test_a_cpu_sweep_is_not_governed_by_it_either(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert (
            sweep_chunk_size(
                max_new_tokens=24576, n_sequences=1600, requested=172, model_id="Qwen/Qwen3.5-2B"
            )
            == 172
        )


class TestChunkSizeAgainstASelfSchedulingBackend:
    """A resident vLLM engine leaves almost no free VRAM, and that must not narrow the batch.

    The regression this pins happened on two rented cards on 2026-08-19: both validator sweeps ran
    the free-VRAM arithmetic while their own vLLM engine held ~90 of 96 GiB, derived a width of 1,
    and decoded a 256-sequence sweep one sequence at a time at 30-42 seconds each -- a ~15-minute
    sweep stretched to 2-3 hours, with nothing failing and nothing warning. The engine's own
    reservation is precisely what makes `mem_get_info` meaningless here.
    """

    ENGINE_RESIDENT_96_GIB_BYTES: ClassVar[tuple[int, int]] = (
        int(0.9 * 1024**3),
        int(95.0 * 1024**3),
    )

    @pytest.fixture
    def pretend_engine_holds_the_card(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: self.ENGINE_RESIDENT_96_GIB_BYTES)
        monkeypatch.setattr(
            torch.cuda, "get_device_name", lambda: "NVIDIA RTX PRO 6000 Blackwell (faked)"
        )

    def test_the_whole_sweep_is_handed_over_despite_a_card_with_no_free_vram(
        self, pretend_engine_holds_the_card: None
    ):
        del pretend_engine_holds_the_card
        assert (
            sweep_chunk_size(
                max_new_tokens=24576, n_sequences=256, model_id=None, schedules_own_batch=True
            )
            == 256
        )

    def test_without_the_flag_the_same_card_collapses_to_one(
        self, pretend_engine_holds_the_card: None
    ):
        """The bug itself, kept as a test so the fix cannot be quietly reverted.

        Asserting the broken number deliberately: it is what the arithmetic says on a card whose
        VRAM is already spoken for, and the point is that only `schedules_own_batch` rescues it.
        """
        del pretend_engine_holds_the_card
        assert sweep_chunk_size(max_new_tokens=24576, n_sequences=256, model_id=None) == 1

    def test_an_explicit_width_still_wins_over_the_handover(
        self, pretend_engine_holds_the_card: None
    ):
        """An operator naming a number is asserting something about the hardware; honour it."""
        del pretend_engine_holds_the_card
        assert (
            sweep_chunk_size(
                max_new_tokens=24576,
                n_sequences=256,
                requested=64,
                model_id=None,
                schedules_own_batch=True,
            )
            == 64
        )

    def test_the_hf_path_is_untouched_by_the_new_branch(self, pretend_engine_holds_the_card: None):
        """The VRAM arithmetic still governs the backend that really does hold every sequence."""
        del pretend_engine_holds_the_card
        assert (
            sweep_chunk_size(max_new_tokens=24576, n_sequences=256, model_id="Qwen/Qwen3.5-2B")
            < 256
        )

    def test_a_sweep_that_persists_per_chunk_caps_the_handover(
        self, pretend_engine_holds_the_card: None
    ):
        """A sweep writing records per chunk needs a bounded chunk, or nothing reaches disk until the end.

        The whole handover and per-record resume are incompatible: one call over all 5,056 sequences
        of the wave-4b pool returns once, at the end, so a box that dies at hour three of six has
        written nothing. The cap is the interval at which finished prompts land, and only a caller
        that persists them asks for it.
        """
        del pretend_engine_holds_the_card
        assert (
            sweep_chunk_size(
                max_new_tokens=32768,
                n_sequences=5056,
                model_id=None,
                schedules_own_batch=True,
                self_scheduling_cap=self_scheduling_chunk_cap(samples_per_prompt=8),
            )
            == 512
        )

    def test_the_cap_ends_on_a_prompt_boundary_at_every_sample_count(self):
        """A chunk that ends mid-prompt would offer half a prompt's samples as a whole record."""
        for samples in (2, 3, 4, 8, 16, 100):
            cap = self_scheduling_chunk_cap(samples_per_prompt=samples)
            assert cap % samples == 0
            assert cap <= SELF_SCHEDULING_CHUNK_SEQUENCES
            assert cap > SELF_SCHEDULING_CHUNK_SEQUENCES - samples
        # More samples per prompt than the cap allows still decodes one whole prompt at a time:
        # there is nothing to write until a prompt is finished.
        assert self_scheduling_chunk_cap(samples_per_prompt=900) == 900
        with pytest.raises(ValueError, match="cannot be sampled"):
            self_scheduling_chunk_cap(samples_per_prompt=0)

    def test_the_cap_never_asks_for_more_work_than_exists(
        self, pretend_engine_holds_the_card: None
    ):
        del pretend_engine_holds_the_card
        assert (
            sweep_chunk_size(
                max_new_tokens=32768,
                n_sequences=64,
                model_id=None,
                schedules_own_batch=True,
                self_scheduling_cap=512,
            )
            == 64
        )

    def test_only_vllm_is_treated_as_self_scheduling(self):
        """The predicate, not just the branch: an HF backend must not claim its own scheduler."""
        assert backend_schedules_own_batch(
            cast("Backend", SimpleNamespace(transport=VLLMBackend.transport))
        )
        assert not backend_schedules_own_batch(
            cast("Backend", SimpleNamespace(transport=HFBackend.transport))
        )
        assert not backend_schedules_own_batch(
            cast("Backend", SimpleNamespace(transport="bedrock"))
        )


class TestOomHalving:
    """The retry path, exercised by a backend that refuses anything wider than it can hold.

    `torch.OutOfMemoryError` is the class the CUDA allocator itself raises (and is
    `torch.cuda.OutOfMemoryError`, the same object), so raising it here drives exactly the branch a
    full card drives -- no GPU required, and nothing else is caught.
    """

    @staticmethod
    def script(n_prompts: int) -> dict[str, list[str]]:
        return {f"sheet-{index}": [f"answer-{index}"] for index in range(n_prompts)}

    def test_a_chunk_that_ooms_is_retried_at_half_width_and_loses_nothing(self):
        script = self.script(10)
        backend = OomAboveWidthBackend(script, fits=2)
        completions = decode_in_chunks(backend, list(script), chunk_size=8)

        assert completions == [f"answer-{index}" for index in range(10)]
        # 8 and 4 are refused, then five chunks of 2 carry the whole list: the narrowed width is
        # kept rather than re-discovered, and the cursor never moved during the failures.
        assert backend.attempted_widths == [8, 4, 2, 2, 2, 2, 2]

    def test_every_completion_stays_with_its_own_prompt_across_a_halving(self):
        rows = [
            matrix_row(prompt_id="one"),
            matrix_row(prompt_id="two", reskin_id="rope-team"),
            matrix_row(prompt_id="three", reskin_id="night-freight"),
        ]
        script = {
            "sheet for one": [PICKS_HOLD],
            "sheet for two": [PICKS_SLASH],
            "sheet for three": [PICKS_HOLD],
        }
        backend = OomAboveWidthBackend(script, fits=3)
        records = sweep_prompts(
            backend, rows, samples_per_prompt=4, prefilled_think=False, chunk_size=12
        )

        assert [record.coop_fraction for record in records] == [1.0, 0.0, 1.0]
        assert 12 in backend.attempted_widths
        assert min(backend.attempted_widths) <= 3

    def test_the_halving_says_loudly_which_width_it_dropped_to(
        self, caplog: pytest.LogCaptureFixture
    ):
        script = self.script(4)
        backend = OomAboveWidthBackend(script, fits=1)
        with caplog.at_level(logging.ERROR, logger="games.chunked_decode"):
            decode_in_chunks(backend, list(script), chunk_size=4)

        assert "old_width=4 new_width=2" in caplog.text
        assert "old_width=2 new_width=1" in caplog.text
        assert any(record.levelno >= logging.ERROR for record in caplog.records)

    def test_the_retry_runs_only_after_the_failed_attempt_has_been_released(self):
        """The bug a capped-allocator sabotage found on 2026-08-18, kept fixed.

        While an exception is being handled, its traceback holds every frame of the failed forward
        pass alive, and with them the activations that pass allocated -- so a retry issued from
        inside the handler runs against memory the first attempt has not given back, and halves all
        the way down to one sequence before raising. `sys.exc_info()` is how that is visible without
        a GPU: it reports the exception being handled anywhere up the stack, so a retry from inside
        the handler shows up here as a `generate` call made while one is still live.
        """
        exception_live_during_call: list[bool] = []

        class RecordsWhetherAnExceptionIsLive(OomAboveWidthBackend):
            def generate(self, prompts: list[str]) -> list[str]:
                exception_live_during_call.append(sys.exc_info()[0] is not None)
                return super().generate(prompts)

        script = self.script(4)
        backend = RecordsWhetherAnExceptionIsLive(script, fits=2)
        completions = decode_in_chunks(backend, list(script), chunk_size=4)

        assert completions == [f"answer-{index}" for index in range(4)]
        assert backend.attempted_widths == [4, 2, 2]
        assert exception_live_during_call == [False, False, False]

    def test_an_oom_at_a_single_sequence_is_raised_rather_than_swallowed(self):
        script = self.script(4)
        backend = OomAboveWidthBackend(script, fits=0)
        with pytest.raises(torch.OutOfMemoryError):
            decode_in_chunks(backend, list(script), chunk_size=4)
        assert backend.attempted_widths == [4, 2, 1]


class TestThrashNarrowing:
    """The half of the recovery nothing raises for, driven through the real `decode_in_chunks`.

    `ThrashVerdict`'s arithmetic is pinned against the measured production points further down this
    file, but none of those tests reach the code that acts on a verdict: they compute the verdict
    themselves. These drive it through the production call path instead, which is where a detector
    that narrows the wrong way, or not at all, becomes visible.

    Offline like the rest: the probe supplies both numbers the detector reads, and
    `torch.cuda.empty_cache()` is a no-op while no CUDA context has been initialised -- the same
    property `TestOomHalving` already relies on.
    """

    @staticmethod
    def script(n_prompts: int) -> dict[str, list[str]]:
        return {f"sheet-{index}": [f"answer-{index}"] for index in range(n_prompts)}

    def test_a_thrashing_chunk_keeps_its_completions_and_narrows_what_follows(self):
        script = self.script(24)
        backend = OomAboveWidthBackend(script, fits=WIDER_THAN_ANY_THRASH_CHUNK)
        completions = decode_in_chunks(
            backend, list(script), chunk_size=8, probe=thrash_probe(backend, sustained=False)
        )

        assert completions == [f"answer-{index}" for index in range(24)]
        # The thrashing chunk of 8 did produce its completions, so they are kept and the cursor
        # advances past them -- re-decoding work already correctly in hand would pay the thrash a
        # second time. Only what follows runs narrowed, and stays narrowed.
        assert backend.attempted_widths == [8, 4, 4, 4, 4]

    def test_the_narrowing_says_loudly_which_width_it_dropped_to(
        self, caplog: pytest.LogCaptureFixture
    ):
        """A silent recovery hides the one thing it proves: the derived chunk size was too large."""
        script = self.script(24)
        backend = OomAboveWidthBackend(script, fits=WIDER_THAN_ANY_THRASH_CHUNK)
        with caplog.at_level(logging.ERROR, logger="games.chunked_decode"):
            decode_in_chunks(
                backend, list(script), chunk_size=8, probe=thrash_probe(backend, sustained=False)
            )

        assert "ALLOCATOR THRASH" in caplog.text
        assert "Narrowing to 4" in caplog.text
        assert any(record.levelno >= logging.ERROR for record in caplog.records)

    def test_pressure_that_never_lets_up_narrows_to_one_sequence_and_still_finishes(self):
        """Width 1 has nothing left to halve, so it has to decode on rather than halve to zero.

        Halving past one hands the backend an empty chunk, and the width reaches `ThrashVerdict`'s
        per-sequence divisor, so the failure is a raise rather than a hang -- which makes "the sweep
        completed, in order, with every prompt" the assertion worth making.
        """
        script = self.script(12)
        backend = OomAboveWidthBackend(script, fits=WIDER_THAN_ANY_THRASH_CHUNK)
        completions = decode_in_chunks(
            backend, list(script), chunk_size=4, probe=thrash_probe(backend, sustained=True)
        )

        assert completions == [f"answer-{index}" for index in range(12)]
        assert backend.attempted_widths == [4, 2, 1, 1, 1, 1, 1, 1]


class TestLocalDecodeModelId:
    def test_only_the_in_process_huggingface_backend_is_sized_from_a_checkpoint(self):
        scripted = ScriptedBackend({"sheet": [PICKS_HOLD]}, model_id="Qwen/Qwen3.5-2B")
        assert local_decode_model_id(scripted) is None

    def test_the_huggingface_transport_hands_over_its_model_id(self):
        class LooksLikeHF(ScriptedBackend):
            transport = HFBackend.transport

        backend = LooksLikeHF({"sheet": [PICKS_HOLD]}, model_id="Qwen/Qwen3.5-2B")
        assert local_decode_model_id(backend) == "Qwen/Qwen3.5-2B"


class TestParseArgs:
    REQUIRED: ClassVar[list[str]] = [
        "--game",
        "twin-pd-group",
        "--grading",
        "group-mix",
        "--model",
        "Qwen/Qwen3.5-4B",
    ]

    def test_defaults(self):
        args = select_prompts._parse_args(self.REQUIRED)
        assert args.game == "twin-pd-group"
        assert args.grading == "group-mix"
        assert args.model == "Qwen/Qwen3.5-4B"
        assert args.backend == "hf"
        assert args.samples_per_prompt == DEFAULT_SAMPLES_PER_PROMPT
        assert args.split == "train"
        # Canonical by default, so an unflagged sweep renders the prompts every measured corpus used.
        assert args.label_print_order == LABEL_PRINT_ORDER_CANONICAL
        assert args.out_dir == DEFAULT_OUT_DIR
        assert args.frozen_opponent_model is None
        assert args.chunk_size is None
        assert args.prefilled_think is None
        assert args.seed == 0
        assert args.min_coop == DEFAULT_MIN_COOP
        assert args.max_coop == DEFAULT_MAX_COOP
        assert args.min_split_std == DEFAULT_MIN_SPLIT_STD
        assert args.min_parseable_fraction == DEFAULT_MIN_PARSEABLE_FRACTION
        # Unset, so the training sampler's own values apply rather than SamplingConfig's defaults.
        assert args.temperature is None
        assert args.max_new_tokens is None

    def test_explicit_values(self):
        args = select_prompts._parse_args(
            [
                *self.REQUIRED,
                "--backend",
                "mock",
                "--samples-per-prompt",
                "16",
                "--temperature",
                "1.0",
                "--max-new-tokens",
                "768",
                "--out-dir",
                "artifacts/games/custom-select",
                "--split",
                "eval",
                "--frozen-opponent-model",
                "openai.gpt-oss-20b",
                "--no-prefilled-think",
                "--min-coop",
                "0.25",
                "--seed",
                "7",
            ]
        )
        assert args.backend == "mock"
        assert args.samples_per_prompt == 16
        assert args.temperature == pytest.approx(1.0)
        assert args.max_new_tokens == 768
        assert args.out_dir == Path("artifacts/games/custom-select")
        assert args.split == "eval"
        assert args.frozen_opponent_model == "openai.gpt-oss-20b"
        assert args.prefilled_think is False
        assert args.min_coop == pytest.approx(0.25)
        assert args.seed == 7

    def test_prefilled_think_flag_is_on_when_given(self):
        args = select_prompts._parse_args([*self.REQUIRED, "--prefilled-think"])
        assert args.prefilled_think is True

    def test_an_unknown_grading_is_refused(self):
        with pytest.raises(SystemExit):
            select_prompts._parse_args(
                ["--game", "twin-pd-group", "--grading", "vibes", "--model", "m"]
            )

    def test_a_local_sweep_turns_thinking_on_by_default(self):
        """backend_cli defaults thinking off; a sweep whose trace is the "before" eval cannot."""
        args = select_prompts._parse_args([*self.REQUIRED, "--backend", "hf"])
        assert args.thinking is None
        select_prompts._require_thinking(args)
        assert args.thinking is True

    def test_turning_thinking_off_is_honoured_but_said_out_loud(
        self, caplog: pytest.LogCaptureFixture
    ):
        args = select_prompts._parse_args([*self.REQUIRED, "--backend", "hf", "--no-thinking"])
        with caplog.at_level(logging.WARNING):
            select_prompts._require_thinking(args)
        assert args.thinking is False
        # The warning has to name the remedy, not just complain: games.train needs the same flag,
        # and without it the sweep describes a policy the trainer never runs.
        assert "--no-thinking" in caplog.text
        assert "never runs" in caplog.text

    def test_a_hosted_backend_keeps_thinking_unset_since_it_templates_its_own_way(self):
        args = select_prompts._parse_args([*self.REQUIRED, "--backend", "bedrock"])
        select_prompts._require_thinking(args)
        assert args.thinking is None

    def test_the_model_name_heuristic_matches_the_families_that_prefill(self):
        assert select_prompts.prefilled_think_from_model_name("Qwen/Qwen3.5-4B")
        assert select_prompts.prefilled_think_from_model_name("Qwen/Qwen3.8-27B")
        assert not select_prompts.prefilled_think_from_model_name("Qwen/Qwen3-0.6B")


class TestArtifacts:
    def records_and_verdicts(self):
        rows = [*counterbalanced_pair("mixed-frame"), *counterbalanced_pair("pure-frame")]
        script: dict[str, list[str]] = {}
        for orientation in ("coop0", "coop1"):
            script[f"mixed-frame--{orientation}"] = [PICKS_HOLD, PICKS_SLASH]
            script[f"pure-frame--{orientation}"] = [PICKS_SLASH]
        records = sweep(rows, script)
        return records, judge_prompts(records)

    def test_the_trace_round_trips_with_the_meta_record_first(self, tmp_path: Path):
        records, _ = self.records_and_verdicts()
        args = select_prompts._parse_args(
            [
                "--game",
                "twin-pd-group",
                "--grading",
                "group-mix",
                "--model",
                "Qwen/Qwen3.5-4B",
                "--backend",
                "mock",
            ]
        )
        backend = ScriptedBackend({"unused": ["x"]})
        path = tmp_path / "sweep.jsonl"
        write_sweep_trace(
            path,
            meta=sweep_meta(
                backend=backend,
                args=args,
                prompt_ids=[record.prompt_id for record in records],
                samples_per_prompt=4,
                prefilled_think=False,
                rows_sha256=None,
            ),
            records=records,
        )

        loaded = read_jsonl(path)
        meta = json_record(loaded, 0)
        assert meta["record_kind"] == META_RECORD_KIND
        assert meta["game"] == "twin-pd-group"
        assert meta["samples_per_prompt"] == 4
        assert meta["prefilled_think"] is False
        # `engine` is empty for a backend that takes no engine kwargs, and present rather than absent
        # so a reader of two traces can tell "no settings" from "a version that did not record them".
        assert meta["backend"] == {
            "model_id": "scripted/policy",
            "transport": "scripted",
            "sampling": None,
            "engine": {},
        }
        assert isinstance(meta["git_sha"], str)
        assert meta["git_sha"]

        assert len(loaded) == len(records) + 1
        first = json_record(loaded, 1)
        assert first["record_kind"] == SWEEP_RECORD_KIND
        assert first["prompt_id"] == records[0].prompt_id
        assert first["row"]["coop_label"] == COOP_LABEL
        assert first["coop_fraction"] == pytest.approx(0.5)
        assert len(first["samples"]) == 4
        assert first["samples"][0]["completion"] == PICKS_HOLD
        assert first["samples"][0]["action"] == "C"

    def test_the_corpus_is_headerless_so_a_dumb_reader_gets_only_rows(self, tmp_path: Path):
        records, _ = self.records_and_verdicts()
        path = tmp_path / "corpus.jsonl"
        write_corpus(path, select_mixed_prompts(records))

        loaded = read_jsonl(path)
        assert [row["prompt_id"] for row in loaded] == [
            "mixed-frame--coop0",
            "mixed-frame--coop1",
        ]
        assert all("record_kind" not in row for row in loaded)

    def test_the_summary_counts_every_prompt_by_reason(self):
        records, verdicts = self.records_and_verdicts()
        summary = cast("dict[str, Any]", selection_summary(records, verdicts))

        assert summary["n_prompts"] == 4
        assert summary["n_kept"] == 2
        assert summary["n_dropped"] == 2
        # The pure frame always writes SLASH, so it reads as unanimous defection in one
        # orientation and unanimous cooperation in the other; both are dropped either way.
        assert summary["counts_by_reason"] == {
            DropReason.KEPT_MIXED: 2,
            DropReason.COOP_FRACTION_BELOW_MIN: 1,
            DropReason.COOP_FRACTION_ABOVE_MAX: 1,
        }
        assert summary["n_samples"] == 16
        assert summary["parse_failure_rate"] == pytest.approx(0.0)
        assert len(summary["verdicts"]) == 4
        # It has to survive the trip to disk, since that is the only form anyone reads it in.
        assert json.loads(json.dumps(summary))["n_kept"] == 2

    def test_the_meta_pins_the_pool_the_sweep_drew_from(self):
        """Two corpora built from different pools are not a controlled comparison.

        Sampling draws from one global RNG stream over the flattened prompt list, and the decode
        chunk boundaries move with the sequence count, so inserting one scenario upstream re-rolls
        every later prompt's samples at the same --seed and a previously-kept prompt can cross a
        threshold for reasons unrelated to the edit. Nothing else in the artifacts can say whether
        two runs are comparable, and per-prompt seeding is not expressible through the backend
        protocol -- `generate` takes no seed.
        """
        args = select_prompts._parse_args(
            ["--game", "twin-pd-group", "--grading", "group-mix", "--model", "m"]
        )
        backend = ScriptedBackend({"unused": ["x"]})

        def meta_for(prompt_ids: list[str]) -> dict[str, Any]:
            return cast(
                "dict[str, Any]",
                sweep_meta(
                    backend=backend,
                    args=args,
                    prompt_ids=prompt_ids,
                    samples_per_prompt=4,
                    prefilled_think=False,
                    rows_sha256=None,
                ),
            )

        swept = meta_for(["one", "two"])
        assert swept["n_prompts"] == 2
        assert (
            swept["prompt_id_order_sha256"]
            != meta_for(["one", "two", "three"])["prompt_id_order_sha256"]
        )
        assert swept["prompt_id_order_sha256"] != meta_for(["two", "one"])["prompt_id_order_sha256"]
        assert swept["prompt_id_order_sha256"] == meta_for(["one", "two"])["prompt_id_order_sha256"]

    def test_the_artifact_stem_names_the_model_without_its_namespace(self):
        args = argparse.Namespace(
            game="twin-pd-group",
            model="Qwen/Qwen3.5-4B",
            label_print_order=LABEL_PRINT_ORDER_CANONICAL,
        )
        stem = select_prompts._artifact_stem("sweep", args, "20260817T010203Z")
        assert stem == "sweep-twin-pd-group-Qwen3.5-4B-20260817T010203Z"

    def test_the_artifact_stem_names_a_swapped_label_print_order(self):
        """Two print orders of one game are two prompt sets; the filenames have to say which."""
        args = argparse.Namespace(
            game="harmony", model="Qwen/Qwen3.5-2B", label_print_order=LABEL_PRINT_ORDER_SWAPPED
        )
        stem = select_prompts._artifact_stem("corpus", args, "20260819T010203Z")
        assert stem == "corpus-harmony-swapped-Qwen3.5-2B-20260819T010203Z"


class TestAgainstTheRealCorpus:
    """End to end on rows from `games.prompts`, which is the seam a hand-built row cannot test.

    Every other test here builds its own rows, so all of them would still pass if this module and
    the corpus builder disagreed about a column name, a sentinel, or which columns a label swap
    changes. These drive the real `generate_prompt_rows` output through the whole path.
    """

    def scripted_policy(self, rows: list[dict[str, object]]) -> ScriptedBackend:
        """Answer each real prompt with its own two labels, so the sweep comes back mixed."""
        return ScriptedBackend(
            {
                cast("str", row["prompt"]): [
                    f"<think>weighing</think><action>{row['label_a']}</action>",
                    f"<think>weighing</think><action>{row['label_b']}</action>",
                ]
                for row in rows
            }
        )

    def test_a_real_matrix_corpus_selects_whole_pairs(self):
        rows = generate_prompt_rows("twin-pd", "group-mix", split="train")
        records = sweep_prompts(
            self.scripted_policy(rows),
            rows,
            samples_per_prompt=4,
            prefilled_think=False,
            chunk_size=8,
        )
        selected = select_mixed_prompts(records)

        # Alternating the two labels is a 50/50 split whichever one is cooperative, so the whole
        # corpus qualifies -- which is what makes "kept in pairs" checkable by counting.
        assert len(selected) == len(rows)
        assert len(rows) % 2 == 0
        pairs = collections.Counter(pair_identity(row) for row in selected)
        assert set(pairs.values()) == {2}

    def test_deleting_one_orientation_of_a_real_frame_is_refused(self):
        """The sabotage the pair coupling exists for, run against the corpus builder's own rows.

        Every counterbalanced game renders both orientations today, so the only way to reach this is
        to break that upstream -- which is exactly the case a warning could not stop.
        """
        rows = generate_prompt_rows("twin-pd", "group-mix", split="train")
        orphaned = [row for row in rows if row["prompt_id"] != rows[1]["prompt_id"]]
        records = sweep_prompts(
            self.scripted_policy(orphaned),
            orphaned,
            samples_per_prompt=4,
            prefilled_think=False,
            chunk_size=8,
        )
        with pytest.raises(ValueError, match="no counterbalanced partner"):
            select_mixed_prompts(records)

    def test_a_real_row_reaches_the_frozen_opponent_path(self):
        rows = generate_prompt_rows("stag-hunt-vs-frozen", "vs-fixed-mix", split="train")
        probs = sample_frozen_opponent(
            rows,
            model_id="scripted/policy",
            samples=4,
            backend=self.scripted_policy(rows),
            prefilled_think=False,
            min_parseable_fraction=DEFAULT_MIN_PARSEABLE_FRACTION,
        )
        assert set(probs) == {cast("str", row["prompt_id"]) for row in rows}
        assert all(0.0 <= prob <= 1.0 for prob in probs.values())

        filled = fill_opponent_probs(rows, probs)
        assert all(row["opp_coop_prob"] != OPP_COOP_PROB_UNSET for row in filled)

    def test_a_real_unilateral_corpus_keys_on_spread_and_needs_no_partner(
        self, caplog: pytest.LogCaptureFixture
    ):
        rows = generate_prompt_rows("dictator", "keep-fraction", split="train")
        # The corpus varies the endowment across rows, so a keep only parses in its own row's range.
        endowments = {cast("str", row["prompt"]): cast("int", row["endowment"]) for row in rows}
        backend = ScriptedBackend(
            {
                prompt: [
                    f"<think>weighing</think><keep>{endowment}</keep>",
                    f"<think>weighing</think><keep>{endowment // 2}</keep>",
                ]
                for prompt, endowment in endowments.items()
            }
        )
        records = sweep_prompts(
            backend, rows, samples_per_prompt=4, prefilled_think=False, chunk_size=8
        )
        with caplog.at_level(logging.WARNING):
            selected = select_mixed_prompts(records)

        assert len(selected) == len(rows)
        assert records[0].kept_fractions == pytest.approx([1.0, 0.5, 1.0, 0.5])
        # A unilateral row has no orientation to pair with, so no partner warning belongs here.
        assert "no counterbalanced partner" not in caplog.text

    def test_a_real_repeated_corpus_parses_its_whole_move_sequence(self):
        rows = generate_prompt_rows("iterated-pd-tft", "iterated-return", split="train")
        n_rounds = cast("int", rows[0]["n_rounds"])
        backend = ScriptedBackend(
            {
                cast("str", row["prompt"]): [
                    "<think>planning</think>" + f"<action>{row['label_a']}</action>" * n_rounds,
                    "<think>planning</think>"
                    + f"<action>{row['label_b']}</action>" * (n_rounds - 1)
                    + f"<action>{row['label_a']}</action>",
                ]
                for row in rows
            }
        )
        records = sweep_prompts(
            backend, rows, samples_per_prompt=4, prefilled_think=False, chunk_size=8
        )
        assert all(record.n_parse_failures == 0 for record in records)
        assert all(len(record.selection_scores) == 4 for record in records)
        assert len(select_mixed_prompts(records)) == len(rows)

    def test_the_real_corpus_writes_all_three_artifacts(self, tmp_path: Path):
        rows = generate_prompt_rows("stag-hunt", "group-mix", split="train")
        backend = self.scripted_policy(rows)
        records = sweep_prompts(
            backend, rows, samples_per_prompt=4, prefilled_think=False, chunk_size=8
        )
        verdicts = judge_prompts(records)
        selected = [
            dict(record.row)
            for record, verdict in zip(records, verdicts, strict=True)
            if verdict.keep
        ]
        args = select_prompts._parse_args(
            ["--game", "stag-hunt", "--grading", "group-mix", "--model", "scripted/policy"]
        )

        trace = tmp_path / "sweep.jsonl"
        corpus = tmp_path / "corpus.jsonl"
        write_sweep_trace(
            trace,
            meta=sweep_meta(
                backend=backend,
                args=args,
                prompt_ids=[record.prompt_id for record in records],
                samples_per_prompt=4,
                prefilled_think=False,
                rows_sha256=None,
            ),
            records=records,
        )
        write_corpus(corpus, selected)

        reloaded = read_jsonl(corpus)
        assert reloaded == selected
        # A corpus row has to arrive at the dataset builder with exactly the columns it expects.
        assert all(frozenset(row) == frozenset(ROW_COLUMNS) for row in reloaded)
        assert len(read_jsonl(trace)) == len(records) + 1

    def test_the_cli_runs_the_whole_path_on_the_mock_backend(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The repo's smoke rule: watch `main` complete locally before trusting a real sweep.

        `--backend mock` samples nothing, so the selection this reports is meaningless and only
        the execution is the claim. It is still the only check that covers argument parsing,
        corpus generation, filename construction and all three writes together.

        The DeltaNet decode bridge is stubbed out because it cannot run twice in one interpreter and
        `test_games_deltanet_kernels.py` imports the Qwen3.5 modeling module in-process, which makes
        the real call raise "too late to bridge" whenever that module was collected first. Ordering
        is what the bridge is about, so it is checked where it can be checked honestly: in a fresh
        interpreter, by `TestEntryPointsBridgeInTime` in that same file.
        """
        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )
        exit_code = select_prompts.main(
            [
                "--game",
                "twin-pd",
                "--grading",
                "group-mix",
                "--model",
                "mock/policy",
                "--backend",
                "mock",
                "--samples-per-prompt",
                "4",
                "--no-prefilled-think",
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert exit_code == 0

        kinds = {path.name.split("-", 1)[0] for path in tmp_path.iterdir()}
        assert kinds == {"sweep", "corpus", "selection", select_prompts.PARTIAL_SUBDIR}
        trace = next(tmp_path.glob("sweep-*.jsonl"))
        assert json_record(read_jsonl(trace), 0)["record_kind"] == META_RECORD_KIND
        # The launch kit finds finished traces with `find <out-dir> -maxdepth 1 -name 'sweep-*.jsonl'`,
        # so the run's partial records have to be invisible to that glob: one trace at depth 1, and
        # the partial one level down under a stem that could not match it either.
        assert [path.name for path in tmp_path.glob("sweep-*.jsonl")] == [trace.name]
        partial = next((tmp_path / select_prompts.PARTIAL_SUBDIR).iterdir())
        assert partial.name.startswith(select_prompts.PARTIAL_STEM_PREFIX)

    def test_the_cli_carries_a_swapped_label_print_order_into_the_rendered_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """`--label-print-order swapped` has to reach the renderer, not just the parsed namespace.

        A flag that parses and then goes nowhere is the failure this experiment cannot survive: the
        sweep would run, cost its GPU time, and report a null that means "the prompts never
        changed". So this reads the print order back off the artefact the sweep writes -- the trace's
        meta line and the rendered row inside every sweep record -- rather than off `args`.
        """
        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )
        exit_code = select_prompts.main(
            [
                "--game",
                "harmony",
                "--grading",
                "group-mix",
                "--model",
                "mock/policy",
                "--backend",
                "mock",
                "--samples-per-prompt",
                "4",
                "--no-prefilled-think",
                "--label-print-order",
                LABEL_PRINT_ORDER_SWAPPED,
                "--out-dir",
                str(tmp_path),
            ]
        )
        assert exit_code == 0

        trace = next(tmp_path.glob("sweep-*.jsonl"))
        assert LABEL_PRINT_ORDER_SWAPPED in trace.name
        records = read_jsonl(trace)
        assert json_record(records, 0)["label_print_order"] == LABEL_PRINT_ORDER_SWAPPED
        swept = records[1:]
        assert swept
        for record in swept:
            row = cast("dict[str, object]", record["row"])
            assert row["label_print_order"] == LABEL_PRINT_ORDER_SWAPPED
            assert LABEL_PRINT_ORDER_SWAPPED in str(record["prompt_id"])
            # The swap is visible in the prose itself: label_b is offered before label_a.
            prompt = str(row["prompt"])
            assert prompt.index(f"<action>{row['label_b']}</action>") < prompt.index(
                f"<action>{row['label_a']}</action>"
            )


class TestTheSweepCompletionBudget:
    """The sweep's own output budget, against the floor the trainer refuses to run below.

    It shipped as a flat 1,024 tokens on a module constant, 16x under the measured floor for the 4B
    and only ever overridden because both stage plans happen to pass --max-new-tokens. A thinking
    trace past the budget emits no closing tag, so the sample is unparseable, and a prompt more than
    half of whose samples land there is dropped as too-few-parseable -- which biases the selected
    corpus toward prompts with unusually short reasoning and says nothing in the log. That is the
    2026-08-17 incident this module already records in a comment: 64 prompts, all dropped.
    """

    REQUIRED: ClassVar[list[str]] = [
        "--game",
        "twin-pd",
        "--grading",
        "group-mix",
        "--model",
        "Qwen/Qwen3.5-4B",
    ]

    def test_the_sampler_carries_each_models_measured_termination_floor(self):
        for model_id in ("Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"):
            assert training_sampler(model_id).max_new_tokens == required_completion_budget(model_id)
        assert training_sampler("Qwen/Qwen3.5-2B").max_new_tokens == 24576
        assert (
            training_sampler("some/unscreened-checkpoint").max_new_tokens
            == MEASURED_TERMINATION_BUDGET
        )

    def test_the_decoding_fields_are_still_grpos_own_generation_defaults(self):
        """The budget is ours; the other three are TRL's, and that provenance split is the point."""
        sampler = training_sampler("Qwen/Qwen3.5-4B")
        assert (sampler.do_sample, sampler.temperature, sampler.top_p, sampler.top_k) == (
            True,
            1.0,
            1.0,
            0,
        )

    def test_training_sampler_accepts_the_configured_training_top_p_and_budget(self):
        sampler = training_sampler(
            "Qwen/Qwen3.5-9B",
            top_p=0.95,
            max_new_tokens=16384,
        )
        assert (sampler.top_p, sampler.max_new_tokens) == (0.95, 16384)
        assert (sampler.do_sample, sampler.temperature, sampler.top_k) == (True, 1.0, 0)

    def test_none_budget_keeps_the_model_specific_default(self):
        assert training_sampler(
            "Qwen/Qwen3.5-9B",
            max_new_tokens=None,
        ).max_new_tokens == required_completion_budget("Qwen/Qwen3.5-9B")

    def test_an_unflagged_sweep_resolves_to_that_budget(self):
        """The resolution `backend_from_args` performs, which is where 1,024 used to survive."""
        args = select_prompts._parse_args([*self.REQUIRED, "--backend", "hf"])
        select_prompts._require_thinking(args)
        sampling = backend_cli.local_sampling_from_args(args, training_sampler(args.model))
        assert sampling.max_new_tokens == required_completion_budget(args.model)

    def test_a_thinking_sweep_below_the_floor_is_refused(self):
        args = select_prompts._parse_args(
            [*self.REQUIRED, "--backend", "hf", "--max-new-tokens", "1024"]
        )
        select_prompts._require_thinking(args)
        with pytest.raises(ValueError, match="allow-short-completions"):
            select_prompts._validate_completion_budget(args)

    def test_the_escape_hatch_warns_rather_than_passing_in_silence(
        self, caplog: pytest.LogCaptureFixture
    ):
        args = select_prompts._parse_args(
            [
                *self.REQUIRED,
                "--backend",
                "hf",
                "--max-new-tokens",
                "1024",
                "--allow-short-completions",
            ]
        )
        select_prompts._require_thinking(args)
        with caplog.at_level(logging.WARNING):
            select_prompts._validate_completion_budget(args)
        assert "1024" in caplog.text
        assert "16384" in caplog.text

    def test_a_thinking_off_sweep_is_exempt_since_it_is_already_labelled_plumbing(self):
        args = select_prompts._parse_args(
            [*self.REQUIRED, "--backend", "hf", "--no-thinking", "--max-new-tokens", "256"]
        )
        select_prompts._require_thinking(args)
        select_prompts._validate_completion_budget(args)

    def test_a_hosted_sweep_is_not_governed_by_a_local_checkpoints_floor(self):
        args = select_prompts._parse_args(
            [*self.REQUIRED, "--backend", "bedrock", "--max-new-tokens", "1024"]
        )
        select_prompts._require_thinking(args)
        select_prompts._validate_completion_budget(args)


class TestTheWholeRunAgainstAHostedOpponent:
    """`main` end to end on a real vs-frozen corpus, with both backends scripted.

    The two backends differ in exactly the way the live path does. The policy is a local checkpoint
    whose chat template prefills `<think>`, so its completions carry the closing tag alone; the
    opponent is a hosted Converse endpoint, whose completions carry no thinking tag at all because
    the reasoning went into a field `BedrockBackend.generate` does not return. One `prefilled_think`
    cannot be right for both, and this is the only test that runs both halves of one run.
    """

    GAME = "stag-hunt-vs-frozen"
    POLICY_MODEL = "Qwen/Qwen3.5-4B"
    OPPONENT_MODEL = "openai.gpt-oss-20b"

    def rows(self) -> list[dict[str, object]]:
        return generate_prompt_rows(self.GAME, select_prompts.GRADING_VS_FIXED_MIX, split="eval")

    def prefilling_policy(self, rows: list[dict[str, object]]) -> ScriptedBackend:
        """Alternate the two labels, in the shape a template-prefilled checkpoint emits."""
        return ScriptedBackend(
            {
                cast("str", row["prompt"]): [
                    f"weighing</think><action>{row['label_a']}</action>",
                    f"weighing</think><action>{row['label_b']}</action>",
                ]
                for row in rows
            },
            model_id=self.POLICY_MODEL,
        )

    def hosted_opponent(
        self, rows: list[dict[str, object]], *, second_sample: str | None = None
    ) -> ScriptedBackend:
        """Answer with a bare action tag, as a hosted endpoint's answer blocks do."""
        return ScriptedBackend(
            {
                cast("str", row["prompt"]): [
                    f"<action>{row['label_a']}</action>",
                    second_sample
                    if second_sample is not None
                    else f"<action>{row['label_b']}</action>",
                ]
                for row in rows
            },
            model_id=self.OPPONENT_MODEL,
        )

    def run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        *,
        opponent: Backend | None,
        grading: str = select_prompts.GRADING_VS_FIXED_MIX,
        game: str | None = None,
    ) -> int:
        rows = generate_prompt_rows(game or self.GAME, grading, split="eval")
        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )
        monkeypatch.setattr(
            select_prompts.backend_cli,
            "backend_from_args",
            lambda *_args, **_kwargs: self.prefilling_policy(rows),
        )
        if opponent is not None:
            monkeypatch.setattr(select_prompts, "build_backend", lambda *_args, **_kwargs: opponent)
        argv = [
            "--game",
            game or self.GAME,
            "--grading",
            grading,
            "--model",
            self.POLICY_MODEL,
            "--split",
            "eval",
            "--samples-per-prompt",
            "2",
            "--frozen-opponent-samples",
            "2",
            # Given explicitly so the arithmetic never reads the shared card in a unit test.
            "--chunk-size",
            "8",
            "--prefilled-think",
            "--out-dir",
            str(tmp_path),
        ]
        if opponent is not None:
            argv += ["--frozen-opponent-model", self.OPPONENT_MODEL]
        return select_prompts.main(argv)

    def test_the_hosted_opponents_answers_reach_the_corpus(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Read under the policy's own prefilled_think, every one of them was truncated thinking."""
        rows = self.rows()
        assert self.run(tmp_path, monkeypatch, opponent=self.hosted_opponent(rows)) == 0

        corpus = read_jsonl(next(tmp_path.glob("corpus-*.jsonl")))
        assert len(corpus) == len(rows)
        assert all(row["opp_coop_prob"] == pytest.approx(0.5) for row in corpus)

    def test_the_trace_carries_both_sweeps_with_the_meta_first(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = self.rows()
        assert self.run(tmp_path, monkeypatch, opponent=self.hosted_opponent(rows)) == 0

        loaded = read_jsonl(next(tmp_path.glob("sweep-*.jsonl")))
        assert json_record(loaded, 0)["record_kind"] == META_RECORD_KIND
        assert collections.Counter(record["record_kind"] for record in loaded) == {
            META_RECORD_KIND: 1,
            SWEEP_RECORD_KIND: len(rows),
            FROZEN_OPPONENT_RECORD_KIND: len(rows),
        }

    def test_the_policy_sweep_survives_the_opponent_pass_failing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The sweep is the irreplaceable artefact; the opponent step must not be able to eat it.

        Sampling the corpus at the training sampler on a rented card is the run's whole cost and its
        only "before" eval, so a throttle in the hosted pass that follows used to discard every
        completion just paid for, leaving nothing at all on disk.
        """
        rows = self.rows()
        with pytest.raises(RuntimeError, match="throttled"):
            self.run(tmp_path, monkeypatch, opponent=ThrottledBackend(self.OPPONENT_MODEL))

        loaded = read_jsonl(next(tmp_path.glob("sweep-*.jsonl")))
        assert json_record(loaded, 0)["record_kind"] == META_RECORD_KIND
        assert len(loaded) == len(rows) + 1
        assert all(json_record(loaded, index)["samples"] for index in range(1, len(loaded)))
        assert all(
            json_record(loaded, index)["record_kind"] == SWEEP_RECORD_KIND
            for index in range(1, len(loaded))
        )
        # No corpus, because the rows still carry the sentinel -- but nothing has to be re-sampled.
        assert not list(tmp_path.glob("corpus-*.jsonl"))

    def test_the_summary_records_what_the_opponent_failed_to_parse(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A cached probability with no parse count beside it cannot be audited afterwards."""
        rows = self.rows()
        opponent = self.hosted_opponent(rows, second_sample="I would rather not commit.")
        assert self.run(tmp_path, monkeypatch, opponent=opponent) == 0

        summary = cast(
            "dict[str, Any]", json.loads(next(tmp_path.glob("selection-*.json")).read_text())
        )
        frozen = cast("dict[str, Any]", summary["frozen_opponent"])
        assert frozen["model_id"] == self.OPPONENT_MODEL
        assert frozen["samples_per_prompt"] == 2
        assert frozen["min_parseable_fraction"] == DEFAULT_MIN_PARSEABLE_FRACTION
        assert frozen["n_parse_failures_by_prompt"] == {
            cast("str", row["prompt_id"]): 1 for row in rows
        }

    def test_a_vs_fixed_mix_sweep_with_no_opponent_is_refused_before_anything_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Otherwise the corpus is written with the sentinel and training raises on its first step.

        The guard that catches an unfilled row lives inside the very block this omission skips.
        """
        with pytest.raises(ValueError, match="--frozen-opponent-model"):
            self.run(tmp_path, monkeypatch, opponent=None)
        assert not list(tmp_path.iterdir())

    def test_a_frozen_opponent_on_a_grading_that_never_reads_it_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        rows = generate_prompt_rows("stag-hunt", select_prompts.GRADING_GROUP_MIX, split="eval")
        with pytest.raises(ValueError, match="group-mix"):
            self.run(
                tmp_path,
                monkeypatch,
                opponent=self.hosted_opponent(rows),
                grading=select_prompts.GRADING_GROUP_MIX,
                game="stag-hunt",
            )
        assert not list(tmp_path.iterdir())


class TestThrashVerdictAgainstMeasuredRuns:
    """Pin the detector's three gates to the runs that actually happened.

    The arithmetic is all this can check -- whether the detector fires under real allocator pressure
    is a GPU sabotage, recorded separately in `docs/scratch/gate-coverage-ledger.md`. These exist
    because the verdict shipped with no tests at all, and because every number below is a real
    observation rather than an invented one: the two production points come from the rented g7e logs
    in `s3://<bucket>/games_rl/step4-20260818/`, the two L4 points from the sabotage harness.
    """

    def test_the_confirmed_production_thrash_fires(self):
        """Width 172 at 24,576: 2588 retries over 93.5 min, the run that burned 93 minutes."""
        verdict = ThrashVerdict(retries=2588, seconds=93.5 * 60, width=172)
        assert verdict.retries_per_minute_per_sequence == pytest.approx(0.1608, abs=1e-3)
        assert verdict.thrashing

    def test_the_healthy_production_run_stays_quiet(self):
        """Width 64, killed prematurely rather than thrashing: 90 retries over 14 min."""
        verdict = ThrashVerdict(retries=90, seconds=14.0 * 60, width=64)
        assert verdict.retries_per_minute_per_sequence == pytest.approx(0.1002, abs=1e-3)
        assert not verdict.thrashing

    def test_real_pressure_on_the_local_card_fires(self):
        """The sabotage at --fraction 0.23: 26 retries in 149.5 s at width 16, nothing raised."""
        assert ThrashVerdict(retries=26, seconds=149.5, width=16).thrashing

    def test_clearing_the_rate_is_not_enough_without_the_retry_floor(self):
        """The sabotage at --fraction 0.24 cleared 0.12 on rate but had only 5 retries.

        Worth pinning because the two gates disagree at the short end: reaching 8 retries at exactly
        threshold rate takes 4.2 minutes at width 16, so short narrow calls can clear the rate and
        still read False. Harmless at production chunk lengths; surprising if you meet it cold.
        """
        verdict = ThrashVerdict(retries=5, seconds=149.5, width=16)
        assert verdict.retries_per_minute_per_sequence >= THRASH_RETRIES_PER_MINUTE_PER_SEQUENCE
        assert not verdict.thrashing

    def test_a_short_burst_is_not_a_trend(self):
        """Eight retries in 20 s reads as 24/min, and must not narrow a width."""
        assert not ThrashVerdict(retries=8, seconds=20.0, width=16).thrashing

    def test_no_elapsed_time_reports_no_rate_rather_than_dividing_by_zero(self):
        assert (
            ThrashVerdict(retries=5, seconds=0.0, width=16).retries_per_minute_per_sequence == 0.0
        )

    def test_the_probe_is_injectable_so_the_arithmetic_needs_no_gpu(self):
        """`AllocatorProbe` is the seam that makes all of the above testable off-card."""
        ticks = iter([100.0, 160.0])
        counts = iter([0, 40])
        probe = AllocatorProbe(clock=lambda: next(ticks), read_retries=lambda: next(counts))
        started, before = probe.clock(), probe.read_retries()
        elapsed, delta = probe.clock() - started, probe.read_retries() - before
        assert ThrashVerdict(retries=delta, seconds=elapsed, width=16).thrashing


class TestARegradeOnlyGradingCannotBeSwept:
    """The format-only placebo's corpus has to come from a regrade, and both refusals say so.

    Selection keeps the prompts whose ACTION distribution is mixed, and a format-only reward is blind
    to the action, so there is no per-sample score to select on. That makes sweeping under it not
    merely wasteful but undefined -- and the arm wants the graded arms' corpus anyway, since sharing
    one selected prompt set is the invariant the whole contrast rests on.

    Two refusals rather than one, because the cheap one is the one that matters. `_parse_sample`
    raises on the first completion, which is after the model is resident and the DeltaNet bridge is
    bound; `_refuse_a_regrade_only_grading` raises on the arguments, before anything is loaded. The
    argument-level check is easy to leave out because argparse accepts the value: `GRADING_CHOICES`
    is derived from the full grading vocabulary on purpose, so that this message is what an operator
    sees instead of a terse "invalid choice".
    """

    ARGV: ClassVar[list[str]] = [
        "--game",
        "twin-pd",
        "--grading",
        GRADING_FORMAT_ONLY,
        "--model",
        "Qwen/Qwen3.5-2B",
    ]

    def test_the_grading_is_still_an_accepted_argument_value(self):
        # The premise of the guard below: if argparse rejected it, the guard would be unreachable and
        # the recovery message would never be shown.
        assert GRADING_FORMAT_ONLY in select_prompts.GRADING_CHOICES
        assert GRADING_FORMAT_ONLY not in select_prompts.SWEEPABLE_GRADINGS
        assert select_prompts._parse_args(self.ARGV).grading == GRADING_FORMAT_ONLY

    def test_the_arguments_are_refused_before_anything_is_loaded(self):
        args = select_prompts._parse_args(self.ARGV)
        with pytest.raises(ValueError, match="cannot be swept") as raised:
            select_prompts._refuse_a_regrade_only_grading(args)
        message = str(raised.value)
        assert "games.regrade_corpus" in message
        assert "GAMES_ARM_SEQ_SWEEP_GRADING" in message

    def test_the_sweep_entry_point_reaches_the_guard_before_it_builds_a_backend(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The wiring, not the guard: a validator nothing calls is a comment.

        Asserted by making the next expensive step explode with a sentinel, so the test says which
        of the two happened first rather than merely that something was raised. Unhooking the guard
        from `main` was tried and does exactly what the guard exists to prevent: the launch path goes
        on to resolve a real chat template and load a 2B checkpoint. A test that measured that by
        waiting would hang, which is why the backend is stubbed instead.

        The kernel bridge is stubbed for a different reason, and it is the honest one: it refuses once
        `transformers.models.qwen3_5.modeling_qwen3_5` has been imported, since a bridge applied after
        that module binds its kernels is a no-op that reports success. That makes it correct and makes
        `main` uncallable from a pytest session that has already imported the modeling module -- which
        this file's sibling tests do. Stubbing it keeps the assertion about argument-order wiring
        rather than about import order.
        """
        monkeypatch.setattr(select_prompts, "bridge_decode_kernel", lambda: {"bridged": False})
        monkeypatch.setattr(
            select_prompts.backend_cli,
            "backend_from_args",
            lambda *args, **kwargs: pytest.fail(
                "the guard was reached too late to save the launch"
            ),
        )
        with pytest.raises(ValueError, match="cannot be swept"):
            select_prompts.main(self.ARGV)

    def test_a_sweepable_grading_passes_the_same_check(self):
        """A guard that refused every grading would pass its own first test."""
        for grading in sorted(select_prompts.SWEEPABLE_GRADINGS):
            args = select_prompts._parse_args(
                ["--game", "twin-pd", "--grading", grading, "--model", "Qwen/Qwen3.5-2B"]
            )
            select_prompts._refuse_a_regrade_only_grading(args)

    def test_sampling_a_format_only_row_refuses_with_the_same_recovery_path(self):
        rows = [matrix_row(prompt_id="p0", reskin_id="r0", grading=GRADING_FORMAT_ONLY)]
        with pytest.raises(ValueError, match="cannot be swept") as raised:
            sweep(rows, {"p0": [PICKS_HOLD]}, samples_per_prompt=2)
        assert "games.regrade_corpus" in str(raised.value)

    def test_an_unknown_grading_still_gets_the_unknown_grading_message(self):
        # The two branches must stay distinguishable: "this grading exists but not for sweeping" is a
        # different operator problem from "this grading does not exist".
        rows = [matrix_row(prompt_id="p0", reskin_id="r0", grading="vibes")]
        with pytest.raises(ValueError, match="unknown grading"):
            sweep(rows, {"p0": [PICKS_HOLD]}, samples_per_prompt=2)


CARE_ONE = care_grading(1)
TRUST_MULTIPLE = 3.0
TRUST_ANNOUNCED_HALF = 0.5


def care_matrix_row(
    prompt_id: str, reskin_id: str, coop_label: str = COOP_LABEL
) -> dict[str, object]:
    """One care-graded matrix row, which answers with an action."""
    row = matrix_row(
        prompt_id=prompt_id, reskin_id=reskin_id, grading=CARE_ONE, coop_label=coop_label
    )
    row[select_prompts.STATED_RETURN_FRACTION_COLUMN] = STATED_RETURN_UNSET
    return row


def care_trust_row(prompt_id: str) -> dict[str, object]:
    """One care-graded announced-rule trust row, which answers with an amount sent."""
    return {
        "prompt": f"sheet for {prompt_id}",
        "prompt_id": prompt_id,
        "game_id": "trust-vs-stated-return",
        "grading": CARE_ONE,
        "payoff_cc": 0.0,
        "payoff_cd": 0.0,
        "payoff_dc": 0.0,
        "payoff_dd": 0.0,
        "label_a": "",
        "label_b": "",
        "coop_label": "",
        "endowment": ENDOWMENT,
        "opp_coop_prob": -1.0,
        "opponent_rule": "",
        "n_rounds": 0,
        "transfer_multiplier": TRUST_MULTIPLE,
        select_prompts.STATED_RETURN_FRACTION_COLUMN: TRUST_ANNOUNCED_HALF,
        "reskin_id": prompt_id,
        "payoff_variant": "return-half",
    }


class TestTheCareFamilyIsSweptPerRowType:
    """One grading, two row types, so the sweep cannot key its parser or its rule on the name alone.

    Wave 4b's corpus is built directly under `care-alpha-1` and holds five matrix games plus the trust
    sender. Keyed on the grading name, every trust row would be parsed for an action it never writes
    and then dropped for cooperating zero times out of eight -- the whole giving game gone from the
    corpus with a plausible drop reason beside it.
    """

    def test_a_matrix_row_answers_with_an_action_and_a_trust_row_with_a_send(self):
        assert select_prompts.row_scores_a_binary_action(CARE_ONE, care_matrix_row("m", "skin"))
        assert not select_prompts.row_scores_a_binary_action(CARE_ONE, care_trust_row("t"))
        # Unchanged for every grading whose name fixes its row type.
        assert select_prompts.row_scores_a_binary_action(
            GRADING_GROUP_MIX, matrix_row(prompt_id="m", reskin_id="skin")
        )
        assert not select_prompts.row_scores_a_binary_action(
            GRADING_KEEP_FRACTION, dictator_row(prompt_id="d", reskin_id="skin")
        )

    def test_each_row_type_is_read_by_its_own_parser(self):
        assert (
            select_prompts.answer_parser_for(CARE_ONE, care_matrix_row("m", "skin"))
            is select_prompts._parse_action_answer
        )
        assert (
            select_prompts.answer_parser_for(CARE_ONE, care_trust_row("t"))
            is select_prompts._parse_stated_rule_trust_answer
        )
        assert select_prompts.answer_parser_for("care-alpha-x", care_trust_row("t")) is None

    def test_a_mixed_care_sweep_scores_both_halves_and_judges_each_by_its_own_rule(self):
        rows = [
            care_matrix_row("care-skin--coop0", "care-skin", COOP_LABEL),
            care_matrix_row("care-skin--coop1", "care-skin", DEFECT_LABEL),
            care_trust_row("care-trust-spread"),
            care_trust_row("care-trust-flat"),
        ]
        mixed_actions = [PICKS_HOLD, PICKS_SLASH, PICKS_HOLD, PICKS_SLASH]
        sends = ["<send>10</send>", "<send>0</send>", "<send>7</send>", "<send>3</send>"]
        flat_sends = ["<send>10</send>"]
        records = sweep(
            rows,
            {
                "care-skin--coop0": mixed_actions,
                "care-skin--coop1": mixed_actions,
                "care-trust-spread": sends,
                "care-trust-flat": flat_sends,
            },
        )
        by_id = {record.prompt_id: record for record in records}
        assert by_id["care-skin--coop0"].coop_fraction == pytest.approx(0.5)
        # The trust rows have no action pair at all, so no cooperation rate is invented for them.
        assert by_id["care-trust-spread"].coop_fraction is None
        assert by_id["care-trust-spread"].selection_scores == pytest.approx([1.0, 0.0, 0.7, 0.3])
        kept = {row["prompt_id"] for row in select_mixed_prompts(records)}
        assert kept == {"care-skin--coop0", "care-skin--coop1", "care-trust-spread"}
        assert reasons(records)["care-trust-flat"] == DropReason.SCORE_SPREAD_BELOW_MIN


class TestTheGradingFlagAcceptsTheCareFamily:
    """`choices=` is a fixed list and the family's alpha is a number, so the flag validates instead."""

    def test_a_family_member_is_an_accepted_argument_value(self):
        args = select_prompts._parse_args(
            ["--game", "twin-pd", "--grading", CARE_ONE, "--model", "Qwen/Qwen3.5-2B"]
        )
        assert args.grading == CARE_ONE

    def test_a_name_no_grading_answers_to_exits(self):
        with pytest.raises(SystemExit):
            select_prompts._parse_args(
                ["--game", "twin-pd", "--grading", "care-alpha-x", "--model", "Qwen/Qwen3.5-2B"]
            )

    def test_a_non_canonical_spelling_exits_naming_the_canonical_one(
        self, capsys: pytest.CaptureFixture[str]
    ):
        # Argparse discards a ValueError's message, so the validator raises ArgumentTypeError to keep
        # the one-character fix in front of the operator.
        with pytest.raises(SystemExit):
            select_prompts._parse_args(
                ["--game", "twin-pd", "--grading", "care-alpha-1.0", "--model", "Qwen/Qwen3.5-2B"]
            )
        assert "care-alpha-1'" in capsys.readouterr().err


class TestTheRowsInput:
    """`--rows` sweeps an authored pool instead of one game's generated split.

    A breadth corpus spans several games under several counterpart framings, which no single `--game`
    can render, so the pool is authored first (`games.breadth_corpus`) and swept as a file. The three
    things that then differ from a generated sweep are all provenance: the pool cannot be reproduced
    from a game id, so the meta record carries the file's digest; the artifacts cannot be named after a
    game, so they are named after the file; and the flags describing the sweep can now disagree with
    the file, so they are compared against it.
    """

    GRADING = care_grading(1)

    def rows_file(self, path: Path, **overrides: object) -> Path:
        rows = [
            {**row, **overrides}
            for row in generate_prompt_rows("twin-pd", self.GRADING, split="train")[:4]
        ]
        write_corpus(path, rows)
        return path

    def args(self, path: Path, *extra: str) -> argparse.Namespace:
        return select_prompts._parse_args(
            [
                "--rows",
                str(path),
                "--grading",
                self.GRADING,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
                *extra,
            ]
        )

    def test_rows_and_game_refuse_to_combine(self, tmp_path: Path):
        args = select_prompts._parse_args(
            [
                "--rows",
                str(self.rows_file(tmp_path / "rows.jsonl")),
                "--game",
                "twin-pd",
                "--grading",
                self.GRADING,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
            ]
        )
        with pytest.raises(ValueError, match="exactly one of --game and --rows"):
            select_prompts._validate_pool_source(args)

    def test_neither_rows_nor_game_is_refused(self):
        args = select_prompts._parse_args(
            ["--grading", self.GRADING, "--model", "mock/policy", "--backend", "mock"]
        )
        with pytest.raises(ValueError, match="exactly one of --game and --rows"):
            select_prompts._validate_pool_source(args)

    def test_a_game_alone_and_rows_alone_are_both_accepted(self, tmp_path: Path):
        select_prompts._validate_pool_source(self.args(self.rows_file(tmp_path / "rows.jsonl")))
        select_prompts._validate_pool_source(
            select_prompts._parse_args(
                [
                    "--game",
                    "twin-pd",
                    "--grading",
                    self.GRADING,
                    "--model",
                    "mock/policy",
                    "--backend",
                    "mock",
                ]
            )
        )

    def test_the_authored_rows_are_loaded_as_written(self, tmp_path: Path):
        path = self.rows_file(tmp_path / "rows.jsonl")
        rows, digest = select_prompts._load_authored_rows(self.args(path))
        assert rows == read_jsonl(path)
        assert digest == select_prompts.rows_file_digest(path)

    def test_rows_graded_differently_from_the_flag_are_refused(self, tmp_path: Path):
        """Selected by one rule and trained by another is exactly what this catches.

        Every completion is parsed under the grading and every verdict keyed on it, so a file of care
        rows swept as group-mix would be judged by the wrong rule and written out claiming the wrong
        reward.
        """
        path = self.rows_file(tmp_path / "rows.jsonl", grading=GRADING_GROUP_MIX)
        with pytest.raises(ValueError, match="but --grading is"):
            select_prompts._load_authored_rows(self.args(path))

    def test_rows_rendered_in_another_print_order_are_refused(self, tmp_path: Path):
        path = self.rows_file(tmp_path / "rows.jsonl", label_print_order=LABEL_PRINT_ORDER_SWAPPED)
        with pytest.raises(ValueError, match="label print orders"):
            select_prompts._load_authored_rows(self.args(path))

    def test_an_empty_rows_file_is_refused(self, tmp_path: Path):
        path = tmp_path / "rows.jsonl"
        write_corpus(path, [])
        with pytest.raises(ValueError, match="holds no rows"):
            select_prompts._load_authored_rows(self.args(path))

    def test_the_meta_record_carries_the_rows_digest_beside_the_pool_hash(self, tmp_path: Path):
        """The pool hash covers the id ORDER; the digest covers the prompts themselves.

        Two rows files could agree on every id while one carried an edited frame or a different
        counterpart paragraph, which is the failure only an authored pool can have.
        """
        path = self.rows_file(tmp_path / "rows.jsonl")
        args = self.args(path)
        rows, digest = select_prompts._load_authored_rows(args)
        meta = sweep_meta(
            backend=ScriptedBackend({"unused": ["x"]}),
            args=args,
            prompt_ids=[str(row["prompt_id"]) for row in rows],
            samples_per_prompt=4,
            prefilled_think=False,
            rows_sha256=digest,
        )
        assert meta["game"] is None
        assert meta["rows_path"] == str(path)
        assert meta["rows_sha256"] == select_prompts.rows_file_digest(path)
        assert meta["prompt_id_order_sha256"]

        edited = self.rows_file(tmp_path / "edited.jsonl", prompt="a different rendering")
        assert select_prompts.rows_file_digest(edited) != meta["rows_sha256"]

    def test_the_digest_covers_the_bytes_that_were_swept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A rows file rebuilt during the sweep must not be recorded as the file the verdicts came from.

        The digest used to be taken again when the meta line was written, hours after the load on a
        real 632-prompt pool, so a `games.breadth_corpus build` into the same out-dir mid-sweep left
        the trace naming a file version nothing had measured. That is the one direction the digest
        exists for, and only a rebuild while a sweep runs can show it.
        """
        rows_path = self.rows_file(tmp_path / "breadth-candidates.jsonl")
        swept_digest = select_prompts.rows_file_digest(rows_path)
        real_sweep = select_prompts.resumable_sweep

        def rebuild_then_sweep(*args: Any, **kwargs: Any) -> select_prompts.SweepPass:
            self.rows_file(rows_path, prompt="a rebuilt rendering")
            return real_sweep(*args, **kwargs)

        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )
        monkeypatch.setattr(select_prompts, "resumable_sweep", rebuild_then_sweep)
        out_dir = tmp_path / "out"
        assert (
            select_prompts.main(
                [
                    "--rows",
                    str(rows_path),
                    "--grading",
                    self.GRADING,
                    "--model",
                    "mock/policy",
                    "--backend",
                    "mock",
                    "--samples-per-prompt",
                    "4",
                    "--no-prefilled-think",
                    "--out-dir",
                    str(out_dir),
                ]
            )
            == 0
        )
        assert select_prompts.rows_file_digest(rows_path) != swept_digest, (
            "the rebuild did not land"
        )
        meta = json_record(read_jsonl(next(out_dir.glob("sweep-breadth-candidates-*.jsonl"))), 0)
        assert meta["rows_sha256"] == swept_digest

    def test_a_meta_record_naming_no_rows_file_may_not_carry_a_digest(self, tmp_path: Path):
        """A digest with no file behind it, or a file with no digest, is provenance about nothing."""
        args = select_prompts._parse_args(
            [
                "--game",
                "twin-pd",
                "--grading",
                GRADING_GROUP_MIX,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
            ]
        )
        with pytest.raises(ValueError, match="rows_sha256"):
            sweep_meta(
                backend=ScriptedBackend({"unused": ["x"]}),
                args=args,
                prompt_ids=["twin-pd--frame--temptation-2--coop0"],
                samples_per_prompt=4,
                prefilled_think=False,
                rows_sha256="deadbeef",
            )
        with pytest.raises(ValueError, match="rows_sha256"):
            sweep_meta(
                backend=ScriptedBackend({"unused": ["x"]}),
                args=self.args(self.rows_file(tmp_path / "rows.jsonl")),
                prompt_ids=["twin-pd--frame--temptation-2--coop0"],
                samples_per_prompt=4,
                prefilled_think=False,
                rows_sha256=None,
            )

    def test_a_generated_sweep_records_no_rows_provenance(self, tmp_path: Path):
        del tmp_path
        args = select_prompts._parse_args(
            [
                "--game",
                "twin-pd",
                "--grading",
                GRADING_GROUP_MIX,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
            ]
        )
        meta = sweep_meta(
            backend=ScriptedBackend({"unused": ["x"]}),
            args=args,
            prompt_ids=["twin-pd--frame--temptation-2--coop0"],
            samples_per_prompt=4,
            prefilled_think=False,
            rows_sha256=None,
        )
        assert meta["game"] == "twin-pd"
        assert meta["rows_path"] is None
        assert meta["rows_sha256"] is None

    def test_the_artifacts_are_named_after_the_rows_file(self, tmp_path: Path):
        args = self.args(self.rows_file(tmp_path / "breadth-candidates.jsonl"))
        stem = select_prompts._artifact_stem("sweep", args, "20260904T000000Z")
        assert stem == "sweep-breadth-candidates-policy-20260904T000000Z"

    def test_the_cli_sweeps_an_authored_pool_end_to_end(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The whole path on the mock backend: parsing, the file read, all three writes.

        `--backend mock` samples nothing, so the selection this reports is meaningless and only the
        execution is the claim. The bridge is stubbed for the reason `TestArtifacts` states.
        """
        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )
        rows_path = self.rows_file(tmp_path / "breadth-candidates.jsonl")
        out_dir = tmp_path / "out"
        exit_code = select_prompts.main(
            [
                "--rows",
                str(rows_path),
                "--grading",
                self.GRADING,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
                "--samples-per-prompt",
                "4",
                "--no-prefilled-think",
                "--out-dir",
                str(out_dir),
            ]
        )
        assert exit_code == 0
        trace = next(out_dir.glob("sweep-breadth-candidates-*.jsonl"))
        meta = json_record(read_jsonl(trace), 0)
        assert meta["rows_sha256"] == select_prompts.rows_file_digest(rows_path)
        assert meta["n_prompts"] == 4
        assert next(out_dir.glob("corpus-breadth-candidates-*.jsonl")).exists()
        assert next(out_dir.glob("selection-breadth-candidates-*.json")).exists()


class DiesMidSweepBackend(ScriptedBackend):
    """A policy that answers `chunks` decode calls and then dies the way a killed box does.

    A raise rather than a `sys.exit`, because what has to be tested is what is ON DISK when the
    process stops: a partial file holding whole prompts and nothing else. The real thing -- SIGKILL to
    a subprocess mid-sweep -- is rehearsed outside pytest, since a killed interpreter cannot report
    anything back to the test that killed it.
    """

    def __init__(
        self, script: dict[str, list[str]], *, chunks: int, model_id: str = "scripted/policy"
    ) -> None:
        super().__init__(script, model_id)
        self.chunks = chunks

    def generate(self, prompts: list[str]) -> list[str]:
        if len(self.batch_sizes) >= self.chunks:
            raise RuntimeError(f"the box died after {self.chunks} decode calls")
        return super().generate(prompts)


class RefusesToDecodeBackend:
    """A policy that raises on any decode at all, so a resume that regenerates nothing can prove it."""

    transport = "scripted"

    def __init__(self, model_id: str = "scripted/policy") -> None:
        self.model_id = model_id

    def generate(self, prompts: list[str]) -> list[str]:
        raise AssertionError(f"asked to decode {len(prompts)} prompts that are already on disk")


class EngineBackend(ScriptedBackend):
    """A scripted policy that also reports the engine settings it was constructed with.

    `VLLMBackend` is the real one and cannot run offline, but the resume only ever reads the settings
    off the backend, which is the seam this stands in for. Quantization and the model-length cap
    change what the engine samples and where it truncates; the memory claim does not.
    """

    def __init__(
        self,
        script: dict[str, list[str]],
        *,
        engine_settings: dict[str, object],
        model_id: str = "scripted/policy",
    ) -> None:
        super().__init__(script, model_id)
        self.engine_settings = dict(engine_settings)


class TestTheSweepResumesPerRecord:
    """A killed sweep leaves its finished prompts on disk, and a relaunch continues from them.

    The wave-4b candidate sweep is 632 rows at eight samples and a 32,768-token budget: three to six
    hours of a rented card, on which nothing used to reach disk until the last chunk returned. Four
    properties carry the resume, each asserted against an artefact rather than a log line: the partial
    file holds WHOLE prompts only; a relaunch's records are the unbroken run's records, line for line
    and in pool order; a launch whose identity drifted is refused by the field that differs; and a
    launch with nothing left to do decodes nothing.

    Four more say the file survives being read twice, which is where the resume was fragile: a torn
    tail is cut off rather than appended onto, so a SECOND death does not make the file unreadable; a
    tear inside a multi-byte character is a torn tail like any other; a second live sweep into the same
    out-dir is refused by its predecessor's lock while a stale lock is taken over; and every resumed
    record's verdicts are re-derived under this session's parser rather than trusted from disk.

    The decode width here is deliberately NOT a multiple of the samples per prompt, because that is
    the case the flush has to survive: an OOM-halved chunk ends mid-prompt, and a partial record
    holding half a prompt's samples would read as a prompt answered four times when it was answered
    twice.
    """

    SAMPLES = 4
    CHUNK = 5

    def rows(self, n_scenarios: int = 3) -> list[dict[str, object]]:
        return [
            row for index in range(n_scenarios) for row in counterbalanced_pair(f"frame-{index}")
        ]

    def script(self, rows: list[dict[str, object]]) -> dict[str, list[str]]:
        return {str(row["prompt"]): [PICKS_HOLD, PICKS_SLASH] for row in rows}

    def identity(
        self, backend: Any, rows: list[dict[str, object]], **overrides: Any
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "backend": backend,
            "rows": rows,
            "samples_per_prompt": self.SAMPLES,
            "prefilled_think": False,
            "thinking": True,
            "grading": GRADING_GROUP_MIX,
            "rows_sha256": None,
            **overrides,
        }
        return select_prompts.sweep_identity(**fields)

    def run(
        self,
        backend: Any,
        rows: list[dict[str, object]],
        partial_path: Path | None,
        **overrides: Any,
    ) -> select_prompts.SweepPass:
        samples = int(overrides.pop("samples_per_prompt", self.SAMPLES))
        chunk = int(overrides.pop("chunk_size", self.CHUNK))
        identity = (
            None
            if partial_path is None
            else self.identity(backend, rows, samples_per_prompt=samples, **overrides)
        )
        return select_prompts.resumable_sweep(
            backend,
            rows,
            samples_per_prompt=samples,
            prefilled_think=False,
            chunk_size=chunk,
            partial_path=partial_path,
            identity=identity,
        )

    def partial_lines(self, path: Path) -> list[dict[str, Any]]:
        return [cast("dict[str, Any]", record) for record in read_jsonl(path)]

    def records_of(self, path: Path) -> list[dict[str, Any]]:
        return [
            line
            for line in self.partial_lines(path)
            if line["record_kind"] == select_prompts.SWEEP_RECORD_KIND
        ]

    def test_a_killed_sweep_leaves_whole_prompts_and_nothing_partial(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        backend = DiesMidSweepBackend(self.script(rows), chunks=2)

        with pytest.raises(RuntimeError, match="the box died"):
            self.run(backend, rows, partial)

        # Two chunks of five completions is ten, which is two whole prompts and half of a third.
        assert backend.batch_sizes == [self.CHUNK, self.CHUNK]
        lines = self.partial_lines(partial)
        assert lines[0]["record_kind"] == select_prompts.PARTIAL_HEADER_RECORD_KIND
        assert lines[1]["record_kind"] == select_prompts.PARTIAL_SESSION_RECORD_KIND
        kept = self.records_of(partial)
        assert [record["prompt_id"] for record in kept] == [row["prompt_id"] for row in rows[:2]]
        assert [record["n_samples"] for record in kept] == [self.SAMPLES, self.SAMPLES]

    def test_a_width_narrower_than_one_prompt_still_writes_only_whole_prompts(self, tmp_path: Path):
        """The OOM-halved case: three sequences per call against four samples per prompt.

        A record is all `samples_per_prompt` samples of one row or it is not there, so the first chunk
        writes nothing at all and the second completes the first prompt. A flush at chunk boundaries
        instead would file a prompt with three samples as though the model had answered three times,
        and every rate over it -- parseable fraction, cooperation rate, score spread -- would be
        computed on a denominator that never existed.
        """
        rows = self.rows(2)
        partial = tmp_path / "partial" / "pending.jsonl"
        backend = DiesMidSweepBackend(self.script(rows), chunks=1)

        with pytest.raises(RuntimeError, match="the box died"):
            self.run(backend, rows, partial, chunk_size=3)
        assert self.records_of(partial) == []

        resumed = self.run(ScriptedBackend(self.script(rows)), rows, partial, chunk_size=3)
        assert [record["n_samples"] for record in self.records_of(partial)] == [self.SAMPLES] * 4
        assert (resumed.n_resumed, resumed.n_generated) == (0, 4)

    def test_the_relaunch_keeps_them_and_lands_the_unbroken_trace(self, tmp_path: Path):
        """The claim the resume rests on: the same records in the same order, whatever split drew them."""
        rows = self.rows()
        unbroken = self.run(ScriptedBackend(self.script(rows)), rows, None)

        partial = tmp_path / "partial" / "pending.jsonl"
        with pytest.raises(RuntimeError, match="the box died"):
            self.run(DiesMidSweepBackend(self.script(rows), chunks=2), rows, partial)
        resumed = self.run(ScriptedBackend(self.script(rows)), rows, partial)

        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in unbroken.records
        ]
        assert (resumed.n_resumed, resumed.n_generated, resumed.n_torn) == (2, 4, 0)
        assert [session["records_resumed"] for session in resumed.sessions] == [0, 2]
        # Every prompt is in the partial file exactly once once the pass completes.
        assert [record["prompt_id"] for record in self.records_of(partial)] == [
            str(row["prompt_id"]) for row in rows
        ]

    def test_the_records_are_in_pool_order_when_the_resumed_ones_are_not_a_prefix(
        self, tmp_path: Path
    ):
        """Assembly is by the pool's order, never by which session decoded what.

        The relaunch's backend is scripted for the MISSING prompts only, so a resume that re-decoded a
        kept prompt raises instead of quietly producing a second answer for it.
        """
        rows = self.rows()
        unbroken = self.run(ScriptedBackend(self.script(rows)), rows, None)
        out_of_order = [unbroken.records[3], unbroken.records[1], unbroken.records[4]]
        partial = tmp_path / "partial" / "pending.jsonl"
        select_prompts.write_partial_header(
            partial, identity=self.identity(ScriptedBackend(self.script(rows)), rows)
        )
        select_prompts._write_partial_lines(
            partial, [record.to_json_dict() for record in out_of_order]
        )

        kept_ids = {record.prompt_id for record in out_of_order}
        remaining = [row for row in rows if row["prompt_id"] not in kept_ids]
        resumed = self.run(ScriptedBackend(self.script(remaining)), rows, partial)

        assert [record.prompt_id for record in resumed.records] == [
            str(row["prompt_id"]) for row in rows
        ]
        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in unbroken.records
        ]
        assert (resumed.n_resumed, resumed.n_generated) == (3, 3)

    def test_a_third_launch_over_a_partial_whose_trace_was_never_written_decodes_nothing(
        self, tmp_path: Path
    ):
        """A complete partial with no trace beside it is the death between the last flush and the write.

        Every row is on disk and none of them is in an artefact, so the launch after it has to keep
        all of them and decode nothing at all. This is the state a live partial file may be in; a
        partial whose trace DID land is retired out of the resume path instead, and
        `TestTheCliResumesItsOwnSweep` is where that is asserted, because only `main` writes a trace.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        first = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)
        third = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)

        assert (third.n_resumed, third.n_generated, third.n_torn) == (len(rows), 0, 0)
        assert [record.to_json_dict() for record in third.records] == [
            record.to_json_dict() for record in first.records
        ]
        assert len(third.sessions) == 3

    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            ({"samples_per_prompt": 6}, "samples_per_prompt"),
            ({"prefilled_think": True}, "prefilled_think"),
            ({"thinking": False}, "thinking"),
            ({"grading": "self"}, "grading"),
            ({"rows_sha256": "a" * 64}, "rows_sha256"),
        ],
    )
    def test_an_identity_that_drifted_is_refused_by_the_field_that_differs(
        self, tmp_path: Path, overrides: dict[str, Any], field: str
    ):
        """A relaunch under other settings is refused, never folded in and never re-swept in silence.

        Under a digest-named partial file each of these would open a SECOND file and re-sweep the whole
        pool -- three to six hours of a rented card, with nothing in the log to say so -- which is why
        the file is named after the pool and the model and the identity is checked inside it.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(ScriptedBackend(self.script(rows)), rows, partial)

        with pytest.raises(ValueError, match=field):
            self.run(ScriptedBackend(self.script(rows)), rows, partial, **overrides)

    def test_a_rewritten_pool_is_refused_on_its_content_and_not_only_its_size(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(ScriptedBackend(self.script(rows)), rows, partial)

        edited = [{**row, "prompt": f"rewritten {row['prompt']}"} for row in rows]
        with pytest.raises(ValueError, match="pool_digest"):
            self.run(ScriptedBackend(self.script(edited)), edited, partial)

    def test_a_model_the_partial_was_not_swept_on_is_refused(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(ScriptedBackend(self.script(rows)), rows, partial)

        other = ScriptedBackend(self.script(rows), model_id="scripted/another-policy")
        with pytest.raises(ValueError, match=r"backend.*scripted/another-policy"):
            self.run(other, rows, partial)

    @pytest.mark.parametrize(
        ("engine_settings", "field"),
        [
            ({"quantization": "fp8"}, "engine_quantization"),
            ({"max_model_len": 8192}, "engine_max_model_len"),
        ],
    )
    def test_an_engine_setting_that_moves_the_policy_refuses_the_resume(
        self, tmp_path: Path, engine_settings: dict[str, object], field: str
    ):
        """The engine's own construction knobs are part of the measurement, not of the plumbing.

        Neither reaches the sampler, so neither was visible to the identity: an fp8 engine is a
        different policy from the bf16 one the flag's own help refuses to compare against, and a
        shorter model-length cap moves where a prompt plus its completion is cut off, which moves the
        parse-failure rate. A relaunch that adds either used to be folded into the same trace.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(EngineBackend(self.script(rows), engine_settings={}), rows, partial)

        drifted = EngineBackend(self.script(rows), engine_settings=engine_settings)
        with pytest.raises(ValueError, match=field):
            self.run(drifted, rows, partial)

    def test_a_relaunch_that_claims_a_different_share_of_the_card_still_resumes(
        self, tmp_path: Path
    ):
        """gpu_memory_utilization sizes the engine's claim on the card and not what it samples.

        It is exactly the knob that legitimately moves when a relaunch lands on a different card, so
        putting it in the identity would refuse the resume on the case rented capacity makes ordinary.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        first = self.run(
            EngineBackend(self.script(rows), engine_settings={"gpu_memory_utilization": 0.9}),
            rows,
            partial,
        )

        smaller_card = EngineBackend(
            self.script(rows), engine_settings={"gpu_memory_utilization": 0.45}
        )
        resumed = self.run(smaller_card, rows, partial)

        assert (resumed.n_resumed, resumed.n_generated) == (len(rows), 0)
        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in first.records
        ]

    def test_the_identity_carries_the_resolved_thinking_flag_and_not_the_tri_state(
        self, tmp_path: Path
    ):
        """`--thinking` is a tri-state so an explicit choice is distinguishable from the default.

        The identity has to record what the sweep RAN under, though, and on a hosted backend an
        absent flag and a spelled-out `--no-thinking` are the same policy: `resolve_thinking` maps
        both to False. Storing the tri-state meant the second phrasing of the same command was
        refused, and re-paying for every prompt a dead box had already decoded.
        """
        rows = self.rows()
        backend = ScriptedBackend(self.script(rows))

        def identity_for(*extra: str) -> dict[str, Any]:
            args = select_prompts._parse_args(
                [
                    "--game",
                    "twin-pd-group",
                    "--grading",
                    GRADING_GROUP_MIX,
                    "--model",
                    "hosted/policy",
                    "--backend",
                    "bedrock",
                    "--out-dir",
                    str(tmp_path),
                    *extra,
                ]
            )
            return select_prompts.sweep_identity_from_args(
                args, backend=backend, rows=rows, prefilled_think=False, rows_sha256=None
            )

        assert identity_for()["thinking"] is False
        assert identity_for() == identity_for("--no-thinking")

    def test_a_resumed_record_is_reparsed_under_this_session_and_the_change_is_counted(
        self, tmp_path: Path
    ):
        """A relaunch must not select on verdicts a parser this tree no longer has computed.

        Selection reads the derived fields -- `parsed`, `selection_score`, the aggregates over them --
        and a resume is exactly when they can have come from other code: a parse bug found at hour
        three is one of the reasons an operator kills a sweep. The completions are the raw material
        and are kept byte for byte; everything derived from them is recomputed, and a record whose
        verdicts moved is counted into the session line so the trace says the resume mixed code states.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        unbroken = self.run(ScriptedBackend(self.script(rows)), rows, partial)

        # The first record as an older parser left it: the completions untouched, every verdict over
        # them wrong.
        lines = partial.read_text(encoding="utf-8").splitlines()
        stale = json.loads(lines[2])
        for sample in stale["samples"]:
            sample["parsed"] = False
            sample["selection_score"] = None
            sample["action"] = None
        lines[2] = json.dumps(stale)
        partial.write_text("\n".join(lines) + "\n", encoding="utf-8")

        resumed = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)

        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in unbroken.records
        ]
        assert resumed.n_reparsed_changed == 1
        assert resumed.sessions[-1]["records_reparsed_changed"] == 1

    def test_a_second_sweep_into_one_out_dir_is_refused_while_the_first_still_runs(
        self, tmp_path: Path
    ):
        """Two writers appending to one partial interleave their bytes INSIDE a line.

        The operator case is a box that looks hung: the kit is relaunched while the old vLLM engine is
        still decoding, and both processes append. That poisons every later launch the same way a torn
        tail did, and neither process reports anything wrong. The lock names its holder so the refusal
        can say which process to kill.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(ScriptedBackend(self.script(rows)), rows, partial)
        lock = select_prompts.partial_sweep_lock_path(partial)
        lock.write_text(
            json.dumps({"pid": os.getpid(), "hostname": socket.gethostname()}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match=f"pid {os.getpid()}"):
            self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)

    @pytest.mark.parametrize("holder", ["dead-pid", "another-host"])
    def test_a_stale_lock_is_taken_over_rather_than_wedging_the_relaunch(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, holder: str
    ):
        """A lock outlives the process that held it whenever the box died with it held.

        Which on a rented box is the ordinary death, so a lock alone may not wedge a relaunch: only a
        holder that is still running on this host may. A pid above the kernel's ceiling can never be
        running, and a lock stamped with another hostname came off a box that is not this one.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        first = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        lock = select_prompts.partial_sweep_lock_path(partial)
        pid_ceiling = int(Path("/proc/sys/kernel/pid_max").read_text(encoding="utf-8"))
        stale = (
            {"pid": pid_ceiling + 1, "hostname": socket.gethostname()}
            if holder == "dead-pid"
            else {"pid": os.getpid(), "hostname": f"not-{socket.gethostname()}"}
        )
        lock.write_text(json.dumps(stale) + "\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=select_prompts.logger.name):
            resumed = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)

        assert (resumed.n_resumed, resumed.n_generated) == (len(rows), 0)
        assert "stale lock" in caplog.text
        # Released when the pass ends, so the launch after this one does not have to take it over too.
        assert not lock.exists()
        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in first.records
        ]

    def test_a_torn_trailing_line_is_dropped_counted_and_regenerated(self, tmp_path: Path):
        """A death mid-append leaves the last line short of its newline, the one line that can be torn."""
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        unbroken = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        whole = partial.read_text(encoding="utf-8")
        torn_at = whole.rindex("\n", 0, len(whole) - 1) + 1
        partial.write_text(whole[: torn_at + 40], encoding="utf-8")

        resumed = self.run(ScriptedBackend(self.script(rows)), rows, partial)

        assert (resumed.n_resumed, resumed.n_generated, resumed.n_torn) == (len(rows) - 1, 1, 1)
        assert resumed.sessions[-1]["records_dropped"] == 1
        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in unbroken.records
        ]

    def test_a_partial_holding_only_a_torn_header_starts_over_rather_than_wedging(
        self, tmp_path: Path
    ):
        """A death between creating the file and writing its first record carries no work to keep."""
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        partial.parent.mkdir(parents=True)
        partial.write_text('{"record_kind": "sweep-partial-hea', encoding="utf-8")

        resumed = self.run(ScriptedBackend(self.script(rows)), rows, partial)

        assert (resumed.n_resumed, resumed.n_generated, resumed.n_torn) == (0, len(rows), 1)
        assert self.partial_lines(partial)[0]["record_kind"] == (
            select_prompts.PARTIAL_HEADER_RECORD_KIND
        )
        # And what it wrote is readable: the torn header was truncated away rather than appended to,
        # so the launch after this one resumes the whole pool instead of refusing the file.
        again = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)
        assert (again.n_resumed, again.n_generated, again.n_torn) == (len(rows), 0, 0)

    def test_a_partial_that_is_only_a_complete_header_is_appended_to(self, tmp_path: Path):
        """The other death right after creation: the header landed whole and nothing followed it.

        Nothing to keep and nothing to repair, so the sweep runs the whole pool and the file stays a
        file later launches can read -- which the launch after it proves by resuming all of it.
        """
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        select_prompts.write_partial_header(
            partial, identity=self.identity(ScriptedBackend(self.script(rows)), rows)
        )

        first = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        assert (first.n_resumed, first.n_generated, first.n_torn) == (0, len(rows), 0)

        second = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)
        assert (second.n_resumed, second.n_generated, second.n_torn) == (len(rows), 0, 0)

    def test_a_relaunch_after_a_repaired_tear_can_itself_be_resumed(self, tmp_path: Path):
        """Three launches across two deaths, which is what a torn tail used to make unreadable.

        The torn line was dropped from the parse and counted but left on disk, so the relaunch's own
        session line landed fused onto the torn bytes as one interior line. That launch complained
        about nothing -- it ran to the end -- and the launch after it refused the whole file, every
        record the first two decoded included. On a 632-row rented sweep that is three to six hours
        thrown away by the SECOND death rather than by the first, which is why one relaunch was never
        enough to see it.
        """
        rows = self.rows()
        unbroken = self.run(ScriptedBackend(self.script(rows)), rows, None)
        partial = tmp_path / "partial" / "pending.jsonl"

        with pytest.raises(RuntimeError, match="the box died"):
            self.run(DiesMidSweepBackend(self.script(rows), chunks=2), rows, partial)
        whole = partial.read_bytes()
        partial.write_bytes(whole[: whole.rindex(b"\n", 0, len(whole) - 1) + 41])
        with pytest.raises(RuntimeError, match="the box died"):
            self.run(DiesMidSweepBackend(self.script(rows), chunks=2), rows, partial)

        third = self.run(ScriptedBackend(self.script(rows)), rows, partial)

        assert (third.n_resumed, third.n_generated, third.n_torn) == (3, 3, 0)
        assert [record.to_json_dict() for record in third.records] == [
            record.to_json_dict() for record in unbroken.records
        ]
        # The tear was counted where it happened and stays counted: repairing the file must not erase
        # the drop from the trace's resume block.
        assert [session["records_dropped"] for session in third.sessions] == [0, 1, 0]

    def test_a_tear_inside_a_multibyte_character_is_read_as_a_torn_tail(self, tmp_path: Path):
        """The partial file is written with ensure_ascii=False, so a tear can split a character.

        Read back as strict UTF-8 text that is a UnicodeDecodeError raised before the torn-line
        handling is ever reached, and the whole file is refused on the FIRST relaunch -- the case the
        resume exists for. The read is on bytes for that reason, and the tail after the last newline
        is a torn line whatever its bytes are.
        """
        rows = self.rows()
        script = {str(row["prompt"]): [MULTIBYTE_PICKS_HOLD, MULTIBYTE_PICKS_SLASH] for row in rows}
        partial = tmp_path / "partial" / "pending.jsonl"
        unbroken = self.run(ScriptedBackend(script), rows, partial)

        raw = partial.read_bytes()
        # One byte into the last multi-byte character on the file's last line.
        partial.write_bytes(raw[: raw.rindex("权".encode()) + 1])

        resumed = self.run(ScriptedBackend(script), rows, partial)
        assert (resumed.n_resumed, resumed.n_generated, resumed.n_torn) == (len(rows) - 1, 1, 1)
        assert [record.to_json_dict() for record in resumed.records] == [
            record.to_json_dict() for record in unbroken.records
        ]

        third = self.run(cast("Any", RefusesToDecodeBackend()), rows, partial)
        assert (third.n_resumed, third.n_generated, third.n_torn) == (len(rows), 0, 0)

    def test_a_corrupt_line_that_is_not_the_trailing_one_stops_the_relaunch(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        self.run(ScriptedBackend(self.script(rows)), rows, partial)
        lines = partial.read_text(encoding="utf-8").splitlines()
        lines[2] = lines[2][: len(lines[2]) // 2]
        partial.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="line 3 is not JSON"):
            self.run(ScriptedBackend(self.script(rows)), rows, partial)

    def test_two_records_for_one_prompt_refuse_rather_than_reporting_a_rate(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        first = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        select_prompts._write_partial_lines(partial, [first.records[0].to_json_dict()])

        with pytest.raises(ValueError, match="two records for prompt"):
            self.run(ScriptedBackend(self.script(rows)), rows, partial)

    def test_a_record_for_a_prompt_outside_this_pool_refuses(self, tmp_path: Path):
        rows = self.rows()
        partial = tmp_path / "partial" / "pending.jsonl"
        first = self.run(ScriptedBackend(self.script(rows)), rows, partial)
        stray = {**first.records[0].to_json_dict(), "prompt_id": "frame-99--coop0"}
        select_prompts._write_partial_lines(partial, [stray])

        with pytest.raises(ValueError, match="frame-99--coop0"):
            self.run(ScriptedBackend(self.script(rows)), rows, partial)

    def test_a_record_rebuilt_from_a_line_reproduces_that_line(self):
        """ "Kept byte for byte" is a claim about the written line, so it is checked on the written line.

        An iterated row on purpose: `action_sequence` and `levels` are tuples in memory and lists once
        written, and they are the only fields a JSON round trip can silently change the type of. The
        comparison is between serialised forms for that reason -- `to_json_dict` hands back the tuple
        and `json.dumps` is what flattens it, so comparing the dicts would fail on a record that
        writes the identical bytes.
        """

        def moves(*labels: str) -> str:
            tags = "".join(f"<action>{label}</action>" for label in labels)
            return f"<think>planning</think>{tags}"

        rows = counterbalanced_pair(
            "repeated-frame",
            grading=select_prompts.GRADING_ITERATED_RETURN,
            game_id="iterated-pd-tft",
            n_rounds=3,
            opponent_rule="tit-for-tat",
        )
        records = sweep(
            rows,
            {
                str(row["prompt_id"]): [
                    moves(COOP_LABEL, DEFECT_LABEL, COOP_LABEL),
                    moves(DEFECT_LABEL, DEFECT_LABEL, DEFECT_LABEL),
                ]
                for row in rows
            },
        )

        for record, row in zip(records, rows, strict=True):
            written = json.dumps(record.to_json_dict())
            rebuilt = PromptSweepRecord.from_json_dict(json.loads(written), row=row)
            assert json.dumps(rebuilt.to_json_dict()) == written
            # The tuple survives as a tuple rather than coming back as a list, which is the field a
            # round trip loses silently: every aggregate over it would still compute.
            assert isinstance(rebuilt.samples[0].action_sequence, tuple)
        # The first orientation names HOLD as the cooperative option, so its moves read C, D, C.
        first = PromptSweepRecord.from_json_dict(
            json.loads(json.dumps(records[0].to_json_dict())), row=rows[0]
        )
        assert first.samples[0].action_sequence == ("C", "D", "C")

    def test_persisting_without_an_identity_is_refused(self, tmp_path: Path):
        """Half the pair is worse than neither: a partial nobody can refuse resumes over anything."""
        rows = self.rows()
        with pytest.raises(ValueError, match="together or neither"):
            select_prompts.resumable_sweep(
                ScriptedBackend(self.script(rows)),
                rows,
                samples_per_prompt=self.SAMPLES,
                prefilled_think=False,
                partial_path=tmp_path / "pending.jsonl",
            )

    def test_the_partial_path_is_named_for_the_pool_and_the_model_under_the_subdirectory(
        self, tmp_path: Path
    ):
        def args_for(*extra: str) -> argparse.Namespace:
            return select_prompts._parse_args(
                [
                    "--game",
                    "twin-pd-group",
                    "--grading",
                    GRADING_GROUP_MIX,
                    "--model",
                    "Qwen/Qwen3.5-9B",
                    "--out-dir",
                    str(tmp_path),
                    *extra,
                ]
            )

        path = select_prompts.partial_sweep_path(args_for())
        assert path.parent == tmp_path / select_prompts.PARTIAL_SUBDIR
        assert path.name == "pending-sweep-twin-pd-group-Qwen3.5-9B.jsonl"
        # No timestamp and no identity digest in the name: a relaunch has to find the file the dead
        # session was writing, and a digest in it would turn a changed sampler into a silent second
        # sweep of the whole pool instead of a refusal.
        assert (
            select_prompts.partial_sweep_path(
                args_for("--label-print-order", LABEL_PRINT_ORDER_SWAPPED)
            ).name
            == "pending-sweep-twin-pd-group-swapped-Qwen3.5-9B.jsonl"
        )


class TestTheCliResumesItsOwnSweep:
    """`main` end to end: a relaunch over a live partial reuses it, and a finished sweep retires it.

    The teeth are in the second launch's decode being made impossible: `iter_decoded_chunks` raises,
    so the launch completes only if it generated nothing at all. Comparing the trace against the
    partial file's own lines is the other half -- a resume that kept the records writes the same
    lines out again.

    A live partial is one whose trace never landed, which is what a death at hour three of six
    leaves. Once the trace IS on disk the partial is retired, because the same command run twice into
    one out-dir otherwise resumes every row, decodes nothing, and writes a fresh timestamped trace,
    corpus and selection summary that are a copy of the first session's completions.
    """

    def launch(self, out_dir: Path, rows_path: Path) -> int:
        return select_prompts.main(
            [
                "--rows",
                str(rows_path),
                "--grading",
                GRADING_GROUP_MIX,
                "--model",
                "mock/policy",
                "--backend",
                "mock",
                "--samples-per-prompt",
                "4",
                "--no-prefilled-think",
                "--out-dir",
                str(out_dir),
            ]
        )

    def rows_file(self, tmp_path: Path) -> Path:
        rows_path = tmp_path / "candidates.jsonl"
        write_corpus(
            rows_path, generate_prompt_rows("twin-pd", GRADING_GROUP_MIX, split="train")[:4]
        )
        return rows_path

    def stub_the_decode_kernel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            select_prompts, "bridge_decode_kernel", lambda: {"bridged": False, "reason": "stubbed"}
        )

    def partial_dir(self, out_dir: Path) -> Path:
        return out_dir / select_prompts.PARTIAL_SUBDIR

    def test_a_relaunch_after_a_death_before_the_trace_keeps_every_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The sweep finished and the box died before the trace was written: nothing may be re-decoded."""
        self.stub_the_decode_kernel(monkeypatch)
        rows_path = self.rows_file(tmp_path)
        out_dir = tmp_path / "out"

        def die_before_the_trace(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise RuntimeError("the box died before the trace was written")

        with monkeypatch.context() as dying:
            dying.setattr(select_prompts, "write_sweep_trace", die_before_the_trace)
            with pytest.raises(RuntimeError, match="the box died"):
                self.launch(out_dir, rows_path)

        partial = self.partial_dir(out_dir) / "pending-sweep-candidates-policy.jsonl"
        kept = [line for line in read_jsonl(partial) if line["record_kind"] == SWEEP_RECORD_KIND]
        assert len(kept) == 4
        assert not list(out_dir.glob("sweep-candidates-*.jsonl"))

        def refuse_to_decode(*args: Any, **kwargs: Any) -> list[list[str]]:
            del args, kwargs
            raise AssertionError("the relaunch decoded prompts that were already on disk")

        monkeypatch.setattr(select_prompts, "iter_decoded_chunks", refuse_to_decode)
        assert self.launch(out_dir, rows_path) == 0

        trace = read_jsonl(next(out_dir.glob("sweep-candidates-*.jsonl")))
        assert trace[1:] == kept, "a resumed launch rewrote the records it was meant to keep"
        resume = json_record(trace, 0)["resume"]
        assert resume["n_sessions"] == 2
        assert resume["records_resumed"] == 4
        assert resume["records_dropped"] == 0
        assert resume["records_reparsed_changed"] == 0
        assert str(resume["partial_path"]).endswith("pending-sweep-candidates-policy.jsonl")
        assert "fresh draw" in str(resume["seeding"])
        assert [session["git_sha"] for session in resume["sessions"]] == [
            git_provenance()["git_sha"]
        ] * 2

    def test_a_finished_sweep_retires_its_partial_so_the_same_command_is_a_fresh_draw(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The identical command twice into one out-dir used to write a duplicate of the first sweep.

        Every row resumed, nothing decoded, and a fresh timestamped trace, corpus and selection
        summary written over the previous session's completions -- a copy that reads as a new
        measurement, and a second corpus that makes `games.plans.resolve_single_corpus` refuse the
        directory as ambiguous. The tells were a log line and `resume.n_sessions`.
        """
        self.stub_the_decode_kernel(monkeypatch)
        rows_path = self.rows_file(tmp_path)
        out_dir = tmp_path / "out"

        assert self.launch(out_dir, rows_path) == 0

        # Retired to a name the launch kit's own `*.jsonl` count under partial/ does not see.
        assert list(self.partial_dir(out_dir).glob("*.jsonl")) == []
        retired = list(
            self.partial_dir(out_dir).glob("pending-sweep-candidates-policy.jsonl.completed-*")
        )
        assert len(retired) == 1
        assert [line["record_kind"] for line in read_jsonl(retired[0])].count(
            SWEEP_RECORD_KIND
        ) == 4

        assert self.launch(out_dir, rows_path) == 0

        # The newest trace: two launches inside one second share a timestamp and so a filename.
        rerun = read_jsonl(max(out_dir.glob("sweep-candidates-*.jsonl")))
        resume = json_record(rerun, 0)["resume"]
        assert (resume["n_sessions"], resume["records_resumed"]) == (1, 0)
