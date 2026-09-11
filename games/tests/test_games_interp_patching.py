"""Offline tests for `games.interp_patching`: the corpus joins, the arm table, and the exact null.

Everything here runs on CPU against a hand-built two-layer model and a char-based tokenizer, so the
whole patch path -- render, bracket, capture, hook, readout, summarise -- is exercised with no
download and no card. The model's second layer is a cumulative sum along the sequence, which is the
minimum mixing needed for a patch at an early position to reach the last-position readout; without
it every patch would read as a no-op for the wrong reason.

Three checks here are the ones the module exists to keep honest, and all three have been watched to
fail:

* the identity arm (side A patched with side A's own activations) must move NO logit at all, exactly.
  Sabotaged by perturbing the identity arm's replacement rows -- the exactness assertion goes red.
* that same null must hold at bf16, which is the dtype every real run computes in. The fake model is
  therefore driven at both dtypes rather than only float32, and the CLI fixture honours the
  `compute_dtype` its namespace names instead of pinning float32 under a config that says otherwise.
  Watched red against the code before `_readout_logits` promoted its row: the patched readout came
  back bf16 beside float32 baselines, so an in-tensor action gap rounded onto the bf16 grid and all
  8 identity cells reported a non-zero `gap_shift` while `max_abs_logit_shift` stayed exactly 0.0.
* a layer index outside the model must be refused before any forward runs. Sabotaged by dropping the
  range check -- a negative index then hooks a layer counted from the end and reports it under the
  index the caller asked for.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch

from games import interp_patching
from games.interp_cells import COMPUTE_DTYPES, STIMULUS_RENDER_TEMPLATED, Stimulus
from games.interp_patching import (
    ANSWER_CLOSER,
    ARM_IDENTITY,
    ARM_MATCHED_NORM,
    ARM_MISMATCHED_PAIR,
    ARM_REAL,
    PATCH_ARMS,
    PLAN_FILENAME,
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    PairRun,
    PatchCorpusError,
    arm_rows,
    build_pair_runs,
    donor_assignments,
    donor_for,
    donor_rows_for_pairs,
    forced_choice_text,
    identity_control_summary,
    layer_rows,
    limit_pairs,
    load_provenance,
    missing_donor_skips,
    pair_readouts,
    patch_identity,
    patch_pair,
    patch_unit_key,
    plan_for_run,
    readout_logits,
    rendered_ids_digest,
    resolve_layers,
    resolve_windows,
    run_patch,
    run_plan,
    summarise_records,
)
from games.interp_stimuli import (
    RENDER_GRADING,
    SET_CAUSAL_VS_FUNCTIONAL,
    SET_COOPERATE_VS_DEFECT,
    SIDE_A,
    SIDE_B,
    TWIN_GAME_ID,
    first_printed_label,
)
from games.parsing import THINK_CLOSE, THINK_OPEN
from games.prompts import LABEL_PRINT_ORDER_CANONICAL, SPLIT_EVAL, generate_prompt_rows
from reward_hacking.interp.jsonl_resume import ResumeMismatchError, ledger_path_for
from reward_hacking.interp.steering import (
    PATCH_WINDOW_GRADER_BODY,
    PATCH_WINDOW_SHARED_PREFIX,
    PatchBaseline,
    capture_patch_prefix,
    run_activation_patch,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

HIDDEN = 8
VOCAB = 128
N_LAYERS = 2
TURN_PREFIX = "<|user|>"
TURN_SUFFIX = f"<|assistant|>{THINK_OPEN}\n"
SHARED_REASONING = "Both sides read the same four cells. "
CONTINUATION = {SIDE_A: "A commits to the joint row.", SIDE_B: "B commits to the other row."}


# --------------------------------------------------------------------------------------
# Fakes: a tokenizer with a chat template, and a two-layer causal LM that mixes positions
# --------------------------------------------------------------------------------------


class TinyTokenizer:
    """Char-based ids plus a chat template that prefills `<think>`, like this family's own."""

    pad_token: str | None = "<pad>"
    eos_token = "<eos>"

    def get_chat_template(self) -> str:
        """No `reasoning_effort` knob, so nothing gets pinned for it."""
        return "{% for message in messages %}...{% endfor %}"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = True,
        **kwargs: object,
    ) -> str:
        """One user turn, then the assistant turn opened at `<think>` when thinking is on."""
        del tokenize, add_generation_prompt, kwargs
        suffix = TURN_SUFFIX if enable_thinking else TURN_SUFFIX.removesuffix(f"{THINK_OPEN}\n")
        return f"{TURN_PREFIX}{messages[0]['content']}{suffix}"

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        """Tokenize one string; nothing is prepended, which is what the capture path asserts."""
        del add_special_tokens
        return {"input_ids": [(ord(char) % (VOCAB - 1)) + 1 for char in text] or [1]}


class _IdentityLayer(torch.nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _CumulativeSumLayer(torch.nn.Module):
    """Causal mixing: position t sums 0..t, so a patch upstream reaches the last-position readout."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(hidden, dim=1)


class _Trunk(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
        self.layers = torch.nn.ModuleList([_IdentityLayer(), _CumulativeSumLayer()])

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs: object
    ) -> SimpleNamespace:
        del attention_mask, kwargs
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class TinyLM(torch.nn.Module):
    """`.model.layers` trunk plus an LM head that honours `logits_to_keep` exactly as HF does."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = _Trunk()
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB)

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: object,
    ) -> SimpleNamespace:
        del kwargs
        outputs = self.model(input_ids, attention_mask)
        kept = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=self.lm_head(outputs.last_hidden_state[:, kept, :]))


# --------------------------------------------------------------------------------------
# A corpus built from the REAL prompt rows, so the label and provenance joins are exercised
# --------------------------------------------------------------------------------------


def real_rows(n: int) -> list[dict[str, Any]]:
    """The first `n` twin-pd eval rows, exactly as the corpus builder would have read them."""
    rows = generate_prompt_rows(
        TWIN_GAME_ID,
        RENDER_GRADING,
        split=SPLIT_EVAL,
        label_print_order=LABEL_PRINT_ORDER_CANONICAL,
    )
    return rows[:n]


def make_corpus(n_pairs: int = 3) -> tuple[list[Stimulus], list[dict[str, Any]]]:
    """Matched-stem pairs plus their provenance sidecar, keyed to real prompt ids."""
    stimuli: list[Stimulus] = []
    provenance: list[dict[str, Any]] = []
    for row in real_rows(n_pairs):
        pair_id = f"{SET_COOPERATE_VS_DEFECT}--{row['prompt_id']}--form-0"
        for side in (SIDE_A, SIDE_B):
            stimulus_id = f"{pair_id}--{side}"
            stimuli.append(
                Stimulus(
                    stimulus_id=stimulus_id,
                    stimulus_set=SET_COOPERATE_VS_DEFECT,
                    side=side,
                    pair_id=pair_id,
                    text=f"a game stem for {row['prompt_id']}",
                    assistant_prefix=SHARED_REASONING + CONTINUATION[side],
                )
            )
            provenance.append(
                {
                    "id": stimulus_id,
                    "set": SET_COOPERATE_VS_DEFECT,
                    "side": side,
                    "pair_id": pair_id,
                    "prompt_id": row["prompt_id"],
                    "reskin_id": row["reskin_id"],
                    "split": SPLIT_EVAL,
                    "payoff_variant": row["payoff_variant"],
                    "label_print_order": row["label_print_order"],
                    "coop_label": row["coop_label"],
                    "first_printed_label": first_printed_label(row),
                    "counterpart_framing": "correlated-instance",
                    "cited_cells": "matched-column",
                }
            )
    return stimuli, provenance


def write_corpus(tmp_path: Path, n_pairs: int = 3) -> tuple[Path, Path]:
    """Write the corpus and its sidecar as JSONL, the way the capture driver reads them."""
    stimuli, provenance = make_corpus(n_pairs)
    stimuli_path = tmp_path / "stimuli.jsonl"
    stimuli_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": stimulus.stimulus_id,
                    "set": stimulus.stimulus_set,
                    "side": stimulus.side,
                    "pair_id": stimulus.pair_id,
                    "text": stimulus.text,
                    "assistant_prefix": stimulus.assistant_prefix,
                }
            )
            for stimulus in stimuli
        )
        + "\n"
    )
    provenance_path = tmp_path / "provenance.jsonl"
    provenance_path.write_text("\n".join(json.dumps(row) for row in provenance) + "\n")
    return stimuli_path, provenance_path


def built_runs(n_pairs: int = 3) -> list[PairRun]:
    """Run the real build path over the synthetic corpus."""
    stimuli, provenance = make_corpus(n_pairs)
    runs, skipped = build_pair_runs(
        cast("Any", TinyTokenizer()),
        stimuli,
        {row["id"]: row for row in provenance},
        sets=[SET_COOPERATE_VS_DEFECT],
    )
    assert not skipped
    return runs


# --------------------------------------------------------------------------------------


class TestForcedChoice:
    def test_the_closer_closes_thinking_and_opens_the_action_tag(self) -> None:
        text = forced_choice_text(f"{TURN_PREFIX}stem{TURN_SUFFIX}some reasoning")
        assert text.endswith("<action>")
        assert THINK_CLOSE in text
        # The reasoning sits inside the thinking block, not after it.
        assert text.index("some reasoning") < text.index(THINK_CLOSE)

    def test_both_sides_of_a_pair_end_with_the_same_closer(self) -> None:
        runs = built_runs(1)
        closer_ids = cast("Any", TinyTokenizer())(ANSWER_CLOSER)["input_ids"]
        for ids in (runs[0].source_ids, runs[0].target_ids):
            assert ids[-len(closer_ids) :].tolist() == closer_ids


class TestProvenanceJoin:
    def test_a_sidecar_missing_a_stimulus_is_refused(self, tmp_path: Path) -> None:
        stimuli, provenance = make_corpus(2)
        path = tmp_path / "p.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in provenance[:-1]) + "\n")
        with pytest.raises(PatchCorpusError, match="does not cover the corpus"):
            load_provenance(path, stimuli)

    def test_an_extra_provenance_row_is_refused(self, tmp_path: Path) -> None:
        stimuli, provenance = make_corpus(2)
        path = tmp_path / "p.jsonl"
        rows = [*provenance, {**provenance[0], "id": "not-in-the-corpus"}]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        with pytest.raises(PatchCorpusError, match="does not cover the corpus"):
            load_provenance(path, stimuli)

    def test_a_repeated_id_is_refused(self, tmp_path: Path) -> None:
        stimuli, provenance = make_corpus(1)
        path = tmp_path / "p.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in [*provenance, provenance[0]]) + "\n")
        with pytest.raises(PatchCorpusError, match="repeats id"):
            load_provenance(path, stimuli)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        stimuli, _ = make_corpus(1)
        with pytest.raises(PatchCorpusError, match="is not a file"):
            load_provenance(tmp_path / "absent.jsonl", stimuli)

    def test_provenance_that_disagrees_with_the_regenerated_row_is_refused(self) -> None:
        """The join failure that would otherwise hand back labels for a different prompt."""
        stimuli, provenance = make_corpus(1)
        by_id = {row["id"]: dict(row) for row in provenance}
        for row in by_id.values():
            row["coop_label"] = "NOT-A-LABEL"
        with pytest.raises(PatchCorpusError, match="the prompt generator has moved"):
            build_pair_runs(
                cast("Any", TinyTokenizer()),
                stimuli,
                by_id,
                sets=[SET_COOPERATE_VS_DEFECT],
            )

    def test_a_prompt_the_generator_does_not_make_is_refused(self) -> None:
        stimuli, provenance = make_corpus(1)
        by_id = {row["id"]: dict(row) for row in provenance}
        for row in by_id.values():
            row["prompt_id"] = "twin-pd--invented--temptation-2--coop0"
        with pytest.raises(PatchCorpusError, match="which the prompt generator does not produce"):
            build_pair_runs(
                cast("Any", TinyTokenizer()), stimuli, by_id, sets=[SET_COOPERATE_VS_DEFECT]
            )


class TestPairSelection:
    def test_a_set_without_action_labels_is_refused(self) -> None:
        stimuli, provenance = make_corpus(1)
        with pytest.raises(PatchCorpusError, match="carry no action labels"):
            build_pair_runs(
                cast("Any", TinyTokenizer()),
                stimuli,
                {row["id"]: row for row in provenance},
                sets=[SET_CAUSAL_VS_FUNCTIONAL],
            )

    def test_a_half_pair_is_refused(self) -> None:
        stimuli, provenance = make_corpus(2)
        kept = [
            stimulus
            for stimulus in stimuli
            if stimulus.side == SIDE_A or "coop1" in stimulus.pair_id
        ]
        with pytest.raises(PatchCorpusError, match="missing a side"):
            build_pair_runs(
                cast("Any", TinyTokenizer()),
                kept,
                {row["id"]: row for row in provenance},
                sets=[SET_COOPERATE_VS_DEFECT],
            )

    def test_colliding_label_tokens_are_skipped_and_counted(self) -> None:
        """A pair whose two labels share a first token cannot be told apart at one position."""

        class _CollidingTokenizer(TinyTokenizer):
            def __call__(
                self, text: str, *, add_special_tokens: bool = True
            ) -> dict[str, list[int]]:
                del add_special_tokens
                # Every string starts with the same id, so the two labels are indistinguishable.
                return {"input_ids": [7, *[(ord(char) % (VOCAB - 1)) + 1 for char in text]]}

        stimuli, provenance = make_corpus(2)
        runs, skipped = build_pair_runs(
            cast("Any", _CollidingTokenizer()),
            stimuli,
            {row["id"]: row for row in provenance},
            sets=[SET_COOPERATE_VS_DEFECT],
        )
        assert not runs
        assert len(skipped) == 2
        assert all("collide" in str(entry["reason"]) for entry in skipped)

    def test_n_pairs_limits_per_set_in_a_fixed_order(self) -> None:
        runs = built_runs(3)
        kept = limit_pairs(runs, 2)
        assert [run.pair_id for run in kept] == sorted(run.pair_id for run in runs)[:2]
        assert limit_pairs(runs, None) == runs


class TestLayerGuard:
    """Sabotage-verified 2026-08-21: with the range check removed both assertions below go red, and
    a `-1` run then gets as far as the activation capture, which refuses it there. Separately
    measured: `residual_intervention` itself accepts `-1` silently and hooks the last layer, so the
    placebo arms -- which supply their own rows and never trigger a capture -- are exactly the path
    that needs this guard rather than the apparatus's."""

    def test_a_layer_past_the_end_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside this 2-layer model"):
            resolve_layers(cast("Any", TinyLM()), "0,5")

    def test_a_negative_layer_is_refused_rather_than_hooking_from_the_end(self) -> None:
        with pytest.raises(ValueError, match="negative index would hook a different layer"):
            resolve_layers(cast("Any", TinyLM()), "-1")

    def test_an_empty_layer_list_is_refused(self) -> None:
        with pytest.raises(ValueError, match="named no layers"):
            resolve_layers(cast("Any", TinyLM()), " , ")

    def test_layers_come_back_sorted_and_deduplicated(self) -> None:
        assert resolve_layers(cast("Any", TinyLM()), "1,0,1") == [0, 1]


class TestWindowNames:
    def test_an_unknown_window_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown windows"):
            resolve_windows(f"{PATCH_WINDOW_GRADER_BODY},not-a-window")


class TestArmRows:
    def test_real_and_identity_write_the_source_rows(self) -> None:
        source, target = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
        generator = torch.Generator().manual_seed(0)
        for arm in (ARM_REAL, ARM_IDENTITY):
            rows = arm_rows(
                arm, source_rows=source, target_rows=target, donor_rows=None, generator=generator
            )
            assert rows is not None
            assert torch.equal(rows, source)

    def test_the_matched_norm_placebo_matches_the_real_edit_norm(self) -> None:
        source, target = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
        rows = arm_rows(
            ARM_MATCHED_NORM,
            source_rows=source,
            target_rows=target,
            donor_rows=None,
            generator=torch.Generator().manual_seed(0),
        )
        assert rows is not None
        assert float((rows - target).norm()) == pytest.approx(
            float((source - target).norm()), abs=1e-4
        )
        assert not torch.allclose(rows, source)

    def test_the_matched_norm_placebo_is_a_no_op_where_the_window_is_identical(self) -> None:
        """The shared-prefix control: a matched-norm perturbation of nothing is nothing."""
        rows = torch.randn(3, HIDDEN)
        placebo = arm_rows(
            ARM_MATCHED_NORM,
            source_rows=rows,
            target_rows=rows,
            donor_rows=None,
            generator=torch.Generator().manual_seed(0),
        )
        assert placebo is not None
        assert torch.equal(placebo, rows)

    def test_a_missing_donor_comes_back_as_none(self) -> None:
        rows = arm_rows(
            ARM_MISMATCHED_PAIR,
            source_rows=torch.randn(2, HIDDEN),
            target_rows=torch.randn(2, HIDDEN),
            donor_rows=None,
            generator=torch.Generator().manual_seed(0),
        )
        assert rows is None

    def test_an_unknown_arm_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown arm"):
            arm_rows(
                "steer:+",
                source_rows=torch.randn(2, HIDDEN),
                target_rows=torch.randn(2, HIDDEN),
                donor_rows=None,
                generator=torch.Generator().manual_seed(0),
            )


class TestDonorRotation:
    def test_a_donor_is_never_the_pair_itself(self) -> None:
        eligible = [0, 1, 2]
        widths = [32, 32, 32]
        for index in eligible:
            assert donor_for(index, eligible, n_positions=32, widths=widths) != index

    def test_a_too_narrow_donor_is_skipped(self) -> None:
        assert donor_for(0, [0, 1, 2], n_positions=32, widths=[32, 4, 32]) == 2

    def test_no_wide_enough_donor_returns_none(self) -> None:
        assert donor_for(0, [0, 1], n_positions=32, widths=[32, 4]) is None

    def test_donor_rows_come_from_another_pair_at_the_same_width(self) -> None:
        runs = built_runs(2)
        plans = [plan_for_run(run, max_narrow_positions=8) for run in runs]
        rows_by_pair = {
            run.pair_id: torch.randn(int(run.source_ids.numel()), HIDDEN) for run in runs
        }
        donors = donor_rows_for_pairs(runs, plans, rows_by_pair, windows=[PATCH_WINDOW_GRADER_BODY])
        for index, run in enumerate(runs):
            donor_id, donor_rows = donors[index][PATCH_WINDOW_GRADER_BODY]
            assert donor_id != run.pair_id
            window = next(w for w in plans[index].windows if w.name == PATCH_WINDOW_GRADER_BODY)
            assert donor_rows.shape[0] == window.n_positions


# --------------------------------------------------------------------------------------
# The integration: one pair, every window, every arm, through the real patch driver
# --------------------------------------------------------------------------------------


def synthetic_rows_by_pair(runs: Sequence[PairRun]) -> dict[str, torch.Tensor]:
    """Per-pair `[seq, HIDDEN]` rows whose value is the position index: a donor table with no model.

    The donor arm only needs rows of the right width from another pair, and reading them off the
    model would make this fixture depend on the capture under test.
    """
    return {
        run.pair_id: torch.stack(
            [
                torch.full((HIDDEN,), float(index + 1))
                for index in range(int(run.source_ids.numel()))
            ]
        )
        for run in runs
    }


def patched_records(
    layer: int = 0, *, windows: Sequence[str] | None = None, dtype: torch.dtype = torch.float32
) -> list[dict[str, Any]]:
    """Patch two pairs at one layer and return the flat records.

    `dtype` is the model's compute dtype and is a real parameter of what the identity null means: a
    bf16 model rounds an in-tensor logit difference onto the bf16 grid, so the exactness assertions
    only have teeth when they are also run at bf16.
    """
    model = TinyLM().to(dtype)
    runs = built_runs(2)
    plans = [plan_for_run(run, max_narrow_positions=8) for run in runs]
    names = list(windows) if windows is not None else [w.name for w in plans[0].windows]
    donors = donor_rows_for_pairs(runs, plans, synthetic_rows_by_pair(runs), windows=names)
    records: list[dict[str, Any]] = []
    for index, (run, plan) in enumerate(zip(runs, plans, strict=True)):
        pair_records, skips = patch_pair(
            cast("Any", model),
            run,
            layer=layer,
            plan=plan,
            windows=names,
            donor_rows_by_window=donors[index],
            seed=index,
            source_rows_all=_source_rows(model, run, layer),
            baseline=pair_readouts(cast("Any", model), run),
            deltanet_kernel=TINY_KERNEL,
        )
        assert not skips
        records.extend(pair_records)
    return records


def _source_rows(model: TinyLM, run: PairRun, layer: int) -> torch.Tensor:
    """Side A's residual at `layer`, the way the driver captures it once per (pair, layer)."""
    ids = run.source_ids.unsqueeze(0)
    return layer_rows(cast("Any", model), ids, torch.ones_like(ids), layer)


class TestPatchPair:
    def test_every_window_gets_every_arm(self) -> None:
        records = patched_records()
        by_window: dict[str, set[str]] = {}
        for record in records:
            by_window.setdefault(str(record["window"]), set()).add(str(record["arm"]))
        assert by_window
        for window, arms in by_window.items():
            assert arms == set(PATCH_ARMS), f"{window} is missing an arm"

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
    def test_the_identity_arm_moves_no_logit_at_all(self, dtype: torch.dtype) -> None:
        """The exact null, at both compute dtypes a real run uses.

        Sabotage-verified twice. Adding 1e-3 to the identity arm's replacement rows makes
        `max_abs_logit_shift` non-zero and the max-shift assertion goes red. And the bf16 case is
        the one that caught the readout-dtype split: before
        `reward_hacking.interp.steering._readout_logits` promoted its row to float32, the two
        baselines were float32 and the patched row was bf16, so `gap_shift` -- a difference taken
        inside one tensor -- rounded onto the bf16 grid on the patched side only. All 8 identity
        cells here read `gap_shift == round_bf16(gap) - gap` (-0.09375 at a gap of 60.09) beside
        `max_abs_logit_shift == 0.0`, reproducing the ~5%-of-pairs artifact seen on the 4B run. The
        float32-only version of this test could not go red on any of it.
        """
        identity = [r for r in patched_records(dtype=dtype) if r["arm"] == ARM_IDENTITY]
        assert identity
        for record in identity:
            assert record["max_abs_logit_shift"] == 0.0
            assert record["gap_shift"] == 0.0
            assert record["patch_delta_norm"] == 0.0

    def test_the_shared_prefix_control_recovers_nothing(self) -> None:
        """Token-for-token identical in both runs, so its real arm can only report zero."""
        control = [
            r for r in patched_records(windows=[PATCH_WINDOW_SHARED_PREFIX]) if r["arm"] == ARM_REAL
        ]
        assert control
        for record in control:
            assert record["patch_delta_norm"] == pytest.approx(0.0, abs=1e-6)
            assert record["max_abs_logit_shift"] == pytest.approx(0.0, abs=1e-5)

    def test_the_real_patch_moves_the_readout_where_the_runs_diverge(self) -> None:
        real = [
            r for r in patched_records(windows=[PATCH_WINDOW_GRADER_BODY]) if r["arm"] == ARM_REAL
        ]
        assert real
        for record in real:
            assert float(record["patch_delta_norm"]) > 0
            assert float(record["max_abs_logit_shift"]) > 0

    def test_the_mismatched_donor_is_named_and_differs_from_the_real_arm(self) -> None:
        records = patched_records(windows=[PATCH_WINDOW_GRADER_BODY])
        by_pair: dict[str, dict[str, dict[str, Any]]] = {}
        for record in records:
            by_pair.setdefault(str(record["pair_id"]), {})[str(record["arm"])] = record
        for pair_id, arms in by_pair.items():
            mismatched = arms[ARM_MISMATCHED_PAIR]
            assert mismatched["donor_pair_id"] not in (None, pair_id)
            assert arms[ARM_REAL]["donor_pair_id"] is None
            assert mismatched["gap_patched"] != arms[ARM_REAL]["gap_patched"]

    def test_an_arm_with_no_donor_is_recorded_as_a_skip(self) -> None:
        model = TinyLM()
        run = built_runs(1)[0]
        plan = plan_for_run(run, max_narrow_positions=8)
        records, skips = patch_pair(
            cast("Any", model),
            run,
            layer=0,
            plan=plan,
            windows=[PATCH_WINDOW_GRADER_BODY],
            donor_rows_by_window={},
            seed=0,
            source_rows_all=_source_rows(model, run, 0),
            baseline=pair_readouts(cast("Any", model), run),
            deltanet_kernel=TINY_KERNEL,
        )
        assert [str(entry["arm"]) for entry in skips] == [ARM_MISMATCHED_PAIR]
        assert {str(record["arm"]) for record in records} == set(PATCH_ARMS) - {ARM_MISMATCHED_PAIR}
        assert skips == missing_donor_skips(run, plan, 0, {}, windows=[PATCH_WINDOW_GRADER_BODY]), (
            "the resume path must re-derive exactly the skips patch_pair records"
        )

    def test_donor_assignments_are_layer_free_and_name_the_donor_positions(self) -> None:
        runs = built_runs(2)
        plans = [plan_for_run(run, max_narrow_positions=8) for run in runs]
        assignments = donor_assignments(runs, plans, windows=[PATCH_WINDOW_GRADER_BODY])
        for index, per_pair in enumerate(assignments):
            assignment = per_pair[PATCH_WINDOW_GRADER_BODY]
            assert assignment.donor_index != index
            window = next(w for w in plans[index].windows if w.name == PATCH_WINDOW_GRADER_BODY)
            assert int(assignment.donor_positions.numel()) == window.n_positions

    def test_every_record_names_the_layer_the_positions_and_the_pair(self) -> None:
        for record in patched_records(layer=1):
            assert record["layer"] == 1
            assert record["pair_id"]
            assert len(record["source_positions"]) == record["n_positions"]
            assert len(record["target_positions"]) == record["n_positions"]


class TestSummaries:
    def _record(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "set": SET_COOPERATE_VS_DEFECT,
            "layer": 3,
            "window": PATCH_WINDOW_GRADER_BODY,
            "arm": ARM_REAL,
            "label_print_order": LABEL_PRINT_ORDER_CANONICAL,
            "counterpart_framing": "correlated-instance",
            "gap_recovery": 0.5,
            "gap_shift": 1.0,
            "max_abs_logit_shift": 2.0,
            "patch_delta_norm": 3.0,
        }
        base.update(overrides)
        return base

    def test_cells_carry_their_denominators_and_both_splits(self) -> None:
        summary = summarise_records(
            [
                self._record(),
                self._record(label_print_order="swapped", gap_recovery=0.1),
                self._record(arm=ARM_MATCHED_NORM, gap_recovery=0.0),
            ]
        )
        cells = {(cell["arm"], cell["window"]): cell for cell in summary["cells"]}
        real = cells[ARM_REAL, PATCH_WINDOW_GRADER_BODY]
        assert real["n"] == 2
        assert real["mean_gap_recovery"] == pytest.approx(0.3)
        assert set(real["by_print_order"]) == {LABEL_PRINT_ORDER_CANONICAL, "swapped"}
        assert real["by_counterpart_framing"]["correlated-instance"]["n"] == 2
        assert cells[ARM_MATCHED_NORM, PATCH_WINDOW_GRADER_BODY]["mean_gap_recovery"] == 0.0

    def test_a_zero_denominator_is_counted_not_averaged(self) -> None:
        summary = summarise_records([self._record(), self._record(gap_recovery=None)])
        cell = summary["cells"][0]
        assert cell["n"] == 2
        assert cell["n_with_gap"] == 1
        assert cell["n_zero_denominator"] == 1
        assert cell["mean_gap_recovery"] == pytest.approx(0.5)

    def test_a_pool_from_two_kernels_is_refused_before_anything_is_averaged(self) -> None:
        """The mixing guard at the summariser, where a records file from two runs arrives.

        The bindings differ on the chunked prefill kernel (fla's against the torch fallback), which is
        the one difference a forward-only record can carry: the decode kernel is not on it.
        """
        fla_chunk = {**TINY_KERNEL}
        torch_chunk = {
            **TINY_KERNEL,
            "chunk_gated_delta_rule": (
                "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule"
            ),
        }
        with pytest.raises(ValueError, match="different Gated DeltaNet kernel bindings"):
            summarise_records(
                [self._record(deltanet_kernel=fla_chunk), self._record(deltanet_kernel=torch_chunk)]
            )

    def test_the_identity_summary_counts_exact_nulls(self) -> None:
        control = identity_control_summary(
            [
                self._record(arm=ARM_IDENTITY, max_abs_logit_shift=0.0),
                self._record(arm=ARM_IDENTITY, max_abs_logit_shift=0.25),
                self._record(arm=ARM_REAL, max_abs_logit_shift=9.0),
            ]
        )
        assert control == {"n": 2, "n_exactly_zero": 1, "worst_max_abs_logit_shift": 0.25}


# --------------------------------------------------------------------------------------
# CLI level, offline
# --------------------------------------------------------------------------------------


def cli_args(tmp_path: Path, command: str, **overrides: Any) -> argparse.Namespace:
    """The namespace the parser would build, with the corpus already on disk."""
    stimuli, provenance = write_corpus(tmp_path, n_pairs=2)
    defaults: dict[str, Any] = {
        "command": command,
        "stimuli": stimuli,
        "provenance": provenance,
        "model": "tiny/base",
        "sets": SET_COOPERATE_VS_DEFECT,
        "stimulus_render": STIMULUS_RENDER_TEMPLATED,
        "no_thinking": False,
        "max_narrow_positions": 8,
    }
    if command == "plan":
        defaults["out"] = tmp_path / PLAN_FILENAME
    else:
        defaults.update(
            adapter=None,
            compute_dtype="bfloat16",
            layers="0",
            windows=f"{PATCH_WINDOW_GRADER_BODY},{PATCH_WINDOW_SHARED_PREFIX}",
            n_pairs=None,
            seed=0,
            deadline=None,
            out_dir=tmp_path / "patch",
        )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


TINY_KERNEL: dict[str, str] = {
    "chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule",
    "causal_conv1d_fn": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_fn",
}
"""The kernel binding a patch record carries: the two kernels a forward with no cache dispatches.

Stated rather than read off the modeling module, so these tests neither import it nor depend on
whether some earlier test in the process bridged it -- and so the resume tests can hand a DIFFERENT
binding to a relaunch and watch the ledger refuse it. The decode pair is absent on purpose: this leg
never dispatches it, so it must not decide whether a relaunch is a continuation.
"""

TINY_KERNELS_BOUND: dict[str, str] = {
    **TINY_KERNEL,
    "recurrent_gated_delta_rule": (
        "transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule"
    ),
    "causal_conv1d_update": "transformers.models.qwen3_5.modeling_qwen3_5.causal_conv1d_update",
}
"""What `bound_deltanet_kernels` reads on an un-bridged box: all four, for the summary's provenance."""

TINY_KERNELS_BOUND_BRIDGED: dict[str, str] = {
    **TINY_KERNELS_BOUND,
    "recurrent_gated_delta_rule": (
        "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule"
    ),
}
"""The same box after the fla decode bridge: only the recurrent kernel moves."""

TINY_IDENTITY: dict[str, Any] = {
    "model": "tiny/base",
    "model_weights_identity": "test:tiny/base",
    "deltanet_kernel": TINY_KERNEL,
    "adapter_dir": None,
}
"""The shape `load_model` records, with the revision lookup a hub id would make stood in for."""


def tiny_loaded_model(model: Any, **identity: Any) -> Any:
    """A `LoadedModel` over a fake, with the identity fields a relaunch is compared on overridable."""
    return interp_patching.LoadedModel(
        model=model,
        identity={**TINY_IDENTITY, **identity},
        deltanet_kernel_bridge={"bridged": False, "reason": "offline test"},
        deltanet_kernels_bound=TINY_KERNELS_BOUND,
    )


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the tokenizer and model loaders for the CPU fakes.

    The fake honours `args.compute_dtype` through the same `COMPUTE_DTYPES` table the real loader
    uses, so the CLI tests run at the dtype their namespace names. A fake pinned to float32 while
    the namespace said `bfloat16` is what let the readout-dtype split live under a green CLI test.
    """
    monkeypatch.setattr(interp_patching, "load_tokenizer", lambda _model_id: TinyTokenizer())
    monkeypatch.setattr(
        interp_patching,
        "load_model",
        lambda args: tiny_loaded_model(TinyLM().to(COMPUTE_DTYPES[args.compute_dtype])),
    )


@pytest.mark.usefixtures("offline")
class TestPlanOffline:
    def test_plan_reports_every_pairs_bracket(self, tmp_path: Path) -> None:
        payload = run_plan(cli_args(tmp_path, "plan"))
        assert len(payload["pairs"]) == 2
        for row in payload["pairs"]:
            assert row["prefix_len"] > 0
            assert row["source_middle_len"] > 0
            assert row["target_middle_len"] > 0
            assert row["suffix_covers_closer"]
            assert PATCH_WINDOW_GRADER_BODY in row["windows"]
        assert payload["denominators"] == {"n_pairs": 2, "n_skipped": 0, "skipped": []}
        assert (tmp_path / PLAN_FILENAME).is_file()


@pytest.mark.usefixtures("offline")
class TestPatchOffline:
    def test_every_record_of_a_cell_names_the_same_kernel_binding(self, tmp_path: Path) -> None:
        """Rank 15's record half: the field is on every record and is one binding for the whole run.

        Sabotage-verified: dropping `deltanet_kernel` from `patch_record`'s returned dict turns the
        first assertion red, and pointing one record's binding at the fused kernel turns the summary
        red through the mixing guard.
        """
        summary = run_patch(cli_args(tmp_path, "patch", layers="0,1"))
        records = [
            json.loads(line)
            for line in (tmp_path / "patch" / RECORDS_FILENAME).read_text().splitlines()
        ]
        assert records
        assert all(record["deltanet_kernel"] == TINY_KERNEL for record in records)
        assert summary["resolved_model"]["deltanet_kernel"] == TINY_KERNEL
        assert summary["deltanet_kernel_bridge"]["bridged"] is False
        assert summary["deltanet_kernels_bound"] == TINY_KERNELS_BOUND

    def test_patch_writes_records_and_a_summary(self, tmp_path: Path) -> None:
        summary = run_patch(cli_args(tmp_path, "patch"))
        out_dir = tmp_path / "patch"
        lines = (out_dir / RECORDS_FILENAME).read_text().splitlines()
        assert len(lines) == summary["denominators"]["n_records"] == len(lines)
        assert summary["denominators"]["n_pairs_run"] == 2
        assert summary["layers"] == [0]
        assert summary["batch_size"] == 1
        assert summary["resolved_model"]["model"] == "tiny/base"
        assert summary["stimuli_sha256"]
        assert summary["identity_control"]["n"] > 0
        assert summary["identity_control"]["worst_max_abs_logit_shift"] == 0.0
        assert (out_dir / SUMMARY_FILENAME).is_file()
        assert summary["stopped_reason"] is None
        windows = {str(record["window"]) for record in map(json.loads, lines)}
        assert windows == {PATCH_WINDOW_GRADER_BODY, PATCH_WINDOW_SHARED_PREFIX}

    def test_a_passed_deadline_stops_and_says_so(self, tmp_path: Path) -> None:
        summary = run_patch(cli_args(tmp_path, "patch", deadline="2000-01-01T00:00:00+00:00"))
        assert summary["stopped_reason"] is not None
        assert summary["denominators"]["n_records"] == 0

    def test_a_short_suffix_is_refused_before_the_model_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the two sides stop ending with the same closer, their readouts ask different
        questions and every window below the divergence is measured against the wrong thing.

        Simulated by giving side B a different closer, which is what a render change would do.
        """

        def side_specific_closer(rendered: str) -> str:
            tail = ANSWER_CLOSER if CONTINUATION[SIDE_B] not in rendered else "<|other|>"
            return rendered + tail

        monkeypatch.setattr(
            interp_patching, "load_model", lambda _args: pytest.fail("loaded the model")
        )
        monkeypatch.setattr(interp_patching, "forced_choice_text", side_specific_closer)
        with pytest.raises(PatchCorpusError, match="not the same question"):
            run_patch(cli_args(tmp_path, "patch"))


class TestPatchBaselineReuse:
    def test_one_pair_pays_for_its_two_un_patched_readouts_once(self) -> None:
        """The baseline is what a patched forward is measured against, and it cannot depend on the
        arm; supplying it is what keeps a four-arm, multi-window sweep affordable."""
        model = TinyLM()
        ids = torch.tensor([[2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        supplied = PatchBaseline(clean=torch.zeros(VOCAB), corrupted=torch.zeros(VOCAB))
        result = run_activation_patch(
            cast("Any", model),
            clean_ids=ids,
            corrupted_ids=ids,
            clean_mask=mask,
            corrupted_mask=mask,
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([1]),
            replacement_rows=torch.zeros(1, HIDDEN),
            baseline=supplied,
            answer_token=3,
        )
        assert torch.equal(result.clean, supplied.clean)
        assert torch.equal(result.corrupted, supplied.corrupted)


class TestReadoutDtype:
    """All three readouts of a cell must sit on ONE grid, whatever dtype the model computes in.

    This is the invariant the identity null rests on. A logit difference taken within one row is
    computed in that row's dtype; the same difference taken across two rows is promoted. So while
    the baselines and the patched row disagree about dtype, `gap_shift` and `max_abs_logit_shift`
    answer the same question on two different grids and an exact null reads non-zero on one of them.
    Watched red against the pre-fix code, where the patched row came back bf16 beside float32
    baselines.
    """

    def test_the_patched_readout_lands_on_the_baselines_grid(self) -> None:
        model = TinyLM().to(torch.bfloat16)
        ids = torch.tensor([[2, 3, 4, 5]])
        mask = torch.ones_like(ids)
        baseline_row = readout_logits(cast("Any", model), ids, mask)
        result = run_activation_patch(
            cast("Any", model),
            clean_ids=ids,
            corrupted_ids=ids,
            clean_mask=mask,
            corrupted_mask=mask,
            layer=0,
            clean_positions=torch.tensor([1]),
            corrupted_positions=torch.tensor([1]),
            replacement_rows=torch.zeros(1, HIDDEN),
            baseline=PatchBaseline(clean=baseline_row, corrupted=baseline_row),
            answer_token=3,
        )
        assert result.patched.dtype == result.clean.dtype == result.corrupted.dtype
        assert result.patched.dtype == torch.float32


class _CountingTrunk:
    """Wraps the fake trunk's forward so every model forward -- capture, readout, patched -- counts."""

    def __init__(self, model: TinyLM) -> None:
        self.model = model
        self.calls = 0
        self._forward = model.model.forward

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self._forward(*args, **kwargs)


class TestForwardBudget:
    """Rank 28 of the hot-path backlog: no forward is spent on a number that cannot change.

    Per pair the two un-patched readouts are read once (they depend on the ids, not the layer), and
    per (pair, layer) side A's residual is captured once (it serves the donor table and every arm),
    side B's once, and each side's replay prefix once (rank 27); every remaining forward is a patched
    one, one per record. Before rank 28 each layer re-read both readouts and re-captured side A inside
    `patch_pair`: three redundant forwards per (pair, layer). Sabotage-verified: re-adding a
    `readout_logits` call inside `patch_pair` puts the count over the bound and this goes red.

    The count is of TRUNK entries, so a replayed patched forward still counts one: what rank 27 buys
    is fewer LAYERS per patched forward, which `TestPatchPrefixReplay` counts instead.
    """

    def test_forwards_equal_readouts_plus_captures_plus_one_per_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = TinyLM().to(torch.bfloat16)
        counter = _CountingTrunk(model)
        monkeypatch.setattr(model.model, "forward", counter)
        monkeypatch.setattr(interp_patching, "load_tokenizer", lambda _model_id: TinyTokenizer())
        monkeypatch.setattr(interp_patching, "load_model", lambda args: tiny_loaded_model(model))
        summary = run_patch(cli_args(tmp_path, "patch", layers="0,1"))
        n_pairs = summary["denominators"]["n_pairs_run"]
        n_layers = len(summary["layers"])
        n_records = summary["denominators"]["n_records"]
        assert n_pairs == 2
        assert n_layers == 2
        assert n_records > 0
        readouts = 2 * n_pairs
        captures = n_layers * 2 * n_pairs
        prefixes = n_layers * 2 * n_pairs
        assert counter.calls == readouts + captures + prefixes + n_records
        assert summary["identity_control"]["worst_max_abs_logit_shift"] == 0.0


class _CrashAfterUnits:
    """A `patch_pair` that dies after `survive` (layer, pair) units, the way a reclaim does."""

    def __init__(self, survive: int) -> None:
        self.survive = survive
        self.calls = 0
        self.armed = True
        self._real = patch_pair

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.armed and self.calls > self.survive:
            raise RuntimeError("simulated reclaim mid-layer")
        return self._real(*args, **kwargs)


@pytest.mark.usefixtures("offline")
class TestPatchResume:
    """Automatic (layer, pair)-level resume (hot-path backlog rank 16), mirroring the steering leg.

    A relaunch carries every complete unit forward, re-derives its skips from the layer-free donor
    assignment, re-runs partial units, and ends with a records file byte-identical to an
    uninterrupted run's -- the per-pair seed is `args.seed + index`, so a re-run unit draws the same
    matched-norm noise. Other-configuration relaunches and tampered files are refused.
    """

    @staticmethod
    def _records(out_dir: Path) -> bytes:
        return (out_dir / RECORDS_FILENAME).read_bytes()

    def test_a_relaunch_resumes_every_unit_and_rewrites_nothing(self, tmp_path: Path) -> None:
        first = run_patch(cli_args(tmp_path, "patch", layers="0,1"))
        first_bytes = self._records(tmp_path / "patch")
        second = run_patch(cli_args(tmp_path, "patch", layers="0,1"))
        assert {(u["layer"], u["pair_id"]) for u in second["resumed_units"]} == {
            (layer, pair_id) for layer in (0, 1) for pair_id in first["denominators"]["pair_ids"]
        }
        assert second["denominators"]["n_records_resumed"] == first["denominators"]["n_records"]
        assert second["denominators"]["skips"] == first["denominators"]["skips"]
        assert second["identity_control"] == first["identity_control"]
        assert second["cells"] == first["cells"]
        assert self._records(tmp_path / "patch") == first_bytes

    def test_a_run_killed_mid_layer_resumes_into_the_uninterrupted_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reference = run_patch(
            cli_args(tmp_path, "patch", layers="0,1", out_dir=tmp_path / "reference")
        )
        crash = _CrashAfterUnits(survive=1)
        monkeypatch.setattr(interp_patching, "patch_pair", crash)
        with pytest.raises(RuntimeError, match="simulated reclaim"):
            run_patch(cli_args(tmp_path, "patch", layers="0,1", out_dir=tmp_path / "interrupted"))
        records_path = tmp_path / "interrupted" / RECORDS_FILENAME
        ledger = json.loads(ledger_path_for(records_path).read_text())
        assert len(ledger["completed_units"]) == 1
        # A torn final line, as a kill between write and flush would leave.
        with records_path.open("ab") as handle:
            handle.write(b'{"pair_id": "torn')

        crash.armed = False
        resumed = run_patch(
            cli_args(tmp_path, "patch", layers="0,1", out_dir=tmp_path / "interrupted")
        )
        assert len(resumed["resumed_units"]) == 1
        assert (
            resumed["denominators"]["n_records_resumed"]
            == ledger["completed_units"][patch_unit_key(0, resumed["resumed_units"][0]["pair_id"])]
        )
        assert resumed["denominators"]["n_records"] == reference["denominators"]["n_records"]
        assert resumed["denominators"]["skips"] == reference["denominators"]["skips"]
        assert resumed["cells"] == reference["cells"]
        assert self._records(tmp_path / "interrupted") == self._records(tmp_path / "reference")

    def test_a_relaunch_under_another_seed_is_refused(self, tmp_path: Path) -> None:
        run_patch(cli_args(tmp_path, "patch"))
        with pytest.raises(ResumeMismatchError, match="seed"):
            run_patch(cli_args(tmp_path, "patch", seed=7))

    def test_a_relaunch_over_other_layers_is_refused(self, tmp_path: Path) -> None:
        """A layer set is part of the plan, not a filter: it changes which units the file holds."""
        run_patch(cli_args(tmp_path, "patch", layers="0"))
        with pytest.raises(ResumeMismatchError, match="layers"):
            run_patch(cli_args(tmp_path, "patch", layers="0,1"))

    def test_a_records_file_that_disagrees_with_its_ledger_is_refused(self, tmp_path: Path) -> None:
        """One record of a complete unit duplicated: a surplus regeneration cannot explain."""
        run_patch(cli_args(tmp_path, "patch"))
        records_path = tmp_path / "patch" / RECORDS_FILENAME
        lines = records_path.read_bytes().splitlines(keepends=True)
        records_path.write_bytes(b"".join([*lines, lines[0]]))
        with pytest.raises(ResumeMismatchError, match="disagrees with its own ledger"):
            run_patch(cli_args(tmp_path, "patch"))

    def test_a_relaunch_on_other_base_weights_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same model name, another resolved revision: the name alone would have resumed."""
        first = run_patch(cli_args(tmp_path, "patch"))
        assert first["resolved_model"]["model_weights_identity"] == "test:tiny/base"
        monkeypatch.setattr(
            interp_patching,
            "load_model",
            lambda args: tiny_loaded_model(
                TinyLM().to(COMPUTE_DTYPES[args.compute_dtype]),
                model_weights_identity="hf:moved",
            ),
        )
        with pytest.raises(ResumeMismatchError, match="model_weights_identity"):
            run_patch(cli_args(tmp_path, "patch"))

    def test_a_relaunch_through_another_prefill_kernel_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same weights through a different chunked kernel are different forwards, so not a resume.

        Refused at relaunch rather than at the summary, which is where a mixed file would otherwise be
        caught: by then the GPU has already been spent on the remaining units through the other kernel.
        The escape is a fresh --out-dir. This is the binding a box without fla installed would carry.
        """
        run_patch(cli_args(tmp_path, "patch"))
        torch_chunk = {
            **TINY_KERNEL,
            "chunk_gated_delta_rule": (
                "transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule"
            ),
        }
        monkeypatch.setattr(
            interp_patching,
            "load_model",
            lambda args: tiny_loaded_model(
                TinyLM().to(COMPUTE_DTYPES[args.compute_dtype]), deltanet_kernel=torch_chunk
            ),
        )
        with pytest.raises(ResumeMismatchError, match="deltanet_kernel"):
            run_patch(cli_args(tmp_path, "patch"))


class TestLoadModelKernelIdentity:
    """`load_model` puts the two prefill kernels in the identity and keeps the full binding beside it.

    Through the real `load_model` with its collaborators stubbed (the weights loader, the revision
    lookup, the bridge and the binding read), because the narrowing lives there: a fake `load_model`
    would only test what the fake claimed. The decode kernel is what the fla bridge re-binds, and a
    patched forward never dispatches it, so a relaunch across the bridge has to resume into
    bit-identical records rather than refuse them -- a 9B patching ledger is ~7 GPU-hours.

    Sabotage-verified: recording ``kernels_bound`` instead of its prefill subset in `load_model` turns
    the relaunch test red with a `deltanet_kernel` mismatch.
    """

    @pytest.fixture
    def real_load_model(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        state: dict[str, Any] = {"bound": TINY_KERNELS_BOUND, "bridged": False}
        monkeypatch.setattr(interp_patching, "load_tokenizer", lambda _model_id: TinyTokenizer())
        monkeypatch.setattr(
            interp_patching,
            "load_adapter_base",
            lambda _model_id, *, dtype, device: TinyLM().to(dtype),
        )
        monkeypatch.setattr(
            interp_patching, "resolve_weights_identity", lambda _model_id: "test:tiny/base"
        )
        monkeypatch.setattr(
            interp_patching,
            "bridge_and_check_decode_kernel",
            lambda: {"bridged": state["bridged"], "reason": "offline test"},
        )
        monkeypatch.setattr(interp_patching, "bound_deltanet_kernels", lambda: dict(state["bound"]))
        return state

    def test_the_identity_carries_the_prefill_kernels_and_the_summary_the_full_binding(
        self, tmp_path: Path, real_load_model: dict[str, Any]
    ) -> None:
        del real_load_model
        summary = run_patch(cli_args(tmp_path, "patch"))
        records = [
            json.loads(line)
            for line in (tmp_path / "patch" / RECORDS_FILENAME).read_text().splitlines()
        ]
        assert summary["resolved_model"]["deltanet_kernel"] == TINY_KERNEL
        assert all(record["deltanet_kernel"] == TINY_KERNEL for record in records)
        assert summary["deltanet_kernels_bound"] == TINY_KERNELS_BOUND

    def test_a_relaunch_across_the_fla_decode_bridge_resumes_into_the_same_file(
        self, tmp_path: Path, real_load_model: dict[str, Any]
    ) -> None:
        first = run_patch(cli_args(tmp_path, "patch", layers="0,1"))
        first_bytes = (tmp_path / "patch" / RECORDS_FILENAME).read_bytes()

        real_load_model["bound"] = TINY_KERNELS_BOUND_BRIDGED
        real_load_model["bridged"] = True
        second = run_patch(cli_args(tmp_path, "patch", layers="0,1"))

        assert second["denominators"]["n_records_resumed"] == first["denominators"]["n_records"]
        assert second["cells"] == first["cells"]
        assert (tmp_path / "patch" / RECORDS_FILENAME).read_bytes() == first_bytes
        assert second["deltanet_kernel_bridge"]["bridged"] is True
        assert second["deltanet_kernels_bound"] == TINY_KERNELS_BOUND_BRIDGED
        assert second["resolved_model"]["deltanet_kernel"] == TINY_KERNEL


class TestPatchIdentityPinsTheRenderedIds:
    """The resume identity digests what every pair tokenized to, in run order."""

    def test_a_changed_token_or_a_reordered_run_changes_the_digest(self) -> None:
        runs = built_runs(2)
        base = rendered_ids_digest(runs)
        assert rendered_ids_digest(list(reversed(runs))) != base
        retokenized = replace(runs[0], target_ids=runs[0].target_ids.clone())
        retokenized.target_ids[-1] = retokenized.target_ids[-1] + 1
        assert rendered_ids_digest([retokenized, runs[1]]) != base

    def test_patch_identity_carries_the_pair_ids_and_the_digest(self, tmp_path: Path) -> None:
        runs = built_runs(2)
        identity = patch_identity(
            cli_args(tmp_path, "patch"),
            {
                "model": "tiny/base",
                "model_weights_identity": "hf:abc",
                "device": "cuda:0",
                "adapter_dir": "/somewhere",
            },
            sets=[SET_COOPERATE_VS_DEFECT],
            layers=[0],
            windows=[PATCH_WINDOW_GRADER_BODY],
            runs=runs,
            stimuli_sha256="abc",
            closer_tokens=1,
        )
        assert identity["pair_ids"] == [run.pair_id for run in runs]
        assert identity["rendered_ids_sha256"] == rendered_ids_digest(runs)
        assert identity["model_identity"] == {
            "model": "tiny/base",
            "model_weights_identity": "hf:abc",
        }, "device and adapter_dir are machine-local and must not enter the identity"


class TestPatchPrefixReplay:
    """Rank 27: a patched forward that replays layers 0..L instead of recomputing them.

    Two claims, and the second is the one that could go wrong quietly. First, the replay is a saving:
    the layers below the patched one run once per (pair, layer) to build the prefix, not once per arm.
    Second, it is exactly the same measurement: the identity arm still reads a max logit shift of
    exactly 0.0 and every record is bit-identical to what the forward-hook path produced. That second
    claim is checked against the hook path itself, computed here from the same inputs, so it cannot
    drift with the implementation under test.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["float32", "bfloat16"])
    @pytest.mark.parametrize("layer", [0, 1])
    def test_records_are_bit_identical_to_the_forward_hook_path(
        self, layer: int, dtype: torch.dtype
    ) -> None:
        """Layer 0 is the case with teeth: the fake's mixing layer runs AFTER the patch there.

        At layer 1 -- the last layer of `TinyLM` -- nothing downstream of the patch remains, so only
        layer 0 checks that a replayed forward's later layers read what a full forward would have
        handed them. Both are run, and both dtypes, since bf16 is where an inexact restore of the
        captured residual would show up.
        """
        names = [PATCH_WINDOW_GRADER_BODY, PATCH_WINDOW_SHARED_PREFIX]
        replayed = patched_records(layer=layer, windows=names, dtype=dtype)
        hooked = _records_without_prefix(names, layer=layer, dtype=dtype)
        assert len(replayed) == len(hooked) > 0
        for replayed_record, hooked_record in zip(replayed, hooked, strict=True):
            assert replayed_record == hooked_record

    def test_the_identity_arm_is_still_exactly_zero_under_the_replay(self) -> None:
        records = patched_records(layer=1)
        summary = identity_control_summary(records)
        assert summary["n"] > 0
        assert summary["n_exactly_zero"] == summary["n"]
        assert summary["worst_max_abs_logit_shift"] == 0.0

    def test_a_prefix_captured_at_another_layer_is_refused(self) -> None:
        model = TinyLM().to(torch.bfloat16)
        run = built_runs(1)[0]
        plan = plan_for_run(run, max_narrow_positions=8)
        window = next(w for w in plan.windows if w.name == PATCH_WINDOW_GRADER_BODY)
        ids = run.target_ids.unsqueeze(0)
        prefix = capture_patch_prefix(
            cast("Any", model), corrupted_ids=ids, corrupted_mask=torch.ones_like(ids), layer=0
        )
        with pytest.raises(ValueError, match="captured at layer 0, not at layer 1"):
            run_activation_patch(
                cast("Any", model),
                clean_ids=run.source_ids.unsqueeze(0),
                corrupted_ids=ids,
                clean_mask=torch.ones_like(run.source_ids).unsqueeze(0),
                corrupted_mask=torch.ones_like(ids),
                layer=1,
                clean_positions=window.clean_positions,
                corrupted_positions=window.corrupted_positions,
                replacement_rows=_source_rows(model, run, 1)[window.clean_positions],
                prefix=prefix,
            )

    def test_a_prefix_captured_over_other_ids_is_refused(self) -> None:
        model = TinyLM().to(torch.bfloat16)
        runs = built_runs(2)
        plan = plan_for_run(runs[0], max_narrow_positions=8)
        window = next(w for w in plan.windows if w.name == PATCH_WINDOW_GRADER_BODY)
        other_ids = runs[1].target_ids.unsqueeze(0)
        prefix = capture_patch_prefix(
            cast("Any", model),
            corrupted_ids=other_ids,
            corrupted_mask=torch.ones_like(other_ids),
            layer=1,
        )
        ids = runs[0].target_ids.unsqueeze(0)
        with pytest.raises(ValueError, match="no layer-1 output was captured"):
            run_activation_patch(
                cast("Any", model),
                clean_ids=runs[0].source_ids.unsqueeze(0),
                corrupted_ids=ids,
                clean_mask=torch.ones_like(runs[0].source_ids).unsqueeze(0),
                corrupted_mask=torch.ones_like(ids),
                layer=1,
                clean_positions=window.clean_positions,
                corrupted_positions=window.corrupted_positions,
                replacement_rows=_source_rows(model, runs[0], 1)[window.clean_positions],
                prefix=prefix,
            )

    def test_the_layers_below_the_patch_run_once_per_pair_not_once_per_arm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The saving itself: layer 0 is entered twice per (pair, layer), once for each side's prefix.

        Sabotage-verified: dropping `prefix=target.prefix` from `patch_pair`'s call puts layer 0 back
        in every arm's forward and this count rises.
        """
        model = TinyLM().to(torch.bfloat16)
        entries = {"count": 0}
        real_forward = model.model.layers[0].forward

        def counting_forward(*args: Any, **kwargs: Any) -> Any:
            entries["count"] += 1
            return real_forward(*args, **kwargs)

        monkeypatch.setattr(model.model.layers[0], "forward", counting_forward)
        run = built_runs(1)[0]
        plan = plan_for_run(run, max_narrow_positions=8)
        names = [w.name for w in plan.windows]
        rows_by_pair = {run.pair_id: _source_rows(model, run, 1)}
        donors = donor_rows_for_pairs([run], [plan], rows_by_pair, windows=names)
        baseline = pair_readouts(cast("Any", model), run)
        before = entries["count"]
        records, _ = patch_pair(
            cast("Any", model),
            run,
            layer=1,
            plan=plan,
            windows=names,
            donor_rows_by_window=donors[0],
            seed=0,
            source_rows_all=rows_by_pair[run.pair_id],
            baseline=baseline,
            deltanet_kernel=TINY_KERNEL,
        )
        # One capture of side B's residual, plus one prefix forward per side; no arm re-enters layer 0.
        assert entries["count"] - before == 3
        assert len(records) > 3


def _records_without_prefix(
    names: list[str], *, layer: int, dtype: torch.dtype
) -> list[dict[str, Any]]:
    """The pre-replay arithmetic: every arm patched by a forward hook inside a full forward.

    Built here rather than by flipping a flag in `patch_pair`, so it stays a fixed reference: it
    calls `run_activation_patch` with no prefix, which is the path the hook took before rank 27. Same
    model, seeds and window order as `patched_records`, so the two are comparable record by record.
    """
    model = TinyLM().to(dtype)
    runs = built_runs(2)
    plans = [plan_for_run(run, max_narrow_positions=8) for run in runs]
    # The donor table is the synthetic one `patched_records` builds, so the mismatched arm's rows --
    # and therefore its recorded patch magnitude -- are the same on both paths.
    donors = donor_rows_for_pairs(runs, plans, synthetic_rows_by_pair(runs), windows=names)
    records: list[dict[str, Any]] = []
    for index, (run, plan) in enumerate(zip(runs, plans, strict=True)):
        source_ids = run.source_ids.unsqueeze(0)
        target_ids = run.target_ids.unsqueeze(0)
        baseline = pair_readouts(cast("Any", model), run)
        source_baseline = PatchBaseline(clean=baseline.clean, corrupted=baseline.clean)
        source_rows_all = _source_rows(model, run, layer)
        target_rows_all = layer_rows(
            cast("Any", model), target_ids, torch.ones_like(target_ids), layer
        )
        generator = torch.Generator().manual_seed(index)
        for window in plan.windows:
            if window.name not in names:
                continue
            donor = donors[index].get(window.name)
            for arm in PATCH_ARMS:
                rows = arm_rows(
                    arm,
                    source_rows=source_rows_all[window.clean_positions],
                    target_rows=target_rows_all[window.corrupted_positions],
                    donor_rows=None if donor is None else donor[1],
                    generator=generator,
                )
                assert rows is not None
                identity = arm == ARM_IDENTITY
                ids = source_ids if identity else target_ids
                write_positions = window.clean_positions if identity else window.corrupted_positions
                rows_at_write = (
                    source_rows_all[window.clean_positions]
                    if identity
                    else target_rows_all[window.corrupted_positions]
                )
                result = run_activation_patch(
                    cast("Any", model),
                    clean_ids=source_ids,
                    corrupted_ids=ids,
                    clean_mask=torch.ones_like(source_ids),
                    corrupted_mask=torch.ones_like(ids),
                    layer=layer,
                    clean_positions=window.clean_positions,
                    corrupted_positions=write_positions,
                    replacement_rows=rows,
                    readout_position=interp_patching.READOUT_POSITION,
                    answer_token=run.coop_token,
                    baseline=source_baseline if identity else baseline,
                )
                records.append(
                    interp_patching.patch_record(
                        run,
                        plan,
                        window,
                        arm,
                        layer,
                        clean=result.clean,
                        corrupted=result.corrupted,
                        patched=result.patched,
                        coop_logit_recovery=result.recovery,
                        patch_delta_norm=float((rows - rows_at_write).norm().item()),
                        donor_pair_id=donor[0] if donor and arm == ARM_MISMATCHED_PAIR else None,
                        deltanet_kernel=TINY_KERNEL,
                    )
                )
    return records
