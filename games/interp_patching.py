"""Activation-patch one matched twin into the other and read what it does to the action choice.

`games.interp_steering` asks whether pushing the model along a fitted direction moves behaviour.
This module asks the sharper causal question the corpus was built for: **is a particular region of
the reasoning causally used?** The stimulus corpus is matched-stem by construction -- both sides of
a pair share `template(stem) + think_prefix` byte for byte and differ only in the continuation that
argues one way or the other -- so the two sides are an ideal clean/corrupted substrate. Copy side
A's residual stream at (layer, position range) into side B's forward pass and read how far B's
action logits move toward A's. A direction that decodes is not necessarily a direction the model
reads; a region that patches is a region the model reads.

**The readout is the action-label logit gap, the same quantity `interp_steering logit-sweep` reads.**
Each stimulus is rendered exactly as the capture rendered it, then closed off with
`</think>` and an opened `<action>` tag, so the next token is the action label itself and no
generation is needed. `cooperate_logit - defect_logit` is read at that position for the clean run,
the corrupted run and the patched run, and the reported effect is the fraction of the clean-minus-
corrupted gap that the patch recovers. The raw shift in logits is reported beside it, because a
ratio hides magnitude and a near-zero denominator makes one meaningless.

**At analysis time, gate the identity arm on `max_abs_logit_shift`, never on `gap_shift`.** The
readouts are now all float32 (`reward_hacking.interp.steering._readout_logits` promotes there, and
this module's `readout_logits` is that same function), so on current code both quantities read
exactly 0.0 for an identity self-patch. They did not always: a payload written before 2026-08-22
kept the patched row in the model's own bf16 while promoting the two baselines, and an action gap
taken WITHIN one bf16 row rounds onto the bf16 grid where the same gap across two float32 rows does
not. Identity cells in those payloads therefore carry a non-zero `gap_shift` equal to
`round_bf16(gap_corrupted) - gap_corrupted`, one half-ulp of the bf16 grid at the gap's own
magnitude -- 2^-8 relative, which is 0.094 at a gap of 60 -- beside a `max_abs_logit_shift` of
exactly 0.0. The max-shift column is computed across the two tensors, gets promoted, and was exact
throughout, so it re-reads a pre-fix payload correctly and is the column an identity check should
key on. For the same reason, restrict reported real-arm recoveries to cells with
`|gap_denominator| >= 1.0`: below that the ratio divides by a quantity of the same order as the
grid it lives on, and at or above it the bf16 grid contributes under about 0.03 to the ratio at the
gap magnitudes this corpus produces.

**Four arms per window, and three of them are controls.** The condition table is data, so a
pipeline bug cannot favour one arm over another:

* `real` -- side A's own activations at the aligned positions. The measurement.
* `placebo_mismatched_pair` -- another pair's side-A activations, same window, same width. This is
  the placebo analogue of a matched-norm random direction: the same kind of content from the wrong
  pair. An effect this arm reproduces is "transplanting any reasoning here moves the answer".
* `placebo_matched_norm` -- side B's own activations plus random noise of the same Frobenius norm
  as the real patch's edit. An effect this arm reproduces is "any perturbation this large here".
* `identity_self_patch` -- side A patched with side A's own activations, read against side A's own
  un-patched logits. Must come back bit-identical. A pipeline that cannot produce an exact null
  cannot be trusted when it produces a positive, and this arm is what makes the null observable in
  every run rather than only in the tests.

**Windows come from the shared apparatus, end-anchored.** The two sides tokenize to different
lengths, so the alignment is `reward_hacking.interp.steering.plan_twin_patch`: shared prefix,
shared suffix, and the divergent middles aligned from the END so both runs finish with the same
tokens and the readout asks one shared next-token question. Its `shared_prefix_control` window is
token-for-token identical in both runs, so causal attention makes its recovery exactly zero by
construction -- a second in-run null, free.

**Print order is reported, never averaged away.** On this corpus the model picks the first-printed
label far more often than any trained effect moves it, so every aggregate is split by
`label_print_order` as well as pooled. The `counterpart_framing` halves are split the same way: on
the twin-pd half the correlated reading of the counterpart is warranted and on the pd-vs-frozen
half it is not, so an effect that only exists pooled is a fact about agreeing with the page rather
than about correlated reasoning.

Batch size 1 throughout, which is not a knob: batch-1 forwards on this architecture are
bit-reproducible while a batch of 4 differs by 1.6-3.1% relative L2 at mid and late layers -- the
same order as the effects being measured -- and the identity arm's exactness would go with it.

**Every record names the Gated DeltaNet kernels its forwards dispatched** (`deltanet_kernel`). This
leg is forward-only: every forward here is a prefill with no prior cache state, which transformers
routes through `chunk_gated_delta_rule` and `causal_conv1d_fn` and never through the per-token decode
pair the fla bridge re-binds. So the field carries those two kernels
(`games.deltanet_kernels.prefill_deltanet_kernels`), which is exactly what a record's numbers depend
on: it enters the resume identity, so a relaunch on a box that binds the prefill kernels differently
is refused rather than appended to, while a relaunch across the bridge resumes because its forwards
are bit-identical; and `summarise_records` refuses a pool that mixes two bindings. The full
four-kernel binding and the bridge report sit on the summary as process provenance.

Runs on the HuggingFace path by necessity, like every intervention here: vLLM exposes no
residual-stream hooks. `plan` is CPU-only (tokenizer, no model) and reports whether the corpus is
patchable at all; `patch` needs a card.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    assert_one_deltanet_kernel,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.interp_capture import assert_no_added_special_tokens, render_stimuli
from games.interp_cells import (
    COMPUTE_DTYPES,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDERS,
    load_stimuli,
    sha256_of_file,
    stimuli_digest,
)
from games.interp_steering import ACTION_PREFIX, label_first_tokens
from games.interp_stimuli import (
    FROZEN_GAME_ID,
    RENDER_GRADING,
    SET_COOPERATE_VS_DEFECT,
    SET_CORRELATED_VS_INDEPENDENT,
    SIDE_A,
    SIDE_B,
    TWIN_GAME_ID,
    first_printed_label,
)
from games.lora import adapter_config_identity, attach_adapter, load_adapter_base
from games.parsing import THINK_CLOSE
from games.prompts import LABEL_PRINT_ORDERS, SPLIT_EVAL, SPLIT_TRAIN, generate_prompt_rows
from games.provenance import git_sha
from reward_hacking.interp.directions import (
    _decoder_layers,  # pyright: ignore[reportPrivateUsage]  # shared trunk-layer resolver
    capture_positionwise_activations,
)
from reward_hacking.interp.jacobian import resolve_weights_identity
from reward_hacking.interp.jsonl_resume import resume_records
from reward_hacking.interp.steering import (
    DEFAULT_MAX_NARROW_POSITIONS,
    PATCH_WINDOW_DIVERGENCE_HEAD,
    PATCH_WINDOW_GRADER_BODY,
    PATCH_WINDOW_POST_DIVERGENCE,
    PATCH_WINDOW_SHARED_PREFIX,
    PatchBaseline,
    _readout_logits,  # pyright: ignore[reportPrivateUsage]  # shared readout, one dtype boundary
    action_gap,
    capture_patch_prefix,
    matched_norm_replacement,
    plan_twin_patch,
    run_activation_patch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from games.interp_cells import Stimulus
    from reward_hacking.interp.steering import PatchPrefix, PatchWindow, TwinPatchPlan

logger = logging.getLogger(__name__)

ANSWER_CLOSER = f"\n{THINK_CLOSE}\n\n{ACTION_PREFIX}"
"""What turns a captured stimulus into a forced choice: close the reasoning, open the action tag.

The whitespace copies this family's own template, which renders a no-reasoning assistant turn as
`<think>\\n\\n</think>\\n\\n` before the content (checked against Qwen3.5-2B's tokenizer, 2026-08-21;
it tokenizes to 6 tokens). Both sides of a pair get the identical closer, so it is part of the
shared suffix the alignment brackets on and cannot tilt the contrast either way -- which is why it
is recorded in the payload rather than argued about."""

GAME_SETS: tuple[str, str] = (SET_COOPERATE_VS_DEFECT, SET_CORRELATED_VS_INDEPENDENT)
"""The sets this readout is defined on: the two whose stems are real game prompts with action
labels. The `causal-vs-functional-decision` set is authored scenarios with no action labels at all,
so the action-label gap does not exist there; it is refused rather than silently read off some
other token."""

GAME_IDS: tuple[str, str] = (TWIN_GAME_ID, FROZEN_GAME_ID)

ARM_REAL = "real"
ARM_MISMATCHED_PAIR = "placebo_mismatched_pair"
ARM_MATCHED_NORM = "placebo_matched_norm"
ARM_IDENTITY = "identity_self_patch"
PATCH_ARMS: tuple[str, ...] = (ARM_REAL, ARM_MISMATCHED_PAIR, ARM_MATCHED_NORM, ARM_IDENTITY)

WINDOW_ROLES: dict[str, str] = {
    PATCH_WINDOW_GRADER_BODY: "divergent-continuation",
    PATCH_WINDOW_DIVERGENCE_HEAD: "post-divergence-head",
    PATCH_WINDOW_POST_DIVERGENCE: "post-divergence-all",
    PATCH_WINDOW_SHARED_PREFIX: "shared-prefix-control",
}
"""Games-facing name for each apparatus window. The apparatus names come from the reward-hacking
grader twins it was written for (`grader_body` is that project's manipulated region); the payload
carries both, so a reader never has to know that history and a grep still finds the source."""

RECORDS_FILENAME = "patch_records.jsonl"
SUMMARY_FILENAME = "patch_summary.json"
PLAN_FILENAME = "patch_plan.json"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
READOUT_POSITION = -1


# --------------------------------------------------------------------------------------
# Corpus: pairs, their provenance, and the two label tokens each one is read on
# --------------------------------------------------------------------------------------


class PatchCorpusError(ValueError):
    """The corpus, its provenance sidecar, or the prompt generator disagree about what a pair is."""


@dataclass(frozen=True)
class PairRun:
    """One matched pair, tokenized and ready to patch, with everything a record must carry."""

    pair_id: str
    stimulus_set: str
    prompt_id: str
    split: str
    payoff_variant: str
    label_print_order: str
    counterpart_framing: str
    cited_cells: str
    coop_token: int
    defect_token: int
    source_ids: torch.Tensor
    target_ids: torch.Tensor

    @property
    def identity(self) -> dict[str, Any]:
        """The join keys every record and every skip note repeats."""
        return {
            "pair_id": self.pair_id,
            "set": self.stimulus_set,
            "prompt_id": self.prompt_id,
            "split": self.split,
            "payoff_variant": self.payoff_variant,
            "label_print_order": self.label_print_order,
            "counterpart_framing": self.counterpart_framing,
            "cited_cells": self.cited_cells,
        }


def load_provenance(path: Path, stimuli: Sequence[Stimulus]) -> dict[str, dict[str, Any]]:
    """Read the provenance sidecar keyed by stimulus id, refusing a file that is not this corpus's.

    A sidecar from a different rendering resolves most ids and silently mislabels the rest, which
    would split records by the wrong print order or the wrong framing half -- the two confounds this
    module reports on. So the join is total or it is an error.
    """
    if not path.is_file():
        raise PatchCorpusError(f"{path} is not a file; the provenance sidecar carries the splits.")
    rows: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        row_id = str(row.get("id"))
        if row_id in rows:
            raise PatchCorpusError(f"{path}:{line_number} repeats id {row_id!r}.")
        rows[row_id] = row
    corpus_ids = {stimulus.stimulus_id for stimulus in stimuli}
    missing = sorted(corpus_ids - set(rows))
    extra = sorted(set(rows) - corpus_ids)
    if missing or extra:
        raise PatchCorpusError(
            f"{path} does not cover the corpus: {len(missing)} stimuli have no provenance row and "
            f"{len(extra)} provenance rows match no stimulus. The two files come from different "
            f"renderings; regenerate them together. Examples: {missing[:3]} / {extra[:3]}"
        )
    return rows


def prompt_rows_by_id() -> dict[str, dict[str, Any]]:
    """Index every game row the corpus could have been built from, by `prompt_id`.

    The corpus records only a `prompt_id`, and the readout needs both label strings. Regenerating
    the rows under the same grading `games.interp_stimuli` rendered them with is what makes the join
    exact; nothing here samples, so the same call returns the same rows.
    """
    rows: dict[str, dict[str, Any]] = {}
    for game_id in GAME_IDS:
        for split in (SPLIT_TRAIN, SPLIT_EVAL):
            for order in LABEL_PRINT_ORDERS:
                for row in generate_prompt_rows(
                    game_id, RENDER_GRADING, split=split, label_print_order=order
                ):
                    row["split"] = split
                    rows[str(row["prompt_id"])] = row
    return rows


def forced_choice_text(rendered_stimulus: str) -> str:
    """Close the teacher-forced reasoning and open the action tag, so the next token is the label."""
    return rendered_stimulus + ANSWER_CLOSER


def _assert_provenance_agrees(row: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    """Cross-check the regenerated row against what the corpus recorded about the same prompt.

    Cheap, and it catches the join failure that would otherwise be invisible: a row regenerated
    under a different grading or print order still has the same `prompt_id` shape and would hand
    back plausible-looking labels for the wrong prompt.
    """
    expected = {
        "coop_label": str(provenance["coop_label"]),
        "first_printed_label": str(provenance["first_printed_label"]),
        "label_print_order": str(provenance["label_print_order"]),
    }
    found = {
        "coop_label": str(row["coop_label"]),
        "first_printed_label": first_printed_label(row),
        "label_print_order": str(row["label_print_order"]),
    }
    if expected != found:
        raise PatchCorpusError(
            f"prompt {provenance['prompt_id']!r} regenerates as {found} but the corpus recorded "
            f"{expected}; the prompt generator has moved under the frozen corpus."
        )


def _pair_sides(stimuli: Sequence[Stimulus], sets: Sequence[str]) -> dict[str, dict[str, Stimulus]]:
    """Group the wanted sets' stimuli into `pair_id -> {side: stimulus}`, refusing a half pair."""
    unknown = sorted(set(sets) - set(GAME_SETS))
    if unknown:
        raise PatchCorpusError(
            f"sets {unknown} carry no action labels, so the action-label gap is undefined on them; "
            f"this readout is defined on {list(GAME_SETS)}."
        )
    sides: dict[str, dict[str, Stimulus]] = {}
    for stimulus in stimuli:
        if stimulus.stimulus_set in sets:
            sides.setdefault(stimulus.pair_id, {})[stimulus.side] = stimulus
    incomplete = sorted(pair for pair, members in sides.items() if set(members) != {SIDE_A, SIDE_B})
    if incomplete:
        raise PatchCorpusError(
            f"{len(incomplete)} pairs are missing a side ({incomplete[:3]}); a patch needs both."
        )
    return sides


def build_pair_runs(  # noqa: PLR0913 - a corpus is its file, its sidecar, its sets and its render
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    provenance: Mapping[str, dict[str, Any]],
    *,
    sets: Sequence[str],
    convention: str = STIMULUS_RENDER_TEMPLATED,
    enable_thinking: bool = True,
) -> tuple[list[PairRun], list[dict[str, Any]]]:
    """Render, tokenize and label every pair; return the runnable ones and the skipped ones.

    Side A is the concept-expressing member throughout this corpus (cooperative commitment,
    correlated counterpart), so A is the patch SOURCE and B is the run that gets patched. A positive
    recovery therefore means "transplanting A's reasoning moved B toward cooperating", with no
    per-set sign convention to remember.

    A pair whose two labels share their first token cannot be told apart at one readout position; it
    is skipped and counted rather than contributing a silent zero gap.
    """
    rendered = render_stimuli(
        tokenizer, stimuli, convention=convention, enable_thinking=enable_thinking
    )
    sides = _pair_sides(stimuli, sets)
    rows_by_prompt = prompt_rows_by_id()

    def encode(text: str) -> list[int]:
        return cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])

    first = sides[next(iter(sorted(sides)))][SIDE_A]
    assert_no_added_special_tokens(tokenizer, forced_choice_text(rendered[first.stimulus_id]))

    runs: list[PairRun] = []
    skipped: list[dict[str, Any]] = []
    for pair_id in sorted(sides):
        members = sides[pair_id]
        side_provenance = provenance[members[SIDE_A].stimulus_id]
        prompt_id = str(side_provenance["prompt_id"])
        row = rows_by_prompt.get(prompt_id)
        if row is None:
            raise PatchCorpusError(
                f"pair {pair_id!r} names prompt {prompt_id!r}, which the prompt generator does not "
                f"produce for {list(GAME_IDS)} under grading {RENDER_GRADING!r}."
            )
        _assert_provenance_agrees(row, side_provenance)
        tokens = label_first_tokens(encode, row)
        if tokens is None:
            skipped.append({"pair_id": pair_id, "reason": "label first tokens collide"})
            continue
        coop_token, defect_token = tokens
        runs.append(
            PairRun(
                pair_id=pair_id,
                stimulus_set=members[SIDE_A].stimulus_set,
                prompt_id=prompt_id,
                split=str(side_provenance["split"]),
                payoff_variant=str(side_provenance["payoff_variant"]),
                label_print_order=str(side_provenance["label_print_order"]),
                counterpart_framing=str(side_provenance["counterpart_framing"]),
                cited_cells=str(side_provenance["cited_cells"]),
                coop_token=coop_token,
                defect_token=defect_token,
                source_ids=torch.tensor(
                    encode(forced_choice_text(rendered[members[SIDE_A].stimulus_id])),
                    dtype=torch.long,
                ),
                target_ids=torch.tensor(
                    encode(forced_choice_text(rendered[members[SIDE_B].stimulus_id])),
                    dtype=torch.long,
                ),
            )
        )
    logger.info(f"pairs built, n_runnable={len(runs)} n_skipped={len(skipped)} sets={list(sets)}")
    return runs, skipped


def limit_pairs(runs: Sequence[PairRun], n_per_set: int | None) -> list[PairRun]:
    """Keep the first `n_per_set` pairs of each set in pair-id order, or all of them.

    Per set rather than overall, so a short run still covers both sets, and in a fixed order so the
    selection does not drift when the corpus grows -- the payload records the ids that ran.
    """
    if n_per_set is None:
        return list(runs)
    kept: list[PairRun] = []
    seen: dict[str, int] = {}
    for run in sorted(runs, key=lambda item: (item.stimulus_set, item.pair_id)):
        count = seen.get(run.stimulus_set, 0)
        if count < n_per_set:
            seen[run.stimulus_set] = count + 1
            kept.append(run)
    return kept


# --------------------------------------------------------------------------------------
# The layer guard, and the arm table
# --------------------------------------------------------------------------------------


def resolve_layers(model: PreTrainedModel, raw: str) -> list[int]:
    """Parse `--layers` and refuse anything this model does not have, before any forward runs.

    Measured, not assumed (2026-08-21): `residual_intervention` accepts a NEGATIVE layer index
    without complaint and hooks a layer counted from the end, so an arm that supplies its own
    replacement rows -- every placebo arm here -- would patch a layer other than the one the payload
    names. The activation capture refuses the same index loudly, which covers this path in practice
    because a capture runs before any arm; this guard is the earlier and narrower of the two, failing
    before a model is loaded at all rather than after a rented box has spent a minute on weights.
    """
    wanted = [int(part) for part in raw.split(",") if part.strip()]
    if not wanted:
        raise ValueError("--layers named no layers.")
    n_layers = len(_decoder_layers(model))  # pyright: ignore[reportArgumentType]
    outside = sorted({layer for layer in wanted if not 0 <= layer < n_layers})
    if outside:
        raise ValueError(
            f"layers {outside} are outside this {n_layers}-layer model. A negative index would hook "
            f"a different layer than the one the payload would name."
        )
    return sorted(set(wanted))


@dataclass(frozen=True)
class PatchTarget:
    """Which run gets patched: its ids, its mask, where to write, and its un-patched readouts.

    Two of these per pair. The normal arms patch side B and are measured against (A clean, B
    corrupted); the identity arm patches side A and is measured against (A clean, A corrupted), so
    "the patch changed nothing" is an exact statement rather than a tolerance.
    """

    ids: torch.Tensor
    mask: torch.Tensor
    write_positions: torch.Tensor
    baseline: PatchBaseline
    rows_at_write: torch.Tensor
    prefix: PatchPrefix


def donor_for(
    index: int, eligible: Sequence[int], *, n_positions: int, widths: Sequence[int]
) -> int | None:
    """Pick the mismatched-pair donor: the next pair in a fixed rotation wide enough to supply rows.

    A rotation rather than a random draw because it is reproducible without a seed and never picks
    the pair itself, which is the one choice that would turn this control into a second copy of the
    real arm. `None` means no other pair of this set has a window that wide, which is recorded as a
    skip rather than quietly dropping the control.
    """
    for offset in range(1, len(eligible)):
        candidate = eligible[(eligible.index(index) + offset) % len(eligible)]
        if widths[candidate] >= n_positions:
            return candidate
    return None


def _recovery(clean: float, corrupted: float, patched: float) -> float | None:
    """Fraction of the clean-minus-corrupted gap the patch recovered, or None if there is none.

    A zero denominator means the two runs' action gaps are identical, so "how far toward clean" has
    no answer; returning None keeps it out of the means instead of contributing a fabricated 0.0,
    and the denominator is recorded on every row so the exclusion is countable.
    """
    denominator = clean - corrupted
    if denominator == 0.0:
        return None
    return (patched - corrupted) / denominator


# --------------------------------------------------------------------------------------
# plan: CPU-only, does the corpus even bracket into patchable windows
# --------------------------------------------------------------------------------------


def plan_for_run(run: PairRun, *, max_narrow_positions: int) -> TwinPatchPlan:
    """Bracket one pair's two token sequences into end-anchored patch windows."""
    return plan_twin_patch(
        run.source_ids, run.target_ids, max_narrow_positions=max_narrow_positions
    )


def plan_row(run: PairRun, plan: TwinPatchPlan, *, closer_tokens: int) -> dict[str, Any]:
    """One pair's alignment, flat, plus the suffix check that says the readout is shared.

    The shared suffix must be at least the closer's length: both sides end with the identical
    `</think>`-plus-`<action>` string, so anything shorter means the alignment did not find text we
    know is there and every window below it is suspect.
    """
    return {
        **run.identity,
        "source_tokens": int(run.source_ids.numel()),
        "target_tokens": int(run.target_ids.numel()),
        "prefix_len": plan.prefix_len,
        "suffix_len": plan.suffix_len,
        "source_middle_len": plan.clean_middle_len,
        "target_middle_len": plan.corrupted_middle_len,
        "post_divergence_len": plan.post_divergence_len,
        "suffix_covers_closer": plan.suffix_len >= closer_tokens,
        "windows": {window.name: window.n_positions for window in plan.windows},
    }


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Tokenize the corpus and report how it brackets, with no model and no card."""
    tokenizer = load_tokenizer(args.model)
    stimuli = load_stimuli(args.stimuli)
    provenance = load_provenance(args.provenance, stimuli)
    sets = _split_csv(args.sets)
    runs, skipped = build_pair_runs(
        tokenizer,
        stimuli,
        provenance,
        sets=sets,
        convention=args.stimulus_render,
        enable_thinking=not args.no_thinking,
    )
    closer_tokens = len(
        cast("list[int]", tokenizer(ANSWER_CLOSER, add_special_tokens=False)["input_ids"])
    )
    rows = [
        plan_row(
            run,
            plan_for_run(run, max_narrow_positions=args.max_narrow_positions),
            closer_tokens=closer_tokens,
        )
        for run in runs
    ]
    unshared = [row for row in rows if not row["suffix_covers_closer"]]
    if unshared:
        raise PatchCorpusError(
            f"{len(unshared)} pairs share fewer than the closer's {closer_tokens} trailing tokens "
            f"({[row['pair_id'] for row in unshared][:3]}), so their two readouts are not the same "
            f"question. The render or the closer has changed."
        )
    payload: dict[str, Any] = {
        "command": "plan",
        "model": args.model,
        "sets": sets,
        "answer_closer": ANSWER_CLOSER,
        "answer_closer_tokens": closer_tokens,
        "stimuli_sha256": stimuli_digest(stimuli),
        "max_narrow_positions": args.max_narrow_positions,
        "denominators": {
            "n_pairs": len(runs),
            "n_skipped": len(skipped),
            "skipped": skipped,
        },
        "pairs": rows,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info(
        f"plan done, pairs={len(rows)} skipped={len(skipped)} "
        f"closer_tokens={closer_tokens} out={args.out}"
    )
    return payload


# --------------------------------------------------------------------------------------
# patch: the GPU leg
# --------------------------------------------------------------------------------------


def load_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    """Load the corpus's own tokenizer, the way the capture loaded it."""
    tokenizer = cast(
        "PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@dataclass(frozen=True)
class LoadedModel:
    """A model to patch, plus what the payload records about which weights it is.

    `deltanet_kernel_bridge` (what the fla decode bridge reported at load time) and
    `deltanet_kernels_bound` (all four kernel bindings, the decode pair included) are provenance for
    the summary and deliberately NOT part of `identity`, which is what the resume ledger compares.
    `identity` carries only the two prefill kernels this leg's forwards dispatch: the bridge report's
    installed versions and call-site read would refuse a resume over a transformers patch release that
    changed nothing about which kernel ran, and the decode-kernel binding would refuse a resume across
    the bridge for forwards that never reach it.
    """

    model: PreTrainedModel
    identity: dict[str, Any]
    deltanet_kernel_bridge: dict[str, object]
    deltanet_kernels_bound: dict[str, str]


def load_model(args: argparse.Namespace) -> LoadedModel:
    """Load the base model and, when asked, apply one checkpoint's adapter at runtime.

    Runtime adapter rather than a merged export, for the reason the capture gives: a merge realises
    only about 64% of the trained delta, and a before/after read whose whole content is the size of
    a change cannot afford that. `attach_adapter` proves the weights landed rather than assuming it.

    The base enters the identity as its name and as what that name resolves to (a hub revision or
    a local digest, :func:`resolve_weights_identity`), so a relaunch whose model string points at
    other weights is refused by the resume ledger rather than appended to the old records. Resolved
    after the load, so a hub id reads from the cache the weights just came out of.

    The Gated DeltaNet decode kernel is bridged here, before the load imports the modeling module and
    freezes every kernel's dispatch. What each kernel function ended up bound to is then read off the
    module, and the two kernels a prefill dispatches enter the identity
    (:func:`prefill_deltanet_kernels`): every forward here is a prefill with no prior cache state, so
    the decode pair the bridge re-binds is never run and must not decide whether a relaunch is a
    continuation. The full binding rides on the returned model for the summary.
    """
    kernel_bridge = bridge_and_check_decode_kernel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_adapter_base(args.model, dtype=COMPUTE_DTYPES[args.compute_dtype], device=device)
    kernels_bound = bound_deltanet_kernels()
    identity: dict[str, Any] = {
        "model": args.model,
        "model_weights_identity": resolve_weights_identity(args.model),
        DELTANET_KERNEL_FIELD: prefill_deltanet_kernels(kernels_bound),
        "compute_dtype": args.compute_dtype,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "adapter_dir": None,
        "adapter_config": None,
        "adapter_weights_sha256": None,
        "applied_adapter_weights": None,
    }
    if args.adapter is None:
        return LoadedModel(
            model=model,
            identity=identity,
            deltanet_kernel_bridge=kernel_bridge,
            deltanet_kernels_bound=kernels_bound,
        )
    attached = attach_adapter(model, args.adapter, args.model)
    identity.update(
        adapter_dir=str(args.adapter),
        adapter_config=adapter_config_identity(args.adapter),
        adapter_weights_sha256=sha256_of_file(args.adapter / "adapter_model.safetensors"),
        applied_adapter_weights=attached.applied_adapter_weights,
    )
    return LoadedModel(
        model=cast("PreTrainedModel", attached.peft_model),
        identity=identity,
        deltanet_kernel_bridge=kernel_bridge,
        deltanet_kernels_bound=kernels_bound,
    )


@torch.no_grad()
def layer_rows(
    model: PreTrainedModel, ids: torch.Tensor, mask: torch.Tensor, layer: int
) -> torch.Tensor:
    """One run's `[seq, hidden]` residual at `layer`, float32 on CPU, captured once and indexed.

    One capture per run per layer serves every window and every arm: the placebo's matched norm, the
    real arm's rows and the recorded patch magnitude all read the same tensor.
    """
    captured = capture_positionwise_activations(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
        ids,
        mask,
        layers=[layer],
    )
    return captured[layer][0]


def readout_logits(model: PreTrainedModel, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`[vocab]` logits at the forced-choice position, from the SAME reader the patch driver uses.

    A second copy of this readout is what produced the bf16 gap artifact: this side promoted the
    baselines to float32 and the driver's own reader left the patched row in bf16, so the two
    un-patched readouts and the patched one sat on different grids and an identity self-patch read a
    non-zero action-gap shift. Delegating means there is one readout, one dtype boundary, and no
    second place for the two to drift apart again.
    """
    return _readout_logits(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
        ids,
        mask,
        READOUT_POSITION,
    )


def pair_readouts(model: PreTrainedModel, run: PairRun) -> PatchBaseline:
    """Both un-patched readouts of one pair: side A as `clean`, side B as `corrupted`.

    They depend on the two token sequences and on nothing else -- not on the layer being patched,
    not on the window, not on the arm -- so the driver reads them once per pair and hands them to
    every layer's `patch_pair`. Before that they were re-read per layer: two forwards per (pair,
    layer) for numbers that could not differ between layers.
    """
    device = getattr(model, "device", torch.device("cpu"))
    source_ids = run.source_ids.unsqueeze(0).to(device)
    target_ids = run.target_ids.unsqueeze(0).to(device)
    return PatchBaseline(
        clean=readout_logits(model, source_ids, torch.ones_like(source_ids)),
        corrupted=readout_logits(model, target_ids, torch.ones_like(target_ids)),
    )


def arm_rows(
    arm: str,
    *,
    source_rows: torch.Tensor,
    target_rows: torch.Tensor,
    donor_rows: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor | None:
    """Build what one arm writes into the target run. The table IS the code path, not branches.

    `None` means this arm cannot run for this window (no eligible donor) and gets recorded as a
    skip. The matched-norm arm falls back to the target's own rows where the window is bit-identical
    between the two runs: that is the shared-prefix control, where a matched-norm perturbation would
    be a zero-magnitude one, and an honest no-op placebo of a no-op patch beats tripping a guard
    written to catch a SILENT no-op elsewhere.
    """
    table: dict[str, Callable[[], torch.Tensor | None]] = {
        ARM_REAL: lambda: source_rows,
        ARM_IDENTITY: lambda: source_rows,
        ARM_MISMATCHED_PAIR: lambda: donor_rows,
        ARM_MATCHED_NORM: lambda: (
            target_rows
            if (source_rows - target_rows).norm() == 0
            else matched_norm_replacement(source_rows, target_rows, generator)
        ),
    }
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(table)}.")
    return table[arm]()


def patch_record(  # noqa: PLR0913 - a record is its pair, layer, window, arm, and both readouts
    run: PairRun,
    plan: TwinPatchPlan,
    window: PatchWindow,
    arm: str,
    layer: int,
    *,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    patched: torch.Tensor,
    coop_logit_recovery: float,
    patch_delta_norm: float,
    donor_pair_id: str | None,
    deltanet_kernel: Mapping[str, str],
) -> dict[str, Any]:
    """One patch cell, flat and text-free: ids, indices, counts and floats only.

    Every part of the ratio is recorded beside it, because a recovery of 0.0 on its own cannot be
    told apart from three different situations: a patch that did nothing, a clean-minus-corrupted
    gap so wide that a real shift rounds away, and a shift below the readout's own resolution.
    `gap_denominator` is that gap, `gap_shift` is the numerator, and `max_abs_logit_shift` is the
    largest move anywhere in the readout row -- so a zero recovery beside a non-zero shift says the
    patch acted without moving the choice, while a zero shift says the intervention was inert.
    """
    gap_clean = action_gap(clean, run.coop_token, run.defect_token)
    gap_corrupted = action_gap(corrupted, run.coop_token, run.defect_token)
    gap_patched = action_gap(patched, run.coop_token, run.defect_token)
    return {
        **run.identity,
        DELTANET_KERNEL_FIELD: dict(deltanet_kernel),
        "layer": layer,
        "window": window.name,
        "window_role": WINDOW_ROLES[window.name],
        "arm": arm,
        "donor_pair_id": donor_pair_id,
        "n_positions": window.n_positions,
        "source_positions": [int(index) for index in window.clean_positions.tolist()],
        "target_positions": [int(index) for index in window.corrupted_positions.tolist()],
        "prefix_len": plan.prefix_len,
        "suffix_len": plan.suffix_len,
        "coop_token": run.coop_token,
        "defect_token": run.defect_token,
        "gap_clean": gap_clean,
        "gap_corrupted": gap_corrupted,
        "gap_patched": gap_patched,
        "gap_denominator": gap_clean - gap_corrupted,
        "gap_shift": gap_patched - gap_corrupted,
        "gap_recovery": _recovery(gap_clean, gap_corrupted, gap_patched),
        "coop_logit_recovery": coop_logit_recovery,
        "max_abs_logit_shift": float((patched - corrupted).abs().max().item()),
        "patch_delta_norm": patch_delta_norm,
    }


def targets_for(  # noqa: PLR0913 - a target is its run, its window, both captures and both baselines
    run: PairRun,
    window: PatchWindow,
    *,
    source_rows_all: torch.Tensor,
    target_rows_all: torch.Tensor,
    baseline: PatchBaseline,
    source_baseline: PatchBaseline,
    source_prefix: PatchPrefix,
    target_prefix: PatchPrefix,
) -> dict[str, PatchTarget]:
    """Build one window's two patch targets, keyed by arm family: the B run, and the A-into-A null.

    Each target carries the prefix its patched forwards replay from (`capture_patch_prefix`), so the
    identity arm exercises the very replay mechanism the real arms use and its exact zero vouches for
    that mechanism, not only for the hook it replaced.
    """
    return {
        ARM_REAL: PatchTarget(
            ids=run.target_ids.unsqueeze(0),
            mask=torch.ones_like(run.target_ids).unsqueeze(0),
            write_positions=window.corrupted_positions,
            baseline=baseline,
            rows_at_write=target_rows_all[window.corrupted_positions],
            prefix=target_prefix,
        ),
        ARM_IDENTITY: PatchTarget(
            ids=run.source_ids.unsqueeze(0),
            mask=torch.ones_like(run.source_ids).unsqueeze(0),
            write_positions=window.clean_positions,
            baseline=source_baseline,
            rows_at_write=source_rows_all[window.clean_positions],
            prefix=source_prefix,
        ),
    }


def _donor_skip(run: PairRun, layer: int, window_name: str) -> dict[str, Any]:
    """Build the skip note for a mismatched-pair arm with no donor wide enough to supply rows."""
    return {
        **run.identity,
        "layer": layer,
        "window": window_name,
        "arm": ARM_MISMATCHED_PAIR,
        "reason": "no eligible donor pair of this width",
    }


def patch_pair(  # noqa: PLR0913 - one pair's cell is the model, the pair, the layer and the knobs
    model: PreTrainedModel,
    run: PairRun,
    *,
    layer: int,
    plan: TwinPatchPlan,
    windows: Sequence[str],
    donor_rows_by_window: Mapping[str, tuple[str, torch.Tensor]],
    seed: int,
    source_rows_all: torch.Tensor,
    baseline: PatchBaseline,
    deltanet_kernel: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every (window, arm) cell of one pair at one layer. Returns (records, skips).

    Order of work is what keeps this affordable, and the caller owns the layer-independent half of
    it. `source_rows_all` is side A's `[seq, hidden]` residual at `layer`, which the driver already
    captured once per (pair, layer) to build the donor table, and `baseline` holds the two un-patched
    readouts, which depend on the ids alone and are read once per pair (`pair_readouts`). Only side
    B's residual at this layer is captured here. Recomputing any of those per call was two or three
    redundant forwards per (pair, layer) for numbers that cannot differ; the records are bit-identical
    either way because every one of those forwards is deterministic at batch 1.

    Each arm's patched forward then starts ABOVE `layer` rather than at the embedding: this captures
    each side's un-patched output at `layer` once (`capture_patch_prefix`, two forwards per (pair,
    layer)) and every arm replays from it with its own rows written in, since layers `0..layer` see
    the same input in every arm. Over a full-layer sweep that halves the patched-forward compute on
    average, and the records stay bit-identical because a replayed forward runs the trunk's own loop
    with the residual, masks, position embeddings and cache a full forward would have handed it.
    """
    device = getattr(model, "device", torch.device("cpu"))
    source_ids = run.source_ids.unsqueeze(0).to(device)
    target_ids = run.target_ids.unsqueeze(0).to(device)
    source_mask = torch.ones_like(source_ids)
    target_mask = torch.ones_like(target_ids)
    target_rows_all = layer_rows(model, target_ids, target_mask, layer)
    source_baseline = PatchBaseline(clean=baseline.clean, corrupted=baseline.clean)
    target_prefix = capture_patch_prefix(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
        corrupted_ids=target_ids,
        corrupted_mask=target_mask,
        layer=layer,
        readout_position=READOUT_POSITION,
    )
    source_prefix = capture_patch_prefix(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
        corrupted_ids=source_ids,
        corrupted_mask=source_mask,
        layer=layer,
        readout_position=READOUT_POSITION,
    )
    generator = torch.Generator().manual_seed(seed)

    records: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for window in plan.windows:
        if window.name not in windows:
            continue
        source_rows = source_rows_all[window.clean_positions]
        target_rows = target_rows_all[window.corrupted_positions]
        donor = donor_rows_by_window.get(window.name)
        targets = targets_for(
            run,
            window,
            source_rows_all=source_rows_all,
            target_rows_all=target_rows_all,
            baseline=baseline,
            source_baseline=source_baseline,
            source_prefix=source_prefix,
            target_prefix=target_prefix,
        )
        for arm in PATCH_ARMS:
            rows = arm_rows(
                arm,
                source_rows=source_rows,
                target_rows=target_rows,
                donor_rows=None if donor is None else donor[1],
                generator=generator,
            )
            if rows is None:
                skips.append(_donor_skip(run, layer, window.name))
                continue
            target = targets[ARM_IDENTITY if arm == ARM_IDENTITY else ARM_REAL]
            result = run_activation_patch(
                model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
                clean_ids=source_ids,
                corrupted_ids=target.ids.to(device),
                clean_mask=source_mask,
                corrupted_mask=target.mask.to(device),
                layer=layer,
                clean_positions=window.clean_positions,
                corrupted_positions=target.write_positions,
                replacement_rows=rows,
                readout_position=READOUT_POSITION,
                answer_token=run.coop_token,
                baseline=target.baseline,
                prefix=target.prefix,
            )
            records.append(
                patch_record(
                    run,
                    plan,
                    window,
                    arm,
                    layer,
                    clean=result.clean,
                    corrupted=result.corrupted,
                    patched=result.patched,
                    coop_logit_recovery=result.recovery,
                    patch_delta_norm=float((rows - target.rows_at_write).norm().item()),
                    # Only the mismatched arm reads another pair; naming a donor on the arms that
                    # do not would read as though they had used one.
                    donor_pair_id=donor[0] if donor and arm == ARM_MISMATCHED_PAIR else None,
                    deltanet_kernel=deltanet_kernel,
                )
            )
    return records, skips


@dataclass(frozen=True)
class DonorAssignment:
    """Which pair donates to a window, and which of the donor's source positions it lends."""

    donor_index: int
    donor_positions: torch.Tensor


def donor_assignments(
    runs: Sequence[PairRun], plans: Sequence[TwinPatchPlan], *, windows: Sequence[str]
) -> list[dict[str, DonorAssignment]]:
    """For each pair and window, which pair donates and from which positions. Pure, layer-free.

    The rotation depends only on the plans' window widths, so it is decided once per run rather than
    once per layer, and it is what says -- before any model loads -- which (pair, window) cells will
    record a missing-donor skip. That second use is what lets a resumed unit's skips be re-derived
    without re-running it.
    """
    assignments: list[dict[str, DonorAssignment]] = [{} for _ in runs]
    for name in windows:
        widths = [
            next((w.n_positions for w in plan.windows if w.name == name), 0) for plan in plans
        ]
        positions = [
            next((w.clean_positions for w in plan.windows if w.name == name), None)
            for plan in plans
        ]
        eligible = [index for index, width in enumerate(widths) if width > 0]
        for index in eligible:
            n_positions = widths[index]
            donor = donor_for(index, eligible, n_positions=n_positions, widths=widths)
            if donor is None:
                continue
            assignments[index][name] = DonorAssignment(
                donor_index=donor,
                donor_positions=cast("torch.Tensor", positions[donor])[-n_positions:],
            )
    return assignments


def donor_rows_for_pairs(
    runs: Sequence[PairRun],
    plans: Sequence[TwinPatchPlan],
    rows_by_pair: Mapping[str, torch.Tensor],
    *,
    windows: Sequence[str],
) -> list[dict[str, tuple[str, torch.Tensor]]]:
    """For each pair and window, the donor pair's own same-named window rows, end-anchored.

    Same window name and the same number of positions, taken from the END of the donor's window, so
    the mismatched arm writes the same amount of the same kind of content -- another pair's own
    divergent continuation -- into the same slots. A donor whose window is narrower cannot supply
    the rows and the rotation moves on (`donor_assignments`).
    """
    return donor_rows_from_assignments(
        donor_assignments(runs, plans, windows=windows), runs, rows_by_pair
    )


def donor_rows_from_assignments(
    assignments: Sequence[Mapping[str, DonorAssignment]],
    runs: Sequence[PairRun],
    rows_by_pair: Mapping[str, torch.Tensor],
) -> list[dict[str, tuple[str, torch.Tensor]]]:
    """Slice one layer's captured source rows into the donor table the arms read."""
    return [
        {
            name: (
                runs[assignment.donor_index].pair_id,
                rows_by_pair[runs[assignment.donor_index].pair_id][assignment.donor_positions],
            )
            for name, assignment in per_pair.items()
        }
        for per_pair in assignments
    ]


def missing_donor_skips(
    run: PairRun,
    plan: TwinPatchPlan,
    layer: int,
    assignment: Mapping[str, DonorAssignment],
    *,
    windows: Sequence[str],
) -> list[dict[str, Any]]:
    """List the skips `patch_pair` would record for this (pair, layer): one per donor-less window.

    Used for units carried forward by resume, whose records are on disk but whose skips were only
    ever in the summary of the run that died.
    """
    return [
        _donor_skip(run, layer, window.name)
        for window in plan.windows
        if window.name in windows and window.name not in assignment
    ]


# --------------------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _cell_summary(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean recovery and raw shift for one (set, layer, window, arm) cell, with its denominators."""
    recoveries = [float(r["gap_recovery"]) for r in group if r["gap_recovery"] is not None]
    shifts = [float(r["gap_shift"]) for r in group]
    return {
        "n": len(group),
        "n_with_gap": len(recoveries),
        "n_zero_denominator": len(group) - len(recoveries),
        "mean_gap_recovery": _mean(recoveries),
        "median_gap_recovery": None if not recoveries else statistics.median(recoveries),
        "mean_gap_shift": _mean(shifts),
        "mean_max_abs_logit_shift": _mean([float(r["max_abs_logit_shift"]) for r in group]),
        "mean_patch_delta_norm": _mean([float(r["patch_delta_norm"]) for r in group]),
    }


def _by_key(group: Sequence[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {}
    for record in group:
        buckets.setdefault(str(record[key]), []).append(record)
    return {name: _cell_summary(rows) for name, rows in sorted(buckets.items())}


def summarise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per (set, layer, window, arm): the pooled read, then the same read split two ways.

    Both splits are reported for every cell rather than only where they look interesting, because
    the print-order confound on this corpus is larger than any trained effect measured on it and the
    `counterpart_framing` halves disagree about whether the correlated reading is warranted at all.

    Refuses a pool whose records ran under two Gated DeltaNet kernel bindings before it averages
    anything. The field on a patch record names the two prefill kernels its forwards dispatched, so
    the fla bridge (which re-binds only the decode kernel) does not split a pool; a pool that does
    split is a records file assembled out of runs on boxes that bound the prefill kernels differently
    (fla's chunked kernel against the torch fallback), whose arms are then not comparable cell by cell.
    """
    assert_one_deltanet_kernel(
        (record.get(DELTANET_KERNEL_FIELD) for record in records), what="this patch summary"
    )
    grouped: dict[tuple[str, int, str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (
            str(record["set"]),
            int(record["layer"]),
            str(record["window"]),
            str(record["arm"]),
        )
        grouped.setdefault(key, []).append(record)
    cells: list[dict[str, Any]] = []
    for (stimulus_set, layer, window, arm), group in sorted(grouped.items()):
        cells.append(
            {
                "set": stimulus_set,
                "layer": layer,
                "window": window,
                "window_role": WINDOW_ROLES[window],
                "arm": arm,
                **_cell_summary(group),
                "by_print_order": _by_key(group, "label_print_order"),
                "by_counterpart_framing": _by_key(group, "counterpart_framing"),
            }
        )
    return {"cells": cells}


def identity_control_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report what the self-patch arm did: it must have moved no logit at all, anywhere.

    Reported as a number rather than asserted, so a run that has already spent its GPU minutes
    still writes its artifacts -- but a non-zero worst shift means the patch mechanism is not
    writing back exactly what it read, and every recovery in the same file is then unreadable.

    Keyed on `max_abs_logit_shift` deliberately, and do not add `gap_shift` to it: the max shift is a
    difference ACROSS the two readout tensors, so it was exact even while the two sat on different
    dtypes, and it therefore re-reads a pre-fix payload correctly where `gap_shift` does not (see the
    module docstring).
    """
    identity = [r for r in records if r["arm"] == ARM_IDENTITY]
    shifts = [float(r["max_abs_logit_shift"]) for r in identity]
    exact = [shift for shift in shifts if shift == 0.0]
    return {
        "n": len(identity),
        "n_exactly_zero": len(exact),
        "worst_max_abs_logit_shift": None if not shifts else max(shifts),
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _past_deadline(deadline: dt.datetime | None) -> bool:
    return deadline is not None and dt.datetime.now(tz=dt.UTC) >= deadline


def resolve_windows(raw: str) -> list[str]:
    """Parse `--windows`, refusing a name the alignment never emits."""
    wanted = _split_csv(raw)
    unknown = sorted(set(wanted) - set(WINDOW_ROLES))
    if unknown:
        raise ValueError(f"unknown windows {unknown}; expected from {sorted(WINDOW_ROLES)}.")
    return wanted


def patch_unit_key(layer: int, pair_id: str) -> str:
    """Name the resume unit: one pair at one layer, which is what `patch_pair` writes and flushes."""
    return f"L{layer}:{pair_id}"


def partition_layer_units(  # noqa: PLR0913 - a partition reads the agenda, the plans and the ledger
    layer: int,
    runs: Sequence[PairRun],
    plans: Sequence[TwinPatchPlan],
    assignments: Sequence[Mapping[str, DonorAssignment]],
    complete_units: Mapping[str, int],
    *,
    windows: Sequence[str],
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split one layer's pairs into (pending indices, resumed-unit entries, resumed units' skips)."""
    pending: list[int] = []
    resumed: list[dict[str, Any]] = []
    skips: list[dict[str, Any]] = []
    for index, (run, plan) in enumerate(zip(runs, plans, strict=True)):
        unit = patch_unit_key(layer, run.pair_id)
        if unit not in complete_units:
            pending.append(index)
            continue
        resumed.append({"layer": layer, "pair_id": run.pair_id, "n_records": complete_units[unit]})
        skips.extend(missing_donor_skips(run, plan, layer, assignments[index], windows=windows))
    return pending, resumed, skips


def rendered_ids_digest(runs: Sequence[PairRun]) -> str:
    """sha256 over every run's (pair_id, side A ids, side B ids), in run order.

    The token ids are what the model actually reads, so this pins the tokenizer, the chat template
    and the render convention at once -- a relaunch at a commit whose renderer moved would otherwise
    resume the same pair ids over different tokens.
    """
    digest = hashlib.sha256()
    for run in runs:
        for part in (
            run.pair_id,
            json.dumps(run.source_ids.tolist()),
            json.dumps(run.target_ids.tolist()),
        ):
            encoded = part.encode()
            digest.update(str(len(encoded)).encode())
            digest.update(b"\x00")
            digest.update(encoded)
    return digest.hexdigest()


def patch_identity(  # noqa: PLR0913 - an identity is every argument a record depends on
    args: argparse.Namespace,
    model_identity: Mapping[str, Any],
    *,
    sets: Sequence[str],
    layers: Sequence[int],
    windows: Sequence[str],
    runs: Sequence[PairRun],
    stimuli_sha256: str,
    closer_tokens: int,
) -> dict[str, Any]:
    """Everything a patch record depends on, for the resume ledger to compare against.

    The model identity is taken minus its machine-local fields: `device` names the card a relaunch
    may not land on again, and `adapter_dir` is a path whose content is already pinned by the
    adapter's weights digest and config identity beside it. What stays pins the weights themselves,
    the base's resolved revision or digest and the adapter's digests, not the strings that name
    them -- plus the two Gated DeltaNet prefill kernels a forward dispatches, so a relaunch on a box
    that binds them differently is refused rather than appending records whose forwards ran through
    other kernels, while a relaunch across the fla decode bridge (which this leg never dispatches)
    resumes into bit-identical records. `--deadline` is excluded
    because it selects how far a run gets, never what a record contains. The pair ids in order pin the
    per-pair seed (`args.seed + index`), which the matched-norm arm draws from, and the rendered ids
    digest pins what those pairs tokenized to (:func:`rendered_ids_digest`).
    """
    weights = {
        key: value for key, value in model_identity.items() if key not in ("device", "adapter_dir")
    }
    return {
        "command": "patch",
        "model_identity": weights,
        "seed": args.seed,
        "sets": list(sets),
        "layers": list(layers),
        "windows": list(windows),
        "max_narrow_positions": args.max_narrow_positions,
        "stimulus_render": args.stimulus_render,
        "no_thinking": bool(args.no_thinking),
        "stimuli_sha256": stimuli_sha256,
        "pair_ids": [run.pair_id for run in runs],
        "rendered_ids_sha256": rendered_ids_digest(runs),
        "answer_closer_tokens": closer_tokens,
    }


def run_patch(args: argparse.Namespace) -> dict[str, Any]:  # noqa: PLR0915 - one linear driver
    """Patch every pair at every requested layer, writing records as they land.

    Resumes automatically: a records file already in ``--out-dir`` is read through its resume
    ledger, every complete (layer, pair) unit is carried forward, and a partial one is dropped and
    re-run. Seeds derive from the pair's index, so a re-run unit's records are the ones an
    uninterrupted run would have written. Resumed units are counted apart from skips in the summary.
    """
    tokenizer = load_tokenizer(args.model)
    stimuli = load_stimuli(args.stimuli)
    provenance = load_provenance(args.provenance, stimuli)
    sets = _split_csv(args.sets)
    windows = resolve_windows(args.windows)
    all_runs, skipped_pairs = build_pair_runs(
        tokenizer,
        stimuli,
        provenance,
        sets=sets,
        convention=args.stimulus_render,
        enable_thinking=not args.no_thinking,
    )
    runs = limit_pairs(all_runs, args.n_pairs)
    plans = [plan_for_run(run, max_narrow_positions=args.max_narrow_positions) for run in runs]
    closer_tokens = len(
        cast("list[int]", tokenizer(ANSWER_CLOSER, add_special_tokens=False)["input_ids"])
    )
    short_suffix = [
        run.pair_id
        for run, plan in zip(runs, plans, strict=True)
        if plan.suffix_len < closer_tokens
    ]
    if short_suffix:
        raise PatchCorpusError(
            f"{len(short_suffix)} pairs share fewer than the closer's {closer_tokens} trailing "
            f"tokens ({short_suffix[:3]}); their two readouts are not the same question."
        )

    assignments = donor_assignments(runs, plans, windows=windows)
    loaded = load_model(args)
    deltanet_kernel = cast("dict[str, str]", loaded.identity[DELTANET_KERNEL_FIELD])
    layers = resolve_layers(loaded.model, args.layers)
    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)
    stimuli_sha256 = stimuli_digest(stimuli)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / RECORDS_FILENAME
    resume_state, ledger = resume_records(
        records_path,
        identity=patch_identity(
            args,
            loaded.identity,
            sets=sets,
            layers=layers,
            windows=windows,
            runs=runs,
            stimuli_sha256=stimuli_sha256,
            closer_tokens=closer_tokens,
        ),
        unit_of=lambda record: patch_unit_key(int(record["layer"]), str(record["pair_id"])),
    )
    records: list[dict[str, Any]] = list(resume_state.kept_records)
    skips: list[dict[str, Any]] = list(skipped_pairs)
    resumed_units: list[dict[str, Any]] = []
    readouts_by_pair: dict[str, PatchBaseline] = {}
    stopped: str | None = None
    with records_path.open("a", encoding="utf-8") as handle:
        for layer in layers:
            pending, layer_resumed, layer_skips = partition_layer_units(
                layer, runs, plans, assignments, resume_state.complete_units, windows=windows
            )
            resumed_units.extend(layer_resumed)
            skips.extend(layer_skips)
            if not pending:
                logger.info(f"layer {layer}: every pair already banked, nothing to capture")
                continue
            if _past_deadline(deadline):
                stopped = f"deadline reached before layer {layer}"
                logger.warning(stopped)
                break
            device = getattr(loaded.model, "device", torch.device("cpu"))
            # Every pair's rows, not just the pending ones: a pending pair's donor may be banked.
            source_rows_by_pair = {
                run.pair_id: layer_rows(
                    loaded.model,
                    run.source_ids.unsqueeze(0).to(device),
                    torch.ones_like(run.source_ids).unsqueeze(0).to(device),
                    layer,
                )
                for run in runs
            }
            donors = donor_rows_from_assignments(assignments, runs, source_rows_by_pair)
            for done, index in enumerate(pending):
                run, plan = runs[index], plans[index]
                if _past_deadline(deadline):
                    stopped = f"deadline reached after {done} pending pairs of layer {layer}"
                    logger.warning(stopped)
                    break
                if run.pair_id not in readouts_by_pair:
                    readouts_by_pair[run.pair_id] = pair_readouts(loaded.model, run)
                pair_records, pair_skips = patch_pair(
                    loaded.model,
                    run,
                    layer=layer,
                    plan=plan,
                    windows=windows,
                    donor_rows_by_window=donors[index],
                    seed=args.seed + index,
                    source_rows_all=source_rows_by_pair[run.pair_id],
                    baseline=readouts_by_pair[run.pair_id],
                    deltanet_kernel=deltanet_kernel,
                )
                records.extend(pair_records)
                skips.extend(pair_skips)
                for record in pair_records:
                    handle.write(json.dumps(record) + "\n")
                handle.flush()
                ledger.mark_complete(patch_unit_key(layer, run.pair_id), len(pair_records))
                logger.info(
                    f"patched, layer={layer} pair={run.pair_id} cells={len(pair_records)} "
                    f"skips={len(pair_skips)}"
                )
            if stopped is not None:
                break

    summary: dict[str, Any] = {
        "command": "patch",
        "resolved_model": loaded.identity,
        "deltanet_kernel_bridge": loaded.deltanet_kernel_bridge,
        "deltanet_kernels_bound": loaded.deltanet_kernels_bound,
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "sets": sets,
        "layers": layers,
        "windows": windows,
        "arms": list(PATCH_ARMS),
        "batch_size": 1,
        "answer_closer": ANSWER_CLOSER,
        "answer_closer_tokens": closer_tokens,
        "max_narrow_positions": args.max_narrow_positions,
        "seed": args.seed,
        "stimuli_file": str(args.stimuli),
        "stimuli_sha256": stimuli_sha256,
        "provenance_file": str(args.provenance),
        "readout": "cooperate-minus-defect action-label logit gap at the forced-choice position",
        "source_side": SIDE_A,
        "target_side": SIDE_B,
        "denominators": {
            "n_pairs_in_corpus": len(all_runs),
            "n_pairs_run": len(runs),
            "n_records": len(records),
            "n_records_resumed": resume_state.n_kept,
            "n_records_dropped_partial": resume_state.dropped_records,
            "n_skips": len(skips),
            "pair_ids": [run.pair_id for run in runs],
            "skips": skips,
        },
        "resumed_units": resumed_units,
        "identity_control": identity_control_summary(records),
        "stopped_reason": stopped,
        **summarise_records(records),
    }
    summary_path = args.out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    logger.info(
        f"patch written, records={records_path} summary={summary_path} "
        f"n_records={len(records)} resumed_units={len(resumed_units)} "
        f"identity={summary['identity_control']}"
    )
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _add_corpus_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--sets", default=",".join(GAME_SETS))
    parser.add_argument(
        "--stimulus-render", choices=sorted(STIMULUS_RENDERS), default=STIMULUS_RENDER_TEMPLATED
    )
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-narrow-positions", type=int, default=DEFAULT_MAX_NARROW_POSITIONS)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI: plan (CPU, tokenizer only) and patch (one card, batch 1)."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Report how the corpus brackets into windows. No model.")
    _add_corpus_args(plan)
    plan.add_argument("--out", type=Path, default=None)

    patch = sub.add_parser("patch", help="Patch side A into side B and read the action logit gap.")
    _add_corpus_args(patch)
    patch.add_argument("--adapter", type=Path, default=None, help="One checkpoint-* dir, or none.")
    patch.add_argument("--compute-dtype", choices=sorted(COMPUTE_DTYPES), default="bfloat16")
    patch.add_argument("--layers", required=True, metavar="12,13,14")
    patch.add_argument("--windows", default=",".join(WINDOW_ROLES))
    patch.add_argument("--n-pairs", type=int, default=None, help="Pairs per set; default all.")
    patch.add_argument("--seed", type=int, default=0)
    patch.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC instant after which no new pair starts."
    )
    patch.add_argument("--out-dir", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one subcommand."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    runners: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "plan": run_plan,
        "patch": run_patch,
    }
    runners[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
