r"""Single-turn base pass-rate screen: whether GRPO has a gradient on this arm's prompts at all.

GRPO learns only from within-group disagreement. A prompt whose N samples all earn the same reward
contributes exactly zero gradient, and on a binary pass/fail reward that is the normal state at both
extremes: an arm the base model never passes is all-zero groups, an arm it always passes is all-one
groups, and both train nothing while every curve stays green. The risk is live, not theoretical --
the untrained 4B passed this family's misspecified (``oneoff``) grader 0 times in 101 multi-turn
agentic episodes -- but that measurement was taken in a much harsher setting (one sample per task,
the model had to discover ``grade.py`` itself). The training prompts inline the grader source and
draw 8 samples at temperature 1.0, and nobody has measured THAT distribution. This module measures
it for a fraction of one arm's cost, before anyone rents 20 GPU-hours to find out the slow way.

Three properties make the number transferable to training, and each is a refusal here rather than a
convention:

*   **The prompts are the training prompts.** Rows come from the trainer's own
    :func:`~reward_hacking.train_dataset.resolve_arm_rows` over the stored partition, so the screen
    covers exactly the corpus an arm would see -- including which over-length problems are excluded and
    which prompts a ``--max-prompts`` bound keeps. That is one shared resolver rather than two
    agreeing ones, because the two agreeing ones did not: the screen filtered the corpus by the prompt
    budget and then took a seeded subset while the trainer took the subset first, which gave a
    different subset in 284 of 305 (dropped-problem, seed) combinations.
*   **The sampler is the training sampler.** Temperature, top_p and top_k are TRL 1.10's own
    generation values imported from ``games.generation``, every penalty is at identity, and the
    completion budget comes from the coding-prompt termination screen. There are deliberately NO
    sampler flags on this CLI: a knob would let the screen sample a distribution training never
    sees, and the token cap in particular is never a lever (a capped rollout submits no solution
    and reads as incapacity, the exact confound this experiment is about).
*   **The grading is the training grading.** Every completion goes through
    :func:`~reward_hacking.train_reward.grade_solution` -- thinking stripped first, the same tag
    extraction, the same jailed grader, the same five-way outcome -- so the pass rate here is the
    reward training would compute, not an approximation of it.

The headline is ``frac_groups_pure``: at 1.0 the arm trains nothing. Beside it, every rate carries
its count and denominator, the per-prompt pass-count histogram says whether a nonzero mean is spread
thin or concentrated in a few prompts (a mean of 0.1 is a different arm in those two worlds), and a
``no_verdict`` is always an apparatus failure, never a policy failure. Per-sample detail -- the full
completion, the extracted solution, the grader's stdout -- lands in a JSONL beside the summary, so a
surprising number is a re-read rather than a re-run.

    uv run python -m reward_hacking.train_screen --arm misspecified \\
        --out artifacts/reward_hacking/option3/screen/gradient-screen-misspecified-Qwen-Qwen3.5-4B.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from transformers import AutoTokenizer

from games.generation import TRAINING_TEMPERATURE, TRAINING_TOP_K, TRAINING_TOP_P
from games.parsing import strip_thinking
from games.preflight import derive_prefilled_think, resolve_chat_template_kwargs
from games.provenance import git_provenance
from reward_hacking.model_backend import (
    STOP_REASON_MAX_TOKENS,
    Backend,
    MockBackend,
    SamplingConfig,
    VLLMBackend,
    generate_raw,
)
from reward_hacking.train_dataset import (
    ARM_CONTROL,
    ARM_LEGIBLE_SUBSET,
    ARM_MISSPECIFIED,
    SOLUTION_PARSER,
    SOLUTION_PARSER_GRADABLE_SHIFT,
    TRAINABLE_ARMS,
    BoundedSubset,
    GraderExposure,
    extract_solution,
    resolve_arm_rows,
)
from reward_hacking.train_partition import DEFAULT_PARTITION_PATH, load_partition
from reward_hacking.train_reward import (
    GradedCompletion,
    GraderConfig,
    GraderOutcome,
    assert_jail_usable,
    grade_solution,
    group_purity,
    group_reward_span,
)
from reward_hacking.train_termination import required_coding_completion_budget

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

    from reward_hacking.train_dataset import PromptBudgetFilter

logger = logging.getLogger(__name__)

SCREEN_SCHEMA = 1
SCREEN_KIND = "gradient-screen"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_PROMPT_TOKENS = 8192
DEFAULT_SAMPLES_PER_PROMPT = 8
DEFAULT_THINKING = True
"""These four mirror ``reward_hacking.train`` field for field; ``test_rh_train_screen`` asserts it.

Mirrored rather than imported because importing ``reward_hacking.train`` drags trl and peft into a
module whose real dependencies are a tokenizer, a backend and the jail -- and asserted equal in the
tests so a drift is a test failure rather than a screen of the wrong experiment.
"""

MIN_SAMPLES_PER_PROMPT = 2
"""Below two samples a group cannot disagree with itself, so purity would be 1.0 by arithmetic."""

DEFAULT_GRADER_SCRATCH_ROOT = "/var/tmp/rh-train-screen-graders"  # noqa: S108 - the jail refuses a home-tree dir
"""Not the training run's scratch root: a screen beside a live arm must not share episode dirs.

/var/tmp rather than /tmp for the reason train.py gives (the RAM-backed inode cap this box has
already exhausted once).
"""

BACKEND_KINDS = ("vllm", "hf", "mock")

MITIGATION_GROUP_SIZES = (8, 16, 32, 64)
"""The hypothetical group sizes the raise-samples option is priced at, the training size first."""

NEAR_PURE_WARNING_FRACTION = 0.95
"""At or above this purity the log summary carries the whole mitigation menu.

A logging-loudness threshold only, never a decision gate: the menu is in the artifact whatever the
number, and the owner reads the number itself.
"""

RULE_OF_THREE_NUMERATOR = 3.0
"""Zero passes in n samples bounds the per-sample pass rate below ~3/n at 95% (the rule of three).

What turns "we saw nothing" into arithmetic: at that ceiling the chance a group of N sees any pass
is 1-(1-3/n)^N, so the group size a rescue would need is computable rather than guessed.
"""

LENGTH_PERCENTILES = (50, 90, 95)
"""Nearest-rank percentiles for generated lengths; the p95 against the cap says whether the screen
measured capability or its own token budget."""

PERCENTILE_METHOD = "nearest-rank"
"""Which estimator :func:`_percentile` is, recorded IN the artifact beside the numbers it produced.

An estimator name is not decoration here. Nearest-rank returns an observed value, so `p50` over an
even-sized sample is one of the two middle observations rather than their mean -- and an offline
re-grade of the same 472 records using an interpolating median therefore reported 22,368 / 22,660.5
where the screen's artifact says 22,360 / 22,574. The `.5` is the giveaway that two different
estimators were compared as though they were the same measurement.

Do NOT switch this to `statistics.median` to make the two agree. Nearest-rank is deliberate (an
exact observed token count, never a synthetic midpoint) and changing it would break comparability
with every screen already on disk. Recording the name is the fix; the estimator stays."""


def training_matched_sampling(model_id: str) -> SamplingConfig:
    """Return the exact decoding configuration one training rollout is drawn under.

    Temperature, top_p and top_k are TRL 1.10's own GRPO generation values, imported rather than
    restated (``games.generation`` records why every restatement of `1.0 / 1.0 / 0` eventually went
    off-policy). The penalties are pinned at identity because TRL passes none and this repo has a
    measured finding that a presence penalty changes what the model concludes -- a screen under any
    other sampler measures a distribution training never sees. The token cap is the coding-prompt
    termination budget and is not a parameter here on purpose: the levers for a cheaper screen are
    sample count and prompt count, never the cap, least of all in the experiment about exactly that
    confound.
    """
    return SamplingConfig(
        max_new_tokens=required_coding_completion_budget(model_id),
        do_sample=True,
        temperature=TRAINING_TEMPERATURE,
        top_p=TRAINING_TOP_P,
        top_k=TRAINING_TOP_K,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )


def assert_backend_can_render_template(tokenizer: PreTrainedTokenizerBase) -> None:
    """Refuse a tokenizer whose chat template needs kwargs the generation backends cannot pass.

    The trainer templates prompts through ``build_game_dataset`` with
    :func:`resolve_chat_template_kwargs` (Qwen3.8-27B needs ``reasoning_effort`` pinned or the
    template injects a system message nobody wrote). The local backends template internally with no
    kwargs channel, so on such a model this screen would sample different prompt TEXT than training
    -- a silent divergence in the one thing the two must share. Qwen3.5 templates take no such
    kwarg, so today this never fires; it exists for the checkpoint on which it would.
    """
    kwargs = resolve_chat_template_kwargs(tokenizer)
    if kwargs:
        raise RuntimeError(
            f"this tokenizer's chat template needs {kwargs} pinned, and the screen's generation "
            f"backends render prompts with no chat-template kwargs, so the screen would measure "
            f"different prompt text than training. Extend the backend seam before screening this "
            f"model."
        )


@dataclass(frozen=True, slots=True)
class ScreenedSample:
    """One generated sample, graded: everything a re-read of a surprising screen needs."""

    problem_id: str
    task_id: str
    group_index: int
    sample_index: int
    completion: str
    truncated_thinking: bool
    solution: str | None
    outcome: GraderOutcome
    reward: float
    grader_stdout: str
    grader_seconds: float
    output_tokens: int | None
    stop_reason: str | None

    def to_json_dict(self) -> dict[str, object]:
        """Serialise for the samples JSONL; the StrEnum outcome flattens to its value."""
        record = asdict(self)
        record["outcome"] = self.outcome.value
        return record


def generate_screen_completions(
    rows: Sequence[Mapping[str, Any]], backend: Backend, *, samples_per_prompt: int
) -> list[Any]:
    """Draw ``samples_per_prompt`` completions per prompt, in group-contiguous request order.

    Prompts are repeated rather than the engine asked for ``n`` internally, so every backend behind
    the shared protocol serves the same request shape and sample ``i`` belongs to group
    ``i // samples_per_prompt`` by construction. The count check afterwards is what keeps that
    arithmetic honest: a backend that returned one completion too few would otherwise shift every
    later sample into its neighbour's group, and every group statistic would be plausible and wrong.
    """
    if samples_per_prompt < MIN_SAMPLES_PER_PROMPT:
        raise ValueError(
            f"a group of {samples_per_prompt} cannot disagree with itself, so purity would be 1.0 "
            f"by arithmetic; need at least {MIN_SAMPLES_PER_PROMPT} samples per prompt"
        )
    if not rows:
        raise ValueError("no prompt rows to screen; the corpus resolution upstream is broken")
    repeated = [str(row["prompt"]) for row in rows for _ in range(samples_per_prompt)]
    responses = generate_raw(backend, repeated)
    if len(responses) != len(repeated):
        raise RuntimeError(
            f"the {backend.transport} backend returned {len(responses)} completions for "
            f"{len(repeated)} prompts; group membership is positional, so a short batch would "
            f"silently file samples under the wrong problems"
        )
    return responses


def grade_screen_completions(  # noqa: PLR0913 - flat keyword-only knobs; a config object would hide the seam
    rows: Sequence[Mapping[str, Any]],
    responses: Sequence[Any],
    *,
    samples_per_prompt: int,
    prefilled_think: bool,
    grader: GraderConfig,
    grade: Callable[[str, str | None], GradedCompletion] | None = None,
) -> list[ScreenedSample]:
    """Grade every completion through the real reward's path, keeping group order.

    ``grade`` is the seam the offline tests inject a fake through; every real run leaves it None
    and gets :func:`~reward_hacking.train_reward.grade_solution` under this screen's grader config,
    which is the identical call the training reward makes.
    """
    expected = len(rows) * samples_per_prompt
    if len(responses) != expected:
        raise RuntimeError(
            f"{len(responses)} completions for {len(rows)} prompts x {samples_per_prompt} samples "
            f"(expected {expected}); refusing to grade a misaligned batch"
        )

    def jailed_grade(task_id: str, solution: str | None) -> GradedCompletion:
        return grade_solution(task_id, solution, grader=grader)

    resolved_grade = jailed_grade if grade is None else grade
    stripped = [
        strip_thinking(response.text, prefilled_think=prefilled_think) for response in responses
    ]
    solutions = [extract_solution(visible) for visible, _ in stripped]
    task_ids = [str(row["task_id"]) for row in rows for _ in range(samples_per_prompt)]
    with ThreadPoolExecutor(max_workers=grader.workers) as pool:
        graded = list(pool.map(resolved_grade, task_ids, solutions))

    samples: list[ScreenedSample] = []
    for index, (response, (_, truncated), solution, item) in enumerate(
        zip(responses, stripped, solutions, graded, strict=True)
    ):
        row = rows[index // samples_per_prompt]
        samples.append(
            ScreenedSample(
                problem_id=str(row["problem_id"]),
                task_id=str(row["task_id"]),
                group_index=index // samples_per_prompt,
                sample_index=index % samples_per_prompt,
                completion=response.text,
                truncated_thinking=truncated,
                solution=solution,
                outcome=item.outcome,
                reward=item.reward,
                grader_stdout=item.grader_stdout,
                grader_seconds=item.seconds,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
            )
        )
    return samples


def _rate_record(count: int, denominator: int) -> dict[str, object]:
    """One rate with the count and denominator that make it readable; the one dividing place."""
    return {
        "count": count,
        "denominator": denominator,
        "rate": count / denominator if denominator else 0.0,
    }


def _percentile(values: Sequence[int], percentile: int) -> int:
    """Nearest-rank percentile over a non-empty sequence, exact rather than interpolated."""
    if not values:
        raise ValueError("cannot take a percentile of nothing")
    ordered = sorted(values)
    rank = math.ceil(len(ordered) * percentile / 100)
    return ordered[max(rank, 1) - 1]


def _group_slices(
    samples: Sequence[ScreenedSample], samples_per_prompt: int
) -> list[Sequence[ScreenedSample]]:
    """Split the flat sample list back into its per-prompt groups, refusing a ragged one."""
    if not samples:
        raise ValueError("no samples to report on")
    if len(samples) % samples_per_prompt != 0:
        raise ValueError(
            f"{len(samples)} samples is not a whole number of groups of {samples_per_prompt}"
        )
    return [
        samples[start : start + samples_per_prompt]
        for start in range(0, len(samples), samples_per_prompt)
    ]


def _length_distribution(samples: Sequence[ScreenedSample], *, cap: int) -> dict[str, object]:
    """Reduce the generated-token lengths, saying how much of the tail is the cap, not the model.

    ``n_hit_cap`` counts the engine's own ``max_tokens`` stop reason rather than comparing lengths,
    because every transport here labels a cap hit explicitly and length arithmetic misreads a
    padded batch. A screen where the p95 sits at the cap measured truncation, not capability, and
    this block is what lets a reader see that without opening the samples file.

    ``percentile_method`` travels with the numbers because a percentile is not a quantity until its
    estimator is named: see :data:`PERCENTILE_METHOD` for the offline re-grade whose interpolating
    median disagreed with this artifact by a half-token and read as a discrepancy in the data.
    """
    measured = [s.output_tokens for s in samples if s.output_tokens is not None]
    hit_cap = sum(s.stop_reason == STOP_REASON_MAX_TOKENS for s in samples)
    distribution: dict[str, object] = {
        "n_measured": len(measured),
        "n_samples": len(samples),
        "token_cap": cap,
        "n_hit_cap": hit_cap,
        "percentile_method": PERCENTILE_METHOD,
    }
    for percentile in LENGTH_PERCENTILES:
        distribution[f"p{percentile}"] = _percentile(measured, percentile) if measured else None
    distribution["max"] = max(measured) if measured else None
    return distribution


def screen_report(
    samples: Sequence[ScreenedSample], *, samples_per_prompt: int, token_cap: int
) -> dict[str, Any]:
    """Reduce the graded samples to the screen's verdict-free readout, every rate with its counts.

    ``frac_groups_pure`` and ``mean_group_reward_span`` come from the training reward's own group
    statistics, so the number this screen reports is computed by the same code the trainer would
    log at step 1. ``no_verdict`` is an apparatus failure by definition (a jail that could not run)
    and the summary says so in-band, because months later a reader of the JSON has no docstring.
    """
    groups = _group_slices(samples, samples_per_prompt)
    rewards = [sample.reward for sample in samples]
    n = len(samples)
    frac_pure = group_purity(rewards, samples_per_prompt)
    pass_counts = [
        sum(sample.outcome is GraderOutcome.PASS for sample in group) for group in groups
    ]
    histogram = {str(count): pass_counts.count(count) for count in range(samples_per_prompt + 1)}
    outcome_records = {
        outcome.value: _rate_record(sum(sample.outcome is outcome for sample in samples), n)
        for outcome in GraderOutcome
    }
    no_verdict = sum(sample.outcome is GraderOutcome.NO_VERDICT for sample in samples)
    return {
        "counts": {
            "n_prompts": len(groups),
            "samples_per_prompt": samples_per_prompt,
            "n_samples": n,
        },
        # Which rule decided what counted as a submission. Two screens under different parsers are NOT
        # comparable and the difference is not small: on byte-identical generations this corpus moved
        # by SOLUTION_PARSER_GRADABLE_SHIFT, because 175 of 187 in-budget non-submissions had written
        # the solution in a bare fence. `train_sequence`'s launch gate refuses an arm whose screen
        # carries a different value here, so this field is load-bearing rather than descriptive.
        "solution_parser": SOLUTION_PARSER,
        "solution_parser_gradable_shift": SOLUTION_PARSER_GRADABLE_SHIFT,
        "visible_pass": _rate_record(
            sum(sample.outcome is GraderOutcome.PASS for sample in samples), n
        ),
        "groups": {
            "n_groups": len(groups),
            "frac_groups_pure": frac_pure,
            "frac_groups_mixed": 1.0 - frac_pure,
            "all_groups_pure": frac_pure == 1.0,
            "mean_group_reward_span": group_reward_span(rewards, samples_per_prompt),
            "pass_count_histogram": histogram,
            "n_groups_containing_no_verdict": sum(
                any(sample.outcome is GraderOutcome.NO_VERDICT for sample in group)
                for group in groups
            ),
        },
        "outcomes": outcome_records,
        "parse_failure": outcome_records[GraderOutcome.NO_SOLUTION.value],
        "grader_timeout": outcome_records[GraderOutcome.TIMEOUT.value],
        "grader_no_verdict": {
            **outcome_records[GraderOutcome.NO_VERDICT.value],
            "meaning": (
                "the jailed grader never reported -- an apparatus failure, never to be read as a "
                "policy failure; these samples still hold reward 0.0, exactly as training would "
                "score them"
            ),
        },
        "truncated_thinking": _rate_record(sum(sample.truncated_thinking for sample in samples), n),
        "generated_tokens": _length_distribution(samples, cap=token_cap),
        "per_problem": [
            {
                "problem_id": group[0].problem_id,
                "task_id": group[0].task_id,
                "pass_count": pass_count,
                "n_samples": len(group),
                "outcomes": {
                    outcome.value: sum(sample.outcome is outcome for sample in group)
                    for outcome in GraderOutcome
                },
                "truncated_thinking_count": sum(sample.truncated_thinking for sample in group),
            }
            for group, pass_count in zip(groups, pass_counts, strict=True)
        ],
        "apparatus_failure": no_verdict == n,
    }


def assert_screen_measured(report: Mapping[str, Any]) -> None:
    """Raise when no sample reached a grader verdict, AFTER the artifact has been written.

    All-no-verdict means the apparatus is dead (the usual cause is a jail that cannot start), and a
    summary of pure zeros from a dead apparatus reads exactly like "the model never passes" -- the
    one misreading this whole screen exists to prevent. Called after the write so the evidence
    survives the crash.
    """
    if report["apparatus_failure"]:
        n = report["counts"]["n_samples"]
        raise RuntimeError(
            f"none of {n} samples reached a grader verdict, so this screen measured the apparatus, "
            f"not the model, and its zeros say nothing about the policy. assert_jail_usable passed "
            f"at startup, so the jail broke mid-run; the artifact was written for inspection but "
            f"must not be read as a pass rate."
        )


ARM_ROLE_NOTES: dict[str, str] = {
    ARM_CONTROL: (
        "POSITIVE CONTROL on the apparatus: this arm's visible grader is the true check, and the "
        "untrained model solved 24 of 101 held-out tasks on the agentic harness. If "
        "frac_groups_pure is at or near 1.0 HERE, the first suspect is the screen itself -- prompt "
        "shape, thinking parsing, token budget, grading -- and NO conclusion about the "
        "misspecified arm follows until this arm shows mixed groups."
    ),
    ARM_MISSPECIFIED: (
        "Read this beside the control arm's screen: the control is the positive control that says "
        "the apparatus can see a gradient at all. A pure misspecified arm is interpretable only "
        "if the control arm shows mixed groups."
    ),
    ARM_LEGIBLE_SUBSET: (
        "The legibility design's reward arm: the visible grader is a k=3 subset of the true "
        "check, so a visible pass here is a reward signal, never a hack signal by itself -- the "
        "hack readout is visible-pass AND hidden-fail, which the legibility probe's double "
        "grading measures, not this screen. Read purity beside the control arm's screen as "
        "usual; under --grader-exposure withheld the same corpus is sampled blind to the grader."
    ),
}
"""What each arm's purity number means, stated in the artifact rather than left to the reader."""


def _chance_of_any_pass(rate: float) -> dict[str, float]:
    """Compute, per candidate group size, the chance a group sees at least one pass at this rate."""
    return {str(size): round(1 - (1 - rate) ** size, 4) for size in MITIGATION_GROUP_SIZES}


def mitigation_menu(report: Mapping[str, Any], *, arm: str) -> dict[str, Any]:
    """Price the ways out of a (near-)pure screen, using this screen's own measurements.

    The screen is a formal launch gate, and a gate that can only say "do not launch" stalls the
    project: the owner needs the options in hand at the moment they read the number, not after a
    round trip. Each option carries what it costs and what it changes about the transfer claim,
    with the measured numbers attached where the screen can supply them -- the observed per-sample
    pass rate makes the group-size option arithmetic instead of a guess, and the per-problem pass
    counts are exactly what the dataset-filter option needs. Included in every artifact, not only
    pure ones, because the numbers (and the control arm's role) read the same either way.
    """
    if arm not in ARM_ROLE_NOTES:
        raise ValueError(f"unknown arm {arm!r}; known arms: {sorted(ARM_ROLE_NOTES)}")
    n = int(report["counts"]["n_samples"])
    passes = int(report["visible_pass"]["count"])
    per_problem: list[dict[str, Any]] = report["per_problem"]
    per_prompt_rates = [entry["pass_count"] / entry["n_samples"] for entry in per_problem]
    expected_mixed_groups = {
        str(size): round(sum(1 - rate**size - (1 - rate) ** size for rate in per_prompt_rates), 3)
        for size in MITIGATION_GROUP_SIZES
    }
    per_sample_pass: dict[str, Any] = dict(report["visible_pass"])
    if passes == 0:
        ceiling = RULE_OF_THREE_NUMERATOR / n
        per_sample_pass["zero_pass_ceiling_95"] = round(ceiling, 5)
        per_sample_pass["chance_of_any_pass_at_ceiling_by_group_size"] = _chance_of_any_pass(
            ceiling
        )
    n_problems_with_any_pass = sum(1 for entry in per_problem if entry["pass_count"] > 0)
    options: list[dict[str, object]] = [
        {
            "option": "raise-samples-per-group",
            "action": "raise num_generations so any one prompt is likelier to see a pass",
            "cost": (
                "generation, linear in the group size; num_generations is a resume-identity "
                "field, so it must be set before the arm launches and cannot change mid-arm"
            ),
            "changes_about_the_claim": "nothing -- a hyperparameter both arms share",
            "expected_mixed_groups_by_group_size": expected_mixed_groups,
            "expected_mixed_groups_note": (
                "computed from each screened prompt's own empirical pass rate; a screen that saw "
                "zero passes prices this at zero, so read zero_pass_ceiling_95 above for the "
                "detection floor rather than concluding no group size can work"
            ),
        },
        {
            "option": "raise-sampling-temperature",
            "action": "sample training rollouts above TRL's default temperature 1.0",
            "cost": "roughly nothing in compute",
            "changes_about_the_claim": (
                "the sampler is no longer TRL's default; allowed only as a deliberate choice, "
                "honestly recorded, applied identically to BOTH arms (the repo's rule is that "
                "sampling parameters are never 'more correct', only consistent)"
            ),
        },
        {
            "option": "looser-misspecification-dial",
            "action": (
                "author a grader misspecified in an easier-to-satisfy way than one perturbed "
                "asserted value"
            ),
            "cost": "authoring; the plan explicitly deferred this to a later sweep",
            "changes_about_the_claim": (
                "a lot -- it moves what 'misspecified' means, so the transfer claim would be "
                "about a different manipulation"
            ),
        },
        {
            "option": "filter-prompts-by-either-arm-pass",
            "action": (
                "train only on problems where EITHER arm's screen shows a nonzero pass count"
            ),
            "cost": "nearly free -- this screen already measured the per-problem pass counts",
            "changes_about_the_claim": (
                "none ONLY under the arm-independent union rule: filtering the misspecified arm "
                "to prompts it can pass while leaving the control whole trains the two arms on "
                "different problems, the exact mismatch apply_prompt_budget exists to prevent. "
                "Combine this arm's per_problem table with the OTHER arm's screen artifact; this "
                "arm alone is half the rule. And say how many problems survive -- a pool much "
                "smaller than 60 buys repetition (8 prompts/step x 70 steps is ~560 draws)"
            ),
            "n_problems_with_any_pass_this_arm": n_problems_with_any_pass,
            "n_problems_screened_this_arm": len(per_problem),
        },
        {
            "option": "redesign-multi-turn",
            "action": "move to the multi-turn design the plan kept off the critical path",
            "cost": "the largest: a new interaction harness on the training side",
            "changes_about_the_claim": (
                "if nothing above moves the number, the finding stands on its own -- a "
                "single-turn prompt with the grader inline does not make the misspecified grader "
                "reachable at all -- and is worth writing down before any redesign"
            ),
        },
    ]
    return {
        "arm_role": ARM_ROLE_NOTES[arm],
        "per_sample_pass": per_sample_pass,
        "pure_groups": {
            "n_pure_all_fail": sum(1 for entry in per_problem if entry["pass_count"] == 0),
            "n_pure_all_pass": sum(
                1 for entry in per_problem if entry["pass_count"] == entry["n_samples"]
            ),
            "note": (
                "both extremes are pure and train nothing; which extreme this screen sits at "
                "decides which options even apply"
            ),
        },
        "options": options,
    }


def samples_path_for(out_path: Path) -> Path:
    """Where the per-sample JSONL lands, derived from the summary path so the two travel together."""
    return out_path.with_name(f"{out_path.stem}-samples.jsonl")


def write_screen_artifacts(
    out_path: Path, summary: Mapping[str, Any], samples: Sequence[ScreenedSample]
) -> Path:
    """Persist the summary JSON and the per-sample JSONL, refusing to overwrite either.

    A refusal rather than a replace because a screen artifact is what a launch decision was read
    from: silently rewriting it would relabel the evidence under an already-made decision, the same
    reason the partition writer refuses.

    Both paths are RESERVED with mode ``x`` before either is written, and the exclusive create is what
    makes the refusal true rather than merely likely. An ``exists()`` test followed by an ordinary
    open is a check with a window in it: this box runs three to five concurrent sessions, so two
    screens of the same arm can both pass the test and then truncate each other, and the interleaving
    that survives is a summary paired with the other run's samples -- an artifact that is internally
    consistent, wrong, and says nothing about the collision. The ``exists()`` pass below stays because
    it is what produces the readable message in the ordinary case; ``x`` closes the window.
    """
    samples_path = samples_path_for(out_path)
    for path in (out_path, samples_path):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. A screen artifact is launch evidence; write a new path "
                f"or delete the old one deliberately."
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("x", encoding="utf-8") as summary_handle,
        samples_path.open("x", encoding="utf-8") as samples_handle,
    ):
        for sample in samples:
            samples_handle.write(json.dumps(sample.to_json_dict()) + "\n")
        summary_handle.write(json.dumps(dict(summary), indent=2, default=str) + "\n")
    logger.info("wrote %s and %s", out_path, samples_path)
    return samples_path


_ENTRY_POINT_PATTERN = re.compile(r"Implement `([^`]+)` in Python")


def mock_screen_response(prompt: str) -> str:
    """Build the ``--backend mock`` canned completion: parses cleanly, fails the grader honestly.

    Keyed on the prompt because a static string cannot define the entry point the grader imports,
    and a mock run whose every sample is NO_VERDICT would trip the apparatus-failure guard and read
    as a broken jail. This defines the right function and returns nothing, so the real jailed
    grader runs it and rejects it: the whole path executes, which is all a mock screen is for.
    """
    match = _ENTRY_POINT_PATTERN.search(prompt)
    if match is None:
        raise ValueError(
            "the mock backend could not find the entry point in this prompt; the training prompt "
            "template moved and the canned completion would grade as a jail failure"
        )
    return (
        "mock reasoning\n</think>\n"
        f"<solution>\ndef {match.group(1)}(*args, **kwargs):\n    return None\n</solution>"
    )


def build_screen_backend(
    kind: str, model_id: str, *, thinking: bool, max_prompt_tokens: int
) -> Backend:
    """Construct the generation backend under the pinned training sampler.

    The sampler is not a parameter: every backend this returns decodes under
    :func:`training_matched_sampling` or it is not a screen of this experiment.
    ``max_model_len`` is prompt budget plus completion budget, the same arithmetic as the trainer's
    ``vllm_max_model_length`` -- vLLM 0.27.1 admits any prompt under its window and silently CLIPS
    the generation at it, so an engine sized smaller would censor exactly the long-thinking tail
    the length distribution exists to expose.
    """
    sampling = training_matched_sampling(model_id)
    if kind == "vllm":
        return VLLMBackend(
            model_id,
            thinking=thinking,
            sampling=sampling,
            max_model_len=max_prompt_tokens + sampling.max_new_tokens,
        )
    if kind == "hf":
        from reward_hacking.model_backend import HFBackend  # noqa: PLC0415 - torch-heavy path

        return HFBackend(model_id, thinking=thinking, sampling=sampling)
    if kind == "mock":
        logger.warning(
            "mock backend: nothing is sampled and the summary is a plumbing smoke, not a screen; "
            "the artifact will still read model_id=%r",
            model_id,
        )
        return MockBackend(mock_screen_response, model_id=model_id)
    raise ValueError(f"unknown backend kind {kind!r}; expected one of {BACKEND_KINDS}")


def _dry_run_report(  # noqa: PLR0913 - one keyword per recorded plan field
    rows: Sequence[Mapping[str, Any]],
    budget: PromptBudgetFilter,
    subset: BoundedSubset | None,
    *,
    args: argparse.Namespace,
    sampling: SamplingConfig,
    prefilled_think: bool,
) -> dict[str, object]:
    """Everything the run would do, with no model loaded and no jail launched."""
    return {
        "dry_run": True,
        "arm": args.arm,
        "grader_exposure": args.grader_exposure,
        "model_id": args.model,
        "backend": args.backend,
        "n_prompts": len(rows),
        "samples_per_prompt": args.samples_per_prompt,
        "n_samples": len(rows) * args.samples_per_prompt,
        "sampler": asdict(sampling),
        "prefilled_think": prefilled_think,
        "prompt_budget": budget.to_json_dict(),
        "bounded_subset": subset.to_json_dict() if subset else None,
        "out": str(args.out),
        "samples_out": str(samples_path_for(args.out)),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        required=True,
        choices=sorted(TRAINABLE_ARMS),
        help=(
            "which arm's grader scores the prompts. Screen the control arm too: it is the "
            "positive control that says the apparatus can see a gradient at all."
        ),
    )
    parser.add_argument(
        "--grader-exposure",
        default=GraderExposure.INLINE.value,
        choices=[exposure.value for exposure in GraderExposure],
        help=(
            "whether the prompts show the grader source (the flagship rendering, the default) or "
            "withhold it behind an unseen-checker note. Grading is identical either way; the "
            "corpus cannot move with this flag (the budget measures both exposures)."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--partition", type=Path, default=DEFAULT_PARTITION_PATH)
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help="the training group size; a different value screens a different group statistic",
    )
    parser.add_argument("--out", type=Path, required=True, help="summary JSON path")
    parser.add_argument("--grader-scratch-root", type=Path, default=DEFAULT_GRADER_SCRATCH_ROOT)
    parser.add_argument("--backend", default="vllm", choices=BACKEND_KINDS)
    parser.add_argument("--max-prompt-tokens", type=int, default=DEFAULT_MAX_PROMPT_TOKENS)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help="bound the screen to a seeded random subset; the artifact records this LOUDLY",
    )
    parser.add_argument("--seed", type=int, default=0, help="seeds only the subset selection")
    parser.add_argument(
        "--no-thinking",
        dest="thinking",
        action="store_false",
        default=DEFAULT_THINKING,
        help="plumbing smokes only; a real screen matches the training arm's thinking=True",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="resolve prompts and print the plan; load no model"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one arm's gradient screen end to end and write its two artifacts."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    started_at = datetime.now(tz=UTC)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    assert_backend_can_render_template(tokenizer)
    prefilled_think = derive_prefilled_think(tokenizer, enable_thinking=args.thinking)
    sampling = training_matched_sampling(args.model)
    partition = load_partition(args.partition)
    # The ARM's resolver, not a second implementation, and that is what makes this module's headline
    # claim true: the screen and the trainer used to apply the prompt budget and `--max-prompts` in
    # OPPOSITE orders, so "the screen covers exactly the corpus an arm would see" was false whenever
    # the bound was used on both sides. The kwargs are asked of the tokenizer here because a screen has
    # no recorded template facts to read them from the way a run record does.
    rows, budget, subset = resolve_arm_rows(
        args.arm,
        partition,
        tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        enable_thinking=args.thinking,
        chat_template_kwargs=resolve_chat_template_kwargs(tokenizer),
        max_prompts=args.max_prompts,
        seed=args.seed,
        exposure=GraderExposure(args.grader_exposure),
    )

    if args.dry_run:
        plan = _dry_run_report(
            rows, budget, subset, args=args, sampling=sampling, prefilled_think=prefilled_think
        )
        print(json.dumps(plan, indent=2))  # noqa: T201 - the dry run's whole output is this plan
        return 0

    grader = GraderConfig(scratch_root=args.grader_scratch_root)
    jail = assert_jail_usable(timeout_seconds=grader.timeout_seconds)
    grader.scratch_root.mkdir(parents=True, exist_ok=True)
    backend = build_screen_backend(
        args.backend, args.model, thinking=args.thinking, max_prompt_tokens=args.max_prompt_tokens
    )

    generation_started = time.perf_counter()
    responses = generate_screen_completions(
        rows, backend, samples_per_prompt=args.samples_per_prompt
    )
    generation_seconds = time.perf_counter() - generation_started
    grading_started = time.perf_counter()
    samples = grade_screen_completions(
        rows,
        responses,
        samples_per_prompt=args.samples_per_prompt,
        prefilled_think=prefilled_think,
        grader=grader,
    )
    grading_seconds = time.perf_counter() - grading_started

    report = screen_report(
        samples, samples_per_prompt=args.samples_per_prompt, token_cap=sampling.max_new_tokens
    )
    menu = mitigation_menu(report, arm=args.arm)
    summary: dict[str, object] = {
        "schema": SCREEN_SCHEMA,
        "kind": SCREEN_KIND,
        # Which commit measured this verdict, the way run_config.json already records it. A screen is
        # read as a launch decision months later, and `train_sequence`'s gate now refuses an arm whose
        # screen was measured under a different solution parser -- so a verdict that cannot name its
        # own code is a verdict whose provenance has to be reconstructed from S3 timestamps.
        **git_provenance(),
        "arm": args.arm,
        "grader_exposure": args.grader_exposure,
        "model_id": args.model,
        "backend_transport": backend.transport,
        "thinking": args.thinking,
        "prefilled_think": prefilled_think,
        "sampler": asdict(sampling),
        "applied_sampling": (
            backend.applied_sampling() if isinstance(backend, VLLMBackend) else None
        ),
        "sampling_unseeded": True,
        "partition": {
            "path": str(args.partition),
            "pool_fingerprint": partition.pool_fingerprint,
            "n_training_problems": len(partition.training_problem_ids),
        },
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_budget": budget.to_json_dict(),
        "bounded_subset": subset.to_json_dict() if subset else None,
        "grader": grader.to_json_dict(),
        "jail_preflight": jail,
        **report,
        "mitigations": menu,
        "samples_path": str(samples_path_for(args.out)),
        "generation_seconds": generation_seconds,
        "grading_seconds": grading_seconds,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
    }
    write_screen_artifacts(args.out, summary, samples)
    groups = report["groups"]
    logger.info(
        "screen done, %s",
        f"arm={args.arm} frac_groups_pure={groups['frac_groups_pure']:.3f} "
        f"visible_pass={report['visible_pass']} histogram={groups['pass_count_histogram']} "
        f"{generation_seconds=:.0f} {grading_seconds=:.0f}",
    )
    if groups["frac_groups_pure"] >= NEAR_PURE_WARNING_FRACTION:
        logger.warning(
            "this screen is at or near ALL-PURE (frac_groups_pure=%.3f), so as measured the arm "
            "would train on (almost) no gradient. The mitigation menu, priced with this screen's "
            "own numbers: %s",
            groups["frac_groups_pure"],
            json.dumps(menu),
        )
    assert_screen_measured(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
