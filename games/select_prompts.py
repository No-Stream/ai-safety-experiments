"""Baseline behavioural sweep, and the mixed-prompt selection it feeds.

GRPO learns from within-group reward disagreement and nothing else: if every completion in a
group scores the same, the advantage is zero and the step carries no gradient no matter how
wrong the behaviour is. So a corpus of prompts the base model answers unanimously is a corpus
that trains nothing while still burning the GPU. This module samples the base policy at the
training sampler, keeps every completion, and hands training only the prompts whose action
distribution is genuinely split.

Three things make the sweep worth more than its selection verdict.

**It is the "before" eval.** Nothing else measures the pre-RL policy on exactly the corpus that
gets trained, so the trace file is written first and in full -- every completion text, not a
count, and on disk before the frozen-opponent pass makes its first hosted call -- and the
selection summary is a derived artefact over it. Re-deciding the thresholds later is then a
re-analysis rather than a re-run.

**A verdict is only reproducible against the same pool.** Sampling draws from one global RNG
stream over the flattened prompt list and the decode chunk boundaries move with the sequence
count, so adding one scenario upstream re-rolls every later prompt's samples even at the same
`--seed`, and a prompt kept before can cross a threshold and leave for reasons unrelated to the
edit. The meta record therefore carries the pool size and a hash of the prompt-id order: two
corpora whose hashes differ are not a controlled comparison, and nothing else can say so.

**Counterbalanced label-mapping pairs are kept or dropped together.** Each scenario is rendered
twice, once with the cooperative option named first and once second, so that a model's
preference for the first-listed option cannot masquerade as a preference for cooperating. Keep
one orientation and drop its partner and that position bias walks straight into the training
corpus, which defeats the point of rendering both.

**The frozen opponent is sampled here, once.** The vs-frozen arms grade against a fixed external
policy; sampling it during the sweep and caching a per-prompt cooperation probability keeps the
training loop hermetic (no API call in a reward function) and is exact rather than approximate,
because a frozen opponent's distribution does not move. The cache records which model and which
sampling config produced it, since that is the only thing that could ever make it stale.

**A sweep resumes per prompt, and its remainder is a fresh draw.** A 632-row pool at eight samples
is 3-6 hours of a rented card, so every prompt's record is appended to a partial file under
`partial/` as its chunk returns, and a relaunch keeps those completions byte for byte and decodes
only the prompts still missing. Resume is keyed on the sweep's own identity -- the pool's content,
the model, the sampler, the engine settings that move what it samples, the samples per prompt, the
thinking convention, the grading -- carried in the partial file's header line and refused by name
when a relaunch disagrees, because folding a rewritten pool's completions into an old file would
report a sweep nobody ran. Four things keep the file itself honest across deaths: one sweep holds it
at a time, a torn trailing line is cut off before the next session appends, every resumed record's
verdicts are re-derived under the parser this session has, and the file is retired once the trace it
fed is on disk, so the same command run twice is a fresh draw rather than a copy of the first. What
resume cannot preserve is
bit-identity: `Backend.generate` takes no per-request seed, and a paged engine's batch composition
moves with the remainder anyway, so the prompts a later session decodes are a fresh draw from the
same policy distribution. That is the same measurement statistically and not the same bytes, which is
why the trace's `resume` block names every session rather than claiming one.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import random
import socket
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import AutoTokenizer

from games import parsing
from games.chunked_decode import (
    backend_schedules_own_batch,
    iter_decoded_chunks,
    local_decode_model_id,
    self_scheduling_chunk_cap,
    sweep_chunk_size,
)
from games.deltanet_kernels import bridge_decode_kernel
from games.generation import TRAINING_TEMPERATURE, TRAINING_TOP_K, TRAINING_TOP_P
from games.payoffs import COOPERATE, STATED_RETURN_UNSET
from games.preflight import (
    default_cuda_allocator_config,
    deltanet_kernel_paths,
    derive_prefilled_think,
)
from games.prompts import (
    LABEL_PRINT_ORDER_CANONICAL,
    LABEL_PRINT_ORDERS,
    OPP_COOP_PROB_UNSET,
    generate_prompt_rows,
)
from games.provenance import git_provenance
from games.rewards import (
    GRADING_GROUP_MIX,
    GRADING_ITERATED_RETURN,
    GRADING_JOINT_WELFARE_GROUP_MIX,
    GRADING_KEEP_FRACTION,
    GRADING_LEVEL_MATCH_RETURN,
    GRADING_MIN_EFFORT_GROUP_MIX,
    GRADING_OTHER_PAYOFF_GROUP_MIX,
    GRADING_SELF,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE,
    GRADING_VS_FIXED_MIX,
    GRADINGS,
    NASH_DEMAND_GRADINGS,
    REGRADE_ONLY_GRADINGS,
    STATED_RETURN_FRACTION_COLUMN,
    THRESHOLD_GOODS_GRADINGS,
    care_alpha_of,
    grading_cli_value,
)
from games.termination import MEASURED_TERMINATION_BUDGET_BY_MODEL, required_completion_budget
from reward_hacking import backend_cli
from reward_hacking.model_backend import (
    OUTPUT_AFFECTING_ENGINE_SETTINGS,
    SamplingConfig,
    build_backend,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Mapping, Sequence

    from reward_hacking.model_backend import Backend

logger = logging.getLogger(__name__)

type Row = dict[str, object]

# `GRADINGS` is a frozenset upstream; the help text wants a stable order. This is the help text's
# list only, never argparse's `choices=`, for two reasons that point the same way. The regrade-only
# gradings stay in it on purpose, because the value of refusing them here is the message that names
# the recovery path (`_refuse_a_regrade_only_grading`) rather than a terse "invalid choice". And the
# care family cannot be listed at all -- its alpha is a number -- so the flag validates through
# `games.rewards.grading_cli_value`, which accepts the family and refuses anything else by name.
GRADING_CHOICES: tuple[str, ...] = tuple(sorted(GRADINGS))
SWEEPABLE_GRADINGS: frozenset[str] = GRADINGS - REGRADE_ONLY_GRADINGS

# Gradings whose completion is a single binary choice, so selection can key on a cooperation
# rate. The others produce a continuous per-sample score and key on its spread instead.
ONE_SHOT_ACTION_GRADINGS = frozenset(
    {
        GRADING_GROUP_MIX,
        GRADING_SELF,
        GRADING_VS_FIXED_MIX,
        GRADING_JOINT_WELFARE_GROUP_MIX,
        GRADING_OTHER_PAYOFF_GROUP_MIX,
    }
)


PROMPT_COLUMN = "prompt"
PROMPT_ID_COLUMN = "prompt_id"
GRADING_COLUMN = "grading"
OPP_COOP_PROB_COLUMN = "opp_coop_prob"

SWEEP_RECORD_KIND = "prompt-sweep"
FROZEN_OPPONENT_RECORD_KIND = "frozen-opponent-sweep"
META_RECORD_KIND = "sweep-meta"
# The two line kinds only a partial file carries: its identity header, and one entry per launch that
# has appended to it. Both are absent from a finished trace, whose meta line carries the same
# sessions inside its `resume` block.
PARTIAL_HEADER_RECORD_KIND = "sweep-partial-header"
PARTIAL_SESSION_RECORD_KIND = "sweep-partial-session"
# Partial records live one level below the artifacts, and the subdirectory is load-bearing rather
# than tidiness: the launch kit lists a run's finished traces with
# `find <out-dir> -maxdepth 1 -name 'sweep-*.jsonl'`, so a partial beside them would read as a
# completed sweep with no meta line. The name inside it cannot match that glob either, in case a
# future reader walks deeper.
PARTIAL_SUBDIR = "partial"
PARTIAL_STEM_PREFIX = "pending-sweep"
# What a retired partial file's name gains once the trace it fed is on disk, and what the lock beside
# a live one is called. Neither may match the kit's `*.jsonl` count under partial/ (see
# `retire_a_completed_partial` and `partial_sweep_lock_path`), so both are suffixes after `.jsonl`.
RETIRED_PARTIAL_SUFFIX = ".completed-"
PARTIAL_LOCK_SUFFIX = ".lock"
# Where `backend_provenance` records what the inference engine was built with. Named because the
# resume identity has to take that block apart rather than compare it whole.
ENGINE_PROVENANCE_KEY = "engine"


def training_sampler(
    model_id: str,
    *,
    top_p: float = TRAINING_TOP_P,
    max_new_tokens: int | None = None,
) -> SamplingConfig:
    """Return the sampler this model's own training run generates with.

    The three decoding fields default to `games.generation`, which carries GRPOConfig's own
    generation defaults in TRL 1.10 and is where `games.train` and every stage plan read them
    too. ``top_p`` and ``max_new_tokens`` can be overridden when matching a training run with
    explicitly configured values; the temperature and top-k remain fixed training constants.
    "At training temperature" means those defaults exactly; SamplingConfig's own defaults are the
    Qwen3.5 card's non-thinking recommendation and would sweep a different policy from the one that
    gets trained.

    The completion budget defaults to our model-specific termination screen, not to TRL's
    `max_completion_length` default of 512. Callers may explicitly provide a different cap when the
    training run uses one; a thinking trace that runs past it emits no closing tag, so the sample is
    unparseable, and the per-prompt truncation counts in the trace make that censoring visible to the
    screen and its downstream audit.
    """
    return SamplingConfig(
        max_new_tokens=(
            required_completion_budget(model_id) if max_new_tokens is None else max_new_tokens
        ),
        do_sample=True,
        temperature=TRAINING_TEMPERATURE,
        top_p=top_p,
        top_k=TRAINING_TOP_K,
    )


# What a hosted frozen opponent's completions can carry. `BedrockBackend.generate` returns the
# answer content blocks only -- Converse splits reasoning into `reasoningContent`, which the text
# join drops -- so a hosted completion never carries a closing `</think>` whatever the served
# model's template does. Read under the POLICY checkpoint's value (True for every Qwen3.5/3.6/3.8)
# each one parses as truncated thinking with no visible answer, and the cache raises on the first
# prompt. No flag, because the opponent is always hosted: `sweep_frozen_opponent` builds Bedrock.
FROZEN_OPPONENT_PREFILLED_THINK = False


@dataclass(frozen=True, slots=True)
class FrozenOpponent:
    """The hosted opponent a run will sample, and the model id its cache will record.

    Two fields rather than one attribute read off the backend, because the whole value of the
    provenance check in `sweep_frozen_opponent` is that the id the cache records and the model the
    backend actually serves are independent statements which can then be compared.
    """

    backend: Backend
    model_id: str


# 1/8 and 7/8: with the default eight samples per prompt, a kept prompt showed at least one
# completion of each kind, which is the weakest condition under which a group can disagree.
DEFAULT_MIN_COOP = 0.125
DEFAULT_MAX_COOP = 0.875
# A floor on the standard deviation of the continuous per-sample score, in the same [0,1] units
# the reward uses. A twentieth of the endowment is a spread GRPO's advantage can see; a dictator
# prompt every sample answers "keep it all" has a spread of exactly zero.
DEFAULT_MIN_SPLIT_STD = 0.05
# Half the samples must parse. Below that the cooperation rate is measured on so few completions
# that it says more about the format than the policy, and the prompt needs rewording, not training.
DEFAULT_MIN_PARSEABLE_FRACTION = 0.5
DEFAULT_SAMPLES_PER_PROMPT = 8
DEFAULT_OUT_DIR = Path("artifacts/games/select")
DEFAULT_SPLIT = "train"

# Columns the counterbalanced label swap itself changes. Everything else is pair identity: see
# `pair_identity`. `reskin_id` is deliberately absent -- it carries the scenario id and is the
# same in both orientations, so it is the sharpest discriminator the key has.
COLUMNS_VARYING_WITHIN_A_PAIR = frozenset(
    {
        PROMPT_COLUMN,
        PROMPT_ID_COLUMN,
        "coop_label",
        "coop_label_index",
        OPP_COOP_PROB_COLUMN,
    }
)

# A spread needs two observations to exist, and a sweep needs two samples to show a split.
MIN_SAMPLES_FOR_SPREAD = 2
# A scenario is rendered once per label orientation, so a pair group holds exactly two rows.
LABEL_ORIENTATIONS_PER_SCENARIO = 2

# Model families whose chat template emits the opening <think> as part of the prompt, so a
# completion can only ever carry the closing tag. Verified for Qwen3.5/3.8 on this box
# (docs/scratch/qwen38-27b-load-check-2026-08-17.md); Qwen3-0.6B does not prefill.
PREFILLED_THINK_MODEL_MARKERS = ("qwen3.5", "qwen3.6", "qwen3.8")

# Canned completions behind `--backend mock`, so the whole CLI runs end to end at zero cost
# before a real sweep is launched. They are shapes, not answers: the action labels are per
# scenario, so most of these parse as failures against a real corpus and the run's value is
# "the path executes and writes its three artefacts", never the selection it reports.
MOCK_RESPONSES: tuple[str, ...] = (
    "<think>weighing the two options</think><action>HOLD</action>",
    "<think>weighing the two options</think><action>PUSH</action>",
    "<think>weighing the two options</think><keep>50</keep>",
    "<think>naming a figure</think><claim>30</claim>",
    "<think>weighing how much to put towards it</think><contribute>3</contribute>",
    "<think>weighing how much to hand over</think><send>4</send>",
    "<think>naming an amount and a share</think><send>7</send><return>40</return>",
    "<think>naming a share only</think><return>25</return>",
    "<think>picking a level</think><level>3</level>",
    (
        "<think>planning five rounds</think>"
        "<level>5</level><level>5</level><level>5</level><level>5</level><level>4</level>"
    ),
    "<think>I would rather not commit either way</think>no answer here",
    "<think>still weighing and running out of budget",
)


class DropReason(StrEnum):
    """Why a prompt was kept or dropped, recorded per prompt so the summary is auditable."""

    KEPT_MIXED = "kept-mixed"
    TOO_FEW_PARSEABLE = "too-few-parseable"
    COOP_FRACTION_BELOW_MIN = "coop-fraction-below-min"
    COOP_FRACTION_ABOVE_MAX = "coop-fraction-above-max"
    SCORE_SPREAD_BELOW_MIN = "score-spread-below-min"
    PARTNER_DROPPED = "counterbalanced-partner-dropped"


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    """One sampled completion and everything parsing could read off it.

    `selection_score` is the per-sample scalar selection keys on, in [0,1]: a cooperation
    indicator for the one-shot action gradings, the kept fraction for the dictator game, the claimed
    fraction for the simultaneous-claim game, the contributed fraction for the shared undertaking, the
    fraction of the stock sent for the trust games, and the cooperative-move fraction for the iterated
    arms. It is None exactly when parsing failed.

    The raw answer is kept beside the score for the numeric games (`kept`, `claim`, `contribution`,
    `sent`) so a later re-analysis can histogram the actual figures -- the claim histogram, the
    exact-half rate and the contribution's distance from the equal share are all questions about the
    integers, and dividing by the total throws that away.
    """

    completion: str
    visible_text: str
    truncated_thinking: bool
    parsed: bool
    selection_score: float | None
    action: str | None = None
    action_sequence: tuple[str, ...] | None = None
    kept: int | None = None
    claim: int | None = None
    contribution: int | None = None
    sent: int | None = None
    return_percentage: int | None = None
    level: int | None = None
    levels: tuple[int, ...] | None = None


def _sample_from_json(stored: Mapping[str, Any]) -> SampleOutcome:
    """Rebuild one sample from its own trace record, restoring the fields JSON flattens to lists.

    `action_sequence` and `levels` are the only two: everything else round-trips as itself. A field
    this does not know about raises out of the constructor rather than being dropped, which is what
    a partial file written by an older schema has to do -- silently losing a column would make a
    resumed record differ from the one it claims to be.
    """
    fields: dict[str, Any] = dict(stored)
    for field in ("action_sequence", "levels"):
        if fields[field] is not None:
            fields[field] = tuple(fields[field])
    return SampleOutcome(**fields)


@dataclass(frozen=True, slots=True)
class PromptSweepRecord:
    """Every sample drawn for one prompt, plus the row that produced it.

    The aggregates are properties rather than stored fields so they cannot drift from the samples
    they summarise; `to_json_dict` materialises them into the trace.
    """

    prompt_id: str
    grading: str
    row: Row
    samples: tuple[SampleOutcome, ...]
    record_kind: str = SWEEP_RECORD_KIND

    @property
    def n_samples(self) -> int:
        """Return how many completions were drawn for this prompt."""
        return len(self.samples)

    @property
    def parsed_samples(self) -> tuple[SampleOutcome, ...]:
        """Return only the samples that yielded a well-formed answer."""
        return tuple(sample for sample in self.samples if sample.parsed)

    @property
    def n_parse_failures(self) -> int:
        """Return how many completions carried no usable answer."""
        return self.n_samples - len(self.parsed_samples)

    @property
    def n_truncated_thinking(self) -> int:
        """Return how many completions ran out of tokens inside the thinking block."""
        return sum(1 for sample in self.samples if sample.truncated_thinking)

    @property
    def parseable_fraction(self) -> float:
        """Return the share of completions that parsed, or 0.0 when nothing was sampled."""
        if not self.samples:
            return 0.0
        return len(self.parsed_samples) / self.n_samples

    @property
    def selection_scores(self) -> list[float]:
        """Return the per-sample selection scalars, over the parseable samples only."""
        return [
            sample.selection_score
            for sample in self.parsed_samples
            if sample.selection_score is not None
        ]

    @property
    def score_std(self) -> float:
        """Return the population standard deviation of the selection scores.

        Population rather than sample standard deviation: these are all the samples there are,
        and the sample form is undefined on a single observation where the population form
        correctly reads zero spread.
        """
        scores = self.selection_scores
        if len(scores) < MIN_SAMPLES_FOR_SPREAD:
            return 0.0
        return statistics.pstdev(scores)

    @property
    def scores_a_binary_action(self) -> bool:
        """Report whether this record's own row answers with one of two labels."""
        return row_scores_a_binary_action(self.grading, self.row)

    @property
    def coop_count(self) -> int | None:
        """Cooperative choices observed: samples one-shot, moves iterated, None for a numeric answer.

        None rather than zero for the games whose answer is a figure: they have no action pair, so a
        cooperation count would be a measurement of nothing printed beside the ones that are real.
        """
        if self.scores_a_binary_action:
            return sum(1 for sample in self.parsed_samples if sample.action == COOPERATE)
        if self.grading == GRADING_ITERATED_RETURN:
            return sum(
                sum(1 for move in sample.action_sequence if move == COOPERATE)
                for sample in self.parsed_samples
                if sample.action_sequence is not None
            )
        return None

    @property
    def coop_total(self) -> int | None:
        """Return the denominator matching `coop_count`, or None when cooperation is undefined."""
        if self.scores_a_binary_action:
            return len(self.parsed_samples)
        if self.grading == GRADING_ITERATED_RETURN:
            return sum(
                len(sample.action_sequence)
                for sample in self.parsed_samples
                if sample.action_sequence is not None
            )
        return None

    @property
    def coop_fraction(self) -> float | None:
        """Return the cooperation rate over parseable samples, or None if it has no meaning."""
        count, total = self.coop_count, self.coop_total
        if count is None or total is None or total == 0:
            return None
        return count / total

    @property
    def kept_fractions(self) -> list[float]:
        """Return the dictator game's kept fractions, and an empty list for every other grading."""
        if self.grading != GRADING_KEEP_FRACTION:
            return []
        return self.selection_scores

    def to_json_dict(self) -> dict[str, object]:
        """Flatten into a self-describing JSON record, aggregates beside the samples."""
        return {
            "record_kind": self.record_kind,
            "prompt_id": self.prompt_id,
            "grading": self.grading,
            "row": dict(self.row),
            "n_samples": self.n_samples,
            "n_parse_failures": self.n_parse_failures,
            "n_truncated_thinking": self.n_truncated_thinking,
            "parseable_fraction": self.parseable_fraction,
            "coop_count": self.coop_count,
            "coop_total": self.coop_total,
            "coop_fraction": self.coop_fraction,
            "kept_fractions": self.kept_fractions,
            "selection_scores": self.selection_scores,
            "score_std": self.score_std,
            "samples": [dataclasses.asdict(sample) for sample in self.samples],
        }

    @classmethod
    def from_json_dict(cls, stored: Mapping[str, Any], *, row: Row) -> PromptSweepRecord:
        """Rebuild a record from a line `to_json_dict` wrote, for the resume path.

        Exact in the only sense that matters: every aggregate in the record is a property over the
        samples, so a rebuilt record re-derives them and `to_json_dict` reproduces the stored line
        byte for byte. That is what "resumed records are kept, never regenerated" has to mean, and
        `TestTheSweepResumesPerRecord` asserts it against a fresh run's line rather than trusting it.

        The row comes from THIS launch's pool rather than from the stored record, because a JSON round
        trip cannot tell a tuple column from a list one, and a resumed row that differs in type from a
        freshly swept one would reach the written corpus. Nothing is lost by preferring the live row:
        the resume refuses outright unless the pool's content digest matches, so the two are the same
        row. `prompt_id` and `grading` are still read off the record, so a caller pairing a record with
        the wrong row is a mismatch this can be checked for rather than one it papers over.
        """
        return cls(
            prompt_id=str(stored["prompt_id"]),
            grading=str(stored["grading"]),
            row=dict(row),
            samples=tuple(_sample_from_json(sample) for sample in stored["samples"]),
            record_kind=str(stored["record_kind"]),
        )


@dataclass(frozen=True, slots=True)
class PromptVerdict:
    """The selection decision for one prompt, and the measurements behind it."""

    prompt_id: str
    keep: bool
    reason: str
    coop_fraction: float | None
    score_std: float
    parseable_fraction: float


def _require(row: Row, column: str) -> object:
    """Return a row column, naming what was actually present when it is missing.

    A missing column here means the corpus builder and this module disagree about the schema,
    which is a caller bug and must not read as a model that failed to answer.
    """
    if column not in row:
        raise KeyError(
            f"row is missing the {column!r} column required by the sweep; "
            f"columns present: {sorted(row)}"
        )
    return row[column]


def _require_str(row: Row, column: str) -> str:
    """Return a row column as a string, refusing another type."""
    value = _require(row, column)
    if not isinstance(value, str):
        raise TypeError(f"row column {column!r} must be a string, got {type(value).__name__}")
    return value


def _require_int(row: Row, column: str) -> int:
    """Return a row column as an int, refusing another type (bool included)."""
    value = _require(row, column)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"row column {column!r} must be an int, got {type(value).__name__}")
    return value


def _require_number(row: Row, column: str) -> float:
    """Return a row column as a float, refusing a non-numeric type (bool included)."""
    value = _require(row, column)
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise TypeError(f"row column {column!r} must be a number, got {type(value).__name__}")
    return float(value)


def row_scores_a_binary_action(grading: str, row: Row) -> bool:
    """Report whether this row's answer is one of two labels, so selection keys on a cooperation rate.

    Every grading but one answers this by its name alone. The care family covers two row types -- its
    matrix rows answer with an action and its announced-rule trust rows with an amount sent -- so for
    those the ROW decides, read off the same `stated_return_fraction` column the reward function's own
    dispatch reads. Judging a mixed care corpus by one rule for both halves would filter the trust
    rows' send spread as though it were a cooperation rate, and every one of them would be dropped
    for having no cooperation at all.
    """
    if grading in ONE_SHOT_ACTION_GRADINGS:
        return True
    if care_alpha_of(grading) is None:
        return False
    return _require_number(row, STATED_RETURN_FRACTION_COLUMN) == STATED_RETURN_UNSET


def prefilled_think_from_model_name(model_id: str) -> bool:
    """Guess whether the model's chat template prefills `<think>`, from its name alone.

    The fallback for backends with no reachable tokenizer (mock, and any hosted endpoint that
    applies its own template). `prefilled_think_from_template` is authoritative and should be
    preferred whenever a tokenizer can be loaded.
    """
    lowered = model_id.casefold()
    return any(marker in lowered for marker in PREFILLED_THINK_MODEL_MARKERS)


def prefilled_think_from_template(model_id: str, *, enable_thinking: bool = True) -> bool:
    """Ask the model's own chat template whether it emits the opening `<think>` in the prompt.

    Loads the tokenizer only, so this costs a config download rather than a model load. Getting
    this wrong is silent in both directions: a completion carrying no tag reads as a plain answer
    when the flag is false and as truncated thinking when it is true, and only one of those is
    the truth for a given template.

    `enable_thinking` must match the setting the sweep itself generates under, which is why it is a
    parameter rather than a constant. Rendering with thinking on while sweeping with it off cost a
    whole sweep on 2026-08-17: the Qwen3.5 templates prefill `<think>` only in thinking mode, so the
    probe said True, every tag-free completion then read as truncated thinking, and all 64 prompts
    were dropped as unparseable with nothing in the log pointing at the cause.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Delegates rather than re-deriving: whether a template "prefills" is not whether `<think>`
    # appears, it is whether the block is left OPEN at the end of the prompt. A containment check
    # gets thinking-off backwards, because Qwen3.5 renders a CLOSED empty block in that mode -- it
    # contains `<think>` while prefilling nothing. Two copies of that predicate diverged once
    # already and cost a sweep, so there is one.
    return derive_prefilled_think(tokenizer, enable_thinking=enable_thinking)


def _regrade_only_refusal(grading: str) -> str:
    """Explain why a sweep cannot select prompts under this grading, and what to do instead."""
    return (
        f"grading {grading!r} cannot be swept: selection keeps the prompts whose ACTION "
        f"distribution is mixed (games.select_prompts scores each sample by the action it names), "
        f"and this grading's reward is blind to the action, so there is no per-sample score to "
        f"select on and 'mixed' would be undefined. Sweep the game under the grading its "
        f"strategically-graded arm trains on -- which is also what keeps the two arms on the same "
        f"prompts, the invariant the whole contrast rests on -- then transform the selected corpus "
        f"with games.regrade_corpus. Through games.arm_sequence that is "
        f"GAMES_ARM_SEQ_SWEEP_GRADING=<the real grading> plus 'regrade' in GAMES_ARM_SEQ_STAGES; "
        f"sweepable gradings: {sorted(SWEEPABLE_GRADINGS)}."
    )


@dataclass(frozen=True)
class _ParsedAnswer:
    """What one grading's parser read off a completion, before the sample-level fields go on.

    Every field is optional because each grading fills only its own: an action for the binary games,
    a figure for the numeric ones. `selection_score` is None exactly when parsing failed, which is
    what `parsed` is derived from.
    """

    selection_score: float | None = None
    action: str | None = None
    action_sequence: tuple[str, ...] | None = None
    kept: int | None = None
    claim: int | None = None
    contribution: int | None = None
    sent: int | None = None
    return_percentage: int | None = None
    level: int | None = None
    levels: tuple[int, ...] | None = None


def _parse_action_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read a binary choice, scored as a cooperation indicator."""
    action = parsing.parse_action(
        visible_text,
        label_a=_require_str(row, "label_a"),
        label_b=_require_str(row, "label_b"),
        coop_label=_require_str(row, "coop_label"),
    )
    if action is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=float(action == COOPERATE), action=action)


def _parse_keep_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read the unilateral split's kept figure, scored as the fraction of the endowment kept."""
    endowment = _require_int(row, "endowment")
    kept = parsing.parse_split(visible_text, endowment=endowment)
    if kept is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=kept / endowment, kept=kept)


def _parse_claim_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read the simultaneous claim, scored as the fraction of the windfall claimed.

    Both gradings of the claim game read the same answer off the same prompt, so selection measures
    one thing for the pair: how spread the claimed fractions are. Keying on the spread rather than on
    a rate is what the continuous path is for -- there is no rate here, and a corpus every sample
    answers with the same figure carries no advantage under either grading.
    """
    windfall = _require_int(row, "windfall")
    claim = parsing.parse_claim(visible_text, windfall=windfall)
    if claim is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=claim / windfall, claim=claim)


def _parse_contribution_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read the figure put towards the shared undertaking, scored as the fraction of the stock put in.

    Both gradings of this game read the same answer off the same prompt, so selection measures one thing
    for the pair: how spread the contributed fractions are. Scored on the ACTION rather than the reward,
    the same deliberate difference the trust gradings make: the group-mix reward depends on the group's
    own realised figures, so selecting on it would judge a prompt by a quantity that moves with whichever
    other samples happened to be drawn beside it, and the two prize variants would be handed
    systematically different corpora -- confounding exactly the between-variant comparison the pair of
    arms exists to make.
    """
    endowment = _require_int(row, "endowment")
    contribution = parsing.parse_contribution(visible_text, endowment=endowment)
    if contribution is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=contribution / endowment, contribution=contribution)


def _parse_move_sequence_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read a whole iterated match, scored as the fraction of its moves that cooperated."""
    moves = parsing.parse_action_sequence(
        visible_text,
        n_rounds=_require_int(row, "n_rounds"),
        label_a=_require_str(row, "label_a"),
        label_b=_require_str(row, "label_b"),
        coop_label=_require_str(row, "coop_label"),
    )
    if moves is None:
        return _ParsedAnswer()
    cooperating = sum(1 for move in moves if move == COOPERATE)
    return _ParsedAnswer(selection_score=cooperating / len(moves), action_sequence=tuple(moves))


def _parse_level_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read a minimum-effort level, scored as its position on the grid.

    The LEVEL rather than the reward, which is the same deliberate choice the trust games make and for
    the same reason: the reward's slope in the level differs per cost ratio (the four variants' spreads
    run 0.100 to 0.400 against the reference distribution), so selecting on reward would demand a 4x
    wider spread from one variant than another and hand the two registered arms systematically
    different corpora -- confounding the between-variant comparison the pair exists to make. The cost
    is that the `min_split_std` floor is in level units here, so it is a fixed bar on how spread the
    answers are rather than on how much reward that spread is worth.
    """
    n_levels = _require_int(row, "n_levels")
    level = parsing.parse_level(visible_text, n_levels=n_levels)
    if level is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=(level - 1) / (n_levels - 1), level=level)


def _parse_level_sequence_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read a whole level-matcher match, scored as the mean position of its levels on the grid.

    All-or-nothing on the round count, as `_parse_move_sequence_answer` is: a simulated return over
    invented rounds measures nothing. The last round's level rides along beside the mean because the
    end-game drop is this arm's headline reading and a mean cannot answer it.
    """
    n_levels = _require_int(row, "n_levels")
    levels = parsing.parse_level_sequence(
        visible_text, n_rounds=_require_int(row, "n_rounds"), n_levels=n_levels
    )
    if levels is None:
        return _ParsedAnswer()
    return _ParsedAnswer(
        selection_score=sum((level - 1) / (n_levels - 1) for level in levels) / len(levels),
        level=levels[-1],
        levels=tuple(levels),
    )


def _parse_stated_rule_trust_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read an announced-rule trust answer: the amount sent, against a rate the prompt stated."""
    endowment = _require_int(row, "endowment")
    sent = parsing.parse_send(visible_text, endowment=endowment)
    if sent is None:
        return _ParsedAnswer()
    return _ParsedAnswer(selection_score=sent / endowment, sent=sent)


def _parse_strategy_method_trust_answer(visible_text: str, row: Row) -> _ParsedAnswer:
    """Read a strategy-method answer: the send and the share promised, or neither.

    All-or-nothing, so a send is never recorded against a rule the completion did not state -- the
    same reason `parse_action_sequence` refuses a partial move list.
    """
    endowment = _require_int(row, "endowment")
    strategy = parsing.parse_trust_strategy(visible_text, endowment=endowment)
    if strategy is None:
        return _ParsedAnswer()
    return _ParsedAnswer(
        selection_score=strategy.sent / endowment,
        sent=strategy.sent,
        return_percentage=strategy.return_percentage,
    )


# One parser per grading, and the ONLY statement of which gradings this sweep can read: a chain of
# branches plus a separately-written vocabulary set is two statements of one list, and those two had
# already drifted apart at wave-2 integration.
ANSWER_PARSER_BY_GRADING: dict[str, Callable[[str, Row], _ParsedAnswer]] = {
    **dict.fromkeys(ONE_SHOT_ACTION_GRADINGS, _parse_action_answer),
    GRADING_KEEP_FRACTION: _parse_keep_answer,
    **dict.fromkeys(NASH_DEMAND_GRADINGS, _parse_claim_answer),
    **dict.fromkeys(THRESHOLD_GOODS_GRADINGS, _parse_contribution_answer),
    GRADING_ITERATED_RETURN: _parse_move_sequence_answer,
    GRADING_TRUSTOR_PAYOFF_STATED_RULE: _parse_stated_rule_trust_answer,
    GRADING_TRUSTOR_PAYOFF_SELF_RULE: _parse_strategy_method_trust_answer,
    GRADING_MIN_EFFORT_GROUP_MIX: _parse_level_answer,
    GRADING_LEVEL_MATCH_RETURN: _parse_level_sequence_answer,
}

# What `_parse_sample` can actually read, derived from the table above rather than restated, because
# it differs from `GRADINGS` exactly when a grading is added upstream without a parser here -- and
# that is the case the refusal has to describe honestly.
PARSEABLE_GRADINGS: frozenset[str] = frozenset(ANSWER_PARSER_BY_GRADING)


def answer_parser_for(grading: str, row: Row) -> Callable[[str, Row], _ParsedAnswer] | None:
    """Return the parser this row's answers are read with, or None if the sweep cannot read them.

    A lookup for every grading whose row type its name fixes, and a per-row choice for the care
    family, which covers both the matrix games and the announced-rule trust sender. One care corpus
    holding both is the point of the family, so the parser cannot be keyed on the grading alone: the
    row's own discriminator picks it, exactly as `games.rewards` picks the scorer.
    """
    tabulated = ANSWER_PARSER_BY_GRADING.get(grading)
    if tabulated is not None:
        return tabulated
    if care_alpha_of(grading) is None:
        return None
    if row_scores_a_binary_action(grading, row):
        return _parse_action_answer
    return _parse_stated_rule_trust_answer


def _parse_sample(completion: str, row: Row, *, prefilled_think: bool) -> SampleOutcome:
    """Parse one completion according to the row's grading, never raising on model output.

    The trust gradings score on the fraction of the stock sent, which is the normalised ACTION rather
    than the reward -- a deliberate difference from the dictator game, where the two coincide. The
    reason is that the trust reward is affine in the send with a slope that differs per announced
    return rate (spreads 0.267 and 0.333), so selecting on the reward would demand a 1.25x wider send
    spread from one variant than the other and hand the two arms systematically different corpora.
    The headline reading across that pair is a comparison BETWEEN the variants, so a variant-dependent
    filter would confound exactly the thing the pair exists to measure. The cost is that the
    `min_split_std` floor is in send units here: 0.05 of the stock is 0.013 of the reward range at the
    lower rate, so this filter is more permissive for the trust games than for the dictator game.
    """
    grading = _require_str(row, GRADING_COLUMN)
    visible_text, truncated_thinking = parsing.strip_thinking(
        completion, prefilled_think=prefilled_think
    )
    if grading in REGRADE_ONLY_GRADINGS:
        raise ValueError(_regrade_only_refusal(grading))
    parse_answer = answer_parser_for(grading, row)
    if parse_answer is None:
        raise ValueError(
            f"unknown grading {grading!r}; this sweep can parse {sorted(PARSEABLE_GRADINGS)} plus "
            f"the care family. "
            f"A grading listed in games.rewards.GRADINGS but missing here has no selection score, "
            f"so it would reach the judge with nothing to judge."
        )
    answer = parse_answer(visible_text, row)
    return SampleOutcome(
        completion=completion,
        visible_text=visible_text,
        truncated_thinking=truncated_thinking,
        parsed=answer.selection_score is not None,
        selection_score=answer.selection_score,
        action=answer.action,
        action_sequence=answer.action_sequence,
        kept=answer.kept,
        claim=answer.claim,
        contribution=answer.contribution,
        sent=answer.sent,
        return_percentage=answer.return_percentage,
        level=answer.level,
        levels=answer.levels,
    )


def pool_digest(rows: Sequence[Row]) -> str:
    """Digest the pool's content and its order, which is what two sessions of one sweep must share.

    Over the parsed rows rather than a file's bytes, because a generated pool has no file: `--game`
    renders its rows out of this repository, and the resume needs one statement that covers both pool
    sources. Every column of every row enters it in the pool's own order, so an edited frame, a
    re-rendered label orientation, an inserted scenario and a reordered file all move it -- which is
    the same set of edits that re-rolls the sampling stream (see `sweep_meta`).
    """
    canonical = "\n".join(json.dumps(row, sort_keys=True, default=str) for row in rows)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sweep_identity(  # noqa: PLR0913 - one identity, and every argument is a field of it
    *,
    backend: Backend,
    rows: Sequence[Row],
    samples_per_prompt: int,
    prefilled_think: bool,
    thinking: bool,
    grading: str,
    rows_sha256: str | None,
) -> dict[str, Any]:
    """List everything two launches must agree on before one may resume the other's partial records.

    "The same measurement" and nothing wider. The pool's content and order, the model and transport,
    the sampler the backend carries, how many samples a prompt gets, which thinking convention its
    completions follow, and the grading every completion is parsed under: change any one and the
    records already on disk describe a policy or a pool this launch is not sweeping. What is
    deliberately absent is everything that only names the artifacts -- the out-dir, the timestamp, the
    split -- and the commit, because a relaunch on a later commit is the ordinary case and the trace
    records each session's revision instead (`_partial_session_record`).

    The result is round-tripped through JSON before it is returned, because it is compared field by
    field against a header read back out of JSON. A sampling config's `stop` is a tuple in memory and
    a list once written, and comparing the live dataclass against the stored one would refuse every
    resume of a sweep whose sampler has one.

    `thinking` is the RESOLVED boolean rather than the tri-state an operator typed. `--thinking` is
    three-valued so an explicit choice can be told from the default, and on a hosted backend an absent
    flag and a spelled-out `--no-thinking` are the same policy (`backend_cli.resolve_thinking` maps
    both to False), so storing the tri-state refused the second phrasing of one command and re-paid
    for every prompt the dead box had already decoded. `sweep_identity_from_args` resolves it.
    """
    provenance = backend_provenance(backend)
    identity: dict[str, Any] = {
        "pool_digest": pool_digest(rows),
        "rows_sha256": rows_sha256,
        "n_prompts": len(rows),
        "grading": grading,
        "samples_per_prompt": samples_per_prompt,
        "prefilled_think": prefilled_think,
        "thinking": thinking,
        "max_new_tokens": _backend_max_new_tokens(backend),
        # The engine block is lifted out of the nested provenance and spread flat below, because only
        # part of it belongs in an identity at all: see `engine_identity_fields`.
        "backend": {
            field: value for field, value in provenance.items() if field != ENGINE_PROVENANCE_KEY
        },
        **engine_identity_fields(backend),
    }
    return json.loads(json.dumps(identity, sort_keys=True, default=str))


def sweep_identity_from_args(
    args: argparse.Namespace,
    *,
    backend: Backend,
    rows: Sequence[Row],
    prefilled_think: bool,
    rows_sha256: str | None,
) -> dict[str, Any]:
    """Build the resume identity from one launch's parsed flags, which is where they are resolved.

    A named seam rather than an inline dict in `main`, because one of the fields has to be RESOLVED
    rather than copied: `args.thinking` is a tri-state and the identity has to record the boolean the
    sweep ran under. Everything else is read straight off the flags or off the backend they built.
    """
    return sweep_identity(
        backend=backend,
        rows=rows,
        samples_per_prompt=args.samples_per_prompt,
        prefilled_think=prefilled_think,
        thinking=backend_cli.resolve_thinking(args),
        grading=args.grading,
        rows_sha256=rows_sha256,
    )


def engine_identity_fields(backend: Backend) -> dict[str, Any]:
    """Name one flat identity field per engine setting that changes what the engine samples.

    These never reached the identity at all: they are construction kwargs of the vLLM engine rather
    than fields of the sampler, so a relaunch that added `--vllm-quantization fp8` or shortened
    `--vllm-max-model-len` was accepted and filed two policies under one trace with every count still
    adding up.

    Flat, one field per setting, None for "unset", and that shape is load-bearing:
    `_refuse_a_foreign_partial` compares field by field against a header read off disk, so a header
    written before these fields existed carries None for each of them and a launch that sets none of
    them still resumes it, while a launch that sets one is refused by that field's own name. A single
    nested dict would compare unequal against every such header and throw away exactly the hours the
    resume exists to save.

    Which settings these are, and why `gpu_memory_utilization` is not one of them, is
    `reward_hacking.model_backend.OUTPUT_AFFECTING_ENGINE_SETTINGS`. Read off the backend rather than
    off the flags for the same reason as the sampler: a caller can derive an engine kwarg from a
    checkpoint (`games.eval_model`) with no flag having been typed.
    """
    settings = _engine_settings(backend)
    return {f"engine_{name}": settings.get(name) for name in OUTPUT_AFFECTING_ENGINE_SETTINGS}


def _engine_settings(backend: Backend) -> dict[str, object]:
    """Return the engine construction settings a backend recorded, empty for one that takes none.

    Reflective for the same reason `backend_provenance` reads the sampler that way: only the vLLM
    backend takes engine kwargs, and the protocol does not carry them.
    """
    return dict(getattr(backend, "engine_settings", {}))


def identity_digest(identity: Mapping[str, Any]) -> str:
    """Digest a sweep identity, so a log line and a header can name it without printing the fields."""
    canonical = json.dumps(dict(identity), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def partial_sweep_path(args: argparse.Namespace) -> Path:
    """Name the file this launch appends its finished records to as they are decoded.

    Under `partial/` because the launch kit lists finished traces with
    `find <out-dir> -maxdepth 1 -name 'sweep-*.jsonl'`, and a partial beside them would read as a
    completed sweep whose meta line is missing. The stem cannot match that glob either, so a reader
    that later walks deeper still cannot mistake one for the other.

    Keyed on the pool and the model rather than on the identity digest, and that is the deliberate
    half. A name carrying the digest would mean a relaunch that changed the samples per prompt, the
    sampler or the pool's own content quietly opened a SECOND partial file and re-swept all 632 rows
    -- three to six hours of a rented card, with nothing in the log to say it happened. Under this
    name such a relaunch lands on the same file and is refused by the field that differs
    (`read_partial_sweep`); moving the old file aside is then the deliberate act that starts a fresh
    sweep, rather than the default nobody chose.
    """
    return (
        args.out_dir / PARTIAL_SUBDIR / f"{PARTIAL_STEM_PREFIX}-{_pool_and_model_stem(args)}.jsonl"
    )


@dataclass(frozen=True, slots=True)
class PartialSweep:
    """What a partial file already holds for this launch, and how it came to hold it.

    `has_header` rather than inferring one from the records, because a file with a header and no
    records at all is a real state -- a session that died inside its first chunk -- and rewriting the
    header over it would discard the session history the trace's resume block is built from.

    `complete_bytes` is where the last complete line ends, so a caller can cut a torn tail off at a
    line boundary before appending to the file (`truncate_a_torn_tail`). A byte offset rather than a
    character count, because the records are written with `ensure_ascii=False`.
    """

    records: tuple[PromptSweepRecord, ...]
    sessions: tuple[dict[str, Any], ...]
    n_torn: int
    has_header: bool
    complete_bytes: int = 0


def read_partial_sweep(
    path: Path,
    *,
    identity: Mapping[str, Any],
    rows: Sequence[Row],
    record_kind: str = SWEEP_RECORD_KIND,
) -> PartialSweep:
    """Read the prompts a previous session finished, refusing a file that is a different sweep's.

    Three things a relaunch has to tell apart, and the whole reason this is not a bare read. A file
    that is this sweep's is resumed. A file whose header disagrees with this launch's identity is
    another sweep's, and the refusal names the field, because appending to it would file two
    measurements under one pool hash with every count still adding up. And a trailing line without its
    newline is a death mid-append: dropped and counted, never parsed, since only the last line can be
    torn -- every earlier one was flushed and fsynced whole.

    A record whose prompt id is not in this pool, and a pool prompt with two records, both refuse.
    Neither can happen while the header matches, which is the point: they are the shapes a hand-edited
    or concatenated file takes, and a resumed rate over two records for one prompt has a denominator
    nobody can state.

    Read as BYTES, and everything after the last newline is the torn tail whatever its bytes are.
    The records are written with `ensure_ascii=False` and Qwen3.5 thinking traces carry non-ASCII
    routinely, so a death can cut the trailing line inside a character: read as strict UTF-8 text that
    is a `UnicodeDecodeError` raised before any of the torn-line handling below is reached, and the
    whole file is refused on the FIRST relaunch -- the case this exists for.

    A torn tail can never be a record that parses, because a line is one `json.dumps` of one record
    and every strict prefix of that is missing its closing brace. The one tail that would parse is a
    whole record short of its newline, and it is dropped and counted like any other rather than kept:
    the writer never terminated the line, and the cost of not trusting it is one prompt re-decoded.
    """
    if not path.exists():
        return PartialSweep((), (), 0, has_header=False)
    raw = path.read_bytes()
    complete_bytes = raw.rfind(b"\n") + 1
    n_torn = 1 if raw[complete_bytes:].strip() else 0
    complete = [line for line in raw[:complete_bytes].decode("utf-8").split("\n") if line.strip()]
    if not complete:
        # Nothing but a torn header: a death between creating the file and writing its first record,
        # so there is no finished work in it to lose and the caller starts it again.
        return PartialSweep((), (), n_torn, has_header=False)
    _refuse_a_foreign_partial(
        _partial_line(complete[0], path=path, number=1), identity=identity, path=path
    )
    rows_by_id = {_require_str(row, PROMPT_ID_COLUMN): row for row in rows}
    records: list[PromptSweepRecord] = []
    sessions: list[dict[str, Any]] = []
    for number, line in enumerate(complete[1:], start=2):
        stored = _partial_line(line, path=path, number=number)
        kind = stored.get("record_kind")
        if kind == PARTIAL_SESSION_RECORD_KIND:
            sessions.append(stored)
            continue
        if kind != record_kind:
            raise ValueError(
                f"{path} line {number} carries record_kind {kind!r}, but a partial sweep file holds "
                f"only {record_kind!r} records, {PARTIAL_SESSION_RECORD_KIND!r} session lines and "
                f"its {PARTIAL_HEADER_RECORD_KIND!r} header. Move the file aside rather than resuming "
                f"over something this cannot read."
            )
        prompt_id = str(stored["prompt_id"])
        row = rows_by_id.get(prompt_id)
        if row is None:
            raise ValueError(
                f"{path} line {number} holds a record for prompt {prompt_id!r}, which is not in this "
                f"launch's pool of {len(rows)} prompts. The header's pool digest matched, so the file "
                f"has been edited or concatenated since; move it aside."
            )
        if any(record.prompt_id == prompt_id for record in records):
            raise ValueError(
                f"{path} holds two records for prompt {prompt_id!r}; a resumed sweep over them would "
                f"report a cooperation rate whose denominator nobody can state. Move the file aside "
                f"or dedupe it deliberately."
            )
        records.append(PromptSweepRecord.from_json_dict(stored, row=row))
    return PartialSweep(
        tuple(records), tuple(sessions), n_torn, has_header=True, complete_bytes=complete_bytes
    )


def _partial_line(line: str, *, path: Path, number: int) -> dict[str, Any]:
    """Parse one complete line of a partial file, naming where a corrupt one is.

    Only a TRAILING line can be torn, so a line that does not parse here is corruption of another
    kind -- an interleaved writer, a truncation at a newline boundary, an edit -- and guessing at it
    would be worse than stopping.
    """
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path} line {number} is not JSON, and it is not the trailing line, so it was not torn "
            f"by a death mid-append. Move the file aside: {error}"
        ) from error
    if not isinstance(parsed, dict):
        raise TypeError(f"{path} line {number} is a {type(parsed).__name__}, not a JSON object")
    return parsed


def _refuse_a_foreign_partial(
    header: Mapping[str, Any], *, identity: Mapping[str, Any], path: Path
) -> None:
    """Refuse a partial file whose header describes a different sweep from this launch's, by field."""
    if header.get("record_kind") != PARTIAL_HEADER_RECORD_KIND:
        raise ValueError(
            f"{path} does not open with a {PARTIAL_HEADER_RECORD_KIND!r} line (its first line is "
            f"record_kind {header.get('record_kind')!r}), so nothing in it can be checked against "
            f"this launch's identity. Move it aside."
        )
    stored = header.get("identity")
    if not isinstance(stored, dict):
        raise TypeError(
            f"{path}'s header carries no identity mapping (got {type(stored).__name__}), so a resume "
            f"could not tell this sweep from another one. Move it aside."
        )
    drifted = [
        f"{field}: {stored.get(field)!r} on disk against {value!r} for this launch"
        for field, value in sorted(identity.items())
        if stored.get(field) != value
    ]
    unknown = sorted(set(stored) - set(identity))
    if unknown:
        drifted.append(
            f"the header also pins {unknown}, which this launch does not compute, so the identity "
            f"itself changed since the file was written"
        )
    if drifted:
        raise ValueError(
            f"refusing to resume {path}: its header describes a different sweep. "
            f"{'; '.join(drifted)}. Its records were drawn from another pool, model, sampler or "
            f"sampling budget, so keeping them would file two measurements under one trace with "
            f"every count still adding up. Move the file aside to sweep this pool from scratch, or "
            f"point --out-dir somewhere else."
        )


def write_partial_header(path: Path, *, identity: Mapping[str, Any]) -> None:
    """Open a partial file with the identity every later session is checked against.

    Truncates, and is only ever called for a path with no usable header: either nothing is there, or
    what is there is a torn header alone, which is a death between creating the file and writing its
    first record and so carries no finished work.
    """
    header = {
        "record_kind": PARTIAL_HEADER_RECORD_KIND,
        "written_at": datetime.now(UTC).isoformat(),
        "identity_sha256": identity_digest(identity),
        "identity": dict(identity),
        **git_provenance(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_partial_lines(path, [header], mode="w")


def _write_partial_lines(
    path: Path, records: Sequence[Mapping[str, Any]], *, mode: str = "a"
) -> None:
    """Append JSON lines to a partial file and force them onto the disk before returning.

    `flush` plus `fsync` rather than trusting the close, because the launch kit syncs this directory
    to S3 on an interval WHILE the sweep runs: a record still in the interpreter's buffer is a record
    that dies with the box. One fsync per chunk, against a chunk that took minutes of a rented card
    to decode.
    """
    with path.open(mode, encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def truncate_a_torn_tail(path: Path, *, complete_bytes: int) -> None:
    """Cut a torn trailing line off a partial file, so this session's appends land on a line boundary.

    The repair the reader could not do. A torn tail was dropped from the parse and counted but left on
    disk, so the relaunch's own session line went straight onto the torn bytes in append mode and made
    one fused interior line. That launch reported nothing wrong and could run to the end; the launch
    AFTER it refused the whole file by name (`_partial_line`), every record the earlier sessions had
    decoded included. Two deaths on a 632-row rented sweep, not one, which is why a single relaunch
    never showed it.

    Called after the read has counted the torn line and never before, because that count is what the
    trace's resume block reports as dropped; and at the byte offset of the last newline, never a
    character count, because the records are written with `ensure_ascii=False`.
    """
    with path.open("r+b") as handle:
        handle.truncate(complete_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    logger.info(f"sweep partial: cut a torn trailing line off {path} at {complete_bytes} bytes")


def retire_a_completed_partial(partial_path: Path, *, timestamp: str) -> Path:
    """Rename a partial file out of the resume path, once the trace it fed is on disk.

    A completed partial left in place made the identical command into a no-op that still wrote a fresh
    timestamped trace, corpus and selection summary over the previous session's completions: a
    duplicate that reads as a new measurement, and a second corpus file that makes
    `games.plans.resolve_single_corpus` refuse the directory as ambiguous. The only tells were a
    `resumed=632 generated=0` log line and the trace's own `resume.n_sessions`.

    Renamed rather than deleted, because the records are the run's raw material, and to a name the
    launch kit's `*.jsonl` count under `partial/` does not match. After the trace is written and never
    before: a death between the last record's flush and the trace write has to stay resumable, which
    is the whole reason the partial file exists.

    The trace's own resume block therefore names the live path rather than this one, the trace having
    been written first; the retired name is that path plus this suffix and the trace's own timestamp.
    """
    retired = partial_path.with_name(f"{partial_path.name}{RETIRED_PARTIAL_SUFFIX}{timestamp}")
    partial_path.rename(retired)
    logger.info(
        f"sweep partial: retired {partial_path} to {retired}; a relaunch here sweeps afresh"
    )
    return retired


def partial_sweep_lock_path(partial_path: Path) -> Path:
    """Name the lock that says a live sweep owns this partial file.

    Beside the file rather than inside it, and suffixed after `.jsonl` so the launch kit's `*.jsonl`
    count under `partial/` cannot read the lock as records.
    """
    return partial_path.with_name(f"{partial_path.name}{PARTIAL_LOCK_SUFFIX}")


@contextlib.contextmanager
def partial_sweep_lock(partial_path: Path) -> Generator[Path]:
    """Hold a partial file for one sweep, refusing a second one that would interleave with it.

    Nothing stopped two sweeps sharing an out-dir, and the operator case is ordinary: a box that looks
    hung, so the kit is relaunched while the old vLLM engine is still decoding into the same file. Both
    processes append, their bytes interleave INSIDE a line, and every later launch refuses the file the
    way a torn tail used to make it refuse -- with neither process reporting anything wrong.

    A lock file and nothing more: no daemon, no lease, no fcntl. Created in `x` mode, which is
    `O_CREAT | O_EXCL`, so two launches racing cannot both believe they own it, and it names its holder
    so the refusal can say which process to kill. Released when the pass ends, however it ends.
    """
    path = partial_sweep_lock_path(partial_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _claim_the_partial(path)
    try:
        yield path
    finally:
        path.unlink()


def _claim_the_partial(path: Path) -> None:
    """Create the lock file, taking over one whose holder is gone and refusing one whose is alive."""
    if path.exists():
        stale = _stale_reason(path)  # Raises rather than returning, when the holder is still alive.
        logger.warning(f"sweep partial: taking over a stale lock, {path} is {stale}")
        path.unlink()
    holder = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "acquired_at": datetime.now(UTC).isoformat(),
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(holder, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _stale_reason(path: Path) -> str:
    """Say why an existing lock may be taken over, and raise when it may not.

    A lock outlives its holder whenever the box died with it held, which on a rented box is the
    ordinary death, so a lock on its own may not wedge a relaunch -- that would turn this from a guard
    into the thing an operator has to work around at hour three. The one case that must refuse is a
    holder still running on THIS host, which is the second live writer.
    """
    held = json.loads(path.read_text(encoding="utf-8"))
    pid = int(held["pid"])
    hostname = str(held["hostname"])
    if hostname != socket.gethostname():
        return f"held on {hostname!r}, which is not this host"
    if not _process_is_running(pid):
        return f"held by pid {pid}, which is no longer running"
    raise ValueError(
        f"refusing to sweep into {path.parent}: {path} is held by pid {pid} on {hostname}, which is "
        f"still running, so that sweep is still appending to this partial file. Two writers interleave "
        f"their bytes inside a line and poison every later launch. Wait for it, kill it, or point "
        f"--out-dir somewhere else."
    )


def _process_is_running(pid: int) -> bool:
    """Say whether a pid is live on this host, which is what tells a stale lock from a held one."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Another user's process: the signal is refused, which is itself proof it is running.
        return True
    return True


def _partial_session_record(
    *, records_resumed: int, records_dropped: int, records_reparsed_changed: int
) -> dict[str, Any]:
    """Record one launch's entry in a partial file: what it inherited, and which code inherited it.

    The revision travels per session because a resumed sweep mixes code states across its records
    (memory: resumed runs mix code states), and a reader of the finished trace has to be able to see
    that from the trace. Written when the session starts rather than when it finishes, so a session
    that died is still named by the next one: the count is how many launches touched this pool, and a
    death is exactly the event that makes it worth counting.

    `records_reparsed_changed` is the mixture measured rather than inferred from the shas: how many of
    the records this session inherited had a verdict move when it re-derived them under its own parser
    (`_reparse_resumed_records`). Zero is the ordinary case and says the code that matters did not
    move; anything above it names a resume whose corpus would have been selected under two parsers.
    """
    return {
        "record_kind": PARTIAL_SESSION_RECORD_KIND,
        "started_at": datetime.now(UTC).isoformat(),
        "records_resumed": records_resumed,
        "records_dropped": records_dropped,
        "records_reparsed_changed": records_reparsed_changed,
        **git_provenance(),
    }


# What a resumed sweep is and is not, carried in the trace rather than left for a reader to work out.
SWEEP_RESUME_SEEDING_NOTE = (
    "no per-request seed: Backend.generate takes none, and a paged engine's batch composition moves "
    "with whatever remainder a session decodes, so the prompts a later session drew are a fresh draw "
    "from the same policy distribution. A trace whose n_sessions is above one is the same measurement "
    "statistically as one unbroken run, and not the same bytes."
)


def sweep_resume_block(
    sessions: Sequence[Mapping[str, Any]], *, partial_path: Path | None
) -> dict[str, Any]:
    """Build the trace meta's `resume` block: every session, and the latest one's counts lifted up.

    Shaped like `games.evals`'s block so one reader answers both: `n_sessions == 1` is a sweep that
    ran unbroken, and anything above it names each launch, what it kept and what it dropped.

    Every session carries its own `git_sha` and `git_tree_dirty`, so a trace states which code states
    a resumed sweep spans. The identity deliberately does not pin the sha -- relaunching on a later
    commit is the ordinary case, and refusing it would make the resume useless -- so the artifact
    records the mixture instead, beside the `records_reparsed_changed` count that says whether the
    mixture reached any verdict.
    """
    latest = sessions[-1]
    return {
        "n_sessions": len(sessions),
        "records_resumed": int(latest["records_resumed"]),
        "records_dropped": int(latest["records_dropped"]),
        "records_reparsed_changed": int(latest["records_reparsed_changed"]),
        "partial_path": None if partial_path is None else str(partial_path),
        "seeding": SWEEP_RESUME_SEEDING_NOTE,
        "sessions": [dict(session) for session in sessions],
    }


@dataclass(frozen=True, slots=True)
class SweepPass:
    """One launch's sweep: every record in pool order, and what this session did and did not draw.

    `records` is in the POOL's order however the work was split across sessions, so a resumed trace
    and an unbroken one are the same file in the same order. The three counts are the accounting the
    repo's resume rule asks for: what was kept from disk, what this session generated, and what it
    dropped as torn -- counted apart, because "skipped" and "resumed" reading as one number is how a
    resume reports work it never did.

    `n_reparsed_changed` is the fourth, and it counts a different thing: how many kept records had a
    verdict move when this session re-derived them under its own parser (`_reparse_resumed_records`).
    """

    records: tuple[PromptSweepRecord, ...]
    n_resumed: int
    n_generated: int
    n_torn: int
    sessions: tuple[dict[str, Any], ...]
    n_reparsed_changed: int = 0


def resumable_sweep(  # noqa: PLR0913 - the sweep's inputs plus where it persists them
    backend: Backend,
    rows: Sequence[Row],
    *,
    samples_per_prompt: int,
    prefilled_think: bool,
    chunk_size: int | None = None,
    record_kind: str = SWEEP_RECORD_KIND,
    partial_path: Path | None = None,
    identity: Mapping[str, Any] | None = None,
) -> SweepPass:
    """Sample every row that is not already on disk, appending each finished prompt as it lands.

    The resumable form of `sweep_prompts`, and the one `main` runs. A 632-row pool at eight samples
    and a 32,768-token budget is 3-6 hours of a rented card, and before this nothing reached disk
    until the last chunk returned, so a box that died at hour three had swept nothing at all. Now a
    prompt's record is written the moment its samples are in hand, a relaunch keeps those records and
    decodes only what is missing, and the launch kit's interval sync to S3 carries them off the box
    for free.

    Granularity is a whole prompt: a partial record holds all `samples_per_prompt` samples of one row
    or it is not there. The decode itself still runs at whatever width the card can hold, so the
    flush lands on row boundaries rather than chunk boundaries -- which is what keeps an OOM-halved
    width from writing half a prompt's samples as though they were all of them.

    Three things happen around the persistence rather than inside the decode, and each closes a way
    the file could be poisoned or misread. The partial is LOCKED for the pass, so a second sweep into
    the same out-dir is refused instead of interleaving its bytes into a line (`partial_sweep_lock`). A
    torn trailing line is cut off before this session's first append, so the append lands on a line
    boundary rather than fusing onto it (`truncate_a_torn_tail`). And every resumed record's verdicts
    are re-derived from its stored completions under this session's parser, because a resume spans code
    states and selection reads those verdicts (`_reparse_resumed_records`).

    `partial_path` and `identity` are given together or not at all. Without them this is the old
    behaviour, nothing is persisted and nothing is resumed, which is what `sweep_frozen_opponent` and
    the offline tests want.
    """
    if (partial_path is None) != (identity is None):
        raise ValueError(
            f"pass partial_path and identity together or neither, got {partial_path=} "
            f"{identity is None=}: a partial file with no identity could not be refused when a later "
            f"launch changed the sampler, and an identity with nowhere to write is not persistence."
        )
    if samples_per_prompt < MIN_SAMPLES_FOR_SPREAD:
        raise ValueError(
            f"a sweep with {samples_per_prompt=} cannot show a split action distribution, "
            "which is the only thing it is for"
        )
    if not rows:
        raise ValueError("no rows to sweep")

    with contextlib.ExitStack() as holding_the_partial:
        if partial_path is not None:
            holding_the_partial.enter_context(partial_sweep_lock(partial_path))
        inherited = (
            read_partial_sweep(partial_path, identity=identity, rows=rows, record_kind=record_kind)
            if partial_path is not None and identity is not None
            else PartialSweep((), (), 0, has_header=False)
        )
        kept, n_reparsed_changed = _reparse_resumed_records(
            inherited.records, prefilled_think=prefilled_think
        )
        resumed_by_id = {record.prompt_id: record for record in kept}
        pending = [row for row in rows if _require_str(row, PROMPT_ID_COLUMN) not in resumed_by_id]
        session = _partial_session_record(
            records_resumed=len(resumed_by_id),
            records_dropped=inherited.n_torn,
            records_reparsed_changed=n_reparsed_changed,
        )
        sessions = (*inherited.sessions, session)
        if partial_path is not None and identity is not None:
            if not inherited.has_header:
                write_partial_header(partial_path, identity=identity)
            elif inherited.n_torn:
                truncate_a_torn_tail(partial_path, complete_bytes=inherited.complete_bytes)
            _write_partial_lines(partial_path, [session])
            logger.info(
                f"sweep partial: path={partial_path} session={len(sessions)} "
                f"identity={identity_digest(identity)[:16]} resumed={len(resumed_by_id)} "
                f"pending={len(pending)} torn={inherited.n_torn} "
                f"reparsed_changed={n_reparsed_changed}"
            )

        generated = _decode_pending_prompts(
            backend,
            pending,
            samples_per_prompt=samples_per_prompt,
            prefilled_think=prefilled_think,
            chunk_size=chunk_size,
            record_kind=record_kind,
            partial_path=partial_path,
        )
    by_id = {**resumed_by_id, **{record.prompt_id: record for record in generated}}
    records = tuple(by_id[_require_str(row, PROMPT_ID_COLUMN)] for row in rows)
    logger.info(
        f"sweep pass: resumed={len(resumed_by_id)} generated={len(generated)} "
        f"torn={inherited.n_torn} n_prompts={len(records)}"
    )
    _log_sweep_totals(records)
    return SweepPass(
        records=records,
        n_resumed=len(resumed_by_id),
        n_generated=len(generated),
        n_torn=inherited.n_torn,
        sessions=sessions,
        n_reparsed_changed=n_reparsed_changed,
    )


def _reparse_resumed_records(
    records: Sequence[PromptSweepRecord], *, prefilled_think: bool
) -> tuple[tuple[PromptSweepRecord, ...], int]:
    """Re-derive every resumed record's verdicts from its stored completions, and count what moved.

    A resume mixes code states by construction: the identity does not pin the commit, because
    relaunching on a later one is the ordinary case, and a parse bug found at hour three is itself one
    of the reasons an operator kills a sweep. Selection then reads exactly the derived fields --
    `parsed`, `selection_score` and the aggregates over them -- so keeping the ones on disk would let a
    corpus be chosen under two parsers with nothing saying so.

    The completions are the raw material and are kept byte for byte; everything computed from them is
    computed again here. `_parse_sample` is a pure function of the completion, the row and
    `prefilled_think`, so on an unchanged parser this reproduces the stored record exactly and the
    count is zero -- which is what makes it safe to do unconditionally rather than only when the shas
    differ, since two dirty trees at one sha are not the same code either.
    """
    reparsed = tuple(
        _record_for_row(
            record.row,
            [sample.completion for sample in record.samples],
            prefilled_think=prefilled_think,
            record_kind=record.record_kind,
        )
        for record in records
    )
    changed = sum(
        1
        for stored, fresh in zip(records, reparsed, strict=True)
        if fresh.to_json_dict() != stored.to_json_dict()
    )
    if changed:
        logger.warning(
            f"sweep partial: {changed} of {len(records)} resumed records had a verdict move under "
            f"this session's parser; the trace's resume block records the count and every session's "
            f"revision, and the corpus below is selected on THIS session's verdicts"
        )
    return reparsed, changed


def _decode_pending_prompts(  # noqa: PLR0913 - the decode's inputs plus where it persists them
    backend: Backend,
    rows: Sequence[Row],
    *,
    samples_per_prompt: int,
    prefilled_think: bool,
    chunk_size: int | None,
    record_kind: str,
    partial_path: Path | None,
) -> list[PromptSweepRecord]:
    """Decode the rows still missing, writing each prompt's record out as its samples complete.

    The backend takes the **raw untemplated** prompt: `HFBackend` applies the chat template itself,
    and `games/dataset.py` stores the templated string for TRL. Passing the templated one here would
    template it twice, which is easy to miss and changes the policy being measured.

    There is no `n` parameter on the backend protocol, so N samples of a prompt are N copies of it in
    the request list, and the flattened list is decoded in chunks because `HFBackend.generate` makes
    exactly one `model.generate` call for whatever it is handed. Completions are buffered only until
    a whole prompt's worth is in hand, so a chunk that ends mid-prompt writes nothing for that prompt
    and the next chunk completes it.
    """
    if not rows:
        return []
    prompts = [_require_str(row, PROMPT_COLUMN) for row in rows for _ in range(samples_per_prompt)]
    chunk = sweep_chunk_size(
        max_new_tokens=_backend_max_new_tokens(backend),
        n_sequences=len(prompts),
        requested=chunk_size,
        model_id=local_decode_model_id(backend),
        schedules_own_batch=backend_schedules_own_batch(backend),
        # A self-scheduling engine is handed the whole list unless this sweep is persisting per
        # chunk, in which case the handover is capped so a death costs one chunk (see
        # `games.chunked_decode.SELF_SCHEDULING_CHUNK_SEQUENCES`).
        self_scheduling_cap=(
            None
            if partial_path is None
            else self_scheduling_chunk_cap(samples_per_prompt=samples_per_prompt)
        ),
    )
    records: list[PromptSweepRecord] = []
    buffered: list[str] = []
    for batch in iter_decoded_chunks(backend, prompts, chunk_size=chunk):
        buffered.extend(batch)
        whole_prompts = len(buffered) // samples_per_prompt
        if not whole_prompts:
            continue
        fresh = [
            _record_for_row(
                rows[len(records) + index],
                buffered[index * samples_per_prompt : (index + 1) * samples_per_prompt],
                prefilled_think=prefilled_think,
                record_kind=record_kind,
            )
            for index in range(whole_prompts)
        ]
        if partial_path is not None:
            _write_partial_lines(partial_path, [record.to_json_dict() for record in fresh])
        records.extend(fresh)
        buffered = buffered[whole_prompts * samples_per_prompt :]
    if buffered:
        raise RuntimeError(
            f"the backend returned {len(prompts) - len(buffered)} completions for {len(prompts)} "
            f"prompts, leaving {len(buffered)} that are not a whole prompt's {samples_per_prompt}; "
            f"the sample-to-row alignment the whole trace depends on is broken"
        )
    if len(records) != len(rows):
        raise RuntimeError(
            f"decoded {len(records)} records for {len(rows)} prompts; the sample-to-row alignment "
            f"the whole trace depends on is broken"
        )
    return records


def _record_for_row(
    row: Row, completions: Sequence[str], *, prefilled_think: bool, record_kind: str
) -> PromptSweepRecord:
    """Parse one prompt's completions into the record both the partial file and the trace hold."""
    return PromptSweepRecord(
        prompt_id=_require_str(row, PROMPT_ID_COLUMN),
        grading=_require_str(row, GRADING_COLUMN),
        row=dict(row),
        samples=tuple(
            _parse_sample(completion, row, prefilled_think=prefilled_think)
            for completion in completions
        ),
        record_kind=record_kind,
    )


def sweep_prompts(  # noqa: PLR0913
    backend: Backend,
    rows: Sequence[Row],
    *,
    samples_per_prompt: int,
    prefilled_think: bool,
    chunk_size: int | None = None,
    record_kind: str = SWEEP_RECORD_KIND,
) -> list[PromptSweepRecord]:
    """Sample every row `samples_per_prompt` times and parse each completion.

    The whole-pool view over `resumable_sweep`, for callers with nothing to resume: the frozen
    opponent, whose pass is short and hosted, and the offline tests. Nothing reaches disk here.
    """
    return list(
        resumable_sweep(
            backend,
            rows,
            samples_per_prompt=samples_per_prompt,
            prefilled_think=prefilled_think,
            chunk_size=chunk_size,
            record_kind=record_kind,
        ).records
    )


def _backend_max_new_tokens(backend: Backend) -> int:
    """Read the backend's output-token budget, for the chunk-size arithmetic.

    The protocol guarantees only `model_id` and `transport`, and the two real backends carry
    differently-named budgets on differently-typed sampling configs, so this reads whichever is
    present and falls back to the training sampler's budget for a backend that samples nothing.
    """
    sampling = getattr(backend, "sampling", None)
    for attribute in ("max_new_tokens", "max_tokens"):
        budget = getattr(sampling, attribute, None)
        if isinstance(budget, int):
            return budget
    return required_completion_budget(backend.model_id)


def _log_sweep_totals(records: Sequence[PromptSweepRecord]) -> None:
    """Log the sweep-wide parse and truncation rates, which shape every number downstream."""
    n_samples = sum(record.n_samples for record in records)
    if not n_samples:
        return
    parse_failures = sum(record.n_parse_failures for record in records)
    truncated = sum(record.n_truncated_thinking for record in records)
    logger.info(
        f"sweep done: n_prompts={len(records)} {n_samples=} "
        f"parse_failure_rate={parse_failures / n_samples:.3f} "
        f"truncated_thinking_rate={truncated / n_samples:.3f}"
    )


def judge_prompt(  # noqa: PLR0911
    record: PromptSweepRecord,
    *,
    min_coop: float = DEFAULT_MIN_COOP,
    max_coop: float = DEFAULT_MAX_COOP,
    min_split_std: float = DEFAULT_MIN_SPLIT_STD,
    min_parseable_fraction: float = DEFAULT_MIN_PARSEABLE_FRACTION,
) -> PromptVerdict:
    """Decide one prompt on its own samples, before any counterbalanced-pair coupling.

    Binary-action gradings key on the cooperation rate lying strictly inside the band; the numeric
    and iterated gradings produce a continuous per-sample score, where a rate says nothing and the
    spread is what GRPO's advantage can actually see. That difference is why wave 1's ceiling-bound
    arms died at selection and the one continuous-score game survived nearly intact: a spread floor
    cannot delete a prompt for being answered correctly, only for being answered unanimously.
    """

    def verdict(*, keep: bool, reason: DropReason) -> PromptVerdict:
        return PromptVerdict(
            prompt_id=record.prompt_id,
            keep=keep,
            reason=reason,
            coop_fraction=record.coop_fraction,
            score_std=record.score_std,
            parseable_fraction=record.parseable_fraction,
        )

    if record.parseable_fraction < min_parseable_fraction:
        return verdict(keep=False, reason=DropReason.TOO_FEW_PARSEABLE)

    if record.scores_a_binary_action:
        coop_fraction = record.coop_fraction
        if coop_fraction is None:
            return verdict(keep=False, reason=DropReason.TOO_FEW_PARSEABLE)
        if coop_fraction < min_coop:
            return verdict(keep=False, reason=DropReason.COOP_FRACTION_BELOW_MIN)
        if coop_fraction > max_coop:
            return verdict(keep=False, reason=DropReason.COOP_FRACTION_ABOVE_MAX)
        return verdict(keep=True, reason=DropReason.KEPT_MIXED)

    if record.score_std < min_split_std:
        return verdict(keep=False, reason=DropReason.SCORE_SPREAD_BELOW_MIN)
    return verdict(keep=True, reason=DropReason.KEPT_MIXED)


def pair_identity(row: Row) -> tuple[object, ...]:
    """Return the key two counterbalanced renderings of one scenario share.

    The two orientations differ only in which neutral label names the cooperative option, so they
    agree on every other column -- `reskin_id` (the scenario id), the payoff variant, the game,
    and both labels. Keying on everything the swap leaves alone, rather than on a named subset,
    means an added column tightens the key instead of being silently ignored.
    """
    return tuple(
        sorted(
            (column, _json_key(value))
            for column, value in row.items()
            if column not in COLUMNS_VARYING_WITHIN_A_PAIR
        )
    )


def is_counterbalanced(row: Row) -> bool:
    """Say whether the row is one of two label orientations rather than a lone prompt.

    The unilateral-split rows carry empty labels: they have no pair of options to swap, so they
    are singletons by construction and must not read as a partner gone missing.
    """
    return bool(_require_str(row, "label_a") and _require_str(row, "label_b"))


def _json_key(value: object) -> object:
    """Make a row value usable inside a dict key, since a row may hold a list or a dict."""
    if isinstance(value, str | int | float | bool | None):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def judge_prompts(
    records: Sequence[PromptSweepRecord],
    *,
    min_coop: float = DEFAULT_MIN_COOP,
    max_coop: float = DEFAULT_MAX_COOP,
    min_split_std: float = DEFAULT_MIN_SPLIT_STD,
    min_parseable_fraction: float = DEFAULT_MIN_PARSEABLE_FRACTION,
) -> list[PromptVerdict]:
    """Judge every prompt, then couple the counterbalanced pairs so they stand or fall together.

    Keeping one orientation of a frame and dropping the other trains on a corpus where the
    cooperative option sits in a particular position more often than chance, so a position bias
    the counterbalancing existed to cancel becomes part of the training signal instead.
    """
    verdicts = {
        record.prompt_id: judge_prompt(
            record,
            min_coop=min_coop,
            max_coop=max_coop,
            min_split_std=min_split_std,
            min_parseable_fraction=min_parseable_fraction,
        )
        for record in records
    }
    _require_unique_prompt_ids(records)

    unpaired: list[str] = []
    for identity, group in _group_by_pair(records).items():
        if len(group) == 1:
            if is_counterbalanced(group[0].row):
                unpaired.append(group[0].prompt_id)
            continue
        if len(group) > LABEL_ORIENTATIONS_PER_SCENARIO:
            raise ValueError(
                f"{len(group)} rows share the counterbalanced-pair key {identity!r}, but a "
                "scenario has exactly two label orientations; the corpus either repeats a "
                "rendering or reuses one label pair across frames, and coupling them would "
                f"drop unrelated prompts together. prompt_ids: {[r.prompt_id for r in group]}"
            )
        if all(verdicts[record.prompt_id].keep for record in group):
            continue
        for record in group:
            if verdicts[record.prompt_id].keep:
                # At most one rewrite per pair, and a frozen PromptVerdict cannot be edited in
                # place, which is what keeps a verdict from drifting after it is recorded.
                verdicts[record.prompt_id] = (
                    dataclasses.replace(  # HARNESS-SCAN-EXEMPT-dataclass-replace-in-loop
                        verdicts[record.prompt_id], keep=False, reason=DropReason.PARTNER_DROPPED
                    )
                )
    if unpaired:
        raise ValueError(
            f"{len(unpaired)} prompts have no counterbalanced partner: {unpaired}. Each scenario "
            f"is rendered in both label orientations precisely so that a preference for the "
            f"first-listed option cannot masquerade as a preference for cooperating, and an orphan "
            f"orientation keeps its own keep verdict and walks that position bias into the "
            f"training corpus. This warned rather than raised until 2026-08-19, which could not "
            f"stop it. Fix the row list that dropped an orientation; every game games.prompts "
            f"renders today ships both."
        )
    return [verdicts[record.prompt_id] for record in records]


def _require_unique_prompt_ids(records: Sequence[PromptSweepRecord]) -> None:
    """Raise on a duplicate prompt_id, which would silently collapse verdicts and corpus rows."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.prompt_id in seen:
            duplicates.add(record.prompt_id)
        seen.add(record.prompt_id)
    if duplicates:
        raise ValueError(
            f"prompt_id must be unique across the sweep; repeated: {sorted(duplicates)}"
        )


def _group_by_pair(
    records: Sequence[PromptSweepRecord],
) -> dict[tuple[object, ...], list[PromptSweepRecord]]:
    """Group records by counterbalanced-pair identity, preserving input order within a group."""
    groups: dict[tuple[object, ...], list[PromptSweepRecord]] = {}
    for record in records:
        groups.setdefault(pair_identity(record.row), []).append(record)
    return groups


def select_mixed_prompts(
    records: Sequence[PromptSweepRecord],
    *,
    min_coop: float = DEFAULT_MIN_COOP,
    max_coop: float = DEFAULT_MAX_COOP,
    min_split_std: float = DEFAULT_MIN_SPLIT_STD,
    min_parseable_fraction: float = DEFAULT_MIN_PARSEABLE_FRACTION,
) -> list[Row]:
    """Return the training rows for the prompts whose action distribution was genuinely split.

    Rows are copies, so a caller filling in `opp_coop_prob` cannot reach back into the sweep
    trace that has already been written.
    """
    verdicts = judge_prompts(
        records,
        min_coop=min_coop,
        max_coop=max_coop,
        min_split_std=min_split_std,
        min_parseable_fraction=min_parseable_fraction,
    )
    return [
        dict(record.row) for record, verdict in zip(records, verdicts, strict=True) if verdict.keep
    ]


def selection_summary(
    records: Sequence[PromptSweepRecord], verdicts: Sequence[PromptVerdict]
) -> dict[str, object]:
    """Count what was kept and dropped, by reason, with the per-prompt verdicts underneath."""
    reasons: dict[str, int] = {}
    for verdict in verdicts:
        reasons[verdict.reason] = reasons.get(verdict.reason, 0) + 1
    n_samples = sum(record.n_samples for record in records)
    return {
        "n_prompts": len(records),
        "n_kept": sum(1 for verdict in verdicts if verdict.keep),
        "n_dropped": sum(1 for verdict in verdicts if not verdict.keep),
        "counts_by_reason": dict(sorted(reasons.items())),
        "n_samples": n_samples,
        "parse_failure_rate": (
            sum(record.n_parse_failures for record in records) / n_samples if n_samples else None
        ),
        "truncated_thinking_rate": (
            sum(record.n_truncated_thinking for record in records) / n_samples
            if n_samples
            else None
        ),
        "verdicts": [dataclasses.asdict(verdict) for verdict in verdicts],
    }


def sweep_frozen_opponent(  # noqa: PLR0913
    rows: Sequence[Row],
    *,
    model_id: str,
    samples: int,
    prefilled_think: bool,
    backend: Backend | None = None,
    chunk_size: int | None = None,
) -> list[PromptSweepRecord]:
    """Sample the frozen opponent on the same prompts the policy sees, keeping every completion.

    The opponent reads the identical sheet: the vs-frozen arms are symmetric games described to
    both sides the same way, so a separately-worded opponent prompt would measure a different
    game from the one being graded. `backend` is injectable so this is testable offline; left
    unset it builds the hosted Bedrock backend named by `model_id`.

    `prefilled_think` carries no default on purpose. It answers "which thinking-tag convention do
    THESE completions follow", and the answer for a hosted opponent is not the answer for the local
    policy -- see `FROZEN_OPPONENT_PREFILLED_THINK`. A default here is what let `main` hand the
    opponent the policy checkpoint's value and make every opponent completion unparseable.
    """
    if not rows:
        raise ValueError("no rows need a frozen opponent")
    wrong_grading = sorted(
        {
            grading
            for row in rows
            if (grading := _require_str(row, GRADING_COLUMN)) != GRADING_VS_FIXED_MIX
        }
    )
    if wrong_grading:
        raise ValueError(
            f"only {GRADING_VS_FIXED_MIX!r} rows consume a cached opponent probability, but rows "
            f"with grading {wrong_grading} were passed; filter before sampling rather than paying "
            "for calls whose result nothing reads"
        )
    opponent = backend if backend is not None else build_backend("bedrock", model_id)
    if opponent.model_id != model_id:
        raise ValueError(
            f"frozen-opponent backend serves {opponent.model_id!r} but the cache would record "
            f"{model_id!r}; the provenance that keeps opp_coop_prob interpretable would be wrong"
        )
    return sweep_prompts(
        opponent,
        rows,
        samples_per_prompt=samples,
        prefilled_think=prefilled_think,
        chunk_size=chunk_size,
        record_kind=FROZEN_OPPONENT_RECORD_KIND,
    )


def frozen_opponent_coop_probs(
    records: Sequence[PromptSweepRecord], *, min_parseable_fraction: float
) -> dict[str, float]:
    """Reduce the opponent's sweep to one cooperation probability per prompt.

    Raises, rather than dropping or defaulting, for any prompt whose opponent answers mostly failed
    to parse. There is no honest default: a 0.5 stand-in would grade the arm against a coin the
    opponent never flipped, and the training loop cannot re-sample because the whole point of the
    cache is that it never calls out.

    The floor is the same one the policy sweep judges prompts by, and it applies here for a stronger
    reason. A policy prompt whose samples mostly failed to parse is dropped as a measurement
    nobody trusts; an opponent prompt in that state is consumed as ground truth, and one surviving
    answer of eight pins the arm's opponent distribution at exactly 0.0 or 1.0 for the whole run.
    """
    empty = sorted(
        (record.prompt_id, len(record.parsed_samples), record.n_samples)
        for record in records
        if record.parseable_fraction < min_parseable_fraction
    )
    if empty:
        raise ValueError(
            f"the frozen opponent produced no parseable answer, or too few for a mix to mean "
            f"anything at a {min_parseable_fraction} floor, for {len(empty)} prompts, so their "
            f"opponent distribution is unknown rather than uncertain. Each entry below is "
            f"(prompt_id, parsed, sampled): {empty}"
        )
    probs: dict[str, float] = {}
    for record in records:
        coop_fraction = record.coop_fraction
        if coop_fraction is None:
            raise ValueError(
                f"prompt {record.prompt_id!r} yields no cooperation rate under grading "
                f"{record.grading!r}, so it cannot supply an opponent probability"
            )
        probs[record.prompt_id] = coop_fraction
    return probs


def sample_frozen_opponent(  # noqa: PLR0913
    rows: Sequence[Row],
    *,
    model_id: str,
    samples: int,
    prefilled_think: bool,
    min_parseable_fraction: float,
    backend: Backend | None = None,
    chunk_size: int | None = None,
) -> dict[str, float]:
    """Return each prompt's frozen-opponent cooperation probability, keyed by prompt_id.

    The convenience view over `sweep_frozen_opponent`; call that directly to keep the opponent's
    completions for the trace, which is the artefact this run should also be producing.
    """
    return frozen_opponent_coop_probs(
        sweep_frozen_opponent(
            rows,
            model_id=model_id,
            samples=samples,
            backend=backend,
            prefilled_think=prefilled_think,
            chunk_size=chunk_size,
        ),
        min_parseable_fraction=min_parseable_fraction,
    )


def fill_opponent_probs(rows: Sequence[Row], coop_probs: dict[str, float]) -> list[Row]:
    """Write the cached opponent probability into each row, refusing to leave one unfilled.

    A vs-fixed-mix row reaching training with the sentinel would make the reward function raise
    mid-run, which is a wasted job; catching it here costs nothing.
    """
    filled: list[Row] = []
    for row in rows:
        prompt_id = _require_str(row, PROMPT_ID_COLUMN)
        if prompt_id not in coop_probs:
            raise KeyError(
                f"no frozen-opponent probability for prompt {prompt_id!r}; a vs-fixed-mix row "
                "with the sentinel would raise inside the reward function mid-training"
            )
        filled.append({**row, OPP_COOP_PROB_COLUMN: coop_probs[prompt_id]})
    return filled


def backend_provenance(backend: Backend) -> dict[str, object]:
    """Describe which model answered and how, for the trace's meta record.

    The sampling config is read reflectively because the protocol does not carry one and the two
    real backends use different config types on purpose (`BedrockSamplingConfig` has no `top_k`
    and no greedy switch, so sharing one type would drop both in silence).

    The engine block is every setting the engine was CONSTRUCTED with, which is a different statement
    from the meta's record of the flags: that one says what was asked for, and a caller can derive an
    engine kwarg from a checkpoint without a flag. Empty for a backend that takes none. Only the
    subset that moves the sampled distribution gates a resume (`engine_identity_fields`).
    """
    sampling = getattr(backend, "sampling", None)
    return {
        "model_id": backend.model_id,
        "transport": backend.transport,
        "sampling": dataclasses.asdict(sampling) if dataclasses.is_dataclass(sampling) else None,  # pyright: ignore[reportArgumentType]
        ENGINE_PROVENANCE_KEY: _engine_settings(backend),
    }


def sweep_meta(  # noqa: PLR0913
    *,
    backend: Backend,
    args: argparse.Namespace,
    prompt_ids: Sequence[str],
    samples_per_prompt: int,
    prefilled_think: bool,
    rows_sha256: str | None,
    frozen_opponent: dict[str, object] | None = None,
    kernel_bridge: dict[str, object] | None = None,
    resume: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Build the trace's first line: everything needed to reproduce or date the sweep.

    `kernel_bridge` is here because most of a sweep's wall clock is decode, and whether decode got
    fla's fused kernel or the pure-torch fallback is otherwise invisible after the fact -- a sweep
    that took twice as long for that reason should say so in its own artifact instead of being
    re-litigated from a wall-clock number later.

    `prompt_ids` rather than a count, because the count alone cannot answer the question the number
    is used for. Sampling draws from one global RNG stream over the flattened prompt list, and the
    decode chunk boundaries move with the sequence count, so a scenario inserted upstream re-rolls
    every later prompt's samples even at the same `--seed`: a prompt kept before can cross a
    threshold and leave the corpus for reasons unrelated to the edit. Two corpora whose pool hash
    differs are therefore not a controlled comparison, and this is the only place that can say so.
    Per-prompt seeding is not the alternative -- `Backend.generate` takes no seed.

    `resume` is the same fact read the other way round, over sessions instead of pools. A sweep that
    died and was relaunched drew its remainder from a stream that started again, so the block names
    every session, what each kept from disk and what each dropped as torn, and states in the trace
    itself that a multi-session sweep is the same measurement statistically and not the same bytes
    (`sweep_resume_block`). None for a caller that persisted nothing, which is a sweep with no partial
    file rather than a sweep that ran unbroken.

    `frozen_opponent` describes the opponent the run INTENDS to sample, because this line is
    written before that pass makes a call: a trace naming an opponent and carrying no
    `frozen-opponent-sweep` records is a run whose opponent pass failed, and the completions above
    it are still the sweep they were.

    `rows_sha256` is passed in rather than taken here, and that is the whole point of the argument.
    This line is written after the sweep, hours after the load on a real pool, so a digest computed
    here would cover whatever the path holds by then -- and a `games.breadth_corpus build` into the
    same out-dir mid-sweep is the ordinary way an operator tweaks a spec. `_load_authored_rows`
    digests the bytes it parsed, which is what "a verdict is only ever readable against the file that
    hashes to it" has to mean.
    """
    if (args.rows is None) != (rows_sha256 is None):
        raise ValueError(
            f"rows_sha256={rows_sha256!r} does not match the pool source --rows={args.rows!r}: an "
            f"authored pool's digest is provenance for the bytes that were swept, so a generated "
            f"sweep has none and a `--rows` sweep must carry the one its load computed."
        )
    return {
        "record_kind": META_RECORD_KIND,
        "written_at": datetime.now(UTC).isoformat(),
        "game": args.game,
        # An authored pool's own provenance, beside the pool hash rather than instead of it: the hash
        # pins the id order the sampling stream depends on, the digest pins the prompts themselves.
        # Both are None for a generated pool, which this repository reproduces from the game id.
        "rows_path": None if args.rows is None else str(args.rows),
        "rows_sha256": rows_sha256,
        "grading": args.grading,
        "split": args.split,
        "label_print_order": args.label_print_order,
        "n_prompts": len(prompt_ids),
        "prompt_id_order_sha256": hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest(),
        "samples_per_prompt": samples_per_prompt,
        "prefilled_think": prefilled_think,
        "backend": backend_provenance(backend),
        "resume": resume,
        "frozen_opponent": frozen_opponent,
        "deltanet_kernel_bridge": kernel_bridge,
        "deltanet_kernel_paths": deltanet_kernel_paths(),
        "args": {key: _json_key(value) for key, value in sorted(vars(args).items())},
        **git_provenance(),
    }


def write_sweep_trace(
    path: Path, *, meta: dict[str, object], records: Iterable[PromptSweepRecord]
) -> None:
    """Write the meta record, then one record per prompt, as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def append_sweep_records(path: Path, records: Iterable[PromptSweepRecord]) -> None:
    """Append records to a trace already on disk, which is what keeps the policy sweep safe.

    The frozen opponent is sampled after the policy, over a hosted API that can throttle or fail,
    and its records used to be handed to `write_sweep_trace` together with the policy's -- so
    anything raising in between discarded every completion the sweep had just paid for. Each
    record carries its own `record_kind`, so one file holding both sweeps stays readable.
    """
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def write_corpus(path: Path, rows: Iterable[Row]) -> None:
    """Write the selected training rows as plain JSONL, one row per line and no meta line.

    Deliberately headerless: `games/dataset.py` feeds these straight into `Dataset.from_list`,
    and a meta line would arrive as a row with every column missing. The provenance lives in the
    selection summary written beside it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    """Read a JSONL file back into a list of records, blank lines skipped."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rows_file_digest(path: Path) -> str:
    """Return the sha256 of an authored rows file, as its own provenance in the sweep's meta record.

    The pool hash beside it covers the ORDER of the prompt ids, which is what the sampling stream
    depends on, and says nothing about the prompts themselves: two rows files could agree on every id
    while one of them carried an edited frame or a different counterpart paragraph. That is exactly the
    failure a `--rows` sweep can have and a `--game` sweep cannot, because a generated pool is
    reproduced from this repository while an authored one is a file on a box. So the digest is of the
    bytes, and a verdict is only ever readable against the file that hashes to it.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_stem(kind: str, args: argparse.Namespace, timestamp: str) -> str:
    """Build a filename stem that says what ran, on which model, and when.

    A non-canonical label print order is named here as well, because two sweeps of one game under
    two print orders are two different prompt sets and a filename that cannot tell them apart is
    how the 2026-08-19 rung miscount happened -- a pooled corpus whose name did not announce what
    was in it. Canonical stems are unchanged, so nothing already on disk is renamed by this.
    """
    return f"{kind}-{_pool_and_model_stem(args)}-{timestamp}"


def _pool_and_model_stem(args: argparse.Namespace) -> str:
    """Name the pool and the model a run swept, which is what every artifact of it shares.

    Split out of `_artifact_stem` because the partial file is named from it too, and that name may
    not carry a timestamp: a relaunch has to find the file the dead session was writing.
    """
    slug = str(args.model).rsplit("/", 1)[-1].replace(":", "-").replace(" ", "-")
    order = (
        ""
        if args.label_print_order == LABEL_PRINT_ORDER_CANONICAL
        else f"-{args.label_print_order}"
    )
    return f"{_pool_name(args)}{order}-{slug}"


def _pool_name(args: argparse.Namespace) -> str:
    """Name the pool a sweep drew from, for the artifact filenames: the game, or the rows file's stem.

    An authored pool spans several games by construction, so no game id could name it; its file's stem
    is the only name it has, and the meta record carries that file's digest beside it.
    """
    if args.game is not None:
        return str(args.game)
    return Path(str(args.rows)).stem


def _selection_thresholds(args: argparse.Namespace) -> dict[str, float]:
    """Collect the selection knobs, so a run's verdicts and its summary cannot disagree."""
    return {
        "min_coop": args.min_coop,
        "max_coop": args.max_coop,
        "min_split_std": args.min_split_std,
        "min_parseable_fraction": args.min_parseable_fraction,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline behavioural sweep at the training sampler, and selection of the prompts "
            "whose action distribution is mixed enough for GRPO to learn from."
        )
    )
    backend_cli.add_backend_args(parser)
    parser.add_argument(
        "--game", default=None, help="Game id to generate prompt rows for; excludes --rows."
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help=(
            "Sweep the rows in this JSONL file instead of generating one game's. For a corpus whose "
            "prompts span several games or several counterpart framings, which no single --game can "
            "render; excludes --game."
        ),
    )
    parser.add_argument(
        "--grading",
        required=True,
        type=grading_cli_value,
        help=(
            f"Reward scheme the rows will be graded by: one of {list(GRADING_CHOICES)}, or a member "
            f"of the care family (care-alpha-<alpha>)."
        ),
    )
    parser.add_argument(
        "--model", required=True, help="HuggingFace id or local path of the policy to sweep."
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help=f"Completions drawn per prompt (default: {DEFAULT_SAMPLES_PER_PROMPT}).",
    )
    parser.add_argument(
        "--split", default=DEFAULT_SPLIT, help=f"Corpus split (default: {DEFAULT_SPLIT})."
    )
    parser.add_argument(
        "--label-print-order",
        choices=LABEL_PRINT_ORDERS,
        default=LABEL_PRINT_ORDER_CANONICAL,
        help=(
            "Which of a frame's two labels the prompts print first. 'swapped' reverses the outcome "
            "table and the answer instruction while the payoff mapping stays put, which is the "
            "control that separates a preference for the first-printed position from a preference "
            f"for the word itself (default: {LABEL_PRINT_ORDER_CANONICAL})."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for the trace, corpus, and summary (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--frozen-opponent-model",
        default=None,
        help="Bedrock model id of the frozen opponent, for the vs-fixed-mix arms.",
    )
    parser.add_argument(
        "--frozen-opponent-samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help="Completions drawn from the frozen opponent per prompt.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Sequences per backend call; derived from free VRAM when omitted.",
    )
    parser.add_argument(
        "--prefilled-think",
        action="store_true",
        default=None,
        help=(
            "Treat the POLICY's chat template as already emitting the opening <think> tag. It does "
            "not govern the frozen opponent, whose hosted completions carry no thinking tag at all."
        ),
    )
    parser.add_argument(
        "--no-prefilled-think",
        dest="prefilled_think",
        action="store_false",
        help="Treat the policy's completions as carrying both thinking tags.",
    )
    parser.add_argument(
        "--allow-short-completions",
        action="store_true",
        help=(
            "Sweep below the model's measured termination budget anyway, for a timing or plumbing "
            "probe. Never for a sweep whose selection anyone will read."
        ),
    )
    parser.add_argument(
        "--min-coop",
        type=float,
        default=DEFAULT_MIN_COOP,
        help="Lowest cooperation rate a kept prompt may show.",
    )
    parser.add_argument(
        "--max-coop",
        type=float,
        default=DEFAULT_MAX_COOP,
        help="Highest cooperation rate a kept prompt may show.",
    )
    parser.add_argument(
        "--min-split-std",
        type=float,
        default=DEFAULT_MIN_SPLIT_STD,
        help="Smallest score spread a kept unilateral-split or repeated prompt may show.",
    )
    parser.add_argument(
        "--min-parseable-fraction",
        type=float,
        default=DEFAULT_MIN_PARSEABLE_FRACTION,
        help="Share of completions that must parse for a prompt to be judged at all.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for python and torch sampling.")
    return parser.parse_args(argv)


def _validate_pool_source(args: argparse.Namespace) -> None:
    """Refuse a sweep that names both pool sources, or neither.

    Both would be a sweep whose artifacts name one pool and whose prompts came from the other, and the
    generated rows would silently win; neither leaves nothing to sample. Argparse cannot express this
    as a mutually exclusive group and still say why, and the why is what the operator needs.
    """
    if (args.game is None) == (args.rows is None):
        raise ValueError(
            "pass exactly one of --game and --rows. --game renders one game's split from this "
            "repository's rosters; --rows sweeps an authored pool that may span several games and "
            "counterpart framings (games.breadth_corpus builds one). Given both, the artifacts would "
            "be named after the game while the prompts came from the file."
        )


def _load_authored_rows(args: argparse.Namespace) -> tuple[list[Row], str]:
    """Read an authored pool with the digest of its bytes, refusing rows the sweep's flags contradict.

    Two checks rather than a bare read. The grading is compared because it is what the corpus written
    out of this sweep will be graded by and what `_parse_sample` reads each completion under, so a file
    of `care-alpha-1` rows swept as `group-mix` would be selected by one rule and trained by another.
    The print order is compared because it reaches the artifact filenames, and a stem claiming an order
    the rows do not carry is the 2026-08-19 miscount again.

    The rows are parsed out of one `read_bytes` and the digest covers those same bytes, because two
    reads of one path are two files whenever anything can rewrite it in between. The sweep's meta line
    then records what was measured rather than what the path holds when the trace is written.
    """
    rows_path: Path = args.rows
    raw = rows_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows: list[Row] = [dict(json.loads(line)) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"--rows {args.rows} holds no rows")
    missing = sorted(
        {
            column
            for row in rows
            for column in (PROMPT_COLUMN, PROMPT_ID_COLUMN, GRADING_COLUMN)
            if column not in row
        }
    )
    if missing:
        raise ValueError(
            f"--rows {args.rows} has rows lacking {missing}; the sweep needs the prompt, its id and "
            f"the grading each row is scored under"
        )
    wrong_grading = sorted({_require_str(row, GRADING_COLUMN) for row in rows} - {args.grading})
    if wrong_grading:
        raise ValueError(
            f"--rows {args.rows} carries rows graded {wrong_grading} but --grading is "
            f"{args.grading!r}; every completion is parsed and every verdict keyed under the grading, "
            f"so a mismatch selects by one rule and trains by another"
        )
    orders = sorted({str(row["label_print_order"]) for row in rows if "label_print_order" in row})
    if orders and orders != [args.label_print_order]:
        raise ValueError(
            f"--rows {args.rows} carries label print orders {orders} but --label-print-order is "
            f"{args.label_print_order!r}; that flag names the artifacts, so a stem would claim a "
            f"rendering these prompts do not have"
        )
    logger.info(f"loaded authored rows, path={args.rows} n_rows={len(rows)} sha256={digest}")
    return rows, digest


def _require_thinking(args: argparse.Namespace) -> None:
    """Default the local backends to thinking on, and say so loudly when a run turns it off.

    `backend_cli`'s shared default is off, because no reward-hacking probe wanted traces. Here it
    has to be on: `games/dataset.py` templates with `enable_thinking=True`, so a sweep with
    thinking off measures a different policy from the one that gets trained, and the whole
    decision-theory signal this project reads lives in the reasoning. Mutating the namespace is
    how the default reaches `backend_from_args`, which resolves it against its own constant.
    """
    if args.backend not in backend_cli.LOCAL_KINDS:
        return
    if args.thinking is None:
        args.thinking = True
        return
    if not args.thinking:
        logger.warning(
            "--no-thinking sweeps a policy the trainer runs only if games.train is ALSO given "
            "--no-thinking, which is its PLUMBING mode. Given on both sides the two match and this "
            "trace is the corpus's 'before' eval; given here alone, the trainer templates with "
            "enable_thinking=True and this trace describes a policy it never runs."
        )


def _validate_completion_budget(args: argparse.Namespace) -> None:
    """Refuse a thinking-on local sweep whose completion budget is below this model's floor.

    A refusal rather than a warning, and for the same reason `games/train.py` refuses it: a warning
    is what the run that cost a night would have printed. The failure is silent by construction --
    a trace cut off inside the thinking block emits no closing tag, the sample reads as a parse
    failure, and a prompt more than half of whose samples land there is dropped as
    too-few-parseable. The selection then reports a smaller corpus with nothing pointing at the
    cause, which is exactly what happened on 2026-08-17 to all 64 prompts of a sweep.

    Only an explicit `--max-new-tokens` can trip this, because the unflagged budget IS the floor
    (see `training_sampler`). Hosted backends are exempt: they apply their own template and their
    own reasoning controls, so a local checkpoint's screen says nothing about them.
    """
    if args.backend not in backend_cli.LOCAL_KINDS or args.thinking is False:
        return
    required = required_completion_budget(args.model)
    budget = args.max_new_tokens if args.max_new_tokens is not None else required
    if budget >= required:
        return
    provenance = (
        "this model's own termination screen"
        if args.model in MEASURED_TERMINATION_BUDGET_BY_MODEL
        else "the generic default, as this model has no screen yet"
    )
    if args.allow_short_completions:
        logger.warning(
            f"sweeping {args.model} at {budget} completion tokens, below the {required} its "
            f"termination screen justifies ({provenance}): traces that run past the budget emit no "
            f"closing tag, so their samples read as parse failures and the prompts they belong to "
            f"are dropped as too-few-parseable. --allow-short-completions was given, so this sweep "
            f"proceeds and its selection is not a measurement"
        )
        return
    raise ValueError(
        f"a thinking-on sweep of {args.model} needs at least {required} completion tokens, got "
        f"{budget}. That floor comes from {provenance}; see MEASURED_TERMINATION_BUDGET_BY_MODEL. "
        f"Below it a rollout is cut off inside its thinking block, the sample is unparseable, and "
        f"the corpus this run selects is silently biased toward prompts with unusually short "
        f"reasoning -- which is also the corpus games.train then refuses to train below this same "
        f"floor. Raise the budget (it is a ceiling, so it costs nothing it does not generate), "
        f"or pass --allow-short-completions for a plumbing probe, or --no-thinking for a sweep "
        f"already labelled as one."
    )


def _validate_frozen_opponent(args: argparse.Namespace) -> None:
    """Refuse a grading and a frozen opponent that do not match, before the sweep spends anything.

    Only vs-fixed-mix rows read a cached opponent probability, and every such row leaves
    `games.prompts` carrying the never-measured sentinel. Without the opponent pass the corpus is
    written with that sentinel still in it and `games/rewards.py` raises on the first reward
    evaluation of the training job -- and the guard against exactly that (`fill_opponent_probs`)
    lives inside the block the omission skips. The mirror case is cheaper to catch here too: the
    opponent's own sweep rejects non-vs-fixed-mix rows, but only after the policy sweep has run.
    """
    if args.grading == GRADING_VS_FIXED_MIX and args.frozen_opponent_model is None:
        raise ValueError(
            f"--grading {GRADING_VS_FIXED_MIX} grades every completion against a cached opponent "
            f"cooperation probability, and nothing else fills it in, so --frozen-opponent-model is "
            f"required. Without it every row is written with opp_coop_prob="
            f"{OPP_COOP_PROB_UNSET} and the training job raises inside its reward function."
        )
    if args.grading != GRADING_VS_FIXED_MIX and args.frozen_opponent_model is not None:
        raise ValueError(
            f"--frozen-opponent-model was given with --grading {args.grading!r}, whose rows never "
            f"read a cached opponent probability. Sampling one would pay for hosted calls whose "
            f"result nothing reads; drop the flag, or sweep under {GRADING_VS_FIXED_MIX!r}."
        )


def _refuse_a_regrade_only_grading(args: argparse.Namespace) -> None:
    """Refuse an unsweepable grading here, where nothing has been loaded and nothing has been paid.

    `_parse_sample` refuses it too, but only once a completion exists -- which is after the model is
    resident, the DeltaNet bridge is bound and generation has begun. On a rented card that is the
    difference between an argument error and a wasted reservation, and the failure mode is quiet in
    the direction that costs: `--grading` validates through `grading_cli_value`, which accepts
    anything `is_grading` knows, and that is the whole vocabulary including the regrade-only
    gradings. They stay in it deliberately, so that refusing one here can name the pass that does
    apply it rather than argparse printing a terse "invalid choice" (see `GRADING_CHOICES`).
    """
    if args.grading in REGRADE_ONLY_GRADINGS:
        raise ValueError(f"--grading {args.grading}: {_regrade_only_refusal(args.grading)}")


def _resolve_prefilled_think(args: argparse.Namespace) -> bool:
    """Settle whether the chat template prefills `<think>`, preferring the template itself."""
    if args.prefilled_think is not None:
        return bool(args.prefilled_think)
    if args.backend in backend_cli.LOCAL_KINDS:
        # `args.thinking` is None when the operator did not choose, which _require_thinking has
        # already defaulted to on; anything but an explicit False means the sweep runs thinking-on.
        return prefilled_think_from_template(args.model, enable_thinking=args.thinking is not False)
    guessed = prefilled_think_from_model_name(args.model)
    logger.warning(
        f"--backend {args.backend} has no reachable chat template; guessing "
        f"prefilled_think={guessed} from the model name. Pass --prefilled-think/"
        "--no-prefilled-think to settle it."
    )
    return guessed


def main(argv: Sequence[str] | None = None) -> int:
    """Sweep the base policy, select the mixed prompts, and write the three artefacts.

    The order of the writes is part of the design rather than incidental. The policy sweep is the
    irreplaceable artefact -- the only "before" eval of the corpus that gets trained, and the run's
    whole GPU cost -- so its trace goes to disk in full the moment judging returns, BEFORE the
    frozen-opponent pass reaches out to a hosted API that can throttle, misparse or refuse. Both
    happening in the other order is how a completed sweep used to be thrown away by a failure in the
    step after it.

    The same argument one level down is why the sweep writes each prompt's record as it lands, into
    `<out-dir>/partial/` (`resumable_sweep`): the trace covers a run that finished, and a rented box
    that dies at hour three of six finishes nothing. A relaunch with the same flags over the same pool
    keeps those records and decodes only what is missing; one with different flags is refused by the
    field that changed rather than quietly re-sweeping.

    The partial file is retired the moment the trace it fed is on disk, and not before, which is the
    same ordering argument one level down again. Before that line a death has to leave the file
    resumable; after it, leaving the file in place made the identical command a no-op that still wrote
    a fresh timestamped trace, corpus and selection summary over the previous session's completions
    (`retire_a_completed_partial`).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    default_cuda_allocator_config()
    # First, before anything can import the Qwen3.5 modeling module: transformers binds each
    # DeltaNet kernel at that module's import time, so a bridge applied afterwards is a no-op that
    # looks like it worked. Decode is three quarters of this sweep's wall clock.
    kernel_bridge = bridge_decode_kernel()
    logger.info(f"deltanet decode bridge: {kernel_bridge}")
    args = _parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    _validate_pool_source(args)
    _require_thinking(args)
    _validate_completion_budget(args)
    _validate_frozen_opponent(args)
    _refuse_a_regrade_only_grading(args)

    if args.rows is not None:
        rows, rows_sha256 = _load_authored_rows(args)
    else:
        rows_sha256 = None
        rows = generate_prompt_rows(
            args.game, args.grading, split=args.split, label_print_order=args.label_print_order
        )
        logger.info(
            f"generated {len(rows)} prompt rows for {args.game=} {args.grading=} {args.split=} "
            f"label_print_order={args.label_print_order!r}"
        )

    backend = backend_cli.backend_from_args(
        args, args.model, local_sampling=training_sampler(args.model), mock_responses=MOCK_RESPONSES
    )
    prefilled_think = _resolve_prefilled_think(args)
    # Built before the sweep, so a bad model id or an unresolvable credential costs nothing rather
    # than surfacing after the card has been paid for.
    opponent = (
        FrozenOpponent(build_backend("bedrock", model_id), model_id)
        if (model_id := args.frozen_opponent_model) is not None
        else None
    )
    frozen_plan: dict[str, object] | None = (
        {
            **backend_provenance(opponent.backend),
            "samples_per_prompt": args.frozen_opponent_samples,
            "prefilled_think": FROZEN_OPPONENT_PREFILLED_THINK,
            "min_parseable_fraction": args.min_parseable_fraction,
        }
        if opponent is not None
        else None
    )

    identity = sweep_identity_from_args(
        args,
        backend=backend,
        rows=rows,
        prefilled_think=prefilled_think,
        rows_sha256=rows_sha256,
    )
    partial_path = partial_sweep_path(args)
    swept = resumable_sweep(
        backend,
        rows,
        samples_per_prompt=args.samples_per_prompt,
        prefilled_think=prefilled_think,
        chunk_size=args.chunk_size,
        partial_path=partial_path,
        identity=identity,
    )
    records = list(swept.records)
    backend_cli.log_token_usage(backend)

    # One judging pass, reused for the corpus and the summary: judging twice would let a future
    # divergence make the summary describe a corpus that was never written.
    thresholds = _selection_thresholds(args)
    verdicts = judge_prompts(records, **thresholds)
    selected = [
        dict(record.row) for record, verdict in zip(records, verdicts, strict=True) if verdict.keep
    ]

    timestamp = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    trace_path = args.out_dir / f"{_artifact_stem('sweep', args, timestamp)}.jsonl"
    corpus_path = args.out_dir / f"{_artifact_stem('corpus', args, timestamp)}.jsonl"
    summary_path = args.out_dir / f"{_artifact_stem('selection', args, timestamp)}.json"

    write_sweep_trace(
        trace_path,
        meta=sweep_meta(
            backend=backend,
            args=args,
            prompt_ids=[record.prompt_id for record in records],
            samples_per_prompt=args.samples_per_prompt,
            prefilled_think=prefilled_think,
            rows_sha256=rows_sha256,
            frozen_opponent=frozen_plan,
            kernel_bridge=kernel_bridge,
            resume=sweep_resume_block(swept.sessions, partial_path=partial_path),
        ),
        records=records,
    )
    logger.info(f"wrote the policy sweep trace to {trace_path} before any hosted call")
    # Only now, with the trace on disk: until this line a death has to leave the partial resumable.
    retire_a_completed_partial(partial_path, timestamp=timestamp)

    frozen_meta: dict[str, object] | None = None
    if opponent is not None:
        # `prefilled_think` is the HOSTED opponent's convention and never the policy's -- see
        # FROZEN_OPPONENT_PREFILLED_THINK for why one value cannot serve both.
        opponent_records = sweep_frozen_opponent(
            selected,
            model_id=opponent.model_id,
            samples=args.frozen_opponent_samples,
            backend=opponent.backend,
            prefilled_think=FROZEN_OPPONENT_PREFILLED_THINK,
            chunk_size=args.chunk_size,
        )
        append_sweep_records(trace_path, opponent_records)
        selected = fill_opponent_probs(
            selected,
            frozen_opponent_coop_probs(
                opponent_records, min_parseable_fraction=args.min_parseable_fraction
            ),
        )
        frozen_meta = {
            **(frozen_plan or {}),
            "n_parse_failures_by_prompt": {
                record.prompt_id: record.n_parse_failures for record in opponent_records
            },
        }
        backend_cli.log_token_usage(opponent.backend)

    write_corpus(corpus_path, selected)

    summary = selection_summary(records, verdicts)
    summary_path.write_text(
        json.dumps(
            {
                "selection": summary,
                "thresholds": thresholds,
                "frozen_opponent": frozen_meta,
                "trace_path": str(trace_path),
                "corpus_path": str(corpus_path),
                **git_provenance(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"selection: n_kept={summary['n_kept']} n_dropped={summary['n_dropped']} "
        f"by_reason={summary['counts_by_reason']}"
    )
    logger.info(f"wrote {trace_path}, {corpus_path}, {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
