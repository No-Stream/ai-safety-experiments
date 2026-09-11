r"""Transplant one arm's reasoning into the other's context, and read whose action comes out.

The interp work established that RL moved a decision axis and that the arms' own chains of thought
load on it. What it cannot say is where the behaviour is carried. Two readings survive every
correlational result so far:

* **Mediated by the text.** Training changed what gets written inside the think block, and the
  action follows the text. Then the same text should produce the same action in either arm's
  weights, and transplanting the group-mix arm's reasoning into the self-graded arm should carry the
  group-mix arm's action across with it.
* **Carried by the weights.** The trace is commentary the action does not depend on. Then a
  transplanted trace changes nothing: each arm answers at its own trained rate whatever reasoning
  precedes the answer.

This module runs the experiment that separates them. For one eval prompt of the requested game
(``--game``, default the arms' own twin-pd; the cross-game legs transplant traces self-generated on
never-trained games, where the behavioural transfer is established and whether it travels as text is
the open question) it takes a completion's ENTIRE thinking block -- verbatim, up to and including
the final ``</think>`` -- and prefills it into the other arm's context, then lets that arm generate
only the answer segment. The recipient's action is read exactly the way the eval battery reads it
(`games.parsing`).

Four source conditions per recipient, and the grid is data rather than branches:

* ``no-prefill`` -- the recipient generates freely. The baseline its own trained rate is read from.
* ``own-reasoning`` -- one of the recipient's OWN traces prefilled back into it. The consistency
  control, and the load-bearing one: it bounds how much *any* prefill perturbs an arm, so a shift
  under a foreign trace is only interesting against it. Without this arm a transplant effect cannot
  be told from "being handed any finished reasoning changes what you answer".
* ``other-arm`` -- the other arm's trace. The measurement.
* ``base-model`` -- an un-adapted step-0 trace. Neither arm's training wrote it, so it says whether
  the recipient follows foreign reasoning in general or specifically the other arm's.

Both directions run, because they are not the same experiment: on this corpus the self-graded arm
cooperates far more often than the group-mix arm, so group-into-self asks whether a defecting trace
suppresses cooperation while self-into-group asks whether a cooperating trace restores it, and
mediation could hold in one direction only.

**What the readout must carry, none of it optional.**

*Actions already named inside the transplanted text are a confound, not noise.* A think block that
ends "Answer: LONG" is not merely reasoning, it is an instruction, and a recipient that copies it
has demonstrated instruction-following rather than mediation. So every record carries whether the
transplant names an action -- its last ``<action>`` tag, and separately which label it mentions last
-- and every rate is reported on the naming and non-naming subsets as well as pooled.

*The print-order confound is reported, never averaged away*, as everywhere else in this project: on
this corpus the model picks the first-printed label more often than any trained effect moves it.
Note the inventory limit this runs into -- the reusable twin-pd traces on disk were generated at the
canonical order only, so a reuse run reads one order and says so in its own plan artifact.

*Truncated source traces are excluded, never transplanted.* A trace that ran into its generation cap
has no ``</think>`` at all, so there is no thinking block to take, and its stored text provably does
not re-encode to the ids the engine wrote. Excluded and counted, so the denominator is visible.

*The transplant must re-tokenise cleanly into the recipient's template.* The prompt render ends
``<|im_start|>assistant\n<think>\n``, so the transplant is appended inside an already-open thinking
block and the seam sits between two token sequences that were never tokenised together.
:func:`assert_clean_boundary` requires ``encode(prompt + transplant) == encode(prompt) +
encode(transplant)``; when it fails the recipient reads a token sequence that no longer contains the
source's own tokens, and every action read off it is a measurement of the seam. Measured to hold on
400 sampled twin-pd traces under ``Qwen/Qwen3.5-2B`` (2026-08-24) and measured to go RED when a
newline is prepended to the transplant, because ``<think>\n`` followed by ``\n`` merges into one
token (198, 198 against 271). ``--sabotage boundary`` reproduces that on demand.

*The answer segment is never capped short.* The observed post-``</think>`` answer on this corpus runs
to a maximum of 15,178 tokens (median 9), so the default budget is
:data:`~games.eval_sampler.DEFAULT_EVAL_MAX_NEW_TOKENS`, above the measured maximum rather than at a
convenient round number. What binds instead is the context a transplant needs: prompt plus the
longest thinking block measured 22,354 tokens, so an engine serving this needs a window above roughly
56,000, and :func:`assert_context_fits` refuses a run whose selected contexts do not fit rather than
letting the engine truncate one silently.

Runs on vLLM, which is the only engine here that both serves a LoRA adapter un-merged (the ruling
this project measures effect sizes under -- a bf16 merge loses about a third of the delta) and
continues from a raw context without re-applying the chat template. ``--backend mock`` is the
offline plumbing path and still runs every assembly guard against the real tokenizer; the
HuggingFace path is refused rather than half-supported, and says why.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from transformers import AutoTokenizer

from games.eval_model import resolve_served_model, verify_served_model
from games.eval_sampler import (
    DEFAULT_EVAL_MAX_NEW_TOKENS,
    SAMPLER_TRAINING_DISTRIBUTION,
    eval_sampling,
)
from games.evals import EVAL_ONLY_GRADING, EVAL_RENDER_GRADING_BY_GAME
from games.parsing import THINK_CLOSE, THINK_OPEN, parse_action, strip_thinking
from games.payoffs import COOPERATE, DEFECT
from games.prompts import (
    EVAL_ONLY_MATRIX_GAME_IDS,
    LABEL_PRINT_ORDERS,
    MATRIX_GAME_IDS,
    SPLIT_EVAL,
    generate_prompt_rows,
)
from games.vllm_teardown import DEFAULT_DRAIN_POLICY, DrainPolicy, release_engine, vram_used_mib
from reward_hacking import backend_cli
from reward_hacking.interp.generation_capture import (
    GEN_ENGINE_VLLM,
    SAMPLING_FIELD_NAMES,
    DroppedKnob,
    ResolvedSampler,
    resolved_sampler_for,
)
from reward_hacking.model_backend import (
    _VLLM_FINISH_REASONS,  # pyright: ignore[reportPrivateUsage]  # one finish-reason vocabulary
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    MockBackend,
    SamplingConfig,
    VLLMBackend,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
    from contextlib import AbstractContextManager
    from typing import TextIO

logger = logging.getLogger(__name__)

DEFAULT_GAME_ID = "twin-pd"
"""The game a bare invocation runs: the arms' own game, which every prior grid read.

The game is a parameter (``--game``) because the cross-game legs transplant traces self-generated on
never-trained games. Everything game-specific in a run -- the eval rows, the source cells' meta
assertion, the artifact labels -- follows the requested game; the integrity checks are unchanged and
key on the REQUEST, so a source cell from any other game is refused exactly as before.
"""

TRANSPLANTABLE_GAME_IDS: tuple[str, ...] = (*MATRIX_GAME_IDS, *EVAL_ONLY_MATRIX_GAME_IDS)
"""The games this module can transplant: everything rendered through the one-shot matrix path.

Those rows carry the two printed labels and a cooperative side, which is what
:func:`games.parsing.parse_action` reads an action against and what the naming detectors key on. The
other renderers answer with a figure (a kept fraction, a demand, a returned share) or over a move
sequence, so "the transplant's action" is undefined for them.
"""


def render_grading_for(game_id: str) -> str:
    """Return the grading ``game_id``'s eval rows are rendered under, from the shared maps.

    Trainable games come from :data:`games.evals.EVAL_RENDER_GRADING_BY_GAME`; the eval-only matrix
    games take :data:`games.evals.EVAL_ONLY_GRADING`, exactly as every other cross-game consumer
    resolves it -- read from the maps rather than restated here, so a grading that moves upstream
    moves this module with it. Prompt text is identical across gradings (a grading is a reward-time
    fact, not a prompt fact), so this only names which registry entry rendered the rows. It is
    checked against each source cell's own meta because a replay that renders under a different
    entry than the traces did would drop a source trace into a prompt string it never answered.
    """
    if game_id not in TRANSPLANTABLE_GAME_IDS:
        raise ValueError(
            f"{game_id!r} is not a game this module can transplant: only the one-shot matrix games "
            f"print two labels for an action to be read against. Expected one of "
            f"{sorted(TRANSPLANTABLE_GAME_IDS)}."
        )
    if game_id in EVAL_RENDER_GRADING_BY_GAME:
        return EVAL_RENDER_GRADING_BY_GAME[game_id]
    return EVAL_ONLY_GRADING


ARM_GROUP_MIX = "twin-pd-group"
ARM_SELF_GRADED = "twin-pd-self"
TRAINED_ARMS: tuple[str, ...] = (ARM_GROUP_MIX, ARM_SELF_GRADED)
TRAINED_STEP = 70
BASE_STEP = 0
POPULATION_BASE = f"base/step-{BASE_STEP}"
POPULATION_GROUP_MIX = f"{ARM_GROUP_MIX}/step-{TRAINED_STEP}"
POPULATION_SELF_GRADED = f"{ARM_SELF_GRADED}/step-{TRAINED_STEP}"
RECIPIENT_POPULATIONS: tuple[str, ...] = (POPULATION_GROUP_MIX, POPULATION_SELF_GRADED)
DONOR_FOR_RECIPIENT: dict[str, str] = {
    POPULATION_GROUP_MIX: POPULATION_SELF_GRADED,
    POPULATION_SELF_GRADED: POPULATION_GROUP_MIX,
}

SOURCE_NO_PREFILL = "no-prefill"
SOURCE_OWN_REASONING = "own-reasoning"
SOURCE_OTHER_ARM = "other-arm"
SOURCE_BASE_MODEL = "base-model"
SOURCE_CONDITIONS: tuple[str, ...] = (
    SOURCE_NO_PREFILL,
    SOURCE_OWN_REASONING,
    SOURCE_OTHER_ARM,
    SOURCE_BASE_MODEL,
)

DEFAULT_SOURCE_PASSES: tuple[str, ...] = ("evals-resample-3e7f227", "evals-resample-5315aea")
"""The two resample passes holding full twin-pd completions per arm at presence_penalty 0.

Both, not one: they are independent draws of the same cells, so a selection over one pass would have
a third of the pool and no way to tell a pass artefact from an arm effect.
"""

DEFAULT_TRACES_PER_PROMPT = 8
DEFAULT_BATCH_SIZE: int | None = None
"""Contexts per engine call: ``None`` is the whole cell, which is the default (`_submission_widths`).

It was 32. Each call is a barrier behind its own slowest continuation, and these are 20k-token
contexts answering under a 65,536-token cap, so a fixed width made every group of 32 wait for its
worst row rather than the cell waiting once. The engine batches internally either way; an explicit
``--batch-size`` now buys persistence granularity (records land per call), not throughput.
"""
DEFAULT_SEED = 0
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"

SABOTAGE_NONE = "none"
SABOTAGE_BOUNDARY = "boundary"
SABOTAGE_VERBATIM = "verbatim"
SABOTAGE_MODES: tuple[str, ...] = (SABOTAGE_NONE, SABOTAGE_BOUNDARY, SABOTAGE_VERBATIM)

BACKEND_VLLM = "vllm"
BACKEND_MOCK = "mock"
SUPPORTED_BACKENDS: tuple[str, ...] = (BACKEND_VLLM, BACKEND_MOCK)

PLAN_FILENAME = "mediation_plan.json"
RECORDS_FILENAME = "mediation_records.jsonl"
SUMMARY_FILENAME = "mediation_summary.json"

DISTRIBUTION_KNOBS: tuple[str, ...] = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
)
"""The sampler fields the recipient must match the source traces on.

``max_new_tokens`` is deliberately absent: a budget censors long completions rather than reweighting
the distribution, and this run needs a wider one than the source traces had because a transplanted
context is already thousands of tokens deep.
"""

GROUP_KEY_NONE = "none"
"""How an absent split value is keyed, so a summary reads ``none`` rather than Python's ``None``."""


# --------------------------------------------------------------------------------------
# The thinking block: what gets transplanted, and the guards on taking it
# --------------------------------------------------------------------------------------


def thinking_segment(completion: str) -> str:
    """Return a completion's thinking block verbatim, up to and including the final ``</think>``.

    The final one, matching :func:`games.parsing.strip_thinking`: a model that writes ``</think>``
    more than once has its answer after the last, so everything before it is deliberation and the
    prefix taken here is exactly the text the source model reasoned with.

    Raises on a completion with no close tag at all. That is a truncated trace, and excluding it is
    the caller's job -- reaching here means a filter was skipped, which would transplant a clipped
    trace and read the result as an arm effect.
    """
    head, separator, _ = completion.rpartition(THINK_CLOSE)
    if not separator:
        raise ValueError(
            f"completion carries no {THINK_CLOSE}, so it has no thinking block to transplant; a "
            f"trace that ran into its generation cap must be excluded before it reaches here "
            f"(head={completion[:80]!r})"
        )
    return head + separator


def _first_divergence(left: Sequence[object], right: Sequence[object]) -> int:
    """Index of the first differing element, or the length of the shorter sequence."""
    for index, (one, other) in enumerate(zip(left, right, strict=False)):
        if one != other:
            return index
    return min(len(left), len(right))


def assert_verbatim_transplant(transplant: str, completion: str) -> None:
    """Raise unless ``transplant`` is a byte-exact prefix of ``completion`` ending at ``</think>``.

    The assembly check. A transplant that has been re-decoded, normalised, stripped or edited by one
    character is no longer the text the source model wrote, and the whole experiment is a claim about
    that exact text reaching the recipient.
    """
    if not transplant.endswith(THINK_CLOSE):
        raise ValueError(
            f"transplant does not end at {THINK_CLOSE}, so the recipient would keep reasoning "
            f"instead of answering (tail={transplant[-60:]!r})"
        )
    if not completion.startswith(transplant):
        raise ValueError(
            f"transplant is not a byte-exact prefix of its source completion, so it is not the text "
            f"the source model wrote. First divergence at character "
            f"{_first_divergence(transplant, completion)}."
        )


class ChatTokenizer(Protocol):
    """The two tokenizer calls this module makes, so a test can supply merges it controls.

    Structural rather than nominal because the real object's signatures are not statically checkable
    here, and because the boundary guard has to be watchable failing -- which needs a tokenizer whose
    merge behaviour is known.
    """

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        """Render one conversation to the string the engine reads."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        """Token ids for ``text``."""
        ...


def render_context(tokenizer: ChatTokenizer, prompt: str) -> str:
    """Render one game prompt as the thinking-on single user turn the source traces were made in.

    Asserts the render leaves the thinking block OPEN. Everything downstream depends on it: the
    transplant is appended to this string, so a template that closed the block (or never opened one)
    would put the source's reasoning in the answer position, where the parser would read its stated
    action as the recipient's own answer and every cell would report perfect mediation.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not rendered.rstrip().endswith(THINK_OPEN):
        raise ValueError(
            f"the rendered prompt does not leave {THINK_OPEN} open (tail={rendered[-60:]!r}), so a "
            f"transplant appended to it would land in the answer position rather than inside the "
            f"reasoning block. This readout assumes the prefilled-think template."
        )
    if THINK_CLOSE in rendered:
        raise ValueError(
            f"the rendered prompt already closes the thinking block, so nothing appended to it is "
            f"reasoning (tail={rendered[-60:]!r})"
        )
    return rendered


def assert_clean_boundary(tokenizer: ChatTokenizer, rendered: str, transplant: str) -> list[int]:
    """Return the context's token ids, refusing a seam that does not re-tokenise cleanly.

    ``encode(rendered + transplant)`` must equal ``encode(rendered) + encode(transplant)``. When it
    does not, the model reads a token sequence that no longer contains the source's own tokens at the
    seam, and the difference is invisible in the decoded string.

    Called BEFORE :func:`assert_verbatim_transplant` on purpose: the verbatim check would also catch
    a boundary sabotage (a prepended character breaks the prefix relation too), so running it first
    would leave this guard never watched failing.
    """
    joint = tokenizer.encode(rendered + transplant)
    prefix = tokenizer.encode(rendered)
    tail = tokenizer.encode(transplant, add_special_tokens=False)
    if joint != prefix + tail:
        raise ValueError(
            f"the transplant does not re-tokenise cleanly into the recipient's template: "
            f"encode(prompt+transplant) has {len(joint)} tokens against "
            f"{len(prefix)}+{len(tail)}={len(prefix) + len(tail)} concatenated, first differing at "
            f"index {_first_divergence(joint, prefix + tail)} (the prompt ends {len(prefix)} tokens "
            f"in). The recipient would read different tokens than the source model wrote, so what "
            f"reaches it is not this text."
        )
    return joint


# --------------------------------------------------------------------------------------
# Source traces: reading them off disk, and what gets excluded
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTrace:
    """One completion available for transplant, with the row facts it must be matched against."""

    replay_id: str
    population: str
    source_arm: str
    pass_name: str
    prompt_id: str
    reskin_id: str
    payoff_variant: str
    label_print_order: str
    label_a: str
    label_b: str
    coop_label: str
    sample_index: int
    completion: str
    thinking: str
    action: str | None

    @property
    def action_stratum(self) -> str:
        """The source's own action as a group key, unparsed getting its own bucket, never a gap."""
        return GROUP_KEY_NONE if self.action is None else self.action


def _population_for(arm: str, step: int) -> str:
    """Which population a cell belongs to. Step 0 is the un-adapted base for BOTH arms."""
    return POPULATION_BASE if step == BASE_STEP else f"{arm}/step-{step}"


def source_cell_paths(
    root: Path, passes: Sequence[str], *, arms: Sequence[str] = TRAINED_ARMS
) -> Iterator[tuple[str, str, int, Path]]:
    """Yield ``(pass, arm, step, path)`` for every presence-penalty-0 cell the passes should hold.

    A missing file raises rather than being skipped: a half-downloaded pass would otherwise read as a
    smaller but honest inventory, and the arm whose file was absent would quietly lose its donor pool.
    """
    for pass_name in passes:
        for arm in arms:
            for step in (BASE_STEP, TRAINED_STEP):
                path = root / pass_name / arm / f"step-{step}-pp0.0.jsonl"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"expected source cell missing: {path}. The reuse inventory is built from "
                        f"all of {list(passes)} x {list(arms)} x steps {BASE_STEP},{TRAINED_STEP}."
                    )
                yield pass_name, arm, step, path


def read_source_sampling(root: Path, passes: Sequence[str]) -> dict[str, Any]:
    """Return the sampler the source cells were generated under, asserting they all agree.

    Read off the traces rather than restated as literals: a replay whose recipient samples under a
    different policy than the donors did is not comparable to the donors' own recorded rates, and an
    artifact that names the sampler from a constant is unfalsifiable from the artifact alone.
    """
    blocks: dict[str, str] = {}
    for pass_name, arm, step, path in source_cell_paths(root, passes):
        with path.open(encoding="utf-8") as handle:
            meta = cast("dict[str, Any]", json.loads(handle.readline()))
        blocks[f"{pass_name}|{arm}|step-{step}"] = json.dumps(meta["sampling"], sort_keys=True)
    distinct = sorted(set(blocks.values()))
    if len(distinct) != 1:
        raise ValueError(f"the source cells disagree on their sampling block: {blocks}")
    return cast("dict[str, Any]", json.loads(distinct[0]))


def assert_sampler_matches_sources(
    sampling: SamplingConfig, source_block: Mapping[str, Any]
) -> None:
    """Raise unless the recipient's distribution knobs match the ones the sources were drawn under.

    Only the distribution knobs, per :data:`DISTRIBUTION_KNOBS`: the completion budget is allowed to
    differ and has to, because a transplanted context starts thousands of tokens deep.
    """
    mismatched = {
        knob: (getattr(sampling, knob), source_block.get(knob))
        for knob in DISTRIBUTION_KNOBS
        if knob in source_block and getattr(sampling, knob) != source_block[knob]
    }
    if mismatched:
        raise ValueError(
            f"the recipient would sample under different knobs than the source traces were drawn "
            f"under: {mismatched}. The no-prefill baseline is compared against the donor arm's own "
            f"recorded rate, so a sampler difference would read as a transplant effect."
        )


@dataclass
class SourceCensus:
    """What the inventory read and dropped, so every rate below has a visible denominator."""

    read: int = 0
    dropped_truncated: int = 0
    dropped_no_close_tag: int = 0
    dropped_unparsed_action: int = 0
    dropped_step0_cross_arm_duplicate: int = 0
    kept: int = 0
    stored_action_disagreements: int = 0

    def as_payload(self) -> dict[str, int]:
        """Flat counts for the artifact."""
        return {
            "read": self.read,
            "dropped_truncated": self.dropped_truncated,
            "dropped_no_close_tag": self.dropped_no_close_tag,
            "dropped_unparsed_action": self.dropped_unparsed_action,
            "dropped_step0_cross_arm_duplicate": self.dropped_step0_cross_arm_duplicate,
            "kept": self.kept,
            "stored_action_disagreements": self.stored_action_disagreements,
        }


def _assert_cell_meta(
    meta: Mapping[str, Any], path: Path, *, game_id: str, render_grading: str
) -> None:
    """Refuse a cell whose meta says it is not the game and rendering this replay assumes."""
    if meta.get("record") != "meta":
        raise ValueError(f"{path} does not open with a meta record; it is not a trace cell.")
    if meta.get("game_id") != game_id:
        raise ValueError(f"{path} holds game_id {meta.get('game_id')!r}, not {game_id!r}.")
    if meta.get("render_grading") != render_grading:
        raise ValueError(
            f"{path} rendered under {meta.get('render_grading')!r}, not {render_grading!r}; its "
            f"completions answer a different prompt string than this replay renders."
        )


def derived_action(record: Mapping[str, Any]) -> str | None:
    """Parse a source record's action from its own completion, the way the eval battery would.

    Re-derived rather than read off the stored ``action`` field, and then cross-checked against it. A
    stored label that disagrees with what the parser makes of the text means one of the two is wrong
    about what the model answered, and every reading here rests on that answer. This is not
    hypothetical: a sibling analysis of these same traces computed its cooperation covariate by
    comparing the stored ``action`` (``C``/``D``) against ``coop_label`` (a word like ``SHORT``),
    which is False for every record ever written.
    """
    visible, _ = strip_thinking(str(record["completion"]), prefilled_think=True)
    return parse_action(
        visible,
        label_a=str(record["label_a"]),
        label_b=str(record["label_b"]),
        coop_label=str(record["coop_label"]),
    )


def _source_trace_from(
    record: Mapping[str, Any], *, pass_name: str, arm: str, step: int, action: str
) -> SourceTrace:
    """Build one :class:`SourceTrace` from a trace record that has passed every exclusion."""
    completion = str(record["completion"])
    return SourceTrace(
        replay_id=(
            f"{pass_name}|{arm}|step-{step}|{record['prompt_id']}|s{record['sample_index']}"
        ),
        population=_population_for(arm, step),
        source_arm=arm,
        pass_name=pass_name,
        prompt_id=str(record["prompt_id"]),
        reskin_id=str(record["reskin_id"]),
        payoff_variant=str(record["payoff_variant"]),
        label_print_order=str(record["label_print_order"]),
        label_a=str(record["label_a"]),
        label_b=str(record["label_b"]),
        coop_label=str(record["coop_label"]),
        sample_index=int(record["sample_index"]),
        completion=completion,
        thinking=thinking_segment(completion),
        action=action,
    )


def _is_cross_arm_step0_duplicate(
    record: Mapping[str, Any], *, pass_name: str, arm: str, seen: dict[str, str]
) -> bool:
    """Whether this step-0 record already arrived under the other arm, remembering it if not.

    Both arms' step-0 legs served the same un-adapted checkpoint and, within a pass, reproduce their
    first sample indices verbatim, so the same completion appears twice. A duplicate found WITHIN one
    arm raises instead of being dropped: that would mean this dedup is discarding real independent
    draws rather than copies.
    """
    digest = hashlib.sha256(str(record["completion"]).encode()).hexdigest()
    key = f"{pass_name}|{record['prompt_id']}|{digest}"
    first_arm = seen.get(key)
    if first_arm is None:
        seen[key] = arm
        return False
    if first_arm == arm:
        raise ValueError(
            f"two records within {pass_name}/{arm}/step-{BASE_STEP} share a completion "
            f"({record['prompt_id']}); the cross-arm dedup would discard a real independent draw."
        )
    return True


def load_source_traces(
    root: Path,
    passes: Sequence[str] = DEFAULT_SOURCE_PASSES,
    *,
    game_id: str = DEFAULT_GAME_ID,
) -> tuple[list[SourceTrace], SourceCensus]:
    """Read every usable completion of ``game_id`` off the resample cells, with its census.

    Four exclusions, each counted:

    * ``truncated_thinking`` traces, and separately any trace with no ``</think>``. The second count
      exists because the two are supposed to coincide, and a divergence would mean the stored flag no
      longer describes the stored text.
    * traces whose own action does not parse, because the follow-the-source reading is undefined
      without it.
    * step-0 records shared between the two arms. Both arms' step-0 legs served the same un-adapted
      checkpoint and, within a pass, reproduce their first sample indices verbatim; counting them
      twice would fill the base pool with copies. A duplicate found WITHIN one arm raises instead,
      because that would mean the dedup is discarding real independent draws.
    """
    render_grading = render_grading_for(game_id)
    kept: list[SourceTrace] = []
    census = SourceCensus()
    seen_step0: dict[str, str] = {}
    for pass_name, arm, step, path in source_cell_paths(root, passes):
        with path.open(encoding="utf-8") as handle:
            _assert_cell_meta(
                cast("dict[str, Any]", json.loads(handle.readline())),
                path,
                game_id=game_id,
                render_grading=render_grading,
            )
            for line in handle:
                record = cast("dict[str, Any]", json.loads(line))
                census.read += 1
                completion = str(record["completion"])
                if record.get("truncated_thinking"):
                    census.dropped_truncated += 1
                    continue
                if THINK_CLOSE not in completion:
                    census.dropped_no_close_tag += 1
                    continue
                if step == BASE_STEP and _is_cross_arm_step0_duplicate(
                    record, pass_name=pass_name, arm=arm, seen=seen_step0
                ):
                    census.dropped_step0_cross_arm_duplicate += 1
                    continue
                action = derived_action(record)
                if action is None:
                    census.dropped_unparsed_action += 1
                    continue
                if record.get("parsed") and str(record.get("action")) != action:
                    census.stored_action_disagreements += 1
                kept.append(
                    _source_trace_from(
                        record, pass_name=pass_name, arm=arm, step=step, action=action
                    )
                )
                census.kept += 1
    if census.stored_action_disagreements:
        raise ValueError(
            f"{census.stored_action_disagreements} source records' stored action disagrees with what "
            f"games.parsing makes of their own completion text. One of the two is wrong about what "
            f"the model answered, and the follow-the-source reading rests on that answer, so this "
            f"has to be understood before any GPU is spent."
        )
    logger.info(f"source inventory: {census.as_payload()}")
    return kept, census


def traces_by_population(traces: Sequence[SourceTrace]) -> dict[str, list[SourceTrace]]:
    """Group traces by population, sorted by ``replay_id`` so selection is order-independent."""
    grouped: dict[str, list[SourceTrace]] = defaultdict(list)
    for trace in traces:
        grouped[trace.population].append(trace)
    return {name: sorted(pool, key=lambda trace: trace.replay_id) for name, pool in grouped.items()}


# --------------------------------------------------------------------------------------
# The condition grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TransplantCell:
    """One (recipient, source condition) cell of the grid."""

    recipient_population: str
    source_condition: str
    source_population: str | None

    @property
    def key(self) -> str:
        """A stable flat key for grouping, e.g. ``twin-pd-self/step-70<-other-arm``."""
        return f"{self.recipient_population}<-{self.source_condition}"

    @property
    def direction(self) -> str | None:
        """``source -> recipient``, or None for the baseline where nothing is transplanted."""
        if self.source_population is None:
            return None
        return f"{self.source_population}->{self.recipient_population}"


def _source_population(recipient: str, condition: str) -> str | None:
    """Which population donates the transplant for one (recipient, condition)."""
    if condition == SOURCE_NO_PREFILL:
        return None
    if condition == SOURCE_OWN_REASONING:
        return recipient
    if condition == SOURCE_OTHER_ARM:
        return DONOR_FOR_RECIPIENT[recipient]
    if condition == SOURCE_BASE_MODEL:
        return POPULATION_BASE
    raise ValueError(f"unknown source condition {condition!r}.")


def transplant_cells(
    recipients: Sequence[str] = RECIPIENT_POPULATIONS,
    conditions: Sequence[str] = SOURCE_CONDITIONS,
) -> list[TransplantCell]:
    """Build the grid: every requested source condition for every requested recipient."""
    unknown_recipient = sorted(set(recipients) - set(RECIPIENT_POPULATIONS))
    if unknown_recipient:
        raise ValueError(
            f"unknown recipient populations {unknown_recipient}; the trained arms are "
            f"{list(RECIPIENT_POPULATIONS)}."
        )
    unknown_condition = sorted(set(conditions) - set(SOURCE_CONDITIONS))
    if unknown_condition:
        raise ValueError(
            f"unknown source conditions {unknown_condition}; expected {list(SOURCE_CONDITIONS)}."
        )
    return [
        TransplantCell(recipient, condition, _source_population(recipient, condition))
        for recipient in recipients
        for condition in conditions
    ]


# --------------------------------------------------------------------------------------
# Prompt rows and source selection
# --------------------------------------------------------------------------------------


def eval_rows(
    orders: Sequence[str] = LABEL_PRINT_ORDERS, *, game_id: str = DEFAULT_GAME_ID
) -> list[dict[str, Any]]:
    """Held-out eval rows of ``game_id`` in the requested print orders, in a stable order."""
    grading = render_grading_for(game_id)
    rows: list[dict[str, Any]] = []
    for order in orders:
        rows.extend(
            generate_prompt_rows(game_id, grading, split=SPLIT_EVAL, label_print_order=order)
        )
    return rows


def assert_row_matches_source(row: Mapping[str, Any], trace: SourceTrace) -> None:
    """Raise unless a source trace answers exactly the row it is about to be transplanted into.

    Pure bookkeeping, and every consequence of getting it wrong is silent: a trace written for a
    swapped rendering dropped into a canonical prompt reads as a print-order effect, and one written
    for another reskin reasons about payoffs the recipient was never shown.
    """
    mismatched = {
        name: (row[name], getattr(trace, name))
        for name in ("prompt_id", "reskin_id", "payoff_variant", "label_print_order", "coop_label")
        if str(row[name]) != str(getattr(trace, name))
    }
    if mismatched:
        raise ValueError(
            f"source trace {trace.replay_id} does not answer the row it would be transplanted into: "
            f"{mismatched}. Transplanting across rows would read as an arm effect."
        )


@dataclass(frozen=True)
class Shortfall:
    """A cell/prompt that could not supply the requested number of source traces."""

    cell_key: str
    prompt_id: str
    requested: int
    available: int
    rejected_over_budget: int

    def as_payload(self) -> dict[str, Any]:
        """Flat form for the artifact."""
        return {
            "cell_key": self.cell_key,
            "prompt_id": self.prompt_id,
            "requested": self.requested,
            "available": self.available,
            "rejected_over_budget": self.rejected_over_budget,
        }


def select_source_traces(  # noqa: PLR0913 - a selection is its pool, its rows, its cell, its size, its seed and its filter
    pool: Sequence[SourceTrace],
    *,
    prompt_ids: Sequence[str],
    cell_key: str,
    n_per_prompt: int,
    seed: int,
    accept: Callable[[SourceTrace], bool] | None = None,
) -> tuple[dict[str, list[SourceTrace]], list[Shortfall]]:
    """Pick ``n_per_prompt`` traces per prompt, deterministically, counting every shortfall.

    Representative rather than balanced on the source's own action, deliberately: the headline
    reading is what a recipient does when handed the donor arm's TYPICAL reasoning, and the mediation
    arithmetic divides a shift by the arms' own behavioural gap, which only means anything if the
    donor sample looks like the donor arm. Every record carries its source's action, so the balanced
    read is a re-analysis rather than a second run.

    ``accept`` is the token-budget predicate. A rejected candidate is skipped and the next taken, so
    a long-trace exclusion thins a cell rather than silently shortening a context -- and the count is
    reported, because that exclusion correlates with trace length and length correlates with what the
    reasoning contains.
    """
    if n_per_prompt < 1:
        raise ValueError(f"n_per_prompt must be positive, got {n_per_prompt}.")
    by_prompt: dict[str, list[SourceTrace]] = defaultdict(list)
    for trace in pool:
        by_prompt[trace.prompt_id].append(trace)
    chosen: dict[str, list[SourceTrace]] = {}
    shortfalls: list[Shortfall] = []
    for prompt_id in prompt_ids:
        candidates = sorted(by_prompt.get(prompt_id, []), key=lambda trace: trace.replay_id)
        random.Random(f"{seed}|{cell_key}|{prompt_id}").shuffle(candidates)
        taken: list[SourceTrace] = []
        rejected = 0
        for candidate in candidates:
            if len(taken) == n_per_prompt:
                break
            if accept is not None and not accept(candidate):
                rejected += 1
                continue
            taken.append(candidate)
        chosen[prompt_id] = taken
        if len(taken) < n_per_prompt:
            shortfalls.append(Shortfall(cell_key, prompt_id, n_per_prompt, len(taken), rejected))
    return chosen, shortfalls


# --------------------------------------------------------------------------------------
# What the transplanted text itself says
# --------------------------------------------------------------------------------------


def transplant_action_tag(trace: SourceTrace) -> str | None:
    """Return the action named by the last ``<action>`` tag INSIDE the thinking block, or None.

    The sharp form of the stated-action confound: a think block that already emitted the answer tag
    is handing the recipient an instruction, and a recipient that repeats it has demonstrated copying
    rather than mediation. Reported as its own split so the pooled rate is never the only number.
    """
    return parse_action(
        trace.thinking, label_a=trace.label_a, label_b=trace.label_b, coop_label=trace.coop_label
    )


def transplant_last_label_mention(trace: SourceTrace) -> str | None:
    """Which of the two labels the thinking block mentions LAST, or None if neither appears.

    The soft form of the same confound, and the common one: traces here overwhelmingly close with
    "Answer: LONG" in prose rather than a tag. Case-sensitive, because these labels print upper-case
    and a lower-case occurrence inside an ordinary word would be a false hit.
    """
    last_a = trace.thinking.rfind(trace.label_a)
    last_b = trace.thinking.rfind(trace.label_b)
    if last_a < 0 and last_b < 0:
        return None
    label = trace.label_a if last_a > last_b else trace.label_b
    return COOPERATE if label == trace.coop_label else DEFECT


# --------------------------------------------------------------------------------------
# Raw continuation: generating from a context the chat template must not touch again
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RawContinuation:
    """What one engine call produced from one already-rendered context."""

    text: str
    stop_reason: str
    n_response_tokens: int | None
    n_context_tokens: int | None


def _vllm_continue_raw(backend: VLLMBackend, contexts: list[str]) -> list[RawContinuation]:
    """Drive the vLLM engine on raw contexts, checking each reply against the context it answers."""
    engine = backend._llm  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]  # raw-prompt path
    params = backend._sampling_params  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    lora = backend._lora_request  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    outputs = cast("list[Any]", engine.generate(contexts, params, lora_request=lora))
    continuations: list[RawContinuation] = []
    for context, output in zip(contexts, outputs, strict=True):
        if output.prompt != context:
            raise RuntimeError(
                f"vLLM returned its replies out of order: reply {len(continuations)} answers a "
                f"context this call did not send at that position. Pairing replies with the wrong "
                f"transplant files every action under the wrong source, silently."
            )
        sequence = output.outputs[0]
        continuations.append(
            RawContinuation(
                text=str(sequence.text),
                stop_reason=_VLLM_FINISH_REASONS.get(
                    sequence.finish_reason, str(sequence.finish_reason)
                ),
                n_response_tokens=len(sequence.token_ids),
                n_context_tokens=len(output.prompt_token_ids),
            )
        )
    return continuations


def continue_raw(backend: object, contexts: list[str]) -> list[RawContinuation]:
    """Continue each context verbatim, with no chat template re-applied.

    The seam this module needs and no backend offers: ``Backend.generate`` wraps its argument as a
    fresh user turn, which would put a transplanted think block inside a user message. vLLM's engine
    takes raw prompt strings directly, so the transplant reaches the model as the assistant's own
    partial turn, which is the only arrangement in which the experiment means anything.
    """
    if isinstance(backend, VLLMBackend):
        return _vllm_continue_raw(backend, contexts)
    if isinstance(backend, MockBackend):
        return [
            RawContinuation(text, STOP_REASON_END_TURN, None, None)
            for text in backend.generate(contexts)
        ]
    raise TypeError(
        f"{type(backend).__name__} cannot continue a raw context. Only vLLM can here: it is the one "
        f"engine that both serves a LoRA adapter un-merged (a bf16 merge loses about a third of the "
        f"trained delta) and continues from a raw string without re-templating it. The HuggingFace "
        f"path is refused rather than half-supported."
    )


# --------------------------------------------------------------------------------------
# Assembling one unit of work
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedUnit:
    """One context ready to be generated from, and the row and source facts it belongs to."""

    cell: TransplantCell
    row: dict[str, Any]
    trace: SourceTrace | None
    transplant: str
    context: str
    n_context_tokens: int
    n_transplant_tokens: int
    sample_index: int


def sabotaged_transplant(transplant: str, mode: str) -> str:
    r"""Return the transplant with the requested violation planted in it.

    ``boundary`` prepends a newline: the render ends ``<think>\n``, so the pair merges into one token
    and the concatenation invariant breaks (measured on ``Qwen/Qwen3.5-2B``: ids 198, 198 against
    271). ``verbatim`` flips one character in the middle, which leaves the seam alone and trips the
    prefix check instead. Both exist so the guards can be watched failing on the real path rather
    than only in a test.
    """
    if mode == SABOTAGE_NONE:
        return transplant
    if mode == SABOTAGE_BOUNDARY:
        return "\n" + transplant
    if mode == SABOTAGE_VERBATIM:
        middle = len(transplant) // 2
        flipped = " " if transplant[middle] != " " else "x"
        return transplant[:middle] + flipped + transplant[middle + 1 :]
    raise ValueError(f"unknown sabotage mode {mode!r}; expected one of {list(SABOTAGE_MODES)}.")


def prepare_unit(  # noqa: PLR0913 - one unit is its tokenizer, its prompt, its cell, its row, its source and its index
    tokenizer: ChatTokenizer,
    rendered: str,
    *,
    cell: TransplantCell,
    row: dict[str, Any],
    trace: SourceTrace | None,
    sample_index: int,
    sabotage: str = SABOTAGE_NONE,
) -> PreparedUnit:
    """Assemble one context, running every assembly guard on the way.

    Guard order is load-bearing and documented on :func:`assert_clean_boundary`.
    """
    if trace is None:
        return PreparedUnit(
            cell=cell,
            row=row,
            trace=None,
            transplant="",
            context=rendered,
            n_context_tokens=len(tokenizer.encode(rendered)),
            n_transplant_tokens=0,
            sample_index=sample_index,
        )
    assert_row_matches_source(row, trace)
    transplant = sabotaged_transplant(trace.thinking, sabotage)
    context_ids = assert_clean_boundary(tokenizer, rendered, transplant)
    assert_verbatim_transplant(transplant, trace.completion)
    return PreparedUnit(
        cell=cell,
        row=row,
        trace=trace,
        transplant=transplant,
        context=rendered + transplant,
        n_context_tokens=len(context_ids),
        n_transplant_tokens=len(tokenizer.encode(transplant, add_special_tokens=False)),
        sample_index=sample_index,
    )


def assert_context_fits(
    units: Sequence[PreparedUnit], *, max_new_tokens: int, max_model_len: int | None
) -> dict[str, int]:
    """Report the context budget the run needs, refusing one the engine cannot hold.

    Refusing rather than trimming: an engine handed a request past its window drops or truncates it,
    and a truncated transplant is a different text than the one whose effect is being measured. The
    error names the filter value that would make the run fit, so the operator chooses the exclusion
    explicitly and it lands in the census.
    """
    if not units:
        raise ValueError("no prepared units, so there is no context budget to check.")
    longest = max(units, key=lambda unit: unit.n_context_tokens)
    needed = longest.n_context_tokens + max_new_tokens
    budget = {
        "max_context_tokens": longest.n_context_tokens,
        "max_transplant_tokens": max(unit.n_transplant_tokens for unit in units),
        "max_new_tokens": max_new_tokens,
        "required_model_len": needed,
    }
    if max_model_len is not None and needed > max_model_len:
        source = "no-prefill" if longest.trace is None else longest.trace.replay_id
        raise ValueError(
            f"the longest selected context is {longest.n_context_tokens} tokens and the answer "
            f"budget is {max_new_tokens}, so this run needs a {needed}-token window against "
            f"--max-model-len {max_model_len}. Either widen the window or pass "
            f"--max-source-response-tokens to exclude the long traces explicitly -- that exclusion "
            f"correlates with trace length, so it has to be a recorded choice rather than a silent "
            f"truncation. Longest source: {source}."
        )
    logger.info(f"context budget: {budget}")
    return budget


# --------------------------------------------------------------------------------------
# Records and summaries
# --------------------------------------------------------------------------------------


def build_record(unit: PreparedUnit, continuation: RawContinuation) -> dict[str, Any]:
    """Turn one generated continuation into the record the analysis reads.

    The recipient's action is parsed off ``transplant + response`` rather than off the response
    alone, so one code path serves both the prefilled cells (whose ``</think>`` arrived in the
    transplant) and the no-prefill baseline (which writes its own). That is also why an action stated
    inside the transplant cannot leak into the recipient's answer by accident:
    :func:`games.parsing.strip_thinking` cuts everything up to the final close tag first.
    """
    row = unit.row
    trace = unit.trace
    full_completion = unit.transplant + continuation.text
    visible, truncated = strip_thinking(full_completion, prefilled_think=True)
    action = parse_action(
        visible,
        label_a=str(row["label_a"]),
        label_b=str(row["label_b"]),
        coop_label=str(row["coop_label"]),
    )
    follows = (
        None if trace is None or action is None or trace.action is None else action == trace.action
    )
    return {
        "condition_key": unit.cell.key,
        "recipient_population": unit.cell.recipient_population,
        "source_condition": unit.cell.source_condition,
        "source_population": unit.cell.source_population,
        "direction": unit.cell.direction,
        "prompt_id": row["prompt_id"],
        "reskin_id": row["reskin_id"],
        "payoff_variant": row["payoff_variant"],
        "label_print_order": row["label_print_order"],
        "coop_label": row["coop_label"],
        "sample_index": unit.sample_index,
        "source_replay_id": None if trace is None else trace.replay_id,
        "source_pass": None if trace is None else trace.pass_name,
        "source_sample_index": None if trace is None else trace.sample_index,
        "source_action": None if trace is None else trace.action,
        "source_action_stratum": GROUP_KEY_NONE if trace is None else trace.action_stratum,
        "transplant_action_tag": None if trace is None else transplant_action_tag(trace),
        "transplant_last_label_mention": (
            None if trace is None else transplant_last_label_mention(trace)
        ),
        "n_context_tokens": unit.n_context_tokens,
        "n_transplant_tokens": unit.n_transplant_tokens,
        "n_response_tokens": continuation.n_response_tokens,
        "stop_reason": continuation.stop_reason,
        "hit_answer_cap": continuation.stop_reason == STOP_REASON_MAX_TOKENS,
        "response_text": continuation.text,
        "parsed_action": action,
        "cooperate": None if action is None else action == COOPERATE,
        "follows_source": follows,
        "truncated_thinking": truncated,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    """Return a rate, or None when the denominator is zero: a zero needs its denominator."""
    return None if denominator == 0 else numerator / denominator


def _cooperation_block(group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cooperation and follow-the-source as k/n over one group of records."""
    parsed = [record for record in group if record["cooperate"] is not None]
    cooperated = sum(1 for record in parsed if bool(record["cooperate"]))
    followed = [record for record in parsed if record["follows_source"] is not None]
    follow_k = sum(1 for record in followed if bool(record["follows_source"]))
    return {
        "n_completions": len(group),
        "n_parsed": len(parsed),
        "n_parse_failures": len(group) - len(parsed),
        "n_truncated_thinking": sum(1 for record in group if bool(record["truncated_thinking"])),
        "n_hit_answer_cap": sum(1 for record in group if bool(record["hit_answer_cap"])),
        "cooperate_k": cooperated,
        "cooperate_rate": _rate(cooperated, len(parsed)),
        "follow_n": len(followed),
        "follow_k": follow_k,
        "follow_rate": _rate(follow_k, len(followed)),
    }


def _group_key(value: object) -> str:
    """Render a split value as a summary key, absent becoming ``none`` rather than ``None``."""
    return GROUP_KEY_NONE if value is None else str(value)


def _split_blocks(group: Sequence[Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    """Cooperation blocks split on one record field, keyed by that field's rendered value."""
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in group:
        buckets[_group_key(record[name])].append(record)
    return {key: _cooperation_block(bucket) for key, bucket in sorted(buckets.items())}


def summarise_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per condition key: pooled rates plus the four splits the readout must carry.

    ``by_print_order`` because the first-printed label dominates any trained effect here;
    ``by_transplant_action_tag`` and ``by_transplant_last_label_mention`` because a transplant that
    names its action is handing over an instruction, and pooling those with the rest would let
    instruction-following read as mediation; ``by_source_action`` because it turns the representative
    sample into the stratified read without another GPU hold.
    """
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["condition_key"])].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for key, group in sorted(grouped.items()):
        entry = _cooperation_block(group)
        entry["by_print_order"] = _split_blocks(group, "label_print_order")
        entry["by_transplant_action_tag"] = _split_blocks(group, "transplant_action_tag")
        entry["by_transplant_last_label_mention"] = _split_blocks(
            group, "transplant_last_label_mention"
        )
        entry["by_source_action"] = _split_blocks(group, "source_action_stratum")
        summary[key] = entry
    return summary


ARM_GAP_FLOOR = 0.02
"""Smallest arm gap a mediated fraction is reported over.

Below it the denominator is noise and the ratio would swing on one flipped action, so the shift is
reported and the fraction is None. Not a threshold anything is gated on -- the shift and both rates
are always there to read.
"""


def _cell_field(
    summary: Mapping[str, Mapping[str, Any]], recipient: str, condition: str, name: str
) -> object:
    """One field of one summary cell, or None when the cell was not run."""
    entry = summary.get(TransplantCell(recipient, condition, None).key)
    return None if entry is None else entry.get(name)


def _cell_rate(
    summary: Mapping[str, Mapping[str, Any]], recipient: str, condition: str
) -> float | None:
    """Return the cooperation rate of one summary cell, or None when the cell was not run."""
    value = _cell_field(summary, recipient, condition, "cooperate_rate")
    return None if value is None else float(cast("float", value))


def _cell_denominator(
    summary: Mapping[str, Mapping[str, Any]], recipient: str, condition: str
) -> dict[str, int | None]:
    """Return the k/n behind one cell's rate, so no difference is quoted without its counts."""
    entry = summary.get(TransplantCell(recipient, condition, None).key)
    if entry is None:
        return {"cooperate_k": None, "n_parsed": None, "n_completions": None}
    return {
        "cooperate_k": int(entry["cooperate_k"]),
        "n_parsed": int(entry["n_parsed"]),
        "n_completions": int(entry["n_completions"]),
    }


def _difference(left: float | None, right: float | None) -> float | None:
    """``left - right``, or None when either side is missing."""
    return None if left is None or right is None else left - right


def _direction_block(
    summary: Mapping[str, Mapping[str, Any]],
    *,
    recipient: str,
    donor: str,
    baseline_from: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble one direction's rates and differences, tolerating a cell that was not run.

    ``baseline_from`` supplies the no-prefill rates when the caller filtered the records down to a
    prefill-only subset: a no-prefill record carries no transplant, so it never survives the
    named-no-action filter, and taking its rate from the filtered summary would report a None
    baseline and silently drop every difference below.
    """
    baselines = summary if baseline_from is None else baseline_from
    no_prefill = _cell_rate(baselines, recipient, SOURCE_NO_PREFILL)
    own = _cell_rate(summary, recipient, SOURCE_OWN_REASONING)
    other = _cell_rate(summary, recipient, SOURCE_OTHER_ARM)
    base = _cell_rate(summary, recipient, SOURCE_BASE_MODEL)
    donor_no_prefill = _cell_rate(baselines, donor, SOURCE_NO_PREFILL)
    arm_gap = _difference(donor_no_prefill, no_prefill)
    shift = _difference(other, own)
    return {
        "cooperate_rate_no_prefill": no_prefill,
        "cooperate_rate_own_reasoning": own,
        "cooperate_rate_other_arm": other,
        "cooperate_rate_base_model": base,
        "cooperate_rate_donor_no_prefill": donor_no_prefill,
        "prefill_perturbation": _difference(own, no_prefill),
        "transplant_shift": shift,
        "base_shift": _difference(base, own),
        "arm_gap": arm_gap,
        "mediated_fraction": (
            None
            if shift is None or arm_gap is None or abs(arm_gap) < ARM_GAP_FLOOR
            else shift / arm_gap
        ),
        "follow_rate_own_reasoning": _cell_field(
            summary, recipient, SOURCE_OWN_REASONING, "follow_rate"
        ),
        "follow_rate_other_arm": _cell_field(summary, recipient, SOURCE_OTHER_ARM, "follow_rate"),
        "follow_rate_base_model": _cell_field(summary, recipient, SOURCE_BASE_MODEL, "follow_rate"),
        "denominators": {
            condition: _cell_denominator(
                baselines if condition == SOURCE_NO_PREFILL else summary, recipient, condition
            )
            for condition in SOURCE_CONDITIONS
        },
    }


def mediation_readings(
    records: Sequence[Mapping[str, Any]], recipients: Sequence[str] = RECIPIENT_POPULATIONS
) -> dict[str, dict[str, Any]]:
    """Return the headline arithmetic per direction, pooled and on the confound-free subset.

    For a recipient R with donor D, four rates and three differences:

    * ``prefill_perturbation`` = R-with-own-reasoning minus R-with-nothing. How much ANY prefill
      moves R. The consistency control, and the baseline the transplant shift is measured from.
    * ``transplant_shift`` = R-with-D's-reasoning minus R-with-own-reasoning. The measurement.
    * ``arm_gap`` = D-with-nothing minus R-with-nothing. The behavioural difference training
      produced, which is what a mediated fraction is a fraction OF.
    * ``mediated_fraction`` = shift / gap, or None when the gap is under :data:`ARM_GAP_FLOOR`.

    ``base_shift`` sits beside them: the same shift under an un-adapted trace, which separates
    "follows any foreign reasoning" from "follows the donor arm's".

    Reported twice -- pooled, and over records whose transplant named no action as a tag or as a last
    label mention. The second is the reading to lead with; the first is what it is measured against.
    """
    pooled = summarise_records(records)
    unnamed = summarise_records(
        [
            record
            for record in records
            if record["transplant_action_tag"] is None
            and record["transplant_last_label_mention"] is None
        ]
    )
    return {
        f"{DONOR_FOR_RECIPIENT[recipient]}->{recipient}": {
            "recipient_population": recipient,
            "donor_population": DONOR_FOR_RECIPIENT[recipient],
            "pooled": _direction_block(
                pooled, recipient=recipient, donor=DONOR_FOR_RECIPIENT[recipient]
            ),
            "transplant_named_no_action": _direction_block(
                unnamed,
                recipient=recipient,
                donor=DONOR_FOR_RECIPIENT[recipient],
                baseline_from=pooled,
            ),
        }
        for recipient in recipients
    }


# --------------------------------------------------------------------------------------
# The plan: what a run would do, before it does any of it
# --------------------------------------------------------------------------------------


def _median(values: Sequence[int]) -> int | None:
    """Integer median, or None over nothing."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def inventory(traces: Sequence[SourceTrace]) -> dict[str, Any]:
    """Per population: how many usable traces, split by prompt, print order, pass and own action."""
    report: dict[str, Any] = {}
    for population, pool in sorted(traces_by_population(traces).items()):
        by_prompt: dict[str, int] = defaultdict(int)
        by_order: dict[str, int] = defaultdict(int)
        by_action: dict[str, int] = defaultdict(int)
        by_pass: dict[str, int] = defaultdict(int)
        for trace in pool:
            by_prompt[trace.prompt_id] += 1
            by_order[trace.label_print_order] += 1
            by_action[trace.action_stratum] += 1
            by_pass[trace.pass_name] += 1
        report[population] = {
            "n_usable": len(pool),
            "n_prompts": len(by_prompt),
            "by_prompt_id": dict(sorted(by_prompt.items())),
            "by_print_order": dict(sorted(by_order.items())),
            "by_source_action": dict(sorted(by_action.items())),
            "by_pass": dict(sorted(by_pass.items())),
            "thinking_chars_median": _median([len(trace.thinking) for trace in pool]),
            "thinking_chars_max": max((len(trace.thinking) for trace in pool), default=0),
        }
    return report


@dataclass(frozen=True)
class MediationPlan:
    """Everything a run would do, assembled before a model is loaded."""

    cells: list[TransplantCell]
    rows: list[dict[str, Any]]
    traces: list[SourceTrace]
    census: SourceCensus
    n_per_prompt: int
    seed: int
    source_sampling: dict[str, Any] = field(default_factory=dict)
    game_id: str = DEFAULT_GAME_ID

    @property
    def prompt_ids(self) -> list[str]:
        """The rendered rows' prompt ids, in the order they were generated."""
        return [str(row["prompt_id"]) for row in self.rows]

    def selections(self) -> dict[str, tuple[dict[str, list[SourceTrace]], list[Shortfall]]]:
        """Per cell key, the chosen source traces per prompt and the shortfalls."""
        pools = traces_by_population(self.traces)
        chosen: dict[str, tuple[dict[str, list[SourceTrace]], list[Shortfall]]] = {}
        for cell in self.cells:
            if cell.source_population is None:
                chosen[cell.key] = ({prompt_id: [] for prompt_id in self.prompt_ids}, [])
                continue
            chosen[cell.key] = select_source_traces(
                pools.get(cell.source_population, []),
                prompt_ids=self.prompt_ids,
                cell_key=cell.key,
                n_per_prompt=self.n_per_prompt,
                seed=self.seed,
            )
        return chosen

    def as_payload(self) -> dict[str, Any]:
        """Return the plan artifact: grid, inventory, planned counts and every shortfall."""
        selections = self.selections()
        cells: list[dict[str, Any]] = []
        shortfalls: list[dict[str, Any]] = []
        for cell in self.cells:
            chosen, cell_shortfalls = selections[cell.key]
            planned = (
                len(self.rows) * self.n_per_prompt
                if cell.source_population is None
                else sum(len(traces) for traces in chosen.values())
            )
            cells.append(
                {
                    "condition_key": cell.key,
                    "recipient_population": cell.recipient_population,
                    "source_condition": cell.source_condition,
                    "source_population": cell.source_population,
                    "direction": cell.direction,
                    "n_prompts": len(self.rows),
                    "n_planned_completions": planned,
                }
            )
            shortfalls.extend(shortfall.as_payload() for shortfall in cell_shortfalls)
        orders = sorted({str(row["label_print_order"]) for row in self.rows})
        return {
            "game_id": self.game_id,
            "render_grading": render_grading_for(self.game_id),
            "split": SPLIT_EVAL,
            "label_print_orders": orders,
            "n_prompt_rows": len(self.rows),
            "prompt_ids": self.prompt_ids,
            "traces_per_prompt": self.n_per_prompt,
            "seed": self.seed,
            "source_sampling": self.source_sampling,
            "source_census": self.census.as_payload(),
            "source_inventory": inventory(self.traces),
            "cells": cells,
            "n_planned_completions_total": sum(
                int(entry["n_planned_completions"]) for entry in cells
            ),
            "shortfalls": shortfalls,
            "print_order_caveat": (
                f"The {self.game_id} source traces exist at the canonical print order only, so "
                f"this run reads one order. Both orders need freshly generated sources."
                if orders == ["canonical"]
                else None
            ),
        }


def recipient_sampling(max_new_tokens: int) -> SamplingConfig:
    """Return the recipient's decoding config: the training distribution at a widened budget.

    The training distribution rather than the vendor preset, for the reason
    :mod:`games.eval_sampler` documents -- the arms were optimised under it, the source traces were
    drawn under it, and the vendor preset's presence penalty removed about three quarters of the
    trained movement this instrument exists to read.
    """
    return replace(
        eval_sampling(SAMPLER_TRAINING_DISTRIBUTION, thinking=True), max_new_tokens=max_new_tokens
    )


def resolve_sampling(args: argparse.Namespace) -> SamplingConfig:
    """Return the config the recipient will actually decode under, with every CLI flag applied.

    One resolution site, because three consumers need the same answer: the context-budget check,
    the engine, and the artifact's ``resolved_sampler``. Two of them reading a different budget than
    the third is how a run passes its own window check and then truncates.

    ``--max-new-tokens`` belongs to :func:`reward_hacking.backend_cli.add_backend_args` rather than
    to this CLI, so its unset default resolves here to
    :data:`~games.eval_sampler.DEFAULT_EVAL_MAX_NEW_TOKENS` -- above the longest answer segment
    measured on this corpus.
    """
    base = recipient_sampling(
        DEFAULT_EVAL_MAX_NEW_TOKENS if args.max_new_tokens is None else int(args.max_new_tokens)
    )
    if args.backend not in backend_cli.LOCAL_KINDS:
        return base
    return backend_cli.local_sampling_from_args(args, base)


MOCK_SAMPLER_REASON = "the mock backend returns canned strings and samples nothing"


def sampler_record(args: argparse.Namespace, sampling: SamplingConfig) -> dict[str, Any]:
    """Return the artifact's sampler record: what the engine ran with, or that nothing did.

    A mock run gets every knob in the DROPPED partition rather than an engine's applied block. The
    alternative -- reporting the config as though ``hf`` had honoured it -- would put a sampler that
    never ran into a file read months later, which is the exact false claim
    :class:`~reward_hacking.interp.generation_capture.ResolvedSampler` exists to prevent.
    """
    if args.backend != BACKEND_VLLM:
        return ResolvedSampler(
            engine=args.backend,
            applied={},
            dropped={
                name: DroppedKnob(requested=getattr(sampling, name), why=MOCK_SAMPLER_REASON)
                for name in SAMPLING_FIELD_NAMES
            },
        ).as_payload()
    return resolved_sampler_for(sampling, engine=GEN_ENGINE_VLLM).as_payload()


def build_plan(args: argparse.Namespace) -> MediationPlan:
    """Read the corpus and the source inventory, and assemble the plan. No model, no GPU."""
    traces, census = load_source_traces(
        args.source_root, tuple(args.source_pass), game_id=args.game
    )
    source_sampling = read_source_sampling(args.source_root, tuple(args.source_pass))
    assert_sampler_matches_sources(resolve_sampling(args), source_sampling)
    rows = eval_rows(tuple(args.label_print_orders), game_id=args.game)
    available = {trace.label_print_order for trace in traces}
    missing = sorted({str(row["label_print_order"]) for row in rows} - available)
    if missing:
        raise ValueError(
            f"no {args.game} source traces exist at print order(s) {missing}, so those rows would "
            f"have a no-prefill baseline and no transplant to compare it against. Pass "
            f"--label-print-orders with an available order, or generate source traces at the "
            f"missing order first. Available: {sorted(available)}."
        )
    return MediationPlan(
        cells=transplant_cells(tuple(args.recipient), tuple(args.source_condition)),
        rows=rows,
        traces=traces,
        census=census,
        n_per_prompt=args.traces_per_prompt,
        seed=args.seed,
        source_sampling=source_sampling,
        game_id=args.game,
    )


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def _past_deadline(deadline: dt.datetime | None) -> bool:
    return deadline is not None and dt.datetime.now(dt.UTC) >= deadline


@dataclass(frozen=True)
class ExecutionConfig:
    """The knobs execution needs, grouped so the driver takes a config rather than nine scalars."""

    out_dir: Path
    max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS
    max_model_len: int | None = None
    max_source_response_tokens: int | None = None
    batch_size: int | None = DEFAULT_BATCH_SIZE
    sabotage: str = SABOTAGE_NONE
    deadline: dt.datetime | None = None
    sampler_payload: dict[str, Any] = field(default_factory=dict)


def prepare_units(
    plan: MediationPlan, tokenizer: ChatTokenizer, config: ExecutionConfig
) -> tuple[list[PreparedUnit], list[Shortfall]]:
    """Assemble and guard every context the run will generate from, before any of it is generated.

    All of it up front rather than per cell, because :func:`assert_context_fits` is only a real check
    over the whole selection: sizing the engine from the first cell and meeting a longer context in
    the fourth is the failure it exists to prevent.
    """
    rendered = {
        str(row["prompt_id"]): render_context(tokenizer, str(row["prompt"])) for row in plan.rows
    }
    pools = traces_by_population(plan.traces)
    rows_by_id = {str(row["prompt_id"]): row for row in plan.rows}

    def accept(trace: SourceTrace) -> bool:
        if config.max_source_response_tokens is None:
            return True
        length = len(tokenizer.encode(trace.thinking, add_special_tokens=False))
        return length <= config.max_source_response_tokens

    units: list[PreparedUnit] = []
    shortfalls: list[Shortfall] = []
    for cell in plan.cells:
        if cell.source_population is None:
            units.extend(
                prepare_unit(
                    tokenizer,
                    rendered[prompt_id],
                    cell=cell,
                    row=rows_by_id[prompt_id],
                    trace=None,
                    sample_index=index,
                )
                for prompt_id in plan.prompt_ids
                for index in range(plan.n_per_prompt)
            )
            continue
        chosen, cell_shortfalls = select_source_traces(
            pools.get(cell.source_population, []),
            prompt_ids=plan.prompt_ids,
            cell_key=cell.key,
            n_per_prompt=plan.n_per_prompt,
            seed=plan.seed,
            accept=accept,
        )
        shortfalls.extend(cell_shortfalls)
        for prompt_id in plan.prompt_ids:
            units.extend(
                prepare_unit(
                    tokenizer,
                    rendered[prompt_id],
                    cell=cell,
                    row=rows_by_id[prompt_id],
                    trace=trace,
                    sample_index=index,
                    sabotage=config.sabotage,
                )
                for index, trace in enumerate(chosen[prompt_id])
            )
    return units, shortfalls


def _submission_widths(n_units: int, batch_size: int | None) -> list[int]:
    """How many contexts each engine call carries: the whole cell by default, or fixed chunks.

    ``None`` is one call for the cell, which is what :data:`DEFAULT_BATCH_SIZE` means and why it is
    the default: every ``engine.generate`` call is a barrier behind its own slowest continuation, and
    these continuations run from 20k-token contexts to a 65,536-token answer cap, so a fixed width
    made the cell wait for the worst row in each group of that many rather than once at the end. The
    engine schedules its own batch either way; the width only decides how often the caller blocks.

    A number narrows that to fixed chunks, which is a persistence knob rather than a throughput one:
    records land per call (see :func:`_generate_cell`), so a chunked cell survives a wall-clock kill
    with its finished chunks on disk where a whole-cell call loses the cell. Kept for a smoke and for
    a box whose deadline is tighter than one cell.

    Statistically neutral, and NOT reproducible-identical: this module's recipient sampler carries no
    per-request seed (`recipient_sampling` leaves `SamplingConfig.seed` at None, and `VLLMBackend`
    passes it through), so requests share the engine's global RNG stream and what a context draws
    depends on what else is in flight. Same distribution, a different draw -- the same grade the eval
    battery's pooled submission carries, and the reason the width is recorded in the summary.
    """
    if batch_size is None:
        return [n_units]
    if batch_size < 1:
        raise ValueError(
            f"--batch-size must be positive or absent (the whole cell), got {batch_size}"
        )
    return [min(batch_size, n_units - start) for start in range(0, n_units, batch_size)]


def _generate_cell(
    backend: object,
    cell_units: Sequence[PreparedUnit],
    *,
    cell_key: str,
    batch_size: int | None,
    handle: TextIO,
) -> list[dict[str, Any]]:
    """Generate one cell, appending each record to the open records file as its call lands.

    One engine call per cell by default (:func:`_submission_widths`), so the cell blocks once on its
    own longest continuation instead of once per chunk. Written and flushed per call rather than at
    the end, so a hold killed by its wall-clock limit leaves the cells it finished behind instead of
    nothing -- and under an explicit ``--batch-size`` the finished chunks of the cell it died in too.
    """
    records: list[dict[str, Any]] = []
    start = 0
    for width in _submission_widths(len(cell_units), batch_size):
        chunk = cell_units[start : start + width]
        start += width
        continuations = continue_raw(backend, [unit.context for unit in chunk])
        for unit, continuation in zip(chunk, continuations, strict=True):
            record = build_record(unit, continuation)
            records.append(record)
            handle.write(json.dumps(record) + "\n")
        handle.flush()
        logger.info(f"cell {cell_key}: {start}/{len(cell_units)} completions")
    return records


def execute_transplant(
    plan: MediationPlan,
    *,
    open_backend: Callable[[str], AbstractContextManager[object]],
    tokenizer: ChatTokenizer,
    config: ExecutionConfig,
) -> dict[str, Any]:
    """Generate every cell and write the records and the summary.

    ``open_backend`` maps a recipient population to a context manager holding the model that answers
    for it, and the loop opens exactly ONE at a time. That is not tidiness: the two recipients are
    different weights, and a vLLM engine claims a fraction of the WHOLE card at construction, so
    holding both would either fail to allocate the second or force both engines onto a KV cache too
    small for the 20k-token contexts this experiment feeds them. Cells are therefore grouped by
    recipient and the engine is torn down between groups -- which the first real run proved is a
    seam and not a formality, dying at it with the first engine still holding 41 of 44 GiB. What
    makes the swap trustworthy now is that :func:`backend_opener`'s teardown polls the card and
    refuses to hand a full one to the next recipient.

    One records file is opened once and held open ACROSS recipients, so the second recipient appends
    behind the first rather than reopening and truncating what the first banked.

    Every context is assembled and guarded, and the whole run's context budget checked, BEFORE the
    first engine loads -- so a run that cannot fit fails in seconds rather than after a model load.
    """
    units, shortfalls = prepare_units(plan, tokenizer, config)
    budget = assert_context_fits(
        units, max_new_tokens=config.max_new_tokens, max_model_len=config.max_model_len
    )

    config.out_dir.mkdir(parents=True, exist_ok=True)
    records_path = config.out_dir / RECORDS_FILENAME
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    by_cell: dict[str, list[PreparedUnit]] = defaultdict(list)
    for unit in units:
        by_cell[unit.cell.key].append(unit)
    cells_by_recipient: dict[str, list[TransplantCell]] = defaultdict(list)
    for cell in plan.cells:
        cells_by_recipient[cell.recipient_population].append(cell)

    with records_path.open("w", encoding="utf-8") as handle:
        for recipient, cells in cells_by_recipient.items():
            runnable = [cell for cell in cells if by_cell.get(cell.key)]
            skipped.extend(
                {"condition_key": cell.key, "skipped": "no units selected"}
                for cell in cells
                if not by_cell.get(cell.key)
            )
            for cell in cells:
                if not by_cell.get(cell.key):
                    logger.warning(f"cell {cell.key} selected no units; skipping")
            if not runnable:
                continue
            if _past_deadline(config.deadline):
                skipped.extend(
                    {"condition_key": cell.key, "skipped": "deadline"} for cell in runnable
                )
                logger.warning(f"deadline passed; skipping every remaining cell of {recipient}")
                continue
            with open_backend(recipient) as backend:
                for cell in runnable:
                    if _past_deadline(config.deadline):
                        skipped.append({"condition_key": cell.key, "skipped": "deadline"})
                        logger.warning(f"deadline passed; skipping cell {cell.key}")
                        continue
                    records.extend(
                        _generate_cell(
                            backend,
                            by_cell[cell.key],
                            cell_key=cell.key,
                            batch_size=config.batch_size,
                            handle=handle,
                        )
                    )

    recipients = sorted({cell.recipient_population for cell in plan.cells})
    summary: dict[str, Any] = {
        "command": "transplant",
        "game_id": plan.game_id,
        "render_grading": render_grading_for(plan.game_id),
        "split": SPLIT_EVAL,
        "label_print_orders": sorted({str(row["label_print_order"]) for row in plan.rows}),
        "traces_per_prompt": plan.n_per_prompt,
        "seed": plan.seed,
        # None is one call per cell; a number is contexts per call, so also how often records land.
        "submission_batch_size": config.batch_size,
        "sabotage": config.sabotage,
        "context_budget": budget,
        "max_source_response_tokens": config.max_source_response_tokens,
        "resolved_sampler": dict(config.sampler_payload),
        "source_sampling": plan.source_sampling,
        "source_census": plan.census.as_payload(),
        "shortfalls": [shortfall.as_payload() for shortfall in shortfalls],
        "n_records": len(records),
        "skipped_cells": skipped,
        "conditions": summarise_records(records),
        "mediation": mediation_readings(records, recipients),
    }
    summary_path = config.out_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(
        f"transplant written, records={records_path} summary={summary_path} "
        f"n_records={len(records)} skipped={len(skipped)}"
    )
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def adapter_for(population: str, adapter_root: Path) -> Path:
    """Return the checkpoint directory serving one recipient population."""
    arm, _, step = population.partition("/step-")
    checkpoint = adapter_root / arm / f"checkpoint-{step}"
    if not (checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"{checkpoint} is not a PEFT adapter directory, so {population} cannot be served "
            f"un-merged. Point --adapter-root at the directory holding <arm>/checkpoint-<step>."
        )
    return checkpoint


def mock_answers(rows: Sequence[Mapping[str, Any]]) -> Callable[[str], str]:
    """Build a scripted answer naming a label the row actually offers, so a mock cell parses.

    A static response list cannot do this: the twin-pd reskins print different label words, so one
    canned tag would be a parse failure on every reskin but one and the smoke would exercise only the
    parse-failure path. Which of the two labels is answered alternates with context length, so a mock
    summary carries both actions rather than a degenerate single-valued rate.
    """

    def answer(context: str) -> str:
        row = next((row for row in rows if str(row["prompt"]) in context), None)
        if row is None:
            raise ValueError(
                "the mock backend was handed a context matching no planned prompt row, so the "
                "rendering under test is not the rendering the plan built."
            )
        labels = (str(row["label_a"]), str(row["label_b"]))
        tag = f"\n\n<action>{labels[len(context) % 2]}</action>"
        # A no-prefill context has an OPEN thinking block, so a real model would close it before
        # answering. Without that the baseline cell reads as truncated, its rate comes back None,
        # and every mediation difference the smoke is meant to exercise silently vanishes.
        if THINK_CLOSE in context:
            return tag
        return f"mock deliberation.{THINK_CLOSE}{tag}"

    return answer


def backend_opener(
    args: argparse.Namespace, plan: MediationPlan, *, policy: DrainPolicy = DEFAULT_DRAIN_POLICY
) -> Callable[[str], AbstractContextManager[object]]:
    """Build the per-recipient backend factory, one engine live at a time.

    A factory rather than a preloaded mapping because a vLLM engine claims a fraction of the whole
    card as it constructs, so two of them cannot coexist on one GPU without starving each other's KV
    cache -- see :func:`execute_transplant`. Each engine is verified to actually be serving its
    adapter (:func:`games.eval_model.verify_served_model`) before it answers anything.

    Teardown goes through :func:`games.vllm_teardown.release_engine`, which shuts vLLM's EngineCore
    subprocess down and then POLLS the card until the memory is back. This used to be ``del backend``
    plus ``torch.cuda.empty_cache()``, and both halves were no-ops: the caller's ``with`` target still
    held the reference, and the memory belonged to a child process the parent's allocator cannot
    reach. The first real run died at exactly this seam with 2.96 of 44.39 GiB free -- see that
    module's docstring for the log.
    """
    sampling = resolve_sampling(args)

    @contextmanager
    def open_backend(population: str) -> Generator[object]:
        if args.backend == BACKEND_MOCK:
            logger.warning(
                f"mock backend: nothing is sampled for {population}, and the summary will read "
                f"engine=mock with every sampling knob reported as dropped"
            )
            yield MockBackend(mock_answers(plan.rows), model_id=f"mock:{population}")
            return
        served = resolve_served_model(
            checkpoint=adapter_for(population, args.adapter_root),
            base_model=args.base_model,
            backend_kind=args.backend,
            merge_root=args.merge_root,
            merge_label=f"{population.replace('/', '-')}-",
        )
        if served.model_id != args.base_model:
            raise ValueError(
                f"{population} resolved to load mode {served.load_mode!r}, which serves "
                f"{served.model_id!r} rather than the base checkpoint. This module renders and "
                f"tokenises against --base-model, so a served id that differs would tokenise the "
                f"transplant seam with a different tokenizer than the engine reads it with."
            )
        extra: dict[str, object] = dict(served.backend_kwargs)
        if args.max_model_len is not None:
            extra["max_model_len"] = args.max_model_len
        # Read BEFORE the engine exists, because the release below asks "did the card come back to
        # where it was". A baseline taken afterwards would include this engine's own claim, so the
        # residue would be zero however much of the card was still held.
        baseline_mib = vram_used_mib()
        logger.info(f"{population} loading onto a card holding {baseline_mib} MiB")
        backend = backend_cli.backend_from_args(
            args, served.model_id, local_sampling=sampling, extra_kwargs=extra
        )
        try:
            # Inside the `try` because raising is what this check is FOR, and before a completion
            # rather than after because base weights served silently make the cell read as an arm
            # that never trained. Outside, its one failure case leaves the card held for everyone.
            verify_served_model(backend, served)
            logger.info(f"{population} served {served.load_mode}: {served.provenance}")
            yield backend
        finally:
            # In a `finally` rather than only on the success path: an engine left up after a failed
            # cell would take the card down with it for every later recipient. A release failure here
            # chains onto whatever stopped the run, so both are in the log.
            release_engine(backend, baseline_mib=baseline_mib, policy=policy)

    return open_backend


def load_tokenizer(base_model: str) -> ChatTokenizer:
    """Load the tokenizer the seam is assembled with, which is the one the engine reads it with.

    Identical by construction rather than by agreement: on the runtime-adapter path the engine is
    built from ``--base-model`` and :class:`~reward_hacking.model_backend.VLLMBackend` builds its own
    tokenizer with this same call on that same id, and :func:`backend_opener` refuses any load mode
    that serves a different id. Loaded here rather than off the engine because every context is
    assembled and guarded before the first engine exists.
    """
    return cast("ChatTokenizer", AutoTokenizer.from_pretrained(base_model))


def run_transplant(args: argparse.Namespace) -> dict[str, Any]:
    """Plan, then either report the plan or run it."""
    if args.backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"--backend {args.backend} cannot continue a raw prefilled context. Use "
            f"{' or '.join(SUPPORTED_BACKENDS)}: vLLM is the one engine that serves a LoRA adapter "
            f"un-merged AND continues from a raw string without re-applying the chat template."
        )
    backend_cli.reject_inapplicable_knobs(args.backend, args)
    plan = build_plan(args)
    payload = plan.as_payload()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / PLAN_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        f"plan written, out_dir={args.out_dir} cells={len(payload['cells'])} "
        f"completions={payload['n_planned_completions_total']} "
        f"shortfalls={len(payload['shortfalls'])}"
    )
    if args.plan_only:
        return payload
    sampling = resolve_sampling(args)
    config = ExecutionConfig(
        out_dir=args.out_dir,
        max_new_tokens=sampling.max_new_tokens,
        max_model_len=args.max_model_len,
        max_source_response_tokens=args.max_source_response_tokens,
        batch_size=args.batch_size,
        sabotage=args.sabotage,
        deadline=None if args.deadline is None else dt.datetime.fromisoformat(args.deadline),
        sampler_payload=sampler_record(args, sampling),
    )
    return execute_transplant(
        plan,
        open_backend=backend_opener(args, plan),
        tokenizer=load_tokenizer(args.base_model),
        config=config,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. ``--plan-only`` is the no-model gate a rented box runs before spending GPU."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    backend_cli.add_backend_args(parser, default=BACKEND_VLLM)
    parser.add_argument(
        "--game",
        default=DEFAULT_GAME_ID,
        choices=TRANSPLANTABLE_GAME_IDS,
        help="Which game's eval rows are rendered and whose source cells are accepted. One-shot "
        "matrix games only; the grading resolves from the shared maps.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=Path("artifacts/games/interp-capture/adapters"),
        help="Holds <arm>/checkpoint-<step> for each recipient population.",
    )
    parser.add_argument("--merge-root", type=Path, default=Path("artifacts/games/merged"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/games/resample-analysis"),
        help="Root holding the resample passes whose completions are transplanted.",
    )
    parser.add_argument("--source-pass", action="append", default=None, metavar="PASS_DIR")
    parser.add_argument("--recipient", action="append", default=None, choices=RECIPIENT_POPULATIONS)
    parser.add_argument(
        "--source-condition", action="append", default=None, choices=SOURCE_CONDITIONS
    )
    parser.add_argument(
        "--label-print-orders",
        action="append",
        default=None,
        choices=LABEL_PRINT_ORDERS,
        help="Rendered print orders. Reusable twin-pd source traces are canonical-only.",
    )
    parser.add_argument("--traces-per-prompt", type=int, default=DEFAULT_TRACES_PER_PROMPT)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Engine context window. Checked against the longest selected context before generating.",
    )
    parser.add_argument(
        "--max-source-response-tokens",
        type=int,
        default=None,
        help="Exclude source traces whose thinking block is longer. Counted, never silent.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Contexts per engine call. Unset is one call per cell, which blocks once on the cell's "
        "longest continuation instead of once per chunk; a number trades that back for records "
        "landing per chunk.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--sabotage",
        choices=SABOTAGE_MODES,
        default=SABOTAGE_NONE,
        help="Plant a violation an assembly guard must refuse, to watch the guard go red.",
    )
    parser.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC; no new cell starts after it."
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def resolve_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CLI, filling the repeatable flags' defaults after the fact.

    ``action="append"`` on a flag with a list default appends to that default, so the defaults are
    applied here instead -- otherwise ``--recipient twin-pd-self/step-70`` would silently run both
    arms.

    ``--thinking`` is deliberately left unset rather than defaulted to true. It selects a branch of
    the backend's own chat template, which the raw-continuation path never calls -- :func:`render_context`
    renders with ``enable_thinking=True`` explicitly and asserts the block came back open. Setting the
    flag here would make ``--backend mock`` fail the knob registry, which is the registry correctly
    saying the mock honours no such knob.
    """
    args = build_parser().parse_args(argv)
    args.source_pass = args.source_pass or list(DEFAULT_SOURCE_PASSES)
    args.recipient = args.recipient or list(RECIPIENT_POPULATIONS)
    args.source_condition = args.source_condition or list(SOURCE_CONDITIONS)
    args.label_print_orders = args.label_print_orders or list(LABEL_PRINT_ORDERS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the transplant CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    run_transplant(resolve_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
