"""Instrument validation for residual steering: can we steer this model at all, and under what conditions.

The reward-hacking steering read came back null with no established cause: two alpha values, two
prompts per cell, effects of opposite sign across concepts, and the largest apparent effect traced
to a single placebo arm. Before any concept axis is steered again, the rig itself has to be shown
to work on a target whose behavioural signature is impossible to miss. That is what this module is.

**The target is output script, and it is the literature's own choice.** Prompts are neutral English
questions; the observable is the fraction of letters in the response that are Han ideographs (or
Cyrillic, for the second direction). arXiv:2608.12334 validates exactly this on **Qwen3.5-2B**, one
rung below our working size and the same architecture family: an English baseline of 0.000 rising to
0.981-0.994 under steering, with a matched-norm random control at 0.027, scored by "fraction of
generated characters in Chinese script" -- pure counting, no judge and no classifier. So a null here
cannot be blamed on the target, and there is a published number to fail against rather than only a
hope. Two things carried over from that paper: its best layers on the 2B were 17-23 of 24, and its
**final layer was a "high-gain perturbation trap" that collapsed into repetitive loops**, which is why
:data:`MAX_LAYER_FRACTION` keeps layer selection out of the top of the trunk. The capability is
certainly present in a bilingual checkpoint, so a shift measures a changed *choice* rather than a
created ability, which is the distinction the whole thread rests on.

**Four things the prior read got wrong that this module fixes by construction.**

1. *Two points, one at the floor of the published safe band and one past its ceiling.* The
   steering literature's damage parameter is the RELATIVE displacement
   ``rho = ||h' - h|| / ||h||``, with a published safe envelope of 0.1 to 0.4 (arXiv:2608.12334).
   ``run_harness`` already sets ``alpha = scale * mean_L2_residual_norm``, so its ``scale`` IS rho,
   and the two values tried were 0.1 and 0.5: the bottom of the safe band and just past the top of
   it, with nothing in between and no way to see a non-monotone response. Dose-response is reported
   as reliably non-monotone, so a two-point sweep can read as a null. This module keeps rho as the
   parameter (:func:`alpha_from_relative_displacement`), sweeps it symmetrically across and beyond
   the safe band, and records the absolute displacement and both norm conventions beside it.
2. *One position regime was tested and it was the compounding one.* ``steering_hook`` has no
   position restriction and the hook stays installed through generation, so every one of up to
   65,536 decode steps was steered and the perturbation compounded into a recurrent state.
   :class:`PositionArm` splits that into prefill-only, all-positions and decode-only.
3. *Means over two items.* Every record here is one (prompt, arm, cell, sample) row carrying its
   raw generation; the readout reports the per-item distribution and an explicit count of items
   that moved the wrong way, because within-concept steerability is documented to be bimodal.
4. *Raw generations were never uploaded and are permanently gone.* The corpus here is generated
   from a template and a common-noun list, so no generation can contain benchmark material and
   every raw response is safe to retain and ship.

**Controls, and an honest account of where each comes from.** A matched-norm Gaussian placebo,
constructed the way arXiv:2608.12334 constructs it -- uniform on the unit sphere, scaled to the norm
of the *actual displacement applied* rather than of the raw direction (on that easy target it read
0.027 against the real axis's 0.994). A matched-norm component *orthogonal* to the direction, because
arXiv:2602.06801 reports orthogonal components achieving indistinguishable impact; that paper has no
unsteered baseline at all, so its result cannot separate "the orthogonal component works too" from
"nothing worked", which is why the alpha-zero arm here is reported in the same units as the other two.
A signed sweep, whose canonical source is CAA (arXiv:2312.06681): a valid direction predicts a
monotone OPPOSITE effect, and a same-sign response at plus and minus alpha means perturbation rather
than direction. And a **shuffled-label direction, which is OUR OWN ADDITION and not field standard**
-- a search of the steering literature found it named nowhere. It permutes the contrast labels before
the difference of means, so it preserves the estimator, the data and the covariance while destroying
the contrast, which makes it strictly stronger than a Gaussian of matched norm. Two things to keep in
view: the handoff attributed this trio to arXiv:2607.13162, but there it appears in the *Limitations*
section as controls the authors did NOT run; and matched-norm random directions are not inert
(arXiv:2509.22067 found 387 of 1000 random directions breaking at least 5 of 100 prompts), so a
placebo arm coming out non-clean is a finding rather than a bug.

**Two tiers, cheap first.** :func:`score_teacher_forced_gap` reads the model's preference between a
Chinese and an English continuation of the same question in one batched forward pass -- no sampling,
no decode loop, ~three orders of magnitude cheaper than generation -- which buys a full layer sweep
and a wide rho sweep. That is a *stated-preference* readout and is labelled as such on every record:
it is a screen for where to look, and generation is the behavioural confirmation. Generation on this
path is slower than it looks, which is the other reason the screen carries the wide sweep: in this
environment fla 0.5.2 exports no ``recurrent_gated_delta_rule``, so transformers' decode-path kernel
lookup misses and **single-token decode silently falls back to the pure-torch Gated DeltaNet loop**
while prefill still uses the Triton chunk kernel.

**The deterministic damage stack, applied in the published order.** (a) rho itself, computable from
activations before anything is generated, against the 0.1-0.4 envelope; (b) the KL divergence of the
next-token distribution at the last prompt position, intervened against not -- Arditi's
``kl_score < 0.1`` is the one published numeric threshold for a deterministic coherence guard;
(c) repeated-n-gram repetition rate (published good ~0.036-0.043, degraded ~0.124-0.126) and
token-level Shannon entropy, which is precisely the signature that pins a generation at its token
cap and would have caught our 65,536 pinning; (d) per-item cap-hit rate, our own symptom, reported
raw. No LLM judge anywhere in it.

**The recurrent-state hazard, and why this module checks it.** ``transformers`` 5.15.0 writes the
Gated DeltaNet state into the cache in place (``cache_utils.DynamicLayer.update_recurrent_state``
does ``self.recurrent_states[state_idx].copy_(...)``, commented "we copy instead of assigning, to
preserve the static address for cudagraphs"; ``modeling_qwen3_5`` line 211 does the same for the
conv state). Two conditions sharing one cache object therefore corrupt each other. Our path builds
a fresh cache per ``generate`` call, so it should be clean -- and :func:`cache_isolation_report`
proves that on the actual model rather than trusting the reading, with a deliberate sabotage arm
that shares a cache and must come back red.
"""

from __future__ import annotations

import logging
import math
import time
import unicodedata
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol, cast

import torch

from reward_hacking.interp.directions import (
    _transformer_trunk,  # pyright: ignore[reportPrivateUsage]  # the shared trunk resolver
    diff_of_means,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.generation_capture import (
    PENALTY_FREE_THINKING_SAMPLING,
    GenerationRecord,
    generate_response,
)
from reward_hacking.interp.steering import Hook, residual_intervention
from reward_hacking.model_backend import SamplingConfig

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Deterministic scorers: what the response is written in, and whether it is intact
# --------------------------------------------------------------------------------------

SCRIPT_HAN = "han"
SCRIPT_CYRILLIC = "cyrillic"
SCRIPT_LATIN = "latin"
TARGET_SCRIPTS: tuple[str, ...] = (SCRIPT_HAN, SCRIPT_CYRILLIC)
"""The scripts a steering target can ask for. Latin is the baseline the model already writes in."""

_SCRIPT_RANGES: dict[str, tuple[tuple[int, int], ...]] = {
    SCRIPT_HAN: ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF)),
    SCRIPT_CYRILLIC: ((0x0400, 0x04FF), (0x0500, 0x052F)),
    SCRIPT_LATIN: ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)),
}
"""Codepoint ranges per script: Unified Ideographs plus Extension A and the compatibility block.

CJK *punctuation* (U+3000-U+303F) is deliberately absent. A Chinese full stop inside an otherwise
English sentence is not the model switching language, and counting it would let a formatting tic
read as a script shift.
"""


def _in_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(low <= codepoint <= high for low, high in ranges)


def count_script_chars(text: str, script: str) -> int:
    """How many characters of ``text`` fall in ``script``'s codepoint ranges."""
    if script not in _SCRIPT_RANGES:
        raise ValueError(f"unknown script {script!r}; expected one of {sorted(_SCRIPT_RANGES)}")
    ranges = _SCRIPT_RANGES[script]
    return sum(1 for char in text if _in_ranges(ord(char), ranges))


def count_other_letters(text: str) -> int:
    """Letters in no tracked script: the collapse-into-random-Unicode signature.

    A steered arm that stops writing language altogether emits letters from scripts nobody asked
    for. Counting them separately keeps that failure out of the target-script numerator *and* out of
    the Latin denominator, so a collapse cannot be read as a successful script shift.
    """
    tracked = tuple(_SCRIPT_RANGES.values())
    return sum(
        1
        for char in text
        if unicodedata.category(char).startswith("L")
        and not any(_in_ranges(ord(char), ranges) for ranges in tracked)
    )


@dataclass(frozen=True)
class ScriptScore:
    """The script composition of one response, counts included so a zero has its denominator.

    ``ratio`` is ``target / (target + latin)``: the share of *language-bearing* characters written
    in the target script, which is 0.0 for an English answer and ~1.0 for a Chinese one. It is
    ``None`` -- never 0.0 -- when the response carries no letters at all, because an empty response
    and a fully-English one are different outcomes and a sweep that silently merges them would read
    a collapse as a null effect.
    """

    script: str
    target_chars: int
    latin_chars: int
    other_letters: int
    total_chars: int
    ratio: float | None

    @property
    def denominator(self) -> int:
        """Language-bearing characters the ratio was computed over."""
        return self.target_chars + self.latin_chars


def script_score(text: str, script: str) -> ScriptScore:
    """Score ``text`` for how much of it is written in ``script`` rather than in Latin."""
    target = count_script_chars(text, script)
    latin = count_script_chars(text, SCRIPT_LATIN)
    denominator = target + latin
    return ScriptScore(
        script=script,
        target_chars=target,
        latin_chars=latin,
        other_letters=count_other_letters(text),
        total_chars=len(text),
        ratio=(target / denominator) if denominator else None,
    )


def distinct_ngram_ratio(token_ids: Sequence[int], n: int = 3) -> float | None:
    """Return unique n-grams over total n-grams: the degeneracy measure, ``None`` when too short.

    A generation that has collapsed into a loop -- the documented high-strength steering failure --
    repeats the same few n-grams, so this falls toward 1/total. Computed on token ids rather than
    characters so it does not depend on how a script segments into words, which is exactly what
    varies across the arms here.
    """
    if len(token_ids) < n:
        return None
    grams = [tuple(token_ids[i : i + n]) for i in range(len(token_ids) - n + 1)]
    return len(set(grams)) / len(grams)


def repetition_rate(token_ids: Sequence[int], n: int = 3) -> float | None:
    """Share of n-grams that are repeats: ``1 - distinct_ngram_ratio``. ``None`` when too short.

    Reported in this direction rather than as the distinct ratio so the number is directly comparable
    to the published figures for this measure -- roughly 0.036-0.043 on intact generations and
    0.124-0.126 on degraded ones (arXiv:2608.12334). Repetition is the signature that pins a
    generation at its token cap, which is the diagnostic our own 65,536-token pinning lacked.
    """
    distinct = distinct_ngram_ratio(token_ids, n)
    return None if distinct is None else 1.0 - distinct


def token_entropy(token_ids: Sequence[int]) -> float | None:
    """Shannon entropy in bits of the empirical token distribution. ``None`` on an empty sequence.

    A second, n-gram-free view of the same degeneration: a generation locked onto a handful of tokens
    has low entropy however its n-grams happen to fall. Computed on ids rather than characters so it
    does not depend on how a script segments into words, which is exactly what varies across the arms
    here.
    """
    if not token_ids:
        return None
    counts: dict[int, int] = {}
    for token in token_ids:
        counts[token] = counts.get(token, 0) + 1
    total = len(token_ids)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def longest_token_run(token_ids: Sequence[int]) -> int:
    """Length of the longest run of one repeated token id. Catches the single-token lock-up."""
    longest = 0
    run = 0
    previous: int | None = None
    for token in token_ids:
        run = run + 1 if token == previous else 1
        previous = token
        longest = max(longest, run)
    return longest


# --------------------------------------------------------------------------------------
# The corpus: generated, never authored, so no committed text is an item
# --------------------------------------------------------------------------------------

PROMPT_TEMPLATE = "In two or three sentences, explain what {subject} is and why it matters."
"""One neutral template. Its natural answer is English prose, which is the 0-ratio baseline."""

CORPUS_SUBJECTS: tuple[str, ...] = (
    "a bridge",
    "a compass",
    "a dictionary",
    "a glacier",
    "a harbour",
    "a keyboard",
    "a library",
    "a magnet",
    "a museum",
    "a pendulum",
    "a printing press",
    "a refrigerator",
    "a telescope",
    "a thermometer",
    "a vaccine",
    "a windmill",
    "an aqueduct",
    "an elevator",
    "photosynthesis",
    "sediment",
    "soap",
    "solar power",
    "the alphabet",
    "the water cycle",
    "a lighthouse",
    "a canal lock",
    "a wheelbarrow",
    "a sundial",
    "a barometer",
    "a seismograph",
    "a microscope",
    "a battery",
    "a bicycle",
    "a camera",
    "a clock",
    "a dam",
    "a fire extinguisher",
    "a fuse",
    "a greenhouse",
    "a hinge",
    "a kiln",
    "a ladder",
    "a lens",
    "a lever",
    "a lock",
    "a loom",
    "a map",
    "a mill",
    "a mirror",
    "a needle",
    "a net",
    "a paddle",
    "a plough",
    "a pulley",
    "a pump",
    "a radiator",
    "a rope",
    "a rudder",
    "a saddle",
    "a sail",
    "a saw",
    "a scale",
    "a screw",
    "a sieve",
    "a spring",
    "a stove",
    "a tent",
    "a thermostat",
    "a tunnel",
    "a turbine",
    "a valve",
    "a wedge",
    "a well",
    "a wheel",
    "a whistle",
    "a zip fastener",
    "an anchor",
    "an anvil",
    "an arch",
    "an axle",
    "an oar",
    "an umbrella",
    "antibiotics",
    "cement",
    "compost",
    "coral",
    "cotton",
    "curdling",
    "dew",
    "distillation",
    "erosion",
    "evaporation",
    "fermentation",
    "fertiliser",
    "filtration",
    "friction",
    "frost",
    "germination",
    "glass",
    "granite",
    "gravity",
    "hail",
    "humidity",
    "ice",
    "insulation",
    "iron",
    "irrigation",
    "kelp",
    "lightning",
    "limestone",
    "lubrication",
    "magnetism",
    "marble",
    "mortar",
    "moss",
    "mould",
    "paper",
    "peat",
    "pollen",
    "pottery",
    "quarrying",
    "rainfall",
    "recycling",
    "refraction",
    "rust",
    "salt",
    "sand",
    "silk",
    "smelting",
    "snow",
    "sonar",
    "steam",
    "steel",
    "tanning",
    "tides",
    "timber",
    "vinegar",
    "weaving",
    "welding",
    "wind",
    "wool",
    "yeast",
    "the compass rose",
    "the food chain",
    "the freezing point",
    "the horizon",
    "the nitrogen cycle",
    "the rock cycle",
    "the seasons",
    "the tide table",
)
"""Common nouns, nothing else. This list plus the template IS the corpus.

150 of them, because 150 >= 128 and 128 contrast pairs is the defensible
published floor for a difference-of-means extraction (Arditi et al. use 128 train plus 32
validation; CAA uses hundreds to thousands; Persona Vectors uses 40 per trait). Our previous
steering cells had TWO prompts, which is the single largest reason that read could not have said
anything either way.

Generated rather than authored, so nothing here is a benchmark item: there is no registered
answer, no planted flaw and no capability being scored, so a future model that memorised every
line of it would be no better at anything this repo measures. That is also why raw generations
from these prompts are safe to upload, which is the property the previous steering runs lost when
their raw material was left on an instance that self-terminated.
"""

LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    SCRIPT_LATIN: "Reply in English.",
    SCRIPT_HAN: "Reply in Chinese.",
    SCRIPT_CYRILLIC: "Reply in Russian.",
}
"""Symmetric instructions for direction fitting.

Symmetric on purpose. Fitting the contrast as "plain prompt" against "prompt asking for Chinese"
would confound the language with the mere presence of an extra instruction, and the fitted direction
would partly encode "I was given a formatting order". Both sides carry an instruction of the same
shape, so the surviving difference is which language, and the steering arms then run on the *plain*
prompt, which neither side of the fit ever saw.
"""


def build_prompts(*, n: int, seed: int, instruction_script: str | None = None) -> list[str]:
    """Deterministically draw ``n`` corpus prompts, optionally with a language instruction appended.

    Sampling without replacement from a seeded permutation rather than taking the first ``n``, so a
    smoke at n=4 and a run at n=16 do not share a prefix that could make an artefact of the first
    four subjects look like a property of the corpus.
    """
    if n > len(CORPUS_SUBJECTS):
        raise ValueError(f"corpus holds {len(CORPUS_SUBJECTS)} subjects; asked for {n}")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(CORPUS_SUBJECTS), generator=generator).tolist()
    chosen = [CORPUS_SUBJECTS[i] for i in order[:n]]
    prompts = [PROMPT_TEMPLATE.format(subject=subject) for subject in chosen]
    if instruction_script is None:
        return prompts
    instruction = LANGUAGE_INSTRUCTIONS[instruction_script]
    return [f"{prompt} {instruction}" for prompt in prompts]


# --------------------------------------------------------------------------------------
# Strength: parameterised in RMS units, because that is what the literature reports
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ResidualScale:
    """The scale a steering displacement is expressed in, at one layer, in every convention we use.

    Three different things get called "the RMS" in this area and confusing them is a factor-of-fifty
    error, so all of them are recorded:

    * ``rms_of_norms`` -- ``sqrt(mean_positions(||h||^2))``, the quadratic mean of the per-position
      residual L2 norms. **This is the denominator of the relative displacement** and the one the
      published safe envelope of 0.1-0.4 is stated against.
    * ``mean_l2`` -- the arithmetic mean of the same norms. What ``run_harness.steering_alpha``
      already uses, kept so every earlier run's alpha stays translatable into these units.
    * ``per_element_rms`` -- ``mean_l2 / sqrt(d)``, the per-*element* root mean square. A different
      quantity by a factor of ``sqrt(2560) ~ 50.6`` at the 4B, recorded only so that a figure quoted
      in those units can be converted rather than silently compared.

    ``median_l2`` comes along because the mean of norms is sensitive to the residual-sink dimensions
    this family has, and a median that disagrees with the mean is worth seeing before trusting either.
    """

    layer: int
    dim: int
    rms_of_norms: float
    mean_l2: float
    median_l2: float
    per_element_rms: float
    sqrt_dim: float


def residual_scale_stats(activations: torch.Tensor, *, layer: int) -> ResidualScale:
    """Measure every norm convention from a ``[seq, d]`` (or ``[batch, seq, d]``) residual capture."""
    flat = activations.reshape(-1, activations.shape[-1]).float()
    dim = int(flat.shape[-1])
    norms = flat.norm(dim=-1)
    mean_l2 = float(norms.mean().item())
    return ResidualScale(
        layer=layer,
        dim=dim,
        rms_of_norms=float(norms.pow(2).mean().sqrt().item()),
        mean_l2=mean_l2,
        median_l2=float(norms.median().item()),
        per_element_rms=mean_l2 / math.sqrt(dim),
        sqrt_dim=math.sqrt(dim),
    )


def alpha_from_relative_displacement(rho: float, residual: ResidualScale) -> float:
    """Absolute steering magnitude for a relative displacement ``rho`` at this layer.

    ``alpha = rho * rms_of_norms``, and because the vector added is ``alpha * unit(direction)`` the
    displacement is exactly ``alpha`` for every arm, so ``rho = ||h' - h|| / ||h||`` on average over
    the measured positions. That ratio is the steering literature's damage parameter, with a published
    safe envelope of 0.1 to 0.4 (arXiv:2608.12334), and it is computable before anything is generated,
    which is what lets it screen coefficients for free.

    A *signed* rho gives the negative-coefficient arm with no separate code path, which matters
    because a valid direction is expected to produce a monotone opposite effect and a same-sign
    response at plus and minus rho is the documented signature of a perturbation rather than a
    direction (CAA, arXiv:2312.06681).
    """
    if residual.rms_of_norms <= 0.0:
        raise ValueError(
            "the residual norm scale must be positive to scale a displacement, got "
            f"{residual.rms_of_norms}"
        )
    return rho * residual.rms_of_norms


SLOPE_GRID_RHOS: tuple[float, ...] = (-2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5)
"""The symmetric seven-point grid the per-item steerability slope is fitted on.

Seven points and symmetric because that is the shape of the statistic: Tan et al.
(arXiv:2407.12404) fit an ordinary-least-squares line per input across a signed multiplier grid and
keep the slope, one scalar per item, which is what makes anti-steerability legible instead of averaged
away. Every control arm runs at least the plus/minus anchors of this grid so a placebo slope exists
per item too -- a real slope with no placebo slope beside it licenses nothing.
"""

DEFAULT_RHOS: tuple[float, ...] = (
    -2.5,
    -1.0,
    -0.25,
    -0.1,
    0.0,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
"""Thirteen relative displacements: a geometric ladder bracketing the published safe band both ways.

Contains :data:`SLOPE_GRID_RHOS` exactly, so the per-item slope is fitted on a symmetric subset while
the positive extension rungs give the dose-response shape. Spans 0.05 to 10 because the interesting
failures sit at both ends and the middle is not where the answer is: below the 0.1-0.4 safe envelope
nothing should move, inside it a real effect should appear, and above it the published behaviour is
direction-dependent collapse rather than a monotone continuation. Dose-response in this literature is
reliably non-monotone -- one reported case went 66.7% to 100% to 16.7% as magnitude rose, another
peaked sharply and collapsed -- so a ladder that stops at the safe ceiling cannot tell "no effect"
from "we never reached the coefficient that works".

It contains the prior run's two points, which sit at rungs 0.1 and 0.5 in these units: the floor of
the safe band and just past its ceiling. That is the retro-diagnosis of the old null -- two points,
one where little should happen and one already into damage, with nothing between them and no way to
see a peak.
"""

MAX_LAYER_FRACTION = 0.8
"""Layer selection stays below this fraction of the trunk depth.

Two independent published reasons, and they agree. arXiv:2608.12334 found its target family's FINAL
layer to be a "high-gain perturbation trap" that collapses into repetitive loops while layers 17-23
of 24 worked. Arditi et al. (arXiv:2406.11717) impose ``layer < 0.8 L`` on direction selection to
avoid unembedding-adjacent directions, which steer the output distribution directly rather than
through the computation. Our own prior causal tier patched at layer 31 of 32 and got recoveries that
turned out to be arithmetic rather than evidence, for the same structural reason.

Late layers are still SWEPT -- the screen covers every layer and the readout shows them -- they are
just not eligible to be chosen as the cell the expensive behavioural units spend their generations on.
"""


# --------------------------------------------------------------------------------------
# Position arms: the comparison this module exists for
# --------------------------------------------------------------------------------------

PositionArm = Literal["prefill", "all", "decode"]
POSITION_ARMS: tuple[PositionArm, ...] = ("prefill", "all", "decode")
"""Where the perturbation is applied.

``prefill`` steers only the pass over the prompt, so the recurrent state and the KV cache are
*seeded* steered and generation then runs clean. ``all`` steers every pass, which is what the
existing ``steering_hook`` does and therefore the only regime tested so far. ``decode`` steers only
the single-token generation passes, leaving the prompt untouched.

The three are not redundant: a published result on the same 3:1 Gated-DeltaNet stack reports
decode-only as a genuine null while all-positions works strongly, on the reading that the
behavioural commitment is computed during prefill and carried in the recurrent state. If that holds
here, prefill-only should reproduce most of the all-positions effect at a fraction of the
perturbation, and the compounding that our 65,536-token traces were subject to is unnecessary.
"""


@dataclass
class HookInvocations:
    """What a steering hook actually did, recorded so "it fired" is never assumed.

    ``sequence_lengths`` holds the position count of every hidden state the hook saw, in order.
    ``selected_positions`` counts the positions the position mask picked out; ``steered_positions``
    counts only those where a NON-ZERO displacement was applied. The split is what makes the
    alpha-zero identity arm checkable: it must select positions (proving the mask and the hook ran on
    exactly the same code path as the steered arms) while steering none (proving it changed nothing).
    Collapsing them into one number makes those two states indistinguishable, and an arm that fired
    and moved nothing is precisely the silent-success shape this repo keeps paying for.
    """

    calls: int = 0
    sequence_lengths: list[int] = field(default_factory=list[int])
    selected_positions: int = 0
    steered_positions: int = 0

    def record(self, seq_len: int, selected: int, *, moved: bool) -> None:
        """Log one hook invocation: how long the pass was, how many positions it picked, did it move."""
        self.calls += 1
        self.sequence_lengths.append(seq_len)
        self.selected_positions += selected
        if moved:
            self.steered_positions += selected

    @property
    def prefill_calls(self) -> int:
        """Invocations over more than one position: prompt passes."""
        return sum(1 for length in self.sequence_lengths if length > 1)

    @property
    def decode_calls(self) -> int:
        """Single-position invocations: autoregressive decode steps."""
        return sum(1 for length in self.sequence_lengths if length == 1)


def generation_position_mask(seq_len: int, arm: PositionArm) -> torch.Tensor | None:
    """Which positions of one generation-time forward pass an ``arm`` steers, or ``None`` for none.

    Discriminated by shape, which is what distinguishes the two phases on this path: with
    ``use_cache=True`` the prompt arrives as one pass of length > 1 and every subsequent decode step
    as a pass of length 1. A one-token prompt would be ambiguous; the chat template makes that
    unreachable here, and :func:`assert_hook_pattern` fails loudly rather than guessing if the
    observed shapes ever stop looking like one prefill followed by single-token decodes.
    """
    is_prefill = seq_len > 1
    if arm == "all":
        return torch.ones(seq_len, dtype=torch.bool)
    if arm == "prefill":
        return torch.ones(seq_len, dtype=torch.bool) if is_prefill else None
    if arm == "decode":
        return None if is_prefill else torch.ones(seq_len, dtype=torch.bool)
    raise ValueError(f"unknown position arm {arm!r}; expected one of {POSITION_ARMS}")


def teacher_forced_position_mask(
    prompt_lens: torch.Tensor, seq_len: int, arm: PositionArm
) -> torch.Tensor:
    """``[batch, seq]`` mask over a single teacher-forced pass, split at each row's prompt boundary.

    A teacher-forced pass holds the prompt and the scored continuation in one tensor, so the arms
    become an index split rather than a shape test. This is a positional *attribution* within one
    forward pass and is NOT the same measurement as the generation-time arms above: nothing is
    carried in a cache across steps here, so it cannot show compounding. It is the cheap screen;
    :func:`run_behavioural_cell` is the confirmation.
    """
    positions = torch.arange(seq_len).unsqueeze(0)
    boundary = prompt_lens.unsqueeze(1)
    if arm == "all":
        return torch.ones(prompt_lens.shape[0], seq_len, dtype=torch.bool)
    if arm == "prefill":
        return positions < boundary
    if arm == "decode":
        return positions >= boundary
    raise ValueError(f"unknown position arm {arm!r}; expected one of {POSITION_ARMS}")


MaskFor = Callable[[int], torch.Tensor | None]


def masked_steering_hook(
    direction: torch.Tensor, alpha: float, mask_for: MaskFor, invocations: HookInvocations
) -> Hook:
    """Forward hook adding ``alpha * unit(direction)`` at the positions ``mask_for`` selects.

    ``mask_for`` receives the pass's sequence length and returns a bool mask over positions
    (broadcast across the batch) or ``None`` to leave the pass alone. Every invocation is recorded
    on ``invocations``, including the ones that steer nothing, so an arm that never fired is
    distinguishable from an arm that fired and had no effect.
    """
    step = unit(direction) * alpha
    moves = bool(step.any().item())

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        seq_len = int(hidden.shape[-2])
        mask = mask_for(seq_len)
        if mask is None:
            invocations.record(seq_len, 0, moved=moves)
            return hidden
        # Align the mask to hidden's axes by adding ONE trailing feature axis and then PREPENDING
        # batch axes. Unsqueezing only at the end turns a 1-D [seq] mask into [seq, 1, 1], which
        # broadcasts against [1, seq, d] to [seq, seq, d] -- the batch axis silently replaced by the
        # sequence length. That produced a shape error inside `generate` rather than a wrong number,
        # which is the only reason it was cheap to find.
        selector = mask.to(hidden.device).unsqueeze(-1)
        while selector.ndim < hidden.ndim:
            selector = selector.unsqueeze(0)
        invocations.record(seq_len, int(mask.sum().item()), moved=moves)
        steered = hidden + selector * step.to(hidden)
        if steered.shape != hidden.shape:
            raise AssertionError(
                f"steering changed the residual's shape from {tuple(hidden.shape)} to "
                f"{tuple(steered.shape)}; the position mask broadcast against the wrong axis"
            )
        return steered

    def hook(module: object, inputs: object, output: object) -> object:
        del module, inputs
        # A decoder layer's output is a Tensor at runtime; the hook signature types it as object.
        if isinstance(output, tuple):
            return (transform(output[0]), *output[1:])  # pyright: ignore[reportArgumentType]
        return transform(output)  # pyright: ignore[reportArgumentType]

    return hook


def assert_hook_pattern(
    invocations: HookInvocations,
    arm: PositionArm,
    *,
    expect_steer: bool,
    prompt_len: int | None = None,
) -> None:
    """Fail unless the hook fired in the exact shape ``arm`` implies. The check with teeth on the arms.

    With ``prompt_len`` supplied this asserts COUNTS, not just presence: the prompt arrives as one pass
    of ``prompt_len`` positions and each generated token after the first as a pass of one, so
    ``prefill`` must steer exactly ``prompt_len`` positions, ``decode`` exactly its own decode-call
    count, and ``all`` their sum. Those identities are what distinguish the three arms mechanically --
    without them "we passed --position-arm decode" is a claim about a flag rather than about what the
    model computed.

    The decode-call count is read from the observed passes rather than predicted from the response
    length, because the exact relationship (whether ``generate`` runs ``n_generated`` or
    ``n_generated - 1`` cached passes) is a property of the transformers version and belongs in an
    artifact, not in an assumption. What IS asserted is that the counts are consistent with each other,
    which is what a mislabelled arm violates.

    ``expect_steer`` is false only for the alpha-zero identity arm, which must fire and steer nothing.
    """
    if invocations.calls == 0:
        raise AssertionError(
            f"the {arm} hook never fired; the layer index is wrong or generation never ran"
        )
    if not expect_steer:
        if invocations.steered_positions != 0:
            raise AssertionError(
                f"the alpha-zero arm steered {invocations.steered_positions} positions; it must "
                "steer none"
            )
        if invocations.selected_positions == 0:
            raise AssertionError(
                "the alpha-zero arm selected no positions at all, so it did not run the same mask "
                "and hook path as the steered arms and is not a matched baseline"
            )
        return
    if invocations.steered_positions == 0:
        raise AssertionError(
            f"the {arm} arm fired {invocations.calls} times and steered ZERO positions, so it is an "
            "unsteered baseline wearing a steered label"
        )
    if arm == "prefill" and invocations.prefill_calls == 0:
        raise AssertionError(
            "the prefill arm saw no multi-position pass, so it steered nothing in prefill"
        )
    if arm == "decode" and invocations.decode_calls == 0:
        raise AssertionError(
            "the decode arm saw no single-position pass, so this path is not decoding step by step "
            "(check use_cache) and the decode arm measures nothing"
        )
    if invocations.prefill_calls > 1:
        raise AssertionError(
            f"the hook saw {invocations.prefill_calls} multi-position passes; this path is expected "
            "to prefill exactly once, so the position accounting does not describe it"
        )
    if prompt_len is not None:
        _assert_position_counts(invocations, arm, prompt_len=prompt_len)


def _assert_position_counts(
    invocations: HookInvocations, arm: PositionArm, *, prompt_len: int
) -> None:
    """Fail unless the steered-position count equals what the arm's definition implies."""
    expected = {
        "prefill": prompt_len,
        "decode": invocations.decode_calls,
        "all": prompt_len + invocations.decode_calls,
    }[arm]
    if invocations.steered_positions != expected:
        raise AssertionError(
            f"the {arm} arm steered {invocations.steered_positions} positions but its definition "
            f"implies {expected} (prompt_len {prompt_len}, decode passes "
            f"{invocations.decode_calls}); the arm is mislabelled or the mask is wrong"
        )


# --------------------------------------------------------------------------------------
# Directions and the four controls
# --------------------------------------------------------------------------------------

ARM_REAL = "real"
ARM_PLACEBO = "placebo_matched_norm"
ARM_SHUFFLED = "shuffled_label"
ARM_ORTHOGONAL = "orthogonal_component"
CONTROL_ARMS: tuple[str, ...] = (ARM_PLACEBO, ARM_SHUFFLED, ARM_ORTHOGONAL)


def shuffled_label_direction(
    positive: torch.Tensor, negative: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Difference of means after permuting the labels across the pooled rows of both classes.

    Strictly stronger than a Gaussian placebo: same data, same per-dimension scale, same covariance
    structure, contrast destroyed. A Gaussian of matched norm controls for "any perturbation this
    large"; this controls for "any difference-of-means computed on these activations", which is the
    one a fitted direction has to beat.
    """
    pooled = torch.cat([positive, negative], dim=0)
    n_positive = int(positive.shape[0])
    order = torch.randperm(int(pooled.shape[0]), generator=generator)
    permuted = pooled[order]
    return diff_of_means(permuted[:n_positive], permuted[n_positive:])


def orthogonal_component_direction(
    reference: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Draw a random direction, project ``reference`` out of it, and rescale to ``reference``'s norm.

    The non-identifiability control. Steering with a component orthogonal to an extracted vector is
    reported to produce behavioural impact indistinguishable from the vector itself, which means a
    positive result against a Gaussian placebo alone does not establish that the *direction* is what
    matters. If this arm moves the observable as much as ``real`` does, the finding is about the
    magnitude and the layer, not about the concept.
    """
    axis = unit(reference)
    raw = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    residual = raw - (raw @ axis) * axis
    if residual.norm() == 0:
        raise ValueError(
            "random draw collapsed onto the reference direction; retry with another seed"
        )
    return residual / residual.norm() * reference.norm()


def control_directions(
    direction: torch.Tensor,
    *,
    positive: torch.Tensor,
    negative: torch.Tensor,
    n_each: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Return the real direction plus ``n_each`` draws of every control, keyed by arm name.

    ``n_each`` draws rather than one, because the prior read's largest apparent effect was produced
    entirely by a single placebo arm: with one draw per control there is no way to tell an outlier
    from a band. Every control is norm-matched to ``direction``, so all arms perturb the residual by
    exactly the same amount and only the content differs.
    """
    generator = torch.Generator().manual_seed(seed)
    arms: dict[str, torch.Tensor] = {ARM_REAL: direction}
    for index in range(n_each):
        arms[f"{ARM_PLACEBO}_{index}"] = matched_norm_random_direction(direction, generator)
        shuffled = shuffled_label_direction(positive, negative, generator)
        arms[f"{ARM_SHUFFLED}_{index}"] = unit(shuffled) * direction.norm()
        arms[f"{ARM_ORTHOGONAL}_{index}"] = orthogonal_component_direction(direction, generator)
    return arms


def arm_family(arm: str) -> str:
    """Strip the draw index off an arm name: ``placebo_matched_norm_2`` -> ``placebo_matched_norm``."""
    if arm == ARM_REAL:
        return ARM_REAL
    for family in CONTROL_ARMS:
        if arm.startswith(f"{family}_"):
            return family
    raise ValueError(
        f"arm {arm!r} belongs to no known family; expected {ARM_REAL} or one of {CONTROL_ARMS}"
    )


# --------------------------------------------------------------------------------------
# The cheap screen: teacher-forced preference between two continuations
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredContinuation:
    """One prompt with the two continuations whose log-probabilities are compared."""

    prompt: str
    target_text: str
    baseline_text: str


def kl_divergence_rows(reference_logits: torch.Tensor, other_logits: torch.Tensor) -> list[float]:
    """Per-row KL(reference || other) in nats over the vocabulary axis of two ``[batch, vocab]`` tensors.

    The deterministic coherence guard the refusal-direction work uses, and the one published numeric
    threshold in this area: ``kl_score < 0.1`` on the last prompt token, intervened against not
    (arXiv:2406.11717). It needs no generation at all, so it screens a coefficient before any tokens
    are spent, and it catches the failure where an intervention has moved the whole output
    distribution rather than the one behaviour asked for.
    """
    reference = torch.log_softmax(reference_logits.float(), dim=-1)
    other = torch.log_softmax(other_logits.float(), dim=-1)
    return (reference.exp() * (reference - other)).sum(dim=-1).cpu().tolist()


@dataclass(frozen=True)
class GapReadout:
    """One screening cell's per-prompt numbers, plus what the hook did and the readout logits.

    ``last_prompt_logits`` is ``[batch, vocab]`` at each row's final PROMPT position -- the
    distribution over the first continuation token -- kept so the caller can compute the KL guard
    against an unsteered readout without paying for another forward pass.

    ``padded_width`` and ``batch_rows`` are on the record because the batch size is a measurement
    parameter here rather than an implementation detail: the same row's log-probability shifts by ~0.1
    to 0.7 nats between batch sizes purely from kernel tiling (see :func:`_encode_scored_batch` for the
    measurements, including the check that rows do not contaminate each other). An artifact that does
    not say which regime it was measured in cannot be compared to another one.
    """

    gaps: list[float]
    target_logprobs: list[float]
    baseline_logprobs: list[float]
    last_prompt_logits: torch.Tensor
    invocations: HookInvocations
    padded_width: int
    batch_rows: int


_FLOAT32_BYTES = 4

LOGIT_BLOCK_BYTES = 3 * 1024**3
"""Target size of ONE float32 block of logits inside :func:`sequence_logprobs`, in bytes.

This is what makes the scorer runnable at the thinking tier at all. The LM head is the memory wall
on this family -- 248,320 vocabulary rows, checked on the real ``Qwen/Qwen3.5-0.8B`` -- and the
scored span at the thinking tier IS the generated response, so bounding the *rows* does nothing:
:func:`self_nll` passes one row holding the whole generation. At ``THINKING_CAP`` that is
65,536 x 248,320 x 4 bytes = 60.6 GiB for a single float32 copy of the logits, which matches the
allocation named in the crash the review recorded for this path ("Tried to allocate 60.65 GiB", after
3,146 seconds of paid GPU time -- that log was not re-read here, but the arithmetic lands on it and
the failure was reproduced at a smaller size, below). Blocking the POSITION axis is the only bound
that holds, and it is applied inside this function so it holds for every caller rather than for one
of the two scoring paths.

**Peak footprint per block is twice this number**, because a block's float32 logits and their
float32 log-softmax are alive at the same time, with a bf16 copy of the same shape alive transiently
while ``.float()`` runs (so ~2.5x at the moment of the cast). At 3 GiB that is a ~6 GiB steady peak
and ~7.5 GiB at the cast, which is roughly what the existing screen configuration already pays: 8
rows by 350 positions is 695,296,000 logits, 2.59 GiB per float32 copy. That choice is deliberate --
see :func:`sequence_logprobs` on why the screen must take exactly one block.

Measured end to end on ``Qwen/Qwen3.5-0.8B`` on a 24 GB L4, 2026-08-24, at one row of 18,000
positions: block width 3,243 over 6 blocks, **11.94 GiB peak allocated** and the score returned,
against the unblocked expression on the same row dying on "Tried to allocate 16.65 GiB". The peak
runs above the 2x block arithmetic because the trunk's own long-sequence workspace and the bf16 head
output sit beside the two float32 copies, so treat 4x the budget as the number to size a card from.

**Not derived from free VRAM, on purpose**, even though the repo's default is to derive memory knobs
from the card present. A block boundary is a summation boundary: the per-position log-probabilities
are summed within a block and the block totals added, so a width that varied with whatever VRAM
happened to be free would make a row's log-probability depend on the machine it was scored on, and
two runs' artifacts would stop being comparable in their last digits. Fixed here, overridable per
call by ``block_positions`` for tests.
"""


class _HasOutputEmbeddings(Protocol):
    """The one method :func:`_output_head` needs off a loaded causal LM.

    Spelled out as a protocol because the module annotates loaded models as
    ``AutoModelForCausalLM`` -- the auto *factory*, whose stub declares none of the
    ``PreTrainedModel`` methods the runtime object actually has. Declaring the assumption here and
    casting to it puts the expected signature in one readable place, which a per-call type suppression
    would not; the runtime object having this method was checked on ``Qwen/Qwen3.5-0.8B`` bare and
    inside a live LoRA wrapper (2026-08-24).
    """

    def get_output_embeddings(self) -> torch.nn.Module | None: ...


def _output_head(model: AutoModelForCausalLM) -> tuple[torch.nn.Module, int]:
    """Resolve the LM head and the vocabulary size that sizes one block of its logits.

    ``get_output_embeddings`` is the accessor rather than ``.lm_head`` because it is what resolves
    correctly in both shapes: ``PeftModel.__getattr__`` forwards an unknown attribute to
    ``base_model``, whose own ``__getattr__`` forwards to the wrapped causal LM, so the call lands on
    the inner model's method and returns whatever module currently occupies the head slot -- the
    LoRA-wrapped head if the adapter targeted it, the bare head otherwise. Verified on
    ``Qwen/Qwen3.5-0.8B`` both bare and inside a live LoRA wrapper (2026-08-24): both resolve to the
    same ``Linear``. ``_unwrap_peft`` would be wrong here for the same reason it is right for the
    trunk -- peeling the wrapper first would hand back the untuned head.

    The vocabulary comes back with it, read off ``get_parameter("weight")`` rather than a config
    field: the head's own weight is what decides the allocation, and both a projecting ``Linear`` and
    a tied embedding store it as ``[vocab, hidden]``. A head with no such parameter raises there,
    which is the honest failure -- there is nothing to project through and no size to derive.
    """
    head = cast("_HasOutputEmbeddings", model).get_output_embeddings()
    if head is None:
        raise AttributeError(
            f"{type(model).__name__}.get_output_embeddings() returned None, so there is no LM head "
            "to project the trunk's hidden states through and no log-probability can be computed"
        )
    return head, int(head.get_parameter("weight").shape[0])


def _logit_block_positions(*, batch: int, vocab: int, block_bytes: int) -> int:
    """Positions of logits that fit in one block at this row count and vocabulary. At least one."""
    per_position = batch * vocab * _FLOAT32_BYTES
    return max(1, block_bytes // per_position)


@torch.no_grad()
def sequence_logprobs(  # noqa: PLR0913 - the batch, its two masks, the readout axis and the bound
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    score_mask: torch.Tensor,
    prompt_lens: torch.Tensor | None = None,
    *,
    block_positions: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum each row's ``score_mask`` log-probabilities, and return its last-prompt-position logits.

    Returns ``([batch], [batch, vocab])``. The second tensor is the raw logits at ``prompt_len - 1``
    per row -- the distribution over the first continuation token -- which is where the KL coherence
    guard is read; when ``prompt_lens`` is ``None`` it comes from the final real position instead. It
    is taken from whichever position block contains that index, so the guard still costs no extra
    forward pass.

    Right padding throughout, which is what makes batching safe on this architecture: the linear
    attention layers carry a recurrent state forward, so a *left*-padded row would run its pad tokens
    through the recurrence before any real token and seed the state with garbage. With right padding
    every real token is computed from real predecessors, and the trailing pads only affect positions
    the mask excludes. ``use_cache=False``: nothing here decodes, so building a cache is pure VRAM
    spent on a tensor that is immediately discarded.

    **One trunk forward, then the head in bounded position blocks.** The trunk is driven directly
    (:func:`~reward_hacking.interp.directions._transformer_trunk`, the module whose forward yields
    ``last_hidden_state``) so the full ``[batch, seq, vocab]`` logits are never materialised at once;
    the head is then applied to ``block_positions`` positions at a time with a running per-row sum.
    See :data:`LOGIT_BLOCK_BYTES` for why the bound lives here and not in one caller.

    **A screen batch is exactly one block, which is what keeps the screening artifacts unchanged.**
    Within a single block the arithmetic is the same expression it always was -- one ``log_softmax``
    over the whole shifted span and one ``.sum``, over a head applied to the whole hidden state -- so
    a stage whose rows and padded width fit in one block scores bit-identically to the unblocked
    version. Only the thinking tier, whose one row is tens of thousands of positions long, spans
    several blocks, and there the alternative was not a slightly different number but no number.
    """
    trunk = _transformer_trunk(model)
    head, vocab = _output_head(model)
    device = model.device  # pyright: ignore[reportAttributeAccessIssue]
    hidden = trunk(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        use_cache=False,
    ).last_hidden_state
    batch, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    if int(hidden.shape[1]) != seq_len:
        raise AssertionError(
            f"the trunk returned {int(hidden.shape[1])} positions for a {seq_len}-position batch, so "
            "the hidden states and the token ids do not describe the same sequence and every "
            "log-probability would be gathered against the wrong target"
        )
    width = (
        _logit_block_positions(batch=batch, vocab=vocab, block_bytes=LOGIT_BLOCK_BYTES)
        if block_positions is None
        else block_positions
    )
    if width < 1:
        raise ValueError(f"block_positions must be at least 1, got {width}")
    if prompt_lens is None:
        readout_index = attention_mask.sum(dim=-1).to(hidden.device) - 1
    else:
        readout_index = prompt_lens.to(hidden.device) - 1
    if not bool(((readout_index >= 0) & (readout_index < seq_len)).all()):
        raise ValueError(
            f"the readout index {readout_index.tolist()} falls outside the {seq_len} positions of "
            "this batch; a row with an empty prompt or an all-pad attention mask would otherwise "
            "read the padded tail and report it as the last prompt position"
        )
    # Position t predicts token t+1, so the last position predicts nothing and is scored by no block.
    predicting = seq_len - 1
    totals = torch.zeros(batch, dtype=torch.float32, device=hidden.device)
    readout = torch.zeros((batch, vocab), dtype=torch.float32, device=hidden.device)
    for start in range(0, seq_len, width):
        stop = min(start + width, seq_len)
        block_logits = head(hidden[:, start:stop, :]).float()
        rows_here = torch.nonzero((readout_index >= start) & (readout_index < stop)).squeeze(-1)
        if int(rows_here.numel()):
            readout[rows_here] = block_logits[rows_here, readout_index[rows_here] - start, :]
        scored_stop = min(stop, predicting)
        if scored_stop <= start:
            continue
        log_probs = torch.log_softmax(block_logits[:, : scored_stop - start, :], dim=-1)
        targets = input_ids[:, start + 1 : scored_stop + 1].to(log_probs.device)
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        keep = score_mask[:, start + 1 : scored_stop + 1].to(gathered.device)
        totals = totals + (gathered * keep).sum(dim=-1)
    return totals, readout


def _encode_scored_batch(
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    continuations: Sequence[str],
    *,
    width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad ``prompt + continuation`` rows to ``width``; return ids, masks and prompt lengths.

    ``width`` is pinned by the caller for a whole call rather than letting each chunk pad to its own
    maximum, so the padded length is one recorded number instead of a function of chunk composition.

    Three things measured on the 0.8B, 2026-08-24, because the first guess was wrong and the third is
    the one that matters:

    * **Batching does NOT leak between rows.** Row 0's summed continuation log-probability came back
      identical to all printed digits (-33.849205) beside an English sibling, a Chinese sibling and a
      degenerate "zzzz" sibling. Right padding keeps every real token causally downstream of real
      tokens only, and the batch does not mix. The batched screen is valid.
    * **Within a fixed batch composition the measurement is bit-exact.** The same pair scored twice
      differs by exactly zero.
    * **The BATCH SIZE is nonetheless a numerical parameter.** The same row scored alone and in a batch
      of two, at the same pinned width, differs by 0.123 nats; across 1, 2 and 4 rows the spread reached
      0.72 nats over a ~40-token continuation. That is kernel tiling and reduction order, not content:
      it is identical whatever the siblings contain. So ``score_batch_rows`` must be held constant
      within a stage, and absolute log-probabilities from different batch sizes must never be pooled --
      which is why :class:`GapRecord` carries both the width and the row count.

    An earlier version of this docstring blamed the padded width for the shift. Pinning the width did
    not remove it, which is how that attribution was caught.
    """
    # Format then encode, rather than `apply_chat_template(tokenize=True)`: that returns a
    # BatchEncoding in transformers 5.15, not a list of ids, and concatenating one to a continuation
    # raises. Going through the string also keeps this byte-identical to what `generate_response`
    # feeds the model, which is the property that makes a teacher-forced score comparable to a
    # generated one.
    prompt_ids = [
        tokenizer(  # pyright: ignore[reportCallIssue]
            tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
            add_special_tokens=False,
        )["input_ids"]
        for prompt in prompts
    ]
    continuation_ids = [
        tokenizer(text, add_special_tokens=False)["input_ids"]  # pyright: ignore[reportCallIssue]
        for text in continuations
    ]
    rows = [p + c for p, c in zip(prompt_ids, continuation_ids, strict=True)]
    longest = max(len(row) for row in rows)
    if width is None:
        width = longest
    elif width < longest:
        raise ValueError(
            f"asked to pad to width {width} but a row is {longest} tokens long; a stage's pinned width "
            "must cover its longest row or the continuation would be truncated"
        )
    pad_id = int(tokenizer.pad_token_id)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    scored = torch.zeros((len(rows), width), dtype=torch.float32)
    for index, (row, prompt) in enumerate(zip(rows, prompt_ids, strict=True)):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        attention[index, : len(row)] = 1
        scored[index, len(prompt) : len(row)] = 1.0
    prompt_lens = torch.tensor([len(p) for p in prompt_ids], dtype=torch.long)
    return ids, attention, scored, prompt_lens


@dataclass(frozen=True)
class GapRecord:
    """One cell of the cheap screen: how far steering moved the target/baseline preference.

    ``gap`` is ``logprob(target continuation) - logprob(baseline continuation)`` under the
    intervention, and ``gap_shift`` is that minus the same quantity with no intervention. Positive
    means steering made the target-script continuation relatively more likely.

    This is a **stated-preference** readout over two supplied continuations, not behaviour: it says
    which of two texts the model prefers, not what it would write. It is here to say *where and how
    hard* to look, and it is labelled on every record so no readout can quietly promote it.
    """

    layer: int
    arm: str
    position_arm: str
    rho: float
    alpha: float
    prompt_index: int
    gap: float
    gap_shift: float
    baseline_gap: float
    target_logprob: float
    baseline_logprob: float
    kl_last_token: float
    padded_width: int
    score_batch_rows: int
    hook_calls: int
    hook_selected_positions: int
    hook_steered_positions: int
    readout: str = "teacher_forced_logprob_gap"


DEFAULT_SCORE_BATCH_ROWS = 8
"""Rows per teacher-forced forward pass, held small for a NUMERICAL reason rather than a memory one.

The memory reason came first and is now handled one level down. A forward over ``[batch, seq]``
projects ``[batch, seq, vocab]`` logits, and this family's vocabulary is 248,320, so 8 rows by 350
positions is 695,296,000 logits: **2.59 GiB per float32 copy**, and 128 rows at the same width is
41.4 GiB per copy. Both figures count ONE copy, which is what an earlier version of this docstring
quoted as the peak and is where it was wrong (it also quoted them in GB while calling them the
budget). :func:`sequence_logprobs` holds the float32 logits and their float32 log-softmax
simultaneously -- 2 copies, 8 bytes per logit -- with a bf16 copy of the same shape alive while
``.float()`` runs, so the moment-of-cast peak is ~2.5x one copy: ~6.5 GiB at 8 rows, ~104 GiB at 128.
On a 48 GB L40S (44.7 GiB) holding ~9.3 GiB of bf16 4B weights, the ~35 GiB left puts the real wall
near 43 rows at that width, not 128.

Chunking over rows no longer carries that bound: :data:`LOGIT_BLOCK_BYTES` caps the transient inside
:func:`sequence_logprobs` by blocking the POSITION axis, which holds however many rows and however
long a span a caller passes -- including the thinking tier's single 65,536-position row, which no row
bound reaches at all. What is left is the reason this stays a stage-level constant: ``batch_rows`` is
a measurement parameter, because the same row's log-probability shifts by ~0.1 to 0.7 nats between
row counts from kernel tiling alone (measured, see :func:`_encode_scored_batch`). Hold it fixed across
every arm and cell of a comparison. The value 8 also keeps a screen batch inside a single position
block, so the screening artifacts are bit-identical to the unblocked scorer.
"""


@dataclass(frozen=True)
class EncodedRows:
    """Right-padded ``prompt + continuation`` rows: ids, attention mask, scored mask, prompt lengths.

    Slicing rows out of a wider encoding is exactly equivalent to encoding those rows alone at the
    same pinned width, because right padding is decided per row: ``_encode_scored_batch`` writes each
    row into a ``[n, width]`` buffer at offset zero and pads the tail, so no row's tensor depends on
    which other rows shared the call. That equivalence is what lets a whole prompt set be encoded
    once and chunked afterwards instead of re-encoded per chunk.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    score_mask: torch.Tensor
    prompt_lens: torch.Tensor

    def slice_rows(self, begin: int, stop: int) -> EncodedRows:
        """Rows ``[begin, stop)`` as their own batch, at the same padded width."""
        return EncodedRows(
            input_ids=self.input_ids[begin:stop],
            attention_mask=self.attention_mask[begin:stop],
            score_mask=self.score_mask[begin:stop],
            prompt_lens=self.prompt_lens[begin:stop],
        )


@dataclass(frozen=True)
class EncodedScoringBatch:
    """One screening prompt set, tokenised once, keyed by nothing a steering cell can change.

    The encoding depends on the prompts, the two continuation texts and the row chunking, and on
    NONE of the cell parameters -- not the layer, the arm, the strength or the position arm. It was
    nonetheless being recomputed inside every cell, twice, purely to derive the padded width: at the
    defaults that is 2 x 128 chat-template renders and tokenisations per cell, across a 32-layer by
    3-strength by multi-arm sweep. Hoisting it into an input rather than caching it inside the scorer
    makes the invariant structural: a cell cannot re-encode what it is handed.

    ``padded_width`` is the same number it always was -- the larger of the two continuation sets'
    natural widths -- so it is unchanged on every :class:`GapRecord` already on disk.
    """

    target: EncodedRows
    baseline: EncodedRows
    padded_width: int
    batch_rows: int

    @property
    def n_rows(self) -> int:
        """Prompts in the set. One row per prompt per continuation side."""
        return int(self.target.input_ids.shape[0])

    def chunks(self) -> Iterator[tuple[EncodedRows, EncodedRows]]:
        """Yield ``(target, baseline)`` row chunks of at most ``batch_rows`` rows each."""
        for begin in range(0, self.n_rows, self.batch_rows):
            stop = begin + self.batch_rows
            yield self.target.slice_rows(begin, stop), self.baseline.slice_rows(begin, stop)


def encode_scoring_batch(
    tokenizer: AutoTokenizer,
    continuations: Sequence[ScoredContinuation],
    *,
    batch_rows: int = DEFAULT_SCORE_BATCH_ROWS,
) -> EncodedScoringBatch:
    """Tokenise a whole screening set once, at ONE padded width covering both continuation sides.

    One width for every chunk and both sides, so the padding regime is a single recorded number
    rather than a function of which rows shared a chunk. It is derived by encoding each side at its
    natural width and reading the shapes, so it cannot drift from what the real passes use, and then
    both sides are re-encoded at the larger of the two.
    """
    if batch_rows < 1:
        raise ValueError(f"batch_rows must be at least 1, got {batch_rows}")
    if not continuations:
        raise ValueError("the cheap screen needs at least one continuation to encode")
    prompts = [item.prompt for item in continuations]
    targets = [item.target_text for item in continuations]
    baselines = [item.baseline_text for item in continuations]
    width = max(
        int(_encode_scored_batch(tokenizer, prompts, targets)[0].shape[1]),
        int(_encode_scored_batch(tokenizer, prompts, baselines)[0].shape[1]),
    )
    return EncodedScoringBatch(
        target=EncodedRows(*_encode_scored_batch(tokenizer, prompts, targets, width=width)),
        baseline=EncodedRows(*_encode_scored_batch(tokenizer, prompts, baselines, width=width)),
        padded_width=width,
        batch_rows=batch_rows,
    )


@dataclass(frozen=True)
class SteeringCell:
    """One cell of a steering sweep: where to intervene, along what, how hard, at which positions.

    Exactly the parameters :class:`EncodedScoringBatch` documents as unable to affect the encoding
    -- the layer, the direction, the strength and the position arm -- so the scorer's two inputs
    split cleanly into what a sweep holds fixed (the encoded batch) and what varies per cell (this).
    ``alpha`` is the absolute residual-stream coefficient, already resolved from a relative ``rho``
    via :func:`alpha_from_relative_displacement`; resolving it needs the fit's residual scales,
    which a cell deliberately does not carry.
    """

    layer: int
    direction: torch.Tensor
    alpha: float
    position_arm: PositionArm


def score_teacher_forced_gap(
    model: AutoModelForCausalLM,
    encoded: EncodedScoringBatch,
    cell: SteeringCell,
) -> GapReadout:
    """Per-prompt ``logprob(target) - logprob(baseline)`` under one steering cell.

    Two batched forward passes per chunk of rows (one per continuation set), which is what makes a
    32-layer by 13-strength by many-arm sweep affordable. The same hook and the same mask logic as the
    behavioural tier, so the two tiers cannot silently disagree about what an arm means. The target
    pass's last-prompt-position logits come back on the readout so the KL coherence guard costs nothing
    extra.

    Takes an :class:`EncodedScoringBatch` rather than a tokenizer and a prompt set: the tokenisation
    is the same for every cell of a sweep, so it belongs to the stage, and passing it in is what stops
    a cell re-deriving it.

    Chunked over rows rather than run as one wide batch: see :data:`DEFAULT_SCORE_BATCH_ROWS`. The
    position mask is rebuilt per chunk and the guard inside ``mask_for`` compares against the chunk's
    width, so a mask built for a different chunk raises rather than steering the wrong positions.

    ``batch_rows`` on the encoded batch is a measurement parameter, not just a memory knob -- the same
    row's log-probability shifts by ~0.1 to 0.7 nats between batch sizes from kernel tiling alone. Hold
    it constant across every arm and cell of a comparison, which is what encoding once per stage now
    enforces, and never pool absolute values measured at different row counts. Rows do not contaminate
    each other; that was checked rather than assumed (see :func:`_encode_scored_batch`).
    """
    invocations = HookInvocations()
    target_logprobs: list[float] = []
    baseline_logprobs: list[float] = []
    readout_chunks: list[torch.Tensor] = []
    for target_chunk, baseline_chunk in encoded.chunks():
        for side, chunk in (("target", target_chunk), ("baseline", baseline_chunk)):
            totals, last_logits = _steered_scored_forward(
                model, chunk, cell, invocations=invocations
            )
            if side == "target":
                target_logprobs.extend(totals.cpu().tolist())
                readout_chunks.append(last_logits.detach().cpu())
            else:
                baseline_logprobs.extend(totals.cpu().tolist())
    return GapReadout(
        gaps=[t - b for t, b in zip(target_logprobs, baseline_logprobs, strict=True)],
        target_logprobs=target_logprobs,
        baseline_logprobs=baseline_logprobs,
        last_prompt_logits=torch.cat(readout_chunks, dim=0),
        invocations=invocations,
        padded_width=encoded.padded_width,
        batch_rows=encoded.batch_rows,
    )


def _steered_scored_forward(
    model: AutoModelForCausalLM,
    chunk: EncodedRows,
    cell: SteeringCell,
    *,
    invocations: HookInvocations,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One chunk's steered teacher-forced pass: summed continuation log-probs and readout logits."""
    ids, attention, scored, prompt_lens = (
        chunk.input_ids,
        chunk.attention_mask,
        chunk.score_mask,
        chunk.prompt_lens,
    )
    mask = teacher_forced_position_mask(prompt_lens, int(ids.shape[1]), cell.position_arm)

    def mask_for(seq_len: int) -> torch.Tensor:
        if seq_len != mask.shape[1]:
            raise AssertionError(
                f"hook saw a {seq_len}-position pass but this chunk's mask covers {mask.shape[1]}; "
                "the teacher-forced pass is not the single forward this mask was built for"
            )
        return mask

    hook = masked_steering_hook(cell.direction, cell.alpha, mask_for, invocations)
    with residual_intervention(model, cell.layer, hook):
        return sequence_logprobs(model, ids, attention, scored, prompt_lens)


# --------------------------------------------------------------------------------------
# The behavioural tier
# --------------------------------------------------------------------------------------

NON_THINKING_CAP = 2048
"""Token cap for the non-thinking behavioural tier.

Not a bound on reasoning: the thinking block is switched OFF for this tier, so what is capped is a
direct answer to a two-or-three-sentence question. The cap is set at roughly an order of magnitude
above the unsteered answer length so that hitting it is a fact about the arm rather than about the
budget, and ``hit_token_cap`` is reported per record so a truncated arm is visible instead of
reading as a short answer. The thinking tier keeps the full 65,536 and is bounded by a deadline and
by arm count, never by a cap.
"""

THINKING_CAP = 65536
"""The thinking tier's cap: the model's own budget, unchanged. Runtime is bounded by the deadline."""


def non_thinking_sampling(
    *, greedy: bool, max_new_tokens: int = NON_THINKING_CAP
) -> SamplingConfig:
    """Build the non-thinking sampling preset, optionally greedy.

    Greedy is the primary behavioural read here: it removes sampling variance from the position and
    strength comparison entirely, so a difference between arms is the intervention. That is safe in
    non-thinking mode; greedy decoding *inside* a thinking block is Qwen3.5's documented repetition
    failure, which is why the thinking tier samples instead.
    """
    base = SamplingConfig.for_thinking(thinking=False)
    return replace(
        base,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        min_p=0.0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
    )


def thinking_sampling(*, max_new_tokens: int = THINKING_CAP) -> SamplingConfig:
    """Build the penalty-free thinking preset at the full cap, matching the rest of the interp path."""
    return replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=max_new_tokens)


@dataclass(frozen=True)
class BehaviouralRecord:
    """One generation under one steering cell, with every deterministic score and the raw text.

    ``response_text`` is retained on the record and written to the run's JSONL: the corpus is
    template-generated, so no response can carry benchmark material, and the previous steering runs
    lost exactly this field to an instance-local directory that was never uploaded.
    """

    layer: int
    arm: str
    position_arm: str
    rho: float
    alpha: float
    prompt_index: int
    sample_index: int
    thinking: bool
    greedy: bool
    prompt_text: str
    response_text: str
    response_tokens: int
    hit_token_cap: bool
    elapsed_seconds: float
    target_script: str
    target_ratio: float | None
    target_chars: int
    latin_chars: int
    other_letters: int
    ratio_denominator: int
    specificity: dict[str, float | None]
    distinct_trigram_ratio: float | None
    repetition_rate: float | None
    token_entropy: float | None
    longest_token_run: int
    self_nll: float | None
    hook_calls: int
    hook_selected_positions: int
    hook_steered_positions: int
    hook_prefill_calls: int
    hook_decode_calls: int
    readout: str = "generated_script_ratio"


@torch.no_grad()
def self_nll(model: AutoModelForCausalLM, record: GenerationRecord) -> float | None:
    """Mean negative log-likelihood of the generated tokens under the UNSTEERED model.

    The deterministic coherence measure. Text the model itself finds unlikely is the signature of a
    damaged generation, and unlike a judge score this needs no second model and no threshold chosen
    by taste: it is comparable across arms because the scoring model is the same clean model every
    time. Run outside any hook, which is what "unsteered" means here -- scoring a steered generation
    under the steering that produced it would measure nothing.
    """
    if record.n_generated == 0:
        return None
    ids = record.full_ids.unsqueeze(0)
    attention = torch.ones_like(ids)
    scored = record.is_response.to(torch.float32).unsqueeze(0)
    total, _ = sequence_logprobs(model, ids, attention, scored)
    return float(-total.item() / record.n_generated)


def _tail_token_ids(record: GenerationRecord) -> list[int]:
    return record.full_ids[record.prompt_len :].tolist()


def score_generation(  # noqa: PLR0913 - a record is its cell, its text and its scores
    record: GenerationRecord,
    *,
    layer: int,
    arm: str,
    position_arm: str,
    rho: float,
    alpha: float,
    prompt_index: int,
    sample_index: int,
    thinking: bool,
    greedy: bool,
    target_script: str,
    elapsed_seconds: float,
    invocations: HookInvocations,
    nll: float | None,
) -> BehaviouralRecord:
    """Turn one generation into a fully scored record. Pure given the record: no model access."""
    primary = script_score(record.response_text, target_script)
    others = {
        script: script_score(record.response_text, script).ratio
        for script in TARGET_SCRIPTS
        if script != target_script
    }
    token_ids = _tail_token_ids(record)
    return BehaviouralRecord(
        layer=layer,
        arm=arm,
        position_arm=position_arm,
        rho=rho,
        alpha=alpha,
        prompt_index=prompt_index,
        sample_index=sample_index,
        thinking=thinking,
        greedy=greedy,
        prompt_text=record.prompt_text,
        response_text=record.response_text,
        response_tokens=record.n_generated,
        hit_token_cap=record.hit_token_cap,
        elapsed_seconds=elapsed_seconds,
        target_script=target_script,
        target_ratio=primary.ratio,
        target_chars=primary.target_chars,
        latin_chars=primary.latin_chars,
        other_letters=primary.other_letters,
        ratio_denominator=primary.denominator,
        specificity=others,
        distinct_trigram_ratio=distinct_ngram_ratio(token_ids),
        repetition_rate=repetition_rate(token_ids),
        token_entropy=token_entropy(token_ids),
        longest_token_run=longest_token_run(token_ids),
        self_nll=nll,
        hook_calls=invocations.calls,
        hook_selected_positions=invocations.selected_positions,
        hook_steered_positions=invocations.steered_positions,
        hook_prefill_calls=invocations.prefill_calls,
        hook_decode_calls=invocations.decode_calls,
    )


def _shuffled_items(prompts: Sequence[str], *, seed: int) -> list[tuple[int, str]]:
    """``(original index, prompt)`` pairs in a seeded random order, so a deadline censors at random.

    Without this a cell that runs out of time always drops the same tail of the item set, and every
    truncated cell in the run is missing the SAME items -- a systematic bias that looks like a
    property of the arms. The original index travels with the prompt so per-item comparisons across
    cells still line up.
    """
    order = torch.randperm(len(prompts), generator=torch.Generator().manual_seed(seed)).tolist()
    return [(int(index), prompts[int(index)]) for index in order]


GENERATION_SEED_STRIDE = 7_919
"""Prime stride between consecutive prompts' generation seeds, so no (prompt, sample) pair collides.

Any value above the largest ``n_samples`` would separate the pairs; a prime keeps neighbouring
prompts' seed blocks from lining up under the base seed as well.
"""


def generation_seed(base: int, *, prompt_index: int, sample_index: int) -> int:
    """Derive the process-global torch seed one generation runs under, per ITEM rather than per cell.

    ``transformers.generate`` has no per-request seed, so it draws from the process-global stream. Per
    cell is not enough: each generation consumes a number of draws that depends on how long it turned
    out to be, so the real arm's third item would start from a different RNG state than the placebo
    arm's third item however carefully the cell was seeded up front. Deriving the seed from
    ``(prompt_index, sample_index)`` alone -- not from the layer, the arm, the position arm or the
    strength -- pairs every cell in the grid item for item, which is the whole premise of the
    matched-norm control band: that a difference between the real direction and a placebo of the same
    magnitude is the intervention rather than two independent samples.
    """
    return base + GENERATION_SEED_STRIDE * prompt_index + sample_index


def run_behavioural_cell(  # noqa: PLR0913 - one cell is the model, the prompts, the axis and the knobs
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    *,
    layer: int,
    arm: str,
    direction: torch.Tensor,
    rho: float,
    alpha: float,
    position_arm: PositionArm,
    thinking: bool,
    greedy: bool,
    target_script: str,
    n_samples: int,
    sampling: SamplingConfig,
    generation_seed_base: int,
    deadline: float | None = None,
    item_order_seed: int = 0,
) -> list[BehaviouralRecord]:
    """Generate and score one (layer, arm, strength, position) cell over every prompt.

    Batch of one per generation, deliberately: left-padded batch generation would run pad tokens
    through the Gated DeltaNet recurrence ahead of real ones, and right padding is not available
    during decoding. Throughput is bought in the cheap screen instead, which batches safely.

    ``generation_seed_base`` is required rather than defaulted, because an unseeded decode is the
    defect: every cell in the grid must draw the same sample per item or the matched-norm control band
    compares two independent samples. The process-global stream is reseeded immediately before each
    generation from :func:`generation_seed`; pass the run's ``--seed`` and nothing else, so real and
    placebo cells at the same layer are paired item for item.

    ``deadline`` is a monotonic timestamp checked BETWEEN generations, never inside one: bounding a
    run by truncating the model's reasoning would make truncation read as an effect of the
    intervention. A cell that runs out of time returns what it finished, and the caller records how
    much of the cell that was.
    """
    records: list[BehaviouralRecord] = []
    for prompt_index, prompt in _shuffled_items(prompts, seed=item_order_seed):
        for sample_index in range(n_samples):
            if deadline is not None and time.monotonic() > deadline:
                logger.warning(
                    f"deadline reached mid-cell: {layer=} {arm=} {position_arm=} {rho=} "
                    f"stopped after {len(records)} of {len(prompts) * n_samples} generations"
                )
                return records
            invocations = HookInvocations()
            hook = masked_steering_hook(
                direction,
                alpha,
                lambda seq_len: generation_position_mask(seq_len, position_arm),
                invocations,
            )
            started = time.monotonic()
            with residual_intervention(model, layer, hook):
                torch.manual_seed(
                    generation_seed(
                        generation_seed_base,
                        prompt_index=prompt_index,
                        sample_index=sample_index,
                    )
                )
                record = generate_response(
                    model, tokenizer, prompt, thinking=thinking, sampling=sampling
                )
            elapsed = time.monotonic() - started
            assert_hook_pattern(
                invocations,
                position_arm,
                expect_steer=alpha != 0.0,
                prompt_len=record.prompt_len,
            )
            records.append(
                score_generation(
                    record,
                    layer=layer,
                    arm=arm,
                    position_arm=position_arm,
                    rho=rho,
                    alpha=alpha,
                    prompt_index=prompt_index,
                    sample_index=sample_index,
                    thinking=thinking,
                    greedy=greedy,
                    target_script=target_script,
                    elapsed_seconds=elapsed,
                    invocations=invocations,
                    nll=self_nll(model, record),
                )
            )
    return records


# --------------------------------------------------------------------------------------
# Rig integrity: the recurrent-state isolation question, with its own sabotage
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class IsolationCheck:
    """One rig-integrity check: what it asserts, what it saw, and whether it held."""

    name: str
    passed: bool
    detail: str


def _greedy_ids(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, *, max_new_tokens: int
) -> list[int]:
    """Greedy-decode ``prompt``, returning the generated ids. One fresh cache, built by ``generate``."""
    chat = tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(chat, return_tensors="pt")  # pyright: ignore[reportCallIssue]
    ids = encoded["input_ids"].to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
    mask = encoded["attention_mask"].to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
    out = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
        input_ids=ids,
        attention_mask=mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,  # pyright: ignore[reportAttributeAccessIssue]
    )
    return out[0].detach().cpu().tolist()[int(ids.shape[1]) :]


CHECK_DETERMINISM = "repeat_determinism"
"""The floor: two identical calls must agree, or a later difference proves nothing."""

CHECK_CONTAMINATION = "steered_run_between"
"""Our real path: a fresh generate per arm must leave the probe's read untouched."""

CHECK_IN_PLACE = "in_place_state_write"
"""The mechanism: advancing one cache mutates its recurrent-state tensor under any alias of it."""

CHECK_BRANCHING = "shared_cache_branching"
"""The faithful hazard: two continuations off one cache, and the first reaches the second."""


def cache_isolation_report(  # noqa: PLR0913 - the model, two prompts, the axis and the sabotage switch
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    probe_prompt: str,
    interference_prompt: str,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
    max_new_tokens: int = 24,
    inject_residue: bool = False,
) -> list[IsolationCheck]:
    """Measure whether one condition leaks recurrent state into the next, and prove the check can fail.

    The hazard is real at source level in the exact versions installed here. transformers 5.15.0's
    ``LinearAttentionLayer.update_recurrent_state`` does
    ``self.recurrent_states[state_idx].copy_(recurrent_states)`` into a tensor pre-allocated by
    ``lazy_initialization`` and pinned with ``mark_static_address``, and the conv-state decode path
    does the same from inside the layer forward. So a cache object -- or a tensor taken off one
    without ``.clone()`` -- reused across two conditions corrupts the second. Reading that is not the
    same as measuring it, which is what these four checks do:

    1. ``repeat_determinism`` -- the same greedy call twice must give the same ids. Without this floor
       a later difference could be nondeterminism rather than contamination, and the whole report
       would be unfalsifiable.
    2. ``steered_run_between`` -- read the probe, run a *strongly steered* generation on a different
       prompt, read the probe again. On our path (a fresh ``generate`` per arm, no ``past_key_values``
       passed) the two reads must be identical.
    3. ``in_place_state_write`` -- prefill a cache, keep BOTH a clone and a live alias of its recurrent
       state, advance the same cache one token, and show the alias moved while the clone did not.
       Expected to observe mutation: it establishes that the hazard exists here, and it is the reason
       any future code that snapshots a state must clone it.
    4. ``shared_cache_branching`` -- prefill once, then branch two continuations off the SAME cache
       object, with the first branch steered in one arm and unsteered in the other, and read the
       second branch both times. Expected to observe corruption. This is checks 2's question asked of
       the other cache discipline: same comparison, one cache instead of two, and the difference
       between the answers is what makes check 2 informative rather than tautological.

    ``inject_residue=True`` is the sabotage for check 2. It applies a residual perturbation on the
    probe's SECOND read, standing in for state left behind by the previous condition, and check 2 must
    then fail. It is a SENSITIVITY sabotage, not a mechanism-faithful one -- it establishes that check
    2 can see a cross-condition influence of that magnitude at all -- and the mechanism-faithful
    demonstration is check 4, which needs no flag because it runs every time. Threading one cache
    through two ``generate`` calls, which would have been the faithful sabotage, is not available:
    it raises ``RuntimeError: Sizes of tensors must match except in dimension 1`` because ``generate``
    derives its position bookkeeping from the cache length (observed on the 0.8B, 2026-08-24).
    """
    checks: list[IsolationCheck] = []
    first = _greedy_ids(model, tokenizer, probe_prompt, max_new_tokens=max_new_tokens)
    second = _greedy_ids(model, tokenizer, probe_prompt, max_new_tokens=max_new_tokens)
    checks.append(
        IsolationCheck(
            name=CHECK_DETERMINISM,
            passed=first == second,
            detail=f"two identical greedy calls agreed on "
            f"{sum(a == b for a, b in zip(first, second, strict=False))} of {len(first)} tokens",
        )
    )

    baseline = _greedy_ids(model, tokenizer, probe_prompt, max_new_tokens=max_new_tokens)
    invocations = HookInvocations()
    interference_hook = masked_steering_hook(
        direction, alpha, lambda seq_len: generation_position_mask(seq_len, "all"), invocations
    )
    with residual_intervention(model, layer, interference_hook):
        _greedy_ids(model, tokenizer, interference_prompt, max_new_tokens=max_new_tokens)
    if inject_residue:
        residue = HookInvocations()
        residue_hook = masked_steering_hook(
            direction,
            alpha,
            lambda seq_len: generation_position_mask(seq_len, "all"),
            residue,
        )
        with residual_intervention(model, layer, residue_hook):
            after = _greedy_ids(model, tokenizer, probe_prompt, max_new_tokens=max_new_tokens)
    else:
        after = _greedy_ids(model, tokenizer, probe_prompt, max_new_tokens=max_new_tokens)
    checks.append(
        IsolationCheck(
            name=CHECK_CONTAMINATION,
            passed=baseline == after,
            detail=(
                f"probe re-read after a steered run on another prompt "
                f"({'RESIDUE INJECTED -- sabotage arm' if inject_residue else 'fresh cache per call'}): "
                f"{sum(a == b for a, b in zip(baseline, after, strict=False))} of {len(baseline)} "
                f"tokens agreed; the interference run's hook fired {invocations.calls} times over "
                f"{invocations.steered_positions} positions at alpha {alpha:.4g}"
            ),
        )
    )
    checks.append(_in_place_write_check(model, tokenizer, probe_prompt))
    checks.append(
        _shared_cache_branching_check(
            model, tokenizer, probe_prompt, layer=layer, direction=direction, alpha=alpha
        )
    )
    return checks


@torch.no_grad()
def _prefill_into(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str, cache: object
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill ``prompt`` into ``cache`` and return (next-token id, last-position logits)."""
    chat = tokenizer.apply_chat_template(  # pyright: ignore[reportAttributeAccessIssue]
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(chat, return_tensors="pt")  # pyright: ignore[reportCallIssue]
    ids = encoded["input_ids"].to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
    mask = encoded["attention_mask"].to(model.device)  # pyright: ignore[reportAttributeAccessIssue]
    out = model(  # pyright: ignore[reportCallIssue]
        input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True
    )
    logits = out.logits[:, -1, :]
    return logits.argmax(dim=-1, keepdim=True), logits


@torch.no_grad()
def _step_on(model: AutoModelForCausalLM, token: torch.Tensor, cache: object) -> torch.Tensor:
    """Advance ``cache`` by one token and return that step's ``[1, vocab]`` logits."""
    out = model(input_ids=token, past_key_values=cache, use_cache=True)  # pyright: ignore[reportCallIssue]
    return out.logits[:, -1, :]


def _shared_cache_branching_check(  # noqa: PLR0913 - the model, prompt, axis and cell knobs
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    layer: int,
    direction: torch.Tensor,
    alpha: float,
) -> IsolationCheck:
    """Branch two continuations off ONE cache and show the first branch reaches the second read.

    Both arms take exactly two steps off the same prefill, so the step count, the position offsets and
    the token fed are identical and the ONLY difference is whether the first branch was steered. If
    the second read differs between arms, the first branch's state was still in the cache -- the
    documented hazard, isolated. ``passed`` is true when that corruption IS observed, because the
    check exists to establish that the hazard is real and detectable on this model.
    """
    invocations = HookInvocations()
    hook = masked_steering_hook(
        direction, alpha, lambda seq_len: generation_position_mask(seq_len, "all"), invocations
    )
    steered_cache = _fresh_cache(model)
    token, _ = _prefill_into(model, tokenizer, prompt, steered_cache)
    with residual_intervention(model, layer, hook):
        _step_on(model, token, steered_cache)
    after_steered = _step_on(model, token, steered_cache)

    clean_cache = _fresh_cache(model)
    token_clean, _ = _prefill_into(model, tokenizer, prompt, clean_cache)
    _step_on(model, token_clean, clean_cache)
    after_clean = _step_on(model, token_clean, clean_cache)

    same_token = bool(torch.equal(token, token_clean))
    max_delta = float((after_steered - after_clean).abs().max().item())
    corrupted = not torch.equal(after_steered, after_clean)
    return IsolationCheck(
        name=CHECK_BRANCHING,
        passed=corrupted and same_token,
        detail=(
            f"two continuations branched off one cache, the first steered at alpha {alpha:.4g}: the "
            f"second read {'DIFFERED' if corrupted else 'was identical'} between arms "
            f"(max logit delta {max_delta:.6g}); the two prefills agreed on their first token: "
            f"{same_token}. Corruption is the expected result and is what makes a fresh cache per "
            f"condition load-bearing; the steered branch's hook fired {invocations.calls} times"
        ),
    )


def _fresh_cache(model: AutoModelForCausalLM) -> object:
    """Build the cache class this model's own forward would build, for the sabotage arm."""
    from transformers import DynamicCache  # noqa: PLC0415 - only the sabotage path needs the class

    return DynamicCache(config=model.config)  # pyright: ignore[reportAttributeAccessIssue]


def _state_slots(initialised: object) -> list[object]:
    """Return the state indices ``initialised`` marks as live, for the dict or the list layout.

    transformers 5.15.0 keys ``recurrent_states`` and ``is_recurrent_states_initialized`` by INT in a
    dict (verified on the real 0.8B: ``{0: True}``), and iterating that dict yields keys, not flags.
    An earlier version of this function did ``enumerate(initialised)``, so ``flag`` was the integer
    key ``0`` -- falsy -- and it returned ``None`` on every cache while looking entirely reasonable.
    Handling both layouts explicitly means a future switch back to lists cannot silently reintroduce
    that.
    """
    if isinstance(initialised, dict):
        return [key for key, flag in initialised.items() if flag]
    if isinstance(initialised, (list, tuple)):
        return [index for index, flag in enumerate(initialised) if flag]
    return []


def _first_recurrent_state(cache: object) -> torch.Tensor | None:
    """First initialised Gated DeltaNet recurrent-state tensor on ``cache``, if any."""
    for entry in getattr(cache, "layers", []):
        states = getattr(entry, "recurrent_states", None)
        initialised = getattr(entry, "is_recurrent_states_initialized", None)
        if states is None or initialised is None:
            continue
        for slot in _state_slots(initialised):
            candidate = states[slot]
            if isinstance(candidate, torch.Tensor):
                return candidate
    return None


def isolation_verdict(checks: Sequence[IsolationCheck], *, sabotage: bool) -> tuple[str, bool]:
    """Turn the isolation checks into a ``(verdict, ok)`` pair, inverting under sabotage.

    Separate from the stage that runs it so the inversion is testable without a model. On the real path
    ``ok`` needs three things at once, and each covers a different way the report could be hollow:

    * ``repeat_determinism`` passed -- the floor. Without it a stable probe read says nothing, because
      an unstable one would have been indistinguishable from contamination.
    * ``steered_run_between`` passed -- our path, a fresh ``generate`` per arm, does not leak.
    * ``shared_cache_branching`` passed, meaning corruption WAS observed when a cache was deliberately
      shared. This is the sabotage built into the ordinary run: it proves the comparison can come out
      the other way, so the second bullet is a measurement rather than a tautology. A run where sharing
      a cache changed nothing has a blind instrument and its clean verdict is worthless.

    Under ``sabotage=True`` (residue injected on the probe's second read) the expectation on
    ``steered_run_between`` INVERTS: it must FAIL. A sabotage arm that comes back green is not a pass,
    it is a check that cannot see what it exists to catch, so ``ok`` is false there. Determinism is not
    required on that path, because the sabotage arm is being asked one question only.
    """
    by_name = {check.name: check for check in checks}
    missing = {CHECK_CONTAMINATION, CHECK_DETERMINISM} - set(by_name)
    if missing:
        raise ValueError(f"isolation report is missing required checks: {sorted(missing)}")
    contaminated = not by_name[CHECK_CONTAMINATION].passed
    if sabotage:
        return ("sabotage-detected" if contaminated else "SABOTAGE-INVISIBLE"), contaminated
    if contaminated:
        return "CONTAMINATED", False
    if not by_name[CHECK_DETERMINISM].passed:
        return "NONDETERMINISTIC", False
    branching = by_name.get(CHECK_BRANCHING)
    if branching is not None and not branching.passed:
        return "BLIND-NO-CORRUPTION-DETECTABLE", False
    return "isolated", True


def tensor_fingerprint(tensor: torch.Tensor) -> tuple[float, float]:
    """Reduce ``tensor`` to ``(sum, sum of squares)`` in double precision, as plain Python floats.

    The point is what the return type is NOT: two floats read out with ``.item()`` are values, so
    nothing about how the tensor was obtained can make them track a later in-place write to it. That
    is what :func:`_in_place_write_check` needs and what comparing a tensor to itself cannot give --
    ``torch.equal(snapshot, snapshot)`` is true for every non-NaN tensor whether or not the snapshot
    was a copy. Two reductions rather than one so a mutation that happened to preserve the sum still
    moves the fingerprint.
    """
    doubled = tensor.double()
    return float(doubled.sum().item()), float(doubled.pow(2).sum().item())


def _in_place_write_check(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str
) -> IsolationCheck:
    """Show that advancing one cache mutates its recurrent-state tensor through any alias of it.

    Plain forwards rather than ``generate``: threading a cache through two ``generate`` calls raises on
    position bookkeeping, while a prefill followed by a single-token step is exactly what decoding
    does and is the legitimate way to advance a cache.

    Passes when mutation IS observed, because the point is to establish that the hazard is real on this
    model at these versions. Two INDEPENDENT comparisons, both against a :func:`tensor_fingerprint`
    taken before the step, and both have to hold:

    * *the write is in place* -- the live alias's own fingerprint moved, so the step wrote into the
      tensor the cache already held rather than into a fresh allocation;
    * *a clone escapes the write* -- the snapshot's own fingerprint did NOT move.

    Each fails on its own defect and neither implies the other, which is what makes the pair a check
    rather than one claim stated twice: a model that reallocated instead of writing in place fails the
    first and passes the second, and a snapshot that is secretly an alias of the state fails the second
    while the first still holds. Both are read through values rather than through the tensors, because
    an aliased snapshot follows the write and every tensor-to-itself comparison stays true regardless
    (``torch.equal(snapshot, snapshot)``, the term this replaced, was true for any non-NaN tensor).

    Together they are the demonstration that future code snapshotting a recurrent state has to
    ``.clone()`` it. If this ever comes back "did NOT mutate", the source comment about preserving a
    static address for cudagraphs has stopped applying and the isolation argument in
    :func:`cache_isolation_report` needs rederiving rather than trusting.
    """
    cache = _fresh_cache(model)
    token, _ = _prefill_into(model, tokenizer, prompt, cache)
    state = _first_recurrent_state(cache)
    if state is None:
        return IsolationCheck(
            name=CHECK_IN_PLACE,
            passed=False,
            detail="no initialised recurrent state found on the cache after a prefill; the cache "
            "layout changed and the isolation argument must be rederived rather than assumed",
        )
    snapshot = state.clone()
    alias = state
    alias_before = tensor_fingerprint(alias)
    snapshot_before = tensor_fingerprint(snapshot)
    _step_on(model, token, cache)
    alias_after = tensor_fingerprint(alias)
    snapshot_after = tensor_fingerprint(snapshot)
    alias_moved = alias_after != alias_before
    clone_independent = snapshot_after == snapshot_before
    return IsolationCheck(
        name=CHECK_IN_PLACE,
        passed=alias_moved and clone_independent,
        detail=(
            f"advancing one cache by a token {'MUTATED' if alias_moved else 'did NOT mutate'} the "
            f"recurrent-state tensor through a live alias (shape {tuple(alias.shape)}, "
            f"dtype {alias.dtype}): its (sum, sum of squares) fingerprint went {alias_before} -> "
            f"{alias_after}. The clone's own fingerprint "
            f"{'held' if clone_independent else f'MOVED from {snapshot_before} to {snapshot_after}'}, "
            f"so it {'did not follow' if clone_independent else 'FOLLOWED'} the write. A mutated alias "
            "beside an unmoved clone is the expected result and is why any snapshot of a recurrent "
            "state must be cloned and why a cache must be fresh per condition"
        ),
    )
