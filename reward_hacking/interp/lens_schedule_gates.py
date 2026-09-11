"""The three schedule gates on the Jacobian-lens fit: memory, tokenizer identity, and restartability.

Beside the architecture gates in ``reward_hacking.interp.lens_deltanet_gates``, three things about
HOW a fit runs decide whether a rented 9B box produces a lens or an OOM, a lens on the wrong ids, or
a lens that cannot survive a restart:

(c) **``dim_batch``.** :func:`sweep_dim_batch` runs the fit's own ``jacobian_for_prompt`` at the
    real window length for each candidate, records peak memory and seconds per prompt, and
    :func:`choose_dim_batch` takes the largest whose RESERVED peak leaves
    :data:`HEADROOM_FRACTION` of the card free. Reserved rather than allocated, because the
    allocator needs the reserved footprint and a fragmented card OOMs on reservation.
(d) **Token identity.** :func:`token_identity` requires the same rendered text to tokenize
    identically under every tokenizer the wave feeds ids to, and requires the check to SEE a
    difference on a tokenizer from another family, or the comparison has never been watched to
    fail.
(e) **Resume equality.** :func:`resume_equality` fits N prompts straight, fits N/2 with a checkpoint,
    resumes the remainder from it, and requires every fp32 accumulator to agree within
    :data:`RESUME_TOLERANCE`; the half fit alone must NOT agree, which is what shows the comparison
    can tell two lenses apart. All three fits go through
    :func:`reward_hacking.interp.jacobian.fit_lens`, so the ``checkpoint_every`` and ``resume`` knobs
    the config threads are the ones under test.

The choice, the comparison and the resume orchestration are plain functions over trials, encoders
and a fitter, so the offline tests drive them on stubs; the GPU parts are the sweep's estimator
calls and the tokenizer loads.
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import torch

from reward_hacking.interp.jacobian import JacobianConfig, fit_lens, resolve_weights_identity
from reward_hacking.interp.lens_deltanet_gates import GateFailureError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

logger = logging.getLogger(__name__)

DIM_BATCH_CANDIDATES: tuple[int, ...] = (16, 8, 4)
HEADROOM_FRACTION = 0.10
RESUME_TOLERANCE = 1e-4
"""Relative Frobenius agreement between a straight and a resumed fit, per layer.

The checkpoint is the exact fp32 running sum, so the only difference two such fits can carry is the
card's own run-to-run reduction noise on the per-prompt Jacobians; the kernel audit's fused-versus-
fused floor says how large that is on the same box.
"""

EXPECT_IDENTICAL = "identical"
EXPECT_DIFFERENT = "different"


# --------------------------------------------------------------------------------------
# (c) dim_batch sweep
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DimBatchTrial:
    """One candidate ``dim_batch`` run through the fit's own per-prompt estimator."""

    dim_batch: int
    completed: bool
    seconds_per_prompt: float | None
    seq_len: int | None
    n_valid_positions: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    error: str | None


@dataclass(frozen=True)
class DimBatchChoice:
    """Which candidate the sweep settled on, and why the others were passed over."""

    chosen: int | None
    budget_bytes: int
    total_bytes: int
    headroom_fraction: float
    rejected: dict[int, str]


def choose_dim_batch(
    trials: Sequence[DimBatchTrial], *, total_bytes: int, headroom_fraction: float
) -> DimBatchChoice:
    """Take the largest completed candidate whose reserved peak leaves the headroom free."""
    budget = int(total_bytes * (1.0 - headroom_fraction))
    rejected: dict[int, str] = {}
    chosen: int | None = None
    for trial in sorted(trials, key=lambda trial: trial.dim_batch, reverse=True):
        if not trial.completed:
            rejected[trial.dim_batch] = trial.error or "did not complete"
            continue
        if trial.peak_reserved_bytes is None or trial.peak_reserved_bytes > budget:
            rejected[trial.dim_batch] = (
                f"reserved peak {trial.peak_reserved_bytes} exceeds the budget {budget} "
                f"({headroom_fraction:.0%} headroom on {total_bytes})"
            )
            continue
        if chosen is None:
            chosen = trial.dim_batch
    return DimBatchChoice(
        chosen=chosen,
        budget_bytes=budget,
        total_bytes=total_bytes,
        headroom_fraction=headroom_fraction,
        rejected=rejected,
    )


@dataclass(frozen=True)
class DimBatchSweepReport:
    """Gate (c): every trial and the choice."""

    max_seq_len: int
    candidates: tuple[int, ...]
    trials: tuple[DimBatchTrial, ...]
    choice: DimBatchChoice

    def failures(self) -> list[str]:
        """Name the refusals: no candidate fits, or a trial ran at a shorter window than asked."""
        failed: list[str] = []
        if self.choice.chosen is None:
            failed.append(
                f"no dim_batch candidate in {self.candidates} fits with headroom: "
                f"{self.choice.rejected}"
            )
        short = [t.dim_batch for t in self.trials if t.completed and t.seq_len != self.max_seq_len]
        if short:
            failed.append(
                f"trials {short} ran at a window shorter than max_seq_len={self.max_seq_len}; the "
                f"sweep prompt is too short for the real window"
            )
        return failed

    @property
    def passed(self) -> bool:
        """A candidate was chosen and every trial ran at the real window length."""
        return not self.failures()

    def as_payload(self) -> dict[str, object]:
        """Return the report block."""
        return {**asdict(self), "failures": self.failures(), "passed": self.passed}


def sweep_dim_batch(  # noqa: PLR0913 - the fit's estimator plus the card's budget
    jl: ModuleType,
    model: object,
    prompt: str,
    *,
    candidates: Sequence[int],
    target_layer: int,
    max_seq_len: int,
    total_bytes: int,
    headroom_fraction: float,
) -> DimBatchSweepReport:
    """Gate (c): ``jacobian_for_prompt`` per candidate at the real window, memory and time recorded.

    An out-of-memory is the one expected failure and is recorded as such (the cache is emptied and
    the sweep moves on); anything else propagates, since a sweep that swallowed a shape error would
    choose a ``dim_batch`` on a measurement of nothing.
    """
    trials: list[DimBatchTrial] = []
    for dim_batch in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            _jacobians, seq_len, n_valid = jl.jacobian_for_prompt(
                model,
                prompt,
                list(range(target_layer)),
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
            )
        except torch.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            trials.append(
                DimBatchTrial(
                    dim_batch=dim_batch,
                    completed=False,
                    seconds_per_prompt=None,
                    seq_len=None,
                    n_valid_positions=None,
                    peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                    peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
                    error=f"OutOfMemoryError: {str(error)[:300]}",
                )
            )
            logger.warning("dim_batch=%d OOM at max_seq_len=%d", dim_batch, max_seq_len)
            continue
        trial = DimBatchTrial(
            dim_batch=dim_batch,
            completed=True,
            seconds_per_prompt=round(time.perf_counter() - started, 2),
            seq_len=int(seq_len),
            n_valid_positions=int(n_valid),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            error=None,
        )
        trials.append(trial)
        logger.info(
            "dim_batch sweep, %s",
            f"{dim_batch=} s/prompt={trial.seconds_per_prompt} "
            f"peak_reserved_GiB={(trial.peak_reserved_bytes or 0) / 2**30:.2f} seq_len={seq_len}",
        )
    choice = choose_dim_batch(trials, total_bytes=total_bytes, headroom_fraction=headroom_fraction)
    logger.info("dim_batch choice, %s", f"chosen={choice.chosen} rejected={choice.rejected}")
    return DimBatchSweepReport(
        max_seq_len=max_seq_len, candidates=tuple(candidates), trials=tuple(trials), choice=choice
    )


# --------------------------------------------------------------------------------------
# (d) Token identity across tokenizers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizerComparison:
    """One tokenizer against the reference over every text."""

    label: str
    identity: str
    expected: str
    n_identical: int
    n_different: int
    first_divergence: dict[str, object] | None


@dataclass(frozen=True)
class TokenIdentityReport:
    """Gate (d): identical where the wave needs identity, different where the check needs teeth."""

    reference: str
    reference_identity: str
    n_texts: int
    comparisons: tuple[TokenizerComparison, ...]

    def failures(self) -> list[str]:
        """Name an expected-identical tokenizer that differs, or an expected-different one that never does."""
        failed: list[str] = []
        for comparison in self.comparisons:
            if comparison.expected == EXPECT_IDENTICAL and comparison.n_different:
                failed.append(
                    f"{comparison.label} tokenizes {comparison.n_different} of {self.n_texts} texts "
                    f"differently from {self.reference}; first divergence "
                    f"{comparison.first_divergence}"
                )
            if comparison.expected == EXPECT_DIFFERENT and not comparison.n_different:
                failed.append(
                    f"{comparison.label} tokenized every text identically to {self.reference}, so "
                    f"the comparison saw no difference anywhere and has not been watched to fail"
                )
        return failed

    @property
    def passed(self) -> bool:
        """Every expectation held."""
        return not self.failures()

    def as_payload(self) -> dict[str, object]:
        """Return the report block."""
        return {**asdict(self), "failures": self.failures(), "passed": self.passed}


def _first_divergence(reference_ids: Sequence[int], other_ids: Sequence[int]) -> dict[str, object]:
    index = next(
        (i for i, (a, b) in enumerate(zip(reference_ids, other_ids, strict=False)) if a != b),
        min(len(reference_ids), len(other_ids)),
    )
    return {
        "token_index": index,
        "reference_len": len(reference_ids),
        "other_len": len(other_ids),
        "reference_id": reference_ids[index] if index < len(reference_ids) else None,
        "other_id": other_ids[index] if index < len(other_ids) else None,
    }


def token_identity(  # noqa: PLR0913 - texts, the encoders, and which are expected to agree or differ
    texts: Sequence[str],
    encoders: Mapping[str, Callable[[str], list[int]]],
    *,
    identities: Mapping[str, str],
    reference: str,
    expected_identical: Sequence[str],
    expected_different: Sequence[str],
) -> TokenIdentityReport:
    """Gate (d) over given encoders: pure, so the mocked test can drive both expectations."""
    if not texts:
        raise GateFailureError("no texts to compare tokenizers on")
    reference_ids = [encoders[reference](text) for text in texts]
    comparisons: list[TokenizerComparison] = []
    for expected, labels in (
        (EXPECT_IDENTICAL, expected_identical),
        (EXPECT_DIFFERENT, expected_different),
    ):
        for label in labels:
            n_different = 0
            first: dict[str, object] | None = None
            for index, text in enumerate(texts):
                other = encoders[label](text)
                if other != reference_ids[index]:
                    n_different += 1
                    if first is None:
                        first = {
                            "text_index": index,
                            **_first_divergence(reference_ids[index], other),
                        }
            comparisons.append(
                TokenizerComparison(
                    label=label,
                    identity=identities[label],
                    expected=expected,
                    n_identical=len(texts) - n_different,
                    n_different=n_different,
                    first_divergence=first,
                )
            )
    report = TokenIdentityReport(
        reference=reference,
        reference_identity=identities[reference],
        n_texts=len(texts),
        comparisons=tuple(comparisons),
    )
    logger.info(
        "token identity, %s",
        f"n_texts={len(texts)} "
        + " ".join(f"{c.label}:{c.n_identical}/{len(texts)}" for c in comparisons),
    )
    return report


def load_tokenizer_encoders(
    specs: Sequence[tuple[str, str | None]],
) -> tuple[dict[str, Callable[[str], list[int]]], dict[str, str]]:
    """Load each ``(model_id, revision)`` tokenizer as an encoder that adds no special tokens.

    No special tokens is the corpus convention: the drawn texts already carry the template's own.
    The identity beside each encoder is the resolved hub commit, so the report says WHICH tokenizer
    files agreed rather than which model ids.
    """
    transformers = importlib.import_module("transformers")
    encoders: dict[str, Callable[[str], list[int]]] = {}
    identities: dict[str, str] = {}
    for model_id, revision in specs:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)

        def encode(text: str, *, tokenizer: object = tokenizer) -> list[int]:
            return cast(
                "list[int]", cast("Any", tokenizer)(text, add_special_tokens=False)["input_ids"]
            )

        encoders[model_id] = encode
        identities[model_id] = resolve_weights_identity(model_id, revision=revision)
    return encoders, identities


# --------------------------------------------------------------------------------------
# (e) Resume equality through fit_lens
# --------------------------------------------------------------------------------------


def lens_relative_diffs(
    a: Mapping[int, torch.Tensor], b: Mapping[int, torch.Tensor]
) -> dict[int, float]:
    """Per layer, ``||A - B||_F / ||B||_F`` between two lenses' Jacobians."""
    if set(a) != set(b):
        raise GateFailureError(f"lenses cover different layers: {sorted(a)} vs {sorted(b)}")
    return {
        layer: ((a[layer].float() - b[layer].float()).norm() / b[layer].float().norm()).item()
        for layer in sorted(b)
    }


@dataclass(frozen=True)
class ResumeEqualityReport:
    """Gate (e): resumed against straight, and the half fit that must differ."""

    n_prompts: int
    n_first_half: int
    checkpoint_every: int
    n_prompts_straight: int
    n_prompts_resumed: int
    n_prompts_half: int
    resumed_vs_straight: dict[int, float]
    half_vs_straight: dict[int, float]
    tolerance: float

    @property
    def max_relative_diff(self) -> float:
        """The worst layer, resumed against straight."""
        return max(self.resumed_vs_straight.values())

    @property
    def half_min_relative_diff(self) -> float:
        """The closest the half fit gets to the straight one; it must stay clear of the tolerance."""
        return min(self.half_vs_straight.values())

    def failures(self) -> list[str]:
        """Name every verdict that did not hold."""
        failed: list[str] = []
        if self.n_prompts_resumed != self.n_prompts or self.n_prompts_straight != self.n_prompts:
            failed.append(
                f"prompt counts: straight={self.n_prompts_straight} "
                f"resumed={self.n_prompts_resumed}, expected {self.n_prompts} for both"
            )
        if self.max_relative_diff > self.tolerance:
            failed.append(
                f"resumed fit differs from the straight fit by {self.max_relative_diff:.3e} "
                f"relative Frobenius (tolerance {self.tolerance})"
            )
        if self.half_min_relative_diff <= self.tolerance:
            failed.append(
                f"the half fit is within tolerance of the straight fit "
                f"({self.half_min_relative_diff:.3e}); the comparison cannot tell two different "
                f"lenses apart"
            )
        return failed

    @property
    def passed(self) -> bool:
        """Resumed equals straight; half does not."""
        return not self.failures()

    def as_payload(self) -> dict[str, object]:
        """Return the report block."""
        return {
            **asdict(self),
            "max_relative_diff": self.max_relative_diff,
            "half_min_relative_diff": self.half_min_relative_diff,
            "failures": self.failures(),
            "passed": self.passed,
        }


def _jacobians_of(lens: object) -> Mapping[int, torch.Tensor]:
    return cast("Mapping[int, torch.Tensor]", cast("Any", lens).jacobians)


def resume_equality(  # noqa: PLR0913 - three fits, a checkpoint and a tolerance
    jl: ModuleType,
    model: object,
    prompts: Sequence[str],
    config: JacobianConfig,
    *,
    work_dir: Path,
    tolerance: float,
) -> ResumeEqualityReport:
    """Gate (e): straight fit of all prompts; first half checkpointed; the rest resumed; compare.

    All three fits go through :func:`fit_lens`, so the knobs the config threads are the ones under
    test. The half fit doubles as the comparison's own falsifier: it is a real, different lens, and
    the check has to see it as one.
    """
    n = len(prompts)
    if n < 2:  # noqa: PLR2004 - a resume needs something before and after the checkpoint
        raise GateFailureError(f"resume equality needs at least 2 prompts, got {n}")
    half = n // 2
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = work_dir / "resume_gate_checkpoint.pt"
    if checkpoint.exists():
        checkpoint.unlink()
    base = replace(config, max_fit_prompts=n, checkpoint_path=None, resume=False)
    straight = fit_lens(base, model, prompts, jl)
    first = fit_lens(
        replace(base, checkpoint_path=checkpoint, checkpoint_every=half, resume=False),
        model,
        prompts[:half],
        jl,
    )
    if not checkpoint.exists():
        raise GateFailureError(f"the half fit wrote no checkpoint at {checkpoint}")
    resumed = fit_lens(
        replace(base, checkpoint_path=checkpoint, checkpoint_every=half, resume=True),
        model,
        prompts,
        jl,
    )
    report = ResumeEqualityReport(
        n_prompts=n,
        n_first_half=half,
        checkpoint_every=half,
        n_prompts_straight=int(cast("Any", straight).n_prompts),
        n_prompts_resumed=int(cast("Any", resumed).n_prompts),
        n_prompts_half=int(cast("Any", first).n_prompts),
        resumed_vs_straight=lens_relative_diffs(_jacobians_of(resumed), _jacobians_of(straight)),
        half_vs_straight=lens_relative_diffs(_jacobians_of(first), _jacobians_of(straight)),
        tolerance=tolerance,
    )
    logger.info(
        "resume equality, %s",
        f"n={n} half={half} max_rel_diff={report.max_relative_diff:.3e} "
        f"half_min_rel_diff={report.half_min_relative_diff:.3e} passed={report.passed}",
    )
    return report
