"""Decode a prompt list under VRAM pressure: how wide a chunk, and what to do when it goes badly.

Three call sites feed one card. `games.select_prompts` sweeps a baseline policy, `games.evals` runs
the eval battery, and `games.screen_thinking` screens termination lengths; all three hand a flat
list of prompts to a backend that builds ONE padded tensor and makes ONE `generate` call, so the
whole list is resident at once. A 200-prompt eight-sample sweep handed over whole is 1,600 sequences
and OOMs on any card.

This lived in the middle of `games/select_prompts.py`, a prompt-selection CLI, for as long as there
was one caller. Nothing here is about prompt selection: it is memory arithmetic over a checkpoint's
architecture, a measured throughput knee, an allocator-thrash detector, and the chunk loop that
narrows a width when a card refuses it. Every constant carries the measurement it came from, because
each one was wrong once in a direction that cost a rented card hours.

**Nothing here hardcodes a memory budget** (`CLAUDE.md`): the width is derived from the VRAM
actually present at the moment of the call, so the same code takes a 24 GiB L4 and a 96 GiB rented
card without either being named.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from games.sizing import checkpoint_sequence_cost
from games.termination import decode_peak_tokens
from reward_hacking.model_backend import HFBackend, VLLMBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from games.sizing import SequenceCost
    from reward_hacking.model_backend import Backend

logger = logging.getLogger(__name__)

BYTES_PER_GIB = 1024**3
# The crude linear VRAM allowance this sizing used to be built on, kept for the one case that has
# nothing better: a backend whose model is not a checkpoint we can read an architecture off. It is
# wrong by 26x for a hybrid-attention 2B -- 8.60 GiB for a sequence whose config-derived cost at the
# same 24,576-token budget is 0.334 GiB -- which is why a 96 GiB card decoded 8 sequences at a time.
# It only ever binds a transport that does not decode on this process's GPU, where the number is
# arbitrary anyway; anything decoding here is sized by `decode_footprint`.
DECODE_VRAM_GIB_PER_KILOTOKEN = 0.35
# Headroom for allocator fragmentation, the CUDA context, and whatever else holds VRAM beside the
# sequences being decoded.
USABLE_VRAM_FRACTION = 0.8
# Measured 2026-08-18 with Qwen3.5-2B in bf16 on real twin-pd sheets (323-325 padded prompt
# tokens), greedy, `min_new_tokens == max_new_tokens` so every row ran to the cap. `peak_gib` is
# `torch.cuda.max_memory_allocated()`, which includes the 3.51 GiB of resident weights, so the
# per-sequence cost is the SLOPE across widths and not the peak divided by the width:
#
#     new tokens   w32     w64     w128    slope GiB/seq   intercept GiB
#            256   5.02    6.51     9.48          0.0465            3.53
#           1024   5.02    6.51     9.48          0.0465            3.53
#           8192   9.39   15.25    26.95          0.1829            3.54
#
# The intercept lands on the resident weights at every budget, so peak allocated is exactly
# `weights + slope * width`: a wider chunk costs linearly and never worse. The slope is IDENTICAL
# at 256 and 1,024 new tokens and then triples by 8,192, which is the shape one flat multiplier over
# the whole arithmetic cannot fit -- and why the old single 1.5 factor, calibrated against the
# short end, drifted low exactly where a real sweep runs. So each term of `SequenceCost` carries its
# own measured factor and the peak is the LARGER of the two regimes rather than their sum, because
# that is what a peak is: prefill sets it below the crossover, the KV cache above it.
#
# Raw records: docs/scratch/measure_decode_footprint_2026-08-18.py, and the knee sweep in
# s3://<bucket>/games_rl/step4-20260818/knee-{short-sequence,8192-token}.log.
#
# Above the crossover the peak grows at 1.90e-5 GiB/token/sequence against the 1.14e-5 the KV
# arithmetic accounts for; the extra is the transient copy `DynamicCache` makes when it concatenates
# the cache at every step, which briefly holds two copies of it.
MEASURED_KV_PEAK_OVER_PREDICTED = 1.70
# Below the crossover the peak is flat in length, and the arithmetic's own prefill term (three
# fp32 copies of q/k/v over the prompt) accounts for 0.007 GiB of the measured 0.047. Calibrated
# at one prompt length on one checkpoint, so it is the weakest constant here -- but it binds only
# below ~1,170 completion tokens, which is smoke-test territory; every real sweep runs at 16,384 or
# more, where the KV regime dominates and this term is not consulted.
MEASURED_PREFILL_PEAK_OVER_PREDICTED = 3.9
# `torch.cuda.mem_get_info` reports what the DRIVER has free, which is the caching allocator's
# RESERVE, while the table above measures what is ALLOCATED. Dividing the first by the second is the
# mistake that put 172 sequences of a 24,576-token budget on a 96 GiB card: 172 x 0.421 GiB = 72.4
# fitted inside the 72.6 GiB budget, and the run then raised a real OOM with only 53.95 GiB
# allocated, because a further 29.44 GiB was reserved and unusable (the allocator's own accounting,
# quoted in the failure at 17:54:20 in stage-logs/sweep-qwen35-2b.log). Episode-only that is 79.88
# GiB of reserve against 50.44 GiB of live data.
#
# Honest about what this factor is not: the reserve is a high-water mark thrown off by the
# grow-and-reallocate pattern above, and it saturates whatever it is given -- the width-32 run at
# 8,192 tokens held only 9.39 GiB allocated and still saw the card down to 0.13 GiB free. So no
# multiplier makes the memory arithmetic sufficient on its own; `knee_width_cap` is what actually
# keeps a sweep off the thrashing part of the curve, and this only stops the arithmetic from
# claiming reserve it will not get.
ALLOCATOR_RESERVE_OVER_ALLOCATED = 1.6
# Prompt length the footprint is budgeted at, matching `games/train.py --max-prompt-tokens`, which
# is the ceiling `games/dataset.py` enforces on the same corpus. Real twin-pd sheets measured 323
# tokens, so this is ~3x conservative and stays right if a longer frame is added later.
SWEEP_PROMPT_TOKEN_ALLOWANCE = 1024
# CPU decoding is bounded by patience rather than VRAM, and only smoke runs land here.
CPU_CHUNK_SEQUENCES = 4

# How many sequences a self-scheduling backend is handed at once when the caller persists results
# per chunk. Nothing about memory: a paged engine admits what fits and queues the rest, so this is
# purely the interval at which finished work reaches disk, and the only cost of narrowing it is the
# idle capacity at each chunk's tail, where the queue empties and concurrency decays to the last
# long sequence. The number balances those two:
#
#   * vLLM 0.27.1 admits at most `SchedulerConfig.DEFAULT_MAX_NUM_SEQS = 128` sequences at once
#     (read out of the installed package, and this repo passes no `max_num_seqs`), so 512 leaves a
#     queue three times the admission limit standing behind what is running. The drain tail then
#     costs at most one admission window's worth of decay per chunk -- roughly 128/512, an eighth
#     of the chunk -- rather than the whole chunk running below capacity.
#   * At the wave-4b sweep's shape (632 candidate rows x 8 samples = 5,056 sequences over 3-6 GPU
#     hours) a 512-sequence chunk is 64 rows, so a box that dies loses at most a tenth of the sweep
#     and about a twentieth on average, against all of it before there was a partial file at all.
#
# Read as sequences rather than rows because that is what the engine schedules; the caller floors it
# to a whole multiple of its samples-per-prompt (`self_scheduling_chunk_cap`) so a chunk boundary is
# also a row boundary and every chunk ends with whole records to write.
SELF_SCHEDULING_CHUNK_SEQUENCES = 512

# Aggregate decode throughput saturates well before the card runs out of memory. At 8,192 new
# tokens on the rented Blackwell, 32 -> 64 sequences bought 1.35x (1228.9 -> 1655.4 tok/s) and
# 64 -> 128 bought only 1.11x (-> 1836.2), while per-sequence throughput fell 38.4 -> 25.87 -> 14.35
# tok/s and GPU utilisation sat at 95-98% from width 32 upward. The card is compute-bound there, so
# past the knee a wider chunk buys almost nothing and loses more work every time one is thrown away.
#
# A bare width is the wrong shape for that knee, and assuming one transfers is precisely the error
# that was made: 64 was measured at 8,192 tokens, and the same width at a 24,576-token budget puts
# three times as many tokens in flight. What transfers is the count of in-flight COMPLETION tokens,
# which is what KV cache, memory traffic and utilisation all scale with. Anchoring on the
# width-64 point -- the widest with a real gain behind it and almost none ahead -- gives
# 64 x 8192 = 524,288, and that one number orders every observation we have:
#
#     width x budget    in-flight tokens   vs knee   outcome
#        32 x  8192              262,144      0.5x   completed 213 s, 1.35x throughput still ahead
#        64 x  8192              524,288      1.0x   completed 317 s, only 1.11x ahead
#       128 x  8192            1,048,576      2.0x   completed 571 s, but paid 100 alloc retries
#        32 x 24576              786,432      1.5x   no chunk in 16 min
#        64 x 24576            1,572,864      3.0x   no chunk in 13.7 min
#        86 x 24576            2,113,536      4.0x   no chunk in 44 min, 2.2x its own healthy ETA
#       172 x 24576            4,227,072      8.1x   raised a real OOM after 50 min
#
# Measured on one card with one checkpoint, so it is a per-card figure: re-run the knee sweep in
# docs/scratch/measure_decode_footprint_2026-08-18.py before trusting it on a different GPU. It is
# deliberately expressed against the *budget* rather than the lengths rows actually reach, because a
# chunk lives until its longest row stops -- see `games.termination.decode_peak_tokens`.
THROUGHPUT_KNEE_SEQUENCE_TOKENS = 524_288

# The rate at which the caching allocator retrying its allocations means a decode has stopped making
# progress, in retries per minute per in-flight sequence. Measured 2026-08-18 across every
# configuration run on the rented Blackwell, counting the allocator's own `memory allocation failed
# with OOM` warnings (one per `num_alloc_retries` increment) against the wall clock of the decode
# they landed in:
#
#     width   budget   retries   minutes    /min   /min/seq   outcome
#        32     8192         6      3.55    1.69     0.0528   completed in 213 s
#        64     8192        21      5.28    3.98     0.0622   completed in 317 s
#       128     8192       100      9.52   10.51     0.0820   completed in 571 s
#        32    24576        50     16.01    3.13     0.0977   no chunk, stopped before its ETA
#        64    24576        90     13.11    6.86     0.1072   no chunk, stopped before its ETA
#        86    24576       558     44.10   12.65     0.1470   no chunk in 2.2x its ETA
#       172    24576      2030     49.93   40.69     0.2366   raised a real OOM after 50 min
#
# The first thing that table says is that the obvious threshold does not work. Retries per MINUTE
# does not separate the two regimes at all: the width-128 run sustained 10.5/min and finished
# normally, while the width-86 run thrashed at 12.6/min -- and the width-32 thrash showed 3.1/min,
# *less* than a width-64 run that completed. Retries scale with how large the allocations being
# requested are, hence with the width, so the rate has to be normalised per in-flight sequence
# before it means anything. Once it is, the ordering is monotone.
#
# 0.12 sits 1.46x above the highest rate any run that completed ever showed (0.0820, width 128 --
# which `knee_width_cap` forbids anyway, so the margin against a width this code will actually
# derive is 1.93x) and 1.23x below the lowest confirmed thrash (0.1470). The two middle rows sit
# below it on purpose: both runs were stopped by hand before their first chunk was even due -- a
# healthy width-64 chunk at 24,576 tokens needs ~16 min and it was killed at 13.7 -- so they are not
# evidence of thrashing, and halving on them would cost throughput for nothing.
THRASH_RETRIES_PER_MINUTE_PER_SEQUENCE = 0.12
# Warmup allowance. A width-32 run that completed cleanly still paid 6 retries, and the whole
# ten-run short-sequence grid paid exactly one between them, so a handful of retries is ordinary
# fragmentation and must never narrow a width.
THRASH_MIN_RETRIES = 8
# A rate over a short window is noise, not a trend: a prefill burst of 8 retries in 20 s reads
# as 24/min. Every decode this catches ran for tens of minutes, and the shortest chunk in the
# knee table took 213 s, so a minute costs nothing and is what makes the reading "sustained".
THRASH_MIN_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class DecodeFootprint:
    """What one in-flight sequence is budgeted to cost, and where that number came from."""

    gib_per_sequence: float
    sizing_tokens: int
    basis: str


def decode_footprint(
    *, max_new_tokens: int, model_id: str | None, cost: SequenceCost | None = None
) -> DecodeFootprint:
    """Budget one in-flight sequence, from the checkpoint's architecture and its termination screen.

    Two things the old flat allowance got wrong, in the same direction. It budgeted the full token
    ceiling for every sequence when no model in the ladder generates anywhere near it, and it priced
    a token at 0.35 GiB per thousand where a whole Qwen3.5-2B sequence at that budget costs 0.334
    GiB. Together they charged 8.6 GiB for a sequence whose real peak is a third of one, so a card
    with 90.8 GiB free decoded eight sequences at a time -- half the card idle, generating at the
    rate this dev box's L4 manages.

    `cost` is injectable so the arithmetic is testable without a config download; left unset it is
    read from the checkpoint. A `model_id` of None means the caller has no checkpoint to read (a
    hosted transport, a mock, a stand-in in a test), and only then does the crude flat allowance
    apply.
    """
    if model_id is None:
        return DecodeFootprint(
            gib_per_sequence=DECODE_VRAM_GIB_PER_KILOTOKEN * max_new_tokens / 1000,
            sizing_tokens=max_new_tokens,
            basis=(
                "no local checkpoint to read an architecture from, so the crude flat allowance of "
                f"{DECODE_VRAM_GIB_PER_KILOTOKEN} GiB per kilotoken at the full token ceiling"
            ),
        )
    sizing_tokens, why = decode_peak_tokens(model_id, ceiling=max_new_tokens)
    resolved = cost if cost is not None else checkpoint_sequence_cost(model_id)
    allocated = calibrated_peak_allocated_gib(
        resolved, prompt_tokens=SWEEP_PROMPT_TOKEN_ALLOWANCE, completion_tokens=sizing_tokens
    )
    return DecodeFootprint(
        gib_per_sequence=allocated * ALLOCATOR_RESERVE_OVER_ALLOCATED,
        sizing_tokens=sizing_tokens,
        basis=(
            f"{model_id} architecture at {SWEEP_PROMPT_TOKEN_ALLOWANCE}+{sizing_tokens} tokens, "
            f"calibrated against the measured decode peaks, predicts {allocated:.3f} GiB/sequence "
            f"allocated, scaled by {ALLOCATOR_RESERVE_OVER_ALLOCATED} for allocator reserve; "
            f"length from {why}"
        ),
    )


def calibrated_peak_allocated_gib(
    cost: SequenceCost, *, prompt_tokens: int, completion_tokens: int
) -> float:
    """Predict one sequence's peak *allocated* VRAM, each term against its own measurement.

    The larger of two regimes rather than their sum, because a peak is a maximum and not a total:
    below the crossover the prefill transient is the high-water mark and the peak does not move with
    output length at all (256 and 1,024 new tokens measured identically at every width), while above
    it the KV cache has overtaken prefill and sets the peak alone. Summing them would double-count
    the crossover and read 10% high at 8,192 tokens, where this instead lands within 3%.

    The recurrent DeltaNet state is in both regimes since it is resident throughout, so it does not
    matter which one wins. Reproduces the measured slopes at 256, 1,024, 2,048 and 8,192 new tokens;
    see `MEASURED_KV_PEAK_OVER_PREDICTED` for the table and where each factor came from.
    """
    recurrent_gib = cost.recurrent_bytes_per_sequence / BYTES_PER_GIB
    prefill_gib = (
        cost.prefill_transient_bytes(prompt_tokens=prompt_tokens) / BYTES_PER_GIB - recurrent_gib
    )
    kv_gib = (
        cost.kv_cache_bytes(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        / BYTES_PER_GIB
    )
    prefill_regime = recurrent_gib + prefill_gib * MEASURED_PREFILL_PEAK_OVER_PREDICTED
    kv_regime = recurrent_gib + kv_gib * MEASURED_KV_PEAK_OVER_PREDICTED
    return max(prefill_regime, kv_regime)


def knee_width_cap(*, max_new_tokens: int) -> int:
    """Cap the chunk width where aggregate throughput stopped rewarding a wider one.

    Expressed in in-flight completion tokens rather than sequences so it travels across token
    budgets: see `THROUGHPUT_KNEE_SEQUENCE_TOKENS` for the measured knee and the table of every
    observation it orders. Never returns zero, because a run that cannot decode one sequence should
    fail in the memory arithmetic with the card's numbers in the message, not here.
    """
    if max_new_tokens < 1:
        raise ValueError(f"a completion budget of {max_new_tokens} tokens decodes nothing")
    return max(1, THROUGHPUT_KNEE_SEQUENCE_TOKENS // max_new_tokens)


def derive_chunk_size(*, free_gib: float, n_sequences: int, footprint: DecodeFootprint) -> int:
    """Fit as many concurrent sequences into the free VRAM as the footprint allows.

    Pure arithmetic over a measured card, so the same call answers "what does this pick on a 96 GiB
    rented card" and "what does it pick on this box's L4" without either being present.
    """
    if footprint.gib_per_sequence <= 0:
        raise ValueError(f"a sequence cannot cost nothing, {footprint=}")
    derived = int(free_gib * USABLE_VRAM_FRACTION / footprint.gib_per_sequence)
    return max(1, min(derived, n_sequences))


def backend_schedules_own_batch(backend: Backend) -> bool:
    """Report whether this backend decides for itself how many sequences are in flight.

    True only for vLLM, which pages its own KV cache and admits sequences up to that budget while
    queueing the rest. For such a backend the chunk width is not a memory decision at all: handing
    it a hundred sequences does not put a hundred in flight, it hands its scheduler a hundred to
    schedule. `HFBackend` is the opposite -- one padded tensor, one `generate` call, every sequence
    resident at once -- which is what the VRAM arithmetic below exists to size.
    """
    return backend.transport == VLLMBackend.transport


def self_scheduling_chunk_cap(*, samples_per_prompt: int) -> int:
    """Return the widest handover to a self-scheduling backend that still ends on a row boundary.

    `SELF_SCHEDULING_CHUNK_SEQUENCES` floored to a whole multiple of the samples drawn per prompt, so
    every chunk a caller writes out holds complete prompts and never half of one. One prompt's worth
    is the floor: a sweep drawing more samples per prompt than the cap allows still has to decode a
    whole prompt before there is anything to write.
    """
    if samples_per_prompt < 1:
        raise ValueError(f"a prompt cannot be sampled {samples_per_prompt} times")
    whole_prompts = (SELF_SCHEDULING_CHUNK_SEQUENCES // samples_per_prompt) * samples_per_prompt
    return max(samples_per_prompt, whole_prompts)


def sweep_chunk_size(  # noqa: PLR0913 - one width, and every argument is an input to it
    *,
    max_new_tokens: int,
    n_sequences: int,
    requested: int | None = None,
    model_id: str | None,
    schedules_own_batch: bool = False,
    self_scheduling_cap: int | None = None,
) -> int:
    """Choose how many sequences to decode per backend call, from the VRAM actually present.

    `HFBackend.generate` builds one padded tensor over the whole prompt list and makes a single
    `model.generate` call, so a 200-prompt eight-sample sweep handed over whole is 1600 sequences
    and will OOM on any card. The chunk size is derived rather than hardcoded because a config
    that assumes this box's 24 GiB cannot take whatever machine happens to be free.

    Free VRAM is read after the model is resident, since the backend is built before any sweep runs,
    so the weights are already out of the number this divides up.

    `self_scheduling_cap` is the one width that is not a hardware statement at all. A caller that
    writes its records out per chunk needs the handover bounded so a death costs one chunk instead of
    the whole pass, and for a paged engine that bound is free: handing it 512 of 5,056 sequences
    still fills its scheduler and queues the rest. Unset, a self-scheduling backend is handed
    everything, which is what a caller that only writes at the end should do -- narrowing the width
    would buy nothing and pay the drain tail at every chunk.

    Two things bound the width, and which one binds says something different. The memory arithmetic
    binds on a small card; the measured knee binds on a large one, where the card runs out
    of useful parallelism long before it runs out of VRAM. A derived width over the knee is clamped,
    because the width is bookkeeping rather than a measurement -- the same call `plan_sizing` makes
    about prompts per step. An *explicitly requested* width over it is refused instead: a human
    naming a number is asserting something about the hardware, and the one time that assertion was
    wrong it cost 93 minutes of a rented card for zero completions.
    """
    if n_sequences < 1:
        raise ValueError(f"nothing to decode, {n_sequences=}")
    # The knee was measured on a GPU decoding in this process. For a hosted transport a chunk is
    # request batching, and on CPU nothing about it applies, so it binds neither.
    decodes_on_this_gpu = model_id is not None and torch.cuda.is_available()
    cap = knee_width_cap(max_new_tokens=max_new_tokens)
    if requested is not None:
        if requested < 1:
            raise ValueError(f"chunk size must be positive, got {requested}")
        if decodes_on_this_gpu and requested > cap:
            raise RuntimeError(
                f"requested chunk width {requested} is past the measured throughput knee of {cap} "
                f"sequences at a {max_new_tokens}-token budget "
                f"({THROUGHPUT_KNEE_SEQUENCE_TOKENS} in-flight completion tokens / "
                f"{max_new_tokens}). Past the knee a wider chunk buys almost nothing -- 64 -> 128 "
                f"sequences bought 1.11x aggregate throughput at 8,192 tokens -- and the allocator "
                f"thrashes instead of decoding: a 172-wide chunk at this budget returned zero "
                f"completions in 93 minutes. Pass --chunk-size {cap} or lower, or leave it unset "
                f"and let the arithmetic derive it. If the knee is truly different on this card, "
                f"re-measure it and move THROUGHPUT_KNEE_SEQUENCE_TOKENS rather than overriding it "
                f"per run."
            )
        return min(requested, n_sequences)
    if schedules_own_batch:
        # Measured 2026-08-19, and the reason this branch exists: two rented-card sweeps ran the
        # VRAM arithmetic below against a resident vLLM engine, which reserves ~90 of 96 GiB by
        # design, so `free_gib` was near zero, the derived width floored at 1, and a paged batching
        # engine decoded one sequence at a time at ~30-42s each -- turning a ~15-minute sweep into
        # 2-3 hours. The engine's own reservation is what made the free-VRAM reading meaningless.
        if self_scheduling_cap is None:
            logger.info(
                f"handing all {n_sequences} sequences to a self-scheduling backend; its paged cache "
                f"admits what fits and queues the rest, so free VRAM says nothing about this width"
            )
            return n_sequences
        width = min(self_scheduling_cap, n_sequences)
        logger.info(
            f"handing a self-scheduling backend {width} of {n_sequences} sequences at a time; its "
            f"paged cache admits what fits and queues the rest, so this width is the interval at "
            f"which finished records reach disk rather than a memory decision"
        )
        return width
    if not torch.cuda.is_available():
        logger.info(f"no CUDA device visible; decoding on CPU in chunks of {CPU_CHUNK_SEQUENCES}")
        return min(CPU_CHUNK_SEQUENCES, n_sequences)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / BYTES_PER_GIB
    footprint = decode_footprint(max_new_tokens=max_new_tokens, model_id=model_id)
    derived = derive_chunk_size(free_gib=free_gib, n_sequences=n_sequences, footprint=footprint)
    chunk = min(derived, cap) if decodes_on_this_gpu else derived
    logger.info(
        f"decode chunk derived from free VRAM: {chunk=} {free_gib=:.1f} "
        f"total_gib={total_bytes / BYTES_PER_GIB:.1f} {max_new_tokens=} "
        f"gib_per_sequence={footprint.gib_per_sequence:.3f} knee_cap={cap} "
        f"memory_derived={derived} device={torch.cuda.get_device_name()} basis={footprint.basis}"
    )
    if chunk < derived:
        logger.warning(
            "the throughput knee, not VRAM, is what limits this decode: %s",
            f"clamped {derived} -> {chunk} sequences because {max_new_tokens} tokens x {derived} "
            f"sequences puts {max_new_tokens * derived} tokens in flight against a knee of "
            f"{THROUGHPUT_KNEE_SEQUENCE_TOKENS}. The card has VRAM for the wider chunk and would "
            f"thrash its allocator rather than use it.",
        )
    return chunk


def local_decode_model_id(backend: Backend) -> str | None:
    """Return the checkpoint whose VRAM this sweep's chunk size has to respect, or None.

    Only `HFBackend` decodes a padded batch inside this process, so only it can be sized from a
    checkpoint's architecture. vLLM runs its own paged cache and evicts finished sequences, and the
    hosted transports decode on someone else's hardware; for those a chunk is request batching and
    the VRAM arithmetic says nothing.
    """
    if backend.transport != HFBackend.transport:
        return None
    return backend.model_id


def allocator_retry_count() -> int:
    """Read how many allocations the caching allocator has had to retry, process-cumulatively.

    `num_alloc_retries` is torch's count of "failed `cudaMalloc` calls that result in a cache flush
    and retry". That absorption is the whole reason `torch.OutOfMemoryError` is not a sufficient
    signal: under the realistic failure -- a width that barely fits and then degrades as its
    sequences lengthen -- the allocator frees cached blocks, retries, succeeds, and raises nothing,
    so a decode can grind for an hour producing nothing with the exception path silent throughout.

    `torch.cuda.memory_stats()` returns an empty mapping until the allocator has been initialised
    (`memory_stats_as_nested_dict` short-circuits on `is_initialized()`), which is why the key is
    read with a default rather than indexed. Zero without CUDA, so a CPU sweep measures a flat line
    and the detector is inert rather than wrong.
    """
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.memory_stats().get("num_alloc_retries", 0))


@dataclass(frozen=True, slots=True)
class AllocatorProbe:
    """Where the thrash detector reads its two numbers.

    Injectable so the detector can be driven on CPU with no GPU and no monkeypatching: a test hands
    over a fake clock and a fake counter and gets the same arithmetic the card drives.
    """

    clock: Callable[[], float] = time.monotonic
    read_retries: Callable[[], int] = allocator_retry_count


LIVE_ALLOCATOR_PROBE = AllocatorProbe()


@dataclass(frozen=True, slots=True)
class ThrashVerdict:
    """What the allocator's retry counter said across one decode call."""

    retries: int
    seconds: float
    width: int

    @property
    def retries_per_minute_per_sequence(self) -> float:
        """Return the retry rate normalised per in-flight sequence, or 0.0 over no time at all."""
        if self.seconds <= 0:
            return 0.0
        return self.retries / (self.seconds / 60.0) / self.width

    @property
    def thrashing(self) -> bool:
        """Report whether this decode spent its time reshuffling VRAM rather than generating."""
        return (
            self.retries >= THRASH_MIN_RETRIES
            and self.seconds >= THRASH_MIN_SECONDS
            and self.retries_per_minute_per_sequence >= THRASH_RETRIES_PER_MINUTE_PER_SEQUENCE
        )

    def __str__(self) -> str:
        """Render every number the verdict rests on, since it is read out of a log."""
        return (
            f"width={self.width} alloc_retries={self.retries} seconds={self.seconds:.1f} "
            f"retries_per_min={self.retries / max(self.seconds / 60.0, 1e-9):.2f} "
            f"retries_per_min_per_sequence={self.retries_per_minute_per_sequence:.4f} "
            f"threshold={THRASH_RETRIES_PER_MINUTE_PER_SEQUENCE}"
        )


@dataclass(frozen=True, slots=True)
class ChunkAttempt:
    """One decode call's outcome: what it produced, and whether it was thrashing while it did.

    `completions` is None exactly when the card refused the chunk outright with a raised OOM.
    `thrashing` covers the other half of the failure space, where nothing was raised at all.
    """

    completions: list[str] | None
    thrashing: bool


def iter_decoded_chunks(
    backend: Backend,
    prompts: Sequence[str],
    *,
    chunk_size: int,
    probe: AllocatorProbe = LIVE_ALLOCATOR_PROBE,
) -> Generator[list[str]]:
    """Yield each chunk's completions as it returns, narrowing the width when one goes badly.

    The streaming form of `decode_in_chunks`, and the seam a caller writes partial results through: a
    sweep that only learns anything when the last chunk returns has nothing on disk when its box
    dies. Everything else about the loop is that function's, including the cursor discipline that
    keeps each completion beside its own prompt.

    A chunk is yielded only once it is in hand, so a caller that persists what it is handed persists
    finished work only, and a raise from a later chunk leaves the earlier ones written.
    """
    decoded = 0
    width = chunk_size
    while decoded < len(prompts):
        batch, width = _decode_chunk_that_fits(backend, prompts[decoded:], width=width, probe=probe)
        if not batch:
            raise RuntimeError(
                f"the backend returned no completions for a chunk of {width} prompts, so this loop "
                f"would never finish; {decoded} of {len(prompts)} decoded so far. A "
                f"hanging sweep is worse than a failed one, which is why this raises."
            )
        decoded += len(batch)
        logger.info(
            f"sweep progress: {decoded}/{len(prompts)} completions "
            f"model={backend.model_id} transport={backend.transport} chunk_width={width}"
        )
        yield batch


def decode_in_chunks(
    backend: Backend,
    prompts: Sequence[str],
    *,
    chunk_size: int,
    probe: AllocatorProbe = LIVE_ALLOCATOR_PROBE,
) -> list[str]:
    """Decode the flattened prompt list chunk by chunk, narrowing the width when one goes badly.

    Completions accumulate in request order and a chunk only advances the cursor once it has
    returned, so a halved retry re-decodes exactly the prompts that failed and every completion
    still lands beside the prompt that produced it. That property is what
    `test_chunking_keeps_each_prompt_with_its_own_completions` asserts, and both recovery paths have
    to hold it or the whole trace is silently misaligned.

    A narrowed width is carried forward rather than reset per chunk: a chunk that went badly tells
    us the sizing was optimistic, and re-learning that at every chunk would pay for a bad decode
    over and over.

    The whole-list view over `iter_decoded_chunks`, for the callers that have nothing to write until
    every completion is in hand.
    """
    completions: list[str] = []
    for batch in iter_decoded_chunks(backend, prompts, chunk_size=chunk_size, probe=probe):
        completions.extend(batch)
    return completions


def _decode_chunk_that_fits(
    backend: Backend, remaining: Sequence[str], *, width: int, probe: AllocatorProbe
) -> tuple[list[str], int]:
    """Decode up to `width` of the remaining prompts, narrowing until the card decodes them well.

    Two ways a width can be wrong and one recovery. A raised OOM means the chunk never produced
    anything, so the same prompts are re-decoded at half the width; recursion rather than a loop so
    the halving is bounded by log2 of the starting width, and the caller's cursor never moves until
    this returns. A thrash verdict means the chunk *did* produce its completions, just far too
    slowly, so those are kept and the narrowing applies to what comes next -- re-decoding work
    already correctly in hand would pay the thrash a second time, which is the exact cost the
    detector exists to avoid.

    `empty_cache` on both paths, and on the thrash path it is the half of the recovery that most
    likely does the work: thrash is fragmentation, and returning the allocator's reserve to the
    driver addresses it directly where a narrower width only reduces the pressure.

    The retry happens here rather than inside `_decode_one_chunk`'s exception handler, and that
    split is load-bearing -- see that function.
    """
    attempt = _decode_one_chunk(backend, remaining, width=width, probe=probe)
    if attempt.completions is None:
        torch.cuda.empty_cache()
        return _decode_chunk_that_fits(backend, remaining, width=width // 2, probe=probe)
    if not attempt.thrashing or width == 1:
        return attempt.completions, width
    torch.cuda.empty_cache()
    return attempt.completions, width // 2


def _decode_one_chunk(
    backend: Backend, remaining: Sequence[str], *, width: int, probe: AllocatorProbe
) -> ChunkAttempt:
    """Decode one chunk, reporting a refusal or a thrash rather than acting on either.

    Reporting a failure through a return value instead of retrying inside the handler is the
    difference between a recovery path that works and one that only looks like it does. While an
    exception is being handled its traceback holds every frame of the failed forward pass alive, and
    with them every activation that pass had allocated -- measured at 0.85 GiB on top of a 2B's 3.51
    GiB of resident weights, more than the chunk that failed. So `empty_cache` called inside the
    handler frees none of it, the halved retry OOMs on memory the first attempt is still holding,
    and the width collapses to 1 and raises. That is what happened on 2026-08-18 when the first
    version of this was sabotaged with a capped allocator. The fix is to let the handler return so
    Python releases the exception, and only then empty the cache and retry. The thrash path inherits
    that property for free, since it is reached with no exception in flight at all.

    Loud rather than quiet on purpose, on both paths. A silent recovery would hide the one thing a
    recovery proves -- that the derived chunk size was too large -- and the sizing arithmetic would
    keep being wrong in the same direction on every future run with nothing in the log to say so.
    """
    retries_before = probe.read_retries()
    started = probe.clock()
    try:
        completions = backend.generate(list(remaining[:width]))
    except torch.OutOfMemoryError as oom:
        if width == 1:
            logger.exception(
                "decode OOM'd at a single sequence, so there is nothing left to halve. The card "
                "cannot hold one sequence at this token budget: use a larger card, or a shorter "
                "budget in a run labelled as not being a measurement."
            )
            raise
        # `error` rather than `exception`, against ruff's advice and deliberately: the useful part
        # of a CUDA OOM is the allocator's own accounting, which is in the message and is repeated
        # here, while the traceback is 140 frames of the model's forward pass and would bury the log
        # this line exists to be found in. The width-1 case above does re-raise, and keeps its.
        logger.error(  # noqa: TRY400
            "decode chunk OOM'd; halving the chunk width and retrying the SAME prompts, %s",
            f"old_width={width} new_width={width // 2} remaining={len(remaining)} "
            f"allocated_gib={torch.cuda.memory_allocated() / BYTES_PER_GIB:.1f}. The derived chunk "
            f"size was too large, which means the footprint arithmetic in decode_footprint "
            f"understated this configuration -- worth a look rather than a shrug. The allocator "
            f"said: {oom}",
        )
        return ChunkAttempt(completions=None, thrashing=False)
    verdict = ThrashVerdict(
        retries=probe.read_retries() - retries_before,
        seconds=probe.clock() - started,
        width=width,
    )
    if not verdict.thrashing:
        return ChunkAttempt(completions=completions, thrashing=False)
    if width == 1:
        logger.error(
            "ALLOCATOR THRASH at a single sequence, so there is no width left to narrow. This "
            "decode is spending its time reshuffling VRAM rather than generating and will not get "
            "faster: %s. Use a larger card, or run the sweep at a shorter budget in a run labelled "
            "as not being a measurement.",
            verdict,
        )
        return ChunkAttempt(completions=completions, thrashing=False)
    logger.error(
        "ALLOCATOR THRASH: the CUDA caching allocator absorbed sustained memory pressure without "
        "ever raising, so this chunk decoded but spent its time freeing and re-acquiring blocks "
        "instead of generating. Narrowing to %d for the remaining prompts and keeping the "
        "completions this chunk already produced. %s. Nothing raised an OutOfMemoryError, which is "
        "exactly why this check exists: the exception path only fires when a width cannot fit at "
        "all, and a width that barely fits and then degrades looks like success. Worth a look at "
        "the footprint arithmetic in decode_footprint rather than a shrug.",
        width // 2,
        verdict,
    )
    return ChunkAttempt(completions=completions, thrashing=True)


def stream_vllm_completions(
    backend: VLLMBackend, prompts: Sequence[str]
) -> Generator[tuple[int, str]]:
    """Yield ``(prompt index, completion text)`` from one pooled submission, in completion order.

    The eval battery's pooled path (`games.evals._run_pooled`). `VLLMBackend.generate_streaming`
    adds every prompt to the engine at once and hands each completion back the moment its own
    sequence stops, whatever else is still in flight -- what lets the trace be appended per record
    and turns a mid-cell death into a loss of zero finished records. This narrows each record to its
    text the way `VLLMBackend.generate` narrows `generate_detailed`; the pairing checks, the RNG
    caveat (statistically neutral relative to per-game calls, byte-identical for the same list on
    the same engine seed) and the engine mechanics live on the backend method.

    The backend's generator is closed the moment this one stops, however it stops, so what the
    engine still had in flight is aborted then rather than whenever the finaliser happens to run --
    the closing idiom `generate_streaming` asks of any consumer that may not drain it to the end.
    """
    if backend.transport != VLLMBackend.transport:
        raise TypeError(
            f"stream_vllm_completions drives a vLLM engine and was handed a {backend.transport!r} "
            f"backend; use games.evals._generate for every other transport."
        )
    with contextlib.closing(backend.generate_streaming(prompts)) as drained:
        for index, tokenized in drained:
            yield index, tokenized.completion.text
