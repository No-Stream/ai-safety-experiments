"""Pre-flight for the interp read: check the fast generation engine feeds the capture pass aligned.

Run this on a box before spending it on a real interp run. It answers the one question the offline
tests cannot, because the answer depends on the machine and the checkpoint rather than on our logic:
do the token ids a live vLLM engine returns line up, position for position, with the HuggingFace
forward pass that captures activations over them?

That question is worth a dedicated check because every way of getting it wrong is silent. A record
whose prompt/response boundary is off by a token still captures cleanly, still pools into a
well-shaped vector, and still produces a run that looks finished -- it just labels the wrong
positions as the model's own reasoning. Nothing downstream can tell.

What it asserts, per generated response:

* the captured activations have one position per token of ``full_ids``, at EVERY layer;
* the response mask is false across the whole prompt and true across the whole response, and its
  count is exactly the generated length;
* decoding ``full_ids`` reproduces the decoded prompt followed by the response text, so the ids the
  capture ran over really are the prompt the engine was asked about plus what it wrote;
* the record's sampler is the one this run resolved for the engine it chose, which is what catches a
  config that never reached the engine.

It also reports generation throughput per engine, so a box that is quietly serving the slow path
shows up as a number rather than as a long afternoon.

Deliberately goes through ``capture_model_then_generator``, the same entry point the real stages use,
so it exercises the load ORDER as well as the alignment: a vLLM engine brought up before the capture
model leaves transformers unable to build a causal-LM head for the same Qwen3.5 checkpoint, and this
would fail in the same place a real run would.

Two engine-startup prerequisites, both observed on this dev box and both fatal at startup rather than
later. vLLM JIT-compiles its flashinfer top-k/top-p sampler, so ``ninja`` must be on ``PATH`` -- it
ships in the venv's ``bin``, which ``scripts/resource-limits.sh`` strips because it runs under
``systemd-run``, so pass ``PATH`` through ``env``. That build also needs the CUDA toolkit headers
(``curand.h``); where they are absent, ``VLLM_USE_FLASHINFER_SAMPLER=0`` falls back to vLLM's native
torch sampler and the engine starts. Worked example, from the repo root:

    scripts/resource-limits.sh --gpu -t 12m -- env \
        PATH="$PWD/.venv/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
        VLLM_USE_FLASHINFER_SAMPLER=0 \
        .venv/bin/python scripts/interp_generation_preflight.py --model-id Qwen/Qwen3.5-0.8B

Prompts are generic and written here rather than drawn from any item corpus, so this script carries no
benchmark material and can be run and its output pasted anywhere.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

import torch

from reward_hacking.interp.generation_capture import (
    DEFAULT_VLLM_GPU_FRACTION,
    GEN_ENGINES,
    PENALTY_FREE_THINKING_SAMPLING,
    GenerationRecord,
    capture_record_activations,
    default_gen_engine,
    resolved_sampler_for,
)
from reward_hacking.interp.run_harness import (
    DEFAULT_MODEL_ID,
    capture_model_then_generator,
    require_record_sampler,
)

logger = logging.getLogger("interp-generation-preflight")

PREFLIGHT_PROMPTS: tuple[str, ...] = (
    "A unit test asserts that summing an empty list gives 1. Briefly, what would you change?",
    "Describe in two sentences how you would check whether a sort function is correct.",
    "A script prints its result but never returns it. Briefly, what would you change?",
    "Explain in two sentences why an empty test suite can report success.",
)
"""Generic prompts, written here rather than taken from any item corpus.

Four rather than one because the check is about a BATCH: the whole point of the fast path is many
sequences in flight, and a batch is where a reply can be paired with the wrong prompt. Their content
does not matter -- only that they are long enough to draw a real continuation and different enough
that a mispairing would show up in the prompt comparison.
"""

DEFAULT_PREFLIGHT_MAX_NEW_TOKENS = 96
"""Short on purpose: this checks that positions line up, not that the model reasons well.

Every assertion here holds at any length, and a long cap would turn a two-minute pre-flight into
something an operator skips. A truncated trace is a perfectly good subject for an alignment check --
it is recorded as truncated and the arithmetic is unchanged.
"""


def check_positions_align(label: str, record: GenerationRecord, layer_count: int) -> None:
    """Refuse a record whose captured positions do not correspond 1:1 to its token ids."""
    if record.n_generated == 0:
        raise RuntimeError(
            f"{label}: the engine returned no response tokens, so there is nothing to align; the "
            "sampler or the token cap is wrong before alignment is even in question"
        )
    if int(record.is_response.shape[0]) != record.seq_len:
        raise RuntimeError(
            f"{label}: the response mask covers {record.is_response.shape[0]} positions for "
            f"{record.seq_len} token ids"
        )
    if bool(record.is_response[: record.prompt_len].any()):
        raise RuntimeError(f"{label}: the response mask is true inside the prompt")
    if not bool(record.is_response[record.prompt_len :].all()):
        raise RuntimeError(f"{label}: the response mask is false inside the response")
    if record.n_generated != record.seq_len - record.prompt_len:
        raise RuntimeError(
            f"{label}: the mask counts {record.n_generated} generated tokens where the boundary "
            f"says {record.seq_len - record.prompt_len}"
        )
    if len(record.token_strings) != record.seq_len:
        raise RuntimeError(
            f"{label}: {len(record.token_strings)} token strings for {record.seq_len} ids, so some "
            "position has no name"
        )
    logger.info(
        "%s: positions aligned over %d layers (seq_len=%d prompt_len=%d generated=%d cap_hit=%s)",
        label,
        layer_count,
        record.seq_len,
        record.prompt_len,
        record.n_generated,
        record.hit_token_cap,
    )


def check_capture_covers_every_position(
    label: str, record: GenerationRecord, positionwise: dict[int, torch.Tensor]
) -> None:
    """Refuse a capture whose per-layer activations are not one row per token id."""
    if not positionwise:
        raise RuntimeError(f"{label}: the capture returned no layers at all")
    for layer, activations in sorted(positionwise.items()):
        if activations.shape[0] != record.seq_len:
            raise RuntimeError(
                f"{label} layer {layer}: captured {activations.shape[0]} positions for "
                f"{record.seq_len} token ids, so every position label past the shorter of the two "
                "names a different token than it did at generation time"
            )


def check_ids_decode_to_the_prompt_and_response(
    label: str, tokenizer: object, record: GenerationRecord
) -> None:
    """Refuse ids that do not decode back to the prompt the engine saw plus what it wrote.

    The end-to-end statement, and the one that would catch a decoded-text round-trip sneaking back in:
    if ``full_ids`` were rebuilt by re-tokenising text, this is where the shifted boundary shows up.
    """
    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]
    whole = decode(record.full_ids.tolist(), skip_special_tokens=True)
    prompt_side = decode(record.full_ids[: record.prompt_len].tolist(), skip_special_tokens=True)
    if whole != prompt_side + record.response_text:
        raise RuntimeError(
            f"{label}: decoding the ids does not reproduce the prompt plus the response. "
            f"ids end {whole[-160:]!r}; prompt+response end "
            f"{(prompt_side + record.response_text)[-160:]!r}"
        )
    if record.prompt_text not in prompt_side:
        raise RuntimeError(
            f"{label}: the prompt this record claims is not inside the ids the capture ran over, so "
            "the record and the measurement are about different text"
        )
    logger.info("%s: ids decode to the prompt plus the response (%d characters)", label, len(whole))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Model, engine and budget: everything that legitimately differs between boxes."""
    parser = argparse.ArgumentParser(
        description="Check that the fast generation engine feeds the HuggingFace capture pass with "
        "every position aligned, on this box and this checkpoint."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="checkpoint to check; use the one the real run will use, since this is a per-box, "
        "per-checkpoint question",
    )
    parser.add_argument("--gen-engine", choices=list(GEN_ENGINES), default=default_gen_engine())
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_PREFLIGHT_MAX_NEW_TOKENS)
    parser.add_argument("--vllm-gpu-fraction", type=float, default=DEFAULT_VLLM_GPU_FRACTION)
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=len(PREFLIGHT_PROMPTS),
        help="how many of the built-in prompts to generate, capped at what there are",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate on the chosen engine, capture on HuggingFace, and refuse any misalignment."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "this pre-flight is GPU-only: it exists to check a live engine against a live capture "
            "pass, and neither runs on the CPU route for this model family"
        )
    prompts = list(PREFLIGHT_PROMPTS[: args.n_prompts])
    if not prompts:
        raise ValueError("--n-prompts must be at least 1, or this would pass without checking one")
    sampling = replace(PENALTY_FREE_THINKING_SAMPLING, max_new_tokens=args.max_new_tokens)
    free_gib, total_gib = (value / 1024**3 for value in torch.cuda.mem_get_info())
    logger.info(
        "checking %s on %s (%.1f of %.1f GiB free), generating on %s, capturing on HuggingFace",
        args.model_id,
        torch.cuda.get_device_name(0),
        free_gib,
        total_gib,
        args.gen_engine,
    )

    model, tokenizer, generator = capture_model_then_generator(
        args.model_id,
        gen_engine=args.gen_engine,
        sampling=sampling,
        vllm_gpu_fraction=args.vllm_gpu_fraction,
    )
    expected_sampler = resolved_sampler_for(sampling, engine=args.gen_engine)
    logger.info("resolved sampler: %s", expected_sampler.as_payload())

    started = time.monotonic()
    records = generator.generate_records(prompts)
    generation_seconds = time.monotonic() - started
    generated_tokens = sum(record.n_generated for record in records)

    for index, (prompt, record) in enumerate(zip(prompts, records, strict=True)):
        label = f"{generator.engine}[{index}]"
        if record.prompt_text != prompt:
            raise RuntimeError(
                f"{label}: this record answers a different prompt than it was paired with, so a "
                "batch of replies came back in an order the caller did not expect"
            )
        require_record_sampler(record, expected_sampler, stage="preflight")
        positionwise = capture_record_activations(model, record)
        check_capture_covers_every_position(label, record, positionwise)
        check_positions_align(label, record, len(positionwise))
        check_ids_decode_to_the_prompt_and_response(label, tokenizer, record)
        del positionwise

    logger.info(
        "generation on %s: %.1fs for %d tokens over %d prompts (%.1f tokens/s)",
        generator.engine,
        generation_seconds,
        generated_tokens,
        len(records),
        generated_tokens / generation_seconds,
    )
    logger.info(
        "PRE-FLIGHT PASSED: %s generates ids that %s captures position for position",
        generator.engine,
        args.model_id,
    )


if __name__ == "__main__":
    main()
