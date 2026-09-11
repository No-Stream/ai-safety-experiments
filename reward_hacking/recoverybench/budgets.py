"""Measured per-model output-token budgets, and a lookup that refuses to guess.

A token cap is not a tuning knob here: it decides how much apparent non-compliance a run
manufactures. At a 16,384-token cap the "no terminal answer" rate across three roster models was
21%, 54% and 85%, none of which was a model property -- the small reasoning models spend 11-19k
tokens of thinking on one hard item and simply never reach the answer line. Raising the cap took the
frontier model to 0 missing answers in 116 replies.

So the caps below are measurements, and :func:`max_tokens_for` raises for a model that has none
rather than falling back to a default. A default is what turns "nobody measured this model" into a
plausible-looking non-compliance rate.

The table is deliberately *narrower* than ``bedrock_batch.BATCH_ROSTER`` and does not line up with
it. Most roster models are absent because nobody measured them, not because they are missing; and
the frontier model is here but not in the roster, because it is reachable only on the live Converse
path. An import-time subset check between the two would therefore be wrong in both directions -- and
worse, it would make adding a batch-capable model for the sibling benchmark break ``import
reward_hacking.recoverybench``. What does connect them is the one number that is a property of a
model rather than a research budget: Nova Micro's API ceiling is defined beside the roster and
imported here, so it exists once.
"""

from __future__ import annotations

from reward_hacking.bedrock_batch import NOVA_MICRO_MAX_TOKENS
from reward_hacking.model_backend import BedrockSamplingConfig

# Enough for a small reasoning model's 11-19k-token thinking trace plus its answer.
SMALL_REASONING_MAX_TOKENS = 24_576

# The hard-band budget: at 30k the frontier model missed no answers in 116 replies.
FRONTIER_MAX_TOKENS = 30_000

MAX_TOKENS_BY_MODEL: dict[str, int] = {
    "us.amazon.nova-micro-v1:0": NOVA_MICRO_MAX_TOKENS,
    "openai.gpt-oss-20b-1:0": SMALL_REASONING_MAX_TOKENS,
    "openai.gpt-oss-120b-1:0": SMALL_REASONING_MAX_TOKENS,
    "qwen.qwen3-235b-a22b-2507-v1:0": SMALL_REASONING_MAX_TOKENS,
    "global.openai.gpt-5.6-luna": FRONTIER_MAX_TOKENS,
}


def bedrock_sampling_for(
    model_id: str, *, reasoning_effort: str | None = None
) -> BedrockSamplingConfig:
    """Build the sampling config for one model at its *measured* cap.

    The only sanctioned way to build a Bedrock sampling config for this bench, and the thing that
    makes the table above load-bearing rather than decorative. ``BedrockSamplingConfig.max_tokens``
    defaults to :data:`~reward_hacking.model_backend.DEFAULT_BEDROCK_MAX_TOKENS`, which is the
    frontier model's measured budget and at or above every other row here: for Nova Micro it is
    three times the ceiling the API hard-rejects above, so a hand-assembled run there fails outright
    rather than measuring anything, and for the small reasoning models it buys more trace than was
    measured while the labels record whichever cap somebody meant. Which direction a hand-assembled
    run goes wrong in has flipped once already -- while that default was 2048 it manufactured
    truncation instead -- and both directions are an argument for the factory: the default is not a
    measurement, whatever its value.

    Going through here also means an unmeasured model raises before a boto3 client exists, let alone
    a bill.
    """
    return BedrockSamplingConfig(
        max_tokens=max_tokens_for(model_id), reasoning_effort=reasoning_effort
    )


def max_tokens_for(model_id: str) -> int:
    """Look up the measured output-token cap for one model, refusing to guess for an unlisted one.

    Widening the roster means measuring the new model and adding a row above, which is a deliberate
    act. Guessing would spend a whole sweep to produce a truncation artifact.
    """
    budget = MAX_TOKENS_BY_MODEL.get(model_id)
    if budget is None:
        msg = (
            f"no measured output-token budget for {model_id!r}; measured models are "
            f"{sorted(MAX_TOKENS_BY_MODEL)}. Probe the model on a hard item and add a row rather "
            "than defaulting, or its truncated replies will read as non-compliance"
        )
        raise ValueError(msg)
    return budget
