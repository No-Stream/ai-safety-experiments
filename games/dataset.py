"""Turn authored prompt rows into the TRL training dataset, templated exactly once.

The design invariant this file exists to hold: **one variant-applied user turn flows through
training and eval unchanged.** The corpus stores the untemplated base text; this builder applies
the selected named variant, then chat-templates the result as a single user turn with no system
prompt, which is byte-identical to what
`reward_hacking.model_backend.HFBackend` produces for the sweeps and evals. Diverging by even a
system message would mean the "before" measurement and the training rollouts saw different
prompts, and nothing downstream would report that.

Two things here are not stylistic:

**Over-length prompts are dropped, never truncated.** TRL 1.10 removed `GRPOConfig`'s
`max_prompt_length`, and its old left-truncation is actively wrong for these games: cutting the
front off a payoff table leaves a prompt whose stated game no longer matches the payoff columns
the reward function grades against, so the reward becomes unreachable and poisons training while
looking fine.

**`enable_thinking` is always passed explicitly.** Template defaults vary non-monotonically
across the model ladder -- Qwen3.5-4B and 3.8-27B prefill `<think>` by default while 3.5-2B and
3.5-0.8B default to a closed empty block, i.e. thinking off (measured 2026-08-17,
`docs/scratch/qwen38-27b-load-check-2026-08-17.md`). Relying on the default would silently turn
thinking off for the smaller arms, and the decision-theory reasoning this project reads lives in
the chain of thought.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset

from games.prompt_variants import apply_prompt_variant

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

RAW_PROMPT_COLUMN = "raw_prompt"
PROMPT_COLUMN = "prompt"

# Injected into the reward function's kwargs by TRL 1.10 (`trainer/grpo_trainer.py:1625`).
TRL_INJECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "log_metric",
        "log_extra",
        "trainer_state",
        "environments",
    }
)

# Written by this builder, so a row arriving with one already set is ambiguous about which wins.
BUILDER_OWNED_COLUMNS: frozenset[str] = frozenset({RAW_PROMPT_COLUMN})

# Set by this builder itself, so a caller repeating one could silently flip thinking off.
TEMPLATE_ARGUMENTS_OWNED_HERE: frozenset[str] = frozenset(
    {
        "tokenize",
        "add_generation_prompt",
        "enable_thinking",
    }
)


def _assert_usable_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject a corpus that cannot be built into a well-formed dataset."""
    if not rows:
        raise ValueError("rows is empty; there is nothing to train on.")

    schema = frozenset(rows[0].keys())
    ragged = [index for index, row in enumerate(rows) if frozenset(row.keys()) != schema]
    if ragged:
        raise ValueError(
            f"All rows must carry the same columns. Rows {ragged} differ from row 0's "
            f"{sorted(schema)}; TRL forwards every column as a parallel list, so a missing key "
            f"would reach the reward function as a hole rather than an error."
        )
    if PROMPT_COLUMN not in schema:
        raise ValueError(
            f"Rows must carry the untemplated prompt in a {PROMPT_COLUMN!r} column, got "
            f"{sorted(schema)}."
        )

    reserved = sorted(schema & (TRL_INJECTED_COLUMNS | BUILDER_OWNED_COLUMNS))
    if reserved:
        raise ValueError(
            f"Column names {reserved} are reserved. TRL injects "
            f"{sorted(TRL_INJECTED_COLUMNS)} into the reward function's kwargs and would "
            f"silently overwrite a dataset column of the same name, and "
            f"{sorted(BUILDER_OWNED_COLUMNS)} is written by this builder."
        )


def build_game_dataset(  # noqa: PLR0913 - the builder's keyword-only controls are one contract
    rows: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int,
    prompt_variant: str,
    enable_thinking: bool = True,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> Dataset:
    """Chat-template every row's raw prompt and return the dataset TRL trains on.

    The templated text lands in `prompt` (what TRL tokenises), the varied user-turn text is
    preserved as `raw_prompt`, and every other column survives untouched so the reward function can
    reconstruct each row's game. Prompt variants are applied before templating and length counting,
    so the budget measures the exact user turn sent to the model. Prompts over
    `max_prompt_tokens` are dropped and counted.

    `chat_template_kwargs` forwards extra template knobs that only some models have. The one
    that matters today is Qwen3.8-27B's `reasoning_effort`: its template injects an unauthored
    system message ("Reasoning effort is set to xhigh...") unless passed `"medium"`, so without
    this the headline 27B arm would train on prompt text nobody wrote, with thinking budgets no
    other arm has. Keys this builder sets itself are rejected rather than merged.
    """
    if max_prompt_tokens <= 0:
        raise ValueError(f"max_prompt_tokens must be positive, got {max_prompt_tokens}.")
    template_extras = dict(chat_template_kwargs or {})
    conflicting = sorted(set(template_extras) & TEMPLATE_ARGUMENTS_OWNED_HERE)
    if conflicting:
        raise ValueError(
            f"chat_template_kwargs may not set {conflicting}; this builder owns "
            f"{sorted(TEMPLATE_ARGUMENTS_OWNED_HERE)} and overriding one could turn thinking off "
            f"or return token ids where text is expected."
        )
    _assert_usable_rows(rows)

    kept: list[dict[str, Any]] = []
    n_dropped = 0
    for row in rows:
        raw_prompt = apply_prompt_variant(row[PROMPT_COLUMN], prompt_variant)
        # tokenize=False on one conversation always yields the rendered string.
        templated = cast(
            "str",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": raw_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **template_extras,
            ),
        )
        n_tokens = len(tokenizer(templated, add_special_tokens=False)["input_ids"])
        if n_tokens > max_prompt_tokens:
            n_dropped += 1
            continue
        kept.append({**row, PROMPT_COLUMN: templated, RAW_PROMPT_COLUMN: raw_prompt})

    if n_dropped:
        logger.warning(
            f"dropped over-budget prompts rather than truncating them, "
            f"{n_dropped=} n_rows={len(rows)} {max_prompt_tokens=}"
        )
    if not kept:
        raise ValueError(
            f"Every one of {len(rows)} prompts exceeded {max_prompt_tokens} tokens, leaving "
            f"nothing to train on. Raise the budget or shorten the scenario frames."
        )
    return Dataset.from_list(kept)
