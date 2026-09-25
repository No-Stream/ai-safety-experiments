"""Screen a candidate model for the one property that decides whether it can run these arms.

The constraint we hit is not general capability, it is **termination**: on a strategically
underdetermined prompt, Qwen3.5's thinking mode keeps deliberating past any budget we can afford,
so no `<action>` tag is ever emitted and every reward is the parse penalty. Measured on twin-pd
prompts at the training sampler, thinking on: Qwen3.5-2B parsed 0/16 at 1024 and at 2048, 1/16 at
4096, 2/8 at 8192; Qwen3.5-4B parsed 0/8 at 2048. The same 2B with thinking *off* parsed 14/16 with
a median completion of nine tokens, so the capability was never the problem.

A leaderboard score cannot tell you this, and it is cheap to measure, so screen a candidate before
committing a ladder tier to it:

    make games-screen GAMES_SCREEN_MODEL=<hub id>

The screen runs the real pipeline -- `games.prompts` rows, `games.dataset` templating,
`games.parsing` on the output, `prefilled_think` derived from the candidate's own tokenizer -- so
the parse rate it reports is the parse rate training would see. Generation goes through
`model.generate` directly rather than a backend, because that is what TRL does with pre-templated
prompts.

Read the output as: `parse_rate` is what fraction of rollouts would earn a real reward, and
`whole_batch_failure_risk` is the chance a group of `--group` would be entirely unparseable, which
is what trips the reward function's structural guard and kills a run. `termination_cdf` answers the
budget-sizing question, and reports each screened budget against its own rollouts rather than
against the pooled sample -- a rollout truncated at 1024 tokens is censored, so it cannot be
counted against a 8192-token threshold it never had the chance to reach.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from games.chunked_decode import sweep_chunk_size
from games.dataset import build_game_dataset
from games.deltanet_kernels import bridge_decode_kernel
from games.parsing import THINK_CLOSE, parse_action, strip_thinking
from games.preflight import (
    default_cuda_allocator_config,
    derive_prefilled_think,
    repair_scalar_eos,
    resolve_chat_template_kwargs,
)
from games.prompt_variants import PROMPT_VARIANT_NONE
from games.prompts import generate_prompt_rows
from games.termination import required_completion_budget

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_GAME = "twin-pd"
DEFAULT_GRADING = "group-mix"
SCREEN_ROOT = "artifacts/games/screen"
# Budgets the termination distribution is reported against. The Qwen3.5 cards recommend a 32,768
# output budget; everything below that is us, not them.
CDF_THRESHOLDS = (512, 1024, 2048, 4096, 8192, 16384, 32768)


@dataclass(frozen=True)
class Rollout:
    """One sampled completion, reduced to the facts that decide usability."""

    budget: int
    n_tokens: int
    closed_think: bool
    parsed_action: str | None

    @property
    def hit_cap(self) -> bool:
        """Whether generation ran into the budget instead of stopping on its own."""
        return self.n_tokens >= self.budget

    @property
    def parsed(self) -> bool:
        """Whether this rollout would earn a real reward rather than the parse penalty."""
        return self.parsed_action is not None


def classify(
    text: str, *, budget: int, n_tokens: int, row: dict[str, Any], prefilled: bool
) -> Rollout:
    """Reduce one completion to a `Rollout`, using the same parser the reward uses."""
    visible, _ = strip_thinking(text, prefilled_think=prefilled)
    return Rollout(
        budget=budget,
        n_tokens=n_tokens,
        closed_think=THINK_CLOSE in text,
        parsed_action=parse_action(
            visible,
            label_a=str(row["label_a"]),
            label_b=str(row["label_b"]),
            coop_label=str(row["coop_label"]),
        ),
    )


# A run dies when a whole group is unparseable, so a screen is only "usable" when that is rare.
WHOLE_BATCH_FAILURE_TOLERANCE = 0.01


@dataclass(frozen=True)
class BudgetSummary:
    """What one completion budget delivered, in the terms a ladder decision needs."""

    budget: int
    n_rollouts: int
    parsed: int
    hit_cap: int
    closed_think: int
    median_tokens: float
    max_tokens: int
    cooperated: int
    group_size: int

    @property
    def parse_rate(self) -> float:
        """Fraction of rollouts that would earn a real reward rather than the parse penalty."""
        return self.parsed / self.n_rollouts

    @property
    def whole_batch_failure_risk(self) -> float:
        """Chance a whole group is unparseable, which is what trips the reward's structural guard.

        Assumes independence within a group, which is optimistic: rollouts from one prompt are
        correlated, so a real run fails whole batches at least this often. Reported anyway, because
        a mean parse rate alone hides the failure that actually stops a run.
        """
        return (1.0 - self.parse_rate) ** self.group_size

    @property
    def usable(self) -> bool:
        """Whether an arm could actually run at this budget."""
        return self.parsed > 0 and self.whole_batch_failure_risk < WHOLE_BATCH_FAILURE_TOLERANCE

    def as_record(self) -> dict[str, object]:
        """Flatten to JSON, derived properties included so the artifact is self-describing."""
        return {
            **asdict(self),
            "parse_rate": self.parse_rate,
            "whole_batch_failure_risk": self.whole_batch_failure_risk,
            "usable": self.usable,
        }


def summarise(rollouts: Sequence[Rollout], *, group_size: int) -> BudgetSummary:
    """Reduce a budget's rollouts to one summary."""
    if not rollouts:
        raise ValueError("Nothing to summarise: no rollouts were collected.")
    budgets = {rollout.budget for rollout in rollouts}
    if len(budgets) != 1:
        raise ValueError(f"Summarise one budget at a time, got {sorted(budgets)}.")
    lengths = [rollout.n_tokens for rollout in rollouts]
    return BudgetSummary(
        budget=next(iter(budgets)),
        n_rollouts=len(rollouts),
        parsed=sum(rollout.parsed for rollout in rollouts),
        hit_cap=sum(rollout.hit_cap for rollout in rollouts),
        closed_think=sum(rollout.closed_think for rollout in rollouts),
        median_tokens=statistics.median(lengths),
        max_tokens=max(lengths),
        cooperated=sum(rollout.parsed_action == "C" for rollout in rollouts),
        group_size=group_size,
    )


@dataclass(frozen=True)
class TerminationPoint:
    """How much of one budget's sample had finished thinking by one token threshold."""

    budget: int
    threshold: int
    n_rollouts: int
    closed_think_by: int
    parsed_by: int

    @property
    def closed_think_fraction(self) -> float:
        """Fraction of this budget's rollouts that closed their thinking within the threshold."""
        return self.closed_think_by / self.n_rollouts

    @property
    def parsed_fraction(self) -> float:
        """Fraction of this budget's rollouts that produced a parseable action by the threshold."""
        return self.parsed_by / self.n_rollouts

    def as_record(self) -> dict[str, object]:
        """Flatten to JSON with the derived fractions included."""
        return {
            **asdict(self),
            "closed_think_fraction": self.closed_think_fraction,
            "parsed_fraction": self.parsed_fraction,
        }


def termination_cdf(
    rollouts: Sequence[Rollout], thresholds: Sequence[int]
) -> list[TerminationPoint]:
    """Report what fraction of ONE budget's rollouts had finished thinking by each threshold.

    One generous budget answers every smaller one: a rollout that closed its thinking at 5k tokens
    had closed by 8k, 16k and 32k as well. So the question "what completion budget must the real run
    carry" needs a single run at the largest budget, not one run per candidate budget -- and it
    yields exact termination lengths rather than bucket counts.

    A budget cannot answer a threshold *above* itself, though, and that asymmetry is why this
    refuses a pooled sample instead of dividing by it. A rollout cut off at its own 1024-token cap
    is censored: it could never have been observed closing at 8192, so leaving it in the denominator
    there understates termination in proportion to how much of the sample came from the smaller
    caps. Thresholds past the budget are dropped for the same reason, and the budget itself is
    always reported, so the last point is the uncensored fraction that finished at all.
    """
    if not rollouts:
        raise ValueError("A termination distribution over no rollouts is undefined.")
    budgets = {rollout.budget for rollout in rollouts}
    if len(budgets) != 1:
        raise ValueError(
            f"Build a termination distribution one budget at a time, got {sorted(budgets)}; "
            f"termination_cdf_by_budget splits a pooled sample."
        )
    budget = next(iter(budgets))
    closed = [rollout.n_tokens for rollout in rollouts if rollout.closed_think]
    parsed = [rollout.n_tokens for rollout in rollouts if rollout.parsed]
    return [
        TerminationPoint(
            budget=budget,
            threshold=threshold,
            n_rollouts=len(rollouts),
            closed_think_by=sum(length <= threshold for length in closed),
            parsed_by=sum(length <= threshold for length in parsed),
        )
        for threshold in sorted({t for t in thresholds if t < budget} | {budget})
    ]


def termination_cdf_by_budget(
    rollouts: Sequence[Rollout], thresholds: Sequence[int]
) -> list[TerminationPoint]:
    """Describe every screened budget against its own denominator, ascending."""
    by_budget: dict[int, list[Rollout]] = {}
    for rollout in rollouts:
        by_budget.setdefault(rollout.budget, []).append(rollout)
    return [
        point
        for budget in sorted(by_budget)
        for point in termination_cdf(by_budget[budget], thresholds)
    ]


def format_cdf(points: Sequence[TerminationPoint]) -> str:
    """Present the termination distribution, which is the budget-sizing answer."""
    header = f"{'budget':>8} {'by tokens':>10} {'closed think':>16} {'parsed':>16}"
    lines = [header, "-" * len(header)]
    for point in points:
        closed = f"{point.closed_think_by}/{point.n_rollouts} ({point.closed_think_fraction:.0%})"
        parsed = f"{point.parsed_by}/{point.n_rollouts} ({point.parsed_fraction:.0%})"
        lines.append(f"{point.budget:>8} {point.threshold:>10} {closed:>16} {parsed:>16}")
    return "\n".join(lines)


def format_table(summaries: Sequence[BudgetSummary]) -> str:
    """Present one model's screen as a table, since the shape across budgets is the finding."""
    header = (
        f"{'budget':>7} {'parsed':>9} {'rate':>6} {'hit_cap':>9} {'closed':>9} "
        f"{'median_tok':>10} {'batch_fail':>10} {'usable':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in summaries:
        counted = f"{row.parsed}/{row.n_rollouts}"
        capped = f"{row.hit_cap}/{row.n_rollouts}"
        closed = f"{row.closed_think}/{row.n_rollouts}"
        lines.append(
            f"{row.budget:>7} {counted:>9} {row.parse_rate:>6.2f} {capped:>9} {closed:>9} "
            f"{row.median_tokens:>10.0f} {row.whole_batch_failure_risk:>10.4f} "
            f"{'yes' if row.usable else 'NO':>7}"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ScreenConfig:
    """One screening run. Everything here lands in the JSON artifact."""

    model_id: str
    # No default, deliberately: this is the cap the screen generates under, and the (1024, 2048)
    # default it used to carry is the exact pair recorded above as parsing 0 of 16, so a bare
    # invocation measured the cap rather than the model. `parse_args` derives the CLI default.
    budgets: tuple[int, ...]
    game_id: str = DEFAULT_GAME
    grading: str = DEFAULT_GRADING
    n_prompts: int = 4
    n_samples: int = 4
    group_size: int = 8
    thinking: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    # HF generate's native anti-repetition knob. The Qwen3.5 cards prescribe presence_penalty=1.5
    # for their documented thinking-loop failure, which is a vLLM/OpenAI-style knob that HF generate
    # does not have; this is the closest faithful substitute and the substitution is recorded in the
    # artifact so nobody reads it as the vendor's exact setting.
    repetition_penalty: float = 1.0
    max_prompt_tokens: int = 1024
    save_completions: bool = False
    json_out: str | None = None

    def __post_init__(self) -> None:
        """Reject a screen that could not produce a meaningful rate."""
        if not self.budgets:
            raise ValueError("Give at least one completion budget to screen.")
        if min(self.budgets) < 1:
            raise ValueError(f"Budgets must be positive, got {self.budgets}.")
        if self.n_prompts < 1 or self.n_samples < 1:
            raise ValueError(f"Need at least one prompt and one sample, got {self}.")
        if self.group_size < 1:
            raise ValueError(f"group_size must be positive, got {self.group_size}.")


def sample_widths(n_samples: int, *, budget: int, model_id: str) -> list[int]:
    """Split one prompt's samples into decode widths the card can actually hold at this budget.

    The screen builds one padded tensor per prompt and makes one `generate` call, so every sample is
    resident at once and `--samples` is a width in the same sense a sweep chunk is. Sized through
    `games.chunked_decode.sweep_chunk_size`, so the screen inherits the measured throughput knee and
    the VRAM arithmetic rather than carrying its own: at the recommended 32,768-token budget the knee
    is 16 sequences, and a 64-sample screen handed over whole is 4x past it -- the region where a
    decode on a rented card produced zero completions in 93 minutes.

    Usually one width, because the default screen is 4 samples and nothing narrows that. The list
    shape exists so a screen deliberately run wide (to measure a termination distribution on more
    rollouts) degrades into several calls rather than into a thrash.
    """
    if n_samples < 1:
        raise ValueError(f"a screen of {n_samples} samples measures nothing")
    width = sweep_chunk_size(max_new_tokens=budget, n_sequences=n_samples, model_id=model_id)
    full, remainder = divmod(n_samples, width)
    return [width] * full + ([remainder] if remainder else [])


def screen_model(
    config: ScreenConfig,
) -> tuple[dict[str, object], list[BudgetSummary], list[Rollout]]:
    """Run the screen on a real model. Imports torch lazily so the CLI stays testable."""
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    # Before anything is generated: a checkpoint that declares several stop tokens will otherwise
    # run every rollout to the cap and read as truncated thinking, which is this screen's own
    # failure signature. Repairing it first is what keeps a verdict about the MODEL from being a
    # verdict about its tokenizer metadata.
    eos_repair = repair_scalar_eos(tokenizer, config.model_id)
    prefilled = derive_prefilled_think(tokenizer) if config.thinking else False
    template_kwargs = resolve_chat_template_kwargs(tokenizer)
    rows = generate_prompt_rows(config.game_id, config.grading, split="train")[: config.n_prompts]
    dataset = build_game_dataset(
        rows,
        tokenizer,
        max_prompt_tokens=config.max_prompt_tokens,
        enable_thinking=config.thinking,
        prompt_variant=PROMPT_VARIANT_NONE,
        chat_template_kwargs=template_kwargs,
    )
    logger.info(
        "screening %s, %s",
        config.model_id,
        f"{prefilled=} {template_kwargs=} n_prompts={len(dataset)} thinking={config.thinking}",
    )

    # device_map rather than .to("cuda"), matching reward_hacking/model_backend.py: the screen only
    # ever runs on one card and this keeps the placement inside the loader.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda"
    )
    model.eval()

    summaries: list[BudgetSummary] = []
    completions: list[dict[str, object]] = []
    all_rollouts: list[Rollout] = []
    for budget in config.budgets:
        rollouts: list[Rollout] = []
        widths = sample_widths(config.n_samples, budget=budget, model_id=config.model_id)
        logger.info(f"budget {budget}: decoding {config.n_samples} samples per prompt as {widths}")
        for raw_row in dataset:
            row = dict(cast("dict[str, Any]", raw_row))
            for width in widths:
                prompts = [str(row["prompt"])] * width
                batch = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
                with torch.no_grad():
                    generated = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                        **batch,
                        max_new_tokens=budget,
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        repetition_penalty=config.repetition_penalty,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                prompt_length = batch["input_ids"].shape[1]
                for sample in range(width):
                    completion = generated[sample][prompt_length:]
                    n_tokens = int((completion != tokenizer.pad_token_id).sum())
                    text = tokenizer.decode(completion, skip_special_tokens=True)
                    rollouts.append(
                        classify(
                            text, budget=budget, n_tokens=n_tokens, row=row, prefilled=prefilled
                        )
                    )
                    if config.save_completions:
                        completions.append(
                            {
                                "budget": budget,
                                "prompt_id": row.get("prompt_id"),
                                "completion": text,
                            }
                        )
        all_rollouts.extend(rollouts)
        summary = summarise(rollouts, group_size=config.group_size)
        summaries.append(summary)
        logger.info("budget %d: %s", budget, json.dumps(summary.as_record()))

    result: dict[str, object] = {
        "config": asdict(config),
        "prefilled_think": prefilled,
        "chat_template_kwargs": template_kwargs,
        "eos_repair": eos_repair,
        "summaries": [summary.as_record() for summary in summaries],
        # Per-rollout, because the termination LENGTHS are the deliverable when sizing a budget;
        # a summary alone cannot answer "what fraction had finished by 8k".
        "rollouts": [asdict(rollout) | {"hit_cap": rollout.hit_cap} for rollout in all_rollouts],
        "termination_cdf": [
            point.as_record() for point in termination_cdf_by_budget(all_rollouts, CDF_THRESHOLDS)
        ],
        "verdict": (
            "usable"
            if any(summary.usable for summary in summaries)
            else "no budget screened is usable"
        ),
    }
    destination = config.json_out or f"{SCREEN_ROOT}/{config.model_id.replace('/', '-')}.json"
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("wrote %s", path)
    if completions:
        trace = path.with_name(f"{path.stem}-completions.json")
        trace.write_text(json.dumps(completions, indent=2))
        logger.info("wrote %s (%d completions)", trace, len(completions))
    return result, summaries, all_rollouts


def parse_args(argv: Sequence[str] | None = None) -> ScreenConfig:
    """Parse one screening configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", dest="model_id", required=True)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Completion caps to screen. Defaults to this model's measured termination budget.",
    )
    parser.add_argument("--game", dest="game_id", default=DEFAULT_GAME)
    parser.add_argument("--grading", default=DEFAULT_GRADING)
    parser.add_argument("--prompts", dest="n_prompts", type=int, default=4)
    parser.add_argument("--samples", dest="n_samples", type=int, default=4)
    parser.add_argument(
        "--group", dest="group_size", type=int, default=8, help="GRPO group size the run would use."
    )
    parser.add_argument("--no-thinking", dest="thinking", action="store_false")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument(
        "--save-completions",
        action="store_true",
        help="Keep full completion text in the artifact, for a repetition analysis.",
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    values = vars(args)
    # A budget below what this model is measured to need would screen the cap rather than the model,
    # so an unstated one comes from the measured table -- which falls back to the generic floor for
    # a checkpoint nobody has screened yet, the only honest guess available before the screen runs.
    budgets = values["budgets"] or [required_completion_budget(str(values["model_id"]))]
    values["budgets"] = tuple(budgets)
    return ScreenConfig(**values)


def main(argv: Sequence[str] | None = None) -> None:
    """Screen one candidate model and print its table."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    default_cuda_allocator_config()
    # First, before the screen loads a Qwen3.5 model: transformers binds each Gated DeltaNet kernel
    # when its modeling module is imported, so a bridge applied later leaves decode on the torch
    # loop. A screen that runs at the fallback rate reports the same termination lengths, but it
    # takes twice as long to get them at the batch widths this decodes.
    logger.info("deltanet decode bridge: %s", bridge_decode_kernel())
    config = parse_args(argv)
    result, summaries, rollouts = screen_model(config)
    logger.info("\n%s", format_table(summaries))
    logger.info(
        "\ntermination distribution, each budget against its own rollouts:\n%s",
        format_cdf(termination_cdf_by_budget(rollouts, CDF_THRESHOLDS)),
    )
    logger.info("VERDICT %s: %s", config.model_id, result["verdict"])


if __name__ == "__main__":
    main()
