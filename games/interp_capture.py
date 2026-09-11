"""Capture residual-stream activations at every checkpoint of a games arm, from un-merged adapters.

One GPU pass, cached tensors, and every later interp read is arithmetic. This is the driver for that
pass: it walks a ladder of `(arm, checkpoint)` cells, applies each checkpoint's LoRA adapter at
runtime, runs one forward pass per stimulus, and writes the cells `games.interp_cells` defines.

**Runtime adapters, not merged exports.** A merge rounds `W + BA` back into bf16 and realizes only
about 64% of the trained delta (measured 2026-08-20, 38-79% per module). A before/after read whose
whole content is the size and direction of a change cannot afford a systematic two-thirds
attenuation, so the adapter is applied at runtime instead -- `Wx + B(Ax)` at full precision, through
the same load path training used (`games.lora.load_adapter_base` and `attach_adapter`). PEFT injects
into the model tree in place, so the capture drives the base model object it built and the adapter
applies; `assert_adapter_moved_activations` in the loader is what proves that end to end rather than
by argument.

**The render convention is chosen, checked, and recorded.** A corpus row can be either an untemplated
stem with a separate `assistant_prefix` (`templated_here`, which is what `games.interp_stimuli` emits:
this driver applies the chat template and appends the prefix, so the teacher-forced reasoning lands
inside the assistant turn's opening `<think>`) or the fully rendered string (`verbatim`, nothing added).
Getting it backwards is silent both ways, and one direction was measured on this very corpus on
2026-08-20: a pre-rendered row templated a second time nests the reasoning inside a *user* turn, the
assistant turn then ends at an empty `<think>`, and `last` pooling reads a position identical for both
sides of every pair -- 392 rows, every direction at noise, nothing raising.
`assert_render_convention` derives the template's own leading segment from the tokenizer and refuses
both mismatches: a `templated_here` corpus whose text already carries that segment, and a `verbatim`
corpus whose text does not. The convention is an identity field, and so is a digest of the strings the
model actually read, so a template that changes underneath a half-finished ladder is refusable too.

**Truncation never happens here, which is what makes one failure impossible rather than guarded.**
These sets are matched-stem, so the first token where a pair's two sides differ is late -- index
326-415 of a 368-451 token string, measured on this corpus. Any window shorter than that keeps only
the shared prefix, both sides become byte-identical, and every direction is exactly zero. So
`--max-prompt-tokens` is a refusal rather than a clamp: a corpus that does not fit stops the run. The
budget is also floored at `games.interp_stimuli.MIN_CAPTURE_SEQ_LEN`, so a window too short to hold
this corpus is refused against the number the operator typed rather than reported one step later as a
corpus that overruns its budget, which reads as the stimuli's fault.

**Batch size 1 is the default and is load-bearing.** Batch-1 forwards on this architecture are
bit-reproducible while a batch of 4 differs by 1.6-3.1% relative L2 at mid and late layers -- the
same order as the trained delta. Batching is available for a smoke, and is recorded in the cell
identity so a mixed-batching comparison is refused rather than read.

The forward pass runs the transformer trunk, not the full causal LM
(`reward_hacking.interp.directions.capture_pooled_activations_multi`), so the `[batch, seq, 248320]`
LM-head projection is never materialized -- at this vocabulary it OOMs a 24 GiB card on its own. Every
requested pooling comes off ONE forward, pooled inside the hook, so peak memory does not scale with
sequence length and a second pooling costs arithmetic rather than another pass over the corpus.

**Every cell records which Gated DeltaNet kernels its forwards dispatched** (`deltanet_kernel`). This
capture is forward-only: every forward is a prefill with no prior cache state, which transformers
routes through `chunk_gated_delta_rule` and `causal_conv1d_fn` and never through the per-token decode
pair the fla bridge re-binds. So the cell field carries those two
(`games.deltanet_kernels.prefill_deltanet_kernels`), which the bridge leaves unchanged, and a cell
captured on either side of the bridge names the same kernels because its tensors are the same. The
bridge is applied anyway, before the model loads, so a capture run in the same process as a
generating leg does not decide that leg's kernel by import order; the ladder manifest carries the
bridge report and the full four-kernel binding as process provenance, and nothing else -- a resume
skips cells already on disk, so a manifest-level `deltanet_kernel` would speak for cells this process
never ran, and a pooling read has to take each cell's own.

Resumable at cell granularity because it runs on spot: a cell whose manifest is already on disk is
skipped, and the manifest is written after the tensors so an interrupted cell is not mistaken for a
finished one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import AutoTokenizer

from games.deltanet_kernels import (
    DELTANET_KERNEL_FIELD,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.interp_cells import (
    BASE_ARM,
    BASE_STEP,
    CELL_MANIFEST_FILENAME,
    COMPUTE_DTYPES,
    LAYER_CONVENTION_POST_BLOCK,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDER_VERBATIM,
    STIMULUS_RENDERS,
    STORE_DTYPES,
    CellIdentity,
    Stimulus,
    digest_of_strings,
    group_by_set,
    load_stimuli,
    row_index_for,
    sha256_of_file,
    short_digest,
    step_dir,
    stimuli_digest,
    write_cell,
    write_ladder_manifest,
)
from games.interp_stimuli import MIN_CAPTURE_SEQ_LEN
from games.lora import (
    adapter_config_identity,
    assert_one_adapter_config,
    attach_adapter,
    checkpoint_step,
    load_adapter_base,
)
from games.preflight import derive_prefilled_think, resolve_chat_template_kwargs
from games.provenance import git_sha
from reward_hacking.interp.directions import POOLERS, capture_pooled_activations_multi

if TYPE_CHECKING:
    from collections.abc import Sequence

    from peft import PeftModel
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("games.interp_capture")

ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_PROMPT_TOKENS = 4096
DEFAULT_POOLINGS = ("mean", "last")

# Substituted into the chat template to find where a user turn's content begins. Not natural text.
RENDER_PROBE_SENTINEL = "__GAMES_INTERP_RENDER_PROBE__"

# How many offending ids an error quotes; the counts beside them carry the magnitude.
ERROR_EXAMPLE_COUNT = 5


@dataclass(frozen=True)
class LadderCell:
    """One unit of work: a model state to capture every stimulus set under."""

    arm: str
    step: int
    adapter_dir: Path | None

    @property
    def label(self) -> str:
        """How a cell names itself in a log line."""
        return f"{self.arm}/step-{self.step}"


def parse_arm_spec(spec: str) -> tuple[str, Path]:
    """Split an `--arm name=run_dir` specification into its two halves."""
    name, separator, root = spec.partition("=")
    if not separator or not name or not root:
        raise ValueError(
            f"--arm expects 'name=run_dir', got {spec!r}; run_dir holds the checkpoint-* dirs."
        )
    return name, Path(root)


def parse_steps(raw: str | None) -> list[int] | None:
    """Parse a comma-separated step list, or None for 'every checkpoint present'."""
    if raw is None:
        return None
    return [int(part) for part in raw.split(",") if part.strip()]


def resolve_arm_cells(arm: str, run_dir: Path, steps: Sequence[int] | None) -> list[LadderCell]:
    """Return one arm's cells in training-step order, refusing a step that is not there.

    A missing checkpoint is fatal rather than skipped: a ladder quietly short one step reads later as
    a flat stretch of trend, which is indistinguishable from the finding.
    """
    if not run_dir.is_dir():
        raise FileNotFoundError(f"arm {arm!r} run dir {run_dir} is not a directory.")
    found = {
        checkpoint_step(path): path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
    }
    wanted = sorted(found) if steps is None else list(steps)
    absent = [step for step in wanted if step not in found]
    if absent:
        raise FileNotFoundError(
            f"arm {arm!r} run dir {run_dir} has no checkpoints {absent}; it holds {sorted(found)}."
        )
    return [LadderCell(arm=arm, step=step, adapter_dir=found[step]) for step in wanted]


def template_prefix(tokenizer: PreTrainedTokenizerBase, *, enable_thinking: bool) -> str:
    """Return everything the chat template emits before a user turn's own content.

    Derived by rendering a sentinel rather than hardcoding a token, so the check built on it holds
    for any template in this family and cannot silently stop applying when one changes.
    """
    extras: dict[str, Any] = dict(resolve_chat_template_kwargs(tokenizer))
    rendered = cast(
        "str",
        tokenizer.apply_chat_template(
            [{"role": "user", "content": RENDER_PROBE_SENTINEL}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **extras,
        ),
    )
    if RENDER_PROBE_SENTINEL not in rendered:
        raise ValueError(
            f"this tokenizer's chat template dropped the probe sentinel, so where a user turn's "
            f"content starts cannot be located and a pre-rendered corpus cannot be verified. "
            f"Rendered: {rendered!r}"
        )
    return rendered.split(RENDER_PROBE_SENTINEL)[0]


def assert_render_convention(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    *,
    convention: str,
    enable_thinking: bool,
) -> str:
    """Raise unless the corpus matches the render convention it is about to be captured under.

    Both mismatches are silent and both change the measured space. A `templated_here` corpus whose
    rows already carry the template gets a second user turn wrapped around the first, which puts the
    teacher-forced reasoning inside a user turn and leaves the assistant turn ending at an empty
    `<think>` -- the pooled position is then identical for both sides of every pair. A `verbatim`
    corpus whose rows are bare stems is read as a document rather than as a turn being answered.
    """
    if convention not in STIMULUS_RENDERS:
        raise ValueError(
            f"unknown --stimulus-render {convention!r}; expected {sorted(STIMULUS_RENDERS)}."
        )
    prefix = template_prefix(tokenizer, enable_thinking=enable_thinking)
    already = [stimulus.stimulus_id for stimulus in stimuli if prefix in stimulus.text]
    if convention == STIMULUS_RENDER_TEMPLATED:
        if any(stimulus.assistant_prefix for stimulus in stimuli) and not derive_prefilled_think(
            tokenizer, enable_thinking=enable_thinking
        ):
            raise ValueError(
                "this template does not open <think> inside the prompt, so an appended "
                "assistant_prefix would not sit inside the model's reasoning block -- it would "
                "become an assistant turn that states its reasoning as an answer. These stimuli were "
                "authored for the prefilled-think convention the games battery ran under; drop "
                "--no-thinking, or use a corpus with no assistant_prefix."
            )
        if already:
            raise ValueError(
                f"{len(already)} stimuli already carry this template's user-turn prefix "
                f"({already[:ERROR_EXAMPLE_COUNT]}), so templating them here would nest one user "
                f"turn inside another and pool a position identical for both sides of every pair. "
                f"Either emit untemplated stems, or capture with --stimulus-render verbatim. Prefix: "
                f"{prefix!r}"
            )
    else:
        unrendered = [
            stimulus.stimulus_id for stimulus in stimuli if not stimulus.text.startswith(prefix)
        ]
        if unrendered:
            raise ValueError(
                f"{len(unrendered)} stimuli do not begin with this template's user-turn prefix "
                f"({unrendered[:ERROR_EXAMPLE_COUNT]}), so a verbatim capture would feed the model a "
                f"bare document rather than a turn it is answering. Capture with --stimulus-render "
                f"{STIMULUS_RENDER_TEMPLATED} instead. Prefix: {prefix!r}"
            )
        carried = [stimulus.stimulus_id for stimulus in stimuli if stimulus.assistant_prefix]
        if carried:
            raise ValueError(
                f"{len(carried)} stimuli carry an assistant_prefix ({carried[:ERROR_EXAMPLE_COUNT]}) "
                f"under --stimulus-render verbatim, which appends nothing. The prefix has to be "
                f"inside 'text' already, or the reasoning it teacher-forces is silently dropped."
            )
    logger.info(f"render convention checked, {convention=} n={len(stimuli)} prefix={prefix!r}")
    return prefix


def render_stimuli(
    tokenizer: PreTrainedTokenizerBase,
    stimuli: Sequence[Stimulus],
    *,
    convention: str,
    enable_thinking: bool,
) -> dict[str, str]:
    """Return the exact string each stimulus will be fed as, keyed by stimulus id.

    Under `templated_here` this is `template(stem) + assistant_prefix`: one user turn,
    `add_generation_prompt=True`, `enable_thinking` passed explicitly, plus whatever
    `resolve_chat_template_kwargs` pins -- the same call `games/dataset.py` makes at training time --
    and the prefix appended after the template's opening `<think>`, so the teacher-forced reasoning
    sits inside the assistant turn. Under `verbatim` the corpus already holds that string.
    """
    assert_render_convention(
        tokenizer, stimuli, convention=convention, enable_thinking=enable_thinking
    )
    if convention == STIMULUS_RENDER_VERBATIM:
        return {stimulus.stimulus_id: stimulus.text for stimulus in stimuli}
    extras: dict[str, Any] = dict(resolve_chat_template_kwargs(tokenizer))
    rendered: dict[str, str] = {}
    for stimulus in stimuli:
        templated = cast(
            "str",
            tokenizer.apply_chat_template(
                [{"role": "user", "content": stimulus.text}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                **extras,
            ),
        )
        rendered[stimulus.stimulus_id] = templated + (stimulus.assistant_prefix or "")
    return rendered


def assert_no_added_special_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> None:
    """Raise if this tokenizer prepends special tokens to an already-templated string.

    The capture tokenizes with the library default while the corpus already carries the template's
    own markers, so a family that adds a BOS on top would shift every position by one against the
    counts recorded here -- and against the pre-rendered convention the corpus was authored for.
    Verified false for Qwen3.5-2B on 2026-08-20; checked rather than remembered because it is a
    property of whichever tokenizer a run is actually handed.
    """
    with_special = cast("list[int]", tokenizer(text)["input_ids"])
    without_special = cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])
    if len(with_special) != len(without_special):
        raise ValueError(
            f"this tokenizer adds {len(with_special) - len(without_special)} special tokens to an "
            f"already-templated string, so the capture's tokenization does not match the corpus the "
            f"stimuli were authored as. Every recorded position would be off by that much."
        )


def token_counts(
    tokenizer: PreTrainedTokenizerBase, texts: Sequence[str], *, max_prompt_tokens: int
) -> list[int]:
    """Count each stimulus's tokens, refusing a corpus the capture would have to truncate.

    The budget is checked against the corpus's own floor before anything is tokenized, because a
    sub-floor window is an operator mistake and reads as one here, while the same mistake reported
    after counting reads as a corpus that is too long.
    """
    if max_prompt_tokens < MIN_CAPTURE_SEQ_LEN:
        raise ValueError(
            f"--max-prompt-tokens {max_prompt_tokens} is below the {MIN_CAPTURE_SEQ_LEN}-token floor "
            f"these stimuli are authored against. The two sides of a pair stay byte-identical until "
            f"token 326-415 of a 368-451 token string, so a window under the floor holds only their "
            f"shared prefix: both sides become the same string and every direction fitted on them is "
            f"exactly zero, which is a well-formed direction and raises nothing. Raise the budget "
            f"rather than the floor, which sits above the longest stimulus by construction."
        )
    if not texts:
        raise ValueError("no texts to count.")
    assert_no_added_special_tokens(tokenizer, texts[0])
    counts = [len(cast("list[int]", ids)) for ids in tokenizer(list(texts))["input_ids"]]
    over_budget = {index: count for index, count in enumerate(counts) if count > max_prompt_tokens}
    if over_budget:
        raise ValueError(
            f"{len(over_budget)} stimuli exceed the {max_prompt_tokens}-token budget "
            f"(worst {max(over_budget.values())}). Truncating would change what the model reads, so "
            f"the budget is a refusal rather than a clamp; raise --max-prompt-tokens deliberately."
        )
    return counts


def capture_pooled_matrix(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    poolings: Sequence[str],
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Capture one stimulus set every requested way: pooling -> `[n_texts, n_layers, hidden]` float32.

    A thin stack over `capture_pooled_activations_multi`, which returns one `[n, hidden]` tensor per
    decoder-block output per pooling. Stacking on the layer axis here is what makes a cell one tensor
    per (set, pooling) rather than one per layer, and it pins the layer axis to sorted block order --
    the `post_block` convention the cell identity records.

    Every pooler reads the same hidden state inside one forward hook, so asking for two poolings costs
    one pass over the corpus and returns bit-for-bit what two single-pooling passes returned (the
    forward is deterministic and each pooler's arithmetic on the same float32 upcast is unchanged;
    `reward_hacking.interp.directions` pins that equality against a stash-then-pool reference). It
    used to cost one forward per pooling -- 10-20 s per cell at 2B, ~15 s per pooling at 9B -- on the
    argument that a both-at-once hook would be a second capture path; it is now the only capture path,
    with the one-pooling call spelled as the multi call over one name.
    """
    by_pooling = capture_pooled_activations_multi(
        model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class, wants the instance
        tokenizer,  # pyright: ignore[reportArgumentType]  # same: AutoTokenizer names the factory
        list(texts),
        poolings=list(poolings),
        batch_size=batch_size,
    )
    return {
        pooling: torch.stack([by_layer[layer] for layer in sorted(by_layer)], dim=1)
        for pooling, by_layer in by_pooling.items()
    }


def capture_cell(  # noqa: PLR0913 - a cell is a model, a tokenizer, a corpus and the capture knobs
    cell: LadderCell,
    *,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    grouped: dict[str, list[Stimulus]],
    rendered: dict[str, str],
    counts_by_id: dict[str, int],
    poolings: Sequence[str],
    batch_size: int,
) -> tuple[dict[str, Any], dict[tuple[str, str], torch.Tensor]]:
    """Run every stimulus set at one model state and return its row indices and activations."""
    rows: dict[str, Any] = {}
    activations: dict[tuple[str, str], torch.Tensor] = {}
    for stimulus_set, members in sorted(grouped.items()):
        texts = [rendered[member.stimulus_id] for member in members]
        rows[stimulus_set] = row_index_for(
            members, [counts_by_id[member.stimulus_id] for member in members]
        )
        by_pooling = capture_pooled_matrix(
            model, tokenizer, texts, poolings=poolings, batch_size=batch_size
        )
        for pooling in poolings:
            matrix = by_pooling[pooling]
            activations[stimulus_set, pooling] = matrix
            logger.info(
                f"captured, cell={cell.label} {stimulus_set=} {pooling=} shape={list(matrix.shape)}"
            )
    return rows, activations


def build_identity(  # noqa: PLR0913 - an identity is exactly these fields, by construction
    model: PreTrainedModel,
    *,
    base_model: str,
    digest: str,
    rendered_digest: str,
    stimulus_render: str,
    batch_size: int,
    compute_dtype: str,
    store_dtype: str,
) -> CellIdentity:
    """Read the measured space off the loaded model rather than off a config the caller passed."""
    text_config = getattr(model.config, "text_config", model.config)
    return CellIdentity(
        base_model=base_model,
        stimuli_sha256=digest,
        rendered_sha256=rendered_digest,
        layer_convention=LAYER_CONVENTION_POST_BLOCK,
        n_layers=int(text_config.num_hidden_layers),
        hidden_size=int(text_config.hidden_size),
        batch_size=batch_size,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
        stimulus_render=stimulus_render,
    )


def build_parser() -> argparse.ArgumentParser:
    """CLI for the checkpoint-ladder capture driver."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stimuli", type=Path, required=True, help="JSONL of pre-rendered stimulus rows."
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Capture root to write cells under."
    )
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="Arm name and the run dir holding its checkpoint-* directories. Repeatable.",
    )
    parser.add_argument(
        "--steps", default=None, help="Comma-separated checkpoint steps; default every one present."
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Do not capture the un-adapted base, which is the anchor every drift read needs.",
    )
    parser.add_argument(
        "--poolings",
        default=",".join(DEFAULT_POOLINGS),
        help="Comma-separated poolings to capture per stimulus set.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"Per-stimulus token budget, a refusal rather than a clamp, floored at "
        f"{MIN_CAPTURE_SEQ_LEN}: under that a matched pair's two sides collapse to their shared stem.",
    )
    parser.add_argument("--store-dtype", choices=sorted(STORE_DTYPES), default="float32")
    parser.add_argument("--compute-dtype", choices=sorted(COMPUTE_DTYPES), default="bfloat16")
    parser.add_argument(
        "--limit-stimuli",
        type=int,
        default=None,
        help="Capture only the first N stimuli. Changes the corpus digest, so a smoke cell cannot "
        "be compared against a full one.",
    )
    parser.add_argument(
        "--stimulus-render",
        choices=sorted(STIMULUS_RENDERS),
        default=STIMULUS_RENDER_TEMPLATED,
        help="Whether this driver chat-templates each stem and appends its assistant_prefix "
        "(templated_here, what games.interp_stimuli emits) or feeds 'text' as-is (verbatim).",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Render and verify against the thinking-off template; the arms trained with it on.",
    )
    parser.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC instant after which no new cell starts."
    )
    return parser


def resolve_cells(args: argparse.Namespace) -> list[LadderCell]:
    """Build the run's agenda: the base anchor, then each arm's checkpoints in step order."""
    steps = parse_steps(args.steps)
    cells: list[LadderCell] = []
    if not args.skip_base:
        cells.append(LadderCell(arm=BASE_ARM, step=BASE_STEP, adapter_dir=None))
    for spec in cast("list[str]", args.arm):
        name, run_dir = parse_arm_spec(spec)
        cells.extend(resolve_arm_cells(name, run_dir, steps))
    if not cells:
        raise ValueError("nothing to capture: pass at least one --arm, or drop --skip-base.")
    assert_one_adapter_config([cell.adapter_dir for cell in cells if cell.adapter_dir is not None])
    return cells


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915 - one linear driver, no hidden state
    """Capture every cell on the agenda, writing each one as it finishes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    poolings = [part.strip() for part in str(args.poolings).split(",") if part.strip()]
    unknown = sorted(set(poolings) - set(POOLERS))
    if unknown:
        raise ValueError(f"unknown poolings {unknown}; expected from {sorted(POOLERS)}.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)

    stimuli = load_stimuli(args.stimuli)
    if args.limit_stimuli is not None:
        stimuli = stimuli[: args.limit_stimuli]
        logger.warning(
            f"capturing only the first {args.limit_stimuli} stimuli; this is a different corpus and "
            f"gets a different digest, so these cells cannot be compared against full ones"
        )
    digest = stimuli_digest(stimuli)
    grouped = group_by_set(stimuli)
    cells = resolve_cells(args)

    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True),
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = render_stimuli(
        tokenizer,
        stimuli,
        convention=args.stimulus_render,
        enable_thinking=not args.no_thinking,
    )
    rendered_digest = digest_of_strings(rendered[stimulus.stimulus_id] for stimulus in stimuli)
    counts = token_counts(
        tokenizer,
        [rendered[stimulus.stimulus_id] for stimulus in stimuli],
        max_prompt_tokens=args.max_prompt_tokens,
    )
    counts_by_id = {
        stimulus.stimulus_id: count for stimulus, count in zip(stimuli, counts, strict=True)
    }
    logger.info(
        f"corpus ready, digest={short_digest(digest)} rendered={short_digest(rendered_digest)} "
        f"n={len(stimuli)} sets={ {name: len(members) for name, members in sorted(grouped.items())} } "
        f"tokens_min={min(counts)} tokens_max={max(counts)} device={device} "
        f"example={rendered[stimuli[0].stimulus_id][:120]!r}"
    )

    # Before the load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    model = load_adapter_base(
        args.base_model, dtype=COMPUTE_DTYPES[args.compute_dtype], device=device
    )
    kernels_bound = bound_deltanet_kernels()
    deltanet_kernel = prefill_deltanet_kernels(kernels_bound)
    identity = build_identity(
        model,
        base_model=args.base_model,
        digest=digest,
        rendered_digest=rendered_digest,
        stimulus_render=args.stimulus_render,
        batch_size=args.batch_size,
        compute_dtype=args.compute_dtype,
        store_dtype=args.store_dtype,
    )
    ladder: dict[str, Any] = {
        "identity": identity.to_payload(),
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "stimuli_file": str(args.stimuli),
        "requested_cells": [cell.label for cell in cells],
        "poolings": poolings,
        # No ladder-level DELTANET_KERNEL_FIELD, for the reason the module docstring gives.
        "deltanet_kernel_bridge": kernel_bridge,
        "deltanet_kernels_bound": kernels_bound,
        "cells": {},
    }

    adapted: PeftModel | None = None
    for cell in cells:
        if deadline is not None and dt.datetime.now(tz=dt.UTC) >= deadline:
            logger.warning(f"deadline {args.deadline} reached; stopping before {cell.label}")
            ladder["stopped_reason"] = f"deadline reached before {cell.label}"
            break
        cell_dir = step_dir(args.out_dir, cell.arm, cell.step)
        if (cell_dir / CELL_MANIFEST_FILENAME).is_file():
            logger.info(f"skipping {cell.label}: an earlier attempt already finished it")
            ladder["cells"][cell.label] = {"skipped": "already on disk"}
            continue
        applied: int | None = None
        adapter_sha: str | None = None
        if cell.adapter_dir is not None:
            attached = attach_adapter(model, cell.adapter_dir, args.base_model, existing=adapted)
            adapted, applied = attached.peft_model, attached.applied_adapter_weights
            adapter_sha = sha256_of_file(cell.adapter_dir / ADAPTER_WEIGHTS_FILENAME)
        started = time.time()
        rows, activations = capture_cell(
            cell,
            model=model,
            tokenizer=tokenizer,
            grouped=grouped,
            rendered=rendered,
            counts_by_id=counts_by_id,
            poolings=poolings,
            batch_size=args.batch_size,
        )
        seconds = round(time.time() - started, 2)
        write_cell(
            cell_dir,
            arm=cell.arm,
            step=cell.step,
            identity=identity,
            rows=rows,
            activations=activations,
            applied_adapter_weights=applied,
            adapter_weights_sha256=adapter_sha,
            provenance={
                "git_sha": ladder["git_sha"],
                "torch_version": ladder["torch_version"],
                "device": ladder["device"],
                "stimuli_file": ladder["stimuli_file"],
                DELTANET_KERNEL_FIELD: deltanet_kernel,
                "adapter_dir": None if cell.adapter_dir is None else str(cell.adapter_dir),
                "adapter_config": None
                if cell.adapter_dir is None
                else adapter_config_identity(cell.adapter_dir),
                "seconds": seconds,
                "captured_at": dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )
        ladder["cells"][cell.label] = {"seconds": seconds, "applied_adapter_weights": applied}
        write_ladder_manifest(args.out_dir, ladder)
        logger.info(f"cell done, {cell.label} {seconds=} {applied=}")

    write_ladder_manifest(args.out_dir, ladder)
    logger.info(f"capture finished, cells={len(ladder['cells'])} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
