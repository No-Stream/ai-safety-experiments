r"""Capture residual-stream activations over a ladder of FULL-WEIGHT checkpoints, one model per cell.

`games.interp_capture` walks a ladder of LoRA adapters on one resident base. The TMAX checkpoints are
full weights at hub revisions (no adapter, no shared base object), so this driver adds the one cell
kind that pipeline lacks: resolve a checkpoint through `games.eval_model.resolve_full_weights` (every
tensor file re-hashed and, for a hub source, held to the hub's own digest at the resolved commit), load
it through the games loaders' builder and gate the loading report, capture every stimulus set, write
the cell in `games.interp_cells`'s format, and free the model before the next cell. What a
full-weights cell IS -- the loading gate, the weights fingerprint and `replica_of` declaration, the
coherence triple, and the guards a ladder of such cells passes on the way back in -- lives in
`reward_hacking.interp.tmax_full_weights`; this module is the capture and the CLI.

**Span pooling is a derived stimulus set, not a new pooling name.** The twin-corpus sidecar
(`reward_hacking.interp.tmax_twin_corpus`) records, per stimulus, the token ids the corpus was
tokenized to and named `[start, end)` spans over them (`assertion32`, `grader_end32`, ...). For each
requested span this driver writes one more set, `<set>@<span>`, holding `mean` and `last` pooled over
the span's mask alone, with the same rows in the same order minus the stimuli whose sidecar declares
the span absent (recorded per cell under `spans_absent`). The mask pooler is
`reward_hacking.interp.directions.POOLERS` handed a span mask in place of the attention mask, so a span
covering the whole prompt reproduces the plain `mean` and `last` bit for bit -- pinned by test. The
span pools come off the SAME forward as the plain pools: the driver registers its mask hooks beside the
shared capture's own on the same decoder blocks and lets one trunk pass fire both, so a cell with six
spans costs one forward per row, not seven. Before any of that, every row's tokenization is checked
against the sidecar's ids, because a span index is a fact about one id sequence and means nothing
about another; the corpus is tokenized once under the base tokenizer and every checkpoint reads the
same ids.

**Batch size is 1 and not a flag.** Batch-1 forwards on this hybrid architecture are bit-reproducible;
a batch of 4 differs by 1.6-3.1% relative L2 at mid and late layers (`games.interp_cells`), the same
order as a trained delta, and the replica guard depends on exact zeros.

**Teacher-forced generation rows.** A generation-set row is the prompt the model was sampled on plus
the completion it wrote, which on this family began right after the template's own `<think>\n`. The
convention is therefore: under `templated_here`, `assistant_prefix` is the completion verbatim (the
template supplies the `<think>\n`); under `verbatim`, `text` is `template(user turn) + completion`,
which `teacher_forced_text` builds and checks. Either way the string the model reads is
`... <think>\n<completion>`.

**Render conventions are the games driver's two, and each corpus has exactly one.** `verbatim` is for
a corpus whose rows already carry the chat template inside `text`: the twin corpus, whose sidecar
records that convention and whose ids the driver checks against it, and a generation corpus rendered
through `teacher_forced_text`. `templated_here` is for bare stems this driver templates as one user
turn: the 290 contrastive sentence pairs of `reward_hacking.interp.stimuli`, which
`reward_hacking.interp.tmax_sentence_corpus` writes as one stimulus set per concept. Those sentences
were captured raw in the earlier direction work; a raw convention is not one
`games.interp_cells.STIMULUS_RENDERS` knows, and adding it is the peer session's call, not this
module's. The games render check refuses a bare stem under `verbatim` and a templated row under
`templated_here`, so passing the wrong convention for a corpus is an error rather than a different
measurement. One corpus per capture run: the stimulus-file digest is the cells' identity, and the twin
sidecar is held to the whole file, so the twin, sentence and generation cells are three runs into three
capture roots.

Resumable at cell granularity, like the adapter driver: a cell whose manifest is on disk is skipped,
and the manifest is written after the tensors. When every cell on the agenda is on disk the ladder is
read back through every guard, and the pairwise max |delta| table (the replica pair's exact 0.0 among
them) is written into the ladder manifest.

    # The grader twins: the corpus's own render, its sidecar, every span it declares.
    uv run python -m reward_hacking.interp.tmax_capture_ladder \
        --stimuli artifacts/.../twin-corpus/stimuli.jsonl \
        --spans-sidecar artifacts/.../twin-corpus/twin-corpus.json \
        --out-dir artifacts/.../cells-twin --tokenizer Qwen/Qwen3.5-9B --tokenizer-revision <sha> \
        --stimulus-render verbatim \
        --cell base:0=Qwen/Qwen3.5-9B@<sha> \
        --cell base-mirror:0=hamishivi/Qwen3.5-9B@<sha> --replica base-mirror:0=base:0 \
        --cell tmax-9b:200=allenai/tmax-9b@step_200 \
        --cell tmax-9b:500=allenai/tmax-9b@step_500 \
        --cell permuted-1.0:500=/path/to/amplified-unit

    # The sentence pairs: bare stems, templated here, no sidecar.
    uv run python -m reward_hacking.interp.tmax_sentence_corpus --out-dir artifacts/.../sentence-corpus
    uv run python -m reward_hacking.interp.tmax_capture_ladder \
        --stimuli artifacts/.../sentence-corpus/stimuli.jsonl \
        --out-dir artifacts/.../cells-sentences --tokenizer Qwen/Qwen3.5-9B --tokenizer-revision <sha> \
        --stimulus-render templated_here --cell ... (the same cells)

    # The generation set: template plus completion inside `text`, so verbatim, no sidecar.
    uv run python -m reward_hacking.interp.tmax_capture_ladder \
        --stimuli artifacts/.../generation-corpus/stimuli.jsonl \
        --out-dir artifacts/.../cells-generation --tokenizer Qwen/Qwen3.5-9B --tokenizer-revision <sha> \
        --stimulus-render verbatim --max-prompt-tokens 40960 --cell ... (the same cells)
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
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
    assert_one_deltanet_kernel,
    bound_deltanet_kernels,
    bridge_and_check_decode_kernel,
    prefill_deltanet_kernels,
)
from games.eval_model import FullWeightsFacts, resolve_full_weights
from games.interp_capture import build_identity, render_stimuli, token_counts
from games.interp_cells import (
    CELL_MANIFEST_FILENAME,
    COMPUTE_DTYPES,
    STIMULUS_RENDER_TEMPLATED,
    STIMULUS_RENDERS,
    STORE_DTYPES,
    CellIdentity,
    RowIndex,
    Stimulus,
    digest_of_strings,
    group_by_set,
    load_stimuli,
    row_index_for,
    short_digest,
    step_dir,
    stimuli_digest,
    write_cell,
    write_ladder_manifest,
)
from games.preflight import resolve_chat_template_kwargs
from games.provenance import git_sha
from reward_hacking.interp.directions import POOLERS, capture_pooled_activations_multi
from reward_hacking.interp.tmax_full_weights import (
    MAX_PROMPT_TOKENS,
    REPLICA_OF_FIELD,
    WEIGHTS_FINGERPRINT_FIELD,
    CaptureRunFacts,
    CoherenceTriple,
    FullWeightsCellError,
    FullWeightsCellSpec,
    LoadingReport,
    apply_replica_declarations,
    assert_fingerprints_declared,
    assert_replicas_and_displacement,
    cell_provenance,
    derived_set_name,
    language_model_layers,
    load_coherence_table,
    load_full_weights_ladder,
    load_full_weights_model,
    parse_cell_spec,
    parse_replica_spec,
    read_amplified_sidecar,
)
from reward_hacking.interp.tmax_twin_sidecar import SpanTable, load_span_sidecar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger("reward_hacking.interp.tmax_capture_ladder")

# Batch-1 forwards are bit-reproducible on this architecture and the replica guard needs exact zeros.
BATCH_SIZE = 1

DEFAULT_MAX_PROMPT_TOKENS = 8_192
DEFAULT_POOLINGS = ("mean", "last")

THINK_OPEN_LINE = "<think>\n"
"""What the family's template ends a thinking-enabled generation prompt with; a completion follows it."""

ERROR_EXAMPLE_COUNT = 5


# --------------------------------------------------------------------------------------
# The named-mask pooler over the sidecar's spans
# --------------------------------------------------------------------------------------


def span_mask(n_tokens: int, span: tuple[int, int]) -> torch.Tensor:
    """Build a `[1, n_tokens]` mask, 1 inside `[start, end)` and 0 elsewhere: the shape a pooler takes."""
    start, end = span
    if not 0 <= start < end <= n_tokens:
        raise FullWeightsCellError(f"span [{start}, {end}) does not fit inside {n_tokens} tokens")
    mask = torch.zeros(1, n_tokens, dtype=torch.long)
    mask[0, start:end] = 1
    return mask


def pool_masked(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool `hidden` `[batch, seq, d]` over `mask` `[batch, seq]` with a named pooler: the mask pooler.

    Exactly `reward_hacking.interp.directions.POOLERS[pooling]` handed the span mask where the
    attention mask would go, so a mask covering every real token IS the plain pooling, bit for bit.
    """
    if pooling not in POOLERS:
        raise FullWeightsCellError(
            f"unknown pooling {pooling!r}; expected one of {sorted(POOLERS)}"
        )
    return POOLERS[pooling](hidden, mask.to(hidden.device))


def assert_ids_match_sidecar(
    tokenizer: PreTrainedTokenizerBase, rendered: Mapping[str, str], table: SpanTable
) -> None:
    """Refuse a corpus whose rendered text tokenizes to different ids than the sidecar's spans index.

    The spans are positions in the id sequence the corpus builder tokenized; the capture tokenizes the
    text again with the tokenizer it was handed. Any drift -- a revision bump, a template change, a
    re-rendered row -- moves every span silently, so equality is required id for id.
    """
    missing = sorted(set(rendered) - set(table.input_ids))
    if missing:
        raise FullWeightsCellError(
            f"{len(missing)} stimuli have no entry in the span sidecar "
            f"({missing[:ERROR_EXAMPLE_COUNT]})"
        )
    for stimulus_id, text in rendered.items():
        ids = tuple(cast("list[int]", tokenizer(text)["input_ids"]))
        if ids != table.input_ids[stimulus_id]:
            raise FullWeightsCellError(
                f"stimulus {stimulus_id!r} tokenizes to {len(ids)} ids under this tokenizer but the "
                f"sidecar recorded {len(table.input_ids[stimulus_id])}; the two sequences differ, so "
                f"every span index for it would point at other tokens. Same tokenizer, same "
                f"revision, same render convention as the corpus builder, or rebuild the corpus."
            )
    logger.info(f"sidecar ids match the capture's tokenization, n={len(rendered)}")


# --------------------------------------------------------------------------------------
# Teacher-forced generation rows
# --------------------------------------------------------------------------------------


def teacher_forced_text(
    tokenizer: PreTrainedTokenizerBase,
    user_text: str,
    completion: str,
    *,
    enable_thinking: bool = True,
) -> str:
    r"""Render `template(user turn) + completion` for a verbatim generation-set row.

    The template's generation prompt has to end with `<think>\n`, because that is where the sampled
    completion began; a template that does not open the thinking block there would put the completion
    outside the reasoning the model wrote it as, silently.
    """
    extras: dict[str, Any] = dict(resolve_chat_template_kwargs(tokenizer))
    rendered = cast(
        "str",
        tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **extras,
        ),
    )
    if not rendered.endswith(THINK_OPEN_LINE):
        raise FullWeightsCellError(
            f"this template's generation prompt ends {rendered[-40:]!r}, not with "
            f"{THINK_OPEN_LINE!r}; a completion appended here would not sit where the model wrote "
            f"it (inside <think>)"
        )
    return rendered + completion


# --------------------------------------------------------------------------------------
# Capture: one forward per row, plain and span pools off the same hidden states
# --------------------------------------------------------------------------------------


def _make_span_hook(
    store: dict[tuple[str, str], dict[int, torch.Tensor]],
    layer: int,
    masks: Mapping[str, torch.Tensor],
    poolings: Sequence[str],
    n_tokens: int,
) -> Callable[[object, object, object], None]:
    """Build a read-only forward hook pooling layer `layer`'s residual over every named span mask."""

    def hook(module: object, inputs: object, output: object) -> None:
        del module, inputs
        hidden = cast("torch.Tensor", output[0] if isinstance(output, tuple) else output)
        if hidden.shape[0] != BATCH_SIZE or hidden.shape[1] != n_tokens:
            raise FullWeightsCellError(
                f"layer {layer} saw a [{hidden.shape[0]}, {hidden.shape[1]}, ...] residual where the "
                f"span masks were built for [1, {n_tokens}]; the masks would index other tokens"
            )
        upcast = hidden.float()
        for span_name, mask in masks.items():
            for pooling in poolings:
                store[span_name, pooling][layer] = pool_masked(upcast, mask, pooling)[0].cpu()

    return hook


def capture_row(  # noqa: PLR0913 - one row is a model, a tokenizer, a text, its spans and the knobs
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    *,
    n_tokens: int,
    poolings: Sequence[str],
    spans: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str], torch.Tensor]]:
    """One forward: pooling -> `[n_layers, hidden]` over the whole prompt, plus (span, pooling) -> same.

    The plain pools come from the shared capture core; the span hooks are registered beside its hooks
    on the same decoder blocks, so both read the one residual that forward produced. Batch size 1.
    """
    layers = language_model_layers(model)
    masks = {name: span_mask(n_tokens, span) for name, span in spans.items()}
    store: dict[tuple[str, str], dict[int, torch.Tensor]] = {
        (name, pooling): {} for name in masks for pooling in poolings
    }
    handles = [
        layer.register_forward_hook(_make_span_hook(store, index, masks, poolings, n_tokens))
        for index, layer in enumerate(layers)
    ]
    try:
        by_pooling = capture_pooled_activations_multi(
            model,  # pyright: ignore[reportArgumentType]  # annotated as the factory class
            tokenizer,  # pyright: ignore[reportArgumentType]  # same
            [text],
            poolings=list(poolings),
            batch_size=BATCH_SIZE,
        )
    finally:
        for handle in handles:
            handle.remove()
    plain = {
        pooling: torch.stack([by_layer[layer][0] for layer in sorted(by_layer)], dim=0)
        for pooling, by_layer in by_pooling.items()
    }
    n_layers = len(layers)
    spanned: dict[tuple[str, str], torch.Tensor] = {}
    for key, by_layer in store.items():
        if sorted(by_layer) != list(range(n_layers)):
            raise FullWeightsCellError(
                f"span {key} was pooled at layers {sorted(by_layer)}, not all {n_layers}"
            )
        spanned[key] = torch.stack([by_layer[layer] for layer in range(n_layers)], dim=0)
    return plain, spanned


@dataclass(frozen=True)
class SpanRow:
    """One stimulus's pooled activations over one span, with the span's width as its token count."""

    stimulus: Stimulus
    width: int
    pooled: dict[str, torch.Tensor]


def capture_full_weights_cell(  # noqa: PLR0913 - a cell is a model, a corpus, its spans and the knobs
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    grouped: Mapping[str, Sequence[Stimulus]],
    rendered: Mapping[str, str],
    counts_by_id: Mapping[str, int],
    poolings: Sequence[str],
    span_table: SpanTable | None,
    span_names: Sequence[str],
) -> tuple[dict[str, RowIndex], dict[tuple[str, str], torch.Tensor], dict[str, dict[str, str]]]:
    """Capture every set at one model state; return rows, activations and the span absences per set.

    Plain sets keep their name; each requested span adds a derived set `<set>@<span>` whose rows are
    the stimuli that declare the span, in corpus order, with the span width as the row's token count.
    """
    rows: dict[str, RowIndex] = {}
    activations: dict[tuple[str, str], torch.Tensor] = {}
    absences: dict[str, dict[str, str]] = {}
    for stimulus_set, members in sorted(grouped.items()):
        plain_rows: list[dict[str, torch.Tensor]] = []
        span_rows: dict[str, list[SpanRow]] = {name: [] for name in span_names}
        for member in members:
            spans = {} if span_table is None else span_table.spans[member.stimulus_id]
            absent = {} if span_table is None else span_table.spans_absent[member.stimulus_id]
            wanted = {name: spans[name] for name in span_names if name in spans}
            for name in span_names:
                if name not in wanted:
                    derived = derived_set_name(stimulus_set, name)
                    absences.setdefault(derived, {})[member.stimulus_id] = absent.get(
                        name, "not declared in the sidecar"
                    )
            plain, spanned = capture_row(
                model,
                tokenizer,
                rendered[member.stimulus_id],
                n_tokens=counts_by_id[member.stimulus_id],
                poolings=poolings,
                spans=wanted,
            )
            plain_rows.append(plain)
            for name, (start, end) in wanted.items():
                pooled = {pooling: spanned[name, pooling] for pooling in poolings}
                span_rows[name].append(SpanRow(stimulus=member, width=end - start, pooled=pooled))
        rows[stimulus_set] = row_index_for(
            members, [counts_by_id[member.stimulus_id] for member in members]
        )
        for pooling in poolings:
            activations[stimulus_set, pooling] = torch.stack(
                [row[pooling] for row in plain_rows], dim=0
            )
        for name, entries in span_rows.items():
            if not entries:
                raise FullWeightsCellError(
                    f"no stimulus in set {stimulus_set!r} declares span {name!r}; a derived set of "
                    f"nothing cannot be written, so drop it from --span-names"
                )
            derived = derived_set_name(stimulus_set, name)
            rows[derived] = row_index_for(
                [entry.stimulus for entry in entries], [entry.width for entry in entries]
            )
            for pooling in poolings:
                activations[derived, pooling] = torch.stack(
                    [entry.pooled[pooling] for entry in entries], dim=0
                )
        logger.info(
            f"captured set, {stimulus_set=} rows={len(members)} spans={list(span_names)} "
            f"shape={list(activations[stimulus_set, poolings[0]].shape)}"
        )
    return rows, activations, absences


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """CLI for the full-weights capture driver."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stimuli", type=Path, required=True, help="Five-key stimulus JSONL.")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Capture root to write cells under."
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        metavar="ARM:STEP=SOURCE",
        help="A checkpoint to capture: repo/name@revision or a local directory. Repeatable.",
    )
    parser.add_argument(
        "--replica",
        action="append",
        default=[],
        metavar="ARM:STEP=ARM:STEP",
        help="Declare a cell a replica (same bytes; the base and its mirror) of another. Repeatable.",
    )
    parser.add_argument(
        "--tokenizer", required=True, help="The tokenizer of record; the base model id."
    )
    parser.add_argument("--tokenizer-revision", default=None, help="Its hub revision, pinned.")
    parser.add_argument(
        "--spans-sidecar",
        type=Path,
        default=None,
        help="twin-corpus.json with per-stimulus ids and spans.",
    )
    parser.add_argument(
        "--span-names",
        default=None,
        help="Comma-separated spans to derive sets for; default every span the sidecar declares.",
    )
    parser.add_argument(
        "--coherence",
        type=Path,
        default=None,
        help="JSON map arm:step -> the coherence triple and its source.",
    )
    parser.add_argument("--poolings", default=",".join(DEFAULT_POOLINGS))
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=DEFAULT_MAX_PROMPT_TOKENS,
        help=f"Per-stimulus token budget, a refusal rather than a clamp, at most {MAX_PROMPT_TOKENS}.",
    )
    parser.add_argument("--store-dtype", choices=sorted(STORE_DTYPES), default="float32")
    parser.add_argument("--compute-dtype", choices=sorted(COMPUTE_DTYPES), default="bfloat16")
    parser.add_argument(
        "--limit-stimuli", type=int, default=None, help="Capture only the first N stimuli."
    )
    parser.add_argument(
        "--stimulus-render", choices=sorted(STIMULUS_RENDERS), default=STIMULUS_RENDER_TEMPLATED
    )
    parser.add_argument(
        "--no-thinking", action="store_true", help="Render against the thinking-off template."
    )
    parser.add_argument(
        "--deadline", default=None, help="ISO-8601 UTC instant after which no new cell starts."
    )
    return parser


def parse_poolings(raw: str) -> list[str]:
    """Split the `--poolings` list and refuse a name the shared capture does not know."""
    poolings = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(poolings) - set(POOLERS))
    if unknown or not poolings:
        raise FullWeightsCellError(
            f"unknown poolings {unknown}; expected a non-empty subset of {sorted(POOLERS)}"
        )
    return poolings


def resolve_agenda(args: argparse.Namespace) -> list[FullWeightsCellSpec]:
    """Return every `--cell` with its `--replica` declaration attached."""
    specs = apply_replica_declarations(
        [parse_cell_spec(spec) for spec in cast("list[str]", args.cell)],
        dict(parse_replica_spec(spec) for spec in cast("list[str]", args.replica)),
    )
    if not specs:
        raise FullWeightsCellError("nothing to capture: pass at least one --cell")
    return specs


@dataclass(frozen=True)
class PreparedCorpus:
    """The corpus as every cell will read it: rendered, counted, digested, with its spans resolved."""

    stimuli: list[Stimulus]
    digest: str
    rendered_digest: str
    grouped: dict[str, list[Stimulus]]
    rendered: dict[str, str]
    counts_by_id: dict[str, int]
    tokenizer: PreTrainedTokenizerBase
    span_table: SpanTable | None
    span_names: list[str]


def resolve_spans(
    args: argparse.Namespace,
    *,
    tokenizer: PreTrainedTokenizerBase,
    rendered: Mapping[str, str],
    digest: str,
) -> tuple[SpanTable | None, list[str]]:
    """Read the span sidecar, hold it to this corpus and tokenizer, and pick the spans to derive."""
    if args.spans_sidecar is None:
        if args.span_names is not None:
            raise FullWeightsCellError("--span-names needs --spans-sidecar")
        return None, []
    table = load_span_sidecar(args.spans_sidecar)
    if table.stimuli_sha256 != digest:
        raise FullWeightsCellError(
            f"the span sidecar was built for corpus {short_digest(table.stimuli_sha256)} but the "
            f"stimuli file digests to {short_digest(digest)}; its spans index another corpus"
        )
    if table.stimulus_render != args.stimulus_render:
        raise FullWeightsCellError(
            f"the span sidecar records render {table.stimulus_render!r}; --stimulus-render is "
            f"{args.stimulus_render!r}, so the capture's ids would not be the sidecar's"
        )
    assert_ids_match_sidecar(tokenizer, rendered, table)
    if args.span_names is None:
        return table, list(table.span_names)
    names = [part.strip() for part in str(args.span_names).split(",") if part.strip()]
    undeclared = sorted(set(names) - set(table.span_names))
    if undeclared:
        raise FullWeightsCellError(
            f"--span-names {undeclared} appear nowhere in the sidecar ({table.span_names})"
        )
    return table, names


def prepare_corpus(args: argparse.Namespace) -> PreparedCorpus:
    """Load the stimuli and the tokenizer of record, render, count, and resolve the spans."""
    stimuli = load_stimuli(args.stimuli)
    if args.limit_stimuli is not None:
        stimuli = stimuli[: args.limit_stimuli]
        logger.warning(
            f"capturing only the first {args.limit_stimuli} stimuli; a different corpus with a "
            f"different digest, so these cells cannot be compared against full ones"
        )
    digest = stimuli_digest(stimuli)
    tokenizer = cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.tokenizer_revision, trust_remote_code=True
        ),
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rendered = render_stimuli(
        tokenizer, stimuli, convention=args.stimulus_render, enable_thinking=not args.no_thinking
    )
    counts = token_counts(
        tokenizer,
        [rendered[stimulus.stimulus_id] for stimulus in stimuli],
        max_prompt_tokens=args.max_prompt_tokens,
    )
    span_table, span_names = resolve_spans(
        args, tokenizer=tokenizer, rendered=rendered, digest=digest
    )
    corpus = PreparedCorpus(
        stimuli=stimuli,
        digest=digest,
        rendered_digest=digest_of_strings(rendered[stimulus.stimulus_id] for stimulus in stimuli),
        grouped=group_by_set(stimuli),
        rendered=rendered,
        counts_by_id={
            stimulus.stimulus_id: count for stimulus, count in zip(stimuli, counts, strict=True)
        },
        tokenizer=tokenizer,
        span_table=span_table,
        span_names=span_names,
    )
    logger.info(
        f"corpus ready, digest={short_digest(digest)} rendered={short_digest(corpus.rendered_digest)} "
        f"n={len(stimuli)} sets={ {name: len(members) for name, members in sorted(corpus.grouped.items())} } "
        f"tokens_min={min(counts)} tokens_max={max(counts)} spans={span_names}"
    )
    return corpus


def _free_model(model: PreTrainedModel) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _peak_vram_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


@dataclass(frozen=True)
class CellOutcome:
    """What one captured cell reports back to the ladder manifest."""

    seconds: float
    peak_vram_bytes: int | None
    deltanet_kernel: dict[str, str]
    kernels_bound: dict[str, str]


def write_full_weights_cell(  # noqa: PLR0913 - a cell on disk is its weights, its tensors and every fact about the run
    cell_dir: Path,
    spec: FullWeightsCellSpec,
    facts: FullWeightsFacts,
    *,
    identity: CellIdentity,
    rows: Mapping[str, RowIndex],
    activations: Mapping[tuple[str, str], torch.Tensor],
    spans_absent: Mapping[str, Mapping[str, str]],
    run: CaptureRunFacts,
    loading_report: LoadingReport,
    deltanet_kernel: Mapping[str, str],
    coherence: CoherenceTriple,
    amplified: Mapping[str, Any] | None,
    seconds: float,
    peak_vram_bytes: int | None,
) -> Path:
    """Write one captured cell in the games format with the full-weights provenance block; return the manifest.

    The whole writer, separable from the capture so the manifest a reader parses can be produced
    without a model: `cell_provenance` is where every field name a CPU reader depends on is spelled.
    """
    return write_cell(
        cell_dir,
        arm=spec.arm,
        step=spec.step,
        identity=identity,
        rows=dict(rows),
        activations=dict(activations),
        applied_adapter_weights=None,
        adapter_weights_sha256=None,
        provenance=cell_provenance(
            spec,
            facts,
            run=run,
            loading_report=loading_report,
            deltanet_kernel=deltanet_kernel,
            amplified=amplified,
            coherence=coherence,
            spans_absent=spans_absent,
            seconds=seconds,
            peak_vram_bytes=peak_vram_bytes,
            captured_at=dt.datetime.now(tz=dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        ),
    )


def capture_and_write_cell(  # noqa: PLR0913 - one cell is its spec, its weights, the corpus and the run's knobs
    spec: FullWeightsCellSpec,
    facts: FullWeightsFacts,
    *,
    corpus: PreparedCorpus,
    args: argparse.Namespace,
    device: torch.device,
    cell_dir: Path,
    run: CaptureRunFacts,
    poolings: Sequence[str],
    coherence: CoherenceTriple,
    amplified: dict[str, Any] | None,
) -> CellOutcome:
    """Load one checkpoint, read its kernel binding, capture every set, free it, and write the cell."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    model, loading_report = load_full_weights_model(
        facts, dtype=COMPUTE_DTYPES[args.compute_dtype], device=device
    )
    kernels_bound = bound_deltanet_kernels()
    deltanet_kernel = prefill_deltanet_kernels(kernels_bound)
    identity = build_identity(
        model,
        base_model=run.identity_base,
        digest=corpus.digest,
        rendered_digest=corpus.rendered_digest,
        stimulus_render=args.stimulus_render,
        batch_size=BATCH_SIZE,
        compute_dtype=args.compute_dtype,
        store_dtype=args.store_dtype,
    )
    rows, activations, absences = capture_full_weights_cell(
        model,
        corpus.tokenizer,
        grouped=corpus.grouped,
        rendered=corpus.rendered,
        counts_by_id=corpus.counts_by_id,
        poolings=poolings,
        span_table=corpus.span_table,
        span_names=corpus.span_names,
    )
    seconds = round(time.time() - started, 2)
    peak = _peak_vram_bytes(device)
    _free_model(model)
    write_full_weights_cell(
        cell_dir,
        spec,
        facts,
        identity=identity,
        rows=rows,
        activations=activations,
        spans_absent=absences,
        run=run,
        loading_report=loading_report,
        deltanet_kernel=deltanet_kernel,
        coherence=coherence,
        amplified=amplified,
        seconds=seconds,
        peak_vram_bytes=peak,
    )
    return CellOutcome(
        seconds=seconds,
        peak_vram_bytes=peak,
        deltanet_kernel=deltanet_kernel,
        kernels_bound=kernels_bound,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve every checkpoint, gate the agenda, then capture each cell and free it."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = build_parser().parse_args(argv)
    poolings = parse_poolings(str(args.poolings))
    if not 0 < args.max_prompt_tokens <= MAX_PROMPT_TOKENS:
        raise FullWeightsCellError(
            f"--max-prompt-tokens {args.max_prompt_tokens} is outside (0, {MAX_PROMPT_TOKENS}]; the "
            f"cap is a 9B prompt plus TMAX's own completion budget, and a longer row is a different "
            f"study"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deadline = None if args.deadline is None else dt.datetime.fromisoformat(args.deadline)
    specs = resolve_agenda(args)
    coherence = {} if args.coherence is None else load_coherence_table(args.coherence)
    unknown_coherence = sorted(set(coherence) - {spec.label for spec in specs})
    if unknown_coherence:
        raise FullWeightsCellError(
            f"--coherence names cells not on the agenda: {unknown_coherence}"
        )
    corpus = prepare_corpus(args)

    resolved: dict[str, tuple[FullWeightsCellSpec, FullWeightsFacts]] = {}
    for spec in specs:
        facts = resolve_full_weights(spec.source)
        resolved[spec.label] = (spec, facts)
        logger.info(
            f"resolved {spec.label} -> {facts.label} fingerprint={short_digest(facts.fingerprint)}"
        )
    assert_fingerprints_declared(resolved)
    amplified = {label: read_amplified_sidecar(facts) for label, (_spec, facts) in resolved.items()}

    # Before the first load, which imports the modeling module and freezes each kernel's dispatch.
    kernel_bridge = bridge_and_check_decode_kernel()
    run = CaptureRunFacts(
        git_sha=git_sha(),
        torch_version=torch.__version__,
        device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        stimuli_file=str(args.stimuli),
        spans_sidecar=None if args.spans_sidecar is None else str(args.spans_sidecar),
        span_names=tuple(corpus.span_names),
        tokenizer_id=str(args.tokenizer),
        tokenizer_revision=cast("str | None", args.tokenizer_revision),
    )
    ladder: dict[str, Any] = {
        "kind": "full-weights-ladder",
        **run.to_payload(),
        "requested_cells": [spec.label for spec in specs],
        "poolings": poolings,
        "batch_size": BATCH_SIZE,
        "deltanet_kernel_bridge": kernel_bridge,
        "cells": {},
    }
    kernels_seen: list[object] = []
    for spec in specs:
        if deadline is not None and dt.datetime.now(tz=dt.UTC) >= deadline:
            logger.warning(f"deadline {args.deadline} reached; stopping before {spec.label}")
            ladder["stopped_reason"] = f"deadline reached before {spec.label}"
            break
        cell_dir = step_dir(args.out_dir, spec.arm, spec.step)
        if (cell_dir / CELL_MANIFEST_FILENAME).is_file():
            logger.info(f"skipping {spec.label}: an earlier attempt already finished it")
            ladder["cells"][spec.label] = {"skipped": "already on disk"}
            continue
        facts = resolved[spec.label][1]
        outcome = capture_and_write_cell(
            spec,
            facts,
            corpus=corpus,
            args=args,
            device=device,
            cell_dir=cell_dir,
            run=run,
            poolings=poolings,
            coherence=coherence.get(
                spec.label, CoherenceTriple.unavailable("no record source declared for this cell")
            ),
            amplified=amplified[spec.label],
        )
        kernels_seen.append(outcome.deltanet_kernel)
        assert_one_deltanet_kernel(kernels_seen, what="this capture run")
        ladder.setdefault("deltanet_kernels_bound", outcome.kernels_bound)
        ladder.setdefault(DELTANET_KERNEL_FIELD, outcome.deltanet_kernel)
        ladder["cells"][spec.label] = {
            "seconds": outcome.seconds,
            "peak_vram_bytes": outcome.peak_vram_bytes,
            WEIGHTS_FINGERPRINT_FIELD: facts.fingerprint,
            REPLICA_OF_FIELD: spec.replica_of,
        }
        write_ladder_manifest(args.out_dir, ladder)
        peak = outcome.peak_vram_bytes
        peak_text = "n/a" if peak is None else f"{peak / 2**30:.2f}"
        logger.info(
            f"cell done, {spec.label} seconds={outcome.seconds} peak_vram_gib={peak_text} "
            f"kernel={outcome.deltanet_kernel}"
        )

    if "stopped_reason" not in ladder:
        loaded = load_full_weights_ladder(args.out_dir, stimuli_sha256=corpus.digest)
        ladder["displacements"] = assert_replicas_and_displacement(loaded.cells)
        for pair, delta in ladder["displacements"].items():
            logger.info(f"DISPLACEMENT {pair}: max|delta|={delta:.6e}")
    write_ladder_manifest(args.out_dir, ladder)
    logger.info(f"capture finished, cells={len(ladder['cells'])} out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
