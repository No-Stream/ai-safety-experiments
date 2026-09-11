r"""The Jacobian-lens fit corpus for the TMAX 9B wave: the text windows a lens is averaged over.

A Jacobian lens is an average over prompts, positions and output dimensions, so what it averages
over decides what it can read. The wave's reads are taken on three kinds of text -- grader source
inlined in a prompt, the tail of a transcript where the solution lands, and the model's own
reasoning -- and a lens fitted on none of them would be a coordinate correction estimated somewhere
else. This module draws that corpus and writes it where ``jlens.fit`` can read it, one string per
window, with every window behind two gates:

* **Round-trip decode.** A window is a run of token ids cut out of a longer document, and
  ``jlens.fit`` re-tokenizes the STRING it is handed. A cut that lands inside a byte-pair merge
  decodes to text whose re-encoding is a different id sequence, so the fit would run on tokens
  nobody chose; measured on this tokenizer, one 512-token window in about thirty fails that way.
  Every window's decoded text has to re-encode to exactly the ids it was cut from; a start that does
  not is shifted within :data:`ROUND_TRIP_SLACK` positions until one does, and a document with no
  such start is skipped and counted. The shift is recorded on the item.
* **Disjointness.** Fit windows, eval windows and anchor prompts never share a source GROUP. A
  reconstruction score read on text the lens was averaged over is not a score, and the twin corpus
  renders each problem six ways that differ by one line, so two renderings of one problem are one
  group (the pair id), and two completions of one problem by one checkpoint are one group (the
  record file and the problem id); the split is at the group level and then asserted over the drawn
  items, not assumed from the draw. Schema 1 corpora were split at the document level and are
  refused by the loader rather than read as if they had been.

``max_seq_len`` is derived from the drawn corpus through :func:`derive_max_seq_len`, never left at
the reference default of 128: the fit truncates every prompt to it, so at 128 a 512-token window
would fit on its first quarter with nothing going red.

**Sources.** The grader class comes from the twin-corpus stimulus file
(``reward_hacking.interp.tmax_twin_corpus`` writes the five-key rows ``games.interp_cells.load_stimuli``
reads); the grader block is the last fenced ``python`` block of a rendering that carries two, since
the training template puts the statement first and the grader second. The tail and reasoning
classes come from legibility-probe record files: a tail is the last window of a record's
``completion``; a reasoning window sits inside the span before ``</think>`` of a record the base
model wrote. Whether the base wrote it is read off the record's SERVED weights
(``model_full_weights`` / ``model_adapter_dir``), never off ``model_id``: the screen serves every
TMAX checkpoint under the base's ``model_id``, which names the tokenizer, so a reader keyed on it
would file RL'd reasoning as the base's. Anchors are whole texts -- rendered prompts and whole completions -- up to
:data:`ANCHOR_MAX_TOKENS`, never truncated. Until the twin corpus lands, ``--stimuli`` may be left
off and the grader class is stood in by :data:`CLASS_CONCEPT_SENTENCES`: the contrastive pairs of
``reward_hacking.interp.stimuli`` packed into windows, which the sidecar says in so many words, so a
lens fitted on the stand-in is never mistaken for one fitted on grader text.

**Privacy.** Every drawn text is benchmark material -- prompts, planted values, model output -- so the
corpus lands under ``artifacts/`` (gitignored) and nothing here logs text, only ids, counts and hashes.

    uv run python -m reward_hacking.interp.tmax_lens_corpus \\
        --records <records.jsonl> [--records ...] [--stimuli <stimuli.jsonl>] --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from transformers import AutoTokenizer

from games.interp_cells import load_stimuli, sha256_of_file
from reward_hacking.interp.jacobian import SeqLenPlan, derive_max_seq_len, digest_strings
from reward_hacking.interp.stimuli import CONCEPTS

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

CORPUS_SCHEMA = 2
"""Schema 2 splits roles at the group level and decides ``base_written`` by the served weights.

Schema 1 split at the document level, so two renderings of one twin problem could sit on either side
of the fit/eval split, and read ``base_written`` off ``model_id``, which every screen record carries
as the base id whatever weights were served. The loader refuses a schema-1 directory by name.
"""
CORPUS_KIND = "tmax-lens-corpus"

WINDOW_TOKENS = 512
N_FIT_WINDOWS = 60
N_EVAL_WINDOWS = 24
N_ANCHOR_PROMPTS = 24
ANCHOR_MAX_TOKENS = 8192
ANCHOR_MIN_TOKENS = 64
"""An anchor shorter than this has almost no positions past the fit's leading-sink exclusion."""
ROUND_TRIP_SLACK = 16

CLASS_GRADER = "grader-text"
CLASS_TAIL = "transcript-tail"
CLASS_REASONING = "base-reasoning"
CLASS_CONCEPT_SENTENCES = "concept-sentences"
WINDOW_CLASSES: tuple[str, ...] = (CLASS_GRADER, CLASS_TAIL, CLASS_REASONING)
FALLBACK_WINDOW_CLASSES: tuple[str, ...] = (CLASS_CONCEPT_SENTENCES, CLASS_TAIL, CLASS_REASONING)

ROLE_FIT = "fit"
ROLE_EVAL = "eval"
ROLE_ANCHOR = "anchor"

SOURCE_STIMULI = "stimuli"
SOURCE_CONCEPT_SENTENCES = "concept-sentences"
SERVED_PROMPT = "rendered-prompt"
"""The ``served`` value of a document nobody generated: a rendered prompt or a packed sentence set."""
THINK_END = "</think>"

RECORD_SERVING_FIELDS: tuple[str, ...] = (
    "model_id",
    "model_load_mode",
    "model_full_weights",
    "model_adapter_dir",
)
"""What a record has to say about the weights that wrote it before ``base_written`` can be decided."""

FIT_WINDOWS_FILENAME = "fit_windows.jsonl"
EVAL_WINDOWS_FILENAME = "eval_windows.jsonl"
ANCHORS_FILENAME = "anchors.jsonl"
SIDECAR_FILENAME = "lens-corpus.json"
DEFAULT_OUT_DIR = Path("artifacts/reward_hacking/tmax-interp/lens-corpus")

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-9B"
DEFAULT_TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
"""The wave's tokenizer of record: the base checkpoint at the commit the twin corpus pins."""

_FENCED_PYTHON = re.compile(r"```python\n.*?\n```", re.DOTALL)


class CorpusDrawError(ValueError):
    """The corpus cannot be drawn as asked: a class short of documents, or an overlap."""


# --------------------------------------------------------------------------------------
# Tokenizer seam
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenCodec:
    """The three tokenizer operations the draw needs, so the draw is testable on a mock.

    ``encode`` adds no special tokens (the documents already carry the template's own), ``decode``
    keeps them, and ``offsets`` gives each token's ``[start, end)`` character span so a character
    position found in the text maps onto a token index.
    """

    encode: Callable[[str], list[int]]
    decode: Callable[[Sequence[int]], str]
    offsets: Callable[[str], list[tuple[int, int]]]


def hf_token_codec(tokenizer: Any) -> TokenCodec:  # noqa: ANN401 - a PreTrainedTokenizerBase, kept duck-typed
    """Wrap a fast HuggingFace tokenizer as a :class:`TokenCodec`."""

    def encode(text: str) -> list[int]:
        return cast("list[int]", tokenizer(text, add_special_tokens=False)["input_ids"])

    def decode(ids: Sequence[int]) -> str:
        return cast("str", tokenizer.decode(list(ids), skip_special_tokens=False))

    def offsets(text: str) -> list[tuple[int, int]]:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(a), int(b)) for a, b in encoded["offset_mapping"]]

    return TokenCodec(encode=encode, decode=decode, offsets=offsets)


def token_index_at_char(offsets: Sequence[tuple[int, int]], char_index: int) -> int:
    """Return the index of the token covering ``char_index``, or the count when it lies past the text."""
    for index, (_start, end) in enumerate(offsets):
        if end > char_index:
            return index
    return len(offsets)


# --------------------------------------------------------------------------------------
# Documents: what a window is cut from
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Document:
    """One source text, tokenized once, with the facts that decide which windows it can supply.

    ``group_id`` names the set of documents that are near-copies of each other -- the six renderings
    of one twin problem, or one checkpoint's completions of one problem -- and the draw hands a group
    to at most one item. ``served`` names the weights that wrote a completion (``repo@revision``), or
    :data:`SERVED_PROMPT` for text nobody generated, so the sidecar can say whose text each class holds.
    """

    doc_id: str
    group_id: str
    source: str
    served: str
    text: str
    ids: tuple[int, ...]
    grader_tokens: tuple[int, int] | None
    reasoning_end_token: int | None
    base_written: bool

    @property
    def n_tokens(self) -> int:
        """How many tokens the whole document is."""
        return len(self.ids)

    def window_classes(self, window_tokens: int) -> tuple[str, ...]:
        """Which window classes this document can supply at ``window_tokens`` width."""
        classes: list[str] = []
        if self.grader_tokens is not None and self.n_tokens >= window_tokens:
            classes.append(CLASS_GRADER)
        if self.source == SOURCE_CONCEPT_SENTENCES and self.n_tokens >= window_tokens:
            classes.append(CLASS_CONCEPT_SENTENCES)
        if self.source not in (SOURCE_STIMULI, SOURCE_CONCEPT_SENTENCES) and (
            self.n_tokens >= window_tokens
        ):
            classes.append(CLASS_TAIL)
        if (
            self.base_written
            and self.reasoning_end_token is not None
            and self.reasoning_end_token >= window_tokens
        ):
            classes.append(CLASS_REASONING)
        return tuple(classes)

    def anchor_eligible(self, *, max_tokens: int, min_tokens: int) -> bool:
        """Whether the whole text can be an anchor prompt: long enough to read, short enough to fit."""
        return min_tokens <= self.n_tokens <= max_tokens


def assert_unique_doc_ids(documents: Sequence[Document], *, what: str) -> None:
    """Refuse a document set with a repeated id: two texts under one id would later read as one.

    Watched to fail on the real screen records, where a problem sampled in two probe cells shares
    problem, group and sample indices and only the cell name tells the two completions apart.
    """
    seen: dict[str, int] = {}
    for document in documents:
        seen[document.doc_id] = seen.get(document.doc_id, 0) + 1
    duplicates = sorted(doc_id for doc_id, count in seen.items() if count > 1)
    if duplicates:
        raise CorpusDrawError(
            f"{what}: {len(duplicates)} document ids occur more than once, e.g. {duplicates[:3]}; "
            f"the id has to name one text"
        )


def grader_char_span(text: str) -> tuple[int, int] | None:
    """Locate the grader block: the last fenced ``python`` block of a text that carries two.

    The training template renders the statement's code block first and the inlined grader second;
    a rendering with a single block (the withheld grader) carries no grader text to window.
    """
    blocks = list(_FENCED_PYTHON.finditer(text))
    if len(blocks) < 2:  # noqa: PLR2004 - statement block plus grader block
        return None
    last = blocks[-1]
    return last.start(), last.end()


def stimulus_documents(path: Path, codec: TokenCodec) -> list[Document]:
    """Every rendered prompt in a five-key stimulus file, with its grader block located.

    The six renderings of one problem share a ``pair_id`` and differ by about one line, so the pair
    is the group: a corpus that fitted on one rendering and scored on another would be scoring on
    its own fit text.
    """
    documents: list[Document] = []
    for stimulus in load_stimuli(path):
        text = stimulus.text
        span = grader_char_span(text)
        grader_tokens: tuple[int, int] | None = None
        if span is not None:
            offsets = codec.offsets(text)
            grader_tokens = (
                token_index_at_char(offsets, span[0]),
                token_index_at_char(offsets, span[1] - 1) + 1,
            )
        documents.append(
            Document(
                doc_id=f"{SOURCE_STIMULI}:{stimulus.stimulus_id}",
                group_id=f"{SOURCE_STIMULI}:{stimulus.pair_id}",
                source=SOURCE_STIMULI,
                served=SERVED_PROMPT,
                text=text,
                ids=tuple(codec.encode(text)),
                grader_tokens=grader_tokens,
                reasoning_end_token=None,
                base_written=False,
            )
        )
    assert_unique_doc_ids(documents, what=str(path))
    logger.info(
        "stimulus documents, %s",
        f"{path=} n={len(documents)} "
        f"n_groups={len({document.group_id for document in documents})} "
        f"n_with_grader={sum(document.grader_tokens is not None for document in documents)}",
    )
    return documents


def served_weights(row: Mapping[str, Any], *, path: Path) -> str:
    """Name the weights that wrote a record: ``repo@revision`` for full weights, else the model id.

    Every serving field has to be present -- a record from before the probe recorded how it was
    served cannot say whose reasoning it holds, and the class this decides is named "base-written".
    """
    missing = [field for field in RECORD_SERVING_FIELDS if field not in row]
    if missing:
        raise CorpusDrawError(
            f"{path}: a record lacks {missing}; without the serving fields the reader cannot tell "
            f"whose weights wrote the completion, and 'base-written' would be a guess"
        )
    if row["model_adapter_dir"] is not None:
        return f"{row['model_id']}+adapter:{row['model_adapter_dir']}"
    if row["model_full_weights"] is not None:
        return str(row["model_full_weights"])
    return str(row["model_id"])


def is_base_written(served: str, *, base_model_id: str) -> bool:
    """Whether ``served`` names the base checkpoint itself: its bare id, or its full weights at any revision.

    The screen writes the base's ``model_id`` on every record whatever weights it served (it names the
    tokenizer), so the decision rests on the served-weights string alone: the base repo at a revision
    (``Qwen/Qwen3.5-9B@main``), never a different repo and never an adapter on top.
    """
    return served == base_model_id or served.split("@", 1)[0] == base_model_id


def record_documents(path: Path, codec: TokenCodec, *, base_model_id: str) -> list[Document]:
    """Every complete completion in a legibility-probe record file.

    A record whose thinking ran out of budget is left out: its completion is a cut-off reasoning
    span with no tail, so it can supply neither class honestly. ``base_written`` is decided by the
    weights that were served (:func:`served_weights` against ``base_model_id``), which is what
    separates base reasoning from the RL'd checkpoints' reasoning. One checkpoint's completions of
    one problem are one group.
    """
    documents: list[Document] = []
    n_truncated = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        completion = str(row["completion"])
        if bool(row["truncated_thinking"]) or not completion.strip():
            n_truncated += 1
            continue
        reasoning_end_token: int | None = None
        think_end = completion.find(THINK_END)
        if think_end >= 0:
            reasoning_end_token = token_index_at_char(codec.offsets(completion), think_end)
        served = served_weights(row, path=path)
        documents.append(
            Document(
                doc_id=(
                    f"{path.stem}:{row['cell']}:{row['problem_id']}:g{row['group_index']}"
                    f":s{row['sample_index']}"
                ),
                group_id=f"{path.stem}:{row['problem_id']}",
                source=f"records:{path.stem}",
                served=served,
                text=completion,
                ids=tuple(codec.encode(completion)),
                grader_tokens=None,
                reasoning_end_token=reasoning_end_token,
                base_written=is_base_written(served, base_model_id=base_model_id),
            )
        )
    assert_unique_doc_ids(documents, what=str(path))
    logger.info(
        "record documents, %s",
        f"{path=} n={len(documents)} {n_truncated=} "
        f"n_groups={len({document.group_id for document in documents})} "
        f"served={_count_by(document.served for document in documents)} "
        f"n_base_written={sum(document.base_written for document in documents)} "
        f"n_with_reasoning={sum(document.reasoning_end_token is not None for document in documents)}",
    )
    return documents


def concept_sentence_documents(codec: TokenCodec, *, window_tokens: int) -> list[Document]:
    """Pack the contrastive pairs into documents at least one window long: the grader stand-in.

    Both sides of every pair, concept by concept, one sentence per line, so a window is a run of
    short statements about grading, concealment and evaluation rather than grader source. Named as
    a stand-in wherever it appears; it exists so the corpus can be drawn before the twin corpus
    lands, not as a substitute for it.
    """
    sentences = [
        sentence
        for pairs in CONCEPTS.values()
        for pair in pairs
        for sentence in (pair.positive, pair.negative)
    ]
    documents: list[Document] = []
    buffer: list[str] = []
    for sentence in sentences:
        buffer.append(sentence)
        text = "\n".join(buffer)
        ids = codec.encode(text)
        if len(ids) >= window_tokens:
            documents.append(
                Document(
                    doc_id=f"{SOURCE_CONCEPT_SENTENCES}:{len(documents):03d}",
                    group_id=f"{SOURCE_CONCEPT_SENTENCES}:{len(documents):03d}",
                    source=SOURCE_CONCEPT_SENTENCES,
                    served=SERVED_PROMPT,
                    text=text,
                    ids=tuple(ids),
                    grader_tokens=None,
                    reasoning_end_token=None,
                    base_written=False,
                )
            )
            buffer = []
    logger.info(
        "concept-sentence documents, %s", f"n={len(documents)} n_sentences={len(sentences)}"
    )
    return documents


# --------------------------------------------------------------------------------------
# Windows: cutting, round-tripping, drawing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusItem:
    """One string handed to the fit or the eval, and exactly where in which document it came from."""

    item_id: str
    role: str
    window_class: str
    source: str
    served: str
    doc_id: str
    group_id: str
    start: int
    end: int
    shift: int
    text: str
    ids_sha256: str

    @property
    def n_tokens(self) -> int:
        """The window's width in tokens, which its text re-encodes to exactly."""
        return self.end - self.start

    def as_row(self) -> dict[str, object]:
        """Return the JSONL row; ``n_tokens`` is written out so a reader needs no tokenizer to plan."""
        return {**asdict(self), "n_tokens": self.n_tokens}


def ids_digest(ids: Sequence[int]) -> str:
    """sha256 of a token-id run, so an item can be checked against a tokenizer later."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def round_trip_start(  # noqa: PLR0913 - a window search is its document, its start, its bounds and its slack
    document: Document,
    codec: TokenCodec,
    *,
    start: int,
    width: int,
    lower: int,
    upper: int,
    slack: int,
    directions: tuple[int, ...] = (1, -1),
) -> tuple[int, str] | None:
    """Find the nearest window start in ``[lower, upper]`` whose decoded text re-encodes to its ids.

    Tries ``start`` first, then alternates outward one position at a time in the given
    ``directions`` (``(1, -1)`` both ways, ``(-1,)`` earlier only) up to ``slack`` steps. Returns the
    start and the decoded text, or ``None`` when no start within reach round-trips.
    """
    upper = min(upper, document.n_tokens - width)
    candidates: list[int] = [start]
    for step in range(1, slack + 1):
        candidates.extend(start + direction * step for direction in directions)
    tried = 0
    for candidate in candidates:
        if not lower <= candidate <= upper:
            continue
        tried += 1
        ids = list(document.ids[candidate : candidate + width])
        text = codec.decode(ids)
        if codec.encode(text) == ids:
            return candidate, text
    logger.warning(
        "no round-tripping window start, %s",
        f"doc_id={document.doc_id} {start=} {width=} {lower=} {upper=} {slack=} {tried=}",
    )
    return None


def _seeded_start(doc_id: str, *, lower: int, upper: int) -> int:
    """Derive a start in ``[lower, upper]`` from the document's identity, never from draw order."""
    if upper < lower:
        raise CorpusDrawError(f"{doc_id}: no room for a window ({lower=} {upper=})")
    draw = int(hashlib.sha256(doc_id.encode()).hexdigest()[:16], 16)
    return lower + draw % (upper - lower + 1)


def cut_window(
    document: Document,
    codec: TokenCodec,
    *,
    window_class: str,
    width: int,
    slack: int,
) -> tuple[int, int, str] | None:
    """Cut one ``window_class`` window of ``width`` tokens out of ``document``: ``(start, shift, text)``.

    The intended start depends on the class: a grader window is centred on the grader block (or
    starts at it when the block is wider than the window), a tail window ends at the document's end
    and may only move earlier, a reasoning window sits at an identity-seeded position inside the
    reasoning span, and a concept-sentence window starts at the text's start. ``shift`` is how far
    the round-trip gate moved it.
    """
    last_start = document.n_tokens - width
    if last_start < 0:
        return None
    lower, upper, directions = 0, last_start, (1, -1)
    if window_class == CLASS_GRADER:
        if document.grader_tokens is None:
            raise CorpusDrawError(
                f"{document.doc_id}: asked for a grader window, has no grader block"
            )
        grader_start, grader_end = document.grader_tokens
        centred = grader_start - max(0, (width - (grader_end - grader_start)) // 2)
        start = min(max(centred, 0), last_start)
    elif window_class == CLASS_TAIL:
        start, directions = last_start, (-1,)
    elif window_class == CLASS_REASONING:
        if document.reasoning_end_token is None:
            raise CorpusDrawError(
                f"{document.doc_id}: asked for a reasoning window, has no </think>"
            )
        upper = min(document.reasoning_end_token - width, last_start)
        start = _seeded_start(document.doc_id, lower=0, upper=upper)
    elif window_class == CLASS_CONCEPT_SENTENCES:
        start = 0
    else:
        raise CorpusDrawError(f"unknown window class {window_class!r}")
    found = round_trip_start(
        document,
        codec,
        start=start,
        width=width,
        lower=lower,
        upper=upper,
        slack=slack,
        directions=directions,
    )
    if found is None:
        return None
    chosen, text = found
    return chosen, chosen - start, text


def hash_order(documents: Iterable[Document]) -> list[Document]:
    """Documents in the order of their identity hash: a draw order that no file order can move."""
    return sorted(
        documents, key=lambda document: hashlib.sha256(document.doc_id.encode()).hexdigest()
    )


def class_quotas(total: int, classes: Sequence[str]) -> dict[str, int]:
    """Split ``total`` across ``classes`` as evenly as possible, the remainder to the first ones."""
    base, remainder = divmod(total, len(classes))
    return {name: base + (1 if index < remainder else 0) for index, name in enumerate(classes)}


@dataclass
class DrawLedger:
    """What the draw skipped and why, so the sidecar can say it rather than the counts implying it."""

    skipped_round_trip: dict[str, int]
    skipped_too_short: int = 0

    def as_payload(self) -> dict[str, object]:
        """Return the sidecar block."""
        return {
            "skipped_round_trip_by_class": dict(self.skipped_round_trip),
            "skipped_too_short": self.skipped_too_short,
        }


def draw_windows(  # noqa: PLR0913 - a draw is its documents, its quotas and its cutting knobs
    documents: Sequence[Document],
    codec: TokenCodec,
    *,
    role: str,
    quotas: Mapping[str, int],
    used: set[str],
    width: int,
    slack: int,
    ledger: DrawLedger,
) -> list[CorpusItem]:
    """Draw ``quotas[class]`` windows per class from unused groups, marking each group used.

    Documents are visited in hash order; a document that cannot supply a round-tripping window for
    the class is skipped and counted, and a class that runs out of documents is a refusal with the
    shortfall named, because a lens fitted on a corpus missing a class is a different experiment.
    ``used`` holds group ids, so once one rendering of a problem is drawn its siblings are out.
    """
    items: list[CorpusItem] = []
    for window_class, quota in quotas.items():
        drawn = 0
        for document in hash_order(documents):
            if drawn == quota:
                break
            if document.group_id in used or window_class not in document.window_classes(width):
                continue
            cut = cut_window(document, codec, window_class=window_class, width=width, slack=slack)
            if cut is None:
                ledger.skipped_round_trip[window_class] = (
                    ledger.skipped_round_trip.get(window_class, 0) + 1
                )
                continue
            start, shift, text = cut
            used.add(document.group_id)
            items.append(
                CorpusItem(
                    item_id=f"{role}:{window_class}:{drawn:03d}",
                    role=role,
                    window_class=window_class,
                    source=document.source,
                    served=document.served,
                    doc_id=document.doc_id,
                    group_id=document.group_id,
                    start=start,
                    end=start + width,
                    shift=shift,
                    text=text,
                    ids_sha256=ids_digest(document.ids[start : start + width]),
                )
            )
            drawn += 1
        if drawn < quota:
            raise CorpusDrawError(
                f"{role} {window_class}: drew {drawn} of {quota} windows; the documents that can "
                f"supply this class are exhausted (round-trip skips so far: "
                f"{ledger.skipped_round_trip.get(window_class, 0)}). Add sources or lower the quota "
                f"deliberately; a corpus short of a class is a different fit."
            )
    return items


def draw_anchors(  # noqa: PLR0913 - the anchor draw is its documents, its bounds and its ledger
    documents: Sequence[Document],
    codec: TokenCodec,
    *,
    count: int,
    used: set[str],
    max_tokens: int,
    min_tokens: int,
    ledger: DrawLedger,
) -> list[CorpusItem]:
    """Draw ``count`` whole-text anchors round-robin over sources, from unused eligible groups.

    Round-robin so the anchors are not all rendered prompts or all completions: each source's
    eligible documents are visited in hash order and one is taken per source per round, and a group
    already drawn on -- by a window or by an earlier anchor -- is passed over.
    """
    eligible: dict[str, list[Document]] = {}
    for document in hash_order(documents):
        if document.group_id in used:
            continue
        if not document.anchor_eligible(max_tokens=max_tokens, min_tokens=min_tokens):
            if document.n_tokens < min_tokens:
                ledger.skipped_too_short += 1
            continue
        eligible.setdefault(document.source, []).append(document)
    items: list[CorpusItem] = []
    sources = sorted(eligible)
    while len(items) < count and any(eligible[source] for source in sources):
        for source in sources:
            document = None if len(items) == count else _pop_unused(eligible[source], used)
            if document is None:
                continue
            text = document.text
            if tuple(codec.encode(text)) != document.ids:
                raise CorpusDrawError(
                    f"{document.doc_id}: the whole text does not re-encode to its ids"
                )
            used.add(document.group_id)
            items.append(
                CorpusItem(
                    item_id=f"{ROLE_ANCHOR}:{len(items):03d}",
                    role=ROLE_ANCHOR,
                    window_class=ROLE_ANCHOR,
                    source=document.source,
                    served=document.served,
                    doc_id=document.doc_id,
                    group_id=document.group_id,
                    start=0,
                    end=document.n_tokens,
                    shift=0,
                    text=text,
                    ids_sha256=ids_digest(document.ids),
                )
            )
    if len(items) < count:
        raise CorpusDrawError(
            f"anchors: drew {len(items)} of {count}; eligible whole texts between {min_tokens} and "
            f"{max_tokens} tokens not already used by a window are exhausted"
        )
    return items


def _pop_unused(queue: list[Document], used: set[str]) -> Document | None:
    """Pop documents off ``queue`` until one whose group is still unused; ``None`` if none is left."""
    while queue:
        document = queue.pop(0)
        if document.group_id not in used:
            return document
    return None


def assert_disjoint(items: Sequence[CorpusItem]) -> None:
    """Refuse a corpus in which two items share a source group.

    One item per group, whatever the roles: two renderings of one problem on either side of the
    fit/eval split would score the lens on its own fit text, and two of them inside one role would
    weight that problem twice. Asserted over the drawn items rather than trusted from the draw: the
    draw marks groups used, and this is the check that the marking held, watched to fail on an eval
    item moved onto a fit item's group.
    """
    by_group: dict[str, list[CorpusItem]] = {}
    for item in items:
        by_group.setdefault(item.group_id, []).append(item)
    for group_id, members in by_group.items():
        if len(members) > 1:
            raise CorpusDrawError(
                f"group {group_id} supplies {len(members)} items "
                f"({', '.join(f'{item.item_id} as {item.role}' for item in members)}); fit, eval "
                f"and anchor texts must come from disjoint groups, one item each"
            )


# --------------------------------------------------------------------------------------
# The corpus: draw, write, load
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LensCorpus:
    """The drawn corpus: three item lists plus the sidecar that says how they were drawn."""

    fit: tuple[CorpusItem, ...]
    eval: tuple[CorpusItem, ...]
    anchors: tuple[CorpusItem, ...]
    sidecar: dict[str, Any]

    @property
    def fit_prompts(self) -> list[str]:
        """The strings ``jlens.fit`` is handed for the window lens."""
        return [item.text for item in self.fit]

    @property
    def eval_prompts(self) -> list[str]:
        """The strings the reconstruction read runs on."""
        return [item.text for item in self.eval]

    @property
    def anchor_prompts(self) -> list[str]:
        """The whole prompts the full-context anchor lens is fitted on."""
        return [item.text for item in self.anchors]

    def plan(self, role: str, *, ceiling: int) -> SeqLenPlan:
        """Derive the ``max_seq_len`` a fit on ``role`` runs at from these items."""
        items = {ROLE_FIT: self.fit, ROLE_EVAL: self.eval, ROLE_ANCHOR: self.anchors}[role]
        return derive_max_seq_len([item.n_tokens for item in items], ceiling=ceiling)


@dataclass(frozen=True)
class DrawSpec:
    """Every knob of a draw, recorded in the sidecar."""

    window_tokens: int = WINDOW_TOKENS
    n_fit: int = N_FIT_WINDOWS
    n_eval: int = N_EVAL_WINDOWS
    n_anchor: int = N_ANCHOR_PROMPTS
    anchor_max_tokens: int = ANCHOR_MAX_TOKENS
    anchor_min_tokens: int = ANCHOR_MIN_TOKENS
    round_trip_slack: int = ROUND_TRIP_SLACK


def draw_corpus(
    documents: Sequence[Document],
    codec: TokenCodec,
    *,
    spec: DrawSpec,
    classes: Sequence[str],
    provenance: Mapping[str, object],
) -> LensCorpus:
    """Draw fit windows, then eval windows, then anchors, each from documents the others left alone."""
    used: set[str] = set()
    ledger = DrawLedger(skipped_round_trip={})
    fit = draw_windows(
        documents,
        codec,
        role=ROLE_FIT,
        quotas=class_quotas(spec.n_fit, classes),
        used=used,
        width=spec.window_tokens,
        slack=spec.round_trip_slack,
        ledger=ledger,
    )
    eval_items = draw_windows(
        documents,
        codec,
        role=ROLE_EVAL,
        quotas=class_quotas(spec.n_eval, classes),
        used=used,
        width=spec.window_tokens,
        slack=spec.round_trip_slack,
        ledger=ledger,
    )
    anchors = draw_anchors(
        documents,
        codec,
        count=spec.n_anchor,
        used=used,
        max_tokens=spec.anchor_max_tokens,
        min_tokens=spec.anchor_min_tokens,
        ledger=ledger,
    )
    items = [*fit, *eval_items, *anchors]
    assert_disjoint(items)
    fit_plan = derive_max_seq_len([item.n_tokens for item in fit], ceiling=spec.window_tokens)
    anchor_plan = derive_max_seq_len(
        [item.n_tokens for item in anchors], ceiling=spec.anchor_max_tokens
    )
    sidecar: dict[str, Any] = {
        "schema": CORPUS_SCHEMA,
        "kind": CORPUS_KIND,
        **dict(provenance),
        "draw": asdict(spec),
        "window_classes": list(classes),
        "grader_class_stand_in": (
            f"{CLASS_CONCEPT_SENTENCES}: the contrastive pairs of reward_hacking.interp.stimuli "
            f"packed into windows, because no twin-corpus stimulus file was given; this is NOT "
            f"grader text"
            if CLASS_CONCEPT_SENTENCES in classes
            else None
        ),
        "n_documents": len(documents),
        "n_documents_by_source": _count_by(document.source for document in documents),
        "n_groups": len({document.group_id for document in documents}),
        "n_groups_by_source": _count_groups_by_source(documents),
        "counts": {
            ROLE_FIT: _count_by(item.window_class for item in fit),
            ROLE_EVAL: _count_by(item.window_class for item in eval_items),
            ROLE_ANCHOR: _count_by(item.source for item in anchors),
        },
        "served_weights": {
            ROLE_FIT: _count_served_by_class(fit),
            ROLE_EVAL: _count_served_by_class(eval_items),
            ROLE_ANCHOR: _count_by(item.served for item in anchors),
        },
        "shifted_windows": sum(item.shift != 0 for item in [*fit, *eval_items]),
        "skipped": ledger.as_payload(),
        "seq_len_plans": {ROLE_FIT: fit_plan.as_payload(), ROLE_ANCHOR: anchor_plan.as_payload()},
        "digests": {
            ROLE_FIT: digest_strings(item.text for item in fit),
            ROLE_EVAL: digest_strings(item.text for item in eval_items),
            ROLE_ANCHOR: digest_strings(item.text for item in anchors),
        },
    }
    logger.info(
        "drew the lens corpus, %s",
        f"fit={len(fit)} eval={len(eval_items)} anchors={len(anchors)} "
        f"fit_max_seq_len={fit_plan.max_seq_len} anchor_max_seq_len={anchor_plan.max_seq_len} "
        f"shifted={sidecar['shifted_windows']} skipped={ledger.as_payload()}",
    )
    return LensCorpus(
        fit=tuple(fit), eval=tuple(eval_items), anchors=tuple(anchors), sidecar=sidecar
    )


def _count_by(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _count_groups_by_source(documents: Iterable[Document]) -> dict[str, int]:
    groups: dict[str, set[str]] = {}
    for document in documents:
        groups.setdefault(document.source, set()).add(document.group_id)
    return {source: len(members) for source, members in sorted(groups.items())}


def _count_served_by_class(items: Iterable[CorpusItem]) -> dict[str, dict[str, int]]:
    """Per window class, whose weights wrote the windows: the check that base-reasoning IS the base's."""
    by_class: dict[str, list[str]] = {}
    for item in items:
        by_class.setdefault(item.window_class, []).append(item.served)
    return {name: _count_by(served) for name, served in sorted(by_class.items())}


_ROLE_FILES: dict[str, str] = {
    ROLE_FIT: FIT_WINDOWS_FILENAME,
    ROLE_EVAL: EVAL_WINDOWS_FILENAME,
    ROLE_ANCHOR: ANCHORS_FILENAME,
}


def write_corpus(corpus: LensCorpus, out_dir: Path) -> Path:
    """Write the three item files and the sidecar, refusing to overwrite a corpus already there."""
    sidecar_path = out_dir / SIDECAR_FILENAME
    if sidecar_path.exists():
        raise FileExistsError(
            f"{out_dir} already holds a lens corpus; a corpus is the identity every lens fitted on "
            f"it carries, so write a new directory or delete this one deliberately"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for role, items in (
        (ROLE_FIT, corpus.fit),
        (ROLE_EVAL, corpus.eval),
        (ROLE_ANCHOR, corpus.anchors),
    ):
        (out_dir / _ROLE_FILES[role]).write_text(
            "".join(json.dumps(item.as_row()) + "\n" for item in items), encoding="utf-8"
        )
    sidecar_path.write_text(json.dumps(corpus.sidecar, indent=1) + "\n", encoding="utf-8")
    reloaded = load_lens_corpus(out_dir)
    if reloaded.sidecar["digests"] != corpus.sidecar["digests"]:
        raise CorpusDrawError(f"{out_dir}: the written corpus read back with different digests")
    logger.info("wrote the lens corpus, %s", f"{out_dir=} digests={corpus.sidecar['digests']}")
    return sidecar_path


def _read_items(path: Path, role: str) -> tuple[CorpusItem, ...]:
    items: list[CorpusItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = cast("dict[str, Any]", json.loads(line))
        row.pop("n_tokens")
        item = CorpusItem(**row)
        if item.role != role:
            raise CorpusDrawError(
                f"{path}: item {item.item_id} is a {item.role} item in the {role} file"
            )
        items.append(item)
    return tuple(items)


def load_lens_corpus(out_dir: Path) -> LensCorpus:
    """Read a written corpus back and verify each role's texts against the sidecar's digests."""
    sidecar = cast("dict[str, Any]", json.loads((out_dir / SIDECAR_FILENAME).read_text()))
    if sidecar.get("kind") != CORPUS_KIND:
        raise CorpusDrawError(
            f"{out_dir} is not a {CORPUS_KIND} directory (kind={sidecar.get('kind')!r})"
        )
    if sidecar.get("schema") != CORPUS_SCHEMA:
        raise CorpusDrawError(
            f"{out_dir} is a schema-{sidecar.get('schema')!r} corpus and this reader is schema "
            f"{CORPUS_SCHEMA}: earlier draws split fit from eval at the document level, so two "
            f"renderings of one twin problem could sit on either side, and filed RL'd reasoning as "
            f"base-written; rebuild the corpus into a fresh directory rather than fitting on this one"
        )
    roles = {role: _read_items(out_dir / filename, role) for role, filename in _ROLE_FILES.items()}
    for role, items in roles.items():
        digest = digest_strings(item.text for item in items)
        if digest != sidecar["digests"][role]:
            raise CorpusDrawError(
                f"{out_dir}: the {role} texts digest to {digest[:12]} but the sidecar says "
                f"{str(sidecar['digests'][role])[:12]}; the corpus on disk is not the one drawn"
            )
    assert_disjoint([item for items in roles.values() for item in items])
    return LensCorpus(
        fit=roles[ROLE_FIT], eval=roles[ROLE_EVAL], anchors=roles[ROLE_ANCHOR], sidecar=sidecar
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def load_codec(model_id: str, revision: str | None) -> TokenCodec:
    """Load the tokenizer of record as a codec (GPU-free; the tokenizer files are cached locally)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)  # pyright: ignore[reportUnknownMemberType]
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"{model_id} loaded a slow tokenizer; the draw needs offset mappings, which only fast "
            f"tokenizers provide"
        )
    return hf_token_codec(tokenizer)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--stimuli",
        type=Path,
        default=None,
        help="the twin-corpus five-key stimulus file; without it the grader class is stood in by "
        "concept sentences and the sidecar says so",
    )
    parser.add_argument(
        "--records",
        type=Path,
        action="append",
        default=[],
        help="legibility-probe record JSONL (repeatable): completions for the tail and reasoning classes",
    )
    parser.add_argument("--base-model-id", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--window-tokens", type=int, default=WINDOW_TOKENS)
    parser.add_argument("--n-fit", type=int, default=N_FIT_WINDOWS)
    parser.add_argument("--n-eval", type=int, default=N_EVAL_WINDOWS)
    parser.add_argument("--n-anchor", type=int, default=N_ANCHOR_PROMPTS)
    parser.add_argument("--anchor-max-tokens", type=int, default=ANCHOR_MAX_TOKENS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Draw the corpus from the given sources and write it under ``--out-dir``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if not args.records:
        raise SystemExit("at least one --records file is needed for the tail and reasoning classes")
    codec = load_codec(cast("str", args.tokenizer), cast("str | None", args.tokenizer_revision))
    spec = DrawSpec(
        window_tokens=cast("int", args.window_tokens),
        n_fit=cast("int", args.n_fit),
        n_eval=cast("int", args.n_eval),
        n_anchor=cast("int", args.n_anchor),
        anchor_max_tokens=cast("int", args.anchor_max_tokens),
    )
    documents: list[Document] = []
    sources: list[dict[str, object]] = []
    stimuli_path = cast("Path | None", args.stimuli)
    if stimuli_path is not None:
        documents.extend(stimulus_documents(stimuli_path, codec))
        sources.append(
            {
                "kind": SOURCE_STIMULI,
                "path": str(stimuli_path),
                "sha256": sha256_of_file(stimuli_path),
            }
        )
        classes: Sequence[str] = WINDOW_CLASSES
    else:
        documents.extend(concept_sentence_documents(codec, window_tokens=spec.window_tokens))
        sources.append(
            {"kind": SOURCE_CONCEPT_SENTENCES, "path": "reward_hacking/interp/stimuli.py"}
        )
        classes = FALLBACK_WINDOW_CLASSES
    for path in cast("list[Path]", args.records):
        documents.extend(
            record_documents(path, codec, base_model_id=cast("str", args.base_model_id))
        )
        sources.append({"kind": "records", "path": str(path), "sha256": sha256_of_file(path)})
    corpus = draw_corpus(
        documents,
        codec,
        spec=spec,
        classes=classes,
        provenance={
            "tokenizer": {"model_id": args.tokenizer, "revision": args.tokenizer_revision},
            "base_model_id": args.base_model_id,
            "sources": sources,
        },
    )
    write_corpus(corpus, cast("Path", args.out_dir))
    logger.info("sidecar: %s", json.dumps(corpus.sidecar, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
