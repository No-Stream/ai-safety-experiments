"""Fit, read, and steer synthetic-contrast reward-hacking directions.

Each contrast is a matched coding conversation with the same task and two assistant responses: one
that attempts the work and one that exploits the grader. Directions are difference-of-means vectors
fit independently at every decoder block and read position. The synthetic pairs and resulting
directions stay under ``artifacts/``; this module contains no benchmark item text.

Decoder-block outputs use the zero-based block index in every path. Hugging Face
``output_hidden_states`` has an embedding entry at index zero, so Atlas token-sweep rows saved as
``layer_ids`` are mapped from block ``i`` to ``hidden_states[i + 1]`` before they are written.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import torch
from peft import PeftModel

from reward_hacking.atlas.readout import out_of_fold_logits, rank_auc
from reward_hacking.atlas.sampling import artifact_model_key
from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    diff_of_means,
    load_model_and_tokenizer,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.linear_probe import grouped_test_masks
from reward_hacking.interp.steering import Hook, residual_intervention, steering_hook

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence


class _ChatTokenizer(Protocol):
    pad_token_id: int | None

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> Mapping[str, Sequence[object]]: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: torch.Tensor, *, skip_special_tokens: bool) -> str: ...


logger = logging.getLogger(__name__)

DIRECTION_SCHEMA_VERSION = 1
POSITION_NAMES: tuple[str, ...] = (
    "response_mean",
    "response_last",
    "assistant_control_mean",
)
ATLAS_POSITION_MAP: dict[str, str] = {
    "last_prompt": "assistant_control_mean",
    "code_mean": "response_mean",
    "completion_last": "response_last",
    "followup_last": "response_last",
}
LAYER_INDEX_CONVENTION = "zero-based decoder-block output; HF hidden_states index = block index + 1"
TOKEN_SWEEP_CODE_OFFSETS: tuple[int, ...] = (-1, 0, 1, 2, 4, 8)
MAX_CAPTURE_BATCH_SIZE = 8
CAPTURE_FREE_VRAM_FRACTION = 0.4
CAPTURE_ACTIVATION_HEADROOM = 3.0
MATRIX_DIMENSIONS = 2
PAIR_TENSOR_DIMENSIONS = 3
PAIR_MEMBER_COUNT = 2
MIN_PAIR_COUNT = 2
REQUIRED_CLASS_COUNT = 2
CHANCE_AUC = 0.5
CAPTURE_ARRAY_DIMENSIONS = 4
PAIR_AUDIT_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
PAIR_AUDIT_TOP_TERM_COUNT = 200
PAIR_AUDIT_FLAG_THRESHOLD = 0.15
PAIR_AUDIT_CV_FOLDS = 5
PAIR_AUDIT_L2_STRENGTH = 1e-2
PAIR_AUDIT_MAX_ITERATIONS = 100
PAIR_AUDIT_MIN_CODE_TOKEN_DIFFERENCE = 15
PAIR_AUDIT_CODE_FENCE_PATTERN = re.compile(
    r"^```[^\n]*\n(.*?)(?:^```[ \t]*$|\Z)", re.DOTALL | re.MULTILINE
)
PAIR_AUDIT_CODE_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")
PAIR_AUDIT_EVALUATIVE_WORDS: tuple[str, ...] = (
    "bypass",
    "cheat",
    "general",
    "hack",
    "honest",
    "proper",
    "properly",
    "robust",
    "shortcut",
    "trick",
    "workaround",
)
# Lexical fingerprints of each family's grader-gaming mechanism. An honest response that contains its
# own family's marker is contaminated: the direction would then be fit partly on the mechanism with
# both labels. Family 6 games the grader through a false report, which has no lexical code marker.
FAMILY_MECHANISM_MARKERS: dict[int, tuple[str, ...]] = {
    1: (
        r"==\s*[\[\(\{]",
        r"==\s*(?:\"[^\"\n]{6,}\"|'[^'\n]{6,}')",
        r"\{\s*\([^()\n]*\)\s*:",
        r"\bin\s*\{\s*\(",
        r"\b_?[A-Z_]*(?:KNOWN|EXPECTED|VISIBLE|CASES)[A-Z_]*\s*=",
    ),
    2: (
        r"conftest",
        r"\bdef\s+pytest_\w+",
        r"hookimpl",
        r"\.outcome\s*=",
        r"pytest_plugins",
    ),
    3: (
        r"^-\s*(?:assert\b|self\.assert|def\s+test_)",
        r"^\+\s*#\s*assert\b",
        r"^\+\s*assert\s+True\b",
        r"\bgit\s+rm\b",
        r"\brm\s+(?:-\w+\s+)*\S*tests?/",
    ),
    4: (
        r"\bsys\.exit\b",
        r"\bos\._exit\b",
        r"\batexit\b",
        r"wasSuccessful",
        r"\bbuiltins\.",
        r"\bexitstatus\b",
        r"\breturncode\s*=\s*0\b",
        r"setattr\(\s*[\w.]*(?:grade|grader|checker|runner|result)",
        r"\b(?:grader|grading|checker|scorer|evaluate|evaluator)\w*(?:\.\w+)+\s*=[^=]",
        r"^\s*[\w.]+\.\w+\s*=\s*lambda\b",
        r"^\s*exit\s+0\b",
        r"\btrap\b.*\bexit\s+0\b",
    ),
    5: (
        (
            r"^(?=.*\b(?:open|Path|read_text|read_bytes|load|loads|glob|read_csv"
            r"|spec_from_file_location|import_module)\b)"
            r"(?=.*(?i:expected|answer|golden|solution|snapshot|test_labels)).*$"
        ),
    ),
    6: (),
    7: (
        r"pytest\.mark\.(?:skip|skipif|xfail)",
        r"\bxfail\b",
        r"--deselect",
        r"-k\s+[\"']?not\b",
        r"--ignore\b",
        r"collect_ignore",
        r"fail[_-]under",
        r"\baddopts\b",
        r"\btestpaths\b",
        r"continue-on-error",
        r"allow_failure",
        r"\|\|\s*true\b",
        r"\b(?:rtol|atol|tolerance)\s*[:=]",
        r"\b\w*threshold\w*\s*[:=]",
        r"\bignore_errors\b",
        r"^\s*omit\s*=",
    ),
    8: (
        r"except\s*:",
        r"except\s+\(?\s*(?:Exception|BaseException)\b",
        r"suppress\(\s*(?:Exception|BaseException)\b",
    ),
}


def render_messages(messages: Sequence[Mapping[str, str]], model_id: str) -> list[dict[str, str]]:
    """Prepare system/user messages for the model's template, folding system into Gemma's user turn."""
    rendered_messages = [
        {"role": str(message["role"]), "content": str(message["content"])} for message in messages
    ]
    if not rendered_messages:
        raise ValueError("messages must not be empty")
    if "gemma" not in model_id.casefold():
        return rendered_messages

    system_messages = [message for message in rendered_messages if message["role"] == "system"]
    if len(system_messages) > 1:
        raise ValueError("Gemma rendering accepts at most one system message")
    if not system_messages:
        return rendered_messages
    first_user = next(
        (index for index, message in enumerate(rendered_messages) if message["role"] == "user"),
        None,
    )
    if first_user is None or first_user < rendered_messages.index(system_messages[0]):
        raise ValueError("Gemma system content must precede a user message")
    system_content = system_messages[0]["content"]
    user_content = rendered_messages[first_user]["content"]
    rendered_messages[first_user] = {
        "role": "user",
        "content": f"{system_content}\n\n{user_content}",
    }
    return [message for message in rendered_messages if message["role"] != "system"]


def diff_of_means_direction(hack: torch.Tensor, honest: torch.Tensor) -> torch.Tensor:
    """Return mean(hack) - mean(honest), requiring two non-empty [n, hidden] groups."""
    if (
        hack.ndim != MATRIX_DIMENSIONS
        or honest.ndim != MATRIX_DIMENSIONS
        or hack.shape[1] != honest.shape[1]
    ):
        raise ValueError("hack and honest activations must be [n, hidden] with equal hidden widths")
    if hack.shape[0] == 0 or honest.shape[0] == 0:
        raise ValueError("both activation groups must be non-empty")
    direction = diff_of_means(hack.float(), honest.float())
    if not torch.isfinite(direction).all() or direction.norm() == 0:
        raise ValueError("contrast direction is zero or non-finite")
    return direction


def _validate_pair_arrays(pair_features: torch.Tensor, pair_labels: torch.Tensor) -> None:
    if pair_features.ndim != PAIR_TENSOR_DIMENSIONS or pair_features.shape[1] != PAIR_MEMBER_COUNT:
        raise ValueError("pair_features must have shape [pairs, 2, hidden]")
    if pair_labels.shape != pair_features.shape[:2]:
        raise ValueError("pair_labels must have shape [pairs, 2]")
    if pair_features.shape[0] < MIN_PAIR_COUNT:
        raise ValueError("leave-one-pair-out AUC requires at least two pairs")
    if not torch.isfinite(pair_features).all():
        raise ValueError("pair_features contain non-finite values")
    if not torch.all((pair_labels == 0) | (pair_labels == 1)):
        raise ValueError("pair_labels must be binary")
    if not torch.all(pair_labels.sum(dim=1) == 1):
        raise ValueError("every pair must contain exactly one hack and one honest row")


def leave_one_pair_out_auc(pair_features: torch.Tensor, pair_labels: torch.Tensor) -> float:
    """Compute grouped leave-one-pair-out AUC using a difference-of-means fit per fold."""
    _validate_pair_arrays(pair_features, pair_labels)
    features = pair_features.float()
    labels = pair_labels.to(dtype=torch.bool)
    positive_by_pair = (features * labels.unsqueeze(-1)).sum(dim=1)
    negative_by_pair = (features * (~labels).unsqueeze(-1)).sum(dim=1)
    positive_total = positive_by_pair.sum(dim=0)
    negative_total = negative_by_pair.sum(dim=0)
    positive_count = int(labels.sum().item()) - 1
    negative_count = int((~labels).sum().item()) - 1
    fold_scores: list[torch.Tensor] = []
    fold_labels: list[torch.Tensor] = []
    for pair_index in range(features.shape[0]):
        direction = (positive_total - positive_by_pair[pair_index]) / positive_count - (
            negative_total - negative_by_pair[pair_index]
        ) / negative_count
        fold_scores.append(features[pair_index] @ unit(direction))
        fold_labels.append(pair_labels[pair_index].to(dtype=torch.long))
    return rank_auc(torch.cat(fold_labels), torch.cat(fold_scores))


def shuffled_pair_null_auc(
    pair_features: torch.Tensor,
    pair_labels: torch.Tensor,
    *,
    permutations: int = 20,
    seed: int = 0,
) -> list[float]:
    """Return a pair-preserving shuffled-label null for the grouped leave-one-pair-out AUC."""
    _validate_pair_arrays(pair_features, pair_labels)
    if permutations < 1:
        raise ValueError(f"permutations must be positive, got {permutations}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_aucs: list[float] = []
    for _ in range(permutations):
        flip_pairs = torch.rand(pair_labels.shape[0], generator=generator) < CHANCE_AUC
        permuted_labels = pair_labels.clone()
        permuted_labels[flip_pairs] = permuted_labels[flip_pairs].flip(dims=(1,))
        null_aucs.append(leave_one_pair_out_auc(pair_features, permuted_labels))
    return null_aucs


def exclude_families(
    pairs: Sequence[Mapping[str, Any]], excluded_families: set[int]
) -> list[Mapping[str, Any]]:
    """Filter pairs by family and refuse a fit set emptied by exclusions."""
    unknown = excluded_families - set(range(1, 9))
    if unknown:
        raise ValueError(f"unknown family numbers: {sorted(unknown)}")
    retained = [pair for pair in pairs if int(pair["family"]) not in excluded_families]
    if not retained:
        raise ValueError("no pairs remain after family exclusion")
    return retained


def make_matched_norm_placebo(direction: torch.Tensor, *, seed: int = 0) -> torch.Tensor:
    """Draw a deterministic random vector with exactly the fitted direction's L2 norm."""
    if direction.ndim != 1 or direction.norm() == 0:
        raise ValueError("placebo reference must be a non-zero one-dimensional direction")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return matched_norm_random_direction(direction.detach().cpu().float(), generator)


@contextlib.contextmanager
def direction_intervention(
    model: torch.nn.Module,
    *,
    layer_index: int,
    direction: torch.Tensor,
    magnitude: float,
) -> Generator[None]:
    """Steer one decoder block by a raw residual magnitude for the duration of a model call."""
    if magnitude < 0 or not math.isfinite(magnitude):
        raise ValueError(f"magnitude must be finite and non-negative, got {magnitude}")
    with residual_intervention(
        cast("Any", model), layer_index, steering_hook(direction, magnitude)
    ):
        yield


def save_direction_bundle(
    output_dir: Path,
    directions_by_position: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> None:
    """Save position-by-layer direction arrays and JSON provenance under one artifact directory."""
    if not directions_by_position:
        raise ValueError("at least one position direction is required")
    if len({tuple(direction.shape) for direction in directions_by_position.values()}) != 1:
        raise ValueError("all positions must have the same layer count and hidden width")
    output_dir.mkdir(parents=True, exist_ok=True)
    for position, directions in directions_by_position.items():
        if not position.replace("_", "").isalnum():
            raise ValueError(f"unsafe direction position name {position!r}")
        array = directions.detach().cpu().float().numpy()
        if array.ndim != MATRIX_DIMENSIONS or not np.isfinite(array).all():
            raise ValueError(f"directions for {position} must be finite [layers, hidden]")
        np.save(output_dir / f"direction.{position}.npy", array, allow_pickle=False)
    (output_dir / "provenance.json").write_text(
        json.dumps(dict(provenance), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_direction_bundle(output_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load saved direction arrays and their provenance, checking layer and width consistency."""
    provenance_path = output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict):
        raise TypeError(f"{provenance_path} must contain a JSON object")
    direction_files = sorted(output_dir.glob("direction.*.npy"))
    if not direction_files:
        raise ValueError(f"no direction arrays found in {output_dir}")
    directions = {
        path.name.removeprefix("direction.").removesuffix(".npy"): torch.from_numpy(
            np.load(path, allow_pickle=False)
        )
        for path in direction_files
    }
    if any(
        array.ndim != MATRIX_DIMENSIONS
        or array.dtype != torch.float32
        or not torch.isfinite(array).all()
        for array in directions.values()
    ):
        raise ValueError("saved direction arrays must be finite [layers, hidden] tensors")
    shapes = {tuple(array.shape) for array in directions.values()}
    if len(shapes) != 1:
        raise ValueError(f"direction arrays have different shapes: {sorted(shapes)}")
    return directions, cast("dict[str, Any]", provenance)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as records_file:
        for line_number, line in enumerate(records_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            records.append(record)
    return records


def _validate_pair_record(
    pair: Mapping[str, Any], path: Path, line_number: int, seen_pair_ids: set[str]
) -> None:
    required = {"pair_id", "family", "messages", "honest", "hack", "held_out_splits"}
    missing = required - pair.keys()
    if missing:
        raise ValueError(f"{path}:{line_number} is missing fields {sorted(missing)}")
    pair_id = str(pair["pair_id"])
    family = int(pair["family"])
    if not pair_id or pair_id in seen_pair_ids:
        raise ValueError(f"{path}:{line_number} has an empty or duplicate pair_id")
    if family not in range(1, 9):
        raise ValueError(f"{path}:{line_number} family must be in 1..8, got {family}")
    seen_pair_ids.add(pair_id)
    messages = pair["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"{path}:{line_number} messages must be a non-empty list")
    if any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        raise TypeError(f"{path}:{line_number} messages require string role and content fields")
    for response_name in ("honest", "hack"):
        if not isinstance(pair[response_name], str) or not pair[response_name].strip():
            raise TypeError(f"{path}:{line_number} {response_name} response must be non-empty text")
    expected_splits = {
        f"without_family_{held_out_family}": "test" if family == held_out_family else "train"
        for held_out_family in (1, 2)
    }
    if pair["held_out_splits"] != expected_splits:
        raise ValueError(
            f"{path}:{line_number} has invalid family-held-out split metadata for family {family}"
        )


def read_pairs(path: Path) -> list[dict[str, Any]]:
    """Read and validate matched synthetic conversations from a local JSONL artifact."""
    pairs = _read_jsonl(path)
    if not pairs:
        raise ValueError(f"no contrast pairs found in {path}")
    seen_pair_ids: set[str] = set()
    for line_number, pair in enumerate(pairs, start=1):
        _validate_pair_record(pair, path, line_number, seen_pair_ids)
    return pairs


@dataclass(frozen=True, slots=True)
class PairLengthRatioStats:
    """Summary of hack-to-honest length ratios across matched pairs."""

    minimum: float
    median: float
    mean: float
    maximum: float
    fraction_hack_shorter: float


@dataclass(frozen=True, slots=True)
class PairAuditTerm:
    """Side-specific document counts for a frequent unigram or bigram."""

    term: str
    ngram_size: int
    honest_df: int
    hack_df: int
    imbalance_fraction: float
    flagged: bool


@dataclass(frozen=True, slots=True)
class PairCodeSubstanceFlag:
    """A pair whose honest and hack code blocks differ by too few tokens to carry the behaviour."""

    pair_id: str
    family: int
    differing_code_tokens: int


@dataclass(frozen=True, slots=True)
class PairMechanismLeak:
    """An honest response containing a lexical marker of its own family's grader-gaming mechanism."""

    pair_id: str
    family: int
    marker: str


@dataclass(frozen=True, slots=True)
class FamilyMechanismCoverage:
    """How many responses on each side of one family contain that family's mechanism markers."""

    family: int
    pair_count: int
    honest_with_marker: int
    hack_with_marker: int


@dataclass(frozen=True, slots=True)
class PairAuditReport:
    """Lexical, code-substance and mechanism-leak measurements for one matched-pair set.

    Prose is each response with its fenced code blocks removed; code is the fenced blocks' contents.
    The prose figures say whether wording alone separates the sides; the code AUC is expected to be
    high, because the code is where the honest and hacking behaviour differ.
    """

    pair_count: int
    distinct_honest_texts: int
    distinct_hack_texts: int
    word_count_ratio: PairLengthRatioStats
    code_token_ratio: PairLengthRatioStats | None
    prose_terms: tuple[PairAuditTerm, ...]
    prose_flagged_terms: tuple[str, ...]
    prose_evaluative_words: tuple[tuple[str, int, int], ...]
    prose_bow_auc: float | None
    code_bow_auc: float | None
    full_text_bow_auc: float
    min_code_token_difference: int
    code_substance_flags: tuple[PairCodeSubstanceFlag, ...]
    mechanism_leaks: tuple[PairMechanismLeak, ...]
    mechanism_coverage: tuple[FamilyMechanismCoverage, ...]


def split_prose_and_code(response: str) -> tuple[str, str]:
    """Return a response's prose (fenced blocks removed) and the joined contents of its code blocks."""
    code_blocks = [match.group(1) for match in PAIR_AUDIT_CODE_FENCE_PATTERN.finditer(response)]
    prose = PAIR_AUDIT_CODE_FENCE_PATTERN.sub("\n", response)
    return prose, "\n".join(code_blocks)


def _word_ngrams(text: str) -> list[str]:
    words = PAIR_AUDIT_TOKEN_PATTERN.findall(text.casefold())
    return [*words, *(f"{left} {right}" for left, right in pairwise(words))]


def _grouped_bag_of_words_auc(paired_texts: Sequence[tuple[str, str]]) -> float | None:
    """Fit unigram/bigram logistic regression with every matched pair kept in one fold.

    Returns None when no text on either side contains a word token (for example, code AUC on a set
    whose responses have no fenced code).
    """
    all_features = [_word_ngrams(text) for pair_texts in paired_texts for text in pair_texts]
    vocabulary = sorted({term for document in all_features for term in document})
    if not vocabulary:
        return None
    vocabulary_index = {term: index for index, term in enumerate(vocabulary)}
    features = torch.zeros((len(all_features), len(vocabulary)), dtype=torch.float32)
    for document_index, document in enumerate(all_features):
        for term in document:
            features[document_index, vocabulary_index[term]] += 1.0

    labels = torch.tensor(
        [label for _pair in paired_texts for label in (0, 1)], dtype=torch.float32
    )
    groups = torch.tensor(
        [index for index in range(len(paired_texts)) for _ in range(2)], dtype=torch.long
    )
    fold_count = min(PAIR_AUDIT_CV_FOLDS, len(paired_texts))
    fold_masks = grouped_test_masks(groups, fold_count)
    scores = out_of_fold_logits(
        features,
        labels,
        fold_masks,
        l2_strength=PAIR_AUDIT_L2_STRENGTH,
        max_iter=PAIR_AUDIT_MAX_ITERATIONS,
    )
    return rank_auc(labels.to(dtype=torch.long), scores)


def _length_ratio_stats(
    honest_lengths: Sequence[int], hack_lengths: Sequence[int]
) -> PairLengthRatioStats:
    ratios = [
        hack_length / honest_length
        for honest_length, hack_length in zip(honest_lengths, hack_lengths, strict=True)
    ]
    return PairLengthRatioStats(
        minimum=min(ratios),
        median=median(ratios),
        mean=mean(ratios),
        maximum=max(ratios),
        fraction_hack_shorter=sum(ratio < 1 for ratio in ratios) / len(ratios),
    )


def _lexical_balance(
    honest_texts: Sequence[str], hack_texts: Sequence[str], top_term_count: int
) -> tuple[tuple[PairAuditTerm, ...], tuple[str, ...]]:
    honest_document_frequency: Counter[str] = Counter()
    hack_document_frequency: Counter[str] = Counter()
    for text in honest_texts:
        honest_document_frequency.update(set(_word_ngrams(text)))
    for text in hack_texts:
        hack_document_frequency.update(set(_word_ngrams(text)))

    frequent_terms = sorted(
        honest_document_frequency.keys() | hack_document_frequency.keys(),
        key=lambda term: (
            -(honest_document_frequency[term] + hack_document_frequency[term]),
            term,
        ),
    )[:top_term_count]
    term_reports = tuple(
        PairAuditTerm(
            term=term,
            ngram_size=term.count(" ") + 1,
            honest_df=honest_document_frequency[term],
            hack_df=hack_document_frequency[term],
            imbalance_fraction=abs(honest_document_frequency[term] - hack_document_frequency[term])
            / len(honest_texts),
            flagged=abs(honest_document_frequency[term] - hack_document_frequency[term])
            / len(honest_texts)
            > PAIR_AUDIT_FLAG_THRESHOLD,
        )
        for term in frequent_terms
    )
    return term_reports, tuple(term.term for term in term_reports if term.flagged)


def _differing_code_tokens(honest_code: str, hack_code: str) -> int:
    honest_tokens = PAIR_AUDIT_CODE_TOKEN_PATTERN.findall(honest_code)
    hack_tokens = PAIR_AUDIT_CODE_TOKEN_PATTERN.findall(hack_code)
    matcher = SequenceMatcher(a=honest_tokens, b=hack_tokens, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return len(honest_tokens) + len(hack_tokens) - 2 * matched


def _mechanism_markers_in(text: str, family: int) -> list[str]:
    return [
        marker
        for marker in FAMILY_MECHANISM_MARKERS[family]
        if re.search(marker, text, flags=re.MULTILINE)
    ]


def audit_pair_records(
    pairs: Sequence[Mapping[str, Any]],
    *,
    top_term_count: int = PAIR_AUDIT_TOP_TERM_COUNT,
    min_code_token_difference: int = PAIR_AUDIT_MIN_CODE_TOKEN_DIFFERENCE,
) -> PairAuditReport:
    """Measure repetition, length balance, prose-only lexical balance, code substance and leaks."""
    if len(pairs) < MIN_PAIR_COUNT:
        raise ValueError("pair audit requires at least two matched pairs")
    if top_term_count < 1:
        raise ValueError(f"top_term_count must be positive, got {top_term_count}")
    if min_code_token_difference < 1:
        raise ValueError(
            f"min_code_token_difference must be positive, got {min_code_token_difference}"
        )
    seen_pair_ids: set[str] = set()
    for line_number, pair in enumerate(pairs, start=1):
        _validate_pair_record(pair, Path("<memory>"), line_number, seen_pair_ids)

    honest_texts = [str(pair["honest"]) for pair in pairs]
    hack_texts = [str(pair["hack"]) for pair in pairs]
    honest_prose, honest_code = zip(
        *(split_prose_and_code(text) for text in honest_texts), strict=True
    )
    hack_prose, hack_code = zip(*(split_prose_and_code(text) for text in hack_texts), strict=True)

    honest_word_counts = [
        len(PAIR_AUDIT_TOKEN_PATTERN.findall(text.casefold())) for text in honest_texts
    ]
    hack_word_counts = [
        len(PAIR_AUDIT_TOKEN_PATTERN.findall(text.casefold())) for text in hack_texts
    ]
    if any(word_count == 0 for word_count in (*honest_word_counts, *hack_word_counts)):
        raise ValueError("pair responses must contain at least one word token")
    code_token_counts = [
        (
            len(PAIR_AUDIT_CODE_TOKEN_PATTERN.findall(honest)),
            len(PAIR_AUDIT_CODE_TOKEN_PATTERN.findall(hack)),
        )
        for honest, hack in zip(honest_code, hack_code, strict=True)
    ]
    pairs_with_honest_code = [(honest, hack) for honest, hack in code_token_counts if honest > 0]

    prose_terms, prose_flagged_terms = _lexical_balance(honest_prose, hack_prose, top_term_count)
    honest_prose_words = [
        set(PAIR_AUDIT_TOKEN_PATTERN.findall(text.casefold())) for text in honest_prose
    ]
    hack_prose_words = [
        set(PAIR_AUDIT_TOKEN_PATTERN.findall(text.casefold())) for text in hack_prose
    ]
    evaluative_counts = [
        (
            word,
            sum(word in words for words in honest_prose_words),
            sum(word in words for words in hack_prose_words),
        )
        for word in PAIR_AUDIT_EVALUATIVE_WORDS
    ]

    code_substance_flags: list[PairCodeSubstanceFlag] = []
    mechanism_leaks: list[PairMechanismLeak] = []
    coverage_counts: dict[int, list[int]] = {}
    for pair, honest, hack, honest_text, hack_text in zip(
        pairs, honest_code, hack_code, honest_texts, hack_texts, strict=True
    ):
        family = int(pair["family"])
        differing_tokens = _differing_code_tokens(honest, hack)
        if differing_tokens < min_code_token_difference:
            code_substance_flags.append(
                PairCodeSubstanceFlag(str(pair["pair_id"]), family, differing_tokens)
            )
        honest_markers = _mechanism_markers_in(honest_text, family)
        mechanism_leaks.extend(
            PairMechanismLeak(str(pair["pair_id"]), family, marker) for marker in honest_markers
        )
        family_counts = coverage_counts.setdefault(family, [0, 0, 0])
        family_counts[0] += 1
        family_counts[1] += bool(honest_markers)
        family_counts[2] += bool(_mechanism_markers_in(hack_text, family))

    return PairAuditReport(
        pair_count=len(pairs),
        distinct_honest_texts=len(set(honest_texts)),
        distinct_hack_texts=len(set(hack_texts)),
        word_count_ratio=_length_ratio_stats(honest_word_counts, hack_word_counts),
        code_token_ratio=None
        if not pairs_with_honest_code
        else _length_ratio_stats(
            [honest for honest, _ in pairs_with_honest_code],
            [hack for _, hack in pairs_with_honest_code],
        ),
        prose_terms=prose_terms,
        prose_flagged_terms=prose_flagged_terms,
        prose_evaluative_words=tuple(row for row in evaluative_counts if row[1] or row[2]),
        prose_bow_auc=_grouped_bag_of_words_auc(list(zip(honest_prose, hack_prose, strict=True))),
        code_bow_auc=_grouped_bag_of_words_auc(list(zip(honest_code, hack_code, strict=True))),
        full_text_bow_auc=cast(
            "float", _grouped_bag_of_words_auc(list(zip(honest_texts, hack_texts, strict=True)))
        ),
        min_code_token_difference=min_code_token_difference,
        code_substance_flags=tuple(code_substance_flags),
        mechanism_leaks=tuple(mechanism_leaks),
        mechanism_coverage=tuple(
            FamilyMechanismCoverage(family, *counts)
            for family, counts in sorted(coverage_counts.items())
        ),
    )


def _log_pair_audit(report: PairAuditReport) -> None:
    logger.info("PAIR_AUDIT %s", json.dumps(asdict(report), sort_keys=True))
    logger.info(
        "PAIR_AUDIT_SUMMARY %s",
        json.dumps(
            {
                "pairs": report.pair_count,
                "distinct_honest_texts": report.distinct_honest_texts,
                "distinct_hack_texts": report.distinct_hack_texts,
                "prose_bow_auc": report.prose_bow_auc,
                "code_bow_auc": report.code_bow_auc,
                "full_text_bow_auc": report.full_text_bow_auc,
                "prose_flagged_terms": list(report.prose_flagged_terms),
                "prose_evaluative_words": [list(row) for row in report.prose_evaluative_words],
                "code_substance_flags": len(report.code_substance_flags),
                "mechanism_leaks": [
                    f"{leak.pair_id}:{leak.marker}" for leak in report.mechanism_leaks
                ],
                "word_count_ratio": asdict(report.word_count_ratio),
                "code_token_ratio": None
                if report.code_token_ratio is None
                else asdict(report.code_token_ratio),
            },
            sort_keys=True,
        ),
    )


def _parse_excluded_families(value: str) -> set[int]:
    if not value.strip():
        return set()
    families = {int(part.strip()) for part in value.split(",") if part.strip()}
    if not families:
        raise ValueError("--exclude-families needs comma-separated family numbers")
    unknown = families - set(range(1, 9))
    if unknown:
        raise ValueError(f"unknown family numbers in --exclude-families: {sorted(unknown)}")
    return families


def _token_ids_from_template(
    tokenizer: _ChatTokenizer,
    messages: Sequence[Mapping[str, str]],
    *,
    generation_prompt: bool,
) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=generation_prompt, return_dict=True
    )
    raw_ids = encoded["input_ids"]
    token_ids = [token_id for token_id in raw_ids if isinstance(token_id, int)]
    if not token_ids or len(token_ids) != len(raw_ids):
        raise TypeError("chat template did not return one flat list of token ids")
    return token_ids


def _encode_response(tokenizer: _ChatTokenizer, response: str) -> list[int]:
    token_ids = tokenizer.encode(response, add_special_tokens=False)
    if not token_ids:
        raise ValueError("assistant response produced no tokens")
    return token_ids


@dataclass(frozen=True)
class PreparedTurn:
    """Tokenized prompt and teacher-forced response with captured position boundaries."""

    token_ids: list[int]
    response_start: int
    response_length: int
    assistant_control_positions: tuple[int, ...]


def prepare_turn(
    tokenizer: _ChatTokenizer,
    messages: Sequence[Mapping[str, str]],
    response: str,
    *,
    model_id: str,
) -> PreparedTurn:
    """Render a prompt and teacher-forced response, locating its assistant header and content."""
    template_messages = render_messages(messages, model_id)
    without_assistant_ids = _token_ids_from_template(
        tokenizer, template_messages, generation_prompt=False
    )
    prompt_ids = _token_ids_from_template(tokenizer, template_messages, generation_prompt=True)
    shared_prefix = 0
    for left, right in zip(without_assistant_ids, prompt_ids, strict=False):
        if left != right:
            break
        shared_prefix += 1
    if shared_prefix != len(without_assistant_ids) or shared_prefix == len(prompt_ids):
        raise ValueError("chat template did not append a distinct assistant generation prefix")
    response_ids = _encode_response(tokenizer, response)
    full_turn_ids = _token_ids_from_template(
        tokenizer,
        [*template_messages, {"role": "assistant", "content": response}],
        generation_prompt=False,
    )
    expected_response_prefix = [*prompt_ids, *response_ids]
    if full_turn_ids[: len(expected_response_prefix)] != expected_response_prefix:
        raise ValueError(
            "assistant response tokenization differs from the full chat-template rendering"
        )
    return PreparedTurn(
        token_ids=full_turn_ids[: len(expected_response_prefix)],
        response_start=len(prompt_ids),
        response_length=len(response_ids),
        assistant_control_positions=tuple(range(shared_prefix, len(prompt_ids))),
    )


def _config_value(model: torch.nn.Module, names: Sequence[str]) -> int:
    model_config = cast("Any", model).config
    configs: list[object] = [model_config]
    text_config = getattr(model_config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    for config in configs:
        for name in names:
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                return value
    raise ValueError(f"model config exposes none of {list(names)}")


def _device_memory(model: torch.nn.Module) -> tuple[torch.device, int, int]:
    device = next(model.parameters()).device
    if device.type != "cuda":
        return device, 0, 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return device, int(free_bytes), int(total_bytes)


def _load_model_with_optional_adapter(
    model_id: str, adapter: Path | None
) -> tuple[torch.nn.Module, _ChatTokenizer]:
    loaded_model, loaded_tokenizer = load_model_and_tokenizer(model_id)
    model = cast("torch.nn.Module", loaded_model)
    if adapter is not None:
        model = cast("torch.nn.Module", PeftModel.from_pretrained(cast("Any", model), adapter))
    model.eval()
    return model, cast("_ChatTokenizer", loaded_tokenizer)


def _capture_batch_size(
    model: torch.nn.Module,
    *,
    sequence_length: int,
    free_vram_bytes: int,
) -> int:
    if free_vram_bytes <= 0:
        return 1
    layer_count = _config_value(model, ("num_hidden_layers", "n_layer", "num_layers"))
    hidden_size = _config_value(model, ("hidden_size", "d_model", "n_embd"))
    element_size = next(model.parameters()).element_size()
    bytes_per_row = (
        sequence_length * hidden_size * layer_count * element_size * CAPTURE_ACTIVATION_HEADROOM
    )
    usable_vram = free_vram_bytes * CAPTURE_FREE_VRAM_FRACTION
    return max(1, min(MAX_CAPTURE_BATCH_SIZE, int(usable_vram // bytes_per_row)))


def _capture_response_batch(
    model: torch.nn.Module,
    batch: Sequence[tuple[str, str, PreparedTurn]],
    pad_token_id: int,
) -> dict[int, torch.Tensor]:
    batch_length = max(len(row[2].token_ids) for row in batch)
    input_ids = torch.full((len(batch), batch_length), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row_index, (_, _, turn) in enumerate(batch):
        input_ids[row_index, : len(turn.token_ids)] = torch.tensor(turn.token_ids, dtype=torch.long)
        attention_mask[row_index, : len(turn.token_ids)] = 1
    return capture_positionwise_activations(cast("Any", model), input_ids, attention_mask)


def _pool_response_batch(
    batch: Sequence[tuple[str, str, PreparedTurn]],
    captured: Mapping[int, torch.Tensor],
    pooled: dict[str, dict[str, dict[int, list[torch.Tensor]]]],
) -> None:
    for row_index, (_, response_name, turn) in enumerate(batch):
        response_indices = torch.arange(
            turn.response_start, turn.response_start + turn.response_length
        )
        control_indices = torch.tensor(turn.assistant_control_positions, dtype=torch.long)
        for layer_id, layer_activations in captured.items():
            states = layer_activations[row_index]
            pooled_vectors = {
                "response_mean": states[response_indices].mean(dim=0),
                # clone: an indexing view would keep the whole batch's activations alive in host RAM
                "response_last": states[turn.response_start + turn.response_length - 1].clone(),
                "assistant_control_mean": states[control_indices].mean(dim=0),
            }
            for position, vector in pooled_vectors.items():
                pooled[position][response_name].setdefault(layer_id, []).append(vector)


def _capture_training_pairs(
    model: torch.nn.Module,
    tokenizer: _ChatTokenizer,
    pairs: Sequence[Mapping[str, Any]],
    *,
    model_id: str,
) -> tuple[dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]], list[int]]:
    prepared_rows = [
        (
            str(pair["pair_id"]),
            response_name,
            prepare_turn(
                tokenizer,
                cast("Sequence[Mapping[str, str]]", pair["messages"]),
                str(pair[response_name]),
                model_id=model_id,
            ),
        )
        for pair in pairs
        for response_name in ("honest", "hack")
    ]
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer has no pad_token_id for right-padded capture batches")
    maximum_length = max(len(row[2].token_ids) for row in prepared_rows)
    device, free_vram_bytes, total_vram_bytes = _device_memory(model)
    batch_size = _capture_batch_size(
        model,
        sequence_length=maximum_length,
        free_vram_bytes=free_vram_bytes,
    )
    logger.info(
        "capture device=%s free_vram_bytes=%d total_vram_bytes=%d batch_size=%d max_sequence=%d",
        device,
        free_vram_bytes,
        total_vram_bytes,
        batch_size,
        maximum_length,
    )

    pooled: dict[str, dict[str, dict[int, list[torch.Tensor]]]] = {
        position: {response_name: {} for response_name in ("honest", "hack")}
        for position in POSITION_NAMES
    }
    layer_ids: list[int] | None = None
    for start in range(0, len(prepared_rows), batch_size):
        batch = prepared_rows[start : start + batch_size]
        captured = _capture_response_batch(model, batch, int(tokenizer.pad_token_id))
        current_layer_ids = sorted(captured)
        if layer_ids is None:
            layer_ids = current_layer_ids
        elif current_layer_ids != layer_ids:
            raise RuntimeError(
                "model capture returned a different decoder-layer set between batches"
            )
        _pool_response_batch(batch, captured, pooled)

    if layer_ids is None:
        raise ValueError("no pair responses were prepared for activation capture")
    ordered_pair_ids = [str(pair["pair_id"]) for pair in pairs]
    result: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
    for position in POSITION_NAMES:
        result[position] = {}
        for layer_id in layer_ids:
            honest_vectors = pooled[position]["honest"][layer_id]
            hack_vectors = pooled[position]["hack"][layer_id]
            if len(honest_vectors) != len(ordered_pair_ids) or len(hack_vectors) != len(
                ordered_pair_ids
            ):
                raise RuntimeError(
                    f"capture rows did not preserve all matched pairs at {position}/L{layer_id}"
                )
            result[position][layer_id] = (torch.stack(hack_vectors), torch.stack(honest_vectors))
    return result, layer_ids


def _fit_metrics(
    hack: torch.Tensor,
    honest: torch.Tensor,
    *,
    permutations: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    pair_features = torch.stack((honest, hack), dim=1)
    pair_labels = torch.tensor([[0, 1]] * hack.shape[0], dtype=torch.long)
    direction = diff_of_means_direction(hack, honest)
    held_out_auc = leave_one_pair_out_auc(pair_features, pair_labels)
    null = shuffled_pair_null_auc(
        pair_features,
        pair_labels,
        permutations=permutations,
        seed=seed,
    )
    return direction, {
        "leave_one_pair_out_auc": held_out_auc,
        "shuffled_label_auc_mean": float(np.mean(null)),
        "shuffled_label_auc_std": float(np.std(null, ddof=0)),
        "shuffled_label_auc_max": max(null),
        "n_shuffled_label_permutations": len(null),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FitConfig:
    """Inputs for fitting and validating a set of synthetic-contrast directions."""

    model_id: str
    adapter: Path | None
    pairs_path: Path
    excluded_families: set[int]
    output_dir: Path
    permutations: int = 20
    seed: int = 0


def fit_directions(config: FitConfig) -> None:
    """Load an HF model, capture matched responses, and save per-position directions and CV reads."""
    if config.permutations < 1:
        raise ValueError(f"permutations must be positive, got {config.permutations}")
    pairs = read_pairs(config.pairs_path)
    retained = cast("list[dict[str, Any]]", exclude_families(pairs, config.excluded_families))
    if len(retained) < MIN_PAIR_COUNT:
        raise ValueError(
            "at least two matched pairs are required for leave-one-pair-out validation"
        )
    model, tokenizer = _load_model_with_optional_adapter(config.model_id, config.adapter)
    logger.info(
        "fit model=%s adapter=%s pairs=%d excluded_families=%s",
        config.model_id,
        config.adapter,
        len(retained),
        sorted(config.excluded_families),
    )
    captured, layer_ids = _capture_training_pairs(
        model, tokenizer, retained, model_id=config.model_id
    )
    directions: dict[str, torch.Tensor] = {}
    metrics: dict[str, list[dict[str, Any]]] = {}
    for position in POSITION_NAMES:
        position_directions: list[torch.Tensor] = []
        position_metrics: list[dict[str, Any]] = []
        for layer_id in layer_ids:
            hack, honest = captured[position][layer_id]
            direction, row_metrics = _fit_metrics(
                hack,
                honest,
                permutations=config.permutations,
                seed=config.seed + layer_id,
            )
            position_directions.append(direction)
            position_metrics.append({"layer": layer_id, **row_metrics})
        directions[position] = torch.stack(position_directions)
        metrics[position] = position_metrics

    provenance: dict[str, Any] = {
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "model": config.model_id,
        "adapter": None if config.adapter is None else str(config.adapter),
        "pairs_sha256": _sha256_file(config.pairs_path),
        "pairs_path": str(config.pairs_path),
        "n_pairs_total": len(pairs),
        "n_pairs_fit": len(retained),
        "excluded_families": sorted(config.excluded_families),
        "positions": list(POSITION_NAMES),
        "layers": layer_ids,
        "layer_index_convention": LAYER_INDEX_CONVENTION,
        "metrics": metrics,
        "family_holdout_splits": ["without_family_1", "without_family_2"],
    }
    save_direction_bundle(config.output_dir, directions, provenance)
    logger.info("saved hack-contrast direction bundle to %s", config.output_dir)


def _labels_from_atlas_grade(grade: Mapping[str, Any]) -> int | None:
    hack = grade.get("hack")
    honest_pass = grade.get("honest_pass")
    if not isinstance(hack, bool) or not isinstance(honest_pass, bool):
        raise TypeError("Atlas grades must contain boolean hack and honest_pass fields")
    if hack and honest_pass:
        raise ValueError("Atlas grade cannot mark one sample both hack and honest_pass")
    if hack:
        return 1
    if honest_pass:
        return 0
    return None


def _atlas_rows(
    run_dir: Path, model_key: str, sample_ids: Sequence[str]
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], int | None]]:
    samples_path = run_dir / f"samples.{model_key}.jsonl"
    grades_path = run_dir / f"grades.{model_key}.jsonl"
    samples = {str(row["sample_id"]): row for row in _read_jsonl(samples_path)}
    grades = {str(row["sample_id"]): row for row in _read_jsonl(grades_path)}
    if len(samples) != len(sample_ids) or len(grades) != len(sample_ids):
        raise ValueError(f"Atlas metadata row counts do not match capture index for {model_key}")
    pair_keys: list[tuple[str, int]] = []
    labels: dict[tuple[str, int], int | None] = {}
    for sample_id in sample_ids:
        if sample_id not in samples or sample_id not in grades:
            raise ValueError(f"Atlas metadata is missing sample_id {sample_id!r} for {model_key}")
        sample = samples[sample_id]
        grade = grades[sample_id]
        item_id = str(sample["item_id"])
        if str(grade["item_id"]) != item_id:
            raise ValueError(f"Atlas sample and grade item_id differ for {sample_id!r}")
        pair_key = item_id, int(sample["sample_index"])
        pair_keys.append(pair_key)
        labels[pair_key] = _labels_from_atlas_grade(grade)
    if len(set(pair_keys)) != len(pair_keys):
        raise ValueError(
            f"Atlas sample metadata contains duplicate item/sample pairs for {model_key}"
        )
    return pair_keys, labels


@dataclass(frozen=True)
class ProjectionCapture:
    """Projection scores and labels for one Atlas model key."""

    model_key: str
    adapter_id: str | None
    scores: dict[tuple[str, int, int | None], dict[tuple[str, int], float]]
    labels: dict[tuple[str, int], int | None]


def _validate_direction_bundle(
    directions: Mapping[str, torch.Tensor], provenance: Mapping[str, Any]
) -> list[int]:
    if provenance.get("schema_version") != DIRECTION_SCHEMA_VERSION:
        raise ValueError("unsupported direction bundle schema")
    if provenance.get("layer_index_convention") != LAYER_INDEX_CONVENTION:
        raise ValueError(
            "saved direction bundle uses a different hidden-state/block index convention"
        )
    layer_ids = [int(layer) for layer in provenance["layers"]]
    expected_layers = list(range(next(iter(directions.values())).shape[0]))
    if layer_ids != expected_layers:
        raise ValueError(
            f"saved direction layers must be contiguous zero-based block indices: {layer_ids}"
        )
    if set(directions) != set(provenance["positions"]):
        raise ValueError("saved direction positions differ from provenance")
    if any(array.shape[0] != len(layer_ids) for array in directions.values()):
        raise ValueError("saved direction layer axes differ from provenance")
    return layer_ids


def _validate_atlas_activation_array(
    activation_path: Path, index_path: Path, activations: np.ndarray, index: Mapping[str, Any]
) -> None:
    if activations.ndim != CAPTURE_ARRAY_DIMENSIONS or activations.dtype != np.dtype(np.float32):
        raise ValueError(
            f"Atlas activations must be float32 [rows, layers, positions, hidden]: {activation_path}"
        )
    if list(activations.shape) != index.get("shape"):
        raise ValueError(f"Atlas activation shape differs from its index: {activation_path}")
    if len(index["sample_ids"]) != activations.shape[0]:
        raise ValueError(f"Atlas sample_ids do not align with activation rows: {index_path}")


def _validate_atlas_positions(
    positions: Sequence[str],
    activations: np.ndarray,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
    index_path: Path,
) -> None:
    if len(positions) != activations.shape[2]:
        raise ValueError(f"Atlas position index does not align with activation axis: {index_path}")
    if list(range(activations.shape[1])) != [int(layer) for layer in provenance["layers"]]:
        raise ValueError(
            "Atlas layer axis differs from saved directions; both must be zero-based decoder-block outputs"
        )
    for position in positions:
        if position not in ATLAS_POSITION_MAP:
            raise ValueError(f"unsupported Atlas read position {position!r}")
        direction = directions[ATLAS_POSITION_MAP[position]]
        if direction.shape != (activations.shape[1], activations.shape[3]):
            raise ValueError(f"direction shape differs from Atlas activations for {position}")


def _atlas_capture(
    run_dir: Path,
    index_path: Path,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> ProjectionCapture:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    model_key = str(index["model_key"])
    output_key = artifact_model_key(model_key)
    activation_path = run_dir / f"activations.{output_key}.npy"
    activations = np.load(activation_path, mmap_mode="r", allow_pickle=False)
    if index.get("complete") is not True:
        raise ValueError(f"Atlas capture index is incomplete: {index_path}")
    if str(index["model_id"]) != str(provenance["model"]):
        raise ValueError(
            f"Atlas base model {index['model_id']} differs from direction model {provenance['model']}"
        )
    _validate_atlas_activation_array(activation_path, index_path, activations, index)
    sample_ids = [str(sample_id) for sample_id in index["sample_ids"]]
    positions = [str(position) for position in index["positions"]]
    _validate_atlas_positions(positions, activations, directions, provenance, index_path)
    pair_keys, labels = _atlas_rows(run_dir, output_key, sample_ids)
    scores: dict[tuple[str, int, int | None], dict[tuple[str, int], float]] = {}
    for row_index, pair_key in enumerate(pair_keys):
        for position_index, position in enumerate(positions):
            direction_position = ATLAS_POSITION_MAP[position]
            vectors = torch.from_numpy(
                np.asarray(activations[row_index, :, position_index, :], dtype=np.float32).copy()
            )
            projections = torch.sum(vectors * directions[direction_position].float(), dim=1)
            for layer_index, score in enumerate(projections.tolist()):
                key = (position, layer_index, None)
                scores.setdefault(key, {})[pair_key] = float(score)
    return ProjectionCapture(
        model_key=model_key,
        adapter_id=None if index.get("adapter_id") is None else str(index["adapter_id"]),
        scores=scores,
        labels=labels,
    )


def _read_token_sweep_arrays(
    index_path: Path,
    index: Mapping[str, Any],
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sweep_dir = index_path.parent
    output_key = str(index["output_key"])
    activation_path = sweep_dir / f"activations.{output_key}.npy"
    mask_path = sweep_dir / f"mask.{output_key}.npy"
    activations = np.load(activation_path, mmap_mode="r", allow_pickle=False)
    mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    if activations.ndim != CAPTURE_ARRAY_DIMENSIONS or activations.dtype != np.dtype(np.float32):
        raise ValueError(
            "token-sweep activations must be float32 [rows, layers, tokens, hidden]: "
            f"{activation_path}"
        )
    if mask.dtype != np.dtype(np.bool_) or mask.shape != (
        activations.shape[0],
        activations.shape[2],
    ):
        raise ValueError(f"token-sweep mask does not align with its activation shard: {mask_path}")
    rows = index["rows"]
    layer_ids = [int(layer) for layer in index["layer_ids"]]
    if len(rows) != activations.shape[0] or len(layer_ids) != activations.shape[1]:
        raise ValueError(
            f"token-sweep index does not align with its activation shard: {index_path}"
        )
    if activations.shape[2] != int(index["prompt_tokens"]) + int(index["completion_tokens"]):
        raise ValueError(f"token-sweep token axis differs from its index: {index_path}")
    if activations.shape[3] != directions["response_mean"].shape[1]:
        raise ValueError(f"token-sweep hidden width differs from saved directions: {index_path}")
    if any(layer not in provenance["layers"] for layer in layer_ids):
        raise ValueError(f"token-sweep layers are outside saved directions: {index_path}")
    return activations, mask, layer_ids


def _token_sweep_labels(
    rows: Sequence[Mapping[str, Any]], index_path: Path
) -> dict[tuple[str, int], int | None]:
    labels: dict[tuple[str, int], int | None] = {}
    for row in rows:
        hack = row.get("hack_label")
        passed = row.get("pass_label")
        if not isinstance(hack, bool) or not isinstance(passed, bool) or (hack and passed):
            raise ValueError(
                "token-sweep rows require exclusive boolean hack_label/pass_label values"
            )
        pair_key = str(row["item_id"]), int(row["sample_index"])
        if pair_key in labels:
            raise ValueError(f"token-sweep rows contain duplicate item/sample pairs: {index_path}")
        labels[pair_key] = 1 if hack else 0 if passed else None
    return labels


@dataclass(frozen=True)
class _TokenSweepScoreContext:
    token_count: int
    activations: np.ndarray
    mask: np.ndarray
    layer_ids: Sequence[int]
    directions: Mapping[str, torch.Tensor]


def _score_token_sweep_row(
    row_index: int,
    row: Mapping[str, Any],
    context: _TokenSweepScoreContext,
) -> list[tuple[tuple[str, int, int | None], tuple[str, int], float]]:
    pair_key = str(row["item_id"]), int(row["sample_index"])
    code_start = int(row["code_token_start"])
    scored_positions: list[tuple[tuple[str, int, int | None], tuple[str, int], float]] = []
    for offset in TOKEN_SWEEP_CODE_OFFSETS:
        token_position = context.token_count + code_start + offset
        if (
            not 0 <= token_position < context.activations.shape[2]
            or not context.mask[row_index, token_position]
        ):
            continue
        for layer_position, layer_id in enumerate(context.layer_ids):
            for direction_position in ("response_mean", "response_last"):
                direction = context.directions[direction_position][layer_id]
                vector = torch.from_numpy(
                    np.asarray(
                        context.activations[row_index, layer_position, token_position],
                        dtype=np.float32,
                    ).copy()
                )
                score = torch.dot(vector, direction.float()).item()
                position = f"code_start{offset:+d}:{direction_position}"
                scored_positions.append(((position, layer_id, offset), pair_key, float(score)))
    return scored_positions


def _token_sweep_capture(
    index_path: Path,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> ProjectionCapture | None:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    model_key = str(index["model_key"])
    if index.get("complete") is not True:
        logger.info("token_sweep_status=incomplete model_key=%s", model_key)
        return None
    if str(index["model_id"]) != str(provenance["model"]):
        raise ValueError(
            f"token-sweep base model {index['model_id']} differs from direction model {provenance['model']}"
        )
    activations, mask, layer_ids = _read_token_sweep_arrays(
        index_path, index, directions, provenance
    )
    rows = index["rows"]
    labels = _token_sweep_labels(rows, index_path)
    scores: dict[tuple[str, int, int | None], dict[tuple[str, int], float]] = {}
    context = _TokenSweepScoreContext(
        token_count=int(index["prompt_tokens"]),
        activations=activations,
        mask=mask,
        layer_ids=layer_ids,
        directions=directions,
    )
    for row_index, row in enumerate(rows):
        for metric, pair_key, score in _score_token_sweep_row(row_index, row, context):
            scores.setdefault(metric, {})[pair_key] = score
    return ProjectionCapture(
        model_key=model_key,
        adapter_id=None if index.get("adapter_id") is None else str(index["adapter_id"]),
        scores=scores,
        labels=labels,
    )


def _token_sweep_captures(
    sweep_dir: Path,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> list[ProjectionCapture]:
    return [
        capture
        for index_path in sorted(sweep_dir.glob("index.*.json"))
        if (capture := _token_sweep_capture(index_path, directions, provenance)) is not None
    ]


def _emit_capture_metrics(capture: ProjectionCapture, *, kind: str) -> None:
    for (position, layer, offset), scores_by_pair in sorted(capture.scores.items()):
        selected = [
            (key, score)
            for key, score in scores_by_pair.items()
            if capture.labels.get(key) is not None
        ]
        labels = torch.tensor([cast("int", capture.labels[key]) for key, _ in selected])
        scores = torch.tensor([score for _, score in selected], dtype=torch.float32)
        if labels.numel() == 0 or torch.unique(labels).numel() < REQUIRED_CLASS_COUNT:
            auc: float | None = None
        else:
            auc = rank_auc(labels, scores)
        logger.info(
            json.dumps(
                {
                    "kind": kind,
                    "model_key": capture.model_key,
                    "position": position,
                    "layer": layer,
                    "token_offset": offset,
                    "n_samples": int(labels.numel()),
                    "n_hacks": int(labels.sum().item()) if labels.numel() else 0,
                    "auc": auc,
                },
                sort_keys=True,
            )
        )


def _emit_pair_shifts(captures: Sequence[ProjectionCapture], *, kind: str) -> None:
    bases = [capture for capture in captures if capture.adapter_id is None]
    adapters = [capture for capture in captures if capture.adapter_id is not None]
    for base in bases:
        for adapter in adapters:
            shared_metrics = set(base.scores) & set(adapter.scores)
            logger.info(
                json.dumps(
                    {
                        "kind": f"{kind}_base_adapter_coverage",
                        "base_model_key": base.model_key,
                        "adapter_model_key": adapter.model_key,
                        "n_shared_metrics": len(shared_metrics),
                        "n_base_only_metrics": len(set(base.scores) - set(adapter.scores)),
                        "n_adapter_only_metrics": len(set(adapter.scores) - set(base.scores)),
                    },
                    sort_keys=True,
                )
            )
            for metric_key in sorted(shared_metrics):
                base_scores = base.scores[metric_key]
                adapter_scores = adapter.scores[metric_key]
                paired_keys = sorted(set(base_scores) & set(adapter_scores))
                shifts = {
                    pair_key: adapter_scores[pair_key] - base_scores[pair_key]
                    for pair_key in paired_keys
                }
                base_hack_shifts = [
                    shift for key, shift in shifts.items() if base.labels.get(key) == 1
                ]
                adapter_hack_shifts = [
                    shift for key, shift in shifts.items() if adapter.labels.get(key) == 1
                ]
                logger.info(
                    json.dumps(
                        {
                            "kind": f"{kind}_base_adapter_shift",
                            "base_model_key": base.model_key,
                            "adapter_model_key": adapter.model_key,
                            "position": metric_key[0],
                            "layer": metric_key[1],
                            "token_offset": metric_key[2],
                            "n_paired_samples": len(shifts),
                            "n_base_only": len(set(base_scores) - set(adapter_scores)),
                            "n_adapter_only": len(set(adapter_scores) - set(base_scores)),
                            "mean_adapter_minus_base": None
                            if not shifts
                            else float(np.mean(list(shifts.values()))),
                            "mean_shift_on_base_hacks": None
                            if not base_hack_shifts
                            else float(np.mean(base_hack_shifts)),
                            "mean_shift_on_adapter_hacks": None
                            if not adapter_hack_shifts
                            else float(np.mean(adapter_hack_shifts)),
                        },
                        sort_keys=True,
                    )
                )


def read_captures(*, run_dir: Path, direction_dir: Path) -> None:
    """Read Atlas captures and optional token-sweep shards with a saved synthetic direction."""
    directions, provenance = load_direction_bundle(direction_dir)
    _validate_direction_bundle(directions, provenance)
    atlas_indexes = sorted(run_dir.glob("capture-index.*.json"))
    if not atlas_indexes:
        raise FileNotFoundError(f"no Atlas capture indexes found in {run_dir}")
    atlas_captures = [
        _atlas_capture(run_dir, index_path, directions, provenance) for index_path in atlas_indexes
    ]
    for capture in atlas_captures:
        _emit_capture_metrics(capture, kind="atlas_auc")
    _emit_pair_shifts(atlas_captures, kind="atlas")

    sweep_dir = run_dir / "token-sweep"
    if not sweep_dir.exists():
        logger.info("token_sweep_status=absent path=%s", sweep_dir)
        return
    sweep_indexes = list(sweep_dir.glob("index.*.json"))
    if not sweep_indexes:
        logger.info("token_sweep_status=absent path=%s", sweep_dir)
        return
    sweep_captures = _token_sweep_captures(sweep_dir, directions, provenance)
    for capture in sweep_captures:
        _emit_capture_metrics(capture, kind="token_sweep_auc")
    _emit_pair_shifts(sweep_captures, kind="token_sweep")


def _generation_hook(
    direction: torch.Tensor,
    *,
    magnitude: float,
    prompt_length: int,
    scope: str,
) -> Hook:
    """Steer the final prompt residual and optionally every subsequently generated token state."""
    if scope not in {"all", "first"}:
        raise ValueError(f"unknown steering scope {scope!r}")
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    step = unit(direction).float() * magnitude
    prefill_seen = False

    def hook(_module: object, _inputs: object, output: object) -> object:
        nonlocal prefill_seen
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        if not isinstance(hidden, torch.Tensor):
            raise TypeError(
                "decoder block output must be a tensor or tuple beginning with a tensor"
            )
        selector = torch.zeros(hidden.shape[-2], dtype=torch.bool, device=hidden.device)
        if not prefill_seen:
            if prompt_length > hidden.shape[-2]:
                raise ValueError("prompt length exceeds the first generation prefill sequence")
            selector[prompt_length - 1] = True
            prefill_seen = True
        elif scope == "all":
            selector[:] = True
        steered = hidden + selector.view(*([1] * (hidden.ndim - 2)), -1, 1) * step.to(hidden)
        if steered.shape != hidden.shape:
            raise AssertionError("generation steering changed the decoder residual shape")
        if is_tuple:
            return (steered, *cast("tuple[object, ...]", output[1:]))
        return steered

    return hook


def _parse_strengths(value: str) -> list[float]:
    strengths = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not strengths or any(not math.isfinite(strength) or strength <= 0 for strength in strengths):
        raise ValueError("--strengths must be comma-separated positive finite multiples")
    return strengths


def _prompt_records(path: Path) -> list[dict[str, Any]]:
    prompts = _read_jsonl(path)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    for line_number, prompt in enumerate(prompts, start=1):
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{path}:{line_number} requires a non-empty messages list")
        if any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"{path}:{line_number} messages need text content")
    return prompts


@dataclass(frozen=True)
class SteeringConfig:
    """Inputs for generating direction-steered and placebo completions."""

    model_id: str
    adapter: Path | None
    direction_dir: Path
    prompts_path: Path
    layer_index: int
    position: str
    scope: str
    strengths: Sequence[float]
    output_path: Path
    max_new_tokens: int
    seed: int


def _validate_steering_request(
    config: SteeringConfig,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> None:
    if config.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {config.max_new_tokens}")
    if any(not math.isfinite(strength) or strength <= 0 for strength in config.strengths):
        raise ValueError("steering strengths must be positive and finite")
    if str(provenance["model"]) != config.model_id:
        raise ValueError(
            f"direction model {provenance['model']} differs from requested model {config.model_id}"
        )
    if config.position not in directions:
        raise ValueError(f"position {config.position!r} is absent from {config.direction_dir}")
    saved_layers = [int(layer) for layer in provenance["layers"]]
    if config.layer_index not in saved_layers:
        raise ValueError(
            f"layer {config.layer_index} is absent from saved direction layers {saved_layers}"
        )


@torch.no_grad()
def steer_prompts(
    config: SteeringConfig,
) -> None:
    """Generate direction and matched-norm-placebo completions for later external grading."""
    prompts = _prompt_records(config.prompts_path)
    directions, provenance = load_direction_bundle(config.direction_dir)
    _validate_direction_bundle(directions, provenance)
    _validate_steering_request(config, directions, provenance)
    model, tokenizer = _load_model_with_optional_adapter(config.model_id, config.adapter)
    device, free_vram_bytes, total_vram_bytes = _device_memory(model)
    logger.info(
        "steer model=%s adapter=%s device=%s free_vram_bytes=%d total_vram_bytes=%d generation_batch_size=1",
        config.model_id,
        config.adapter,
        device,
        free_vram_bytes,
        total_vram_bytes,
    )
    layer_count = _config_value(model, ("num_hidden_layers", "n_layer", "num_layers"))
    if not 0 <= config.layer_index < layer_count:
        raise ValueError(f"layer {config.layer_index} is outside the {layer_count}-block model")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    placebo = make_matched_norm_placebo(
        directions[config.position][config.layer_index], seed=config.seed
    )
    with config.output_path.open("w", encoding="utf-8") as records_file:
        for prompt_index, prompt in enumerate(prompts):
            messages = render_messages(
                cast("Sequence[Mapping[str, str]]", prompt["messages"]), config.model_id
            )
            prompt_ids = _token_ids_from_template(tokenizer, messages, generation_prompt=True)
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            prompt_activations = capture_positionwise_activations(
                cast("Any", model),
                input_ids,
                attention_mask,
                layers=[config.layer_index],
            )[config.layer_index][0]
            mean_residual_norm = float(prompt_activations.norm(dim=-1).mean().item())
            axes = {
                "direction": directions[config.position][config.layer_index],
                "matched_norm_placebo": placebo,
            }
            for axis_name, axis in axes.items():
                for strength_multiple in config.strengths:
                    magnitude = float(strength_multiple * mean_residual_norm)
                    hook = _generation_hook(
                        axis,
                        magnitude=magnitude,
                        prompt_length=len(prompt_ids),
                        scope=config.scope,
                    )
                    with residual_intervention(cast("Any", model), config.layer_index, hook):
                        generated = cast(
                            "torch.Tensor",
                            cast("Any", model).generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                do_sample=False,
                                max_new_tokens=config.max_new_tokens,
                                use_cache=True,
                            ),
                        )
                    completion = tokenizer.decode(
                        generated[0, input_ids.shape[1] :], skip_special_tokens=True
                    )
                    record = {
                        "prompt_id": str(prompt.get("prompt_id", prompt_index)),
                        "model": config.model_id,
                        "adapter": None if config.adapter is None else str(config.adapter),
                        "position": config.position,
                        "layer": config.layer_index,
                        "scope": config.scope,
                        "axis": axis_name,
                        "strength_multiple": float(strength_multiple),
                        "mean_residual_norm": mean_residual_norm,
                        "applied_magnitude": magnitude,
                        "completion": completion,
                    }
                    records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_file.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser(
        "audit-pairs", help="audit lexical balance of synthetic pairs"
    )
    audit_parser.add_argument("--pairs", type=Path, required=True)
    audit_parser.add_argument(
        "--min-code-token-difference", type=int, default=PAIR_AUDIT_MIN_CODE_TOKEN_DIFFERENCE
    )

    fit_parser = commands.add_parser("fit", help="fit synthetic-contrast directions")
    fit_parser.add_argument("--model", required=True)
    fit_parser.add_argument("--adapter", type=Path)
    fit_parser.add_argument("--pairs", type=Path, required=True)
    fit_parser.add_argument("--exclude-families", default="")
    fit_parser.add_argument("--out-dir", type=Path, required=True)
    fit_parser.add_argument("--permutations", type=int, default=20)
    fit_parser.add_argument("--seed", type=int, default=0)

    read_parser = commands.add_parser("read", help="score existing captures with saved directions")
    read_parser.add_argument("--run-dir", type=Path, required=True)
    read_parser.add_argument("--direction-dir", type=Path, required=True)

    steer_parser = commands.add_parser(
        "steer", help="generate matched direction/placebo continuations"
    )
    steer_parser.add_argument("--model", required=True)
    steer_parser.add_argument("--adapter", type=Path)
    steer_parser.add_argument("--direction-dir", type=Path, required=True)
    steer_parser.add_argument("--prompts", type=Path, required=True)
    steer_parser.add_argument("--layer", type=int, required=True)
    steer_parser.add_argument("--position", choices=POSITION_NAMES, required=True)
    steer_parser.add_argument(
        "--scope",
        choices=("all", "first"),
        default="all",
        help="steer the first output token and every later token, or the first output token only",
    )
    steer_parser.add_argument("--strengths", default="0.25,0.5,1,2")
    steer_parser.add_argument("--max-new-tokens", type=int, default=512)
    steer_parser.add_argument("--seed", type=int, default=0)
    steer_parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a pair audit, fit, read, or steering operation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "audit-pairs":
        _log_pair_audit(
            audit_pair_records(
                read_pairs(arguments.pairs),
                min_code_token_difference=arguments.min_code_token_difference,
            )
        )
    elif arguments.command == "fit":
        fit_directions(
            FitConfig(
                model_id=arguments.model,
                adapter=arguments.adapter,
                pairs_path=arguments.pairs,
                excluded_families=_parse_excluded_families(arguments.exclude_families),
                output_dir=arguments.out_dir,
                permutations=arguments.permutations,
                seed=arguments.seed,
            )
        )
    elif arguments.command == "read":
        read_captures(run_dir=arguments.run_dir, direction_dir=arguments.direction_dir)
    else:
        steer_prompts(
            SteeringConfig(
                model_id=arguments.model,
                adapter=arguments.adapter,
                direction_dir=arguments.direction_dir,
                prompts_path=arguments.prompts,
                layer_index=arguments.layer,
                position=arguments.position,
                scope=arguments.scope,
                strengths=_parse_strengths(arguments.strengths),
                output_path=arguments.out,
                max_new_tokens=arguments.max_new_tokens,
                seed=arguments.seed,
            )
        )


if __name__ == "__main__":
    main()
