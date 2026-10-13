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
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

import numpy as np
import torch
from peft import PeftModel

from experiment_sampling import add_oversize_reason_argument, log_sampling_grid
from grpo.gpu_memory import gpu_memory_info
from reward_hacking.atlas.readout import out_of_fold_logits, rank_auc
from reward_hacking.atlas.sampling import artifact_model_key
from reward_hacking.interp.directions import (
    capture_positionwise_activations,
    diff_of_means,
    load_model_and_tokenizer,
    matched_norm_random_direction,
    unit,
)
from reward_hacking.interp.jsonl_resume import ResumeLedger, resume_records
from reward_hacking.interp.lens_fit_runtime import weights_fingerprint
from reward_hacking.interp.linear_probe import grouped_test_masks
from reward_hacking.interp.steering import Hook, residual_intervention, steering_hook
from reward_hacking.jsonl_io import read_jsonl_objects

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence


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
STEER_POSITION_NAMES: tuple[str, ...] = (*POSITION_NAMES, "last_prompt")
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
GENERATION_FREE_VRAM_FRACTION = 0.65
GENERATION_KV_CACHE_HEADROOM = 1.5
GENERATION_RESIDUAL_NDIM = 3
GENERATION_TOKEN_BATCH_NDIM = 2
MAX_STEERING_NORM_MULTIPLE = 4.0
STEERING_STRENGTH_UNIT = "multiple of the unsteered residual L2 norm at the last prompt token"
CLASS_DIFFERENCE_STRENGTH_UNIT = (
    "multiple of the layer's difference-of-means vector norm in the direction bundle"
)
STRENGTH_UNITS = {
    "last_prompt_norm": STEERING_STRENGTH_UNIT,
    "class_difference": CLASS_DIFFERENCE_STRENGTH_UNIT,
}
MATRIX_DIMENSIONS = 2
PAIR_TENSOR_DIMENSIONS = 3
PAIR_MEMBER_COUNT = 2
MIN_PAIR_COUNT = 2
MIN_ITEMS_PER_CLASS_FOR_LOO = 2
DEFAULT_ITEM_NULL_DRAWS = 50
ITEM_HOLDOUT_FRACTION = 0.3
ITEM_CLASS_FRACTION_THRESHOLD = 0.75
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
    pairs = read_jsonl_objects(path)
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
    free_bytes, total_bytes = gpu_memory_info(torch, device)
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


def _steering_adapter_weights_identity(
    model: torch.nn.Module, adapter_path: Path | None
) -> str | None:
    """Fingerprint effective adapter settings and loaded adapter parameters."""
    if adapter_path is None:
        return None
    config_path = adapter_path / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"adapter config must contain an object: {config_path}")
    provenance_fields = {"base_model_name_or_path", "revision", "peft_version"}
    semantic_config = {key: value for key, value in config.items() if key not in provenance_fields}
    adapter_parameters = [
        (name, parameter)
        for name, parameter in sorted(model.named_parameters())
        if ".lora_" in name or "modules_to_save." in name
    ]
    if not adapter_parameters:
        raise ValueError(f"loaded adapter at {adapter_path} has no supported adapter parameters")
    digest = hashlib.sha256(json.dumps(semantic_config, sort_keys=True).encode())
    for name, parameter in adapter_parameters:
        value = parameter.detach().contiguous()
        digest.update(name.encode())
        digest.update(b"\x00")
        digest.update(str(value.dtype).encode())
        digest.update(b"\x00")
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(b"\x00")
        for start in range(0, value.numel(), 4_194_304):
            chunk = value.view(-1)[start : start + 4_194_304].to("cpu")
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _steering_prompt_tokens_sha256(
    item_ids: Sequence[str], prompt_token_ids: Sequence[Sequence[int]]
) -> str:
    content = [
        {"item_id": item_id, "token_ids": list(token_ids)}
        for item_id, token_ids in zip(item_ids, prompt_token_ids, strict=True)
    ]
    serialized = json.dumps(content, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _direction_bundle_content_identity(
    directions: Mapping[str, torch.Tensor], layer_ids: Sequence[int]
) -> dict[str, object]:
    """Identify loaded direction values independently of NumPy container serialization."""
    positions: dict[str, dict[str, object]] = {}
    for position, direction in sorted(directions.items()):
        canonical = direction.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(canonical.dtype).encode())
        digest.update(b"\x00")
        digest.update(json.dumps(list(canonical.shape)).encode())
        digest.update(b"\x00")
        digest.update(canonical.view(torch.uint8).numpy().tobytes())
        positions[position] = {
            "dtype": str(canonical.dtype),
            "shape": list(canonical.shape),
            "values_sha256": digest.hexdigest(),
        }
    return {"layers": list(layer_ids), "positions": positions}


def _legacy_steering_identity_compatible(  # noqa: PLR0911 - legacy identity fields have separate compatibility conditions
    wanted: Mapping[str, Any],
    found: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_prompt_tokens: Mapping[str, Sequence[int]] | None = None,
) -> bool:
    """Migrate old path/byte identities when their stored prompt tokens confirm semantics."""
    if found.get("stage") != wanted.get("stage"):
        return False
    direction_identity = wanted.get("direction_bundle_content_identity")
    if not isinstance(direction_identity, Mapping) or not direction_identity.get("positions"):
        return False
    semantic_core = {
        "stage",
        "selected_item_ids",
        "layers",
        "position",
        "scope",
        "axes",
        "strengths",
        "samples_per_prompt",
        "sampling_budget",
        "temperature",
        "max_new_tokens",
        "seed",
        "strength_unit",
        "allow_large",
    }
    if semantic_core - set(wanted) or semantic_core - set(found):
        return False
    ignored_fields = {
        "adapter",
        "adapter_weights_identity",
        "direction_bundle_content_identity",
        "direction_bundle_sha256",
        "model",
        "model_weights_identity",
        "prompts_sha256",
        "selected_prompt_tokens_sha256",
        "selected_prompt_token_ids",
    }
    shared_fields = (set(wanted) & set(found)) - ignored_fields
    if any(wanted[field] != found[field] for field in shared_fields):
        return False

    if expected_prompt_tokens is None:
        raw_expected = wanted.get("selected_prompt_token_ids")
        if not isinstance(raw_expected, list):
            return not records
        expected_prompt_tokens = {
            str(row["item_id"]): cast("Sequence[int]", row["token_ids"]) for row in raw_expected
        }
    if not records:
        return True
    for record in records:
        item_id = str(record.get("item_id", record.get("prompt_id", "")))
        expected_tokens = expected_prompt_tokens.get(item_id)
        if expected_tokens is None or record.get("prompt_token_ids") != list(expected_tokens):
            return False
    return True


@dataclass(frozen=True)
class FitItemsConfig:
    """Inputs for fitting a last-prompt direction from an Atlas token sweep."""

    run_dir: Path
    model_key: str
    output_dir: Path
    top_layers: int | None = None
    null_draws: int = DEFAULT_ITEM_NULL_DRAWS
    seed: int = 0


def _item_loo_auc(features: torch.Tensor, labels: torch.Tensor) -> float:
    if features.ndim != MATRIX_DIMENSIONS or labels.shape != (features.shape[0],):
        raise ValueError("item features and labels must have shapes [items, hidden] and [items]")
    positive_count = int((labels == 1).sum().item())
    negative_count = int((labels == 0).sum().item())
    if min(positive_count, negative_count) < MIN_ITEMS_PER_CLASS_FOR_LOO:
        raise ValueError("leave-one-item-out AUC requires at least two fit items per class")
    predictions: list[torch.Tensor] = []
    for held_out_index in range(features.shape[0]):
        training_mask = torch.ones(features.shape[0], dtype=torch.bool)
        training_mask[held_out_index] = False
        training_features = features[training_mask]
        training_labels = labels[training_mask]
        direction = diff_of_means_direction(
            training_features[training_labels == 1], training_features[training_labels == 0]
        )
        predictions.append(torch.dot(features[held_out_index], unit(direction)))
    return rank_auc(labels, torch.stack(predictions))


def _item_hash_order(item_ids: Sequence[str]) -> list[str]:
    return sorted(
        item_ids, key=lambda item_id: (hashlib.sha256(item_id.encode()).digest(), item_id)
    )


def _item_sweep_identity(model_key: str) -> tuple[str, str, str]:
    if model_key.startswith("base--reads--"):
        adapter_key = model_key.removeprefix("base--reads--")
        if not adapter_key:
            raise ValueError("base--reads-- model keys require an adapter suffix")
        return model_key, "base", adapter_key
    return artifact_model_key(model_key), model_key, model_key


def _validate_item_sweep_identity(  # noqa: PLR0913
    index: Mapping[str, Any],
    *,
    output_key: str,
    expected_model_key: str,
    expected_samples_from: str,
    model_key: str,
    index_path: Path,
) -> None:
    if index.get("complete") is not True:
        raise ValueError(f"token-sweep index is incomplete: {index_path}")
    if index.get("output_key") != output_key:
        raise ValueError(f"token-sweep output key does not match {index_path}")
    if (
        index.get("model_key") != expected_model_key
        or index.get("samples_from") != expected_samples_from
    ):
        raise ValueError(f"token-sweep model identity does not match --model-key {model_key!r}")


def _validate_item_sweep_layout(
    activations: np.ndarray, index: Mapping[str, Any], index_path: Path, activation_path: Path
) -> None:
    if activations.ndim != CAPTURE_ARRAY_DIMENSIONS or activations.dtype != np.dtype(np.float32):
        raise ValueError("token-sweep activations must be float32 [rows, layers, tokens, hidden]")
    if list(activations.shape) != index.get("shape"):
        raise ValueError(f"token-sweep activation shape differs from its index: {activation_path}")
    rows = index.get("rows")
    layer_ids = [int(layer_id) for layer_id in index.get("layer_ids", [])]
    identity = index.get("identity")
    if not isinstance(rows, list) or len(rows) != activations.shape[0]:
        raise ValueError(f"token-sweep rows do not align with activations: {index_path}")
    if (
        len(layer_ids) != activations.shape[1]
        or layer_ids != sorted(set(layer_ids))
        or any(layer_id < 0 for layer_id in layer_ids)
    ):
        raise ValueError(f"token-sweep layer_ids do not align with activation layers: {index_path}")
    if not isinstance(identity, dict) or identity.get("layer_ids") != layer_ids:
        raise ValueError(
            f"token-sweep layer_ids must be zero-based decoder-block outputs per {LAYER_INDEX_CONVENTION}"
        )
    prompt_tokens = int(index["prompt_tokens"])
    if prompt_tokens < 1:
        raise ValueError(f"token-sweep prompt token count must be positive: {index_path}")
    if identity.get("prompt_tokens") != prompt_tokens:
        raise ValueError(f"token-sweep prompt token count differs from its index: {index_path}")
    if activations.shape[2] != prompt_tokens + int(index["completion_tokens"]):
        raise ValueError(f"token-sweep token axis differs from its index: {index_path}")
    position_layout = index.get("position_layout", {})
    if (
        position_layout.get("prompt_offsets") != list(range(-prompt_tokens, 0))
        or position_layout.get("prompt_padding") != "left"
        or position_layout.get("activation_alignment")
        != "hidden state after the token at that position"
    ):
        raise ValueError("token-sweep position layout does not identify the final prompt token")


def _item_fit_rows(run_dir: Path, model_key: str) -> tuple[np.ndarray, dict[str, Any], Path, str]:
    output_key, expected_model_key, expected_samples_from = _item_sweep_identity(model_key)
    index_path = run_dir / "token-sweep" / f"index.{output_key}.json"
    activation_path = run_dir / "token-sweep" / f"activations.{output_key}.npy"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    activations = np.load(activation_path, mmap_mode="r", allow_pickle=False)
    _validate_item_sweep_identity(
        index,
        output_key=output_key,
        expected_model_key=expected_model_key,
        expected_samples_from=expected_samples_from,
        model_key=model_key,
        index_path=index_path,
    )
    _validate_item_sweep_layout(activations, index, index_path, activation_path)
    return activations, index, index_path, output_key


@dataclass(frozen=True)
class ItemSweepSummary:
    """Per-item means, label fractions, and activation consistency diagnostics."""

    item_means: dict[str, np.ndarray]
    hack_fractions: dict[str, float]
    honest_pass_fractions: dict[str, float]
    max_within_item_abs_delta: float


def _aggregate_item_sweep(
    activations: np.ndarray, rows: Sequence[Mapping[str, Any]], prompt_token_index: int
) -> ItemSweepSummary:
    item_sums: dict[str, np.ndarray] = {}
    item_reference_activations: dict[str, np.ndarray] = {}
    item_sample_counts: Counter[str] = Counter()
    item_hack_counts: Counter[str] = Counter()
    item_pass_counts: Counter[str] = Counter()
    seen_samples: set[tuple[str, int]] = set()
    max_within_item_abs_delta = 0.0
    for row_index, row in enumerate(rows):
        item_id = str(row["item_id"])
        if not item_id:
            raise ValueError("token-sweep item ids must be non-empty")
        sample_key = (item_id, int(row["sample_index"]))
        if sample_key in seen_samples:
            raise ValueError(f"token-sweep rows contain duplicate item/sample pair {sample_key}")
        seen_samples.add(sample_key)
        hack_label = row.get("hack_label")
        pass_label = row.get("pass_label")
        if (
            not isinstance(hack_label, bool)
            or not isinstance(pass_label, bool)
            or (hack_label and pass_label)
        ):
            raise ValueError("token-sweep rows need exclusive boolean hack_label/pass_label values")
        item_activations = np.asarray(
            activations[row_index, :, prompt_token_index, :], dtype=np.float32
        ).copy()
        if not np.isfinite(item_activations).all():
            raise ValueError(f"non-finite last-prompt activations for item {item_id}")
        reference_activations = item_reference_activations.get(item_id)
        if reference_activations is None:
            item_reference_activations[item_id] = item_activations.copy()
        else:
            max_within_item_abs_delta = max(
                max_within_item_abs_delta,
                float(np.max(np.abs(item_activations - reference_activations))),
            )
        item_sums[item_id] = (
            item_sums.get(item_id, np.zeros_like(item_activations)) + item_activations
        )
        item_sample_counts[item_id] += 1
        item_hack_counts[item_id] += int(hack_label)
        item_pass_counts[item_id] += int(pass_label)

    if not item_sample_counts:
        raise ValueError("token-sweep index contains no item rows")
    return ItemSweepSummary(
        item_means={
            item_id: item_sums[item_id] / item_sample_counts[item_id]
            for item_id in item_sample_counts
        },
        hack_fractions={
            item_id: item_hack_counts[item_id] / item_sample_counts[item_id]
            for item_id in item_sample_counts
        },
        honest_pass_fractions={
            item_id: item_pass_counts[item_id] / item_sample_counts[item_id]
            for item_id in item_sample_counts
        },
        max_within_item_abs_delta=max_within_item_abs_delta,
    )


def _held_out_item_ids(item_ids: Sequence[str]) -> list[str]:
    if len(item_ids) < MIN_ITEMS_PER_CLASS_FOR_LOO:
        raise ValueError("fit-items requires at least two distinct items")
    ordered_item_ids = _item_hash_order(item_ids)
    held_out_count = min(len(item_ids) - 1, max(1, int(len(item_ids) * ITEM_HOLDOUT_FRACTION)))
    return ordered_item_ids[:held_out_count]


def _fit_class_item_ids(
    summary: ItemSweepSummary, held_out_item_ids: Sequence[str]
) -> tuple[list[str], list[str]]:
    held_out_set = set(held_out_item_ids)
    hack_item_ids = [
        item_id
        for item_id in summary.item_means
        if item_id not in held_out_set
        and summary.hack_fractions[item_id] >= ITEM_CLASS_FRACTION_THRESHOLD
    ]
    honest_item_ids = [
        item_id
        for item_id in summary.item_means
        if item_id not in held_out_set
        and summary.honest_pass_fractions[item_id] >= ITEM_CLASS_FRACTION_THRESHOLD
    ]
    if set(hack_item_ids) & set(honest_item_ids):
        raise ValueError("an item cannot meet both the hack and honest-pass fraction thresholds")
    if min(len(hack_item_ids), len(honest_item_ids)) < MIN_ITEMS_PER_CLASS_FOR_LOO:
        raise ValueError("fit-items requires at least two non-held-out items in each class")
    return hack_item_ids, honest_item_ids


def _fit_item_layer(  # noqa: PLR0913
    item_means: Mapping[str, np.ndarray],
    layer_position: int,
    layer_id: int,
    hack_item_ids: Sequence[str],
    honest_item_ids: Sequence[str],
    *,
    null_draws: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    hack_features = torch.from_numpy(
        np.stack([item_means[item_id][layer_position] for item_id in hack_item_ids])
    )
    honest_features = torch.from_numpy(
        np.stack([item_means[item_id][layer_position] for item_id in honest_item_ids])
    )
    direction = diff_of_means_direction(hack_features, honest_features)
    fit_features = torch.cat((hack_features, honest_features), dim=0)
    fit_labels = torch.tensor([1] * len(hack_item_ids) + [0] * len(honest_item_ids))
    loo_auc = _item_loo_auc(fit_features, fit_labels)
    null_aucs = _random_direction_null_aucs(
        fit_features, fit_labels, direction, null_draws=null_draws, seed=seed + layer_id
    )
    metric = {
        "layer": layer_id,
        "n_hack_items": len(hack_item_ids),
        "n_honest_items": len(honest_item_ids),
        "leave_one_item_out_auc": loo_auc,
        "random_direction_null_auc": null_aucs,
        "random_direction_null_auc_mean": float(np.mean(null_aucs)),
        "random_direction_null_auc_max": max(null_aucs),
    }
    return direction, metric


def _random_direction_null_aucs(
    features: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    *,
    null_draws: int,
    seed: int,
) -> list[float]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_aucs: list[float] = []
    for _ in range(null_draws):
        random_direction = torch.randn(direction.shape, generator=generator)
        matched_norm_direction = unit(random_direction) * direction.norm()
        null_aucs.append(rank_auc(labels, features @ unit(matched_norm_direction)))
    return null_aucs


def _rank_item_layers(layer_metrics: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(metric["layer"])
        for metric in sorted(
            layer_metrics,
            key=lambda metric: (-float(metric["leave_one_item_out_auc"]), int(metric["layer"])),
        )
    ]


def fit_item_directions(config: FitItemsConfig) -> None:
    """Fit a held-out Atlas item direction from final-prompt-token sweep activations."""
    if config.null_draws < 1:
        raise ValueError(f"null_draws must be positive, got {config.null_draws}")
    activations, index, index_path, output_key = _item_fit_rows(config.run_dir, config.model_key)
    layer_ids = [int(layer_id) for layer_id in index["layer_ids"]]
    item_summary = _aggregate_item_sweep(
        activations, index["rows"], int(index["prompt_tokens"]) - 1
    )
    item_ids = sorted(item_summary.item_means)
    held_out_item_ids = _held_out_item_ids(item_ids)
    hack_item_ids, honest_item_ids = _fit_class_item_ids(item_summary, held_out_item_ids)
    fitted_layers = [
        _fit_item_layer(
            item_summary.item_means,
            layer_position,
            layer_id,
            hack_item_ids,
            honest_item_ids,
            null_draws=config.null_draws,
            seed=config.seed,
        )
        for layer_position, layer_id in enumerate(layer_ids)
    ]
    all_directions = [direction for direction, _ in fitted_layers]
    layer_metrics = [metric for _, metric in fitted_layers]

    ranked_layer_ids = _rank_item_layers(layer_metrics)
    selected_layer_count = len(layer_ids) if config.top_layers is None else config.top_layers
    if not 1 <= selected_layer_count <= len(layer_ids):
        raise ValueError(f"top_layers must be between 1 and {len(layer_ids)}")
    selected_layers = set(ranked_layer_ids[:selected_layer_count])
    saved_layer_ids = [layer_id for layer_id in layer_ids if layer_id in selected_layers]
    layer_position_by_id = {layer_id: position for position, layer_id in enumerate(layer_ids)}
    directions = torch.stack(
        [all_directions[layer_position_by_id[layer_id]] for layer_id in saved_layer_ids]
    )
    provenance: dict[str, Any] = {
        "schema_version": DIRECTION_SCHEMA_VERSION,
        "model": str(index["model_id"]),
        "adapter": index.get("adapter_id"),
        "source": "atlas_token_sweep_item_contrast",
        "source_model_key": config.model_key,
        "source_output_key": output_key,
        "source_index": str(index_path),
        "source_index_sha256": _sha256_file(index_path),
        "position": "last_prompt",
        "positions": ["last_prompt"],
        "layers": saved_layer_ids,
        "layer_index_convention": LAYER_INDEX_CONVENTION,
        "prompt_token_index": int(index["prompt_tokens"]) - 1,
        "item_fraction_threshold": ITEM_CLASS_FRACTION_THRESHOLD,
        "holdout_fraction": ITEM_HOLDOUT_FRACTION,
        "holdout_selection": "ascending SHA-256(item_id), first max(1, floor(0.30 * n_items))",
        "held_out_item_ids": held_out_item_ids,
        "fit_item_ids": sorted([*hack_item_ids, *honest_item_ids]),
        "n_items_total": len(item_ids),
        "n_items_held_out": len(held_out_item_ids),
        "n_items_excluded_from_fit": len(item_ids)
        - len(held_out_item_ids)
        - len(hack_item_ids)
        - len(honest_item_ids),
        "max_within_item_last_prompt_abs_delta": item_summary.max_within_item_abs_delta,
        "null_draws": config.null_draws,
        "seed": config.seed,
        "metrics": {"last_prompt": layer_metrics},
    }
    save_direction_bundle(config.output_dir, {"last_prompt": directions}, provenance)
    logger.info(
        "fit-items model_key=%s items=%d held_out=%d saved_layers=%s aucs=%s",
        config.model_key,
        len(item_ids),
        len(held_out_item_ids),
        saved_layer_ids,
        [
            (metric["layer"], round(float(metric["leave_one_item_out_auc"]), 3))
            for metric in layer_metrics
        ],
    )


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
    samples = {str(row["sample_id"]): row for row in read_jsonl_objects(samples_path)}
    grades = {str(row["sample_id"]): row for row in read_jsonl_objects(grades_path)}
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
    schema_version = provenance.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version < 1
    ):
        raise ValueError(f"invalid direction bundle semantic layout: schema tag {schema_version!r}")
    if provenance.get("layer_index_convention") != LAYER_INDEX_CONVENTION:
        raise ValueError(
            "saved direction bundle uses a different hidden-state/block index convention"
        )
    positions = provenance.get("positions")
    layers = provenance.get("layers")
    if (
        not directions
        or not isinstance(positions, list)
        or any(not isinstance(position, str) for position in positions)
        or len(positions) != len(set(positions))
        or not isinstance(layers, list)
        or any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers)
    ):
        raise ValueError("invalid direction bundle semantic layout: required fields are malformed")
    layer_ids = cast("list[int]", layers)
    if not layer_ids or layer_ids != sorted(set(layer_ids)) or layer_ids[0] < 0:
        raise ValueError(
            f"saved direction layers must be sorted zero-based block indices: {layer_ids}"
        )
    if set(directions) != set(positions):
        raise ValueError("saved direction positions differ from provenance")
    if any(
        array.ndim != MATRIX_DIMENSIONS
        or array.shape[0] != len(layer_ids)
        or array.shape[1] < 1
        or not torch.isfinite(array).all()
        for array in directions.values()
    ):
        raise ValueError("saved direction layer axes differ from provenance")
    return layer_ids


def _direction_row_index(layer_ids: Sequence[int], layer_id: int) -> int:
    try:
        return layer_ids.index(layer_id)
    except ValueError as error:
        raise ValueError(
            f"layer {layer_id} is absent from saved direction layers {list(layer_ids)}"
        ) from error


def _atlas_direction_name(position: str, directions: Mapping[str, torch.Tensor]) -> str | None:
    if position not in ATLAS_POSITION_MAP:
        raise ValueError(f"unsupported Atlas read position {position!r}")
    direction_name = position if position in directions else ATLAS_POSITION_MAP[position]
    return direction_name if direction_name in directions else None


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
    layer_ids = [int(layer) for layer in provenance["layers"]]
    if any(layer_id >= activations.shape[1] for layer_id in layer_ids):
        raise ValueError("saved direction layer is outside the Atlas activation layer axis")
    matched_positions = 0
    for position in positions:
        direction_name = _atlas_direction_name(position, directions)
        if direction_name is None:
            continue
        matched_positions += 1
        direction = directions[direction_name]
        if direction.shape != (len(layer_ids), activations.shape[3]):
            raise ValueError(f"direction shape differs from Atlas activations for {position}")
    if not matched_positions:
        raise ValueError("Atlas capture has no read positions supported by the direction bundle")


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
    direction_layer_ids = [int(layer_id) for layer_id in provenance["layers"]]
    for row_index, pair_key in enumerate(pair_keys):
        for position_index, position in enumerate(positions):
            direction_position = _atlas_direction_name(position, directions)
            if direction_position is None:
                continue
            for direction_row, layer_id in enumerate(direction_layer_ids):
                vector = torch.from_numpy(
                    np.asarray(
                        activations[row_index, layer_id, position_index, :], dtype=np.float32
                    ).copy()
                )
                score = torch.dot(vector, directions[direction_position][direction_row].float())
                key = (position, layer_id, None)
                scores.setdefault(key, {})[pair_key] = float(score.item())
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
    direction_row_by_layer: Mapping[int, int]
    directions: Mapping[str, torch.Tensor]


def _score_token_sweep_row(
    row_index: int,
    row: Mapping[str, Any],
    context: _TokenSweepScoreContext,
) -> list[tuple[tuple[str, int, int | None], tuple[str, int], float]]:
    pair_key = str(row["item_id"]), int(row["sample_index"])
    scored_positions: list[tuple[tuple[str, int, int | None], tuple[str, int], float]] = []
    if row["code_token_start"] is None:
        return scored_positions
    code_start = int(row["code_token_start"])
    for offset in TOKEN_SWEEP_CODE_OFFSETS:
        token_position = context.token_count + code_start + offset
        if (
            not 0 <= token_position < context.activations.shape[2]
            or not context.mask[row_index, token_position]
        ):
            continue
        for layer_position, layer_id in enumerate(context.layer_ids):
            for direction_position in ("response_mean", "response_last"):
                direction = context.directions[direction_position][
                    context.direction_row_by_layer[layer_id]
                ]
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
    if not {"response_mean", "response_last"} <= set(directions):
        logger.info("token_sweep_status=positions_unavailable model_key=%s", model_key)
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
        direction_row_by_layer={
            int(layer_id): row_index for row_index, layer_id in enumerate(provenance["layers"])
        },
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


def _generation_hidden(output: object) -> tuple[torch.Tensor, bool]:
    is_tuple = isinstance(output, tuple)
    hidden = output[0] if is_tuple else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("decoder block output must be a tensor or tuple beginning with a tensor")
    return hidden, is_tuple


def _generation_prefill_selector(
    hidden: torch.Tensor, prompt_lengths: Sequence[int]
) -> torch.Tensor:
    if any(prompt_length > hidden.shape[-2] for prompt_length in prompt_lengths):
        raise ValueError("prompt length exceeds the first generation prefill sequence")
    selector = torch.zeros(hidden.shape[:2], dtype=hidden.dtype, device=hidden.device)
    prompt_lengths_tensor = torch.tensor(prompt_lengths, device=hidden.device)
    left_padding = hidden.shape[-2] - prompt_lengths_tensor
    last_prompt_positions = left_padding + prompt_lengths_tensor - 1
    selector[torch.arange(hidden.shape[0], device=hidden.device), last_prompt_positions] = 1
    return selector.unsqueeze(-1)


def _replace_generation_hidden(
    output: object, *, is_tuple: bool, steered_hidden: torch.Tensor
) -> object:
    if is_tuple:
        output_tuple = cast("tuple[object, ...]", output)
        return (steered_hidden, *output_tuple[1:])
    return steered_hidden


def _apply_generation_steering(
    hidden: torch.Tensor,
    row_steps: torch.Tensor,
    *,
    prompt_lengths: Sequence[int],
    scope: str,
    prefill_seen: bool,
) -> tuple[torch.Tensor | None, bool]:
    if not prefill_seen:
        selector = _generation_prefill_selector(hidden, prompt_lengths)
        return hidden + selector * row_steps, True
    if scope == "all":
        return hidden + row_steps, prefill_seen
    return None, prefill_seen


def _generation_hook(
    direction: torch.Tensor,
    *,
    magnitudes: Sequence[float],
    prompt_lengths: Sequence[int],
    scope: str,
) -> Hook:
    """Steer each row's final prompt residual and optionally later generated token states."""
    if scope not in {"all", "first"}:
        raise ValueError(f"unknown steering scope {scope!r}")
    if not magnitudes or len(magnitudes) != len(prompt_lengths):
        raise ValueError("generation steering requires one magnitude and prompt length per row")
    if any(prompt_length < 1 for prompt_length in prompt_lengths):
        raise ValueError("prompt lengths must be positive")
    if any(not math.isfinite(magnitude) for magnitude in magnitudes):
        raise ValueError("generation steering magnitudes must be finite")
    direction_step = unit(direction).float()
    row_magnitudes = torch.tensor(magnitudes, dtype=torch.float32)
    prefill_seen = False
    row_steps: torch.Tensor | None = None

    def hook(_module: object, _inputs: object, output: object) -> object:
        nonlocal prefill_seen, row_steps
        hidden, is_tuple = _generation_hidden(output)
        if hidden.ndim != GENERATION_RESIDUAL_NDIM or hidden.shape[0] != len(prompt_lengths):
            raise ValueError(
                "generation steering expects hidden states shaped [batch, sequence, width]"
            )
        if row_steps is None:
            row_steps = row_magnitudes.to(hidden).view(-1, 1, 1) * direction_step.to(hidden).view(
                1, 1, -1
            )
        steered, prefill_seen = _apply_generation_steering(
            hidden,
            row_steps,
            prompt_lengths=prompt_lengths,
            scope=scope,
            prefill_seen=prefill_seen,
        )
        if steered is None:
            return output
        if steered.shape != hidden.shape:
            raise AssertionError("generation steering changed the decoder residual shape")
        return _replace_generation_hidden(output, is_tuple=is_tuple, steered_hidden=steered)

    return hook


def _left_pad_prompt_batch(
    prompt_token_ids: Sequence[Sequence[int]], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad prompt ids so each row's final prompt token shares the final column."""
    if not prompt_token_ids or any(not tokens for tokens in prompt_token_ids):
        raise ValueError("generation batches require at least one non-empty prompt")
    if pad_token_id < 0:
        raise ValueError(f"pad_token_id must be non-negative, got {pad_token_id}")
    sequence_width = max(len(tokens) for tokens in prompt_token_ids)
    input_ids = torch.full(
        (len(prompt_token_ids), sequence_width), pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, tokens in enumerate(prompt_token_ids):
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device)
        left_padding = sequence_width - len(tokens)
        input_ids[row_index, left_padding:] = token_tensor
        attention_mask[row_index, left_padding:] = 1
    return input_ids, attention_mask


def _validate_steering_magnitude(
    applied_magnitude: float, steered_position_norm: float, *, allow_large: bool
) -> None:
    """Refuse interventions larger than four times the local, unsteered residual norm."""
    if not math.isfinite(applied_magnitude):
        raise ValueError("applied steering magnitude must be finite")
    if not math.isfinite(steered_position_norm) or steered_position_norm < 0:
        raise ValueError("steered-position norm must be finite and non-negative")
    if (
        not allow_large
        and abs(applied_magnitude) > MAX_STEERING_NORM_MULTIPLE * steered_position_norm
    ):
        raise ValueError(
            f"applied magnitude {abs(applied_magnitude):.6g} exceeds the "
            f"{MAX_STEERING_NORM_MULTIPLE:g}x steered-position norm guard "
            f"({steered_position_norm:.6g}); pass --allow-large to override"
        )


def _completion_coherence(
    completion_token_ids: Sequence[int], completion_text: str
) -> dict[str, float | bool]:
    """Summarize token diversity near the end of a completion and code-fence presence."""
    recent_token_ids = completion_token_ids[-64:]
    distinct_fraction = (
        len(set(recent_token_ids)) / len(recent_token_ids) if recent_token_ids else 0.0
    )
    return {
        "distinct_token_fraction_last_64": distinct_fraction,
        "contains_code_block": "```" in completion_text,
    }


def _prompt_residual_norms(activations: torch.Tensor) -> tuple[float, float | None]:
    """Return the last-token norm and the median norm after excluding the first prompt token."""
    if activations.ndim != MATRIX_DIMENSIONS or activations.shape[0] == 0:
        raise ValueError("prompt residual activations must have shape [sequence, width]")
    token_norms = torch.linalg.vector_norm(activations, dim=-1).tolist()
    last_token_norm = float(token_norms[-1])
    median_after_first = (
        float(median(float(norm) for norm in token_norms[1:])) if len(token_norms) > 1 else None
    )
    return last_token_norm, median_after_first


def _generation_batch_size(
    model: torch.nn.Module,
    *,
    free_vram_bytes: int,
    max_sequence_length: int,
    row_count: int,
) -> int:
    """Estimate a safe HF generation batch from free VRAM and the model's KV-cache shape."""
    if row_count < 1 or max_sequence_length < 1:
        raise ValueError("generation row count and sequence length must be positive")
    if free_vram_bytes <= 0:
        parameter_device = next(model.parameters()).device
        if parameter_device.type == "cpu":
            return row_count
        raise ValueError("cannot size a CUDA generation batch without free VRAM")
    layer_count = _config_value(model, ("num_hidden_layers", "n_layer", "num_layers"))
    key_value_heads = _config_value(model, ("num_key_value_heads",))
    head_dimension = _config_value(model, ("head_dim",))
    element_size = next(model.parameters()).element_size()
    bytes_per_row = (
        2
        * layer_count
        * key_value_heads
        * head_dimension
        * max_sequence_length
        * element_size
        * GENERATION_KV_CACHE_HEADROOM
    )
    usable_vram = free_vram_bytes * GENERATION_FREE_VRAM_FRACTION
    estimated_rows = max(1, int(usable_vram // bytes_per_row))
    return min(row_count, estimated_rows)


@dataclass(frozen=True)
class _GenerationBatchConfig:
    layer_index: int
    direction: torch.Tensor
    magnitudes: Sequence[float]
    prompt_lengths: Sequence[int]
    scope: str
    seed: int
    temperature: float
    max_new_tokens: int
    do_sample: bool = True


def _generate_steered_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_config: _GenerationBatchConfig,
) -> torch.Tensor:
    """Generate one batch with a per-row last-prompt intervention and deterministic RNG state."""
    device = input_ids.device
    rng_devices = [device.index] if device.type == "cuda" else []
    hook = _generation_hook(
        batch_config.direction,
        magnitudes=batch_config.magnitudes,
        prompt_lengths=batch_config.prompt_lengths,
        scope=batch_config.scope,
    )
    with (
        torch.random.fork_rng(devices=rng_devices),
        residual_intervention(cast("Any", model), batch_config.layer_index, hook),
    ):
        torch.manual_seed(batch_config.seed)
        generated = cast(
            "torch.Tensor",
            cast("Any", model).generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=batch_config.do_sample,
                temperature=batch_config.temperature,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                repetition_penalty=1.0,
                max_new_tokens=batch_config.max_new_tokens,
                use_cache=True,
            ),
        )
    if generated.ndim != GENERATION_TOKEN_BATCH_NDIM or generated.shape[0] != input_ids.shape[0]:
        raise ValueError("model.generate returned a malformed batch")
    if generated.shape[1] < input_ids.shape[1]:
        raise ValueError("model.generate returned fewer tokens than its padded prompt batch")
    return generated


def _steering_batch_seed(sample_seeds: Sequence[int]) -> int:
    """Derive one stable sampling seed from the sample identities in a generation batch."""
    if not sample_seeds:
        raise ValueError("generation batch seed requires at least one sample identity")
    digest = hashlib.sha256("\0".join(str(seed) for seed in sample_seeds).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _eos_token_ids(model: torch.nn.Module) -> frozenset[int]:
    generation_config = getattr(cast("Any", model), "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset({eos_token_id})
    if isinstance(eos_token_id, list) and all(
        isinstance(token_id, int) for token_id in eos_token_id
    ):
        return frozenset(eos_token_id)
    raise TypeError("generation_config.eos_token_id must be an integer or a list of integers")


def _trim_completion_at_eos(
    completion_token_ids: Sequence[int], eos_token_ids: frozenset[int]
) -> list[int]:
    """Drop batch padding after the first emitted end-of-sequence token."""
    for token_index, token_id in enumerate(completion_token_ids):
        if token_id in eos_token_ids:
            return list(completion_token_ids[: token_index + 1])
    return list(completion_token_ids)


def _parse_strengths(value: str) -> list[float]:
    strengths = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not strengths or any(not math.isfinite(strength) for strength in strengths):
        raise ValueError("--strengths must be comma-separated finite multiples")
    return strengths


def _prompt_records(
    path: Path,
    *,
    held_out_item_ids: Sequence[str] | None = None,
    max_prompts: int | None = None,
    item_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    prompts = read_jsonl_objects(path)
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    _validate_prompt_records(path, prompts)
    if max_prompts is not None and max_prompts < 1:
        raise ValueError(f"max_prompts must be positive, got {max_prompts}")
    if item_ids is not None:
        if held_out_item_ids is None:
            raise ValueError(
                "an item subset needs a direction bundle that records its held-out items"
            )
        not_held_out = sorted(set(item_ids) - set(held_out_item_ids))
        if not_held_out:
            raise ValueError(
                f"requested items are not held out from the direction fit: {not_held_out[:5]}"
            )
        held_out_item_ids = list(item_ids)
    if held_out_item_ids is not None:
        prompts = _select_held_out_prompts(path, prompts, held_out_item_ids, max_prompts)
    elif max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"no prompts remain after item selection from {path}")
    return prompts


def _validate_prompt_records(path: Path, prompts: Sequence[Mapping[str, Any]]) -> None:
    for line_number, prompt in enumerate(prompts, start=1):
        messages = prompt.get("messages")
        rendered_prompt = prompt.get("prompt")
        if isinstance(rendered_prompt, str):
            continue
        if not isinstance(messages, list) or not messages:
            raise ValueError(
                f"{path}:{line_number} requires rendered prompt text or a messages list"
            )
        if any(
            not isinstance(message, dict) or not isinstance(message.get("content"), str)
            for message in messages
        ):
            raise ValueError(f"{path}:{line_number} messages need text content")


def _select_held_out_prompts(
    path: Path,
    prompts: Sequence[dict[str, Any]],
    held_out_item_ids: Sequence[str],
    max_prompts: int | None,
) -> list[dict[str, Any]]:
    prompt_by_item_id = {
        str(prompt["item_id"]): prompt for prompt in prompts if "item_id" in prompt
    }
    if len(prompt_by_item_id) != sum("item_id" in prompt for prompt in prompts):
        raise ValueError(f"{path} contains duplicate Atlas item ids")
    selected_ids = list(held_out_item_ids)
    if max_prompts is not None:
        selected_ids = selected_ids[:max_prompts]
    missing_ids = sorted(set(selected_ids) - set(prompt_by_item_id))
    if missing_ids:
        raise ValueError(f"held-out item ids are missing from {path}: {missing_ids[:5]}")
    return [prompt_by_item_id[item_id] for item_id in selected_ids]


def _tokenize_prompt_record(
    tokenizer: _ChatTokenizer,
    prompt: Mapping[str, Any],
    *,
    model_id: str,
) -> list[int]:
    rendered_prompt = prompt.get("prompt")
    if isinstance(rendered_prompt, str):
        token_ids = [
            int(token_id)
            for token_id in tokenizer.encode(rendered_prompt, add_special_tokens=False)
        ]
    else:
        messages = render_messages(
            cast("Sequence[Mapping[str, str]]", prompt["messages"]), model_id
        )
        token_ids = _token_ids_from_template(tokenizer, messages, generation_prompt=True)
    if not token_ids:
        raise ValueError("prompt tokenization produced an empty sequence")
    return token_ids


def _validate_rendered_prompt_model(
    prompts_path: Path, prompts: Sequence[Mapping[str, Any]], model_id: str
) -> None:
    if not any(isinstance(prompt.get("prompt"), str) for prompt in prompts):
        return
    metadata_path = prompts_path.with_name("prompt-metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"{metadata_path} must contain a JSON object")
    if metadata.get("model_id") != model_id:
        raise ValueError(
            f"Atlas prompts were rendered for {metadata.get('model_id')!r}, not {model_id!r}"
        )


@dataclass(frozen=True)
class SteeringConfig:
    """Inputs for generating direction-steered and placebo completions."""

    model_id: str
    adapter: Path | None
    direction_dir: Path
    prompts_path: Path
    position: str
    scope: str
    strengths: Sequence[float]
    output_path: Path
    max_new_tokens: int
    seed: int
    layer_index: int | None = None
    best_layers: int | None = None
    max_prompts: int | None = None
    samples_per_prompt: int = 1
    temperature: float = 1.0
    allow_large: bool = False
    layer_indices: Sequence[int] | None = None
    # last_prompt_norm suits models without massive-activation channels; on Gemma-3 the last-token norm is
    # dominated by one channel, so steering is scaled by the hack-vs-honest class difference instead.
    strength_unit: str = "last_prompt_norm"
    item_ids: Sequence[str] | None = None
    oversize_reason: str | None = None


@dataclass(frozen=True)
class _SteeringSample:
    prompt_ids: list[int]
    item_id: str
    sample_index: int
    sample_id: str
    sample_seed: int
    steered_position_norm: float
    median_prompt_token_norm: float | None
    applied_magnitude: float


@dataclass(frozen=True)
class _SteeringRuntime:
    config: SteeringConfig
    model: torch.nn.Module
    tokenizer: _ChatTokenizer
    directions: Mapping[str, torch.Tensor]
    saved_layer_ids: list[int]
    selected_layers: list[int]
    prompts: list[dict[str, Any]]
    prompt_token_ids: list[list[int]]
    item_ids: list[str]
    residual_norms_by_layer: dict[int, list[tuple[float, float | None]]]
    device: torch.device
    pad_token_id: int
    generation_batch_size: int
    eos_token_ids: frozenset[int]
    sampling_provenance: dict[str, object]
    model_weights_identity: str | None = None
    adapter_weights_identity: str | None = None


@dataclass(frozen=True)
class _SteeringCell:
    layer_id: int
    axis_name: str
    axis: torch.Tensor
    strength_multiple: float
    samples: list[_SteeringSample]


@dataclass
class _SteeringProgress:
    completed_sample_ids: set[str]
    ledger: ResumeLedger
    records_file: TextIO


@dataclass(frozen=True)
class _GeneratedSteeringBatch:
    samples: list[_SteeringSample]
    seed: int
    input_width: int
    token_ids: torch.Tensor


def _validate_steering_request(
    config: SteeringConfig,
    directions: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
) -> None:
    if config.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {config.max_new_tokens}")
    if config.samples_per_prompt < 1:
        raise ValueError(f"samples_per_prompt must be positive, got {config.samples_per_prompt}")
    if not math.isfinite(config.temperature) or config.temperature <= 0:
        raise ValueError(f"temperature must be positive and finite, got {config.temperature}")
    if not config.strengths or any(not math.isfinite(strength) for strength in config.strengths):
        raise ValueError("steering strengths must be non-empty and finite")
    _validate_steering_layer_selection(config)
    if config.max_prompts is not None and config.max_prompts < 1:
        raise ValueError(f"max_prompts must be positive, got {config.max_prompts}")
    if str(provenance["model"]) != config.model_id:
        raise ValueError(
            f"direction model {provenance['model']} differs from requested model {config.model_id}"
        )
    if config.position not in directions:
        raise ValueError(f"position {config.position!r} is absent from {config.direction_dir}")


def _validate_steering_layer_selection(config: SteeringConfig) -> None:
    layer_selection_count = sum(
        selection is not None
        for selection in (config.layer_index, config.layer_indices, config.best_layers)
    )
    if layer_selection_count != 1:
        raise ValueError("select exactly one of layer_index, layer_indices, or best_layers")
    if config.layer_indices is not None and (
        not config.layer_indices
        or any(layer < 0 for layer in config.layer_indices)
        or len(set(config.layer_indices)) != len(config.layer_indices)
    ):
        raise ValueError("layer_indices must be a non-empty sequence of unique non-negative layers")
    if config.best_layers is not None and config.best_layers < 1:
        raise ValueError(f"best_layers must be positive, got {config.best_layers}")


def _selected_steering_layers(
    config: SteeringConfig,
    provenance: Mapping[str, Any],
) -> list[int]:
    saved_layers = [int(layer_id) for layer_id in provenance["layers"]]
    if config.layer_indices is not None:
        selected = list(config.layer_indices)
    elif config.layer_index is not None:
        selected = [config.layer_index]
    else:
        metrics_by_position = provenance.get("metrics", {}).get(config.position, [])
        ranked_metrics = sorted(
            metrics_by_position,
            key=lambda metric: (
                -float(metric["leave_one_item_out_auc"]),
                int(metric["layer"]),
            ),
        )
        requested_count = cast("int", config.best_layers)
        selected = [int(metric["layer"]) for metric in ranked_metrics[:requested_count]]
        if len(selected) != requested_count:
            raise ValueError("direction provenance has too few leave-one-item-out layer metrics")
    missing_layers = [layer_id for layer_id in selected if layer_id not in saved_layers]
    if missing_layers:
        raise ValueError(f"requested layers are absent from saved directions: {missing_layers}")
    if not selected:
        raise ValueError("no steering layers were selected")
    return selected


def _parse_layer_indices(value: str) -> list[int]:
    layer_indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not layer_indices or any(layer < 0 for layer in layer_indices):
        raise ValueError("--layers must be comma-separated non-negative layer indices")
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("--layers cannot contain duplicate layer indices")
    return layer_indices


def _steering_sample_seed(seed: int, item_id: str, sample_index: int, sample_count: int) -> int:
    if (
        sample_index < 0
        or sample_count < 1
        or sample_index >= sample_count
        or sample_count >= 2**31 - 1
    ):
        raise ValueError("sample_index must be inside the positive sample_count")
    digest = hashlib.sha256(f"{seed}\0{item_id}".encode()).digest()
    seed_base = int.from_bytes(digest[:8], "big") % (2**31 - 1 - sample_count)
    return seed_base + sample_index


def _prepare_steering_runtime(config: SteeringConfig) -> _SteeringRuntime:
    directions, provenance = load_direction_bundle(config.direction_dir)
    _validate_direction_bundle(directions, provenance)
    _validate_steering_request(config, directions, provenance)
    saved_layer_ids = [int(layer_id) for layer_id in provenance["layers"]]
    selected_layers = _selected_steering_layers(config, provenance)
    held_out_ids = provenance.get("held_out_item_ids")
    if held_out_ids is not None and (
        not isinstance(held_out_ids, list)
        or any(not isinstance(item_id, str) for item_id in held_out_ids)
        or len(set(held_out_ids)) != len(held_out_ids)
    ):
        raise ValueError("direction provenance held_out_item_ids must be a unique string list")
    prompts = _prompt_records(
        config.prompts_path,
        held_out_item_ids=cast("Sequence[str] | None", held_out_ids),
        max_prompts=config.max_prompts,
        item_ids=config.item_ids,
    )
    _validate_rendered_prompt_model(config.prompts_path, prompts, config.model_id)
    sampling_provenance = log_sampling_grid(
        {
            (f"layer={layer_id}/axis={axis_name}/strength={strength_multiple:g}"): len(prompts)
            * config.samples_per_prompt
            for layer_id in selected_layers
            for axis_name in ("direction", "matched_norm_placebo")
            for strength_multiple in config.strengths
        },
        oversize_reason=config.oversize_reason,
    )
    model, tokenizer = _load_model_with_optional_adapter(config.model_id, config.adapter)
    model_weights_identity = weights_fingerprint(config.model_id, model, config.adapter)
    adapter_weights_identity = _steering_adapter_weights_identity(model, config.adapter)
    device, free_vram_bytes, total_vram_bytes = _device_memory(model)
    layer_count = _config_value(model, ("num_hidden_layers", "n_layer", "num_layers"))
    invalid_layers = [layer_id for layer_id in selected_layers if not 0 <= layer_id < layer_count]
    if invalid_layers:
        raise ValueError(
            f"steering layers {invalid_layers} are outside the {layer_count}-block model"
        )
    if tokenizer.pad_token_id is None:
        raise ValueError("batched steering requires a tokenizer pad_token_id")
    prompt_token_ids = [
        _tokenize_prompt_record(tokenizer, prompt, model_id=config.model_id) for prompt in prompts
    ]
    item_ids = [
        str(prompt.get("item_id", prompt.get("prompt_id", prompt_index)))
        for prompt_index, prompt in enumerate(prompts)
    ]
    max_prompt_length = max(len(tokens) for tokens in prompt_token_ids)
    generation_batch_size = _generation_batch_size(
        model,
        free_vram_bytes=free_vram_bytes,
        max_sequence_length=max_prompt_length + config.max_new_tokens,
        row_count=len(prompts) * config.samples_per_prompt,
    )
    logger.info(
        "steer model=%s adapter=%s device=%s free_vram_bytes=%d total_vram_bytes=%d "
        "max_prompt_tokens=%d max_new_tokens=%d generation_batch_size=%d",
        config.model_id,
        config.adapter,
        device,
        free_vram_bytes,
        total_vram_bytes,
        max_prompt_length,
        config.max_new_tokens,
        generation_batch_size,
    )
    residual_norms_by_layer: dict[int, list[tuple[float, float | None]]] = {}
    for layer_id in selected_layers:
        layer_norms: list[tuple[float, float | None]] = []
        for prompt_tokens in prompt_token_ids:
            input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            activations = capture_positionwise_activations(
                cast("Any", model), input_ids, torch.ones_like(input_ids), layers=[layer_id]
            )[layer_id][0]
            layer_norms.append(_prompt_residual_norms(activations))
        residual_norms_by_layer[layer_id] = layer_norms
    return _SteeringRuntime(
        config=config,
        model=model,
        tokenizer=tokenizer,
        directions=directions,
        saved_layer_ids=saved_layer_ids,
        selected_layers=selected_layers,
        prompts=prompts,
        prompt_token_ids=prompt_token_ids,
        item_ids=item_ids,
        residual_norms_by_layer=residual_norms_by_layer,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
        generation_batch_size=generation_batch_size,
        eos_token_ids=_eos_token_ids(model),
        sampling_provenance=sampling_provenance,
        model_weights_identity=model_weights_identity,
        adapter_weights_identity=adapter_weights_identity,
    )


def _steering_resume_identity(runtime: _SteeringRuntime) -> dict[str, object]:
    config = runtime.config
    return {
        "stage": "hack_contrast_steer",
        "model_weights_identity": runtime.model_weights_identity,
        "adapter_weights_identity": runtime.adapter_weights_identity,
        "direction_bundle_content_identity": _direction_bundle_content_identity(
            runtime.directions, runtime.saved_layer_ids
        ),
        "selected_item_ids": runtime.item_ids,
        "selected_prompt_tokens_sha256": _steering_prompt_tokens_sha256(
            runtime.item_ids, runtime.prompt_token_ids
        ),
        "layers": runtime.selected_layers,
        "position": config.position,
        "scope": config.scope,
        "axes": ["direction", "matched_norm_placebo"],
        "strengths": list(config.strengths),
        "samples_per_prompt": config.samples_per_prompt,
        "sampling_budget": runtime.sampling_provenance,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "seed": config.seed,
        "strength_unit": STRENGTH_UNITS[config.strength_unit],
        "generation_batch_size": runtime.generation_batch_size,
        "allow_large": config.allow_large,
    }


def _build_steering_samples(
    runtime: _SteeringRuntime, *, layer_id: int, axis_name: str, strength_multiple: float
) -> list[_SteeringSample]:
    config = runtime.config
    layer_norms = runtime.residual_norms_by_layer[layer_id]
    if config.strength_unit not in STRENGTH_UNITS:
        raise ValueError(
            f"unknown strength unit {config.strength_unit!r}; choose from {sorted(STRENGTH_UNITS)}"
        )
    class_difference_norm = float(
        runtime.directions[config.position][runtime.saved_layer_ids.index(layer_id)].norm().item()
    )

    def reference_norm(steered_position_norm: float) -> float:
        return (
            class_difference_norm
            if config.strength_unit == "class_difference"
            else steered_position_norm
        )

    samples: list[_SteeringSample] = []
    for sample_index in range(config.samples_per_prompt):
        for prompt_index, prompt_tokens in enumerate(runtime.prompt_token_ids):
            steered_position_norm, median_prompt_token_norm = layer_norms[prompt_index]
            applied_magnitude = float(strength_multiple * reference_norm(steered_position_norm))
            _validate_steering_magnitude(
                applied_magnitude,
                steered_position_norm,
                allow_large=config.allow_large,
            )
            item_id = runtime.item_ids[prompt_index]
            sample_id = f"L{layer_id}:{axis_name}:{strength_multiple:g}:{item_id}:{sample_index}"
            samples.append(
                _SteeringSample(
                    prompt_ids=prompt_tokens,
                    item_id=item_id,
                    sample_index=sample_index,
                    sample_id=sample_id,
                    sample_seed=_steering_sample_seed(
                        config.seed, item_id, sample_index, config.samples_per_prompt
                    ),
                    steered_position_norm=steered_position_norm,
                    median_prompt_token_norm=median_prompt_token_norm,
                    applied_magnitude=applied_magnitude,
                )
            )
    return samples


def _generate_steering_batch(
    runtime: _SteeringRuntime, cell: _SteeringCell, samples: list[_SteeringSample]
) -> _GeneratedSteeringBatch:
    batch_seed = _steering_batch_seed([sample.sample_seed for sample in samples])
    input_ids, attention_mask = _left_pad_prompt_batch(
        [sample.prompt_ids for sample in samples],
        pad_token_id=runtime.pad_token_id,
        device=runtime.device,
    )
    token_ids = _generate_steered_batch(
        runtime.model,
        input_ids,
        attention_mask,
        _GenerationBatchConfig(
            layer_index=cell.layer_id,
            direction=cell.axis,
            magnitudes=[sample.applied_magnitude for sample in samples],
            prompt_lengths=[len(sample.prompt_ids) for sample in samples],
            scope=runtime.config.scope,
            seed=batch_seed,
            temperature=runtime.config.temperature,
            max_new_tokens=runtime.config.max_new_tokens,
        ),
    )
    return _GeneratedSteeringBatch(
        samples=samples,
        seed=batch_seed,
        input_width=input_ids.shape[1],
        token_ids=token_ids,
    )


def _steering_record(
    runtime: _SteeringRuntime,
    cell: _SteeringCell,
    batch: _GeneratedSteeringBatch,
    batch_index: int,
) -> dict[str, object]:
    config = runtime.config
    sample = batch.samples[batch_index]
    completion_token_ids = _trim_completion_at_eos(
        [int(token_id) for token_id in batch.token_ids[batch_index, batch.input_width :]],
        runtime.eos_token_ids,
    )
    completion_text = runtime.tokenizer.decode(
        torch.tensor(completion_token_ids), skip_special_tokens=True
    )
    return {
        "sample_id": sample.sample_id,
        "prompt_id": sample.item_id,
        "item_id": sample.item_id,
        "sample_index": sample.sample_index,
        "seed": batch.seed,
        "generation_batch_seed": batch.seed,
        "generation_batch_size": len(batch.samples),
        "model": config.model_id,
        "adapter": None if config.adapter is None else str(config.adapter),
        "position": config.position,
        "layer": cell.layer_id,
        "scope": config.scope,
        "axis": cell.axis_name,
        "strength_multiple": cell.strength_multiple,
        "strength_unit": STRENGTH_UNITS[runtime.config.strength_unit],
        "steered_position_norm": sample.steered_position_norm,
        "median_residual_norm_excluding_first_prompt_token": sample.median_prompt_token_norm,
        "applied_magnitude": sample.applied_magnitude,
        "temperature": config.temperature,
        "top_p": 1.0,
        "top_k": 0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": config.max_new_tokens,
        "prompt_token_ids": sample.prompt_ids,
        "completion_token_ids": completion_token_ids,
        "completion": completion_text,
        "coherence": _completion_coherence(completion_token_ids, completion_text),
    }


def _run_steering_cell(
    runtime: _SteeringRuntime, cell: _SteeringCell, progress: _SteeringProgress
) -> None:
    batch_size = runtime.generation_batch_size
    for batch_start in range(0, len(cell.samples), batch_size):
        samples = cell.samples[batch_start : batch_start + batch_size]
        if all(sample.sample_id in progress.completed_sample_ids for sample in samples):
            continue
        batch = _generate_steering_batch(runtime, cell, samples)
        for batch_index, sample in enumerate(batch.samples):
            if sample.sample_id in progress.completed_sample_ids:
                continue
            record = _steering_record(runtime, cell, batch, batch_index)
            progress.records_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            progress.records_file.flush()
            progress.ledger.mark_record_complete(sample.sample_id)
            progress.completed_sample_ids.add(sample.sample_id)


@torch.no_grad()
def steer_prompts(
    config: SteeringConfig,
) -> None:
    """Generate direction and matched-norm-placebo completions for later external grading."""
    runtime = _prepare_steering_runtime(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_prompt_tokens = dict(zip(runtime.item_ids, runtime.prompt_token_ids, strict=True))
    resume, ledger = resume_records(
        config.output_path,
        identity=_steering_resume_identity(runtime),
        unit_of=lambda record: str(record["sample_id"]),
        record_key_of=lambda record: str(record["sample_id"]),
        legacy_identity_compatible=lambda wanted, found, records: (
            _legacy_steering_identity_compatible(
                wanted,
                found,
                records,
                expected_prompt_tokens=expected_prompt_tokens,
            )
        ),
    )
    completed_sample_ids = set(ledger.completed_records)
    logger.info(
        "steer_resume completed_records=%d prompts=%d layers=%s",
        resume.n_kept,
        len(runtime.prompts),
        runtime.selected_layers,
    )
    with config.output_path.open("a", encoding="utf-8") as records_file:
        progress = _SteeringProgress(completed_sample_ids, ledger, records_file)
        for layer_id in runtime.selected_layers:
            direction = runtime.directions[config.position][
                _direction_row_index(runtime.saved_layer_ids, layer_id)
            ]
            placebo = make_matched_norm_placebo(direction, seed=config.seed + layer_id)
            axes = (("direction", direction), ("matched_norm_placebo", placebo))
            for axis_name, axis in axes:
                for strength_multiple in config.strengths:
                    cell = _SteeringCell(
                        layer_id,
                        axis_name,
                        axis,
                        float(strength_multiple),
                        _build_steering_samples(
                            runtime,
                            layer_id=layer_id,
                            axis_name=axis_name,
                            strength_multiple=float(strength_multiple),
                        ),
                    )
                    _run_steering_cell(runtime, cell, progress)


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

    fit_items_parser = commands.add_parser(
        "fit-items", help="fit a held-out item direction at the last prompt token"
    )
    fit_items_parser.add_argument("--run-dir", type=Path, required=True)
    fit_items_parser.add_argument("--model-key", required=True)
    fit_items_parser.add_argument("--out-dir", type=Path, required=True)
    fit_items_parser.add_argument("--top-layers", type=int)
    fit_items_parser.add_argument("--null-draws", type=int, default=DEFAULT_ITEM_NULL_DRAWS)
    fit_items_parser.add_argument("--seed", type=int, default=0)

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
    layer_selection = steer_parser.add_mutually_exclusive_group(required=True)
    layer_selection.add_argument("--layer", type=int)
    layer_selection.add_argument("--layers", type=_parse_layer_indices)
    layer_selection.add_argument("--best-layers", type=int)
    steer_parser.add_argument("--position", choices=STEER_POSITION_NAMES, required=True)
    steer_parser.add_argument(
        "--scope",
        choices=("all", "first"),
        default="all",
        help="steer the first output token and every later token, or the first output token only",
    )
    steer_parser.add_argument("--strengths", default="0.25,0.5,1,2")
    steer_parser.add_argument(
        "--strength-unit", choices=sorted(STRENGTH_UNITS), default="last_prompt_norm"
    )
    steer_parser.add_argument(
        "--item-ids",
        help="comma-separated subset of the bundle's held-out items to steer, in this order",
    )
    steer_parser.add_argument("--samples-per-prompt", type=int, default=1)
    add_oversize_reason_argument(steer_parser)
    steer_parser.add_argument("--temperature", type=float, default=1.0)
    steer_parser.add_argument("--max-prompts", type=int)
    steer_parser.add_argument("--max-new-tokens", type=int, default=512)
    steer_parser.add_argument("--seed", type=int, default=0)
    steer_parser.add_argument(
        "--allow-large",
        action="store_true",
        help="allow steering magnitudes above four times the last-prompt residual norm",
    )
    steer_parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run a pair audit, fit, item fit, read, or steering operation."""
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
    elif arguments.command == "fit-items":
        fit_item_directions(
            FitItemsConfig(
                run_dir=arguments.run_dir,
                model_key=arguments.model_key,
                output_dir=arguments.out_dir,
                top_layers=arguments.top_layers,
                null_draws=arguments.null_draws,
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
                position=arguments.position,
                scope=arguments.scope,
                strengths=_parse_strengths(arguments.strengths),
                output_path=arguments.out,
                max_new_tokens=arguments.max_new_tokens,
                seed=arguments.seed,
                layer_index=arguments.layer,
                best_layers=arguments.best_layers,
                max_prompts=arguments.max_prompts,
                samples_per_prompt=arguments.samples_per_prompt,
                temperature=arguments.temperature,
                allow_large=arguments.allow_large,
                strength_unit=arguments.strength_unit,
                item_ids=None if arguments.item_ids is None else arguments.item_ids.split(","),
                layer_indices=arguments.layers,
                oversize_reason=arguments.oversize_reason,
            )
        )


if __name__ == "__main__":
    main()
