"""CLI parsing and private corpus loading for the generic Jacobian-lens fit."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from reward_hacking.interp.jacobian import digest_strings

LAYER_RANGE_PART_COUNT = 3


@dataclass(frozen=True)
class FitCorpus:
    """The selected prompt prefix and hashes that identify it without storing its text."""

    prompts: tuple[str, ...]
    source_sha256: str
    selected_text_sha256: str


@dataclass(frozen=True)
class FitOptions:
    """Validated CLI inputs for one generic lens fit."""

    model: str
    corpus: Path
    n_prompts: int
    source_layers: tuple[int, ...]
    target_layer: int
    dim_batch: int
    max_seq_len: int
    out_dir: Path
    adapter: Path | None = None
    halves: bool = False
    checkpoint_every: int = 5

    def __post_init__(self) -> None:
        """Reject CLI settings that cannot describe a fit."""
        if self.n_prompts <= 0:
            raise ValueError(f"n_prompts must be positive, got {self.n_prompts}")
        if self.dim_batch <= 0:
            raise ValueError(f"dim_batch must be positive, got {self.dim_batch}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be positive, got {self.checkpoint_every}")
        if not self.source_layers:
            raise ValueError("source_layers must select at least one layer")


def parse_layer_range(value: str) -> list[int]:
    """Parse a half-open ``start:stop:step`` layer range, such as ``8:44:2``."""
    parts = value.split(":")
    if len(parts) != LAYER_RANGE_PART_COUNT:
        raise ValueError(f"source layers must use start:stop:step, got {value!r}")
    try:
        start, stop, step = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(f"source layers must use integer start:stop:step, got {value!r}") from exc
    if step <= 0:
        raise ValueError(f"source layer step must be positive, got {step}")
    layers = list(range(start, stop, step))
    if not layers:
        raise ValueError(f"source layer range {value!r} selects no layers")
    return layers


def read_fit_corpus(path: Path, *, n_prompts: int) -> FitCorpus:
    """Read an ordered JSONL corpus, cap at its first ``n_prompts`` rows, and retain hashes only."""
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}")
    source_digest = hashlib.sha256()
    prompts: list[str] = []
    with path.open("rb") as corpus_file:
        for line_number, raw_line in enumerate(corpus_file, start=1):
            source_digest.update(raw_line)
            try:
                row = json.loads(raw_line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError(f"{path}: line {line_number} is not valid JSON") from exc
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("text"), str)
                or not row["text"].strip()
            ):
                raise ValueError(
                    f"{path}: line {line_number} must contain a non-empty string field 'text'"
                )
            if len(prompts) < n_prompts:
                prompts.append(row["text"])
    if len(prompts) < n_prompts:
        raise ValueError(
            f"corpus has {len(prompts)} rows with text, fewer than --n-prompts={n_prompts}"
        )
    return FitCorpus(
        prompts=tuple(prompts),
        source_sha256=source_digest.hexdigest(),
        selected_text_sha256=digest_strings(prompts),
    )


def build_parser(*, description: str | None = None) -> argparse.ArgumentParser:
    """Build the public generic fit CLI."""
    parser = argparse.ArgumentParser(
        description=description or "Fit a Jacobian lens to an ordered text JSONL corpus."
    )
    parser.add_argument(
        "--model", required=True, help="Hugging Face causal-LM model id or local path"
    )
    parser.add_argument("--adapter", type=Path, help="optional PEFT adapter directory")
    parser.add_argument(
        "--corpus", type=Path, required=True, help="JSONL corpus with one text field per row"
    )
    parser.add_argument(
        "--n-prompts", type=int, required=True, help="number of ordered corpus rows to fit"
    )
    parser.add_argument(
        "--source-layers",
        type=parse_layer_range,
        required=True,
        help="half-open start:stop:step, for example 8:44:2",
    )
    parser.add_argument("--target-layer", type=int, required=True, help="jlens target block index")
    parser.add_argument(
        "--dim-batch", type=int, required=True, help="number of output dimensions per backward"
    )
    parser.add_argument(
        "--max-seq-len", type=int, required=True, help="token cap for each corpus row"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="output directory for lens and provenance"
    )
    parser.add_argument(
        "--halves", action="store_true", help="fit odd/even rows and merge their lenses"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="write the jlens accumulator every N prompts",
    )
    return parser


def options_from_args(args: argparse.Namespace) -> FitOptions:
    """Convert argparse's namespace into validated fit options."""
    return FitOptions(
        model=args.model,
        corpus=args.corpus,
        n_prompts=args.n_prompts,
        source_layers=tuple(args.source_layers),
        target_layer=args.target_layer,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        out_dir=args.out_dir,
        adapter=args.adapter,
        halves=args.halves,
        checkpoint_every=args.checkpoint_every,
    )
