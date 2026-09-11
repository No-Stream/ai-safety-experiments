"""Token frequencies from the released rollouts, and the frequency-residualised token-row ranking.

Ranking embedding or lm_head rows by how far they moved is a frequency table with extra steps: the
rows that move most are the rows whose tokens appear most, end-of-turn markers first. The real
search is the residual after frequency is regressed out. The frequencies come from the released
TMAX training rollouts, which store prompts and responses as token ids in the same vocabulary, so
nothing is re-tokenised; they are counted twice, as **inputs** (prompt plus response: every
position feeds the embedding row of its token) and as **targets** (response only: the loss, and so
the lm_head rows' gradient, is on response tokens).

Rows are read one line at a time and only the two token arrays are parsed out of each; the rest
of a row (per-token logprobs, tool outputs) is megabytes the count does not need.
"""

from __future__ import annotations

import json
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from tokenizers import Tokenizer

from reward_hacking.tmax.weight_delta import row_norms

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import torch
    from numpy.typing import NDArray

TOKEN_ROW_NDIM = 2

TOKEN_ARRAY_KEYS: tuple[str, ...] = ("prompt_tokens", "response_tokens")
_TOKEN_ARRAY_PATTERNS: dict[str, re.Pattern[str]] = {
    key: re.compile(rf'"{key}"\s*:\s*\[') for key in TOKEN_ARRAY_KEYS
}


class FrequencyRole(StrEnum):
    """Which count a token row is trained through: inputs for embeddings, targets for lm_head."""

    INPUT = "input"
    TARGET = "target"


def _token_ids(line: str, key: str) -> list[int]:
    """Pull one token-id array out of a rollout row without parsing the row's logprobs."""
    match = _TOKEN_ARRAY_PATTERNS[key].search(line)
    if match is None:
        raise ValueError(f"rollout row has no {key!r} array")
    end = line.index("]", match.end())
    body = line[match.end() : end].strip()
    return cast("list[int]", json.loads(f"[{body}]")) if body else []


def count_tokens_in_file(
    path: Path, vocab_size: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    """Input (prompt plus response) and target (response) token counts over one rollouts JSONL."""
    input_counts = np.zeros(vocab_size, dtype=np.int64)
    target_counts = np.zeros(vocab_size, dtype=np.int64)
    n_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prompt = np.asarray(_token_ids(line, "prompt_tokens"), dtype=np.int64)
            response = np.asarray(_token_ids(line, "response_tokens"), dtype=np.int64)
            for ids in (prompt, response):
                if ids.size and (int(ids.max()) >= vocab_size or int(ids.min()) < 0):
                    raise ValueError(
                        f"{path}: token id outside [0, {vocab_size}) in row {n_rows}; the "
                        f"rollouts were not produced with this vocabulary"
                    )
            input_counts += np.bincount(np.concatenate([prompt, response]), minlength=vocab_size)
            target_counts += np.bincount(response, minlength=vocab_size)
            n_rows += 1
    return input_counts, target_counts, n_rows


@dataclass(frozen=True)
class TokenCounts:
    """Token frequencies over a rollout corpus, split by the role a token row is trained through."""

    input_counts: NDArray[np.int64]
    target_counts: NDArray[np.int64]
    n_rows: int
    sources: tuple[str, ...]

    @property
    def vocab_size(self) -> int:
        """Number of token ids the counts cover."""
        return int(self.input_counts.shape[0])

    @property
    def n_input_tokens(self) -> int:
        """Total input positions counted (prompt plus response)."""
        return int(self.input_counts.sum())

    @property
    def n_target_tokens(self) -> int:
        """Total response positions counted."""
        return int(self.target_counts.sum())

    def counts_for(self, role: FrequencyRole) -> NDArray[np.int64]:
        """Return the count vector a token row of the given role trains through."""
        return self.input_counts if role == FrequencyRole.INPUT else self.target_counts

    def frame(self) -> pl.DataFrame:
        """Both counts as one frame keyed by token id."""
        return pl.DataFrame(
            {
                "token_id": np.arange(self.vocab_size, dtype=np.int64),
                "input_count": self.input_counts,
                "target_count": self.target_counts,
            }
        )

    def write(self, path: Path) -> None:
        """Write the counts as parquet and the provenance (rows, totals, files) as a JSON sibling."""
        self.frame().write_parquet(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "n_rows": self.n_rows,
                    "n_input_tokens": self.n_input_tokens,
                    "n_target_tokens": self.n_target_tokens,
                    "sources": list(self.sources),
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read(cls, path: Path) -> TokenCounts:
        """Read back what :meth:`write` wrote."""
        frame = pl.read_parquet(path)
        meta = cast(
            "dict[str, object]", json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        )
        return cls(
            input_counts=frame["input_count"].to_numpy().astype(np.int64),
            target_counts=frame["target_count"].to_numpy().astype(np.int64),
            n_rows=int(cast("int", meta["n_rows"])),
            sources=tuple(cast("list[str]", meta["sources"])),
        )


def rollout_files(rollouts_dir: Path) -> tuple[Path, ...]:
    """Every ``*_rollouts_*.jsonl`` under the directory, sorted so the count is reproducible."""
    files = tuple(sorted(rollouts_dir.rglob("*_rollouts_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"{rollouts_dir} holds no *_rollouts_*.jsonl file")
    return files


def count_rollout_tokens(
    files: Sequence[Path], *, vocab_size: int, max_workers: int = 8
) -> TokenCounts:
    """Count token ids across rollout files in parallel processes, one file per task.

    Workers are spawned rather than forked: the caller has torch loaded and multi-threaded, and a
    forked child of a threaded process can deadlock on a lock some other thread held.
    """
    input_counts = np.zeros(vocab_size, dtype=np.int64)
    target_counts = np.zeros(vocab_size, dtype=np.int64)
    n_rows = 0
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(max_workers, len(files)), mp_context=context) as pool:
        for file_inputs, file_targets, file_rows in pool.map(
            count_tokens_in_file, files, [vocab_size] * len(files)
        ):
            input_counts += file_inputs
            target_counts += file_targets
            n_rows += file_rows
    return TokenCounts(
        input_counts=input_counts,
        target_counts=target_counts,
        n_rows=n_rows,
        sources=tuple(str(path) for path in files),
    )


@dataclass(frozen=True)
class TokenRowSite:
    """Which checkpoint's token-row matrix a frame describes, and the count its rows train through."""

    checkpoint: str
    module: str
    role: FrequencyRole


def token_row_frame(
    site: TokenRowSite,
    *,
    base32: torch.Tensor,
    delta32: torch.Tensor,
    counts: TokenCounts | None,
) -> pl.DataFrame:
    """Per-token row movement of an embedding-shaped delta beside the frequency its rows train through.

    ``frequency`` is the count for the site's role; both counts travel so the other can be
    consulted. Without counts the frame carries movement only and no residual can be taken from it.
    """
    module = site.module
    if delta32.ndim != TOKEN_ROW_NDIM or base32.shape != delta32.shape:
        raise ValueError(f"{module} is not a token-row matrix pair: {tuple(delta32.shape)}")
    n_rows = delta32.shape[0]
    row_norm = row_norms(delta32).numpy()
    base_row_norm = row_norms(base32).numpy()
    frame = pl.DataFrame(
        {
            "token_id": np.arange(n_rows, dtype=np.int64),
            "row_norm": row_norm,
            "base_row_norm": base_row_norm,
        }
    ).with_columns(
        pl.lit(site.checkpoint).alias("checkpoint"),
        pl.lit(module).alias("module"),
        pl.lit(str(site.role)).alias("frequency_role"),
        (pl.col("row_norm") / pl.col("base_row_norm")).alias("relative_row_delta"),
    )
    if counts is None:
        return frame
    if counts.vocab_size != n_rows:
        raise ValueError(
            f"token counts cover {counts.vocab_size} ids but {module} has {n_rows} rows"
        )
    return frame.with_columns(
        pl.Series("input_count", counts.input_counts),
        pl.Series("target_count", counts.target_counts),
        pl.Series("frequency", counts.counts_for(site.role)),
    )


@dataclass(frozen=True)
class FrequencyFit:
    """The log-frequency regression a row ranking was residualised against."""

    degree: int
    coefficients: tuple[float, ...]
    r_squared: float
    n_fit: int
    n_zero_frequency: int
    n_zero_movement: int
    n_zero_frequency_moved: int


def residualise_on_log_frequency(
    rows: pl.DataFrame, *, degree: int = 2
) -> tuple[pl.DataFrame, FrequencyFit]:
    """Regress ``log(row_norm)`` on a polynomial in ``log(frequency)`` and keep the residual.

    Rows with zero frequency or zero movement cannot enter a log-log fit; they are counted in the
    fit record (a moved row with zero frequency is itself a finding: no input ever reached that
    embedding row, so whatever moved it was not the data) and carry a null residual.
    """
    frequency = rows["frequency"].to_numpy().astype(np.float64)
    norm = rows["row_norm"].to_numpy().astype(np.float64)
    fit_mask = (frequency > 0) & (norm > 0)
    n_fit = int(fit_mask.sum())
    if n_fit <= degree + 1:
        raise ValueError(
            f"only {n_fit} rows have both frequency and movement; cannot fit degree {degree}"
        )
    x = np.log(frequency[fit_mask])
    y = np.log(norm[fit_mask])
    design = np.vander(x, degree + 1)
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted_fit = design @ coefficients
    ss_res = float(((y - predicted_fit) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    log_norm = np.full(norm.shape, np.nan)
    log_norm[norm > 0] = np.log(norm[norm > 0])
    predicted = np.full(norm.shape, np.nan)
    predicted[fit_mask] = predicted_fit
    residual = np.full(norm.shape, np.nan)
    residual[fit_mask] = y - predicted_fit
    out = rows.with_columns(
        pl.Series("log_row_norm", log_norm).fill_nan(None),
        pl.Series("predicted_log_row_norm", predicted).fill_nan(None),
        pl.Series("frequency_residual", residual).fill_nan(None),
    )
    fit = FrequencyFit(
        degree=degree,
        coefficients=tuple(float(c) for c in coefficients),
        r_squared=1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        n_fit=n_fit,
        n_zero_frequency=int((frequency == 0).sum()),
        n_zero_movement=int((norm == 0).sum()),
        n_zero_frequency_moved=int(((frequency == 0) & (norm > 0)).sum()),
    )
    return out, fit


def token_strings(tokenizer_json: Path, token_ids: Sequence[int]) -> list[tuple[str, str]]:
    """``(piece, decoded)`` per id: the vocabulary entry and the text it decodes to."""
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    out: list[tuple[str, str]] = []
    for token_id in token_ids:
        piece = tokenizer.id_to_token(token_id)
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
        out.append((piece if piece is not None else "", decoded))
    return out


def with_token_strings(rows: pl.DataFrame, tokenizer_json: Path) -> pl.DataFrame:
    """Attach ``piece`` and ``text`` columns for the (few) token ids in ``rows``."""
    ids: list[int] = rows.get_column("token_id").cast(pl.Int64).to_list()
    strings = token_strings(tokenizer_json, ids)
    return rows.with_columns(
        pl.Series("piece", [piece for piece, _ in strings]),
        pl.Series("text", [text for _, text in strings]),
    )


def top_rows(rows: pl.DataFrame, *, by: str, k: int, descending: bool = True) -> pl.DataFrame:
    """Return the ``k`` rows with the largest (or smallest) defined value of ``by``.

    A display cut of a ranked table, never a computation input: every statistic in this package
    is computed over the full frame before a ranking is shown.
    """
    return rows.filter(pl.col(by).is_not_null()).sort(by, descending=descending).head(k)
