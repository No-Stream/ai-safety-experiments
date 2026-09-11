r"""Weight-space geometry of a released RL delta: where it landed, what shape it has, how deltas relate.

The behavioural screen reads what TMAX's RL taught from the outside; this module reads it off the
weights, CPU-only and inference-free, so the same loop can run over every released checkpoint the
registry names (:mod:`reward_hacking.tmax.artifacts`). Four questions, each a table:

* **Where did the update land?** Per tensor, ``||W_rl - W_base||_F / ||W_base||_F`` and the
  fraction of entries that changed at all, rolled up by layer and by module family (attention,
  DeltaNet in/out projections, DeltaNet state parameters, MLP, embeddings, lm_head, norms).
* **What shape does it have?** The delta's singular spectrum per 2-D tensor: stable rank
  (``||D||_F^2 / s_1^2``), effective rank (``exp`` of the entropy of the singular values normalised
  to sum to one, Roy & Vetterli 2007), that entropy divided by ``log n`` so it reads on ``[0, 1]``,
  and the share of squared singular-value mass in the top 1, 8 and 64 directions. The base tensor's
  own shape is computed once beside it, so "low rank" is read against the tensor RL started from.
* **Do two deltas point the same way?** Per tensor, the cosine between two checkpoints' deltas, the
  least-squares scale ``alpha`` that best rebuilds the later delta from the earlier one, and the
  residual share (``1 - cos^2`` exactly for that fit; reported because it is the number a reader
  asks for). Every cosine is read against the double-rounding floor below.
* **Which token rows moved?** ``embed_tokens`` and ``lm_head`` row norms regressed on log token
  frequency (:mod:`reward_hacking.tmax.token_frequency`), so the ranking is not a frequency table
  with extra steps.

**The double-rounding floor** every cosine is read against, and why it is near zero for a
representable bf16 base while the reported ``offset`` variant is not, is documented in
:mod:`reward_hacking.tmax.weight_floor`. **Dtype alignment** (``rl - round(base, rl.dtype)``) is in
:mod:`reward_hacking.tmax.weight_delta`; spectra are float64 and documented in
:mod:`reward_hacking.tmax.weight_spectra`.

**Memory.** Tensors stream one name at a time through the lazy reader
:func:`reward_hacking.tmax.amplify.align_tensor_sets` opens; peak is the largest tensor (the
vocab-sized embedding, 4 GB in float32) a handful of times over. Run it under
``scripts/resource-limits.sh`` with a memory cap regardless. Every table is appended per tensor and
a relaunch skips what is already on disk, counting resumed tensors separately.

    uv run python -m reward_hacking.tmax.weight_geometry run \\
        --base Qwen/Qwen3.5-9B --base-revision main \\
        --rl allenai/tmax-9b@step_200 --rl allenai/tmax-9b@step_500 \\
        --rollouts-dir /var/tmp/tmax-rollouts/decompressed \\
        --out artifacts/tmax/weight_geometry/9b-step200-step500
    uv run python -m reward_hacking.tmax.weight_geometry render \\
        --out artifacts/tmax/weight_geometry/9b-step200-step500
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from games.eval_model import FullWeightsFacts, FullWeightsSource, resolve_full_weights
from reward_hacking.tmax.amplify import AlignedTensorReader, align_tensor_sets
from reward_hacking.tmax.token_frequency import (
    FrequencyRole,
    TokenCounts,
    TokenRowSite,
    count_rollout_tokens,
    rollout_files,
    token_row_frame,
)
from reward_hacking.tmax.weight_delta import (
    AlignedDelta,
    aligned_delta,
    frobenius_norm,
    row_norms,
    tensor_movement,
)
from reward_hacking.tmax.weight_floor import (
    DEFAULT_FLOOR_PAIRS,
    cosine_from_inner,
    delta_relationship,
)
from reward_hacking.tmax.weight_geometry_tables import render_readout
from reward_hacking.tmax.weight_layout import (
    BASE_SPECTRA_FILENAME,
    EMBED_TOKENS_KEY,
    GATE_HEAD_MODULES,
    GATE_HEADS_FILENAME,
    PAIRS_FILENAME,
    RUN_MANIFEST_FILENAME,
    TENSORS_FILENAME,
    TOKEN_COUNTS_FILENAME,
    TOKEN_ROW_MODULES,
    ModuleClass,
    TensorSite,
    classify_tensor,
    token_rows_filename,
)
from reward_hacking.tmax.weight_spectra import (
    DEFAULT_TOP_K_SPECTRUM,
    GRAM_MIN_DIM,
    MATRIX_NDIM,
    as_matrix,
    spectral_shape,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

FREQUENCY_ROLES: dict[ModuleClass, FrequencyRole] = {
    ModuleClass.EMBED_TOKENS: FrequencyRole.INPUT,
    ModuleClass.LM_HEAD: FrequencyRole.TARGET,
}
"""Embedding rows train through every input position; lm_head rows through response targets only."""


# --------------------------------------------------------------------------------------
# DeltaNet gates, per recurrent head
# --------------------------------------------------------------------------------------


def gate_head_records(
    site: TensorSite, checkpoint: str, delta: AlignedDelta
) -> list[dict[str, object]]:
    """One row per recurrent head of a gate projection (rows) or state parameter (scalars)."""
    base32, delta32 = delta.base32, delta.delta32
    records: list[dict[str, object]] = []
    if base32.ndim == MATRIX_NDIM:
        base_norms = row_norms(base32)
        delta_norms = row_norms(delta32)
        inner = (base32.double() * delta32.double()).sum(dim=1)
        changed = (delta32 != 0).float().mean(dim=1)
        for head in range(base32.shape[0]):
            base_norm = float(base_norms[head])
            delta_norm = float(delta_norms[head])
            records.append(
                {
                    "checkpoint": checkpoint,
                    **site.fields(),
                    "head": head,
                    "base_norm": base_norm,
                    "delta_norm": delta_norm,
                    "relative_delta": delta_norm / base_norm if base_norm else 0.0,
                    "cosine_delta_base": cosine_from_inner(
                        float(inner[head]), base_norm, delta_norm
                    ),
                    "changed_fraction": float(changed[head]),
                    "base_value": None,
                    "delta_value": None,
                }
            )
        return records
    if base32.ndim != 1:
        raise ValueError(f"{site.name} has {base32.ndim} dims; gate tables read rows or scalars")
    for head in range(base32.shape[0]):
        base_value = float(base32[head])
        delta_value = float(delta32[head])
        records.append(
            {
                "checkpoint": checkpoint,
                **site.fields(),
                "head": head,
                "base_norm": abs(base_value),
                "delta_norm": abs(delta_value),
                "relative_delta": abs(delta_value) / abs(base_value) if base_value else 0.0,
                "cosine_delta_base": None,
                "changed_fraction": 1.0 if delta_value != 0.0 else 0.0,
                "base_value": base_value,
                "delta_value": delta_value,
            }
        )
    return records


# --------------------------------------------------------------------------------------
# The run: stream every tensor once, append every table, resume what is on disk
# --------------------------------------------------------------------------------------


def _existing_keys(path: Path, key_fields: Sequence[str]) -> set[tuple[str, ...]]:
    if not path.is_file():
        return set()
    keys: set[tuple[str, ...]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = cast("dict[str, object]", json.loads(line))
                keys.add(tuple(str(record[field]) for field in key_fields))
    return keys


def _append_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Append records as JSON lines; nothing to append leaves the file untouched (or absent)."""
    if not records:
        return
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")


def _prefixed(prefix: str, values: Mapping[str, object]) -> dict[str, object]:
    return {f"{prefix}{key}": value for key, value in values.items()}


@dataclass(frozen=True)
class GeometryRun:
    """Everything one invocation computes: a base, the RL'd checkpoints against it, where to write."""

    base: FullWeightsFacts
    rl: tuple[FullWeightsFacts, ...]
    out_dir: Path
    n_floor_pairs: int = DEFAULT_FLOOR_PAIRS
    top_k_spectrum: int = DEFAULT_TOP_K_SPECTRUM

    def __post_init__(self) -> None:
        """Refuse an empty or ambiguous checkpoint list and a floor with no draws."""
        if not self.rl:
            raise ValueError("a geometry run needs at least one RL'd checkpoint")
        labels = [facts.label for facts in self.rl]
        if len(set(labels)) != len(labels):
            raise ValueError(f"RL'd checkpoint labels must be distinct, got {labels}")
        if self.n_floor_pairs < 1:
            raise ValueError("n_floor_pairs must be at least 1")

    @property
    def labels(self) -> tuple[str, ...]:
        """The RL'd checkpoints' labels, in the order given (earliest first)."""
        return tuple(facts.label for facts in self.rl)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """Every ordered pair ``(earlier, later)`` in the order the checkpoints were given."""
        n = len(self.rl)
        return tuple((i, j) for i in range(n) for j in range(i + 1, n))

    def manifest(self) -> dict[str, object]:
        """Describe what the run was computed over, for the sidecar beside the tables."""

        def facts_record(facts: FullWeightsFacts) -> dict[str, object]:
            return {
                "label": facts.label,
                "snapshot_dir": str(facts.snapshot_dir),
                "commit_sha": facts.commit_sha,
                "weights_fingerprint": facts.fingerprint,
            }

        return {
            "kind": "weight-geometry",
            "base": facts_record(self.base),
            "rl": [facts_record(facts) for facts in self.rl],
            "pairs": [[self.labels[i], self.labels[j]] for i, j in self.pairs],
            "n_floor_pairs": self.n_floor_pairs,
            "top_k_spectrum": self.top_k_spectrum,
            "gram_min_dim": GRAM_MIN_DIM,
            "torch_version": torch.__version__,
        }


@dataclass(frozen=True)
class RunSummary:
    """What one invocation did: tensors seen, computed, and skipped as already on disk."""

    n_tensors: int
    n_computed: int
    n_resumed: int
    seconds: float


def open_readers(run: GeometryRun) -> tuple[tuple[str, ...], tuple[AlignedTensorReader, ...]]:
    """Align every RL'd checkpoint against the base and require them to share one tensor set."""
    aligned = tuple(
        align_tensor_sets(base_dir=run.base.snapshot_dir, rl_dir=facts.snapshot_dir)
        for facts in run.rl
    )
    names = aligned[0].names
    for facts, sets in zip(run.rl[1:], aligned[1:], strict=True):
        if sets.names != names:
            raise ValueError(
                f"{facts.label} carries a different tensor set from {run.rl[0].label}; the "
                f"checkpoints of one run must share a layout"
            )
    return names, tuple(sets.open() for sets in aligned)


def vocab_size_of(reader: AlignedTensorReader) -> int:
    """Rows of the embedding matrix, read from the shard header."""
    return reader.shape_of(EMBED_TOKENS_KEY)[0]


@dataclass(frozen=True)
class _TensorWork:
    """Which outputs one tensor still owes, given what is already on disk."""

    tensors: list[str]
    pairs: list[tuple[int, int]]
    base: bool
    gates: list[str]
    rows: list[str]

    @property
    def any(self) -> bool:
        """Whether anything at all is still owed."""
        return bool(self.tensors or self.pairs or self.base or self.gates or self.rows)


def _tensor_work(run: GeometryRun, site: TensorSite, done: _Done) -> _TensorWork:
    labels = run.labels
    name = site.name
    return _TensorWork(
        tensors=[label for label in labels if (label, name) not in done.tensors],
        pairs=[(i, j) for i, j in run.pairs if (labels[i], labels[j], name) not in done.pairs],
        base=(name,) not in done.base,
        gates=(
            [label for label in labels if (label, name) not in done.gates]
            if site.module in GATE_HEAD_MODULES
            else []
        ),
        rows=(
            [
                label
                for label in labels
                if not (run.out_dir / token_rows_filename(label, site.module)).is_file()
            ]
            if site.module in TOKEN_ROW_MODULES
            else []
        ),
    )


@dataclass(frozen=True)
class _Done:
    tensors: set[tuple[str, ...]]
    base: set[tuple[str, ...]]
    pairs: set[tuple[str, ...]]
    gates: set[tuple[str, ...]]

    @classmethod
    def read(cls, out: Path) -> _Done:
        return cls(
            tensors=_existing_keys(out / TENSORS_FILENAME, ("checkpoint", "name")),
            base=_existing_keys(out / BASE_SPECTRA_FILENAME, ("name",)),
            pairs=_existing_keys(out / PAIRS_FILENAME, ("checkpoint_a", "checkpoint_b", "name")),
            gates=_existing_keys(out / GATE_HEADS_FILENAME, ("checkpoint", "name")),
        )


def _load_deltas(
    readers: Sequence[AlignedTensorReader], labels: Sequence[str], name: str
) -> tuple[torch.Tensor, dict[str, AlignedDelta]]:
    """Read one tensor from every checkpoint: the raw base in float32 and each aligned delta."""
    deltas: dict[str, AlignedDelta] = {}
    first_base, first_rl = readers[0].pair(name)
    base_raw32 = first_base.to(torch.float32)
    deltas[labels[0]] = aligned_delta(first_base, first_rl)
    del first_base, first_rl
    for label, reader in zip(labels[1:], readers[1:], strict=True):
        base_tensor, rl_tensor = reader.pair(name)
        deltas[label] = aligned_delta(base_tensor, rl_tensor)
        del base_tensor, rl_tensor
    return base_raw32, deltas


def _base_spectrum_record(
    site: TensorSite, base_raw32: torch.Tensor, top_k: int
) -> dict[str, object]:
    record: dict[str, object] = {
        **site.fields(),
        "shape": list(base_raw32.shape),
        "norm": frobenius_norm(base_raw32),
    }
    matrix = as_matrix(base_raw32)
    if matrix is not None:
        record.update(asdict(spectral_shape(matrix, top_k=top_k)))
    return record


def _movement_record(
    label: str, site: TensorSite, delta: AlignedDelta, top_k: int
) -> dict[str, object]:
    row: dict[str, object] = {
        "checkpoint": label,
        **site.fields(),
        **asdict(tensor_movement(delta)),
    }
    matrix = as_matrix(delta.delta32)
    if matrix is not None:
        row.update(_prefixed("delta_", asdict(spectral_shape(matrix, top_k=top_k))))
    return row


def _analyze_tensor(
    run: GeometryRun,
    site: TensorSite,
    work: _TensorWork,
    readers: Sequence[AlignedTensorReader],
    token_counts: TokenCounts | None,
) -> None:
    """Compute and append everything one tensor still owes."""
    out = run.out_dir
    labels = run.labels
    base_raw32, deltas = _load_deltas(readers, labels, site.name)
    if work.base:
        _append_jsonl(
            out / BASE_SPECTRA_FILENAME,
            [_base_spectrum_record(site, base_raw32, run.top_k_spectrum)],
        )
    del base_raw32
    _append_jsonl(
        out / TENSORS_FILENAME,
        [
            _movement_record(label, site, deltas[label], run.top_k_spectrum)
            for label in work.tensors
        ],
    )
    _append_jsonl(
        out / PAIRS_FILENAME,
        [
            {
                "checkpoint_a": labels[i],
                "checkpoint_b": labels[j],
                **site.fields(),
                **delta_relationship(
                    name=site.name,
                    a=deltas[labels[i]],
                    b=deltas[labels[j]],
                    n_floor_pairs=run.n_floor_pairs,
                ).record(),
            }
            for i, j in work.pairs
        ],
    )
    _append_jsonl(
        out / GATE_HEADS_FILENAME,
        [
            record
            for label in work.gates
            for record in gate_head_records(site, label, deltas[label])
        ],
    )
    for label in work.rows:
        frame = token_row_frame(
            TokenRowSite(
                checkpoint=label, module=str(site.module), role=FREQUENCY_ROLES[site.module]
            ),
            base32=deltas[label].base32,
            delta32=deltas[label].delta32,
            counts=token_counts,
        )
        frame.write_parquet(out / token_rows_filename(label, site.module))


def run_geometry(run: GeometryRun, *, token_counts: TokenCounts | None) -> RunSummary:
    """Stream every aligned tensor once and append each table; skip tensors already fully on disk."""
    started = time.perf_counter()
    out = run.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(run.manifest(), indent=1) + "\n", encoding="utf-8"
    )
    if token_counts is not None and not (out / TOKEN_COUNTS_FILENAME).is_file():
        token_counts.write(out / TOKEN_COUNTS_FILENAME)
    names, readers = open_readers(run)
    done = _Done.read(out)
    n_resumed = 0
    n_computed = 0
    for name in names:
        site = classify_tensor(name)
        work = _tensor_work(run, site, done)
        if not work.any:
            n_resumed += 1
            continue
        tensor_started = time.perf_counter()
        _analyze_tensor(run, site, work, readers, token_counts)
        n_computed += 1
        logger.info(
            f"{name}: layer={site.layer} module={site.module} tensors={len(work.tensors)} "
            f"pairs={len(work.pairs)} seconds={time.perf_counter() - tensor_started:.1f}"
        )
    summary = RunSummary(
        n_tensors=len(names),
        n_computed=n_computed,
        n_resumed=n_resumed,
        seconds=time.perf_counter() - started,
    )
    logger.info(
        f"geometry run complete, n_tensors={summary.n_tensors} computed={summary.n_computed} "
        f"resumed={summary.n_resumed} seconds={summary.seconds:.0f} out={out}"
    )
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_checkpoint_spec(spec: str) -> FullWeightsSource:
    """``repo@revision`` for a hub checkpoint, or an existing local directory."""
    if Path(spec).is_dir():
        return FullWeightsSource.parse(spec, None)
    if "@" not in spec:
        raise ValueError(
            f"{spec!r}: a hub checkpoint needs repo@revision (or name a local directory)"
        )
    repo_id, revision = spec.rsplit("@", 1)
    return FullWeightsSource.parse(repo_id, revision)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="compute every table for a base and its RL'd checkpoints")
    run.add_argument(
        "--base", required=True, help="base checkpoint: hub repo id or local directory"
    )
    run.add_argument("--base-revision", default=None, help="hub revision of --base")
    run.add_argument(
        "--rl",
        action="append",
        required=True,
        help="RL'd checkpoint as repo@revision or a local directory; repeat, earliest first",
    )
    run.add_argument(
        "--out", type=Path, required=True, help="output directory (resumes if present)"
    )
    run.add_argument(
        "--rollouts-dir",
        type=Path,
        default=None,
        help="directory of released rollouts JSONL for token frequencies; omitted = no residual",
    )
    run.add_argument("--n-floor-pairs", type=int, default=DEFAULT_FLOOR_PAIRS)
    run.add_argument("--top-k-spectrum", type=int, default=DEFAULT_TOP_K_SPECTRUM)
    run.add_argument("--count-workers", type=int, default=8)
    render = commands.add_parser("render", help="render the markdown tables from a finished run")
    render.add_argument("--out", type=Path, required=True, help="the run's output directory")
    render.add_argument("--top-k-tokens", type=int, default=40)
    render.add_argument("--to", type=Path, default=None, help="write here instead of stdout")
    return parser.parse_args(argv)


def _token_counts_for(args: argparse.Namespace, run: GeometryRun) -> TokenCounts | None:
    counts_path = run.out_dir / TOKEN_COUNTS_FILENAME
    if counts_path.is_file():
        counts = TokenCounts.read(counts_path)
        logger.info(f"token counts resumed from {counts_path}, n_rows={counts.n_rows}")
        return counts
    if args.rollouts_dir is None:
        return None
    _, readers = open_readers(run)
    files = rollout_files(args.rollouts_dir)
    counts = count_rollout_tokens(
        files, vocab_size=vocab_size_of(readers[0]), max_workers=args.count_workers
    )
    logger.info(
        f"token counts from {len(files)} rollout file(s): rows={counts.n_rows} "
        f"input_tokens={counts.n_input_tokens} target_tokens={counts.n_target_tokens}"
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    """Resolve the checkpoints (fetching and verifying hub sources), count tokens, run, or render."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    if args.command == "render":
        text = render_readout(args.out, top_k_tokens=args.top_k_tokens)
        if args.to is None:
            print(text)  # noqa: T201 - the render's whole purpose is stdout
        else:
            args.to.write_text(text + "\n", encoding="utf-8")
        return 0
    base = resolve_full_weights(FullWeightsSource.parse(args.base, args.base_revision))
    rl = tuple(resolve_full_weights(parse_checkpoint_spec(spec)) for spec in args.rl)
    run = GeometryRun(
        base=base,
        rl=rl,
        out_dir=args.out,
        n_floor_pairs=args.n_floor_pairs,
        top_k_spectrum=args.top_k_spectrum,
    )
    run_geometry(run, token_counts=_token_counts_for(args, run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
