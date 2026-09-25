"""Matching-sampler screen for the frozen cooperation training corpus.

This is the small live screen required before the cooperation-generalization treatment. It samples
every row in the explicit frozen training JSONL with the training sampler, keeps the complete raw
records, and reports observed reward variance and termination by family and stratum. It deliberately
does not select or filter prompts: a pure group is evidence about the proposed corpus and remains in
the trace and its audit.

The screen is separate from :mod:`games.select_prompts`'s CLI because that CLI retires its partial
file and starts a fresh timestamped sweep after completion. This screen has stable artifact names,
so a completed relaunch validates the existing trace and returns without sampling again.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from games import cooperation_corpus, select_prompts
from games.deltanet_kernels import bridge_decode_kernel
from games.generation import TRAINING_TOP_P
from games.preflight import default_cuda_allocator_config
from games.prompt_variants import (
    PROMPT_VARIANT_NONE,
    PROMPT_VARIANTS,
    apply_prompt_variant,
)
from games.provenance import git_provenance
from games.select_prompts import META_RECORD_KIND, SWEEP_RECORD_KIND
from reward_hacking import backend_cli
from reward_hacking.interp.jacobian import resolve_weights_identity

if TYPE_CHECKING:
    from collections.abc import Sequence

    from games.select_prompts import Row
    from reward_hacking.model_backend import Backend, SamplingConfig

logger = logging.getLogger(__name__)

SCREEN_SCHEMA = "cooperation-matching-sampler-screen/v1"
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"
DEFAULT_COMPLETION_TOKENS = select_prompts.required_completion_budget(DEFAULT_MODEL_ID)
DEFAULT_SAMPLES_PER_PROMPT = select_prompts.DEFAULT_SAMPLES_PER_PROMPT
DEFAULT_OUTPUT_DIR = Path("artifacts/games/cooperation-generalization/matching-screen")
TRACE_FILENAME = "matching-sampler-screen.jsonl"
SUMMARY_FILENAME = "matching-sampler-summary.json"
PARTIAL_DIRECTORY = "partial"
PARTIAL_FILENAME = "matching-sampler-screen.jsonl"
REQUIRED_ROW_COUNT = 48
EXPECTED_STRATUM_COUNT = 10
MIN_SAMPLES_PER_PROMPT = 2
EXPECTED_FAMILY_COUNTS = {
    "twin-pd": 16,
    "stag-hunt": 16,
    "trust-vs-stated-return": 16,
}


@dataclass(frozen=True, slots=True)
class ScreenConfig:
    """The screen's frozen runtime choices and artifact location."""

    corpus_path: Path
    output_dir: Path
    model_id: str = DEFAULT_MODEL_ID
    samples_per_prompt: int = DEFAULT_SAMPLES_PER_PROMPT
    thinking: bool = True
    prefilled_think: bool = True
    chunk_size: int | None = None
    training_top_p: float = TRAINING_TOP_P
    training_max_new_tokens: int = DEFAULT_COMPLETION_TOKENS
    prompt_variant: str = PROMPT_VARIANT_NONE

    def __post_init__(self) -> None:
        """Reject settings that cannot measure a matching group or match training."""
        if self.samples_per_prompt < MIN_SAMPLES_PER_PROMPT:
            raise ValueError(
                f"samples_per_prompt must be at least 2 to measure within-group variance, got "
                f"{self.samples_per_prompt}"
            )
        if not self.model_id:
            raise ValueError("model_id must be non-empty")
        if not self.thinking:
            raise ValueError(
                "the cooperation matching screen is pinned to thinking mode; use the 32768-token "
                "training sampler rather than screening a different policy"
            )
        if not self.prefilled_think:
            raise ValueError(
                "the Qwen3.5 training template pre-fills the opening thinking tag; a screen with a "
                "different completion convention is not comparable to training"
            )
        if self.chunk_size is not None and self.chunk_size < 1:
            raise ValueError(f"chunk_size must be positive when supplied, got {self.chunk_size}")
        if not 0.0 < self.training_top_p <= 1.0:
            raise ValueError(f"training_top_p must be in (0, 1], got {self.training_top_p}")
        if self.training_max_new_tokens < 1:
            raise ValueError(
                f"training_max_new_tokens must be positive, got {self.training_max_new_tokens}"
            )
        if self.prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"unknown prompt variant {self.prompt_variant!r}; expected one of "
                f"{list(PROMPT_VARIANTS)}"
            )


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """Paths and accounting returned by one screen invocation."""

    trace_path: Path
    summary_path: Path
    partial_path: Path
    n_resumed: int
    n_generated: int
    n_torn: int
    noop: bool = False


def trace_path(config: ScreenConfig) -> Path:
    """Return the stable completed trace path for this screen."""
    return config.output_dir / TRACE_FILENAME


def summary_path(config: ScreenConfig) -> Path:
    """Return the stable post-screen audit path for this screen."""
    return config.output_dir / SUMMARY_FILENAME


def partial_path(config: ScreenConfig) -> Path:
    """Return the durable resumable raw-record path for this screen."""
    return config.output_dir / PARTIAL_DIRECTORY / PARTIAL_FILENAME


def _canonical(value: object) -> object:
    """Round-trip JSON values so identity comparisons use one representation."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _canonical_mapping(value: object, *, context: str) -> dict[str, Any]:
    """Return a JSON-canonical mapping while preserving string-keyed identity fields."""
    canonical = _canonical(value)
    if not isinstance(canonical, dict) or any(not isinstance(key, str) for key in canonical):
        raise TypeError(f"{context} must be a JSON object")
    return dict(canonical)


def _prompt_order_digest(rows: Sequence[Row]) -> str:
    return hashlib.sha256(
        "\n".join(str(row["prompt_id"]) for row in rows).encode("utf-8")
    ).hexdigest()


def _parse_rows(path: Path, raw: bytes) -> list[Row]:
    """Parse one frozen JSONL corpus without changing its row order."""
    rows: list[Row] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"{path} line {line_number} is not a JSON object")
        rows.append(dict(payload))
    return rows


def _validate_frozen_rows(rows: Sequence[Row]) -> None:
    """Check complete corpus membership and identity before any backend is used."""
    if len(rows) != REQUIRED_ROW_COUNT:
        raise ValueError(
            f"frozen training corpus must contain exactly {REQUIRED_ROW_COUNT} rows, got {len(rows)}; "
            "the matching screen cannot silently screen a subset"
        )
    required_columns = (
        "prompt",
        "prompt_id",
        "grading",
        "game_id",
        "payoff_variant",
        "framing_id",
    )
    missing = sorted({column for row in rows for column in required_columns if column not in row})
    if missing:
        raise ValueError(f"frozen training corpus lacks required columns {missing}")
    prompt_ids = [str(row["prompt_id"]) for row in rows]
    if any(not prompt_id for prompt_id in prompt_ids):
        raise ValueError("frozen training corpus contains an empty prompt_id")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("frozen training corpus repeats a prompt_id")
    if any(not isinstance(row["prompt"], str) or not row["prompt"] for row in rows):
        raise ValueError("frozen training corpus contains a missing or empty prompt")
    family_counts = {
        game_id: sum(str(row["game_id"]) == game_id for row in rows)
        for game_id in EXPECTED_FAMILY_COUNTS
    }
    if family_counts != EXPECTED_FAMILY_COUNTS:
        raise ValueError(
            f"frozen training corpus family counts are {family_counts}, expected "
            f"{EXPECTED_FAMILY_COUNTS}; no family may be silently omitted"
        )
    gradings = {str(row["grading"]) for row in rows}
    if len(gradings) != 1:
        raise ValueError(f"frozen training corpus carries multiple gradings: {sorted(gradings)}")
    if gradings != {cooperation_corpus.GRADING}:
        raise ValueError(
            f"frozen training corpus grading is {sorted(gradings)}, expected the planned "
            f"{cooperation_corpus.GRADING!r} care grading"
        )
    cooperation_corpus.assert_matrix_balance(rows)
    strata = cooperation_corpus.expected_stratum_keys(rows)
    if len(strata) != EXPECTED_STRATUM_COUNT:
        raise ValueError(
            f"frozen training corpus has {len(strata)} strata, expected all "
            f"{EXPECTED_STRATUM_COUNT} proposed strata; the screen cannot silently replace or "
            "filter strata"
        )


def load_frozen_training_rows(path: Path) -> tuple[list[Row], str]:
    """Load and validate the complete 48-row training corpus, returning its byte digest.

    The digest covers the exact bytes parsed, while the validation checks the row-level identity and
    all three family counts. The screen never derives a replacement corpus and never drops rows
    after this function returns.
    """
    if not path.is_file():
        raise FileNotFoundError(f"frozen training corpus does not exist: {path}")
    raw = path.read_bytes()
    rows = _parse_rows(path, raw)
    _validate_frozen_rows(rows)
    return rows, hashlib.sha256(raw).hexdigest()


def validate_backend_kind(kind: str) -> None:
    """Refuse hosted transports before backend construction or any paid call."""
    if kind in backend_cli.LOCAL_KINDS or kind == "mock":
        return
    raise ValueError(
        f"cooperation matching screen requires a local backend, got {kind!r}; hosted backends and "
        "cloud inference are outside this screen"
    )


def validate_backend_transport(backend: Backend) -> None:
    """Refuse a hosted backend even when a caller bypasses the CLI parser."""
    if backend.transport in backend_cli.HOSTED_KINDS | {"bedrock-converse"}:
        raise ValueError(
            f"cooperation matching screen requires a local backend, got transport "
            f"{backend.transport!r}; hosted inference is outside this screen"
        )


def _sampler_identity(backend: Backend, *, expected: SamplingConfig) -> dict[str, Any]:
    """Return and validate the effective sampler, including the offline mock fallback."""
    provenance = select_prompts.backend_provenance(backend)
    sampling = provenance["sampling"]
    if sampling is None:
        if backend.transport != "mock":
            raise ValueError(
                f"backend {backend.transport!r} exposes no sampler identity; refusing a trace that "
                "cannot prove it used the training sampler"
            )
        sampling = dataclasses.asdict(expected)
    if not isinstance(sampling, Mapping):
        raise TypeError(
            f"backend sampler provenance must be a mapping, got {type(sampling).__name__}"
        )
    sampler_identity = _canonical_mapping(sampling, context="backend sampler provenance")
    expected_identity = _canonical_mapping(dataclasses.asdict(expected), context="training sampler")
    if sampler_identity != expected_identity:
        raise ValueError(
            "backend sampler does not match the exact training sampler; "
            f"actual={sampler_identity!r}, expected={expected_identity!r}"
        )
    return sampler_identity


def _backend_provenance(backend: Backend, sampler: Mapping[str, Any]) -> dict[str, object]:
    provenance = dict(select_prompts.backend_provenance(backend))
    provenance["sampling"] = dict(sampler)
    return provenance


def _apply_prompt_variant(rows: Sequence[Row], variant: str) -> list[Row]:
    """Apply the configured stimulus variant to every already-validated corpus row."""
    applied_rows: list[Row] = []
    for row in rows:
        prompt = row["prompt"]
        if not isinstance(prompt, str):
            raise TypeError("validated frozen training row has a non-string prompt")
        applied_rows.append({**row, "prompt": apply_prompt_variant(prompt, variant)})
    return applied_rows


def screen_identity(  # noqa: PLR0913 - the identity names each resume axis explicitly
    *,
    backend: Backend,
    config: ScreenConfig,
    rows: Sequence[Row],
    rows_sha256: str,
    model_weights_identity: str,
    sampler_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the identity that gates partial resume and completed relaunches."""
    if backend.model_id != config.model_id:
        raise ValueError(
            f"backend serves model {backend.model_id!r}, but the screen is configured for "
            f"{config.model_id!r}"
        )
    if not model_weights_identity:
        raise ValueError("model_weights_identity must be non-empty")
    identity = select_prompts.sweep_identity(
        backend=backend,
        rows=rows,
        samples_per_prompt=config.samples_per_prompt,
        prefilled_think=config.prefilled_think,
        thinking=config.thinking,
        grading=str(rows[0]["grading"]),
        rows_sha256=rows_sha256,
    )
    identity["schema"] = SCREEN_SCHEMA
    identity["model_id"] = config.model_id
    identity["model_weights_identity"] = model_weights_identity
    identity["prompt_id_order_sha256"] = _prompt_order_digest(rows)
    identity["prompt_pool_digest"] = select_prompts.pool_digest(rows)
    identity["sampler_identity"] = dict(sampler_identity)
    identity["samples_per_prompt"] = config.samples_per_prompt
    identity["training_top_p"] = config.training_top_p
    identity["training_max_new_tokens"] = config.training_max_new_tokens
    identity["prompt_variant"] = config.prompt_variant
    return _canonical_mapping(identity, context="screen identity")


def _trace_meta(  # noqa: PLR0913 - the trace header records every identity axis explicitly
    *,
    backend: Backend,
    config: ScreenConfig,
    rows: Sequence[Row],
    rows_sha256: str,
    model_weights_identity: str,
    sampler_identity: Mapping[str, Any],
    identity: Mapping[str, Any],
    sweep: select_prompts.SweepPass,
    partial: Path,
    kernel_bridge: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the prompt-, weight- and sampler-complete final trace header."""
    return {
        "record_kind": META_RECORD_KIND,
        "schema": SCREEN_SCHEMA,
        "written_at": datetime.now(UTC).isoformat(),
        "corpus_path": str(config.corpus_path),
        "corpus_sha256": rows_sha256,
        # The corpus helper's provenance gate uses the shared `rows_sha256` spelling. Keep the
        # screen-specific corpus field above as well so readers can distinguish the input role.
        "rows_sha256": rows_sha256,
        "prompt_pool_digest": select_prompts.pool_digest(rows),
        "prompt_id_order_sha256": _prompt_order_digest(rows),
        "n_prompts": len(rows),
        "samples_per_prompt": config.samples_per_prompt,
        "thinking": config.thinking,
        "prefilled_think": config.prefilled_think,
        "prompt_variant": config.prompt_variant,
        "model_weights_identity": model_weights_identity,
        "sampler_identity": dict(sampler_identity),
        "backend": _backend_provenance(backend, sampler_identity),
        "screen_identity": dict(identity),
        "resume": select_prompts.sweep_resume_block(sweep.sessions, partial_path=partial),
        "deltanet_kernel_bridge": None if kernel_bridge is None else dict(kernel_bridge),
        **git_provenance(),
    }


def _write_jsonl_atomically(
    path: Path,
    *,
    meta: Mapping[str, object],
    records: Sequence[select_prompts.PromptSweepRecord],
) -> None:
    """Write a completed trace through a same-directory temporary file."""
    if path.exists():
        raise FileExistsError(f"completed screen trace already exists at {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    select_prompts.write_sweep_trace(temporary, meta=dict(meta), records=records)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_summary_atomically(
    *,
    trace: Path,
    corpus: Path,
    output: Path,
    model_id: str,
    sampler_identity: Mapping[str, Any],
) -> None:
    """Write the production-scorer post-screen audit without exposing prompt text."""
    if output.exists():
        raise FileExistsError(f"completed screen summary already exists at {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    cooperation_corpus.write_post_screen_summary(
        trace,
        corpus,
        temporary,
        expected_model_id=model_id,
        expected_sampler_identity=sampler_identity,
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(output)


def _validate_completed_trace(  # noqa: PLR0913 - all completed trace identities are checked
    *,
    config: ScreenConfig,
    rows: Sequence[Row],
    rows_sha256: str,
    model_weights_identity: str,
    sampler_identity: Mapping[str, Any],
    identity: Mapping[str, Any],
    trace: Path,
) -> dict[str, Any]:
    """Validate an existing complete trace and return its deterministically derived summary."""
    entries = select_prompts.read_jsonl(trace)
    if not entries or entries[0].get("record_kind") != META_RECORD_KIND:
        raise ValueError(f"completed screen trace {trace} has no leading sweep-meta record")
    meta = entries[0]
    if meta.get("schema") != SCREEN_SCHEMA:
        raise ValueError(f"completed screen trace {trace} has an incompatible schema")
    checks = {
        "corpus_sha256": (meta.get("corpus_sha256"), rows_sha256),
        "prompt_pool_digest": (meta.get("prompt_pool_digest"), select_prompts.pool_digest(rows)),
        "prompt_id_order_sha256": (meta.get("prompt_id_order_sha256"), _prompt_order_digest(rows)),
        "model_weights_identity": (meta.get("model_weights_identity"), model_weights_identity),
        "screen_identity": (meta.get("screen_identity"), dict(identity)),
    }
    for name, (found, expected) in checks.items():
        if _canonical(found) != _canonical(expected):
            raise ValueError(f"completed screen {name} differs from the requested identity")
    return cooperation_corpus.summarize_sweep_trace(
        trace,
        expected_prompt_ids=[str(row["prompt_id"]) for row in rows],
        expected_model_id=config.model_id,
        expected_sampler_identity=sampler_identity,
        expected_rows_sha256=rows_sha256,
        expected_strata=cooperation_corpus.expected_stratum_keys(rows),
    ).to_json_dict()


def _validate_completed_artifacts(  # noqa: PLR0913 - all completed artifact identities are checked
    *,
    config: ScreenConfig,
    rows: Sequence[Row],
    rows_sha256: str,
    model_weights_identity: str,
    sampler_identity: Mapping[str, Any],
    identity: Mapping[str, Any],
    trace: Path,
    summary: Path,
) -> None:
    """Validate an existing final trace and summary before returning the no-op result."""
    expected_summary = _validate_completed_trace(
        config=config,
        rows=rows,
        rows_sha256=rows_sha256,
        model_weights_identity=model_weights_identity,
        sampler_identity=sampler_identity,
        identity=identity,
        trace=trace,
    )
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    if _canonical(summary_payload) != _canonical(expected_summary):
        raise ValueError(f"completed screen summary {summary} does not match its trace")


def run_matching_screen(
    backend: Backend,
    config: ScreenConfig,
    *,
    model_weights_identity: str | None = None,
    kernel_bridge: Mapping[str, object] | None = None,
) -> ScreenResult:
    """Run or resume the full frozen-corpus matching screen with a supplied backend.

    Passing a backend makes the operation straightforward to exercise offline. The CLI constructs
    the real local backend through :func:`reward_hacking.backend_cli.backend_from_args`; this
    function owns persistence and never creates a hosted fallback.
    """
    validate_backend_transport(backend)
    rows, rows_sha256 = load_frozen_training_rows(config.corpus_path)
    rows = _apply_prompt_variant(rows, config.prompt_variant)
    expected_sampler = select_prompts.training_sampler(
        config.model_id,
        top_p=config.training_top_p,
        max_new_tokens=config.training_max_new_tokens,
    )
    sampler_identity = _sampler_identity(backend, expected=expected_sampler)
    weights_identity = (
        model_weights_identity
        if model_weights_identity is not None
        else resolve_backend_weights_identity(backend, config.model_id)
    )
    identity = screen_identity(
        backend=backend,
        config=config,
        rows=rows,
        rows_sha256=rows_sha256,
        model_weights_identity=weights_identity,
        sampler_identity=sampler_identity,
    )
    trace = trace_path(config)
    summary = summary_path(config)
    partial = partial_path(config)
    if trace.exists() or summary.exists():
        if summary.exists() and not trace.exists():
            raise RuntimeError(
                f"screen summary {summary} exists without its completed trace {trace}; "
                "refusing to trust or overwrite the orphaned summary"
            )
        if summary.exists():
            _validate_completed_artifacts(
                config=config,
                rows=rows,
                rows_sha256=rows_sha256,
                model_weights_identity=weights_identity,
                sampler_identity=sampler_identity,
                identity=identity,
                trace=trace,
                summary=summary,
            )
        else:
            _validate_completed_trace(
                config=config,
                rows=rows,
                rows_sha256=rows_sha256,
                model_weights_identity=weights_identity,
                sampler_identity=sampler_identity,
                identity=identity,
                trace=trace,
            )
            _write_summary_atomically(
                trace=trace,
                corpus=config.corpus_path,
                output=summary,
                model_id=config.model_id,
                sampler_identity=sampler_identity,
            )
        return ScreenResult(trace, summary, partial, len(rows), 0, 0, noop=True)

    if select_prompts.rows_file_digest(config.corpus_path) != rows_sha256:
        raise RuntimeError(
            f"frozen training corpus changed while preparing the screen: {config.corpus_path}"
        )
    swept = select_prompts.resumable_sweep(
        backend,
        rows,
        samples_per_prompt=config.samples_per_prompt,
        prefilled_think=config.prefilled_think,
        chunk_size=config.chunk_size,
        record_kind=SWEEP_RECORD_KIND,
        partial_path=partial,
        identity=identity,
    )
    if len(swept.records) != len(rows):
        raise RuntimeError(
            f"matching screen returned {len(swept.records)} records for {len(rows)} frozen rows; "
            "the screen never filters the training corpus"
        )
    if select_prompts.rows_file_digest(config.corpus_path) != rows_sha256:
        raise RuntimeError(
            f"frozen training corpus changed during the screen: {config.corpus_path}"
        )
    meta = _trace_meta(
        backend=backend,
        config=config,
        rows=rows,
        rows_sha256=rows_sha256,
        model_weights_identity=weights_identity,
        sampler_identity=sampler_identity,
        identity=identity,
        sweep=swept,
        partial=partial,
        kernel_bridge=kernel_bridge,
    )
    _write_jsonl_atomically(trace, meta=meta, records=swept.records)
    _write_summary_atomically(
        trace=trace,
        corpus=config.corpus_path,
        output=summary,
        model_id=config.model_id,
        sampler_identity=sampler_identity,
    )
    return ScreenResult(
        trace,
        summary,
        partial,
        swept.n_resumed,
        swept.n_generated,
        swept.n_torn,
    )


def resolve_backend_weights_identity(backend: Backend, model_id: str) -> str:
    """Resolve the exact local weights identity, with an explicit mock marker for tests/smoke."""
    if backend.transport == "mock":
        return f"mock:{model_id}"
    model_path = getattr(backend, "model_path", None)
    source = model_id if model_path is None else str(model_path)
    return resolve_weights_identity(source)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen every frozen cooperation training row at the matching local sampler."
    )
    parser.add_argument(
        "--corpus", "--training-corpus", dest="corpus_path", type=Path, required=True
    )
    parser.add_argument(
        "--out-dir", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hub id or immutable local snapshot from which the backend loads weights.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Canonical model identity used to select and record the measured training sampler.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=DEFAULT_SAMPLES_PER_PROMPT,
        help=f"Completions per frozen training row (default: {DEFAULT_SAMPLES_PER_PROMPT}).",
    )
    parser.add_argument(
        "--training-top-p",
        type=float,
        default=TRAINING_TOP_P,
        help=f"Top-p used by the training run being screened (default: {TRAINING_TOP_P}).",
    )
    parser.add_argument(
        "--training-max-new-tokens",
        type=int,
        default=DEFAULT_COMPLETION_TOKENS,
        help=(
            "Completion cap used by the training run being screened "
            f"(default: {DEFAULT_COMPLETION_TOKENS})."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=PROMPT_VARIANTS,
        default=PROMPT_VARIANT_NONE,
        help="Named prompt edit applied before sampling (default: none).",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    backend_cli.add_backend_args(parser, default="hf")
    return parser.parse_args(argv)


def _configured_training_sampler(args: argparse.Namespace) -> SamplingConfig:
    """Build the sampler named by the screen's explicit training configuration flags."""
    return select_prompts.training_sampler(
        args.model_id,
        top_p=args.training_top_p,
        max_new_tokens=args.training_max_new_tokens,
    )


def _prepare_cli_args(args: argparse.Namespace) -> None:
    """Pin the CLI to the 9B thinking-on research sampler before model construction."""
    validate_backend_kind(args.backend)
    if args.model_id is None:
        if args.model != DEFAULT_MODEL_ID:
            raise ValueError(
                "--model-id is required when --model names a non-default load source; sampler "
                "semantics cannot be inferred from a local snapshot path"
            )
        args.model_id = args.model
    if args.thinking is False:
        raise ValueError(
            "the cooperation matching screen requires --thinking and the configured training "
            "completion budget"
        )
    args.thinking = True
    expected = _configured_training_sampler(args)
    effective = backend_cli.local_sampling_from_args(args, expected)
    if effective != expected:
        raise ValueError(
            "the cooperation matching screen must use the exact training sampler; remove decoding "
            "overrides or record a separately reviewed research configuration"
        )
    if args.model_id != DEFAULT_MODEL_ID:
        logger.warning(
            "screen model %s differs from the planned %s; the model identity is retained in every "
            "artifact and this is not a 9B research result",
            args.model_id,
            DEFAULT_MODEL_ID,
        )


def _build_backend(args: argparse.Namespace) -> Backend:
    """Build a backend labelled by the canonical model while loading the requested snapshot."""
    return backend_cli.backend_from_args(
        args,
        args.model_id,
        local_sampling=_configured_training_sampler(args),
        mock_responses=select_prompts.MOCK_RESPONSES,
        extra_kwargs={"model_path": args.model} if args.model != args.model_id else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local 9B matching screen from the command line."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
    )
    default_cuda_allocator_config()
    args = _parse_args(argv)
    _prepare_cli_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    kernel_bridge = bridge_decode_kernel()
    backend = _build_backend(args)
    weights_identity = resolve_backend_weights_identity(backend, args.model_id)
    result = run_matching_screen(
        backend,
        ScreenConfig(
            corpus_path=args.corpus_path,
            output_dir=args.output_dir,
            model_id=args.model_id,
            samples_per_prompt=args.samples_per_prompt,
            chunk_size=args.chunk_size,
            training_top_p=args.training_top_p,
            training_max_new_tokens=args.training_max_new_tokens,
            prompt_variant=args.prompt_variant,
        ),
        model_weights_identity=weights_identity,
        kernel_bridge=kernel_bridge,
    )
    logger.info(
        "matching screen %s: trace=%s summary=%s resumed=%d generated=%d torn=%d",
        "reused" if result.noop else "complete",
        result.trace_path,
        result.summary_path,
        result.n_resumed,
        result.n_generated,
        result.n_torn,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
