"""Build CPU-only Jacobian-lens subspace bundles for residual ablations.

The bundle contains one orthonormal row basis per fitted source layer.  A row is the
first-order residual direction for increasing one selected lens token's pre-norm logit:
``J_l.T @ (W_U[token] * (1 + final_norm_gain))``.  The model trunk is never loaded here;
only its final RMSNorm, lm-head, and tokenizer are read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from games.interp_cells import sha256_of_file
from games.workspace_readout import LoadedLens, StoredUnembed, load_lens, load_unembed

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_PLACEBO_SUFFIX = "-placebo"
MATRIX_NDIM = 2
_BAND_RE = re.compile(r"(?:L)?(-?\d+)(?::|-L)(-?\d+)")


def _parse_source_band(value: str, *, source_layers: Sequence[int]) -> tuple[int, int]:
    """Parse a half-open layer band against the fitted source-layer range."""
    if not source_layers:
        raise ValueError("lens has no fitted source layers")
    match = _BAND_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid band {value!r}; expected START:END")
    n_layers = max(source_layers) + 1
    start, end = (int(match.group(index)) for index in (1, 2))
    if start < 0:
        start += n_layers
    if end < 0:
        end += n_layers
    if not 0 <= start < end <= n_layers:
        raise ValueError(f"band {value!r} resolves to [{start}, {end}), outside 0..{n_layers}")
    return start, end


def token_ids_from_tokenizer(
    tokenizer: PreTrainedTokenizerBase, tokens: Sequence[str]
) -> list[int]:
    """Resolve strings against the exact decoded vocabulary used by workspace readout."""
    if not tokens or any(not token for token in tokens):
        raise ValueError("at least one token is required")
    wanted = set(tokens)
    matches: dict[str, list[int]] = {token: [] for token in wanted}
    for token_id in range(int(tokenizer.vocab_size)):
        decoded = str(tokenizer.decode([token_id]))
        if decoded in matches:
            matches[decoded].append(token_id)
    ambiguous = {token: ids for token, ids in matches.items() if len(ids) != 1}
    if ambiguous:
        raise ValueError(f"token strings must each identify one decoded vocabulary id: {ambiguous}")
    return [matches[token][0] for token in tokens]


def _load_tokenizer(model_path: Path) -> PreTrainedTokenizerBase:
    """Load the model tokenizer without constructing a model or allocating CUDA memory."""
    transformers = importlib.import_module("transformers")
    return cast(
        "PreTrainedTokenizerBase", transformers.AutoTokenizer.from_pretrained(str(model_path))
    )


def _load_analysis_tokens(payload: object, *, analysis_key: str, top_n: int) -> list[str]:
    """Extract a bounded risen-token list from a workspace-readout analysis object."""
    if top_n <= 0:
        raise ValueError("top_n must be positive when loading from analysis")
    if not isinstance(payload, dict):
        raise TypeError("analysis JSON must be an object")
    paired = payload.get("paired_differences")
    if not isinstance(paired, dict) or analysis_key not in paired:
        raise ValueError(f"analysis JSON has no paired_differences key {analysis_key!r}")
    entry = paired[analysis_key]
    if not isinstance(entry, dict) or not isinstance(entry.get("risen"), list):
        raise TypeError(f"analysis key {analysis_key!r} has no risen list")
    selected: list[str] = []
    for row in cast("list[object]", entry["risen"])[:top_n]:
        if not isinstance(row, dict) or not isinstance(row.get("token"), str):
            raise TypeError(f"analysis key {analysis_key!r} contains a malformed risen row")
        selected.append(str(row["token"]))
    if not selected:
        raise ValueError(f"analysis key {analysis_key!r} has no risen tokens")
    return selected


def load_token_spec(
    path: Path,
    *,
    analysis_key: str | None = None,
    top_n: int | None = None,
) -> list[str]:
    """Load either a JSON token list or the first ``top_n`` risen tokens from an analysis file."""
    if (analysis_key is None) != (top_n is None):
        raise ValueError("analysis_key and top_n must be supplied together")
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if analysis_key is not None:
        if top_n is None:
            raise ValueError("analysis_key and top_n must be supplied together")
        return _load_analysis_tokens(payload, analysis_key=analysis_key, top_n=top_n)
    if not isinstance(payload, list) or not all(isinstance(token, str) for token in payload):
        raise ValueError("tokens JSON must be a list of token strings")
    tokens = cast("list[str]", payload)
    if not tokens:
        raise ValueError("tokens JSON must contain at least one token")
    return tokens


def effective_token_vectors(
    lens: LoadedLens,
    unembed: StoredUnembed,
    token_ids: Sequence[int],
    *,
    layers: Sequence[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Return ``J_l.T @ (W_U[token] * (1 + g))`` rows for each requested layer."""
    selected_layers = tuple(lens.source_layers if layers is None else layers)
    if not selected_layers:
        raise ValueError("at least one fitted layer is required")
    if not token_ids:
        raise ValueError("at least one token id is required")
    if unembed.norm_weight.ndim != 1 or unembed.lm_head_weight.ndim != MATRIX_NDIM:
        raise ValueError("unembed tensors must be one-dimensional norm and two-dimensional head")
    if (
        unembed.norm_weight.shape[0] != lens.d_model
        or unembed.lm_head_weight.shape[1] != lens.d_model
    ):
        raise ValueError(
            f"lens d_model {lens.d_model} disagrees with unembed shapes "
            f"{tuple(unembed.norm_weight.shape)} and {tuple(unembed.lm_head_weight.shape)}"
        )
    vocabulary_size = int(unembed.lm_head_weight.shape[0])
    if any(token_id < 0 or token_id >= vocabulary_size for token_id in token_ids):
        raise ValueError(f"token ids must be in 0..{vocabulary_size - 1}")
    effective_rows = unembed.lm_head_weight.float()[list(token_ids)] * (
        1.0 + unembed.norm_weight.float()
    ).unsqueeze(0)
    vectors: dict[int, torch.Tensor] = {}
    for layer in selected_layers:
        if layer not in lens.jacobians:
            raise ValueError(f"layer {layer} is not in the fitted lens")
        jacobian = lens.jacobians[layer].float().cpu()
        vectors[int(layer)] = effective_rows.float().cpu() @ jacobian
    return vectors


def _canonicalize_rows(rows: torch.Tensor) -> torch.Tensor:
    """Fix each basis row's sign at its largest-magnitude coordinate."""
    result = rows.clone()
    for index in range(result.shape[0]):
        pivot = int(result[index].abs().argmax().item())
        if result[index, pivot] < 0:
            result[index] *= -1
    return result


def orthonormalize_rows(rows: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Return an orthonormal basis for the row span and its effective rank."""
    if rows.ndim != MATRIX_NDIM or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError(f"rows must be a non-empty matrix, got shape {tuple(rows.shape)}")
    cpu_rows = rows.detach().float().cpu().contiguous()
    if not bool(torch.isfinite(cpu_rows).all()):
        raise ValueError("rows contain non-finite values")
    rank = int(torch.linalg.matrix_rank(cpu_rows).item())
    if rank <= 0:
        raise ValueError("rows have zero effective rank")
    _, _, right_singular_vectors = torch.linalg.svd(cpu_rows, full_matrices=False)
    row_space = right_singular_vectors[:rank]
    q, _ = torch.linalg.qr(row_space.T, mode="reduced")
    basis = _canonicalize_rows(q.T.contiguous())
    if not torch.allclose(basis @ basis.T, torch.eye(rank), rtol=1e-5, atol=1e-5):
        raise RuntimeError("orthonormalization produced a non-orthonormal basis")
    return basis, rank


def build_real_bundle(
    lens: LoadedLens,
    unembed: StoredUnembed,
    *,
    token_ids: Sequence[int],
    band: tuple[int, int],
) -> tuple[dict[int, torch.Tensor], dict[int, int]]:
    """Build a rank-reduced, orthonormal real-token bundle over a source-layer band."""
    start, end = band
    if start < 0 or end <= start:
        raise ValueError(f"invalid band {band!r}")
    layers = tuple(layer for layer in lens.source_layers if start <= layer < end)
    if not layers:
        raise ValueError(f"band {band!r} contains no fitted source layers")
    raw_vectors = effective_token_vectors(lens, unembed, token_ids, layers=layers)
    bundle: dict[int, torch.Tensor] = {}
    ranks: dict[int, int] = {}
    for layer in layers:
        basis, rank = orthonormalize_rows(raw_vectors[layer])
        bundle[layer] = basis
        ranks[layer] = rank
    return bundle, ranks


def build_placebo_bundle(
    real_bundle: Mapping[int, torch.Tensor], *, identity_sha256: str
) -> dict[int, torch.Tensor]:
    """Build a deterministic per-layer Gaussian placebo from the real identity digest."""
    if not real_bundle:
        raise ValueError("real bundle is empty")
    placebo: dict[int, torch.Tensor] = {}
    for layer in sorted(real_bundle):
        real_basis = real_bundle[layer]
        if real_basis.ndim != MATRIX_NDIM or real_basis.shape[0] == 0 or real_basis.shape[1] == 0:
            raise ValueError(
                f"real bundle layer {layer} has invalid shape {tuple(real_basis.shape)}"
            )
        seed_digest = hashlib.sha256(f"{identity_sha256}\0{layer}".encode()).digest()
        seed = int.from_bytes(seed_digest[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_rows = torch.randn(
            real_basis.shape,
            generator=generator,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        basis, rank = orthonormalize_rows(random_rows)
        if rank != real_basis.shape[0]:
            raise RuntimeError(f"placebo layer {layer} unexpectedly has rank {rank}")
        placebo[int(layer)] = basis
    return placebo


def load_bundle(path: Path) -> dict[int, torch.Tensor]:
    """Load and validate a CPU orthonormal subspace bundle."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} does not hold a non-empty layer bundle")
    bundle: dict[int, torch.Tensor] = {}
    d_model: int | None = None
    for layer, basis in cast("dict[object, object]", payload).items():
        if (
            not isinstance(layer, int)
            or layer < 0
            or not isinstance(basis, torch.Tensor)
            or basis.ndim != MATRIX_NDIM
        ):
            raise ValueError(f"{path} must map int layer -> 2-D tensor")
        if basis.shape[0] == 0 or basis.shape[1] == 0:
            raise ValueError(f"{path} layer {layer} has an empty basis")
        if d_model is None:
            d_model = int(basis.shape[1])
        elif basis.shape[1] != d_model:
            raise ValueError(f"{path} has inconsistent d_model dimensions")
        basis_cpu = basis.float().cpu().contiguous()
        if not bool(torch.isfinite(basis_cpu).all()):
            raise ValueError(f"{path} layer {layer} contains non-finite values")
        if not torch.allclose(
            basis_cpu @ basis_cpu.T,
            torch.eye(basis_cpu.shape[0]),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise ValueError(f"{path} layer {layer} basis is not orthonormal")
        bundle[layer] = basis_cpu
    return bundle


def _identity_digest(identity: Mapping[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_bundle(
    path: Path, bundle: Mapping[int, torch.Tensor], sidecar: Mapping[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {int(layer): basis.float().cpu().contiguous() for layer, basis in bundle.items()}, path
    )
    path.with_suffix(".json").write_text(
        json.dumps(dict(sidecar), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_bundle(  # noqa: PLR0913
    *,
    lens_path: Path,
    model_path: Path,
    token_strings: Sequence[str],
    band: tuple[int, int],
    out: Path,
    placebo_out: Path | None = None,
) -> dict[str, object]:
    """Build the real bundle and deterministic sibling placebo from local CPU artifacts."""
    if placebo_out is None:
        placebo_out = out.with_name(out.stem + DEFAULT_PLACEBO_SUFFIX + out.suffix)
    if out.resolve() == placebo_out.resolve():
        raise ValueError("real and placebo bundle paths must differ")
    lens = load_lens(lens_path)
    unembed = load_unembed(model_path)
    token_ids = token_ids_from_tokenizer(_load_tokenizer(model_path), token_strings)
    bundle, ranks = build_real_bundle(lens, unembed, token_ids=token_ids, band=band)
    identity: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "tokens": list(token_strings),
        "token_ids": token_ids,
        "band": list(band),
        "layers": sorted(bundle),
        "d_model": lens.d_model,
        "lens_sha256": sha256_of_file(lens_path),
    }
    real_identity_sha256 = _identity_digest(identity)
    real_sidecar = {
        **identity,
        "kind": "real",
        "effective_rank": {str(layer): rank for layer, rank in ranks.items()},
        "identity_sha256": real_identity_sha256,
    }
    _write_bundle(out, bundle, real_sidecar)
    placebo = build_placebo_bundle(bundle, identity_sha256=real_identity_sha256)
    placebo_sidecar = {
        **identity,
        "kind": "placebo",
        "effective_rank": {str(layer): int(basis.shape[0]) for layer, basis in placebo.items()},
        "real_identity_sha256": real_identity_sha256,
        "identity_sha256": _identity_digest(
            {"kind": "placebo", "real_identity_sha256": real_identity_sha256}
        ),
    }
    _write_bundle(placebo_out, placebo, placebo_sidecar)
    logger.info("workspace bundles written: real=%s placebo=%s", out, placebo_out)
    return real_sidecar


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-bundle", help="Build real and placebo CPU bundles.")
    build.add_argument("--lens", type=Path, required=True)
    build.add_argument("--model-path", type=Path, required=True)
    tokens = build.add_mutually_exclusive_group(required=True)
    tokens.add_argument("--tokens-json", type=Path)
    tokens.add_argument("--from-analysis", type=Path)
    build.add_argument("--key", default=None, help="paired_differences key for --from-analysis")
    build.add_argument("--top-n", type=int, default=None)
    build.add_argument("--band", required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--placebo-out", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the CPU-only bundle builder."""
    args = _parser().parse_args(argv)
    if args.tokens_json is not None:
        if args.key is not None or args.top_n is not None:
            raise ValueError("--key and --top-n require --from-analysis")
        token_strings = load_token_spec(args.tokens_json)
    else:
        if args.key is None or args.top_n is None:
            raise ValueError("--from-analysis requires --key and --top-n")
        token_strings = load_token_spec(
            args.from_analysis,
            analysis_key=args.key,
            top_n=args.top_n,
        )
    lens = load_lens(args.lens)
    band = _parse_source_band(args.band, source_layers=lens.source_layers)
    build_bundle(
        lens_path=args.lens,
        model_path=args.model_path,
        token_strings=token_strings,
        band=band,
        out=args.out,
        placebo_out=args.placebo_out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
