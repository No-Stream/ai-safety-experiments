"""CPU analysis of how two same-model fitted Jacobian lenses differ."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict, cast

import torch
from safetensors import safe_open

logger: logging.Logger = logging.getLogger(__name__)

MODEL_ID = "google/gemma-3-12b-it"
CONTENT_LAYERS = (12, 26, 40)
CONTENT_DIRECTIONS = 8
TOKEN_COUNT = 20
SUBSPACE_KS = (16, 64)
RANDOM_SEED = 20261005
OVERSAMPLING = 16
POWER_ITERATIONS = 4
MATRIX_DIMENSIONS = 2
EXACT_SVD_THRESHOLD = 256
EMBEDDING_KEY = "language_model.model.embed_tokens.weight"
FINAL_NORM_KEY = "language_model.model.norm.weight"


class JacobianLensLike(Protocol):
    """Minimum lens interface needed for same-model matrix comparison."""

    jacobians: dict[int, torch.Tensor]
    source_layers: list[int]
    n_prompts: int
    d_model: int

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Transport one residual direction from its source layer."""
        ...


class TokenizerLike(Protocol):
    """Tokenizer interface used to map output-logit ids to vocabulary tokens."""

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Return one vocabulary token string for each supplied token id."""
        ...


class MatrixPairMetrics(TypedDict):
    """Scalar matrix metrics and principal angles for one source layer."""

    layer: int
    relative_frobenius: float
    cosine: float
    principal_angles_degrees: dict[str, list[float]]


class LensPairMetrics(TypedDict):
    """Layer metrics for a pair of fitted lenses."""

    before_n_prompts: int
    after_n_prompts: int
    layers: dict[str, MatrixPairMetrics]


@dataclass(frozen=True)
class ReadoutAssets:
    """CPU weights and tokenizer used to decode transported residual directions."""

    embedding: torch.Tensor
    norm_weight: torch.Tensor
    epsilon: float
    tokenizer: TokenizerLike


@dataclass(frozen=True)
class ComparisonRequest:
    """Paths and model identity used for one rendered comparison."""

    model: str
    before_path: Path
    after_path: Path
    noise_path: Path | None


def _validate_lens_pair(before: JacobianLensLike, after: JacobianLensLike) -> None:
    if before.d_model != after.d_model:
        raise ValueError(f"lens widths differ: {before.d_model} != {after.d_model}")
    if before.source_layers != after.source_layers:
        raise ValueError(
            f"lens source layers differ: {before.source_layers!r} != {after.source_layers!r}"
        )
    for layer in before.source_layers:
        before_matrix = before.jacobians[layer]
        after_matrix = after.jacobians[layer]
        expected_shape = (before.d_model, before.d_model)
        if (
            tuple(before_matrix.shape) != expected_shape
            or tuple(after_matrix.shape) != expected_shape
        ):
            raise ValueError(f"layer {layer} matrices must have shape {expected_shape}")


def _leading_singular_vectors(
    matrix: torch.Tensor,
    *,
    count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.ndim != MATRIX_DIMENSIONS:
        raise ValueError(f"SVD input must be a matrix, got shape {tuple(matrix.shape)}")
    if count <= 0:
        raise ValueError(f"singular vector count must be positive, got {count}")

    matrix = matrix.float().cpu()
    limit = min(matrix.shape)
    count = min(count, limit)
    if limit <= EXACT_SVD_THRESHOLD or count == limit:
        left, singular_values, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
        return left[:, :count], singular_values[:count], right_transpose[:count, :].T

    q = min(limit - 1, count + OVERSAMPLING)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        left, singular_values, right = torch.svd_lowrank(matrix, q=q, niter=POWER_ITERATIONS)
    order = torch.argsort(singular_values, descending=True)[:count]
    left = left[:, order]
    singular_values = singular_values[order]
    right = right[:, order]
    return left, singular_values, right


def _left_singular_subspace(matrix: torch.Tensor, *, count: int, seed: int) -> torch.Tensor:
    left, _, _ = _leading_singular_vectors(matrix, count=count, seed=seed)
    return left


def compare_matrix_pair(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    layer: int,
    seed: int = RANDOM_SEED,
) -> MatrixPairMetrics:
    """Compute matrix-change, cosine, and top-left-subspace angle metrics."""
    if (
        before.ndim != MATRIX_DIMENSIONS
        or before.shape != after.shape
        or before.shape[0] != before.shape[1]
    ):
        raise ValueError(
            "Jacobian matrices must have matching square shapes, got "
            f"{tuple(before.shape)} and {tuple(after.shape)}"
        )
    before = before.float().cpu()
    after = after.float().cpu()
    before_norm = torch.linalg.vector_norm(before)
    after_norm = torch.linalg.vector_norm(after)
    if before_norm == 0 or after_norm == 0:
        raise ValueError(f"layer {layer} has a zero Jacobian norm; comparison is undefined")

    delta = after - before
    relative_frobenius = torch.linalg.vector_norm(delta) / before_norm
    cosine = torch.sum(before * after) / (before_norm * after_norm)
    principal_angles: dict[str, list[float]] = {}
    max_k = max(SUBSPACE_KS)
    before_basis = _left_singular_subspace(before, count=max_k, seed=seed)
    after_basis = _left_singular_subspace(after, count=max_k, seed=seed)
    for k in SUBSPACE_KS:
        effective_k = min(k, before.shape[0])
        principal_cosines = torch.linalg.svdvals(
            before_basis[:, :effective_k].T @ after_basis[:, :effective_k]
        ).clamp_(0.0, 1.0)
        angles = torch.rad2deg(torch.arccos(principal_cosines))
        principal_angles[str(k)] = [float(angle) for angle in angles]

    return {
        "layer": layer,
        "relative_frobenius": float(relative_frobenius),
        "cosine": float(cosine),
        "principal_angles_degrees": principal_angles,
    }


def compare_lens_pair(
    before: JacobianLensLike,
    after: JacobianLensLike,
    *,
    seed: int = RANDOM_SEED,
) -> LensPairMetrics:
    """Compare all shared source layers from two fitted Jacobian lenses."""
    _validate_lens_pair(before, after)
    layers: dict[str, MatrixPairMetrics] = {
        str(layer): compare_matrix_pair(
            before.jacobians[layer], after.jacobians[layer], layer=layer, seed=seed
        )
        for layer in before.source_layers
    }
    return {
        "before_n_prompts": before.n_prompts,
        "after_n_prompts": after.n_prompts,
        "layers": layers,
    }


def top_change_singular_vectors(
    delta: torch.Tensor,
    *,
    count: int = CONTENT_DIRECTIONS,
    seed: int = RANDOM_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return leading left vectors, values and right vectors of a Jacobian change.

    The sign is canonicalized so the largest-magnitude entry in each left vector
    is positive. The vector and its negation remain the same mathematical axis.
    """
    left, singular_values, right = _leading_singular_vectors(delta, count=count, seed=seed)
    for index in range(left.shape[1]):
        pivot = torch.argmax(torch.abs(left[:, index]))
        if left[pivot, index] < 0:
            left[:, index].neg_()
            right[:, index].neg_()
    return left, singular_values, right


def gemma_rms_norm(vector: torch.Tensor, weight: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    """Apply Gemma RMSNorm, including its `(1 + weight)` gain convention."""
    if vector.shape[-1] != weight.numel():
        raise ValueError(
            f"norm width {weight.numel()} does not match vector width {vector.shape[-1]}"
        )
    if epsilon <= 0:
        raise ValueError(f"RMSNorm epsilon must be positive, got {epsilon}")
    float_vector = vector.float()
    float_weight = weight.to(device=float_vector.device, dtype=torch.float32)
    rms = torch.rsqrt(float_vector.square().mean(dim=-1, keepdim=True) + epsilon)
    return float_vector * rms * (1.0 + float_weight)


def _top_token_scores(
    logits: torch.Tensor,
    tokenizer: TokenizerLike,
    *,
    count: int = TOKEN_COUNT,
) -> dict[str, list[dict[str, float | str]]]:
    vocab_count = logits.numel()
    count = min(count, vocab_count)
    positive_scores, positive_ids = torch.topk(logits, count)
    negative_scores, negative_ids = torch.topk(logits, count, largest=False)
    positive_tokens = tokenizer.convert_ids_to_tokens([int(token_id) for token_id in positive_ids])
    negative_tokens = tokenizer.convert_ids_to_tokens([int(token_id) for token_id in negative_ids])
    if len(positive_tokens) != count or len(negative_tokens) != count:
        raise ValueError("tokenizer returned an unexpected number of token strings")
    return {
        "positive_tokens": [
            {"token": token, "score": float(score)}
            for token, score in zip(positive_tokens, positive_scores, strict=True)
        ],
        "negative_tokens": [
            {"token": token, "score": float(-score)}
            for token, score in zip(negative_tokens, negative_scores, strict=True)
        ],
    }


def _decode_direction(
    direction: torch.Tensor,
    delta_jacobian: torch.Tensor,
    readout: ReadoutAssets,
) -> dict[str, list[dict[str, float | str]]]:
    transported = direction.float() @ delta_jacobian.float().T
    normed = gemma_rms_norm(transported, readout.norm_weight, epsilon=readout.epsilon)
    logits = readout.embedding.float() @ normed
    return _top_token_scores(logits, readout.tokenizer)


def decode_change_contents(
    before: JacobianLensLike,
    after: JacobianLensLike,
    *,
    readout: ReadoutAssets,
    layers: tuple[int, ...] = CONTENT_LAYERS,
    seed: int = RANDOM_SEED,
) -> dict[str, object]:
    """Decode right singular directions of `J_after - J_before` at selected layers."""
    _validate_lens_pair(before, after)
    if readout.embedding.ndim != MATRIX_DIMENSIONS or readout.embedding.shape[1] != before.d_model:
        raise ValueError(
            f"embedding shape {tuple(readout.embedding.shape)} is incompatible with lens width {before.d_model}"
        )
    if readout.norm_weight.ndim != 1 or readout.norm_weight.numel() != before.d_model:
        raise ValueError(f"final norm width must be {before.d_model}")

    decoded: dict[str, object] = {}
    for layer in layers:
        if layer not in before.source_layers:
            raise ValueError(f"content layer {layer} is missing from the fitted lenses")
        delta = after.jacobians[layer].float().cpu() - before.jacobians[layer].float().cpu()
        _left, singular_values, right = top_change_singular_vectors(
            delta, count=CONTENT_DIRECTIONS, seed=seed + layer
        )
        directions: list[dict[str, object]] = []
        for index in range(right.shape[1]):
            direction_readout = _decode_direction(right[:, index], delta, readout)
            directions.append(
                {
                    "rank": index + 1,
                    "singular_value": float(singular_values[index]),
                    "left_vector_orientation": "largest-magnitude component positive",
                    **direction_readout,
                }
            )

        generator = torch.Generator(device="cpu").manual_seed(seed + layer + 1_000_003)
        random_direction = torch.randn(before.d_model, generator=generator)
        random_direction /= torch.linalg.vector_norm(random_direction)
        random_control_readout = _decode_direction(random_direction, delta, readout)
        decoded[str(layer)] = {
            "difference_operator": "J_after - J_before",
            "decoded_readout": "Gemma unembedding of (J_after - J_before) @ direction",
            "directions": directions,
            "matched_norm_random_control": {
                "direction_norm": float(torch.linalg.vector_norm(random_direction)),
                **random_control_readout,
            },
        }
    return decoded


def _model_cache_snapshot(model: str) -> Path:
    if model != MODEL_ID:
        raise ValueError(f"this readout loader only supports {MODEL_ID!r}, got {model!r}")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-3-12b-it"
    revision_path = cache_root / "refs" / "main"
    if not revision_path.is_file():
        raise FileNotFoundError(f"offline Gemma cache reference is missing: {revision_path}")
    snapshot = cache_root / "snapshots" / revision_path.read_text().strip()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"offline Gemma snapshot is missing: {snapshot}")
    return snapshot


def _load_tensor_from_snapshot(snapshot: Path, tensor_key: str) -> torch.Tensor:
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Gemma safetensors index is missing: {index_path}")
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if tensor_key not in weight_map:
        raise KeyError(f"Gemma checkpoint index does not contain {tensor_key!r}")
    shard_path = snapshot / weight_map[tensor_key]
    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        return shard.get_tensor(tensor_key).to(dtype=torch.float32).contiguous()


def load_gemma_readout_weights(model: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Load Gemma's tied embedding matrix and final RMSNorm weight from local safetensors."""
    snapshot = _model_cache_snapshot(model)
    config = json.loads((snapshot / "config.json").read_text())
    hidden_size = int(config["text_config"]["hidden_size"])
    epsilon = float(config["text_config"].get("rms_norm_eps", 1e-6))
    embedding = _load_tensor_from_snapshot(snapshot, EMBEDDING_KEY)
    norm_weight = _load_tensor_from_snapshot(snapshot, FINAL_NORM_KEY)
    if tuple(embedding.shape)[1] != hidden_size or norm_weight.numel() != hidden_size:
        raise ValueError("cached Gemma embedding or final norm does not match config hidden_size")
    return embedding, norm_weight, epsilon


def load_tokenizer(model: str) -> TokenizerLike:
    """Load the tokenizer from the already-cached model snapshot without Hub access."""
    snapshot = _model_cache_snapshot(model)
    transformers = importlib.import_module("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )
    return cast("TokenizerLike", tokenizer)


def load_lens(path: str | Path) -> JacobianLensLike:
    """Load a local fitted lens through the pinned external `jlens` package."""
    jlens = importlib.import_module("jlens")
    lens = jlens.JacobianLens.load(str(path))
    return cast("JacobianLensLike", lens)


def _fit_provenance(path: str | Path) -> dict[str, object] | None:
    provenance_path = Path(path).with_suffix(".provenance.json")
    if not provenance_path.is_file():
        return None
    provenance = json.loads(provenance_path.read_text())
    approximate_key = "final_max_d_mean_approx"
    exact_key = "final_max_d_mean"
    convergence = provenance.get(exact_key)
    approximate = False
    if convergence is None and approximate_key in provenance:
        convergence = provenance[approximate_key]
        approximate = True
    return {
        "prompts_fitted": provenance.get("prompts_fitted"),
        "final_max_d_mean": convergence,
        "final_max_d_mean_approximate": approximate,
        "run_notes": provenance.get("run_notes"),
        "source": provenance_path.name,
    }


def _infer_noise_lens(after_path: Path) -> Path | None:
    if "-n30-" not in after_path.name:
        return None
    noise_path = after_path.with_name(after_path.name.replace("-n30-", "-n2-", 1))
    if not noise_path.is_file():
        raise FileNotFoundError(
            f"the n=2 smoke lens is required as a fit-noise reference: {noise_path}"
        )
    return noise_path


def _mean_max_angles(metrics: dict[str, Any], k: int) -> tuple[float, float]:
    angles = metrics["principal_angles_degrees"][str(k)]
    if not angles:
        return 0.0, 0.0
    return sum(angles) / len(angles), max(angles)


def _markdown_table_cell(metrics: dict[str, Any] | None, k: int) -> str:
    if metrics is None:
        return "—"
    mean_angle, max_angle = _mean_max_angles(metrics, k)
    return f"{mean_angle:.2f} / {max_angle:.2f}°"


def _render_matrix_table(summary: dict[str, Any]) -> list[str]:
    comparisons = summary["comparisons"]
    before_after = comparisons["before_after"]["layers"]
    after_noise = comparisons["after_noise"]
    noise_layers = None if after_noise is None else after_noise["layers"]
    table_header = (
        "| Layer | Base vs after rel. Frobenius | Cosine | k=16 mean / max angle | "
        "k=64 mean / max angle | Fit-noise rel. Frobenius | Noise cosine | "
        "Noise k=16 mean / max | Noise k=64 mean / max |"
    )
    lines = [
        (
            "The matrix comparison uses full source-layer Jacobians. Principal angles use a seeded "
            "randomized SVD with 16 oversampling dimensions and 4 power iterations on large matrices."
        ),
        "The n=2 comparison is a fit-noise reference when an independent n=30 after lens is available.",
        "",
        table_header,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer in sorted(before_after, key=int):
        main_metrics = before_after[layer]
        noise_metrics = None if noise_layers is None else noise_layers[layer]
        lines.append(
            "| {layer} | {relative:.5f} | {cosine:.6f} | {angle16} | {angle64} | "
            "{noise_relative} | {noise_cosine} | {noise_angle16} | {noise_angle64} |".format(
                layer=layer,
                relative=main_metrics["relative_frobenius"],
                cosine=main_metrics["cosine"],
                angle16=_markdown_table_cell(main_metrics, 16),
                angle64=_markdown_table_cell(main_metrics, 64),
                noise_relative=(
                    "—" if noise_metrics is None else f"{noise_metrics['relative_frobenius']:.5f}"
                ),
                noise_cosine=("—" if noise_metrics is None else f"{noise_metrics['cosine']:.6f}"),
                noise_angle16=_markdown_table_cell(noise_metrics, 16),
                noise_angle64=_markdown_table_cell(noise_metrics, 64),
            )
        )

    if after_noise is None:
        reason = summary["noise_floor"]["reason_unavailable"]
        lines.extend(["", f"Fit-noise reference: unavailable. {reason}"])
    else:
        lines.extend(["", f"Fit-noise reference lens: `{summary['noise_floor']['lens']}`"])
    return lines


def _render_fit_provenance(summary: dict[str, Any]) -> list[str]:
    lines = ["", "## Fit convergence"]
    for label in ("before", "after", "noise"):
        provenance = summary["fit_provenance"].get(label)
        if provenance is None:
            lines.append(f"- {label}: no provenance sidecar found.")
        else:
            convergence = provenance["final_max_d_mean"]
            if convergence is None:
                convergence_text = "no `max_d_mean` statistic recorded"
            else:
                approximate_mark = "≈" if provenance["final_max_d_mean_approximate"] else ""
                convergence_text = f"final `max_d_mean`={approximate_mark}{convergence}"
            lines.append(f"- {label}: prompts={provenance['prompts_fitted']}, {convergence_text}.")
            if provenance["run_notes"]:
                lines.append(f"  Fit note: {provenance['run_notes']}")
    return lines


def _render_decoded_contents(contents: dict[str, Any]) -> list[str]:
    lines = ["", "## Decoded Jacobian-change directions"]
    for layer in sorted(contents, key=int):
        layer_contents = contents[layer]
        lines.extend(["", f"### Layer {layer}"])
        for direction in layer_contents["directions"]:
            lines.append(
                f"- Direction {direction['rank']} (singular value {direction['singular_value']:.6g})"
            )
            for polarity in ("positive_tokens", "negative_tokens"):
                tokens = ", ".join(
                    f"`{entry['token']}` ({entry['score']:.4g})" for entry in direction[polarity]
                )
                lines.append(f"  - {polarity.removesuffix('_tokens')}: {tokens}")
        control = layer_contents["matched_norm_random_control"]
        lines.append("- Matched-norm random control")
        for polarity in ("positive_tokens", "negative_tokens"):
            tokens = ", ".join(
                f"`{entry['token']}` ({entry['score']:.4g})" for entry in control[polarity]
            )
            lines.append(f"  - {polarity.removesuffix('_tokens')}: {tokens}")
    return lines


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the JSON comparison report as a human-readable Markdown summary."""
    lines = [
        "# Gemma Jacobian lens comparison",
        "",
        f"Model: `{summary['model']}`",
        f"Before lens: `{summary['before_lens']}`",
        f"After lens: `{summary['after_lens']}`",
        "",
    ]
    lines.extend(_render_matrix_table(summary))
    lines.extend(_render_fit_provenance(summary))
    lines.extend(_render_decoded_contents(summary["contents"]))
    return "\n".join(lines) + "\n"


def run_comparison(
    before: JacobianLensLike,
    after: JacobianLensLike,
    request: ComparisonRequest,
    *,
    noise_lens: JacobianLensLike | None,
    readout: ReadoutAssets,
) -> dict[str, Any]:
    """Build metrics, decoded contents, and fit metadata for one lens pair."""
    comparisons: dict[str, Any] = {"before_after": compare_lens_pair(before, after)}
    if noise_lens is None:
        after_is_smoke = "-n2-" in request.after_path.name
        noise_reason = (
            "the supplied after lens is itself the n=2 smoke lens; an independent after-versus-smoke "
            "fit-noise comparison requires the n=30 lens."
            if after_is_smoke
            else "no independent n=2 smoke lens was supplied or inferred."
        )
        comparisons["after_noise"] = None
    else:
        comparisons["after_noise"] = compare_lens_pair(after, noise_lens)
        noise_reason = None

    return {
        "schema_version": 1,
        "model": request.model,
        "before_lens": request.before_path.name,
        "after_lens": request.after_path.name,
        "lens_metadata": {
            "before": {"n_prompts": before.n_prompts, "d_model": before.d_model},
            "after": {"n_prompts": after.n_prompts, "d_model": after.d_model},
            "noise": (
                None
                if noise_lens is None
                else {"n_prompts": noise_lens.n_prompts, "d_model": noise_lens.d_model}
            ),
        },
        "fit_provenance": {
            "before": _fit_provenance(request.before_path),
            "after": _fit_provenance(request.after_path),
            "noise": None if request.noise_path is None else _fit_provenance(request.noise_path),
        },
        "noise_floor": {
            "lens": None if request.noise_path is None else request.noise_path.name,
            "label": "n=2 fit-noise reference",
            "reason_unavailable": noise_reason,
        },
        "comparisons": comparisons,
        "contents": decode_change_contents(before, after, readout=readout),
        "readout": {
            "norm": "Gemma3RMSNorm: float32 RMS normalization followed by (1 + weight)",
            "epsilon": readout.epsilon,
            "unembedding": "tied language-model embedding matrix; no input-embedding sqrt(hidden_size) scale",
            "direction_source": "right singular vectors of J_after - J_before",
            "direction_decode": "embedding @ GemmaRMSNorm((J_after - J_before) @ direction)",
            "matched_norm_control": (
                "unit-norm Gaussian residual direction decoded through the same difference operator"
            ),
        },
        "subspace_estimator": {
            "method": "torch.svd_lowrank",
            "oversampling": OVERSAMPLING,
            "power_iterations": POWER_ITERATIONS,
            "seed": RANDOM_SEED,
            "note": "small matrices use exact torch.linalg.svd",
        },
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two fitted Jacobian lenses of one base model."
    )
    parser.add_argument("--before", required=True, help="path to the before-RL Jacobian lens")
    parser.add_argument("--after", required=True, help="path to the after-RL Jacobian lens")
    parser.add_argument(
        "--noise-lens", help="optional n=2 smoke lens used as a fit-noise reference"
    )
    parser.add_argument(
        "--model", default=MODEL_ID, help="base model id; only cached Gemma-3-12B-it is supported"
    )
    parser.add_argument(
        "--out-dir", required=True, help="directory for comparison.json and summary.md"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the offline CPU comparison CLI and write JSON plus Markdown artifacts."""
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    before_path = Path(args.before)
    after_path = Path(args.after)
    noise_path = Path(args.noise_lens) if args.noise_lens else _infer_noise_lens(after_path)

    before = load_lens(before_path)
    after = load_lens(after_path)
    noise_lens = None if noise_path is None else load_lens(noise_path)
    embedding, norm_weight, epsilon = load_gemma_readout_weights(args.model)
    tokenizer = load_tokenizer(args.model)
    request = ComparisonRequest(
        model=args.model,
        before_path=before_path,
        after_path=after_path,
        noise_path=noise_path,
    )
    readout = ReadoutAssets(
        embedding=embedding,
        norm_weight=norm_weight,
        epsilon=epsilon,
        tokenizer=tokenizer,
    )
    summary = run_comparison(
        before,
        after,
        request,
        noise_lens=noise_lens,
        readout=readout,
    )

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "summary.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    markdown_path.write_text(render_markdown(summary))
    logger.info("wrote lens comparison JSON to %s and markdown to %s", json_path, markdown_path)


if __name__ == "__main__":
    main()
