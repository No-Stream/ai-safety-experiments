"""Roll-ups and the markdown rendering of a :mod:`reward_hacking.tmax.weight_geometry` run.

Everything here is Polars over the JSONL and parquet a run appends, recomputed from the artifacts
each time so a readout never drifts from what is on disk. The roll-ups are Frobenius-consistent: a
group's relative delta is ``sqrt(sum ||D||^2) / sqrt(sum ||W||^2)`` and a group's cosine is
``sum <a, b> / sqrt(sum ||a||^2 sum ||b||^2)``, both the statistic of the concatenated group rather
than an average of per-tensor ratios, and the double-rounding floors aggregate the same way per
pair index before the maximum is taken.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl

from reward_hacking.tmax.token_frequency import (
    residualise_on_log_frequency,
    top_rows,
    with_token_strings,
)
from reward_hacking.tmax.weight_layout import (
    BASE_SPECTRA_FILENAME,
    GATE_HEAD_MODULES,
    GATE_HEADS_FILENAME,
    PAIRS_FILENAME,
    RUN_MANIFEST_FILENAME,
    TENSORS_FILENAME,
    ModuleClass,
    token_rows_filename,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

SHAPE_COLUMNS: tuple[str, ...] = (
    "delta_stable_rank",
    "delta_effective_rank",
    "delta_sv_entropy_normalized",
    "delta_top1_energy",
    "delta_top8_energy",
    "delta_top64_energy",
)


LARGE_VALUE = 1000.0
SMALL_VALUE = 1e-3


def load_tensors(out_dir: Path) -> pl.DataFrame:
    """One row per (checkpoint, tensor): movement plus the delta's spectral shape."""
    return pl.read_ndjson(out_dir / TENSORS_FILENAME, infer_schema_length=None)


def load_base_spectra(out_dir: Path) -> pl.DataFrame:
    """One row per tensor: the base weight's own spectral shape."""
    return pl.read_ndjson(out_dir / BASE_SPECTRA_FILENAME, infer_schema_length=None)


def load_pairs(out_dir: Path) -> pl.DataFrame:
    """One row per (earlier checkpoint, later checkpoint, tensor): the delta relationship."""
    return pl.read_ndjson(out_dir / PAIRS_FILENAME, infer_schema_length=None)


def load_gate_heads(out_dir: Path) -> pl.DataFrame:
    """One row per (checkpoint, DeltaNet gate tensor, recurrent head)."""
    return pl.read_ndjson(out_dir / GATE_HEADS_FILENAME, infer_schema_length=None)


def movement_by(tensors: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    """Frobenius roll-up of movement per group (see the module docstring for why not a mean)."""
    keys = ["checkpoint", *by]
    return (
        tensors.group_by(keys)
        .agg(
            pl.len().alias("n_tensors"),
            pl.col("n_elements").sum(),
            (pl.col("base_norm") ** 2).sum().sqrt().alias("base_norm"),
            (pl.col("delta_norm") ** 2).sum().sqrt().alias("delta_norm"),
            pl.col("n_changed").sum(),
            pl.col("n_one_ulp").sum(),
        )
        .with_columns(
            (pl.col("delta_norm") / pl.col("base_norm")).alias("relative_delta"),
            (pl.col("n_changed") / pl.col("n_elements")).alias("changed_fraction"),
            (pl.col("n_one_ulp") / pl.col("n_changed")).alias("one_ulp_share_of_changed"),
        )
        .sort(keys)
    )


def spectral_summary_by(tensors: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    """Medians of the delta's shape statistics per group, over tensors that have a spectrum."""
    keys = ["checkpoint", *by]
    return (
        tensors.filter(pl.col("delta_stable_rank").is_not_null())
        .group_by(keys)
        .agg(
            pl.len().alias("n_tensors"),
            pl.col("delta_n_singular_values").median().alias("n_singular_values_median"),
            *[pl.col(column).median().alias(f"{column}_median") for column in SHAPE_COLUMNS],
        )
        .sort(keys)
    )


def base_versus_delta_shape(tensors: pl.DataFrame, base_spectra: pl.DataFrame) -> pl.DataFrame:
    """Per tensor, the delta's shape beside the base tensor's, for the same statistics."""
    base_columns = base_spectra.select(
        "name",
        pl.col("stable_rank").alias("base_stable_rank"),
        pl.col("effective_rank").alias("base_effective_rank"),
        pl.col("sv_entropy_normalized").alias("base_sv_entropy_normalized"),
        pl.col("top64_energy").alias("base_top64_energy"),
    )
    return tensors.join(base_columns, on="name", how="left")


def _floor_cosines_by(pairs: pl.DataFrame, by: Sequence[str], kind: str) -> pl.DataFrame:
    """One floor kind per group: cosine of the concatenated draws per pair index, then the max."""
    keys = ["checkpoint_a", "checkpoint_b", *by]
    inner, norm_a, norm_b = (
        f"floor_{kind}_inner",
        f"floor_{kind}_norm_a",
        f"floor_{kind}_norm_b",
    )
    exploded = (
        pairs.select(*keys, inner, norm_a, norm_b)
        .with_columns(pl.int_ranges(0, pl.col(inner).list.len()).alias("pair_index"))
        .explode([inner, norm_a, norm_b, "pair_index"], empty_as_null=True)
    )
    per_index = (
        exploded.group_by([*keys, "pair_index"])
        .agg(
            pl.col(inner).sum().alias("inner"),
            (pl.col(norm_a) ** 2).sum().sqrt().alias("norm_a"),
            (pl.col(norm_b) ** 2).sum().sqrt().alias("norm_b"),
        )
        .with_columns((pl.col("inner") / (pl.col("norm_a") * pl.col("norm_b"))).alias("cosine"))
    )
    return per_index.group_by(keys).agg(pl.col("cosine").max().alias(f"floor_{kind}_max"))


def relationship_by(pairs: pl.DataFrame, by: Sequence[str]) -> pl.DataFrame:
    """Group-level cosine, scale and residual between two deltas, each read against its floor.

    The group ``alpha`` is the least-squares scale over the concatenation,
    ``sum <a, b> / sum ||a||^2``; ``n_clearing_floor`` counts the group's tensors whose own cosine
    cleared their own floor, which is the finer-grained reading.
    """
    keys = ["checkpoint_a", "checkpoint_b", *by]
    real = (
        pairs.group_by(keys)
        .agg(
            pl.len().alias("n_tensors"),
            pl.col("inner").sum(),
            (pl.col("norm_a") ** 2).sum().alias("norm_a_squared"),
            (pl.col("norm_b") ** 2).sum().alias("norm_b_squared"),
            pl.col("clears_floor").cast(pl.Int64).sum().alias("n_clearing_floor"),
        )
        .with_columns(
            (
                pl.col("inner")
                / (pl.col("norm_a_squared").sqrt() * pl.col("norm_b_squared").sqrt())
            ).alias("cosine"),
            (pl.col("inner") / pl.col("norm_a_squared")).alias("alpha_b_on_a"),
            (pl.col("norm_b_squared") / pl.col("norm_a_squared"))
            .sqrt()
            .alias("norm_ratio_b_over_a"),
        )
        .with_columns((1.0 - pl.col("cosine") ** 2).alias("residual_share"))
    )
    joined = real
    for kind in ("rms", "sparsity", "offset"):
        joined = joined.join(_floor_cosines_by(pairs, by, kind), on=keys, how="left")
    return (
        joined.with_columns(
            pl.max_horizontal("floor_rms_max", "floor_sparsity_max").alias("floor_max")
        )
        .with_columns((pl.col("cosine") > pl.col("floor_max")).alias("clears_floor"))
        .sort(keys)
    )


def _format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    if value == 0.0:
        return "0"
    if abs(value) >= LARGE_VALUE or abs(value) < SMALL_VALUE:
        return f"{value:.3e}"
    return f"{value:.4g}"


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_format_cell(v) for v in cast("Sequence[object]", value))
    return str(value).replace("|", "\\|").replace("\n", "\\n")


def markdown_table(frame: pl.DataFrame, columns: Sequence[str] | None = None) -> str:
    """Render a (small, already reduced) frame as a GitHub-flavoured markdown table."""
    names = list(columns) if columns is not None else frame.columns
    selected = frame.select(names)
    column_values: list[list[object]] = [column.to_list() for column in selected.get_columns()]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join("---" for _ in names) + "|"]
    lines.extend(
        "| " + " | ".join(_format_cell(v) for v in row) + " |"
        for row in zip(*column_values, strict=True)
    )
    return "\n".join(lines)


def _tokenizer_json_for(manifest: Mapping[str, object], checkpoint: str) -> Path:
    for entry in cast("list[dict[str, object]]", manifest["rl"]):
        if entry["label"] == checkpoint:
            return Path(cast("str", entry["snapshot_dir"])) / "tokenizer.json"
    raise KeyError(f"{checkpoint} is not a checkpoint of this run")


def _section(parts: list[str], heading: str, *blocks: str) -> None:
    parts.append(heading)
    parts.append("")
    for block in blocks:
        parts.append(block)
        parts.append("")


def _movement_sections(parts: list[str], tensors: pl.DataFrame, checkpoints: Sequence[str]) -> None:
    _section(
        parts,
        "## 1. Movement by module family",
        markdown_table(
            movement_by(tensors, ["family"]),
            [
                "checkpoint",
                "family",
                "n_tensors",
                "relative_delta",
                "changed_fraction",
                "one_ulp_share_of_changed",
            ],
        ),
    )
    _section(
        parts,
        "### Movement by module",
        markdown_table(
            movement_by(tensors, ["module"]),
            ["checkpoint", "module", "n_tensors", "relative_delta", "changed_fraction"],
        ),
    )
    in_stack = tensors.filter(pl.col("layer").is_not_null())
    by_layer = movement_by(in_stack, ["layer"])
    _section(
        parts,
        "### Movement by layer (all families), relative delta per checkpoint",
        markdown_table(
            by_layer.pivot(on="checkpoint", index="layer", values="relative_delta").sort("layer")
        ),
    )
    cells = movement_by(in_stack, ["layer", "family"])
    for checkpoint in checkpoints:
        _section(
            parts,
            f"### Largest 24 layer-by-family cells, `{checkpoint}`",
            markdown_table(
                top_rows(
                    cells.filter(pl.col("checkpoint") == checkpoint), by="relative_delta", k=24
                ),
                ["layer", "family", "relative_delta", "changed_fraction"],
            ),
        )
    for checkpoint in checkpoints:
        _section(
            parts,
            f"### Twenty most-moved tensors, `{checkpoint}`",
            markdown_table(
                top_rows(
                    tensors.filter(pl.col("checkpoint") == checkpoint), by="relative_delta", k=20
                ),
                ["name", "relative_delta", "changed_fraction", "cosine_delta_base"],
            ),
        )


def _spectral_sections(parts: list[str], tensors: pl.DataFrame, base_spectra: pl.DataFrame) -> None:
    _section(
        parts,
        "## 2. Spectral shape of the delta, medians by family",
        markdown_table(
            spectral_summary_by(tensors, ["family"]),
            [
                "checkpoint",
                "family",
                "n_tensors",
                "n_singular_values_median",
                "delta_stable_rank_median",
                "delta_effective_rank_median",
                "delta_sv_entropy_normalized_median",
                "delta_top1_energy_median",
                "delta_top8_energy_median",
                "delta_top64_energy_median",
            ],
        ),
    )
    joined = base_versus_delta_shape(tensors, base_spectra)
    _section(
        parts,
        "### Delta shape against the base tensor's own shape, medians by family",
        markdown_table(
            joined.filter(pl.col("delta_stable_rank").is_not_null())
            .group_by(["checkpoint", "family"])
            .agg(
                pl.col("delta_effective_rank").median(),
                pl.col("base_effective_rank").median(),
                pl.col("delta_top64_energy").median(),
                pl.col("base_top64_energy").median(),
                pl.col("delta_sv_entropy_normalized").median(),
                pl.col("base_sv_entropy_normalized").median(),
            )
            .sort(["checkpoint", "family"])
        ),
    )
    _section(
        parts,
        "### Spectral shape by layer, medians over the layer's matrices",
        markdown_table(
            spectral_summary_by(tensors.filter(pl.col("layer").is_not_null()), ["layer"]),
            [
                "checkpoint",
                "layer",
                "delta_stable_rank_median",
                "delta_effective_rank_median",
                "delta_top64_energy_median",
            ],
        ),
    )


def _relationship_sections(parts: list[str], pairs: pl.DataFrame) -> None:
    _section(
        parts,
        "## 3. Relationship between deltas, by family",
        markdown_table(
            relationship_by(pairs, ["family"]),
            [
                "checkpoint_a",
                "checkpoint_b",
                "family",
                "n_tensors",
                "cosine",
                "floor_rms_max",
                "floor_sparsity_max",
                "floor_offset_max",
                "clears_floor",
                "n_clearing_floor",
                "alpha_b_on_a",
                "norm_ratio_b_over_a",
                "residual_share",
            ],
        ),
    )
    _section(
        parts,
        "### Whole language model",
        markdown_table(
            relationship_by(pairs, []),
            [
                "checkpoint_a",
                "checkpoint_b",
                "n_tensors",
                "cosine",
                "floor_max",
                "floor_offset_max",
                "n_clearing_floor",
                "alpha_b_on_a",
                "norm_ratio_b_over_a",
                "residual_share",
            ],
        ),
    )
    _section(
        parts,
        "### By layer",
        markdown_table(
            relationship_by(pairs.filter(pl.col("layer").is_not_null()), ["layer"]),
            [
                "layer",
                "cosine",
                "floor_max",
                "n_clearing_floor",
                "alpha_b_on_a",
                "residual_share",
            ],
        ),
    )
    columns = [
        "name",
        "cosine",
        "floor_rms_max",
        "floor_sparsity_max",
        "floor_offset_max",
        "margin_over_floor",
        "alpha_b_on_a",
    ]
    _section(
        parts,
        "### Per-tensor extremes: ten lowest, then ten highest margins over the floor",
        markdown_table(top_rows(pairs, by="margin_over_floor", k=10, descending=False), columns),
        markdown_table(top_rows(pairs, by="margin_over_floor", k=10), columns),
    )


def _token_row_sections(
    parts: list[str],
    out_dir: Path,
    manifest: Mapping[str, object],
    checkpoints: Sequence[str],
    top_k_tokens: int,
) -> None:
    parts.append("## 4. Token rows")
    parts.append("")
    token_columns = ["token_id", "piece", "text", "row_norm", "frequency", "frequency_residual"]
    for checkpoint in checkpoints:
        for module in (ModuleClass.EMBED_TOKENS, ModuleClass.LM_HEAD):
            path = out_dir / token_rows_filename(checkpoint, module)
            if not path.is_file():
                continue
            rows = pl.read_parquet(path)
            tokenizer_json = _tokenizer_json_for(manifest, checkpoint)
            has_frequency = "frequency" in rows.columns
            _section(
                parts,
                f"### `{checkpoint}` / `{module}`: ten largest raw row movements (the frequency "
                f"artefact, for contrast)",
                markdown_table(
                    with_token_strings(top_rows(rows, by="row_norm", k=10), tokenizer_json),
                    [
                        "token_id",
                        "piece",
                        "text",
                        "row_norm",
                        *(["frequency"] if has_frequency else []),
                    ],
                ),
            )
            if not has_frequency:
                _section(parts, "No token counts were supplied, so no frequency residual.")
                continue
            residualised, fit = residualise_on_log_frequency(rows)
            _section(
                parts,
                f"Log-log fit of row norm on frequency, degree {fit.degree}: R^2 = "
                f"{fit.r_squared:.3f} over {fit.n_fit} rows with both frequency and movement; "
                f"{fit.n_zero_frequency} rows never appear in the corpus, {fit.n_zero_movement} "
                f"rows did not move, and {fit.n_zero_frequency_moved} rows moved without ever "
                f"appearing.",
            )
            _section(
                parts,
                f"Top {top_k_tokens} rows by frequency residual (moved more than frequency "
                f"predicts):",
                markdown_table(
                    with_token_strings(
                        top_rows(residualised, by="frequency_residual", k=top_k_tokens),
                        tokenizer_json,
                    ),
                    token_columns,
                ),
            )
            _section(
                parts,
                "Ten rows by most negative residual (moved less than frequency predicts):",
                markdown_table(
                    with_token_strings(
                        top_rows(residualised, by="frequency_residual", k=10, descending=False),
                        tokenizer_json,
                    ),
                    token_columns,
                ),
            )
            moved_unseen = rows.filter((pl.col("frequency") == 0) & (pl.col("row_norm") > 0))
            if moved_unseen.height:
                _section(
                    parts,
                    f"Ten largest movers among rows with zero corpus frequency "
                    f"({moved_unseen.height} such rows):",
                    markdown_table(
                        with_token_strings(
                            top_rows(moved_unseen, by="row_norm", k=10), tokenizer_json
                        ),
                        ["token_id", "piece", "text", "row_norm", "input_count", "target_count"],
                    ),
                )


def _gate_sections(
    parts: list[str], tensors: pl.DataFrame, gates: pl.DataFrame, checkpoints: Sequence[str]
) -> None:
    gate_tensors = tensors.filter(
        pl.col("module").is_in([str(module) for module in GATE_HEAD_MODULES])
    )
    _section(
        parts,
        "## 5. DeltaNet gates: gate projections and state parameters per layer",
        markdown_table(
            gate_tensors.sort(["checkpoint", "module", "layer"]),
            [
                "checkpoint",
                "module",
                "layer",
                "relative_delta",
                "changed_fraction",
                "cosine_delta_base",
                "delta_stable_rank",
                "delta_top8_energy",
            ],
        ),
    )
    head_columns = [
        "layer",
        "module",
        "head",
        "relative_delta",
        "cosine_delta_base",
        "changed_fraction",
    ]
    for checkpoint in checkpoints:
        _section(
            parts,
            f"### Twenty most-moved gate heads, `{checkpoint}`",
            markdown_table(
                top_rows(
                    gates.filter(
                        (pl.col("checkpoint") == checkpoint)
                        & pl.col("cosine_delta_base").is_not_null()
                    ),
                    by="relative_delta",
                    k=20,
                ),
                head_columns,
            ),
        )
    _section(
        parts,
        "### A_log and dt_bias per head, twenty largest absolute changes",
        markdown_table(
            top_rows(
                gates.filter(pl.col("delta_value").is_not_null()).with_columns(
                    pl.col("delta_value").abs().alias("abs_delta_value")
                ),
                by="abs_delta_value",
                k=20,
            ),
            ["checkpoint", "layer", "module", "head", "base_value", "delta_value"],
        ),
    )


def render_readout(out_dir: Path, *, top_k_tokens: int = 40) -> str:
    """Every table of the readout as one markdown document, computed from the artifacts on disk."""
    manifest = cast(
        "dict[str, object]",
        json.loads((out_dir / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8")),
    )
    tensors = load_tensors(out_dir)
    base_spectra = load_base_spectra(out_dir)
    checkpoint_values: list[object] = (
        tensors.get_column("checkpoint").unique(maintain_order=True).to_list()
    )
    checkpoints = [str(value) for value in checkpoint_values]
    parts: list[str] = []
    _section(
        parts,
        f"# Weight geometry tables: `{out_dir.name}`",
        f"Base `{json.dumps(manifest['base'])}` against {len(checkpoints)} checkpoint(s): "
        + ", ".join(f"`{c}`" for c in checkpoints)
        + ".",
    )
    _movement_sections(parts, tensors, checkpoints)
    _spectral_sections(parts, tensors, base_spectra)
    if (out_dir / PAIRS_FILENAME).is_file():
        _relationship_sections(parts, load_pairs(out_dir))
    _token_row_sections(parts, out_dir, manifest, checkpoints, top_k_tokens)
    if (out_dir / GATE_HEADS_FILENAME).is_file():
        _gate_sections(parts, tensors, load_gate_heads(out_dir), checkpoints)
    return "\n".join(parts)
