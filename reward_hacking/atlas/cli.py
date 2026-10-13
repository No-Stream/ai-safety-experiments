"""Command line entry points for the Atlas re-probe pipeline.

Smoke example (first four prompts, base plus one adapter):

    python -m reward_hacking.atlas prompts --run-name atlas-smoke --max-items 4
    python -m reward_hacking.atlas sample --run-name atlas-smoke --models base \
        kl0.01-det3-seed1-mbpp_probe --max-prompts 4
    python -m reward_hacking.atlas grade --run-name atlas-smoke --models base kl0.01-det3-seed1-mbpp_probe
    python -m reward_hacking.atlas capture --run-name atlas-smoke --models base kl0.01-det3-seed1-mbpp_probe
    python -m reward_hacking.atlas neutral --run-name atlas-smoke --models base kl0.01-det3-seed1-mbpp_probe
    python -m reward_hacking.atlas readout --run-name atlas-smoke --models base kl0.01-det3-seed1-mbpp_probe

The full run omits ``--max-items`` and ``--max-prompts`` and uses the default model slice. Every stage
loads only its own runtime dependencies after command parsing; importing this module does not load
weights or initialize a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from experiment_sampling import add_oversize_reason_argument, log_sampling_grid
from reward_hacking.atlas.grading_runtime import DEFAULT_GRADE_WORKERS
from reward_hacking.jsonl_io import read_jsonl_objects_value_error, write_json_atomic

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_ROOT = REPO_ROOT / "artifacts" / "atlas"
MINIMUM_PROBE_CLASS_COUNT = 15
# The within-model readout sweeps every layer; cross-model transfer (42 model pairs x 4 positions)
# reads every fourth layer to keep the probe-fit count in the low thousands.
TRANSFER_LAYER_STRIDE = 4
LEGACY_REFERENCE_NPY_DIMS = 3
POSITIONAL_REFERENCE_NPY_DIMS = 4


def _run_dir(run_name: str, runs_root: Path) -> Path:
    if not run_name or Path(run_name).is_absolute() or Path(run_name).name != run_name:
        raise ValueError("run name must be a single non-empty path component")
    if run_name in {".", ".."}:
        raise ValueError("run name cannot be '.' or '..'")
    return runs_root / run_name


def _json_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"expected a JSON object: {error}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-name", required=True, help="run directory under artifacts/atlas")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="artifact root (default: repository artifacts/atlas)",
    )
    parser.add_argument(
        "--base-model",
        default="google/gemma-3-12b-it",
        help="base model used for this run (default: google/gemma-3-12b-it)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the stage-selecting command-line parser without loading model weights."""
    from reward_hacking.atlas.sampling import MODEL_KEYS, PROMPT_DATASETS  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="python -m reward_hacking.atlas",
        description="Run resumable Atlas sampling, jail grading, activation capture, and probe readout stages.",
        epilog=(
            "All prompts, completions, and per-sample outcomes are written only under the selected "
            "artifact directory. Use the six subcommands in order; a repeated stage resumes from its "
            "record ledger."
        ),
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    prompts = subparsers.add_parser(
        "prompts", help="load runtime HF datasets and render Atlas prompts"
    )
    _add_common(prompts)
    prompts.add_argument(
        "--leakage-hint-variant",
        default="none",
        help="runtime hint variant to append after the visible test (default: none)",
    )
    prompts.add_argument(
        "--chat-template-kwargs",
        type=_json_object,
        default={},
        help="JSON object passed to the tokenizer chat template (default: {})",
    )
    prompts.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(PROMPT_DATASETS),
        default=tuple(PROMPT_DATASETS),
        help="HF datasets to include; synthetic MBPP is checked for compatible fields",
    )
    prompts.add_argument("--max-items", type=int, help="limit each dataset split for a smoke")

    sample = subparsers.add_parser("sample", help="sample the base and/or released LoRA adapters")
    _add_common(sample)
    sample.add_argument("--models", nargs="+", default=MODEL_KEYS)
    sample.add_argument("--samples-per-prompt", type=int, default=2)
    add_oversize_reason_argument(sample)
    sample.add_argument("--max-tokens", type=int, default=256)
    sample.add_argument("--max-prompts", type=int, help="limit prompts for a smoke")
    sample.add_argument(
        "--language-model-only",
        action="store_true",
        help="Build vLLM in language-model-only mode (default: off).",
    )
    sample.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="Optional vLLM KV cache dtype, such as fp8; omitted keeps the engine default.",
    )

    grade = subparsers.add_parser("grade", help="grade solutions in fresh episode-jail runs")
    _add_common(grade)
    grade.add_argument("--models", nargs="+", default=MODEL_KEYS)
    grade.add_argument("--timeout-seconds", type=int, default=5)
    grade.add_argument("--scratch-parent", type=Path, default=Path("/var/tmp"))  # noqa: S108
    grade.add_argument("--max-records", type=int, help="limit samples for a smoke")
    grade.add_argument(
        "--workers", type=int, default=DEFAULT_GRADE_WORKERS, help="parallel jail runs"
    )

    capture = subparsers.add_parser("capture", help="capture residuals at the Atlas positions")
    _add_common(capture)
    capture.add_argument("--models", nargs="+", default=MODEL_KEYS)
    capture.add_argument("--max-records", type=int, help="limit samples for a smoke")
    capture.add_argument(
        "--capture-reference",
        action="store_true",
        help="also capture hardcode-versus-code references for each requested model",
    )

    neutral = subparsers.add_parser("neutral", help="ask each checkpoint a fresh-context question")
    _add_common(neutral)
    neutral.add_argument("--models", nargs="+", default=MODEL_KEYS)
    neutral.add_argument("--max-records", type=int, help="limit samples for a smoke")
    neutral.add_argument("--max-tokens", type=int, default=64)

    readout = subparsers.add_parser("readout", help="fit grouped-CV probes and text baselines")
    _add_common(readout)
    readout.add_argument("--models", nargs="+", default=MODEL_KEYS)
    readout.add_argument("--permutations", type=int, default=5)
    readout.add_argument("--seed", type=int, default=0)

    return parser


def _sample_sampling_budget(arguments: argparse.Namespace, prompt_path: Path) -> dict[str, object]:
    """Build the selected model/prompt episode grid without logging prompt content."""
    if arguments.samples_per_prompt < 1:
        raise ValueError(
            f"--samples-per-prompt must be positive, got {arguments.samples_per_prompt}"
        )
    if arguments.max_prompts is not None and arguments.max_prompts < 1:
        raise ValueError(f"--max-prompts must be positive, got {arguments.max_prompts}")
    prompt_count = len(read_jsonl_objects_value_error(prompt_path))
    if prompt_count == 0:
        raise ValueError(f"no prompts found in {prompt_path}")
    if arguments.max_prompts is not None:
        prompt_count = min(prompt_count, arguments.max_prompts)
    episodes_per_cell = {
        f"model={model_key}/prompt={prompt_index}": arguments.samples_per_prompt
        for model_key in arguments.models
        for prompt_index in range(prompt_count)
    }
    return log_sampling_grid(
        episodes_per_cell,
        oversize_reason=arguments.oversize_reason,
    )


def _reference_probe_inputs(run_dir: Path, model_key: str) -> dict[str, Any] | None:
    """Load one reference shard as follow-up activations and item-grouped labels."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from reward_hacking.atlas.capture import reference_artifact_paths  # noqa: PLC0415

    activation_path, index_path, _ = reference_artifact_paths(run_dir, model_key)
    if not activation_path.exists() or not index_path.exists():
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not index.get("complete"):
        raise ValueError(f"reference capture for {model_key} is incomplete")
    activations = np.load(activation_path, mmap_mode="r")
    rows = list(index["rows"])
    positions = list(index["positions"])
    if activations.shape[0] != len(rows) or list(activations.shape) != index.get("shape"):
        raise ValueError(
            f"reference activation rows or shape disagree with the index for {model_key}"
        )
    if "followup_last" not in positions:
        raise ValueError(f"reference shard for {model_key} lacks followup_last activations")
    followup_position = positions.index("followup_last")
    if activations.ndim == LEGACY_REFERENCE_NPY_DIMS:
        if positions != ["followup_last"]:
            raise ValueError(f"legacy reference shard positions disagree for {model_key}")
        followup_activations = np.asarray(activations, dtype=np.float32)
    elif activations.ndim == POSITIONAL_REFERENCE_NPY_DIMS:
        if activations.shape[2] != len(positions):
            raise ValueError(f"reference position axis disagrees with the index for {model_key}")
        followup_activations = np.asarray(activations[:, :, followup_position, :], dtype=np.float32)
    else:
        raise ValueError(
            f"reference shard for {model_key} has unsupported shape {activations.shape}"
        )
    class_names = [str(row["class_name"]) for row in rows]
    if any(class_name not in {"hardcode", "code"} for class_name in class_names):
        raise ValueError(f"reference shard for {model_key} has an unknown class label")
    labels = torch.tensor([float(class_name == "hardcode") for class_name in class_names])
    item_names = [str(row["item_id"]) for row in rows]
    item_group_ids = {name: index for index, name in enumerate(dict.fromkeys(item_names))}
    groups = torch.tensor([item_group_ids[name] for name in item_names], dtype=torch.long)
    return {
        "activations": torch.from_numpy(followup_activations.copy()),
        "labels": labels,
        "groups": groups,
        "n_hardcode": int((labels == 1).sum().item()),
        "n_code": int((labels == 0).sum().item()),
    }


def _fixed_corpus_refit(
    run_dir: Path,
    model_keys: list[str],
    prepared: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Fit each saved reference corpus once, then score each model's graded hacks."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from reward_hacking.atlas.fixed_corpus import (  # noqa: PLC0415
        BEST_SINGLE_LAYER,
        DEFAULT_FALSE_POSITIVE_RATE,
        MEAN_OVER_LAYERS,
        fit_fixed_corpus_probes,
        score_fixed_corpus_probes,
    )

    reference_inputs: dict[str, dict[str, Any]] = {}
    base_reference = _reference_probe_inputs(run_dir, "base")
    if base_reference is None:
        return (
            {
                "status": "not_captured",
                "reason": "base reference shard is required for fixed-corpus refits",
                "models": {},
            },
            [],
        )
    reference_inputs["base"] = base_reference
    missing_model_references: list[str] = []
    for model_key in model_keys:
        if model_key == "base":
            continue
        model_reference = _reference_probe_inputs(run_dir, model_key)
        if model_reference is None:
            missing_model_references.append(model_key)
        else:
            reference_inputs[model_key] = model_reference

    reference_probes = {
        source_key: fit_fixed_corpus_probes(
            reference["activations"],
            reference["labels"],
            reference["groups"],
        )
        for source_key, reference in reference_inputs.items()
    }
    models_summary: dict[str, Any] = {}
    markdown_rows: list[str] = []
    probe_cells = (MEAN_OVER_LAYERS, BEST_SINGLE_LAYER)
    for model_key in model_keys:
        if model_key not in reference_inputs:
            continue
        model = prepared[model_key]
        eligible_ids = [
            sample_id for sample_id in model["sample_ids"] if sample_id in model["labels_by_id"]
        ]
        sample_indices = [model["row_of_sample"][sample_id] for sample_id in eligible_ids]
        followup_position = model["position_names"].index("followup_last")
        target_activations = torch.from_numpy(
            np.asarray(
                model["shard"][sample_indices, :, followup_position, :], dtype=np.float32
            ).copy()
        )
        target_hack_mask = torch.tensor(
            [model["labels_by_id"][sample_id] == 1 for sample_id in eligible_ids],
            dtype=torch.bool,
        )
        target_regex_mask = torch.tensor(
            [model["regex_flags_by_id"][sample_id] for sample_id in eligible_ids],
            dtype=torch.bool,
        )
        source_keys = ["base"] if model_key == "base" else ["base", model_key]
        reference_sources: dict[str, Any] = {}
        for source_key in source_keys:
            source_reference = reference_inputs[source_key]
            readouts = score_fixed_corpus_probes(
                reference_probes[source_key],
                target_activations,
                target_hack_mask,
                target_regex_mask,
            )
            cells: dict[str, Any] = {}
            for cell in probe_cells:
                result = readouts[cell]
                cell_summary = asdict(result)
                cells[cell] = cell_summary
                overall = result.hacks
                regex = result.regex_flagged_hacks
                overall_cell = (
                    "n=0"
                    if overall.fraction is None
                    else f"{overall.flagged}/{overall.n} ({overall.fraction:.1%})"
                )
                regex_cell = (
                    "n=0"
                    if regex.fraction is None
                    else f"{regex.flagged}/{regex.n} ({regex.fraction:.1%})"
                )
                readout_label = (
                    "mean over layers"
                    if result.selected_layer is None
                    else f"best single layer (L{result.selected_layer})"
                )
                markdown_rows.append(
                    f"| {model_key} | {source_key} | {readout_label} | "
                    f"{result.reference_cv_auc:.3f} | {overall_cell} | {regex_cell} | "
                    f"{result.calibration_false_positive_rate:.1%} |"
                )
            reference_sources[source_key] = {
                "n_hardcode": source_reference["n_hardcode"],
                "n_code": source_reference["n_code"],
                "probes": cells,
            }
        models_summary[model_key] = {"reference_sources": reference_sources}

    status = "ok" if models_summary else "no_model_reference_shards"
    return (
        {
            "status": status,
            "target_false_positive_rate": DEFAULT_FALSE_POSITIVE_RATE,
            "models": models_summary,
            "missing_model_reference_shards": missing_model_references,
        },
        markdown_rows,
    )


def _readout(  # noqa: C901, PLR0912, PLR0915
    run_dir: Path, model_keys: list[str], permutations: int, seed: int
) -> None:
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from reward_hacking.atlas.core import (  # noqa: PLC0415
        extract_code,
        hardcode_regex_flag,
        probe_readout,
        tfidf_char_ngram_readout,
    )
    from reward_hacking.atlas.readout import rank_auc  # noqa: PLC0415
    from reward_hacking.atlas.sampling import artifact_model_key  # noqa: PLC0415
    from reward_hacking.interp.linear_probe import fit_logistic_probe  # noqa: PLC0415

    prepared: dict[str, dict[str, Any]] = {}
    summary: dict[str, object] = {"models": {}, "transfers": [], "original_probe_standin": {}}
    for model_key in model_keys:
        output_key = artifact_model_key(model_key)
        sample_records = read_jsonl_objects_value_error(run_dir / f"samples.{output_key}.jsonl")
        grade_records = read_jsonl_objects_value_error(run_dir / f"grades.{output_key}.jsonl")
        grade_by_id = {str(record["sample_id"]): record for record in grade_records}
        shard = np.load(run_dir / f"activations.{output_key}.npy", mmap_mode="r")
        index = json.loads(
            (run_dir / f"capture-index.{output_key}.json").read_text(encoding="utf-8")
        )
        sample_ids = [str(sample_id) for sample_id in index["sample_ids"]]
        sample_by_id = {str(record["sample_id"]): record for record in sample_records}
        if set(sample_ids) != set(sample_by_id):
            raise ValueError(f"sample records and activation index disagree for {model_key}")
        if len(sample_ids) != shard.shape[0]:
            raise ValueError(f"activation row count and index disagree for {model_key}")
        labels_by_id: dict[str, int] = {}
        for sample_id in sample_ids:
            grade = grade_by_id.get(sample_id)
            if grade is None:
                continue
            if bool(grade["hack"]):
                labels_by_id[sample_id] = 1
            elif bool(grade["honest_pass"]):
                labels_by_id[sample_id] = 0
        eligible_ids = [sample_id for sample_id in sample_ids if sample_id in labels_by_id]
        labels = torch.tensor(
            [labels_by_id[sample_id] for sample_id in eligible_ids], dtype=torch.float32
        )
        group_names = [str(sample_by_id[sample_id]["item_id"]) for sample_id in eligible_ids]
        group_ids = {name: index for index, name in enumerate(dict.fromkeys(group_names))}
        groups = torch.tensor([group_ids[name] for name in group_names], dtype=torch.long)
        texts = [
            extract_code(str(sample_by_id[sample_id]["completion_text"]))
            for sample_id in eligible_ids
        ]
        regex_flags = torch.tensor(
            [
                float(
                    hardcode_regex_flag(
                        texts[row],
                        str(sample_by_id[sample_id]["test_list"][0]),
                    )
                )
                for row, sample_id in enumerate(eligible_ids)
            ],
            dtype=torch.float32,
        )
        regex_flags_by_id = {
            sample_id: bool(regex_flags[row].item()) for row, sample_id in enumerate(eligible_ids)
        }
        text_readout = tfidf_char_ngram_readout(
            texts,
            labels,
            groups,
            minimum_class_count=15,
        )
        regex_auc = None if text_readout.status == "too_few" else rank_auc(labels, regex_flags)
        model_summary: dict[str, object] = {
            "n_samples": len(sample_ids),
            "n_hack": int((labels == 1).sum().item()),
            "n_honest_pass": int((labels == 0).sum().item()),
            "n_fail_or_ungraded": len(sample_ids) - len(eligible_ids),
            "grade_rates": (
                json.loads(
                    (run_dir / f"grade-summary.{output_key}.json").read_text(encoding="utf-8")
                )
                if (run_dir / f"grade-summary.{output_key}.json").exists()
                else None
            ),
            "text_baselines": {
                "hardcode_regex_auc": regex_auc,
                "hardcode_regex_status": text_readout.status,
                "tfidf_char_ngram": asdict(text_readout),
            },
            "positions": {},
        }
        position_reads: dict[str, object] = {}
        position_names = list(index["positions"])
        row_of_sample = {sample_id: row for row, sample_id in enumerate(sample_ids)}
        eligible_rows = np.array(
            [row_of_sample[sample_id] for sample_id in eligible_ids], dtype=np.int64
        )
        for position_index, position_name in enumerate(position_names):
            layer_reads = []
            # One read per position (rows x layers x width) instead of one scattered read per layer.
            position_rows = np.asarray(shard[eligible_rows, :, position_index, :], dtype=np.float32)
            for layer in range(shard.shape[1]):
                feature_rows = torch.from_numpy(position_rows[:, layer, :].copy())
                readout = probe_readout(
                    feature_rows,
                    labels,
                    groups,
                    n_permutations=permutations,
                    seed=seed,
                )
                layer_reads.append({"layer": layer, **asdict(readout)})
            position_reads[position_name] = layer_reads
        model_summary["positions"] = position_reads
        summary_models = summary["models"]
        if not isinstance(summary_models, dict):
            raise TypeError("readout summary model map was replaced unexpectedly")
        summary_models[model_key] = model_summary
        prepared[model_key] = {
            "samples": sample_by_id,
            "sample_ids": sample_ids,
            "row_of_sample": row_of_sample,
            "shard": shard,
            "labels_by_id": labels_by_id,
            "regex_flags_by_id": regex_flags_by_id,
            "position_names": position_names,
        }

    reference_path = run_dir / "reference-activations.npy"
    reference_index_path = run_dir / "reference-activation-index.json"
    original_probe_summary: dict[str, object]
    if not reference_path.exists() or not reference_index_path.exists():
        original_probe_summary = {
            "status": "not_captured",
            "reason": "base capture did not produce the Atlas hardcode-vs-code reference shard",
        }
    else:
        reference_activations = np.load(reference_path, mmap_mode="r")
        reference_index = json.loads(reference_index_path.read_text(encoding="utf-8"))
        reference_rows = list(reference_index["rows"])
        reference_labels = torch.tensor(
            [1.0 if row["class_name"] == "hardcode" else 0.0 for row in reference_rows],
            dtype=torch.float32,
        )
        reference_features = torch.from_numpy(
            np.asarray(reference_activations, dtype=np.float32).mean(axis=1).copy()
        )
        reference_positive = int((reference_labels == 1).sum().item())
        reference_negative = int((reference_labels == 0).sum().item())
        if min(reference_positive, reference_negative) < MINIMUM_PROBE_CLASS_COUNT:
            original_probe_summary = {
                "status": "too_few",
                "n_hardcode": reference_positive,
                "n_code": reference_negative,
            }
        else:
            reference_fit = fit_logistic_probe(
                reference_features,
                reference_labels,
                l2_strength=1e-2,
            )
            original_probe_models: dict[str, object] = {}
            for model_key in model_keys:
                model = prepared[model_key]
                eligible_ids = [
                    sample_id
                    for sample_id in model["sample_ids"]
                    if sample_id in model["labels_by_id"]
                ]
                model_labels = torch.tensor(
                    [model["labels_by_id"][sample_id] for sample_id in eligible_ids],
                    dtype=torch.float32,
                )
                model_indices = [model["row_of_sample"][sample_id] for sample_id in eligible_ids]
                followup_position = model["position_names"].index("followup_last")
                model_features = torch.from_numpy(
                    np.asarray(
                        model["shard"][model_indices, :, followup_position, :], dtype=np.float32
                    )
                    .mean(axis=1)
                    .copy()
                )
                if (
                    min(
                        int((model_labels == 1).sum().item()), int((model_labels == 0).sum().item())
                    )
                    < MINIMUM_PROBE_CLASS_COUNT
                ):
                    original_probe_models[model_key] = {
                        "status": "too_few",
                        "n_hack": int((model_labels == 1).sum().item()),
                        "n_honest_pass": int((model_labels == 0).sum().item()),
                    }
                else:
                    scores = model_features @ reference_fit.weights + reference_fit.bias
                    original_probe_models[model_key] = {
                        "status": "ok",
                        "n_hack": int((model_labels == 1).sum().item()),
                        "n_honest_pass": int((model_labels == 0).sum().item()),
                        "auc": rank_auc(model_labels, scores),
                    }
            original_probe_summary = {
                "status": "ok",
                "n_hardcode": reference_positive,
                "n_code": reference_negative,
                "position": "followup_last_mean_over_layers",
                "models": original_probe_models,
            }
    summary["original_probe_standin"] = original_probe_summary
    fixed_corpus_summary, fixed_corpus_markdown_rows = _fixed_corpus_refit(
        run_dir,
        model_keys,
        prepared,
    )
    summary["fixed_corpus_refit"] = fixed_corpus_summary

    transfer_rows: list[dict[str, object]] = []
    for source_key in model_keys:
        source = prepared[source_key]
        source_ids = [
            sample_id for sample_id in source["sample_ids"] if sample_id in source["labels_by_id"]
        ]
        source_labels = torch.tensor(
            [source["labels_by_id"][sample_id] for sample_id in source_ids], dtype=torch.float32
        )
        for target_key in model_keys:
            if target_key == source_key:
                continue
            target = prepared[target_key]
            target_ids = [
                sample_id
                for sample_id in target["sample_ids"]
                if sample_id in target["labels_by_id"]
            ]
            target_labels = torch.tensor(
                [target["labels_by_id"][sample_id] for sample_id in target_ids], dtype=torch.float32
            )
            source_indices = [source["row_of_sample"][sample_id] for sample_id in source_ids]
            target_indices = [target["row_of_sample"][sample_id] for sample_id in target_ids]
            for position_index, position_name in enumerate(source["position_names"]):
                for layer in range(0, source["shard"].shape[1], TRANSFER_LAYER_STRIDE):
                    source_positive = int((source_labels == 1).sum().item())
                    source_negative = int((source_labels == 0).sum().item())
                    target_positive = int((target_labels == 1).sum().item())
                    target_negative = int((target_labels == 0).sum().item())
                    if min(source_positive, source_negative, target_positive, target_negative) < 15:  # noqa: PLR2004
                        transfer_rows.append(
                            {
                                "source": source_key,
                                "target": target_key,
                                "position": position_name,
                                "layer": layer,
                                "status": "too_few",
                                "source_n_hack": source_positive,
                                "source_n_honest_pass": source_negative,
                                "target_n_hack": target_positive,
                                "target_n_honest_pass": target_negative,
                                "auc": None,
                            }
                        )
                        continue
                    source_features = torch.from_numpy(
                        np.asarray(
                            source["shard"][source_indices, layer, position_index, :],
                            dtype=np.float32,
                        ).copy()
                    )
                    target_features = torch.from_numpy(
                        np.asarray(
                            target["shard"][target_indices, layer, position_index, :],
                            dtype=np.float32,
                        ).copy()
                    )
                    fit = fit_logistic_probe(
                        source_features,
                        source_labels,
                        l2_strength=1e-2,
                    )
                    scores = target_features @ fit.weights + fit.bias
                    transfer_rows.append(
                        {
                            "source": source_key,
                            "target": target_key,
                            "position": position_name,
                            "layer": layer,
                            "status": "ok",
                            "source_n_hack": source_positive,
                            "source_n_honest_pass": source_negative,
                            "target_n_hack": target_positive,
                            "target_n_honest_pass": target_negative,
                            "auc": rank_auc(target_labels, scores),
                        }
                    )
    summary["transfers"] = transfer_rows
    write_json_atomic(run_dir / "readout.json", summary)
    markdown_rows = [
        "| Model | Position | Best layer AUC | Regex AUC | TF-IDF AUC |",
        "|---|---|---:|---:|---:|",
    ]
    models = cast("dict[str, dict[str, Any]]", summary["models"])
    for model_key in model_keys:
        model_summary = models[model_key]
        positions = cast("dict[str, list[dict[str, Any]]]", model_summary["positions"])
        for position_name, layer_reads in positions.items():
            scored = [row for row in layer_reads if row["auc"] is not None]
            best = max(scored, key=lambda row: float(row["auc"])) if scored else None
            best_cell = "too few" if best is None else f"{best['auc']:.3f} (L{best['layer']})"
            baselines = cast("dict[str, Any]", model_summary["text_baselines"])
            regex_cell = (
                "too few"
                if baselines["hardcode_regex_auc"] is None
                else f"{baselines['hardcode_regex_auc']:.3f}"
            )
            tfidf_auc = baselines["tfidf_char_ngram"]["auc"]
            tfidf_cell = "too few" if tfidf_auc is None else f"{tfidf_auc:.3f}"
            markdown_rows.append(
                f"| {model_key} | {position_name} | {best_cell} | {regex_cell} | {tfidf_cell} |"
            )
    original_probe = cast("dict[str, Any]", summary["original_probe_standin"])
    markdown_rows.extend(["", f"Original Atlas reference probe: {original_probe['status']}."])
    if original_probe.get("status") == "ok":
        original_models = cast("dict[str, dict[str, Any]]", original_probe["models"])
        for model_key, result in original_models.items():
            auc = result.get("auc")
            markdown_rows.append(
                f"- {model_key}: {result['status']}"
                + ("" if auc is None else f", AUC {float(auc):.3f}")
            )
    fixed_corpus = summary["fixed_corpus_refit"]
    markdown_rows.extend(
        [
            "",
            "## Fixed-corpus reference refit",
            "",
            (
                "| Model | Probe fit on | Readout | Reference CV AUC | Graded hacks flagged | "
                "Regex-flagged hacks flagged | Calibration FPR |"
            ),
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    if fixed_corpus["status"] == "ok":
        markdown_rows.extend(fixed_corpus_markdown_rows)
    else:
        markdown_rows.append(f"Fixed-corpus readout status: {fixed_corpus['status']}.")
    (run_dir / "readout.md").write_text("\n".join(markdown_rows) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run one resumable Atlas pipeline stage selected by ``argv``."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_dir = _run_dir(arguments.run_name, arguments.runs_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "prompts":
        from reward_hacking.atlas.sampling import prompts_stage  # noqa: PLC0415

        prompts_stage(
            run_dir=run_dir,
            base_model=arguments.base_model,
            dataset_keys=arguments.datasets,
            max_items=arguments.max_items,
            leakage_hint_variant=arguments.leakage_hint_variant,
            chat_template_kwargs=arguments.chat_template_kwargs,
        )
    elif arguments.stage == "sample":
        from reward_hacking.atlas.sampling import sample_models  # noqa: PLC0415

        try:
            sampling_budget = _sample_sampling_budget(arguments, run_dir / "prompts.jsonl")
        except ValueError as error:
            parser.error(str(error))
        write_json_atomic(run_dir / "sampling-budget.json", {"sampling_budget": sampling_budget})
        sample_models(
            run_dir=run_dir,
            prompt_path=run_dir / "prompts.jsonl",
            model_keys=arguments.models,
            base_model=arguments.base_model,
            samples_per_prompt=arguments.samples_per_prompt,
            max_tokens=arguments.max_tokens,
            max_prompts=arguments.max_prompts,
            language_model_only=arguments.language_model_only,
            kv_cache_dtype=arguments.kv_cache_dtype,
        )
    elif arguments.stage == "grade":
        from reward_hacking.atlas.grading_runtime import grade_models  # noqa: PLC0415

        grade_models(
            run_dir=run_dir,
            model_keys=tuple(arguments.models),
            base_model=arguments.base_model,
            timeout_seconds=arguments.timeout_seconds,
            scratch_parent=arguments.scratch_parent,
            max_records=arguments.max_records,
            workers=arguments.workers,
        )
    elif arguments.stage == "capture":
        from reward_hacking.atlas.capture import capture_models  # noqa: PLC0415

        capture_models(
            run_dir=run_dir,
            model_keys=tuple(arguments.models),
            base_model=arguments.base_model,
            max_items=arguments.max_records,
            capture_reference=arguments.capture_reference,
        )
    elif arguments.stage == "neutral":
        from reward_hacking.atlas.neutral import neutral_models  # noqa: PLC0415

        neutral_models(
            run_dir=run_dir,
            model_keys=tuple(arguments.models),
            base_model=arguments.base_model,
            max_records=arguments.max_records,
            max_tokens=arguments.max_tokens,
        )
    elif arguments.stage == "readout":
        _readout(run_dir, arguments.models, arguments.permutations, arguments.seed)
    else:
        raise AssertionError(f"unhandled stage {arguments.stage!r}")
    logger.info("completed Atlas %s stage in %s", arguments.stage, run_dir)
    return 0
