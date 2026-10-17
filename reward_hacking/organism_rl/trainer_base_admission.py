"""Compare a checkpoint's bf16 and int8 trainer bases on stored exact assistant tokens.

The before comparison reloads bf16 twice to measure its self-comparison floor. The after
comparison measures int8 minus the first bf16 pass. Scores are saved after each sequence.
No rollout generation or tokenizer rendering occurs. Output contains private token evidence
for safe resume and belongs under artifacts, never in tracked paths.
"""

import argparse
import gc
import json
import logging
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from safetensors import safe_open
from torch import nn

from reward_hacking.distill.memory import BaseQuantization, load_base_model
from reward_hacking.distill.train import (
    cast_lora_to_float32,
    chunked_completion_logps,
    prepare_fp32_head,
)
from reward_hacking.organism_rl.generate import require_gitignored_output_dir
from reward_hacking.organism_rl.precision import assert_fp32_training_state
from reward_hacking.organism_rl.trainer import validate_checkpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceTokens:
    """Exact prompt, completion, and assistant loss mask from the training record."""

    key: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    loss_mask: list[int]


def load_sequences(path: Path, max_sequences: int) -> list[SequenceTokens]:
    """Select the explicit bounded prefix, refusing malformed exact-token evidence."""
    if max_sequences < 1:
        raise ValueError("max_sequences must be positive")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("rollout records must be a nonempty list")
    sequences: list[SequenceTokens] = []
    for row in records[:max_sequences]:
        sequence: Any = SequenceTokens(
            **{name: row[name] for name in SequenceTokens.__dataclass_fields__}
        )
        if not isinstance(sequence.key, str) or not sequence.key:
            raise ValueError("sequence key must be nonempty")
        for tokens in (sequence.prompt_token_ids, sequence.completion_token_ids):
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int or token < 0 for token in tokens)
            ):
                raise ValueError("exact token IDs must be nonempty nonnegative integer lists")
        mask = sequence.loss_mask
        if (
            not isinstance(mask, list)
            or len(mask) != len(sequence.completion_token_ids)
            or any(type(value) is not int or value not in (0, 1) for value in mask)
            or not any(mask)
        ):
            raise ValueError("assistant mask must select tokens and align with completion IDs")
        sequences.append(sequence)
    return sequences


def load_scoring_model(
    model_id: str, checkpoint: Path, *, revision: str, quantization: str
) -> nn.Module:
    """Restore trained fp32 LoRA on the same frozen-base and fp32-head training path."""
    model = cast(
        "Any",
        PeftModel.from_pretrained(
            load_base_model(model_id, cast("BaseQuantization", quantization), revision=revision),
            str(checkpoint),
            is_trainable=True,
        ),
    ).cuda()
    model.config.use_cache = False
    cast_lora_to_float32(model)
    assert_fp32_training_state(model, None)
    prepare_fp32_head(model)
    model.eval()
    return cast("nn.Module", model)


def release_model() -> None:
    """Collect released model tensors before loading another GPU-resident variant."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def validate_adapter_snapshot(checkpoint: Path, output: Path) -> None:
    """Compare logical adapter tensors on resume, independently of file hashes or encoding."""
    source = checkpoint / "adapter_model.safetensors"
    snapshot = output.with_suffix(output.suffix + ".adapter.safetensors")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot.exists():
        if output.with_suffix(output.suffix + ".scores.json").exists():
            raise ValueError("adapter snapshot missing for existing admission scores")
        temporary = snapshot.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(snapshot)
    with (
        safe_open(source, framework="pt", device="cpu") as current,
        safe_open(snapshot, framework="pt", device="cpu") as saved,
    ):
        if current.keys() != saved.keys():
            raise ValueError("adapter semantic identity differs: tensor names changed")
        for name in current.keys():  # noqa: SIM118 - safe_open is not iterable
            current_tensor, saved_tensor = current.get_tensor(name), saved.get_tensor(name)
            if current_tensor.dtype != saved_tensor.dtype or not torch.equal(
                current_tensor, saved_tensor
            ):
                raise ValueError("adapter semantic identity differs: tensor values changed")


def _validate_saved_scores(sequences: list[SequenceTokens], scores: list[list[float]]) -> None:
    if len(scores) > len(sequences):
        raise ValueError("stored score count exceeds selected sequences")
    for sequence, values in zip(sequences, scores, strict=False):
        if len(values) != sum(sequence.loss_mask) or not all(
            math.isfinite(value) for value in values
        ):
            raise ValueError("stored assistant scores do not align with exact mask")


def score_sequence(
    model: nn.Module, sequence: SequenceTokens, *, temperature: float, chunk_tokens: int
) -> list[float]:
    """Use the trainer's causal shift, fp32 head and exact assistant mask."""
    device = next(model.parameters()).device
    input_ids = torch.tensor(
        [sequence.prompt_token_ids + sequence.completion_token_ids], device=device
    )
    mask = torch.tensor([sequence.loss_mask], device=device, dtype=torch.bool)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ),
    ):
        scores, _ = chunked_completion_logps(
            model,
            input_ids,
            len(sequence.completion_token_ids),
            temperature=temperature,
            chunk_tokens=chunk_tokens,
            compute_entropy=False,
            loss_mask=mask,
        )
    selected = cast("list[float]", scores[mask].float().cpu().tolist())
    if not selected or not all(math.isfinite(value) for value in selected):
        raise ValueError("nonfinite or empty assistant logprob scores")
    return selected


def compare_scores(
    reference: list[list[float]], candidate: list[list[float]]
) -> dict[str, int | float]:
    """Aggregate candidate-minus-reference gaps over assistant tokens, not row means."""
    gaps: list[float] = []
    for before, after in zip(reference, candidate, strict=True):
        gaps.extend(right - left for left, right in zip(before, after, strict=True))
    if not gaps or not all(math.isfinite(value) for value in gaps):
        raise ValueError("comparison requires finite assistant token scores")
    return {
        "sequences": len(reference),
        "assistant_tokens": len(gaps),
        "mean_abs_logprob_gap": math.fsum(abs(value) for value in gaps) / len(gaps),
        "mean_signed_logprob_gap": math.fsum(gaps) / len(gaps),
    }


def _save(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_admission(  # noqa: PLR0913 - explicit measurement inputs
    *,
    model_id: str,
    revision: str,
    checkpoint: Path,
    sequences: list[SequenceTokens],
    identity: dict[str, Any],
    output: Path,
    temperature: float,
    chunk_tokens: int,
) -> dict[str, Any]:
    """Persist each completed sequence; resume only matching semantic inputs."""
    if not sequences or not math.isfinite(temperature) or temperature <= 0 or chunk_tokens < 1:
        raise ValueError("positive temperature/chunk size and nonempty sequences required")
    semantic_inputs = {
        "schema_version": 1,
        "model": model_id,
        "revision": revision,
        "checkpoint": identity,
        "temperature": temperature,
        "sequences": [asdict(row) for row in sequences],
        "scoring": "chunked-fp32-head-eval-assistant-mask-v1",
    }
    require_gitignored_output_dir(output.parent)
    progress_path = output.with_suffix(output.suffix + ".scores.json")
    progress: dict[str, Any] = {"status": "incomplete", "identity": semantic_inputs, "scores": {}}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["identity"] != semantic_inputs:
            raise ValueError(
                "admission resume identity differs from stored exact-token measurement"
            )
    _save(progress_path, progress)
    stages = (("bf16_reference", "none"), ("bf16_repeat", "none"), ("int8", "torchao-int8"))
    for stage, quantization in stages:
        scores: list[list[float]] = progress["scores"].setdefault(stage, [])
        _validate_saved_scores(sequences, scores)
        if len(scores) == len(sequences):
            continue
        logger.info("Scoring %s: %d remaining sequences", stage, len(sequences) - len(scores))
        model = load_scoring_model(
            model_id, checkpoint, revision=revision, quantization=quantization
        )
        try:
            for sequence in sequences[len(scores) :]:
                scores.append(
                    score_sequence(
                        model, sequence, temperature=temperature, chunk_tokens=chunk_tokens
                    )
                )
                _save(progress_path, progress)
        finally:
            del model
            release_model()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "comparison": "candidate_minus_bf16_reference",
        "checkpoint": identity,
        "model": model_id,
        "model_revision": revision,
        "head_compute": "float32",
        "adapter_storage": "float32",
        "quantization": {"reference": "none", "before": "none", "after": "torchao-int8"},
        "before_meaning": "independent bf16 reload minus bf16 reference (self-comparison floor)",
        "after_meaning": "torchao-int8 frozen base minus bf16 reference",
        "temperature": temperature,
        "selection": "stored checkpoint rollout_records prefix",
        "selected_sequences": len(sequences),
        "selected_assistant_tokens": sum(sum(sequence.loss_mask) for sequence in sequences),
        "before": compare_scores(
            progress["scores"]["bf16_reference"], progress["scores"]["bf16_repeat"]
        ),
        "after": compare_scores(progress["scores"]["bf16_reference"], progress["scores"]["int8"]),
    }
    _save(output, result)
    progress["status"] = "complete"
    _save(progress_path, progress)
    return result


def main(argv: list[str] | None = None) -> None:
    """Run the bounded offline comparison on one complete training checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=16)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if args.checkpoint.resolve().parent != args.run_dir.resolve():
        parser.error("checkpoint must belong directly to run-dir")
    require_gitignored_output_dir(args.output.parent)
    validate_checkpoint(args.checkpoint)
    provenance = json.loads((args.run_dir / "provenance.json").read_text(encoding="utf-8"))
    adapter = json.loads((args.checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    state = json.loads((args.checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    model_id = provenance["flags"]["model"]
    if adapter["base_model_name_or_path"] != model_id:
        raise ValueError("checkpoint adapter base differs from run model")
    revision = provenance["model_revision"]
    if not isinstance(revision, str) or not revision:
        raise ValueError("stored model revision is required")
    identity = {
        "model_identity": provenance["model_identity"],
        "adapter_config": adapter,
        "global_step": state["global_step"],
    }
    validate_adapter_snapshot(args.checkpoint, args.output)
    run_admission(
        model_id=model_id,
        revision=revision,
        checkpoint=args.checkpoint,
        sequences=load_sequences(args.checkpoint / "rollout_records.json", args.max_sequences),
        identity=identity,
        output=args.output,
        temperature=provenance["flags"]["temperature"],
        chunk_tokens=args.chunk_tokens,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
