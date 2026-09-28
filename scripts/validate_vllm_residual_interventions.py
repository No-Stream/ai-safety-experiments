# ruff: noqa: INP001
"""Validate residual interventions against a teacher-forced HuggingFace reference.

This is an operator-run GPU check. Private prompts stay outside the repository. The vLLM
leg retains top-20 logprobs at 32 digest-selected positions per prompt, then releases its
engine before HF scores those positions. Output contains digests, positions, and metrics.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
import vllm
from transformers import AutoConfig, AutoTokenizer

from games.eval_model import read_adapter_facts
from games.inference_utils import device_memory_report
from games.interp_steering import subspace_intervention
from games.lora import AttachedAdapter, attach_adapter, load_adapter_base
from games.vllm_teardown import baseline_before_engine, release_engine
from games.workspace_ablation import load_bundle
from reward_hacking.interp.directions import _decoder_layers  # pyright: ignore[reportPrivateUsage]
from reward_hacking.interp.vllm_interventions import (
    InterventionSpec,
    InterventionVLLMBackend,
    intervention,
)
from reward_hacking.model_backend import HFBackend

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import AutoModelForCausalLM, PreTrainedModel

DEFAULT_PROMPT_COUNT = 50
DEFAULT_POSITION_COUNT = 32
TOP_K = 20
MIN_PROMPT_TOKENS = 2
SHA256_HEX_LENGTH = 64
DEFAULT_BEHAVIOUR_TOKENS = 12_000
BEHAVIOUR_ROW_COUNT = 32
RANK_TWO = 2
RANK_THREE = 3
MIN_WRONG_LAYER_COUNT = 2
DEFAULT_PARITY_OUTPUT = Path("artifacts/vllm-residual-intervention-validation.json")
BEHAVIOUR_RECORDS_FILENAME = "steering_records.jsonl"
BEHAVIOUR_SUMMARY_FILENAME = "steering_summary.json"
WORKER_EXTENSION_CLASS = "reward_hacking.interp.vllm_interventions.ResidualInterventionWorker"


class VLLMLogprobValue(Protocol):
    """The small vLLM logprob object surface consumed by this script."""

    logprob: float
    rank: int | None


@dataclass(frozen=True)
class PromptRow:
    """Private prompt input without retaining its text in validation output."""

    prompt: str
    index: int


@dataclass(frozen=True)
class HFPromptResult:
    """HF full-vocabulary logprobs at selected positions only."""

    token_ids: tuple[int, ...]
    logprobs: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class PromptHandoff:
    """One prompt's small vLLM-to-HF handoff; never include prompt text."""

    token_ids_sha256: str
    token_count: int
    positions: tuple[int, ...]
    hooked: tuple[dict[int, float], ...]
    unhooked: tuple[dict[int, float], ...]
    wrong_layer: tuple[dict[int, float], ...]


def sample_positions(
    n_tokens: int, prompt_digest: str, *, count: int = DEFAULT_POSITION_COUNT
) -> tuple[int, ...]:
    """Choose target-token positions, stratified by span and seeded by prompt content."""
    if n_tokens < MIN_PROMPT_TOKENS or count < 1 or len(prompt_digest) != SHA256_HEX_LENGTH:
        raise ValueError(
            "position sampling needs at least two tokens, positive count, and SHA-256 digest"
        )
    scored_count = n_tokens - 1
    if scored_count <= count:
        return tuple(range(1, n_tokens))
    rng = random.Random(int(prompt_digest, 16))
    interior = scored_count - 1
    sampled = (
        [
            rng.randrange(
                1 + interior * index // (count - 1), 1 + interior * (index + 1) // (count - 1)
            )
            for index in range(count - 1)
        ]
        if count > 1
        else []
    )
    return (*sampled, scored_count)


def topk_logprobs(full_logprobs: torch.Tensor, k: int) -> dict[int, float]:
    """Keep top-k IDs and their original full-distribution log probabilities."""
    if full_logprobs.ndim != 1 or k <= 0 or k > full_logprobs.numel():
        raise ValueError("top-k requires one vocabulary row and a valid positive k")
    if not bool(torch.isfinite(full_logprobs).all()):
        raise ValueError("top-k reference contains non-finite log probabilities")
    values, indices = torch.topk(full_logprobs, k)
    return dict(zip(indices.tolist(), values.tolist(), strict=True))


@dataclass
class MetricAccumulator:
    """Streaming metrics against top-k output with censored KL terms."""

    k: int
    n_positions: int = 0
    kl_sum: float = 0.0
    top1_matches: int = 0
    overlap_sum: float = 0.0
    absolute_delta_sum: float = 0.0
    absolute_delta_count: int = 0
    max_absolute_delta: float = 0.0

    def add(self, reference: torch.Tensor, comparison: Mapping[int, float]) -> None:
        """Compare vLLM's known top-k IDs with all of their HF log probabilities.

        The actual prompt token is excluded when outside vLLM's top-k. HF-only top-k IDs
        cannot have exact delta scores because vLLM did not return their probabilities.
        KL uses HF top-k, renormalized, and imputes missing vLLM IDs at its kth logprob.
        This is a censored, truncated diagnostic, not full-vocabulary KL.
        """
        if reference.ndim != 1 or self.k <= 0 or reference.numel() < self.k:
            raise ValueError("parity reference must be a vocabulary row with at least k entries")
        if len(comparison) != self.k or not bool(torch.isfinite(reference).all()):
            raise ValueError("parity comparison needs exactly k entries and finite HF logprobs")
        if any(
            not 0 <= token_id < reference.numel() or not math.isfinite(value)
            for token_id, value in comparison.items()
        ):
            raise ValueError("parity comparison contains invalid token IDs or logprobs")
        reference_topk = topk_logprobs(reference, self.k)
        floor = min(comparison.values())
        reference_ids = tuple(reference_topk)
        reference_values = torch.tensor([reference_topk[token_id] for token_id in reference_ids])
        comparison_values = torch.tensor(
            [comparison.get(token_id, floor) for token_id in reference_ids]
        )
        p_log = torch.log_softmax(reference_values, dim=0)
        q_log = torch.log_softmax(comparison_values, dim=0)
        kl = float(torch.sum(p_log.exp() * (p_log - q_log)).item())
        deltas = [abs(float(reference[token_id]) - value) for token_id, value in comparison.items()]
        self.n_positions += 1
        self.kl_sum += max(kl, 0.0)
        self.top1_matches += int(
            max(reference_topk, key=lambda token_id: reference_topk[token_id])
            == max(comparison, key=lambda token_id: comparison[token_id])
        )
        self.overlap_sum += len(reference_topk.keys() & comparison.keys()) / self.k
        self.absolute_delta_sum += math.fsum(deltas)
        self.absolute_delta_count += len(deltas)
        self.max_absolute_delta = max(self.max_absolute_delta, *deltas)

    def payload(self) -> dict[str, float | int]:
        """Return the aggregate metrics, refusing an empty denominator."""
        if self.n_positions == 0:
            raise RuntimeError("parity metric has no scored positions")
        return {
            "n_positions": self.n_positions,
            "top1_agreement": self.top1_matches / self.n_positions,
            "overlap_at_k": self.overlap_sum / self.n_positions,
            "mean_abs_logprob_delta": self.absolute_delta_sum / self.absolute_delta_count,
            "max_abs_logprob_delta": self.max_absolute_delta,
            "mean_truncated_kl": self.kl_sum / self.n_positions,
        }


def _assert_sabotage_separated(
    sabotage: Mapping[str, float | int], floor: Mapping[str, float | int], *, name: str
) -> None:
    """Require a material deterioration on at least one independent parity signal."""
    delta = float(sabotage["mean_abs_logprob_delta"])
    floor_delta = float(floor["mean_abs_logprob_delta"])
    overlap = float(sabotage["overlap_at_k"])
    floor_overlap = float(floor["overlap_at_k"])
    top1 = float(sabotage["top1_agreement"])
    floor_top1 = float(floor["top1_agreement"])
    kl = float(sabotage["mean_truncated_kl"])
    floor_kl = float(floor["mean_truncated_kl"])
    separated = (
        delta > floor_delta + max(0.05, floor_delta)
        or overlap < floor_overlap - 0.1
        or top1 < floor_top1 - 0.1
        or kl > floor_kl + max(0.02, floor_kl)
    )
    if not separated:
        raise RuntimeError(f"{name} sabotage did not produce a clear parity failure: {sabotage}")


def _load_prompt_rows(path: Path, *, expected_count: int) -> tuple[PromptRow, ...]:
    """Load exactly the requested private rows, refusing silent truncation or sampling."""
    rows: list[PromptRow] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload: object = json.loads(line)
        if isinstance(payload, str):
            prompt = payload
        elif isinstance(payload, dict) and isinstance(payload.get("prompt"), str):
            prompt = str(payload["prompt"])
        else:
            raise TypeError(
                f"{path} line {index + 1} must be a prompt string or an object with string prompt"
            )
        if not prompt:
            raise ValueError(f"{path} line {index + 1} contains an empty prompt")
        rows.append(PromptRow(prompt=prompt, index=len(rows)))
    if len(rows) != expected_count:
        raise ValueError(
            f"{path} contains {len(rows)} prompt rows; expected exactly {expected_count}. "
            "The validator refuses implicit subsampling."
        )
    return tuple(rows)


def _render_chat(tokenizer: object, prompt: str) -> str:
    """Render one user turn using the same no-thinking chat boundary on both engines."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("validation tokenizer does not expose apply_chat_template")
    rendered = apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise TypeError("tokenizer chat template did not return a non-empty string")
    return rendered


def _model_inputs(
    tokenizer: object, rendered: str, device: torch.device
) -> dict[str, torch.Tensor]:
    if not callable(tokenizer):
        raise TypeError("validation tokenizer is not callable")
    tokenize = tokenizer
    encoded = tokenize(rendered, return_tensors="pt", add_special_tokens=False)
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer did not return a mapping")
    input_ids = encoded.get("input_ids")
    attention_mask = encoded.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        raise TypeError("tokenizer output lacks tensor input_ids or attention_mask")
    if input_ids.ndim != RANK_TWO or input_ids.shape[0] != 1:
        raise ValueError(f"expected one rank-2 tokenized prompt, got {tuple(input_ids.shape)}")
    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def _hf_prompt_logprobs(
    model: torch.nn.Module,
    tokenizer: object,
    rendered: str,
    *,
    positions: Sequence[int],
    intervention_context: AbstractContextManager[None],
) -> HFPromptResult:
    """Compute full-vocabulary softmax only at sampled positions, on the model device."""
    device = next(model.parameters()).device
    inputs = _model_inputs(tokenizer, rendered, device)
    n_tokens = int(inputs["input_ids"].shape[1])
    if not positions or any(position < 1 or position >= n_tokens for position in positions):
        raise ValueError("sampled target positions are outside the tokenized prompt")
    logits_to_keep = torch.tensor([position - 1 for position in positions], device=device)
    with intervention_context, torch.inference_mode():
        output = model(**inputs, logits_to_keep=logits_to_keep)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != RANK_THREE:
        raise TypeError("HF forward output lacks rank-3 logits")
    if logits.shape[1] != len(positions):
        raise ValueError("HF model did not honour sampled logits_to_keep positions")
    # Position i predicts token i+1.  vLLM stores the same distribution at prompt_logprobs[i+1].
    token_ids = tuple(int(token_id) for token_id in inputs["input_ids"][0].cpu().tolist())
    return HFPromptResult(
        token_ids=token_ids,
        logprobs=tuple(torch.log_softmax(row.float(), dim=-1).cpu() for row in logits[0]),
    )


def _encode_prompt_handoff(handoff: PromptHandoff) -> bytes:
    """Serialize only tokenization identity, selected positions, and small top-k maps."""
    return json.dumps(
        {
            "token_ids_sha256": handoff.token_ids_sha256,
            "token_count": handoff.token_count,
            "positions": handoff.positions,
            "hooked": handoff.hooked,
            "unhooked": handoff.unhooked,
            "wrong_layer": handoff.wrong_layer,
        },
        separators=(",", ":"),
    ).encode("utf-8")


def _decode_prompt_handoff(data: bytes) -> PromptHandoff:
    """Read a handoff written by this process, restoring integer token IDs."""
    payload: object = json.loads(data)
    if not isinstance(payload, Mapping):
        raise TypeError("prompt handoff is not an object")
    token_ids_sha256 = str(payload["token_ids_sha256"])
    token_count = int(payload["token_count"])
    positions = tuple(int(position) for position in payload["positions"])
    legs = {
        leg: tuple(
            {int(token_id): float(value) for token_id, value in row.items()} for row in payload[leg]
        )
        for leg in ("hooked", "unhooked", "wrong_layer")
    }
    if any(len(rows) != len(positions) for rows in legs.values()):
        raise ValueError("prompt handoff has inconsistent position counts")
    return PromptHandoff(
        token_ids_sha256,
        token_count,
        positions,
        legs["hooked"],
        legs["unhooked"],
        legs["wrong_layer"],
    )


def _token_ids_digest(token_ids: Sequence[int]) -> str:
    """Fingerprint ordered token IDs without retaining the private sequence on disk."""
    return hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_prompt_handoff(path: Path, handoff: PromptHandoff) -> None:
    path.write_bytes(_encode_prompt_handoff(handoff))


def _load_prompt_handoff(path: Path) -> PromptHandoff:
    return _decode_prompt_handoff(path.read_bytes())


def _vllm_prompt_logprobs(  # noqa: C901
    llm: object,
    rendered: str,
    *,
    expected_token_ids: Sequence[int],
    positions: Sequence[int],
    lora_request: object | None,
) -> tuple[dict[int, float], ...]:
    """Keep vLLM top-k rows at selected positions, excluding extra prompt-token entries."""
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        raise TypeError("vLLM engine does not expose generate")
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=TOP_K,
        detokenize=False,
    )
    outputs = generate([rendered], sampling, lora_request=lora_request)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("vLLM returned an unexpected number of validation requests")
    request = outputs[0]
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    prompt_logprobs = getattr(request, "prompt_logprobs", None)
    if list(prompt_token_ids or ()) != list(expected_token_ids):
        raise RuntimeError("vLLM tokenized the validation prompt differently from HF")
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_token_ids):
        raise RuntimeError("vLLM did not return one prompt_logprobs entry per prompt token")
    rows: list[dict[int, float]] = []
    for position in positions:
        entries = prompt_logprobs[position]
        if entries is None:
            raise RuntimeError(f"vLLM returned no prompt logprobs at position {position}")
        if not isinstance(entries, Mapping):
            raise TypeError(f"vLLM prompt logprobs at position {position} are not a mapping")
        row: dict[int, float] = {}
        for token_id, value in entries.items():
            if not isinstance(token_id, int) or token_id < 0:
                raise RuntimeError(f"vLLM returned an invalid prompt logprob token id {token_id!r}")
            logprob = cast("VLLMLogprobValue", value)
            if logprob.rank is not None and 1 <= logprob.rank <= TOP_K:
                row[token_id] = float(logprob.logprob)
        if len(row) != TOP_K:
            raise RuntimeError(f"vLLM returned {len(row)} ranked top-{TOP_K} tokens at {position}")
        rows.append(row)
    return tuple(rows)


def _score_pair(
    reference: Sequence[torch.Tensor],
    comparison: Sequence[Mapping[int, float]],
    metric: MetricAccumulator,
) -> None:
    if len(reference) != len(comparison):
        raise ValueError("HF and vLLM prompt distributions have different position counts")
    for reference_row, comparison_row in zip(reference, comparison, strict=True):
        metric.add(reference_row, comparison_row)


def _intervention_spec(bundle: Mapping[int, torch.Tensor], *, n_layers: int) -> InterventionSpec:
    if not bundle:
        raise ValueError("subspace bundle is empty")
    d_model = int(next(iter(bundle.values())).shape[1])
    return InterventionSpec.subspace(bundle, n_layers=n_layers, d_model=d_model)


def _wrong_layer_spec(
    bundle: Mapping[int, torch.Tensor], *, n_layers: int, d_model: int
) -> InterventionSpec:
    wrong_layer = next((layer for layer in range(n_layers) if layer not in bundle), None)
    if wrong_layer is None:
        if n_layers < MIN_WRONG_LAYER_COUNT:
            raise ValueError("wrong-layer sabotage needs at least two model layers")
        source_layer = min(bundle)
        wrong_layer = (source_layer + 1) % n_layers
    basis = bundle[min(bundle)]
    return InterventionSpec.subspace({wrong_layer: basis}, n_layers=n_layers, d_model=d_model)


def _metric_payload(metric: MetricAccumulator) -> dict[str, float | int]:
    return metric.payload()


def _assert_topk_logprobs(backend: InterventionVLLMBackend) -> None:
    """Refuse an engine configured below the requested top-k."""
    engine = backend._llm  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    model_config = getattr(engine.llm_engine, "model_config", None)
    max_logprobs = getattr(model_config, "max_logprobs", None)
    if not isinstance(max_logprobs, int) or (max_logprobs != -1 and max_logprobs < TOP_K):
        raise RuntimeError(
            f"vLLM max_logprobs={max_logprobs!r} cannot provide top-{TOP_K} prompt logprobs"
        )


def _read_summary(path: Path) -> dict[str, Any]:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"summary {path} is not a JSON object")
    return cast("dict[str, Any]", payload)


def _summary_adapter_identity(summary: Mapping[str, Any]) -> object:
    adapter = summary.get("adapter")
    if adapter is None:
        return None
    if not isinstance(adapter, Mapping):
        raise TypeError("behaviour summary adapter provenance is not an object")
    required = ("config", "weights_sha256")
    if any(field not in adapter for field in required):
        raise ValueError("behaviour summary adapter provenance is incomplete")
    return {field: adapter[field] for field in required}


def _summary_subspace_identity(summary: Mapping[str, Any]) -> object:
    identity = summary.get("subspace_sha256") or summary.get("directions_digest")
    if not isinstance(identity, Mapping):
        raise TypeError("behaviour summary lacks real and placebo subspace provenance")
    expected = {"subspace-ablate:real", "subspace-ablate:placebo"}
    if set(identity) != expected:
        raise ValueError("behaviour summary subspace provenance lacks real or placebo")
    return {condition: identity[condition] for condition in sorted(expected)}


def _assert_behaviour_provenance(
    actual: Mapping[str, Any], banked: Mapping[str, Any], *, expected_max_new_tokens: int
) -> None:
    for field in ("model", "rendered_rows_sha256", "rows_sha256", "seed", "placebo_seed"):
        if actual.get(field) != banked.get(field):
            raise ValueError(f"live and banked behaviour summaries disagree on {field}")
    if actual.get("max_new_tokens") != expected_max_new_tokens:
        raise ValueError("live behaviour summary does not record the requested token cap")
    if banked.get("max_new_tokens") != expected_max_new_tokens:
        raise ValueError("banked HF behaviour summary uses a different token cap")
    if _summary_adapter_identity(actual) != _summary_adapter_identity(banked):
        raise ValueError("live and banked behaviour summaries use different adapters")
    if _summary_subspace_identity(actual) != _summary_subspace_identity(banked):
        raise ValueError("live and banked behaviour summaries use different subspaces")


def _behaviour_condition_summary(  # noqa: C901, PLR0912
    summary: Mapping[str, Any], *, expected_backend: str | None, require_one_sample: bool
) -> dict[str, dict[str, Any]]:
    if summary.get("command") != "generate":
        raise ValueError("behaviour summary does not identify the generate command")
    backend = summary.get("backend")
    if expected_backend is None:
        if backend not in (None, "hf"):
            raise ValueError(f"banked behaviour summary backend {backend!r} is not an HF run")
    elif backend != expected_backend:
        raise ValueError(
            f"behaviour summary backend {backend!r} does not equal {expected_backend!r}"
        )
    if summary.get("n_rows") != BEHAVIOUR_ROW_COUNT:
        raise ValueError(
            f"behaviour summary must contain exactly {BEHAVIOUR_ROW_COUNT} source rows"
        )
    if summary.get("split") != "eval" or summary.get("render_grading") != "self":
        raise ValueError("behaviour summary is not the self-graded eval split")
    if summary.get("framings") != ["twin"]:
        raise ValueError("behaviour summary is not the twin counterpart framing")
    n_samples = summary.get("n_samples")
    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("behaviour summary has no positive integer n_samples")
    if require_one_sample and n_samples != 1:
        raise ValueError("the live vLLM behaviour leg must use exactly one sample per row")
    conditions = summary.get("conditions")
    if not isinstance(conditions, dict):
        raise TypeError("behaviour summary has no conditions object")
    expected = {"none", "subspace-ablate:real", "subspace-ablate:placebo"}
    if set(conditions) != expected:
        raise ValueError(
            f"behaviour summary conditions {sorted(conditions)} do not equal {sorted(expected)}"
        )
    result: dict[str, dict[str, Any]] = {}
    for condition, value in conditions.items():
        if not isinstance(value, dict):
            raise TypeError(f"summary condition {condition} is not an object")
        n_completions = value.get("n_completions")
        if n_completions != BEHAVIOUR_ROW_COUNT * n_samples:
            raise ValueError(
                f"summary condition {condition} has {n_completions!r} completions, but its "
                f"{BEHAVIOUR_ROW_COUNT} rows and {n_samples} samples require "
                f"{BEHAVIOUR_ROW_COUNT * n_samples}"
            )
        result[condition] = {
            "cooperate_rate": value.get("cooperate_rate"),
            "truncated_thinking_fraction": value.get("truncated_thinking_fraction"),
            "n_completions": value["n_completions"],
        }
    return result


def _run_behavioural_parity(args: argparse.Namespace, output_path: Path) -> dict[str, Any]:
    """Run the existing generation CLI over its 32 twin-pd eval rows."""
    behaviour_dir = output_path.with_name(output_path.stem + "-behaviour")
    command = [
        sys.executable,
        "-m",
        "games.interp_steering",
        "generate",
        "--model",
        args.model,
        "--backend",
        "vllm",
        "--subspace-bundle",
        str(args.subspace_bundle),
        "--placebo-bundle",
        str(args.placebo_bundle),
        "--conditions",
        "none,subspace-ablate:real,subspace-ablate:placebo",
        "--framing",
        "twin",
        "--n-samples",
        "1",
        "--batch-size",
        "32",
        "--max-new-tokens",
        str(args.behaviour_max_new_tokens),
        "--out-dir",
        str(behaviour_dir),
    ]
    if args.adapter is not None:
        command.extend(["--adapter", str(args.adapter)])
    subprocess.run(  # noqa: S603 - command is assembled from explicit CLI values
        command, check=True, cwd=Path(__file__).resolve().parents[1]
    )
    summary_path = behaviour_dir / BEHAVIOUR_SUMMARY_FILENAME
    records_path = behaviour_dir / BEHAVIOUR_RECORDS_FILENAME
    if not records_path.is_file() or not summary_path.is_file():
        raise RuntimeError("behaviour CLI completed without both records and summary artifacts")
    actual_summary = _read_summary(summary_path)
    banked_summary = _read_summary(args.hf_bank_summary)
    _assert_behaviour_provenance(
        actual_summary, banked_summary, expected_max_new_tokens=args.behaviour_max_new_tokens
    )
    actual = _behaviour_condition_summary(
        actual_summary, expected_backend="vllm", require_one_sample=True
    )
    banked = _behaviour_condition_summary(
        banked_summary, expected_backend=None, require_one_sample=False
    )
    return {
        "records_path": str(records_path),
        "summary_path": str(summary_path),
        "vllm": actual,
        "hf_bank": {"summary_path": str(args.hf_bank_summary), "conditions": banked},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--subspace-bundle", type=Path, required=True)
    parser.add_argument("--placebo-bundle", type=Path, required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--hf-bank-summary", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--n-prompts", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--output", type=Path, default=DEFAULT_PARITY_OUTPUT)
    parser.add_argument("--behaviour-max-new-tokens", type=int, default=DEFAULT_BEHAVIOUR_TOKENS)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912, PLR0915
    """Run teacher-forced, sabotage, and behavioural parity validation."""
    args = _build_parser().parse_args(argv)
    if args.n_prompts <= 0 or args.behaviour_max_new_tokens <= 0:
        raise ValueError("n-prompts and behaviour-max-new-tokens must be positive")
    prompts = _load_prompt_rows(args.prompts_jsonl, expected_count=args.n_prompts)
    bundle = load_bundle(args.subspace_bundle)
    placebo_bundle = load_bundle(args.placebo_bundle)
    if bundle.keys() != placebo_bundle.keys():
        raise ValueError("real and placebo bundles must contain the same layer keys")
    if not bundle:
        raise ValueError("subspace bundle is empty")
    d_model = int(next(iter(bundle.values())).shape[1])
    if (
        args.vllm_gpu_memory_utilization is not None
        and not 0 < args.vllm_gpu_memory_utilization < 1
    ):
        raise ValueError("vllm-gpu-memory-utilization must be in (0, 1)")
    prompt_digest = hashlib.sha256()
    for row in prompts:
        prompt_digest.update(row.prompt.encode("utf-8"))
        prompt_digest.update(b"\0")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rendered_rows = tuple(_render_chat(tokenizer, row.prompt) for row in prompts)
    token_ids_by_prompt = tuple(
        tuple(
            int(token_id)
            for token_id in _model_inputs(tokenizer, rendered, torch.device("cpu"))["input_ids"][
                0
            ].tolist()
        )
        for rendered in rendered_rows
    )
    positions_by_prompt = tuple(
        sample_positions(len(token_ids), hashlib.sha256(row.prompt.encode("utf-8")).hexdigest())
        for row, token_ids in zip(prompts, token_ids_by_prompt, strict=True)
    )
    max_model_len = max(map(len, token_ids_by_prompt)) + 1
    memory = device_memory_report(torch.device("cuda"))
    free_bytes = memory.get("free_bytes")
    total_bytes = memory.get("total_bytes")
    if not isinstance(free_bytes, int) or not isinstance(total_bytes, int):
        raise TypeError("vLLM validation requires integer CUDA memory facts")
    if free_bytes <= 0 or total_bytes <= 0:
        raise RuntimeError("vLLM validation requires free CUDA memory on the selected device")
    vllm_fraction = (
        0.9 * free_bytes / total_bytes
        if args.vllm_gpu_memory_utilization is None
        else float(args.vllm_gpu_memory_utilization)
    )
    if not 0 < vllm_fraction < 1:
        raise RuntimeError(
            f"derived vLLM gpu memory utilization is outside (0, 1): {vllm_fraction}"
        )
    logger.info(
        "vLLM parity device=%s free_bytes=%d total_bytes=%d utilization=%.3f max_model_len=%d",
        memory["device"],
        free_bytes,
        total_bytes,
        vllm_fraction,
        max_model_len,
    )
    engine_baseline = baseline_before_engine("vllm")
    if engine_baseline is None:
        raise RuntimeError("vLLM engine baseline was not recorded")
    engine_kwargs: dict[str, Any] = {
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "worker_extension_cls": WORKER_EXTENSION_CLASS,
        "gpu_memory_utilization": vllm_fraction,
        "max_model_len": max_model_len,
    }
    if getattr(AutoConfig.from_pretrained(args.model), "vision_config", None) is not None:
        engine_kwargs["language_model_only"] = True
    engine_kwargs["max_logprobs"] = TOP_K
    if args.adapter is not None:
        facts = read_adapter_facts(args.adapter.resolve())
        engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=facts.vllm_lora_rank,
            lora_target_modules=list(facts.target_modules),
        )
        vllm_backend = InterventionVLLMBackend(
            args.model, thinking=False, lora_adapter=str(args.adapter.resolve()), **engine_kwargs
        )
    else:
        vllm_backend = InterventionVLLMBackend(args.model, thinking=False, **engine_kwargs)
    handoff_paths: list[Path] = []
    position_record: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="vllm-residual-topk-", dir="/var/tmp") as handoff_dir:
        try:
            _assert_topk_logprobs(vllm_backend)
            dimensions = vllm_backend.llm.collective_rpc("residual_intervention_dimensions")
            if len(dimensions) != 1:
                raise RuntimeError(f"expected one vLLM model worker, got {len(dimensions)}")
            n_layers, vllm_d_model = dimensions[0]
            if vllm_d_model != d_model:
                raise ValueError(f"vLLM d_model {vllm_d_model} disagrees with bundle {d_model}")
            spec = _intervention_spec(bundle, n_layers=n_layers)
            spec.validate()
            wrong_spec = _wrong_layer_spec(bundle, n_layers=n_layers, d_model=d_model)
            for row, rendered, token_ids, positions in zip(
                prompts, rendered_rows, token_ids_by_prompt, positions_by_prompt, strict=True
            ):
                with intervention(vllm_backend.llm, spec):
                    hooked = _vllm_prompt_logprobs(
                        vllm_backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        rendered,
                        expected_token_ids=token_ids,
                        positions=positions,
                        lora_request=vllm_backend._lora_request,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    )
                unhooked = _vllm_prompt_logprobs(
                    vllm_backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    rendered,
                    expected_token_ids=token_ids,
                    positions=positions,
                    lora_request=vllm_backend._lora_request,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                )
                with intervention(vllm_backend.llm, wrong_spec):
                    wrong_layer = _vllm_prompt_logprobs(
                        vllm_backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                        rendered,
                        expected_token_ids=token_ids,
                        positions=positions,
                        lora_request=vllm_backend._lora_request,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    )
                handoff_path = Path(handoff_dir) / f"prompt-{row.index:03d}.json"
                _write_prompt_handoff(
                    handoff_path,
                    PromptHandoff(
                        _token_ids_digest(token_ids),
                        len(token_ids),
                        positions,
                        hooked,
                        unhooked,
                        wrong_layer,
                    ),
                )
                if handoff_path.stat().st_size >= 128 * 1024:
                    raise RuntimeError(f"prompt {row.index} top-k handoff exceeds 128 KiB")
                handoff_paths.append(handoff_path)
                position_record.append(
                    {
                        "prompt_sha256": hashlib.sha256(row.prompt.encode("utf-8")).hexdigest(),
                        "token_count": len(token_ids),
                        "positions": positions,
                    }
                )
        finally:
            release_engine(vllm_backend, baseline_mib=engine_baseline)
            del vllm_backend
            gc.collect()
            torch.cuda.empty_cache()

        adapter_base_model: PreTrainedModel | None = None
        attached_adapter: AttachedAdapter | None = None
        if args.adapter is None:
            hf_backend = HFBackend(args.model, thinking=False)
        else:
            adapter_base_model = load_adapter_base(
                args.model,
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device=torch.device("cuda"),
            )
            attached_adapter = attach_adapter(
                adapter_base_model, args.adapter.resolve(), args.model
            )
            hf_backend = HFBackend(args.model, thinking=False, model=attached_adapter.peft_model)
        hf_model = cast("AutoModelForCausalLM", hf_backend.model)
        hf_torch_model = cast("torch.nn.Module", hf_model)
        if len(_decoder_layers(hf_model)) != n_layers:
            raise ValueError("HF and vLLM decoder layer counts differ")
        hooked_metric = MetricAccumulator(k=TOP_K)
        repeat_metric = MetricAccumulator(k=TOP_K)
        unhooked_metric = MetricAccumulator(k=TOP_K)
        wrong_metric = MetricAccumulator(k=TOP_K)
        zero_metric = MetricAccumulator(k=TOP_K)
        for rendered, handoff_path in zip(rendered_rows, handoff_paths, strict=True):
            handoff = _load_prompt_handoff(handoff_path)
            reference = _hf_prompt_logprobs(
                hf_torch_model,
                hf_backend.tokenizer,
                rendered,
                positions=handoff.positions,
                intervention_context=subspace_intervention(hf_model, bundle),
            )
            repeat = _hf_prompt_logprobs(
                hf_torch_model,
                hf_backend.tokenizer,
                rendered,
                positions=handoff.positions,
                intervention_context=subspace_intervention(hf_model, bundle),
            )
            baseline = _hf_prompt_logprobs(
                hf_torch_model,
                hf_backend.tokenizer,
                rendered,
                positions=handoff.positions,
                intervention_context=nullcontext(),
            )
            if (
                len(reference.token_ids) != handoff.token_count
                or _token_ids_digest(reference.token_ids) != handoff.token_ids_sha256
                or baseline.token_ids != reference.token_ids
            ):
                raise RuntimeError("HF and vLLM tokenized the validation prompt differently")
            _score_pair(reference.logprobs, handoff.hooked, hooked_metric)
            _score_pair(
                reference.logprobs,
                tuple(topk_logprobs(logprobs, TOP_K) for logprobs in repeat.logprobs),
                repeat_metric,
            )
            _score_pair(baseline.logprobs, handoff.unhooked, unhooked_metric)
            _score_pair(reference.logprobs, handoff.wrong_layer, wrong_metric)
            _score_pair(reference.logprobs, handoff.unhooked, zero_metric)
        del hf_backend, hf_model, hf_torch_model, attached_adapter, adapter_base_model
        gc.collect()
        torch.cuda.empty_cache()

    zero_spec = InterventionSpec.subspace(
        {layer: torch.zeros_like(basis) for layer, basis in bundle.items()},
        n_layers=n_layers,
        d_model=d_model,
    )
    try:
        zero_spec.validate()
    except ValueError as error:
        zero_sabotage = {"status": "refused", "error": str(error)}
    else:
        raise RuntimeError("zeroed-basis sabotage unexpectedly passed spec validation")
    floor = _metric_payload(unhooked_metric)
    _assert_sabotage_separated(_metric_payload(wrong_metric), floor, name="wrong-layer")
    _assert_sabotage_separated(_metric_payload(zero_metric), floor, name="zero-basis")

    behaviour = _run_behavioural_parity(args, args.output)
    result = {
        "schema": "vllm-residual-intervention-validation/v2",
        "model": args.model,
        "prompt_count": len(prompts),
        "prompt_sha256": prompt_digest.hexdigest(),
        "bundle": str(args.subspace_bundle),
        "teacher_forced": {
            "top_k": TOP_K,
            "positions_per_prompt": DEFAULT_POSITION_COUNT,
            "sampled_positions": position_record,
            "hf_vs_vllm": _metric_payload(hooked_metric),
            "hf_vs_hf_noise_floor": _metric_payload(repeat_metric),
            "hf_unhooked_vs_vllm_unhooked_noise_floor": floor,
            "wrong_layer_sabotage": _metric_payload(wrong_metric),
            "zero_basis_noop_sabotage": _metric_payload(zero_metric),
            "zeroed_basis_sabotage": zero_sabotage,
        },
        "behavioural": behaviour,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("vLLM residual validation written to %s", args.output)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    raise SystemExit(main())
