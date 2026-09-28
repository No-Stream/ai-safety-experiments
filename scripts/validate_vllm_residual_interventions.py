# ruff: noqa: INP001
"""Validate residual interventions against a teacher-forced HuggingFace reference.

This is an operator-run GPU check.  It deliberately keeps the prompt corpus and model paths
outside the repository: ``--prompts-jsonl`` is a private runtime input and the output contains
only digests and aggregate metrics.  The vLLM leg uses ``prompt_logprobs=-1``.  A vLLM build that
does not return a complete distribution fails before any parity number is reported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
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
from transformers import AutoConfig

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


@dataclass(frozen=True)
class PromptRow:
    """Private prompt input without retaining its text in validation output."""

    prompt: str
    index: int


@dataclass(frozen=True)
class HFPromptResult:
    """HF teacher-forcing distributions and the exact token ids they scored."""

    token_ids: tuple[int, ...]
    logprobs: tuple[torch.Tensor, ...]


@dataclass
class MetricAccumulator:
    """Streaming parity statistics over full-vocabulary position distributions."""

    n_positions: int = 0
    kl_sum: float = 0.0
    kl_max: float = 0.0
    top1_matches: int = 0

    def add(self, reference: torch.Tensor, comparison: torch.Tensor) -> None:
        """Add one full-vocabulary position to the running metrics."""
        if reference.shape != comparison.shape or reference.ndim != 1:
            raise ValueError(
                "parity distributions must be matching rank-1 vocabulary vectors; "
                f"got {tuple(reference.shape)} and {tuple(comparison.shape)}"
            )
        if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(comparison).all()):
            raise ValueError("parity distributions contain non-finite log probabilities")
        reference_probability = reference.exp()
        kl = float((reference_probability * (reference - comparison)).sum().item())
        self.n_positions += 1
        self.kl_sum += kl
        self.kl_max = max(self.kl_max, kl)
        self.top1_matches += int(torch.argmax(reference) == torch.argmax(comparison))

    def payload(self) -> dict[str, float | int]:
        """Return the aggregate metrics, refusing an empty denominator."""
        if self.n_positions == 0:
            raise RuntimeError("parity metric has no scored positions")
        return {
            "n_positions": self.n_positions,
            "mean_kl": self.kl_sum / self.n_positions,
            "max_kl": self.kl_max,
            "top1_agreement": self.top1_matches / self.n_positions,
        }


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
    intervention_context: AbstractContextManager[None],
) -> HFPromptResult:
    """Return one CPU log-probability vector per scored prompt position from HF."""
    device = next(model.parameters()).device
    inputs = _model_inputs(tokenizer, rendered, device)
    with intervention_context, torch.inference_mode():
        output = model(**inputs)
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != RANK_THREE:
        raise TypeError("HF forward output lacks rank-3 logits")
    if logits.shape[1] != inputs["input_ids"].shape[1]:
        raise ValueError("HF logits sequence length differs from tokenized prompt")
    # Position i predicts token i+1.  vLLM stores the same distribution at prompt_logprobs[i+1].
    token_ids = tuple(int(token_id) for token_id in inputs["input_ids"][0].cpu().tolist())
    return HFPromptResult(
        token_ids=token_ids,
        logprobs=tuple(torch.log_softmax(row.float(), dim=-1).cpu() for row in logits[0, :-1]),
    )


def _load_hf_prompt_result(path: Path) -> HFPromptResult:
    """Load one spooled HF result without retaining the full prompt set in host memory."""
    payload: object = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"spooled HF result {path} is not a mapping")
    raw_token_ids = payload.get("token_ids")
    raw_logprobs = payload.get("logprobs")
    if not isinstance(raw_token_ids, (list, tuple)) or not all(
        isinstance(token_id, int) for token_id in raw_token_ids
    ):
        raise TypeError(f"spooled HF result {path} has invalid token ids")
    if not isinstance(raw_logprobs, (list, tuple)) or not all(
        isinstance(row, torch.Tensor) for row in raw_logprobs
    ):
        raise TypeError(f"spooled HF result {path} has invalid logprob rows")
    token_ids = tuple(raw_token_ids)
    logprobs = tuple(raw_logprobs)
    if len(token_ids) != len(logprobs) + 1:
        raise ValueError(f"spooled HF result {path} has inconsistent token and logprob lengths")
    return HFPromptResult(token_ids=token_ids, logprobs=logprobs)


def _vllm_prompt_logprobs(  # noqa: C901
    llm: object,
    rendered: str,
    *,
    expected_token_ids: Sequence[int],
    expected_vocab_size: int,
    intervention_context: AbstractContextManager[None],
) -> tuple[torch.Tensor, ...]:
    """Read complete prompt distributions from one vLLM request, failing on truncation."""
    generate = getattr(llm, "generate", None)
    if not callable(generate):
        raise TypeError("vLLM engine does not expose generate")
    sampling = vllm.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=1,
        prompt_logprobs=-1,
        detokenize=False,
    )
    with intervention_context:
        outputs = generate([rendered], sampling)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("vLLM returned an unexpected number of validation requests")
    request = outputs[0]
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    prompt_logprobs = getattr(request, "prompt_logprobs", None)
    if list(prompt_token_ids or ()) != list(expected_token_ids):
        raise RuntimeError("vLLM tokenized the validation prompt differently from HF")
    if prompt_logprobs is None or len(prompt_logprobs) != len(expected_token_ids):
        raise RuntimeError(
            "vLLM did not return one prompt_logprobs entry per prompt token; "
            "full-distribution parity cannot be measured"
        )
    rows: list[torch.Tensor] = []
    for position in range(1, len(expected_token_ids)):
        entries = prompt_logprobs[position]
        if entries is None:
            raise RuntimeError(f"vLLM returned no prompt logprobs at position {position}")
        if not isinstance(entries, Mapping) or len(entries) != expected_vocab_size:
            raise RuntimeError(
                "vLLM prompt_logprobs=-1 did not return the complete vocabulary at position "
                f"{position}: got {len(entries) if isinstance(entries, Mapping) else type(entries)} "
                f"entries, expected {expected_vocab_size}"
            )
        row = torch.full((expected_vocab_size,), -torch.inf, dtype=torch.float32)
        for token_id, value in entries.items():
            if not isinstance(token_id, int) or not 0 <= token_id < expected_vocab_size:
                raise RuntimeError(f"vLLM returned an invalid prompt logprob token id {token_id!r}")
            logprob_value = cast("VLLMLogprobValue", value).logprob
            row[token_id] = float(logprob_value)
        if not bool(torch.isfinite(row).all()):
            raise RuntimeError(
                f"vLLM prompt logprobs at position {position} are not a complete finite distribution"
            )
        rows.append(torch.log_softmax(row, dim=-1))
    return tuple(rows)


def _score_pair(
    reference: Sequence[torch.Tensor], comparison: Sequence[torch.Tensor], metric: MetricAccumulator
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


def _assert_full_vocab_logprobs(
    backend: InterventionVLLMBackend, *, expected_vocab_size: int
) -> None:
    """Refuse a vLLM engine whose max-logprobs guard would truncate ``-1`` requests."""
    engine = backend._llm  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    model_config = getattr(engine.llm_engine, "model_config", None)
    max_logprobs = getattr(model_config, "max_logprobs", None)
    if max_logprobs not in (-1, expected_vocab_size):
        raise RuntimeError(
            "vLLM cannot provide a full prompt distribution: its max_logprobs is "
            f"{max_logprobs!r}, while the model vocabulary has {expected_vocab_size} entries. "
            "The validator refuses to replace full-vocabulary KL with a top-k approximation."
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

    adapter_base_model: PreTrainedModel | None = None
    attached_adapter: AttachedAdapter | None = None
    hf_backend: HFBackend
    if args.adapter is None:
        hf_backend = HFBackend(args.model, thinking=False)
    else:
        adapter_base_model = load_adapter_base(
            args.model,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device=torch.device("cuda"),
        )
        attached_adapter = attach_adapter(adapter_base_model, args.adapter.resolve(), args.model)
        hf_backend = HFBackend(args.model, thinking=False, model=attached_adapter.peft_model)

    hf_model = cast("AutoModelForCausalLM", hf_backend.model)
    hf_torch_model = cast("torch.nn.Module", hf_model)
    hf_tokenizer = hf_backend.tokenizer
    layer_count = len(_decoder_layers(hf_model))
    n_layers = layer_count
    spec = _intervention_spec(bundle, n_layers=n_layers)
    spec.validate()
    noise_metric = MetricAccumulator()
    reference_temp = tempfile.TemporaryDirectory(prefix="vllm-residual-reference-")
    reference_paths: list[Path] = []
    prompt_token_lengths: list[int] = []
    expected_vocab_size: int | None = None
    for row in prompts:
        rendered = _render_chat(hf_tokenizer, row.prompt)
        reference = _hf_prompt_logprobs(
            hf_torch_model,
            hf_tokenizer,
            rendered,
            intervention_context=subspace_intervention(hf_model, bundle),
        )
        repeat = _hf_prompt_logprobs(
            hf_torch_model,
            hf_tokenizer,
            rendered,
            intervention_context=subspace_intervention(hf_model, bundle),
        )
        _score_pair(reference.logprobs, repeat.logprobs, noise_metric)
        if not reference.logprobs:
            raise RuntimeError(f"prompt row {row.index} produced no next-token positions")
        row_vocab_size = reference.logprobs[0].shape[0]
        if expected_vocab_size is None:
            expected_vocab_size = row_vocab_size
        elif row_vocab_size != expected_vocab_size:
            raise ValueError("HF prompt rows have inconsistent vocabulary sizes")
        reference_path = Path(reference_temp.name) / f"prompt-{row.index:03d}.pt"
        torch.save(
            {"token_ids": reference.token_ids, "logprobs": reference.logprobs},
            reference_path,
        )
        reference_paths.append(reference_path)
        prompt_token_lengths.append(len(reference.token_ids))
    if not reference_paths or expected_vocab_size is None:
        raise RuntimeError("teacher-forced prompt set produced no next-token positions")

    if args.adapter is not None:
        adapter_base_model = None
        attached_adapter = None
    del hf_backend
    del hf_model
    del hf_torch_model
    gc.collect()
    torch.cuda.empty_cache()

    prompt_token_length = max(prompt_token_lengths)
    max_model_len = prompt_token_length + 1
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
    engine_kwargs: dict[str, Any] = {
        "enforce_eager": True,
        "worker_extension_cls": WORKER_EXTENSION_CLASS,
        "gpu_memory_utilization": vllm_fraction,
        "max_model_len": max_model_len,
    }
    if getattr(AutoConfig.from_pretrained(args.model), "vision_config", None) is not None:
        engine_kwargs["language_model_only"] = True
    if args.adapter is not None:
        facts = read_adapter_facts(args.adapter.resolve())
        engine_kwargs.update(
            enable_lora=True,
            max_lora_rank=facts.vllm_lora_rank,
            lora_target_modules=list(facts.target_modules),
        )
        vllm_backend = InterventionVLLMBackend(
            args.model,
            thinking=False,
            max_logprobs=-1,
            lora_adapter=str(args.adapter.resolve()),
            **engine_kwargs,
        )
    else:
        vllm_backend = InterventionVLLMBackend(
            args.model,
            thinking=False,
            max_logprobs=-1,
            **engine_kwargs,
        )
    dimensions = vllm_backend.llm.collective_rpc("residual_intervention_dimensions")
    if len(dimensions) != 1:
        raise RuntimeError(f"expected one vLLM model worker, got {len(dimensions)}")
    vllm_n_layers, vllm_d_model = dimensions[0]
    if vllm_n_layers != n_layers or vllm_d_model != d_model:
        raise ValueError(
            f"vLLM dimensions {(vllm_n_layers, vllm_d_model)} disagree with HF/bundle "
            f"dimensions {(n_layers, d_model)}"
        )
    vllm_metric = MetricAccumulator()
    sabotage_metric = MetricAccumulator()
    zero_sabotage: dict[str, str] = {"status": "not-run"}
    try:
        _assert_full_vocab_logprobs(
            vllm_backend,
            expected_vocab_size=expected_vocab_size,
        )
        with intervention(vllm_backend.llm, spec):
            for row, reference_path in zip(prompts, reference_paths, strict=True):
                hf_result = _load_hf_prompt_result(reference_path)
                rendered = _render_chat(vllm_backend.tokenizer, row.prompt)
                vllm_rows = _vllm_prompt_logprobs(
                    vllm_backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                    rendered,
                    expected_token_ids=hf_result.token_ids,
                    expected_vocab_size=hf_result.logprobs[0].shape[0],
                    intervention_context=nullcontext(),
                )
                _score_pair(hf_result.logprobs, vllm_rows, vllm_metric)

        wrong_spec = _wrong_layer_spec(bundle, n_layers=n_layers, d_model=d_model)
        first_row = prompts[0]
        first_hf_result = _load_hf_prompt_result(reference_paths[0])
        rendered = _render_chat(vllm_backend.tokenizer, first_row.prompt)
        with intervention(vllm_backend.llm, wrong_spec):
            wrong_rows = _vllm_prompt_logprobs(
                vllm_backend._llm,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
                rendered,
                expected_token_ids=first_hf_result.token_ids,
                expected_vocab_size=expected_vocab_size,
                intervention_context=nullcontext(),
            )
        _score_pair(first_hf_result.logprobs, wrong_rows, sabotage_metric)
        wrong_payload = _metric_payload(sabotage_metric)
        if wrong_payload["mean_kl"] <= max(10.0 * _metric_payload(noise_metric)["mean_kl"], 1e-5):
            raise RuntimeError(
                f"wrong-layer sabotage did not produce a clear parity failure: {wrong_payload}"
            )

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
    finally:
        if engine_baseline is None:
            raise RuntimeError("vLLM engine baseline was not recorded")
        try:
            release_engine(vllm_backend, baseline_mib=engine_baseline)
        finally:
            del vllm_backend
            gc.collect()
            reference_temp.cleanup()

    behaviour = _run_behavioural_parity(args, args.output)
    result = {
        "schema": "vllm-residual-intervention-validation/v1",
        "model": args.model,
        "prompt_count": len(prompts),
        "prompt_sha256": prompt_digest.hexdigest(),
        "bundle": str(args.subspace_bundle),
        "teacher_forced": {
            "hf_vs_vllm": _metric_payload(vllm_metric),
            "hf_vs_hf_noise_floor": _metric_payload(noise_metric),
            "wrong_layer_sabotage": _metric_payload(sabotage_metric),
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
